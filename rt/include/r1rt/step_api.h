/* r1rt/step_api.h - the C ABI of the real-time core library (DESIGN §2 principle 2: a library with step() first,
 * a process second). The r1rt process calls it from its 500 Hz writer thread; the simulator (r1sim lockstep, via the
 * nanobind module r1rt_py, sessions 2-4) calls the same functions with simulated time, so the same code is tested.
 *
 * Contract:
 *   - plain C, no exceptions cross it (the implementation catches nothing because it throws nothing: -fno-exceptions
 *     in RT code is the plan), no allocation inside r1rt_core_step(), constant time per call;
 *   - the core's other inputs (stop channel, upper-body command ring, base command, hands contact, ownership token)
 *     and its outputs besides the command (state, heartbeat, mirror, telemetry/policy rings) live in the shared
 *     segment given at creation (shm_layout.h): a real mapping on the robot, ordinary memory in the simulator;
 *   - r1rt_core_step() fills `cmd_out`; cmd_out->publish == 0 means "write nothing this tick". The caller (the DDS
 *     layer) maps the 26 joints to the 35 hg slots (R1RT_HG_SLOT_MAP_INIT), adds the CRC32 and writes, after checking
 *     the ownership token once more immediately before the write.
 *
 * Session 1: the ABI and the segment functions. Session 3 (ABI 2): the core is real - the 500 Hz writer step
 * (DESIGN §3.1: state -> estimate -> checks §4.2 -> policy action / inline policy (rung 1) / STAND_FREEZE (rung 3) ->
 * upper body HOLD -> limits -> LowCmd), the 50 Hz policy thread's iteration (r1rt_core_policy_step: observation +
 * network = r1rt::PolicyRunner, the C++ port of v2's WalkPolicy), operator requests, the telemetry record, the state
 * seqlock, the heartbeat and the guard mirror in the segment. NEVER an automatic damp: DAMP is entered only by an
 * operator request (R1RT_REQ_DAMP / the stop channel's damp level). ABI 3 (session 3, the `wire` step): the
 * upper-body executor (r1rt::UpperBodyExecutor, DESIGN §3.3) runs inside the writer step for arms + head - records
 * from the segment's upper_cmd ring (r1d -> r1rt; r1rt_segment_push_upper_cmd is the client side), the per-group
 * TRAJECTORY keepalives (r1rt_segment_upper_keepalive), the stop channel's epoch; every stop the core raises itself
 * (R1RT_REQ_STOP, entering STAND_FREEZE) is posted on the stop channel, so the epoch r1d stamps on its records is the
 * one the executor checks. ABI 4 (session 4): the takeover sequence MONITOR -> TAKEOVER_MIRROR -> TAKEOVER_BLEND ->
 * BALANCE (DESIGN §3.5; v2's proven gates and blend, r1mc2/runtime.py:335-412, with v3's failure branches: never a
 * damp) - requests R1RT_REQ_TAKEOVER / _RELEASED / _ABORT, the built-in controller's last rt/lowcmd as an input
 * (r1rt_core_set_builtin_cmd), the gates in r1rt_config; and the race-free hand-back between the writer and the
 * policy thread (every policy result is computed on a private copy of the committed runner state and committed only
 * if nothing else committed since it read that state: a late thread result can never overwrite a newer inline one,
 * and vice versa). ABI 5 (session 5, layout 2): the mode-request mailbox in the segment (several clients -> the core,
 * read every writer tick: r1rt_segment_request_mode / r1rt_segment_mode_reply; DAMP only from the operator
 * requester), ON_STAND (R1RT_REQ_ON_STAND: from MONITOR through the takeover's MIRROR -> BLEND into a stiff hold of
 * the measured pose, then per-joint L0 for legs + waist through the upper_cmd ring's leg / waist groups),
 * R1RT_FAULT_GUARD_ABSENT wired to r1guard's heartbeat, the guard mirror's blend / freeze / waist fields.
 * Not yet: the guard's hand-back of the token.
 */
#ifndef R1RT_STEP_API_H
#define R1RT_STEP_API_H

#include <stddef.h>
#include <stdint.h>

#include "r1rt/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* bump on any change to these declarations or to a struct in types.h (2 = session 3: the policy interface, the
 * stand-freeze pose, requests, policy_step, status, fault injection; 3 = session 3: the upper-body executor in the
 * writer - status fields, the segment's upper_cmd / keepalive client functions; 4 = session 4: the takeover sequence -
 * config gates, requests, r1rt_core_set_builtin_cmd, status fields - and the policy hand-back's commit counters) */
#define R1RT_ABI_VERSION 8u  /* 5 = session 5: the mode-request mailbox, ON_STAND (config gains, request, status), the
                                guard-absent alarm, R1RT_E_AGAIN / R1RT_E_ABANDONED; 6 = session 6: R1RT_REQ_RECLAIM
                                (the operator's hand-back from r1guard in ON_STAND), the stop channel's damp level
                                operator-only, status fields, the guard heartbeat's flags byte (types.h); 7 = session 8
                                (HW1 fix #29 / #31): config resume_gap_ns, the resume / stop status fields, the
                                mirror's tau_ff (types.h, segment layout 3); 8 = session 9 (U19 / D25): the state-age
                                limit + the stale rung (config, status, R1RT_FAULT_STATE_OLD) */

/* status codes */
#define R1RT_OK 0
#define R1RT_E_NOT_IMPLEMENTED (-1)
#define R1RT_E_INVALID_ARG (-2)
#define R1RT_E_ABI_MISMATCH (-3)
#define R1RT_E_BAD_SEGMENT (-4)
#define R1RT_E_NOT_OWNER (-5)
#define R1RT_E_REFUSED (-6)      /* a request not allowed in this mode / state (e.g. BALANCE without a policy) */
#define R1RT_E_POLICY_LOAD (-7)  /* create(): the policy weights file could not be read / does not fit the params */
#define R1RT_E_AGAIN (-8)        /* the mode-request mailbox is busy (another client's request is pending: retry), or
                                    a reply is not there yet */
#define R1RT_E_ABANDONED (-9)    /* a mailbox request the client claimed but did not publish within
                                    kModeClaimTimeoutNs: the core dropped it (nothing was done) */

/* The policy interface (DESIGN §3.2), filled from the policy's ONNX metadata + r1mc_policy.json by the caller
 * (r1d from the validated profile; the simulator from r1mc3.policy.ReferencePolicy): the core parses no metadata.
 * Semantics = v2 r1mc2/policy.py WalkPolicy (Manip2-Lean subset + the Manip3 waist term). */
#define R1RT_POLICY_MAX_JOINTS 26
#define R1RT_POLICY_MAX_TERMS 9
enum {
  R1RT_OBS_BASE_ANG_VEL = 1,     /* IMU gyro, 3 */
  R1RT_OBS_PROJECTED_GRAVITY = 2,/* R^T (0, 0, -1), 3 */
  R1RT_OBS_COMMAND = 3,          /* (vx, vy, wz) clipped, 3 */
  R1RT_OBS_LEAN = 4,             /* ramped lean, 1 */
  R1RT_OBS_WAIST_CMD = 5,        /* ramped waist roll, yaw, 2 */
  R1RT_OBS_PHASE = 6,            /* sin / cos of the gait phase, zeroed when |cmd| < 0.1, 2 */
  R1RT_OBS_JOINT_POS = 7,        /* q - default, n_joints */
  R1RT_OBS_JOINT_VEL = 8,        /* dq, n_joints */
  R1RT_OBS_ACTIONS = 9           /* previous raw action, n_act */
};
typedef struct r1rt_policy_params {
  uint32_t n_joints;          /* joints the policy observes (24 for Manip2-Lean) */
  uint32_t n_act;             /* joints it acts on (14 = legs + waist), <= R1RT_NUM_POLICY_ACTIONS */
  uint32_t n_terms;           /* entries used in obs_terms */
  uint32_t has_lean;          /* the "lean" observation exists (ramped lean command) */
  uint32_t has_waist;         /* the "waist_cmd" observation exists (Manip3) */
  int32_t obs_terms[R1RT_POLICY_MAX_TERMS];      /* R1RT_OBS_* in observation order */
  int32_t joint_index[R1RT_POLICY_MAX_JOINTS];   /* SDK index of policy joint i */
  int32_t act_pos[R1RT_NUM_POLICY_ACTIONS];      /* position (within the policy joints) of action k */
  uint32_t _pad0;
  double default_pos[R1RT_POLICY_MAX_JOINTS];
  double kp[R1RT_POLICY_MAX_JOINTS];
  double kd[R1RT_POLICY_MAX_JOINTS];
  double action_scale[R1RT_NUM_POLICY_ACTIONS];
  double kp_other[R1RT_NUM_JOINTS];  /* gains of the SDK joints the policy does not observe (v2: KP_STIFF) */
  double kd_other[R1RT_NUM_JOINTS];
  double period_s;            /* gait period (0.86) */
  double step_dt_s;           /* policy period (0.02) */
  double cmd_lo[3], cmd_hi[3];
  double lean_lo, lean_hi;    /* rad */
  double lean_rate;           /* rad/s of the lean ramp (4 deg/s) */
  double waist_lo[2], waist_hi[2];
  double waist_rate[2];       /* rad/s */
} r1rt_policy_params;

/* Everything the core needs at creation; values come from the validated robot profile (config/, every value with
 * provenance) - the core reads no files. Joint vectors in SDK order. */
typedef struct r1rt_config {
  uint32_t abi_version;        /* = R1RT_ABI_VERSION */
  uint32_t struct_size;        /* = sizeof(r1rt_config) */
  uint32_t writer_rate_hz;     /* 500 (200 fallback; 200-1000 allowed) */
  uint32_t policy_rate_hz;     /* 50, as trained */
  double q_min[R1RT_NUM_JOINTS];
  double q_max[R1RT_NUM_JOINTS];
  double dq_max[R1RT_NUM_JOINTS];
  double tau_max[R1RT_NUM_JOINTS];  /* clamp at 0.6 x TAU_MAX as v2 */
  double kp_max[R1RT_NUM_JOINTS];
  double kd_max[R1RT_NUM_JOINTS];
  double stop_time_s;          /* trajectory keepalive expiry -> decelerate within this (default 0.15) */
  double arm_tilt_stop_rad;    /* 4 deg over the start of an arm motion */
  double fall_tilt_rad;        /* 30 deg: event only, the policy keeps running */
  int64_t lowstate_late_ns;    /* 10 ms */
  int64_t lowstate_stale_ns;   /* 100 ms: resend the last balanced command */
  int64_t lowstate_lost_ns;    /* 1 s: fault (no limp) */
  int64_t policy_max_age_ns;   /* 40 ms: run the policy inline */
  void* segment;               /* the shared segment (r1rt_segment_init'ed), owned by the caller */
  size_t segment_size;
  /* ---- ABI 2 (session 3) ---- */
  const char* policy_weights_path; /* R1W1 file (tools/policy_convert.py); read in create() only. NULL = no policy
                                      (BALANCE is refused) */
  r1rt_policy_params policy;
  double stand_q[R1RT_NUM_JOINTS];  /* STAND_FREEZE pose: the CoM-centred double-support stand (S1: POSE_STAND
                                       itself falls backwards under a plain PD; r1sim.model.centred_stand_pose) */
  double stand_kp[R1RT_NUM_JOINTS];
  double stand_kd[R1RT_NUM_JOINTS];
  double stand_ramp_s;              /* 0.3-0.5 s ramp from the last command to stand_q */
  double upper_kp[R1RT_NUM_JOINTS]; /* HOLD gains of arms + head (joints 14-25; the other entries unused) */
  double upper_kd[R1RT_NUM_JOINTS];
  double action_abs_max;            /* a raw action beyond +-this (or non-finite) is INVALID -> STAND_FREEZE */
  double damp_kd[R1RT_NUM_JOINTS];  /* DAMP (operator only): kp 0, this kd */
  int64_t loop_gap_ns;              /* writer iteration gap counted as a loop_gap event (2 periods) */
  /* ---- ABI 4 (session 4): the takeover (DESIGN §3.5; v2 r1mc2/runtime.py:335-412). 0 = the default in brackets ---- */
  double takeover_blend_s;          /* BLEND mirror -> ours [1.0 s: v2's takeover default (into=stand; 0.3 s into=walk),
                                       and the length at which rung 3 still catches an invalid policy right after the
                                       takeover in the simulator (0.3 s / instant did not, S4)] */
  double takeover_tilt_max_rad;     /* gate: tilt [12 deg] */
  double takeover_dq_max;           /* gate: max |dq| over the 26 joints [0.5 rad/s] */
  double takeover_temp_max_c;       /* gate: every motor below this [60 C] */
  int64_t takeover_state_max_age_ns;/* gate: lowstate age [50 ms] */
  int64_t builtin_max_age_ns;       /* gate: the built-in's last rt/lowcmd younger than this [100 ms] ... */
  double builtin_min_leg_kp;        /* ... and stiff: every leg kp above this [10] (stand lock 200, its walk 84 / 31) */
  double mirror_max_dev_rad;        /* the built-in command is mirrored only if every joint is within this of the
                                       measured pose and every motor enabled [0.6 rad]; otherwise MIRROR holds the
                                       measured pose at the stand gains (v2: its HOLD tick) */
  int64_t takeover_mirror_alarm_ns; /* MIRROR longer than this without RELEASED / ABORT: alarm only [5 s] */
  uint32_t takeover_require_guard;  /* 1 = refuse a takeover without a fresh r1guard heartbeat (robot profiles);
                                       0 = no guard (the simulator's single-core tests) */
  uint32_t _pad_abi4;
  int64_t guard_max_age_ns;         /* the guard heartbeat's maximum age for that gate [50 ms] */
  /* ---- ABI 5 (session 5): ON_STAND (DESIGN §3.5: the robot hanging on its stand, feet off the floor) ---- */
  double on_stand_kp[R1RT_NUM_JOINTS]; /* HOLD gains of legs + waist (joints 0-13) in ON_STAND and its blend; all 0 =
                                          the stand gains (stand_kp / stand_kd); arms + head use upper_kp / upper_kd */
  double on_stand_kd[R1RT_NUM_JOINTS];
  /* ---- ABI 5 (session 5, the process wrapper; DESIGN §2.1: "refuses takeover if real-time scheduling was not
   * granted") ---- */
  uint32_t rt_granted;              /* the process's grants at create (SCHED_FIFO + mlockall + affinity): 1 / 0; later
                                       changes: r1rt_core_set_rt_granted. Copied into every heartbeat */
  uint32_t takeover_require_rt;     /* 1 = TAKEOVER / ON_STAND refused (R1RT_TAKEOVER_RT) and R1RT_FAULT_RT_NOT_GRANTED
                                       raised (an alarm) while rt_granted is 0 (robot profiles); 0 = the simulator */
  /* ---- ABI 7 (session 8, HW1 fix #29: the resume after a long silence) ---- */
  int64_t resume_gap_ns;            /* a writer silence longer than this [100 ms] is a STOP (the process stopped /
                                       stalled: SIGSTOP, a whole-machine stall): the board holds the last LowCmd only
                                       ~400 ms and then releases into a damped sink (HW1 board-hold test), so the next
                                       command must not snap back to the old target - the core re-enters through a
                                       blend from the MEASURED pose (ON_STAND and every upper-body HOLD: over
                                       takeover_blend_s; STAND_FREEZE: its ramp restarted from the measured pose;
                                       in-flight upper / lower motions stopped). BALANCE's legs are NOT blended (the
                                       policy keeps its authority, DESIGN D3). Below the board's hold the blend starts
                                       at the held pose: harmless. r1guard applies the same rule (its own silence, and
                                       a takeover of a core silent longer than this) */
  /* ---- ABI 8 (session 9, U19 / D25: the state-age limit + the "hold the stand" rung; docs/sim/STATE_AGE.md) ---- */
  int64_t state_age_limit_ns;       /* the robot state is OLD when its age exceeds this [10 ms: the user's target, 11 Oct
                                       2026; B7: a constant extra 15 ms of latency degrades, 20 ms falls; HW1 T5: age
                                       p99 1.0 / max 3.0 ms]. Age = the receive age, or the board-tick CONTENT age
                                       (r1rt/state_age.hpp) when that clock is valid and larger. Always measured:
                                       R1RT_FAULT_STATE_OLD (an alarm) + the status counters, whatever stale_rung is */
  int64_t state_age_sustain_ns;     /* ... continuously for this long while BALANCING -> the rung [60 ms] */
  int64_t stale_return_live_ns;     /* stale_rung 2: the state live (age <= the limit) this long, after the freeze ramp
                                       ended, before BALANCE comes back by itself [500 ms] */
  double board_tick_hz;             /* the board's tick rate for the content age [1000: HW1 999.5-1000.5 /s] */
  uint32_t stale_rung;              /* 0 = OFF (as before session 9: the policy runs on the old state until
                                       lowstate_stale_ns, then the last balanced command is resent - rung 4);
                                       1 = HOLD THE STAND: STAND_FREEZE (the 0.3-0.5 s ramp to the CoM-centred stand at
                                       stand gains, arms / head HOLD; never limp; it needs no state) - BALANCE comes
                                       back by the operator (ENTER_BALANCE / r1 resume words);
                                       2 = the same, then BALANCE again by itself once the state is live again
                                       (stale_return_live_ns). The robot profile ships 0 until the user's OK */
  uint32_t _pad_abi8;
} r1rt_config;

/* Operator / service requests (r1rt_core_request). */
enum {
  R1RT_REQ_ENTER_BALANCE = 1, /* MONITOR / STAND_FREEZE -> BALANCE: takes the ownership token, runs the policy once
                                 inline for a bumpless start (the end of a takeover; tests). Refused without a valid
                                 policy or without fresh state */
  R1RT_REQ_STOP = 2,          /* base command zero, arms decelerate to HOLD, keep balancing (posted on the stop
                                 channel's stop level, so the stop epoch grows: older upper-body records are refused) */
  R1RT_REQ_RESUME = 3,        /* clears a STOP */
  R1RT_REQ_DAMP = 4,          /* DAMP: kp 0 + damping. OPERATOR ONLY - nothing in the core ever requests it */
  R1RT_REQ_STAND_FREEZE = 5,  /* operator: freeze into the stand (also entered automatically on an invalid policy) */
  /* ABI 4: the takeover sequence (DESIGN §3.5). r1d runs it: TAKEOVER, then the motion-switcher's ReleaseMode RPC in
   * r1d (never in an RT thread; v2 waited <= 4 s), then RELEASED (it succeeded) or ABORT (it failed / timed out). */
  R1RT_REQ_TAKEOVER = 6,          /* MONITOR -> TAKEOVER_MIRROR after the gates (R1RT_E_REFUSED + status
                                     takeover_refusal otherwise; nothing changes): takes the ownership token and
                                     republishes a snapshot of the built-in's last command */
  R1RT_REQ_TAKEOVER_RELEASED = 7, /* TAKEOVER_MIRROR -> TAKEOVER_BLEND: the policy starts (entry action inline), the
                                     command blends linearly from the snapshot to ours over takeover_blend_s, then
                                     BALANCE. Faults during the blend do not stop it (alarm); an invalid policy action
                                     freezes into the stand (rung 3) from the blended command */
  R1RT_REQ_TAKEOVER_ABORT = 8,    /* TAKEOVER_MIRROR -> MONITOR: the release did not succeed, the built-in still
                                     drives the motors - stop competing (publish nothing, the token back), alarm.
                                     Refused in BLEND / BALANCE (the built-in has released: stopping would be limp) */
  /* ABI 5: ON_STAND (DESIGN §3.5). r1d runs it like a takeover: ON_STAND (the gates for a hanging robot; MIRROR of
   * the built-in's command), ReleaseMode, then TAKEOVER_RELEASED (BLEND into a stiff hold of the measured pose ->
   * ON_STAND) or TAKEOVER_ABORT. In ON_STAND legs + waist follow the lower groups' L0 records (HOLD / STREAM /
   * TRAJECTORY, legs only on the stand), arms + head the upper-body executor; no policy runs; it is left only by the
   * operator's DAMP (the session-end sequence). */
  R1RT_REQ_ON_STAND = 9,
  /* ABI 6 (session 6; DESIGN §4.4, §14.1 T6): the OPERATOR's hand-back from r1guard. Accepted only from
   * R1RT_REQUESTER_OPERATOR (the mailbox; R1RT_REFUSE_NOT_OPERATOR otherwise), only while r1guard holds the token
   * (R1RT_REFUSE_TOKEN), only with the core in ON_STAND (the guard took over from ON_STAND or from the sequence into it:
   * holding, not balancing - a balancing hand-back is new and unproven, R1RT_REFUSE_MODE) and with fresh state and both
   * executors ready (R1RT_REFUSE_STATE). The core takes the token back (CAS R1GUARD -> R1RT, a new generation), both
   * executors HOLD at the targets the guard was holding (the core's last published command), and the command blends
   * over takeover_blend_s from exactly what the guard sent (no velocity / feed-forward, the guard's gains) into the
   * ON_STAND hold (TAKEOVER_BLEND, target ON_STAND, then ON_STAND). r1guard sees the token gone, stops writing (at most
   * one stale packet: the same held command) and re-arms (standby, warm, a new takeover possible). */
  R1RT_REQ_RECLAIM = 10,
  /* ABI 6 (session 6; HW1 runbook S10, the user's U3: the fast STAND_FREEZE ramp seen on the stand first): in ON_STAND,
   * R1RT_REQ_STAND_FREEZE from the OPERATOR (R1RT_REFUSE_NOT_OPERATOR otherwise) ramps legs + waist from the last command
   * to the stand pose over stand_ramp_s at the stand gains, then holds (arms / head stop to HOLD) - only when every leg /
   * waist target is within R1RT_ON_STAND_FREEZE_MAX_DEV_RAD of the stand pose (R1RT_REFUSE_FAR) and with fresh state.
   * A freeze entered from ON_STAND never leads to BALANCE (ENTER_BALANCE refused: never balance a hanging robot);
   * R1RT_REQ_STAND_HOLD returns it to the ON_STAND hold (a blend over takeover_blend_s from the freeze's command into a
   * hold of that command, then ON_STAND; no release - the built-in is long gone). */
  R1RT_REQ_STAND_HOLD = 11
};
#define R1RT_ON_STAND_FREEZE_MAX_DEV_RAD 0.15 /* the S10 test starts 0.10 rad from the stand pose (ADDITIONS S5) */

/* Why a mailbox request was refused (ModeReply.refusal) when it is not a takeover gate (those use R1RT_TAKEOVER_*). */
enum {
  R1RT_REFUSE_NONE = 0,
  R1RT_REFUSE_NOT_OPERATOR = 20, /* DAMP from a requester other than R1RT_REQUESTER_OPERATOR */
  R1RT_REFUSE_MODE = 21,         /* not allowed in the current mode (Core::request said R1RT_E_REFUSED) */
  R1RT_REFUSE_UNKNOWN = 22,      /* unknown request code */
  R1RT_REFUSE_TOKEN = 23,        /* ABI 6, RECLAIM: r1guard does not hold the ownership token (nothing to take back) */
  R1RT_REFUSE_STATE = 24,        /* ABI 6, RECLAIM / STAND_HOLD: no fresh lowstate, or an executor not ready */
  R1RT_REFUSE_FAR = 25           /* ABI 6, STAND_FREEZE in ON_STAND: a leg / waist target farther than
                                    R1RT_ON_STAND_FREEZE_MAX_DEV_RAD from the stand pose (a large fast motion) */
};

/* Why the last R1RT_REQ_TAKEOVER was refused (r1rt_core_status.takeover_refusal; 0 = accepted). */
enum {
  R1RT_TAKEOVER_OK = 0,
  R1RT_TAKEOVER_NOT_MONITOR = 1,     /* only from MONITOR */
  R1RT_TAKEOVER_NO_POLICY = 2,       /* no policy weights / executor */
  R1RT_TAKEOVER_STATE = 3,           /* no lowstate younger than takeover_state_max_age_ns */
  R1RT_TAKEOVER_TILT = 4,            /* tilted beyond takeover_tilt_max_rad */
  R1RT_TAKEOVER_MOVING = 5,          /* a joint faster than takeover_dq_max */
  R1RT_TAKEOVER_MOTOR_ERROR = 6,     /* a motor error word is set */
  R1RT_TAKEOVER_HOT = 7,             /* a motor at or above takeover_temp_max_c */
  R1RT_TAKEOVER_GUARD = 8,           /* takeover_require_guard and no fresh r1guard heartbeat */
  R1RT_TAKEOVER_BUILTIN = 9,         /* the built-in is not visibly holding the legs (no fresh, stiff rt/lowcmd) */
  R1RT_TAKEOVER_TOKEN = 10,          /* the ownership token is held by someone else */
  R1RT_TAKEOVER_RT = 11,             /* ABI 5: takeover_require_rt and the process's real-time grants missing */
  R1RT_TAKEOVER_STAND_SESSION = 12   /* ABI 6: the process runs a STAND SESSION (r1rt --stand-session, DESIGN §3.5's
                                        start flag): no entry into BALANCE for the process's lifetime - TAKEOVER and
                                        ENTER_BALANCE refused with this code (also ENTER_BALANCE's mailbox refusal) */
};

/* Fault injection at code points (DESIGN §12.6; simulation and tests only - r1d never calls it). */
enum {
  R1RT_INJECT_POLICY_NAN = 1,          /* the network output is NaN in [t_start, t_end) (thread and inline) */
  R1RT_INJECT_POLICY_OUT_OF_RANGE = 2  /* the network output is `value` in the window */
};

/* Counters and the latest values, for tests, telemetry and r1mon (read by the caller's thread; plain copy). */
typedef struct r1rt_core_status {
  uint64_t ticks;
  uint64_t policy_thread_runs;   /* actions published by the policy thread */
  uint64_t inline_runs;          /* policy steps run inline by the writer (rung 1) */
  uint64_t inline_entries;       /* times the writer went inline */
  uint64_t invalid_actions;      /* non-finite / out-of-range network outputs (not used) */
  uint64_t lowstate_late;        /* ticks with lowstate age > late threshold */
  uint64_t stale_resends;        /* ticks the last balanced command was resent (age > stale) */
  uint64_t clamp_total;
  uint64_t loop_gaps;
  uint64_t freeze_entries;
  uint64_t telemetry_drops;      /* ring full (nobody draining: lockstep without r1log) */
  int64_t last_action_t_ns;
  int64_t lowstate_age_ns;
  uint32_t faults;               /* R1RT_FAULT_* this tick */
  uint32_t faults_seen;          /* OR of every tick since create */
  double tilt_rad;
  double gait_phase;             /* 0..1 */
  double lean;                   /* the ramped lean the policy sees, rad */
  r1rt_mode mode;
  uint8_t inline_active;
  uint8_t stop_active;
  uint8_t policy_loaded;
  uint8_t exec_ready;            /* the upper-body executor initialised (BALANCE is refused without it) */
  uint8_t upper_last_reject;     /* the executor's last refusal reason (ExecReject, executor.hpp) */
  uint8_t _pad0;
  /* ---- ABI 3: the upper-body executor (per group: arm_left, arm_right, head) ---- */
  uint64_t upper_accepted;       /* upper_cmd records the executor accepted */
  uint64_t upper_rejected;       /* ... and refused (reason: upper_last_reject) */
  uint64_t upper_stop_epoch;     /* the stop epoch the executor holds (records stamped older are refused) */
  uint8_t upper_source[R1RT_NUM_UPPER_GROUPS];   /* r1rt_source in effect */
  uint8_t upper_event[R1RT_NUM_UPPER_GROUPS];    /* the group's last ExecEvent (guards 1-15, stops 16+, done 32) */
  uint8_t upper_stopping[R1RT_NUM_UPPER_GROUPS]; /* 1 = decelerating to HOLD */
  uint8_t upper_tick_event;      /* the first event raised in the last writer tick (= telemetry guard_event) */
  uint8_t _pad1[6];
  uint64_t upper_cmd_id[R1RT_NUM_UPPER_GROUPS];  /* the record each group follows (0 = HOLD of its own) */
  /* ---- ABI 4: the takeover and the policy hand-back ---- */
  uint64_t takeover_requests;    /* R1RT_REQ_TAKEOVER received */
  uint64_t takeover_aborts;      /* R1RT_REQ_TAKEOVER_ABORT accepted */
  uint64_t takeover_alarm_ticks; /* ticks with R1RT_FAULT_TAKEOVER_ALARM */
  int64_t builtin_age_ns;        /* age of the built-in's last rt/lowcmd (INT64_MAX = never received) */
  double blend_progress;         /* 0 in MIRROR, 0..1 in BLEND, 1 after */
  double mirror_dev_rad;         /* max |snapshot q - measured q| when the takeover started */
  uint64_t policy_commits;       /* policy results committed (thread + inline + entry) = the hand-back generation */
  uint64_t policy_discarded;     /* results computed on a state another commit had replaced: dropped, never used */
  uint8_t takeover_refusal;      /* R1RT_TAKEOVER_*: the last TAKEOVER request's verdict */
  uint8_t takeover_used_mirror;  /* 1 = MIRROR republishes the built-in's command, 0 = our hold at the measured pose;
                                    2 = (ABI 6, ON_STAND from a soft built-in: zero torque / damping / disabled slots)
                                    the measured pose at the built-in's own gains, blended up to the hold */
  uint8_t _pad2[6];
  /* ---- ABI 5 (session 5) ---- */
  uint64_t mode_requests;        /* mailbox requests processed (incl. refused) */
  uint64_t mode_abandoned;       /* mailbox claims abandoned (claimed, never published) */
  uint64_t mode_last_seq;        /* the last mailbox request processed */
  uint64_t guard_absent_ticks;   /* publishing ticks with R1RT_FAULT_GUARD_ABSENT */
  int64_t guard_age_ns;          /* r1guard's heartbeat age at the last tick (INT64_MAX = never seen) */
  uint8_t takeover_target;       /* the mode the running TAKEOVER_* sequence ends in: BALANCE or ON_STAND */
  uint8_t on_stand_refusal;      /* R1RT_TAKEOVER_*: the last ON_STAND request's verdict */
  uint8_t lower_source[R1RT_NUM_CMD_GROUPS - R1RT_NUM_UPPER_GROUPS];   /* legs + waist (ON_STAND): r1rt_source */
  uint8_t lower_event[R1RT_NUM_CMD_GROUPS - R1RT_NUM_UPPER_GROUPS];    /* ... their last ExecEvent */
  uint8_t lower_stopping[R1RT_NUM_CMD_GROUPS - R1RT_NUM_UPPER_GROUPS];
  uint8_t hands_pose_used;       /* 1 = the arm model used r1hands' finger joints + payloads this tick */
  uint8_t _pad3[4];
  uint64_t lower_cmd_id[R1RT_NUM_CMD_GROUPS - R1RT_NUM_UPPER_GROUPS];
  uint64_t lower_accepted;       /* lower-group records accepted / refused (ON_STAND only; refused elsewhere) */
  uint64_t lower_rejected;
  int32_t max_wake_lateness_ns;  /* the largest wake-up lateness / iteration time the process reported */
  int32_t max_iteration_ns;
  uint8_t rt_granted;            /* the process's real-time grants as last set */
  uint8_t _pad4[7];
  /* ---- ABI 6 (session 6) ---- */
  uint64_t damp_ignored;         /* stop-channel DAMP requests NOT from R1RT_REQUESTER_OPERATOR: acknowledged, ignored
                                    (DESIGN §3.5 / §5.1: DAMP is operator-only; the mailbox refuses them too) */
  uint64_t reclaims;             /* R1RT_REQ_RECLAIM accepted (the token back from r1guard) */
  uint8_t reclaim_refusal;       /* R1RT_REFUSE_*: the last RECLAIM's verdict (0 = accepted) */
  uint8_t freeze_refusal;        /* R1RT_REFUSE_*: the last STAND_FREEZE / STAND_HOLD request's verdict (0 = accepted) */
  uint8_t freeze_from_stand;     /* 1 = the current STAND_FREEZE was entered from ON_STAND (no BALANCE from it) */
  uint8_t motor_error_groups;    /* bit g = upper group g (arm_left / arm_right / head) HELD for a motor error word
                                    (DESIGN §4.2: HOLD that limb; its records refused until the operator's RESUME with
                                    the word cleared); legs / waist: alarm only (R1RT_FAULT_MOTOR_ERROR) */
  uint8_t temp_max_joint;        /* the hottest joint this tick (SDK index) */
  uint8_t stand_session;         /* 1 = a stand session (r1rt --stand-session): BALANCE is never entered */
  uint8_t _pad5[2];
  uint32_t motor_error_joints;   /* bit j = joint j's motor error word is nonzero this tick */
  int32_t temp_max_c;            /* the hottest motor temperature this tick (deg C, either sensor) */
  uint64_t motor_error_holds;    /* upper groups put in HOLD for a motor error */
  uint64_t over_temp_ticks;      /* ticks with R1RT_FAULT_OVER_TEMP (a warning: temperature >= takeover_temp_max_c) */
  /* ---- ABI 7 (session 8: HW1 fix #29 the resume after a long silence, #31 stops vs overruns) ---- */
  uint64_t stops;                /* writer silences longer than resume_gap_ns (deliberate test SIGSTOPs or whole-
                                    process stalls): counted apart from the scheduler's overruns (loop_gaps) */
  int64_t stop_ns_max;           /* the longest such silence */
  int64_t stop_ns_total;         /* their sum (r1rt's skipped deadlines inside them = this / the writer period) */
  uint64_t resume_blends;        /* resume blends started from the measured pose (after a stop, once fresh state came) */
  uint64_t resume_waits_expired; /* stops after which no fresh state came within R1RT_RESUME_STATE_WAIT_NS: the normal
                                    stale handling then (the last command resent) */
  uint64_t resume_uncaught;      /* blends started after R1RT_RESUME_STATE_WAIT_NS from a sample fresh by its stamp but
                                    whose board tick never reached the extrapolated tick (r1rt::ResumeClock) */
  double resume_progress;        /* 0..1 while a resume blend runs, 1 = none running */
  uint8_t resume_waiting;        /* 1 = a stop was seen, waiting for fresh state before publishing (ON_STAND /
                                    STAND_FREEZE / the sequence into ON_STAND) */
  uint8_t _pad6[7];
  uint64_t resume_drain_waits;   /* session 9 (#29 hazard ii): stops whose wait ran past R1RT_RESUME_STATE_WAIT_NS
                                    because the samples coming in were still the stop's backlog (fresh stamp, board
                                    tick provably old): the wait went on, up to R1RT_RESUME_DRAIN_MAX_NS */
  /* ---- ABI 8 (session 9: the state-age limit, docs/sim/STATE_AGE.md) ---- */
  int64_t state_content_age_ns;     /* the newest sample's board-tick content age (-1 = no valid tick clock) */
  int64_t state_content_age_max_ns; /* its maximum since create (while valid) */
  uint64_t state_old_ticks;         /* ticks with the state older than state_age_limit_ns (R1RT_FAULT_STATE_OLD) */
  uint64_t state_old_episodes;      /* episodes of it that lasted state_age_sustain_ns (counted with the rung OFF too) */
  int64_t state_old_ns_max;         /* the longest such episode so far */
  uint64_t stale_rung_entries;      /* times the rung froze a balancing robot into the stand */
  uint64_t stale_rung_returns;      /* times BALANCE came back by itself afterwards (stale_rung 2) */
  uint64_t content_age_relearns;    /* the board-tick clock re-based on a stable elevated offset (a new time base) */
  uint8_t stale_freeze;             /* 1 = the current STAND_FREEZE was entered by the state-age rung */
  uint8_t content_age_valid;        /* 1 = the board-tick clock is valid this tick */
  uint8_t _pad7[6];
} r1rt_core_status;
#define R1RT_DEFAULT_STATE_AGE_LIMIT_NS 10000000LL   /* state_age_limit_ns when 0: 10 ms */
#define R1RT_DEFAULT_STATE_AGE_SUSTAIN_NS 60000000LL /* state_age_sustain_ns when 0: 60 ms */
#define R1RT_DEFAULT_STALE_RETURN_LIVE_NS 500000000LL /* stale_return_live_ns when 0: 0.5 s */
#define R1RT_DEFAULT_BOARD_TICK_HZ 1000.0            /* board_tick_hz when 0 */
#define R1RT_DEFAULT_RESUME_GAP_NS 100000000LL /* r1rt_config.resume_gap_ns when 0: 100 ms (a quarter of the board's
                                                  ~400 ms hold, HW1; 50 x r1rt's worst real lateness 0.34 ms) */
#define R1RT_RESUME_SETTLE_NS 10000000LL /* ... the sample's board tick at the tick extrapolated from before the stop
                                           (r1rt/resume.hpp): the receive thread drains the samples queued during the
                                           stop, each stamped at its RECEIVE time - a stale pose with a fresh stamp (found
                                           with the processes, session 8); AND, only while the ResumeClock has no learned
                                           tick rate, at least this long after the stop's end (session 9: with a rate the
                                           first live sample ends the wait - #29's open race, a stale pre-stop packet
                                           acted >= 10 ms) */
#define R1RT_RESUME_STATE_WAIT_NS 50000000LL /* after a stop: wait at most this long for a lowstate younger than
                                                lowstate_late_ns before giving up (the receive thread was stopped too:
                                                the state of the stop's start must not be the blend's origin) */
#define R1RT_RESUME_DRAIN_MAX_NS 400000000LL /* session 9 (#29 hazard ii): ... but while FRESH samples keep coming whose
                                                board tick has not caught up (the DDS backlog of the stop still
                                                draining: each stamped at receive, its pose from the stop's start), the
                                                wait goes on up to this long - a blend from such a sample is the snap.
                                                400 ms = the robot's worst backlog (a 4 MB request, Linux doubles it:
                                                ~3400 LowState samples of ~2.4 KB) at the slowest drain measured (~100
                                                us / sample, macOS under load); after it, a still-uncaught fresh sample
                                                is taken as live (a board tick that moved off the learned rate) */

typedef struct r1rt_core r1rt_core; /* opaque */

/* The ABI version this library was built with (compare with R1RT_ABI_VERSION of the caller's header). */
uint32_t r1rt_abi_version(void);

/* Shared segment: size to map; placement-construct (creator only; zeroes everything, sets magic/version and the
 * slot map, then `ready`); check before use (magic, layout version, size, ready). */
size_t r1rt_segment_size(void);
int32_t r1rt_segment_init(void* base, size_t size, int32_t creator_pid, int64_t t_monotonic_ns);
int32_t r1rt_segment_check(const void* base, size_t size);

/* Core lifetime. create() validates the config (and may allocate: it runs before the RT loop starts). Returns NULL
 * and sets *status_out (if given) on error. destroy(NULL) is a no-op. */
r1rt_core* r1rt_core_create(const r1rt_config* config, int32_t* status_out);
void r1rt_core_destroy(r1rt_core* core);

/* One writer tick. t_monotonic_ns: CLOCK_MONOTONIC (robot) or simulated time (lockstep). state may be NULL when no
 * new lowstate arrived since the last tick. telemetry_out (optional) receives this tick's record. Returns R1RT_OK or
 * an error code; on ANY return cmd_out is fully written (publish = 0 when nothing must be sent). */
int32_t r1rt_core_step(r1rt_core* core, const r1rt_lowstate_in* state, int64_t t_monotonic_ns,
                       r1rt_lowcmd_out* cmd_out, r1rt_telemetry_record* telemetry_out);

/* Current mode (r1rt_mode values); R1RT_MODE_MONITOR for a NULL core. */
r1rt_mode r1rt_core_mode(const r1rt_core* core);

/* ONE iteration of the 50 Hz policy thread (DESIGN §3.2): reads the core's latest state and base command, builds
 * the observation, runs the network and publishes the action with its time into the core's policy mailbox. Does
 * nothing outside BALANCE, while the state is stale, or while the writer runs the policy inline and has not handed
 * back (the thread's result would race the writer's; it resumes at its next iteration after the hand-back). */
int32_t r1rt_core_policy_step(r1rt_core* core, int64_t t_monotonic_ns);

/* Operator / service requests (R1RT_REQ_*): R1RT_OK, R1RT_E_INVALID_ARG (unknown) or R1RT_E_REFUSED (not allowed
 * in this mode / state). */
int32_t r1rt_core_request(r1rt_core* core, int32_t request, int64_t t_monotonic_ns);

/* The base command (r1d's TTL'd velocity / lean command), written into the segment's base_cmd seqlock. */
int32_t r1rt_core_set_base_cmd(r1rt_core* core, const r1rt_base_cmd* cmd);

/* The built-in controller's last rt/lowcmd (Unitree's own writer; the DDS layer passes only samples from foreign
 * writers, never our own), mapped to the 26 joints, with its receive time. Called by the DDS receive thread (a
 * seqlock inside the core: single writer = that thread) or by the simulator. Used for the takeover gates and the
 * MIRROR snapshot, and watched afterwards (R1RT_FAULT_BUILTIN_ACTIVE). */
int32_t r1rt_core_set_builtin_cmd(r1rt_core* core, const r1rt_lowcmd_out* cmd, int64_t t_rx_ns);

/* Simulation / tests only: a fault at a code point (R1RT_INJECT_*), active for t_start_ns <= t < t_end_ns. */
int32_t r1rt_core_inject(r1rt_core* core, int32_t kind, int64_t t_start_ns, int64_t t_end_ns, double value);

/* Counters and latest values (a copy). */
int32_t r1rt_core_get_status(const r1rt_core* core, r1rt_core_status* out);

/* ABI 5 (session 5): the process wrapper's facts, from the WRITER thread (plain stores the next step reads).
 * set_rt_granted: the real-time grants (1 / 0) now - copied into every heartbeat; with takeover_require_rt a 0 refuses
 * TAKEOVER / ON_STAND (R1RT_TAKEOVER_RT) and raises R1RT_FAULT_RT_NOT_GRANTED (an alarm only).
 * set_loop_timing: called before each r1rt_core_step - this tick's wake-up lateness and the PREVIOUS iteration's
 * duration (wake-up to after the DDS write; the current one is not finished when the record is written), both in ns;
 * they land in this tick's telemetry record (wake_lateness_ns / iteration_ns) and the status maxima. */
int32_t r1rt_core_set_rt_granted(r1rt_core* core, int32_t granted);
/* ABI 6 (session 6, DESIGN §3.5 "start flag --on-stand"): a STAND SESSION - from this call on, for the core's
 * lifetime, every entry into BALANCE is refused (R1RT_TAKEOVER_STAND_SESSION: TAKEOVER, ENTER_BALANCE); ON_STAND, the
 * stand freeze / hold, RECLAIM, DAMP, STOP unchanged. It cannot be cleared: on = 1 sets it (R1RT_OK), on = 0 is
 * R1RT_E_REFUSED once it is set (R1RT_OK while it was never set). r1rt sets it before its threads start
 * (--stand-session); the rt heartbeat's flags byte carries it (R1RT_RT_FLAG_STAND_SESSION). */
int32_t r1rt_core_set_stand_session(r1rt_core* core, int32_t on);
int32_t r1rt_core_set_loop_timing(r1rt_core* core, int32_t wake_lateness_ns, int32_t prev_iteration_ns);

/* Client side of the segment's upper-body inputs (r1d's role; the simulator and tests use them too). Each is
 * single-producer: one r1d thread. push: R1RT_OK, R1RT_E_REFUSED when the ring is full (the record is dropped and
 * counted by nothing - r1d retries), R1RT_E_BAD_SEGMENT. keepalive: renews the TRAJECTORY deadman of `group` for
 * record cmd_id (the core passes the latest stamp to the executor every tick). stop_epoch: the stop channel's epoch
 * (the sum of its stop / halt / damp requests) - stamp it on every record (r1rt_upper_cmd.stop_epoch). */
int32_t r1rt_segment_push_upper_cmd(void* base, const r1rt_upper_cmd* cmd);
int32_t r1rt_segment_upper_keepalive(void* base, uint32_t group, uint64_t cmd_id, int64_t t_ns);
uint64_t r1rt_segment_stop_epoch(const void* base);

/* ABI 5: the mode-request mailbox (shm_layout.h ModeMailbox), the client side - any number of clients, lock-free.
 * request_mode: R1RT_OK (*seq_out = the request's sequence number, > 0), R1RT_E_AGAIN (another request is pending:
 * retry), R1RT_E_ABANDONED (the core dropped this claim: the client stalled > kModeClaimTimeoutNs between claim and
 * publish; nothing was done), R1RT_E_INVALID_ARG, R1RT_E_BAD_SEGMENT. The core processes it at its next writer tick.
 * mode_reply: R1RT_OK when request `seq` was processed (*status = its result: R1RT_OK / R1RT_E_REFUSED /
 * R1RT_E_INVALID_ARG / R1RT_E_NOT_OWNER / R1RT_E_ABANDONED; *refusal = R1RT_TAKEOVER_* or R1RT_REFUSE_*;
 * *mode_after = the core's mode right after it), R1RT_E_AGAIN while pending, R1RT_E_INVALID_ARG for a sequence that
 * was never issued or whose reply was overwritten (more than kModeReplySlots = 64 requests later). */
int32_t r1rt_segment_request_mode(void* base, uint32_t req, uint32_t requester, int64_t t_ns, uint64_t* seq_out);
int32_t r1rt_segment_mode_reply(const void* base, uint64_t seq, int32_t* status, int32_t* refusal,
                                uint32_t* mode_after);

/* The hg LowCmd_ image of a command (26 joints -> 35 motor slots with R1RT_HG_SLOT_MAP_INIT, unused slots 0,
 * mode_pr / mode_machine, CRC32 last): what the DDS layer writes. `out` must point to sizeof(r1rt::hg::LowCmd)
 * bytes (1004). Returns the CRC. */
uint32_t r1rt_fill_hg_lowcmd(const r1rt_lowcmd_out* cmd, void* out, size_t out_size);

#ifdef __cplusplus
}
#endif

#endif /* R1RT_STEP_API_H */
