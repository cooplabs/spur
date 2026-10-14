/* r1rt/types.h - C-compatible payload types shared by r1rt, r1guard, r1d, r1hands, r1log and the simulator.
 *
 * Plain data only: fixed sizes, no pointers, explicit padding, native little-endian (arm64 / aarch64 / x86-64).
 * These structs travel inside the shared-memory segment (shm_layout.h) and through the C step API (step_api.h).
 * Every size is pinned by a static assertion (bottom of this file): changing a struct means bumping
 * R1RT_LAYOUT_VERSION and R1RT_ABI_VERSION. Design: docs/architecture/RT_CORE_INTERFACES.md, DESIGN.md §3, §9.
 *
 * Joint vectors are the 26 R1 joints in SDK order (r1sim.joints.JOINTS: 12 legs, waist roll/yaw, 2 x 5 arm,
 * head pitch/yaw). The DDS layer maps them to the 35 hg LowCmd_/LowState_ motor slots with R1RT_HG_SLOT_MAP_INIT
 * (confirmed on our robot from the 28 Sep 2026 read-only lowstate capture; v2 r1mc2/joints.py:26-28).
 */
#ifndef R1RT_TYPES_H
#define R1RT_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define R1RT_NUM_JOINTS 26
#define R1RT_HG_NUM_SLOTS 35          /* unitree_hg LowCmd_.motor_cmd length */
#define R1RT_MODE_MACHINE_R1 1        /* R1 26-DoF (observed on our robot) */
#define R1RT_MODE_PR 0                /* ankles commanded as pitch/roll (observed) */
#define R1RT_NUM_POLICY_ACTIONS 14    /* legs + waist (R1MC-Manip2-Lean) */
#define R1RT_MAX_POLICY_OBS 128       /* live policy: 74; Manip3: 76 */
#define R1RT_MAX_GROUP_JOINTS 6       /* a leg (6, ON_STAND); an arm uses 5, the head and the waist 2 (layout 2: was 5) */
#define R1RT_MAX_KNOTS 8              /* trajectory knots per upper-body command record */
#define R1RT_REMOTE_BYTES 40          /* hg LowState_.wireless_remote */
#define R1RT_NUM_HANDS 2

/* joint index (SDK order) -> hg motor slot; slots 14, 20, 21, 27, 28, 31-34 are unused on the R1 */
#define R1RT_HG_SLOT_MAP_INIT \
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30 }

/* ---- enumerations (stored as fixed-width integers; C enums have no fixed size) ---- */

/* r1rt modes (DESIGN §3.5). Only DAMP is limp, and it is entered only by the operator. */
typedef uint16_t r1rt_mode;
enum {
  R1RT_MODE_MONITOR = 0,         /* nothing published; Unitree's controller in charge */
  R1RT_MODE_TAKEOVER_MIRROR = 1, /* mirror Unitree's last lowcmd */
  R1RT_MODE_TAKEOVER_BLEND = 2,  /* blend into ours */
  R1RT_MODE_BALANCE = 3,         /* policy owns legs + waist, executor owns arms + head */
  R1RT_MODE_GUARDED = 4,         /* r1guard owns the token (new, unproven) */
  R1RT_MODE_STAND_FREEZE = 5,    /* no valid policy: ramp to the stand and hold (new, unproven) */
  R1RT_MODE_ON_STAND = 6,        /* robot hanging on its stand: per-joint L0, stiff hold (new, unproven) */
  R1RT_MODE_OVERLAY_ARM_SDK = 7, /* Unitree balances, we publish rt/arm_sdk (new, unproven) */
  R1RT_MODE_DAMP = 8             /* operator only */
};

/* joint groups of the upper_cmd ring (DESIGN §3.3): arms + head (the upper-body executor, BALANCE and ON_STAND) and,
 * session 5, legs + waist (the lower executor, ON_STAND only: the robot hanging on its stand, DESIGN §3.5 / §5.1 L0) */
typedef uint8_t r1rt_group;
enum {
  R1RT_GROUP_ARM_LEFT = 0,   /* joints 14-18 */
  R1RT_GROUP_ARM_RIGHT = 1,  /* joints 19-23 */
  R1RT_GROUP_HEAD = 2,       /* joints 24-25 */
  R1RT_NUM_UPPER_GROUPS = 3,
  R1RT_GROUP_LEG_LEFT = 3,   /* joints 0-5 (ON_STAND only) */
  R1RT_GROUP_LEG_RIGHT = 4,  /* joints 6-11 (ON_STAND only) */
  R1RT_GROUP_WAIST = 5,      /* joints 12-13 (ON_STAND only) */
  R1RT_NUM_CMD_GROUPS = 6
};

/* the one source a group follows at a time (DESIGN §3.3) */
typedef uint8_t r1rt_source;
enum { R1RT_SOURCE_HOLD = 0, R1RT_SOURCE_TRAJECTORY = 1, R1RT_SOURCE_STREAM = 2, R1RT_SOURCE_FREEDRIVE = 3 };

/* stop channel levels (DESIGN §4.2) */
enum { R1RT_STOP_STOP = 0, R1RT_STOP_HALT = 1, R1RT_STOP_DAMP = 2, R1RT_NUM_STOP_LEVELS = 3 };

/* ownership of rt/lowcmd (DESIGN §2 principle 1, §4.3) */
typedef uint8_t r1rt_owner;
enum { R1RT_OWNER_NONE = 0, R1RT_OWNER_R1RT = 1, R1RT_OWNER_R1GUARD = 2 };

/* who asks (stop channel requester ids and the mode-request mailbox, session 5). The low ids equal the owners, so the
 * core's own stop channel requests keep their id. DAMP through the mailbox is accepted only from OPERATOR (DESIGN §5.1:
 * `damp` only from an operator client - the CLI's / r1d's operator connection). */
enum {
  R1RT_REQUESTER_NONE = 0,
  R1RT_REQUESTER_R1RT = 1,
  R1RT_REQUESTER_R1GUARD = 2,
  R1RT_REQUESTER_R1D = 3,       /* the service (goals, the takeover sequence, ON_STAND) */
  R1RT_REQUESTER_OPERATOR = 4,  /* an operator client (r1 CLI / the operator's r1d connection): DAMP allowed */
  R1RT_REQUESTER_SIM = 5,       /* the simulator / test harnesses (DAMP refused, as any non-operator) */
  R1RT_NUM_REQUESTERS = 6
};

/* fault bits (DESIGN §4.2); a fault is an event + alarm, never an automatic damp */
enum {
  R1RT_FAULT_LOWSTATE_LATE = 1u << 0,   /* > 10 ms */
  R1RT_FAULT_LOWSTATE_STALE = 1u << 1,  /* > 100 ms: last balanced command resent */
  R1RT_FAULT_LOWSTATE_LOST = 1u << 2,   /* > 1 s */
  R1RT_FAULT_CRC = 1u << 3,
  R1RT_FAULT_TILT_ARM_STOP = 1u << 4,   /* 4 deg over the start of an arm motion */
  R1RT_FAULT_TILT_FALL = 1u << 5,       /* 30 deg: the policy keeps running */
  R1RT_FAULT_JOINT_CLAMP = 1u << 6,
  R1RT_FAULT_VELOCITY_CLAMP = 1u << 7,
  R1RT_FAULT_TORQUE_CLAMP = 1u << 8,
  R1RT_FAULT_MOTOR_ERROR = 1u << 9,
  R1RT_FAULT_POLICY_LATE = 1u << 10,    /* > 40 ms: inline policy (rung 1) */
  R1RT_FAULT_POLICY_INVALID = 1u << 11, /* STAND_FREEZE (rung 3) */
  R1RT_FAULT_LOOP_GAP = 1u << 12,       /* > 2 writer periods */
  R1RT_FAULT_NOT_OWNER = 1u << 13,
  R1RT_FAULT_GUARD_ABSENT = 1u << 14,   /* blocks takeover / ON_STAND; session 5: also raised (alarm only) in a
                                           publishing mode while r1guard's heartbeat is stale or missing, when the
                                           profile has a guard (takeover_require_guard) */
  R1RT_FAULT_RT_NOT_GRANTED = 1u << 15, /* SCHED_FIFO not granted: takeover refused */
  /* session 4 (ABI 4): the takeover sequence (DESIGN §3.5). Alarms only: they never change what is published. */
  R1RT_FAULT_TAKEOVER_ALARM = 1u << 16, /* a fault during MIRROR / BLEND (kept mirroring / finished the blend), a
                                           MIRROR longer than takeover_mirror_alarm_ns, or the takeover aborted */
  R1RT_FAULT_BUILTIN_ACTIVE = 1u << 17, /* the built-in controller's rt/lowcmd still arrives while we blend /
                                           balance: two writers (the release did not take; the operator decides) */
  R1RT_FAULT_OVER_TEMP = 1u << 18,      /* session 6 (DESIGN §4.1 over-temperature): a joint's temperature at or above
                                           takeover_temp_max_c (60 C) - a WARNING only, never an action on stiffness */
  R1RT_FAULT_STATE_OLD = 1u << 19       /* session 9 (U19 / D25): the robot state older than state_age_limit_ns (the
                                           receive age or the board-tick content age) - an alarm; the stale rung acts
                                           on it only when enabled and sustained (step_api.h stale_rung) */
};

/* ---- payloads ---- */

/* The robot state as received (one hg LowState_, already CRC-checked and mapped to the 26 joints). */
typedef struct r1rt_lowstate_in {
  int64_t t_rx_ns;            /* CLOCK_MONOTONIC receive time */
  uint32_t board_tick;        /* hg LowState_.tick (~1 kHz board counter) */
  uint8_t mode_pr;            /* 0 = PR ankles */
  uint8_t mode_machine;       /* 1 = R1 26-DoF */
  uint8_t crc_ok;             /* 1 = the DDS layer's CRC32 check passed */
  uint8_t _pad0;
  double q[R1RT_NUM_JOINTS];  /* rad, joint side */
  double dq[R1RT_NUM_JOINTS]; /* rad/s */
  double ddq[R1RT_NUM_JOINTS];
  double tau_est[R1RT_NUM_JOINTS];             /* N m (board estimate; calibration per joint is open) */
  int16_t temperature[R1RT_NUM_JOINTS][2];     /* hg MotorState_.temperature[2], deg C */
  float voltage[R1RT_NUM_JOINTS];              /* V */
  uint32_t motor_state[R1RT_NUM_JOINTS];       /* hg MotorState_.motorstate (error word; bit meaning open) */
  double imu_quat[4];         /* w x y z */
  double imu_gyro[3];         /* rad/s */
  double imu_acc[3];          /* m/s^2 */
  double imu_rpy[3];          /* rad */
  uint8_t wireless_remote[R1RT_REMOTE_BYTES];
} r1rt_lowstate_in;

/* The state as the core sees it, published every writer tick (shared-memory seqlock "state"). */
typedef struct r1rt_state {
  uint64_t seq;               /* writer tick that published it */
  int64_t t_core_ns;          /* monotonic publish time */
  r1rt_lowstate_in in;
  double projected_gravity[3];
  double tilt_rad;            /* from the learned stance (v2 semantics) */
  int64_t lowstate_age_ns;
  uint32_t faults;            /* R1RT_FAULT_* active this tick */
  r1rt_mode mode;
  r1rt_owner token_owner;
  uint8_t _pad0;
} r1rt_state;

/* One command the writer sends (26 joints; the DDS layer fills the 35 hg slots, the CRC and writes). */
typedef struct r1rt_lowcmd_out {
  uint8_t publish;            /* 0 = write NOTHING this tick (MONITOR, not owner, stub core) */
  uint8_t mode_pr;
  uint8_t mode_machine;
  uint8_t _pad0[5];
  uint8_t enable[R1RT_NUM_JOINTS]; /* hg motor_cmd.mode (1 = enabled) */
  uint8_t _pad1[6];
  double q[R1RT_NUM_JOINTS];
  double dq[R1RT_NUM_JOINTS];
  double kp[R1RT_NUM_JOINTS];
  double kd[R1RT_NUM_JOINTS];
  double tau_ff[R1RT_NUM_JOINTS]; /* board law: tau = kp (q - q_m) + kd (dq - dq_m) + tau_ff */
} r1rt_lowcmd_out;

/* Writer heartbeat (r1rt every tick; r1guard every cycle, in its own slot). */
typedef struct r1rt_heartbeat {
  uint64_t seq;               /* +1 per beat */
  int64_t t_ns;               /* monotonic time of this beat */
  uint64_t lowstate_tick;     /* board tick of the state used */
  uint32_t rate_hz;           /* writer rate (500; 200 fallback) */
  uint32_t faults;
  float blend_progress;       /* 0..1 in TAKEOVER_BLEND */
  r1rt_mode mode;
  uint8_t rt_granted;         /* 1 = SCHED_FIFO + mlockall granted */
  uint8_t guard_flags;        /* ABI 6 (session 6; was padding, the bytes unchanged): the writer's flags byte -
                                 in r1guard's heartbeat R1RT_GUARD_FLAG_* + the mode it took over from in bits 4-7;
                                 in r1rt's heartbeat R1RT_RT_FLAG_* (the stand session) */
} r1rt_heartbeat;

/* r1guard's heartbeat flags (ABI 6): what r1d's status shows ("the guard armed"). */
#define R1RT_GUARD_FLAG_ARMED 0x01u      /* takeover enabled (0 = observe only: heartbeat + shadow, never takes over) */
#define R1RT_GUARD_FLAG_ACTIVE 0x02u     /* holds the token and publishes (mode GUARDED) */
#define R1RT_GUARD_FLAG_DAMP 0x04u       /* active and publishing the OPERATOR's damp */
#define R1RT_GUARD_FLAG_NOT_OWNER 0x08u  /* active, but the token is no longer its own (not writing) */
#define R1RT_GUARD_FLAG_MODE_FROM_SHIFT 4 /* bits 4-7: the r1rt_mode it took over from (while active) */
/* r1rt's heartbeat flags (the same byte in the core's own heartbeat; ABI 6): */
#define R1RT_RT_FLAG_STAND_SESSION 0x01u /* r1rt runs a stand session (--stand-session): BALANCE is never entered */
#define R1RT_RT_FLAG_STOP_ACTIVE 0x02u   /* session 13: a stop / halt is latched (until a resume): the base command -
                                           the policy's lean - is zero (HW2 day 2's F12: r1d accepted `lean 3` and the
                                           core used 0 after a client's stop; r1d could not see the latch) */

/* What the guard mirrors every cycle to stay warm (DESIGN §4.3: bumpless takeover). */
typedef struct r1rt_mirror {
  uint64_t seq;
  int64_t t_ns;
  int64_t last_action_t_ns;
  double gait_phase;
  double last_action[R1RT_NUM_POLICY_ACTIONS];
  double base_cmd[4];         /* vx, vy, wz, lean as used this tick */
  double q_target[R1RT_NUM_JOINTS]; /* last commanded q (upper-body targets included) */
  double kp[R1RT_NUM_JOINTS];
  double kd[R1RT_NUM_JOINTS];
  float blend_progress;
  r1rt_mode mode;
  uint8_t policy_valid;
  uint8_t freeze_invalid;     /* layout 2: STAND_FREEZE because no valid policy (1) or the operator's request (0) */
  /* ---- layout 2 (session 5): what the guard needs to continue the core EXACTLY (GUARD.md "not solid" 2) ---- */
  int64_t blend_t0_ns;        /* TAKEOVER_BLEND: when the blend started (the core's clock); 0 outside a blend */
  int64_t blend_ns;           /* the core's blend length (takeover_blend_s after its default) */
  int64_t freeze_t0_ns;       /* STAND_FREEZE: when the ramp started; 0 outside the freeze */
  int64_t freeze_ramp_ns;     /* the core's stand_ramp_s (after its default) */
  double waist[2];            /* the policy runner's ramped waist command (Manip3; 0 for Manip2-Lean) */
  double waist_target[2];
  uint8_t takeover_target;    /* the mode a TAKEOVER_* sequence ends in: BALANCE (3) or ON_STAND (6) */
  uint8_t _pad1[7];
  /* ---- layout 3 (session 8, HW1 fix #28): the feed-forward of the last command - r1guard holds the arms / head
   *      with it (the core's arm gravity feed-forward, +-1.2-1.6 N m at the shoulders: without it T6 moved the arms
   *      0.004-0.020 rad on the robot) ---- */
  double tau_ff[R1RT_NUM_JOINTS];
  /* (layout 3, HW1 fix #31) the core's stops - writer silences > resume_gap_ns, deliberate (test SIGSTOPs) or
   * stalls - apart from the scheduler's overruns; r1d reads them here (SegmentView.mirror()) for `r1 status` */
  uint64_t stops;
  int64_t stop_ns_max;
  uint64_t resume_blends;     /* resume blends from the measured pose started (fix #29) */
  uint8_t resume_waiting;     /* 1 = a stop was seen, fresh state awaited (nothing published in ON_STAND / freeze) */
  uint8_t _pad2[7];
} r1rt_mirror;

/* Base command from r1d (policy command; TTL <= 0.5 s, expiry = zero). Single writer: r1d. */
typedef struct r1rt_base_cmd {
  uint64_t seq;
  int64_t t_issue_ns;
  int64_t ttl_ns;
  double vx, vy, wz;          /* m/s, m/s, rad/s */
  double lean;                /* rad (policy lean command) */
  double waist_roll, waist_yaw; /* rad (Manip3 waist command; ignored by Manip2-Lean) */
  uint32_t requester;
  uint32_t _pad0;
} r1rt_base_cmd;

/* Contact summary from r1hands (single writer), read by the RT trajectory guards every tick. */
typedef struct r1rt_hands_contact {
  uint64_t seq;
  int64_t t_ns;
  uint32_t pad_contact[R1RT_NUM_HANDS];  /* bit k = Dex3 pad k (0-8) in contact */
  uint8_t finger_stalled[R1RT_NUM_HANDS]; /* bit per finger: thumb 0, middle 1, index 2 */
  uint8_t finger_blocked[R1RT_NUM_HANDS];
  uint8_t grip_state[R1RT_NUM_HANDS];    /* OPEN .. HOLDING (DESIGN §6.2) */
  uint8_t valid;                         /* 0 = hands state stale or hands absent */
  uint8_t _pad0;
} r1rt_hands_contact;

/* Layout 4 (session 10): the upper-body (and lower, ON_STAND) executors' per-group status and the state-age
 * counters, published by r1rt every writer tick (single writer) -> r1d. On real processes this is how r1d learns how
 * a goal ended (DONE vs the guard that stopped it) and whether the core refused its record; the lockstep core gives
 * r1d the same numbers through Core.status() (r1rt_core_status's upper_* / lower_* / state-age fields). */
typedef struct r1rt_exec_status {
  uint64_t seq;               /* writer tick that published it */
  int64_t t_ns;               /* monotonic publish time (a reader judges r1rt alive from it) */
  uint64_t upper_accepted;    /* upper_cmd records the upper executor accepted ... */
  uint64_t upper_rejected;    /* ... and refused (reason: last_reject) */
  uint64_t stop_epoch;        /* the stop epoch the executor holds */
  uint64_t lower_accepted;    /* legs / waist records (ON_STAND) accepted / refused */
  uint64_t lower_rejected;
  uint64_t cmd_id[R1RT_NUM_CMD_GROUPS];  /* the record each group follows (0 = HOLD of its own); groups as r1rt_group */
  uint8_t source[R1RT_NUM_CMD_GROUPS];   /* r1rt_source in effect */
  uint8_t event[R1RT_NUM_CMD_GROUPS];    /* the group's last ExecEvent (guards 1-15, stops 16+, done 32) */
  uint8_t stopping[R1RT_NUM_CMD_GROUPS]; /* 1 = decelerating to HOLD */
  uint8_t last_reject;        /* the executor's last refusal reason (ExecReject; the lower executor's if it refused last) */
  uint8_t tick_event;         /* the first event raised in this tick (= telemetry guard_event) */
  uint8_t motor_error_groups; /* bit g = upper group g HELD for a motor error word */
  uint8_t content_age_valid;  /* 1 = the board-tick clock is valid this tick (session 9) */
  uint8_t stale_freeze;       /* 1 = the current STAND_FREEZE was entered by the state-age rung */
  uint8_t stale_rung;         /* the configured rung (0 = off, 1, 2; D34: 0 for HW2) */
  int64_t state_content_age_ns;     /* the newest sample's board-tick content age (-1 = no valid tick clock) */
  int64_t state_content_age_max_ns; /* its maximum since create */
  uint64_t state_old_ticks;         /* ticks with R1RT_FAULT_STATE_OLD */
  uint64_t state_old_episodes;      /* episodes that lasted state_age_sustain_ns (rung OFF too) */
  int64_t state_old_ns_max;         /* the longest such episode */
  uint64_t stale_rung_entries;      /* times the rung froze a balancing robot into the stand */
  uint64_t content_age_relearns;    /* the board-tick clock re-based */
} r1rt_exec_status;

/* One knot of a jerk-limited trajectory (sampled by r1rt at the writer rate; quintic Hermite between knots). */
typedef struct r1rt_knot {
  int64_t t_ns;               /* from the trajectory start */
  double q[R1RT_MAX_GROUP_JOINTS];
  double dq[R1RT_MAX_GROUP_JOINTS];
  double ddq[R1RT_MAX_GROUP_JOINTS];
} r1rt_knot;

/* Guards evaluated every tick on a TRAJECTORY (DESIGN §3.3); they travel with it. 0 = guard off. */
typedef struct r1rt_traj_guards {
  double surface_z_m;         /* lowest fingertip may not go below this (base frame), NaN/0 with bit off */
  float lag_rad;              /* joint lag over the expected-lag model */
  float tau_nm;               /* torque estimate threshold (calibrated joints only) */
  uint32_t contact_mask;      /* pads that count (r1rt_hands_contact.pad_contact bits) */
  uint8_t contact_all;        /* 1 = every pad in the mask, 0 = any */
  uint8_t enable;             /* bit 0 lag, 1 torque, 2 contact, 3 surface */
  uint8_t hand;               /* 0 left, 1 right */
  uint8_t _pad0;
} r1rt_traj_guards;

/* One upper-body command record (ring "upper_cmd", single producer r1d -> single consumer r1rt). A trajectory
 * longer than R1RT_MAX_KNOTS knots is sent as consecutive records with segment_index 0, 1, ... and last_segment. */
typedef struct r1rt_upper_cmd {
  uint64_t cmd_id;            /* r1d goal / stream id */
  uint64_t stop_epoch;        /* stop-channel requests seen by r1d when issued; older records are dropped */
  int64_t t_issue_ns;
  int64_t t_start_ns;         /* TRAJECTORY start (monotonic); 0 = on receipt */
  int64_t ttl_ns;             /* STREAM: expiry after t_issue (<= 300 ms arms); TRAJECTORY: keepalive period */
  r1rt_group group;
  r1rt_source source;
  uint8_t n_joints;           /* 5 arm, 2 head */
  uint8_t n_knots;
  uint16_t segment_index;
  uint8_t last_segment;
  uint8_t _pad0;
  double q[R1RT_MAX_GROUP_JOINTS];      /* HOLD / STREAM / FREEDRIVE setpoints */
  double dq[R1RT_MAX_GROUP_JOINTS];
  double kp[R1RT_MAX_GROUP_JOINTS];
  double kd[R1RT_MAX_GROUP_JOINTS];
  double tau_ff[R1RT_MAX_GROUP_JOINTS];
  r1rt_knot knots[R1RT_MAX_KNOTS];      /* TRAJECTORY */
  r1rt_traj_guards guards;
} r1rt_upper_cmd;

/* One record per writer tick (ring "telemetry", r1rt -> r1log; DESIGN §9 "RT records"). float32 to halve the
 * volume (~1.2 kB x 500 Hz). */
typedef struct r1rt_telemetry_record {
  uint64_t seq;               /* writer tick */
  int64_t t_wake_ns;          /* monotonic wake-up */
  int64_t t_send_ns;          /* monotonic time of the write (0 = nothing written) */
  int64_t t_state_rx_ns;      /* receive time of the state used */
  uint32_t board_tick;
  int32_t wake_lateness_ns;
  int32_t iteration_ns;
  uint32_t faults;
  uint32_t clamp_count;       /* clamps this tick */
  r1rt_mode mode;
  r1rt_owner token_owner;
  uint8_t guard_event;        /* 0 none; else a guard id (DESIGN §3.3) */
  float q[R1RT_NUM_JOINTS];
  float dq[R1RT_NUM_JOINTS];
  float tau_est[R1RT_NUM_JOINTS];
  float q_cmd[R1RT_NUM_JOINTS];
  float dq_cmd[R1RT_NUM_JOINTS];
  float kp[R1RT_NUM_JOINTS];
  float kd[R1RT_NUM_JOINTS];
  float tau_ff[R1RT_NUM_JOINTS];
  float imu_quat[4];
  float imu_gyro[3];
  float imu_acc[3];
  int16_t temperature[R1RT_NUM_JOINTS][2];
  uint32_t motor_state[R1RT_NUM_JOINTS];
} r1rt_telemetry_record;

/* One policy step (ring "policy", 50 Hz). */
typedef struct r1rt_policy_record {
  uint64_t seq;
  int64_t t_obs_ns;
  int64_t t_action_ns;
  uint32_t n_obs;
  uint8_t inline_run;         /* 1 = run inline by the writer (rung 1) */
  uint8_t _pad0[3];
  float obs[R1RT_MAX_POLICY_OBS];
  float action[R1RT_NUM_POLICY_ACTIONS];
} r1rt_policy_record;

/* Session 5: the hands' joint positions and the declared payloads (r1hands -> r1rt; DESIGN §3.4: the arm model and
 * the fingertip FK with the real hand pose, the surface guard's input). Single writer: r1hands. */
typedef struct r1rt_hands_pose {
  uint64_t seq;
  int64_t t_ns;
  double q[R1RT_NUM_HANDS][7];   /* Dex3-1 joints per hand (thumb 0-2, middle 3-4, index 5-6), rad */
  double payload_kg[R1RT_NUM_HANDS]; /* the declared payload held by each hand (0 = none) */
  uint8_t valid;                 /* 0 = hands state stale / absent */
  uint8_t _pad0[7];
} r1rt_hands_pose;

/* Session 5: one r1guard cycle (ring "guard_telemetry", r1guard -> r1log; the guard's own full-rate record at its
 * 1 kHz cycle, standby and active). float32 like the core's record. */
typedef struct r1rt_guard_record {
  uint64_t seq;               /* guard cycle */
  int64_t t_ns;               /* the cycle's monotonic time */
  int64_t t_state_rx_ns;      /* receive time of the guard's own lowstate */
  int64_t rt_beat_age_ns;     /* the core's heartbeat age this cycle (-1 never seen) */
  uint64_t rt_beat_seq;
  uint32_t board_tick;
  uint32_t faults;            /* R1RT_FAULT_* of the guard this cycle */
  r1rt_mode mode;             /* GUARDED while active, MONITOR on standby */
  r1rt_mode mode_from;        /* the core's mode (mirror) */
  r1rt_mode act_mode;         /* what the guard publishes */
  r1rt_owner token_owner;
  uint8_t event;              /* GuardEvent (guard.hpp) of the cycle */
  uint8_t active;
  uint8_t published;          /* 1 = a command went out this cycle */
  uint8_t shadow_valid;
  uint8_t _pad0[5];
  float shadow_dev_rad;
  float tilt_rad;
  float q_cmd[R1RT_NUM_JOINTS];   /* the command (published or not; 0 on standby) */
  float kp[R1RT_NUM_JOINTS];
  float kd[R1RT_NUM_JOINTS];
} r1rt_guard_record;

/* ---- pinned sizes (a change here is a layout + ABI version bump) ---- */
#ifdef __cplusplus
#define R1RT_STATIC_ASSERT static_assert
#else
#define R1RT_STATIC_ASSERT _Static_assert
#endif
R1RT_STATIC_ASSERT(sizeof(double) == 8 && sizeof(float) == 4, "IEEE-754 sizes");
R1RT_STATIC_ASSERT(sizeof(r1rt_lowstate_in) == 1304, "r1rt_lowstate_in layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_state) == 1368, "r1rt_state layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_lowcmd_out) == 1080, "r1rt_lowcmd_out layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_heartbeat) == 40, "r1rt_heartbeat layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_mirror) == 1120, "r1rt_mirror layout changed");  /* layout 3: + tau_ff, stops */
R1RT_STATIC_ASSERT(sizeof(r1rt_base_cmd) == 80, "r1rt_base_cmd layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_hands_contact) == 32, "r1rt_hands_contact layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_knot) == 152, "r1rt_knot layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_traj_guards) == 24, "r1rt_traj_guards layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_upper_cmd) == 1528, "r1rt_upper_cmd layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_telemetry_record) == 1136, "r1rt_telemetry_record layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_policy_record) == 600, "r1rt_policy_record layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_hands_pose) == 152, "r1rt_hands_pose layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_guard_record) == 384, "r1rt_guard_record layout changed");
R1RT_STATIC_ASSERT(sizeof(r1rt_exec_status) == 184, "r1rt_exec_status layout changed");  /* layout 4 */

#ifdef __cplusplus
}
#endif

#endif /* R1RT_TYPES_H */
