// r1rt::Guard - r1guard, the hot standby of the real-time core (DESIGN.md §2.1, §4.3 rung 2; GUARD.md).
//
// It shares the core's shared-memory segment (shm::Segment) and, every cycle (1 kHz):
//   1. takes its OWN lowstate (on the robot: its own DDS reader; in lockstep: the simulator's state),
//   2. writes its heartbeat (guard_heartbeat: the core and r1d see that a standby exists),
//   3. reads the core's heartbeat and mirror; while the core is alive it keeps a WARM shadow policy: at the policy rate
//      it restores its own PolicyRunner from the mirror (the core's last raw action, gait phase, lean, targets) and
//      computes the action it would publish now (a non-finite / out-of-range shadow is flagged in its heartbeat),
//   4. when the core's heartbeat is older than `stale_periods` writer periods (5 = 10 ms at 500 Hz) and the token is
//      the core's, takes the ownership token by CAS and publishes from the same cycle, by the mode the core was in
//      (DESIGN §4.3 table): BALANCE -> its policy with base command zero (the lean posture kept), arms / head HOLD at
//      the core's last targets and gains; STAND_FREEZE -> the ramp from the core's last command to the stand, then
//      hold; TAKEOVER_MIRROR / BLEND -> a blend from the core's last command into its policy; ON_STAND -> the core's
//      last command held; DAMP -> damp (the operator's state); MONITOR / GUARDED / OVERLAY_ARM_SDK -> nothing (no
//      rt/lowcmd of ours was being published, or not built). NEVER an automatic damp or zero torque.
//   5. records forensics at the takeover (the core's last heartbeat, mirror, token words, its own state; on Linux
//      capture_proc_forensics() adds /proc/<pid>/task/*/{comm,wchan,syscall,stack} - file I/O, called by the process
//      wrapper AFTER the cycle's write, never inside cycle()).
//
// The guard never hands the token back by itself (hand-back = an operator decision, new and unproven: DESIGN §4.4).
// Session 6: the operator's hand-back (R1RT_REQ_RECLAIM, ON_STAND only) is done by the CORE (CAS R1GUARD -> R1RT); the
// guard sees the token owned by r1rt again with a fresh core heartbeat, stops writing and RE-ARMS (standby, warm, a
// new takeover possible). While it holds the token it honours the OPERATOR's damp: a new stop-channel DAMP request
// whose requester is R1RT_REQUESTER_OPERATOR (baselined at attach and, at the takeover, at the core's acknowledged
// count: the core acted on the older ones), or a live core (not the owner) that entered DAMP - then it publishes damp
// (the operator's state; it stays damp). A non-operator damp request is counted and ignored.
// Its policy is the same PolicyRunner code as the core's (bumpless by construction: the same observation code, state
// and network); an invalid action of its own -> STAND_FREEZE (rung 3), never damp.
//
// Real-time rules: init() validates, loads the policy weights (the only file access) and may allocate; cycle() does
// not allocate, throw, block, or touch files (a counting operator new test pins it: rt/tests/test_guard.cpp).
#pragma once

#include <cstddef>
#include <cstdint>

#include "r1rt/policy_mlp.hpp"
#include "r1rt/policy_runner.hpp"
#include "r1rt/resume.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/state_age.hpp"
#include "r1rt/step_api.h"

namespace r1rt {

struct GuardParams {
  std::uint32_t cycle_rate_hz = 1000;                        // the heartbeat check rate (DESIGN §2.1: 1 kHz)
  std::uint32_t stale_periods = shm::kHeartbeatStalePeriods;  // the core's heartbeat older than this many periods
  double takeover_blend_s = 0.3;  // from TAKEOVER_* when the mirror carries no blend length (a layout-1 core):
                                  // the core's last command -> the guard's policy. Layout 2: the core's own blend
  std::uint8_t takeover_enabled = 1;  // 0 = observe only (heartbeat + shadow, never takes the token)
  std::uint8_t _pad[7] = {};
};

// What happened at the guard this cycle (status / telemetry; the first event of the cycle).
enum GuardEvent : std::uint8_t {
  kGuardEventNone = 0,
  kGuardEventTakeover = 1,        // token taken, publishing from this cycle
  kGuardEventCasFailed = 2,       // the heartbeat was stale but the token changed hands under us: no takeover
  kGuardEventTokenLost = 3,       // active, but the token is not ours any more (an operator took it): not writing
  kGuardEventShadowInvalid = 4,   // the shadow action is non-finite / out of range (standby: flagged only)
  kGuardEventFreeze = 5,          // active: an invalid action of its own -> STAND_FREEZE (rung 3)
  kGuardEventModeNotHandled = 6,  // stale core in a mode the guard does not publish for (MONITOR / OVERLAY / GUARDED)
  kGuardEventRearmed = 7,         // session 6: the operator's hand-back (the token r1rt's again, the core alive): standby
  kGuardEventOperatorDamp = 8,    // session 6: active, the operator's damp -> publishing damp
  kGuardEventDampIgnored = 9,     // session 6: a stop-channel damp request not from the operator (ignored)
};

struct GuardProcThread {
  std::int32_t tid;
  std::int32_t err;   // 0, or the errno of the first failed read (stack needs CAP_SYS_ADMIN: EACCES is normal)
  char comm[32];
  char wchan[64];
  char syscall[96];
  char stack[512];
};

// The takeover's evidence (DESIGN §9 "stall evidence"; to the flight recorder by the process wrapper / r1log).
struct GuardForensics {
  std::uint8_t captured = 0;           // 1 = filled at a takeover
  std::int64_t t_takeover_ns = 0;
  std::uint64_t rt_beat_seq = 0;       // the core's last heartbeat
  std::int64_t rt_beat_t_ns = 0;
  std::int64_t rt_silence_ns = 0;      // t_takeover - rt_beat_t: how long the core had been silent
  r1rt_mode rt_mode = R1RT_MODE_MONITOR;
  std::uint32_t rt_faults = 0;
  std::uint64_t rt_lowstate_tick = 0;
  std::uint32_t rt_rate_hz = 0;
  std::uint8_t rt_granted = 0;
  std::uint64_t mirror_seq = 0;        // the last mirror the guard holds
  std::int64_t mirror_t_ns = 0;
  std::int64_t mirror_last_action_t_ns = 0;
  std::uint8_t mirror_policy_valid = 0;
  r1rt_mode mode_from = R1RT_MODE_MONITOR;  // the mode the guard took over from (the mirror's)
  std::uint64_t token_before = 0;
  std::uint64_t token_after = 0;
  std::int32_t rt_pid = 0;             // the segment's creator (r1rt)
  std::int64_t guard_lowstate_age_ns = 0;
  double tilt_rad = 0.0;               // from the guard's own lowstate
  std::uint8_t shadow_valid = 0;       // the shadow at the moment of the takeover
  double shadow_dev_rad = 0.0;         // max |shadow target - the core's target| over the policy's action joints
  // capture_proc_forensics(): -1 not attempted, 0 ok, errno (ENOSYS off Linux)
  std::int32_t proc_err = -1;
  std::int32_t n_threads = 0;
  GuardProcThread threads[8] = {};
};

struct GuardStatus {
  std::uint64_t cycles;
  std::uint64_t publishes;            // cycles with a command published
  std::uint64_t shadow_steps;         // shadow policy steps (standby)
  std::uint64_t shadow_invalid;       // ... with an invalid output
  std::uint64_t policy_steps;         // policy steps while active
  std::uint64_t invalid_actions;      // ... with an invalid output (-> STAND_FREEZE)
  std::uint64_t takeovers;
  std::uint64_t cas_failures;
  std::uint64_t token_lost_cycles;
  std::uint64_t stale_resends;        // active, own lowstate stale: the last command resent
  std::uint64_t telemetry_drops;      // session 5: guard cycle records the ring could not take (r1log behind / absent)
  std::int64_t rt_beat_age_ns;        // the core's heartbeat age at the latest cycle (-1: never seen)
  std::int64_t lowstate_age_ns;       // the guard's own lowstate age (-1: none)
  std::int64_t t_takeover_ns;         // 0 = never
  double shadow_dev_rad;              // latest shadow deviation (see GuardForensics)
  double shadow_dev_max_rad;          // max over standby
  // The bumpless check: when the core publishes an action computed at the same instant as a shadow step, the guard
  // compares the core's targets with its shadow's (same code, state, inputs: equal to the network's rounding).
  std::uint64_t shadow_matches;       // shadow steps compared with the core's action of the same instant
  double shadow_match_err_rad;        // latest max |core target - shadow target| over the action joints
  double shadow_match_err_max_rad;
  double tilt_rad;
  std::uint32_t faults;               // R1RT_FAULT_* this cycle (LOWSTATE_*, POLICY_INVALID, NOT_OWNER, TILT_FALL)
  std::uint32_t faults_seen;
  r1rt_mode mode;                     // GUARDED while active, MONITOR on standby
  r1rt_mode mode_from;                // the core's mode the guard took over from
  r1rt_mode act_mode;                 // what the guard publishes: BALANCE / STAND_FREEZE / ON_STAND / DAMP / ...
  std::uint8_t active;
  std::uint8_t shadow_valid;
  std::uint8_t event;                 // GuardEvent of the latest cycle
  std::uint8_t policy_loaded;
  // session 6
  std::uint64_t rearms;               // hand-backs seen (the operator's RECLAIM through the core): back to standby
  std::uint64_t operator_damps;       // the operator's damp acted on while active
  std::uint64_t damp_ignored;         // stop-channel damp requests not from the operator (while active)
  std::uint8_t armed;                 // takeover enabled (0 = observe only)
  std::uint8_t _pad_s6[7];
  // session 8 (HW1 fix #29): the resume after a silence (the core's resume_gap_ns: the board let go after ~400 ms)
  std::uint64_t stops;                // its own cycle silences > resume_gap_ns while it held the token
  std::uint64_t resume_takeovers;     // takeovers of a core silent > resume_gap_ns (the guard was stalled too)
  std::uint64_t resume_blends;        // resume blends started from the measured pose (once its state was fresh)
  std::uint64_t resume_waits_expired; // ... no fresh state within R1RT_RESUME_STATE_WAIT_NS: the plain hold then
  std::uint64_t resume_uncaught;      // ... fresh by its stamp, the board tick never caught up: from the latest pose
  std::uint8_t resume_waiting;        // 1 = holding back its first command until fresh state
  std::uint8_t _pad_s8[7];
  // session 9 (#29's open race): the last resume wait, measured - after a whole-machine stop the DDS receive path drains
  // the lowstate samples queued during the stop (each stamped fresh, its board tick old) before a live one arrives; the
  // wait = how long a stale command that slipped out before the stop stays the board's command
  std::int64_t resume_wait_ns_last;            // begin_resume -> the blend's first command (or the wait's end)
  std::uint64_t resume_wait_samples_last;      // new lowstate samples seen during that wait
  std::uint64_t resume_wait_old_samples_last;  // ... of them not caught up with the board's tick (queued in the stop)
  std::uint64_t resume_drain_waits;  // waits that ran past R1RT_RESUME_STATE_WAIT_NS on a backlog still draining (hazard ii)
  // session 9 (U19 / D25): the state-age limit on its own lowstate (the core's rule, docs/sim/STATE_AGE.md)
  std::int64_t state_content_age_ns;  // its newest sample's board-tick content age (-1 = no valid tick clock)
  std::uint64_t state_old_episodes;   // episodes older than state_age_limit_ns for state_age_sustain_ns
  std::uint64_t stale_rung_entries;   // times its stale rung froze it into the stand (balancing after a takeover)
  std::uint8_t stale_freeze;          // 1 = its current STAND_FREEZE was entered by the stale rung
  std::uint8_t _pad_s9[7];
};

class Guard {
 public:
  Guard() noexcept = default;
  Guard(const Guard&) = delete;
  Guard& operator=(const Guard&) = delete;

  // cfg = the SAME profile the core runs (rates, limits, policy, stand pose, damp kd; cfg.segment = the core's
  // segment, already initialised). Loads the weights (cfg.policy_weights_path; required: a guard without a policy
  // cannot balance - R1RT_E_POLICY_LOAD). May allocate.
  int32_t init(const r1rt_config& cfg, const GuardParams& gp, char* err, std::size_t errlen) noexcept;

  // One guard cycle at t_ns (CLOCK_MONOTONIC, the core's clock): state = a new lowstate (nullptr: none this cycle).
  // `out` is always written; out->publish = 1 only while the guard owns the token and has something to send.
  int32_t cycle(const r1rt_lowstate_in* state, std::int64_t t_ns, r1rt_lowcmd_out* out) noexcept;

  // Fault injection on the guard's OWN network output (simulation / tests): R1RT_INJECT_POLICY_NAN /
  // R1RT_INJECT_POLICY_OUT_OF_RANGE in [t0, t1), both the shadow and the active policy.
  int32_t inject(std::int32_t kind, std::int64_t t0, std::int64_t t1, double value) noexcept;

  bool active() const noexcept { return active_; }
  std::uint64_t takeovers() const noexcept { return st_.takeovers; }  // the process wrapper: one forensics line each
  // session 5 (the process wrapper): its real-time grants, copied into every guard heartbeat (init: cfg.rt_granted)
  void set_rt_granted(bool granted) noexcept { rt_granted_ = granted; }
  const GuardForensics& forensics() const noexcept { return fx_; }
  GuardForensics& forensics_for_capture() noexcept { return fx_; }
  void status(GuardStatus* out) const noexcept { *out = st_; }

 private:
  struct Injection {
    std::int32_t kind;
    std::int64_t t0, t1;
    double value;
  };
  static constexpr int kMaxInjections = 8;
  static constexpr int kUpperFirst = 14;

  static void filter_thunk(void* ctx, double* action, int n) noexcept;
  bool fresh(std::int64_t t_ns) const noexcept;
  // false: the mirror holds no usable policy state. allow_blend (the takeover only): also the blend into BALANCE
  // (the standby shadow stays on BALANCE mirrors: during a blend the core's targets are not the policy's)
  bool runner_from_mirror(const r1rt_mirror& mi, bool allow_blend = false) noexcept;
  bool run_policy(std::int64_t t_ns, double* dev) noexcept;  // one act() on the guard's lowstate, cmd zero
  void take_over(std::int64_t t_ns, std::uint64_t token_word, const r1rt_heartbeat& hb) noexcept;
  void enter_freeze(std::int64_t t_ns, const double* q0, const double* kp0, const double* kd0) noexcept;
  void build(std::int64_t t_ns, r1rt_lowcmd_out* out) noexcept;
  void apply_limits(r1rt_lowcmd_out* out) noexcept;
  void write_heartbeat(std::int64_t t_ns) noexcept;
  void check_operator_damp(std::int64_t t_ns) noexcept;  // session 6: active only
  void rearm(std::int64_t t_ns) noexcept;                // session 6: the operator's hand-back seen
  void push_record(std::int64_t t_ns, const r1rt_lowcmd_out& out) noexcept;  // session 5: the guard telemetry ring

  r1rt_config cfg_{};
  GuardParams gp_{};
  shm::Segment* seg_ = nullptr;
  PolicyMlp net_;
  PolicyRunner runner_;
  bool policy_loaded_ = false;
  std::int64_t policy_period_ns_ = 20'000'000;
  std::int64_t writer_period_ns_ = 2'000'000;

  r1rt_lowstate_in latest_{};
  bool have_state_ = false;
  double tilt_ = 0.0;

  r1rt_heartbeat hb_{};        // the core's last heartbeat read
  bool have_hb_ = false;
  r1rt_mirror mirror_{};       // the core's last mirror read (standby)
  bool have_mirror_ = false;
  std::int64_t next_shadow_ns_ = 0;
  std::int64_t shadow_t_ns_ = -1;  // the latest valid shadow step (for the bumpless check), -1 = none pending
  double shadow_target_[R1RT_NUM_POLICY_ACTIONS] = {};

  bool active_ = false;
  std::uint64_t token_word_ = 0;
  std::uint64_t damp_seen_ = 0;  // session 6: the stop channel's DAMP-level request count already accounted for
  r1rt_mode act_mode_ = R1RT_MODE_MONITOR;
  std::int64_t next_policy_ns_ = 0;
  bool policy_started_ = false;
  // the command the guard takes over from (the core's last, from the mirror) and the arms / head HOLD
  double from_q_[R1RT_NUM_JOINTS] = {};
  double from_kp_[R1RT_NUM_JOINTS] = {};
  double from_kd_[R1RT_NUM_JOINTS] = {};
  double from_tau_[R1RT_NUM_JOINTS] = {};  // session 8 (fix #28): the core's last feed-forward (the mirror, layout 3)
  std::int64_t blend_t0_ = 0;
  double blend_s_ = 0.0;  // 0 = no blend
  // session 6: a sequence INTO ON_STAND taken over in MIRROR / BLEND: the core's gain ramp continued to the ON_STAND
  // hold gains (legs + waist: the config's on_stand gains, all 0 = the stand gains), the targets held - never a policy
  bool stand_blend_ = false;
  double hold_kp_[R1RT_NUM_JOINTS] = {};
  double hold_kd_[R1RT_NUM_JOINTS] = {};
  std::int64_t freeze_t0_ = 0;
  double freeze_ramp_s_ = 0.4;   // this freeze's ramp length (the rest of the core's ramp when continuing it)
  bool freeze_invalid_ = false;  // STAND_FREEZE because no valid policy (vs continuing the core's freeze)
  double freeze_q0_[R1RT_NUM_JOINTS] = {};
  double freeze_kp0_[R1RT_NUM_JOINTS] = {};
  double freeze_kd0_[R1RT_NUM_JOINTS] = {};
  r1rt_lowcmd_out last_cmd_{};
  bool have_cmd_ = false;

  Injection inj_[kMaxInjections] = {};
  int n_inj_ = 0;
  std::int64_t now_ns_ = 0;
  std::uint64_t hb_seq_ = 0;

  GuardForensics fx_{};
  GuardStatus st_{};
  bool rt_granted_ = false;

  // session 8 (HW1 fix #29): the same resume rule as the core (core.cpp begin_resume): a silence > resume_gap_ns -
  // its own while active, or the core's at a takeover (= the guard was stalled with it) - re-enters through a blend
  // from the pose measured after it (ON_STAND every joint; arms + head otherwise; STAND_FREEZE restarts its ramp)
  void begin_resume(std::int64_t t_ns, const double* cmd_q) noexcept;
  double pre_stop_q_[R1RT_NUM_JOINTS] = {};  // the pose measured before this cycle's sample (a stop's start)
  double resume_off_[R1RT_NUM_JOINTS] = {};  // the pre-stop hold offset (the command - pre_stop_q_), bounded
  std::int64_t last_cycle_ns_ = 0;
  bool have_cycle_ = false;
  bool resume_wait_ = false;
  bool resume_active_ = false;
  bool resume_drain_counted_ = false;  // session 9: this wait already counted in resume_drain_waits
  std::int64_t resume_wait_t0_ = 0;
  std::int64_t resume_t0_ = 0;
  double resume_s_ = 1.0;
  double resume_q0_[R1RT_NUM_JOINTS] = {};
  ResumeClock rclock_{};             // its own lowstate live after a stop (resume.hpp)
  std::uint32_t prev_tick_ = 0;      // the sample before this cycle's (a stop's start)
  std::int64_t prev_rx_ = 0;
  bool had_state_ = false;
  // session 9 (U19 / D25): the state-age limit (the core's rule; no automatic return in the guard - as its rung 3)
  ContentAgeClock age_clock_{};
  std::int64_t old_since_ns_ = 0;
  bool old_counted_ = false;
  bool stale_act_ = false;   // this cycle: balancing on state old for the sustain time, the rung enabled
  bool stale_freeze_ = false;
};

// Linux: reads /proc/<pid>/task/*/{comm, wchan, syscall, stack} (up to 8 threads) into f. File I/O: call it OUTSIDE
// cycle() (the process wrapper, after the takeover's first write). Returns 0 or an errno; ENOSYS off Linux.
std::int32_t capture_proc_forensics(std::int32_t pid, GuardForensics* f) noexcept;

}  // namespace r1rt
