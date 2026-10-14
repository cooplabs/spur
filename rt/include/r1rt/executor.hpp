// r1rt::UpperBodyExecutor - the upper-body executor of the real-time core (DESIGN.md §3.3).
//
// Per joint group (arm_left = joints 14-18, arm_right = 19-23, head = 24-25, SDK order) exactly one source at a time:
//
//   HOLD        hold the last target (or the measured pose on entry) with the group's hold gains + the record's tau_ff
//   TRAJECTORY  a time-parameterised path: quintic Hermite through r1rt_knot (q, dq, ddq), sampled at the caller's tick;
//               long paths come as consecutive records (segment_index 0, 1, ..., last_segment); r1d must renew a
//               keepalive every keepalive_ns (the record's ttl_ns, at most the config's) - on expiry the group
//               decelerates to a stop within stop_time_s and HOLDs (the deadman). Guards travel with the path.
//   STREAM      per-joint setpoints (q, dq, kp, kd, tau_ff) with a TTL <= stream_ttl_max_ns (300 ms for the arms);
//               expiry -> decelerate and HOLD.
//   FREEDRIVE   kp 0, a small kd, gravity feed-forward (hand-guiding); OPERATOR ONLY (a separate call, never a ring
//               record) and timed. It is refused until a gravity feed-forward is supplied every tick
//               (set_gravity_ff): without the built-in arm model (DESIGN §3.4, not built yet) a kp-0 arm would fall.
//
// Guards, evaluated every tick on a TRAJECTORY (their parameters travel with it, r1rt_traj_guards):
//   lag      v2's "arm_blocked" expected-lag test, ported (r1mc2 controller.py:2198-2261 at git r21_try 2a4a165;
//            V2_INVENTORY App. B §4.3): per joint, along its commanded motion only, (e - e0).sign(v) beyond
//            friction / kp + |v| latency + kd / kp |v - v_sent| + lag_rad, after a grace time, lasting persist_s;
//            e = last tick's commanded q - measured q, e0 = the hold lag averaged before the path began, of which a
//            head start AHEAD along the motion is ignored up to lag_head_start_rad (v2 guard_ahead_max, bounded).
//   torque   |tau_est - its mean over the first torque_baseline_s| > tau_nm on a CALIBRATED joint of the group,
//            lasting torque_persist_s (v2's finger-force guard pattern, controller.py:2145-2196). A torque guard on a
//            group without a calibrated joint is refused at submit (it could never trip).
//   contact  r1rt_hands_contact.pad_contact of guards.hand: any / all of contact_mask, counting only pads that ROSE
//            after the path began (a pad already in contact at the start counts once it has cleared), after
//            contact_grace_s, lasting contact_persist_s (v2 controller.py:2085-2134). Hands state missing / invalid /
//            older than hands_max_age_ns while a contact guard runs = a stop too (event CONTACT_STALE).
//   surface  the lowest fingertip below surface_z_m: needs the built-in fingertip FK (DESIGN §3.4), which does not
//            exist yet. The hook is here (surface_guard_hit), INACTIVE; a record that enables it is REFUSED at
//            submit, so nobody relies on a guard that cannot trip.
//   tilt     (arms, TRAJECTORY and STREAM) the body tilt rises arm_tilt_stop_rad (4 deg, v2 ARM_TILT_STOP_DEG,
//            controller.py:3410) over its value when the motion began.
// On a guard: decelerate within guard_stop_time_s (a few ms, bounded by ddq_stop), HOLD, record the event.
//
// Every stop (guard, deadman, TTL, operator stop, a starved trajectory) uses one smooth profile per joint: the
// velocity falls from v0 to 0 as v0 (1 - 3 s^2 + 2 s^3), s = t / T (continuous velocity AND acceleration at both
// ends; the peak deceleration 1.5 |v0| / T <= ddq_stop; travel v0 T / 2). All joints of a group stop together.
//
// Limits on every output: q within [q_min, q_max]; the target moves at most dq_max per second (a position jump in
// a HOLD / STREAM record is approached at dq_max, a path that would exceed it is held back); |dq| <= dq_max; gains
// within [0, kp_max] / [0, kd_max] (a record outside is refused). Each clamp is counted (clamp_count()).
//
// Real-time rules: init() is the only function that allocates (the knot buffers); submit(), keepalive(), stop(),
// set_speed_scale(), set_gravity_ff() and step() allocate nothing, throw nothing and run in constant time (checked by
// rt/tests/test_executor.cpp with a counting operator new). Not thread-safe: the writer thread owns it.
#pragma once

#include <cstdint>
#include <vector>

#include "r1rt/types.h"

namespace r1rt {

// Events per group (last_guard(), tick_event()): guards 1-15, other stop reasons 16+.
enum ExecEvent : std::uint8_t {
  kEventNone = 0,
  kGuardLag = 1,
  kGuardTorque = 2,
  kGuardContact = 3,
  kGuardSurface = 4,       // never raised (the hook is inactive, see above)
  kGuardTilt = 5,
  kGuardContactStale = 6,  // a contact guard without fresh hands state
  kEventDeadman = 16,      // TRAJECTORY keepalive expired
  kEventStreamTtl = 17,    // STREAM TTL expired
  kEventOperatorStop = 18, // stop()
  kEventStarved = 19,      // the trajectory clock reached the last received knot before last_segment
  kEventFreedriveEnd = 20, // FREEDRIVE timed out / gravity ff went stale
  kEventBadState = 21,     // a non-finite measured q / dq / tau_est / tilt while a guard needed it
  kEventMotorError = 22,   // session 6: a motor error word on one of the group's joints (DESIGN §4.2: HOLD that limb)
  kEventResume = 23,       // session 8 (HW1 fix #29): the writer resumed after a stop (> resume_gap_ns silent): the
                           // motion is aborted to HOLD ("aborted: resume after silence"); the board had let go
  kEventDone = 32,         // a TRAJECTORY reached its last knot (not a stop)
};

// Why submit() refused a record (last_reject()).
enum ExecReject : std::uint8_t {
  kRejectNone = 0,
  kRejectNotReady,
  kRejectGroup,
  kRejectSource,         // unknown source, or FREEDRIVE through the ring (operator only)
  kRejectNJoints,
  kRejectNonFinite,
  kRejectGains,
  kRejectStopEpoch,      // issued before the last operator stop
  kRejectTtl,            // STREAM ttl <= 0, > stream_ttl_max_ns, or already expired
  kRejectKnots,          // count, times not increasing, outside the limits, |dq| > dq_max
  kRejectSegment,        // a segment that does not continue the active trajectory (cmd_id / index / order)
  kRejectTooLong,        // more knots than max_traj_knots
  kRejectStart,          // the first knot is more than traj_start_tol_rad away from the current target
  kRejectTorqueGuard,    // torque guard on a group without a calibrated joint
  kRejectSurfaceGuard,   // surface guard requested (inactive until the fingertip FK exists)
  kRejectContactGuard,   // contact guard with an empty mask or a bad hand index
  kRejectNoGravityFF,    // FREEDRIVE without a fresh gravity feed-forward (the built-in arm model, DESIGN §3.4)
  kRejectGroupHeld,      // session 6: the group is held for a motor error until the operator clears it (resume)
};

struct ExecutorConfig {
  // Session 5: the group table. An executor runs three groups whose record ids are group_base .. group_base + 2: the
  // upper body (the default: arm_left 14-18, arm_right 19-23, head 24-25; the tilt stop on the arms) or, for ON_STAND,
  // the lower body (r1_lower_executor_config: leg_left 0-5, leg_right 6-11, waist 12-13; no tilt stop - the stand
  // holds the robot). Only the joints of its groups are used / written.
  std::uint8_t group_base = R1RT_GROUP_ARM_LEFT;
  std::int32_t group_first[R1RT_NUM_UPPER_GROUPS] = {14, 19, 24};
  std::int32_t group_n[R1RT_NUM_UPPER_GROUPS] = {5, 5, 2};
  std::uint8_t group_tilt_stop[R1RT_NUM_UPPER_GROUPS] = {1, 1, 0};
  // Per joint, SDK order; only the indices of the groups above are used.
  double hold_kp[R1RT_NUM_JOINTS] = {};
  double hold_kd[R1RT_NUM_JOINTS] = {};
  double q_min[R1RT_NUM_JOINTS] = {};
  double q_max[R1RT_NUM_JOINTS] = {};
  double dq_max[R1RT_NUM_JOINTS] = {};
  double ddq_stop[R1RT_NUM_JOINTS] = {};   // the largest deceleration a stop may use, rad/s^2
  double kp_max[R1RT_NUM_JOINTS] = {};
  double kd_max[R1RT_NUM_JOINTS] = {};
  double lag_friction_nm[R1RT_NUM_JOINTS] = {};  // static friction of the expected-lag model (v2 guard_friction_nm)
  std::uint8_t tau_calibrated[R1RT_NUM_JOINTS] = {};  // 1 = tau_est is calibrated (torque guards allowed)
  double stop_time_s = 0.15;               // deadman / TTL / operator stop
  double guard_stop_time_s = 0.02;         // a guard stop ("within a few ms"), still bounded by ddq_stop
  std::int64_t keepalive_ns = 50'000'000;  // TRAJECTORY deadman period (a record may ask for a shorter one)
  std::int64_t stream_ttl_max_ns = 300'000'000;
  std::int64_t tick_ns = 2'000'000;        // the writer period: bounds the per-tick motion after a loop gap
  // v2's lag guard (controller.py:325-350; the hardware-tuned values of 7-8 Oct)
  double lag_head_start_rad = 0.03;        // guard_ahead_max (the contract's "lag_head_start_s" is in rad: v2's unit)
  double lag_latency_s = 0.015;            // guard_latency_s
  double lag_grace_s = 0.2;                // guard_grace_s
  double lag_persist_s = 0.06;             // guard_persist_s
  double hold_lag_window_s = 0.1;          // the hold-lag baseline average (controller.py:1985-1998)
  double torque_baseline_s = 0.15;         // v2 finger guard: tau0 = mean over the first 0.15 s
  double torque_persist_s = 0.04;          // v2 guard_finger_persist_s
  double contact_grace_s = 0.15;           // v2: no guard hit in the first 0.15 s of a guarded path
  double contact_persist_s = 0.03;         // v2 guard_pad_persist_s
  std::int64_t hands_max_age_ns = 100'000'000;
  double arm_tilt_stop_rad = 0.06981317007977318;  // 4 deg (v2 ARM_TILT_STOP_DEG)
  double traj_start_tol_rad = 0.05;        // v2 _start_arm_path: within 0.05 rad of the path's start
  double speed_ramp_s = 0.3;               // pause / resume / speed changes ramp the clock rate (v2 arm_pause_s)
  double freedrive_kd_max = 1.0;           // "a small kd"
  std::int64_t freedrive_max_ns = 120'000'000'000;  // a FREEDRIVE lasts at most this long (timed, operator only)
  std::int64_t gravity_ff_max_age_ns = 10'000'000;  // FREEDRIVE needs a gravity ff at most this old
  std::uint32_t max_traj_knots = 512;      // per group (preallocated in init)
};

// The R1 defaults (v2 values with their sources; estimates marked in executor.cpp).
ExecutorConfig r1_default_executor_config();
// Session 5: the legs + waist executor of ON_STAND (the robot hanging on its stand; DESIGN §3.5): the same limits table
// (URDF - 0.05 rad), hold gains given by the caller (the profile's ON_STAND gains), dq_max 2 rad/s and ddq_stop
// 20 rad/s^2 (estimates as the arms'), the issue-#51 stiction as the lag model's friction, no tilt stop.
ExecutorConfig r1_lower_executor_config(const double hold_kp[R1RT_NUM_JOINTS], const double hold_kd[R1RT_NUM_JOINTS]);

class UpperBodyExecutor {
 public:
  static constexpr int kFirstJoint = 14;
  static constexpr int kNumGroups = R1RT_NUM_UPPER_GROUPS;

  UpperBodyExecutor() = default;

  // Validates and copies the config, allocates the knot buffers. R1RT_OK or R1RT_E_INVALID_ARG.
  std::int32_t init(const ExecutorConfig& cfg);
  bool ready() const noexcept { return ready_; }

  // Every group HOLDs the measured pose with its hold gains; clears sources, stops, guards, speed scales.
  void reset(const double q_meas[R1RT_NUM_JOINTS], std::int64_t t_ns) noexcept;

  // The record group ids this executor runs (cfg.group_base .. + kNumGroups - 1).
  bool owns_group(std::uint32_t group) const noexcept {
    return group >= cfg_.group_base && group < static_cast<std::uint32_t>(cfg_.group_base) + kNumGroups;
  }
  // One record from the upper_cmd ring. R1RT_OK or R1RT_E_INVALID_ARG (reason: last_reject()).
  std::int32_t submit(const r1rt_upper_cmd& c, std::int64_t t_ns) noexcept;
  // The TRAJECTORY deadman: the group(s) running trajectory cmd_id may run until t_ns + its keepalive period (a
  // renewal never shortens the deadline, so the core can pass r1d's last stamp every tick).
  void keepalive(std::uint64_t cmd_id, std::int64_t t_ns) noexcept;
  // Operator stop: every group decelerates to HOLD; records issued before it (stop_epoch) are refused afterwards.
  void stop(std::int64_t t_ns) noexcept;
  // The same, driven by the stop channel's epoch (shm StopChannel::epoch(), the sum of all stop / halt / damp
  // requests, which r1d stamps on its records): when it grew, every group stops and the executor's epoch becomes it.
  void stop_at_epoch(std::uint64_t epoch, std::int64_t t_ns) noexcept;
  // Session 6 (DESIGN §4.2 "arm / head motor error: HOLD that limb"): ONE group decelerates to HOLD (stiff, its hold
  // gains; a FREEDRIVE ends in a HOLD where it is) with `event`, and while `held` its new records are refused
  // (kRejectGroupHeld). set_group_held(g, false) only lifts the refusal (the group stays in HOLD).
  void stop_group(r1rt_group g, std::int64_t t_ns, std::uint8_t event) noexcept;
  void set_group_held(r1rt_group g, bool held) noexcept;
  bool group_held(r1rt_group g) const noexcept;
  std::uint64_t stop_epoch() const noexcept { return stop_epoch_; }
  // Trajectory clock rate of a group in [0, 1] (0 = pause; ramps over speed_ramp_s). Guards keep running.
  void set_speed_scale(r1rt_group g, double s, std::int64_t t_ns) noexcept;
  // The built-in arm model's gravity torque for the measured pose (joints 14-25), added to every output's tau_ff.
  void set_gravity_ff(const double tau[R1RT_NUM_JOINTS], std::int64_t t_ns) noexcept;
  // OPERATOR ONLY: hand-guiding for duration_ns (<= freedrive_max_ns) at kd (<= freedrive_kd_max), kp 0.
  // R1RT_E_INVALID_ARG without a fresh gravity feed-forward, or with bad arguments.
  std::int32_t freedrive(r1rt_group g, double kd, std::int64_t duration_ns, std::int64_t t_ns) noexcept;

  // One writer tick. Writes ONLY the indices of its groups (14-25 for the upper body). Before reset() the first call
  // resets at q.
  void step(std::int64_t t_ns, const double q[R1RT_NUM_JOINTS], const double dq[R1RT_NUM_JOINTS],
            const double tau_est[R1RT_NUM_JOINTS], const r1rt_hands_contact* hands, double tilt_rad,
            double q_out[R1RT_NUM_JOINTS], double dq_out[R1RT_NUM_JOINTS], double kp_out[R1RT_NUM_JOINTS],
            double kd_out[R1RT_NUM_JOINTS], double tau_ff_out[R1RT_NUM_JOINTS]) noexcept;

  r1rt_source source(r1rt_group g) const noexcept;
  bool stopping(r1rt_group g) const noexcept;          // decelerating after a stop / guard / expiry
  std::uint8_t last_guard(r1rt_group g) const noexcept; // the last ExecEvent of the group
  std::uint8_t tick_event() const noexcept { return tick_event_; }  // first event raised in the last step (telemetry)
  std::uint64_t active_cmd(r1rt_group g) const noexcept;
  std::uint32_t clamp_count() const noexcept { return clamp_count_; }  // since init
  std::uint32_t tick_clamps() const noexcept { return tick_clamps_; }  // in the last step
  std::uint8_t last_reject() const noexcept { return last_reject_; }
  double guard_margin(r1rt_group g) const noexcept;    // lag guard: the largest (lag - allowed) seen on the path

  // The SURFACE guard's hook (DESIGN §3.4): always false until the built-in fingertip FK exists.
  static bool surface_guard_hit(const r1rt_traj_guards& g, const double q[R1RT_NUM_JOINTS]) noexcept;

 private:
  enum class Mode : std::uint8_t { kHold, kTrajectory, kStream, kFreedrive, kStopping };

  struct Group {
    int first = 0;
    int n = 0;
    bool tilt_stop = false;  // the arms' tilt stop applies to this group
    Mode mode = Mode::kHold;
    r1rt_source src = R1RT_SOURCE_HOLD;  // reported source
    std::uint64_t cmd_id = 0;
    std::uint8_t event = kEventNone;
    bool held = false;  // session 6: records refused (a motor error) until the operator clears it
    // the commanded target (what was sent last tick)
    double q[R1RT_MAX_GROUP_JOINTS] = {};
    double v[R1RT_MAX_GROUP_JOINTS] = {};
    double kp[R1RT_MAX_GROUP_JOINTS] = {};
    double kd[R1RT_MAX_GROUP_JOINTS] = {};
    double tau_ff[R1RT_MAX_GROUP_JOINTS] = {};
    double gkp[R1RT_MAX_GROUP_JOINTS] = {};  // the gains kp / kd ramp to (over 0.1 s)
    double gkd[R1RT_MAX_GROUP_JOINTS] = {};
    // HOLD / STREAM setpoint the target moves to at dq_max
    double goal[R1RT_MAX_GROUP_JOINTS] = {};
    double goal_v[R1RT_MAX_GROUP_JOINTS] = {};
    std::int64_t expire_ns = 0;          // STREAM TTL / TRAJECTORY keepalive / FREEDRIVE end
    std::int64_t keepalive_period_ns = 0;
    // TRAJECTORY
    std::vector<r1rt_knot> knots;        // preallocated (max_traj_knots)
    std::uint32_t n_knots = 0;
    std::uint16_t next_segment = 0;
    bool complete = false;               // last_segment received
    std::uint32_t seg = 0;               // the knot interval in use
    std::int64_t t_start_ns = 0;
    double clock_s = 0.0;                // trajectory time
    double rate = 1.0, rate_from = 1.0, rate_to = 1.0;
    std::int64_t rate_t0_ns = 0;
    r1rt_traj_guards guards = {};
    // guard state
    double t_path_s = 0.0;               // real time since the path began (grace windows)
    double e0[R1RT_MAX_GROUP_JOINTS] = {};
    bool e0_set = false;
    double lag_since_s = -1.0;
    double margin = -1.0;
    double tau0[R1RT_MAX_GROUP_JOINTS] = {};
    double tau0_n = 0.0;
    double torque_since_s = -1.0;
    std::uint32_t pads_start = 0;
    std::uint32_t pads_armed = 0;
    bool pads_started = false;
    double contact_since_s = -1.0;
    std::int64_t hands_bad_since_ns = -1;
    double tilt0 = 0.0;
    bool tilt0_set = false;
    double v_sent[R1RT_MAX_GROUP_JOINTS] = {};
    // hold-lag baseline (while holding)
    double hold_lag[R1RT_MAX_GROUP_JOINTS] = {};
    double hold_lag_s = 0.0;
    // a stop in progress
    double stop_q0[R1RT_MAX_GROUP_JOINTS] = {};
    double stop_v0[R1RT_MAX_GROUP_JOINTS] = {};
    std::int64_t stop_t0_ns = 0;
    double stop_T = 0.0;                          // the group's stop time (the longest joint's)
    double stop_Tj[R1RT_MAX_GROUP_JOINTS] = {};   // per joint (shorter only where a limit is near)
    double stop_clock_s = 0.0;
    double vq[R1RT_MAX_GROUP_JOINTS] = {};        // how fast the target actually moved last tick
  };

  void enter_hold(Group& g, const double* q_target) noexcept;
  void begin_stop(Group& g, std::int64_t t_ns, double stop_time_s, std::uint8_t event) noexcept;
  void reset_guards(Group& g, double tilt_rad) noexcept;
  bool sample_trajectory(Group& g, double* q, double* v) noexcept;
  static void stop_sample(const Group& g, double* q, double* v) noexcept;
  std::uint8_t check_guards(Group& g, std::int64_t t_ns, double dt, const double* q, const double* dq,
                            const double* tau_est, const r1rt_hands_contact* hands, double tilt_rad) noexcept;
  void raise(Group& g, std::uint8_t ev) noexcept;
  std::int32_t reject(ExecReject r) noexcept;

  ExecutorConfig cfg_{};
  Group groups_[R1RT_NUM_UPPER_GROUPS];
  bool ready_ = false;
  bool primed_ = false;  // reset() happened
  std::int64_t t_prev_ns_ = 0;
  bool have_t_prev_ = false;
  std::uint64_t stop_epoch_ = 0;
  double gravity_ff_[R1RT_NUM_JOINTS] = {};
  std::int64_t gravity_ff_t_ns_ = 0;
  bool have_gravity_ff_ = false;
  std::uint32_t clamp_count_ = 0;
  std::uint32_t tick_clamps_ = 0;
  std::uint8_t tick_event_ = kEventNone;
  std::uint8_t last_reject_ = kRejectNone;
  int joints_[R1RT_NUM_JOINTS] = {};  // the joints of the groups (init), n_joints_ of them
  int n_joints_ = 0;
  Group* group_of(std::uint32_t id) noexcept { return owns_group(id) ? &groups_[id - cfg_.group_base] : nullptr; }
  const Group* group_of(std::uint32_t id) const noexcept {
    return owns_group(id) ? &groups_[id - cfg_.group_base] : nullptr;
  }
};

}  // namespace r1rt
