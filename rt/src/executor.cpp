// r1rt::UpperBodyExecutor - see rt/include/r1rt/executor.hpp and docs/architecture/EXECUTOR.md.
#include "r1rt/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "r1rt/step_api.h"

namespace r1rt {

namespace {

constexpr double kNsToS = 1e-9;
constexpr double kGainRampS = 0.1;  // gains move to a new target over 0.1 s (bumpless HOLD <-> source changes)
constexpr std::uint32_t kAllPads = (1u << 9) - 1u;  // Dex3: 9 pads

bool finite(double x) noexcept { return std::isfinite(x); }

double clampd(double x, double lo, double hi) noexcept { return x < lo ? lo : (x > hi ? hi : x); }

double smoothstep(double s) noexcept {
  s = clampd(s, 0.0, 1.0);
  return s * s * (3.0 - 2.0 * s);
}

double sgn(double x) noexcept { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }

}  // namespace

ExecutorConfig r1_default_executor_config() {
  // Sources: r1mc2 (v2) at git r21_try 2a4a165 - joints.py (URDF limits, the 0.05 rad soft margin, head yaw +-1.5,
  // KP_STIFF / KD_STIFF), arms.py:34-35 (KP_ARM 60 60 45 45 30, KD_ARM 2 2 2 2 1.5: the arm gains under the
  // manipulation policy, used on hardware 5-9 Oct), controller.py:325-350 (the lag guard). ESTIMATES (until HW1):
  // dq_max 2 rad/s (v2 planned arm paths at <= 1 rad/s, arm_joint_vmax), ddq_stop 20 rad/s^2 (a 1 rad/s stop in
  // 75 ms), kd_max 2x the largest kd v2 used, the head's lag friction. tau_calibrated all 0: body tau_est is not
  // calibrated (types.h), so torque guards are refused until it is.
  ExecutorConfig c;
  const double qmin[R1RT_NUM_JOINTS] = {-2.9322, -1.0472, -2.7402, -0.1745, -0.8727, -0.2618, -2.9322, -1.7453, -2.7402,
                                        -0.1745, -0.8727, -0.2618, -0.5236, -2.618,  -3.1416, -0.2269, -1.9199, -0.9756,
                                        -1.9199, -3.1416, -2.4785, -1.9199, -0.9756, -1.9199, -0.6283, -2.0071};
  const double qmax[R1RT_NUM_JOINTS] = {2.5482, 1.7453, 2.7402, 2.4260, 0.5760, 0.2618, 2.5482, 1.0472, 2.7402,
                                        2.4260, 0.5760, 0.2618, 0.5236, 2.618,  2.0944, 2.4784, 1.9199, 2.1852,
                                        1.9199, 2.0944, 0.2268, 1.9199, 2.1852, 1.9199, 0.6283, 2.0071};
  const double arm_kp[5] = {60.0, 60.0, 45.0, 45.0, 30.0};
  const double arm_kd[5] = {2.0, 2.0, 2.0, 2.0, 1.5};
  const double arm_kp_max[5] = {100.0, 100.0, 100.0, 100.0, 50.0};  // KP_STIFF arms
  const double arm_fric[5] = {2.5, 2.5, 0.2, 0.2, 0.2};             // guard_friction_nm
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    c.q_min[j] = qmin[j] + 0.05;
    c.q_max[j] = qmax[j] - 0.05;
    c.dq_max[j] = 2.0;
    c.ddq_stop[j] = 20.0;
  }
  c.q_min[25] = -1.5;  // head yaw (cable safety, joints.py SOFT_MIN / SOFT_MAX)
  c.q_max[25] = 1.5;
  for (int side = 0; side < 2; ++side) {
    for (int k = 0; k < 5; ++k) {
      const int j = 14 + 5 * side + k;
      c.hold_kp[j] = arm_kp[k];
      c.hold_kd[j] = arm_kd[k];
      c.kp_max[j] = arm_kp_max[k];
      c.kd_max[j] = 4.0;
      c.lag_friction_nm[j] = arm_fric[k];
    }
  }
  c.hold_kp[24] = 50.0;  // head: KP_STIFF / KD_STIFF (the core passes the policy metadata's head gains instead)
  c.hold_kd[24] = 2.0;
  c.hold_kp[25] = 10.0;
  c.hold_kd[25] = 0.1;
  c.kp_max[24] = 50.0;
  c.kp_max[25] = 10.0;
  c.kd_max[24] = 4.0;
  c.kd_max[25] = 0.2;
  c.lag_friction_nm[24] = 0.2;
  c.lag_friction_nm[25] = 0.2;
  return c;
}

ExecutorConfig r1_lower_executor_config(const double hold_kp[R1RT_NUM_JOINTS], const double hold_kd[R1RT_NUM_JOINTS]) {
  ExecutorConfig c = r1_default_executor_config();  // the limits table (URDF - 0.05), dq_max 2, ddq_stop 20
  c.group_base = R1RT_GROUP_LEG_LEFT;
  const std::int32_t first[3] = {0, 6, 12};
  const std::int32_t n[3] = {6, 6, 2};
  for (int g = 0; g < 3; ++g) {
    c.group_first[g] = first[g];
    c.group_n[g] = n[g];
    c.group_tilt_stop[g] = 0;  // the stand holds the robot: the arms' tilt stop does not apply to the legs
  }
  for (int j = 0; j < 14; ++j) {
    c.hold_kp[j] = hold_kp[j];
    c.hold_kd[j] = hold_kd[j];
    // gains a record may ask for: up to 2x the hold gains (the core's kp_max / kd_max still clamp the output)
    c.kp_max[j] = 2.0 * hold_kp[j];
    c.kd_max[j] = 2.0 * hold_kd[j] > 1.0 ? 2.0 * hold_kd[j] : 1.0;
    // the lag model's static friction: unitree_rl_mjlab issue #51's stiction (2.5 N m hips / knees / waist, 1.5
    // ankles), the same estimate the simulator's nominal legs carry (HW1's torque ramp measures it)
    const bool ankle = j == 4 || j == 5 || j == 10 || j == 11;
    c.lag_friction_nm[j] = ankle ? 1.5 : 2.5;
  }
  return c;
}

std::int32_t UpperBodyExecutor::init(const ExecutorConfig& cfg) {
  ready_ = false;
  n_joints_ = 0;
  bool seen[R1RT_NUM_JOINTS] = {};
  if (cfg.group_base + kNumGroups > R1RT_NUM_CMD_GROUPS) return R1RT_E_INVALID_ARG;
  for (int gi = 0; gi < kNumGroups; ++gi) {
    const int f = cfg.group_first[gi];
    const int n = cfg.group_n[gi];
    if (f < 0 || n <= 0 || n > R1RT_MAX_GROUP_JOINTS || f + n > R1RT_NUM_JOINTS) return R1RT_E_INVALID_ARG;
    for (int j = f; j < f + n; ++j) {
      if (seen[j]) return R1RT_E_INVALID_ARG;  // groups do not overlap
      seen[j] = true;
      joints_[n_joints_++] = j;
    }
  }
  for (int ji = 0; ji < n_joints_; ++ji) {
    const int j = joints_[ji];
    const double v[] = {cfg.hold_kp[j], cfg.hold_kd[j], cfg.q_min[j],  cfg.q_max[j],  cfg.dq_max[j],
                        cfg.ddq_stop[j], cfg.kp_max[j], cfg.kd_max[j], cfg.lag_friction_nm[j]};
    for (double x : v) {
      if (!finite(x)) return R1RT_E_INVALID_ARG;
    }
    if (!(cfg.q_min[j] < cfg.q_max[j]) || cfg.dq_max[j] <= 0.0 || cfg.ddq_stop[j] <= 0.0) return R1RT_E_INVALID_ARG;
    // a HOLD is never limp: kp > 0 on every joint of the groups
    if (cfg.hold_kp[j] <= 0.0 || cfg.hold_kp[j] > cfg.kp_max[j]) return R1RT_E_INVALID_ARG;
    if (cfg.hold_kd[j] < 0.0 || cfg.hold_kd[j] > cfg.kd_max[j] || cfg.lag_friction_nm[j] < 0.0) return R1RT_E_INVALID_ARG;
  }
  const double pos[] = {cfg.stop_time_s,       cfg.guard_stop_time_s, cfg.lag_latency_s,  cfg.lag_persist_s,
                        cfg.hold_lag_window_s, cfg.torque_baseline_s, cfg.torque_persist_s, cfg.contact_persist_s,
                        cfg.arm_tilt_stop_rad, cfg.traj_start_tol_rad, cfg.speed_ramp_s};
  for (double x : pos) {
    if (!finite(x) || x <= 0.0) return R1RT_E_INVALID_ARG;
  }
  const double nonneg[] = {cfg.lag_head_start_rad, cfg.lag_grace_s, cfg.contact_grace_s, cfg.freedrive_kd_max};
  for (double x : nonneg) {
    if (!finite(x) || x < 0.0) return R1RT_E_INVALID_ARG;
  }
  if (cfg.keepalive_ns <= 0 || cfg.stream_ttl_max_ns <= 0 || cfg.tick_ns <= 0 || cfg.hands_max_age_ns <= 0 ||
      cfg.freedrive_max_ns <= 0 || cfg.gravity_ff_max_age_ns <= 0)
    return R1RT_E_INVALID_ARG;
  if (cfg.max_traj_knots < 2u * R1RT_MAX_KNOTS || cfg.max_traj_knots > 65535u) return R1RT_E_INVALID_ARG;
  cfg_ = cfg;
  for (int gi = 0; gi < kNumGroups; ++gi) {
    Group& g = groups_[gi];
    g = Group{};
    g.first = cfg.group_first[gi];
    g.n = cfg.group_n[gi];
    g.tilt_stop = cfg.group_tilt_stop[gi] != 0;
    g.knots.assign(cfg.max_traj_knots, r1rt_knot{});  // the only allocation
  }
  primed_ = false;
  have_t_prev_ = false;
  have_gravity_ff_ = false;
  stop_epoch_ = 0;
  clamp_count_ = 0;
  tick_clamps_ = 0;
  tick_event_ = kEventNone;
  last_reject_ = kRejectNone;
  ready_ = true;
  return R1RT_OK;
}

void UpperBodyExecutor::reset(const double q_meas[R1RT_NUM_JOINTS], std::int64_t t_ns) noexcept {
  if (!ready_) return;
  for (Group& g : groups_) {
    for (int k = 0; k < g.n; ++k) {
      const int j = g.first + k;
      const double fallback = primed_ ? g.q[k] : clampd(0.0, cfg_.q_min[j], cfg_.q_max[j]);
      g.q[k] = finite(q_meas[j]) ? clampd(q_meas[j], cfg_.q_min[j], cfg_.q_max[j]) : fallback;
      g.v[k] = 0.0;
      g.kp[k] = cfg_.hold_kp[j];
      g.kd[k] = cfg_.hold_kd[j];
      g.gkp[k] = cfg_.hold_kp[j];
      g.gkd[k] = cfg_.hold_kd[j];
      g.tau_ff[k] = 0.0;
      g.goal[k] = g.q[k];
      g.goal_v[k] = 0.0;
      g.v_sent[k] = 0.0;
      g.vq[k] = 0.0;
      g.hold_lag[k] = 0.0;
    }
    g.mode = Mode::kHold;
    g.src = R1RT_SOURCE_HOLD;
    g.cmd_id = 0;
    g.event = kEventNone;
    g.n_knots = 0;
    g.complete = false;
    g.rate = g.rate_from = g.rate_to = 1.0;
    g.rate_t0_ns = t_ns;
    g.hold_lag_s = 0.0;
    g.margin = -1.0;
  }
  t_prev_ns_ = t_ns;
  have_t_prev_ = true;
  primed_ = true;
}

std::int32_t UpperBodyExecutor::reject(ExecReject r) noexcept {
  last_reject_ = r;
  return R1RT_E_INVALID_ARG;
}

void UpperBodyExecutor::raise(Group& g, std::uint8_t ev) noexcept {
  g.event = ev;
  if (tick_event_ == kEventNone) tick_event_ = ev;
}

void UpperBodyExecutor::reset_guards(Group& g, double tilt_rad) noexcept {
  g.t_path_s = 0.0;
  g.lag_since_s = -1.0;
  g.margin = -1.0;
  g.tau0_n = 0.0;
  g.torque_since_s = -1.0;
  g.pads_start = 0;
  g.pads_armed = 0;
  g.pads_started = false;
  g.contact_since_s = -1.0;
  g.tilt0 = tilt_rad;
  g.tilt0_set = finite(tilt_rad);
  for (int k = 0; k < g.n; ++k) g.tau0[k] = 0.0;
}

void UpperBodyExecutor::enter_hold(Group& g, const double* q_target) noexcept {
  g.mode = Mode::kHold;
  g.src = R1RT_SOURCE_HOLD;
  for (int k = 0; k < g.n; ++k) {
    const int j = g.first + k;
    g.q[k] = q_target[k];
    g.goal[k] = q_target[k];
    g.goal_v[k] = 0.0;
    g.v[k] = 0.0;
    g.gkp[k] = cfg_.hold_kp[j];
    g.gkd[k] = cfg_.hold_kd[j];
  }
  g.hold_lag_s = 0.0;
}

void UpperBodyExecutor::begin_stop(Group& g, std::int64_t t_ns, double stop_time_s, std::uint8_t event) noexcept {
  raise(g, event);
  // v0 = how fast the TARGET was moving (not a stream's dq feed-forward, which need not match its setpoints), so the
  // position stays continuous in velocity through the stop
  double T = stop_time_s;
  bool moving = false;
  for (int k = 0; k < g.n; ++k) {
    const int j = g.first + k;
    g.stop_q0[k] = g.q[k];
    g.stop_v0[k] = finite(g.vq[k]) ? g.vq[k] : 0.0;
    if (g.stop_v0[k] != 0.0) moving = true;
    // peak deceleration of the profile = 1.5 |v0| / T <= ddq_stop
    T = std::max(T, 1.5 * std::fabs(g.stop_v0[k]) / cfg_.ddq_stop[j]);
  }
  if (!moving) {
    double q[R1RT_MAX_GROUP_JOINTS];
    for (int k = 0; k < g.n; ++k) q[k] = g.q[k];
    enter_hold(g, q);
    return;
  }
  for (int k = 0; k < g.n; ++k) {
    // all joints stop together (T), except one that would run past its limit (travel v0 T / 2): it stops sooner,
    // as soon as its deceleration bound allows; if even that is too far, the position clamp is the last word
    const int j = g.first + k;
    const double v0 = std::fabs(g.stop_v0[k]);
    double Tj = T;
    if (v0 > 0.0) {
      const double room = g.stop_v0[k] > 0.0 ? cfg_.q_max[j] - g.q[k] : g.q[k] - cfg_.q_min[j];
      const double t_room = 2.0 * std::max(room, 0.0) / v0;
      if (t_room < Tj) Tj = std::max(t_room, 1.5 * v0 / cfg_.ddq_stop[j]);
    }
    g.stop_Tj[k] = std::max(Tj, 1e-6);
  }
  g.mode = Mode::kStopping;
  g.src = R1RT_SOURCE_HOLD;  // the group no longer follows its source
  g.stop_t0_ns = t_ns;
  g.stop_T = T;
  g.stop_clock_s = 0.0;
  for (int k = 0; k < g.n; ++k) {  // hold gains at the end of the stop (ramped)
    g.gkp[k] = cfg_.hold_kp[g.first + k];
    g.gkd[k] = cfg_.hold_kd[g.first + k];
  }
}

std::int32_t UpperBodyExecutor::submit(const r1rt_upper_cmd& c, std::int64_t t_ns) noexcept {
  if (!ready_ || !primed_) return reject(kRejectNotReady);
  if (!owns_group(c.group)) return reject(kRejectGroup);
  Group& g = groups_[c.group - cfg_.group_base];
  if (g.held) return reject(kRejectGroupHeld);  // session 6: a motor error holds the group until the operator clears it
  if (c.n_joints != g.n) return reject(kRejectNJoints);
  if (c.stop_epoch < stop_epoch_) return reject(kRejectStopEpoch);
  if (c.source != R1RT_SOURCE_HOLD && c.source != R1RT_SOURCE_TRAJECTORY && c.source != R1RT_SOURCE_STREAM)
    return reject(kRejectSource);  // FREEDRIVE is operator-only (freedrive()), never a ring record

  // gains + feed-forward (HOLD / STREAM / TRAJECTORY): per joint either both 0 (= the hold gains) or kp in
  // (0, kp_max] and kd in [0, kd_max]: only FREEDRIVE may have kp 0
  double kp[R1RT_MAX_GROUP_JOINTS], kd[R1RT_MAX_GROUP_JOINTS];
  for (int k = 0; k < g.n; ++k) {
    const int j = g.first + k;
    if (!finite(c.kp[k]) || !finite(c.kd[k]) || !finite(c.tau_ff[k])) return reject(kRejectNonFinite);
    if (c.kp[k] == 0.0 && c.kd[k] == 0.0) {
      kp[k] = cfg_.hold_kp[j];
      kd[k] = cfg_.hold_kd[j];
    } else {
      if (!(c.kp[k] > 0.0) || c.kp[k] > cfg_.kp_max[j] || c.kd[k] < 0.0 || c.kd[k] > cfg_.kd_max[j])
        return reject(kRejectGains);
      kp[k] = c.kp[k];
      kd[k] = c.kd[k];
    }
  }

  if (c.source == R1RT_SOURCE_HOLD) {
    for (int k = 0; k < g.n; ++k) {
      if (!finite(c.q[k])) return reject(kRejectNonFinite);
    }
    const bool was_hold = g.mode == Mode::kHold;
    g.mode = Mode::kHold;
    g.src = R1RT_SOURCE_HOLD;
    g.cmd_id = c.cmd_id;
    for (int k = 0; k < g.n; ++k) {
      const int j = g.first + k;
      const double goal = clampd(c.q[k], cfg_.q_min[j], cfg_.q_max[j]);
      if (goal != c.q[k]) ++clamp_count_;
      g.goal[k] = goal;
      g.goal_v[k] = 0.0;
      g.gkp[k] = kp[k];
      g.gkd[k] = kd[k];
      g.tau_ff[k] = c.tau_ff[k];
    }
    if (!was_hold) g.hold_lag_s = 0.0;
    last_reject_ = kRejectNone;
    return R1RT_OK;
  }

  if (c.source == R1RT_SOURCE_STREAM) {
    for (int k = 0; k < g.n; ++k) {
      if (!finite(c.q[k]) || !finite(c.dq[k])) return reject(kRejectNonFinite);
    }
    if (c.ttl_ns <= 0 || c.ttl_ns > cfg_.stream_ttl_max_ns || c.t_issue_ns + c.ttl_ns <= t_ns) return reject(kRejectTtl);
    if (g.mode != Mode::kStream || g.cmd_id != c.cmd_id) reset_guards(g, std::nan(""));
    g.mode = Mode::kStream;
    g.src = R1RT_SOURCE_STREAM;
    g.cmd_id = c.cmd_id;
    g.expire_ns = c.t_issue_ns + c.ttl_ns;
    for (int k = 0; k < g.n; ++k) {
      const int j = g.first + k;
      const double goal = clampd(c.q[k], cfg_.q_min[j], cfg_.q_max[j]);
      if (goal != c.q[k]) ++clamp_count_;
      g.goal[k] = goal;
      g.goal_v[k] = c.dq[k];
      g.gkp[k] = kp[k];
      g.gkd[k] = kd[k];
      g.tau_ff[k] = c.tau_ff[k];
    }
    last_reject_ = kRejectNone;
    return R1RT_OK;
  }

  // ---- TRAJECTORY
  if (c.n_knots < 1 || c.n_knots > R1RT_MAX_KNOTS) return reject(kRejectKnots);
  for (int i = 0; i < c.n_knots; ++i) {
    const r1rt_knot& kn = c.knots[i];
    if (kn.t_ns < 0) return reject(kRejectKnots);
    for (int k = 0; k < g.n; ++k) {
      if (!finite(kn.q[k]) || !finite(kn.dq[k]) || !finite(kn.ddq[k])) return reject(kRejectNonFinite);
      const int j = g.first + k;
      if (kn.q[k] < cfg_.q_min[j] || kn.q[k] > cfg_.q_max[j] || std::fabs(kn.dq[k]) > cfg_.dq_max[j])
        return reject(kRejectKnots);
    }
    if (i > 0 && kn.t_ns <= c.knots[i - 1].t_ns) return reject(kRejectKnots);
  }
  const bool continuation = c.segment_index > 0;
  if (continuation) {
    if (g.mode != Mode::kTrajectory || g.cmd_id != c.cmd_id || g.complete || c.segment_index != g.next_segment)
      return reject(kRejectSegment);
    if (c.knots[0].t_ns <= g.knots[g.n_knots - 1].t_ns) return reject(kRejectSegment);
    if (g.n_knots + c.n_knots > cfg_.max_traj_knots) return reject(kRejectTooLong);
    std::memcpy(&g.knots[g.n_knots], c.knots, sizeof(r1rt_knot) * c.n_knots);
    g.n_knots += c.n_knots;
    g.next_segment = static_cast<std::uint16_t>(g.next_segment + 1u);
    g.complete = c.last_segment != 0;
    g.expire_ns = t_ns + g.keepalive_period_ns;  // a segment renews the deadman too
    last_reject_ = kRejectNone;
    return R1RT_OK;
  }
  if (c.knots[0].t_ns != 0) return reject(kRejectKnots);
  if (c.n_knots < 2 && c.last_segment) return reject(kRejectKnots);
  for (int k = 0; k < g.n; ++k) {
    if (std::fabs(c.knots[0].q[k] - g.q[k]) > cfg_.traj_start_tol_rad) return reject(kRejectStart);
  }
  // guards
  const r1rt_traj_guards& gd = c.guards;
  if (gd.enable & 0x8u) return reject(kRejectSurfaceGuard);  // needs the fingertip FK (DESIGN §3.4): not built
  if (gd.enable & 0x1u) {
    if (!std::isfinite(gd.lag_rad) || gd.lag_rad <= 0.0f) return reject(kRejectGains);
  }
  if (gd.enable & 0x2u) {
    bool any_cal = false;
    for (int k = 0; k < g.n; ++k) any_cal = any_cal || cfg_.tau_calibrated[g.first + k] != 0;
    if (!any_cal || !std::isfinite(gd.tau_nm) || gd.tau_nm <= 0.0f) return reject(kRejectTorqueGuard);
  }
  if (gd.enable & 0x4u) {
    if (gd.hand >= R1RT_NUM_HANDS || (gd.contact_mask & kAllPads) == 0 || (gd.contact_mask & ~kAllPads) != 0)
      return reject(kRejectContactGuard);
  }
  const bool was_hold = g.mode == Mode::kHold;
  g.mode = Mode::kTrajectory;
  g.src = R1RT_SOURCE_TRAJECTORY;
  g.cmd_id = c.cmd_id;
  std::memcpy(&g.knots[0], c.knots, sizeof(r1rt_knot) * c.n_knots);
  g.n_knots = c.n_knots;
  g.next_segment = 1;
  g.complete = c.last_segment != 0;
  g.seg = 0;
  g.t_start_ns = c.t_start_ns > 0 ? c.t_start_ns : t_ns;
  g.clock_s = 0.0;
  g.keepalive_period_ns = c.ttl_ns > 0 ? std::min(c.ttl_ns, cfg_.keepalive_ns) : cfg_.keepalive_ns;
  g.expire_ns = t_ns + g.keepalive_period_ns;
  g.guards = gd;
  for (int k = 0; k < g.n; ++k) {
    g.gkp[k] = kp[k];
    g.gkd[k] = kd[k];
    g.tau_ff[k] = c.tau_ff[k];
  }
  reset_guards(g, std::nan(""));
  // the lag baseline: the hold lag averaged before the path (fresh: >= 0.05 s of holding, v2), else the first tick's
  g.e0_set = was_hold && g.hold_lag_s >= 0.05;
  for (int k = 0; k < g.n; ++k) g.e0[k] = g.e0_set ? g.hold_lag[k] : 0.0;
  g.hold_lag_s = 0.0;
  last_reject_ = kRejectNone;
  return R1RT_OK;
}

void UpperBodyExecutor::keepalive(std::uint64_t cmd_id, std::int64_t t_ns) noexcept {
  if (!ready_) return;
  for (Group& g : groups_) {
    // never shortens the deadline: the core may pass r1d's last keepalive stamp every tick (UpperKeepalive)
    if (g.mode == Mode::kTrajectory && g.cmd_id == cmd_id) g.expire_ns = std::max(g.expire_ns, t_ns + g.keepalive_period_ns);
  }
}

void UpperBodyExecutor::stop_at_epoch(std::uint64_t epoch, std::int64_t t_ns) noexcept {
  if (!ready_ || epoch <= stop_epoch_) return;
  stop(t_ns);
  stop_epoch_ = epoch;
}

void UpperBodyExecutor::stop(std::int64_t t_ns) noexcept {
  if (!ready_) return;
  ++stop_epoch_;
  if (!primed_) return;
  for (Group& g : groups_) {
    if (g.mode == Mode::kStopping) {
      raise(g, kEventOperatorStop);
      continue;
    }
    if (g.mode == Mode::kFreedrive) {
      double q[R1RT_MAX_GROUP_JOINTS];
      for (int k = 0; k < g.n; ++k) q[k] = g.q[k];
      raise(g, kEventOperatorStop);
      enter_hold(g, q);
      continue;
    }
    begin_stop(g, t_ns, cfg_.stop_time_s, kEventOperatorStop);  // HOLD at rest -> stays HOLD at its target
  }
}

void UpperBodyExecutor::stop_group(r1rt_group gi, std::int64_t t_ns, std::uint8_t event) noexcept {
  if (!ready_ || !primed_ || !owns_group(gi)) return;
  Group& g = groups_[gi - cfg_.group_base];
  if (g.mode == Mode::kStopping) {
    raise(g, event);
    return;
  }
  if (g.mode == Mode::kFreedrive) {
    double q[R1RT_MAX_GROUP_JOINTS];
    for (int k = 0; k < g.n; ++k) q[k] = g.q[k];
    raise(g, event);
    enter_hold(g, q);
    return;
  }
  begin_stop(g, t_ns, cfg_.stop_time_s, event);  // decelerate to HOLD at the hold gains (a HOLD stays where it is)
}

void UpperBodyExecutor::set_group_held(r1rt_group gi, bool held) noexcept {
  if (!ready_ || !owns_group(gi)) return;
  groups_[gi - cfg_.group_base].held = held;
}

bool UpperBodyExecutor::group_held(r1rt_group gi) const noexcept {
  return ready_ && owns_group(gi) && groups_[gi - cfg_.group_base].held;
}

void UpperBodyExecutor::set_speed_scale(r1rt_group gi, double s, std::int64_t t_ns) noexcept {
  if (!ready_ || !owns_group(gi) || !finite(s)) return;
  Group& g = groups_[gi - cfg_.group_base];
  const double u = (static_cast<double>(t_ns - g.rate_t0_ns) * kNsToS) / cfg_.speed_ramp_s;
  g.rate_from = g.rate_from + (g.rate_to - g.rate_from) * smoothstep(u);
  g.rate_to = clampd(s, 0.0, 1.0);
  g.rate_t0_ns = t_ns;
}

void UpperBodyExecutor::set_gravity_ff(const double tau[R1RT_NUM_JOINTS], std::int64_t t_ns) noexcept {
  if (!ready_) return;
  for (int ji = 0; ji < n_joints_; ++ji) {
    if (!finite(tau[joints_[ji]])) {
      have_gravity_ff_ = false;  // a bad model output is no feed-forward at all
      return;
    }
  }
  for (int ji = 0; ji < n_joints_; ++ji) gravity_ff_[joints_[ji]] = tau[joints_[ji]];
  gravity_ff_t_ns_ = t_ns;
  have_gravity_ff_ = true;
}

std::int32_t UpperBodyExecutor::freedrive(r1rt_group gi, double kd, std::int64_t duration_ns, std::int64_t t_ns) noexcept {
  if (!ready_ || !primed_) return reject(kRejectNotReady);
  if (!owns_group(gi)) return reject(kRejectGroup);
  if (!finite(kd) || kd < 0.0 || kd > cfg_.freedrive_kd_max || duration_ns <= 0 || duration_ns > cfg_.freedrive_max_ns)
    return reject(kRejectGains);
  if (!have_gravity_ff_ || t_ns - gravity_ff_t_ns_ > cfg_.gravity_ff_max_age_ns) return reject(kRejectNoGravityFF);
  Group& g = groups_[gi - cfg_.group_base];
  g.mode = Mode::kFreedrive;
  g.src = R1RT_SOURCE_FREEDRIVE;
  g.cmd_id = 0;
  g.expire_ns = t_ns + duration_ns;
  for (int k = 0; k < g.n; ++k) {
    g.gkp[k] = 0.0;
    g.gkd[k] = kd;
    g.tau_ff[k] = 0.0;  // gravity only
    g.v[k] = 0.0;
  }
  last_reject_ = kRejectNone;
  return R1RT_OK;
}

void UpperBodyExecutor::stop_sample(const Group& g, double* q, double* v) noexcept {
  // velocity v0 (1 - 3 s^2 + 2 s^3), s = t / T_j: continuous velocity and acceleration at both ends
  for (int k = 0; k < g.n; ++k) {
    const double s = clampd(g.stop_clock_s / g.stop_Tj[k], 0.0, 1.0);
    const double pos = s - s * s * s + 0.5 * s * s * s * s;  // its integral
    const double vel = 1.0 - 3.0 * s * s + 2.0 * s * s * s;
    q[k] = g.stop_q0[k] + g.stop_v0[k] * g.stop_Tj[k] * pos;
    v[k] = g.stop_v0[k] * vel;
  }
}

bool UpperBodyExecutor::surface_guard_hit(const r1rt_traj_guards&, const double*) noexcept {
  // INACTIVE: the lowest fingertip's z needs the built-in fingertip forward kinematics (DESIGN §3.4, generated from
  // the simulator's MJCF; not built yet). submit() refuses a record that enables this guard.
  return false;
}

bool UpperBodyExecutor::sample_trajectory(Group& g, double* q, double* v) noexcept {
  // returns false when the clock has passed the last received knot (done or starved)
  const double tau = g.clock_s;
  const r1rt_knot* K = g.knots.data();
  const std::uint32_t n = g.n_knots;
  const double t_last = static_cast<double>(K[n - 1].t_ns) * kNsToS;
  if (tau >= t_last) {
    for (int k = 0; k < g.n; ++k) {
      q[k] = K[n - 1].q[k];
      v[k] = K[n - 1].dq[k];
    }
    return false;
  }
  while (g.seg + 1 < n && static_cast<double>(K[g.seg + 1].t_ns) * kNsToS <= tau) ++g.seg;
  const r1rt_knot& a = K[g.seg];
  const r1rt_knot& b = K[g.seg + 1];
  const double t0 = static_cast<double>(a.t_ns) * kNsToS;
  const double h = static_cast<double>(b.t_ns - a.t_ns) * kNsToS;
  const double s = clampd((tau - t0) / h, 0.0, 1.0);
  for (int k = 0; k < g.n; ++k) {
    // quintic Hermite: position, velocity and acceleration match both knots
    const double p0 = a.q[k], v0 = a.dq[k] * h, a0 = a.ddq[k] * h * h;
    const double p1 = b.q[k], v1 = b.dq[k] * h, a1 = b.ddq[k] * h * h;
    const double d = p1 - p0;
    const double c3 = 10.0 * d - 6.0 * v0 - 4.0 * v1 - 1.5 * a0 + 0.5 * a1;
    const double c4 = -15.0 * d + 8.0 * v0 + 7.0 * v1 + 1.5 * a0 - a1;
    const double c5 = 6.0 * d - 3.0 * v0 - 3.0 * v1 - 0.5 * a0 + 0.5 * a1;
    q[k] = p0 + s * (v0 + s * (0.5 * a0 + s * (c3 + s * (c4 + s * c5))));
    const double ds = v0 + s * (a0 + s * (3.0 * c3 + s * (4.0 * c4 + s * 5.0 * c5)));
    v[k] = ds / h;
  }
  return true;
}

std::uint8_t UpperBodyExecutor::check_guards(Group& g, std::int64_t t_ns, double dt, const double* q, const double* dq,
                                             const double* tau_est, const r1rt_hands_contact* hands,
                                             double tilt_rad) noexcept {
  const bool is_arm = g.tilt_stop;  // the arms (v2's tilt stop); session 5: a group-table flag
  const bool traj = g.mode == Mode::kTrajectory;
  g.t_path_s += dt;
  if (is_arm) {  // tilt stop: arms, TRAJECTORY and STREAM (v2 ARM_TILT_STOP_DEG)
    if (!finite(tilt_rad)) return kEventBadState;
    if (!g.tilt0_set) {
      g.tilt0 = tilt_rad;
      g.tilt0_set = true;
    }
    if (tilt_rad - g.tilt0 > cfg_.arm_tilt_stop_rad) return kGuardTilt;
  }
  if (!traj) return kEventNone;
  const r1rt_traj_guards& gd = g.guards;
  if (gd.enable & 0x1u) {  // lag (v2 controller.py:2198-2261)
    double e[R1RT_MAX_GROUP_JOINTS];
    for (int k = 0; k < g.n; ++k) {
      const int j = g.first + k;
      if (!finite(q[j]) || !finite(dq[j])) return kEventBadState;
      e[k] = g.q[k] - q[j];  // last tick's command - measured
    }
    if (!g.e0_set) {
      for (int k = 0; k < g.n; ++k) g.e0[k] = e[k];
      g.e0_set = true;
    }
    if (g.t_path_s >= cfg_.lag_grace_s) {
      double best = -1e300;
      bool best_moving = false;
      for (int k = 0; k < g.n; ++k) {
        const int j = g.first + k;
        const double kp = std::max(g.kp[k], 1.0);
        const double fric = cfg_.lag_friction_nm[j] / kp;
        const double v = g.v[k];
        const double d = std::fabs(v) > 0.01 ? sgn(v) : 0.0;
        const double head = std::max(-g.e0[k] * d, 0.0);
        const double along = e[k] * d - std::max(g.e0[k] * d, 0.0) + std::max(head - cfg_.lag_head_start_rad, 0.0);
        const double expect = fric + std::fabs(v) * cfg_.lag_latency_s + g.kd[k] / kp * std::fabs(v - g.v_sent[k]);
        const double over = d != 0.0 ? along - expect - static_cast<double>(gd.lag_rad) : -1.0;
        if (over > best) {
          best = over;
          best_moving = d != 0.0;
        }
      }
      if (best_moving) g.margin = std::max(g.margin, best);
      if (best <= 0.0) {
        g.lag_since_s = -1.0;
      } else {
        if (g.lag_since_s < 0.0) g.lag_since_s = g.t_path_s;
        if (g.t_path_s - g.lag_since_s >= cfg_.lag_persist_s - 1e-12) return kGuardLag;
      }
    }
  }
  if (gd.enable & 0x2u) {  // torque on calibrated joints (v2's finger-force guard pattern)
    double dev = 0.0;
    for (int k = 0; k < g.n; ++k) {
      const int j = g.first + k;
      if (!cfg_.tau_calibrated[j]) continue;
      if (!finite(tau_est[j])) return kEventBadState;
      if (g.t_path_s <= cfg_.torque_baseline_s) {
        g.tau0[k] += (tau_est[j] - g.tau0[k]) / (g.tau0_n + 1.0);
      } else {
        dev = std::max(dev, std::fabs(tau_est[j] - g.tau0[k]));
      }
    }
    if (g.t_path_s <= cfg_.torque_baseline_s) {
      g.tau0_n += 1.0;
    } else if (dev > static_cast<double>(gd.tau_nm)) {
      if (g.torque_since_s < 0.0) g.torque_since_s = g.t_path_s;
      if (g.t_path_s - g.torque_since_s >= cfg_.torque_persist_s - 1e-12) return kGuardTorque;
    } else {
      g.torque_since_s = -1.0;
    }
  }
  if (gd.enable & 0x4u) {  // contact (v2 controller.py:2085-2134: a RISE, persisting, not in the first 0.15 s)
    const bool stale = hands == nullptr || hands->valid == 0 || t_ns - hands->t_ns > cfg_.hands_max_age_ns;
    if (stale) return kGuardContactStale;
    const std::uint32_t mask = gd.contact_mask & kAllPads;
    const std::uint32_t pads = hands->pad_contact[gd.hand] & mask;
    if (!g.pads_started) {
      g.pads_start = pads;
      g.pads_armed = mask & ~pads;
      g.pads_started = true;
    }
    g.pads_armed |= mask & ~pads;  // a pad that was in contact at the start counts once it has cleared
    const std::uint32_t counted = pads & g.pads_armed;
    const bool hit = gd.contact_all ? counted == mask : counted != 0;
    if (hit && g.t_path_s >= cfg_.contact_grace_s) {
      if (g.contact_since_s < 0.0) g.contact_since_s = g.t_path_s;
      if (g.t_path_s - g.contact_since_s >= cfg_.contact_persist_s - 1e-12) return kGuardContact;
    } else {
      g.contact_since_s = -1.0;
    }
  }
  if ((gd.enable & 0x8u) && surface_guard_hit(gd, q)) return kGuardSurface;
  return kEventNone;
}

void UpperBodyExecutor::step(std::int64_t t_ns, const double q[R1RT_NUM_JOINTS], const double dq[R1RT_NUM_JOINTS],
                             const double tau_est[R1RT_NUM_JOINTS], const r1rt_hands_contact* hands, double tilt_rad,
                             double q_out[R1RT_NUM_JOINTS], double dq_out[R1RT_NUM_JOINTS],
                             double kp_out[R1RT_NUM_JOINTS], double kd_out[R1RT_NUM_JOINTS],
                             double tau_ff_out[R1RT_NUM_JOINTS]) noexcept {
  if (!ready_) return;
  if (!primed_) reset(q, t_ns);
  tick_event_ = kEventNone;
  tick_clamps_ = 0;
  double dt = have_t_prev_ ? static_cast<double>(t_ns - t_prev_ns_) * kNsToS : 0.0;
  if (!(dt > 0.0)) dt = 0.0;
  // after a loop gap the targets move as if at most two writer periods had passed (no jump on the next tick)
  dt = std::min(dt, 2.0 * static_cast<double>(cfg_.tick_ns) * kNsToS);
  t_prev_ns_ = t_ns;
  have_t_prev_ = true;
  const bool gravity_fresh = have_gravity_ff_ && t_ns - gravity_ff_t_ns_ <= cfg_.gravity_ff_max_age_ns;

  for (Group& g : groups_) {
    double qn[R1RT_MAX_GROUP_JOINTS], vn[R1RT_MAX_GROUP_JOINTS];
    for (int k = 0; k < g.n; ++k) {
      qn[k] = g.q[k];
      vn[k] = g.v[k];
    }

    // ---- expiries and guards (they may turn the group into a stop)
    if (g.mode == Mode::kTrajectory && t_ns >= g.expire_ns) {
      begin_stop(g, t_ns, cfg_.stop_time_s, kEventDeadman);
    } else if (g.mode == Mode::kStream && t_ns >= g.expire_ns) {
      begin_stop(g, t_ns, cfg_.stop_time_s, kEventStreamTtl);
    } else if (g.mode == Mode::kFreedrive && (t_ns >= g.expire_ns || !gravity_fresh)) {
      for (int k = 0; k < g.n; ++k) qn[k] = finite(q[g.first + k]) ? q[g.first + k] : g.q[k];
      raise(g, kEventFreedriveEnd);
      enter_hold(g, qn);
    }
    if (g.mode == Mode::kTrajectory || g.mode == Mode::kStream) {
      const std::uint8_t ev = check_guards(g, t_ns, dt, q, dq, tau_est, hands, tilt_rad);
      if (ev != kEventNone) begin_stop(g, t_ns, cfg_.guard_stop_time_s, ev);
    }

    // ---- the next target of the source in force
    switch (g.mode) {
      case Mode::kTrajectory: {
        if (t_ns >= g.t_start_ns) {
          const double u = (static_cast<double>(t_ns - g.rate_t0_ns) * kNsToS) / cfg_.speed_ramp_s;
          g.rate = g.rate_from + (g.rate_to - g.rate_from) * smoothstep(u);
          // only the time since the start counts (session 3 `wire`: the whole tick was added on the start tick,
          // so the target ran one writer period ahead of the knots)
          const double since_start = static_cast<double>(t_ns - g.t_start_ns) * kNsToS;
          g.clock_s += std::min(dt, since_start) * g.rate;
        }
        double qs[R1RT_MAX_GROUP_JOINTS], vs[R1RT_MAX_GROUP_JOINTS];
        const bool inside = sample_trajectory(g, qs, vs);
        const double rate = t_ns >= g.t_start_ns ? g.rate : 0.0;
        for (int k = 0; k < g.n; ++k) {
          qn[k] = qs[k];
          vn[k] = vs[k] * rate;
        }
        if (!inside) {
          if (g.complete && [&] {
                for (int k = 0; k < g.n; ++k)
                  if (std::fabs(vn[k]) > 1e-9) return false;
                return true;
              }()) {
            // done at rest: HOLD the last knot (the target reaches it through the limits below)
            raise(g, kEventDone);
            g.mode = Mode::kHold;
            g.src = R1RT_SOURCE_HOLD;
            for (int k = 0; k < g.n; ++k) {
              g.goal[k] = qn[k];
              g.goal_v[k] = 0.0;
              g.gkp[k] = cfg_.hold_kp[g.first + k];
              g.gkd[k] = cfg_.hold_kd[g.first + k];
              vn[k] = 0.0;
            }
            g.hold_lag_s = 0.0;
          } else {
            // the clock passed the last knot while moving: the last segment never came (starved), or a path that
            // ends in motion - decelerate from where the target is
            begin_stop(g, t_ns, cfg_.stop_time_s, g.complete ? kEventDone : kEventStarved);
            if (g.mode == Mode::kStopping) {
              g.stop_clock_s += dt;
              stop_sample(g, qn, vn);
            } else {
              for (int k = 0; k < g.n; ++k) {
                qn[k] = g.q[k];
                vn[k] = 0.0;
              }
            }
          }
        }
        break;
      }
      case Mode::kStopping: {
        g.stop_clock_s += dt;
        stop_sample(g, qn, vn);
        if (g.stop_clock_s >= g.stop_T) {
          for (int k = 0; k < g.n; ++k) {
            const int j = g.first + k;
            qn[k] = clampd(qn[k], cfg_.q_min[j], cfg_.q_max[j]);
          }
          enter_hold(g, qn);
          for (int k = 0; k < g.n; ++k) vn[k] = 0.0;
        }
        break;
      }
      case Mode::kStream: {
        for (int k = 0; k < g.n; ++k) {
          qn[k] = g.goal[k];
          vn[k] = g.goal_v[k];
        }
        break;
      }
      case Mode::kHold: {
        // approach the goal (a HOLD record's target) with |v| <= dq_max and |dv| <= ddq_stop dt per tick, braking in
        // discrete time (the largest speed from which steps of a dt less each still stop at the goal); at rest still
        for (int k = 0; k < g.n; ++k) {
          const int j = g.first + k;
          const double err = g.goal[k] - g.q[k];
          if (err == 0.0 && g.v[k] == 0.0) {
            qn[k] = g.q[k];
            vn[k] = 0.0;
            continue;
          }
          const double a = cfg_.ddq_stop[j];
          const double adt = a * dt;
          if (dt <= 0.0) {
            qn[k] = g.q[k];
            vn[k] = g.v[k];
            continue;
          }
          if (std::fabs(err) <= adt * dt && std::fabs(g.v[k]) <= adt) {  // one small last step
            qn[k] = g.goal[k];
            vn[k] = 0.0;
            continue;
          }
          const double v_brake = adt * (std::sqrt(0.25 + 2.0 * std::fabs(err) / (adt * dt)) - 0.5);
          const double v_des = sgn(err) * std::min(cfg_.dq_max[j], v_brake);
          const double v = g.v[k] + clampd(v_des - g.v[k], -adt, adt);
          qn[k] = g.q[k] + v * dt;
          vn[k] = v;
        }
        // the hold-lag baseline for the next guarded path (v2 _track_hold_lag)
        bool ok = true;
        for (int k = 0; k < g.n; ++k) ok = ok && finite(q[g.first + k]);
        if (ok && dt > 0.0) {
          const double a = std::min(1.0, dt / cfg_.hold_lag_window_s);
          for (int k = 0; k < g.n; ++k) {
            const double e = g.q[k] - q[g.first + k];
            g.hold_lag[k] = g.hold_lag_s == 0.0 ? e : g.hold_lag[k] + a * (e - g.hold_lag[k]);
          }
          g.hold_lag_s += dt;
        }
        break;
      }
      case Mode::kFreedrive: {
        for (int k = 0; k < g.n; ++k) {
          const int j = g.first + k;
          qn[k] = finite(q[j]) ? clampd(q[j], cfg_.q_min[j], cfg_.q_max[j]) : g.q[k];
          vn[k] = 0.0;
        }
        break;
      }
    }

    // ---- limits: velocity, the per-tick motion of the target, position; gains ramp to their targets
    for (int k = 0; k < g.n; ++k) {
      const int j = g.first + k;
      if (!finite(qn[k])) qn[k] = g.q[k];
      if (!finite(vn[k])) vn[k] = 0.0;
      if (vn[k] > cfg_.dq_max[j] || vn[k] < -cfg_.dq_max[j]) {
        vn[k] = clampd(vn[k], -cfg_.dq_max[j], cfg_.dq_max[j]);
        ++tick_clamps_;
      }
      if (g.mode != Mode::kFreedrive) {
        const double room = cfg_.dq_max[j] * dt * (1.0 + 1e-9) + 1e-12;
        const double d = qn[k] - g.q[k];
        if (d > room || d < -room) {
          qn[k] = g.q[k] + clampd(d, -room, room);
          ++tick_clamps_;
        }
      }
      if (qn[k] < cfg_.q_min[j] || qn[k] > cfg_.q_max[j]) {
        qn[k] = clampd(qn[k], cfg_.q_min[j], cfg_.q_max[j]);
        if ((qn[k] <= cfg_.q_min[j] && vn[k] < 0.0) || (qn[k] >= cfg_.q_max[j] && vn[k] > 0.0)) vn[k] = 0.0;
        ++tick_clamps_;
      }
      g.vq[k] = dt > 0.0 ? (qn[k] - g.q[k]) / dt : 0.0;
      if (g.mode == Mode::kHold && g.goal[k] == qn[k] && vn[k] == 0.0) g.vq[k] = 0.0;  // arrived: at rest
      g.q[k] = qn[k];
      g.v[k] = vn[k];
      g.v_sent[k] = vn[k];
      const double rkp = std::max(g.gkp[k], cfg_.hold_kp[j]) * dt / kGainRampS;
      const double rkd = std::max(g.gkd[k], cfg_.hold_kd[j]) * dt / kGainRampS + 1e-12;
      g.kp[k] += clampd(g.gkp[k] - g.kp[k], -rkp, rkp);
      g.kd[k] += clampd(g.gkd[k] - g.kd[k], -rkd, rkd);

      q_out[j] = g.q[k];
      dq_out[j] = g.v[k];
      kp_out[j] = g.kp[k];
      kd_out[j] = g.kd[k];
      tau_ff_out[j] = g.tau_ff[k] + (gravity_fresh ? gravity_ff_[j] : 0.0);
    }
  }
  clamp_count_ += tick_clamps_;
}

r1rt_source UpperBodyExecutor::source(r1rt_group g) const noexcept {
  const Group* p = group_of(g);
  return p != nullptr ? p->src : static_cast<r1rt_source>(R1RT_SOURCE_HOLD);  // GCC: -Werror=extra
}

bool UpperBodyExecutor::stopping(r1rt_group g) const noexcept {
  const Group* p = group_of(g);
  return p != nullptr && p->mode == Mode::kStopping;
}

std::uint8_t UpperBodyExecutor::last_guard(r1rt_group g) const noexcept {
  const Group* p = group_of(g);
  return p != nullptr ? p->event : static_cast<std::uint8_t>(kEventNone);  // GCC: -Werror=extra
}

std::uint64_t UpperBodyExecutor::active_cmd(r1rt_group g) const noexcept {
  const Group* p = group_of(g);
  return p != nullptr ? p->cmd_id : 0;
}

double UpperBodyExecutor::guard_margin(r1rt_group g) const noexcept {
  const Group* p = group_of(g);
  return p != nullptr ? p->margin : -1.0;
}

}  // namespace r1rt
