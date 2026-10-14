// r1rt::PolicyRunner (policy_runner.hpp). Ported from r1mc3.policy.reference.ReferencePolicy (itself a port of
// r1mc2 (v2) r1mc2/policy.py WalkPolicy at git r21_try 2a4a165); changes: C++, fixed-size arrays, an invalid network
// output is not used (see the header). Statement order follows the reference line by line (compiled with
// -ffp-contract=off) so the results match it to the network's rounding.
#include "r1rt/policy_runner.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace r1rt {

namespace {

constexpr double kPi = 3.14159265358979323846;  // == numpy.pi

int term_size(int term, const r1rt_policy_params& p) noexcept {
  switch (term) {
    case R1RT_OBS_BASE_ANG_VEL:
    case R1RT_OBS_PROJECTED_GRAVITY:
    case R1RT_OBS_COMMAND:
      return 3;
    case R1RT_OBS_LEAN:
      return 1;
    case R1RT_OBS_WAIST_CMD:
    case R1RT_OBS_PHASE:
      return 2;
    case R1RT_OBS_JOINT_POS:
    case R1RT_OBS_JOINT_VEL:
      return static_cast<int>(p.n_joints);
    case R1RT_OBS_ACTIONS:
      return static_cast<int>(p.n_act);
    default:
      return -1;
  }
}

double clip(double x, double lo, double hi) noexcept {
  // numpy.clip(x, lo, hi) = minimum(maximum(x, lo), hi)
  const double y = x < lo ? lo : x;
  return y > hi ? hi : y;
}

// numpy.linalg.norm of a short vector: sqrt(dot(x, x)); only compared against thresholds.
double norm2(double a, double b) noexcept { return std::sqrt(a * a + b * b); }
double norm3(double a, double b, double c) noexcept { return std::sqrt(a * a + b * b + c * c); }

bool fail(char* err, std::size_t errlen, const char* msg) noexcept {
  if (err != nullptr && errlen > 0) std::snprintf(err, errlen, "%s", msg);
  return false;
}

}  // namespace

void projected_gravity(const double quat[4], double g[3]) noexcept {
  // mju_quat2Mat (MuJoCo 3.5.0 engine_util_spatial.c), then R^T (0, 0, -1) = -(R[6], R[7], R[8]).
  if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    g[0] = 0.0;
    g[1] = 0.0;
    g[2] = -1.0;
    return;
  }
  const double q00 = quat[0] * quat[0];
  const double q01 = quat[0] * quat[1];
  const double q02 = quat[0] * quat[2];
  const double q11 = quat[1] * quat[1];
  const double q13 = quat[1] * quat[3];
  const double q22 = quat[2] * quat[2];
  const double q23 = quat[2] * quat[3];
  const double q33 = quat[3] * quat[3];
  const double r6 = 2 * (q13 - q02);
  const double r7 = 2 * (q23 + q01);
  const double r8 = q00 - q11 - q22 + q33;
  g[0] = -r6;
  g[1] = -r7;
  g[2] = -r8;
}

double quat_cos_tilt(const double quat[4]) noexcept {
  const double g8 = quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] + quat[3] * quat[3];
  return g8;
}

bool PolicyRunner::init(const r1rt_policy_params& params, const PolicyMlp* net, char* err,
                        std::size_t errlen) noexcept {
  net_ = nullptr;
  if (net == nullptr || !net->loaded()) return fail(err, errlen, "PolicyRunner: no loaded network");
  const r1rt_policy_params& p = params;
  if (p.n_joints == 0 || p.n_joints > kMaxJoints) return fail(err, errlen, "PolicyRunner: n_joints out of range");
  if (p.n_act == 0 || p.n_act > kMaxAct || static_cast<int>(p.n_act) != PolicyMlp::kAct)
    return fail(err, errlen, "PolicyRunner: n_act does not match the network");
  if (p.n_terms == 0 || p.n_terms > R1RT_POLICY_MAX_TERMS) return fail(err, errlen, "PolicyRunner: n_terms");
  int n = 0;
  for (std::uint32_t k = 0; k < p.n_terms; ++k) {
    const int s = term_size(p.obs_terms[k], p);
    if (s < 0) return fail(err, errlen, "PolicyRunner: unknown observation term");
    if (p.obs_terms[k] == R1RT_OBS_LEAN && !p.has_lean) return fail(err, errlen, "PolicyRunner: lean term without lean");
    if (p.obs_terms[k] == R1RT_OBS_WAIST_CMD && !p.has_waist)
      return fail(err, errlen, "PolicyRunner: waist term without waist");
    n += s;
  }
  if (n != PolicyMlp::kObs || n > kMaxObs) return fail(err, errlen, "PolicyRunner: observation length != network input");
  for (std::uint32_t i = 0; i < p.n_joints; ++i)
    if (p.joint_index[i] < 0 || p.joint_index[i] >= R1RT_NUM_JOINTS)
      return fail(err, errlen, "PolicyRunner: joint_index out of range");
  for (std::uint32_t k = 0; k < p.n_act; ++k)
    if (p.act_pos[k] < 0 || p.act_pos[k] >= static_cast<int>(p.n_joints))
      return fail(err, errlen, "PolicyRunner: act_pos out of range");
  if (!(p.step_dt_s > 0.0) || !(p.period_s > 0.0)) return fail(err, errlen, "PolicyRunner: period / step_dt");
  p_ = p;
  n_obs_ = n;
  for (std::uint32_t k = 0; k < p.n_act; ++k) act_idx26_[k] = p.joint_index[p.act_pos[k]];
  net_ = net;
  lean_target_ = 0.0;
  waist_target_[0] = waist_target_[1] = 0.0;
  reset();
  return true;
}

void PolicyRunner::reset() noexcept {
  for (int k = 0; k < kMaxAct; ++k) {
    last_action_[k] = 0.0;
    last_raw_[k] = 0.0;
    target_[k] = k < static_cast<int>(p_.n_act) ? p_.default_pos[p_.act_pos[k]] : 0.0;
  }
  phase_t_ = 0.0;
  t_last_ = 0.0;
  t_policy_ = 0.0;
  has_t_last_ = false;
  has_t_policy_ = false;
  lean_ = 0.0;
  waist_[0] = waist_[1] = 0.0;
  steps_ = 0;
}

double PolicyRunner::gait_phase() const noexcept { return std::fmod(phase_t_, p_.period_s) / p_.period_s; }

void PolicyRunner::observe(const double* q, const double* dq, const double quat[4], const double gyro[3],
                           const double cmd[3], double* obs) const noexcept {
  double g[3];
  projected_gravity(quat, g);
  const double ph = std::fmod(phase_t_, p_.period_s) / p_.period_s;  // Python float % (positive operands)
  double phase[2] = {std::sin(2 * kPi * ph), std::cos(2 * kPi * ph)};
  if (norm3(cmd[0], cmd[1], cmd[2]) < 0.1) phase[0] = phase[1] = 0.0;
  const int nj = static_cast<int>(p_.n_joints);
  const int na = static_cast<int>(p_.n_act);
  int o = 0;
  for (std::uint32_t k = 0; k < p_.n_terms; ++k) {
    switch (p_.obs_terms[k]) {
      case R1RT_OBS_BASE_ANG_VEL:
        for (int i = 0; i < 3; ++i) obs[o++] = gyro[i];
        break;
      case R1RT_OBS_PROJECTED_GRAVITY:
        for (int i = 0; i < 3; ++i) obs[o++] = g[i];
        break;
      case R1RT_OBS_COMMAND:
        for (int i = 0; i < 3; ++i) obs[o++] = cmd[i];
        break;
      case R1RT_OBS_LEAN:
        obs[o++] = lean_;
        break;
      case R1RT_OBS_WAIST_CMD:
        obs[o++] = waist_[0];
        obs[o++] = waist_[1];
        break;
      case R1RT_OBS_PHASE:
        obs[o++] = phase[0];
        obs[o++] = phase[1];
        break;
      case R1RT_OBS_JOINT_POS:
        for (int i = 0; i < nj; ++i) obs[o++] = q[i] - p_.default_pos[i];
        break;
      case R1RT_OBS_JOINT_VEL:
        for (int i = 0; i < nj; ++i) obs[o++] = dq[i];
        break;
      case R1RT_OBS_ACTIONS:
        for (int i = 0; i < na; ++i) obs[o++] = last_action_[i];
        break;
      default:
        break;
    }
  }
}

bool PolicyRunner::act(const double* q, const double* dq, const double quat[4], const double gyro[3],
                       const double cmd[3], double* obs_out, double abs_max) noexcept {
  double c[3];
  for (int i = 0; i < 3; ++i) c[i] = clip(cmd[i], p_.cmd_lo[i], p_.cmd_hi[i]);
  const bool walking = (norm2(c[0], c[1]) + std::fabs(c[2])) > 0.1;
  if (p_.has_lean) {  // as in training: lean_rate toward the target, 0 while walking
    const double goal = walking ? 0.0 : lean_target_;
    const double stp = p_.lean_rate * p_.step_dt_s;
    lean_ += clip(goal - lean_, -stp, stp);
  }
  if (p_.has_waist) {  // R1MC-Manip3
    for (int i = 0; i < 2; ++i) {
      const double wgoal = walking ? 0.0 : waist_target_[i];
      const double wstp = p_.waist_rate[i] * p_.step_dt_s;
      waist_[i] = waist_[i] + clip(wgoal - waist_[i], -wstp, wstp);
    }
  }
  double obs[kMaxObs];
  observe(q, dq, quat, gyro, c, obs);
  std::memcpy(last_obs_, obs, sizeof(double) * static_cast<std::size_t>(n_obs_));
  if (obs_out != nullptr) std::memcpy(obs_out, obs, sizeof(double) * static_cast<std::size_t>(n_obs_));
  double a[kMaxAct];
  net_->run(obs, a);
  const int na = static_cast<int>(p_.n_act);
  if (filter_ != nullptr) filter_(filter_ctx_, a, na);
  bool ok = true;
  for (int k = 0; k < na; ++k) {
    last_raw_[k] = a[k];
    if (!std::isfinite(a[k]) || (abs_max > 0.0 && std::fabs(a[k]) > abs_max)) ok = false;
  }
  if (!ok) return false;
  for (int k = 0; k < na; ++k) {
    last_action_[k] = a[k];
    target_[k] = p_.default_pos[p_.act_pos[k]] + p_.action_scale[k] * a[k];
  }
  steps_ += 1;
  return true;
}

void PolicyRunner::advance_clock(double t) noexcept {
  if (has_t_last_) phase_t_ += t - t_last_;
  t_last_ = t;
  has_t_last_ = true;
}

bool PolicyRunner::step(const double q26[R1RT_NUM_JOINTS], const double dq26[R1RT_NUM_JOINTS], const double quat[4],
                        const double gyro[3], const double cmd[3], double t, double q26_out[R1RT_NUM_JOINTS],
                        double kp26_out[R1RT_NUM_JOINTS], double kd26_out[R1RT_NUM_JOINTS]) noexcept {
  const int nj = static_cast<int>(p_.n_joints);
  double q[kMaxJoints];
  double dq[kMaxJoints];
  for (int i = 0; i < nj; ++i) {
    q[i] = q26[p_.joint_index[i]];
    dq[i] = dq26[p_.joint_index[i]];
  }
  advance_clock(t);
  bool ran = false;
  if (!has_t_policy_ || t - t_policy_ >= p_.step_dt_s - 1e-4) {
    t_policy_ = !has_t_policy_ ? t : t_policy_ + p_.step_dt_s;
    has_t_policy_ = true;
    if (t - t_policy_ > p_.step_dt_s) t_policy_ = t;
    act(q, dq, quat, gyro, cmd);
    ran = true;
  }
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    q26_out[j] = q26[j];
    kp26_out[j] = p_.kp_other[j];
    kd26_out[j] = p_.kd_other[j];
  }
  for (int k = 0; k < static_cast<int>(p_.n_act); ++k) q26_out[act_idx26_[k]] = target_[k];
  for (int i = 0; i < nj; ++i) {
    kp26_out[p_.joint_index[i]] = p_.kp[i];
    kd26_out[p_.joint_index[i]] = p_.kd[i];
  }
  return ran;
}

bool PolicyRunner::set_lean_target(double rad) noexcept {
  if (!p_.has_lean) return std::fabs(rad) < 1e-9;  // the reference accepts 0 for a policy without lean
  if (!(p_.lean_lo - 1e-6 <= rad && rad <= p_.lean_hi + 1e-6)) return false;
  lean_target_ = rad;
  return true;
}

bool PolicyRunner::set_waist_target(double roll, double yaw) noexcept {
  if (!std::isfinite(roll) || !std::isfinite(yaw)) return false;
  if (!p_.has_waist) return std::fabs(roll) < 1e-9 && std::fabs(yaw) < 1e-9;
  const double v[2] = {roll, yaw};
  for (int i = 0; i < 2; ++i)
    if (!(p_.waist_lo[i] - 1e-6 <= v[i] && v[i] <= p_.waist_hi[i] + 1e-6)) return false;
  waist_target_[0] = roll;
  waist_target_[1] = yaw;
  return true;
}

void PolicyRunner::save(State& s) const noexcept {
  std::memcpy(s.last_action, last_action_, sizeof(last_action_));
  std::memcpy(s.target, target_, sizeof(target_));
  s.lean = lean_;
  s.lean_target = lean_target_;
  s.waist[0] = waist_[0];
  s.waist[1] = waist_[1];
  s.waist_target[0] = waist_target_[0];
  s.waist_target[1] = waist_target_[1];
  s.phase_t = phase_t_;
  s.t_last = t_last_;
  s.t_policy = t_policy_;
  s.steps = steps_;
  s.has_t_last = has_t_last_ ? 1 : 0;
  s.has_t_policy = has_t_policy_ ? 1 : 0;
}

void PolicyRunner::restore(const State& s) noexcept {
  std::memcpy(last_action_, s.last_action, sizeof(last_action_));
  std::memcpy(target_, s.target, sizeof(target_));
  lean_ = s.lean;
  lean_target_ = s.lean_target;
  waist_[0] = s.waist[0];
  waist_[1] = s.waist[1];
  waist_target_[0] = s.waist_target[0];
  waist_target_[1] = s.waist_target[1];
  phase_t_ = s.phase_t;
  t_last_ = s.t_last;
  t_policy_ = s.t_policy;
  steps_ = s.steps;
  has_t_last_ = s.has_t_last != 0;
  has_t_policy_ = s.has_t_policy != 0;
}

}  // namespace r1rt
