// r1rt::PolicyRunner - the balance policy's observation, schedule and action logic in C++ (DESIGN.md §3.2).
//
// A port of r1mc3.policy.reference.ReferencePolicy (= v2's WalkPolicy, r1mc2/policy.py at git r21_try 2a4a165,
// bit-identical in Python), on top of r1rt::PolicyMlp:
//   observe()  - the observation vector in the metadata's term order (base_ang_vel, projected_gravity, command, lean,
//                waist_cmd, phase, joint_pos, joint_vel, actions; Manip2-Lean uses 74 values);
//   act()      - one 50 Hz policy step: clip the command, ramp the lean / waist toward their targets (0 while
//                walking), observe, run the network, keep the raw action for the next observation, target =
//                default + scale * action;
//   step()     - the reference's per-tick controller interface (phase clock, the 50 Hz schedule inside a faster loop,
//                the 26-joint q / kp / kd), used by the golden test; the core uses advance_clock() + act().
// Every operation is the reference's, in the same order, compiled without floating-point contraction
// (-ffp-contract=off, rt/core.cmake), so the C++ result differs from Python only by the network's summation order
// (PolicyMlp vs NumPy: ~1e-14); golden tests: tests/unit/test_policy_runner.py (vs ReferencePolicy through the
// binding, <= 1e-12) and rt/tests/test_policy_runner.cpp (the committed sequence tests/assets/policy_runner/).
//
// ONE deliberate difference from v2 (DESIGN §4.2-4.3; S2 "Not solid" 2): a network output that is not finite or
// lies beyond +-abs_max is NOT used - act() returns false and leaves last_action / target unchanged, so a NaN never
// enters the next observation (v2 fed it back and was never valid again). The core then freezes (rung 3).
//
// Real-time rules: init() validates; nothing allocates afterwards (the network is PolicyMlp::run, which allocates
// nothing); no exceptions; constant time per call.
#pragma once

#include <cstddef>
#include <cstdint>

#include "r1rt/policy_mlp.hpp"
#include "r1rt/step_api.h"

namespace r1rt {

class PolicyRunner {
 public:
  static constexpr int kMaxObs = R1RT_MAX_POLICY_OBS;
  static constexpr int kMaxJoints = R1RT_POLICY_MAX_JOINTS;
  static constexpr int kMaxAct = R1RT_NUM_POLICY_ACTIONS;
  // Hook on the raw network output (fault injection in simulation: policy NaN / out of range). nullptr = none.
  using ActionFilter = void (*)(void* ctx, double* action, int n);

  // Checks the params (sizes, indices, the observation length == the network's input) and keeps a pointer to the
  // loaded network (owned by the caller, must outlive the runner). false + reason on error.
  bool init(const r1rt_policy_params& params, const PolicyMlp* net, char* err, std::size_t errlen) noexcept;
  bool ready() const noexcept { return net_ != nullptr; }

  // ReferencePolicy.reset(): no previous action, phase 0, no schedule yet, target = the default pose, ramps at 0
  // (the lean / waist TARGETS are kept, as in the reference).
  void reset() noexcept;

  // Observation from policy-order q, dq (n_joints each), the IMU quaternion (w, x, y, z), gyro and the (already
  // clipped) command. obs must hold n_obs() values.
  void observe(const double* q, const double* dq, const double quat[4], const double gyro[3], const double cmd[3],
               double* obs) const noexcept;

  // One policy step (ReferencePolicy.act). cmd is clipped here. Writes the observation it used into obs_out (n_obs()
  // values, nullable). Returns false when the output is invalid (see above): nothing is updated except the ramps.
  bool act(const double* q, const double* dq, const double quat[4], const double gyro[3], const double cmd[3],
           double* obs_out = nullptr, double abs_max = 0.0) noexcept;

  // The phase clock as the reference advances it (phase_t += t - t_last; the first call only sets t_last).
  void advance_clock(double t) noexcept;

  // ReferencePolicy.step() (every tick of a faster loop): the phase clock, the 50 Hz schedule (t_policy rule), and
  // the 26-joint command: q = the measured q with the action joints at the target, kp / kd = the policy's for its
  // joints, kp_other / kd_other elsewhere. Returns true if the network ran this call.
  bool step(const double q26[R1RT_NUM_JOINTS], const double dq26[R1RT_NUM_JOINTS], const double quat[4],
            const double gyro[3], const double cmd[3], double t, double q26_out[R1RT_NUM_JOINTS],
            double kp26_out[R1RT_NUM_JOINTS], double kd26_out[R1RT_NUM_JOINTS]) noexcept;

  // set_body_targets(lean_deg=...): false (nothing changed) when the policy has no lean or rad is outside its range.
  bool set_lean_target(double rad) noexcept;
  bool set_waist_target(double roll, double yaw) noexcept;

  void set_action_filter(ActionFilter f, void* ctx) noexcept {
    filter_ = f;
    filter_ctx_ = ctx;
  }

  const r1rt_policy_params& params() const noexcept { return p_; }
  int n_obs() const noexcept { return n_obs_; }
  int n_joints() const noexcept { return static_cast<int>(p_.n_joints); }
  int n_act() const noexcept { return static_cast<int>(p_.n_act); }
  const double* target() const noexcept { return target_; }          // n_act values (SDK joints act_idx26())
  const double* last_action() const noexcept { return last_action_; } // raw network output used last
  const double* last_raw() const noexcept { return last_raw_; }       // the latest output, valid or not
  const double* last_obs() const noexcept { return last_obs_; }       // the observation of the latest act()
  const int* act_idx26() const noexcept { return act_idx26_; }
  double lean() const noexcept { return lean_; }
  double lean_target() const noexcept { return lean_target_; }
  const double* waist() const noexcept { return waist_; }
  double phase_t() const noexcept { return phase_t_; }
  double gait_phase() const noexcept;  // (phase_t % period) / period
  std::uint64_t steps() const noexcept { return steps_; }

  // The full runner state that a bumpless hand-over between the policy thread and the writer must carry.
  struct State {
    double last_action[kMaxAct];
    double target[kMaxAct];
    double lean;
    double lean_target;
    double waist[2];
    double waist_target[2];
    double phase_t;
    double t_last;
    double t_policy;
    std::uint64_t steps;
    std::uint8_t has_t_last;
    std::uint8_t has_t_policy;
  };
  void save(State& s) const noexcept;
  void restore(const State& s) noexcept;

 private:
  r1rt_policy_params p_{};
  const PolicyMlp* net_ = nullptr;
  int n_obs_ = 0;
  int act_idx26_[kMaxAct] = {};
  double last_action_[kMaxAct] = {};
  double last_raw_[kMaxAct] = {};
  double last_obs_[kMaxObs] = {};
  double target_[kMaxAct] = {};
  double lean_ = 0.0;
  double lean_target_ = 0.0;
  double waist_[2] = {0.0, 0.0};
  double waist_target_[2] = {0.0, 0.0};
  double phase_t_ = 0.0;
  double t_last_ = 0.0;
  double t_policy_ = 0.0;
  bool has_t_last_ = false;
  bool has_t_policy_ = false;
  std::uint64_t steps_ = 0;
  ActionFilter filter_ = nullptr;
  void* filter_ctx_ = nullptr;
};

// Projected gravity R^T (0, 0, -1) from a (w, x, y, z) quaternion, with MuJoCo's mju_quat2Mat arithmetic (the
// reference calls it); also used by the core's estimate.
void projected_gravity(const double quat[4], double g[3]) noexcept;
// cos of the tilt (R[2][2] = the body z axis' world z component).
double quat_cos_tilt(const double quat[4]) noexcept;

}  // namespace r1rt
