// r1rt::PolicyRunner against the Python reference (r1mc3.policy.ReferencePolicy = v2's WalkPolicy) on the committed
// golden sequence (tools/policy_runner_golden.py: S2's 1,600-tick scripted sequence): every observation, target,
// action, the lean ramp and the phase clock within 1e-12; the 26-joint command composition; the invalid-output rule.
#include "r1rt/policy_runner.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "test_core_util.hpp"

namespace {

using r1rt::PolicyMlp;
using r1rt::PolicyRunner;

constexpr double kTol = 1e-12;

struct Loaded {
  PolicyMlp net;
  PolicyRunner runner;
  r1rt_test::Golden g;
  bool ok = false;
  std::string why;
  Loaded() {
    if (!r1rt_test::read_golden(R1RT_POLICY_RUNNER_GOLDEN, g, why)) return;
    char err[256] = {0};
    if (!net.load(R1RT_POLICY_WEIGHTS, err, sizeof err)) {
      why = err;
      return;
    }
    if (!runner.init(g.params, &net, err, sizeof err)) {
      why = err;
      return;
    }
    ok = true;
  }
};

TEST(PolicyRunner, MatchesTheReferenceOnTheGoldenSequence) {
  Loaded L;
  if (!L.ok && L.why.find("cannot open") != std::string::npos && L.why.find(".r1w") != std::string::npos)
    GTEST_SKIP() << "no policy weights (run CMake under uv run): " << L.why;
  ASSERT_TRUE(L.ok) << L.why;
  const auto& g = L.g;
  ASSERT_EQ(g.ticks.size(), 1600u);
  ASSERT_EQ(L.runner.n_obs(), 74);
  std::size_t row = 0;
  double max_obs = 0, max_tgt = 0, max_act = 0, max_lean = 0, max_phase = 0;
  for (std::size_t k = 0; k < g.ticks.size(); ++k) {
    const auto& t = g.ticks[k];
    if (!std::isnan(t.lean_req)) {
      const bool accepted = L.runner.set_lean_target(t.lean_req);
      EXPECT_EQ(accepted, k == 100) << "tick " << k;  // 3 deg accepted, 7 deg refused (0..5 deg)
    }
    double q26[R1RT_NUM_JOINTS], kp26[R1RT_NUM_JOINTS], kd26[R1RT_NUM_JOINTS];
    const bool ran = L.runner.step(t.q, t.dq, t.quat, t.gyro, t.cmd, t.t, q26, kp26, kd26);
    const bool want = row < g.rows.size() && g.rows[row].tick == static_cast<int>(k);
    ASSERT_EQ(ran, want) << "tick " << k;
    // composition: legs + waist = the held target, the rest = the measured q; the policy's gains on its joints
    for (int j = 0; j < 14; ++j) EXPECT_EQ(q26[j], L.runner.target()[j]);
    for (int j = 14; j < R1RT_NUM_JOINTS; ++j) EXPECT_EQ(q26[j], t.q[j]);
    for (int j = 0; j < 24; ++j) EXPECT_EQ(kp26[j], g.params.kp[j]);
    EXPECT_EQ(kp26[24], g.params.kp_other[24]);
    if (!ran) continue;
    const auto& r = g.rows[row++];
    for (int i = 0; i < 74; ++i) max_obs = std::fmax(max_obs, std::fabs(L.runner.last_obs()[i] - r.obs[i]));
    for (int i = 0; i < 14; ++i) {
      max_tgt = std::fmax(max_tgt, std::fabs(L.runner.target()[i] - r.target[i]));
      max_act = std::fmax(max_act, std::fabs(L.runner.last_action()[i] - r.last_action[i]));
    }
    max_lean = std::fmax(max_lean, std::fabs(L.runner.lean() - r.lean));
    max_phase = std::fmax(max_phase, std::fabs(L.runner.phase_t() - r.phase_t));
  }
  EXPECT_EQ(row, g.rows.size());
  EXPECT_GT(row, 150u);
  std::printf("PolicyRunner vs reference over %zu policy steps: max |obs| %.3g, |target| %.3g, |action| %.3g, "
              "|lean| %.3g, |phase_t| %.3g\n", row, max_obs, max_tgt, max_act, max_lean, max_phase);
  EXPECT_LE(max_obs, kTol);
  EXPECT_LE(max_tgt, kTol);
  EXPECT_LE(max_act, kTol);
  EXPECT_LE(max_lean, kTol);
  EXPECT_LE(max_phase, kTol);
  EXPECT_NEAR(L.runner.lean(), 3.0 * 3.14159265358979323846 / 180.0, 1e-12);  // the 3 deg lean was reached
}

void nan_filter(void*, double* a, int n) {
  for (int i = 0; i < n; ++i) a[i] = std::numeric_limits<double>::quiet_NaN();
}

TEST(PolicyRunner, AnInvalidOutputIsNotUsedAndNeverEntersTheObservation) {
  Loaded L;
  if (!L.ok) GTEST_SKIP() << L.why;
  const auto& t = L.g.ticks[700];
  double q[24], dq[24];
  for (int i = 0; i < 24; ++i) {
    q[i] = t.q[i];
    dq[i] = t.dq[i];
  }
  ASSERT_TRUE(L.runner.act(q, dq, t.quat, t.gyro, t.cmd));
  double act0[14], tgt0[14];
  for (int i = 0; i < 14; ++i) {
    act0[i] = L.runner.last_action()[i];
    tgt0[i] = L.runner.target()[i];
  }
  L.runner.set_action_filter(&nan_filter, nullptr);
  EXPECT_FALSE(L.runner.act(q, dq, t.quat, t.gyro, t.cmd));
  for (int i = 0; i < 14; ++i) {
    EXPECT_EQ(L.runner.last_action()[i], act0[i]);  // v2 kept the NaN here (S2 "Not solid" 2); v3 does not
    EXPECT_EQ(L.runner.target()[i], tgt0[i]);
    EXPECT_TRUE(std::isnan(L.runner.last_raw()[i]));
  }
  L.runner.set_action_filter(nullptr, nullptr);
  double obs[74];
  EXPECT_TRUE(L.runner.act(q, dq, t.quat, t.gyro, t.cmd, obs));
  for (double v : obs) EXPECT_TRUE(std::isfinite(v));
  EXPECT_FALSE(L.runner.act(q, dq, t.quat, t.gyro, t.cmd, nullptr, 1e-6));  // beyond abs_max: invalid too
}

TEST(PolicyRunner, InitRefusesBadParams) {
  Loaded L;
  if (!L.ok) GTEST_SKIP() << L.why;
  PolicyRunner r;
  char err[128];
  r1rt_policy_params p = L.g.params;
  p.n_joints = 23;  // the observation would be 72 long
  EXPECT_FALSE(r.init(p, &L.net, err, sizeof err));
  p = L.g.params;
  p.joint_index[3] = 40;
  EXPECT_FALSE(r.init(p, &L.net, err, sizeof err));
  p = L.g.params;
  p.obs_terms[0] = 77;
  EXPECT_FALSE(r.init(p, &L.net, err, sizeof err));
  EXPECT_FALSE(r.init(L.g.params, nullptr, err, sizeof err));
  EXPECT_TRUE(r.init(L.g.params, &L.net, err, sizeof err)) << err;
}

}  // namespace
