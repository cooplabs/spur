// Test helpers for the core / policy-runner tests: the golden sequence file (tools/policy_runner_golden.py, "R1PR"
// v1) and a core configuration built from its policy params (the same values r1sim.v3_core.core_config uses).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "r1rt/step_api.h"

namespace r1rt_test {

struct Tick {
  double t;
  double lean_req;  // NaN = none
  double q[R1RT_NUM_JOINTS], dq[R1RT_NUM_JOINTS], quat[4], gyro[3], cmd[3];
};
struct PolicyRow {
  int tick;
  std::vector<double> obs, target, last_action;
  double lean, phase_t;
};
struct Golden {
  r1rt_policy_params params{};
  std::vector<Tick> ticks;
  std::vector<PolicyRow> rows;
  int n_obs = 0;
};

inline bool read_golden(const char* path, Golden& g, std::string& err) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    err = std::string("cannot open ") + path;
    return false;
  }
  std::vector<unsigned char> b;
  unsigned char buf[65536];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
  std::fclose(f);
  if (b.size() < 24 || std::memcmp(b.data(), "R1PR", 4) != 0) {
    err = "not an R1PR file";
    return false;
  }
  std::uint32_t h[5];
  std::memcpy(h, b.data() + 4, sizeof h);
  if (h[0] != 1) {
    err = "R1PR version";
    return false;
  }
  const std::uint32_t n_ticks = h[1], n_policy = h[2], n_obs = h[3], n_params = h[4];
  std::size_t off = 24;
  auto rd = [&](double* out, std::size_t k) {
    std::memcpy(out, b.data() + off, k * sizeof(double));
    off += k * sizeof(double);
  };
  std::vector<double> p(n_params);
  rd(p.data(), n_params);
  // r1sim.v3_core.PARAMS_LAYOUT
  std::size_t i = 0;
  r1rt_policy_params& q = g.params;
  auto take = [&](std::size_t k) {
    const double* v = p.data() + i;
    i += k;
    return v;
  };
  q.n_joints = static_cast<std::uint32_t>(*take(1));
  q.n_act = static_cast<std::uint32_t>(*take(1));
  q.n_terms = static_cast<std::uint32_t>(*take(1));
  q.has_lean = static_cast<std::uint32_t>(*take(1));
  q.has_waist = static_cast<std::uint32_t>(*take(1));
  const double* v = take(R1RT_POLICY_MAX_TERMS);
  for (int k = 0; k < R1RT_POLICY_MAX_TERMS; ++k) q.obs_terms[k] = static_cast<std::int32_t>(v[k]);
  v = take(R1RT_POLICY_MAX_JOINTS);
  for (int k = 0; k < R1RT_POLICY_MAX_JOINTS; ++k) q.joint_index[k] = static_cast<std::int32_t>(v[k]);
  v = take(R1RT_NUM_POLICY_ACTIONS);
  for (int k = 0; k < R1RT_NUM_POLICY_ACTIONS; ++k) q.act_pos[k] = static_cast<std::int32_t>(v[k]);
  std::memcpy(q.default_pos, take(R1RT_POLICY_MAX_JOINTS), sizeof q.default_pos);
  std::memcpy(q.kp, take(R1RT_POLICY_MAX_JOINTS), sizeof q.kp);
  std::memcpy(q.kd, take(R1RT_POLICY_MAX_JOINTS), sizeof q.kd);
  std::memcpy(q.action_scale, take(R1RT_NUM_POLICY_ACTIONS), sizeof q.action_scale);
  std::memcpy(q.kp_other, take(R1RT_NUM_JOINTS), sizeof q.kp_other);
  std::memcpy(q.kd_other, take(R1RT_NUM_JOINTS), sizeof q.kd_other);
  q.period_s = *take(1);
  q.step_dt_s = *take(1);
  std::memcpy(q.cmd_lo, take(3), sizeof q.cmd_lo);
  std::memcpy(q.cmd_hi, take(3), sizeof q.cmd_hi);
  q.lean_lo = *take(1);
  q.lean_hi = *take(1);
  q.lean_rate = *take(1);
  std::memcpy(q.waist_lo, take(2), sizeof q.waist_lo);
  std::memcpy(q.waist_hi, take(2), sizeof q.waist_hi);
  std::memcpy(q.waist_rate, take(2), sizeof q.waist_rate);
  if (i != n_params) {
    err = "params layout mismatch";
    return false;
  }
  g.ticks.resize(n_ticks);
  for (auto& t : g.ticks) {
    rd(&t.t, 1);
    rd(&t.lean_req, 1);
    float fl[62];
    std::memcpy(fl, b.data() + off, sizeof fl);
    off += sizeof fl;
    int k = 0;
    for (double& x : t.q) x = fl[k++];
    for (double& x : t.dq) x = fl[k++];
    for (double& x : t.quat) x = fl[k++];
    for (double& x : t.gyro) x = fl[k++];
    for (double& x : t.cmd) x = fl[k++];
  }
  g.n_obs = static_cast<int>(n_obs);
  g.rows.resize(n_policy);
  for (auto& r : g.rows) {
    double tk;
    rd(&tk, 1);
    r.tick = static_cast<int>(tk);
    r.obs.resize(n_obs);
    rd(r.obs.data(), n_obs);
    r.target.resize(q.n_act);
    rd(r.target.data(), q.n_act);
    r.last_action.resize(q.n_act);
    rd(r.last_action.data(), q.n_act);
    rd(&r.lean, 1);
    rd(&r.phase_t, 1);
  }
  if (off != b.size()) {
    err = "trailing bytes";
    return false;
  }
  return true;
}

}  // namespace r1rt_test
