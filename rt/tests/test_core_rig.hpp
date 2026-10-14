// The core test rig (shared by test_core.cpp, test_core_takeover.cpp, test_core_threads.cpp): a core built from the
// golden sequence's policy params on its own segment, a quiet standing state, one writer tick + the policy thread on
// its 50 Hz schedule. Header-only, in an anonymous namespace (each test translation unit gets its own copy).
#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"
#include "test_core_util.hpp"

namespace {

constexpr std::int64_t kMs = 1'000'000;
constexpr std::int64_t kTick = 2 * kMs;  // 500 Hz

struct Rig {
  r1rt_test::Golden g;
  bool ok = false;
  std::string why;
  void* seg = nullptr;
  std::size_t seg_size = 0;
  r1rt_config cfg{};
  r1rt_core* core = nullptr;
  r1rt_lowstate_in st{};
  r1rt_lowcmd_out out{};
  r1rt_telemetry_record tel{};
  std::int64_t t = 1'000 * kMs;
  int k = 0;  // writer ticks since entry

  // tweak: changes the config before the core is created (e.g. the ABI 4 takeover gates)
  explicit Rig(bool with_policy = true, void (*tweak)(r1rt_config&) = nullptr) {
    if (!r1rt_test::read_golden(R1RT_POLICY_RUNNER_GOLDEN, g, why)) return;
    seg_size = (r1rt_segment_size() + 63) / 64 * 64;
    seg = std::aligned_alloc(64, seg_size);
    r1rt_segment_init(seg, seg_size, 1, 0);
    cfg.abi_version = R1RT_ABI_VERSION;
    cfg.struct_size = sizeof(r1rt_config);
    cfg.writer_rate_hz = 500;
    cfg.policy_rate_hz = 50;
    cfg.stop_time_s = 0.15;
    cfg.arm_tilt_stop_rad = 4.0 * M_PI / 180.0;
    cfg.fall_tilt_rad = 30.0 * M_PI / 180.0;
    cfg.lowstate_late_ns = 10 * kMs;
    cfg.lowstate_stale_ns = 100 * kMs;
    cfg.lowstate_lost_ns = 1000 * kMs;
    cfg.policy_max_age_ns = 40 * kMs;
    cfg.segment = seg;
    cfg.segment_size = seg_size;
    cfg.policy_weights_path = with_policy ? R1RT_POLICY_WEIGHTS : nullptr;
    cfg.policy = g.params;
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
      cfg.q_min[j] = -2.5;
      cfg.q_max[j] = 2.5;
      cfg.dq_max[j] = 30.0;
      cfg.tau_max[j] = 50.0;
      cfg.kp_max[j] = 500.0;
      cfg.kd_max[j] = 20.0;
      cfg.stand_q[j] = (j < 24 ? g.params.default_pos[j] : 0.0) + (j == 4 || j == 10 ? -0.066 : 0.0);
      cfg.stand_kp[j] = 200.0;
      cfg.stand_kd[j] = 3.0;
      cfg.upper_kp[j] = 60.0;
      cfg.upper_kd[j] = 2.0;
      cfg.damp_kd[j] = 3.0;
    }
    cfg.stand_ramp_s = 0.4;
    cfg.action_abs_max = 50.0;
    cfg.loop_gap_ns = 4 * kMs;
    if (tweak != nullptr) tweak(cfg);
    std::int32_t status = 0;
    core = r1rt_core_create(&cfg, &status);
    if (core == nullptr) {
      why = "create: " + std::to_string(status);
      return;
    }
    // a quiet standing state: the policy's default pose, upright
    for (int j = 0; j < 24; ++j) st.q[j] = g.params.default_pos[j];
    st.imu_quat[0] = 1.0;
    st.crc_ok = 1;
    st.mode_machine = 1;
    ok = true;
  }
  ~Rig() {
    r1rt_core_destroy(core);
    std::free(seg);
  }
  r1rt::shm::Segment& segment() { return *static_cast<r1rt::shm::Segment*>(seg); }
  // one writer tick (state delivered unless `withhold`), then the policy thread's iteration on its schedule
  std::int32_t tick(bool withhold = false, bool thread = true) {
    t += kTick;
    st.t_rx_ns = t;
    st.board_tick = static_cast<std::uint32_t>(t / kMs);
    const std::int32_t rc = r1rt_core_step(core, withhold ? nullptr : &st, t, &out, &tel);
    ++k;
    if (thread && k % 10 == 0) r1rt_core_policy_step(core, t);
    return rc;
  }
  bool enter() {
    t += kTick;
    st.t_rx_ns = t;
    r1rt_core_step(core, &st, t, &out, &tel);
    k = 0;
    return r1rt_core_request(core, R1RT_REQ_ENTER_BALANCE, t) == R1RT_OK;
  }
  r1rt_core_status status() const {
    r1rt_core_status s{};
    r1rt_core_get_status(core, &s);
    return s;
  }
};

}  // namespace

#define NEED_RIG(r)                        \
  if (!(r).ok) {                           \
    if ((r).why.find("create: -7") == 0) { \
      GTEST_SKIP() << "no policy weights"; \
    }                                      \
    FAIL() << (r).why;                     \
  }
