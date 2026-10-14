// The step C ABI: argument checks, and a core created without a policy stays in MONITOR and never publishes
// (session 1 pinned the stub; session 3's core: tests/test_core.cpp).
#include "r1rt/step_api.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

extern "C" int r1rt_c_abi_check(void);

namespace {

struct Fixture {
  void* base = nullptr;
  std::size_t size = 0;
  r1rt_config cfg{};
  Fixture() {
    size = (r1rt_segment_size() + 63) / 64 * 64;
    base = std::aligned_alloc(64, size);
    r1rt_segment_init(base, size, 1, 0);
    cfg.abi_version = R1RT_ABI_VERSION;
    cfg.struct_size = sizeof(r1rt_config);
    cfg.writer_rate_hz = 500;
    cfg.policy_rate_hz = 50;
    cfg.stop_time_s = 0.15;
    cfg.arm_tilt_stop_rad = 4.0 * M_PI / 180.0;
    cfg.fall_tilt_rad = 30.0 * M_PI / 180.0;
    cfg.lowstate_late_ns = 10'000'000;
    cfg.lowstate_stale_ns = 100'000'000;
    cfg.lowstate_lost_ns = 1'000'000'000;
    cfg.policy_max_age_ns = 40'000'000;
    cfg.segment = base;
    cfg.segment_size = size;
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {  // ABI 2: limits are validated at create
      cfg.q_min[j] = -1.0;
      cfg.q_max[j] = 1.0;
      cfg.dq_max[j] = 10.0;
      cfg.tau_max[j] = 10.0;
      cfg.kp_max[j] = 100.0;
      cfg.kd_max[j] = 10.0;
    }
  }
  ~Fixture() { std::free(base); }
};

TEST(StepApi, CHeadersCompileAsC11AndBehave) { EXPECT_EQ(r1rt_c_abi_check(), 0); }

TEST(StepApi, AbiVersion) { EXPECT_EQ(r1rt_abi_version(), R1RT_ABI_VERSION); }

TEST(StepApi, CreateValidatesConfig) {
  Fixture f;
  int32_t st = 0;
  EXPECT_EQ(r1rt_core_create(nullptr, &st), nullptr);
  EXPECT_EQ(st, R1RT_E_INVALID_ARG);
  r1rt_config bad = f.cfg;
  bad.abi_version = R1RT_ABI_VERSION + 1;
  EXPECT_EQ(r1rt_core_create(&bad, &st), nullptr);
  EXPECT_EQ(st, R1RT_E_ABI_MISMATCH);
  bad = f.cfg;
  bad.writer_rate_hz = 2000;
  EXPECT_EQ(r1rt_core_create(&bad, &st), nullptr);
  EXPECT_EQ(st, R1RT_E_INVALID_ARG);
  bad = f.cfg;
  bad.writer_rate_hz = 510;  // not a multiple of the policy rate
  EXPECT_EQ(r1rt_core_create(&bad, &st), nullptr);
  bad = f.cfg;
  bad.segment_size = 64;
  EXPECT_EQ(r1rt_core_create(&bad, &st), nullptr);
  EXPECT_EQ(st, R1RT_E_BAD_SEGMENT);
  r1rt_core_destroy(nullptr);  // no-op
}

TEST(StepApi, MonitorWithoutAPolicyNeverPublishes) {
  Fixture f;
  int32_t st = -99;
  r1rt_core* core = r1rt_core_create(&f.cfg, &st);
  ASSERT_NE(core, nullptr);
  ASSERT_EQ(st, R1RT_OK);
  EXPECT_EQ(r1rt_core_mode(core), R1RT_MODE_MONITOR);
  r1rt_lowstate_in state{};
  state.board_tick = 77;
  state.crc_ok = 1;  // a state whose CRC failed is not used (R1RT_FAULT_CRC)
  state.t_rx_ns = 1'000;
  r1rt_lowcmd_out cmd;
  std::memset(&cmd, 0xFF, sizeof(cmd));  // garbage in: everything must be overwritten
  r1rt_telemetry_record tel{};
  for (int k = 1; k <= 3; ++k) {
    EXPECT_EQ(r1rt_core_step(core, &state, 2'000'000LL * k, &cmd, &tel), R1RT_OK);
    EXPECT_EQ(cmd.publish, 0);
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) EXPECT_EQ(cmd.enable[j], 0);
    EXPECT_EQ(tel.seq, static_cast<std::uint64_t>(k));
    EXPECT_EQ(tel.board_tick, 77u);
    EXPECT_EQ(tel.mode, R1RT_MODE_MONITOR);
    EXPECT_EQ(tel.token_owner, R1RT_OWNER_NONE);
  }
  EXPECT_EQ(r1rt_core_step(core, nullptr, 0, nullptr, nullptr), R1RT_E_INVALID_ARG);
  EXPECT_EQ(r1rt_core_step(core, nullptr, 0, &cmd, nullptr), R1RT_OK);  // no new state: fine
  EXPECT_EQ(r1rt_core_request(core, R1RT_REQ_ENTER_BALANCE, 0), R1RT_E_REFUSED);  // no policy loaded
  EXPECT_EQ(cmd.publish, 0);
  r1rt_core_destroy(core);
}

}  // namespace
