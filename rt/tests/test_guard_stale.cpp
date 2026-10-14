// Session 9 (U19 / D25): r1guard applies the core's state-age rule on its own lowstate while it BALANCES after a
// takeover (docs/sim/STATE_AGE.md): stale_rung 0 = the last command resent after lowstate_stale_ns (as before); 1 / 2 =
// the stand after state_age_sustain_ns of state older than state_age_limit_ns (no automatic return in the guard, as
// its rung 3). Never limp, one writer.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "r1rt/guard.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"
#include "test_core_util.hpp"

namespace {

constexpr std::int64_t kMs = 1'000'000;

struct Pair {
  r1rt_test::Golden g;
  bool ok = false;
  std::string why;
  void* seg = nullptr;
  r1rt_config cfg{};
  r1rt_core* core = nullptr;
  r1rt::Guard guard;
  r1rt_lowstate_in st{};
  r1rt_lowcmd_out core_out{};
  r1rt_lowcmd_out guard_out{};
  r1rt_telemetry_record tel{};
  std::int64_t t = 1'000 * kMs;
  bool core_stalled = false;
  bool guard_gets_state = true;
  int two_writers = 0;

  explicit Pair(std::uint32_t stale_rung) {
    if (!r1rt_test::read_golden(R1RT_POLICY_RUNNER_GOLDEN, g, why)) return;
    const std::size_t seg_size = (r1rt_segment_size() + 63) / 64 * 64;
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
    cfg.policy_weights_path = R1RT_POLICY_WEIGHTS;
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
    cfg.stale_rung = stale_rung;
    std::int32_t status = 0;
    core = r1rt_core_create(&cfg, &status);
    if (core == nullptr) {
      why = "create: " + std::to_string(status);
      return;
    }
    char err[256] = {0};
    if (guard.init(cfg, r1rt::GuardParams{}, err, sizeof err) != R1RT_OK) {
      why = std::string("guard init: ") + err;
      return;
    }
    for (int j = 0; j < 24; ++j) st.q[j] = g.params.default_pos[j];
    st.imu_quat[0] = 1.0;
    st.crc_ok = 1;
    st.mode_machine = 1;
    ok = true;
  }
  ~Pair() {
    r1rt_core_destroy(core);
    std::free(seg);
  }
  r1rt::GuardStatus gstatus() const {
    r1rt::GuardStatus s{};
    guard.status(&s);
    return s;
  }
  void step() {  // 1 ms: the core's writer on even ms (+ its policy thread every 20 ms) unless stalled, the guard
    t += kMs;
    st.t_rx_ns = t;
    st.board_tick = static_cast<std::uint32_t>(t / kMs);
    core_out.publish = 0;
    if (!core_stalled && (t / kMs) % 2 == 0) {
      r1rt_core_step(core, &st, t, &core_out, &tel);
      if ((t / kMs) % 20 == 0) r1rt_core_policy_step(core, t);
    }
    guard.cycle(guard_gets_state ? &st : nullptr, t, &guard_out);
    if (core_out.publish && guard_out.publish) ++two_writers;
  }
  void run_ms(int ms) {
    for (int i = 0; i < ms; ++i) step();
  }
  bool enter() {
    t += kMs;
    if ((t / kMs) % 2 != 0) t += kMs;
    st.t_rx_ns = t;
    r1rt_core_step(core, &st, t, &core_out, &tel);
    return r1rt_core_request(core, R1RT_REQ_ENTER_BALANCE, t) == R1RT_OK;
  }
};

bool legs_stiff(const r1rt_lowcmd_out& c) {
  for (int j = 0; j < 14; ++j)
    if (!(c.kp[j] > 1.0)) return false;
  return c.publish != 0;
}

#define NEED_PAIR(p)                                                        \
  if (!(p).ok) {                                                            \
    if ((p).why.find("create: -7") == 0 || (p).why.find("guard init") == 0) \
      GTEST_SKIP() << "no policy weights: " << (p).why;                     \
    FAIL() << (p).why;                                                      \
  }

}  // namespace

TEST(GuardStale, BalancingAfterATakeoverItFreezesIntoTheStandOnOldStateNeverLimp) {
  Pair p(1);
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(300);
  p.core_stalled = true;
  p.run_ms(50);  // the takeover (~10 ms), then its own policy on fresh state
  ASSERT_TRUE(p.guard.active());
  ASSERT_EQ(p.gstatus().act_mode, R1RT_MODE_BALANCE);
  const std::int64_t t_last = p.t;
  p.guard_gets_state = false;  // its DDS reader stalled too
  std::int64_t t_freeze = 0;
  for (int i = 0; i < 150; ++i) {
    p.step();
    EXPECT_TRUE(legs_stiff(p.guard_out));
    if (t_freeze == 0 && p.gstatus().act_mode == R1RT_MODE_STAND_FREEZE) t_freeze = p.t;
  }
  ASSERT_NE(t_freeze, 0);
  EXPECT_GE(t_freeze - t_last, 70 * kMs);
  EXPECT_LE(t_freeze - t_last, 72 * kMs);
  const r1rt::GuardStatus s = p.gstatus();
  EXPECT_EQ(s.stale_rung_entries, 1u);
  EXPECT_EQ(s.stale_freeze, 1);
  EXPECT_EQ(s.stale_resends, 0u);  // frozen before the 100 ms resend rung
  EXPECT_EQ(p.two_writers, 0);
  p.guard_gets_state = true;
  p.run_ms(600);
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_STAND_FREEZE);  // no automatic return in the guard
  for (int j = 0; j < 12; ++j) EXPECT_NEAR(p.guard_out.q[j], p.cfg.stand_q[j], 1e-9);
}

TEST(GuardStale, RungOffItResendsAsBefore) {
  Pair p(0);
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(300);
  p.core_stalled = true;
  p.run_ms(50);
  ASSERT_TRUE(p.guard.active());
  p.guard_gets_state = false;
  p.run_ms(150);
  const r1rt::GuardStatus s = p.gstatus();
  EXPECT_EQ(s.act_mode, R1RT_MODE_BALANCE);
  EXPECT_GT(s.stale_resends, 0u);
  EXPECT_EQ(s.stale_rung_entries, 0u);
  EXPECT_EQ(s.state_old_episodes, 1u);  // counted with the rung off
  EXPECT_TRUE(legs_stiff(p.guard_out));
}
