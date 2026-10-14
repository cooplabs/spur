// r1rt::Guard (r1guard) next to a real r1rt::Core on ONE segment, on a synthetic state stream (no simulator): standby
// (heartbeat, warm shadow, nothing published), the takeover on a stale core heartbeat (CAS, <= 15 ms, bumpless, by the
// core's mode: BALANCE / STAND_FREEZE / DAMP / TAKEOVER blend / ON_STAND), one writer at all times (the core after its
// stall sees the token gone), never an automatic damp, the guard's own invalid action -> STAND_FREEZE, the token taken
// away from the guard, observe-only mode, forensics, and no allocation per cycle. The same on the simulated robot:
// tests/integration/test_v3_guard.py.
#include <gtest/gtest.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "r1rt/guard.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"
#include "test_core_util.hpp"

// Counting allocator for this test binary (the no-allocation test).
static std::atomic<long> g_news{0};
void* operator new(std::size_t n) {
  g_news.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
// The nothrow / array forms as well, so EVERY new / delete pair of this binary goes through malloc / free: libstdc++'s
// std::stable_sort (inside GoogleTest) uses the nothrow form, and with only the plain forms replaced ASan reports
// alloc-dealloc-mismatch (its own nothrow new, our free) - session 6, the Linux build check (GCC 9 / libstdc++).
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  g_news.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(n ? n : 1);
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return ::operator new(n, std::nothrow); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

namespace {

constexpr std::int64_t kMs = 1'000'000;

bool legs_stiff(const r1rt_lowcmd_out& c) {
  for (int j = 0; j < 14; ++j)
    if (!(c.kp[j] > 1.0)) return false;
  return true;
}

// The core (500 Hz writer, 50 Hz policy thread) and the guard (1 kHz) on one segment; t advances 1 ms per step().
struct Pair {
  r1rt_test::Golden g;
  bool ok = false;
  std::string why;
  void* seg = nullptr;
  std::size_t seg_size = 0;
  r1rt_config cfg{};
  r1rt_core* core = nullptr;
  r1rt::Guard guard;
  r1rt_lowstate_in st{};
  r1rt_lowcmd_out core_out{};
  r1rt_lowcmd_out guard_out{};
  r1rt_telemetry_record tel{};
  std::int64_t t = 1'000 * kMs;
  std::int64_t core_last_tick = 0;
  bool core_stalled = false;
  bool policy_thread_stalled = false;  // only the core's policy thread (the core's writer goes inline: rung 1)
  bool guard_gets_state = true;
  int two_writers = 0;  // ticks on which both published (must stay 0)

  explicit Pair(const r1rt::GuardParams& gp = r1rt::GuardParams{}) {
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
    std::int32_t status = 0;
    core = r1rt_core_create(&cfg, &status);
    if (core == nullptr) {
      why = "create: " + std::to_string(status);
      return;
    }
    char err[256] = {0};
    const std::int32_t rc = guard.init(cfg, gp, err, sizeof err);
    if (rc != R1RT_OK) {
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
  r1rt::shm::Segment& segment() { return *static_cast<r1rt::shm::Segment*>(seg); }
  r1rt::GuardStatus gstatus() const {
    r1rt::GuardStatus s{};
    guard.status(&s);
    return s;
  }
  r1rt_core_status cstatus() const {
    r1rt_core_status s{};
    r1rt_core_get_status(core, &s);
    return s;
  }
  // One millisecond: the core's writer on even ms (+ its policy thread every 20 ms) unless stalled, then the guard.
  void step() {
    t += kMs;
    st.t_rx_ns = t;
    st.board_tick = static_cast<std::uint32_t>(t / kMs);
    core_out.publish = 0;
    if (!core_stalled && (t / kMs) % 2 == 0) {
      r1rt_core_step(core, &st, t, &core_out, &tel);
      core_last_tick = t;
      if ((t / kMs) % 20 == 0 && !policy_thread_stalled) r1rt_core_policy_step(core, t);
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

#define NEED_PAIR(p)                                                    \
  if (!(p).ok) {                                                        \
    if ((p).why.find("create: -7") == 0 || (p).why.find("guard init") == 0) \
      GTEST_SKIP() << "no policy weights: " << (p).why;                 \
    FAIL() << (p).why;                                                  \
  }

TEST(Guard, InitNeedsTheSegmentAndAPolicy) {
  r1rt_config c{};
  c.writer_rate_hz = 500;
  c.policy_rate_hz = 50;
  r1rt::Guard gd;
  char err[128] = {0};
  EXPECT_EQ(gd.init(c, r1rt::GuardParams{}, err, sizeof err), R1RT_E_INVALID_ARG) << err;
  Pair p;
  NEED_PAIR(p);
  r1rt_config no_policy = p.cfg;
  no_policy.policy_weights_path = nullptr;
  r1rt::Guard g2;
  EXPECT_EQ(g2.init(no_policy, r1rt::GuardParams{}, err, sizeof err), R1RT_E_POLICY_LOAD);
  EXPECT_NE(std::string(err).find("cannot balance"), std::string::npos);
}

TEST(Guard, StandbyPublishesNothingBeatsAndKeepsAWarmShadow) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(1000);
  const auto s = p.gstatus();
  EXPECT_EQ(s.active, 0);
  EXPECT_EQ(s.publishes, 0u);
  EXPECT_EQ(s.takeovers, 0u);
  EXPECT_EQ(s.mode, R1RT_MODE_MONITOR);
  EXPECT_GE(s.shadow_steps, 49u);  // 50 Hz over 1 s
  EXPECT_EQ(s.shadow_invalid, 0u);
  EXPECT_EQ(s.shadow_valid, 1);
  EXPECT_LE(s.rt_beat_age_ns, 2 * kMs);
  r1rt_heartbeat hb{};
  ASSERT_TRUE(p.segment().guard_heartbeat.try_read(hb));
  EXPECT_EQ(hb.seq, s.cycles);
  EXPECT_EQ(hb.mode, R1RT_MODE_MONITOR);
  EXPECT_EQ(hb.rate_hz, 1000u);
  EXPECT_EQ(p.two_writers, 0);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(p.segment().token.load()), R1RT_OWNER_R1RT);
}

TEST(Guard, TakesOverAStalledBalancingCoreWithin15MsBumplessAndTheCoreStopsWriting) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(1000);
  const r1rt_lowcmd_out core_last = p.core_out;
  ASSERT_EQ(core_last.publish, 1);
  p.core_stalled = true;
  const std::int64_t t_last = p.core_last_tick;
  std::int64_t t_take = 0;
  r1rt_lowcmd_out first{};
  for (int i = 0; i < 40 && t_take == 0; ++i) {
    p.step();
    if (p.guard_out.publish) {
      t_take = p.t;
      first = p.guard_out;
    }
  }
  ASSERT_GT(t_take, 0) << "no takeover";
  // stale = older than 5 writer periods (10 ms): the first 1 kHz cycle after that publishes
  EXPECT_GT(t_take - t_last, 10 * kMs);
  EXPECT_LE(t_take - t_last, 15 * kMs);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(p.segment().token.load()), R1RT_OWNER_R1GUARD);
  // bumpless: the guard's first command is the core's policy continued (legs + waist targets within a policy
  // step's change; arms / head exactly the core's last targets and gains)
  double dmax = 0.0;
  for (int j = 0; j < 14; ++j) dmax = std::fmax(dmax, std::fabs(first.q[j] - core_last.q[j]));
  EXPECT_LT(dmax, 0.05);
  for (int j = 14; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_DOUBLE_EQ(first.q[j], core_last.q[j]) << j;
    EXPECT_DOUBLE_EQ(first.kp[j], core_last.kp[j]) << j;
    EXPECT_DOUBLE_EQ(first.kd[j], core_last.kd[j]) << j;
  }
  for (int j = 0; j < 14; ++j) {
    EXPECT_DOUBLE_EQ(first.kp[j], core_last.kp[j]) << j;
    EXPECT_DOUBLE_EQ(first.kd[j], core_last.kd[j]) << j;
  }
  const auto fx = p.guard.forensics();
  EXPECT_EQ(fx.captured, 1);
  EXPECT_EQ(fx.mode_from, R1RT_MODE_BALANCE);
  EXPECT_EQ(fx.rt_beat_t_ns, t_last);
  EXPECT_EQ(fx.rt_silence_ns, t_take - t_last);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(fx.token_before), R1RT_OWNER_R1RT);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(fx.token_after), R1RT_OWNER_R1GUARD);
  EXPECT_EQ(r1rt::shm::OwnershipToken::generation_of(fx.token_after),
            r1rt::shm::OwnershipToken::generation_of(fx.token_before) + 1);
  EXPECT_EQ(fx.mirror_policy_valid, 1);
  EXPECT_EQ(fx.proc_err, -1);  // the /proc capture is the process wrapper's, outside the cycle
  // the guard keeps balancing at its policy rate; the core comes back and finds the token gone: it does not write
  p.run_ms(4000);
  p.core_stalled = false;
  for (int i = 0; i < 1000; ++i) {
    p.step();
    if ((p.t / kMs) % 2 == 0) {
      EXPECT_EQ(p.core_out.publish, 0);
      EXPECT_NE(p.cstatus().faults & R1RT_FAULT_NOT_OWNER, 0u);
    }
    ASSERT_EQ(p.guard_out.publish, 1);
    ASSERT_TRUE(legs_stiff(p.guard_out));
  }
  const auto s = p.gstatus();
  EXPECT_EQ(s.active, 1);
  EXPECT_EQ(s.act_mode, R1RT_MODE_BALANCE);
  EXPECT_EQ(s.takeovers, 1u);
  EXPECT_GE(s.policy_steps, 250u);  // 50 Hz over ~5 s
  EXPECT_EQ(s.invalid_actions, 0u);
  EXPECT_EQ(p.two_writers, 0);
  r1rt_heartbeat hb{};
  ASSERT_TRUE(p.segment().guard_heartbeat.try_read(hb));
  EXPECT_EQ(hb.mode, R1RT_MODE_GUARDED);
}

TEST(Guard, TakesOverFromTheCoresInlineRungBumpless) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(500);
  p.policy_thread_stalled = true;  // the writer runs the policy inline from ~40 ms on
  p.run_ms(300);
  ASSERT_EQ(p.cstatus().inline_active, 1);
  const r1rt_lowcmd_out core_last = p.core_out;
  p.core_stalled = true;  // now the writer too
  p.run_ms(12);
  ASSERT_TRUE(p.guard.active());
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_BALANCE);
  double dmax = 0.0;
  for (int j = 0; j < 14; ++j) dmax = std::fmax(dmax, std::fabs(p.guard_out.q[j] - core_last.q[j]));
  EXPECT_LT(dmax, 0.05);
  EXPECT_EQ(p.guard.forensics().mirror_policy_valid, 1);
  EXPECT_NE(p.guard.forensics().rt_faults & R1RT_FAULT_POLICY_LATE, 0u);  // the core's last beat: on the inline rung
  EXPECT_EQ(p.gstatus().shadow_invalid, 0u);  // the shadow kept following the core's inline actions
}

TEST(Guard, AShortCoreHiccupBelowTheThresholdIsNotATakeover) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(300);
  p.core_stalled = true;  // 4 writer ticks missing: the heartbeat at most 10 ms old (stale is > 10 ms)
  p.run_ms(9);
  p.core_stalled = false;
  p.run_ms(300);
  const auto s = p.gstatus();
  EXPECT_EQ(s.takeovers, 0u);
  EXPECT_EQ(s.publishes, 0u);
  EXPECT_EQ(p.cstatus().faults & R1RT_FAULT_NOT_OWNER, 0u);  // the core still writes
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(p.segment().token.load()), R1RT_OWNER_R1RT);
}

TEST(Guard, NothingToTakeOverFromAMonitoringCore) {
  Pair p;
  NEED_PAIR(p);
  p.run_ms(200);  // the core in MONITOR (never took the token)
  p.core_stalled = true;
  p.run_ms(500);
  const auto s = p.gstatus();
  EXPECT_EQ(s.takeovers, 0u);
  EXPECT_EQ(s.publishes, 0u);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(p.segment().token.load()), R1RT_OWNER_NONE);
}

TEST(Guard, ContinuesTheCoresStandFreezeRampAndHolds) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(500);
  ASSERT_EQ(r1rt_core_request(p.core, R1RT_REQ_STAND_FREEZE, p.t), R1RT_OK);
  p.run_ms(100);  // mid-ramp (0.4 s)
  const r1rt_lowcmd_out core_last = p.core_out;
  p.core_stalled = true;
  p.run_ms(20);
  ASSERT_TRUE(p.guard.active());
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_STAND_FREEZE);
  double d0 = 0.0;
  for (int j = 0; j < 14; ++j) d0 = std::fmax(d0, std::fabs(p.guard_out.q[j] - core_last.q[j]));
  EXPECT_LT(d0, 0.05);  // continuous with the core's last ramp command
  p.run_ms(1000);
  for (int j = 0; j < 14; ++j) {
    EXPECT_NEAR(p.guard_out.q[j], p.cfg.stand_q[j], 1e-9) << j;
    EXPECT_NEAR(p.guard_out.kp[j], p.cfg.stand_kp[j], 1e-9) << j;
  }
  EXPECT_TRUE(legs_stiff(p.guard_out));
  EXPECT_EQ(p.gstatus().faults & R1RT_FAULT_POLICY_INVALID, 0u);  // continuing the operator's freeze, not a fault
}

TEST(Guard, AnOperatorsDampStaysDampAndNothingElseEverDamps) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(300);
  ASSERT_EQ(r1rt_core_request(p.core, R1RT_REQ_DAMP, p.t), R1RT_OK);  // the operator
  p.run_ms(10);
  p.core_stalled = true;
  p.run_ms(50);
  ASSERT_TRUE(p.guard.active());
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_DAMP);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_EQ(p.guard_out.kp[j], 0.0);
    EXPECT_EQ(p.guard_out.kd[j], p.cfg.damp_kd[j]);
  }
}

TEST(Guard, ItsOwnInvalidActionAtTheTakeoverFreezesIntoTheStandNeverDamps) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(500);
  // NaN on the guard's network output from now for 1 s (the core's own policy is untouched)
  ASSERT_EQ(p.guard.inject(R1RT_INJECT_POLICY_NAN, p.t, p.t + 1000 * kMs, 0.0), R1RT_OK);
  p.run_ms(200);  // standby: the shadow is invalid - flagged, nothing published, the core unaffected
  auto s = p.gstatus();
  EXPECT_GT(s.shadow_invalid, 5u);
  EXPECT_EQ(s.shadow_valid, 0);
  EXPECT_NE(s.faults & R1RT_FAULT_POLICY_INVALID, 0u);
  EXPECT_EQ(s.publishes, 0u);
  EXPECT_EQ(p.core_out.publish, 1);
  r1rt_heartbeat hb{};
  ASSERT_TRUE(p.segment().guard_heartbeat.try_read(hb));
  EXPECT_NE(hb.faults & R1RT_FAULT_POLICY_INVALID, 0u);  // visible to the core / r1d
  p.core_stalled = true;
  p.run_ms(20);
  ASSERT_TRUE(p.guard.active());
  s = p.gstatus();
  EXPECT_EQ(s.act_mode, R1RT_MODE_STAND_FREEZE);
  EXPECT_GE(s.invalid_actions, 1u);
  for (int i = 0; i < 2000; ++i) {  // past the injection window: still frozen (nothing leaves it automatically)
    p.step();
    ASSERT_EQ(p.guard_out.publish, 1);
    ASSERT_TRUE(legs_stiff(p.guard_out));
  }
  s = p.gstatus();
  EXPECT_EQ(s.act_mode, R1RT_MODE_STAND_FREEZE);
  EXPECT_NE(s.faults & R1RT_FAULT_POLICY_INVALID, 0u);
  for (int j = 0; j < 14; ++j) EXPECT_NEAR(p.guard_out.q[j], p.cfg.stand_q[j], 1e-9) << j;
}

TEST(Guard, StopsWritingWhenTheTokenIsTakenAway) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(300);
  p.core_stalled = true;
  p.run_ms(30);
  ASSERT_EQ(p.guard_out.publish, 1);
  std::uint64_t w = p.segment().token.load();
  ASSERT_TRUE(p.segment().token.try_transfer(w, R1RT_OWNER_NONE));  // e.g. the operator's tool
  p.run_ms(5);
  EXPECT_EQ(p.guard_out.publish, 0);
  const auto s = p.gstatus();
  EXPECT_NE(s.faults & R1RT_FAULT_NOT_OWNER, 0u);
  EXPECT_GE(s.token_lost_cycles, 5u);
}

TEST(Guard, ObserveOnlyNeverTakesTheToken) {
  r1rt::GuardParams gp;
  gp.takeover_enabled = 0;
  Pair p(gp);
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(200);
  p.core_stalled = true;
  p.run_ms(500);
  EXPECT_EQ(p.gstatus().takeovers, 0u);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(p.segment().token.load()), R1RT_OWNER_R1RT);
}

TEST(Guard, WithoutItsOwnStateItHoldsTheCoresLastCommand) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(300);
  const r1rt_lowcmd_out core_last = p.core_out;
  p.guard_gets_state = false;  // its DDS reader stalled too
  p.core_stalled = true;
  p.run_ms(150);  // > 100 ms: its own state is stale
  ASSERT_TRUE(p.guard.active());
  ASSERT_EQ(p.guard_out.publish, 1);
  EXPECT_GT(p.gstatus().stale_resends, 0u);
  EXPECT_TRUE(legs_stiff(p.guard_out));
  // the guard's last policy command (or the core's last one) is held: close to the core's last
  for (int j = 0; j < 14; ++j) EXPECT_NEAR(p.guard_out.q[j], core_last.q[j], 0.05) << j;
}

// A core caught mid-takeover (the mirror says TAKEOVER_BLEND; the core's takeover sequence is rtcore's session-4
// work) and on ON_STAND: written into the segment directly.
void fake_core(Pair& p, r1rt_mode mode, float progress, const double q[R1RT_NUM_JOINTS]) {
  auto& seg = p.segment();
  std::uint64_t w = seg.token.load();
  if (r1rt::shm::OwnershipToken::owner_of(w) != R1RT_OWNER_R1RT) {
    ASSERT_TRUE(seg.token.try_transfer(w, R1RT_OWNER_R1RT));
  }
  r1rt_heartbeat hb{};
  hb.seq = 7;
  hb.t_ns = p.t;
  hb.rate_hz = 500;
  hb.mode = mode;
  hb.blend_progress = progress;
  seg.rt_heartbeat.write(hb);
  r1rt_mirror mi{};
  mi.seq = 7;
  mi.t_ns = p.t;
  mi.mode = mode;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    mi.q_target[j] = q[j];
    mi.kp[j] = 150.0;
    mi.kd[j] = 4.0;
  }
  seg.mirror.write(mi);
}

TEST(Guard, FromATakeoverBlendItBlendsTheRestIntoItsPolicy) {
  r1rt::GuardParams gp;
  gp.takeover_blend_s = 0.3;
  Pair p(gp);
  NEED_PAIR(p);
  p.core_stalled = true;  // only the fake core below
  double q[R1RT_NUM_JOINTS];
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) q[j] = p.st.q[j] + 0.01;
  fake_core(p, R1RT_MODE_TAKEOVER_BLEND, 0.5f, q);
  p.run_ms(11);
  ASSERT_TRUE(p.guard.active());
  const r1rt_lowcmd_out first = p.guard_out;
  for (int j = 0; j < 14; ++j) {
    EXPECT_NEAR(first.q[j], q[j], 1e-9) << j;  // s = 0 at the takeover cycle: the core's last command
    EXPECT_NEAR(first.kp[j], 150.0, 1e-9) << j;
  }
  p.run_ms(200);  // > 0.3 x (1 - 0.5) = 0.15 s: the guard's own policy (its gains)
  const auto& prm = p.g.params;
  for (std::uint32_t i = 0; i < prm.n_joints; ++i) {
    if (prm.joint_index[i] < 14) {
      EXPECT_NEAR(p.guard_out.kp[prm.joint_index[i]], prm.kp[i], 1e-9);
    }
  }
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_BALANCE);
  EXPECT_TRUE(legs_stiff(p.guard_out));
}

TEST(Guard, FromOnStandItHoldsTheLastCommand) {
  Pair p;
  NEED_PAIR(p);
  p.core_stalled = true;
  double q[R1RT_NUM_JOINTS];
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) q[j] = 0.02 * j;
  fake_core(p, R1RT_MODE_ON_STAND, 0.0f, q);
  p.run_ms(500);
  ASSERT_TRUE(p.guard.active());
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_ON_STAND);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_NEAR(p.guard_out.q[j], q[j], 1e-12) << j;
    EXPECT_NEAR(p.guard_out.kp[j], 150.0, 1e-12) << j;
  }
}

// ---- session 5 (layout 2): the guard continues the core EXACTLY from the mirror's blend / freeze fields ----

// The built-in's stand lock as the DDS layer hands it to the core (every motor enabled, legs kp 200).
r1rt_lowcmd_out stand_lock_cmd(const r1rt_lowstate_in& st, double leg_kp = 200.0) {
  r1rt_lowcmd_out b{};
  b.publish = 1;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    b.enable[j] = 1;
    b.q[j] = st.q[j] + 0.01;
    b.kp[j] = j < 12 ? leg_kp : 60.0;
    b.kd[j] = 3.0;
  }
  return b;
}

// MONITOR -> MIRROR (the stand lock) -> RELEASED -> `blend_ms` into the blend (target BALANCE or ON_STAND).
bool feed_builtin(Pair& p, double leg_kp) {
  const r1rt_lowcmd_out b = stand_lock_cmd(p.st, leg_kp);
  return r1rt_core_set_builtin_cmd(p.core, &b, p.t) == R1RT_OK;
}

bool into_blend(Pair& p, int blend_ms, std::int32_t first_request, double leg_kp = 200.0) {
  p.run_ms(2);
  if (!feed_builtin(p, leg_kp)) return false;
  if (r1rt_core_request(p.core, first_request, p.t) != R1RT_OK) return false;
  for (int i = 0; i < 50; ++i) {
    p.step();
    if (!feed_builtin(p, leg_kp)) return false;
  }
  if (r1rt_core_request(p.core, R1RT_REQ_TAKEOVER_RELEASED, p.t) != R1RT_OK) return false;
  p.run_ms(blend_ms);
  return r1rt_core_mode(p.core) == R1RT_MODE_TAKEOVER_BLEND;
}

TEST(Guard, ContinuesTheCoresFreezeRampExactlyEndingWhenTheCoresWould) {
  // Two identical pairs: in one the core stalls 100 ms into the operator's 0.4 s freeze ramp; in the other it runs
  // on. The guard's legs + waist commands must equal the live core's (the same ramp, ending at freeze_t0 + 0.4 s) -
  // session 4's guard restarted a full 0.4 s ramp from its takeover (ending ~0.11 s later, a different line).
  Pair stalled;
  Pair live;
  NEED_PAIR(stalled);
  NEED_PAIR(live);
  for (Pair* p : {&stalled, &live}) {
    ASSERT_TRUE(p->enter());
    p->run_ms(500);
    ASSERT_EQ(r1rt_core_request(p->core, R1RT_REQ_STAND_FREEZE, p->t), R1RT_OK);
    p->run_ms(100);
  }
  stalled.core_stalled = true;
  double worst = 0.0;
  int compared = 0;
  for (int i = 0; i < 600; ++i) {
    stalled.step();
    live.step();
    if (!stalled.guard.active() || !live.core_out.publish) continue;  // compare at the live core's ticks
    for (int j = 0; j < 14; ++j) {
      worst = std::fmax(worst, std::fabs(stalled.guard_out.q[j] - live.core_out.q[j]));
      worst = std::fmax(worst, std::fabs(stalled.guard_out.kp[j] - live.core_out.kp[j]));
    }
    ++compared;
  }
  ASSERT_GT(compared, 250);
  EXPECT_LT(worst, 1e-9);  // the same ramp to the rounding
  EXPECT_EQ(stalled.gstatus().act_mode, R1RT_MODE_STAND_FREEZE);
  EXPECT_EQ(stalled.gstatus().faults & R1RT_FAULT_POLICY_INVALID, 0u);  // the operator's freeze, not a fault
}

TEST(Guard, ContinuesTheCoresTakeoverBlendExactlyIntoItsPolicy) {
  // The core stalls 400 ms into its 1.0 s blend from the built-in's stand lock (legs kp 200) into the policy (legs kp
  // 84.302 / ankles 31.385). The guard continues the SAME ramp: its leg gains equal (1 - a) 200 + a kp_policy with
  // a = (t - blend_t0) / 1.0 s, reaching the policy's exactly at blend_t0 + 1.0 s (session 4: a 0.3 s x (1 - a)
  // blend from the takeover, i.e. done ~0.19 s after the stall - a gain step the core would not have made).
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(into_blend(p, 400, R1RT_REQ_TAKEOVER));
  r1rt_mirror mi{};
  ASSERT_TRUE(p.segment().mirror.try_read(mi));
  const std::int64_t blend_t0 = mi.blend_t0_ns;
  ASSERT_GT(blend_t0, 0);
  ASSERT_EQ(mi.blend_ns, 1000 * kMs);
  const r1rt_lowcmd_out core_last = p.core_out;
  p.core_stalled = true;
  const auto& prm = p.g.params;
  double worst_kp = 0.0, first_dq = -1.0;
  for (int i = 0; i < 800; ++i) {
    p.step();
    if (!p.guard.active()) continue;
    if (first_dq < 0.0) {
      first_dq = 0.0;
      for (int j = 0; j < 14; ++j) first_dq = std::fmax(first_dq, std::fabs(p.guard_out.q[j] - core_last.q[j]));
    }
    const double a = std::fmin(1.0, static_cast<double>(p.t - blend_t0) / 1e9);
    for (std::uint32_t k = 0; k < prm.n_joints; ++k) {
      const int j = prm.joint_index[k];
      if (j >= 12) continue;  // the legs (the snapshot's legs had kp 200)
      const double expect = (1.0 - a) * 200.0 + a * prm.kp[k];
      worst_kp = std::fmax(worst_kp, std::fabs(p.guard_out.kp[j] - expect));
    }
    EXPECT_TRUE(legs_stiff(p.guard_out));
  }
  EXPECT_LT(worst_kp, 1e-6);  // the core's ramp, continued (float mirror kp: ~1e-13)
  EXPECT_LT(first_dq, 0.02);   // no step at the takeover
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_BALANCE);
  EXPECT_EQ(p.gstatus().mode_from, R1RT_MODE_TAKEOVER_BLEND);
  EXPECT_EQ(p.two_writers, 0);
}

TEST(Guard, ASequenceIntoOnStandContinuesTheCoresGainRampToTheHoldNeverBalanced) {
  // The core stalls 400 ms into the 1 s blend into ON_STAND from a SOFT built-in (legs kp 0: Unitree's zero torque /
  // damping on the stand). The guard continues the core's gain ramp: its leg gains equal (1 - a) 0 + a 200 with
  // a = (t - blend_t0) / 1 s, reaching the hold gains (the stand gains here: no on_stand gains in the config) exactly at
  // the core's blend end; the targets held where the mirror had them; no policy on a hanging robot. Session 5 / 80cac26
  // held the mirror's command as it was: legs at kp ~80 (here) - after a stall in MIRROR / early BLEND from zero torque
  // they stayed kp ~0, limp by our software (fails on 80cac26).
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(into_blend(p, 400, R1RT_REQ_ON_STAND, 0.0));
  r1rt_mirror mi{};
  ASSERT_TRUE(p.segment().mirror.try_read(mi));
  const std::int64_t blend_t0 = mi.blend_t0_ns;
  ASSERT_EQ(mi.blend_ns, 1000 * kMs);
  const r1rt_lowcmd_out core_last = p.core_out;
  p.core_stalled = true;
  double worst = 0.0, max_step = 0.0, prev_kp = -1.0;
  for (int i = 0; i < 900; ++i) {
    p.step();
    if (!p.guard.active()) continue;
    const double a = std::fmin(1.0, static_cast<double>(p.t - blend_t0) / 1e9);
    for (int j = 0; j < 12; ++j) worst = std::fmax(worst, std::fabs(p.guard_out.kp[j] - a * 200.0));
    if (prev_kp >= 0.0) max_step = std::fmax(max_step, std::fabs(p.guard_out.kp[0] - prev_kp));
    prev_kp = p.guard_out.kp[0];
  }
  EXPECT_LT(worst, 1e-6);  // the core's own ramp, continued
  EXPECT_LT(max_step, 200.0 / 1000.0 * 1.5);  // per 1 ms guard cycle: the ramp's step, no jump
  for (int j = 0; j < 14; ++j) {
    EXPECT_NEAR(p.guard_out.kp[j], 200.0, 1e-9) << j;  // stiff at the end: the hold
    EXPECT_NEAR(p.guard_out.q[j], core_last.q[j], 1e-9) << j;  // the targets held
  }
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_ON_STAND);
  EXPECT_EQ(p.gstatus().policy_steps, 0u);
  EXPECT_EQ(p.two_writers, 0);
}

TEST(Guard, AStallInMirrorOfAnOnStandEntryRampsTheWholeBlendToTheHold) {
  // The core stalls in MIRROR (ReleaseMode still running, a = 0) of an entry from a soft built-in: the guard ramps the
  // legs from the snapshot's kp 0 to the hold over the core's whole blend length from its takeover - never left soft.
  Pair p;
  NEED_PAIR(p);
  p.run_ms(2);
  ASSERT_TRUE(feed_builtin(p, 0.0));
  ASSERT_EQ(r1rt_core_request(p.core, R1RT_REQ_ON_STAND, p.t), R1RT_OK);
  for (int i = 0; i < 50; ++i) {
    p.step();
    ASSERT_TRUE(feed_builtin(p, 0.0));
  }
  ASSERT_EQ(r1rt_core_mode(p.core), R1RT_MODE_TAKEOVER_MIRROR);
  for (int j = 0; j < 12; ++j) ASSERT_EQ(p.core_out.kp[j], 0.0) << j;
  p.core_stalled = true;
  double max_step = 0.0, prev_kp = -1.0;
  std::int64_t t_take = 0;
  for (int i = 0; i < 1200; ++i) {
    p.step();
    if (!p.guard.active()) continue;
    if (t_take == 0) t_take = p.t;
    if (prev_kp >= 0.0) max_step = std::fmax(max_step, std::fabs(p.guard_out.kp[3] - prev_kp));
    prev_kp = p.guard_out.kp[3];
    if (p.t - t_take == 500 * kMs) {
      EXPECT_NEAR(p.guard_out.kp[3], 100.0, 0.5);  // halfway
    }
  }
  EXPECT_LT(max_step, 200.0 / 1000.0 * 1.5);
  for (int j = 0; j < 12; ++j) EXPECT_NEAR(p.guard_out.kp[j], 200.0, 1e-9) << j;
  EXPECT_EQ(p.gstatus().mode_from, R1RT_MODE_TAKEOVER_MIRROR);
  EXPECT_EQ(p.gstatus().policy_steps, 0u);
}

TEST(Guard, ItsCycleRecordsGoToItsOwnRing) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(100);
  auto& ring = p.segment().guard_telemetry;
  EXPECT_EQ(ring.size(), 100u);  // one per guard cycle (standby too)
  r1rt_guard_record rec{};
  std::uint64_t last = 0;
  int balancing = 0;
  while (ring.try_pop(rec)) {
    EXPECT_GT(rec.seq, last);
    last = rec.seq;
    EXPECT_EQ(rec.active, 0);
    balancing += rec.mode_from == R1RT_MODE_BALANCE ? 1 : 0;  // the core's mode as the mirror shows it
  }
  EXPECT_GE(balancing, 98);
  p.core_stalled = true;
  p.run_ms(30);
  bool saw_takeover = false;
  while (ring.try_pop(rec)) {
    if (rec.event == r1rt::kGuardEventTakeover) {
      saw_takeover = true;
      EXPECT_EQ(rec.published, 1);
      EXPECT_EQ(rec.token_owner, R1RT_OWNER_R1GUARD);
      EXPECT_GT(rec.kp[0], 1.0f);
    }
  }
  EXPECT_TRUE(saw_takeover);
  EXPECT_EQ(p.gstatus().telemetry_drops, 0u);
}

TEST(Guard, ItsHeartbeatCarriesTheProcesssRealTimeGrants) {
  Pair p;
  NEED_PAIR(p);
  p.run_ms(2);
  r1rt_heartbeat hb{};
  ASSERT_TRUE(p.segment().guard_heartbeat.try_read(hb));
  EXPECT_EQ(hb.rt_granted, 0);  // lockstep / not granted (session 4 always wrote 0)
  p.guard.set_rt_granted(true);
  p.run_ms(1);
  ASSERT_TRUE(p.segment().guard_heartbeat.try_read(hb));
  EXPECT_EQ(hb.rt_granted, 1);
}

TEST(Guard, NoAllocationPerCycleIncludingTheTakeover) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(100);
  const long before = g_news.load();
  for (int i = 0; i < 300; ++i) {
    p.guard.cycle(&p.st, p.t + i * kMs, &p.guard_out);  // standby with shadow steps
  }
  p.t += 300 * kMs;
  for (int i = 0; i < 1000; ++i) {  // the core silent: takeover + 1 s of policy
    p.t += kMs;
    p.st.t_rx_ns = p.t;
    p.guard.cycle(&p.st, p.t, &p.guard_out);
  }
  EXPECT_EQ(g_news.load() - before, 0);
  EXPECT_TRUE(p.guard.active());
  // cycle cost on this build (information only; Debug + sanitizers, the network at -O2)
  using clock = std::chrono::steady_clock;
  const auto c0 = clock::now();
  for (int i = 0; i < 1000; ++i) {
    p.t += kMs;
    p.st.t_rx_ns = p.t;
    p.guard.cycle(&p.st, p.t, &p.guard_out);
  }
  const double us = std::chrono::duration<double, std::micro>(clock::now() - c0).count() / 1000.0;
  std::printf("guard cycle, active (policy at 50 Hz inside): mean %.2f us\n", us);
}

// The core (writer + its policy iteration in one thread, as in lockstep) and the guard in ANOTHER thread on one
// segment, on the monotonic clock: the shared-memory protocol (seqlocks, the token CAS) under real concurrency - the
// Linux CI's TSan job runs it. Timing is deliberately loose (stale after 100 writer periods = 200 ms) so a loaded host
// cannot make the guard take over before the deliberate 600 ms stall; the 10 ms threshold is the lockstep tests'.
TEST(Guard, TwoThreadsOneSegmentTheTakeoverAndOneWriterAtATime) {
  r1rt::GuardParams gp;
  gp.stale_periods = 100;
  Pair p(gp);
  NEED_PAIR(p);
  using clock = std::chrono::steady_clock;
  const auto t_origin = clock::now();
  auto now_ns = [&] {
    return 1'000 * kMs + std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t_origin).count();
  };
  {
    p.t = now_ns();
    p.st.t_rx_ns = p.t;
    r1rt_core_step(p.core, &p.st, p.t, &p.core_out, &p.tel);
    ASSERT_EQ(r1rt_core_request(p.core, R1RT_REQ_ENTER_BALANCE, p.t), R1RT_OK);
  }
  std::atomic<bool> stall{false}, done{false};
  std::atomic<std::int64_t> core_last_pub{0}, core_first_pub_after_resume{0}, guard_first_pub{0};
  std::atomic<long> core_ticks{0}, guard_cycles{0};
  std::atomic<bool> resumed{false};
  std::thread core_thread([&] {
    r1rt_lowstate_in st = p.st;
    r1rt_lowcmd_out out{};
    r1rt_telemetry_record tel{};
    long k = 0;
    while (!done.load()) {
      if (!stall.load()) {
        const std::int64_t t = now_ns();
        st.t_rx_ns = t;
        r1rt_core_step(p.core, &st, t, &out, &tel);
        if (++k % 10 == 0) r1rt_core_policy_step(p.core, t);
        core_ticks.fetch_add(1);
        if (out.publish) {
          core_last_pub.store(t);
          if (resumed.load() && core_first_pub_after_resume.load() == 0) core_first_pub_after_resume.store(t);
        }
      }
      std::this_thread::sleep_for(std::chrono::microseconds(2000));
    }
  });
  std::thread guard_thread([&] {
    r1rt_lowstate_in st = p.st;
    r1rt_lowcmd_out out{};
    while (!done.load()) {
      const std::int64_t t = now_ns();
      st.t_rx_ns = t;
      p.guard.cycle(&st, t, &out);
      guard_cycles.fetch_add(1);
      if (out.publish && guard_first_pub.load() == 0) guard_first_pub.store(t);
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  const bool early = guard_first_pub.load() != 0;
  stall.store(true);
  const std::int64_t t_stall = now_ns();
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  resumed.store(true);
  stall.store(false);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  done.store(true);
  core_thread.join();
  guard_thread.join();
  ASSERT_FALSE(early) << "the guard took over before the stall (host overloaded?)";
  ASSERT_GT(core_ticks.load(), 100);
  ASSERT_GT(guard_cycles.load(), 200);
  const auto s = p.gstatus();
  EXPECT_EQ(s.takeovers, 1u);
  EXPECT_TRUE(p.guard.active());
  EXPECT_GT(guard_first_pub.load(), t_stall);
  EXPECT_LT(core_last_pub.load(), guard_first_pub.load());   // the core's last write came before the guard's first
  EXPECT_EQ(core_first_pub_after_resume.load(), 0);          // and it never wrote again after its stall
  EXPECT_GE(p.guard.forensics().rt_silence_ns, 200 * kMs);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(p.segment().token.load()), R1RT_OWNER_R1GUARD);
}

// ---- session 6: the operator's damp through the guard, the operator's hand-back (RECLAIM) and the guard re-arming ----

// MONITOR -> ON_STAND (the built-in damping the legs on the stand, as Unitree does there) -> ON_STAND reached.
bool into_on_stand(Pair& p) {
  if (!into_blend(p, 0, R1RT_REQ_ON_STAND, 0.0)) return false;
  p.run_ms(1100);
  return r1rt_core_mode(p.core) == R1RT_MODE_ON_STAND;
}

// One mode request through the segment's mailbox as `requester`; the core answers on its next tick(s).
bool mailbox(Pair& p, std::uint32_t req, std::uint32_t requester, std::int32_t* status, std::int32_t* refusal) {
  std::uint64_t seq = 0;
  if (r1rt_segment_request_mode(p.seg, req, requester, p.t, &seq) != R1RT_OK) return false;
  for (int i = 0; i < 10; ++i) {
    p.step();
    std::uint32_t after = 0;
    if (r1rt_segment_mode_reply(p.seg, seq, status, refusal, &after) == R1RT_OK) return true;
  }
  return false;
}

bool is_damp(const r1rt_lowcmd_out& c, const r1rt_config& cfg) {
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j)
    if (c.kp[j] != 0.0 || c.kd[j] != cfg.damp_kd[j]) return false;
  return true;
}

r1rt_heartbeat guard_beat(Pair& p) {
  r1rt_heartbeat hb{};
  p.segment().guard_heartbeat.try_read(hb);
  return hb;
}

TEST(Guard, HonoursTheOperatorsStopChannelDampWhileItHoldsTheTokenOnStand) {
  // The core stalls in ON_STAND, the guard holds its last command. The operator's `r1 damp` (r1d posts the stop
  // channel's damp level with the operator's id) must reach the motors through the guard; any other requester's damp
  // never does. Session 5's guard ignored the stop channel: the legs stayed stiff (fails on 26e8bce).
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(into_on_stand(p));
  p.core_stalled = true;
  p.run_ms(50);
  ASSERT_TRUE(p.guard.active());
  ASSERT_EQ(p.gstatus().act_mode, R1RT_MODE_ON_STAND);
  ASSERT_TRUE(legs_stiff(p.guard_out));
  p.segment().stop.request(R1RT_STOP_DAMP, R1RT_REQUESTER_R1D, p.t);  // not the operator: ignored
  p.run_ms(5);
  EXPECT_TRUE(legs_stiff(p.guard_out));
  EXPECT_EQ(p.gstatus().damp_ignored, 1u);
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_ON_STAND);
  p.segment().stop.request(R1RT_STOP_DAMP, R1RT_REQUESTER_OPERATOR, p.t);
  p.step();
  EXPECT_EQ(p.guard_out.publish, 1);
  EXPECT_TRUE(is_damp(p.guard_out, p.cfg));  // the same cycle
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_DAMP);
  EXPECT_EQ(p.gstatus().operator_damps, 1u);
  const r1rt_heartbeat hb = guard_beat(p);
  EXPECT_NE(hb.guard_flags & R1RT_GUARD_FLAG_DAMP, 0u);
  EXPECT_NE(hb.guard_flags & R1RT_GUARD_FLAG_ACTIVE, 0u);
  p.run_ms(200);  // it stays damp (the operator's state)
  EXPECT_TRUE(is_damp(p.guard_out, p.cfg));
  EXPECT_EQ(p.two_writers, 0);
}

TEST(Guard, HonoursTheOperatorsDampPostedDuringTheStallBeforeItsTakeover) {
  // A damp posted while the core is already frozen (inside the 10 ms before the takeover): the core never
  // acknowledged it, so the guard acts on it from its first cycles. In BALANCE (the guard's policy) as well.
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(300);
  p.core_stalled = true;
  p.run_ms(3);
  ASSERT_FALSE(p.guard.active());
  p.segment().stop.request(R1RT_STOP_DAMP, R1RT_REQUESTER_OPERATOR, p.t);
  p.run_ms(20);
  ASSERT_TRUE(p.guard.active());
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_DAMP);
  EXPECT_TRUE(is_damp(p.guard_out, p.cfg));
}

TEST(Guard, AnOldDampRequestTheCoreHandledNeverActsAtALaterTakeover) {
  // The operator's damp in MONITOR (the core refuses it: nothing of ours to damp, but acknowledges it); later the core
  // balances and stalls: the takeover must not replay that old request.
  Pair p;
  NEED_PAIR(p);
  p.run_ms(10);
  p.segment().stop.request(R1RT_STOP_DAMP, R1RT_REQUESTER_OPERATOR, p.t);
  p.run_ms(10);
  ASSERT_EQ(p.segment().stop.level[R1RT_STOP_DAMP].acked.load(), 1u);
  ASSERT_EQ(r1rt_core_mode(p.core), R1RT_MODE_MONITOR);
  ASSERT_TRUE(p.enter());
  p.run_ms(300);
  p.core_stalled = true;
  p.run_ms(50);
  ASSERT_TRUE(p.guard.active());
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_BALANCE);
  EXPECT_EQ(p.gstatus().operator_damps, 0u);
  EXPECT_TRUE(legs_stiff(p.guard_out));
}

TEST(Guard, FollowsALiveCoreTheOperatorDampedThroughTheMailbox) {
  // After SIGCONT the core is alive but no longer the owner; the operator's DAMP through the mailbox puts the core in
  // DAMP (it publishes nothing): the guard, still the writer, follows into damp.
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(into_on_stand(p));
  p.core_stalled = true;
  p.run_ms(50);
  ASSERT_TRUE(p.guard.active());
  p.core_stalled = false;  // SIGCONT
  p.run_ms(20);
  EXPECT_EQ(p.core_out.publish, 0);
  std::int32_t status = 0, refusal = 0;
  ASSERT_TRUE(mailbox(p, R1RT_REQ_DAMP, R1RT_REQUESTER_R1D, &status, &refusal));
  EXPECT_EQ(refusal, R1RT_REFUSE_NOT_OPERATOR);
  p.run_ms(5);
  EXPECT_TRUE(legs_stiff(p.guard_out));
  ASSERT_TRUE(mailbox(p, R1RT_REQ_DAMP, R1RT_REQUESTER_OPERATOR, &status, &refusal));
  EXPECT_EQ(status, R1RT_OK);
  EXPECT_EQ(r1rt_core_mode(p.core), R1RT_MODE_DAMP);
  p.run_ms(3);
  EXPECT_TRUE(is_damp(p.guard_out, p.cfg));
  EXPECT_EQ(p.core_out.publish, 0);  // still one writer: the guard
  EXPECT_EQ(p.two_writers, 0);
}

TEST(Guard, TheOperatorsReclaimOnStandIsBumplessAndTheGuardReArmsAndCatchesTheNextStall) {
  // HW1's T6 in lockstep: SIGSTOP the core in ON_STAND -> the guard holds -> SIGCONT: the core silent (not the owner)
  // -> the operator's RECLAIM through the mailbox -> the core publishes again from exactly what the guard sent (the
  // first command equals the guard's last; the blend ends in the ON_STAND hold), the guard stops writing and re-arms
  // -> a second stall is taken over again. Session 5 had no hand-back (R1RT_REQ_RECLAIM unknown: fails on 26e8bce).
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(into_on_stand(p));
  const r1rt_lowcmd_out core_last = p.core_out;
  p.core_stalled = true;
  p.run_ms(100);
  ASSERT_TRUE(p.guard.active());
  p.core_stalled = false;  // SIGCONT
  p.run_ms(50);
  ASSERT_EQ(p.core_out.publish, 0);
  ASSERT_EQ(p.guard_out.publish, 1);
  const r1rt_lowcmd_out guard_last = p.guard_out;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) ASSERT_NEAR(guard_last.q[j], core_last.q[j], 1e-12) << j;
  // not the operator: refused, nothing changes
  std::int32_t status = 0, refusal = 0;
  ASSERT_TRUE(mailbox(p, R1RT_REQ_RECLAIM, R1RT_REQUESTER_R1D, &status, &refusal));
  EXPECT_EQ(status, R1RT_E_REFUSED);
  EXPECT_EQ(refusal, R1RT_REFUSE_NOT_OPERATOR);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(p.segment().token.load()), R1RT_OWNER_R1GUARD);
  EXPECT_EQ(p.guard_out.publish, 1);
  // the operator: the core's tick takes the token back and publishes in that same step; the guard re-arms in it
  std::uint64_t seq = 0;
  ASSERT_EQ(r1rt_segment_request_mode(p.seg, R1RT_REQ_RECLAIM, R1RT_REQUESTER_OPERATOR, p.t, &seq), R1RT_OK);
  int steps = 0;
  while (p.core_out.publish == 0 && steps < 10) {
    p.step();
    ++steps;
  }
  ASSERT_EQ(p.core_out.publish, 1);
  std::uint32_t after = 0;
  ASSERT_EQ(r1rt_segment_mode_reply(p.seg, seq, &status, &refusal, &after), R1RT_OK);
  EXPECT_EQ(status, R1RT_OK);
  EXPECT_EQ(after, static_cast<std::uint32_t>(R1RT_MODE_TAKEOVER_BLEND));
  EXPECT_EQ(p.guard_out.publish, 0);  // the same step: the guard saw the token gone
  EXPECT_FALSE(p.guard.active());
  EXPECT_EQ(p.gstatus().rearms, 1u);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(p.segment().token.load()), R1RT_OWNER_R1RT);
  double max_dq = 0.0, max_dkp = 0.0, max_dkd = 0.0;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    max_dq = std::fmax(max_dq, std::fabs(p.core_out.q[j] - guard_last.q[j]));
    max_dkp = std::fmax(max_dkp, std::fabs(p.core_out.kp[j] - guard_last.kp[j]));
    max_dkd = std::fmax(max_dkd, std::fabs(p.core_out.kd[j] - guard_last.kd[j]));
    // session 8 (fix #28): the guard's hold carries the core's last arm / head feed-forward (none on legs / waist);
    // the blend starts from exactly that
    EXPECT_NEAR(p.core_out.tau_ff[j], guard_last.tau_ff[j], 1e-12) << j;
    if (j < 14) {  // braces: GCC 9 -Werror=dangling-else (the gtest macro is an if / else; the Linux build check)
      EXPECT_EQ(p.core_out.tau_ff[j], 0.0) << j;
    }
  }
  EXPECT_LT(max_dq, 1e-12);  // bumpless: the first command after the hand-back IS the guard's last
  EXPECT_LT(max_dkp, 1e-9);
  EXPECT_LT(max_dkd, 1e-9);
  // the blend into the ON_STAND hold: no position step anywhere, legs stiff throughout, then ON_STAND
  r1rt_lowcmd_out prev = p.core_out;
  double step_max = 0.0;
  for (int i = 0; i < 1100; ++i) {
    p.step();
    if (p.core_out.publish) {
      ASSERT_TRUE(legs_stiff(p.core_out));
      for (int j = 0; j < R1RT_NUM_JOINTS; ++j) step_max = std::fmax(step_max, std::fabs(p.core_out.q[j] - prev.q[j]));
      prev = p.core_out;
    }
    ASSERT_EQ(p.guard_out.publish, 0);
  }
  EXPECT_LT(step_max, 1e-6);
  EXPECT_EQ(r1rt_core_mode(p.core), R1RT_MODE_ON_STAND);
  EXPECT_EQ(p.cstatus().reclaims, 1u);
  // re-armed: the next stall is caught again (the second T6)
  const r1rt_lowcmd_out core_last2 = p.core_out;
  p.core_stalled = true;
  p.run_ms(50);
  ASSERT_TRUE(p.guard.active());
  EXPECT_EQ(p.gstatus().takeovers, 2u);
  EXPECT_EQ(p.gstatus().act_mode, R1RT_MODE_ON_STAND);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) EXPECT_NEAR(p.guard_out.q[j], core_last2.q[j], 1e-12) << j;
  EXPECT_EQ(p.two_writers, 0);
}

TEST(Guard, ReclaimIsRefusedUnlessTheGuardHoldsTheTokenAndTheCoreIsOnStand) {
  // From BALANCE (the guard balances with its own policy): a hand-back between two balancing policies is new and
  // unproven (DESIGN §4.4) - refused; with the core owning the token: nothing to take back.
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(p.enter());
  p.run_ms(300);
  std::int32_t status = 0, refusal = 0;
  ASSERT_TRUE(mailbox(p, R1RT_REQ_RECLAIM, R1RT_REQUESTER_OPERATOR, &status, &refusal));
  EXPECT_EQ(status, R1RT_E_REFUSED);
  EXPECT_EQ(refusal, R1RT_REFUSE_TOKEN);
  p.core_stalled = true;
  p.run_ms(50);
  ASSERT_TRUE(p.guard.active());
  p.core_stalled = false;
  p.run_ms(20);
  ASSERT_TRUE(mailbox(p, R1RT_REQ_RECLAIM, R1RT_REQUESTER_OPERATOR, &status, &refusal));
  EXPECT_EQ(status, R1RT_E_REFUSED);
  EXPECT_EQ(refusal, R1RT_REFUSE_MODE);
  EXPECT_EQ(p.cstatus().reclaim_refusal, R1RT_REFUSE_MODE);
  EXPECT_TRUE(p.guard.active());  // the guard keeps balancing
  EXPECT_EQ(p.guard_out.publish, 1);
  EXPECT_EQ(p.core_out.publish, 0);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(p.segment().token.load()), R1RT_OWNER_R1GUARD);
}

TEST(Guard, ReclaimNeedsFreshState) {
  Pair p;
  NEED_PAIR(p);
  ASSERT_TRUE(into_on_stand(p));
  p.core_stalled = true;
  p.run_ms(50);
  ASSERT_TRUE(p.guard.active());
  p.core_stalled = false;
  p.st.t_rx_ns = 0;
  // the core and the guard without state for 150 ms (> the 100 ms stale threshold)
  for (int i = 0; i < 150; ++i) {
    p.t += kMs;
    if ((p.t / kMs) % 2 == 0) r1rt_core_step(p.core, nullptr, p.t, &p.core_out, &p.tel);
    p.guard.cycle(nullptr, p.t, &p.guard_out);
  }
  ASSERT_EQ(r1rt_core_request(p.core, R1RT_REQ_RECLAIM, p.t), R1RT_E_REFUSED);
  EXPECT_EQ(p.cstatus().reclaim_refusal, R1RT_REFUSE_STATE);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(p.segment().token.load()), R1RT_OWNER_R1GUARD);
}

TEST(Guard, ItsHeartbeatSaysArmedOrObserveOnlyAndWhatItTookOverFrom) {
  Pair p;
  NEED_PAIR(p);
  p.run_ms(5);
  EXPECT_EQ(guard_beat(p).guard_flags, R1RT_GUARD_FLAG_ARMED);
  EXPECT_EQ(p.gstatus().armed, 1);
  ASSERT_TRUE(into_on_stand(p));
  p.core_stalled = true;
  p.run_ms(50);
  const std::uint8_t f = guard_beat(p).guard_flags;
  EXPECT_EQ(f & 0x0Fu, R1RT_GUARD_FLAG_ARMED | R1RT_GUARD_FLAG_ACTIVE);
  EXPECT_EQ(f >> R1RT_GUARD_FLAG_MODE_FROM_SHIFT, static_cast<unsigned>(R1RT_MODE_ON_STAND));
  r1rt::GuardParams obs;
  obs.takeover_enabled = 0;
  Pair q(obs);
  NEED_PAIR(q);
  q.run_ms(5);
  EXPECT_EQ(guard_beat(q).guard_flags, 0u);  // observe only: not armed
  EXPECT_EQ(q.gstatus().armed, 0);
}

TEST(Guard, ProcForensicsOnLinuxOnly) {
  r1rt::GuardForensics f{};
#if defined(__linux__)
  ASSERT_EQ(r1rt::capture_proc_forensics(static_cast<std::int32_t>(::getpid()), &f), 0);
  EXPECT_EQ(f.proc_err, 0);
  ASSERT_GE(f.n_threads, 1);
  EXPECT_GT(f.threads[0].tid, 0);
  EXPECT_GT(std::strlen(f.threads[0].comm), 0u);
  EXPECT_GT(std::strlen(f.threads[0].syscall), 0u);  // "running" or a syscall number + registers
#else
  EXPECT_EQ(r1rt::capture_proc_forensics(1, &f), ENOSYS);
  EXPECT_EQ(f.proc_err, ENOSYS);
  EXPECT_EQ(f.n_threads, 0);
  GTEST_SKIP() << "no /proc task files on this OS: the capture runs on Linux (CI) only";
#endif
}

}  // namespace
