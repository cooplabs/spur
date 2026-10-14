// The real-time core (r1rt::Core through the step C ABI) on a synthetic state stream - no simulator: modes and
// requests, the policy thread vs the inline rung (rung 1), STAND_FREEZE on an invalid action (rung 3), the lowstate
// age thresholds, limits, the ownership token, the stop channel, the hg image + CRC, and no allocation per tick.
// The same behaviours on the simulated robot: tests/integration/test_v3_core.py.
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include "r1rt/crc32.hpp"
#include "r1rt/executor.hpp"
#include "r1rt/hg_msgs.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"
#include "test_core_rig.hpp"
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

TEST(Core, MonitorPublishesNothingAndBalanceNeedsAPolicyAndFreshState) {
  Rig r;
  NEED_RIG(r);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(r.tick(), R1RT_OK);
    EXPECT_EQ(r.out.publish, 0);
  }
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_MONITOR);
  EXPECT_EQ(r1rt_core_request(r.core, R1RT_REQ_DAMP, r.t), R1RT_E_REFUSED);  // nothing of ours to damp
  // stale state: refused
  for (int i = 0; i < 60; ++i) r.tick(/*withhold=*/true);
  EXPECT_EQ(r1rt_core_request(r.core, R1RT_REQ_ENTER_BALANCE, r.t), R1RT_E_REFUSED);
  ASSERT_TRUE(r.enter());
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(r.segment().token.load()), R1RT_OWNER_R1RT);

  Rig none(/*with_policy=*/false);
  ASSERT_TRUE(none.core != nullptr);
  none.tick();
  EXPECT_EQ(r1rt_core_request(none.core, R1RT_REQ_ENTER_BALANCE, none.t), R1RT_E_REFUSED);
}

TEST(Core, BalanceCommandIsThePolicyTargetPlusTheUpperHold) {
  Rig r;
  NEED_RIG(r);
  r.st.q[15] = 0.31;  // an arm away from the default: HOLD keeps the entry pose
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 100; ++i) ASSERT_EQ(r.tick(), R1RT_OK);
  EXPECT_EQ(r.out.publish, 1);
  EXPECT_EQ(r.out.mode_machine, 1);
  EXPECT_EQ(r.out.mode_pr, 0);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) EXPECT_EQ(r.out.enable[j], 1);
  for (int j = 0; j < 14; ++j) {
    EXPECT_EQ(r.out.kp[j], r.g.params.kp[j]);
    EXPECT_EQ(r.out.kd[j], r.g.params.kd[j]);
    EXPECT_LT(std::fabs(r.out.q[j] - r.g.params.default_pos[j]), 0.5);  // a policy target near the stance
  }
  EXPECT_EQ(r.out.q[15], 0.31);
  EXPECT_EQ(r.out.kp[15], 60.0);
  const auto s = r.status();
  EXPECT_EQ(s.inline_runs, 0u);  // the thread kept up: never inline
  EXPECT_EQ(s.policy_thread_runs, 10u);
  EXPECT_EQ(s.faults_seen & R1RT_FAULT_POLICY_LATE, 0u);
  EXPECT_EQ(s.faults_seen & (R1RT_FAULT_LOWSTATE_LATE | R1RT_FAULT_LOWSTATE_STALE), 0u);
}

TEST(Core, AStalledPolicyThreadIsReplacedInlineThenHandedBack) {
  // The A7 stall (4.018 s, the 9 Oct fall) moved into v3's policy thread: the writer runs the policy itself.
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 50; ++i) r.tick();
  const auto before = r.status();
  const std::int64_t t_last_thread = before.last_action_t_ns;
  int first_inline_tick = -1;
  const int stall_ticks = 2009;  // 4.018 s at 500 Hz
  for (int i = 0; i < stall_ticks; ++i) {
    r.tick(false, /*thread=*/false);
    ASSERT_EQ(r.out.publish, 1);
    if (first_inline_tick < 0 && r.status().inline_runs > 0) first_inline_tick = i;
  }
  const auto during = r.status();
  ASSERT_GE(first_inline_tick, 0);
  const std::int64_t age_at_inline = r.t - static_cast<std::int64_t>(stall_ticks - 1 - first_inline_tick) * kTick -
                                     t_last_thread;
  EXPECT_GT(age_at_inline, 40 * kMs);
  EXPECT_LE(age_at_inline, 40 * kMs + kTick);  // within one writer tick of the 40 ms threshold
  EXPECT_EQ(during.inline_entries, 1u);
  EXPECT_NEAR(static_cast<double>(during.inline_runs), 4.018 / 0.02, 3.0);  // every policy period
  EXPECT_TRUE(during.inline_active);
  EXPECT_EQ(during.mode, R1RT_MODE_BALANCE);
  // the thread is back: two beats, then the writer hands back and the thread computes again
  for (int i = 0; i < 60; ++i) r.tick();
  const auto after = r.status();
  EXPECT_FALSE(after.inline_active);
  EXPECT_GT(after.policy_thread_runs, during.policy_thread_runs);
  EXPECT_EQ(after.inline_entries, 1u);
  EXPECT_EQ(after.mode, R1RT_MODE_BALANCE);
}

TEST(Core, AnInvalidActionFreezesIntoTheStandNeverDamps) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 25; ++i) r.tick();
  ASSERT_EQ(r1rt_core_inject(r.core, R1RT_INJECT_POLICY_NAN, r.t, r.t + 100 * kMs, 0.0), R1RT_OK);
  int frozen_at = -1;
  for (int i = 0; i < 400; ++i) {  // 0.8 s
    r.tick();
    ASSERT_EQ(r.out.publish, 1);
    ASSERT_NE(r1rt_core_mode(r.core), R1RT_MODE_DAMP);
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) ASSERT_TRUE(std::isfinite(r.out.q[j]));
    if (frozen_at < 0 && r1rt_core_mode(r.core) == R1RT_MODE_STAND_FREEZE) frozen_at = i;
  }
  ASSERT_GE(frozen_at, 0);
  EXPECT_LE(frozen_at, 11);  // the next policy step produced the NaN, the next writer tick froze
  // after the 0.4 s ramp: the stand pose at stand gains (legs + waist); the arms still HOLD
  for (int j = 0; j < 14; ++j) {
    EXPECT_NEAR(r.out.q[j], r.cfg.stand_q[j], 1e-12) << j;
    EXPECT_EQ(r.out.kp[j], 200.0);
  }
  EXPECT_EQ(r.out.kp[14], 60.0);
  const auto s = r.status();
  EXPECT_EQ(s.freeze_entries, 1u);
  EXPECT_GE(s.invalid_actions, 1u);
  EXPECT_TRUE(s.faults & R1RT_FAULT_POLICY_INVALID);  // the reason stays visible while frozen
  // the injection is over, but the freeze holds until the operator (no automatic return)
  for (int i = 0; i < 100; ++i) r.tick();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_STAND_FREEZE);
  EXPECT_EQ(r1rt_core_request(r.core, R1RT_REQ_ENTER_BALANCE, r.t), R1RT_OK);  // the operator's call
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
}

TEST(Core, LowstateAgeLateStaleLostKeepsPublishing) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 30; ++i) r.tick();
  // 15 ms: late, not stale
  std::uint32_t seen = 0;
  for (int i = 0; i < 8; ++i) {
    r.tick(true);
    seen |= r.status().faults;
  }
  EXPECT_TRUE(seen & R1RT_FAULT_LOWSTATE_LATE);
  EXPECT_FALSE(seen & R1RT_FAULT_LOWSTATE_STALE);
  r.tick();
  EXPECT_EQ(r.status().faults & R1RT_FAULT_LOWSTATE_LATE, 0u);
  for (int i = 0; i < 30; ++i) r.tick();
  // 150 ms: stale -> the last balanced command is resent unchanged
  r1rt_lowcmd_out last_fresh{};
  seen = 0;
  for (int i = 0; i < 75; ++i) {
    r.tick(true);
    const std::uint32_t f = r.status().faults;
    seen |= f;
    if (!(f & R1RT_FAULT_LOWSTATE_STALE)) last_fresh = r.out;
    if (f & R1RT_FAULT_LOWSTATE_STALE) {
      ASSERT_EQ(r.out.publish, 1);
      for (int j = 0; j < R1RT_NUM_JOINTS; ++j) ASSERT_EQ(r.out.q[j], last_fresh.q[j]);
    }
  }
  EXPECT_TRUE(seen & R1RT_FAULT_LOWSTATE_STALE);
  EXPECT_FALSE(seen & R1RT_FAULT_LOWSTATE_LOST);
  EXPECT_GT(r.status().stale_resends, 0u);
  for (int i = 0; i < 30; ++i) r.tick();
  EXPECT_EQ(r.status().faults & R1RT_FAULT_LOWSTATE_STALE, 0u);
  // 1.5 s: lost - a fault, still publishing the last balanced command, never damping
  seen = 0;
  for (int i = 0; i < 750; ++i) {
    r.tick(true);
    seen |= r.status().faults;
    ASSERT_EQ(r.out.publish, 1);
    ASSERT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  }
  EXPECT_TRUE(seen & R1RT_FAULT_LOWSTATE_LOST);
  for (int i = 0; i < 30; ++i) r.tick();  // state is back: the policy runs again
  const auto s = r.status();
  EXPECT_EQ(s.faults & (R1RT_FAULT_LOWSTATE_LOST | R1RT_FAULT_LOWSTATE_STALE), 0u);
  EXPECT_EQ(s.mode, R1RT_MODE_BALANCE);
}

TEST(Core, DampOnlyByTheOperatorThroughTheStopChannel) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 20; ++i) r.tick();
  r.segment().stop.request(R1RT_STOP_STOP, 7, r.t);
  r.tick();
  EXPECT_TRUE(r.status().stop_active);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);  // stop keeps balancing
  EXPECT_EQ(r.segment().stop.level[R1RT_STOP_STOP].acked.load(), 1u);
  EXPECT_EQ(r1rt_core_request(r.core, R1RT_REQ_RESUME, r.t), R1RT_OK);
  // session 6: the stop channel's damp level is OPERATOR ONLY too (DESIGN §3.5 / §5.1) - until session 6 any
  // client's damp-level request (here requester 7; r1d's id 3 likewise) limped the robot (fails on 26e8bce)
  r.segment().stop.request(R1RT_STOP_DAMP, 7, r.t);
  r.tick();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  EXPECT_EQ(r.status().damp_ignored, 1u);
  EXPECT_EQ(r.segment().stop.level[R1RT_STOP_DAMP].acked.load(), 1u);  // acknowledged (seen), not acted on
  for (int j = 0; j < 12; ++j) EXPECT_GT(r.out.kp[j], 0.0) << j;
  r.segment().stop.request(R1RT_STOP_DAMP, R1RT_REQUESTER_R1D, r.t);
  r.tick();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  EXPECT_EQ(r.status().damp_ignored, 2u);
  r.segment().stop.request(R1RT_STOP_DAMP, R1RT_REQUESTER_OPERATOR, r.t);
  r.tick();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_DAMP);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_EQ(r.out.kp[j], 0.0);
    EXPECT_EQ(r.out.kd[j], 3.0);
  }
  EXPECT_EQ(r.out.publish, 1);
}

TEST(Core, LimitsClampAndCount) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 20; ++i) r.tick();
  // an action of 40 (inside abs_max 50): target = default + 0.178 * 40 = ~6.8 rad -> clamped to +-2.5
  ASSERT_EQ(r1rt_core_inject(r.core, R1RT_INJECT_POLICY_OUT_OF_RANGE, r.t, r.t + 30 * kMs, 40.0), R1RT_OK);
  std::uint32_t seen = 0;
  for (int i = 0; i < 20; ++i) {
    r.tick();
    seen |= r.status().faults;
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) ASSERT_LE(std::fabs(r.out.q[j]), 2.5);
  }
  EXPECT_TRUE(seen & R1RT_FAULT_JOINT_CLAMP);
  EXPECT_GT(r.status().clamp_total, 0u);
  // beyond abs_max: invalid -> STAND_FREEZE
  ASSERT_EQ(r1rt_core_inject(r.core, R1RT_INJECT_POLICY_OUT_OF_RANGE, r.t, r.t + 30 * kMs, 80.0), R1RT_OK);
  for (int i = 0; i < 20; ++i) r.tick();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_STAND_FREEZE);
}

TEST(Core, NoWriteWithoutTheOwnershipToken) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  r.tick();
  EXPECT_EQ(r.out.publish, 1);
  std::uint64_t w = r.segment().token.load();
  ASSERT_TRUE(r.segment().token.try_transfer(w, R1RT_OWNER_R1GUARD));  // r1guard took over
  r.tick();
  EXPECT_EQ(r.out.publish, 0);
  EXPECT_TRUE(r.status().faults & R1RT_FAULT_NOT_OWNER);
}

TEST(Core, LoopGapIsCounted) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  r.tick();
  r.t += 10 * kMs;  // the writer slept 10 ms too long
  r.tick();
  EXPECT_TRUE(r.status().faults & R1RT_FAULT_LOOP_GAP);
  EXPECT_EQ(r.status().loop_gaps, 1u);
}

TEST(Core, HgImageMapsTheSlotsAndCarriesUnitreesCrc) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 12; ++i) r.tick();
  r1rt::hg::LowCmd m;
  const std::uint32_t crc = r1rt_fill_hg_lowcmd(&r.out, &m, sizeof m);
  EXPECT_EQ(crc, m.crc);
  EXPECT_EQ(crc, r1rt::hg_crc(m));
  EXPECT_EQ(m.mode_machine, 1);
  const int slots[R1RT_NUM_JOINTS] = R1RT_HG_SLOT_MAP_INIT;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_EQ(m.motor_cmd[slots[j]].q, static_cast<float>(r.out.q[j]));
    EXPECT_EQ(m.motor_cmd[slots[j]].kp, static_cast<float>(r.out.kp[j]));
    EXPECT_EQ(m.motor_cmd[slots[j]].mode, 1);
  }
  for (int s : {14, 20, 21, 27, 28, 31, 32, 33, 34}) EXPECT_EQ(m.motor_cmd[s].mode, 0);
  m.motor_cmd[0].q += 1e-3f;
  EXPECT_NE(r1rt::hg_crc(m), crc);
  EXPECT_EQ(r1rt_fill_hg_lowcmd(&r.out, &m, 10), 0u);  // too small: refused
}

// ---- the upper-body executor inside the writer (session 3 `wire`) ----

// A right-arm TRAJECTORY record: one quintic from the arm's current target by `delta` on every joint over `dur_ns`
// (8 knots, exact samples of q0 + delta (10 s^3 - 15 s^4 + 6 s^5)).
r1rt_upper_cmd right_arm_move(const r1rt_lowcmd_out& last, double delta, std::int64_t dur_ns, std::uint64_t id,
                              std::uint64_t epoch, std::int64_t t_issue) {
  r1rt_upper_cmd c{};
  c.cmd_id = id;
  c.stop_epoch = epoch;
  c.t_issue_ns = t_issue;
  c.ttl_ns = 50 * kMs;
  c.group = R1RT_GROUP_ARM_RIGHT;
  c.source = R1RT_SOURCE_TRAJECTORY;
  c.n_joints = 5;
  c.n_knots = R1RT_MAX_KNOTS;
  c.last_segment = 1;
  const double T = static_cast<double>(dur_ns) * 1e-9;
  for (int i = 0; i < R1RT_MAX_KNOTS; ++i) {
    const double s = static_cast<double>(i) / (R1RT_MAX_KNOTS - 1);
    c.knots[i].t_ns = static_cast<std::int64_t>(std::llround(s * static_cast<double>(dur_ns)));
    for (int k = 0; k < 5; ++k) {
      c.knots[i].q[k] = last.q[19 + k] + delta * (10 * s * s * s - 15 * s * s * s * s + 6 * s * s * s * s * s);
      c.knots[i].dq[k] = delta * (30 * s * s - 60 * s * s * s + 30 * s * s * s * s) / T;
      c.knots[i].ddq[k] = delta * (60 * s - 180 * s * s + 120 * s * s * s) / (T * T);
    }
  }
  return c;
}

TEST(Core, UpperTrajectoryFromTheRingFollowsTheKnotsAndTheDeadmanStopsIt) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 10; ++i) r.tick();
  EXPECT_EQ(r.status().exec_ready, 1);
  const r1rt_lowcmd_out before = r.out;
  const std::uint64_t epoch = r1rt_segment_stop_epoch(r.seg);
  const r1rt_upper_cmd c = right_arm_move(before, 0.2, 1000 * kMs, 7, epoch, r.t);
  ASSERT_EQ(r1rt_segment_push_upper_cmd(r.seg, &c), R1RT_OK);
  ASSERT_EQ(r1rt_segment_upper_keepalive(r.seg, R1RT_GROUP_ARM_RIGHT, 7, r.t), R1RT_OK);
  const std::int64_t t_start = r.t + kTick;  // received (and started) at the next writer tick
  // r1d renews every 20 ms for 0.5 s: the arm follows the quintic exactly (the knots sample it)
  double max_err = 0.0;
  for (int i = 0; i < 250; ++i) {
    if (i % 10 == 0) r1rt_segment_upper_keepalive(r.seg, R1RT_GROUP_ARM_RIGHT, 7, r.t + kTick);
    r.tick();
    const double s = static_cast<double>(r.t - t_start) * 1e-9 / 1.0;
    const double want = before.q[19] + 0.2 * (10 * s * s * s - 15 * s * s * s * s + 6 * s * s * s * s * s);
    max_err = std::fmax(max_err, std::fabs(r.out.q[19] - want));
    EXPECT_EQ(r.out.kp[19], 60.0);
  }
  auto st = r.status();
  EXPECT_EQ(st.upper_accepted, 1u);
  EXPECT_EQ(st.upper_source[R1RT_GROUP_ARM_RIGHT], R1RT_SOURCE_TRAJECTORY);
  EXPECT_EQ(st.upper_cmd_id[R1RT_GROUP_ARM_RIGHT], 7u);
  EXPECT_LT(max_err, 1e-9);
  // no more keepalives: the deadman (50 ms) stops it within stop_time (0.15 s), smoothly, then HOLD
  const double q_at = r.out.q[19];
  double last = q_at;
  bool moved_back = false;
  std::uint8_t ev = 0;
  for (int i = 0; i < 150; ++i) {
    r.tick();
    if (r.tel.guard_event != 0 && ev == 0) ev = r.tel.guard_event;
    moved_back = moved_back || r.out.q[19] < last - 1e-12;  // the move was increasing: a stop never reverses
    last = r.out.q[19];
  }
  st = r.status();
  EXPECT_EQ(ev, 16);  // kEventDeadman, in the telemetry record's guard_event
  EXPECT_EQ(st.upper_event[R1RT_GROUP_ARM_RIGHT], 16);
  EXPECT_EQ(st.upper_source[R1RT_GROUP_ARM_RIGHT], R1RT_SOURCE_HOLD);
  EXPECT_EQ(st.upper_stopping[R1RT_GROUP_ARM_RIGHT], 0);
  EXPECT_FALSE(moved_back);
  EXPECT_LT(last, before.q[19] + 0.2);  // stopped short of the goal
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  // the other groups never moved
  for (int j = 14; j < 19; ++j) EXPECT_EQ(r.out.q[j], before.q[j]);
}

TEST(Core, RecordsIssuedBeforeAStopAreRefusedAndTheStopKeepsBalancing) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 10; ++i) r.tick();
  const std::uint64_t old_epoch = r1rt_segment_stop_epoch(r.seg);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_STOP, r.t), R1RT_OK);  // posted on the stop channel
  EXPECT_EQ(r1rt_segment_stop_epoch(r.seg), old_epoch + 1);
  r.tick();
  EXPECT_EQ(r.status().upper_stop_epoch, old_epoch + 1);
  const r1rt_upper_cmd stale = right_arm_move(r.out, 0.1, 500 * kMs, 3, old_epoch, r.t);
  ASSERT_EQ(r1rt_segment_push_upper_cmd(r.seg, &stale), R1RT_OK);
  r.tick();
  auto st = r.status();
  EXPECT_EQ(st.upper_rejected, 1u);
  EXPECT_EQ(st.upper_last_reject, 7);  // kRejectStopEpoch
  EXPECT_EQ(st.stop_active, 1);
  const r1rt_upper_cmd fresh = right_arm_move(r.out, 0.1, 500 * kMs, 4, r1rt_segment_stop_epoch(r.seg), r.t);
  ASSERT_EQ(r1rt_segment_push_upper_cmd(r.seg, &fresh), R1RT_OK);
  r1rt_segment_upper_keepalive(r.seg, R1RT_GROUP_ARM_RIGHT, 4, r.t + kTick);
  r.tick();
  st = r.status();
  EXPECT_EQ(st.upper_accepted, 1u);
  EXPECT_EQ(st.upper_source[R1RT_GROUP_ARM_RIGHT], R1RT_SOURCE_TRAJECTORY);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  // the operator's stop through the channel stops a running group
  for (int i = 0; i < 20; ++i) {
    if (i % 10 == 0) r1rt_segment_upper_keepalive(r.seg, R1RT_GROUP_ARM_RIGHT, 4, r.t + kTick);
    r.tick();
  }
  r.segment().stop.request(R1RT_STOP_HALT, 9, r.t);
  for (int i = 0; i < 100; ++i) {
    r1rt_segment_upper_keepalive(r.seg, R1RT_GROUP_ARM_RIGHT, 4, r.t + kTick);  // keepalives cannot revive it
    r.tick();
  }
  st = r.status();
  EXPECT_EQ(st.upper_event[R1RT_GROUP_ARM_RIGHT], 18);  // kEventOperatorStop
  EXPECT_EQ(st.upper_source[R1RT_GROUP_ARM_RIGHT], R1RT_SOURCE_HOLD);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
}

TEST(Core, StandFreezeStopsTheArmsThroughTheStopChannel) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 10; ++i) r.tick();
  const r1rt_upper_cmd c = right_arm_move(r.out, 0.3, 1000 * kMs, 5, r1rt_segment_stop_epoch(r.seg), r.t);
  ASSERT_EQ(r1rt_segment_push_upper_cmd(r.seg, &c), R1RT_OK);
  const std::uint64_t epoch0 = r1rt_segment_stop_epoch(r.seg);
  r1rt_core_inject(r.core, R1RT_INJECT_POLICY_NAN, r.t + 200 * kMs, r.t + 220 * kMs, 0.0);
  for (int i = 0; i < 300; ++i) {
    if (i % 10 == 0) r1rt_segment_upper_keepalive(r.seg, R1RT_GROUP_ARM_RIGHT, 5, r.t + kTick);
    r.tick();
  }
  const auto st = r.status();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_STAND_FREEZE);
  EXPECT_EQ(st.freeze_entries, 1u);
  EXPECT_EQ(r1rt_segment_stop_epoch(r.seg), epoch0 + 1);  // the core's own stop went through the channel
  EXPECT_EQ(st.upper_stop_epoch, epoch0 + 1);
  EXPECT_EQ(st.upper_event[R1RT_GROUP_ARM_RIGHT], 18);
  EXPECT_EQ(st.upper_source[R1RT_GROUP_ARM_RIGHT], R1RT_SOURCE_HOLD);
  EXPECT_EQ(st.upper_stopping[R1RT_GROUP_ARM_RIGHT], 0);
  EXPECT_EQ(r.out.publish, 1);
  for (int j = 19; j < 24; ++j) EXPECT_EQ(r.out.kp[j], 60.0);  // HOLD gains, never limp
}

TEST(Core, NoAllocationInTheWriterOrThePolicyThread) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 20; ++i) r.tick();
  const long before = g_news.load();
  for (int i = 0; i < 1000; ++i) r.tick();  // incl. 100 policy-thread iterations
  for (int i = 0; i < 100; ++i) r.tick(false, false);  // incl. inline runs
  // the executor in the loop: a trajectory from the ring, keepalives, a stop through the channel
  const r1rt_upper_cmd c = right_arm_move(r.out, 0.2, 400 * kMs, 11, r1rt_segment_stop_epoch(r.seg), r.t);
  r1rt_segment_push_upper_cmd(r.seg, &c);
  for (int i = 0; i < 100; ++i) {
    if (i % 10 == 0) r1rt_segment_upper_keepalive(r.seg, R1RT_GROUP_ARM_RIGHT, 11, r.t + kTick);
    r.tick();
  }
  r1rt_core_request(r.core, R1RT_REQ_STOP, r.t);
  for (int i = 0; i < 100; ++i) r.tick();
  EXPECT_EQ(g_news.load() - before, 0);
  EXPECT_GT(r.status().inline_runs, 0u);
  EXPECT_EQ(r.status().upper_accepted, 1u);
}

TEST(Core, NoAllocationInTheTakeover) {
  // ABI 4: the built-in's command, the gates, MIRROR, RELEASED, BLEND and BALANCE (and an ABORT) allocate nothing
  Rig r;
  NEED_RIG(r);
  r1rt_lowcmd_out b{};
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    b.enable[j] = 1;
    b.q[j] = r.st.q[j] + 0.02;
    b.kp[j] = 200.0;
    b.kd[j] = 3.0;
  }
  for (int i = 0; i < 5; ++i) r.tick();
  const long before = g_news.load();
  r1rt_core_set_builtin_cmd(r.core, &b, r.t);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER, r.t), R1RT_OK);
  for (int i = 0; i < 20; ++i) r.tick();
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_ABORT, r.t), R1RT_OK);
  r.tick();
  r1rt_core_set_builtin_cmd(r.core, &b, r.t);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER, r.t), R1RT_OK);
  for (int i = 0; i < 20; ++i) r.tick();
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_RELEASED, r.t), R1RT_OK);
  for (int i = 0; i < 520; ++i) r.tick();  // the 1 s blend, then BALANCE
  EXPECT_EQ(g_news.load() - before, 0);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
}

}  // namespace

// ---- session 6: DESIGN §4.2 motor error words (arm / head: HOLD that limb; legs / waist: alarm only) and the
// over-temperature warning (DESIGN §4.1; alarm only). Session 5 raised MOTOR_ERROR and nothing else, and had no
// temperature warning (both fail on 26e8bce).
namespace {

r1rt_upper_cmd arm_stream(Rig& r, std::uint8_t group, const double* q, std::uint64_t id) {
  r1rt_upper_cmd c{};
  c.cmd_id = id;
  c.stop_epoch = r1rt_segment_stop_epoch(r.seg);
  c.t_issue_ns = r.t;
  c.ttl_ns = 300 * kMs;
  c.group = group;
  c.source = R1RT_SOURCE_STREAM;
  c.n_joints = 5;
  for (int k = 0; k < 5; ++k) c.q[k] = q[k];
  return c;
}

std::int32_t push_arm(Rig& r, std::uint8_t group, const double* q, std::uint64_t id) {
  const r1rt_upper_cmd c = arm_stream(r, group, q, id);
  return r1rt_segment_push_upper_cmd(r.seg, &c);
}

}  // namespace

TEST(Core, S6_AnArmMotorErrorHoldsThatArmStiffAndRefusesItsRecordsUntilTheOperatorsResume) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 20; ++i) r.tick();
  double ql[5], qr[5];
  for (int k = 0; k < 5; ++k) {
    ql[k] = r.out.q[14 + k];
    qr[k] = r.out.q[19 + k];
  }
  ql[1] += 0.1;  // the left shoulder roll out 0.1 rad (a stream the left arm follows)
  ASSERT_EQ(push_arm(r, R1RT_GROUP_ARM_LEFT, ql, 1), R1RT_OK);
  for (int i = 0; i < 10; ++i) r.tick();
  ASSERT_GT(r.out.q[15], ql[1] - 0.1 + 0.005);  // moving
  r.st.motor_state[16] = 1;  // an error word on a left-arm motor
  r.tick();
  auto s = r.status();
  EXPECT_NE(s.faults & R1RT_FAULT_MOTOR_ERROR, 0u);
  EXPECT_EQ(s.motor_error_groups, 1u << R1RT_GROUP_ARM_LEFT);
  EXPECT_EQ(s.motor_error_joints, 1u << 16);
  EXPECT_EQ(s.upper_event[R1RT_GROUP_ARM_LEFT], r1rt::kEventMotorError);
  for (int i = 0; i < 100; ++i) r.tick();  // the stop time (0.15 s) and more: at rest in HOLD
  const double held = r.out.q[15];
  for (int j = 14; j < 19; ++j) EXPECT_GT(r.out.kp[j], 1.0) << j;  // stiff, never limp
  for (int j = 0; j < 12; ++j) EXPECT_GT(r.out.kp[j], 1.0) << j;   // the legs keep balancing
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  // its records are refused; the other arm's are not
  const std::uint64_t rej0 = r.status().upper_rejected;
  ql[1] += 0.1;
  ASSERT_EQ(push_arm(r, R1RT_GROUP_ARM_LEFT, ql, 2), R1RT_OK);
  qr[1] -= 0.05;
  ASSERT_EQ(push_arm(r, R1RT_GROUP_ARM_RIGHT, qr, 3), R1RT_OK);
  for (int i = 0; i < 20; ++i) r.tick();
  EXPECT_EQ(r.status().upper_rejected, rej0 + 1);
  EXPECT_NEAR(r.out.q[15], held, 1e-12);  // still held
  EXPECT_LT(r.out.q[20], qr[1] + 0.05 - 0.005);  // the right arm follows its stream
  // RESUME while the word is still set: stays held
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_RESUME, r.t), R1RT_OK);
  EXPECT_EQ(r.status().motor_error_groups, 1u << R1RT_GROUP_ARM_LEFT);
  // the word cleared, then the operator's RESUME: records accepted again (the arm stays in HOLD until one comes)
  r.st.motor_state[16] = 0;
  r.tick();
  EXPECT_EQ(r.status().motor_error_groups, 1u << R1RT_GROUP_ARM_LEFT);  // nothing clears it by itself
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_RESUME, r.t), R1RT_OK);
  r.tick();
  EXPECT_EQ(r.status().motor_error_groups, 0u);
  ASSERT_EQ(push_arm(r, R1RT_GROUP_ARM_LEFT, ql, 4), R1RT_OK);
  for (int i = 0; i < 20; ++i) r.tick();
  EXPECT_GT(r.out.q[15], held + 0.005);
  EXPECT_EQ(r.status().motor_error_holds, 1u);
}

TEST(Core, S6_ALegMotorErrorIsAnAlarmOnlyTheLegsStayStiff) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 20; ++i) r.tick();
  r.st.motor_state[3] = 1;  // the left knee
  for (int i = 0; i < 50; ++i) r.tick();
  const auto s = r.status();
  EXPECT_NE(s.faults & R1RT_FAULT_MOTOR_ERROR, 0u);
  EXPECT_EQ(s.motor_error_groups, 0u);  // no group held: the policy keeps balancing
  EXPECT_EQ(s.motor_error_holds, 0u);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  for (int j = 0; j < 12; ++j) EXPECT_GT(r.out.kp[j], 1.0) << j;
}

TEST(Core, S6_OverTemperatureIsAWarningOnlyFromTheTakeoverGate) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 20; ++i) r.tick();
  r1rt_lowcmd_out before = r.out;
  r.st.temperature[1][0] = 59;  // below the gate (60 C)
  r.tick();
  EXPECT_EQ(r.status().faults & R1RT_FAULT_OVER_TEMP, 0u);
  EXPECT_EQ(r.status().temp_max_c, 59);
  r.st.temperature[1][1] = 61;  // the hip roll's second sensor
  r.tick();
  auto s = r.status();
  EXPECT_NE(s.faults & R1RT_FAULT_OVER_TEMP, 0u);
  EXPECT_EQ(s.temp_max_c, 61);
  EXPECT_EQ(s.temp_max_joint, 1);
  EXPECT_EQ(s.over_temp_ticks, 1u);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_EQ(r.out.kp[j], before.kp[j]) << j;  // no action on stiffness
    EXPECT_EQ(r.out.kd[j], before.kd[j]) << j;
  }
  r.st.temperature[1][1] = 30;
  r.tick();
  EXPECT_EQ(r.status().faults & R1RT_FAULT_OVER_TEMP, 0u);
}


// Session 10 (layout 4): r1d on real processes has no Core.status() - it reads the executor's per-group status from
// the segment's exec_status region, which must say what the status says, every tick: the record a group follows, the
// DONE / guard / deadman event that ended it, the refusals, the state-age counters. (Fails to build on the base: no
// exec_status region / r1rt_exec_status type.)
TEST(Core, ExecStatusInTheSegmentSaysWhatTheExecutorDid) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 10; ++i) r.tick();
  auto same = [&r](const char* when) {
    r1rt_exec_status es{};
    ASSERT_TRUE(r.segment().exec_status.try_read(es)) << when;
    const r1rt_core_status st = r.status();
    EXPECT_EQ(es.seq, st.ticks) << when;
    EXPECT_EQ(es.t_ns, r.t) << when;  // published at the tick's own time
    EXPECT_EQ(es.upper_accepted, st.upper_accepted) << when;
    EXPECT_EQ(es.upper_rejected, st.upper_rejected) << when;
    EXPECT_EQ(es.last_reject, st.upper_last_reject) << when;
    EXPECT_EQ(es.stop_epoch, st.upper_stop_epoch) << when;
    for (int g = 0; g < R1RT_NUM_UPPER_GROUPS; ++g) {
      EXPECT_EQ(es.cmd_id[g], st.upper_cmd_id[g]) << when << " group " << g;
      EXPECT_EQ(es.source[g], st.upper_source[g]) << when << " group " << g;
      EXPECT_EQ(es.event[g], st.upper_event[g]) << when << " group " << g;
      EXPECT_EQ(es.stopping[g], st.upper_stopping[g]) << when << " group " << g;
    }
    for (int g = 0; g < R1RT_NUM_CMD_GROUPS - R1RT_NUM_UPPER_GROUPS; ++g) {
      EXPECT_EQ(es.cmd_id[R1RT_NUM_UPPER_GROUPS + g], st.lower_cmd_id[g]) << when;
      EXPECT_EQ(es.source[R1RT_NUM_UPPER_GROUPS + g], st.lower_source[g]) << when;
    }
    EXPECT_EQ(es.state_content_age_ns, st.state_content_age_ns) << when;
    EXPECT_EQ(es.state_old_ticks, st.state_old_ticks) << when;
    EXPECT_EQ(es.state_old_episodes, st.state_old_episodes) << when;
    EXPECT_EQ(es.content_age_valid, st.content_age_valid) << when;
  };
  same("idle");
  const r1rt_upper_cmd c = right_arm_move(r.out, 0.2, 1000 * kMs, 7, r1rt_segment_stop_epoch(r.seg), r.t);
  ASSERT_EQ(r1rt_segment_push_upper_cmd(r.seg, &c), R1RT_OK);
  for (int i = 0; i < 20; ++i) {
    if (i % 10 == 0) r1rt_segment_upper_keepalive(r.seg, R1RT_GROUP_ARM_RIGHT, 7, r.t + kTick);
    r.tick();
  }
  same("running");
  r1rt_exec_status es{};
  ASSERT_TRUE(r.segment().exec_status.try_read(es));
  EXPECT_EQ(es.cmd_id[R1RT_GROUP_ARM_RIGHT], 7u);
  EXPECT_EQ(es.source[R1RT_GROUP_ARM_RIGHT], R1RT_SOURCE_TRAJECTORY);
  EXPECT_EQ(es.upper_accepted, 1u);
  for (int i = 0; i < 150; ++i) r.tick();  // no keepalive: the deadman stops it
  same("deadman");
  ASSERT_TRUE(r.segment().exec_status.try_read(es));
  EXPECT_EQ(es.event[R1RT_GROUP_ARM_RIGHT], 16);  // kEventDeadman
  EXPECT_EQ(es.source[R1RT_GROUP_ARM_RIGHT], R1RT_SOURCE_HOLD);
  // a record stamped before a stop is refused: the count and the reason reach the segment
  const std::uint64_t old_epoch = r1rt_segment_stop_epoch(r.seg);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_STOP, r.t), R1RT_OK);
  r.tick();
  const r1rt_upper_cmd stale = right_arm_move(r.out, 0.1, 500 * kMs, 3, old_epoch, r.t);
  ASSERT_EQ(r1rt_segment_push_upper_cmd(r.seg, &stale), R1RT_OK);
  r.tick();
  same("refused");
  ASSERT_TRUE(r.segment().exec_status.try_read(es));
  EXPECT_EQ(es.upper_rejected, 1u);
  EXPECT_EQ(es.last_reject, 7);  // kRejectStopEpoch
  EXPECT_EQ(es.stale_rung, 0);    // the configured rung (the rig's config: OFF)
}

// Session 13 (HW2 day 2's F12: r1d accepted `r1 lean 3` but the robot never leaned - a client's `stop` six minutes
// before had latched, the stop zeroes the base command until a resume, and r1d could not see the latch): r1rt's
// heartbeat flags carry the latched stop (R1RT_RT_FLAG_STOP_ACTIVE) - set by the stop, cleared by the resume - and a
// lean renewed through the stop stays 0 until the resume, then ramps (4 deg/s) to it.
TEST(Core, TheLatchedStopIsInTheHeartbeatAndHoldsTheLeanAtZero) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 20; ++i) r.tick();
  const auto flags = [&r]() -> unsigned {
    r1rt_heartbeat hb{};
    EXPECT_TRUE(r.segment().rt_heartbeat.try_read(hb));
    return hb.guard_flags;
  };
  std::uint64_t seq = 0;
  const auto lean3 = [&r, &seq]() {
    r1rt_base_cmd b{};
    b.seq = ++seq;
    b.t_issue_ns = r.t;
    b.ttl_ns = 500 * kMs;
    b.lean = 3.0 * M_PI / 180.0;
    r.segment().base_cmd.write(b);
  };
  EXPECT_EQ(flags() & R1RT_RT_FLAG_STOP_ACTIVE, 0u);
  r.segment().stop.request(R1RT_STOP_STOP, R1RT_REQUESTER_R1D, r.t);
  r.tick();
  EXPECT_NE(flags() & R1RT_RT_FLAG_STOP_ACTIVE, 0u);
  for (int i = 0; i < 500; ++i) {  // 1 s of the lean renewed (r1d's tick) while stopped
    lean3();
    r.tick();
  }
  EXPECT_NEAR(r.status().lean, 0.0, 1e-12);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_RESUME, r.t), R1RT_OK);
  r.tick();
  EXPECT_EQ(flags() & R1RT_RT_FLAG_STOP_ACTIVE, 0u);
  for (int i = 0; i < 500; ++i) {  // 1 s after the resume: the ramp reaches 3 deg in 0.75 s
    lean3();
    r.tick();
  }
  EXPECT_NEAR(r.status().lean, 3.0 * M_PI / 180.0, 1e-9);
  // ... and r1log's policy record carries the observation the network saw (base: all zeros): the lean term (index 9
  // for this policy: ang vel 3, gravity 3, command 3, lean 1) = the ramped lean, the gravity term = the IMU's
  r1rt_policy_record rec{};
  r1rt_policy_record last{};
  int n = 0;
  while (r.segment().policy.try_pop(rec)) {
    last = rec;
    ++n;
  }
  ASSERT_GT(n, 0);
  EXPECT_EQ(last.n_obs, 74u);
  EXPECT_NEAR(last.obs[9], 3.0 * M_PI / 180.0, 1e-6);
  EXPECT_NEAR(last.obs[5], -1.0, 1e-6);  // projected gravity z of the upright rig
}
