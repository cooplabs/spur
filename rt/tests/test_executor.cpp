// r1rt::UpperBodyExecutor (DESIGN.md §3.3): every source, the deadman / TTL / operator-stop paths, the guards, the
// stop profile and its bounds, pause / resume, multi-segment trajectories, limits, bad input, determinism, and the
// no-allocation rule of the RT calls.
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>

#include "r1rt/executor.hpp"
#include "r1rt/step_api.h"

// ---------------------------------------------------------------- a counting global operator new (this test binary)
namespace {
std::atomic<long> g_news{0};
}  // namespace

void* operator new(std::size_t n) {
  g_news.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n ? n : 1)) return p;
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

using r1rt::ExecutorConfig;
using r1rt::UpperBodyExecutor;

constexpr std::int64_t kTick = 2'000'000;  // 500 Hz
constexpr double kDt = 0.002;
constexpr int N = R1RT_NUM_JOINTS;

// A closed loop around the executor: the "robot" follows the command exactly (or as a test says).
struct Rig {
  ExecutorConfig cfg = r1rt::r1_default_executor_config();
  UpperBodyExecutor ex;
  double q[N] = {}, dq[N] = {}, tau[N] = {};
  double qo[N], dqo[N], kpo[N], kdo[N], ffo[N];
  double tilt = 0.0;
  r1rt_hands_contact hands{};
  bool hands_on = true;
  std::int64_t t = 1'000'000'000;
  bool follow = true;   // measured q := commanded q every tick
  int blocked = -1;     // a joint whose measured q stays where it is

  explicit Rig(bool do_init = true) {
    for (int j = 0; j < N; ++j) {
      qo[j] = dqo[j] = kpo[j] = kdo[j] = ffo[j] = -777.0;  // sentinels: indices 0-13 must stay untouched
    }
    const double pose[12] = {0.2, 0.19, 0.0, 0.8, 0.0, 0.2, -0.19, 0.0, 0.8, 0.0, 0.0, 0.0};
    for (int k = 0; k < 12; ++k) q[14 + k] = pose[k];
    hands.valid = 1;
    if (do_init) {
      EXPECT_EQ(ex.init(cfg), R1RT_OK);
      ex.reset(q, t);
    }
  }
  void step() {
    t += kTick;
    hands.t_ns = t;
    ex.step(t, q, dq, tau, hands_on ? &hands : nullptr, tilt, qo, dqo, kpo, kdo, ffo);
    if (follow) {
      for (int j = 14; j < N; ++j) {
        if (j == blocked) continue;
        dq[j] = (qo[j] - q[j]) / kDt;
        q[j] = qo[j];
      }
    }
  }
  void run(double s) {
    const int n = static_cast<int>(std::lround(s / kDt));
    for (int i = 0; i < n; ++i) step();
  }
};

r1rt_upper_cmd base_cmd(r1rt_group g, r1rt_source src, std::uint64_t id, std::int64_t t) {
  r1rt_upper_cmd c;
  std::memset(&c, 0, sizeof(c));
  c.cmd_id = id;
  c.group = g;
  c.source = src;
  c.n_joints = g == R1RT_GROUP_HEAD ? 2 : 5;
  c.t_issue_ns = t;
  return c;
}

int first_joint(r1rt_group g) { return g == R1RT_GROUP_ARM_LEFT ? 14 : (g == R1RT_GROUP_ARM_RIGHT ? 19 : 24); }

// A trajectory record for group g: n knots, knot i at i * dt_knot seconds, q = q0 + amp * (1 - cos(pi s)) / 2 on
// every joint (s = i / (n_total - 1)), with the matching dq / ddq.
r1rt_upper_cmd traj(r1rt_group g, std::uint64_t id, std::int64_t t, const double* q0, double amp, int i0, int n,
                    int n_total, double dt_knot, std::uint16_t seg, bool last) {
  r1rt_upper_cmd c = base_cmd(g, R1RT_SOURCE_TRAJECTORY, id, t);
  c.n_knots = static_cast<std::uint8_t>(n);
  c.segment_index = seg;
  c.last_segment = last ? 1 : 0;
  const double T = dt_knot * (n_total - 1);
  for (int i = 0; i < n; ++i) {
    const int ii = i0 + i;
    const double tt = ii * dt_knot;
    c.knots[i].t_ns = static_cast<std::int64_t>(std::llround(tt * 1e9));
    for (int k = 0; k < c.n_joints; ++k) {
      const double w = M_PI / T;
      c.knots[i].q[k] = q0[k] + amp * 0.5 * (1.0 - std::cos(w * tt));
      c.knots[i].dq[k] = amp * 0.5 * w * std::sin(w * tt);
      c.knots[i].ddq[k] = amp * 0.5 * w * w * std::cos(w * tt);
    }
  }
  return c;
}

double cosine_q(double q0, double amp, double T, double tt) {
  return q0 + amp * 0.5 * (1.0 - std::cos(M_PI / T * std::min(tt, T)));
}

// ---------------------------------------------------------------- config / reset

TEST(Executor, InitRefusesBadConfigs) {
  UpperBodyExecutor ex;
  ExecutorConfig c = r1rt::r1_default_executor_config();
  EXPECT_EQ(ex.init(c), R1RT_OK);
  auto bad = [&](auto mutate) {
    ExecutorConfig b = r1rt::r1_default_executor_config();
    mutate(b);
    UpperBodyExecutor e2;
    return e2.init(b) == R1RT_E_INVALID_ARG && !e2.ready();
  };
  EXPECT_TRUE(bad([](ExecutorConfig& b) { b.hold_kp[15] = 0.0; }));  // a limp HOLD
  EXPECT_TRUE(bad([](ExecutorConfig& b) { b.hold_kp[15] = b.kp_max[15] + 1.0; }));
  EXPECT_TRUE(bad([](ExecutorConfig& b) { b.q_min[20] = b.q_max[20]; }));
  EXPECT_TRUE(bad([](ExecutorConfig& b) { b.dq_max[24] = std::nan(""); }));
  EXPECT_TRUE(bad([](ExecutorConfig& b) { b.ddq_stop[19] = 0.0; }));
  EXPECT_TRUE(bad([](ExecutorConfig& b) { b.stop_time_s = 0.0; }));
  EXPECT_TRUE(bad([](ExecutorConfig& b) { b.keepalive_ns = 0; }));
  EXPECT_TRUE(bad([](ExecutorConfig& b) { b.max_traj_knots = 4; }));
}

TEST(Executor, ResetHoldsTheMeasuredPoseAndWritesOnlyTheUpperBody) {
  Rig r;
  r.step();
  for (int j = 0; j < 14; ++j) {
    EXPECT_EQ(r.qo[j], -777.0);
    EXPECT_EQ(r.kpo[j], -777.0);
  }
  for (int j = 14; j < N; ++j) {
    EXPECT_DOUBLE_EQ(r.qo[j], r.q[j]);
    EXPECT_EQ(r.dqo[j], 0.0);
    EXPECT_DOUBLE_EQ(r.kpo[j], r.cfg.hold_kp[j]);
    EXPECT_DOUBLE_EQ(r.kdo[j], r.cfg.hold_kd[j]);
    EXPECT_EQ(r.ffo[j], 0.0);
  }
  for (int g = 0; g < 3; ++g) EXPECT_EQ(r.ex.source(static_cast<r1rt_group>(g)), R1RT_SOURCE_HOLD);
}

TEST(Executor, FirstStepWithoutResetHoldsTheMeasuredPose) {
  Rig r(false);
  ASSERT_EQ(r.ex.init(r.cfg), R1RT_OK);
  r.q[16] = 0.33;
  r.step();
  EXPECT_DOUBLE_EQ(r.qo[16], 0.33);
  EXPECT_GT(r.kpo[16], 0.0);
}

TEST(Executor, NotInitialisedWritesNothingAndRefusesRecords) {
  Rig r(false);
  r.step();
  for (int j = 0; j < N; ++j) EXPECT_EQ(r.qo[j], -777.0);
  const auto c = base_cmd(R1RT_GROUP_HEAD, R1RT_SOURCE_HOLD, 1, r.t);
  EXPECT_EQ(r.ex.submit(c, r.t), R1RT_E_INVALID_ARG);
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectNotReady);
}

// ---------------------------------------------------------------- HOLD

TEST(Executor, HoldRecordMovesSmoothlyToItsGoalAndStays) {
  Rig r;
  auto c = base_cmd(R1RT_GROUP_ARM_RIGHT, R1RT_SOURCE_HOLD, 7, r.t);
  for (int k = 0; k < 5; ++k) c.q[k] = r.q[19 + k] + 0.5;
  c.q[1] = r.q[20];  // right shoulder roll stays (its upper limit is 0.18)
  c.tau_ff[0] = 1.25;
  ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  double v_prev = 0.0, q_prev = r.q[19];
  double max_dv = 0.0, max_v = 0.0;
  for (int i = 0; i < 1000; ++i) {
    r.step();
    const double v = (r.qo[19] - q_prev) / kDt;
    max_dv = std::max(max_dv, std::fabs(v - v_prev));
    max_v = std::max(max_v, std::fabs(r.dqo[19]));
    EXPECT_LE(r.qo[19], c.q[0] + 1e-9);
    v_prev = v;
    q_prev = r.qo[19];
  }
  EXPECT_NEAR(r.qo[19], c.q[0], 1e-9);
  EXPECT_EQ(r.dqo[19], 0.0);
  EXPECT_LE(max_v, r.cfg.dq_max[19] + 1e-9);
  EXPECT_LE(max_dv, r.cfg.ddq_stop[19] * kDt + 1e-6);
  EXPECT_DOUBLE_EQ(r.ffo[19], 1.25);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_RIGHT), R1RT_SOURCE_HOLD);
}

TEST(Executor, HoldGoalBeyondTheLimitIsClampedAndCounted) {
  Rig r;
  auto c = base_cmd(R1RT_GROUP_HEAD, R1RT_SOURCE_HOLD, 1, r.t);
  c.q[0] = 5.0;
  c.q[1] = 0.0;
  const auto n0 = r.ex.clamp_count();
  ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  r.run(2.0);
  EXPECT_DOUBLE_EQ(r.qo[24], r.cfg.q_max[24]);
  EXPECT_GT(r.ex.clamp_count(), n0);
}

TEST(Executor, GainsRampBumplessly) {
  Rig r;
  auto c = base_cmd(R1RT_GROUP_ARM_LEFT, R1RT_SOURCE_HOLD, 1, r.t);
  for (int k = 0; k < 5; ++k) {
    c.q[k] = r.q[14 + k];
    c.kp[k] = 20.0;
    c.kd[k] = 1.0;
  }
  ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  double prev = r.cfg.hold_kp[14];
  for (int i = 0; i < 100; ++i) {
    r.step();
    EXPECT_LE(std::fabs(r.kpo[14] - prev), r.cfg.hold_kp[14] * kDt / 0.1 + 1e-9);
    prev = r.kpo[14];
  }
  EXPECT_DOUBLE_EQ(r.kpo[14], 20.0);
}

// ---------------------------------------------------------------- TRAJECTORY

TEST(Executor, TrajectoryPassesThroughItsKnotsAndEndsInHold) {
  Rig r;
  const double* q0 = &r.q[14];
  double q00[5];
  std::memcpy(q00, q0, sizeof(q00));
  const auto c = traj(R1RT_GROUP_ARM_LEFT, 11, r.t, q00, 0.3, 0, 6, 6, 0.2, 0, true);  // 1.0 s
  ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY);
  double maxerr = 0.0, maxverr = 0.0;
  for (int i = 1; i <= 520; ++i) {
    r.ex.keepalive(11, r.t);
    r.step();
    const double tt = i * kDt;
    if (tt <= 1.0) {
      maxerr = std::max(maxerr, std::fabs(r.qo[14] - cosine_q(q00[0], 0.3, 1.0, tt)));
      const double v = 0.3 * 0.5 * M_PI * std::sin(M_PI * tt);
      maxverr = std::max(maxverr, std::fabs(r.dqo[14] - v));
    }
  }
  // a quintic Hermite through exact samples of a cosine: a few 1e-5 rad between the knots
  EXPECT_LT(maxerr, 5e-5);
  EXPECT_LT(maxverr, 5e-4);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_HOLD);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kEventDone);
  EXPECT_NEAR(r.qo[14], q00[0] + 0.3, 1e-12);
  EXPECT_EQ(r.dqo[14], 0.0);
  EXPECT_EQ(r.qo[19], r.q[19]);  // the other arm did not move
}

TEST(Executor, TrajectoryKnotTimesAreHitExactly) {
  Rig r;
  auto c = base_cmd(R1RT_GROUP_HEAD, R1RT_SOURCE_TRAJECTORY, 3, r.t);
  c.n_knots = 3;
  c.last_segment = 1;
  const double qs[3] = {r.q[24], r.q[24] + 0.1, r.q[24] + 0.15};
  for (int i = 0; i < 3; ++i) {
    c.knots[i].t_ns = i * 100'000'000;  // 0, 0.1, 0.2 s = 50 ticks each
    c.knots[i].q[0] = qs[i];
    c.knots[i].q[1] = r.q[25];
  }
  ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  for (int i = 1; i <= 50; ++i) {
    r.ex.keepalive(3, r.t);
    r.step();
  }
  EXPECT_NEAR(r.qo[24], qs[1], 1e-9);
  for (int i = 1; i <= 50; ++i) {
    r.ex.keepalive(3, r.t);
    r.step();
  }
  EXPECT_NEAR(r.qo[24], qs[2], 1e-9);
}

TEST(Executor, MultiSegmentTrajectoryIsOneSmoothPath) {
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[19], sizeof(q00));
  const int n_total = 22;  // 3 records: 8 + 8 + 6 knots, 0.1 s apart = 2.1 s
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_RIGHT, 5, r.t, q00, -0.4, 0, 8, n_total, 0.1, 0, false), r.t), R1RT_OK);
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_RIGHT, 5, r.t, q00, -0.4, 8, 8, n_total, 0.1, 1, false), r.t), R1RT_OK);
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_RIGHT, 5, r.t, q00, -0.4, 16, 6, n_total, 0.1, 2, true), r.t), R1RT_OK);
  double maxerr = 0.0;
  for (int i = 1; i <= 1100; ++i) {
    r.ex.keepalive(5, r.t);
    r.step();
    maxerr = std::max(maxerr, std::fabs(r.qo[19] - cosine_q(q00[0], -0.4, 2.1, i * kDt)));
  }
  EXPECT_LT(maxerr, 5e-5);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_RIGHT), r1rt::kEventDone);
  EXPECT_EQ(r.ex.tick_clamps(), 0u);
}

TEST(Executor, SegmentsMustContinueTheActiveTrajectory) {
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 9, r.t, q00, 0.2, 0, 8, 20, 0.1, 0, false), r.t), R1RT_OK);
  // wrong index, wrong id, overlapping times, a segment 0 that does not start at t = 0
  EXPECT_NE(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 9, r.t, q00, 0.2, 8, 8, 20, 0.1, 2, false), r.t), R1RT_OK);
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectSegment);
  EXPECT_NE(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 10, r.t, q00, 0.2, 8, 8, 20, 0.1, 1, false), r.t), R1RT_OK);
  EXPECT_NE(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 9, r.t, q00, 0.2, 7, 8, 20, 0.1, 1, false), r.t), R1RT_OK);
  auto late = traj(R1RT_GROUP_ARM_LEFT, 12, r.t, q00, 0.2, 1, 8, 20, 0.1, 0, false);
  EXPECT_NE(r.ex.submit(late, r.t), R1RT_OK);
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectKnots);
  EXPECT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 9, r.t, q00, 0.2, 8, 8, 20, 0.1, 1, false), r.t), R1RT_OK);
}

TEST(Executor, AStarvedTrajectoryStopsSmoothly) {
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  // the first of two segments only: the clock reaches its last knot (0.7 s) while moving
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 4, r.t, q00, 0.6, 0, 8, 12, 0.1, 0, false), r.t), R1RT_OK);
  for (int i = 0; i < 500; ++i) {  // starved at 0.7 s, then a 0.15 s stop
    r.ex.keepalive(4, r.t);
    r.step();
  }
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kEventStarved);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_HOLD);
  EXPECT_FALSE(r.ex.stopping(R1RT_GROUP_ARM_LEFT));
  EXPECT_EQ(r.dqo[14], 0.0);
}

void run_ka(Rig& r, std::uint64_t id, double s) {
  const int n = static_cast<int>(std::lround(s / kDt));
  for (int i = 0; i < n; ++i) {
    r.ex.keepalive(id, r.t);
    r.step();
  }
}

TEST(Executor, TrajectoryMustStartAtTheCurrentTarget) {
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  q00[2] += 0.06;  // > traj_start_tol_rad
  EXPECT_NE(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 1, r.t, q00, 0.1, 0, 4, 4, 0.1, 0, true), r.t), R1RT_OK);
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectStart);
}

TEST(Executor, DeadmanStopsATrajectoryWithoutKeepalive) {
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 21, r.t, q00, 0.8, 0, 8, 8, 0.3, 0, true), r.t), R1RT_OK);
  const std::int64_t t0 = r.t;
  // keepalives for 0.5 s, then r1d "dies"
  while (r.t - t0 < 500'000'000) {
    r.ex.keepalive(21, r.t);
    r.step();
  }
  const std::int64_t t_dead = r.t;
  double v_prev = r.dqo[14], max_dv = 0.0;
  bool stopped_seen = false;
  std::int64_t t_hold = 0;
  for (int i = 0; i < 300; ++i) {
    r.step();
    if (r.ex.stopping(R1RT_GROUP_ARM_LEFT)) stopped_seen = true;
    max_dv = std::max(max_dv, std::fabs(r.dqo[14] - v_prev));
    v_prev = r.dqo[14];
    if (t_hold == 0 && stopped_seen && !r.ex.stopping(R1RT_GROUP_ARM_LEFT)) t_hold = r.t;
  }
  EXPECT_TRUE(stopped_seen);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kEventDeadman);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_HOLD);
  EXPECT_EQ(r.dqo[14], 0.0);
  // the deadman fires at keepalive_ns after the last renewal; the stop lasts stop_time_s (or longer if ddq_stop
  // demands it)
  EXPECT_GT(t_hold - t_dead, r.cfg.keepalive_ns);
  EXPECT_LE(t_hold - t_dead, r.cfg.keepalive_ns + 300'000'000);
  EXPECT_LE(max_dv, r.cfg.ddq_stop[14] * kDt * 1.0001 + 1e-9);
  // a late keepalive does not restart it
  r.ex.keepalive(21, r.t);
  r.step();
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_HOLD);
}

TEST(Executor, AnOldKeepaliveStampPassedEveryTickDoesNotExtendTheDeadline) {
  // the core reads r1d's UpperKeepalive {t_ns, cmd_id} every tick and passes it on: a stamp that stops changing
  // must let the deadman fire keepalive_ns after it
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 31, r.t, q00, 0.5, 0, 5, 5, 0.5, 0, true), r.t), R1RT_OK);
  std::int64_t stamp = r.t;
  for (int i = 0; i < 100; ++i) {  // r1d renews every 20 ms for 0.2 s
    if (i % 10 == 0) stamp = r.t;
    r.ex.keepalive(31, stamp);
    r.step();
  }
  const std::int64_t last = stamp;
  while (r.ex.source(R1RT_GROUP_ARM_LEFT) == R1RT_SOURCE_TRAJECTORY && r.t < last + 1'000'000'000) {
    r.ex.keepalive(31, last);  // the same old stamp, every tick
    r.step();
  }
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kEventDeadman);
  EXPECT_LE(r.t - last, r.cfg.keepalive_ns + kTick);
  EXPECT_GE(r.t - last, r.cfg.keepalive_ns);
}

TEST(Executor, StopEpochFromTheStopChannel) {
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[19], sizeof(q00));
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_RIGHT, 1, r.t, q00, -0.3, 0, 5, 5, 0.5, 0, true), r.t), R1RT_OK);
  run_ka(r, 1, 0.3);
  r.ex.stop_at_epoch(0, r.t);  // unchanged: nothing happens
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_RIGHT), R1RT_SOURCE_TRAJECTORY);
  r.ex.stop_at_epoch(3, r.t);  // two stops and a halt arrived since the last tick
  EXPECT_TRUE(r.ex.stopping(R1RT_GROUP_ARM_RIGHT));
  EXPECT_EQ(r.ex.stop_epoch(), 3u);
  auto c = base_cmd(R1RT_GROUP_HEAD, R1RT_SOURCE_HOLD, 2, r.t);
  c.q[0] = r.q[24];
  c.q[1] = r.q[25];
  c.stop_epoch = 2;
  EXPECT_NE(r.ex.submit(c, r.t), R1RT_OK);
  c.stop_epoch = 3;
  EXPECT_EQ(r.ex.submit(c, r.t), R1RT_OK);
}

TEST(Executor, ARecordMayAskForAShorterKeepalivePeriod) {
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  auto c = traj(R1RT_GROUP_ARM_LEFT, 2, r.t, q00, 0.3, 0, 5, 5, 0.5, 0, true);
  c.ttl_ns = 20'000'000;
  ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  r.run(0.024);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kEventDeadman);
}

// ---------------------------------------------------------------- STREAM

TEST(Executor, StreamFollowsSetpointsAndStopsWhenTheTtlExpires) {
  Rig r;
  double q = r.q[24];
  for (int i = 0; i < 100; ++i) {
    auto c = base_cmd(R1RT_GROUP_HEAD, R1RT_SOURCE_STREAM, 77, r.t);
    q += 0.002;  // 1 rad/s
    c.q[0] = q;
    c.q[1] = r.q[25];
    c.dq[0] = 1.0;
    c.ttl_ns = 100'000'000;
    ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
    r.step();
    EXPECT_DOUBLE_EQ(r.qo[24], q);
    EXPECT_DOUBLE_EQ(r.dqo[24], 1.0);
  }
  EXPECT_EQ(r.ex.source(R1RT_GROUP_HEAD), R1RT_SOURCE_STREAM);
  const double q_last = r.qo[24];
  // no more records: the target keeps the last setpoint (a stream is not extrapolated) with the record's dq
  // feed-forward until the TTL (100 ms) expires, then HOLDs there
  r.run(0.09);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_HEAD), R1RT_SOURCE_STREAM);
  EXPECT_DOUBLE_EQ(r.qo[24], q_last);
  EXPECT_DOUBLE_EQ(r.dqo[24], 1.0);
  r.run(0.02);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_HEAD), r1rt::kEventStreamTtl);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_HEAD), R1RT_SOURCE_HOLD);
  EXPECT_EQ(r.dqo[24], 0.0);
  EXPECT_DOUBLE_EQ(r.qo[24], q_last);
}

TEST(Executor, AStreamThatExpiresWhileItsTargetMovesDeceleratesSmoothly) {
  Rig r;
  auto c = base_cmd(R1RT_GROUP_HEAD, R1RT_SOURCE_STREAM, 78, r.t);
  c.q[0] = r.q[24] + 0.5;  // far: the target travels at dq_max (2 rad/s) for 0.25 s
  c.q[1] = r.q[25];
  c.ttl_ns = 100'000'000;
  ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  double q_prev = r.q[24], v_prev = 0.0, max_acc = 0.0;
  bool stopped = false;
  for (int i = 0; i < 200; ++i) {
    r.step();
    const double v = (r.qo[24] - q_prev) / kDt;
    if (stopped || r.ex.stopping(R1RT_GROUP_HEAD)) {
      if (stopped) max_acc = std::max(max_acc, std::fabs(v - v_prev) / kDt);
      stopped = true;
    }
    q_prev = r.qo[24];
    v_prev = v;
  }
  EXPECT_TRUE(stopped);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_HEAD), r1rt::kEventStreamTtl);
  EXPECT_LE(max_acc, r.cfg.ddq_stop[24] * 1.0001 + 1e-6);
  EXPECT_LT(r.qo[24], r.q[24] + 1e-12 + 0.5);  // it stopped short of the far setpoint
  EXPECT_EQ(r.dqo[24], 0.0);
}

TEST(Executor, StreamTtlBounds) {
  Rig r;
  auto c = base_cmd(R1RT_GROUP_ARM_LEFT, R1RT_SOURCE_STREAM, 1, r.t);
  for (int k = 0; k < 5; ++k) c.q[k] = r.q[14 + k];
  c.ttl_ns = 301'000'000;  // > 300 ms
  EXPECT_NE(r.ex.submit(c, r.t), R1RT_OK);
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectTtl);
  c.ttl_ns = 0;
  EXPECT_NE(r.ex.submit(c, r.t), R1RT_OK);
  c.ttl_ns = 50'000'000;
  c.t_issue_ns = r.t - 60'000'000;  // already expired when it arrives
  EXPECT_NE(r.ex.submit(c, r.t), R1RT_OK);
  c.t_issue_ns = r.t;
  EXPECT_EQ(r.ex.submit(c, r.t), R1RT_OK);
}

TEST(Executor, AStreamJumpIsApproachedAtDqMax) {
  Rig r;
  auto c = base_cmd(R1RT_GROUP_ARM_LEFT, R1RT_SOURCE_STREAM, 1, r.t);
  for (int k = 0; k < 5; ++k) c.q[k] = r.q[14 + k];
  c.q[3] += 0.5;
  c.ttl_ns = 300'000'000;
  ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  const double q0 = r.q[17];
  r.step();
  EXPECT_NEAR(r.qo[17] - q0, r.cfg.dq_max[17] * kDt, 1e-9);
  EXPECT_GT(r.ex.tick_clamps(), 0u);
}

// ---------------------------------------------------------------- FREEDRIVE (operator only)

TEST(Executor, FreedriveNeedsAFreshGravityFeedForwardAndIsTimed) {
  Rig r;
  auto c = base_cmd(R1RT_GROUP_ARM_LEFT, R1RT_SOURCE_FREEDRIVE, 1, r.t);
  EXPECT_NE(r.ex.submit(c, r.t), R1RT_OK);  // never through the ring
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectSource);
  EXPECT_NE(r.ex.freedrive(R1RT_GROUP_ARM_LEFT, 0.5, 1'000'000'000, r.t), R1RT_OK);  // no gravity model yet
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectNoGravityFF);
  double grav[N] = {};
  for (int j = 14; j < N; ++j) grav[j] = 0.1 * j;
  r.ex.set_gravity_ff(grav, r.t);
  EXPECT_NE(r.ex.freedrive(R1RT_GROUP_ARM_LEFT, 5.0, 1'000'000'000, r.t), R1RT_OK);  // kd too large
  ASSERT_EQ(r.ex.freedrive(R1RT_GROUP_ARM_LEFT, 0.5, 400'000'000, r.t), R1RT_OK);
  r.follow = false;
  for (int i = 0; i < 150; ++i) {
    r.ex.set_gravity_ff(grav, r.t + kTick);
    r.q[15] += 0.001;  // the operator moves the arm
    r.step();
  }
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_FREEDRIVE);
  EXPECT_EQ(r.kpo[15], 0.0);
  EXPECT_DOUBLE_EQ(r.kdo[15], 0.5);
  EXPECT_DOUBLE_EQ(r.qo[15], r.q[15]);
  EXPECT_DOUBLE_EQ(r.ffo[15], grav[15]);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_RIGHT), R1RT_SOURCE_HOLD);
  for (int i = 0; i < 60; ++i) {
    r.ex.set_gravity_ff(grav, r.t + kTick);
    r.step();
  }
  // timed out: HOLD where the operator left it, gains ramping back
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_HOLD);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kEventFreedriveEnd);
  EXPECT_DOUBLE_EQ(r.qo[15], r.q[15]);
  r.run(0.2);
  EXPECT_DOUBLE_EQ(r.kpo[15], r.cfg.hold_kp[15]);
}

TEST(Executor, FreedriveEndsWhenTheGravityFeedForwardGoesStale) {
  Rig r;
  double grav[N] = {};
  r.ex.set_gravity_ff(grav, r.t);
  ASSERT_EQ(r.ex.freedrive(R1RT_GROUP_ARM_RIGHT, 0.2, 10'000'000'000, r.t), R1RT_OK);
  r.run(0.05);  // no fresh gravity ff (max age 10 ms)
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_RIGHT), R1RT_SOURCE_HOLD);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_RIGHT), r1rt::kEventFreedriveEnd);
  EXPECT_GT(r.kpo[19], 0.0);
}

// ---------------------------------------------------------------- operator stop, pause / resume

TEST(Executor, OperatorStopHaltsEveryGroupAndRefusesOlderRecords) {
  Rig r;
  double ql[5], qr[5];
  std::memcpy(ql, &r.q[14], sizeof(ql));
  std::memcpy(qr, &r.q[19], sizeof(qr));
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 1, r.t, ql, 0.5, 0, 6, 6, 0.4, 0, true), r.t), R1RT_OK);
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_RIGHT, 2, r.t, qr, -0.5, 0, 6, 6, 0.4, 0, true), r.t), R1RT_OK);
  for (int i = 0; i < 300; ++i) {
    r.ex.keepalive(1, r.t);
    r.ex.keepalive(2, r.t);
    r.step();
  }
  EXPECT_EQ(r.ex.stop_epoch(), 0u);
  r.ex.stop(r.t);
  EXPECT_EQ(r.ex.stop_epoch(), 1u);
  EXPECT_TRUE(r.ex.stopping(R1RT_GROUP_ARM_LEFT));
  EXPECT_TRUE(r.ex.stopping(R1RT_GROUP_ARM_RIGHT));
  for (int i = 0; i < 200; ++i) {
    r.ex.keepalive(1, r.t);  // keepalives no longer matter
    r.step();
  }
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_HOLD);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kEventOperatorStop);
  EXPECT_EQ(r.dqo[14], 0.0);
  EXPECT_EQ(r.dqo[19], 0.0);
  auto old = base_cmd(R1RT_GROUP_HEAD, R1RT_SOURCE_HOLD, 3, r.t);
  old.q[0] = r.q[24];
  old.q[1] = r.q[25];
  old.stop_epoch = 0;
  EXPECT_NE(r.ex.submit(old, r.t), R1RT_OK);
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectStopEpoch);
  old.stop_epoch = 1;
  EXPECT_EQ(r.ex.submit(old, r.t), R1RT_OK);
}

TEST(Executor, PauseStopsTheTrajectoryClockAndResumeContinues) {
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 8, r.t, q00, 0.4, 0, 8, 8, 0.3, 0, true), r.t), R1RT_OK);  // 2.1 s
  auto tick = [&](int n) {
    for (int i = 0; i < n; ++i) {
      r.ex.keepalive(8, r.t);
      r.step();
    }
  };
  tick(250);  // 0.5 s
  r.ex.set_speed_scale(R1RT_GROUP_ARM_LEFT, 0.0, r.t);
  tick(200);  // the 0.3 s ramp + 0.1 s still
  const double q_paused = r.qo[14];
  tick(250);
  EXPECT_DOUBLE_EQ(r.qo[14], q_paused);  // still, but guarded and alive (TRAJECTORY)
  EXPECT_EQ(r.dqo[14], 0.0);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY);
  r.ex.set_speed_scale(R1RT_GROUP_ARM_LEFT, 1.0, r.t);
  tick(2000);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kEventDone);
  EXPECT_NEAR(r.qo[14], q00[0] + 0.4, 1e-12);
}

// ---------------------------------------------------------------- guards

r1rt_upper_cmd guarded(Rig& r, std::uint8_t enable, double amp = 0.4, double T = 2.0) {
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  auto c = traj(R1RT_GROUP_ARM_LEFT, 50, r.t, q00, amp, 0, 6, 6, T / 5.0, 0, true);
  c.guards.enable = enable;
  c.guards.lag_rad = 0.035f;  // v2 guard_lag
  return c;
}

void run_guarded(Rig& r, double s) {
  const int n = static_cast<int>(std::lround(s / kDt));
  for (int i = 0; i < n; ++i) {
    r.ex.keepalive(50, r.t);
    r.step();
  }
}

TEST(Executor, LagGuardStopsABlockedArm) {
  Rig r;
  ASSERT_EQ(r.ex.submit(guarded(r, 0x1), r.t), R1RT_OK);
  r.blocked = 14;  // the shoulder pitch hits something: its measured angle stays
  run_guarded(r, 1.0);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kGuardLag);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_HOLD);
  EXPECT_GT(r.ex.guard_margin(R1RT_GROUP_ARM_LEFT), 0.0);
}

TEST(Executor, LagGuardIgnoresTheExpectedLagAndAHoldOffset) {
  // the robot follows 0.02 rad behind along the motion + the left yaw holds 0.025 rad off its command (v2: 0.020-0.027
  // rad on 7 Oct): neither is "blocked"
  Rig r;
  r.follow = false;
  auto follow_lagging = [&](int n) {
    for (int i = 0; i < n; ++i) {
      r.ex.keepalive(50, r.t);
      r.step();
      for (int j = 14; j < N; ++j) {
        const double lag = r.dqo[j] > 0.01 ? 0.02 : (r.dqo[j] < -0.01 ? -0.02 : 0.0);
        r.q[j] = r.qo[j] - lag - (j == 16 ? 0.025 : 0.0);
      }
    }
  };
  follow_lagging(200);  // hold: the baseline e0 learns the yaw's offset
  ASSERT_EQ(r.ex.submit(guarded(r, 0x1, 0.4, 1.5), r.t), R1RT_OK);
  follow_lagging(900);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kEventDone);
  EXPECT_LT(r.ex.guard_margin(R1RT_GROUP_ARM_LEFT), 0.0);
}

// v2's own worked case (controller.py:2236-2244, session 9, 8 Oct): the right shoulder pitch began a lift 0.052 rad
// AHEAD of its command (the table bore the arm) and ended 0.024 behind. The bounded head start (0.03 rad) is ignored,
// the rest counts: along = 0.024 + 0.022 = 0.046 against ~0.08 allowed -> not blocked. A head start of 0.11 counts
// 0.08 more and stops the arm. Here on the left shoulder pitch (same gains, friction 2.5 N.m, kp 60).
double head_start_case(double ahead, bool* finished) {
  Rig r;
  r.follow = false;
  auto plant = [&](double offset_14) {
    for (int j = 14; j < N; ++j) r.q[j] = r.qo[j];
    r.q[14] = r.qo[14] + offset_14;
  };
  // HOLD 1 s with the joint `ahead` along the coming motion (the hold-lag average starts at the first sample, 0
  // here, and converges with a 0.1 s window: after 1 s it is within 4e-5 of the offset)
  for (int i = 0; i < 500; ++i) {
    r.step();
    plant(+ahead);
  }
  double q00[5];
  std::memcpy(q00, &r.qo[14], sizeof(q00));
  auto c = traj(R1RT_GROUP_ARM_LEFT, 90, r.t, q00, 0.3, 0, 6, 6, 0.4, 0, true);  // +0.3 rad in 2 s
  c.guards.enable = 0x1;
  c.guards.lag_rad = 0.035f;
  EXPECT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  for (int i = 0; i < 1100; ++i) {
    r.ex.keepalive(90, r.t);
    r.step();
    plant(-0.024);  // 0.024 behind along the motion
  }
  *finished = r.ex.last_guard(R1RT_GROUP_ARM_LEFT) == r1rt::kEventDone;
  EXPECT_TRUE(*finished || r.ex.last_guard(R1RT_GROUP_ARM_LEFT) == r1rt::kGuardLag);
  return r.ex.guard_margin(R1RT_GROUP_ARM_LEFT);
}

TEST(Executor, LagGuardReproducesV2sBoundedHeadStartCase) {
  bool finished = false;
  const double m = head_start_case(0.052, &finished);
  EXPECT_TRUE(finished);
  // over = along - friction / kp - |v| latency - lag = 0.046 - 2.5 / 60 - 0.015 |v| - 0.035, largest where the joint
  // moves slowest (|v| just above 0.01 rad/s at the ends of the path): -0.0307 - 0.015 |v|
  const double base = 0.046 - 2.5 / 60.0 - 0.035;
  EXPECT_LT(m, base - 0.015 * 0.01 + 1e-9);
  EXPECT_GT(m, base - 0.015 * 0.08);
  const double m2 = head_start_case(0.11, &finished);
  EXPECT_FALSE(finished) << "a head start beyond guard_ahead_max counts as lag again";
  EXPECT_GT(m2, 0.0);
}

TEST(Executor, LagGuardWaitsForTheGraceTimeAndPersistence) {
  Rig r;
  ASSERT_EQ(r.ex.submit(guarded(r, 0x1), r.t), R1RT_OK);
  r.blocked = 14;
  run_guarded(r, 0.19);  // inside guard_grace_s 0.2
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY);
}

TEST(Executor, TorqueGuardNeedsACalibratedJointAndTrips) {
  Rig r;
  EXPECT_NE(r.ex.submit(guarded(r, 0x2), r.t), R1RT_OK);  // no calibrated joint (the default config)
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectTorqueGuard);
  Rig c;
  c.cfg.tau_calibrated[17] = 1;
  ASSERT_EQ(c.ex.init(c.cfg), R1RT_OK);
  c.ex.reset(c.q, c.t);
  auto cmd = guarded(c, 0x2);
  cmd.guards.tau_nm = 0.5f;
  ASSERT_EQ(c.ex.submit(cmd, c.t), R1RT_OK);
  c.tau[17] = 1.0;  // a steady load: the baseline
  run_guarded(c, 0.4);
  EXPECT_EQ(c.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY);
  c.tau[17] = 1.4;  // +0.4 < 0.5
  run_guarded(c, 0.1);
  EXPECT_EQ(c.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY);
  c.tau[17] = 1.6;  // +0.6 for > 0.04 s
  run_guarded(c, 0.03);
  EXPECT_EQ(c.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY);
  run_guarded(c, 0.02);
  EXPECT_EQ(c.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kGuardTorque);
}

TEST(Executor, ContactGuardAnyAndAllCountOnlyRisingPads) {
  for (int all = 0; all <= 1; ++all) {
    Rig r;
    auto c = guarded(r, 0x4);
    c.guards.hand = 0;
    c.guards.contact_mask = (1u << 1) | (1u << 3);  // two tips
    c.guards.contact_all = static_cast<std::uint8_t>(all);
    r.hands.pad_contact[0] = 1u << 3;  // tip 3 already touching at the start
    ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
    run_guarded(r, 0.3);
    EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY) << "a pad touching at the start is no rise";
    r.hands.pad_contact[0] |= 1u << 1;  // tip 1 rises
    run_guarded(r, 0.05);
    if (all) {
      EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY) << "'all' still needs tip 3 to rise";
      r.hands.pad_contact[0] &= ~(1u << 3);
      run_guarded(r, 0.01);
      r.hands.pad_contact[0] |= 1u << 3;
      run_guarded(r, 0.05);
    }
    EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kGuardContact) << "all=" << all;
  }
}

TEST(Executor, ContactGuardGraceAndPersistence) {
  Rig r;
  auto c = guarded(r, 0x4);
  c.guards.contact_mask = 1u << 0;
  ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  run_guarded(r, 0.05);
  r.hands.pad_contact[0] = 1u;  // a contact inside the first 0.15 s: not yet
  run_guarded(r, 0.08);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY);
  r.hands.pad_contact[0] = 0u;
  run_guarded(r, 0.1);
  r.hands.pad_contact[0] = 1u;  // a 20 ms blip: no stop (persist 30 ms)
  run_guarded(r, 0.02);
  r.hands.pad_contact[0] = 0u;
  run_guarded(r, 0.02);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY);
  r.hands.pad_contact[0] = 1u;
  run_guarded(r, 0.04);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kGuardContact);
}

TEST(Executor, ContactGuardWithoutFreshHandsStateStops) {
  Rig r;
  auto c = guarded(r, 0x4);
  c.guards.contact_mask = 1u;
  ASSERT_EQ(r.ex.submit(c, r.t), R1RT_OK);
  run_guarded(r, 0.1);
  r.hands_on = false;
  run_guarded(r, 0.01);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kGuardContactStale);
  Rig s;
  auto c2 = guarded(s, 0x4);
  c2.guards.contact_mask = 1u;
  ASSERT_EQ(s.ex.submit(c2, s.t), R1RT_OK);
  run_guarded(s, 0.1);
  s.hands.valid = 0;
  run_guarded(s, 0.01);
  EXPECT_EQ(s.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kGuardContactStale);
}

TEST(Executor, ContactGuardArgumentsAreChecked) {
  Rig r;
  auto c = guarded(r, 0x4);
  c.guards.contact_mask = 0;
  EXPECT_NE(r.ex.submit(c, r.t), R1RT_OK);
  c.guards.contact_mask = 1u << 9;  // only 9 pads
  EXPECT_NE(r.ex.submit(c, r.t), R1RT_OK);
  c.guards.contact_mask = 1u;
  c.guards.hand = 2;
  EXPECT_NE(r.ex.submit(c, r.t), R1RT_OK);
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectContactGuard);
}

TEST(Executor, SurfaceGuardIsRefusedUntilTheFingertipModelExists) {
  Rig r;
  auto c = guarded(r, 0x8);
  c.guards.surface_z_m = 0.76;
  EXPECT_NE(r.ex.submit(c, r.t), R1RT_OK);
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectSurfaceGuard);
  double q[N] = {};
  EXPECT_FALSE(UpperBodyExecutor::surface_guard_hit(c.guards, q));
}

TEST(Executor, TiltStopsArmMotionButNotTheHead) {
  Rig r;
  r.tilt = 0.01;
  ASSERT_EQ(r.ex.submit(guarded(r, 0x0), r.t), R1RT_OK);  // no guard bits: the tilt stop applies to every arm motion
  auto h = base_cmd(R1RT_GROUP_HEAD, R1RT_SOURCE_STREAM, 60, r.t);
  h.q[0] = r.q[24];
  h.q[1] = r.q[25];
  h.ttl_ns = 300'000'000;
  ASSERT_EQ(r.ex.submit(h, r.t), R1RT_OK);
  run_guarded(r, 0.2);
  r.tilt = 0.01 + 0.06;  // 3.4 deg over the start: no stop
  run_guarded(r, 0.05);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_ARM_LEFT), R1RT_SOURCE_TRAJECTORY);
  r.tilt = 0.01 + 0.075;  // 4.3 deg
  r.step();
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kGuardTilt);
  EXPECT_EQ(r.ex.source(R1RT_GROUP_HEAD), R1RT_SOURCE_STREAM);
}

TEST(Executor, AGuardStopIsFastButBoundedByDdqStop) {
  Rig r;
  for (int j = 14; j < 19; ++j) r.cfg.ddq_stop[j] = 30.0;  // the group stops together: its slowest joint sets T
  ASSERT_EQ(r.ex.init(r.cfg), R1RT_OK);
  r.ex.reset(r.q, r.t);
  ASSERT_EQ(r.ex.submit(guarded(r, 0x0, 1.0, 1.6), r.t), R1RT_OK);  // peak ~1 rad/s at 0.8 s
  run_guarded(r, 0.8);
  const double v0 = r.dqo[14];
  ASSERT_GT(v0, 0.9);
  r.tilt = 0.2;
  double v_prev = v0, max_dv = 0.0;
  int ticks = 0;
  do {
    r.step();
    max_dv = std::max(max_dv, std::fabs(r.dqo[14] - v_prev));
    v_prev = r.dqo[14];
    ++ticks;
  } while (r.ex.stopping(R1RT_GROUP_ARM_LEFT) && ticks < 1000);
  // T = max(guard_stop_time_s 0.02, 1.5 v0 / ddq_stop): ~0.05 s, not the 0.15 s of a deadman stop
  EXPECT_LE(ticks, static_cast<int>(1.5 * v0 / 30.0 / kDt) + 2);
  EXPECT_LE(max_dv, 30.0 * kDt * 1.0001);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kGuardTilt);
}

TEST(Executor, NonFiniteStateDuringAGuardedPathStops) {
  Rig r;
  ASSERT_EQ(r.ex.submit(guarded(r, 0x1), r.t), R1RT_OK);
  run_guarded(r, 0.1);
  r.q[18] = std::nan("");
  r.follow = false;
  run_guarded(r, 0.01);
  EXPECT_EQ(r.ex.last_guard(R1RT_GROUP_ARM_LEFT), r1rt::kEventBadState);
  for (int j = 14; j < N; ++j) EXPECT_TRUE(std::isfinite(r.qo[j]));
}

// ---------------------------------------------------------------- bad input

TEST(Executor, BadRecordsAreRefusedAndChangeNothing) {
  Rig r;
  r.run(0.01);
  double before[N];
  std::memcpy(before, r.qo, sizeof(before));
  auto c = base_cmd(R1RT_GROUP_ARM_LEFT, R1RT_SOURCE_HOLD, 1, r.t);
  for (int k = 0; k < 5; ++k) c.q[k] = r.q[14 + k];
  auto expect_reject = [&](r1rt_upper_cmd x, std::uint8_t why) {
    EXPECT_EQ(r.ex.submit(x, r.t), R1RT_E_INVALID_ARG);
    EXPECT_EQ(r.ex.last_reject(), why);
  };
  auto x = c;
  x.group = 3;
  expect_reject(x, r1rt::kRejectGroup);
  x = c;
  x.n_joints = 2;
  expect_reject(x, r1rt::kRejectNJoints);
  x = c;
  x.source = 9;
  expect_reject(x, r1rt::kRejectSource);
  x = c;
  x.q[2] = std::nan("");
  expect_reject(x, r1rt::kRejectNonFinite);
  x = c;
  x.tau_ff[1] = INFINITY;
  expect_reject(x, r1rt::kRejectNonFinite);
  x = c;
  x.kp[0] = 0.0;
  x.kd[0] = 1.0;  // kp 0 = limp: only FREEDRIVE
  expect_reject(x, r1rt::kRejectGains);
  x = c;
  x.kp[0] = 500.0;
  expect_reject(x, r1rt::kRejectGains);
  x = c;
  x.kp[0] = 30.0;
  x.kd[0] = -1.0;
  expect_reject(x, r1rt::kRejectGains);
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  auto t = traj(R1RT_GROUP_ARM_LEFT, 2, r.t, q00, 0.3, 0, 4, 4, 0.2, 0, true);
  t.knots[2].q[1] = 9.0;  // outside the limits
  expect_reject(t, r1rt::kRejectKnots);
  t = traj(R1RT_GROUP_ARM_LEFT, 2, r.t, q00, 0.3, 0, 4, 4, 0.2, 0, true);
  t.knots[1].dq[0] = 5.0;  // > dq_max
  expect_reject(t, r1rt::kRejectKnots);
  t = traj(R1RT_GROUP_ARM_LEFT, 2, r.t, q00, 0.3, 0, 4, 4, 0.2, 0, true);
  t.knots[2].t_ns = t.knots[1].t_ns;
  expect_reject(t, r1rt::kRejectKnots);
  t = traj(R1RT_GROUP_ARM_LEFT, 2, r.t, q00, 0.3, 0, 4, 4, 0.2, 0, true);
  t.knots[3].ddq[4] = std::nan("");
  expect_reject(t, r1rt::kRejectNonFinite);
  t.n_knots = 0;
  expect_reject(t, r1rt::kRejectKnots);
  t = traj(R1RT_GROUP_ARM_LEFT, 2, r.t, q00, 0.3, 0, 4, 4, 0.2, 0, true);
  t.guards.enable = 0x1;
  t.guards.lag_rad = -1.0f;
  expect_reject(t, r1rt::kRejectGains);
  r.step();
  for (int j = 14; j < N; ++j) EXPECT_DOUBLE_EQ(r.qo[j], before[j]);
  for (int g = 0; g < 3; ++g) EXPECT_EQ(r.ex.source(static_cast<r1rt_group>(g)), R1RT_SOURCE_HOLD);
}

TEST(Executor, TooLongATrajectoryIsRefused) {
  Rig r;
  r.cfg.max_traj_knots = 2 * R1RT_MAX_KNOTS;
  ASSERT_EQ(r.ex.init(r.cfg), R1RT_OK);
  r.ex.reset(r.q, r.t);
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 1, r.t, q00, 0.3, 0, 8, 30, 0.1, 0, false), r.t), R1RT_OK);
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 1, r.t, q00, 0.3, 8, 8, 30, 0.1, 1, false), r.t), R1RT_OK);
  EXPECT_NE(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 1, r.t, q00, 0.3, 16, 8, 30, 0.1, 2, false), r.t), R1RT_OK);
  EXPECT_EQ(r.ex.last_reject(), r1rt::kRejectTooLong);
}

TEST(Executor, ALoopGapDoesNotJumpTheTargets) {
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  ASSERT_EQ(r.ex.submit(traj(R1RT_GROUP_ARM_LEFT, 1, r.t, q00, 0.5, 0, 6, 6, 0.2, 0, true), r.t), R1RT_OK);
  run_guarded(r, 0.3);
  const double q_before = r.qo[14];
  r.t += 200'000'000;  // the writer stalled 0.2 s
  r.ex.keepalive(1, r.t);
  r.step();
  EXPECT_LE(std::fabs(r.qo[14] - q_before), r.cfg.dq_max[14] * 2.0 * kDt + 1e-9);
}

// ---------------------------------------------------------------- properties: limits, smoothness, determinism

struct Fuzz {
  std::mt19937_64 rng;
  explicit Fuzz(std::uint64_t seed) : rng(seed) {}
  double u(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
  int i(int a, int b) { return std::uniform_int_distribution<int>(a, b)(rng); }
};

// A random session: HOLD / STREAM / TRAJECTORY records (some bad), keepalives that sometimes stop, stops, pauses,
// tilt and contact changes. Returns a checksum of every output.
double fuzz_session(std::uint64_t seed, bool check) {
  Rig r;
  Fuzz f(seed);
  double sum = 0.0;
  double v_prev[N] = {}, q_prev[N];
  std::memcpy(q_prev, r.q, sizeof(q_prev));
  std::uint64_t id = 100;
  bool alive = true;
  for (int i = 0; i < 6000; ++i) {
    const int what = f.i(0, 199);
    const auto g = static_cast<r1rt_group>(f.i(0, 2));
    const int j0 = first_joint(g);
    const int n = g == R1RT_GROUP_HEAD ? 2 : 5;
    if (what < 3) {
      auto c = base_cmd(g, R1RT_SOURCE_HOLD, ++id, r.t);
      c.stop_epoch = r.ex.stop_epoch();
      for (int k = 0; k < n; ++k) c.q[k] = r.qo[j0 + k] + f.u(-1.0, 1.0);
      r.ex.submit(c, r.t);
    } else if (what < 6) {
      auto c = base_cmd(g, R1RT_SOURCE_STREAM, ++id, r.t);
      c.stop_epoch = r.ex.stop_epoch();
      for (int k = 0; k < n; ++k) {
        c.q[k] = r.qo[j0 + k] + f.u(-0.3, 0.3);
        c.dq[k] = f.u(-3.0, 3.0);
      }
      c.ttl_ns = f.i(10, 300) * 1'000'000;
      r.ex.submit(c, r.t);
    } else if (what < 9) {
      double q00[5] = {};
      for (int k = 0; k < n; ++k) q00[k] = r.qo[j0 + k];
      auto c = traj(g, ++id, r.t, q00, f.u(-0.6, 0.6), 0, f.i(2, 8), 8, f.u(0.05, 0.4), 0, f.i(0, 3) > 0);
      c.stop_epoch = r.ex.stop_epoch();
      c.guards.enable = static_cast<std::uint8_t>(f.i(0, 1) | (f.i(0, 1) << 2));
      c.guards.lag_rad = 0.035f;
      c.guards.contact_mask = 1u << f.i(0, 8);
      r.ex.submit(c, r.t);
      alive = f.i(0, 4) > 0;
    } else if (what == 9) {
      r.ex.stop(r.t);
    } else if (what == 10) {
      r.ex.set_speed_scale(g, f.u(-0.5, 1.5), r.t);
    } else if (what == 11) {
      r.tilt = f.u(0.0, 0.1);
    } else if (what == 12) {
      r.hands.pad_contact[0] = static_cast<std::uint32_t>(f.i(0, 511));
    }
    if (alive) {
      for (std::uint64_t k = id - 3; k <= id; ++k) r.ex.keepalive(k, r.t);
    }
    bool was_stopping[3];
    for (int g2 = 0; g2 < 3; ++g2) was_stopping[g2] = r.ex.stopping(static_cast<r1rt_group>(g2));
    r.step();
    for (int j = 14; j < N; ++j) {
      sum += r.qo[j] * 1.1 + r.dqo[j] * 1.3 + r.kpo[j] * 1.7 + r.kdo[j] * 1.9 + r.ffo[j];
      if (check) {
        EXPECT_GE(r.qo[j], r.cfg.q_min[j]);
        EXPECT_LE(r.qo[j], r.cfg.q_max[j]);
        EXPECT_LE(std::fabs(r.dqo[j]), r.cfg.dq_max[j] + 1e-12);
        EXPECT_LE(std::fabs(r.qo[j] - q_prev[j]), r.cfg.dq_max[j] * kDt * (1.0 + 1e-9) + 1e-10);
        EXPECT_GT(r.kpo[j], 0.0) << "never limp outside FREEDRIVE";
        EXPECT_LE(r.kpo[j], r.cfg.kp_max[j] + 1e-12);
        // a stop moves the target with a bounded deceleration (the position's second difference), unless it
        // reaches a joint limit (then the clamp wins)
        const auto grp = static_cast<r1rt_group>(j < 19 ? 0 : (j < 24 ? 1 : 2));
        const bool stop_now = r.ex.stopping(grp);
        const double vq = (r.qo[j] - q_prev[j]) / kDt;
        const bool at_limit = r.qo[j] <= r.cfg.q_min[j] || r.qo[j] >= r.cfg.q_max[j];
        if (stop_now && was_stopping[grp] && !at_limit) {
          EXPECT_LE(std::fabs(vq - v_prev[j]), r.cfg.ddq_stop[j] * kDt * 1.0001 + 1e-6) << "seed " << seed;
        }
      }
      v_prev[j] = (r.qo[j] - q_prev[j]) / kDt;
      q_prev[j] = r.qo[j];
    }
  }
  return sum;
}

TEST(Executor, RandomSessionsStayInsideEveryLimit) {
  for (std::uint64_t seed = 1; seed <= 6; ++seed) fuzz_session(seed, true);
}

TEST(Executor, IsDeterministic) {
  EXPECT_EQ(fuzz_session(42, false), fuzz_session(42, false));
  EXPECT_NE(fuzz_session(42, false), fuzz_session(43, false));
}

TEST(Executor, TheRtCallsAllocateNothing) {
  Rig r;
  double q00[5];
  std::memcpy(q00, &r.q[14], sizeof(q00));
  const auto t1 = traj(R1RT_GROUP_ARM_LEFT, 1, r.t, q00, 0.3, 0, 8, 16, 0.1, 0, false);
  const auto t2 = traj(R1RT_GROUP_ARM_LEFT, 1, r.t, q00, 0.3, 8, 8, 16, 0.1, 1, true);
  double grav[N] = {};
  const long before = g_news.load();
  for (int rep = 0; rep < 20; ++rep) {
    r.ex.submit(t1, r.t);
    r.ex.submit(t2, r.t);
    auto s = base_cmd(R1RT_GROUP_HEAD, R1RT_SOURCE_STREAM, 2, r.t);
    s.q[0] = r.qo[24];
    s.q[1] = r.qo[25];
    s.ttl_ns = 100'000'000;
    r.ex.submit(s, r.t);
    r.ex.set_gravity_ff(grav, r.t);
    r.ex.set_speed_scale(R1RT_GROUP_ARM_LEFT, 0.5, r.t);
    for (int i = 0; i < 500; ++i) {
      r.ex.keepalive(1, r.t);
      r.step();
    }
    r.ex.stop(r.t);
    r.run(0.2);
  }
  EXPECT_EQ(g_news.load(), before) << "an RT call allocated";
}

}  // namespace
