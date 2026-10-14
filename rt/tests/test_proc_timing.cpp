// The processes' periodic loop (rt_thread.hpp): absolute deadlines (no drift), lateness measured, a long stall skips
// the missed deadlines instead of bursting; the numbers of THIS machine printed ([ info ]; macOS has no SCHED_FIFO).
#include <gtest/gtest.h>

#include <cstdio>

#include "r1rt/rt_thread.hpp"

namespace {
using namespace r1rt::proc;

TEST(ProcTiming, PeriodicKeepsAbsoluteDeadlinesAndMeasuresLateness) {
  const RtGrant g = request_rt_thread(90, 7, 2'000'000, 1'000'000);
  std::printf("[ info ] os %s: fifo %d (errno %d) affinity %d (errno %d) time-constraint %d\n", os_name(), g.fifo,
              g.fifo_errno, g.affinity, g.affinity_errno, g.time_constraint);
  constexpr int kTicks = 1000;  // 2 s at 500 Hz
  const std::int64_t t0 = now_ns() + 2'000'000;
  Periodic per(2'000'000, t0);
  TimingStats st;
  std::int64_t last = 0;
  for (int i = 0; i < kTicks; ++i) {
    std::int64_t late = 0;
    last = per.wait(&late);
    EXPECT_GE(late, 0);
    st.add(late);
  }
  char buf[512];
  std::printf("[ info ] 500 Hz wake-up lateness: %s; skipped %llu\n", st.json(buf, sizeof buf),
              static_cast<unsigned long long>(per.skipped()));
  // No drift: the k-th wake is the k-th deadline plus its lateness (deadlines are absolute) - the loop ends within a
  // few ms of t0 + 999 periods even though every single wake is late, unless deadlines were skipped (counted).
  const std::int64_t expect_last = t0 + static_cast<std::int64_t>(kTicks - 1 + static_cast<int>(per.skipped())) * 2'000'000;
  EXPECT_LT(last - expect_last, 20'000'000) << "under heavy load a single late wake can be long; a drift grows";
  EXPECT_EQ(st.n, static_cast<std::uint64_t>(kTicks));
  EXPECT_LT(st.mean_ns(), 2'000'000.0) << "mean lateness above one period: the loop cannot keep 500 Hz here";
}

// The integrator's session-6 CI caught one 500 Hz wake 5 us BEFORE its deadline (lateness -5000 ns) on macOS:
// mach_wait_until can return before the CLOCK_MONOTONIC deadline (the interval is converted between two clocks at
// one instant; an interrupted wait returns early too). An absolute-deadline loop must never act early.
TEST(ProcTiming, SleepUntilNeverReturnsBeforeTheDeadline) {
  int early = 0;
  std::int64_t worst = 0;
  for (int i = 0; i < 20000; ++i) {
    const std::int64_t t = now_ns() + 20'000 + (i % 10) * 15'000;  // 20-155 us ahead
    sleep_until(t);
    const std::int64_t d = now_ns() - t;
    if (d < 0) {
      ++early;
      if (d < worst) worst = d;
    }
  }
  std::printf("[ info ] sleep_until: %d of 20000 returned early (worst %lld ns)\n", early, static_cast<long long>(worst));
  EXPECT_EQ(early, 0);
}

TEST(ProcTiming, ALongStallSkipsMissedDeadlinesInsteadOfBursting) {
  Periodic per(2'000'000, now_ns() + 2'000'000);
  std::int64_t late = 0;
  per.wait(&late);
  sleep_for(50'000'000);  // 25 periods
  const std::int64_t w = per.wait(&late);
  EXPECT_GE(late, 45'000'000);
  EXPECT_GE(per.skipped(), 23u);
  EXPECT_GT(per.next_deadline_ns(), w);  // the next deadline is in the future: no catch-up burst
  EXPECT_LE(per.next_deadline_ns() - w, 2'000'000);
}

TEST(ProcTiming, StatsHistogramEdges) {
  TimingStats st;
  st.add(10'000);       // < 50 us
  st.add(60'000);       // < 100 us
  st.add(1'500'000);    // < 2 ms
  st.add(30'000'000);   // >= 20 ms
  EXPECT_EQ(st.hist[0], 1u);
  EXPECT_EQ(st.hist[1], 1u);
  EXPECT_EQ(st.hist[5], 1u);
  EXPECT_EQ(st.hist[9], 1u);
  EXPECT_EQ(st.max_ns, 30'000'000);
}

}  // namespace
