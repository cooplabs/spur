// r1rt::ResumeClock (session 8, HW1 fix #29): a sample drained from the stop's backlog (fresh receive stamp, old board
// tick) is NOT live; the current one is; without a learned tick rate everything counts as live.
#include "r1rt/resume.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

constexpr std::int64_t kMs = 1'000'000;

// A 1 kHz board tick received every 2 ms for `ms` milliseconds from (tick0, t0).
void feed(r1rt::ResumeClock& c, std::uint32_t tick0, std::int64_t t0, int ms) {
  for (int k = 0; k <= ms; k += 2) c.live_sample(tick0 + static_cast<std::uint32_t>(k), t0 + k * kMs);
}

}  // namespace

TEST(ResumeClock, LearnsTheTickRateFromLiveSamples) {
  r1rt::ResumeClock c;
  EXPECT_FALSE(c.has_rate());
  feed(c, 1000, 0, 500);
  ASSERT_TRUE(c.has_rate());
  EXPECT_NEAR(c.rate_per_ns() * 1e6, 1.0, 1e-9);  // 1 tick per ms
}

TEST(ResumeClock, ABacklogSampleIsNotLiveTheCurrentOneIs) {
  r1rt::ResumeClock c;
  feed(c, 1000, 0, 500);                // live up to tick 1500 at t = 500 ms
  c.mark_stop(1500, 500 * kMs);         // the process stopped there for 2 s
  const std::int64_t now = 2510 * kMs;  // the first tick after the stop
  EXPECT_FALSE(c.caught_up(1502, now));  // drained from the socket: 2 s old, stamped "now"
  EXPECT_FALSE(c.caught_up(3400, now));  // still 110 ms of board time behind
  EXPECT_TRUE(c.caught_up(3495, now));   // within the 20 ms slack
  EXPECT_TRUE(c.caught_up(3510, now));
}

TEST(ResumeClock, WrapsAroundTheTickCounter) {
  r1rt::ResumeClock c;
  const std::uint32_t t0 = 0xFFFFFF00u;
  feed(c, t0, 0, 400);
  c.mark_stop(t0 + 400u, 400 * kMs);
  EXPECT_FALSE(c.caught_up(t0 + 402u, 1400 * kMs));
  EXPECT_TRUE(c.caught_up(t0 + 1400u, 1400 * kMs));  // wrapped past 2^32
}

TEST(ResumeClock, WithoutARateEverythingIsLive) {
  r1rt::ResumeClock c;
  c.live_sample(5, 0);
  c.mark_stop(5, 0);
  EXPECT_TRUE(c.caught_up(5, 3000 * kMs));
}
