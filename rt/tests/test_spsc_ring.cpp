// SpscRing: FIFO semantics, full/empty, drop counting, wrap-around, and 2-thread stress tests.
#include "r1rt/spsc_ring.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <thread>

#include "r1rt/types.h"

namespace {

struct Item {
  std::uint64_t seq;
  std::uint64_t payload[7];
};

Item make_item(std::uint64_t seq) {
  Item it{};
  it.seq = seq;
  for (std::uint64_t i = 0; i < 7; ++i) it.payload[i] = seq ^ (i * 0x5555ull);
  return it;
}

bool consistent(const Item& it) {
  for (std::uint64_t i = 0; i < 7; ++i)
    if (it.payload[i] != (it.seq ^ (i * 0x5555ull))) return false;
  return true;
}

TEST(SpscRing, EmptyPopFails) {
  r1rt::SpscRing<Item, 4> r;
  Item out{};
  EXPECT_FALSE(r.try_pop(out));
  EXPECT_TRUE(r.empty());
  EXPECT_EQ(r.drops(), 0u);
}

TEST(SpscRing, FifoOrderAndFullDropsCounted) {
  r1rt::SpscRing<Item, 4> r;
  for (std::uint64_t k = 0; k < 4; ++k) ASSERT_TRUE(r.try_push(make_item(k)));
  EXPECT_EQ(r.size(), 4u);
  EXPECT_FALSE(r.try_push(make_item(100)));  // full: never waits, drops
  EXPECT_FALSE(r.try_push(make_item(101)));
  EXPECT_EQ(r.drops(), 2u);
  Item out{};
  for (std::uint64_t k = 0; k < 4; ++k) {
    ASSERT_TRUE(r.try_pop(out));
    EXPECT_EQ(out.seq, k);
  }
  EXPECT_FALSE(r.try_pop(out));
  EXPECT_EQ(r.pushed(), 4u);
  EXPECT_EQ(r.popped(), 4u);
}

TEST(SpscRing, WrapAroundManyTimes) {
  r1rt::SpscRing<Item, 4> r;
  std::uint64_t next_in = 0, next_out = 0;
  Item out{};
  for (int round = 0; round < 1000; ++round) {
    const int n = 1 + round % 4;
    for (int i = 0; i < n; ++i) ASSERT_TRUE(r.try_push(make_item(next_in++)));
    for (int i = 0; i < n; ++i) {
      ASSERT_TRUE(r.try_pop(out));
      ASSERT_EQ(out.seq, next_out++);
      ASSERT_TRUE(consistent(out));
    }
  }
  EXPECT_EQ(r.drops(), 0u);
  EXPECT_TRUE(r.empty());
}

TEST(SpscRing, TwoThreadStressLossless) {
  // The consumer keeps up; the producer retries on full (a test choice: r1rt itself never retries).
  constexpr std::uint64_t kItems = 1'000'000;
  auto r = std::make_unique<r1rt::SpscRing<Item, 1024>>();
  std::uint64_t bad = 0, received = 0;
  std::thread consumer([&] {
    Item out{};
    std::uint64_t expect = 0;
    while (expect < kItems) {
      if (r->try_pop(out)) {
        if (out.seq != expect || !consistent(out)) ++bad;
        ++expect;
        ++received;
      }
    }
  });
  for (std::uint64_t k = 0; k < kItems;)
    if (r->try_push(make_item(k))) ++k;
  consumer.join();
  EXPECT_EQ(received, kItems);
  EXPECT_EQ(bad, 0u);
}

TEST(SpscRing, TwoThreadStressProducerNeverWaits) {
  // The real-time pattern: push once per tick, never retry; every attempt is either delivered or counted.
  constexpr std::uint64_t kAttempts = 1'000'000;
  auto r = std::make_unique<r1rt::SpscRing<Item, 64>>();
  std::atomic<bool> done{false};
  std::uint64_t bad = 0, received = 0, last = 0;
  bool first = true;
  std::thread consumer([&] {
    Item out{};
    for (;;) {
      if (r->try_pop(out)) {
        if (!consistent(out) || (!first && out.seq <= last)) ++bad;  // strictly increasing, gaps = drops
        first = false;
        last = out.seq;
        ++received;
      } else if (done.load(std::memory_order_acquire) && r->empty()) {
        break;
      }
    }
  });
  std::uint64_t delivered = 0;
  for (std::uint64_t k = 0; k < kAttempts; ++k) delivered += r->try_push(make_item(k)) ? 1 : 0;
  done.store(true, std::memory_order_release);
  consumer.join();
  EXPECT_EQ(bad, 0u);
  EXPECT_EQ(received, delivered);
  EXPECT_EQ(delivered + r->drops(), kAttempts);
  std::printf("[ ring stress ] attempts %llu, delivered %llu, dropped %llu\n",
              static_cast<unsigned long long>(kAttempts), static_cast<unsigned long long>(delivered),
              static_cast<unsigned long long>(r->drops()));
}

TEST(SpscRing, TelemetryRecordsAcrossThreads) {
  // The real record type (1136 B) through a 4096-slot ring, as r1rt -> r1log.
  constexpr std::uint64_t kRecords = 100'000;
  auto r = std::make_unique<r1rt::SpscRing<r1rt_telemetry_record, 4096>>();
  std::uint64_t bad = 0;
  std::thread consumer([&] {
    r1rt_telemetry_record rec{};
    std::uint64_t expect = 0;
    while (expect < kRecords) {
      if (r->try_pop(rec)) {
        if (rec.seq != expect || rec.q[25] != static_cast<float>(expect % 1000) ||
            rec.motor_state[0] != static_cast<std::uint32_t>(expect))
          ++bad;
        ++expect;
      }
    }
  });
  r1rt_telemetry_record rec{};
  for (std::uint64_t k = 0; k < kRecords;) {
    rec.seq = k;
    rec.q[25] = static_cast<float>(k % 1000);
    rec.motor_state[0] = static_cast<std::uint32_t>(k);
    if (r->try_push(rec)) ++k;
  }
  consumer.join();
  EXPECT_EQ(bad, 0u);
}

}  // namespace
