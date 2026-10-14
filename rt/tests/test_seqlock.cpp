// Seqlock: single-thread semantics, the bounded reader, and a 2-thread torn-read stress test.
#include "r1rt/seqlock.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <cstdint>
#include <thread>

namespace r1rt {
struct SeqlockTestAccess {
  template <typename T>
  static void set_seq(Seqlock<T>& s, std::uint64_t v) {
    s.seq_.store(v, std::memory_order_release);
  }
};
}  // namespace r1rt

namespace {

struct Odd13 {  // a size that is not a multiple of 8
  std::uint8_t b[13];
};

struct Payload {  // ~0.25 kB with a checksum: a torn read cannot pass the check
  std::uint64_t seq;
  std::uint64_t a[30];
  std::uint64_t checksum;
};

Payload make_payload(std::uint64_t seq) {
  Payload p{};
  p.seq = seq;
  std::uint64_t sum = seq;
  for (std::uint64_t i = 0; i < 30; ++i) {
    p.a[i] = seq * 0x9E3779B97F4A7C15ull + i;
    sum ^= p.a[i] + i;
  }
  p.checksum = sum;
  return p;
}

bool consistent(const Payload& p) {
  std::uint64_t sum = p.seq;
  for (std::uint64_t i = 0; i < 30; ++i) {
    if (p.a[i] != p.seq * 0x9E3779B97F4A7C15ull + i) return false;
    sum ^= p.a[i] + i;
  }
  return sum == p.checksum;
}

TEST(Seqlock, UnwrittenReadsZeroWithSequenceZero) {
  r1rt::Seqlock<Payload> s;
  Payload p = make_payload(7);
  std::uint64_t seq = 99;
  ASSERT_TRUE(s.try_read(p, 4, &seq));
  EXPECT_EQ(seq, 0u);
  EXPECT_EQ(p.seq, 0u);
  EXPECT_EQ(s.writes(), 0u);
}

TEST(Seqlock, WriteThenRead) {
  r1rt::Seqlock<Payload> s;
  for (std::uint64_t k = 1; k <= 5; ++k) {
    s.write(make_payload(k));
    Payload p{};
    std::uint64_t seq = 0;
    ASSERT_TRUE(s.try_read(p, 1, &seq));
    EXPECT_EQ(p.seq, k);
    EXPECT_TRUE(consistent(p));
    EXPECT_EQ(seq, 2 * k);
    EXPECT_EQ(s.writes(), k);
  }
}

TEST(Seqlock, SizeNotMultipleOfEight) {
  r1rt::Seqlock<Odd13> s;
  static_assert(r1rt::Seqlock<Odd13>::kWords == 2);
  Odd13 in{};
  for (int i = 0; i < 13; ++i) in.b[i] = static_cast<std::uint8_t>(i * 17 + 3);
  s.write(in);
  Odd13 out{};
  ASSERT_TRUE(s.try_read(out));
  for (int i = 0; i < 13; ++i) EXPECT_EQ(out.b[i], in.b[i]);
}

TEST(Seqlock, ReaderGivesUpWhenWriterDiedMidWrite) {
  r1rt::Seqlock<Payload> s;
  s.write(make_payload(1));
  r1rt::SeqlockTestAccess::set_seq(s, 3);  // odd forever: the writer stopped inside write()
  Payload p{};
  EXPECT_FALSE(s.try_read(p, 1000));       // bounded: returns instead of spinning forever
}

TEST(Seqlock, TwoThreadStressHammeringWriterNoTornReads) {
  r1rt::Seqlock<Payload> s;
  constexpr std::uint64_t kWrites = 1'000'000;
  std::atomic<bool> done{false};
  std::uint64_t ok = 0, failed = 0, torn = 0, backwards = 0, last = 0;
  std::thread reader([&] {
    Payload p{};
    std::uint64_t seq = 0;
    while (!done.load(std::memory_order_acquire)) {
      if (s.try_read(p, 256, &seq)) {
        if (seq == 0) continue;  // never written yet: the all-zero payload (not a torn read)
        ++ok;
        if (!consistent(p)) ++torn;
        if (p.seq < last) ++backwards;
        last = p.seq;
      } else {
        ++failed;
      }
    }
  });
  for (std::uint64_t k = 1; k <= kWrites; ++k) s.write(make_payload(k));
  done.store(true, std::memory_order_release);
  reader.join();
  Payload final_p{};
  ASSERT_TRUE(s.try_read(final_p));
  EXPECT_EQ(final_p.seq, kWrites);
  EXPECT_EQ(torn, 0u);
  EXPECT_EQ(backwards, 0u);
  EXPECT_GT(ok, 0u);
  std::printf("[ seqlock stress ] writes %llu, reads ok %llu, gave up %llu, torn %llu\n",
              static_cast<unsigned long long>(kWrites), static_cast<unsigned long long>(ok),
              static_cast<unsigned long long>(failed), static_cast<unsigned long long>(torn));
}

}  // namespace

namespace {

TEST(Seqlock, TwoThreadStressPacedWriterReadersSucceed) {
  // Closer to the real pattern (one write per writer tick): the writer pauses ~2 us between writes, so most reads
  // succeed and every successful read is checked. Two readers.
  r1rt::Seqlock<Payload> s;
  constexpr std::uint64_t kWrites = 200'000;
  std::atomic<bool> done{false};
  struct Stats {
    std::uint64_t ok = 0, failed = 0, torn = 0, backwards = 0, last = 0;
  } st[2];
  auto reader = [&](Stats& r) {
    Payload p{};
    std::uint64_t seq = 0;
    while (!done.load(std::memory_order_acquire)) {
      if (s.try_read(p, 64, &seq)) {
        if (seq == 0) continue;  // never written yet: the all-zero payload (not a torn read)
        ++r.ok;
        if (!consistent(p)) ++r.torn;
        if (p.seq < r.last) ++r.backwards;
        r.last = p.seq;
      } else {
        ++r.failed;
      }
    }
  };
  std::thread r0(reader, std::ref(st[0])), r1(reader, std::ref(st[1]));
  for (std::uint64_t k = 1; k <= kWrites; ++k) {
    s.write(make_payload(k));
    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(2);
    while (std::chrono::steady_clock::now() < until) {
    }
  }
  done.store(true, std::memory_order_release);
  r0.join();
  r1.join();
  for (const Stats& r : st) {
    EXPECT_EQ(r.torn, 0u);
    EXPECT_EQ(r.backwards, 0u);
    EXPECT_GT(r.ok, kWrites / 10);
    std::printf("[ seqlock paced ] writes %llu, reads ok %llu, gave up %llu, torn %llu\n",
                static_cast<unsigned long long>(kWrites), static_cast<unsigned long long>(r.ok),
                static_cast<unsigned long long>(r.failed), static_cast<unsigned long long>(r.torn));
  }
}

}  // namespace
