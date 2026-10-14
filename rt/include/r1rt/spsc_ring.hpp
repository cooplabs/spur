// r1rt/spsc_ring.hpp - bounded single-producer single-consumer ring for a trivially copyable T, usable in shared
// memory (fixed size, no pointers, placement-constructed once by the segment's creator).
//
// The real-time side never waits: try_push fails when the ring is full and counts the drop (DESIGN §9: "rings drop
// with a counter and the RT core never waits"); try_pop fails when it is empty.
//
// Indices are free-running 64-bit counters (no wrap-around in practice: 2^64 pushes), slot = index & (N - 1).
// Ordering (Lamport queue): the producer writes the slot, then publishes head with a release store; the consumer
// reads head with acquire, reads the slot, then frees it with a release store of tail; the producer reads tail with
// acquire before reusing a slot. Each counter has one writer. head, tail and the counters sit on separate 64-byte
// cache lines (a constant: libc++ lacks std::hardware_destructive_interference_size).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace r1rt {

inline constexpr std::size_t kCacheLine = 64;

template <typename T, std::size_t N>
class SpscRing {
  static_assert(std::is_trivially_copyable_v<T>, "SpscRing<T>: T must be trivially copyable");
  static_assert(N >= 2 && (N & (N - 1)) == 0, "SpscRing: capacity must be a power of two");
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "64-bit atomics must be lock-free");

 public:
  static constexpr std::size_t kCapacity = N;

  SpscRing() noexcept : slots_{} {
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
    drops_.store(0, std::memory_order_relaxed);
  }
  SpscRing(const SpscRing&) = delete;
  SpscRing& operator=(const SpscRing&) = delete;

  // Producer only. Wait-free. false = full (the item is dropped and counted).
  bool try_push(const T& item) noexcept {
    const std::uint64_t h = head_.load(std::memory_order_relaxed);
    const std::uint64_t t = tail_.load(std::memory_order_acquire);
    if (h - t >= N) {
      drops_.store(drops_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);  // one writer
      return false;
    }
    slots_[h & (N - 1)] = item;
    head_.store(h + 1, std::memory_order_release);
    return true;
  }

  // Consumer only. Wait-free. false = empty.
  bool try_pop(T& out) noexcept {
    const std::uint64_t t = tail_.load(std::memory_order_relaxed);
    const std::uint64_t h = head_.load(std::memory_order_acquire);
    if (t == h) return false;
    out = slots_[t & (N - 1)];
    tail_.store(t + 1, std::memory_order_release);
    return true;
  }

  // Approximate when called from a third party; exact from the producer (upper bound) or consumer (lower bound).
  std::size_t size() const noexcept {
    const std::uint64_t t = tail_.load(std::memory_order_acquire);
    const std::uint64_t h = head_.load(std::memory_order_acquire);
    return static_cast<std::size_t>(h - t);
  }
  bool empty() const noexcept { return size() == 0; }
  std::uint64_t pushed() const noexcept { return head_.load(std::memory_order_acquire); }
  std::uint64_t popped() const noexcept { return tail_.load(std::memory_order_acquire); }
  std::uint64_t drops() const noexcept { return drops_.load(std::memory_order_relaxed); }

 private:
  alignas(kCacheLine) std::atomic<std::uint64_t> head_;   // written by the producer
  alignas(kCacheLine) std::atomic<std::uint64_t> drops_;  // written by the producer
  alignas(kCacheLine) std::atomic<std::uint64_t> tail_;   // written by the consumer
  alignas(kCacheLine) T slots_[N];
};

}  // namespace r1rt
