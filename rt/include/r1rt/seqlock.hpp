// r1rt/seqlock.hpp - single-writer sequence lock for a trivially copyable T, usable in shared memory.
//
// Writer (exactly one thread in one process): never blocks, never allocates, constant time.
// Readers (any number, any process): copy the value and retry if a write overlapped; the number of tries is bounded,
// so a reader never spins forever (e.g. when the writer process died in the middle of a write and left the
// sequence odd); it reports failure instead and the caller treats the value as stale.
//
// Memory ordering (H.-J. Boehm, "Can seqlocks get along with programming language memory models?", MSPC 2012,
// the fence-based variant): the payload is stored as relaxed atomic 64-bit words, so concurrent reads and writes
// are not data races (ThreadSanitizer-clean), and
//   writer: seq <- s+1 (relaxed); fence(release); words <- v (relaxed); seq <- s+2 (release)
//   reader: s1 <- seq (acquire); v <- words (relaxed); fence(acquire); s2 <- seq (relaxed); ok iff s1 == s2, even
// A reader that sees s1 == s2 (even) read words that were all written before the release store of s1 and none
// written after the odd store of the next write (the release fence orders that odd store before the new words).
//
// No pointers inside: the object can live in a mapped segment (placement-constructed once by its creator).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace r1rt {

struct SeqlockTestAccess;  // unit tests only (simulates a writer that died mid-write)

static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "64-bit atomics must be lock-free (address-free)");

template <typename T>
class Seqlock {
  static_assert(std::is_trivially_copyable_v<T>, "Seqlock<T>: T must be trivially copyable");

 public:
  static constexpr std::size_t kWords = (sizeof(T) + 7) / 8;

  Seqlock() noexcept {
    seq_.store(0, std::memory_order_relaxed);
    for (auto& w : words_) w.store(0, std::memory_order_relaxed);
  }
  Seqlock(const Seqlock&) = delete;
  Seqlock& operator=(const Seqlock&) = delete;

  // Single writer only. Wait-free.
  void write(const T& value) noexcept {
    std::uint64_t buf[kWords] = {};
    std::memcpy(buf, &value, sizeof(T));
    const std::uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    for (std::size_t i = 0; i < kWords; ++i) words_[i].store(buf[i], std::memory_order_relaxed);
    seq_.store(s + 2, std::memory_order_release);
  }

  // Any reader. Returns true and fills `out` with a consistent snapshot, or false after `max_tries` overlapped
  // attempts (out is then unspecified). `seq_out`, if given, receives the sequence of the snapshot (even; 0 = never
  // written).
  bool try_read(T& out, int max_tries = 64, std::uint64_t* seq_out = nullptr) const noexcept {
    std::uint64_t buf[kWords];
    for (int attempt = 0; attempt < max_tries; ++attempt) {
      const std::uint64_t s1 = seq_.load(std::memory_order_acquire);
      if (s1 & 1u) continue;  // a write is in progress
      for (std::size_t i = 0; i < kWords; ++i) buf[i] = words_[i].load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      const std::uint64_t s2 = seq_.load(std::memory_order_relaxed);
      if (s1 == s2) {
        std::memcpy(&out, buf, sizeof(T));
        if (seq_out != nullptr) *seq_out = s1;
        return true;
      }
    }
    return false;
  }

  // Number of completed writes (cheap liveness probe; does not read the payload).
  std::uint64_t writes() const noexcept { return seq_.load(std::memory_order_acquire) / 2; }

 private:
  friend struct SeqlockTestAccess;
  alignas(64) std::atomic<std::uint64_t> seq_;
  std::atomic<std::uint64_t> words_[kWords];
};

}  // namespace r1rt
