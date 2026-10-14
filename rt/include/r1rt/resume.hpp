// r1rt::ResumeClock - session 8 (HW1 fix #29): is the lowstate in hand LIVE after a stop?
//
// After a stop (the whole process SIGSTOPped, or stalled) the receive thread drains the samples queued during the stop
// (the UDP socket buffer: cyclonedds.xml asks for >= 4 MB; the reader keeps the last one) and stamps each with its
// RECEIVE time - a pose from the start of the stop arrives with a fresh stamp (found with the processes on the DDS
// bridge, session 8: the blend started from it and snapped back anyway). The board's own tick says how old a sample
// really is: the clock learns the tick rate from live samples (ticks per ns, whatever the board's unit), and after a stop
// a sample counts as live only when its tick has reached the tick extrapolated from before the stop. Without a learned
// rate (too early, or a board whose tick does not move) it answers "live" (the settle time then is the only guard).
// Header-only, no allocation, noexcept: the writer / guard cycles call it.
#pragma once

#include <cstdint>

namespace r1rt {

class ResumeClock {
 public:
  // A sample from a normal (not stopped) tick: learn the rate over windows of >= kWindowNs.
  void live_sample(std::uint32_t tick, std::int64_t t_rx_ns) noexcept {
    if (!have_ref_) {
      ref_tick_ = tick;
      ref_t_ = t_rx_ns;
      have_ref_ = true;
      return;
    }
    const std::int64_t dt = t_rx_ns - ref_t_;
    if (dt < kWindowNs) return;
    const auto dtick = static_cast<std::uint32_t>(tick - ref_tick_);  // modulo 2^32
    if (dt <= kMaxWindowNs && dtick > 0) {
      rate_ = static_cast<double>(dtick) / static_cast<double>(dt);
      have_rate_ = true;
    }
    ref_tick_ = tick;
    ref_t_ = t_rx_ns;
  }

  // The stop's start: the last live sample (tick, its receive time).
  void mark_stop(std::uint32_t last_tick, std::int64_t last_t_rx_ns) noexcept {
    stop_tick_ = last_tick;
    stop_t_ = last_t_rx_ns;
    have_ref_ = false;  // the window across the stop is no rate sample
  }

  // After mark_stop: has `tick` reached the board's tick now (t_now_ns), within kSlackNs of board time?
  bool caught_up(std::uint32_t tick, std::int64_t t_now_ns) const noexcept {
    if (!have_rate_) return true;
    const double expect = rate_ * static_cast<double>(t_now_ns - stop_t_ - kSlackNs);
    if (!(expect > 0.0)) return true;
    const auto got = static_cast<std::uint32_t>(tick - stop_tick_);
    return static_cast<double>(got) >= expect;
  }

  bool has_rate() const noexcept { return have_rate_; }
  double rate_per_ns() const noexcept { return rate_; }

  static constexpr std::int64_t kWindowNs = 200'000'000;     // learn over >= 0.2 s of live samples
  static constexpr std::int64_t kMaxWindowNs = 2'000'000'000;  // a longer window crossed a gap: not a rate sample
  static constexpr std::int64_t kSlackNs = 20'000'000;        // a sample <= 20 ms of board time old counts as live

 private:
  std::uint32_t ref_tick_ = 0;
  std::int64_t ref_t_ = 0;
  bool have_ref_ = false;
  double rate_ = 0.0;
  bool have_rate_ = false;
  std::uint32_t stop_tick_ = 0;
  std::int64_t stop_t_ = 0;
};

}  // namespace r1rt
