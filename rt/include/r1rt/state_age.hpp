// r1rt::ContentAgeClock - session 9 (U19 / D25, the user's state-age limit ~10 ms sustained): how old the robot
// state's CONTENT is, from the board's own tick, not only from when it was received.
//
// Why: the receive time says when a sample reached r1rt, not when the board measured it. A receive path that is only
// LATE - samples queued in the socket after a receive-thread stall, a backlog the reader drains - hands the core old
// poses with fresh stamps (session 8 found exactly that after a SIGSTOP: r1rt/resume.hpp). B7 (session 8) found the
// balance policy's limit is such latency: a constant extra 10 ms passes, 15 ms degrades, 20 ms falls.
//
// The model: board time = tick / tick_hz (HW1, 11 Oct 2026: the R1 board's tick runs at 999.5-1000.5 /s over the stand
// tests' recordings, with ~5 % duplicate ticks at r1rt's 1052 samples/s; the simulator 1000 exactly). Each sample's
// receive offset = t_rx - board time; the BASELINE = the fastest path seen: it follows a lower offset at once and
// creeps up (at most kDriftPerS, the two clocks' drift and a tick-rate error) only while the state is healthy (the
// offset within half the limit of it) - so a real backlog is never absorbed as drift. Content age of the newest sample
// at t_now = t_now - (its board time + the baseline) = its receive age + how much later than the fastest path it came.
// No usable model (too few samples, a tick that stopped moving while samples arrive, a wild jump) -> valid() false and
// the caller uses the receive age alone. Header-only, no allocation, noexcept: the writer cycle calls it.
#pragma once

#include <cstdint>

namespace r1rt {

class ContentAgeClock {
 public:
  // tick_hz: the board tick rate; healthy_ns: the baseline creeps up only while a sample is within this of it.
  void configure(double tick_hz, std::int64_t healthy_ns) noexcept {
    ns_per_tick_ = tick_hz > 0.0 ? 1e9 / tick_hz : 1e6;
    healthy_ns_ = healthy_ns > 0 ? healthy_ns : 5'000'000;
    have_ = false;
  }

  // A NEW sample (call once per received sample): its board tick and its receive time.
  void sample(std::uint32_t tick, std::int64_t t_rx_ns) noexcept {
    if (!have_) {
      rebase(tick, t_rx_ns);
      return;
    }
    // signed, modulo 2^32: a sample OLDER than the last one (a late / reordered sample, or the simulator's state
    // delay switching on) steps back - it keeps the model and simply reads old
    const auto d = static_cast<std::int32_t>(tick - last_tick_);
    if (d > kMaxJumpTicks || d < -kMaxJumpTicks || t_rx_ns < last_t_) {  // a board restart / wild tick: start over
      rebase(tick, t_rx_ns);
      return;
    }
    if (d == 0) {
      // a duplicate tick (r1rt receives ~1052 samples/s of a 1 kHz tick); a tick that stops moving while samples keep
      // arriving is no clock (a board whose tick does not run): invalid until it moves again
      if (t_rx_ns - moved_t_ > kStuckNs) stuck_ = true;
      return;
    }
    if (stuck_) {
      rebase(tick, t_rx_ns);
      return;
    }
    tick64_ += d;
    last_tick_ = tick;
    if (d > 0) moved_t_ = t_rx_ns;
    const double off = static_cast<double>(t_rx_ns) - static_cast<double>(tick64_) * ns_per_tick_;
    const double dt = static_cast<double>(t_rx_ns - last_t_);
    last_t_ = t_rx_ns;
    if (off < base_) {
      base_ = off;
    } else if (off - base_ < static_cast<double>(healthy_ns_)) {
      const double up = base_ + kDriftPerS * dt;
      base_ = up < off ? up : off;
    }
    // a STABLE elevated offset for kRelearnNs (every sample late by the same amount, +-kRelearnSpreadNs) is a new
    // time base, not a backlog (a board tick reset, the simulator's real-time re-anchor): re-based, counted. A growing
    // or jittering lateness never is. (A real constant backlog is then read as fresh after kRelearnNs: the rung has
    // acted long before; documented in docs/sim/STATE_AGE.md.)
    if (off - base_ >= static_cast<double>(healthy_ns_)) {
      const double lo = off < elev_min_ ? off : elev_min_;
      const double hi = off > elev_max_ ? off : elev_max_;
      if (elev_since_ == 0 || hi - lo > 2.0 * kRelearnSpreadNs) {
        elev_since_ = t_rx_ns;  // (re)start the window at this level
        elev_min_ = elev_max_ = off;
      } else {
        elev_min_ = lo;
        elev_max_ = hi;
        if (t_rx_ns - elev_since_ >= kRelearnNs) {
          if (hold_) {  // session 11: not during a freeze (hold_relearn)
            ++relearns_held_;
            elev_since_ = t_rx_ns;
          } else {
            base_ = elev_min_;
            elev_since_ = 0;
            ++relearns_;
          }
        }
      }
    } else {
      elev_since_ = 0;
    }
    if (n_ < kMinSamples) ++n_;
  }

  std::uint64_t relearns() const noexcept { return relearns_; }

  // session 11 (S10's not-solid item, STATE_AGE.md "Session 11"): while held (the core holds it for as long as it is
  // in STAND_FREEZE) a stable elevated offset is NEVER taken as a new time base - a state that stayed late through a
  // freeze keeps reading late, so the operator's ENTER_BALANCE (`r1 balance`) is judged on the true content age, not on
  // a re-based one. The window restarts when the hold ends: a new time base needs a full kRelearnNs after the freeze.
  // Wild jumps / a stuck tick still start over (a board restart is not a backlog).
  void hold_relearn(bool on) noexcept {
    if (on != hold_) elev_since_ = 0;
    hold_ = on;
  }
  bool relearn_held() const noexcept { return hold_; }
  std::uint64_t relearns_held() const noexcept { return relearns_held_; }  // relearns a hold refused

  bool valid() const noexcept { return have_ && !stuck_ && n_ >= kMinSamples; }

  // The content age of the newest sample at t_now (ns); only meaningful when valid().
  std::int64_t age_ns(std::int64_t t_now_ns) const noexcept {
    return static_cast<std::int64_t>(static_cast<double>(t_now_ns) -
                                     (static_cast<double>(tick64_) * ns_per_tick_ + base_));
  }

  static constexpr double kDriftPerS = 1e-3;              // the baseline creeps up at most 1 ms per second
  static constexpr std::int32_t kMaxJumpTicks = 600'000;  // > 10 min of ticks at once: not the same clock
  static constexpr std::int64_t kStuckNs = 20'000'000;     // the tick unchanged for 20 ms while samples arrive
  static constexpr int kMinSamples = 100;                  // ~0.1 s of samples before the model is used
  static constexpr std::int64_t kRelearnNs = 2'000'000'000;  // a stable elevated offset this long = a new time base
  static constexpr double kRelearnSpreadNs = 1.5e6;           // "stable": the window's offsets span <= 3 ms

 private:
  void rebase(std::uint32_t tick, std::int64_t t_rx_ns) noexcept {
    have_ = true;
    stuck_ = false;
    last_tick_ = tick;
    tick64_ = 0;
    last_t_ = t_rx_ns;
    moved_t_ = t_rx_ns;
    base_ = static_cast<double>(t_rx_ns);
    n_ = 0;
    elev_since_ = 0;
  }

  double ns_per_tick_ = 1e6;
  std::int64_t healthy_ns_ = 5'000'000;
  bool have_ = false;
  bool stuck_ = false;
  std::uint32_t last_tick_ = 0;
  std::int64_t tick64_ = 0;
  std::int64_t last_t_ = 0;
  std::int64_t moved_t_ = 0;
  double base_ = 0.0;
  int n_ = 0;
  std::int64_t elev_since_ = 0;
  double elev_min_ = 0.0;
  double elev_max_ = 0.0;
  std::uint64_t relearns_ = 0;
  bool hold_ = false;
  std::uint64_t relearns_held_ = 0;
};

}  // namespace r1rt
