// r1rt/rt_thread.hpp - what the r1rt / r1guard PROCESSES need from the OS (session 5; DESIGN.md §2.1, §3.1, §11.1):
// the monotonic clock, absolute-deadline sleeps, the real-time scheduling request (SCHED_FIFO + priority + CPU
// affinity + mlockall) with an honest report of what was granted, and the loop-timing statistics every periodic
// thread records (wake-up lateness, iteration time). Set-up functions may allocate / fail; the per-tick functions
// (now_ns, sleep_until, Periodic::wait, TimingStats::add) do not allocate and never throw.
//
// Linux (the robot's Jetson): clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME), pthread_setschedparam(SCHED_FIFO),
// pthread_setaffinity_np, mlockall(MCL_CURRENT | MCL_FUTURE). Granted only with the RT limits of the systemd unit
// (LimitRTPRIO, LimitMEMLOCK; DESIGN §11.1) - otherwise EPERM is reported, never hidden (v2 fell back silently).
// macOS (this Mac: the simulator rehearsal only): no SCHED_FIFO for a user process, no CPU affinity, no mlockall -
// `rt_granted` is false and that is stated; the periodic threads ask for the Mach time-constraint policy instead
// (what audio threads use: it does reduce wake-up jitter under load, it is not a real-time guarantee) and sleep with
// mach_wait_until on an absolute deadline converted from CLOCK_MONOTONIC.
#pragma once

#include <cstddef>
#include <cstdint>

namespace r1rt::proc {

// CLOCK_MONOTONIC in ns (the clock of every r1rt timestamp; the simulator passes simulated time instead).
std::int64_t now_ns() noexcept;

// Sleep until CLOCK_MONOTONIC reaches t_ns (returns at once if it already has). Absolute: no drift accumulates.
void sleep_until(std::int64_t t_ns) noexcept;

// Plain relative sleep (fault injection, bounded waits at start-up).
void sleep_for(std::int64_t dt_ns) noexcept;

// What a thread asked for and what it got (logged in the ready line; DESIGN §2.1 "logs the granted policy").
struct RtGrant {
  bool fifo = false;              // SCHED_FIFO at `priority`
  int fifo_errno = 0;
  int priority = 0;
  bool affinity = false;          // pinned to `cpu`
  int affinity_errno = 0;
  int cpu = -1;                   // -1 = not requested
  bool time_constraint = false;   // macOS only: THREAD_TIME_CONSTRAINT_POLICY
  int time_constraint_err = 0;
};

// Request real-time scheduling for the CALLING thread: SCHED_FIFO `priority` (Linux), CPU `cpu` (>= 0, Linux), and on
// macOS the time-constraint policy for a thread with this period / computation budget.
RtGrant request_rt_thread(int priority, int cpu, std::int64_t period_ns, std::int64_t computation_ns) noexcept;

// mlockall(MCL_CURRENT | MCL_FUTURE): true when granted; *err = errno otherwise (ENOSYS / EPERM / ...).
bool lock_memory(int* err) noexcept;

// "linux" / "darwin" / "other".
const char* os_name() noexcept;

// Wake-up lateness / iteration-time statistics (no allocation): count, mean, max and a fixed histogram.
struct TimingStats {
  static constexpr int kBuckets = 10;
  // upper edges in ns: 50 us, 100 us, 250 us, 500 us, 1 ms, 2 ms, 5 ms, 10 ms, 20 ms, +inf
  static constexpr std::int64_t kEdges[kBuckets - 1] = {50'000,    100'000,   250'000,    500'000,   1'000'000,
                                                        2'000'000, 5'000'000, 10'000'000, 20'000'000};
  std::uint64_t n = 0;
  std::int64_t max_ns = 0;
  double sum_ns = 0.0;
  std::uint64_t hist[kBuckets] = {};

  void add(std::int64_t v_ns) noexcept;
  double mean_ns() const noexcept { return n ? sum_ns / static_cast<double>(n) : 0.0; }
  // {"n":..,"mean_us":..,"max_us":..,"hist_us":{"<50":..,...}} into buf (no allocation); returns buf.
  const char* json(char* buf, std::size_t len) const noexcept;
};

// An absolute-deadline periodic loop: wait() sleeps to the next deadline and returns the wake-up time; lateness =
// wake - deadline. A wake more than one period late skips the missed deadlines (counted in `skipped`) instead of
// running a burst of catch-up iterations.
class Periodic {
 public:
  Periodic(std::int64_t period_ns, std::int64_t start_ns) noexcept : period_(period_ns), next_(start_ns) {}
  std::int64_t wait(std::int64_t* lateness_ns) noexcept;
  std::int64_t period_ns() const noexcept { return period_; }
  std::int64_t next_deadline_ns() const noexcept { return next_; }
  std::uint64_t skipped() const noexcept { return skipped_; }

 private:
  std::int64_t period_;
  std::int64_t next_;
  std::uint64_t skipped_ = 0;
};

}  // namespace r1rt::proc
