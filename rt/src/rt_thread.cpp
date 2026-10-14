// The OS side of the r1rt / r1guard processes: clock, absolute sleeps, real-time requests (rt_thread.hpp).
#include "r1rt/rt_thread.hpp"

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>

#include <cstdio>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#endif

namespace r1rt::proc {

std::int64_t now_ns() noexcept {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

#if defined(__APPLE__)
namespace {
// mach_absolute_time ticks <-> ns (1:1 on Apple silicon is NOT guaranteed: 125/3 on M-series).
mach_timebase_info_data_t timebase() noexcept {
  static mach_timebase_info_data_t tb = [] {
    mach_timebase_info_data_t t{};
    mach_timebase_info(&t);
    return t;
  }();
  return tb;
}
std::uint64_t ns_to_abs(std::int64_t ns) noexcept {
  const auto tb = timebase();
  return ns <= 0 ? 0 : static_cast<std::uint64_t>(ns) * tb.denom / tb.numer;  // ns <= seconds: no overflow
}
}  // namespace
#endif

void sleep_until(std::int64_t t_ns) noexcept {
#if defined(__linux__)
  timespec ts{};
  ts.tv_sec = static_cast<time_t>(t_ns / 1'000'000'000);
  ts.tv_nsec = static_cast<long>(t_ns % 1'000'000'000);
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
  }
#elif defined(__APPLE__)
  // CLOCK_MONOTONIC (mach_continuous_time) and mach_absolute_time differ by the time the Mac slept; convert the
  // remaining interval, not the absolute value, then wait on mach's own absolute clock.
  // mach_wait_until can return before the CLOCK_MONOTONIC deadline (session 6: one 500 Hz wake 5 us early in the
  // integrator's CI; an interrupted wait returns early too, KERN_ABORTED): wait again on the remainder, never act early.
  for (;;) {
    const std::int64_t dt = t_ns - now_ns();
    if (dt <= 0) return;
    mach_wait_until(mach_absolute_time() + ns_to_abs(dt));
  }
#else
  const std::int64_t dt = t_ns - now_ns();
  if (dt > 0) sleep_for(dt);
#endif
}

void sleep_for(std::int64_t dt_ns) noexcept {
  if (dt_ns <= 0) return;
  timespec ts{};
  ts.tv_sec = static_cast<time_t>(dt_ns / 1'000'000'000);
  ts.tv_nsec = static_cast<long>(dt_ns % 1'000'000'000);
  timespec rem{};
  while (nanosleep(&ts, &rem) == -1 && errno == EINTR) ts = rem;
}

RtGrant request_rt_thread(int priority, int cpu, std::int64_t period_ns, std::int64_t computation_ns) noexcept {
  RtGrant g;
  g.priority = priority;
  g.cpu = cpu;
#if defined(__linux__)
  sched_param sp{};
  sp.sched_priority = priority;
  const int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
  g.fifo = rc == 0;
  g.fifo_errno = rc;
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    const int ra = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    g.affinity = ra == 0;
    g.affinity_errno = ra;
  }
  (void)period_ns;
  (void)computation_ns;
#elif defined(__APPLE__)
  // No SCHED_FIFO / affinity for user processes on macOS: report them as not granted (ENOTSUP), ask for the Mach
  // time-constraint policy (period, computation budget, constraint = the period) instead.
  g.fifo_errno = ENOTSUP;
  if (cpu >= 0) g.affinity_errno = ENOTSUP;
  thread_time_constraint_policy_data_t pol{};
  pol.period = static_cast<uint32_t>(ns_to_abs(period_ns));
  pol.computation = static_cast<uint32_t>(ns_to_abs(computation_ns));
  pol.constraint = static_cast<uint32_t>(ns_to_abs(period_ns));
  pol.preemptible = 1;
  const kern_return_t kr = thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                                             reinterpret_cast<thread_policy_t>(&pol),
                                             THREAD_TIME_CONSTRAINT_POLICY_COUNT);
  g.time_constraint = kr == KERN_SUCCESS;
  g.time_constraint_err = static_cast<int>(kr);
  (void)priority;
#else
  g.fifo_errno = ENOTSUP;
  (void)period_ns;
  (void)computation_ns;
#endif
  return g;
}

bool lock_memory(int* err) noexcept {
#if defined(__linux__)
  if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
    if (err) *err = 0;
    return true;
  }
  if (err) *err = errno;
  return false;
#else
  if (err) *err = ENOTSUP;  // macOS: mlockall is not supported for user processes
  return false;
#endif
}

const char* os_name() noexcept {
#if defined(__linux__)
  return "linux";
#elif defined(__APPLE__)
  return "darwin";
#else
  return "other";
#endif
}

void TimingStats::add(std::int64_t v_ns) noexcept {
  n += 1;
  sum_ns += static_cast<double>(v_ns);
  if (v_ns > max_ns) max_ns = v_ns;
  int b = 0;
  while (b < kBuckets - 1 && v_ns >= kEdges[b]) ++b;
  hist[b] += 1;
}

const char* TimingStats::json(char* buf, std::size_t len) const noexcept {
  std::snprintf(buf, len,
                "{\"n\":%llu,\"mean_us\":%.1f,\"max_us\":%.1f,\"hist_us\":{\"<50\":%llu,\"<100\":%llu,\"<250\":%llu,"
                "\"<500\":%llu,\"<1000\":%llu,\"<2000\":%llu,\"<5000\":%llu,\"<10000\":%llu,\"<20000\":%llu,"
                "\">=20000\":%llu}}",
                static_cast<unsigned long long>(n), mean_ns() / 1e3, static_cast<double>(max_ns) / 1e3,
                static_cast<unsigned long long>(hist[0]), static_cast<unsigned long long>(hist[1]),
                static_cast<unsigned long long>(hist[2]), static_cast<unsigned long long>(hist[3]),
                static_cast<unsigned long long>(hist[4]), static_cast<unsigned long long>(hist[5]),
                static_cast<unsigned long long>(hist[6]), static_cast<unsigned long long>(hist[7]),
                static_cast<unsigned long long>(hist[8]), static_cast<unsigned long long>(hist[9]));
  return buf;
}

std::int64_t Periodic::wait(std::int64_t* lateness_ns) noexcept {
  sleep_until(next_);
  const std::int64_t wake = now_ns();
  const std::int64_t late = wake - next_;
  if (lateness_ns) *lateness_ns = late;
  next_ += period_;
  if (wake >= next_) {  // the next deadline already passed: skip the missed ones (no catch-up burst)
    const std::int64_t missed = (wake - next_) / period_ + 1;
    next_ += missed * period_;
    skipped_ += static_cast<std::uint64_t>(missed);
  }
  return wake;
}

}  // namespace r1rt::proc
