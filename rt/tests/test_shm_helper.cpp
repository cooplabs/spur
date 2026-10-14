// Helper process for the two-process shared-memory tests (test_shm.cpp spawns it with posix_spawn).
//
//   r1rt_shm_helper read_state <name> <min_distinct>
//       attach, then read the `state` seqlock until <min_distinct> different snapshots were seen (10 s cap); every
//       snapshot must be internally consistent (all fields derived from its seq). Exit 0 ok, 1 torn read,
//       2 attach failed, 3 timeout, 4 usage.
//   r1rt_shm_helper write_guard_heartbeat <name> <count>
//       attach, write the `guard_heartbeat` seqlock <count> times (seq 1..count, the other fields derived from seq).
//       Exit 0 ok, 2 attach failed, 4 usage.
//   r1rt_shm_helper attach <name>
//       attach only; exit = 10 + the ShmError code (10 = ok).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "r1rt/shm.hpp"
#include "test_shm_pattern.hpp"

int main(int argc, char** argv) {
  if (argc < 3) return 4;
  const char* mode = argv[1];
  const char* name = argv[2];
  r1rt::ShmSegment seg;
  const r1rt::ShmError e = r1rt::ShmSegment::attach(name, seg);
  if (std::strcmp(mode, "attach") == 0) return 10 + static_cast<int>(e);
  if (e != r1rt::ShmError::kOk) {
    std::fprintf(stderr, "attach %s: %s\n", name, r1rt::shm_error_str(e));
    return 2;
  }
  if (argc < 4) return 4;
  const long n = std::strtol(argv[3], nullptr, 10);
  if (std::strcmp(mode, "read_state") == 0) {
    const auto t0 = std::chrono::steady_clock::now();
    long distinct = 0;
    std::uint64_t last_seq = 0;
    r1rt_state s{};
    while (distinct < n) {
      if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) return 3;
      if (!seg.segment()->state.try_read(s)) continue;
      if (s.seq == 0) continue;  // not written yet
      if (!r1rt_test::state_consistent(s)) {
        std::fprintf(stderr, "torn state at seq %llu\n", static_cast<unsigned long long>(s.seq));
        return 1;
      }
      if (s.seq != last_seq) {
        last_seq = s.seq;
        ++distinct;
      }
    }
    return 0;
  }
  if (std::strcmp(mode, "write_guard_heartbeat") == 0) {
    for (long k = 1; k <= n; ++k) seg.segment()->guard_heartbeat.write(r1rt_test::heartbeat_for(static_cast<std::uint64_t>(k)));
    return 0;
  }
  return 4;
}
