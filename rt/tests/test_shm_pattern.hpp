// Test patterns for the two-process shared-memory tests: every field of a snapshot is a function of its seq, so a
// reader can tell a consistent snapshot from a torn one.
#pragma once

#include <cstdint>
#include <cstring>

#include "r1rt/types.h"

namespace r1rt_test {

inline r1rt_state state_for(std::uint64_t k) {
  r1rt_state s;
  std::memset(&s, 0, sizeof(s));
  s.seq = k;
  s.t_core_ns = static_cast<std::int64_t>(k * 3);
  for (int i = 0; i < R1RT_NUM_JOINTS; ++i) {
    s.in.q[i] = static_cast<double>(k) + i;
    s.in.tau_est[i] = -static_cast<double>(k);
  }
  s.in.imu_quat[0] = static_cast<double>(k) * 0.5;
  s.lowstate_age_ns = static_cast<std::int64_t>(k) * 7;
  s.faults = static_cast<std::uint32_t>(k & 0xFFFFFFFFu);
  return s;
}

inline bool state_consistent(const r1rt_state& s) {
  const r1rt_state want = state_for(s.seq);
  for (int i = 0; i < R1RT_NUM_JOINTS; ++i)
    if (s.in.q[i] != want.in.q[i] || s.in.tau_est[i] != want.in.tau_est[i]) return false;
  return s.t_core_ns == want.t_core_ns && s.in.imu_quat[0] == want.in.imu_quat[0] &&
         s.lowstate_age_ns == want.lowstate_age_ns && s.faults == want.faults;
}

inline r1rt_heartbeat heartbeat_for(std::uint64_t k) {
  r1rt_heartbeat h;
  std::memset(&h, 0, sizeof(h));
  h.seq = k;
  h.t_ns = static_cast<std::int64_t>(k * 7);
  h.lowstate_tick = k * k;
  h.rate_hz = static_cast<std::uint32_t>(k % 1000u);
  h.faults = static_cast<std::uint32_t>(~k & 0xFFFFFFFFu);
  return h;
}

inline bool heartbeat_consistent(const r1rt_heartbeat& h) {
  const r1rt_heartbeat want = heartbeat_for(h.seq);
  return h.t_ns == want.t_ns && h.lowstate_tick == want.lowstate_tick && h.rate_hz == want.rate_hz &&
         h.faults == want.faults;
}

}  // namespace r1rt_test
