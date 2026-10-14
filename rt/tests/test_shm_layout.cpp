// The shared-memory segment: pinned layout, init/check, the ownership-token protocol and the stop channel.
#include "r1rt/shm_layout.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "r1rt/step_api.h"

namespace {

using r1rt::shm::OwnershipToken;
using r1rt::shm::Segment;

// A segment in ordinary memory, as the simulator's lockstep mode uses it.
struct HeapSegment {
  void* base = nullptr;
  std::size_t size = 0;
  HeapSegment() {
    size = (r1rt_segment_size() + 63) / 64 * 64;
    base = std::aligned_alloc(64, size);
  }
  ~HeapSegment() { std::free(base); }
  Segment* seg() { return static_cast<Segment*>(base); }
};

TEST(ShmLayout, RegionsAndSizes) {
  EXPECT_EQ(r1rt_segment_size(), sizeof(Segment));
  EXPECT_LT(sizeof(Segment), 8u * 1024u * 1024u);  // fits comfortably in a locked mapping
  EXPECT_EQ(sizeof(OwnershipToken), 64u);
  EXPECT_EQ(offsetof(Segment, header), 0u);
  std::printf("[ shm layout ] segment %zu B: header@%zu token@%zu rt_heartbeat@%zu guard_heartbeat@%zu state@%zu "
              "mirror@%zu base_cmd@%zu hands_contact@%zu stop@%zu keepalive@%zu upper_cmd@%zu telemetry@%zu "
              "policy@%zu mode_req@%zu hands_pose@%zu guard_telemetry@%zu\n",
              sizeof(Segment), offsetof(Segment, header), offsetof(Segment, token), offsetof(Segment, rt_heartbeat),
              offsetof(Segment, guard_heartbeat), offsetof(Segment, state), offsetof(Segment, mirror),
              offsetof(Segment, base_cmd), offsetof(Segment, hands_contact), offsetof(Segment, stop),
              offsetof(Segment, keepalive), offsetof(Segment, upper_cmd), offsetof(Segment, telemetry),
              offsetof(Segment, policy), offsetof(Segment, mode_req), offsetof(Segment, hands_pose),
              offsetof(Segment, guard_telemetry));
  for (std::size_t off : {offsetof(Segment, token), offsetof(Segment, rt_heartbeat),
                          offsetof(Segment, guard_heartbeat), offsetof(Segment, state), offsetof(Segment, mirror),
                          offsetof(Segment, base_cmd), offsetof(Segment, hands_contact), offsetof(Segment, stop),
                          offsetof(Segment, keepalive), offsetof(Segment, upper_cmd), offsetof(Segment, telemetry),
                          offsetof(Segment, policy), offsetof(Segment, mode_req), offsetof(Segment, hands_pose),
                          offsetof(Segment, guard_telemetry)})
    EXPECT_EQ(off % 64, 0u);
  // layout 2 appended its new regions; header .. mirror keep their layout-1 offsets (the mirror itself grew 808 ->
  // 880 B, the upper_cmd records 1296 -> 1528 B and the keepalives to 6 groups, so the regions after it moved)
  EXPECT_EQ(offsetof(Segment, token), 192u);
  EXPECT_EQ(offsetof(Segment, rt_heartbeat), 256u);
  EXPECT_EQ(offsetof(Segment, state), 384u);
  EXPECT_EQ(offsetof(Segment, mirror), 1792u);
  EXPECT_GT(offsetof(Segment, mode_req), offsetof(Segment, policy));
  // layout 4 (session 10): the executor status APPENDED after the guard ring - every layout-3 offset unchanged
  EXPECT_GT(offsetof(Segment, exec_status), offsetof(Segment, guard_telemetry));
  EXPECT_EQ(offsetof(Segment, exec_status) % 64, 0u);
  EXPECT_EQ(sizeof(r1rt_exec_status), 184u);
  EXPECT_EQ(R1RT_LAYOUT_VERSION, 4u);  // 3 = session 8 (the mirror's tau_ff); 4 = session 10 (the executor status)
}

TEST(ShmLayout, InitCheckAndHeader) {
  HeapSegment h;
  ASSERT_NE(h.base, nullptr);
  EXPECT_EQ(r1rt_segment_check(h.base, h.size), R1RT_E_BAD_SEGMENT);  // not initialised (garbage / not ready)
  ASSERT_EQ(r1rt_segment_init(h.base, h.size, 4242, 123456789), R1RT_OK);
  ASSERT_EQ(r1rt_segment_check(h.base, h.size), R1RT_OK);
  const auto& hd = h.seg()->header;
  EXPECT_EQ(hd.magic, r1rt::shm::kMagic);
  EXPECT_EQ(std::memcmp(&hd.magic, "R1MC3SHM", 8), 0);
  EXPECT_EQ(hd.layout_version, R1RT_LAYOUT_VERSION);
  EXPECT_EQ(hd.creator_pid, 4242);
  EXPECT_EQ(hd.created_ns, 123456789);
  const std::uint32_t v2_slots[26] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                      13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30};  // v2 joints.py SLOT
  for (int i = 0; i < 26; ++i) EXPECT_EQ(hd.hg_slot_map[i], v2_slots[i]) << i;
  EXPECT_EQ(OwnershipToken::owner_of(h.seg()->token.load()), R1RT_OWNER_NONE);
  EXPECT_TRUE(h.seg()->upper_cmd.empty());
  EXPECT_TRUE(h.seg()->telemetry.empty());
  EXPECT_EQ(h.seg()->stop.epoch(), 0u);
  // corruptions are refused
  EXPECT_EQ(r1rt_segment_check(h.base, h.size - 64), R1RT_E_BAD_SEGMENT);
  h.seg()->header.magic ^= 1u;
  EXPECT_EQ(r1rt_segment_check(h.base, h.size), R1RT_E_BAD_SEGMENT);
  h.seg()->header.magic ^= 1u;
  h.seg()->header.layout_version += 1;
  EXPECT_EQ(r1rt_segment_check(h.base, h.size), R1RT_E_BAD_SEGMENT);
  h.seg()->header.layout_version -= 1;
  EXPECT_EQ(r1rt_segment_check(h.base, h.size), R1RT_OK);
  EXPECT_EQ(r1rt_segment_init(nullptr, h.size, 1, 0), R1RT_E_INVALID_ARG);
  EXPECT_EQ(r1rt_segment_init(h.base, 64, 1, 0), R1RT_E_INVALID_ARG);
}

TEST(OwnershipTokenProtocol, TakeoverGuardAndOperatorReturn) {
  HeapSegment h;
  ASSERT_EQ(r1rt_segment_init(h.base, h.size, 1, 0), R1RT_OK);
  OwnershipToken& tok = h.seg()->token;
  // r1rt takes ownership at takeover
  std::uint64_t rt_word = tok.load();
  ASSERT_TRUE(tok.try_transfer(rt_word, R1RT_OWNER_R1RT));
  EXPECT_TRUE(tok.holds(rt_word));
  EXPECT_EQ(OwnershipToken::owner_of(rt_word), R1RT_OWNER_R1RT);
  // the guard sees a stale heartbeat and CASes from the word it observed
  std::uint64_t guard_view = tok.load();
  ASSERT_TRUE(tok.try_transfer(guard_view, R1RT_OWNER_R1GUARD));
  EXPECT_FALSE(tok.holds(rt_word));  // r1rt's check-before-write now fails: it must not write
  EXPECT_TRUE(tok.holds(guard_view));
  // a stale CAS (r1rt waking up and trying to re-take with its old word) cannot succeed: no ABA
  std::uint64_t stale = rt_word;
  EXPECT_FALSE(tok.try_transfer(stale, R1RT_OWNER_R1RT));
  EXPECT_EQ(stale, guard_view);  // failure reports the current word
  // the operator returns ownership (generation keeps increasing)
  std::uint64_t op_view = tok.load();
  ASSERT_TRUE(tok.try_transfer(op_view, R1RT_OWNER_R1RT));
  EXPECT_EQ(OwnershipToken::generation_of(op_view), 3u);
  EXPECT_FALSE(tok.holds(guard_view));
}

TEST(OwnershipTokenProtocol, ConcurrentClaimsExactlyOneWins) {
  for (int trial = 0; trial < 2000; ++trial) {
    OwnershipToken tok{};
    tok.word.store(OwnershipToken::pack(R1RT_OWNER_R1RT, 1), std::memory_order_relaxed);
    const std::uint64_t seen = tok.load();
    std::atomic<int> wins{0};
    std::atomic<bool> go{false};
    auto claim = [&](r1rt_owner who) {
      std::uint64_t w = seen;
      while (!go.load(std::memory_order_acquire)) {
      }
      if (tok.try_transfer(w, who)) wins.fetch_add(1);
    };
    std::thread a(claim, R1RT_OWNER_R1GUARD), b(claim, R1RT_OWNER_NONE);
    go.store(true, std::memory_order_release);
    a.join();
    b.join();
    ASSERT_EQ(wins.load(), 1) << trial;
  }
}

TEST(StopChannel, RequestsFromManyClientsAreCountedExactly) {
  HeapSegment h;
  ASSERT_EQ(r1rt_segment_init(h.base, h.size, 1, 0), R1RT_OK);
  auto& stop = h.seg()->stop;
  constexpr int kThreads = 4, kEach = 10'000;
  std::vector<std::thread> clients;
  for (int t = 0; t < kThreads; ++t)
    clients.emplace_back([&, t] {
      for (int i = 0; i < kEach; ++i) stop.request(t % 2 == 0 ? R1RT_STOP_STOP : R1RT_STOP_HALT, 100u + t, i);
    });
  for (auto& c : clients) c.join();
  EXPECT_EQ(stop.level[R1RT_STOP_STOP].requests.load(), 2u * kEach);
  EXPECT_EQ(stop.level[R1RT_STOP_HALT].requests.load(), 2u * kEach);
  EXPECT_EQ(stop.level[R1RT_STOP_DAMP].requests.load(), 0u);
  EXPECT_EQ(stop.epoch(), 4u * kEach);
  // r1rt acknowledges what it acted on
  const std::uint64_t seen = stop.level[R1RT_STOP_STOP].requests.load(std::memory_order_acquire);
  stop.level[R1RT_STOP_STOP].acked.store(seen, std::memory_order_release);
  EXPECT_EQ(stop.level[R1RT_STOP_STOP].acked.load(), seen);
  stop.request(R1RT_STOP_DAMP, 1u, 77);
  EXPECT_EQ(stop.level[R1RT_STOP_DAMP].last_requester.load(), 1u);
  EXPECT_EQ(stop.level[R1RT_STOP_DAMP].last_request_ns.load(), 77);
}

TEST(ShmLayout, SeqlockAndRingRegionsWorkInTheSegment) {
  HeapSegment h;
  ASSERT_EQ(r1rt_segment_init(h.base, h.size, 1, 0), R1RT_OK);
  Segment& s = *h.seg();
  r1rt_heartbeat hb{};
  hb.seq = 5;
  hb.t_ns = 1000;
  hb.mode = R1RT_MODE_BALANCE;
  hb.rate_hz = 500;
  s.rt_heartbeat.write(hb);
  r1rt_heartbeat got{};
  ASSERT_TRUE(s.rt_heartbeat.try_read(got));
  EXPECT_EQ(got.seq, 5u);
  EXPECT_EQ(got.mode, R1RT_MODE_BALANCE);
  r1rt_upper_cmd cmd{};
  cmd.cmd_id = 9;
  cmd.group = R1RT_GROUP_ARM_RIGHT;
  cmd.source = R1RT_SOURCE_TRAJECTORY;
  cmd.n_knots = 3;
  cmd.stop_epoch = s.stop.epoch();
  ASSERT_TRUE(s.upper_cmd.try_push(cmd));
  r1rt_upper_cmd out{};
  ASSERT_TRUE(s.upper_cmd.try_pop(out));
  EXPECT_EQ(out.cmd_id, 9u);
  EXPECT_EQ(out.group, R1RT_GROUP_ARM_RIGHT);
}

}  // namespace
