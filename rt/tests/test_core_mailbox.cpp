// The mode-request mailbox (session 5, layout 2 / ABI 5; shm_layout.h ModeMailbox): several clients -> the core, read
// every writer tick inside r1rt_core_step. Every request answered exactly once, in order; BUSY while one is pending;
// DAMP only from the operator requester; a claim that is never published is abandoned, so the mailbox cannot block.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <set>
#include <thread>
#include <vector>

#include "test_core_rig.hpp"

namespace {

using MB = r1rt::shm::ModeMailbox;

std::int32_t reply(Rig& r, std::uint64_t seq, std::int32_t* status, std::int32_t* refusal, std::uint32_t* mode) {
  return r1rt_segment_mode_reply(r.seg, seq, status, refusal, mode);
}

TEST(CoreMailbox, ARequestIsAnsweredAtTheNextWriterTick) {
  Rig r;
  NEED_RIG(r);
  r.tick();
  std::uint64_t seq = 0;
  ASSERT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_RESUME, R1RT_REQUESTER_R1D, r.t, &seq), R1RT_OK);
  EXPECT_EQ(seq, 1u);
  std::int32_t st = 99, why = 99;
  std::uint32_t mode = 99;
  EXPECT_EQ(reply(r, seq, &st, &why, &mode), R1RT_E_AGAIN);  // pending until the core's tick
  EXPECT_EQ(reply(r, 2, &st, &why, &mode), R1RT_E_INVALID_ARG);  // never issued
  r.tick();
  ASSERT_EQ(reply(r, seq, &st, &why, &mode), R1RT_OK);
  EXPECT_EQ(st, R1RT_OK);
  EXPECT_EQ(why, R1RT_REFUSE_NONE);
  EXPECT_EQ(mode, static_cast<std::uint32_t>(R1RT_MODE_MONITOR));
  EXPECT_EQ(r.status().mode_requests, 1u);
  EXPECT_EQ(r.status().mode_last_seq, 1u);
  EXPECT_EQ(MB::phase_of(r.segment().mode_req.state.load()), MB::kFree);
}

TEST(CoreMailbox, BusyWhileOnePendsThenInOrder) {
  Rig r;
  NEED_RIG(r);
  r.tick();
  std::uint64_t a = 0, b = 0;
  ASSERT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_STOP, R1RT_REQUESTER_R1D, r.t, &a), R1RT_OK);
  EXPECT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_RESUME, R1RT_REQUESTER_R1D, r.t, &b), R1RT_E_AGAIN);
  EXPECT_EQ(b, 0u);
  r.tick();
  ASSERT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_RESUME, R1RT_REQUESTER_R1D, r.t, &b), R1RT_OK);
  EXPECT_EQ(b, a + 1);
  r.tick();
  std::int32_t st = 0;
  ASSERT_EQ(reply(r, a, &st, nullptr, nullptr), R1RT_OK);
  EXPECT_EQ(st, R1RT_OK);
  ASSERT_EQ(reply(r, b, &st, nullptr, nullptr), R1RT_OK);
  EXPECT_EQ(st, R1RT_OK);
  EXPECT_EQ(r.status().stop_active, 0);  // STOP then RESUME, in that order
}

TEST(CoreMailbox, DampOnlyFromTheOperator) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 20; ++i) r.tick();
  std::uint64_t seq = 0;
  for (std::uint32_t who : {static_cast<std::uint32_t>(R1RT_REQUESTER_R1D), static_cast<std::uint32_t>(R1RT_REQUESTER_SIM),
                            static_cast<std::uint32_t>(R1RT_REQUESTER_NONE)}) {
    ASSERT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_DAMP, who, r.t, &seq), R1RT_OK);
    r.tick();
    std::int32_t st = 0, why = 0;
    std::uint32_t mode = 0;
    ASSERT_EQ(reply(r, seq, &st, &why, &mode), R1RT_OK);
    EXPECT_EQ(st, R1RT_E_REFUSED) << who;
    EXPECT_EQ(why, R1RT_REFUSE_NOT_OPERATOR) << who;
    EXPECT_EQ(mode, static_cast<std::uint32_t>(R1RT_MODE_BALANCE)) << who;
  }
  ASSERT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_DAMP, R1RT_REQUESTER_OPERATOR, r.t, &seq), R1RT_OK);
  r.tick();
  std::int32_t st = -1;
  std::uint32_t mode = 0;
  ASSERT_EQ(reply(r, seq, &st, nullptr, &mode), R1RT_OK);
  EXPECT_EQ(st, R1RT_OK);
  EXPECT_EQ(mode, static_cast<std::uint32_t>(R1RT_MODE_DAMP));
}

TEST(CoreMailbox, RefusalsCarryTheGateAndTheReason) {
  Rig r;
  NEED_RIG(r);
  r.tick();
  std::uint64_t seq = 0;
  // TAKEOVER without the built-in's rt/lowcmd: the takeover gate's code comes back
  ASSERT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_TAKEOVER, R1RT_REQUESTER_R1D, r.t, &seq), R1RT_OK);
  r.tick();
  std::int32_t st = 0, why = 0;
  std::uint32_t mode = 0;
  ASSERT_EQ(reply(r, seq, &st, &why, &mode), R1RT_OK);
  EXPECT_EQ(st, R1RT_E_REFUSED);
  EXPECT_EQ(why, R1RT_TAKEOVER_BUILTIN);
  EXPECT_EQ(mode, static_cast<std::uint32_t>(R1RT_MODE_MONITOR));
  // RELEASED outside MIRROR: refused by the mode
  ASSERT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_TAKEOVER_RELEASED, R1RT_REQUESTER_R1D, r.t, &seq), R1RT_OK);
  r.tick();
  ASSERT_EQ(reply(r, seq, &st, &why, &mode), R1RT_OK);
  EXPECT_EQ(st, R1RT_E_REFUSED);
  EXPECT_EQ(why, R1RT_REFUSE_MODE);
  // an unknown code
  ASSERT_EQ(r1rt_segment_request_mode(r.seg, 77, R1RT_REQUESTER_R1D, r.t, &seq), R1RT_OK);
  r.tick();
  ASSERT_EQ(reply(r, seq, &st, &why, &mode), R1RT_OK);
  EXPECT_EQ(st, R1RT_E_INVALID_ARG);
  EXPECT_EQ(why, R1RT_REFUSE_UNKNOWN);
  // a bad requester id is refused at the client side
  EXPECT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_STOP, R1RT_NUM_REQUESTERS, r.t, &seq), R1RT_E_INVALID_ARG);
}

TEST(CoreMailbox, AClaimNeverPublishedIsAbandonedAndTheMailboxFreesItself) {
  Rig r;
  NEED_RIG(r);
  r.tick();
  auto& mb = r.segment().mode_req;
  // a client claims (FREE -> WRITING) and dies before publishing
  std::uint64_t s = mb.state.load();
  ASSERT_TRUE(mb.state.compare_exchange_strong(s, MB::pack(1, MB::kWriting)));
  std::uint64_t seq = 0;
  EXPECT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_STOP, R1RT_REQUESTER_R1D, r.t, &seq), R1RT_E_AGAIN);
  for (int i = 0; i < 45; ++i) r.tick();  // 90 ms: not yet
  EXPECT_EQ(MB::phase_of(mb.state.load()), MB::kWriting);
  for (int i = 0; i < 10; ++i) r.tick();  // > 100 ms since the core first saw the claim
  EXPECT_EQ(mb.state.load(), MB::pack(1, MB::kFree));
  std::int32_t st = 0;
  ASSERT_EQ(reply(r, 1, &st, nullptr, nullptr), R1RT_OK);
  EXPECT_EQ(st, R1RT_E_ABANDONED);
  EXPECT_EQ(r.status().mode_abandoned, 1u);
  // the late client's publish fails (its claim is gone): what r1rt_segment_request_mode reports as ABANDONED
  std::uint64_t expect = MB::pack(1, MB::kWriting);
  EXPECT_FALSE(mb.state.compare_exchange_strong(expect, MB::pack(1, MB::kReady)));
  // and the mailbox works again
  ASSERT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_STOP, R1RT_REQUESTER_R1D, r.t, &seq), R1RT_OK);
  EXPECT_EQ(seq, 2u);
  r.tick();
  ASSERT_EQ(reply(r, 2, &st, nullptr, nullptr), R1RT_OK);
  EXPECT_EQ(st, R1RT_OK);
}

TEST(CoreMailbox, ManyClientsRaceEveryRequestAnsweredExactlyOnce) {
  // 4 client threads x 150 requests (STOP / RESUME, each tagged with a unique time stamp), retrying while BUSY, each
  // waiting for its own reply; the writer ticks on another thread. Every sequence number is issued once, every request
  // is processed once (the core's count), every client finds its own reply (the tag it sent comes back).
  Rig r;
  NEED_RIG(r);
  r.tick();
  constexpr int kClients = 4;
  constexpr int kEach = 150;
  std::atomic<bool> done{false};
  std::thread writer([&] {
    while (!done.load()) {
      r.tick();
      std::this_thread::sleep_for(std::chrono::microseconds(30));
    }
  });
  std::vector<std::vector<std::uint64_t>> seqs(kClients);
  std::atomic<int> bad_replies{0}, lost_replies{0}, busy{0};
  std::vector<std::thread> clients;
  for (int c = 0; c < kClients; ++c) {
    clients.emplace_back([&, c] {
      for (int k = 0; k < kEach; ++k) {
        const std::int64_t tag = 1'000'000'000LL * (c + 1) + k;
        const std::uint32_t req = (k % 2 == 0) ? R1RT_REQ_STOP : R1RT_REQ_RESUME;
        std::uint64_t seq = 0;
        std::int32_t rc;
        while ((rc = r1rt_segment_request_mode(r.seg, req, R1RT_REQUESTER_R1D, tag, &seq)) == R1RT_E_AGAIN) {
          busy.fetch_add(1);
          std::this_thread::yield();
        }
        ASSERT_EQ(rc, R1RT_OK);
        seqs[c].push_back(seq);
        for (;;) {
          std::int32_t st = 0;
          const std::int32_t q = r1rt_segment_mode_reply(r.seg, seq, &st, nullptr, nullptr);
          if (q == R1RT_E_AGAIN) {
            std::this_thread::yield();
            continue;
          }
          if (q != R1RT_OK) {
            lost_replies.fetch_add(1);
            break;
          }
          r1rt::shm::ModeReply rep{};
          ASSERT_TRUE(r.segment().mode_req.replies[seq % r1rt::shm::kModeReplySlots].try_read(rep));
          if (rep.seq == seq && (rep.t_req_ns != tag || rep.req != req || st != R1RT_OK)) bad_replies.fetch_add(1);
          break;
        }
      }
    });
  }
  for (auto& t : clients) t.join();
  done.store(true);
  writer.join();
  std::set<std::uint64_t> all;
  for (const auto& v : seqs)
    for (std::uint64_t s : v) EXPECT_TRUE(all.insert(s).second) << "sequence " << s << " issued twice";
  ASSERT_EQ(all.size(), static_cast<std::size_t>(kClients * kEach));
  EXPECT_EQ(*all.begin(), 1u);
  EXPECT_EQ(*all.rbegin(), static_cast<std::uint64_t>(kClients * kEach));  // contiguous: no sequence skipped
  EXPECT_EQ(r.status().mode_requests, static_cast<std::uint64_t>(kClients * kEach));
  EXPECT_EQ(r.status().mode_abandoned, 0u);
  EXPECT_EQ(bad_replies.load(), 0);  // never somebody else's reply
  // a reply is kept for the next 64 requests (by design); a client descheduled that long finds it overwritten
  // (INVALID_ARG, never a wrong answer). Under this test's load it should not happen; reported, bounded loosely.
  EXPECT_LE(lost_replies.load(), kClients * kEach / 50);
  std::printf("[ mailbox race ] %d requests, %d busy retries, %d replies overwritten before read\n", kClients * kEach,
              busy.load(), lost_replies.load());
}

}  // namespace
