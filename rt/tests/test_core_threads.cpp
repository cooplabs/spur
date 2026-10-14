// The writer / policy-thread hand-back with two REAL threads (session 4; core.hpp "Threads"): every policy result is
// computed on a private runner from the committed state and committed only if the generation is unchanged, so
//   - a thread result computed before an inline run can never overwrite it (discarded, counted),
//   - an inline run that loses the race to a thread result uses that result,
//   - commits are totally ordered: the policy ring carries consecutive generations with non-decreasing action times.
// The first two are forced deterministically with the core's commit hook (a real thread parked exactly between the
// read of the committed state and its commit, as a stall would leave it); the third hammers the hand-back from two
// threads with random stalls. These run under ThreadSanitizer in the Linux CI job (cpp-tsan; macOS 26's sanitizer
// runtimes do not start) - a data race anywhere in the writer / thread interface fails that job.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>

#include "r1rt/core.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"
#include "test_core_rig.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Park {  // a rendezvous between the parked thread and the main thread
  std::atomic<int> parked{0};   // 1 = the hooked side is parked in the hook
  std::atomic<int> release{0};  // 1 = it may go on
  std::atomic<int> once{0};
  int source = r1rt::Core::kSourceThread;
};

void park_hook(void* ctx, int source) {
  Park* p = static_cast<Park*>(ctx);
  if (source != p->source || p->once.exchange(1) != 0) return;
  p->parked.store(1, std::memory_order_release);
  const auto t0 = Clock::now();
  while (p->release.load(std::memory_order_acquire) == 0 && Clock::now() - t0 < std::chrono::seconds(20))
    std::this_thread::yield();
}

void wait_for(const std::atomic<int>& flag) {
  const auto t0 = Clock::now();
  while (flag.load(std::memory_order_acquire) == 0 && Clock::now() - t0 < std::chrono::seconds(20))
    std::this_thread::yield();
}

int drain(r1rt::shm::Segment& seg, r1rt_policy_record* last) {
  int n = 0;
  r1rt_policy_record rec;
  while (seg.policy.try_pop(rec)) {
    *last = rec;
    ++n;
  }
  return n;
}

TEST(CoreThreads, ALateThreadResultIsDiscardedNeverOverwritingTheInlineOne) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 30; ++i) r.tick();  // the thread on its schedule
  r1rt_policy_record last{};
  drain(r.segment(), &last);
  Park p;
  r1rt::core_of(r.core)->set_commit_hook(&park_hook, &p);
  const std::int64_t t_thread = r.t;
  // the policy thread starts an iteration, computes, and stalls right before its commit
  std::thread th([&] { r1rt_core_policy_step(r.core, t_thread); });
  wait_for(p.parked);
  const r1rt_core_status s0 = r.status();
  for (int i = 0; i < 40; ++i) r.tick(false, /*thread=*/false);  // 80 ms: the writer goes inline (rung 1)
  const r1rt_core_status s1 = r.status();
  ASSERT_GE(s1.inline_runs, 2u);
  p.release.store(1, std::memory_order_release);  // the thread wakes up and tries to commit its old result
  th.join();
  const r1rt_core_status s2 = r.status();
  EXPECT_EQ(s2.policy_discarded, s0.policy_discarded + 1);  // dropped
  EXPECT_EQ(s2.policy_commits, s1.policy_commits);         // nothing committed by the late thread
  EXPECT_EQ(s2.policy_thread_runs, s0.policy_thread_runs);
  ASSERT_EQ(drain(r.segment(), &last), static_cast<int>(s1.policy_commits - s0.policy_commits));
  EXPECT_EQ(last.inline_run, 1);  // the newest action is still the writer's inline one
  EXPECT_EQ(last.seq, s2.policy_commits);
  EXPECT_GT(last.t_action_ns, t_thread);
  r.tick();
  EXPECT_EQ(r.status().faults & R1RT_FAULT_POLICY_INVALID, 0u);
  r1rt::core_of(r.core)->set_commit_hook(nullptr, nullptr);
  for (int i = 0; i < 100; ++i) r.tick();  // hand-back: the thread computes again
  EXPECT_GT(r.status().policy_thread_runs, s2.policy_thread_runs);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
}

TEST(CoreThreads, AnInlineRunThatLosesTheRaceUsesTheThreadsResult) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 30; ++i) r.tick();
  r1rt_policy_record last{};
  drain(r.segment(), &last);
  // the thread computes and parks before its commit; the writer goes inline and parks before ITS commit; the thread
  // commits first (it read the committed state earlier, so the writer's result is the stale one now)
  Park pt;
  Park pw;
  pw.source = r1rt::Core::kSourceInline;
  struct Both {
    Park* t;
    Park* w;
  } both{&pt, &pw};
  r1rt::core_of(r.core)->set_commit_hook(
      [](void* ctx, int source) {
        Both* b = static_cast<Both*>(ctx);
        park_hook(b->t, source);
        park_hook(b->w, source);
      },
      &both);
  const std::int64_t t_thread = r.t;
  std::atomic<int> thread_done{0};
  std::thread th([&] {
    r1rt_core_policy_step(r.core, t_thread);
    thread_done.store(1, std::memory_order_release);
  });
  wait_for(pt.parked);
  const r1rt_core_status s0 = r.status();
  // the writer ticks until it goes inline; its inline run parks in the hook - on a helper thread (the writer)
  std::thread writer([&] {
    for (int i = 0; i < 40 && pw.parked.load(std::memory_order_acquire) == 0; ++i) r.tick(false, false);
  });
  wait_for(pw.parked);
  pt.release.store(1, std::memory_order_release);  // the thread commits (it wins the CAS)
  wait_for(thread_done);
  pw.release.store(1, std::memory_order_release);  // the writer's commit now finds the generation moved: discarded
  writer.join();
  th.join();
  const r1rt_core_status s1 = r.status();
  EXPECT_EQ(s1.policy_discarded, s0.policy_discarded + 1);
  EXPECT_EQ(s1.policy_thread_runs, s0.policy_thread_runs + 1);
  ASSERT_GE(drain(r.segment(), &last), 1);
  EXPECT_EQ(last.inline_run, 0);  // the action in use is the thread's
  EXPECT_EQ(last.t_action_ns, t_thread);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
  r1rt::core_of(r.core)->set_commit_hook(nullptr, nullptr);
}

TEST(CoreThreads, TwoThreadsHammerTheHandBackCommitsStayTotallyOrdered) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  auto& seg = r.segment();
  std::atomic<std::int64_t> now{r.t};
  std::atomic<bool> stop{false};
  std::atomic<bool> stop_log{false};  // after the policy thread has joined (its last push must be counted)
  std::atomic<long> seq_gaps{0}, t_back{0}, records{0};
  std::thread policy([&] {  // the 50 Hz policy thread with random stalls (a GIL-free stand-in for a page-fault storm)
    std::mt19937 rng(7);
    std::int64_t next = 0;
    while (!stop.load(std::memory_order_acquire)) {
      const std::int64_t t = now.load(std::memory_order_acquire);
      if (t < next) {
        std::this_thread::yield();
        continue;
      }
      r1rt_core_policy_step(r.core, t);
      next = t + 20 * kMs;
      if (rng() % 16 == 0) std::this_thread::sleep_for(std::chrono::microseconds(300 + rng() % 2000));
    }
  });
  std::thread log([&] {  // r1log's role: the single consumer of the policy ring
    r1rt_policy_record rec;
    std::uint64_t last_seq = 0;
    std::int64_t last_t = 0;
    auto take = [&] {
      while (seg.policy.try_pop(rec)) {
        if (last_seq != 0 && rec.seq != last_seq + 1) seq_gaps.fetch_add(1);
        if (rec.t_action_ns < last_t) t_back.fetch_add(1);
        last_seq = rec.seq;
        last_t = rec.t_action_ns;
        records.fetch_add(1);
      }
    };
    while (!stop_log.load(std::memory_order_acquire)) {
      take();
      std::this_thread::yield();
    }
    take();
  });
  int bad_mode = 0, not_published = 0;
  for (int i = 0; i < 20000; ++i) {  // the writer: 40 s of simulated time
    r.t += kTick;
    r.st.t_rx_ns = r.t;
    r1rt_core_step(r.core, &r.st, r.t, &r.out, &r.tel);
    now.store(r.t, std::memory_order_release);
    if (r1rt_core_mode(r.core) != R1RT_MODE_BALANCE) ++bad_mode;
    if (r.out.publish != 1) ++not_published;
    const auto t0 = Clock::now();
    while (Clock::now() - t0 < std::chrono::microseconds(20)) {
    }
  }
  stop.store(true, std::memory_order_release);
  policy.join();
  stop_log.store(true, std::memory_order_release);
  log.join();
  const r1rt_core_status s = r.status();
  std::printf("hand-back hammer: %llu commits (thread %llu, inline %llu), %llu discarded, %llu inline entries, "
              "%ld ring records, ring drops %llu\n",
              static_cast<unsigned long long>(s.policy_commits), static_cast<unsigned long long>(s.policy_thread_runs),
              static_cast<unsigned long long>(s.inline_runs), static_cast<unsigned long long>(s.policy_discarded),
              static_cast<unsigned long long>(s.inline_entries), records.load(),
              static_cast<unsigned long long>(seg.policy.drops()));
  EXPECT_EQ(bad_mode, 0);
  EXPECT_EQ(not_published, 0);
  EXPECT_EQ(seq_gaps.load(), 0);  // every generation pushed once, in order
  EXPECT_EQ(t_back.load(), 0);    // action times never go backwards
  EXPECT_EQ(static_cast<std::uint64_t>(records.load()) + seg.policy.drops(), s.policy_commits);
  // both paths ran and raced each other (the ordering checks above are the test): ~1,860 thread runs on a quiet host;
  // GitHub's 4-vCPU runners: 67-1,863 across runs (ASan and TSan) with every ordering check holding
  EXPECT_GT(s.policy_thread_runs, 10u);
  EXPECT_GT(s.inline_entries, 0u);  // the stalls did push the writer onto the inline rung
  EXPECT_EQ(s.faults_seen & R1RT_FAULT_POLICY_INVALID, 0u);
}

}  // namespace
