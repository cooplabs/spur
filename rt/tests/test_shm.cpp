// r1shm on POSIX shared memory (shm.hpp): create / attach / header checks / names / stale segments, and two real
// processes sharing the segment (the helper is spawned with posix_spawn).
#include <gtest/gtest.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fcntl.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "r1rt/shm.hpp"
#include "r1rt/step_api.h"
#include "test_shm_pattern.hpp"

extern char** environ;

namespace {

using r1rt::ShmError;
using r1rt::ShmSegment;

// A unique, short name per test (macOS: <= 31 chars incl. '/').
std::string test_name(const char* tag) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/r1t.%d.%s", static_cast<int>(getpid()), tag);
  return buf;
}

pid_t spawn_helper(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  std::string exe = R1RT_SHM_HELPER;
  argv.push_back(exe.data());
  std::vector<std::string> copy = args;
  for (auto& a : copy) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = -1;
  if (posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv.data(), environ) != 0) return -1;
  return pid;
}

int wait_exit(pid_t pid) {
  int status = 0;
  if (waitpid(pid, &status, 0) != pid) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
}

TEST(Shm, NameRules) {
  EXPECT_EQ(ShmSegment::check_name("/r1mc3.rt"), ShmError::kOk);
  EXPECT_EQ(ShmSegment::check_name(R1RT_SHM_NAME), ShmError::kOk);
  EXPECT_EQ(ShmSegment::check_name("r1mc3.rt"), ShmError::kBadName);   // no leading '/'
  EXPECT_EQ(ShmSegment::check_name("/a/b"), ShmError::kBadName);       // second '/'
  EXPECT_EQ(ShmSegment::check_name("/"), ShmError::kBadName);
  EXPECT_EQ(ShmSegment::check_name(nullptr), ShmError::kBadName);
  const std::string ok31 = "/" + std::string(30, 'a');                 // 31 chars: the macOS limit
  const std::string bad32 = "/" + std::string(31, 'a');
  EXPECT_EQ(ShmSegment::check_name(ok31.c_str()), ShmError::kOk);
  EXPECT_EQ(ShmSegment::check_name(bad32.c_str()), ShmError::kBadName);
  ShmSegment s;
  EXPECT_EQ(ShmSegment::create(bad32.c_str(), s), ShmError::kBadName);
  EXPECT_FALSE(s.is_open());
}

TEST(Shm, CreateAttachClose) {
  const std::string name = test_name("ca");
  ShmSegment::unlink(name.c_str());
  ShmSegment owner;
  ASSERT_EQ(ShmSegment::create(name.c_str(), owner), ShmError::kOk);
  ASSERT_TRUE(owner.is_open());
  EXPECT_TRUE(owner.owner());
  EXPECT_EQ(owner.size(), sizeof(r1rt::shm::Segment));
  EXPECT_EQ(r1rt_segment_check(owner.data(), owner.size()), R1RT_OK);
  EXPECT_EQ(owner.segment()->header.creator_pid, getpid());
  std::printf("[ info ] segment %zu B; mlock %s (errno %d)\n", owner.size(), owner.locked() ? "OK" : "refused",
              owner.mlock_errno());

  ShmSegment reader;
  ASSERT_EQ(ShmSegment::attach(name.c_str(), reader), ShmError::kOk);
  EXPECT_FALSE(reader.owner());
  EXPECT_GE(reader.size(), sizeof(r1rt::shm::Segment));
  // Both mappings see the same memory.
  owner.segment()->state.write(r1rt_test::state_for(42));
  r1rt_state s{};
  ASSERT_TRUE(reader.segment()->state.try_read(s));
  EXPECT_EQ(s.seq, 42u);
  EXPECT_TRUE(r1rt_test::state_consistent(s));

  // Move keeps one owner.
  ShmSegment moved(std::move(owner));
  EXPECT_FALSE(owner.is_open());
  EXPECT_TRUE(moved.owner());
  moved.close();  // the owner unlinks the name
  ShmSegment again;
  EXPECT_EQ(ShmSegment::attach(name.c_str(), again), ShmError::kNotFound);
  moved.close();  // twice is safe
}

TEST(Shm, AttachMissing) {
  ShmSegment s;
  EXPECT_EQ(ShmSegment::attach(test_name("none").c_str(), s), ShmError::kNotFound);
}

TEST(Shm, StaleSegmentNeedsExplicitReplace) {
  const std::string name = test_name("st");
  ShmSegment::unlink(name.c_str());
  ShmSegment first;
  ASSERT_EQ(ShmSegment::create(name.c_str(), first, r1rt::kNoMlock), ShmError::kOk);
  ShmSegment second;
  EXPECT_EQ(ShmSegment::create(name.c_str(), second), ShmError::kExists);  // never two owners by accident
  EXPECT_FALSE(second.is_open());
  EXPECT_EQ(ShmSegment::create(name.c_str(), second, r1rt::kReplaceStale | r1rt::kNoMlock), ShmError::kOk);
  // The old mapping stays valid memory but is no longer reachable by name; attach finds the new one.
  second.segment()->state.write(r1rt_test::state_for(7));
  ShmSegment reader;
  ASSERT_EQ(ShmSegment::attach(name.c_str(), reader), ShmError::kOk);
  r1rt_state s{};
  ASSERT_TRUE(reader.segment()->state.try_read(s));
  EXPECT_EQ(s.seq, 7u);
  first.close();  // the replaced creator must NOT unlink the name, which now belongs to `second`
  ShmSegment still;
  EXPECT_EQ(ShmSegment::attach(name.c_str(), still), ShmError::kOk);
  second.close();
  ShmSegment gone;
  EXPECT_EQ(ShmSegment::attach(name.c_str(), gone), ShmError::kNotFound);
}

// Header mismatches are refused with a specific error (the owner corrupts its own header in place).
TEST(Shm, HeaderMismatchRefused) {
  const std::string name = test_name("hm");
  ShmSegment::unlink(name.c_str());
  ShmSegment owner;
  ASSERT_EQ(ShmSegment::create(name.c_str(), owner, r1rt::kNoMlock), ShmError::kOk);
  auto& h = owner.segment()->header;
  struct Case {
    const char* what;
    ShmError want;
    void (*corrupt)(r1rt::shm::SegmentHeader&);
    void (*restore)(r1rt::shm::SegmentHeader&);
  };
  const Case cases[] = {
      {"magic", ShmError::kBadMagic, [](auto& x) { x.magic ^= 1u; }, [](auto& x) { x.magic ^= 1u; }},
      {"version", ShmError::kBadVersion, [](auto& x) { x.layout_version += 1; },
       [](auto& x) { x.layout_version -= 1; }},
      {"size", ShmError::kBadSize, [](auto& x) { x.segment_size += 64; }, [](auto& x) { x.segment_size -= 64; }},
      {"joints", ShmError::kBadJoints, [](auto& x) { x.num_joints += 1; }, [](auto& x) { x.num_joints -= 1; }},
      {"abi", ShmError::kAbiMismatch, [](auto& x) { x.abi_version += 1; }, [](auto& x) { x.abi_version -= 1; }},
      {"ready", ShmError::kNotReady, [](auto& x) { x.ready.store(0); }, [](auto& x) { x.ready.store(1); }},
  };
  for (const auto& c : cases) {
    c.corrupt(h);
    ShmSegment r;
    EXPECT_EQ(ShmSegment::attach(name.c_str(), r), c.want) << c.what;
    EXPECT_FALSE(r.is_open()) << c.what;
    c.restore(h);
  }
  ShmSegment ok;
  EXPECT_EQ(ShmSegment::attach(name.c_str(), ok), ShmError::kOk);
}

// An object smaller than the segment (e.g. created by an older build) is refused before it is mapped.
TEST(Shm, TooSmallRefused) {
  const std::string name = test_name("sm");
  shm_unlink(name.c_str());
  const int fd = shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(ftruncate(fd, 4096), 0);
  close(fd);
  ShmSegment r;
  EXPECT_EQ(ShmSegment::attach(name.c_str(), r), ShmError::kTooSmall);
  EXPECT_EQ(ShmSegment::unlink(name.c_str()), ShmError::kOk);
}

// The torn-read detector used by the helper: a snapshot mixing two writes is caught.
TEST(ShmTwoProcess, DetectorCatchesMixedSnapshots) {
  r1rt_state s = r1rt_test::state_for(5);
  EXPECT_TRUE(r1rt_test::state_consistent(s));
  s.in.q[25] = r1rt_test::state_for(6).in.q[25];  // the tail of the next write
  EXPECT_FALSE(r1rt_test::state_consistent(s));
  r1rt_heartbeat h = r1rt_test::heartbeat_for(9);
  h.faults = r1rt_test::heartbeat_for(10).faults;
  EXPECT_FALSE(r1rt_test::heartbeat_consistent(h));
}

// Two processes: this one writes the state seqlock every ~10 us (200x the 500 Hz writer's rate), the helper attaches
// and must see >= 20000 distinct, never torn snapshots. (Unpaced back-to-back writes of the 1.4 kB state starve the
// reader's bounded retries - by design a reader gives up rather than spin; the real writer writes every 2 ms.)
TEST(ShmTwoProcess, ChildReadsStateWhileParentWrites) {
  const std::string name = test_name("tp");
  ShmSegment::unlink(name.c_str());
  ShmSegment owner;
  ASSERT_EQ(ShmSegment::create(name.c_str(), owner, r1rt::kNoMlock), ShmError::kOk);
  const pid_t pid = spawn_helper({"read_state", name, "20000"});
  ASSERT_GT(pid, 0);
  std::uint64_t k = 0;
  int status = 0;
  const auto t0 = std::chrono::steady_clock::now();
  pid_t done = 0;
  while ((done = waitpid(pid, &status, WNOHANG)) == 0) {
    for (int i = 0; i < 100; ++i) {
      owner.segment()->state.write(r1rt_test::state_for(++k));
      const auto tw = std::chrono::steady_clock::now();
      while (std::chrono::steady_clock::now() - tw < std::chrono::microseconds(10)) {
      }
    }
    ASSERT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(20)) << "helper did not finish";
  }
  ASSERT_EQ(done, pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0) << "1 = torn read, 2 = attach failed, 3 = timeout";
  std::printf("[ info ] parent wrote %llu states while the child read\n", static_cast<unsigned long long>(k));
}

// The reverse direction: the helper writes the guard heartbeat (r1guard's region), this process reads it.
TEST(ShmTwoProcess, ParentReadsHeartbeatWhileChildWrites) {
  const std::string name = test_name("hb");
  ShmSegment::unlink(name.c_str());
  ShmSegment owner;
  ASSERT_EQ(ShmSegment::create(name.c_str(), owner, r1rt::kNoMlock), ShmError::kOk);
  constexpr long kCount = 300000;
  const pid_t pid = spawn_helper({"write_guard_heartbeat", name, std::to_string(kCount)});
  ASSERT_GT(pid, 0);
  std::uint64_t reads = 0, torn = 0, last = 0;
  int status = 0;
  pid_t done = 0;
  while ((done = waitpid(pid, &status, WNOHANG)) == 0) {
    r1rt_heartbeat h{};
    if (owner.segment()->guard_heartbeat.try_read(h) && h.seq != 0) {
      ++reads;
      if (!r1rt_test::heartbeat_consistent(h) || h.seq < last) ++torn;
      last = h.seq;
    }
  }
  ASSERT_EQ(done, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  r1rt_heartbeat fin{};
  ASSERT_TRUE(owner.segment()->guard_heartbeat.try_read(fin));
  EXPECT_EQ(fin.seq, static_cast<std::uint64_t>(kCount));
  EXPECT_TRUE(r1rt_test::heartbeat_consistent(fin));
  EXPECT_EQ(torn, 0u);
  EXPECT_GT(reads, 0u);
  std::printf("[ info ] %llu concurrent heartbeat reads, 0 torn\n", static_cast<unsigned long long>(reads));
}

// The helper's attach sees the same header checks (a version mismatch is refused in another process too).
TEST(ShmTwoProcess, ChildRefusesVersionMismatch) {
  const std::string name = test_name("vm");
  ShmSegment::unlink(name.c_str());
  ShmSegment owner;
  ASSERT_EQ(ShmSegment::create(name.c_str(), owner, r1rt::kNoMlock), ShmError::kOk);
  pid_t pid = spawn_helper({"attach", name});
  ASSERT_GT(pid, 0);
  EXPECT_EQ(wait_exit(pid), 10 + static_cast<int>(ShmError::kOk));
  owner.segment()->header.layout_version += 1;
  pid = spawn_helper({"attach", name});
  ASSERT_GT(pid, 0);
  EXPECT_EQ(wait_exit(pid), 10 + static_cast<int>(ShmError::kBadVersion));
  owner.segment()->header.layout_version -= 1;
}

// Session 13 (HW2 day 2: r1guard started 8 ms after r1rt and exited "attach /r1mc3.rt: segment smaller than expected"
// once in four starts): create() is shm_open THEN ftruncate THEN init - attach() between them sees a 0-byte object
// (kTooSmall), then a zeroed header (kNotReady). attach_wait() rides both out, bounded; an object that STAYS too small
// (another build's) is refused after too_small_max_ns, not after the whole timeout.
TEST(Shm, AttachWaitRidesOutASegmentBeingSized) {
  const std::string name = test_name("aw");
  shm_unlink(name.c_str());
  const int fd = shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);  // the creator right after its shm_open
  ASSERT_GE(fd, 0);
  ShmSegment plain;
  EXPECT_EQ(ShmSegment::attach(name.c_str(), plain), ShmError::kTooSmall);  // what r1guard exited on (base)
  std::thread creator([fd] {
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    const std::size_t size = sizeof(r1rt::shm::Segment);
    EXPECT_EQ(ftruncate(fd, static_cast<off_t>(size)), 0);
    void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    EXPECT_NE(base, MAP_FAILED);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));  // sized, the header not ready yet
    if (base != MAP_FAILED) {
      EXPECT_EQ(r1rt_segment_init(base, size, static_cast<std::int32_t>(getpid()), 1), R1RT_OK);
      munmap(base, size);
    }
    close(fd);
  });
  ShmSegment s;
  ShmSegment::AttachWait w{};
  const ShmError e = ShmSegment::attach_wait(name.c_str(), s, 2'000'000'000LL, &w);
  creator.join();
  EXPECT_EQ(e, ShmError::kOk);
  EXPECT_GT(w.too_small, 0);           // it saw the 0-byte object ...
  EXPECT_GT(w.tries, w.too_small);     // ... and the not-ready header after it
  EXPECT_LT(w.waited_ns, 1'000'000'000LL);
  s.close();
  EXPECT_EQ(ShmSegment::unlink(name.c_str()), ShmError::kOk);

  // stays too small (a 4 KB object): refused after too_small_max_ns (0.1 s here), not the 5 s timeout
  const int fd2 = shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
  ASSERT_GE(fd2, 0);
  ASSERT_EQ(ftruncate(fd2, 4096), 0);
  close(fd2);
  ShmSegment t;
  ShmSegment::AttachWait w2{};
  EXPECT_EQ(ShmSegment::attach_wait(name.c_str(), t, 5'000'000'000LL, &w2, 100'000'000LL), ShmError::kTooSmall);
  EXPECT_LT(w2.waited_ns, 1'000'000'000LL);
  EXPECT_EQ(ShmSegment::unlink(name.c_str()), ShmError::kOk);

  // nothing there at all: kNotFound after the timeout
  ShmSegment u;
  ShmSegment::AttachWait w3{};
  EXPECT_EQ(ShmSegment::attach_wait(name.c_str(), u, 50'000'000LL, &w3), ShmError::kNotFound);
  EXPECT_GE(w3.waited_ns, 50'000'000LL);
}

}  // namespace
