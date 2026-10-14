// r1shm on POSIX shared memory (shm.hpp).
#include "r1rt/shm.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <ctime>

#include "r1rt/step_api.h"

namespace r1rt {

namespace {

std::int64_t monotonic_ns() noexcept {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<std::int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

}  // namespace

const char* shm_error_str(ShmError e) noexcept {
  switch (e) {
    case ShmError::kOk: return "ok";
    case ShmError::kBadName: return "bad name (needs '/name', <= 31 chars, no other '/')";
    case ShmError::kExists: return "segment exists (stale? pass kReplaceStale to replace it)";
    case ShmError::kNotFound: return "no such segment";
    case ShmError::kOpen: return "shm_open failed";
    case ShmError::kTruncate: return "ftruncate failed";
    case ShmError::kStat: return "fstat failed";
    case ShmError::kMap: return "mmap failed";
    case ShmError::kTooSmall: return "segment smaller than expected";
    case ShmError::kNotReady: return "segment not ready (creator constructing or died)";
    case ShmError::kBadMagic: return "bad magic (not an r1mc3 segment)";
    case ShmError::kBadVersion: return "layout version mismatch";
    case ShmError::kBadSize: return "segment size mismatch";
    case ShmError::kBadJoints: return "joint count mismatch";
    case ShmError::kAbiMismatch: return "ABI version mismatch";
    case ShmError::kInit: return "segment init failed";
    case ShmError::kNotOpen: return "segment not open";
  }
  return "unknown";
}

ShmSegment::ShmSegment(ShmSegment&& other) noexcept { *this = static_cast<ShmSegment&&>(other); }

ShmSegment& ShmSegment::operator=(ShmSegment&& other) noexcept {
  if (this != &other) {
    close();
    base_ = other.base_;
    size_ = other.size_;
    owner_ = other.owner_;
    locked_ = other.locked_;
    mlock_errno_ = other.mlock_errno_;
    errno_ = other.errno_;
    std::memcpy(name_, other.name_, sizeof(name_));
    other.base_ = nullptr;
    other.size_ = 0;
    other.owner_ = false;
    other.locked_ = false;
    other.name_[0] = '\0';
  }
  return *this;
}

ShmError ShmSegment::check_name(const char* name) noexcept {
  if (name == nullptr || name[0] != '/') return ShmError::kBadName;
  const std::size_t n = std::strlen(name);
  if (n < 2 || n > kMaxNameLen) return ShmError::kBadName;
  if (std::strchr(name + 1, '/') != nullptr) return ShmError::kBadName;
  return ShmError::kOk;
}

ShmError ShmSegment::check_header(const void* base, std::size_t size) noexcept {
  using shm::Segment;
  if (base == nullptr) return ShmError::kNotOpen;
  if (size < sizeof(Segment)) return ShmError::kTooSmall;
  const auto* seg = static_cast<const Segment*>(base);
  const auto& h = seg->header;
  if (h.ready.load(std::memory_order_acquire) != 1) return ShmError::kNotReady;
  if (h.magic != shm::kMagic) return ShmError::kBadMagic;
  if (h.layout_version != shm::kLayoutVersion) return ShmError::kBadVersion;
  if (h.segment_size != sizeof(Segment)) return ShmError::kBadSize;
  if (h.num_joints != R1RT_NUM_JOINTS) return ShmError::kBadJoints;
  if (h.abi_version != R1RT_ABI_VERSION) return ShmError::kAbiMismatch;
  return ShmError::kOk;
}

ShmError ShmSegment::create(const char* name, ShmSegment& out, unsigned flags) noexcept {
  if (out.is_open()) return ShmError::kNotOpen;
  if (check_name(name) != ShmError::kOk) return ShmError::kBadName;
  const std::size_t size = sizeof(shm::Segment);
  int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (fd < 0 && errno == EEXIST && (flags & kReplaceStale) != 0) {
    shm_unlink(name);
    fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
  }
  if (fd < 0) {
    out.errno_ = errno;
    return errno == EEXIST ? ShmError::kExists : ShmError::kOpen;
  }
  auto fail = [&](ShmError e, void* base) {
    out.errno_ = errno;
    if (base != nullptr) munmap(base, size);
    ::close(fd);
    shm_unlink(name);
    return e;
  };
  if (ftruncate(fd, static_cast<off_t>(size)) != 0) return fail(ShmError::kTruncate, nullptr);
  void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (base == MAP_FAILED) return fail(ShmError::kMap, nullptr);
  if (r1rt_segment_init(base, size, static_cast<int32_t>(getpid()), monotonic_ns()) != R1RT_OK)
    return fail(ShmError::kInit, base);
  ::close(fd);  // the mapping keeps the object alive
  out.base_ = base;
  out.size_ = size;
  out.owner_ = true;
  std::memcpy(out.name_, name, std::strlen(name) + 1);
  if ((flags & kNoMlock) == 0) {
    if (mlock(base, size) == 0) {
      out.locked_ = true;
    } else {
      out.mlock_errno_ = errno;
    }
  }
  return ShmError::kOk;
}

ShmError ShmSegment::attach(const char* name, ShmSegment& out) noexcept {
  if (out.is_open()) return ShmError::kNotOpen;
  if (check_name(name) != ShmError::kOk) return ShmError::kBadName;
  const int fd = shm_open(name, O_RDWR, 0600);
  if (fd < 0) {
    out.errno_ = errno;
    return errno == ENOENT ? ShmError::kNotFound : ShmError::kOpen;
  }
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    out.errno_ = errno;
    ::close(fd);
    return ShmError::kStat;
  }
  // macOS rounds shm objects up to whole pages, so the object may be larger than the segment, never smaller.
  const std::size_t size = static_cast<std::size_t>(st.st_size);
  if (size < sizeof(shm::Segment)) {
    ::close(fd);
    return ShmError::kTooSmall;
  }
  void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (base == MAP_FAILED) {
    out.errno_ = errno;
    return ShmError::kMap;
  }
  const ShmError e = check_header(base, size);
  if (e != ShmError::kOk) {
    munmap(base, size);
    return e;
  }
  out.base_ = base;
  out.size_ = size;
  out.owner_ = false;
  std::memcpy(out.name_, name, std::strlen(name) + 1);
  return ShmError::kOk;
}

ShmError ShmSegment::attach_wait(const char* name, ShmSegment& out, std::int64_t timeout_ns, AttachWait* report,
                                 std::int64_t too_small_max_ns, std::int64_t poll_ns) noexcept {
  AttachWait w{};
  const std::int64_t t0 = monotonic_ns();
  std::int64_t too_small_since = -1;  // when the current run of kTooSmall answers began
  ShmError e = ShmError::kNotFound;
  for (;;) {
    e = attach(name, out);
    w.tries += 1;
    w.last = e;
    const std::int64_t now = monotonic_ns();
    w.waited_ns = now - t0;
    if (e == ShmError::kOk) break;
    if (e == ShmError::kTooSmall) {
      w.too_small += 1;
      if (too_small_since < 0) too_small_since = now;
      if (now - too_small_since > too_small_max_ns) break;  // stays too small: not a segment being created
    } else {
      too_small_since = -1;
      if (e != ShmError::kNotFound && e != ShmError::kNotReady) break;
    }
    if (now - t0 > timeout_ns) break;
    timespec ts{};
    ts.tv_sec = static_cast<time_t>(poll_ns / 1000000000LL);
    ts.tv_nsec = static_cast<long>(poll_ns % 1000000000LL);
    nanosleep(&ts, nullptr);
  }
  if (report != nullptr) *report = w;
  return e;
}

ShmError ShmSegment::unlink(const char* name) noexcept {
  if (check_name(name) != ShmError::kOk) return ShmError::kBadName;
  if (shm_unlink(name) != 0) return errno == ENOENT ? ShmError::kNotFound : ShmError::kOpen;
  return ShmError::kOk;
}

// True if `name` still refers to the object mapped at `mine` (same creator pid and creation time in the header). A
// creator whose segment was replaced (kReplaceStale by another process) must not unlink the replacement's name.
static bool name_is_mine(const char* name, const shm::Segment* mine) noexcept {
  const int fd = shm_open(name, O_RDONLY, 0600);
  if (fd < 0) return false;
  void* p = mmap(nullptr, sizeof(shm::SegmentHeader), PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) return false;
  const auto* h = static_cast<const shm::SegmentHeader*>(p);
  const bool same = h->magic == mine->header.magic && h->creator_pid == mine->header.creator_pid &&
                    h->created_ns == mine->header.created_ns;
  munmap(p, sizeof(shm::SegmentHeader));
  return same;
}

void ShmSegment::close() noexcept {
  if (base_ == nullptr) return;
  if (owner_ && name_is_mine(name_, segment())) shm_unlink(name_);
  if (locked_) munlock(base_, size_);
  munmap(base_, size_);
  base_ = nullptr;
  size_ = 0;
  owner_ = false;
  locked_ = false;
  name_[0] = '\0';
}

}  // namespace r1rt
