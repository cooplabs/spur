// r1rt/shm.hpp - the shared-memory segment (shm_layout.h) on a real POSIX shared-memory object (C++17).
//
// r1rt creates `R1RT_SHM_NAME` at start (in MONITOR, before anything is published): shm_open(O_CREAT | O_EXCL),
// ftruncate to sizeof(Segment), mmap, placement-construct with r1rt_segment_init() (step_api.h), then mlock the
// mapping where the OS permits it (not fatal: the result is reported, the RT set-up of DESIGN §11.1 decides). Every
// other process attaches: shm_open, fstat (size), mmap, and checks the header (ready flag, magic, layout version,
// segment size, joint count, ABI version) before use; any mismatch is refused with a specific error, never "fixed".
//
// Names: a leading '/', no other '/', at most 31 characters including the '/' (macOS PSHMNAMLEN; Linux allows more,
// the limit is applied everywhere so a name that works on Linux also works on the Mac).
//
// A segment left behind by a crashed creator is "stale": create() refuses it (kExists) unless kReplaceStale is given,
// so two r1rt processes can never both believe they own the segment. No exceptions; set-up only (never in RT paths).
#pragma once

#include <cstddef>
#include <cstdint>

#include "r1rt/shm_layout.h"

namespace r1rt {

enum class ShmError : int {
  kOk = 0,
  kBadName,       // not "/name", too long, or contains a second '/'
  kExists,        // create(): the object exists (stale segment?) and kReplaceStale was not given
  kNotFound,      // attach(): no such object
  kOpen,          // shm_open failed otherwise (errno in ShmSegment::last_errno())
  kTruncate,      // ftruncate failed
  kStat,          // fstat failed
  kMap,           // mmap failed
  kTooSmall,      // the object is smaller than the expected size
  kNotReady,      // header.ready != 1 (creator still constructing, or died during construction)
  kBadMagic,      // not an r1mc3 segment
  kBadVersion,    // layout version mismatch
  kBadSize,       // header.segment_size != the expected size
  kBadJoints,     // header.num_joints != R1RT_NUM_JOINTS
  kAbiMismatch,   // header.abi_version != R1RT_ABI_VERSION
  kInit,          // r1rt_segment_init failed
  kNotOpen,       // operation on a closed segment
};

const char* shm_error_str(ShmError e) noexcept;

enum ShmCreateFlags : unsigned {
  kShmNone = 0,
  kReplaceStale = 1u << 0,  // unlink an existing object of the same name first (operator decision / tests)
  kNoMlock = 1u << 1,       // do not try to mlock the mapping
};

class ShmSegment {
 public:
  static constexpr std::size_t kMaxNameLen = 31;  // incl. the leading '/' (macOS PSHMNAMLEN)

  ShmSegment() noexcept = default;
  ~ShmSegment() { close(); }
  ShmSegment(const ShmSegment&) = delete;
  ShmSegment& operator=(const ShmSegment&) = delete;
  ShmSegment(ShmSegment&& other) noexcept;
  ShmSegment& operator=(ShmSegment&& other) noexcept;

  static ShmError check_name(const char* name) noexcept;

  // Create + map + initialise a Segment (r1rt). `out` must be closed. On error nothing is left behind.
  static ShmError create(const char* name, ShmSegment& out, unsigned flags = kShmNone) noexcept;
  // Map an existing Segment read-write and verify its header (every other process).
  static ShmError attach(const char* name, ShmSegment& out) noexcept;
  // What attach_wait() went through (the attaching process's start words).
  struct AttachWait {
    int tries = 0;                // attach() calls
    int too_small = 0;            // of them answered kTooSmall (the creator between shm_open and ftruncate)
    std::int64_t waited_ns = 0;   // from the first call to the last
    ShmError last = ShmError::kOk;
  };
  // Session 13 (HW2 day 2: r1guard started 8 ms after r1rt and exited "segment smaller than expected" once in four
  // starts): attach, waiting (bounded by timeout_ns) while the creator makes the segment. create() is shm_open(O_CREAT)
  // THEN ftruncate THEN init - an attach in between sees a 0-byte object (kTooSmall) or a zeroed header (kNotReady).
  // kNotFound / kNotReady are retried until timeout_ns, kTooSmall for at most too_small_max_ns of it (a segment that
  // STAYS too small is another build's, not one being made); any other error at once. Polls every poll_ns. Set-up only.
  static ShmError attach_wait(const char* name, ShmSegment& out, std::int64_t timeout_ns, AttachWait* report = nullptr,
                              std::int64_t too_small_max_ns = 2'000'000'000LL,
                              std::int64_t poll_ns = 5'000'000LL) noexcept;
  // Verify a mapped header against this build (also used by attach).
  static ShmError check_header(const void* base, std::size_t size) noexcept;
  // Remove a name (no-op error kNotFound if absent).
  static ShmError unlink(const char* name) noexcept;

  // Unmap; the owner (creator) also unlinks the name. Safe to call twice.
  void close() noexcept;

  bool is_open() const noexcept { return base_ != nullptr; }
  bool owner() const noexcept { return owner_; }
  bool locked() const noexcept { return locked_; }  // mlock succeeded (creator only tries)
  int mlock_errno() const noexcept { return mlock_errno_; }
  int last_errno() const noexcept { return errno_; }
  std::size_t size() const noexcept { return size_; }
  void* data() const noexcept { return base_; }
  shm::Segment* segment() const noexcept { return static_cast<shm::Segment*>(base_); }
  const char* name() const noexcept { return name_; }

 private:
  void* base_ = nullptr;
  std::size_t size_ = 0;
  bool owner_ = false;
  bool locked_ = false;
  int mlock_errno_ = 0;
  int errno_ = 0;
  char name_[kMaxNameLen + 1] = {};
};

}  // namespace r1rt
