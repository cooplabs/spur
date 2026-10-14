// r1rt/shm_layout.h - the shared-memory segment between r1rt, r1guard, r1d, r1hands and r1log (C++17).
//
// One POSIX shared-memory object, R1RT_SHM_NAME, created and placement-constructed by r1rt at start (in MONITOR,
// before anything is published) with r1rt_segment_init() (step_api.h); everyone else maps it and checks it with
// r1rt_segment_check() (magic, layout version, size) before use. In the simulator's lockstep mode the same struct
// lives in ordinary process memory, so the same code runs (DESIGN §2 principle 2).
//
// Rules (docs/architecture/RT_CORE_INTERFACES.md):
//   * fixed size, no pointers, only lock-free (address-free) atomics; every region starts on a 64-byte line;
//   * every region has exactly ONE writer process, except the stop channel (any client; fetch_add) and the
//     ownership token (CAS);
//   * nobody ever waits on anyone: seqlock readers retry a bounded number of times, ring producers drop + count;
//   * a layout change bumps R1RT_LAYOUT_VERSION (readers refuse a mismatch).
// Payload structs are plain C (types.h) so other languages can read them from the offsets pinned below.
#pragma once

#ifndef __cplusplus
#error "shm_layout.h is C++17; C code uses types.h and step_api.h"
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "r1rt/seqlock.hpp"
#include "r1rt/spsc_ring.hpp"
#include "r1rt/types.h"

#define R1RT_SHM_NAME "/r1mc3.rt"
#define R1RT_LAYOUT_VERSION 4u  /* 2 = session 5: the mode-request mailbox, legs / waist command groups (6-joint
                                    records, 6 keepalives), the mirror's blend / freeze / waist fields, the hands
                                    pose, the guard telemetry ring; 3 = session 8: the mirror's tau_ff (HW1 fix #28);
                                    4 = session 10: the executor status region appended (r1rt -> r1d: per-group
                                    events / cmd ids / refusals + the state-age counters) */

namespace r1rt::shm {

// "R1MC3SHM" as little-endian bytes.
inline constexpr std::uint64_t kMagic = 0x4D485333434D3152ull;
inline constexpr std::uint32_t kLayoutVersion = R1RT_LAYOUT_VERSION;

// Ring capacities (records). Telemetry: 4096 x 1136 B = ~4.7 MB, ~8 s at 500 Hz before r1log must have drained it.
inline constexpr std::size_t kUpperCmdCapacity = 256;
inline constexpr std::size_t kTelemetryCapacity = 4096;
inline constexpr std::size_t kPolicyCapacity = 512;
// Session 5: r1guard's own record per 1 kHz cycle (384 B): 4096 = ~4 s before r1log must have drained it (1.6 MB).
inline constexpr std::size_t kGuardTelemetryCapacity = 4096;
// The mode-request mailbox's reply history (replies[seq % kModeReplySlots]): a client finds its reply up to this many
// requests later (one request per writer tick at most: >= 128 ms at 500 Hz).
inline constexpr std::size_t kModeReplySlots = 64;
// A request a client claimed (WRITING) but never published is abandoned by the core after this long (the client died
// between the claim and the publish), so the mailbox cannot block forever.
inline constexpr std::int64_t kModeClaimTimeoutNs = 100'000'000;

// The guard takes over when r1rt's heartbeat is older than this many writer periods (10 ms at 500 Hz; DESIGN §4.3).
inline constexpr std::uint32_t kHeartbeatStalePeriods = 5;

// Written once by the creator before `ready` is set (release); read-only afterwards.
struct alignas(kCacheLine) SegmentHeader {
  std::uint64_t magic;
  std::uint32_t layout_version;
  std::uint32_t abi_version;     // R1RT_ABI_VERSION of the creator
  std::uint64_t segment_size;    // sizeof(Segment)
  std::int64_t created_ns;       // CLOCK_MONOTONIC
  std::int32_t creator_pid;      // r1rt
  std::uint32_t num_joints;      // R1RT_NUM_JOINTS
  std::uint32_t writer_rate_hz;  // nominal (heartbeat carries the live value)
  std::uint32_t hg_slot_map[R1RT_NUM_JOINTS];
  std::atomic<std::uint32_t> ready;  // 1 = fully constructed
};

// Owner of rt/lowcmd: one 64-bit word = owner (bits 0-7) | generation (bits 8-63), changed only by CAS.
// r1rt takes it NONE -> R1RT at takeover; r1guard CASes R1RT -> R1GUARD when r1rt's heartbeat is stale; an operator
// command returns it (generation + 1 each time, so a stale CAS can never succeed: no ABA). Each writer checks
// `holds(my_word)` immediately before every DDS write; the residual race is one packet (DESIGN §4.3).
struct alignas(kCacheLine) OwnershipToken {
  std::atomic<std::uint64_t> word;

  static constexpr std::uint64_t pack(r1rt_owner owner, std::uint64_t generation) noexcept {
    return (generation << 8) | static_cast<std::uint64_t>(owner);
  }
  static constexpr r1rt_owner owner_of(std::uint64_t w) noexcept { return static_cast<r1rt_owner>(w & 0xFFu); }
  static constexpr std::uint64_t generation_of(std::uint64_t w) noexcept { return w >> 8; }

  std::uint64_t load() const noexcept { return word.load(std::memory_order_acquire); }
  bool holds(std::uint64_t my_word) const noexcept { return load() == my_word; }
  // CAS expected -> (to, generation(expected) + 1). On success `expected` becomes the new word (the caller's
  // token); on failure it receives the current word.
  bool try_transfer(std::uint64_t& expected, r1rt_owner to) noexcept {
    const std::uint64_t desired = pack(to, generation_of(expected) + 1);
    if (word.compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
      expected = desired;
      return true;
    }
    return false;
  }
};

// Stop channel (DESIGN §4.2, R-c: a stop never depends on the thread it stops). Any client may request; r1rt reads
// every tick and acknowledges; r1guard reads it too. Per level a free-running request counter (fetch_add: lock-free,
// multi-writer) + who/when (best effort, last writer wins), and r1rt's acknowledged count.
struct alignas(kCacheLine) StopLevel {
  std::atomic<std::uint64_t> requests;      // any client: fetch_add(1, release) AFTER writing who/when
  std::atomic<std::int64_t> last_request_ns;
  std::atomic<std::uint32_t> last_requester;
  std::atomic<std::uint64_t> acked;         // r1rt: the request count it has acted on
  std::atomic<std::int64_t> acked_ns;
};
struct alignas(kCacheLine) StopChannel {
  StopLevel level[R1RT_NUM_STOP_LEVELS];    // R1RT_STOP_STOP, _HALT, _DAMP

  void request(int lvl, std::uint32_t requester, std::int64_t t_ns) noexcept {
    StopLevel& s = level[lvl];
    s.last_requester.store(requester, std::memory_order_relaxed);
    s.last_request_ns.store(t_ns, std::memory_order_relaxed);
    s.requests.fetch_add(1, std::memory_order_release);
  }
  // Sum of all requests: the "stop epoch" r1d stamps on upper-body commands (older ones are dropped).
  std::uint64_t epoch() const noexcept {
    std::uint64_t e = 0;
    for (const auto& s : level) e += s.requests.load(std::memory_order_acquire);
    return e;
  }
};

// TRAJECTORY deadman (DESIGN §3.3): r1d renews per group every 50 ms; r1rt decelerates to HOLD when it expires.
// Single writer: r1d. Layout 2: six groups (arms + head, and legs + waist for ON_STAND).
struct alignas(kCacheLine) UpperKeepalive {
  std::atomic<std::int64_t> t_ns[R1RT_NUM_CMD_GROUPS];
  std::atomic<std::uint64_t> cmd_id[R1RT_NUM_CMD_GROUPS];
};

// The mode-request mailbox (session 5; DESIGN §2 principle 3: shared memory between our processes, nobody waits):
// operator / service requests (R1RT_REQ_*) from SEVERAL clients (r1d, the r1 CLI, test harnesses) to the core, read
// by the core every writer tick inside r1rt_core_step (so the process wrapper and the lockstep driver add nothing).
//
// One request slot with a state word = (seq << 2) | phase, phase FREE 0 / WRITING 1 / READY 2:
//   client: load s; phase(s) == FREE else BUSY (retry later); CAS s -> (seq(s)+1, WRITING) (a lost CAS = BUSY);
//           write the payload (relaxed atomics); CAS (seq, WRITING) -> (seq, READY) (release) - a failed CAS means
//           the core abandoned the claim (the client stalled longer than kModeClaimTimeoutNs): ABANDONED.
//   core:   load s (acquire); READY -> read the payload, act (Core::request), write the reply into
//           replies[seq % kModeReplySlots] (a seqlock: the core is its single writer), then store (seq, FREE)
//           (release). WRITING for longer than kModeClaimTimeoutNs -> CAS (seq, WRITING) -> (seq, FREE) + an
//           ABANDONED reply.
// Every request is therefore answered exactly once, in order, at most one per writer tick; a client that sees BUSY
// retries (requests are rare: takeover steps, stops, ON_STAND). Lock-free for everyone; the core never waits.
struct ModeReply {
  std::uint64_t seq;          // the request's sequence number (0 = slot never written)
  std::int64_t t_req_ns;      // the request's own time stamp
  std::int64_t t_done_ns;     // the core's tick that processed it
  std::int32_t status;        // R1RT_OK / R1RT_E_REFUSED / R1RT_E_INVALID_ARG / R1RT_E_NOT_OWNER / R1RT_E_ABANDONED
  std::int32_t refusal;       // R1RT_TAKEOVER_* for TAKEOVER / ON_STAND, R1RT_REFUSE_* otherwise (0 = none)
  std::uint32_t req;
  std::uint32_t requester;
  std::uint16_t mode_after;   // r1rt_mode after the request
  std::uint8_t _pad[6];
};
static_assert(sizeof(ModeReply) == 48, "ModeReply layout");
struct alignas(kCacheLine) ModeMailbox {
  static constexpr std::uint64_t kFree = 0, kWriting = 1, kReady = 2;
  static constexpr std::uint64_t pack(std::uint64_t seq, std::uint64_t phase) noexcept { return (seq << 2) | phase; }
  static constexpr std::uint64_t seq_of(std::uint64_t w) noexcept { return w >> 2; }
  static constexpr std::uint64_t phase_of(std::uint64_t w) noexcept { return w & 3u; }

  std::atomic<std::uint64_t> state;        // (seq << 2) | phase
  std::atomic<std::uint32_t> req;          // payload (written while WRITING)
  std::atomic<std::uint32_t> requester;
  std::atomic<std::int64_t> t_ns;
  std::atomic<std::int64_t> arg;           // reserved (0)
  alignas(kCacheLine) Seqlock<ModeReply> replies[kModeReplySlots];  // the core writes, clients read
};


// The whole segment. Region (writer -> readers):
struct Segment {
  SegmentHeader header;                                          // creator (r1rt), once
  OwnershipToken token;                                          // CAS: r1rt / r1guard / operator via r1d
  Seqlock<r1rt_heartbeat> rt_heartbeat;                          // r1rt every tick -> r1guard, r1d, r1mon
  Seqlock<r1rt_heartbeat> guard_heartbeat;                       // r1guard every cycle -> r1rt (takeover gate), r1d
  Seqlock<r1rt_state> state;                                     // r1rt every tick -> r1d, r1guard, r1hands
  Seqlock<r1rt_mirror> mirror;                                   // r1rt every tick -> r1guard (hot standby)
  Seqlock<r1rt_base_cmd> base_cmd;                               // r1d -> r1rt (TTL)
  Seqlock<r1rt_hands_contact> hands_contact;                     // r1hands 100 Hz -> r1rt guards
  StopChannel stop;                                              // any client -> r1rt, r1guard
  UpperKeepalive keepalive;                                      // r1d -> r1rt
  SpscRing<r1rt_upper_cmd, kUpperCmdCapacity> upper_cmd;         // r1d -> r1rt
  SpscRing<r1rt_telemetry_record, kTelemetryCapacity> telemetry; // r1rt -> r1log
  SpscRing<r1rt_policy_record, kPolicyCapacity> policy;          // r1rt -> r1log
  // ---- layout 2 (session 5): appended, so every layout-1 offset above is unchanged ----
  ModeMailbox mode_req;                                          // clients (r1d, CLI, harnesses) -> r1rt
  Seqlock<r1rt_hands_pose> hands_pose;                           // r1hands -> r1rt (arm model / surface guard)
  SpscRing<r1rt_guard_record, kGuardTelemetryCapacity> guard_telemetry;  // r1guard -> r1log
  // ---- layout 4 (session 10): appended, every layout-3 offset above is unchanged ----
  Seqlock<r1rt_exec_status> exec_status;                         // r1rt every tick -> r1d (goal ends, guard names)
};

static_assert(std::is_standard_layout_v<Segment>, "the segment must be standard layout (offsetof)");
static_assert(alignof(Segment) == kCacheLine, "regions start on cache lines");
static_assert(offsetof(Segment, header) == 0, "header first");
static_assert(offsetof(Segment, token) % kCacheLine == 0 && offsetof(Segment, rt_heartbeat) % kCacheLine == 0 &&
                  offsetof(Segment, state) % kCacheLine == 0 && offsetof(Segment, stop) % kCacheLine == 0 &&
                  offsetof(Segment, upper_cmd) % kCacheLine == 0 && offsetof(Segment, telemetry) % kCacheLine == 0 &&
                  offsetof(Segment, mode_req) % kCacheLine == 0 && offsetof(Segment, hands_pose) % kCacheLine == 0 &&
                  offsetof(Segment, guard_telemetry) % kCacheLine == 0 &&
                  offsetof(Segment, exec_status) % kCacheLine == 0,
              "every region on its own cache line");
static_assert(std::atomic<std::uint32_t>::is_always_lock_free && std::atomic<std::int64_t>::is_always_lock_free,
              "lock-free atomics only");

}  // namespace r1rt::shm
