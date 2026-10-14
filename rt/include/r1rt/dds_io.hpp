// r1rt/dds_io.hpp - the DDS layer of the r1rt / r1guard processes (session 5; DESIGN.md §2.1, §3.1 steps 2 / 9, §12.4):
// CycloneDDS-CXX 0.10.2 with Unitree's hg types generated from our IDL (rt/idl, r1hg_dds).
//
//   rt/lowstate  -> a receive thread (the process's "DDS receive thread", DESIGN §2.1) takes every sample, checks
//                   Unitree's CRC (hg_crc over the canonical image), maps the 35 motor slots to the 26 joints
//                   (R1RT_HG_SLOT_MAP_INIT) and stamps CLOCK_MONOTONIC at receipt -> a seqlock the writer thread reads
//                   without waiting (the latest state wins; a state with a bad CRC is passed on with crc_ok = 0 so the
//                   core counts it and does not use it);
//   rt/lowcmd    <- the writer: r1rt_fill_hg_lowcmd (26 joints -> 35 slots, CRC) -> field copy into LowCmd_ -> write;
//                   the ownership token is checked by the caller immediately before write();
//   rt/lowcmd    -> (r1rt only) the BUILT-IN controller's commands for the takeover (MIRROR snapshot, the gates,
//                   BUILTIN_ACTIVE): samples from FOREIGN writers only. Our own writers (r1rt's and r1guard's) carry the
//                   DDS user_data "r1mc3:<role>"; a publication whose matched-publication QoS carries that prefix is
//                   ours and its samples are ignored (looked up once per publication handle, cached).
//
// Network configuration: CycloneDDS's own CYCLONEDDS_URI (the robot: bound to the motor NIC, DESIGN §11.3; tests and
// the simulator: the loopback interface, unicast). No exceptions leave this layer: DDS errors become return codes /
// counters; set-up failures are reported by Bus::ok() + Bus::error().
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "r1rt/seqlock.hpp"
#include "r1rt/step_api.h"
#include "r1rt/types.h"

namespace unitree_hg::msg::dds_ {
class LowState_;
class LowCmd_;
}  // namespace unitree_hg::msg::dds_

namespace r1rt::dds_io {

namespace uhg = unitree_hg::msg::dds_;

inline constexpr const char* kTopicLowState = "rt/lowstate";
inline constexpr const char* kTopicLowCmd = "rt/lowcmd";
inline constexpr const char* kUserDataPrefix = "r1mc3:";  // our writers' user_data: "r1mc3:r1rt", "r1mc3:r1guard"

// ---- conversions (pure; no DDS entities needed) ----
// A received LowState_ -> the core's input; crc_ok = (Unitree's CRC over the message == its crc field).
void to_lowstate_in(const uhg::LowState_& m, std::int64_t t_rx_ns, r1rt_lowstate_in* out) noexcept;
// Unitree's CRC of a LowState_ (what the board puts in .crc).
std::uint32_t lowstate_crc(const uhg::LowState_& m) noexcept;
// A received LowCmd_ (the built-in's) -> 26 joints (enable = motor mode, publish = 1). Returns its CRC verdict.
bool to_lowcmd_out(const uhg::LowCmd_& m, r1rt_lowcmd_out* out) noexcept;
// Our command -> the LowCmd_ we write (35 slots, mode_pr / mode_machine, CRC). Returns the CRC.
std::uint32_t to_dds(const r1rt_lowcmd_out& cmd, uhg::LowCmd_* out) noexcept;

// Counters of the DDS layer (read by the process's status line; relaxed atomics, written by one thread each).
struct BusCounters {
  std::atomic<std::uint64_t> lowstate_rx{0};
  std::atomic<std::uint64_t> lowstate_crc_bad{0};
  std::atomic<std::uint64_t> builtin_rx{0};       // foreign rt/lowcmd samples passed on
  std::atomic<std::uint64_t> own_lowcmd_rx{0};    // rt/lowcmd samples from our own writers (ignored)
  std::atomic<std::uint64_t> builtin_crc_bad{0};  // foreign rt/lowcmd with a bad CRC: dropped (a corrupt command is
                                                  // never a MIRROR snapshot)
  std::atomic<std::uint64_t> writes{0};
  std::atomic<std::uint64_t> write_errors{0};
  std::atomic<std::uint64_t> foreign_writers{0};  // distinct foreign rt/lowcmd publications seen
  std::atomic<std::uint64_t> lookup_failures{0};  // a publication's QoS could not be read (not cached; retried)
};

// The built-in's command callback (r1rt: r1rt_core_set_builtin_cmd). Called on the receive thread.
using BuiltinFn = void (*)(void* ctx, const r1rt_lowcmd_out& cmd, std::int64_t t_rx_ns);

struct BusOptions {
  int domain = 0;
  const char* role = "r1rt";     // user_data "r1mc3:<role>" on our writer
  bool write_lowcmd = true;      // a rt/lowcmd writer
  bool read_builtin = false;     // a rt/lowcmd reader for foreign writers (r1rt only)
  BuiltinFn on_builtin = nullptr;
  void* builtin_ctx = nullptr;
};

class Bus {
 public:
  explicit Bus(const BusOptions& opt) noexcept;
  ~Bus();
  Bus(const Bus&) = delete;
  Bus& operator=(const Bus&) = delete;

  bool ok() const noexcept { return ok_; }
  const std::string& error() const noexcept { return err_; }

  // The receive loop (run it on its own thread; returns when `stop` becomes true; wakes at least every 50 ms).
  void run_receive(const std::atomic<bool>& stop) noexcept;

  // The latest lowstate if a new one arrived since `*seen` (then *seen is updated). Never waits.
  bool latest(r1rt_lowstate_in* out, std::uint64_t* seen) const noexcept;

  // Write one command (the caller checked the token). false = a DDS error (counted).
  bool write(const r1rt_lowcmd_out& cmd) noexcept;

  // Is the publication with this instance handle one of ours (user_data "r1mc3:")? Cached; receive thread only.
  bool publication_is_ours(std::uint64_t publication_handle) noexcept;

  const BusCounters& counters() const noexcept { return cnt_; }
  // The instance handle of our own writer (0 without one) - the fake robot / tests identify it in sample infos.
  std::uint64_t writer_handle() const noexcept { return writer_handle_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  BusOptions opt_;
  bool ok_ = false;
  std::string err_;
  Seqlock<r1rt_lowstate_in> state_;
  std::atomic<std::uint64_t> state_count_{0};
  BusCounters cnt_;
  std::uint64_t writer_handle_ = 0;
  static constexpr int kMaxPubs = 32;
  std::uint64_t pub_handle_[kMaxPubs] = {};
  std::uint8_t pub_ours_[kMaxPubs] = {};
  int n_pubs_ = 0;
};

}  // namespace r1rt::dds_io
