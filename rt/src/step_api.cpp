// The C ABI of the real-time core (step_api.h): the shared-segment functions (session 1) and the core (session 3:
// r1rt::Core, rt/src/core.cpp). The ABI layer only checks arguments and forwards.
#include "r1rt/step_api.h"

#include <cstring>
#include <new>

#include "r1rt/core.hpp"
#include "r1rt/shm_layout.h"

namespace {

constexpr std::uint32_t kSlotMap[R1RT_NUM_JOINTS] = R1RT_HG_SLOT_MAP_INIT;

void fill_no_publish(r1rt_lowcmd_out* out) noexcept {
  std::memset(out, 0, sizeof(*out));  // publish = 0: write nothing (also enable = 0, but nothing is sent at all)
  out->mode_pr = R1RT_MODE_PR;
  out->mode_machine = R1RT_MODE_MACHINE_R1;
}

}  // namespace

struct r1rt_core {
  r1rt::Core core;
};

r1rt::Core* r1rt::core_of(r1rt_core* c) noexcept { return c == nullptr ? nullptr : &c->core; }

extern "C" {

uint32_t r1rt_abi_version(void) { return R1RT_ABI_VERSION; }

size_t r1rt_segment_size(void) { return sizeof(r1rt::shm::Segment); }

int32_t r1rt_segment_init(void* base, size_t size, int32_t creator_pid, int64_t t_monotonic_ns) {
  using r1rt::shm::Segment;
  if (base == nullptr || size < sizeof(Segment)) return R1RT_E_INVALID_ARG;
  if (reinterpret_cast<std::uintptr_t>(base) % alignof(Segment) != 0) return R1RT_E_INVALID_ARG;
  std::memset(base, 0, sizeof(Segment));
  auto* seg = new (base) Segment();  // value-initialises every region (atomics 0, rings empty, seqlocks unwritten)
  auto& h = seg->header;
  h.magic = r1rt::shm::kMagic;
  h.layout_version = r1rt::shm::kLayoutVersion;
  h.abi_version = R1RT_ABI_VERSION;
  h.segment_size = sizeof(Segment);
  h.created_ns = t_monotonic_ns;
  h.creator_pid = creator_pid;
  h.num_joints = R1RT_NUM_JOINTS;
  h.writer_rate_hz = 0;
  for (int i = 0; i < R1RT_NUM_JOINTS; ++i) h.hg_slot_map[i] = kSlotMap[i];
  seg->token.word.store(r1rt::shm::OwnershipToken::pack(R1RT_OWNER_NONE, 0), std::memory_order_relaxed);
  h.ready.store(1, std::memory_order_release);
  return R1RT_OK;
}

int32_t r1rt_segment_check(const void* base, size_t size) {
  using r1rt::shm::Segment;
  if (base == nullptr || size < sizeof(Segment)) return R1RT_E_BAD_SEGMENT;
  const auto* seg = static_cast<const Segment*>(base);
  if (seg->header.ready.load(std::memory_order_acquire) != 1) return R1RT_E_BAD_SEGMENT;
  if (seg->header.magic != r1rt::shm::kMagic || seg->header.layout_version != r1rt::shm::kLayoutVersion ||
      seg->header.segment_size != sizeof(Segment) || seg->header.num_joints != R1RT_NUM_JOINTS)
    return R1RT_E_BAD_SEGMENT;
  if (seg->header.abi_version != R1RT_ABI_VERSION) return R1RT_E_ABI_MISMATCH;
  return R1RT_OK;
}

int32_t r1rt_segment_push_upper_cmd(void* base, const r1rt_upper_cmd* cmd) {
  if (cmd == nullptr) return R1RT_E_INVALID_ARG;
  if (r1rt_segment_check(base, sizeof(r1rt::shm::Segment)) != R1RT_OK) return R1RT_E_BAD_SEGMENT;
  auto* seg = static_cast<r1rt::shm::Segment*>(base);
  return seg->upper_cmd.try_push(*cmd) ? R1RT_OK : R1RT_E_REFUSED;
}

int32_t r1rt_segment_upper_keepalive(void* base, uint32_t group, uint64_t cmd_id, int64_t t_ns) {
  if (group >= R1RT_NUM_CMD_GROUPS) return R1RT_E_INVALID_ARG;
  if (r1rt_segment_check(base, sizeof(r1rt::shm::Segment)) != R1RT_OK) return R1RT_E_BAD_SEGMENT;
  auto* seg = static_cast<r1rt::shm::Segment*>(base);
  // the stamp first, then the id it belongs to (the core reads the id, then the stamp: a stamp newer than its id is
  // harmless - a renewal never shortens the deadline, and the executor ignores ids it is not running)
  seg->keepalive.t_ns[group].store(t_ns, std::memory_order_relaxed);
  seg->keepalive.cmd_id[group].store(cmd_id, std::memory_order_release);
  return R1RT_OK;
}

uint64_t r1rt_segment_stop_epoch(const void* base) {
  if (r1rt_segment_check(base, sizeof(r1rt::shm::Segment)) != R1RT_OK) return 0;
  return static_cast<const r1rt::shm::Segment*>(base)->stop.epoch();
}

int32_t r1rt_segment_request_mode(void* base, uint32_t req, uint32_t requester, int64_t t_ns, uint64_t* seq_out) {
  // The client side of the mailbox (shm_layout.h ModeMailbox): claim FREE -> WRITING by CAS, write the payload,
  // publish WRITING -> READY by CAS (a failed publish = the core abandoned the claim). Never waits.
  using MB = r1rt::shm::ModeMailbox;
  if (seq_out != nullptr) *seq_out = 0;
  if (requester >= R1RT_NUM_REQUESTERS) return R1RT_E_INVALID_ARG;
  if (r1rt_segment_check(base, sizeof(r1rt::shm::Segment)) != R1RT_OK) return R1RT_E_BAD_SEGMENT;
  MB& mb = static_cast<r1rt::shm::Segment*>(base)->mode_req;
  std::uint64_t s = mb.state.load(std::memory_order_acquire);
  if (MB::phase_of(s) != MB::kFree) return R1RT_E_AGAIN;
  const std::uint64_t seq = MB::seq_of(s) + 1;
  const std::uint64_t writing = MB::pack(seq, MB::kWriting);
  if (!mb.state.compare_exchange_strong(s, writing, std::memory_order_acq_rel, std::memory_order_acquire))
    return R1RT_E_AGAIN;  // another client claimed it first
  mb.req.store(req, std::memory_order_relaxed);
  mb.requester.store(requester, std::memory_order_relaxed);
  mb.t_ns.store(t_ns, std::memory_order_relaxed);
  mb.arg.store(0, std::memory_order_relaxed);
  std::uint64_t expect = writing;
  if (!mb.state.compare_exchange_strong(expect, MB::pack(seq, MB::kReady), std::memory_order_release,
                                        std::memory_order_relaxed))
    return R1RT_E_ABANDONED;
  if (seq_out != nullptr) *seq_out = seq;
  return R1RT_OK;
}

int32_t r1rt_segment_mode_reply(const void* base, uint64_t seq, int32_t* status, int32_t* refusal,
                                uint32_t* mode_after) {
  using r1rt::shm::kModeReplySlots;
  if (seq == 0) return R1RT_E_INVALID_ARG;
  if (r1rt_segment_check(base, sizeof(r1rt::shm::Segment)) != R1RT_OK) return R1RT_E_BAD_SEGMENT;
  const auto& mb = static_cast<const r1rt::shm::Segment*>(base)->mode_req;
  r1rt::shm::ModeReply r{};
  if (!mb.replies[seq % kModeReplySlots].try_read(r)) return R1RT_E_AGAIN;  // being written right now
  if (r.seq == seq) {
    if (status != nullptr) *status = r.status;
    if (refusal != nullptr) *refusal = r.refusal;
    if (mode_after != nullptr) *mode_after = r.mode_after;
    return R1RT_OK;
  }
  if (r.seq > seq) return R1RT_E_INVALID_ARG;  // overwritten by a later request
  // not processed yet - or never issued (beyond the last claimed sequence)
  const std::uint64_t issued = r1rt::shm::ModeMailbox::seq_of(mb.state.load(std::memory_order_acquire));
  return seq <= issued ? R1RT_E_AGAIN : R1RT_E_INVALID_ARG;
}

r1rt_core* r1rt_core_create(const r1rt_config* config, int32_t* status_out) {
  auto fail = [status_out](int32_t code) -> r1rt_core* {
    if (status_out != nullptr) *status_out = code;
    return nullptr;
  };
  if (config == nullptr) return fail(R1RT_E_INVALID_ARG);
  if (config->abi_version != R1RT_ABI_VERSION || config->struct_size != sizeof(r1rt_config))
    return fail(R1RT_E_ABI_MISMATCH);
  if (config->writer_rate_hz < 200 || config->writer_rate_hz > 1000 || config->policy_rate_hz == 0 ||
      config->writer_rate_hz % config->policy_rate_hz != 0)
    return fail(R1RT_E_INVALID_ARG);
  if (r1rt_segment_check(config->segment, config->segment_size) != R1RT_OK) return fail(R1RT_E_BAD_SEGMENT);
  auto* core = new (std::nothrow) r1rt_core();
  if (core == nullptr) return fail(R1RT_E_INVALID_ARG);
  const int32_t st = core->core.init(*config, nullptr, 0);
  if (st != R1RT_OK) {
    delete core;
    return fail(st);
  }
  if (status_out != nullptr) *status_out = R1RT_OK;
  return core;
}

void r1rt_core_destroy(r1rt_core* core) { delete core; }

int32_t r1rt_core_step(r1rt_core* core, const r1rt_lowstate_in* state, int64_t t_monotonic_ns,
                       r1rt_lowcmd_out* cmd_out, r1rt_telemetry_record* telemetry_out) {
  if (cmd_out == nullptr) return R1RT_E_INVALID_ARG;
  fill_no_publish(cmd_out);
  if (core == nullptr) return R1RT_E_INVALID_ARG;
  return core->core.step(state, t_monotonic_ns, cmd_out, telemetry_out);
}

r1rt_mode r1rt_core_mode(const r1rt_core* core) {
  // the enum constant cast to the typedef: GCC rejects an enumeral / non-enumeral conditional (-Werror=extra)
  return core == nullptr ? static_cast<r1rt_mode>(R1RT_MODE_MONITOR) : core->core.mode();
}

int32_t r1rt_core_policy_step(r1rt_core* core, int64_t t_monotonic_ns) {
  if (core == nullptr) return R1RT_E_INVALID_ARG;
  return core->core.policy_step(t_monotonic_ns);
}

int32_t r1rt_core_request(r1rt_core* core, int32_t request, int64_t t_monotonic_ns) {
  if (core == nullptr) return R1RT_E_INVALID_ARG;
  return core->core.request(request, t_monotonic_ns);
}

int32_t r1rt_core_set_base_cmd(r1rt_core* core, const r1rt_base_cmd* cmd) {
  if (core == nullptr || cmd == nullptr) return R1RT_E_INVALID_ARG;
  return core->core.set_base_cmd(*cmd);
}

int32_t r1rt_core_set_builtin_cmd(r1rt_core* core, const r1rt_lowcmd_out* cmd, int64_t t_rx_ns) {
  if (core == nullptr || cmd == nullptr) return R1RT_E_INVALID_ARG;
  return core->core.set_builtin_cmd(*cmd, t_rx_ns);
}

int32_t r1rt_core_inject(r1rt_core* core, int32_t kind, int64_t t_start_ns, int64_t t_end_ns, double value) {
  if (core == nullptr) return R1RT_E_INVALID_ARG;
  return core->core.inject(kind, t_start_ns, t_end_ns, value);
}

int32_t r1rt_core_set_stand_session(r1rt_core* core, int32_t on) {
  if (core == nullptr) return R1RT_E_INVALID_ARG;
  return core->core.set_stand_session(on != 0) ? R1RT_OK : R1RT_E_REFUSED;
}

int32_t r1rt_core_set_rt_granted(r1rt_core* core, int32_t granted) {
  if (core == nullptr) return R1RT_E_INVALID_ARG;
  core->core.set_rt_granted(granted != 0);
  return R1RT_OK;
}

int32_t r1rt_core_set_loop_timing(r1rt_core* core, int32_t wake_lateness_ns, int32_t prev_iteration_ns) {
  if (core == nullptr) return R1RT_E_INVALID_ARG;
  core->core.set_loop_timing(wake_lateness_ns, prev_iteration_ns);
  return R1RT_OK;
}

int32_t r1rt_core_get_status(const r1rt_core* core, r1rt_core_status* out) {
  if (core == nullptr || out == nullptr) return R1RT_E_INVALID_ARG;
  core->core.status(out);
  return R1RT_OK;
}

}  // extern "C"
