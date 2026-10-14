// r1rt::Core - the real-time core behind the step C ABI (step_api.h): the 500 Hz writer step and the 50 Hz policy
// thread's iteration (DESIGN.md §3.1-§3.2, §4.2-§4.3).
//
// Modes (§3.5): MONITOR (nothing published) -> TAKEOVER_MIRROR (republish a snapshot of the built-in controller's last
// rt/lowcmd while r1d runs ReleaseMode) -> TAKEOVER_BLEND (linear blend from the snapshot to our BALANCE command over
// takeover_blend_s) -> BALANCE; STAND_FREEZE; DAMP (operator only). R1RT_REQ_ENTER_BALANCE (MONITOR -> BALANCE
// directly) stays for the simulator's single-core tests. The takeover's failure branches are v3's (§3.5): a gate fails
// -> MONITOR unchanged; a fault during MIRROR -> keep mirroring, alarm; ABORT (release failed) -> MONITOR, the token
// back, alarm; a fault during BLEND -> finish the blend, alarm; an invalid policy action in BLEND -> STAND_FREEZE
// from the blended command (rung 3). Nothing in a takeover ever damps.
//
// Writer step, in order (§3.1): loop timing -> the latest lowstate (age: late > 10 ms counted, stale > 100 ms = the
// last BALANCED command is resent, lost > 1 s = a fault, still publishing) -> estimate (projected gravity, tilt) ->
// checks (§4.2: operator stops, lowstate age, tilt (30 deg = an event only: the policy keeps running), motor error
// words (event + alarm; nothing drops stiffness), policy age / validity) -> legs + waist: the latest valid policy
// action (zero-order hold); older than policy_max_age (40 ms) -> the writer runs the policy INLINE (rung 1, bumpless:
// the same runner state, observation code and phase clock) until the policy thread is back; an invalid action
// (non-finite / beyond action_abs_max) -> STAND_FREEZE (rung 3: a ramp to the CoM-centred stand at stand gains, then
// hold) -> arms + head: the upper-body executor (r1rt::UpperBodyExecutor, DESIGN §3.3: HOLD / TRAJECTORY / STREAM
// from the upper_cmd ring, keepalives, stop epochs, guards) -> hard limits (clamp + count) -> LowCmd
// (26 joints, mode_machine 1, PR ankles) -> ownership token check -> telemetry record, state seqlock, heartbeat,
// guard mirror. NEVER an automatic damp: DAMP is entered only by an operator request.
//
// Threads: in lockstep (the simulator, tests) step() and policy_step() are called in a fixed order from one thread.
// On the robot they run in two threads (writer FIFO 90, policy FIFO 80). The hand-back between them (session 4) is
// race-free: the policy runner's state lives in ONE committed slot (a seqlock written only by the winner of a
// generation CAS); the policy thread and the writer's inline rung each compute on their OWN runner restored from the
// committed state they read, and commit only if the generation is still the one they read (compare-and-swap g ->
// g + 1). So a thread result computed before an inline run can never overwrite it (it is discarded and counted), an
// inline run that lost the race to a thread result uses that result, and the commits (and the policy-ring pushes,
// made between the CAS and the slot write) are totally ordered. The thread reads the state through the segment's
// state seqlock, never the writer's members; the flags it shares with the writer are atomics. The thread still
// computes nothing while the writer is inline (it beats; the writer hands back after kHandbackBeats beats), so the
// inline rung's behaviour is unchanged. rt/tests/test_core_threads.cpp hammers it from two threads (TSan in the Linux
// CI job; macOS sanitizer runtimes do not start).
//
// No allocation after init(), no exceptions, no files / network / Python in step() / policy_step() (a counting
// operator new test pins it: rt/tests/test_core.cpp).
#pragma once

#include <atomic>
#include <cstdint>

#include "r1rt/executor.hpp"
#include "r1rt/policy_mlp.hpp"
#include "r1rt/policy_runner.hpp"
#include "r1rt/resume.hpp"
#include "r1rt/seqlock.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/state_age.hpp"
#include "r1rt/step_api.h"

namespace r1rt {

class Core {
 public:
  Core() noexcept = default;
  Core(const Core&) = delete;
  Core& operator=(const Core&) = delete;

  // Validates the config, loads the policy weights (the only file access; may allocate). R1RT_OK or an error.
  int32_t init(const r1rt_config& cfg, char* err, std::size_t errlen) noexcept;

  int32_t step(const r1rt_lowstate_in* state, std::int64_t t_ns, r1rt_lowcmd_out* out,
               r1rt_telemetry_record* tel) noexcept;
  // The built-in controller's last rt/lowcmd (the DDS receive thread; single writer of its seqlock).
  int32_t set_builtin_cmd(const r1rt_lowcmd_out& cmd, std::int64_t t_rx_ns) noexcept;
  int32_t policy_step(std::int64_t t_ns) noexcept;
  int32_t request(std::int32_t req, std::int64_t t_ns) noexcept;
  int32_t set_base_cmd(const r1rt_base_cmd& cmd) noexcept;
  int32_t inject(std::int32_t kind, std::int64_t t0, std::int64_t t1, double value) noexcept;
  void status(r1rt_core_status* out) const noexcept;
  // session 5 (the process wrapper, writer thread): the real-time grants; this tick's loop timing (step_api.h)
  void set_rt_granted(bool granted) noexcept { rt_granted_ = granted; }
  // session 6: the stand session (r1rt --stand-session): latched - false once true is refused
  bool set_stand_session(bool on) noexcept {
    if (!on && stand_session_) return false;
    stand_session_ = stand_session_ || on;
    return true;
  }
  void set_loop_timing(std::int32_t wake_lateness_ns, std::int32_t prev_iteration_ns) noexcept {
    wake_lateness_ns_ = wake_lateness_ns;
    prev_iteration_ns_ = prev_iteration_ns;
  }
  r1rt_mode mode() const noexcept { return mode_.load(std::memory_order_acquire); }

  // Test hook (tests only, nullptr otherwise): called inside a policy computation between the read of the committed
  // state and the commit, with the source (kSourceThread / kSourceInline / kSourceEntry) - lets a single-threaded test
  // interleave a writer tick exactly where a real stall would put it.
  using CommitHook = void (*)(void* ctx, int source);
  void set_commit_hook(CommitHook hook, void* ctx) noexcept {
    commit_hook_ = hook;
    commit_hook_ctx_ = ctx;
  }
  static constexpr int kSourceThread = 0;  // who ran the policy step
  static constexpr int kSourceInline = 1;
  static constexpr int kSourceEntry = 2;   // the first step at ENTER_BALANCE / TAKEOVER_RELEASED

 private:
  struct Mailbox {  // one committed policy result: the action the writer uses
    double target[R1RT_NUM_POLICY_ACTIONS];
    double raw[R1RT_NUM_POLICY_ACTIONS];
    std::int64_t t_action_ns;
    std::uint64_t seq;           // = the commit generation
    double gait_phase;           // of the committing runner, after the step (status / mirror)
    double lean;
    std::uint8_t valid;
    std::uint8_t inline_run;
    std::uint8_t _pad[6];
  };
  struct Committed {  // the committed slot: the runner state the next step continues from + its result
    PolicyRunner::State rs;
    Mailbox mb;
    std::uint64_t gen;           // 0 = nothing committed yet
    std::uint8_t has_state;
    std::uint8_t _pad[7];
  };
  struct Builtin {  // the built-in controller's last rt/lowcmd
    r1rt_lowcmd_out cmd;
    std::int64_t t_rx_ns;
    std::uint64_t n;             // samples received (0 = none)
  };
  struct FilterCtx {
    const Core* core;
    std::int64_t t_ns;
  };
  struct Injection {
    std::int32_t kind;
    std::int64_t t0, t1;
    double value;
  };
  static constexpr int kMaxInjections = 8;

  bool fresh(std::int64_t t_ns) const noexcept;
  void current_cmd(std::int64_t t_ns, double cmd[3], double* lean) const noexcept;
  // One policy step on the source's own runner from the committed state, committed iff the generation is unchanged.
  static constexpr int kPolicyValid = 0;
  static constexpr int kPolicyInvalid = 1;    // committed, but the action is invalid (rung 3)
  static constexpr int kPolicyDiscarded = 2;  // another commit came first: dropped, never used
  static constexpr int kPolicyBusy = 3;       // the committed slot could not be read (a commit in progress)
  int run_policy(std::int64_t t_ns, int source, const r1rt_lowstate_in& in) noexcept;
  void read_mailbox(Mailbox& mb) noexcept;  // the writer's view of the committed action (last good copy on a miss)
  // BALANCE's command (also BLEND's "ours"): kBal* below; out = the policy's action + (by the caller) the executor
  static constexpr int kBalOk = 0;
  static constexpr int kBalFroze = 1;  // an invalid action: STAND_FREEZE entered, out = its ramp
  static constexpr int kBalStale = 2;  // no fresh state: out = the last balanced command, resent
  int balance_tick(std::int64_t t_ns, r1rt_lowcmd_out* out, std::uint32_t* faults) noexcept;
  // R1RT_TAKEOVER_*; on OK the token is ours, the snapshot taken, MIRROR entered. target = BALANCE (a takeover) or
  // ON_STAND (session 5: the robot hanging on its stand - no policy needed, the built-in need not be stiff)
  int takeover_gate(std::int64_t t_ns, r1rt_mode target) noexcept;
  void on_stand_command(r1rt_lowcmd_out* out, std::int64_t t_ns) noexcept;  // legs + waist L0 + the upper body
  bool lower_active() const noexcept;  // the lower executor publishes legs + waist (ON_STAND, or its blend)
  void sync_counters() noexcept;                  // the policy thread's atomic counters -> st_
  void enter_freeze(std::int64_t t_ns, bool operator_request = false) noexcept;
  // session 9 (U19 / D25): the state-age limit. state_age_step: every tick, the age (receive / board-tick content),
  // the STATE_OLD alarm, the sustained episodes; true = the rung should act now (BALANCE, enabled, sustained).
  bool state_age_step(std::int64_t t_ns, std::int64_t rx_age, bool new_sample, std::uint32_t* faults) noexcept;
  void stale_return_step(std::int64_t t_ns) noexcept;  // stale_rung 2: BALANCE again once the state is live
  void build_balance(r1rt_lowcmd_out* out, const Mailbox& mb) const noexcept;
  void build_freeze(r1rt_lowcmd_out* out, std::int64_t t_ns) const noexcept;
  void build_damp(r1rt_lowcmd_out* out) const noexcept;
  void upper_step(r1rt_lowcmd_out* out, std::int64_t t_ns) noexcept;  // the executor's tick (joints 14-25)
  void internal_stop(std::int64_t t_ns) noexcept;  // a stop the core itself raises, through the stop channel
  std::uint32_t apply_limits(r1rt_lowcmd_out* out, std::uint32_t* faults) noexcept;
  void poll_stop_channel(std::int64_t t_ns) noexcept;
  // Session 5: the mode-request mailbox (shm_layout.h ModeMailbox) - at most one request per tick, answered once.
  void poll_mode_mailbox(std::int64_t t_ns) noexcept;
  void write_mode_reply(std::uint64_t seq, std::uint32_t req, std::uint32_t requester, std::int64_t t_req,
                        std::int64_t t_ns, std::int32_t status, std::int32_t refusal) noexcept;
  void publish_shared(std::int64_t t_ns, const r1rt_lowcmd_out& out, std::uint32_t faults, std::uint32_t clamps,
                      std::int64_t age, r1rt_telemetry_record* tel, std::int64_t t_wake) noexcept;
  static void filter_thunk(void* ctx, double* action, int n) noexcept;

  r1rt_config cfg_{};
  shm::Segment* seg_ = nullptr;
  PolicyMlp net_;               // stateless: both runners share it (PolicyMlp::run has no mutable state)
  PolicyRunner thread_runner_;  // the policy thread's private runner
  PolicyRunner writer_runner_;  // the writer's (inline rung, entry) private runner
  FilterCtx thread_filter_{};
  FilterCtx writer_filter_{};
  bool policy_loaded_ = false;
  std::int64_t period_ns_ = 2'000'000;
  std::int64_t policy_period_ns_ = 20'000'000;

  std::atomic<r1rt_mode> mode_{R1RT_MODE_MONITOR};  // written by the writer / requests, read by the policy thread
  std::uint64_t token_word_ = 0;  // our ownership word while we own rt/lowcmd
  bool own_token_ = false;

  r1rt_lowstate_in latest_{};
  bool have_state_ = false;
  double gravity_[3] = {0.0, 0.0, -1.0};
  double tilt_ = 0.0;

  Seqlock<Committed> committed_;           // single logical writer: the winner of the generation CAS
  std::atomic<std::uint64_t> gen_{0};      // the committed generation (CAS g -> g + 1 before writing the slot)
  Mailbox last_mb_{};                      // the writer's last good read of the committed action
  std::atomic<bool> inline_active_{false}; // the writer runs the policy (rung 1); the thread only beats meanwhile
  std::int64_t next_inline_ns_ = 0;
  std::atomic<std::uint64_t> thread_beats_{0};  // policy-thread iterations seen while inline
  std::uint64_t beats_at_inline_ = 0;
  // counters the policy thread touches (copied into the status by the writer / status())
  std::atomic<std::uint64_t> n_thread_runs_{0};
  std::atomic<std::uint64_t> n_invalid_{0};
  std::atomic<std::uint64_t> n_commits_{0};
  std::atomic<std::uint64_t> n_discarded_{0};
  std::atomic<std::int64_t> last_action_t_ns_{0};
  CommitHook commit_hook_ = nullptr;
  void* commit_hook_ctx_ = nullptr;

  // the takeover (DESIGN §3.5)
  Seqlock<Builtin> builtin_;               // single writer: set_builtin_cmd (the DDS receive thread)
  std::uint64_t builtin_n_ = 0;            // set_builtin_cmd's own sample counter
  std::atomic<std::int64_t> builtin_t_rx_ns_{0};
  r1rt_lowcmd_out mirror_cmd_{};           // the snapshot MIRROR republishes and BLEND starts from
  std::int64_t mirror_t0_ = 0;
  std::int64_t blend_t0_ = 0;
  std::int64_t released_t_ns_ = 0;         // RELEASED / ENTER_BALANCE: built-in samples after it = two writers
  std::int64_t blend_ns_ = 300'000'000;  // takeover_blend_s in ns (an exact 1.0 at the end of the blend)
  double blend_progress_ = 0.0;
  bool alarm_pending_ = false;             // a one-shot TAKEOVER_ALARM for the next tick (ABORT)

  r1rt_lowcmd_out last_cmd_{};             // the last command built (any mode)
  r1rt_lowcmd_out balanced_cmd_{};         // the last BALANCE command built with fresh state
  bool have_balanced_ = false;
  // the upper-body executor (arms + head); init() refuses nothing on its failure, ENTER_BALANCE does
  UpperBodyExecutor exec_;
  bool exec_ok_ = false;
  // session 5: the legs + waist executor of ON_STAND (r1_lower_executor_config; records of the leg / waist groups)
  UpperBodyExecutor exec_lower_;
  bool exec_lower_ok_ = false;
  double on_stand_kp_[R1RT_NUM_JOINTS] = {};  // ON_STAND hold gains (legs + waist from the config, arms + head upper_*)
  double on_stand_kd_[R1RT_NUM_JOINTS] = {};
  std::atomic<std::uint8_t> takeover_target_{R1RT_MODE_BALANCE};  // where TAKEOVER_* ends (the policy thread reads)
  r1rt_heartbeat guard_hb_{};              // the last good read of r1guard's heartbeat (GUARD_ABSENT)
  bool have_guard_hb_ = false;
  double last_waist_[2] = {};               // the committed runner's waist state (the guard mirror; Manip3)
  double last_waist_target_[2] = {};
  r1rt_hands_contact hands_last_{};        // the last good read of the hands_contact seqlock (seq != 0)
  r1rt_hands_pose hands_pose_{};           // session 5: the last good read of the hands_pose seqlock
  bool have_hands_pose_ = false;
  bool have_hands_ = false;
  std::uint8_t tick_event_ = 0;            // the executor's first event this tick (telemetry guard_event)
  std::uint32_t tick_exec_clamps_ = 0;

  std::int64_t freeze_t0_ = 0;
  bool freeze_invalid_ = false;  // entered because no valid policy (vs the operator's request)
  bool freeze_from_stand_ = false;  // session 6: the operator's freeze from ON_STAND (S10): never into BALANCE
  bool stand_session_ = false;      // session 6: r1rt --stand-session - no BALANCE for the process's lifetime
  std::uint8_t motor_err_groups_ = 0;  // session 6: upper groups held for a motor error word (bit = r1rt_group)
  void motor_error_holds(std::int64_t t_ns) noexcept;  // session 6: DESIGN §4.2 - arm / head motor error -> HOLD
  double freeze_q0_[R1RT_NUM_JOINTS] = {};
  double freeze_kp0_[R1RT_NUM_JOINTS] = {};
  double freeze_kd0_[R1RT_NUM_JOINTS] = {};
  // session 10 (stale): the freeze's stand target (legs + waist) = stand_q with both ankle pitches shifted for the
  // current arms (r1rt/stand_pose.hpp); = stand_q for a hanging robot (ON_STAND) and when the shift cannot be computed
  double freeze_stand_q_[R1RT_NUM_JOINTS] = {};
  double freeze_ankle_shift_ = 0.0;

  std::atomic<bool> stop_active_{false};  // read by the policy thread (base command zero while stopped)
  std::uint64_t stop_seen_[R1RT_NUM_STOP_LEVELS] = {};
  bool rt_granted_ = false;               // the process's real-time grants (lockstep: 0)
  std::int32_t wake_lateness_ns_ = 0;      // this tick's, from the process (set_loop_timing)
  std::int32_t prev_iteration_ns_ = 0;
  std::uint64_t claim_seen_seq_ = 0;     // the mailbox's WRITING claim first seen at claim_seen_ns_ (abandon timer)
  std::int64_t claim_seen_ns_ = 0;

  Injection inj_[kMaxInjections] = {};
  // session 6: published with release / read with acquire - r1rt's --fault-policy-nan injects from the writer thread
  // at BALANCE entry while the policy thread runs (an entry is complete before the count covers it; never modified)
  std::atomic<int> n_inj_{0};

  // session 8 (HW1 fix #29): the resume after a stop (a writer silence > resume_gap_ns). resume_wait_: a stop was seen,
  // fresh state awaited (ON_STAND / STAND_FREEZE / the sequence into ON_STAND publish nothing meanwhile: the old
  // command would be the snap); resume_active_: the blend from resume_q0_ (the measured pose) to the mode's command.
  void begin_resume(std::int64_t t_ns, std::int64_t gap_ns) noexcept;
  void start_resume_blend(std::int64_t t_ns) noexcept;
  bool resume_wait_ = false;
  bool resume_active_ = false;
  bool resume_drain_counted_ = false;  // session 9: this stop's wait already counted in resume_drain_waits
  std::int64_t resume_wait_t0_ = 0;
  std::int64_t resume_t0_ = 0;
  double resume_q0_[R1RT_NUM_JOINTS] = {};
  // session 8 follow-up: the pre-stop hold offset (the last command - the pose measured at the last tick before the
  // stop): the blend starts at the measured pose + it, so the holding torque is continuous at the resume
  double pre_stop_q_[R1RT_NUM_JOINTS] = {};
  double resume_off_[R1RT_NUM_JOINTS] = {};
  ResumeClock rclock_{};  // is the lowstate live after a stop (its board tick vs the tick extrapolated from before)

  std::int64_t last_tick_ns_ = 0;
  bool have_tick_ = false;
  std::uint64_t ticks_ = 0;
  std::uint64_t hb_seq_ = 0;
  r1rt_core_status st_{};

  // session 9 (U19 / D25): the state-age limit + the "hold the stand" rung (docs/sim/STATE_AGE.md)
  ContentAgeClock age_clock_{};      // the board-tick content age of the newest sample
  std::int64_t old_since_ns_ = 0;    // the state has been older than state_age_limit_ns since (0 = it is not)
  bool old_counted_ = false;         // this episode already counted as sustained
  std::int64_t live_since_ns_ = 0;   // stale_rung 2: the state live again since (0 = not live)
  std::int32_t balance_refusal_ = 0;  // session 11: why the last ENTER_BALANCE was refused (R1RT_REFUSE_*)
  bool stale_freeze_ = false;        // the current STAND_FREEZE was entered by the stale rung
  std::int64_t trig_age_ns_ = 0;     // the age the limit compares this tick
};

// The Core behind a C handle (C++ callers only: tests set the commit hook through it). Defined in step_api.cpp.
Core* core_of(r1rt_core* c) noexcept;

}  // namespace r1rt
