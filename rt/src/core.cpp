// r1rt::Core (core.hpp): the writer step, the policy thread's iteration, requests, status. DESIGN.md §3.1-§3.2,
// §4.2-§4.3; the safety numbers (10 ms / 100 ms / 1 s, 40 ms, 30 deg) come in through r1rt_config.
#include "r1rt/core.hpp"
#include "r1rt/arm_model.hpp"
#include "r1rt/stand_pose.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace r1rt {

namespace {

constexpr int kUpperFirst = 14;  // arms 14-23, head 24-25 (SDK order)
// session 6: a built-in leg / waist slot below this kp counts as soft (zero torque / damping) for ON_STAND's snapshot
constexpr double kSoftKp = 1.0;
constexpr std::uint32_t kCoreRequester = R1RT_OWNER_R1RT;  // stop-channel requester id of the core's own stops
constexpr int kMaxUpperRecordsPerTick = 16;  // ring drain bound per writer tick (constant time; the rest next tick)
constexpr std::int64_t kNever = std::numeric_limits<std::int64_t>::max();
constexpr int kHandbackBeats = 2;  // policy-thread iterations seen alive before the writer hands back
constexpr std::int64_t kHandsPoseMaxAgeNs = 100'000'000;  // r1hands at 100 Hz: 10 periods (as hands_max_age_ns)

void base_fill(r1rt_lowcmd_out* out) noexcept {
  std::memset(out, 0, sizeof(*out));
  out->mode_pr = R1RT_MODE_PR;
  out->mode_machine = R1RT_MODE_MACHINE_R1;
}

bool finite_array(const double* a, int n) noexcept {
  for (int i = 0; i < n; ++i)
    if (!std::isfinite(a[i])) return false;
  return true;
}

double clampd(double x, double lo, double hi) noexcept { return x < lo ? lo : (x > hi ? hi : x); }

// Faults that raise the takeover alarm while in MIRROR / BLEND (DESIGN §3.5: they change nothing that is published).
constexpr std::uint32_t kTakeoverFaults = R1RT_FAULT_LOWSTATE_STALE | R1RT_FAULT_LOWSTATE_LOST | R1RT_FAULT_CRC |
                                          R1RT_FAULT_TILT_FALL | R1RT_FAULT_MOTOR_ERROR | R1RT_FAULT_POLICY_LATE |
                                          R1RT_FAULT_POLICY_INVALID;

// v2's takeover gates and blend (r1mc2/runtime.py:335-412, server.py:136-137), the defaults of the ABI 4 config fields.
// The blend: v2's `takeover` default (into=stand: 1.0 s; into=walk: 0.3 s). 1.0 s also because rung 3 needs it: in the
// simulator an invalid policy action in the policy's first ~0.3 s after a takeover froze into a fall with a 0.3 s or
// an instant switch and stayed up with 1.0 s (tests/integration/test_v3_core_takeover.py, S4 notes) - the snapshot's
// share keeps the robot near the stand the freeze returns to.
constexpr double kDefaultBlendS = 1.0;
constexpr double kDefaultTiltMaxRad = 12.0 * M_PI / 180.0;
constexpr double kDefaultTakeoverDqMax = 0.5;
constexpr double kDefaultTempMaxC = 60.0;
constexpr std::int64_t kDefaultStateMaxAgeNs = 50'000'000;
constexpr std::int64_t kDefaultBuiltinMaxAgeNs = 100'000'000;
constexpr double kDefaultBuiltinMinLegKp = 10.0;
constexpr double kDefaultMirrorMaxDevRad = 0.6;
constexpr std::int64_t kDefaultMirrorAlarmNs = 5'000'000'000;
constexpr std::int64_t kDefaultGuardMaxAgeNs = 50'000'000;
constexpr double kResumeOffsetMaxRad = 0.15;  // session 8: the pre-stop hold offset carried into a resume, at most
constexpr int kLegs = 12;

void copy_command(r1rt_lowcmd_out* out, const r1rt_lowcmd_out& src) noexcept {
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    out->q[j] = src.q[j];
    out->dq[j] = src.dq[j];
    out->kp[j] = src.kp[j];
    out->kd[j] = src.kd[j];
    out->tau_ff[j] = src.tau_ff[j];
  }
}

}  // namespace

int32_t Core::init(const r1rt_config& cfg, char* err, std::size_t errlen) noexcept {
  auto fail = [&](int32_t code, const char* msg) {
    if (err != nullptr && errlen > 0) std::snprintf(err, errlen, "%s", msg);
    return code;
  };
  cfg_ = cfg;
  cfg_.policy_weights_path = nullptr;  // read here only
  seg_ = static_cast<shm::Segment*>(cfg.segment);
  period_ns_ = 1'000'000'000LL / cfg.writer_rate_hz;
  policy_period_ns_ = 1'000'000'000LL / cfg.policy_rate_hz;
  if (cfg_.loop_gap_ns <= 0) cfg_.loop_gap_ns = 2 * period_ns_;
  if (!(cfg_.stand_ramp_s > 0.0)) cfg_.stand_ramp_s = 0.4;
  if (!finite_array(cfg.stand_q, R1RT_NUM_JOINTS) || !finite_array(cfg.stand_kp, R1RT_NUM_JOINTS) ||
      !finite_array(cfg.stand_kd, R1RT_NUM_JOINTS) || !finite_array(cfg.upper_kp, R1RT_NUM_JOINTS) ||
      !finite_array(cfg.upper_kd, R1RT_NUM_JOINTS))
    return fail(R1RT_E_INVALID_ARG, "config: non-finite stand / upper values");
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) freeze_stand_q_[j] = cfg_.stand_q[j];  // session 10: until a freeze
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    if (!(cfg.q_min[j] < cfg.q_max[j])) return fail(R1RT_E_INVALID_ARG, "config: q_min >= q_max");
    if (!(cfg.kp_max[j] >= 0.0 && cfg.kd_max[j] >= 0.0 && cfg.tau_max[j] >= 0.0 && cfg.dq_max[j] > 0.0))
      return fail(R1RT_E_INVALID_ARG, "config: limits");
  }
  if (!(cfg_.takeover_blend_s > 0.0)) cfg_.takeover_blend_s = kDefaultBlendS;
  if (!(cfg_.takeover_tilt_max_rad > 0.0)) cfg_.takeover_tilt_max_rad = kDefaultTiltMaxRad;
  if (!(cfg_.takeover_dq_max > 0.0)) cfg_.takeover_dq_max = kDefaultTakeoverDqMax;
  if (!(cfg_.takeover_temp_max_c > 0.0)) cfg_.takeover_temp_max_c = kDefaultTempMaxC;
  if (cfg_.takeover_state_max_age_ns <= 0) cfg_.takeover_state_max_age_ns = kDefaultStateMaxAgeNs;
  if (cfg_.builtin_max_age_ns <= 0) cfg_.builtin_max_age_ns = kDefaultBuiltinMaxAgeNs;
  if (!(cfg_.builtin_min_leg_kp > 0.0)) cfg_.builtin_min_leg_kp = kDefaultBuiltinMinLegKp;
  if (!(cfg_.mirror_max_dev_rad > 0.0)) cfg_.mirror_max_dev_rad = kDefaultMirrorMaxDevRad;
  if (cfg_.takeover_mirror_alarm_ns <= 0) cfg_.takeover_mirror_alarm_ns = kDefaultMirrorAlarmNs;
  if (cfg_.guard_max_age_ns <= 0) cfg_.guard_max_age_ns = kDefaultGuardMaxAgeNs;
  if (cfg_.resume_gap_ns <= 0) cfg_.resume_gap_ns = R1RT_DEFAULT_RESUME_GAP_NS;
  // session 9 (U19 / D25): the state-age limit (0 = the default); stale_rung > 2 is not a rung
  if (cfg_.state_age_limit_ns <= 0) cfg_.state_age_limit_ns = R1RT_DEFAULT_STATE_AGE_LIMIT_NS;
  if (cfg_.state_age_sustain_ns <= 0) cfg_.state_age_sustain_ns = R1RT_DEFAULT_STATE_AGE_SUSTAIN_NS;
  if (cfg_.stale_return_live_ns <= 0) cfg_.stale_return_live_ns = R1RT_DEFAULT_STALE_RETURN_LIVE_NS;
  if (!(cfg_.board_tick_hz > 0.0)) cfg_.board_tick_hz = R1RT_DEFAULT_BOARD_TICK_HZ;
  if (cfg_.stale_rung > 2) return fail(R1RT_E_INVALID_ARG, "config: stale_rung must be 0 (off), 1 or 2");
  age_clock_.configure(cfg_.board_tick_hz, cfg_.state_age_limit_ns / 2);
  st_.state_content_age_ns = -1;
  blend_ns_ = static_cast<std::int64_t>(std::llround(cfg_.takeover_blend_s * 1e9));
  rt_granted_ = cfg.rt_granted != 0;
  if (cfg.policy_weights_path != nullptr) {
    if (!net_.load(cfg.policy_weights_path, err, errlen)) return R1RT_E_POLICY_LOAD;
    if (!thread_runner_.init(cfg.policy, &net_, err, errlen)) return R1RT_E_POLICY_LOAD;
    if (!writer_runner_.init(cfg.policy, &net_, err, errlen)) return R1RT_E_POLICY_LOAD;
    thread_filter_ = FilterCtx{this, 0};
    writer_filter_ = FilterCtx{this, 0};
    thread_runner_.set_action_filter(&Core::filter_thunk, &thread_filter_);
    writer_runner_.set_action_filter(&Core::filter_thunk, &writer_filter_);
    policy_loaded_ = true;
  }
  // The upper-body executor (DESIGN §3.3): v2's R1 values (r1_default_executor_config, with their sources) with this
  // config's arm / head HOLD gains, limits intersected with the core's, the stop time and the arm tilt stop.
  ExecutorConfig ec = r1_default_executor_config();
  for (int j = kUpperFirst; j < R1RT_NUM_JOINTS; ++j) {
    ec.hold_kp[j] = cfg.upper_kp[j];
    ec.hold_kd[j] = cfg.upper_kd[j];
    ec.q_min[j] = cfg.q_min[j] > ec.q_min[j] ? cfg.q_min[j] : ec.q_min[j];
    ec.q_max[j] = cfg.q_max[j] < ec.q_max[j] ? cfg.q_max[j] : ec.q_max[j];
    ec.dq_max[j] = cfg.dq_max[j] < ec.dq_max[j] ? cfg.dq_max[j] : ec.dq_max[j];
    const double kp_hi = ec.kp_max[j] > cfg.upper_kp[j] ? ec.kp_max[j] : cfg.upper_kp[j];
    const double kd_hi = ec.kd_max[j] > cfg.upper_kd[j] ? ec.kd_max[j] : cfg.upper_kd[j];
    ec.kp_max[j] = kp_hi < cfg.kp_max[j] ? kp_hi : cfg.kp_max[j];
    ec.kd_max[j] = kd_hi < cfg.kd_max[j] ? kd_hi : cfg.kd_max[j];
  }
  if (cfg.stop_time_s > 0.0) ec.stop_time_s = cfg.stop_time_s;
  if (cfg.arm_tilt_stop_rad > 0.0) ec.arm_tilt_stop_rad = cfg.arm_tilt_stop_rad;
  ec.tick_ns = period_ns_;
  exec_ok_ = exec_.init(ec) == R1RT_OK;  // allocates its knot buffers (here, before the RT loop, only)
  // Session 5, ON_STAND (DESIGN §3.5): legs + waist hold at on_stand_kp / kd (all 0 in the config = the stand gains),
  // arms + head at the upper HOLD gains; the lower executor runs the legs / waist groups' L0 records on the stand.
  bool on_stand_given = false;
  for (int j = 0; j < kUpperFirst; ++j) on_stand_given = on_stand_given || cfg.on_stand_kp[j] != 0.0;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    const bool lower = j < kUpperFirst;
    on_stand_kp_[j] = lower ? (on_stand_given ? cfg.on_stand_kp[j] : cfg.stand_kp[j]) : cfg.upper_kp[j];
    on_stand_kd_[j] = lower ? (on_stand_given ? cfg.on_stand_kd[j] : cfg.stand_kd[j]) : cfg.upper_kd[j];
    if (!std::isfinite(on_stand_kp_[j]) || !std::isfinite(on_stand_kd_[j]))
      return fail(R1RT_E_INVALID_ARG, "config: non-finite on_stand gains");
  }
  ExecutorConfig el = r1_lower_executor_config(on_stand_kp_, on_stand_kd_);
  for (int j = 0; j < kUpperFirst; ++j) {
    el.q_min[j] = cfg.q_min[j] > el.q_min[j] ? cfg.q_min[j] : el.q_min[j];
    el.q_max[j] = cfg.q_max[j] < el.q_max[j] ? cfg.q_max[j] : el.q_max[j];
    el.dq_max[j] = cfg.dq_max[j] < el.dq_max[j] ? cfg.dq_max[j] : el.dq_max[j];
    el.kp_max[j] = el.kp_max[j] < cfg.kp_max[j] ? el.kp_max[j] : cfg.kp_max[j];
    el.kd_max[j] = el.kd_max[j] < cfg.kd_max[j] ? el.kd_max[j] : cfg.kd_max[j];
    if (el.hold_kp[j] > el.kp_max[j]) el.kp_max[j] = el.hold_kp[j];  // the hold is never refused (core limits clamp)
    if (el.hold_kd[j] > el.kd_max[j]) el.kd_max[j] = el.hold_kd[j];
  }
  if (cfg.stop_time_s > 0.0) el.stop_time_s = cfg.stop_time_s;
  el.tick_ns = period_ns_;
  exec_lower_ok_ = exec_lower_.init(el) == R1RT_OK;  // ON_STAND is refused when this failed (e.g. zero hold gains)
  base_fill(&last_cmd_);
  base_fill(&balanced_cmd_);
  base_fill(&mirror_cmd_);
  st_.policy_loaded = policy_loaded_ ? 1 : 0;
  st_.exec_ready = exec_ok_ ? 1 : 0;
  return R1RT_OK;
}

void Core::filter_thunk(void* ctx, double* action, int n) noexcept {
  // ctx = the computing runner's FilterCtx (its own time: the thread and the writer never share it). inj_ entries are
  // written by inject() and published by n_inj_ (release; session 6: r1rt injects from the writer thread at BALANCE).
  const FilterCtx* fc = static_cast<const FilterCtx*>(ctx);
  const Core* c = fc->core;
  const int n_inj = c->n_inj_.load(std::memory_order_acquire);
  for (int k = 0; k < n_inj; ++k) {
    const Injection& in = c->inj_[k];
    if (fc->t_ns < in.t0 || fc->t_ns >= in.t1) continue;
    for (int i = 0; i < n; ++i) {
      if (in.kind == R1RT_INJECT_POLICY_NAN) action[i] = std::numeric_limits<double>::quiet_NaN();
      if (in.kind == R1RT_INJECT_POLICY_OUT_OF_RANGE) action[i] = in.value;
    }
  }
}

bool Core::fresh(std::int64_t t_ns) const noexcept {
  return have_state_ && t_ns - latest_.t_rx_ns <= cfg_.lowstate_stale_ns;
}

void Core::current_cmd(std::int64_t t_ns, double cmd[3], double* lean) const noexcept {
  cmd[0] = cmd[1] = cmd[2] = 0.0;
  *lean = 0.0;
  r1rt_base_cmd b{};
  if (stop_active_.load(std::memory_order_acquire) || !seg_->base_cmd.try_read(b) || b.seq == 0) return;
  if (t_ns - b.t_issue_ns > b.ttl_ns) return;  // expired: zero (types.h: TTL <= 0.5 s, expiry = zero)
  if (!std::isfinite(b.vx) || !std::isfinite(b.vy) || !std::isfinite(b.wz) || !std::isfinite(b.lean)) return;
  cmd[0] = b.vx;
  cmd[1] = b.vy;
  cmd[2] = b.wz;
  *lean = b.lean;
}

int Core::run_policy(std::int64_t t_ns, int source, const r1rt_lowstate_in& in) noexcept {
  // The hand-back protocol (core.hpp "Threads"): read the committed slot (generation g), compute on this source's
  // own runner restored from it, then commit iff the generation is still g (CAS g -> g + 1; the slot write follows the
  // CAS, the policy-ring push sits between the two, so pushes and slot writes are totally ordered across the threads).
  PolicyRunner& r = source == kSourceThread ? thread_runner_ : writer_runner_;
  FilterCtx& fc = source == kSourceThread ? thread_filter_ : writer_filter_;
  Committed c{};
  if (!committed_.try_read(c)) return kPolicyBusy;
  if (c.has_state != 0 && t_ns < c.mb.t_action_ns) {
    // our clock is older than the committed result (the thread took its time, then the writer committed before it
    // read the slot): computing would run the phase clock backwards - drop it like a lost race
    n_discarded_.fetch_add(1, std::memory_order_relaxed);
    return kPolicyDiscarded;
  }
  if (c.has_state != 0) r.restore(c.rs);
  if (source == kSourceEntry || c.has_state == 0) r.reset();  // a fresh start (reset keeps the lean / waist targets)
  const r1rt_policy_params& p = r.params();
  const int nj = static_cast<int>(p.n_joints);
  double q[R1RT_POLICY_MAX_JOINTS];
  double dq[R1RT_POLICY_MAX_JOINTS];
  for (int i = 0; i < nj; ++i) {
    q[i] = in.q[p.joint_index[i]];
    dq[i] = in.dq[p.joint_index[i]];
  }
  double cmd[3];
  double lean = 0.0;
  current_cmd(t_ns, cmd, &lean);
  r.set_lean_target(lean);  // refused (kept) when outside the policy's range
  fc.t_ns = t_ns;
  r.advance_clock(static_cast<double>(t_ns) * 1e-9);
  const bool ok = r.act(q, dq, in.imu_quat, in.imu_gyro, cmd, nullptr, cfg_.action_abs_max);
  if (commit_hook_ != nullptr) commit_hook_(commit_hook_ctx_, source);  // tests: a stall exactly here
  Committed n{};
  r.save(n.rs);
  n.has_state = 1;
  n.gen = c.gen + 1;
  for (int k = 0; k < r.n_act(); ++k) {
    n.mb.target[k] = r.target()[k];
    n.mb.raw[k] = r.last_raw()[k];
  }
  n.mb.t_action_ns = t_ns;
  n.mb.seq = n.gen;
  n.mb.gait_phase = r.gait_phase();
  n.mb.lean = r.lean();
  n.mb.valid = ok ? 1 : 0;
  n.mb.inline_run = source == kSourceThread ? 0 : 1;
  std::uint64_t expect = c.gen;
  if (!gen_.compare_exchange_strong(expect, n.gen, std::memory_order_acq_rel, std::memory_order_acquire)) {
    n_discarded_.fetch_add(1, std::memory_order_relaxed);  // computed on a replaced state: never used
    return kPolicyDiscarded;
  }
  r1rt_policy_record rec{};
  rec.seq = n.gen;
  rec.t_obs_ns = in.t_rx_ns;
  rec.t_action_ns = t_ns;
  rec.n_obs = static_cast<std::uint32_t>(r.n_obs());
  rec.inline_run = n.mb.inline_run;
  // session 13: the observation the network saw (types.h / LOGGER.md "observation, action") - never filled before, so
  // HW2 day 2's r1log could not show what the policy's lean input was on the robot (F12)
  const double* ob = r.last_obs();
  for (int k = 0; k < r.n_obs() && k < R1RT_MAX_POLICY_OBS; ++k) rec.obs[k] = static_cast<float>(ob[k]);
  for (int k = 0; k < r.n_act(); ++k) rec.action[k] = static_cast<float>(n.mb.raw[k]);
  seg_->policy.try_push(rec);
  committed_.write(n);
  n_commits_.fetch_add(1, std::memory_order_relaxed);
  if (!ok) n_invalid_.fetch_add(1, std::memory_order_relaxed);
  if (source == kSourceInline) st_.inline_runs += 1;  // the writer's own counter
  if (source == kSourceThread) n_thread_runs_.fetch_add(1, std::memory_order_relaxed);
  last_action_t_ns_.store(t_ns, std::memory_order_relaxed);
  return ok ? kPolicyValid : kPolicyInvalid;
}

void Core::read_mailbox(Mailbox& mb) noexcept {
  Committed c{};
  if (committed_.try_read(c)) {
    last_mb_ = c.mb;
    if (c.has_state != 0) {  // the runner's waist state, for the guard mirror (session 5)
      for (int i = 0; i < 2; ++i) {
        last_waist_[i] = c.rs.waist[i];
        last_waist_target_[i] = c.rs.waist_target[i];
      }
    }
  }
  mb = last_mb_;
}

void Core::enter_freeze(std::int64_t t_ns, bool operator_request) noexcept {
  // Start the ramp from what was last commanded (legs + waist), so the freeze is continuous with the last command.
  const bool have_cmd = last_cmd_.publish != 0;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    freeze_q0_[j] = have_cmd ? last_cmd_.q[j] : latest_.q[j];
    freeze_kp0_[j] = have_cmd ? last_cmd_.kp[j] : cfg_.stand_kp[j];
    freeze_kd0_[j] = have_cmd ? last_cmd_.kd[j] : cfg_.stand_kd[j];
    if (!std::isfinite(freeze_q0_[j])) freeze_q0_[j] = latest_.q[j];
  }
  // session 10 (stale; STATE_AGE.md "Session 10: the freeze pose"): the stand the legs ramp to keeps the whole-body
  // CoM where the configured stand has it for the arms / head as they are now (they only decelerate to HOLD): one
  // ankle-pitch shift for both ankles (table_ready arms: +0.0105 rad; the frozen robot 2.4-2.5 deg instead of 5.7-6.5).
  // Only on a robot standing on its feet: a hanging robot (ON_STAND, or a takeover into ON_STAND) keeps stand_q as
  // HW1 ran it. No shift computable (bad input, beyond kMaxFreezeAnkleShift) = stand_q, the behaviour before.
  constexpr double kMaxFreezeAnkleShift = 0.08;  // rad; arms straight forward (shoulder pitch -0.6) need 0.047
  const r1rt_mode m_prev = mode_.load(std::memory_order_relaxed);
  const bool hanging = m_prev == R1RT_MODE_ON_STAND ||
                       ((m_prev == R1RT_MODE_TAKEOVER_MIRROR || m_prev == R1RT_MODE_TAKEOVER_BLEND) &&
                        takeover_target_.load(std::memory_order_relaxed) == R1RT_MODE_ON_STAND);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) freeze_stand_q_[j] = cfg_.stand_q[j];
  freeze_ankle_shift_ = 0.0;
  if (!hanging) {
    double upper[12];
    for (int k = 0; k < 12; ++k) upper[k] = freeze_q0_[stand::kUpper0 + k];
    freeze_ankle_shift_ = stand::ankle_shift_for_arms(cfg_.stand_q, upper, kMaxFreezeAnkleShift);
    freeze_stand_q_[stand::kAnkleLeft] += freeze_ankle_shift_;
    freeze_stand_q_[stand::kAnkleRight] += freeze_ankle_shift_;
  }
  freeze_t0_ = t_ns;
  freeze_invalid_ = !operator_request;
  mode_.store(R1RT_MODE_STAND_FREEZE, std::memory_order_release);
  inline_active_.store(false, std::memory_order_release);
  st_.freeze_entries += 1;
  internal_stop(t_ns);  // rung 3: the arms and the head decelerate to HOLD too
}

void Core::motor_error_holds(std::int64_t t_ns) noexcept {
  // DESIGN §4.2: "Arm/head motor: HOLD that limb." A nonzero error word on any joint of an upper group (arm_left
  // 14-18, arm_right 19-23, head 24-25) puts that group in HOLD once (decelerating at the stop time, at its hold gains -
  // stiff, never limp) and refuses its new records until the operator's RESUME with the word cleared. Legs / waist: no
  // action (the meaning of the bits is open, RES §8 Q10; the operator decides). In every mode the executor runs in.
  static constexpr int kFirst[R1RT_NUM_UPPER_GROUPS] = {14, 19, 24};
  static constexpr int kN[R1RT_NUM_UPPER_GROUPS] = {5, 5, 2};
  if (!exec_ok_) return;
  for (int g = 0; g < R1RT_NUM_UPPER_GROUPS; ++g) {
    bool bad = false;
    for (int k = 0; k < kN[g]; ++k) bad = bad || latest_.motor_state[kFirst[g] + k] != 0;
    const std::uint8_t bit = static_cast<std::uint8_t>(1u << g);
    if (bad && (motor_err_groups_ & bit) == 0) {
      motor_err_groups_ |= bit;
      exec_.stop_group(static_cast<r1rt_group>(g), t_ns, kEventMotorError);
      exec_.set_group_held(static_cast<r1rt_group>(g), true);
      st_.motor_error_holds += 1;
    }
  }
  st_.motor_error_groups = motor_err_groups_;
}

void Core::internal_stop(std::int64_t t_ns) noexcept {
  // Posted on the stop channel (the core as a client of its own channel), so the epoch grows exactly as for an
  // operator stop: r1d sees it and stamps its next records with the new epoch, and the executor refuses records issued
  // before the stop. A direct exec_.stop() would advance the executor's epoch past the channel's and every later
  // record would be refused. The channel's stop level also zeroes the base command (stop_active_, as REQ_STOP).
  seg_->stop.request(R1RT_STOP_STOP, kCoreRequester, t_ns);
  stop_active_.store(true, std::memory_order_release);
  exec_.stop_at_epoch(seg_->stop.epoch(), t_ns);
}

bool Core::lower_active() const noexcept {
  const r1rt_mode m = mode_.load(std::memory_order_relaxed);
  return exec_lower_ok_ && (m == R1RT_MODE_ON_STAND ||
                            (m == R1RT_MODE_TAKEOVER_BLEND &&
                             takeover_target_.load(std::memory_order_relaxed) == R1RT_MODE_ON_STAND));
}

void Core::upper_step(r1rt_lowcmd_out* out, std::int64_t t_ns) noexcept {
  // DESIGN §3.1 step 7 / EXECUTOR.md "wiring": the stop epoch, r1d's records (bounded drain), the keepalives, the
  // hands' contact summary (the last good copy; the executor judges its age), then one executor tick writing joints
  // 14-25 of the command. The core's hard limits (step 8) still apply to its output. Session 5: records of the leg /
  // waist groups go to the lower executor - accepted only while it publishes (ON_STAND and its blend), refused and
  // counted otherwise (the policy owns the legs in BALANCE); in ON_STAND it writes joints 0-13 after the upper body.
  const std::uint64_t epoch = seg_->stop.epoch();
  exec_.stop_at_epoch(epoch, t_ns);
  const bool lower = lower_active();
  if (exec_lower_ok_) exec_lower_.stop_at_epoch(epoch, t_ns);
  r1rt_upper_cmd rec;
  for (int k = 0; k < kMaxUpperRecordsPerTick && seg_->upper_cmd.try_pop(rec); ++k) {
    if (exec_lower_ok_ && exec_lower_.owns_group(rec.group)) {
      if (lower && exec_lower_.submit(rec, t_ns) == R1RT_OK) {
        st_.lower_accepted += 1;
      } else {
        st_.lower_rejected += 1;
        st_.upper_last_reject = lower ? exec_lower_.last_reject() : static_cast<std::uint8_t>(kRejectGroup);
      }
      continue;
    }
    if (exec_.submit(rec, t_ns) == R1RT_OK) {
      st_.upper_accepted += 1;
    } else {
      st_.upper_rejected += 1;
      st_.upper_last_reject = exec_.last_reject();
    }
  }
  for (int g = 0; g < R1RT_NUM_CMD_GROUPS; ++g) {
    const std::uint64_t id = seg_->keepalive.cmd_id[g].load(std::memory_order_acquire);
    if (id == 0) continue;
    const std::int64_t ka = seg_->keepalive.t_ns[g].load(std::memory_order_relaxed);
    if (g < R1RT_NUM_UPPER_GROUPS) {
      exec_.keepalive(id, ka);
    } else if (lower) {
      exec_lower_.keepalive(id, ka);
    }
  }
  r1rt_hands_contact hc{};
  if (seg_->hands_contact.try_read(hc) && hc.seq != 0) {
    hands_last_ = hc;
    have_hands_ = true;
  }
  // DESIGN §3.4: the built-in arm model's gravity torque for the measured pose (ARM_MODEL.md); without it the
  // feed-forward goes stale after gravity_ff_max_age_ns and the executor adds nothing (FREEDRIVE ends)
  // session 5: with r1hands' pose + declared payloads (the hands_pose seqlock; fresh within 100 ms) the model uses the
  // real finger joints and the payloads; without them the open hand and no payload (as before)
  r1rt_hands_pose hp{};
  if (seg_->hands_pose.try_read(hp) && hp.seq != 0) {
    hands_pose_ = hp;
    have_hands_pose_ = true;
  }
  const bool pose_ok = have_hands_pose_ && hands_pose_.valid != 0 && t_ns - hands_pose_.t_ns <= kHandsPoseMaxAgeNs &&
                       t_ns >= hands_pose_.t_ns;
  double g_tau[R1RT_NUM_JOINTS];
  if (arm::gravity_ff26(latest_.q, latest_.imu_quat, pose_ok ? hands_pose_.q[0] : nullptr,
                        pose_ok ? hands_pose_.q[1] : nullptr, pose_ok ? hands_pose_.payload_kg : nullptr, g_tau))
    exec_.set_gravity_ff(g_tau, t_ns);
  st_.hands_pose_used = pose_ok ? 1 : 0;
  exec_.step(t_ns, latest_.q, latest_.dq, latest_.tau_est, have_hands_ ? &hands_last_ : nullptr, tilt_, out->q,
             out->dq, out->kp, out->kd, out->tau_ff);
  tick_event_ = exec_.tick_event();
  tick_exec_clamps_ = exec_.tick_clamps();
  if (lower) {  // ON_STAND: legs + waist (joints 0-13) from their L0 groups (no hands, no tilt stop on the stand)
    exec_lower_.step(t_ns, latest_.q, latest_.dq, latest_.tau_est, nullptr, tilt_, out->q, out->dq, out->kp, out->kd,
                     out->tau_ff);
    if (tick_event_ == 0) tick_event_ = exec_lower_.tick_event();
    tick_exec_clamps_ += exec_lower_.tick_clamps();
  }
}

void Core::on_stand_command(r1rt_lowcmd_out* out, std::int64_t t_ns) noexcept {
  upper_step(out, t_ns);  // every joint: the lower executor (0-13) and the upper-body executor (14-25)
}

void Core::build_balance(r1rt_lowcmd_out* out, const Mailbox& mb) const noexcept {
  const r1rt_policy_params& p = writer_runner_.params();
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    out->q[j] = latest_.q[j];
    out->kp[j] = p.kp_other[j];
    out->kd[j] = p.kd_other[j];
  }
  const int* idx = writer_runner_.act_idx26();
  for (int k = 0; k < writer_runner_.n_act(); ++k) out->q[idx[k]] = mb.target[k];
  for (std::uint32_t i = 0; i < p.n_joints; ++i) {
    out->kp[p.joint_index[i]] = p.kp[i];
    out->kd[p.joint_index[i]] = p.kd[i];
  }
  // joints 14-25: the executor (upper_step), after this
}

void Core::build_freeze(r1rt_lowcmd_out* out, std::int64_t t_ns) const noexcept {
  const double s = clampd(static_cast<double>(t_ns - freeze_t0_) * 1e-9 / cfg_.stand_ramp_s, 0.0, 1.0);
  for (int j = 0; j < kUpperFirst; ++j) {
    out->q[j] = freeze_q0_[j] + s * (freeze_stand_q_[j] - freeze_q0_[j]);  // session 10: the arms' stand
    out->kp[j] = freeze_kp0_[j] + s * (cfg_.stand_kp[j] - freeze_kp0_[j]);
    out->kd[j] = freeze_kd0_[j] + s * (cfg_.stand_kd[j] - freeze_kd0_[j]);
  }
  // joints 14-25: the executor (upper_step: stopped on entry, then HOLD), after this
}

void Core::build_damp(r1rt_lowcmd_out* out) const noexcept {
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    out->q[j] = have_state_ ? latest_.q[j] : 0.0;
    out->kp[j] = 0.0;
    out->kd[j] = cfg_.damp_kd[j];
  }
}

std::uint32_t Core::apply_limits(r1rt_lowcmd_out* out, std::uint32_t* faults) noexcept {
  std::uint32_t n = 0;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    double v = clampd(out->q[j], cfg_.q_min[j], cfg_.q_max[j]);
    if (v != out->q[j]) {
      ++n;
      *faults |= R1RT_FAULT_JOINT_CLAMP;
      out->q[j] = v;
    }
    v = clampd(out->dq[j], -cfg_.dq_max[j], cfg_.dq_max[j]);
    if (v != out->dq[j]) {
      ++n;
      *faults |= R1RT_FAULT_VELOCITY_CLAMP;
      out->dq[j] = v;
    }
    v = clampd(out->tau_ff[j], -cfg_.tau_max[j], cfg_.tau_max[j]);
    if (v != out->tau_ff[j]) {
      ++n;
      *faults |= R1RT_FAULT_TORQUE_CLAMP;
      out->tau_ff[j] = v;
    }
    v = clampd(out->kp[j], 0.0, cfg_.kp_max[j]);
    if (v != out->kp[j]) {
      ++n;
      out->kp[j] = v;
    }
    v = clampd(out->kd[j], 0.0, cfg_.kd_max[j]);
    if (v != out->kd[j]) {
      ++n;
      out->kd[j] = v;
    }
  }
  st_.clamp_total += n;
  return n;
}

void Core::poll_stop_channel(std::int64_t t_ns) noexcept {
  for (int lvl = 0; lvl < R1RT_NUM_STOP_LEVELS; ++lvl) {
    shm::StopLevel& s = seg_->stop.level[lvl];
    const std::uint64_t req = s.requests.load(std::memory_order_acquire);
    if (req == stop_seen_[lvl]) continue;
    stop_seen_[lvl] = req;
    if (lvl == R1RT_STOP_DAMP) {
      // the operator's damp (DESIGN §4.2 stop channel) - OPERATOR ONLY (§3.5 / §5.1), as through the mailbox: until
      // session 6 any client's damp-level request limped the robot. last_requester is best effort (last writer wins:
      // two clients posting in the same tick leave one id), so a non-operator request racing the operator's can hide
      // it - the operator then repeats the word; a non-operator id is never acted on.
      if (s.last_requester.load(std::memory_order_relaxed) == R1RT_REQUESTER_OPERATOR) {
        request(R1RT_REQ_DAMP, t_ns);
      } else {
        st_.damp_ignored += 1;
      }
    } else {
      stop_active_.store(true, std::memory_order_release);  // stop / halt: base command zero, arms HOLD, keep balancing
    }
    s.acked.store(req, std::memory_order_release);
    s.acked_ns.store(t_ns, std::memory_order_relaxed);
  }
}

void Core::write_mode_reply(std::uint64_t seq, std::uint32_t req, std::uint32_t requester, std::int64_t t_req,
                            std::int64_t t_ns, std::int32_t status, std::int32_t refusal) noexcept {
  shm::ModeReply r{};
  r.seq = seq;
  r.t_req_ns = t_req;
  r.t_done_ns = t_ns;
  r.status = status;
  r.refusal = refusal;
  r.req = req;
  r.requester = requester;
  r.mode_after = mode_.load(std::memory_order_relaxed);
  seg_->mode_req.replies[seq % shm::kModeReplySlots].write(r);  // the core is the single writer of the replies
  st_.mode_last_seq = seq;
}

void Core::poll_mode_mailbox(std::int64_t t_ns) noexcept {
  // shm_layout.h ModeMailbox: READY -> act + reply + FREE; a WRITING claim older than kModeClaimTimeoutNs (by this
  // core's clock, from when it was first seen) -> abandoned (FREE + an ABANDONED reply). Never waits.
  using MB = shm::ModeMailbox;
  MB& mb = seg_->mode_req;
  std::uint64_t s = mb.state.load(std::memory_order_acquire);
  const std::uint64_t seq = MB::seq_of(s);
  const std::uint64_t phase = MB::phase_of(s);
  if (phase == MB::kWriting) {
    if (claim_seen_seq_ != seq) {
      claim_seen_seq_ = seq;
      claim_seen_ns_ = t_ns;
      return;
    }
    if (t_ns - claim_seen_ns_ < shm::kModeClaimTimeoutNs) return;
    if (mb.state.compare_exchange_strong(s, MB::pack(seq, MB::kFree), std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
      write_mode_reply(seq, 0, 0, 0, t_ns, R1RT_E_ABANDONED, R1RT_REFUSE_NONE);
      st_.mode_abandoned += 1;
    }
    return;
  }
  if (phase != MB::kReady) return;
  const std::uint32_t req = mb.req.load(std::memory_order_relaxed);
  const std::uint32_t requester = mb.requester.load(std::memory_order_relaxed);
  const std::int64_t t_req = mb.t_ns.load(std::memory_order_relaxed);
  std::int32_t status = R1RT_OK;
  std::int32_t refusal = R1RT_REFUSE_NONE;
  // session 6: the freeze on the stand (S10) is the operator's test too
  const bool stand_freeze =
      req == R1RT_REQ_STAND_FREEZE && mode_.load(std::memory_order_relaxed) == R1RT_MODE_ON_STAND;
  if ((req == R1RT_REQ_DAMP || req == R1RT_REQ_RECLAIM || stand_freeze) && requester != R1RT_REQUESTER_OPERATOR) {
    status = R1RT_E_REFUSED;  // DESIGN §5.1: damp only from an operator client; session 6: the hand-back too
    refusal = R1RT_REFUSE_NOT_OPERATOR;
    if (req == R1RT_REQ_RECLAIM) st_.reclaim_refusal = R1RT_REFUSE_NOT_OPERATOR;
    if (stand_freeze) st_.freeze_refusal = R1RT_REFUSE_NOT_OPERATOR;
  } else {
    status = request(static_cast<std::int32_t>(req), t_ns);
    if (req == R1RT_REQ_TAKEOVER) {
      refusal = st_.takeover_refusal;
    } else if (req == R1RT_REQ_ON_STAND) {
      refusal = st_.on_stand_refusal;
    } else if (req == R1RT_REQ_RECLAIM) {
      refusal = st_.reclaim_refusal;
    } else if (req == R1RT_REQ_ENTER_BALANCE && status != R1RT_OK && stand_session_) {
      refusal = R1RT_TAKEOVER_STAND_SESSION;
    } else if (req == R1RT_REQ_ENTER_BALANCE && status == R1RT_E_REFUSED) {
      refusal = balance_refusal_;  // session 11: MODE, or STATE (no fresh state / executor not ready / state too old)
    } else if ((req == R1RT_REQ_STAND_FREEZE || req == R1RT_REQ_STAND_HOLD) && status != R1RT_OK &&
               st_.freeze_refusal != R1RT_REFUSE_NONE) {
      refusal = st_.freeze_refusal;
    } else if (status == R1RT_E_INVALID_ARG) {
      refusal = R1RT_REFUSE_UNKNOWN;
    } else if (status != R1RT_OK) {
      refusal = R1RT_REFUSE_MODE;
    }
  }
  write_mode_reply(seq, req, requester, t_req, t_ns, status, refusal);
  st_.mode_requests += 1;
  mb.state.store(MB::pack(seq, MB::kFree), std::memory_order_release);
}

void Core::begin_resume(std::int64_t t_ns, std::int64_t gap_ns) noexcept {
  // Session 8 (HW1 fix #29; board-hold test: the board holds the last LowCmd ~400 ms, then a damped release): this
  // tick ends a writer silence longer than resume_gap_ns. Counted as a STOP (#31: apart from the scheduler's overruns).
  // Every executor motion in flight is aborted to HOLD (kEventResume: the time base jumped by the stop, a trajectory
  // would continue from a point its limb never reached); the resume blend starts once fresh state is in.
  st_.stops += 1;
  st_.stop_ns_total += gap_ns;
  if (gap_ns > st_.stop_ns_max) st_.stop_ns_max = gap_ns;
  const r1rt_mode m = mode_.load(std::memory_order_relaxed);
  const bool owner = own_token_ && seg_->token.holds(token_word_);
  if (!owner || m == R1RT_MODE_MONITOR || m == R1RT_MODE_DAMP || m == R1RT_MODE_TAKEOVER_MIRROR) {
    // nothing of ours holds a target: not the owner (r1guard took over during the stop and kept the board fed - the
    // operator's RECLAIM blends from exactly what it sent), MONITOR / DAMP, or MIRROR (the built-in's own command)
    resume_wait_ = false;
    resume_active_ = false;
    return;
  }
  if (exec_ok_) {
    for (int g = 0; g < R1RT_NUM_UPPER_GROUPS; ++g) exec_.stop_group(static_cast<r1rt_group>(g), t_ns, kEventResume);
  }
  if (exec_lower_ok_) {
    for (int g = R1RT_NUM_UPPER_GROUPS; g < R1RT_NUM_CMD_GROUPS; ++g)
      exec_lower_.stop_group(static_cast<r1rt_group>(g), t_ns, kEventResume);
  }
  // The pre-stop hold offset (session 8 follow-up, the integrator's HW1-stand rehearsal: a 100 / 300 ms pause - the board
  // HELD - gave a 3 N m hip step + a 0.013 rad dip when the blend started at the bare measured pose): the board was
  // applying kp (q_cmd - q) before the stop (~3 N m of gravity on a hanging hip at kp 200); starting the blend at the
  // measured pose + that offset keeps the holding torque continuous - no sag: the start IS the old target (nothing
  // moves); a sag d: the extra kp d builds over the blend. Bounded (kResumeOffsetMaxRad) against a non-steady lag.
  const bool have_cmd = last_cmd_.publish != 0;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    const double off = have_cmd ? last_cmd_.q[j] - pre_stop_q_[j] : 0.0;
    resume_off_[j] = std::isfinite(off) ? clampd(off, -kResumeOffsetMaxRad, kResumeOffsetMaxRad) : 0.0;
  }
  resume_wait_ = true;
  resume_active_ = false;
  resume_drain_counted_ = false;
  resume_wait_t0_ = t_ns;
}

void Core::start_resume_blend(std::int64_t t_ns) noexcept {
  resume_wait_ = false;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) resume_q0_[j] = latest_.q[j] + resume_off_[j];
  resume_t0_ = t_ns;
  resume_active_ = true;
  st_.resume_blends += 1;
  if (mode_.load(std::memory_order_relaxed) == R1RT_MODE_STAND_FREEZE) {
    // the freeze's ramp again, from the measured pose at the gains last sent (stand_ramp_s): its target is the stand
    const bool have_cmd = last_cmd_.publish != 0;
    for (int j = 0; j < kUpperFirst; ++j) {
      freeze_q0_[j] = resume_q0_[j];  // the measured pose + the pre-stop hold offset
      freeze_kp0_[j] = have_cmd ? last_cmd_.kp[j] : cfg_.stand_kp[j];
      freeze_kd0_[j] = have_cmd ? last_cmd_.kd[j] : cfg_.stand_kd[j];
    }
    freeze_t0_ = t_ns;
  }
}

int Core::balance_tick(std::int64_t t_ns, r1rt_lowcmd_out* out, std::uint32_t* faults) noexcept {
  if (*faults & R1RT_FAULT_LOWSTATE_STALE) {
    // > 100 ms without state: resend the last command built from fresh state (DESIGN §4.2); no policy on stale
    // state (the thread and the inline rung wait too).
    *out = balanced_cmd_;  // the arms too: the executor waits for state (its clock bounds the motion after)
    st_.stale_resends += 1;
    return kBalStale;
  }
  Mailbox mb{};
  read_mailbox(mb);
  if (inline_active_.load(std::memory_order_relaxed) &&
      thread_beats_.load(std::memory_order_acquire) - beats_at_inline_ >= static_cast<std::uint64_t>(kHandbackBeats)) {
    inline_active_.store(false, std::memory_order_release);  // the thread is back: it computes from its next iteration
  }
  if (mb.seq != 0 && mb.valid == 0) {
    *faults |= R1RT_FAULT_POLICY_INVALID;
    enter_freeze(t_ns);  // rung 3: an invalid action (no valid policy anywhere: the inline net is the same)
    build_freeze(out, t_ns);
    return kBalFroze;
  }
  bool inline_now = inline_active_.load(std::memory_order_relaxed);
  if (t_ns - mb.t_action_ns > cfg_.policy_max_age_ns || (inline_now && t_ns >= next_inline_ns_)) {
    *faults |= R1RT_FAULT_POLICY_LATE;
    if (!inline_now) {
      beats_at_inline_ = thread_beats_.load(std::memory_order_acquire);
      inline_active_.store(true, std::memory_order_release);
      st_.inline_entries += 1;
    }
    // rung 1: the policy inline in this thread from the committed runner state (bumpless), then every policy period;
    // if the thread committed first (a late result that made it), that result is used instead (discarded here)
    const int r = run_policy(t_ns, kSourceInline, latest_);
    next_inline_ns_ = t_ns + policy_period_ns_;
    read_mailbox(mb);
    if (r == kPolicyInvalid) {
      *faults |= R1RT_FAULT_POLICY_INVALID;
      enter_freeze(t_ns);
      build_freeze(out, t_ns);
      return kBalFroze;
    }
  } else if (inline_now) {
    *faults |= R1RT_FAULT_POLICY_LATE;  // still on the inline rung (between its runs)
  }
  build_balance(out, mb);
  return kBalOk;
}

bool Core::state_age_step(std::int64_t t_ns, std::int64_t rx_age, bool new_sample, std::uint32_t* faults) noexcept {
  // session 9 (U19 / D25): the age the limit judges = the receive age, or the board-tick content age when that clock
  // is valid and larger (a late receive path: fresh stamps, old poses - r1rt/state_age.hpp). Measured in every mode
  // (an alarm, counted); the rung (the caller) acts only in BALANCE and only when enabled.
  // session 11: a freeze never re-bases the board-tick clock (state_age.hpp hold_relearn): a state that stayed late
  // through STAND_FREEZE keeps reading late until it is really live again - the operator's `r1 balance` is judged on it
  age_clock_.hold_relearn(mode_.load(std::memory_order_relaxed) == R1RT_MODE_STAND_FREEZE);
  if (new_sample) age_clock_.sample(latest_.board_tick, latest_.t_rx_ns);
  const bool cvalid = have_state_ && age_clock_.valid();
  const std::int64_t content = cvalid ? age_clock_.age_ns(t_ns) : -1;
  st_.state_content_age_ns = content;
  st_.content_age_valid = cvalid ? 1 : 0;
  st_.content_age_relearns = age_clock_.relearns();
  if (cvalid && content > st_.state_content_age_max_ns) st_.state_content_age_max_ns = content;
  trig_age_ns_ = (cvalid && content > rx_age) ? content : rx_age;
  if (!have_state_ || trig_age_ns_ <= cfg_.state_age_limit_ns) {
    if (old_since_ns_ != 0 && old_counted_) {
      const std::int64_t len = t_ns - old_since_ns_;
      if (len > st_.state_old_ns_max) st_.state_old_ns_max = len;
    }
    old_since_ns_ = 0;
    old_counted_ = false;
    return false;
  }
  *faults |= R1RT_FAULT_STATE_OLD;
  st_.state_old_ticks += 1;
  if (old_since_ns_ == 0) old_since_ns_ = t_ns;
  const std::int64_t len = t_ns - old_since_ns_;
  const bool sustained = len >= cfg_.state_age_sustain_ns;
  if (sustained) {
    if (!old_counted_) st_.state_old_episodes += 1;
    old_counted_ = true;
    if (len > st_.state_old_ns_max) st_.state_old_ns_max = len;
  }
  return sustained && cfg_.stale_rung != 0;
}

void Core::stale_return_step(std::int64_t t_ns) noexcept {
  // the freeze ended some other way (the operator's ENTER_BALANCE / DAMP / a takeover): no longer the rung's
  if (mode_.load(std::memory_order_relaxed) != R1RT_MODE_STAND_FREEZE) {
    stale_freeze_ = false;
    live_since_ns_ = 0;
    return;
  }
  if (cfg_.stale_rung != 2) return;  // rung 1: the operator brings BALANCE back
  const bool live = have_state_ && trig_age_ns_ <= cfg_.state_age_limit_ns;
  if (!live) {
    live_since_ns_ = 0;
    return;
  }
  if (live_since_ns_ == 0) live_since_ns_ = t_ns;
  const bool ramp_done = static_cast<double>(t_ns - freeze_t0_) * 1e-9 >= cfg_.stand_ramp_s;
  if (!ramp_done || t_ns - live_since_ns_ < cfg_.stale_return_live_ns) return;
  // the operator's ENTER_BALANCE from STAND_FREEZE (fresh state, a valid first action from the reset runner)
  if (request(R1RT_REQ_ENTER_BALANCE, t_ns) == R1RT_OK) st_.stale_rung_returns += 1;
  // refused (e.g. an invalid first action: request() froze again, rung 3): no more automatic tries - the operator
  stale_freeze_ = false;
  live_since_ns_ = 0;
}

int32_t Core::step(const r1rt_lowstate_in* state, std::int64_t t_ns, r1rt_lowcmd_out* out,
                   r1rt_telemetry_record* tel) noexcept {
  base_fill(out);
  ticks_ += 1;
  std::uint32_t faults = 0;
  const std::int64_t gap = have_tick_ ? t_ns - last_tick_ns_ : 0;
  if (have_tick_ && gap > cfg_.loop_gap_ns) {
    faults |= R1RT_FAULT_LOOP_GAP;
    st_.loop_gaps += 1;
  }
  const bool stopped = have_tick_ && gap > cfg_.resume_gap_ns;  // session 8 (fix #29): this tick ends a stop
  have_tick_ = true;
  last_tick_ns_ = t_ns;

  // 2. the latest lowstate and its age
  const std::uint32_t prev_tick = latest_.board_tick;  // session 8: the last sample before this tick (a stop's start)
  const std::int64_t prev_rx = latest_.t_rx_ns;
  const bool had_state = have_state_;
  if (stopped && had_state)  // the pose at the last tick before the stop (this tick's sample may be after it)
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) pre_stop_q_[j] = latest_.q[j];
  if (state != nullptr && state->crc_ok) {
    latest_ = *state;
    have_state_ = true;
    if (!stopped && !resume_wait_) rclock_.live_sample(latest_.board_tick, latest_.t_rx_ns);
  } else if (state != nullptr) {
    faults |= R1RT_FAULT_CRC;
  }
  const std::int64_t age = have_state_ ? t_ns - latest_.t_rx_ns : kNever;
  if (age > cfg_.lowstate_late_ns) {
    faults |= R1RT_FAULT_LOWSTATE_LATE;
    st_.lowstate_late += 1;
  }
  if (age > cfg_.lowstate_stale_ns) faults |= R1RT_FAULT_LOWSTATE_STALE;
  if (age > cfg_.lowstate_lost_ns) faults |= R1RT_FAULT_LOWSTATE_LOST;
  // session 9 (U19 / D25): the content age (board tick) next to the receive age; STATE_OLD; sustained episodes
  const bool stale_act =
      state_age_step(t_ns, age, state != nullptr && state->crc_ok && (!had_state || latest_.t_rx_ns != prev_rx), &faults);

  // 2b. session 8 (HW1 fix #29): the resume after a stop. The blend's origin is the pose measured AFTER the stop (a
  //     lowstate younger than lowstate_late_ns: the receive thread may have been stopped too); until it comes (at most
  //     R1RT_RESUME_STATE_WAIT_NS) the modes that hold a fixed target publish nothing (their old command = the snap).
  if (stopped) {
    if (had_state) rclock_.mark_stop(prev_tick, prev_rx);
    begin_resume(t_ns, gap);
  }
  if (resume_wait_) {
    const bool fresh_now = have_state_ && age <= cfg_.lowstate_late_ns;
    // session 9 (#29's open race): the wait ends at the FIRST live sample. With a learned tick rate the board's own tick
    // proves a sample is current (ResumeClock), so no time floor: a pre-stop packet that slipped out inside the write
    // path (a stop between the stale check and the write: no user-space check can stop it) acts only until this tick's
    // command replaces it (~1-2 ms at the robot's 1052/s). Without a rate the 10 ms settle stays the only guard.
    const bool settled = rclock_.has_rate() || t_ns - resume_wait_t0_ >= R1RT_RESUME_SETTLE_NS;
    const bool caught = rclock_.caught_up(latest_.board_tick, t_ns);
    // session 9 (#29 hazard ii): a fresh sample whose tick is provably behind (a learned rate, not caught up) is the
    // stop's backlog still draining - its pose is from the stop's start; a blend from it is the snap. Keep waiting
    // (publishing nothing, as above) while such samples come, up to R1RT_RESUME_DRAIN_MAX_NS.
    const bool draining = fresh_now && rclock_.has_rate() && !caught &&
                          t_ns - resume_wait_t0_ <= R1RT_RESUME_DRAIN_MAX_NS;
    if (fresh_now && settled && caught) {
      start_resume_blend(t_ns);
    } else if (t_ns - resume_wait_t0_ > R1RT_RESUME_STATE_WAIT_NS) {
      if (draining) {
        if (!resume_drain_counted_) {
          st_.resume_drain_waits += 1;
          resume_drain_counted_ = true;
        }
      } else if (fresh_now) {  // fresh by its stamp, the tick never caught up within the drain limit: the latest pose
        start_resume_blend(t_ns);
        st_.resume_uncaught += 1;
      } else {
        resume_wait_ = false;  // no fresh state: the normal stale handling (the last command resent, rung 4)
        st_.resume_waits_expired += 1;
      }
    }
  }

  // 3-4. inputs and the estimate
  poll_stop_channel(t_ns);
  poll_mode_mailbox(t_ns);  // session 5: requests from r1d / the CLI / harnesses, acted on with this tick's state
  if (have_state_) {
    projected_gravity(latest_.imu_quat, gravity_);
    tilt_ = std::acos(clampd(-gravity_[2], -1.0, 1.0));
    if (tilt_ > cfg_.fall_tilt_rad) faults |= R1RT_FAULT_TILT_FALL;  // an event: the policy keeps running
    // motor error words: an alarm; arms / head -> that group HOLDs (stiff) and refuses records (session 6, below);
    // legs / waist: nothing that changes stiffness (the policy / the stand hold continues; the operator decides)
    std::uint32_t err = 0;
    std::int32_t tmax = -1000;
    int tj = 0;
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
      if (latest_.motor_state[j] != 0) err |= 1u << j;
      for (int k = 0; k < 2; ++k) {
        if (latest_.temperature[j][k] > tmax) {
          tmax = latest_.temperature[j][k];
          tj = j;
        }
      }
    }
    if (err != 0) faults |= R1RT_FAULT_MOTOR_ERROR;
    st_.motor_error_joints = err;
    st_.temp_max_c = tmax;
    st_.temp_max_joint = static_cast<std::uint8_t>(tj);
    // session 6 (DESIGN §4.1 over-temperature): a WARNING from the takeover's temperature gate up; never an action
    if (static_cast<double>(tmax) >= cfg_.takeover_temp_max_c) {
      faults |= R1RT_FAULT_OVER_TEMP;
      st_.over_temp_ticks += 1;
    }
    motor_error_holds(t_ns);
  }

  if (alarm_pending_) {
    faults |= R1RT_FAULT_TAKEOVER_ALARM;
    alarm_pending_ = false;
  }
  // the built-in controller still writing rt/lowcmd after it released (two writers): a sample received after the
  // release (RELEASED / ENTER_BALANCE) and still fresh - an alarm, nothing else
  // session 9 (U19 / D25): the "hold the stand" rung - a balancing robot whose state stayed older than the limit for
  // state_age_sustain_ns freezes into the stand (STAND_FREEZE: a ramp at stand gains that needs no state; arms / head
  // decelerate to HOLD). Never limp. stale_rung 2: back to BALANCE by itself once the state is live again.
  if (stale_act && mode_.load(std::memory_order_relaxed) == R1RT_MODE_BALANCE) {
    enter_freeze(t_ns);
    freeze_invalid_ = false;  // not an invalid policy: the reason is the state (stale_freeze in the status)
    stale_freeze_ = true;
    live_since_ns_ = 0;
    st_.stale_rung_entries += 1;
  }
  if (stale_freeze_) stale_return_step(t_ns);
  const std::int64_t b_rx = builtin_t_rx_ns_.load(std::memory_order_acquire);
  const r1rt_mode m_now = mode_.load(std::memory_order_relaxed);
  if (b_rx != 0 && b_rx > released_t_ns_ && t_ns - b_rx <= cfg_.builtin_max_age_ns && m_now != R1RT_MODE_MONITOR &&
      m_now != R1RT_MODE_TAKEOVER_MIRROR)
    faults |= R1RT_FAULT_BUILTIN_ACTIVE;

  // 5-7. the mode's command
  bool publish = false;
  bool upper = false;  // the executor writes arms + head this tick (BALANCE with fresh state, STAND_FREEZE)
  bool resent = false;  // out = an earlier command resent (stale state): never blended again (session 8)
  tick_event_ = 0;
  tick_exec_clamps_ = 0;
  const bool fixed_target = m_now == R1RT_MODE_ON_STAND || m_now == R1RT_MODE_STAND_FREEZE ||
                            (m_now == R1RT_MODE_TAKEOVER_BLEND &&
                             takeover_target_.load(std::memory_order_relaxed) == R1RT_MODE_ON_STAND);
  const bool hold_off = resume_wait_ && fixed_target;  // session 8: a stop, no fresh state yet - nothing published
  switch (hold_off ? static_cast<r1rt_mode>(R1RT_MODE_MONITOR) : m_now) {
    case R1RT_MODE_BALANCE:
      publish = true;
      upper = balance_tick(t_ns, out, &faults) != kBalStale;
      resent = !upper;
      break;
    case R1RT_MODE_TAKEOVER_MIRROR:
      // DESIGN §3.5: republish the snapshot while r1d runs ReleaseMode; a fault changes nothing (alarm), nor does a
      // long wait (alarm after takeover_mirror_alarm_ns: r1d decides RELEASED / ABORT, the operator watches).
      publish = true;
      copy_command(out, mirror_cmd_);
      if ((faults & kTakeoverFaults) != 0 || t_ns - mirror_t0_ > cfg_.takeover_mirror_alarm_ns)
        faults |= R1RT_FAULT_TAKEOVER_ALARM;
      break;
    case R1RT_MODE_TAKEOVER_BLEND: {
      // (1 - a) snapshot + a ours, a = elapsed / blend_s (v2 runtime.py:247-256); ours = BALANCE's command with the
      // executor's arms. Faults finish the blend (alarm); an invalid action freezes from the blended command (rung 3).
      publish = true;
      if (takeover_target_.load(std::memory_order_relaxed) == R1RT_MODE_ON_STAND) {
        // ON_STAND (session 5): ours = the stiff hold of the pose measured at RELEASED (+ any L0 records) - no
        // policy; stale state resends the last blended command (alarm), the clock goes on
        blend_progress_ = clampd(static_cast<double>(t_ns - blend_t0_) / static_cast<double>(blend_ns_), 0.0, 1.0);
        if (faults & R1RT_FAULT_LOWSTATE_STALE) {
          copy_command(out, last_cmd_);
          st_.stale_resends += 1;
          faults |= R1RT_FAULT_TAKEOVER_ALARM;
          resent = true;
          break;
        }
        on_stand_command(out, t_ns);
        const double a = blend_progress_;
        for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
          out->q[j] = (1.0 - a) * mirror_cmd_.q[j] + a * out->q[j];
          out->dq[j] = (1.0 - a) * mirror_cmd_.dq[j] + a * out->dq[j];
          out->kp[j] = (1.0 - a) * mirror_cmd_.kp[j] + a * out->kp[j];
          out->kd[j] = (1.0 - a) * mirror_cmd_.kd[j] + a * out->kd[j];
          out->tau_ff[j] = (1.0 - a) * mirror_cmd_.tau_ff[j] + a * out->tau_ff[j];
        }
        if ((faults & kTakeoverFaults) != 0) faults |= R1RT_FAULT_TAKEOVER_ALARM;
        if (a >= 1.0) mode_.store(R1RT_MODE_ON_STAND, std::memory_order_release);
        break;
      }
      const int r = balance_tick(t_ns, out, &faults);
      blend_progress_ = clampd(static_cast<double>(t_ns - blend_t0_) / static_cast<double>(blend_ns_), 0.0, 1.0);
      if (r == kBalFroze) {  // mode_ is STAND_FREEZE now; its ramp starts from the last (blended) command
        faults |= R1RT_FAULT_TAKEOVER_ALARM;
        upper = true;
        break;
      }
      if (r == kBalStale) {  // out = the last blended command, resent; the blend clock runs on
        faults |= R1RT_FAULT_TAKEOVER_ALARM;
        resent = true;
        break;
      }
      upper_step(out, t_ns);
      const double a = blend_progress_;
      for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
        out->q[j] = (1.0 - a) * mirror_cmd_.q[j] + a * out->q[j];
        out->dq[j] = (1.0 - a) * mirror_cmd_.dq[j] + a * out->dq[j];
        out->kp[j] = (1.0 - a) * mirror_cmd_.kp[j] + a * out->kp[j];
        out->kd[j] = (1.0 - a) * mirror_cmd_.kd[j] + a * out->kd[j];
        out->tau_ff[j] = (1.0 - a) * mirror_cmd_.tau_ff[j] + a * out->tau_ff[j];
      }
      if ((faults & kTakeoverFaults) != 0) faults |= R1RT_FAULT_TAKEOVER_ALARM;
      if (a >= 1.0) mode_.store(R1RT_MODE_BALANCE, std::memory_order_release);
      break;
    }
    case R1RT_MODE_STAND_FREEZE:
      publish = true;
      if (freeze_invalid_) faults |= R1RT_FAULT_POLICY_INVALID;  // no valid policy: the reason stays visible
      build_freeze(out, t_ns);
      upper = have_state_;
      break;
    case R1RT_MODE_ON_STAND:
      // DESIGN §3.5 (session 5): the robot hanging on its stand - every joint from the executors' L0 (HOLD at the
      // ON_STAND gains by default); no policy. Stale state: the last command resent (no motion on stale state).
      publish = true;
      if (faults & R1RT_FAULT_LOWSTATE_STALE) {
        copy_command(out, last_cmd_);
        st_.stale_resends += 1;
        resent = true;
        break;
      }
      on_stand_command(out, t_ns);
      break;
    case R1RT_MODE_DAMP:
      publish = true;
      build_damp(out);
      break;
    default:
      break;  // MONITOR (and the modes not built yet): nothing published
  }

  if (cfg_.takeover_require_rt != 0 && !rt_granted_) faults |= R1RT_FAULT_RT_NOT_GRANTED;  // an alarm (session 5)
  // session 5 (DESIGN §4.4): r1guard's heartbeat. In a publishing mode a stale / missing guard is an ALARM
  // (R1RT_FAULT_GUARD_ABSENT; nothing else changes) when the profile has a guard (takeover_require_guard)
  {
    r1rt_heartbeat ghb{};
    if (seg_->guard_heartbeat.try_read(ghb) && ghb.seq != 0) {
      guard_hb_ = ghb;
      have_guard_hb_ = true;
    }
    st_.guard_age_ns = have_guard_hb_ ? t_ns - guard_hb_.t_ns : kNever;
    if (publish && cfg_.takeover_require_guard != 0 && st_.guard_age_ns > cfg_.guard_max_age_ns) {
      faults |= R1RT_FAULT_GUARD_ABSENT;
      st_.guard_absent_ticks += 1;
    }
  }

  if (upper) upper_step(out, t_ns);
  if (hold_off) {
    // nothing published this tick (the reason stays visible); the mirror (the guard's start if the core stalls
    // again right now) carries the measured pose at the gains / feed-forward last sent - holding it does not snap
    faults |= R1RT_FAULT_LOOP_GAP;
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
      out->q[j] = have_state_ ? latest_.q[j] : last_cmd_.q[j];
      out->kp[j] = last_cmd_.kp[j];
      out->kd[j] = last_cmd_.kd[j];
      out->tau_ff[j] = last_cmd_.tau_ff[j];
    }
  }

  // 7b. session 8 (HW1 fix #29): the resume blend - from the pose measured after the stop to the mode's command over
  //     takeover_blend_s (the targets only: the mode's gains and feed-forward stand, so the hold builds up smoothly from
  //     the sagged pose). Every joint in ON_STAND and the sequence into it; arms + head elsewhere (STAND_FREEZE's legs
  //     restart their own ramp from the measured pose; BALANCE's legs are the policy's - D3 - never blended).
  if (resume_active_) {
    const r1rt_mode m_cmd = mode_.load(std::memory_order_relaxed);
    const bool all = m_cmd == R1RT_MODE_ON_STAND ||
                     (m_cmd == R1RT_MODE_TAKEOVER_BLEND &&
                      takeover_target_.load(std::memory_order_relaxed) == R1RT_MODE_ON_STAND);
    const bool arms = all || m_cmd == R1RT_MODE_BALANCE || m_cmd == R1RT_MODE_TAKEOVER_BLEND ||
                      m_cmd == R1RT_MODE_STAND_FREEZE;
    const double a = clampd(static_cast<double>(t_ns - resume_t0_) / static_cast<double>(blend_ns_), 0.0, 1.0);
    if (publish && !resent && arms) {
      for (int j = all ? 0 : kUpperFirst; j < R1RT_NUM_JOINTS; ++j) {
        out->q[j] = (1.0 - a) * resume_q0_[j] + a * out->q[j];
        out->dq[j] = a * out->dq[j];
      }
    }
    st_.resume_progress = a;
    if (a >= 1.0) resume_active_ = false;
  }

  // 8. hard limits
  std::uint32_t clamps = 0;
  if (publish) clamps = apply_limits(out, &faults) + tick_exec_clamps_;

  // 9. the command + the ownership token immediately before the write
  if (publish) {
    if (own_token_ && seg_->token.holds(token_word_)) {
      out->publish = 1;
      for (int j = 0; j < R1RT_NUM_JOINTS; ++j) out->enable[j] = 1;
    } else {
      out->publish = 0;
      faults |= R1RT_FAULT_NOT_OWNER;
    }
  }
  if (out->publish) {
    last_cmd_ = *out;
    const r1rt_mode m_out = mode_.load(std::memory_order_relaxed);
    if ((m_out == R1RT_MODE_BALANCE || m_out == R1RT_MODE_TAKEOVER_BLEND) && !(faults & R1RT_FAULT_LOWSTATE_STALE)) {
      balanced_cmd_ = *out;
      have_balanced_ = true;
    }
  }

  // 10. telemetry, state, heartbeat, mirror
  publish_shared(t_ns, *out, faults, clamps, age, tel, t_ns);
  return R1RT_OK;
}

void Core::publish_shared(std::int64_t t_ns, const r1rt_lowcmd_out& out, std::uint32_t faults, std::uint32_t clamps,
                          std::int64_t age, r1rt_telemetry_record* tel, std::int64_t t_wake) noexcept {
  const r1rt_mode mode = mode_.load(std::memory_order_relaxed);
  Mailbox mb{};
  read_mailbox(mb);
  st_.ticks = ticks_;
  st_.faults = faults;
  st_.faults_seen |= faults;
  st_.lowstate_age_ns = age;
  st_.tilt_rad = tilt_;
  st_.mode = mode;
  st_.inline_active = inline_active_.load(std::memory_order_relaxed) ? 1 : 0;
  st_.stop_active = stop_active_.load(std::memory_order_relaxed) ? 1 : 0;
  st_.gait_phase = policy_loaded_ ? mb.gait_phase : 0.0;
  st_.lean = policy_loaded_ ? mb.lean : 0.0;
  if (faults & R1RT_FAULT_TAKEOVER_ALARM) st_.takeover_alarm_ticks += 1;
  const std::int64_t b_rx = builtin_t_rx_ns_.load(std::memory_order_relaxed);
  st_.builtin_age_ns = b_rx == 0 ? kNever : t_ns - b_rx;
  const double progress = mode == R1RT_MODE_TAKEOVER_BLEND                                   ? blend_progress_
                          : (mode == R1RT_MODE_MONITOR || mode == R1RT_MODE_TAKEOVER_MIRROR) ? 0.0
                                                                                              : 1.0;
  st_.blend_progress = progress;
  sync_counters();
  st_.upper_stop_epoch = exec_.stop_epoch();
  st_.upper_tick_event = tick_event_;
  for (int g = 0; g < R1RT_NUM_UPPER_GROUPS; ++g) {
    const auto gr = static_cast<r1rt_group>(g);
    st_.upper_source[g] = exec_.source(gr);
    st_.upper_event[g] = exec_.last_guard(gr);
    st_.upper_stopping[g] = exec_.stopping(gr) ? 1 : 0;
    st_.upper_cmd_id[g] = exec_.active_cmd(gr);
  }
  st_.takeover_target = takeover_target_.load(std::memory_order_relaxed);
  st_.freeze_from_stand = (mode == R1RT_MODE_STAND_FREEZE && freeze_from_stand_) ? 1 : 0;
  st_.stale_freeze = (mode == R1RT_MODE_STAND_FREEZE && stale_freeze_) ? 1 : 0;  // session 9
  st_.stand_session = stand_session_ ? 1 : 0;
  for (int g = 0; g < R1RT_NUM_CMD_GROUPS - R1RT_NUM_UPPER_GROUPS; ++g) {
    const auto gr = static_cast<r1rt_group>(R1RT_NUM_UPPER_GROUPS + g);
    st_.lower_source[g] = exec_lower_.source(gr);
    st_.lower_event[g] = exec_lower_.last_guard(gr);
    st_.lower_stopping[g] = exec_lower_.stopping(gr) ? 1 : 0;
    st_.lower_cmd_id[g] = exec_lower_.active_cmd(gr);
  }

  r1rt_state s{};
  s.seq = ticks_;
  s.t_core_ns = t_ns;
  s.in = latest_;
  for (int i = 0; i < 3; ++i) s.projected_gravity[i] = gravity_[i];
  s.tilt_rad = tilt_;
  s.lowstate_age_ns = age;
  s.faults = faults;
  s.mode = mode;
  s.token_owner = shm::OwnershipToken::owner_of(seg_->token.load());
  seg_->state.write(s);

  r1rt_heartbeat hb{};
  hb.seq = ++hb_seq_;
  hb.t_ns = t_ns;
  hb.lowstate_tick = latest_.board_tick;
  hb.rate_hz = cfg_.writer_rate_hz;
  hb.faults = faults;
  hb.blend_progress = static_cast<float>(progress);
  hb.mode = mode;
  hb.rt_granted = rt_granted_ ? 1 : 0;  // the process wrapper's grants (session 5; lockstep: 0)
  // session 6: r1rt's flags (the stand session); session 13: + the latched stop (the base command / lean is zero)
  hb.guard_flags = static_cast<std::uint8_t>((stand_session_ ? R1RT_RT_FLAG_STAND_SESSION : 0u) |
                                             (stop_active_.load(std::memory_order_relaxed) ? R1RT_RT_FLAG_STOP_ACTIVE : 0u));
  seg_->rt_heartbeat.write(hb);

  r1rt_mirror mi{};
  mi.seq = ticks_;
  mi.t_ns = t_ns;
  mi.last_action_t_ns = mb.t_action_ns;
  mi.gait_phase = st_.gait_phase;
  for (int k = 0; k < R1RT_NUM_POLICY_ACTIONS; ++k) mi.last_action[k] = mb.raw[k];
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    mi.q_target[j] = out.q[j];
    mi.kp[j] = out.kp[j];
    mi.kd[j] = out.kd[j];
    mi.tau_ff[j] = out.tau_ff[j];  // layout 3 (session 8, fix #28): the guard holds the arms / head with it
  }
  mi.base_cmd[3] = st_.lean;
  mi.blend_progress = static_cast<float>(progress);
  mi.mode = mode;
  mi.policy_valid = mb.valid;
  // layout 2 (session 5): what the guard needs to continue the core exactly (GUARD.md "not solid" 2)
  mi.freeze_invalid = (mode == R1RT_MODE_STAND_FREEZE && freeze_invalid_) ? 1 : 0;
  mi.blend_t0_ns = mode == R1RT_MODE_TAKEOVER_BLEND ? blend_t0_ : 0;
  mi.blend_ns = blend_ns_;
  mi.freeze_t0_ns = mode == R1RT_MODE_STAND_FREEZE ? freeze_t0_ : 0;
  mi.freeze_ramp_ns = static_cast<std::int64_t>(std::llround(cfg_.stand_ramp_s * 1e9));
  for (int i = 0; i < 2; ++i) {
    mi.waist[i] = last_waist_[i];
    mi.waist_target[i] = last_waist_target_[i];
  }
  mi.takeover_target = takeover_target_.load(std::memory_order_relaxed);
  mi.stops = st_.stops;  // layout 3 (session 8, #31)
  mi.stop_ns_max = st_.stop_ns_max;
  mi.resume_blends = st_.resume_blends;
  mi.resume_waiting = resume_wait_ ? 1 : 0;
  seg_->mirror.write(mi);

  // layout 4 (session 10): the executors' per-group status + the state-age counters for r1d on real processes (the
  // lockstep core gives r1d the same numbers through Core.status(); st_ was filled above this tick)
  r1rt_exec_status es{};
  es.seq = ticks_;
  es.t_ns = t_ns;
  es.upper_accepted = st_.upper_accepted;
  es.upper_rejected = st_.upper_rejected;
  es.stop_epoch = st_.upper_stop_epoch;
  es.lower_accepted = st_.lower_accepted;
  es.lower_rejected = st_.lower_rejected;
  for (int g = 0; g < R1RT_NUM_UPPER_GROUPS; ++g) {
    es.cmd_id[g] = st_.upper_cmd_id[g];
    es.source[g] = st_.upper_source[g];
    es.event[g] = st_.upper_event[g];
    es.stopping[g] = st_.upper_stopping[g];
  }
  for (int g = 0; g < R1RT_NUM_CMD_GROUPS - R1RT_NUM_UPPER_GROUPS; ++g) {
    es.cmd_id[R1RT_NUM_UPPER_GROUPS + g] = st_.lower_cmd_id[g];
    es.source[R1RT_NUM_UPPER_GROUPS + g] = st_.lower_source[g];
    es.event[R1RT_NUM_UPPER_GROUPS + g] = st_.lower_event[g];
    es.stopping[R1RT_NUM_UPPER_GROUPS + g] = st_.lower_stopping[g];
  }
  es.last_reject = st_.upper_last_reject;
  es.tick_event = st_.upper_tick_event;
  es.motor_error_groups = st_.motor_error_groups;
  es.content_age_valid = st_.content_age_valid;
  es.stale_freeze = st_.stale_freeze;
  es.stale_rung = static_cast<std::uint8_t>(cfg_.stale_rung);
  es.state_content_age_ns = st_.state_content_age_ns;
  es.state_content_age_max_ns = st_.state_content_age_max_ns;
  es.state_old_ticks = st_.state_old_ticks;
  es.state_old_episodes = st_.state_old_episodes;
  es.state_old_ns_max = st_.state_old_ns_max;
  es.stale_rung_entries = st_.stale_rung_entries;
  es.content_age_relearns = st_.content_age_relearns;
  seg_->exec_status.write(es);

  r1rt_telemetry_record r{};
  r.seq = ticks_;
  r.t_wake_ns = t_wake;
  r.wake_lateness_ns = wake_lateness_ns_;  // the process's (session 5; lockstep: 0)
  r.iteration_ns = prev_iteration_ns_;     // the PREVIOUS iteration's duration (this one is not finished yet)
  if (wake_lateness_ns_ > st_.max_wake_lateness_ns) st_.max_wake_lateness_ns = wake_lateness_ns_;
  if (prev_iteration_ns_ > st_.max_iteration_ns) st_.max_iteration_ns = prev_iteration_ns_;
  st_.rt_granted = rt_granted_ ? 1 : 0;
  r.t_send_ns = out.publish ? t_ns : 0;
  r.t_state_rx_ns = latest_.t_rx_ns;
  r.board_tick = latest_.board_tick;
  r.faults = faults;
  r.clamp_count = clamps;
  r.guard_event = tick_event_;
  r.mode = mode;
  r.token_owner = s.token_owner;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    r.q[j] = static_cast<float>(latest_.q[j]);
    r.dq[j] = static_cast<float>(latest_.dq[j]);
    r.tau_est[j] = static_cast<float>(latest_.tau_est[j]);
    r.q_cmd[j] = static_cast<float>(out.q[j]);
    r.dq_cmd[j] = static_cast<float>(out.dq[j]);
    r.kp[j] = static_cast<float>(out.kp[j]);
    r.kd[j] = static_cast<float>(out.kd[j]);
    r.tau_ff[j] = static_cast<float>(out.tau_ff[j]);
    r.temperature[j][0] = latest_.temperature[j][0];
    r.temperature[j][1] = latest_.temperature[j][1];
    r.motor_state[j] = latest_.motor_state[j];
  }
  for (int i = 0; i < 4; ++i) r.imu_quat[i] = static_cast<float>(latest_.imu_quat[i]);
  for (int i = 0; i < 3; ++i) {
    r.imu_gyro[i] = static_cast<float>(latest_.imu_gyro[i]);
    r.imu_acc[i] = static_cast<float>(latest_.imu_acc[i]);
  }
  if (!seg_->telemetry.try_push(r)) st_.telemetry_drops += 1;
  if (tel != nullptr) *tel = r;
}

int32_t Core::policy_step(std::int64_t t_ns) noexcept {
  // The 50 Hz policy thread's iteration. Touches only: mode_ / inline_active_ / thread_beats_ (atomics), the
  // segment's state and base_cmd seqlocks, its own runner, the committed slot through the CAS protocol and its
  // counters (atomics). Never the writer's members.
  const r1rt_mode m = mode_.load(std::memory_order_acquire);
  if ((m != R1RT_MODE_BALANCE && m != R1RT_MODE_TAKEOVER_BLEND) || !policy_loaded_) return R1RT_OK;  // idle
  if (m == R1RT_MODE_TAKEOVER_BLEND && takeover_target_.load(std::memory_order_acquire) == R1RT_MODE_ON_STAND)
    return R1RT_OK;  // the blend into ON_STAND runs no policy
  if (inline_active_.load(std::memory_order_acquire)) {
    thread_beats_.fetch_add(1, std::memory_order_acq_rel);  // alive; the writer hands back after kHandbackBeats
    return R1RT_OK;
  }
  r1rt_state s{};
  if (!seg_->state.try_read(s) || s.seq == 0) return R1RT_OK;
  if (t_ns - s.in.t_rx_ns > cfg_.lowstate_stale_ns) return R1RT_OK;  // no policy on stale state
  run_policy(t_ns, kSourceThread, s.in);
  return R1RT_OK;
}

void Core::sync_counters() noexcept {
  st_.policy_thread_runs = n_thread_runs_.load(std::memory_order_relaxed);
  st_.invalid_actions = n_invalid_.load(std::memory_order_relaxed);
  st_.policy_commits = n_commits_.load(std::memory_order_relaxed);
  st_.policy_discarded = n_discarded_.load(std::memory_order_relaxed);
  st_.last_action_t_ns = last_action_t_ns_.load(std::memory_order_relaxed);
}

int32_t Core::set_builtin_cmd(const r1rt_lowcmd_out& cmd, std::int64_t t_rx_ns) noexcept {
  Builtin b{};
  b.cmd = cmd;
  b.t_rx_ns = t_rx_ns;
  b.n = ++builtin_n_;
  builtin_.write(b);
  builtin_t_rx_ns_.store(t_rx_ns, std::memory_order_release);
  return R1RT_OK;
}

int Core::takeover_gate(std::int64_t t_ns, r1rt_mode target) noexcept {
  // v2's gates (r1mc2/runtime.py:341-372), in its order, plus the guard (DESIGN §2.1: its absence blocks takeover)
  // and the token. Nothing changes unless every gate passes.
  // ON_STAND (session 5; the robot hanging on its stand, feet off the floor): the same gates and the same mirror /
  // blend, except (a) no policy is needed (nothing balances on the stand; the executors must be ready), and (b) the
  // built-in's rt/lowcmd must be fresh (something to mirror, and the release must take over from a live writer) but
  // NOT stiff: on the stand Unitree's damping mode (leg kp 0) is a normal start - mirroring it and blending into the
  // stiff hold stiffens the hanging legs over the blend. Kept for the stand: the tilt gate (a hanging robot tilted
  // > 12 deg is not hanging straight), the motion gate (a swinging robot is not taken), motor errors, temperatures,
  // the guard (it holds the last command if the core stalls) and the token.
  const bool stand = target == R1RT_MODE_ON_STAND;
  if (mode_.load(std::memory_order_relaxed) != R1RT_MODE_MONITOR) return R1RT_TAKEOVER_NOT_MONITOR;
  if (stand ? !(exec_ok_ && exec_lower_ok_) : (!policy_loaded_ || !exec_ok_)) return R1RT_TAKEOVER_NO_POLICY;
  if (!have_state_ || t_ns - latest_.t_rx_ns > cfg_.takeover_state_max_age_ns) return R1RT_TAKEOVER_STATE;
  double g[3];
  projected_gravity(latest_.imu_quat, g);
  if (std::acos(clampd(-g[2], -1.0, 1.0)) > cfg_.takeover_tilt_max_rad) return R1RT_TAKEOVER_TILT;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j)
    if (!(std::fabs(latest_.dq[j]) <= cfg_.takeover_dq_max)) return R1RT_TAKEOVER_MOVING;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j)
    if (latest_.motor_state[j] != 0) return R1RT_TAKEOVER_MOTOR_ERROR;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j)
    for (int k = 0; k < 2; ++k)
      if (static_cast<double>(latest_.temperature[j][k]) >= cfg_.takeover_temp_max_c) return R1RT_TAKEOVER_HOT;
  if (cfg_.takeover_require_guard != 0) {
    r1rt_heartbeat hb{};
    if (!seg_->guard_heartbeat.try_read(hb) || hb.seq == 0 || t_ns - hb.t_ns > cfg_.guard_max_age_ns)
      return R1RT_TAKEOVER_GUARD;
  }
  // DESIGN §2.1: no takeover (nor ON_STAND) without the real-time grants the profile requires (session 5)
  if (cfg_.takeover_require_rt != 0 && !rt_granted_) return R1RT_TAKEOVER_RT;
  // the built-in is visibly holding the legs: a fresh rt/lowcmd with stiff legs (stand lock 200; its walk 84 / 31)
  Builtin b{};
  if (!builtin_.try_read(b) || b.n == 0 || t_ns - b.t_rx_ns > cfg_.builtin_max_age_ns) return R1RT_TAKEOVER_BUILTIN;
  for (int j = 0; j < kLegs && !stand; ++j)
    if (!(b.cmd.kp[j] > cfg_.builtin_min_leg_kp)) return R1RT_TAKEOVER_BUILTIN;
  std::uint64_t w = seg_->token.load();
  if (shm::OwnershipToken::owner_of(w) != R1RT_OWNER_NONE) return R1RT_TAKEOVER_TOKEN;
  if (!seg_->token.try_transfer(w, R1RT_OWNER_R1RT)) return R1RT_TAKEOVER_TOKEN;
  token_word_ = w;
  own_token_ = true;
  // The snapshot (v2 runtime.py:376-382): the built-in's command when every motor is enabled and every joint target is
  // within mirror_max_dev of the measured pose (stand lock sits within ~0.04 rad); otherwise a hold of the measured
  // pose at the stand gains (v2: its HOLD tick).
  bool use = true;
  double dev = 0.0;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    const double d = std::fabs(b.cmd.q[j] - latest_.q[j]);
    if (b.cmd.enable[j] == 0 || !std::isfinite(b.cmd.q[j]) || !std::isfinite(b.cmd.dq[j]) ||
        !std::isfinite(b.cmd.kp[j]) || !std::isfinite(b.cmd.kd[j]) || !std::isfinite(b.cmd.tau_ff[j]))
      use = false;
    if (d > dev || !std::isfinite(d)) dev = d;
  }
  if (!(dev < cfg_.mirror_max_dev_rad)) use = false;
  // Session 6 (rehearse R7; the user's U13 / D22: "a 1 s blend into a stiff hold"): on the stand Unitree starts in ZERO
  // TORQUE (kp 0 / kd 0) or a damping mode, and a slot may be disabled. A soft slot's target means nothing (no torque
  // follows from it; zero torque's targets need not be near the hanging pose), so mirroring it would blend the legs
  // towards junk targets as the gains rise, and the old fallback (the measured pose at the ON_STAND gains from the first
  // MIRROR tick) stiffened the legs 0 -> hold in ONE tick (a swinging leg, |dq| <= 0.5 passes the gate, stopped
  // abruptly). When every leg / waist slot is soft (disabled, or kp < kSoftKp), the snapshot is per joint: a soft or
  // disabled slot = the MEASURED pose at the built-in's own gains (disabled = kp 0 / kd 0: zero torque, what that motor
  // already does); a stiff slot (e.g. arms still held) = the built-in's command as it is. The BLEND then ramps the gains
  // from what the motors have now to the ON_STAND hold over takeover_blend_s. A stiff built-in (stand lock) and the
  // takeover into BALANCE (its gate needs a stiff built-in) are unchanged.
  bool soft = stand;
  for (int j = 0; j < kUpperFirst && soft; ++j) {
    const bool on = b.cmd.enable[j] != 0 && std::isfinite(b.cmd.kp[j]);
    if (on && b.cmd.kp[j] >= kSoftKp) soft = false;
  }
  base_fill(&mirror_cmd_);
  if (soft) {
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
      const bool on = b.cmd.enable[j] != 0 && std::isfinite(b.cmd.q[j]) && std::isfinite(b.cmd.dq[j]) &&
                      std::isfinite(b.cmd.kp[j]) && std::isfinite(b.cmd.kd[j]) && std::isfinite(b.cmd.tau_ff[j]);
      if (on && b.cmd.kp[j] >= kSoftKp) {  // a stiff slot: as the built-in sends it
        mirror_cmd_.q[j] = b.cmd.q[j];
        mirror_cmd_.dq[j] = b.cmd.dq[j];
        mirror_cmd_.kp[j] = b.cmd.kp[j];
        mirror_cmd_.kd[j] = b.cmd.kd[j];
        mirror_cmd_.tau_ff[j] = b.cmd.tau_ff[j];
        continue;
      }
      mirror_cmd_.q[j] = latest_.q[j];
      mirror_cmd_.kp[j] = on ? clampd(b.cmd.kp[j], 0.0, cfg_.kp_max[j]) : 0.0;
      mirror_cmd_.kd[j] = on ? clampd(b.cmd.kd[j], 0.0, cfg_.kd_max[j]) : 0.0;
      mirror_cmd_.tau_ff[j] = on ? b.cmd.tau_ff[j] : 0.0;
    }
  } else if (use) {
    copy_command(&mirror_cmd_, b.cmd);
  } else {
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
      mirror_cmd_.q[j] = latest_.q[j];
      mirror_cmd_.kp[j] = stand ? on_stand_kp_[j] : cfg_.stand_kp[j];
      mirror_cmd_.kd[j] = stand ? on_stand_kd_[j] : cfg_.stand_kd[j];
    }
  }
  takeover_target_.store(static_cast<std::uint8_t>(target), std::memory_order_release);
  st_.mirror_dev_rad = dev;
  st_.takeover_used_mirror = soft ? 2 : (use ? 1 : 0);  // 2 = session 6: soft slots at the measured pose (their gains)
  mirror_t0_ = t_ns;
  blend_progress_ = 0.0;
  mode_.store(R1RT_MODE_TAKEOVER_MIRROR, std::memory_order_release);
  return R1RT_TAKEOVER_OK;
}

int32_t Core::request(std::int32_t req, std::int64_t t_ns) noexcept {
  switch (req) {
    case R1RT_REQ_ENTER_BALANCE: {
      // session 11: the refusal's reason goes into the mailbox reply (balance_refusal_; no status field, no ABI change)
      balance_refusal_ = R1RT_REFUSE_MODE;
      if (stand_session_) return R1RT_E_REFUSED;  // session 6: a stand session never balances (the mailbox: code 12)
      const r1rt_mode m = mode_.load(std::memory_order_relaxed);
      if (m == R1RT_MODE_DAMP) return R1RT_E_REFUSED;  // from DAMP only through a takeover
      if (m == R1RT_MODE_ON_STAND) return R1RT_E_REFUSED;  // never balance a robot hanging on its stand
      if (m == R1RT_MODE_STAND_FREEZE && freeze_from_stand_) return R1RT_E_REFUSED;  // session 6: the same robot
      if (m == R1RT_MODE_TAKEOVER_MIRROR || m == R1RT_MODE_TAKEOVER_BLEND) return R1RT_E_REFUSED;  // the sequence runs
      if (!policy_loaded_) return R1RT_E_REFUSED;
      balance_refusal_ = R1RT_REFUSE_STATE;
      if (!exec_ok_ || !fresh(t_ns)) return R1RT_E_REFUSED;
      // session 11 (S10's not-solid item; STATE_AGE.md "Session 11"): never into BALANCE on a state older than the
      // state-age limit NOW (the receive age, or the board-tick content age when valid - the clock does not re-base
      // during a freeze, so a state that stayed late reads late). Measured whatever stale_rung is: a refusal, never an
      // action - the robot stays in its freeze (stand gains, never limp) and the operator tries again.
      if (have_state_ && trig_age_ns_ > cfg_.state_age_limit_ns) return R1RT_E_REFUSED;
      balance_refusal_ = R1RT_REFUSE_MODE;
      if (!own_token_) {
        std::uint64_t w = seg_->token.load();
        if (shm::OwnershipToken::owner_of(w) != R1RT_OWNER_NONE) return R1RT_E_NOT_OWNER;
        if (!seg_->token.try_transfer(w, R1RT_OWNER_R1RT)) return R1RT_E_NOT_OWNER;
        token_word_ = w;
        own_token_ = true;
      }
      exec_.reset(latest_.q, t_ns);  // the executor starts publishing the upper body: every group HOLDs the measured pose
      inline_active_.store(false, std::memory_order_release);
      // the first action (from a reset runner), so the first BALANCE tick has one; computed before the mode switch,
      // so the policy thread (idle outside BALANCE / BLEND) cannot race it
      if (run_policy(t_ns, kSourceEntry, latest_) != kPolicyValid) {
        enter_freeze(t_ns);
        balance_refusal_ = R1RT_REFUSE_MODE;  // r1d reads freeze_invalid: the first action was invalid, frozen again
        return R1RT_E_REFUSED;
      }
      balance_refusal_ = R1RT_REFUSE_NONE;
      released_t_ns_ = t_ns;
      mode_.store(R1RT_MODE_BALANCE, std::memory_order_release);
      return R1RT_OK;
    }
    case R1RT_REQ_TAKEOVER: {
      st_.takeover_requests += 1;
      // session 6: a stand session (r1rt --stand-session) refuses every entry into BALANCE, before any gate
      const int why = stand_session_ ? R1RT_TAKEOVER_STAND_SESSION : takeover_gate(t_ns, R1RT_MODE_BALANCE);
      st_.takeover_refusal = static_cast<std::uint8_t>(why);
      return why == R1RT_TAKEOVER_OK ? R1RT_OK : R1RT_E_REFUSED;
    }
    case R1RT_REQ_ON_STAND: {  // session 5 (DESIGN §3.5): the same sequence, ending in ON_STAND
      const int why = takeover_gate(t_ns, R1RT_MODE_ON_STAND);
      st_.on_stand_refusal = static_cast<std::uint8_t>(why);
      return why == R1RT_TAKEOVER_OK ? R1RT_OK : R1RT_E_REFUSED;
    }
    case R1RT_REQ_TAKEOVER_RELEASED: {
      // The built-in has let go (r1d's ReleaseMode succeeded): from here only forward - BLEND, then BALANCE (or
      // STAND_FREEZE on an invalid action). Accepted with stale state too (the first action then comes from the
      // inline rung once state is fresh; until then the snapshot is resent).
      if (mode_.load(std::memory_order_relaxed) != R1RT_MODE_TAKEOVER_MIRROR) return R1RT_E_REFUSED;
      exec_.reset(latest_.q, t_ns);  // arms + head: HOLD the measured pose, blended in from the snapshot
      inline_active_.store(false, std::memory_order_release);
      if (takeover_target_.load(std::memory_order_relaxed) == R1RT_MODE_ON_STAND) {
        // ON_STAND: legs + waist HOLD the measured pose too (the lower executor); no policy is started
        exec_lower_.reset(latest_.q, t_ns);
        blend_t0_ = t_ns;
        released_t_ns_ = t_ns;
        blend_progress_ = 0.0;
        mode_.store(R1RT_MODE_TAKEOVER_BLEND, std::memory_order_release);
        return R1RT_OK;
      }
      balanced_cmd_ = last_cmd_;  // a stale-state resend during the blend repeats the last published command
      if (balanced_cmd_.publish == 0) copy_command(&balanced_cmd_, mirror_cmd_);
      have_balanced_ = true;
      blend_t0_ = t_ns;
      released_t_ns_ = t_ns;
      blend_progress_ = 0.0;
      if (fresh(t_ns) && run_policy(t_ns, kSourceEntry, latest_) == kPolicyInvalid) {
        enter_freeze(t_ns);  // rung 3 from the snapshot (last_cmd_): no valid policy; never back to MONITOR
        return R1RT_E_REFUSED;
      }
      mode_.store(R1RT_MODE_TAKEOVER_BLEND, std::memory_order_release);
      return R1RT_OK;
    }
    case R1RT_REQ_TAKEOVER_ABORT: {
      // ReleaseMode failed: the built-in still drives the motors - stop competing (DESIGN §3.5). Only from MIRROR.
      if (mode_.load(std::memory_order_relaxed) != R1RT_MODE_TAKEOVER_MIRROR) return R1RT_E_REFUSED;
      std::uint64_t w = token_word_;
      if (own_token_) seg_->token.try_transfer(w, R1RT_OWNER_NONE);  // the token back (nobody of ours writes)
      own_token_ = false;
      mode_.store(R1RT_MODE_MONITOR, std::memory_order_release);
      st_.takeover_aborts += 1;
      st_.faults_seen |= R1RT_FAULT_TAKEOVER_ALARM;
      alarm_pending_ = true;
      return R1RT_OK;
    }
    case R1RT_REQ_STOP:
      internal_stop(t_ns);  // base command zero, arms decelerate to HOLD, keep balancing
      return R1RT_OK;
    case R1RT_REQ_RESUME:
      stop_active_.store(false, std::memory_order_release);
      // session 6: the operator clears the motor-error HOLD of every upper group whose error words are now zero (a
      // group whose word is still set stays held); the group stays in HOLD - only its records are accepted again
      if (exec_ok_ && motor_err_groups_ != 0) {
        static constexpr int kFirst[R1RT_NUM_UPPER_GROUPS] = {14, 19, 24};
        static constexpr int kN[R1RT_NUM_UPPER_GROUPS] = {5, 5, 2};
        for (int g = 0; g < R1RT_NUM_UPPER_GROUPS; ++g) {
          const std::uint8_t bit = static_cast<std::uint8_t>(1u << g);
          if ((motor_err_groups_ & bit) == 0) continue;
          bool bad = false;
          for (int k = 0; k < kN[g]; ++k) bad = bad || latest_.motor_state[kFirst[g] + k] != 0;
          if (bad) continue;
          motor_err_groups_ = static_cast<std::uint8_t>(motor_err_groups_ & ~bit);
          exec_.set_group_held(static_cast<r1rt_group>(g), false);
        }
        st_.motor_error_groups = motor_err_groups_;
      }
      return R1RT_OK;
    case R1RT_REQ_DAMP:
      // nothing of ours is published in MONITOR: nothing to damp
      if (mode_.load(std::memory_order_relaxed) == R1RT_MODE_MONITOR) return R1RT_E_REFUSED;
      mode_.store(R1RT_MODE_DAMP, std::memory_order_release);
      inline_active_.store(false, std::memory_order_release);
      freeze_from_stand_ = false;
      return R1RT_OK;
    case R1RT_REQ_STAND_FREEZE: {
      const r1rt_mode m = mode_.load(std::memory_order_relaxed);
      st_.freeze_refusal = R1RT_REFUSE_NONE;
      if (m == R1RT_MODE_ON_STAND) {
        // session 6 (HW1 S10, U3): the operator's freeze ON THE STAND - the real torque step of the freeze ramp seen
        // on a hanging robot. Only near the stand pose (the test moves the legs to 0.10 rad from it first), with fresh
        // state; the ramp starts from the last command (enter_freeze), legs + waist end at the stand pose and gains.
        if (!fresh(t_ns)) {
          st_.freeze_refusal = R1RT_REFUSE_STATE;
          return R1RT_E_REFUSED;
        }
        for (int j = 0; j < kUpperFirst; ++j) {
          if (!(std::fabs(last_cmd_.q[j] - cfg_.stand_q[j]) <= R1RT_ON_STAND_FREEZE_MAX_DEV_RAD)) {
            st_.freeze_refusal = R1RT_REFUSE_FAR;
            return R1RT_E_REFUSED;
          }
        }
        enter_freeze(t_ns, true);
        freeze_from_stand_ = true;
        return R1RT_OK;
      }
      if (m != R1RT_MODE_BALANCE && m != R1RT_MODE_TAKEOVER_BLEND) return R1RT_E_REFUSED;
      // the freeze ramps the legs to the standing pose: not on a robot hanging on its stand (ON_STAND's blend)
      if (m == R1RT_MODE_TAKEOVER_BLEND && takeover_target_.load(std::memory_order_relaxed) == R1RT_MODE_ON_STAND)
        return R1RT_E_REFUSED;
      enter_freeze(t_ns, true);
      return R1RT_OK;
    }
    case R1RT_REQ_STAND_HOLD: {
      // session 6: back from the operator's freeze on the stand to the ON_STAND hold - a blend from the freeze's last
      // command into a hold of that command (no position step; the gains go to the ON_STAND hold gains), then ON_STAND
      st_.freeze_refusal = R1RT_REFUSE_NONE;
      if (mode_.load(std::memory_order_relaxed) != R1RT_MODE_STAND_FREEZE || !freeze_from_stand_) {
        st_.freeze_refusal = R1RT_REFUSE_MODE;
        return R1RT_E_REFUSED;
      }
      if (!fresh(t_ns) || !exec_ok_ || !exec_lower_ok_) {
        st_.freeze_refusal = R1RT_REFUSE_STATE;
        return R1RT_E_REFUSED;
      }
      copy_command(&mirror_cmd_, last_cmd_);
      exec_.reset(last_cmd_.q, t_ns);
      exec_lower_.reset(last_cmd_.q, t_ns);
      takeover_target_.store(static_cast<std::uint8_t>(R1RT_MODE_ON_STAND), std::memory_order_release);
      blend_t0_ = t_ns;
      released_t_ns_ = t_ns;
      blend_progress_ = 0.0;
      freeze_from_stand_ = false;
      mode_.store(R1RT_MODE_TAKEOVER_BLEND, std::memory_order_release);
      return R1RT_OK;
    }
    case R1RT_REQ_RECLAIM: {
      // session 6 (DESIGN §4.4, §14.1 T6): the operator takes ownership back from r1guard - only in ON_STAND (the
      // guard holds the core's last command there: holding, not balancing). From BALANCE the guard runs its own policy
      // and a hand-back between two balancing policies is new and unproven: refused. The mailbox already checked the
      // requester (operator only); this direct path is the simulator's / tests'.
      const std::uint64_t w0 = seg_->token.load();
      std::uint8_t why = R1RT_REFUSE_NONE;
      if (shm::OwnershipToken::owner_of(w0) != R1RT_OWNER_R1GUARD) {
        why = R1RT_REFUSE_TOKEN;
      } else if (mode_.load(std::memory_order_relaxed) != R1RT_MODE_ON_STAND) {
        why = R1RT_REFUSE_MODE;
      } else if (!fresh(t_ns) || !exec_ok_ || !exec_lower_ok_) {
        why = R1RT_REFUSE_STATE;
      }
      st_.reclaim_refusal = why;
      if (why != R1RT_REFUSE_NONE) return R1RT_E_REFUSED;
      // What the guard is sending: it took over from the core's mirror = the core's last PUBLISHED command (last_cmd_:
      // after SIGCONT the core computed but published nothing, so last_cmd_ is still the stall's last packet) and
      // holds it in ON_STAND exactly as Guard::build does - every joint at those targets and gains, no velocity, no
      // feed-forward on the legs / waist, the arms / head with the last feed-forward (session 8, fix #28) and a zero
      // gain replaced by the HOLD gain. The blend starts from that.
      base_fill(&mirror_cmd_);
      for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
        mirror_cmd_.q[j] = std::isfinite(last_cmd_.q[j]) ? last_cmd_.q[j] : latest_.q[j];
        mirror_cmd_.kp[j] = last_cmd_.kp[j];
        mirror_cmd_.kd[j] = last_cmd_.kd[j];
        if (j >= kUpperFirst && std::isfinite(last_cmd_.tau_ff[j])) mirror_cmd_.tau_ff[j] = last_cmd_.tau_ff[j];
        if (j >= kUpperFirst && !(mirror_cmd_.kp[j] > 0.0)) mirror_cmd_.kp[j] = cfg_.upper_kp[j];
        if (j >= kUpperFirst && !(mirror_cmd_.kd[j] > 0.0)) mirror_cmd_.kd[j] = cfg_.upper_kd[j];
      }
      std::uint64_t w = w0;
      if (!seg_->token.try_transfer(w, R1RT_OWNER_R1RT)) {  // the token moved under us (never expected): nothing changes
        st_.reclaim_refusal = R1RT_REFUSE_TOKEN;
        return R1RT_E_REFUSED;
      }
      token_word_ = w;
      own_token_ = true;
      // both executors HOLD the held targets (records r1d queued while the guard owned were drained by the non-owner
      // ticks after SIGCONT; the reset discards what they did)
      exec_.reset(mirror_cmd_.q, t_ns);
      exec_lower_.reset(mirror_cmd_.q, t_ns);
      takeover_target_.store(static_cast<std::uint8_t>(R1RT_MODE_ON_STAND), std::memory_order_release);
      blend_t0_ = t_ns;
      released_t_ns_ = t_ns;
      blend_progress_ = 0.0;
      mode_.store(R1RT_MODE_TAKEOVER_BLEND, std::memory_order_release);
      st_.reclaims += 1;
      return R1RT_OK;
    }
    default:
      return R1RT_E_INVALID_ARG;
  }
}

int32_t Core::set_base_cmd(const r1rt_base_cmd& cmd) noexcept {
  seg_->base_cmd.write(cmd);
  return R1RT_OK;
}

int32_t Core::inject(std::int32_t kind, std::int64_t t0, std::int64_t t1, double value) noexcept {
  if (kind != R1RT_INJECT_POLICY_NAN && kind != R1RT_INJECT_POLICY_OUT_OF_RANGE) return R1RT_E_INVALID_ARG;
  const int n = n_inj_.load(std::memory_order_relaxed);  // one injecting thread at a time (the caller's contract)
  if (n >= kMaxInjections || t1 <= t0) return R1RT_E_INVALID_ARG;
  inj_[n] = Injection{kind, t0, t1, value};
  n_inj_.store(n + 1, std::memory_order_release);
  return R1RT_OK;
}

void Core::status(r1rt_core_status* out) const noexcept {
  *out = st_;  // the writer's copy, with the counters the policy thread touches read fresh
  out->policy_thread_runs = n_thread_runs_.load(std::memory_order_relaxed);
  out->invalid_actions = n_invalid_.load(std::memory_order_relaxed);
  out->policy_commits = n_commits_.load(std::memory_order_relaxed);
  out->policy_discarded = n_discarded_.load(std::memory_order_relaxed);
  out->last_action_t_ns = last_action_t_ns_.load(std::memory_order_relaxed);
  out->resume_waiting = resume_wait_ ? 1 : 0;
  if (!resume_active_) out->resume_progress = 1.0;
}

}  // namespace r1rt
