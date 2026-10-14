// r1rt::Guard (guard.hpp): r1guard's cycle - its own lowstate, heartbeats, the warm shadow policy, the takeover by
// CAS on a stale core heartbeat and the by-mode command of DESIGN.md §4.3. Never an automatic damp.
#include "r1rt/guard.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace r1rt {

namespace {

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

// Modes in which the core publishes rt/lowcmd (the guard continues them); MONITOR publishes nothing, GUARDED is the
// guard's own, OVERLAY_ARM_SDK publishes rt/arm_sdk (Unitree balances; the guard's arm_sdk continuation is not built).
bool core_publishes(r1rt_mode m) noexcept {
  switch (m) {
    case R1RT_MODE_TAKEOVER_MIRROR:
    case R1RT_MODE_TAKEOVER_BLEND:
    case R1RT_MODE_BALANCE:
    case R1RT_MODE_STAND_FREEZE:
    case R1RT_MODE_ON_STAND:
    case R1RT_MODE_DAMP:
      return true;
    default:
      return false;
  }
}

}  // namespace

int32_t Guard::init(const r1rt_config& cfg, const GuardParams& gp, char* err, std::size_t errlen) noexcept {
  auto fail = [&](int32_t code, const char* msg) {
    if (err != nullptr && errlen > 0) std::snprintf(err, errlen, "%s", msg);
    return code;
  };
  if (cfg.segment == nullptr) return fail(R1RT_E_INVALID_ARG, "guard: no segment (the core's, initialised)");
  if (cfg.writer_rate_hz == 0 || cfg.policy_rate_hz == 0 || gp.cycle_rate_hz == 0 || gp.stale_periods == 0)
    return fail(R1RT_E_INVALID_ARG, "guard: rates / stale_periods");
  if (!(gp.takeover_blend_s >= 0.0)) return fail(R1RT_E_INVALID_ARG, "guard: takeover_blend_s");
  if (!finite_array(cfg.stand_q, R1RT_NUM_JOINTS) || !finite_array(cfg.stand_kp, R1RT_NUM_JOINTS) ||
      !finite_array(cfg.stand_kd, R1RT_NUM_JOINTS) || !finite_array(cfg.upper_kp, R1RT_NUM_JOINTS) ||
      !finite_array(cfg.upper_kd, R1RT_NUM_JOINTS) || !finite_array(cfg.damp_kd, R1RT_NUM_JOINTS))
    return fail(R1RT_E_INVALID_ARG, "guard: non-finite stand / upper / damp values");
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    if (!(cfg.q_min[j] < cfg.q_max[j])) return fail(R1RT_E_INVALID_ARG, "guard: q_min >= q_max");
    if (!(cfg.kp_max[j] >= 0.0 && cfg.kd_max[j] >= 0.0 && cfg.tau_max[j] >= 0.0 && cfg.dq_max[j] > 0.0))
      return fail(R1RT_E_INVALID_ARG, "guard: limits");
  }
  if (cfg.policy_weights_path == nullptr)
    return fail(R1RT_E_POLICY_LOAD, "guard: no policy weights (a guard without a policy cannot balance)");
  cfg_ = cfg;
  cfg_.policy_weights_path = nullptr;  // read here only
  gp_ = gp;
  seg_ = static_cast<shm::Segment*>(cfg.segment);
  writer_period_ns_ = 1'000'000'000LL / cfg.writer_rate_hz;
  policy_period_ns_ = 1'000'000'000LL / cfg.policy_rate_hz;
  if (!(cfg_.stand_ramp_s > 0.0)) cfg_.stand_ramp_s = 0.4;
  if (cfg_.resume_gap_ns <= 0) cfg_.resume_gap_ns = R1RT_DEFAULT_RESUME_GAP_NS;  // session 8 (the core's default)
  // session 9 (U19 / D25): the state-age limit with the core's defaults (0 = default; stale_rung > 2 refused)
  if (cfg_.state_age_limit_ns <= 0) cfg_.state_age_limit_ns = R1RT_DEFAULT_STATE_AGE_LIMIT_NS;
  if (cfg_.state_age_sustain_ns <= 0) cfg_.state_age_sustain_ns = R1RT_DEFAULT_STATE_AGE_SUSTAIN_NS;
  if (!(cfg_.board_tick_hz > 0.0)) cfg_.board_tick_hz = R1RT_DEFAULT_BOARD_TICK_HZ;
  if (cfg_.stale_rung > 2) return fail(R1RT_E_INVALID_ARG, "guard: stale_rung must be 0, 1 or 2");
  age_clock_.configure(cfg_.board_tick_hz, cfg_.state_age_limit_ns / 2);
  resume_s_ = cfg_.takeover_blend_s > 0.0 ? cfg_.takeover_blend_s : 1.0;      // the core's blend length (default 1 s)
  if (!net_.load(cfg.policy_weights_path, err, errlen)) return R1RT_E_POLICY_LOAD;
  if (!runner_.init(cfg.policy, &net_, err, errlen)) return R1RT_E_POLICY_LOAD;
  runner_.set_action_filter(&Guard::filter_thunk, this);
  policy_loaded_ = true;
  rt_granted_ = cfg.rt_granted != 0;
  base_fill(&last_cmd_);
  st_ = GuardStatus{};
  st_.rt_beat_age_ns = -1;
  st_.lowstate_age_ns = -1;
  st_.mode = R1RT_MODE_MONITOR;
  st_.mode_from = R1RT_MODE_MONITOR;
  st_.act_mode = R1RT_MODE_MONITOR;
  st_.policy_loaded = 1;
  st_.armed = gp_.takeover_enabled != 0 ? 1 : 0;
  // session 6: the ON_STAND hold gains exactly as the core computes them (core.cpp init: legs + waist from on_stand_kp /
  // kd when any is given, else the stand gains) - where a takeover of a sequence into ON_STAND ends
  bool on_stand_given = false;
  for (int j = 0; j < kUpperFirst; ++j) on_stand_given = on_stand_given || cfg.on_stand_kp[j] != 0.0;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    hold_kp_[j] = on_stand_given ? cfg.on_stand_kp[j] : cfg.stand_kp[j];
    hold_kd_[j] = on_stand_given ? cfg.on_stand_kd[j] : cfg.stand_kd[j];
    if (!std::isfinite(hold_kp_[j]) || !std::isfinite(hold_kd_[j])) {
      hold_kp_[j] = cfg.stand_kp[j];
      hold_kd_[j] = cfg.stand_kd[j];
    }
  }
  fx_ = GuardForensics{};
  // session 6: damp requests posted before the guard attached are not its business (the core acted on them)
  damp_seen_ = seg_->stop.level[R1RT_STOP_DAMP].requests.load(std::memory_order_acquire);
  return R1RT_OK;
}

void Guard::filter_thunk(void* ctx, double* action, int n) noexcept {
  const Guard* g = static_cast<const Guard*>(ctx);
  for (int k = 0; k < g->n_inj_; ++k) {
    const Injection& in = g->inj_[k];
    if (g->now_ns_ < in.t0 || g->now_ns_ >= in.t1) continue;
    for (int i = 0; i < n; ++i) {
      if (in.kind == R1RT_INJECT_POLICY_NAN) action[i] = std::nan("");
      if (in.kind == R1RT_INJECT_POLICY_OUT_OF_RANGE) action[i] = in.value;
    }
  }
}

int32_t Guard::inject(std::int32_t kind, std::int64_t t0, std::int64_t t1, double value) noexcept {
  if (kind != R1RT_INJECT_POLICY_NAN && kind != R1RT_INJECT_POLICY_OUT_OF_RANGE) return R1RT_E_INVALID_ARG;
  if (n_inj_ >= kMaxInjections || t1 <= t0) return R1RT_E_INVALID_ARG;
  inj_[n_inj_++] = Injection{kind, t0, t1, value};
  return R1RT_OK;
}

bool Guard::fresh(std::int64_t t_ns) const noexcept {
  return have_state_ && t_ns - latest_.t_rx_ns <= cfg_.lowstate_stale_ns;
}

bool Guard::runner_from_mirror(const r1rt_mirror& mi, bool allow_blend) noexcept {
  // The mirror carries what the observation needs for a bumpless step: the core's last raw action (the "actions"
  // term), the gait phase (phase term; zeroed while standing anyway), the ramped lean, and the last targets. The
  // guard holds the lean where it is (lean target = the current lean: base command zero, posture kept).
  // BALANCE, and (session 5) the blend into BALANCE: the core's runner runs there too (from its RELEASED reset)
  const bool balancing = mi.mode == R1RT_MODE_BALANCE || (allow_blend && mi.mode == R1RT_MODE_TAKEOVER_BLEND &&
                                                           mi.takeover_target != R1RT_MODE_ON_STAND);
  if (mi.seq == 0 || !balancing || mi.policy_valid == 0) return false;
  const int na = runner_.n_act();
  if (!finite_array(mi.last_action, na) || !finite_array(mi.q_target, R1RT_NUM_JOINTS) ||
      !std::isfinite(mi.base_cmd[3]) || !std::isfinite(mi.gait_phase))
    return false;
  PolicyRunner::State s{};
  runner_.save(s);
  const int* idx = runner_.act_idx26();
  for (int k = 0; k < na; ++k) {
    s.last_action[k] = mi.last_action[k];
    s.target[k] = mi.q_target[idx[k]];
  }
  s.lean = mi.base_cmd[3];
  s.lean_target = mi.base_cmd[3];
  for (int i = 0; i < 2; ++i) {  // layout 2: the core's waist state (Manip3; 0 for Manip2-Lean)
    s.waist[i] = std::isfinite(mi.waist[i]) ? mi.waist[i] : 0.0;
    s.waist_target[i] = std::isfinite(mi.waist_target[i]) ? mi.waist_target[i] : 0.0;
  }
  s.phase_t = mi.gait_phase * runner_.params().period_s;
  s.t_last = 0.0;
  s.t_policy = 0.0;
  s.has_t_last = 0;
  s.has_t_policy = 0;
  runner_.restore(s);
  return true;
}

bool Guard::run_policy(std::int64_t t_ns, double* dev) noexcept {
  const r1rt_policy_params& p = runner_.params();
  const int nj = static_cast<int>(p.n_joints);
  double q[R1RT_POLICY_MAX_JOINTS];
  double dq[R1RT_POLICY_MAX_JOINTS];
  for (int i = 0; i < nj; ++i) {
    q[i] = latest_.q[p.joint_index[i]];
    dq[i] = latest_.dq[p.joint_index[i]];
  }
  const double cmd[3] = {0.0, 0.0, 0.0};  // DESIGN §4.3: base command zero
  now_ns_ = t_ns;
  runner_.advance_clock(static_cast<double>(t_ns) * 1e-9);
  const bool ok = runner_.act(q, dq, latest_.imu_quat, latest_.imu_gyro, cmd, nullptr, cfg_.action_abs_max);
  if (dev != nullptr) {
    double d = 0.0;
    if (ok) {
      const int* idx = runner_.act_idx26();
      for (int k = 0; k < runner_.n_act(); ++k) {
        const double e = std::fabs(runner_.target()[k] - mirror_.q_target[idx[k]]);
        if (e > d) d = e;
      }
    }
    *dev = d;
  }
  return ok;
}

void Guard::enter_freeze(std::int64_t t_ns, const double* q0, const double* kp0, const double* kd0) noexcept {
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    freeze_q0_[j] = std::isfinite(q0[j]) ? q0[j] : latest_.q[j];
    freeze_kp0_[j] = std::isfinite(kp0[j]) ? kp0[j] : cfg_.stand_kp[j];
    freeze_kd0_[j] = std::isfinite(kd0[j]) ? kd0[j] : cfg_.stand_kd[j];
  }
  freeze_t0_ = t_ns;
  freeze_ramp_s_ = cfg_.stand_ramp_s;
  act_mode_ = R1RT_MODE_STAND_FREEZE;
  blend_s_ = 0.0;
}

void Guard::take_over(std::int64_t t_ns, std::uint64_t token_word, const r1rt_heartbeat& hb) noexcept {
  active_ = true;
  token_word_ = token_word;
  const r1rt_mode from = have_mirror_ ? mirror_.mode : hb.mode;
  // The command the core was last sending (the mirror's targets and gains): the start of every ramp / blend and the
  // arms' and head's HOLD. A non-finite entry (never expected) falls back to the measured pose / the HOLD gains.
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    const bool ok = have_mirror_ && std::isfinite(mirror_.q_target[j]) && std::isfinite(mirror_.kp[j]) &&
                    std::isfinite(mirror_.kd[j]);
    from_q_[j] = ok ? mirror_.q_target[j] : latest_.q[j];
    from_kp_[j] = ok ? mirror_.kp[j] : (j >= kUpperFirst ? cfg_.upper_kp[j] : cfg_.stand_kp[j]);
    from_kd_[j] = ok ? mirror_.kd[j] : (j >= kUpperFirst ? cfg_.upper_kd[j] : cfg_.stand_kd[j]);
    // session 8 (HW1 fix #28): the core's last feed-forward (its arm gravity model's torque for the held pose);
    // a non-finite entry (never expected) -> 0; the limits clamp it as every command's (apply_limits)
    from_tau_[j] = have_mirror_ && std::isfinite(mirror_.tau_ff[j]) ? mirror_.tau_ff[j] : 0.0;
  }
  // Arms / head never limp under the guard (outside DAMP): a zero HOLD gain becomes the configured HOLD gain.
  for (int j = kUpperFirst; j < R1RT_NUM_JOINTS; ++j) {
    if (!(from_kp_[j] > 0.0)) from_kp_[j] = cfg_.upper_kp[j];
    if (!(from_kd_[j] > 0.0)) from_kd_[j] = cfg_.upper_kd[j];
  }
  blend_s_ = 0.0;
  stand_blend_ = false;
  policy_started_ = false;
  next_policy_ns_ = t_ns;  // the first policy step in this cycle
  // session 6: damp requests the core acknowledged were acted on by the core (its mode / mirror says DAMP then); the
  // ones after its last acknowledgement - e.g. posted during the stall - are the guard's to honour (operator only)
  {
    const std::uint64_t acked = seg_->stop.level[R1RT_STOP_DAMP].acked.load(std::memory_order_acquire);
    if (acked > damp_seen_) damp_seen_ = acked;
  }
  switch (from) {
    case R1RT_MODE_BALANCE:
      act_mode_ = R1RT_MODE_BALANCE;
      if (!runner_from_mirror(mirror_)) {  // no usable policy state to continue: rung 3 from the core's last command
        enter_freeze(t_ns, from_q_, from_kp_, from_kd_);
        freeze_invalid_ = true;
      }
      break;
    case R1RT_MODE_TAKEOVER_MIRROR:
    case R1RT_MODE_TAKEOVER_BLEND: {
      if (have_mirror_ && mirror_.takeover_target == R1RT_MODE_ON_STAND) {
        // session 5: a sequence into ON_STAND (the robot hanging on its stand): nothing balances there - never a
        // policy on a hanging robot. Session 6 (DESIGN §4.3 "continues the same ramp"; the soft start from Unitree's
        // zero torque): the core's command was (1 - a) snapshot + a hold, linear in a - the guard continues the GAINS
        // from the mirror's command at a_L to the ON_STAND hold gains over (1 - a_L) blend_ns (ending when the core's
        // blend would have), the targets held where the mirror had them (the soft snapshot is the measured pose, so
        // the core's own targets move by the pose's change between the gate and RELEASED only). MIRROR (a = 0, before
        // RELEASED): the whole blend from now. Session 5 held the mirror's command as it was - soft (kp 0) for good
        // after a stall early in an entry from zero torque.
        act_mode_ = R1RT_MODE_ON_STAND;
        stand_blend_ = true;
        if (from == R1RT_MODE_TAKEOVER_BLEND && mirror_.blend_ns > 0) {
          const double a_l = clampd(static_cast<double>(mirror_.t_ns - mirror_.blend_t0_ns) /
                                        static_cast<double>(mirror_.blend_ns), 0.0, 1.0);
          blend_s_ = (1.0 - a_l) * static_cast<double>(mirror_.blend_ns) * 1e-9;
          blend_t0_ = mirror_.t_ns;
        } else {
          blend_s_ = mirror_.blend_ns > 0 ? static_cast<double>(mirror_.blend_ns) * 1e-9 : gp_.takeover_blend_s;
          blend_t0_ = t_ns;
        }
        if (!(blend_s_ > 0.0)) blend_s_ = 0.0;
        if (blend_s_ == 0.0) {  // the blend already at its end: the hold gains now
          for (int j = 0; j < kUpperFirst; ++j) {
            from_kp_[j] = hold_kp_[j];
            from_kd_[j] = hold_kd_[j];
          }
        }
        break;
      }
      // The core was ramping from Unitree's command into its policy; the guard continues THAT ramp (session 5,
      // layout 2): the core's command was (1 - a) snapshot + a ours with a = (t - blend_t0) / blend_ns, and the
      // mirror holds it at a_L (the mirror's time). With the guard's policy as "ours", the same ramp continued from
      // the mirror's command is from + (a - a_L) / (1 - a_L) (ours - from): a blend that starts at the mirror's time
      // and lasts (1 - a_L) blend_ns, ending exactly when the core's would have. MIRROR (the blend not begun): the
      // core's whole blend length from now. The guard's runner continues the core's (BLEND) or starts fresh (MIRROR,
      // as at RELEASED).
      act_mode_ = R1RT_MODE_BALANCE;
      if (!runner_from_mirror(mirror_, true)) runner_.reset();
      const bool blending = have_mirror_ && from == R1RT_MODE_TAKEOVER_BLEND && mirror_.blend_ns > 0;
      if (blending) {
        const double a_l = clampd(static_cast<double>(mirror_.t_ns - mirror_.blend_t0_ns) /
                                      static_cast<double>(mirror_.blend_ns), 0.0, 1.0);
        blend_s_ = (1.0 - a_l) * static_cast<double>(mirror_.blend_ns) * 1e-9;
        blend_t0_ = mirror_.t_ns;
      } else {
        const double full_s =
            have_mirror_ && mirror_.blend_ns > 0 ? static_cast<double>(mirror_.blend_ns) * 1e-9 : gp_.takeover_blend_s;
        const double progress = clampd(static_cast<double>(hb.blend_progress), 0.0, 1.0);
        blend_s_ = full_s * (1.0 - progress);
        blend_t0_ = t_ns;
      }
      if (!(blend_s_ > 0.0)) blend_s_ = 0.0;
      break;
    }
    case R1RT_MODE_STAND_FREEZE:
      enter_freeze(t_ns, from_q_, from_kp_, from_kd_);  // the ramp from the core's last command to the stand, hold
      if (have_mirror_ && mirror_.freeze_t0_ns != 0 && mirror_.freeze_ramp_ns > 0) {
        // session 5 (layout 2): continue the core's ramp EXACTLY - it is linear towards a fixed stand, so from the
        // mirror's command at s_L = (mirror t - freeze_t0) / ramp the rest is the same line, starting at the
        // mirror's time and lasting (1 - s_L) ramp (ending when the core's would have)
        const double ramp_s = static_cast<double>(mirror_.freeze_ramp_ns) * 1e-9;
        const double s_l = clampd(static_cast<double>(mirror_.t_ns - mirror_.freeze_t0_ns) * 1e-9 / ramp_s, 0.0, 1.0);
        freeze_t0_ = mirror_.t_ns;
        freeze_ramp_s_ = (1.0 - s_l) * ramp_s;
        freeze_invalid_ = mirror_.freeze_invalid != 0;  // the core's reason stays visible
      }
      break;
    case R1RT_MODE_ON_STAND:
      act_mode_ = R1RT_MODE_ON_STAND;  // the last command held at its (stiff) gains
      break;
    case R1RT_MODE_DAMP:
      act_mode_ = R1RT_MODE_DAMP;  // the operator's state: it stays damp
      break;
    default:
      act_mode_ = R1RT_MODE_ON_STAND;  // not reached (take_over is called for core_publishes() modes only)
      break;
  }
  fx_.captured = 1;
  fx_.t_takeover_ns = t_ns;
  fx_.rt_beat_seq = hb.seq;
  fx_.rt_beat_t_ns = hb.t_ns;
  fx_.rt_silence_ns = t_ns - hb.t_ns;
  fx_.rt_mode = hb.mode;
  fx_.rt_faults = hb.faults;
  fx_.rt_lowstate_tick = hb.lowstate_tick;
  fx_.rt_rate_hz = hb.rate_hz;
  fx_.rt_granted = hb.rt_granted;
  fx_.mirror_seq = mirror_.seq;
  fx_.mirror_t_ns = mirror_.t_ns;
  fx_.mirror_last_action_t_ns = mirror_.last_action_t_ns;
  fx_.mirror_policy_valid = mirror_.policy_valid;
  fx_.mode_from = from;
  fx_.token_after = token_word;
  fx_.rt_pid = seg_->header.creator_pid;
  fx_.guard_lowstate_age_ns = have_state_ ? t_ns - latest_.t_rx_ns : -1;
  fx_.tilt_rad = tilt_;
  fx_.shadow_valid = st_.shadow_valid;
  fx_.shadow_dev_rad = st_.shadow_dev_rad;
  fx_.proc_err = -1;
  fx_.n_threads = 0;
  st_.takeovers += 1;
  st_.t_takeover_ns = t_ns;
  st_.mode_from = from;
  st_.event = kGuardEventTakeover;
  // session 8 (HW1 fix #29): a core silent longer than resume_gap_ns at the takeover means the guard was stalled too
  // (an alive guard takes over after stale_periods writer periods, ~10 ms): the board has (maybe) let go, the limbs
  // sagged - the core's last command at full stiffness would be the snap. The resume rule.
  resume_wait_ = false;
  resume_active_ = false;
  if (t_ns - hb.t_ns > cfg_.resume_gap_ns) {
    st_.resume_takeovers += 1;
    begin_resume(t_ns, from_q_);  // the core's last command vs the pose before the guard's own stop
  }
}

void Guard::begin_resume(std::int64_t t_ns, const double* cmd_q) noexcept {
  if (act_mode_ == R1RT_MODE_DAMP) return;  // the operator's damp: no target held
  // the pre-stop hold offset (as the core: core.cpp begin_resume) - the holding torque continuous at the resume
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    const double off = have_state_ ? cmd_q[j] - pre_stop_q_[j] : 0.0;
    resume_off_[j] = std::isfinite(off) ? clampd(off, -0.15, 0.15) : 0.0;
  }
  resume_wait_ = true;
  resume_active_ = false;
  resume_wait_t0_ = t_ns;
  resume_drain_counted_ = false;
  st_.resume_wait_samples_last = 0;  // session 9: the wait measured (GuardStatus)
  st_.resume_wait_old_samples_last = 0;
  st_.resume_wait_ns_last = 0;
}

void Guard::build(std::int64_t t_ns, r1rt_lowcmd_out* out) noexcept {
  const r1rt_policy_params& p = runner_.params();
  switch (act_mode_) {
    case R1RT_MODE_BALANCE: {
      if (stale_act_) {
        // session 9 (U19 / D25): the core's "hold the stand" rung - balancing on state older than the limit for the
        // sustain time -> the stand, from what it was sending (as its rung 3; nothing leaves it automatically)
        if (st_.event == kGuardEventNone) st_.event = kGuardEventFreeze;
        if (have_cmd_) {
          enter_freeze(t_ns, last_cmd_.q, last_cmd_.kp, last_cmd_.kd);
        } else {
          enter_freeze(t_ns, from_q_, from_kp_, from_kd_);
        }
        freeze_invalid_ = false;
        stale_freeze_ = true;
        st_.stale_rung_entries += 1;
        build(t_ns, out);
        return;
      }
      if (!fresh(t_ns)) {
        // no policy on stale state (as the core): the last command again (rung 4), or the core's last one
        if (have_cmd_) {
          *out = last_cmd_;
        } else {
          for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
            out->q[j] = from_q_[j];
            out->kp[j] = from_kp_[j];
            out->kd[j] = from_kd_[j];
          }
        }
        st_.stale_resends += 1;
        return;
      }
      if (t_ns >= next_policy_ns_) {
        const bool ok = run_policy(t_ns, nullptr);
        st_.policy_steps += 1;
        next_policy_ns_ = policy_started_ ? next_policy_ns_ + policy_period_ns_ : t_ns + policy_period_ns_;
        if (next_policy_ns_ <= t_ns) next_policy_ns_ = t_ns + policy_period_ns_;
        policy_started_ = true;
        if (!ok) {
          // rung 3 in the guard: its own action is invalid -> the stand, from what it was sending
          st_.invalid_actions += 1;
          if (st_.event == kGuardEventNone) st_.event = kGuardEventFreeze;
          if (have_cmd_) {
            enter_freeze(t_ns, last_cmd_.q, last_cmd_.kp, last_cmd_.kd);
          } else {
            enter_freeze(t_ns, from_q_, from_kp_, from_kd_);
          }
          freeze_invalid_ = true;  // nothing leaves STAND_FREEZE automatically (as in the core)
          build(t_ns, out);
          return;
        }
      }
      for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
        out->q[j] = latest_.q[j];
        out->kp[j] = p.kp_other[j];
        out->kd[j] = p.kd_other[j];
      }
      const int* idx = runner_.act_idx26();
      for (int k = 0; k < runner_.n_act(); ++k) out->q[idx[k]] = runner_.target()[k];
      for (std::uint32_t i = 0; i < p.n_joints; ++i) {
        out->kp[p.joint_index[i]] = p.kp[i];
        out->kd[p.joint_index[i]] = p.kd[i];
      }
      if (blend_s_ > 0.0) {
        const double s = clampd(static_cast<double>(t_ns - blend_t0_) * 1e-9 / blend_s_, 0.0, 1.0);
        for (int j = 0; j < kUpperFirst; ++j) {
          out->q[j] = from_q_[j] + s * (out->q[j] - from_q_[j]);
          out->kp[j] = from_kp_[j] + s * (out->kp[j] - from_kp_[j]);
          out->kd[j] = from_kd_[j] + s * (out->kd[j] - from_kd_[j]);
        }
        if (s >= 1.0) blend_s_ = 0.0;
      }
      break;
    }
    case R1RT_MODE_STAND_FREEZE: {
      if (freeze_invalid_) st_.faults |= R1RT_FAULT_POLICY_INVALID;  // the reason stays visible
      const double s =
          freeze_ramp_s_ > 0.0 ? clampd(static_cast<double>(t_ns - freeze_t0_) * 1e-9 / freeze_ramp_s_, 0.0, 1.0) : 1.0;
      for (int j = 0; j < kUpperFirst; ++j) {
        out->q[j] = freeze_q0_[j] + s * (cfg_.stand_q[j] - freeze_q0_[j]);
        out->kp[j] = freeze_kp0_[j] + s * (cfg_.stand_kp[j] - freeze_kp0_[j]);
        out->kd[j] = freeze_kd0_[j] + s * (cfg_.stand_kd[j] - freeze_kd0_[j]);
      }
      break;
    }
    case R1RT_MODE_ON_STAND: {
      // session 6: after a takeover of the sequence INTO ON_STAND, the core's gain ramp continued to the hold gains
      const double s = stand_blend_ && blend_s_ > 0.0
                           ? clampd(static_cast<double>(t_ns - blend_t0_) * 1e-9 / blend_s_, 0.0, 1.0)
                           : -1.0;
      for (int j = 0; j < kUpperFirst; ++j) {
        out->q[j] = from_q_[j];
        out->kp[j] = s < 0.0 ? from_kp_[j] : from_kp_[j] + s * (hold_kp_[j] - from_kp_[j]);
        out->kd[j] = s < 0.0 ? from_kd_[j] : from_kd_[j] + s * (hold_kd_[j] - from_kd_[j]);
      }
      if (s >= 1.0) {  // the ramp's end: hold at the hold gains from now on
        for (int j = 0; j < kUpperFirst; ++j) {
          from_kp_[j] = hold_kp_[j];
          from_kd_[j] = hold_kd_[j];
        }
        blend_s_ = 0.0;
        stand_blend_ = false;
      }
      break;
    }
    case R1RT_MODE_DAMP:
      for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
        out->q[j] = have_state_ ? latest_.q[j] : from_q_[j];
        out->kp[j] = 0.0;
        out->kd[j] = cfg_.damp_kd[j];
      }
      return;  // the arms / head damp too (the operator's state)
    default:
      return;
  }
  // arms + head: HOLD at the core's last targets, gains and feed-forward (no velocity). Session 8 (HW1 fix #28): the
  // feed-forward was 0 until session 8 - the arms lost their gravity feed-forward (+-1.2-1.6 N m at the shoulders) at
  // every takeover and moved 0.004-0.020 rad on the robot (T6, 11 Oct 2026); the core's arm model gives it for the
  // pose the guard holds (the core's last target), so the core's last value is the right one for the whole hold
  for (int j = kUpperFirst; j < R1RT_NUM_JOINTS; ++j) {
    out->q[j] = from_q_[j];
    out->dq[j] = 0.0;
    out->kp[j] = from_kp_[j];
    out->kd[j] = from_kd_[j];
    out->tau_ff[j] = from_tau_[j];
  }
}

void Guard::apply_limits(r1rt_lowcmd_out* out) noexcept {
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    double v = clampd(out->q[j], cfg_.q_min[j], cfg_.q_max[j]);
    if (v != out->q[j]) st_.faults |= R1RT_FAULT_JOINT_CLAMP;
    out->q[j] = v;
    v = clampd(out->dq[j], -cfg_.dq_max[j], cfg_.dq_max[j]);
    if (v != out->dq[j]) st_.faults |= R1RT_FAULT_VELOCITY_CLAMP;
    out->dq[j] = v;
    v = clampd(out->tau_ff[j], -cfg_.tau_max[j], cfg_.tau_max[j]);
    if (v != out->tau_ff[j]) st_.faults |= R1RT_FAULT_TORQUE_CLAMP;
    out->tau_ff[j] = v;
    out->kp[j] = clampd(out->kp[j], 0.0, cfg_.kp_max[j]);
    out->kd[j] = clampd(out->kd[j], 0.0, cfg_.kd_max[j]);
  }
}

void Guard::check_operator_damp(std::int64_t t_ns) noexcept {
  // Session 6 (the operator's damp must reach the motors while the guard owns them; DAMP stays operator-only):
  // (1) the stop channel's DAMP level: a request after damp_seen_ whose requester is the operator (last_requester is
  //     best effort - last writer wins - so a non-operator request racing the operator's in the same cycle can hide it:
  //     the operator repeats the word; a non-operator id is never acted on);
  // (2) a live core (fresh heartbeat; the token is the guard's, so the core publishes nothing) that entered DAMP: only
  //     the operator puts the core there (the mailbox refuses DAMP from anyone else, the stop channel ignores it).
  shm::StopLevel& lvl = seg_->stop.level[R1RT_STOP_DAMP];
  bool damp = false;
  const std::uint64_t req = lvl.requests.load(std::memory_order_acquire);
  if (req != damp_seen_) {
    damp_seen_ = req;
    if (lvl.last_requester.load(std::memory_order_relaxed) == R1RT_REQUESTER_OPERATOR) {
      damp = true;
    } else {
      st_.damp_ignored += 1;
      if (st_.event == kGuardEventNone) st_.event = kGuardEventDampIgnored;
    }
  }
  if (have_hb_ && hb_.mode == R1RT_MODE_DAMP) {
    const std::int64_t period = hb_.rate_hz > 0 ? 1'000'000'000LL / hb_.rate_hz : writer_period_ns_;
    if (t_ns - hb_.t_ns <= static_cast<std::int64_t>(gp_.stale_periods) * period) damp = true;
  }
  if (damp && act_mode_ != R1RT_MODE_DAMP) {
    act_mode_ = R1RT_MODE_DAMP;  // the operator's state: it stays damp (nothing leaves it automatically)
    blend_s_ = 0.0;
    stand_blend_ = false;
    st_.operator_damps += 1;
    st_.event = kGuardEventOperatorDamp;
  }
}

void Guard::rearm(std::int64_t t_ns) noexcept {
  // Session 6: the operator's hand-back (R1RT_REQ_RECLAIM, done by the core: CAS R1GUARD -> R1RT) - the token is r1rt's
  // again and its heartbeat is fresh. Back to standby: no command, the shadow warm again from the next cycle's mirror,
  // a new takeover possible on the next stale heartbeat.
  active_ = false;
  token_word_ = 0;
  act_mode_ = R1RT_MODE_MONITOR;
  have_cmd_ = false;
  resume_wait_ = false;  // session 8
  resume_active_ = false;
  blend_s_ = 0.0;
  stand_blend_ = false;
  freeze_invalid_ = false;
  stale_freeze_ = false;  // session 9
  policy_started_ = false;
  have_mirror_ = false;
  shadow_t_ns_ = -1;
  next_shadow_ns_ = t_ns;
  damp_seen_ = seg_->stop.level[R1RT_STOP_DAMP].requests.load(std::memory_order_acquire);
  st_.rearms += 1;
  st_.event = kGuardEventRearmed;
}

void Guard::write_heartbeat(std::int64_t t_ns) noexcept {
  r1rt_heartbeat hb{};
  hb.seq = ++hb_seq_;
  hb.t_ns = t_ns;
  hb.lowstate_tick = latest_.board_tick;
  hb.rate_hz = gp_.cycle_rate_hz;
  hb.faults = st_.faults;
  hb.blend_progress = 0.0f;
  if (active_ && blend_s_ > 0.0)
    hb.blend_progress = static_cast<float>(clampd(static_cast<double>(t_ns - blend_t0_) * 1e-9 / blend_s_, 0.0, 1.0));
  hb.mode = active_ ? R1RT_MODE_GUARDED : R1RT_MODE_MONITOR;
  hb.rt_granted = rt_granted_ ? 1 : 0;  // the process wrapper's grants (session 5; lockstep: 0)
  // session 6: armed / active / damp / not owner + the mode taken over from (r1d's status)
  std::uint32_t f = gp_.takeover_enabled != 0 ? R1RT_GUARD_FLAG_ARMED : 0u;
  if (active_) {
    f |= R1RT_GUARD_FLAG_ACTIVE;
    if (act_mode_ == R1RT_MODE_DAMP) f |= R1RT_GUARD_FLAG_DAMP;
    if (st_.faults & R1RT_FAULT_NOT_OWNER) f |= R1RT_GUARD_FLAG_NOT_OWNER;
    f |= (static_cast<std::uint32_t>(st_.mode_from) & 0x0Fu) << R1RT_GUARD_FLAG_MODE_FROM_SHIFT;
  }
  hb.guard_flags = static_cast<std::uint8_t>(f);
  seg_->guard_heartbeat.write(hb);
}

void Guard::push_record(std::int64_t t_ns, const r1rt_lowcmd_out& out) noexcept {
  // One record per cycle into the segment's guard_telemetry ring (layout 2; single producer = this guard, consumer =
  // r1log). A full ring drops the record and counts it (the guard never waits).
  r1rt_guard_record r{};
  r.seq = st_.cycles;
  r.t_ns = t_ns;
  r.t_state_rx_ns = have_state_ ? latest_.t_rx_ns : 0;
  r.rt_beat_age_ns = st_.rt_beat_age_ns;
  r.rt_beat_seq = have_hb_ ? hb_.seq : 0;
  r.board_tick = latest_.board_tick;
  r.faults = st_.faults;
  r.mode = st_.mode;
  r.mode_from = active_ ? st_.mode_from : (have_mirror_ ? mirror_.mode : static_cast<r1rt_mode>(R1RT_MODE_MONITOR));
  r.act_mode = st_.act_mode;
  r.token_owner = shm::OwnershipToken::owner_of(seg_->token.load());
  r.event = st_.event;
  r.active = active_ ? 1 : 0;
  r.published = out.publish;
  r.shadow_valid = st_.shadow_valid;
  r.shadow_dev_rad = static_cast<float>(st_.shadow_dev_rad);
  r.tilt_rad = static_cast<float>(tilt_);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    r.q_cmd[j] = static_cast<float>(out.q[j]);
    r.kp[j] = static_cast<float>(out.kp[j]);
    r.kd[j] = static_cast<float>(out.kd[j]);
  }
  if (!seg_->guard_telemetry.try_push(r)) st_.telemetry_drops += 1;
}

int32_t Guard::cycle(const r1rt_lowstate_in* state, std::int64_t t_ns, r1rt_lowcmd_out* out) noexcept {
  base_fill(out);
  st_.cycles += 1;
  st_.faults = 0;
  st_.event = kGuardEventNone;
  // session 8 (HW1 fix #29): its own silence while it holds the token (a stall of r1guard itself)
  const bool own_gap = have_cycle_ && t_ns - last_cycle_ns_ > cfg_.resume_gap_ns;  // stopped / stalled itself
  const bool own_stop = own_gap && active_;
  have_cycle_ = true;
  last_cycle_ns_ = t_ns;
  prev_tick_ = latest_.board_tick;
  prev_rx_ = latest_.t_rx_ns;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) pre_stop_q_[j] = latest_.q[j];
  had_state_ = have_state_;
  if (own_gap && had_state_) rclock_.mark_stop(prev_tick_, prev_rx_);

  // 1. its own lowstate, its age, the tilt
  if (state != nullptr && state->crc_ok) {
    latest_ = *state;
    have_state_ = true;
    if (!own_gap && !resume_wait_) rclock_.live_sample(latest_.board_tick, latest_.t_rx_ns);
  } else if (state != nullptr) {
    st_.faults |= R1RT_FAULT_CRC;
  }
  const std::int64_t age = have_state_ ? t_ns - latest_.t_rx_ns : -1;
  st_.lowstate_age_ns = age;
  if (!have_state_ || age > cfg_.lowstate_late_ns) st_.faults |= R1RT_FAULT_LOWSTATE_LATE;
  if (!have_state_ || age > cfg_.lowstate_stale_ns) st_.faults |= R1RT_FAULT_LOWSTATE_STALE;
  if (!have_state_ || age > cfg_.lowstate_lost_ns) st_.faults |= R1RT_FAULT_LOWSTATE_LOST;
  // session 9 (U19 / D25): the core's state-age rule on its own lowstate (receive age, or the board-tick content age)
  if (state != nullptr && state->crc_ok && (!had_state_ || latest_.t_rx_ns != prev_rx_))
    age_clock_.sample(latest_.board_tick, latest_.t_rx_ns);
  {
    const bool cvalid = have_state_ && age_clock_.valid();
    const std::int64_t content = cvalid ? age_clock_.age_ns(t_ns) : -1;
    st_.state_content_age_ns = content;
    const std::int64_t trig = (cvalid && content > age) ? content : age;
    stale_act_ = false;
    if (have_state_ && trig > cfg_.state_age_limit_ns) {
      st_.faults |= R1RT_FAULT_STATE_OLD;
      if (old_since_ns_ == 0) old_since_ns_ = t_ns;
      if (t_ns - old_since_ns_ >= cfg_.state_age_sustain_ns) {
        if (!old_counted_) st_.state_old_episodes += 1;
        old_counted_ = true;
        stale_act_ = cfg_.stale_rung != 0;
      }
    } else {
      old_since_ns_ = 0;
      old_counted_ = false;
    }
  }
  if (have_state_) {
    double g[3];
    projected_gravity(latest_.imu_quat, g);
    tilt_ = std::acos(clampd(-g[2], -1.0, 1.0));
    if (tilt_ > cfg_.fall_tilt_rad) st_.faults |= R1RT_FAULT_TILT_FALL;  // an event: the policy keeps running
  }
  st_.tilt_rad = tilt_;

  // 2. the core's heartbeat and mirror (a torn read keeps the previous copy)
  r1rt_heartbeat hb{};
  if (seg_->rt_heartbeat.try_read(hb) && hb.seq != 0) {
    hb_ = hb;
    have_hb_ = true;
  }
  st_.rt_beat_age_ns = have_hb_ ? t_ns - hb_.t_ns : -1;
  if (!active_) {
    r1rt_mirror mi{};
    if (seg_->mirror.try_read(mi) && mi.seq != 0) {
      mirror_ = mi;
      have_mirror_ = true;
    }
    if (shadow_t_ns_ >= 0 && have_mirror_ && mirror_.policy_valid != 0 && mirror_.last_action_t_ns == shadow_t_ns_) {
      // the core's action of the shadow's instant: the takeover would have continued it exactly this closely
      const int* idx = runner_.act_idx26();
      double e = 0.0;
      for (int k = 0; k < runner_.n_act(); ++k) {
        const double d = std::fabs(mirror_.q_target[idx[k]] - shadow_target_[k]);
        if (!(d <= e)) e = d;  // NaN-safe
      }
      st_.shadow_matches += 1;
      st_.shadow_match_err_rad = e;
      if (!(e <= st_.shadow_match_err_max_rad)) st_.shadow_match_err_max_rad = e;
      shadow_t_ns_ = -1;
    }

    // 3. the shadow policy at the policy rate (warm: restored from the mirror every time, so it follows the core)
    if (t_ns >= next_shadow_ns_) {
      next_shadow_ns_ = t_ns + policy_period_ns_;
      if (have_mirror_ && fresh(t_ns) && runner_from_mirror(mirror_)) {
        double dev = 0.0;
        const bool ok = run_policy(t_ns, &dev);
        st_.shadow_steps += 1;
        st_.shadow_valid = ok ? 1 : 0;
        if (ok) {
          st_.shadow_dev_rad = dev;
          if (dev > st_.shadow_dev_max_rad) st_.shadow_dev_max_rad = dev;
          // comparable with the core's action of this instant only if the mirror did not already carry it (then
          // the shadow computed the step AFTER the core's, from the core's new action)
          shadow_t_ns_ = mirror_.last_action_t_ns < t_ns ? t_ns : -1;
          for (int k = 0; k < runner_.n_act(); ++k) shadow_target_[k] = runner_.target()[k];
        } else {
          st_.shadow_invalid += 1;
          st_.event = kGuardEventShadowInvalid;
        }
      } else {
        st_.shadow_valid = 0;  // nothing to shadow (the core not balancing, or no fresh state)
      }
    }
    if (st_.shadow_valid == 0 && have_mirror_ && mirror_.mode == R1RT_MODE_BALANCE && fresh(t_ns))
      st_.faults |= R1RT_FAULT_POLICY_INVALID;  // the standby could not continue the core's policy right now

    // 4. the takeover: the core's heartbeat older than stale_periods of its writer period, the token the core's
    if (have_hb_ && gp_.takeover_enabled) {
      const std::int64_t period = hb_.rate_hz > 0 ? 1'000'000'000LL / hb_.rate_hz : writer_period_ns_;
      const bool stale = t_ns - hb_.t_ns > static_cast<std::int64_t>(gp_.stale_periods) * period;
      if (stale) {
        std::uint64_t w = seg_->token.load();
        const r1rt_mode from = have_mirror_ ? mirror_.mode : hb_.mode;
        if (shm::OwnershipToken::owner_of(w) == R1RT_OWNER_R1RT) {
          if (!core_publishes(from)) {
            st_.event = kGuardEventModeNotHandled;
          } else {
            const std::uint64_t before = w;
            if (seg_->token.try_transfer(w, R1RT_OWNER_R1GUARD)) {
              take_over(t_ns, w, hb_);
              fx_.token_before = before;
            } else {
              st_.cas_failures += 1;
              st_.event = kGuardEventCasFailed;
            }
          }
        }
      }
    }
  }

  // 5. the command (active only), the limits, the token immediately before the write. Session 6: the token gone
  //    to r1rt with the core alive = the operator's hand-back -> re-arm; otherwise the operator's damp is honoured.
  if (active_ && !seg_->token.holds(token_word_)) {
    const std::uint64_t w = seg_->token.load();
    const std::int64_t period = have_hb_ && hb_.rate_hz > 0 ? 1'000'000'000LL / hb_.rate_hz : writer_period_ns_;
    const bool core_alive = have_hb_ && t_ns - hb_.t_ns <= static_cast<std::int64_t>(gp_.stale_periods) * period;
    if (shm::OwnershipToken::owner_of(w) == R1RT_OWNER_R1RT && core_alive) rearm(t_ns);
  }
  if (active_ && own_stop) {
    st_.stops += 1;
    begin_resume(t_ns, have_cmd_ ? last_cmd_.q : from_q_);
  }
  bool hold_off = false;  // session 8: a resume without fresh state yet - nothing published (the old command = the snap)
  if (active_ && resume_wait_) {
    const bool fresh_now = have_state_ && age >= 0 && age <= cfg_.lowstate_late_ns;
    // session 9 (#29's open race): as the core - the first live sample ends the wait (no 10 ms floor once the tick
    // rate is known), so the guard's first write after a whole-machine stop overrides a stale core packet at once
    const bool settled = rclock_.has_rate() || t_ns - resume_wait_t0_ >= R1RT_RESUME_SETTLE_NS;
    const bool caught = rclock_.caught_up(latest_.board_tick, t_ns);
    const bool live = fresh_now && settled && caught;
    const bool late = t_ns - resume_wait_t0_ > R1RT_RESUME_STATE_WAIT_NS;
    // session 9 (#29 hazard ii, as the core): past the 50 ms, a fresh sample whose tick is provably behind is the
    // stop's backlog still draining (its pose from the stop's start) - keep holding back, up to the drain limit
    const bool draining = late && fresh_now && rclock_.has_rate() && !caught &&
                          t_ns - resume_wait_t0_ <= R1RT_RESUME_DRAIN_MAX_NS;
    if (draining && !resume_drain_counted_) {
      st_.resume_drain_waits += 1;
      resume_drain_counted_ = true;
    }
    if (state != nullptr) {
      st_.resume_wait_samples_last += 1;
      if (!caught) st_.resume_wait_old_samples_last += 1;
    }
    st_.resume_wait_ns_last = t_ns - resume_wait_t0_;
    if (live || (late && fresh_now && !draining)) {
      if (!live) st_.resume_uncaught += 1;
      resume_wait_ = false;
      resume_active_ = true;
      resume_t0_ = t_ns;
      for (int j = 0; j < R1RT_NUM_JOINTS; ++j) resume_q0_[j] = latest_.q[j] + resume_off_[j];
      st_.resume_blends += 1;
      if (act_mode_ == R1RT_MODE_STAND_FREEZE) {  // the freeze ramp again from the measured pose (the gains last sent)
        enter_freeze(t_ns, resume_q0_, have_cmd_ ? last_cmd_.kp : from_kp_, have_cmd_ ? last_cmd_.kd : from_kd_);
      }
    } else if (late && !draining) {
      resume_wait_ = false;
      st_.resume_waits_expired += 1;
    } else if (act_mode_ == R1RT_MODE_ON_STAND || act_mode_ == R1RT_MODE_STAND_FREEZE) {
      hold_off = true;
    }
  }
  if (active_ && hold_off) st_.faults |= R1RT_FAULT_LOOP_GAP;
  if (active_ && !hold_off) {
    if (seg_->token.holds(token_word_)) check_operator_damp(t_ns);
    build(t_ns, out);
    if (resume_active_) {  // the blend from the measured pose to the hold's targets (gains / feed-forward unchanged)
      const double a = clampd(static_cast<double>(t_ns - resume_t0_) * 1e-9 / resume_s_, 0.0, 1.0);
      const int first = act_mode_ == R1RT_MODE_ON_STAND ? 0 : (act_mode_ == R1RT_MODE_DAMP ? R1RT_NUM_JOINTS : kUpperFirst);
      for (int j = first; j < R1RT_NUM_JOINTS; ++j) {
        out->q[j] = (1.0 - a) * resume_q0_[j] + a * out->q[j];
        out->dq[j] = a * out->dq[j];
      }
      if (a >= 1.0) resume_active_ = false;
    }
    apply_limits(out);
    if (seg_->token.holds(token_word_)) {
      out->publish = 1;
      for (int j = 0; j < R1RT_NUM_JOINTS; ++j) out->enable[j] = 1;
      last_cmd_ = *out;
      have_cmd_ = true;
      st_.publishes += 1;
    } else {
      out->publish = 0;
      st_.faults |= R1RT_FAULT_NOT_OWNER;
      st_.token_lost_cycles += 1;
      if (st_.event == kGuardEventNone) st_.event = kGuardEventTokenLost;
    }
  }

  // 6. its heartbeat, its status
  st_.faults_seen |= st_.faults;
  st_.active = active_ ? 1 : 0;
  st_.mode = active_ ? R1RT_MODE_GUARDED : R1RT_MODE_MONITOR;
  st_.act_mode = active_ ? act_mode_ : static_cast<r1rt_mode>(R1RT_MODE_MONITOR);  // GCC: -Werror=extra
  st_.resume_waiting = resume_wait_ ? 1 : 0;
  st_.stale_freeze = (active_ && act_mode_ == R1RT_MODE_STAND_FREEZE && stale_freeze_) ? 1 : 0;  // session 9
  write_heartbeat(t_ns);
  push_record(t_ns, *out);
  return R1RT_OK;
}

}  // namespace r1rt
