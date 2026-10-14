// r1rt_py.Guard: the Python binding of r1rt::Guard (r1guard, guard.hpp) for the simulator's lockstep (simulation /
// validation only; r1sim/v3_guard.py drives it). The guard runs on the SAME segment as an r1rt_py.Core: the module
// init in r1rt_py.cpp calls bind_guard(m, segment_of) with a function that returns a Core's segment.
//
// The dict -> r1rt_config / lowstate helpers below repeat r1rt_py.cpp's (those are file-local there); the keys are
// the same (r1sim.v3_core.core_config), so the guard runs exactly the core's profile.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>

#include "r1rt/guard.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"

namespace nb = nanobind;

namespace {

using Vec26 = nb::ndarray<nb::numpy, double, nb::shape<R1RT_NUM_JOINTS>>;

void* (*g_segment_of)(nb::handle) = nullptr;  // set by bind_guard (r1rt_py.cpp: a Core -> its segment)

void read_vec(nb::handle h, double* out, std::size_t n, const char* what) {
  nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> a;
  if (nb::try_cast(h, a, false) && a.shape(0) == n) {
    std::memcpy(out, a.data(), n * sizeof(double));
    return;
  }
  std::size_t i = 0;
  for (nb::handle v : h) {
    if (i >= n) throw nb::value_error((std::string(what) + ": too many values").c_str());
    out[i++] = nb::cast<double>(v);
  }
  if (i != n) throw nb::value_error((std::string(what) + ": expected " + std::to_string(n) + " values").c_str());
}

template <typename I>
void read_ivec(nb::handle h, I* out, std::size_t n, const char* what) {
  double tmp[64];
  read_vec(h, tmp, n, what);
  for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<I>(tmp[i]);
}

double get_d(const nb::dict& d, const char* k) {
  if (!d.contains(k)) throw nb::key_error(k);
  return nb::cast<double>(d[k]);
}
std::int64_t get_i(const nb::dict& d, const char* k) {
  if (!d.contains(k)) throw nb::key_error(k);
  return nb::cast<std::int64_t>(d[k]);
}

r1rt_policy_params read_params(const nb::dict& p) {
  r1rt_policy_params q{};
  q.n_joints = static_cast<std::uint32_t>(get_i(p, "n_joints"));
  q.n_act = static_cast<std::uint32_t>(get_i(p, "n_act"));
  q.n_terms = static_cast<std::uint32_t>(get_i(p, "n_terms"));
  q.has_lean = static_cast<std::uint32_t>(get_i(p, "has_lean"));
  q.has_waist = static_cast<std::uint32_t>(get_i(p, "has_waist"));
  if (q.n_joints > R1RT_POLICY_MAX_JOINTS || q.n_act > R1RT_NUM_POLICY_ACTIONS || q.n_terms > R1RT_POLICY_MAX_TERMS)
    throw nb::value_error("policy params: sizes out of range");
  read_ivec(p["obs_terms"], q.obs_terms, q.n_terms, "obs_terms");
  read_ivec(p["joint_index"], q.joint_index, q.n_joints, "joint_index");
  read_ivec(p["act_pos"], q.act_pos, q.n_act, "act_pos");
  read_vec(p["default_pos"], q.default_pos, q.n_joints, "default_pos");
  read_vec(p["kp"], q.kp, q.n_joints, "kp");
  read_vec(p["kd"], q.kd, q.n_joints, "kd");
  read_vec(p["action_scale"], q.action_scale, q.n_act, "action_scale");
  read_vec(p["kp_other"], q.kp_other, R1RT_NUM_JOINTS, "kp_other");
  read_vec(p["kd_other"], q.kd_other, R1RT_NUM_JOINTS, "kd_other");
  q.period_s = get_d(p, "period_s");
  q.step_dt_s = get_d(p, "step_dt_s");
  read_vec(p["cmd_lo"], q.cmd_lo, 3, "cmd_lo");
  read_vec(p["cmd_hi"], q.cmd_hi, 3, "cmd_hi");
  q.lean_lo = get_d(p, "lean_lo");
  q.lean_hi = get_d(p, "lean_hi");
  q.lean_rate = get_d(p, "lean_rate");
  read_vec(p["waist_lo"], q.waist_lo, 2, "waist_lo");
  read_vec(p["waist_hi"], q.waist_hi, 2, "waist_hi");
  read_vec(p["waist_rate"], q.waist_rate, 2, "waist_rate");
  return q;
}

Vec26 vec26(const double* src) {
  double* out = new double[R1RT_NUM_JOINTS];
  std::memcpy(out, src, sizeof(double) * R1RT_NUM_JOINTS);
  nb::capsule owner(out, [](void* q) noexcept { delete[] static_cast<double*>(q); });
  return Vec26(out, {static_cast<std::size_t>(R1RT_NUM_JOINTS)}, owner);
}

struct PyGuard {
  nb::object core_ref;  // keeps the Core (the owner of the segment) alive
  std::string weights;
  r1rt::Guard guard;
  r1rt_lowcmd_out last{};

  PyGuard(const nb::dict& c, nb::handle core, const nb::dict& params) {
    if (g_segment_of == nullptr) throw std::runtime_error("r1rt_py.Guard: module not initialised");
    void* seg = g_segment_of(core);  // raises TypeError unless `core` is an r1rt_py.Core
    core_ref = nb::borrow(core);
    r1rt_config cfg{};
    cfg.abi_version = R1RT_ABI_VERSION;
    cfg.struct_size = sizeof(r1rt_config);
    cfg.writer_rate_hz = static_cast<std::uint32_t>(get_i(c, "writer_hz"));
    cfg.policy_rate_hz = static_cast<std::uint32_t>(get_i(c, "policy_hz"));
    read_vec(c["q_min"], cfg.q_min, R1RT_NUM_JOINTS, "q_min");
    read_vec(c["q_max"], cfg.q_max, R1RT_NUM_JOINTS, "q_max");
    read_vec(c["dq_max"], cfg.dq_max, R1RT_NUM_JOINTS, "dq_max");
    read_vec(c["tau_max"], cfg.tau_max, R1RT_NUM_JOINTS, "tau_max");
    read_vec(c["kp_max"], cfg.kp_max, R1RT_NUM_JOINTS, "kp_max");
    read_vec(c["kd_max"], cfg.kd_max, R1RT_NUM_JOINTS, "kd_max");
    cfg.stop_time_s = get_d(c, "stop_time_s");
    cfg.arm_tilt_stop_rad = get_d(c, "arm_tilt_stop_rad");
    cfg.fall_tilt_rad = get_d(c, "fall_tilt_rad");
    cfg.lowstate_late_ns = get_i(c, "lowstate_late_ns");
    cfg.lowstate_stale_ns = get_i(c, "lowstate_stale_ns");
    cfg.lowstate_lost_ns = get_i(c, "lowstate_lost_ns");
    cfg.policy_max_age_ns = get_i(c, "policy_max_age_ns");
    cfg.segment = seg;
    cfg.segment_size = r1rt_segment_size();
    if (c.contains("weights") && !c["weights"].is_none()) {
      weights = nb::cast<std::string>(c["weights"]);
      cfg.policy_weights_path = weights.c_str();
    }
    cfg.policy = read_params(nb::cast<nb::dict>(c["policy"]));
    read_vec(c["stand_q"], cfg.stand_q, R1RT_NUM_JOINTS, "stand_q");
    read_vec(c["stand_kp"], cfg.stand_kp, R1RT_NUM_JOINTS, "stand_kp");
    read_vec(c["stand_kd"], cfg.stand_kd, R1RT_NUM_JOINTS, "stand_kd");
    cfg.stand_ramp_s = get_d(c, "stand_ramp_s");
    read_vec(c["upper_kp"], cfg.upper_kp, R1RT_NUM_JOINTS, "upper_kp");
    read_vec(c["upper_kd"], cfg.upper_kd, R1RT_NUM_JOINTS, "upper_kd");
    read_vec(c["damp_kd"], cfg.damp_kd, R1RT_NUM_JOINTS, "damp_kd");
    cfg.action_abs_max = get_d(c, "action_abs_max");
    cfg.loop_gap_ns = get_i(c, "loop_gap_ns");
    if (c.contains("resume_gap_ns")) cfg.resume_gap_ns = get_i(c, "resume_gap_ns");  // session 8 (fix #29)
    if (c.contains("takeover_blend_s")) cfg.takeover_blend_s = get_d(c, "takeover_blend_s");
    if (c.contains("rt_granted")) cfg.rt_granted = static_cast<std::uint32_t>(get_i(c, "rt_granted"));  // session 5
    // session 9 (U19 / D25): the state-age limit + the stale rung (absent = the defaults, the rung OFF)
    if (c.contains("state_age_limit_ns")) cfg.state_age_limit_ns = get_i(c, "state_age_limit_ns");
    if (c.contains("state_age_sustain_ns")) cfg.state_age_sustain_ns = get_i(c, "state_age_sustain_ns");
    if (c.contains("board_tick_hz")) cfg.board_tick_hz = get_d(c, "board_tick_hz");
    if (c.contains("stale_rung")) cfg.stale_rung = static_cast<std::uint32_t>(get_i(c, "stale_rung"));
    r1rt::GuardParams gp;
    if (params.contains("cycle_hz")) gp.cycle_rate_hz = static_cast<std::uint32_t>(get_i(params, "cycle_hz"));
    if (params.contains("stale_periods"))
      gp.stale_periods = static_cast<std::uint32_t>(get_i(params, "stale_periods"));
    if (params.contains("takeover_blend_s")) gp.takeover_blend_s = get_d(params, "takeover_blend_s");
    if (params.contains("takeover_enabled"))
      gp.takeover_enabled = static_cast<std::uint8_t>(get_i(params, "takeover_enabled") != 0 ? 1 : 0);
    char err[512] = {0};
    const std::int32_t rc = guard.init(cfg, gp, err, sizeof err);
    if (rc != R1RT_OK) throw std::runtime_error("r1rt::Guard::init: status " + std::to_string(rc) + ": " + err);
  }

  std::tuple<int, Vec26, Vec26, Vec26, Vec26, Vec26> cycle(std::int64_t t_ns, nb::handle state) {
    r1rt_lowstate_in in{};
    const bool has = !state.is_none();
    if (has) {
      nb::dict d = nb::cast<nb::dict>(state);
      in.t_rx_ns = get_i(d, "t_rx_ns");
      in.board_tick = static_cast<std::uint32_t>(get_i(d, "board_tick"));
      in.mode_pr = R1RT_MODE_PR;
      in.mode_machine = R1RT_MODE_MACHINE_R1;
      in.crc_ok = 1;
      read_vec(d["q"], in.q, R1RT_NUM_JOINTS, "q");
      read_vec(d["dq"], in.dq, R1RT_NUM_JOINTS, "dq");
      read_vec(d["tau_est"], in.tau_est, R1RT_NUM_JOINTS, "tau_est");
      read_vec(d["quat"], in.imu_quat, 4, "quat");
      read_vec(d["gyro"], in.imu_gyro, 3, "gyro");
      read_vec(d["acc"], in.imu_acc, 3, "acc");
      if (d.contains("motor_state")) read_ivec(d["motor_state"], in.motor_state, R1RT_NUM_JOINTS, "motor_state");
    }
    {
      nb::gil_scoped_release release;
      guard.cycle(has ? &in : nullptr, t_ns, &last);
    }
    return {static_cast<int>(last.publish), vec26(last.q), vec26(last.dq), vec26(last.kp), vec26(last.kd),
            vec26(last.tau_ff)};
  }

  nb::dict status() const {
    r1rt::GuardStatus s{};
    guard.status(&s);
    nb::dict d;
    d["cycles"] = s.cycles;
    d["publishes"] = s.publishes;
    d["shadow_steps"] = s.shadow_steps;
    d["shadow_invalid"] = s.shadow_invalid;
    d["policy_steps"] = s.policy_steps;
    d["invalid_actions"] = s.invalid_actions;
    d["takeovers"] = s.takeovers;
    d["cas_failures"] = s.cas_failures;
    d["token_lost_cycles"] = s.token_lost_cycles;
    d["stale_resends"] = s.stale_resends;
    d["telemetry_drops"] = s.telemetry_drops;
    d["rt_beat_age_ns"] = s.rt_beat_age_ns;
    d["lowstate_age_ns"] = s.lowstate_age_ns;
    d["t_takeover_ns"] = s.t_takeover_ns;
    d["shadow_dev_rad"] = s.shadow_dev_rad;
    d["shadow_dev_max_rad"] = s.shadow_dev_max_rad;
    d["shadow_matches"] = s.shadow_matches;
    d["shadow_match_err_rad"] = s.shadow_match_err_rad;
    d["shadow_match_err_max_rad"] = s.shadow_match_err_max_rad;
    d["tilt_rad"] = s.tilt_rad;
    d["faults"] = s.faults;
    d["faults_seen"] = s.faults_seen;
    d["mode"] = static_cast<int>(s.mode);
    d["mode_from"] = static_cast<int>(s.mode_from);
    d["act_mode"] = static_cast<int>(s.act_mode);
    d["active"] = static_cast<int>(s.active);
    d["shadow_valid"] = static_cast<int>(s.shadow_valid);
    d["event"] = static_cast<int>(s.event);
    d["policy_loaded"] = static_cast<int>(s.policy_loaded);
    d["rearms"] = s.rearms;  // session 6
    d["operator_damps"] = s.operator_damps;
    d["damp_ignored"] = s.damp_ignored;
    d["armed"] = static_cast<int>(s.armed);
    d["stops"] = s.stops;  // session 8 (fix #29)
    d["resume_takeovers"] = s.resume_takeovers;
    d["resume_blends"] = s.resume_blends;
    d["resume_waits_expired"] = s.resume_waits_expired;
    d["resume_uncaught"] = s.resume_uncaught;
    d["resume_drain_waits"] = s.resume_drain_waits;  // session 9 (#29 hazard ii)
    d["resume_wait_ns_last"] = s.resume_wait_ns_last;
    d["resume_wait_samples_last"] = s.resume_wait_samples_last;
    d["resume_wait_old_samples_last"] = s.resume_wait_old_samples_last;
    d["resume_waiting"] = static_cast<int>(s.resume_waiting);
    d["state_content_age_ns"] = s.state_content_age_ns;  // session 9 (the state-age limit)
    d["state_old_episodes"] = s.state_old_episodes;
    d["stale_rung_entries"] = s.stale_rung_entries;
    d["stale_freeze"] = static_cast<int>(s.stale_freeze);
    return d;
  }

  nb::dict forensics() const {
    const r1rt::GuardForensics& f = guard.forensics();
    nb::dict d;
    d["captured"] = static_cast<int>(f.captured);
    d["t_takeover_ns"] = f.t_takeover_ns;
    d["rt_beat_seq"] = f.rt_beat_seq;
    d["rt_beat_t_ns"] = f.rt_beat_t_ns;
    d["rt_silence_ns"] = f.rt_silence_ns;
    d["rt_mode"] = static_cast<int>(f.rt_mode);
    d["rt_faults"] = f.rt_faults;
    d["rt_lowstate_tick"] = f.rt_lowstate_tick;
    d["rt_rate_hz"] = f.rt_rate_hz;
    d["rt_granted"] = static_cast<int>(f.rt_granted);
    d["mirror_seq"] = f.mirror_seq;
    d["mirror_t_ns"] = f.mirror_t_ns;
    d["mirror_last_action_t_ns"] = f.mirror_last_action_t_ns;
    d["mirror_policy_valid"] = static_cast<int>(f.mirror_policy_valid);
    d["mode_from"] = static_cast<int>(f.mode_from);
    d["token_before"] = f.token_before;
    d["token_after"] = f.token_after;
    d["rt_pid"] = f.rt_pid;
    d["guard_lowstate_age_ns"] = f.guard_lowstate_age_ns;
    d["tilt_rad"] = f.tilt_rad;
    d["shadow_valid"] = static_cast<int>(f.shadow_valid);
    d["shadow_dev_rad"] = f.shadow_dev_rad;
    d["proc_err"] = f.proc_err;
    nb::list threads;
    for (int i = 0; i < f.n_threads; ++i) {
      const r1rt::GuardProcThread& t = f.threads[i];
      nb::dict td;
      td["tid"] = t.tid;
      td["err"] = t.err;
      td["comm"] = std::string(t.comm);
      td["wchan"] = std::string(t.wchan);
      td["syscall"] = std::string(t.syscall);
      td["stack"] = std::string(t.stack);
      threads.append(td);
    }
    d["threads"] = threads;
    return d;
  }
};

}  // namespace

// Called once from r1rt_py.cpp's module init (after Core is registered).
void bind_guard(nb::module_& m, void* (*segment_of)(nb::handle)) {
  g_segment_of = segment_of;
  m.attr("GUARD_EVENT_TAKEOVER") = static_cast<int>(r1rt::kGuardEventTakeover);
  m.attr("GUARD_EVENT_SHADOW_INVALID") = static_cast<int>(r1rt::kGuardEventShadowInvalid);
  m.attr("GUARD_EVENT_FREEZE") = static_cast<int>(r1rt::kGuardEventFreeze);
  nb::class_<PyGuard>(m, "Guard")
      .def(nb::init<const nb::dict&, nb::handle, const nb::dict&>(), nb::arg("config"), nb::arg("core"),
           nb::arg("params") = nb::dict(),
           "r1guard on `core`'s segment; config = the core's (r1sim.v3_core.core_config); params: cycle_hz (1000), "
           "stale_periods (5), takeover_blend_s (0.3), takeover_enabled (1)")
      .def("cycle", &PyGuard::cycle, nb::arg("t_ns"), nb::arg("state").none(),
           "One guard cycle: state dict (or None) -> (publish, q, dq, kp, kd, tau_ff)")
      .def(
          "inject",
          [](PyGuard& g, int kind, std::int64_t t0, std::int64_t t1, double value) {
            return g.guard.inject(kind, t0, t1, value);
          },
          nb::arg("kind"), nb::arg("t_start_ns"), nb::arg("t_end_ns"), nb::arg("value") = 0.0,
          "fault injection on the guard's own network output (INJECT_POLICY_NAN / _OUT_OF_RANGE)")
      .def("status", &PyGuard::status)
      .def("forensics", &PyGuard::forensics)
      .def(
          "capture_proc",
          [](PyGuard& g, std::int32_t pid) {
            return r1rt::capture_proc_forensics(pid, &g.guard.forensics_for_capture());
          },
          nb::arg("pid"), "Linux: /proc/<pid>/task/* into the forensics (ENOSYS elsewhere); outside the cycle")
      .def_prop_ro("active", [](const PyGuard& g) { return g.guard.active(); });
}
