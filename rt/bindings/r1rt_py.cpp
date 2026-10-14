// r1rt_py: the Python binding of the C++ real-time core for the simulator's lockstep (simulation / validation only).
//
// Session 3: r1rt_py.Core (the step C ABI: writer step, policy-thread iteration, requests, status, fault injection;
// r1sim/v3_core.py drives it) and r1rt_py.PolicyRunner (r1rt::PolicyRunner, the observation + schedule + action
// logic; the golden test against r1mc3.policy.ReferencePolicy). Both release the GIL around the C++ calls.
//
// Session 2: r1rt::PolicyMlp:
//   import r1rt_py                      # sim/r1sim/cpp_policy.py finds the module in the C++ build tree
//   net = r1rt_py.PolicyMlp()
//   net.load("policy_weights.r1w")      # tools/policy_convert.py output; raises RuntimeError with the C++ reason
//   act = net.run(obs)                  # obs: float64 (74,) -> float64 (14,), the raw network output
//
// Simulation / validation only: the robot's RT path never loads Python (DESIGN.md §3.1). run() releases the GIL
// around the network evaluation (PolicyMlp::run is const and keeps no mutable state, so concurrent calls on one
// loaded instance are safe); the only copy is the (14,) output array. run_into() writes into a caller's array
// instead (no allocation per call).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <tuple>

#include "r1rt/policy_mlp.hpp"
#include "r1rt/hg_msgs.hpp"
#include "r1rt/policy_runner.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"

namespace nb = nanobind;
using r1rt::PolicyMlp;

namespace {

using ObsIn = nb::ndarray<const double, nb::shape<PolicyMlp::kObs>, nb::c_contig, nb::device::cpu>;
using ActOut = nb::ndarray<double, nb::shape<PolicyMlp::kAct>, nb::c_contig, nb::device::cpu>;
using ActArray = nb::ndarray<nb::numpy, double, nb::shape<PolicyMlp::kAct>>;
using ObsIn26 = nb::ndarray<const double, nb::shape<R1RT_NUM_JOINTS>, nb::c_contig, nb::device::cpu>;

void require_loaded(const PolicyMlp& p) {
  if (!p.loaded()) throw std::runtime_error("PolicyMlp: no weights loaded (call load(path) first)");
}

std::string hex(const std::array<std::uint8_t, 32>& b) {
  static const char* digits = "0123456789abcdef";
  std::string s;
  s.reserve(64);
  for (std::uint8_t v : b) {
    s.push_back(digits[v >> 4]);
    s.push_back(digits[v & 0xF]);
  }
  return s;
}

using Vec26 = nb::ndarray<nb::numpy, double, nb::shape<R1RT_NUM_JOINTS>>;

// A Python sequence / numpy array of numbers -> n doubles (exactly n, else ValueError).
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

// The policy params dict (r1sim.v3_core.policy_params) -> r1rt_policy_params; only the first n entries of each
// array are given (the rest stay 0).
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

template <std::size_t N>
nb::ndarray<nb::numpy, double, nb::ndim<1>> vecn(const double* src, std::size_t n) {
  double* out = new double[N];
  std::memcpy(out, src, sizeof(double) * n);
  nb::capsule owner(out, [](void* q) noexcept { delete[] static_cast<double*>(q); });
  return nb::ndarray<nb::numpy, double, nb::ndim<1>>(out, {n}, owner);
}

// An upper-body record (r1sim.v3_core builds it) -> r1rt_upper_cmd. Keys: cmd_id, group, source, n_joints; optional:
// stop_epoch (default: the segment's current epoch), t_issue_ns, t_start_ns, ttl_ns, segment_index, last_segment,
// q / dq / kp / kd / tau_ff (n_joints each), knots [(t_ns, q, dq, ddq), ...] (<= R1RT_MAX_KNOTS), guards {enable,
// lag_rad, tau_nm, contact_mask, contact_all, hand, surface_z_m}.
r1rt_upper_cmd read_upper(const nb::dict& d, std::uint64_t epoch_now) {
  r1rt_upper_cmd c{};
  auto opt_i = [&](const char* k, std::int64_t dflt) { return d.contains(k) ? get_i(d, k) : dflt; };
  c.cmd_id = static_cast<std::uint64_t>(get_i(d, "cmd_id"));
  c.stop_epoch = static_cast<std::uint64_t>(opt_i("stop_epoch", static_cast<std::int64_t>(epoch_now)));
  c.t_issue_ns = opt_i("t_issue_ns", 0);
  c.t_start_ns = opt_i("t_start_ns", 0);
  c.ttl_ns = opt_i("ttl_ns", 0);
  c.group = static_cast<r1rt_group>(get_i(d, "group"));
  c.source = static_cast<r1rt_source>(get_i(d, "source"));
  c.n_joints = static_cast<std::uint8_t>(get_i(d, "n_joints"));
  c.segment_index = static_cast<std::uint16_t>(opt_i("segment_index", 0));
  c.last_segment = static_cast<std::uint8_t>(opt_i("last_segment", 1));
  const std::size_t n = c.n_joints;
  if (n == 0 || n > R1RT_MAX_GROUP_JOINTS) throw nb::value_error("upper record: n_joints out of range");
  if (d.contains("q")) read_vec(d["q"], c.q, n, "q");
  if (d.contains("dq")) read_vec(d["dq"], c.dq, n, "dq");
  if (d.contains("kp")) read_vec(d["kp"], c.kp, n, "kp");
  if (d.contains("kd")) read_vec(d["kd"], c.kd, n, "kd");
  if (d.contains("tau_ff")) read_vec(d["tau_ff"], c.tau_ff, n, "tau_ff");
  if (d.contains("knots")) {
    std::size_t i = 0;
    for (nb::handle kh : d["knots"]) {
      if (i >= R1RT_MAX_KNOTS) throw nb::value_error("upper record: more than R1RT_MAX_KNOTS knots");
      nb::sequence kn = nb::cast<nb::sequence>(kh);
      c.knots[i].t_ns = nb::cast<std::int64_t>(kn[0]);
      read_vec(kn[1], c.knots[i].q, n, "knot q");
      read_vec(kn[2], c.knots[i].dq, n, "knot dq");
      read_vec(kn[3], c.knots[i].ddq, n, "knot ddq");
      ++i;
    }
    c.n_knots = static_cast<std::uint8_t>(i);
  }
  if (d.contains("guards")) {
    nb::dict g = nb::cast<nb::dict>(d["guards"]);
    c.guards.enable = static_cast<std::uint8_t>(g.contains("enable") ? get_i(g, "enable") : 0);
    c.guards.lag_rad = static_cast<float>(g.contains("lag_rad") ? get_d(g, "lag_rad") : 0.0);
    c.guards.tau_nm = static_cast<float>(g.contains("tau_nm") ? get_d(g, "tau_nm") : 0.0);
    c.guards.contact_mask = static_cast<std::uint32_t>(g.contains("contact_mask") ? get_i(g, "contact_mask") : 0);
    c.guards.contact_all = static_cast<std::uint8_t>(g.contains("contact_all") ? get_i(g, "contact_all") : 0);
    c.guards.hand = static_cast<std::uint8_t>(g.contains("hand") ? get_i(g, "hand") : 0);
    c.guards.surface_z_m = g.contains("surface_z_m") ? get_d(g, "surface_z_m") : 0.0;
  }
  return c;
}

}  // namespace

// Session 5: the same record dict -> r1rt_upper_cmd for SegmentView.push_upper (r1rt_py_shm.cpp; r1d on r1rt's real
// segment).
r1rt_upper_cmd r1rt_py_read_upper(const nb::dict& d, std::uint64_t epoch_now) { return read_upper(d, epoch_now); }

namespace {

// The core with its own shared segment (ordinary memory: the lockstep mode of DESIGN §2 principle 2).
struct PyCore {
  void* seg = nullptr;
  std::size_t seg_size = 0;
  r1rt_core* core = nullptr;
  r1rt_lowcmd_out last{};
  r1rt_telemetry_record tel{};

  explicit PyCore(const nb::dict& c) {
    seg_size = (r1rt_segment_size() + 63) / 64 * 64;
    seg = std::aligned_alloc(64, seg_size);
    if (seg == nullptr) throw std::bad_alloc();
    r1rt_segment_init(seg, seg_size, 0, 0);
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
    cfg.segment_size = seg_size;
    std::string weights;
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
    // ABI 4 (session 4): the takeover gates; absent keys = 0 = the core's defaults (v2's values, step_api.h)
    auto opt_d = [&](const char* k) { return c.contains(k) ? get_d(c, k) : 0.0; };
    auto opt_i = [&](const char* k) { return c.contains(k) ? get_i(c, k) : std::int64_t{0}; };
    cfg.takeover_blend_s = opt_d("takeover_blend_s");
    cfg.takeover_tilt_max_rad = opt_d("takeover_tilt_max_rad");
    cfg.takeover_dq_max = opt_d("takeover_dq_max");
    cfg.takeover_temp_max_c = opt_d("takeover_temp_max_c");
    cfg.takeover_state_max_age_ns = opt_i("takeover_state_max_age_ns");
    cfg.builtin_max_age_ns = opt_i("builtin_max_age_ns");
    cfg.builtin_min_leg_kp = opt_d("builtin_min_leg_kp");
    cfg.mirror_max_dev_rad = opt_d("mirror_max_dev_rad");
    cfg.takeover_mirror_alarm_ns = opt_i("takeover_mirror_alarm_ns");
    cfg.takeover_require_guard = static_cast<std::uint32_t>(opt_i("takeover_require_guard"));
    cfg.resume_gap_ns = opt_i("resume_gap_ns");  // ABI 7 (session 8, fix #29): absent = 0 = 100 ms
    cfg.guard_max_age_ns = opt_i("guard_max_age_ns");
    // ABI 5 (session 5): ON_STAND's legs + waist hold gains (absent = 0 = the stand gains)
    if (c.contains("on_stand_kp")) read_vec(c["on_stand_kp"], cfg.on_stand_kp, R1RT_NUM_JOINTS, "on_stand_kp");
    if (c.contains("on_stand_kd")) read_vec(c["on_stand_kd"], cfg.on_stand_kd, R1RT_NUM_JOINTS, "on_stand_kd");
    cfg.rt_granted = static_cast<std::uint32_t>(opt_i("rt_granted"));
    cfg.takeover_require_rt = static_cast<std::uint32_t>(opt_i("takeover_require_rt"));
    // ABI 8 (session 9, U19 / D25): the state-age limit + the stale rung (absent = 0 = the defaults; the rung OFF)
    cfg.state_age_limit_ns = opt_i("state_age_limit_ns");
    cfg.state_age_sustain_ns = opt_i("state_age_sustain_ns");
    cfg.stale_return_live_ns = opt_i("stale_return_live_ns");
    cfg.board_tick_hz = opt_d("board_tick_hz");
    cfg.stale_rung = static_cast<std::uint32_t>(opt_i("stale_rung"));
    std::int32_t st = 0;
    core = r1rt_core_create(&cfg, &st);
    if (core == nullptr) {
      std::free(seg);
      seg = nullptr;
      throw std::runtime_error("r1rt_core_create failed: status " + std::to_string(st));
    }
  }
  ~PyCore() {
    r1rt_core_destroy(core);
    std::free(seg);
  }
  PyCore(const PyCore&) = delete;
  PyCore& operator=(const PyCore&) = delete;

  std::tuple<int, Vec26, Vec26, Vec26, Vec26, Vec26> step(std::int64_t t_ns, nb::handle state) {
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
      if (d.contains("temperature")) {
        double tmp[R1RT_NUM_JOINTS];
        read_vec(d["temperature"], tmp, R1RT_NUM_JOINTS, "temperature");
        for (int j = 0; j < R1RT_NUM_JOINTS; ++j) in.temperature[j][0] = in.temperature[j][1] = static_cast<std::int16_t>(tmp[j]);
      }
    }
    std::int32_t rc;
    {
      nb::gil_scoped_release release;
      rc = r1rt_core_step(core, has ? &in : nullptr, t_ns, &last, &tel);
    }
    if (rc != R1RT_OK) throw std::runtime_error("r1rt_core_step: status " + std::to_string(rc));
    return {static_cast<int>(last.publish), vec26(last.q), vec26(last.dq), vec26(last.kp), vec26(last.kd),
            vec26(last.tau_ff)};
  }

  int policy_step(std::int64_t t_ns) {
    nb::gil_scoped_release release;
    return r1rt_core_policy_step(core, t_ns);
  }

  nb::dict status() const {
    r1rt_core_status s{};
    r1rt_core_get_status(core, &s);
    nb::dict d;
    d["ticks"] = s.ticks;
    d["policy_thread_runs"] = s.policy_thread_runs;
    d["inline_runs"] = s.inline_runs;
    d["inline_entries"] = s.inline_entries;
    d["invalid_actions"] = s.invalid_actions;
    d["lowstate_late"] = s.lowstate_late;
    d["stale_resends"] = s.stale_resends;
    d["clamp_total"] = s.clamp_total;
    d["loop_gaps"] = s.loop_gaps;
    d["freeze_entries"] = s.freeze_entries;
    d["telemetry_drops"] = s.telemetry_drops;
    d["last_action_t_ns"] = s.last_action_t_ns;
    d["lowstate_age_ns"] = s.lowstate_age_ns;
    d["faults"] = s.faults;
    d["faults_seen"] = s.faults_seen;
    d["tilt_rad"] = s.tilt_rad;
    d["gait_phase"] = s.gait_phase;
    d["lean"] = s.lean;
    d["mode"] = static_cast<int>(s.mode);
    d["inline_active"] = static_cast<int>(s.inline_active);
    d["stop_active"] = static_cast<int>(s.stop_active);
    d["policy_loaded"] = static_cast<int>(s.policy_loaded);
    d["publish"] = static_cast<int>(last.publish);
    d["clamp_count"] = tel.clamp_count;
    d["guard_event"] = static_cast<int>(tel.guard_event);
    d["exec_ready"] = static_cast<int>(s.exec_ready);
    d["upper_accepted"] = s.upper_accepted;
    d["upper_rejected"] = s.upper_rejected;
    d["upper_last_reject"] = static_cast<int>(s.upper_last_reject);
    d["upper_stop_epoch"] = s.upper_stop_epoch;
    d["upper_tick_event"] = static_cast<int>(s.upper_tick_event);
    nb::list src, ev, stopping, ids;
    for (int g = 0; g < R1RT_NUM_UPPER_GROUPS; ++g) {
      src.append(static_cast<int>(s.upper_source[g]));
      ev.append(static_cast<int>(s.upper_event[g]));
      stopping.append(static_cast<int>(s.upper_stopping[g]));
      ids.append(s.upper_cmd_id[g]);
    }
    d["upper_source"] = src;
    d["upper_event"] = ev;
    d["upper_stopping"] = stopping;
    d["upper_cmd_id"] = ids;
    d["takeover_requests"] = s.takeover_requests;
    d["takeover_aborts"] = s.takeover_aborts;
    d["takeover_alarm_ticks"] = s.takeover_alarm_ticks;
    d["builtin_age_ns"] = s.builtin_age_ns;
    d["blend_progress"] = s.blend_progress;
    d["mirror_dev_rad"] = s.mirror_dev_rad;
    d["policy_commits"] = s.policy_commits;
    d["policy_discarded"] = s.policy_discarded;
    d["takeover_refusal"] = static_cast<int>(s.takeover_refusal);
    d["takeover_used_mirror"] = static_cast<int>(s.takeover_used_mirror);
    // ABI 5 (session 5)
    d["mode_requests"] = s.mode_requests;
    d["mode_abandoned"] = s.mode_abandoned;
    d["mode_last_seq"] = s.mode_last_seq;
    d["guard_absent_ticks"] = s.guard_absent_ticks;
    d["guard_age_ns"] = s.guard_age_ns;
    d["takeover_target"] = static_cast<int>(s.takeover_target);
    d["on_stand_refusal"] = static_cast<int>(s.on_stand_refusal);
    nb::list lsrc, lev, lstop, lids;
    for (int g = 0; g < R1RT_NUM_CMD_GROUPS - R1RT_NUM_UPPER_GROUPS; ++g) {
      lsrc.append(static_cast<int>(s.lower_source[g]));
      lev.append(static_cast<int>(s.lower_event[g]));
      lstop.append(static_cast<int>(s.lower_stopping[g]));
      lids.append(s.lower_cmd_id[g]);
    }
    d["lower_source"] = lsrc;
    d["lower_event"] = lev;
    d["lower_stopping"] = lstop;
    d["lower_cmd_id"] = lids;
    d["hands_pose_used"] = static_cast<int>(s.hands_pose_used);
    d["max_wake_lateness_ns"] = s.max_wake_lateness_ns;
    d["max_iteration_ns"] = s.max_iteration_ns;
    d["rt_granted"] = static_cast<int>(s.rt_granted);
    d["damp_ignored"] = s.damp_ignored;  // ABI 6
    d["reclaims"] = s.reclaims;
    d["reclaim_refusal"] = static_cast<int>(s.reclaim_refusal);
    d["freeze_refusal"] = static_cast<int>(s.freeze_refusal);
    d["freeze_from_stand"] = static_cast<int>(s.freeze_from_stand);
    d["stand_session"] = static_cast<int>(s.stand_session);
    d["motor_error_groups"] = static_cast<int>(s.motor_error_groups);
    d["motor_error_joints"] = s.motor_error_joints;
    d["motor_error_holds"] = s.motor_error_holds;
    d["temp_max_c"] = s.temp_max_c;
    d["temp_max_joint"] = static_cast<int>(s.temp_max_joint);
    d["over_temp_ticks"] = s.over_temp_ticks;
    d["stops"] = s.stops;  // ABI 7 (session 8: fix #29 / #31)
    d["stop_ns_max"] = s.stop_ns_max;
    d["stop_ns_total"] = s.stop_ns_total;
    d["resume_blends"] = s.resume_blends;
    d["resume_waits_expired"] = s.resume_waits_expired;
    d["resume_uncaught"] = s.resume_uncaught;
    d["resume_drain_waits"] = s.resume_drain_waits;  // session 9 (#29 hazard ii)
    d["resume_progress"] = s.resume_progress;
    d["resume_waiting"] = static_cast<int>(s.resume_waiting);
    d["state_content_age_ns"] = s.state_content_age_ns;  // ABI 8 (session 9: the state-age limit)
    d["state_content_age_max_ns"] = s.state_content_age_max_ns;
    d["content_age_valid"] = static_cast<int>(s.content_age_valid);
    d["state_old_ticks"] = s.state_old_ticks;
    d["state_old_episodes"] = s.state_old_episodes;
    d["state_old_ns_max"] = s.state_old_ns_max;
    d["stale_rung_entries"] = s.stale_rung_entries;
    d["stale_rung_returns"] = s.stale_rung_returns;
    d["stale_freeze"] = static_cast<int>(s.stale_freeze);
    d["content_age_relearns"] = s.content_age_relearns;
    d["lower_accepted"] = s.lower_accepted;
    d["lower_rejected"] = s.lower_rejected;
    return d;
  }

  nb::bytes hg_lowcmd() const {
    unsigned char buf[2048];
    std::memset(buf, 0, sizeof buf);
    const std::size_t n = sizeof(r1rt::hg::LowCmd);
    r1rt_fill_hg_lowcmd(&last, buf, n);
    return nb::bytes(reinterpret_cast<const char*>(buf), n);
  }
};

struct PyRunner {
  r1rt::PolicyMlp net;
  r1rt::PolicyRunner runner;
  PyRunner(const std::string& weights, const nb::dict& params) {
    char err[512] = {0};
    if (!net.load(weights.c_str(), err, sizeof err)) throw std::runtime_error(std::string("PolicyRunner: ") + err);
    if (!runner.init(read_params(params), &net, err, sizeof err))
      throw std::runtime_error(std::string("PolicyRunner: ") + err);
  }
};

}  // namespace

NB_MODULE(r1rt_py, m) {
  m.doc() = "Python binding of r1rt::PolicyMlp, the C++ balance-policy runner (simulation / validation only)";
  m.attr("K_OBS") = PolicyMlp::kObs;
  m.attr("K_ACT") = PolicyMlp::kAct;
  m.attr("WEIGHTS_VERSION") = PolicyMlp::kVersion;

  nb::class_<PolicyMlp>(m, "PolicyMlp")
      .def(nb::init<>())
      .def(
          "load",
          [](PolicyMlp& p, const std::string& path) {
            char err[512] = {0};
            if (!p.load(path.c_str(), err, sizeof err)) throw std::runtime_error(std::string("PolicyMlp.load: ") + err);
          },
          nb::arg("path"), "Read an R1W1 weights file (tools/policy_convert.py); RuntimeError with the reason on failure")
      .def_prop_ro("loaded", &PolicyMlp::loaded)
      .def(
          "run",
          [](const PolicyMlp& p, ObsIn obs) {
            require_loaded(p);
            double* out = new double[PolicyMlp::kAct];
            nb::capsule owner(out, [](void* q) noexcept { delete[] static_cast<double*>(q); });
            {
              nb::gil_scoped_release release;
              p.run(obs.data(), out);
            }
            return ActArray(out, {static_cast<std::size_t>(PolicyMlp::kAct)}, owner);
          },
          nb::arg("obs"), "obs float64 (74,) -> raw network output float64 (14,); releases the GIL")
      .def(
          "run_into",
          [](const PolicyMlp& p, ObsIn obs, ActOut act) {
            require_loaded(p);
            nb::gil_scoped_release release;
            p.run(obs.data(), act.data());
          },
          nb::arg("obs"), nb::arg("act"), "as run(), writing into a caller-owned float64 (14,) array")
      .def_prop_ro("onnx_sha256", [](const PolicyMlp& p) { return hex(p.onnx_sha256()); },
                   "sha256 (hex) of the ONNX file the weights were converted from")
      .def_prop_ro("elu_alpha", &PolicyMlp::elu_alpha);

  m.attr("ABI_VERSION") = R1RT_ABI_VERSION;
  m.attr("LAYOUT_VERSION") = R1RT_LAYOUT_VERSION;
  // Session 5: the codes clients use with the mode-request mailbox (step_api.h / types.h), by name.
  m.def("requests", []() {
    nb::dict d;
    d["enter_balance"] = static_cast<int>(R1RT_REQ_ENTER_BALANCE);
    d["stop"] = static_cast<int>(R1RT_REQ_STOP);
    d["resume"] = static_cast<int>(R1RT_REQ_RESUME);
    d["damp"] = static_cast<int>(R1RT_REQ_DAMP);
    d["stand_freeze"] = static_cast<int>(R1RT_REQ_STAND_FREEZE);
    d["takeover"] = static_cast<int>(R1RT_REQ_TAKEOVER);
    d["takeover_released"] = static_cast<int>(R1RT_REQ_TAKEOVER_RELEASED);
    d["takeover_abort"] = static_cast<int>(R1RT_REQ_TAKEOVER_ABORT);
    d["on_stand"] = static_cast<int>(R1RT_REQ_ON_STAND);
    d["reclaim"] = static_cast<int>(R1RT_REQ_RECLAIM);  // ABI 6: the operator's hand-back from r1guard (ON_STAND)
    d["stand_hold"] = static_cast<int>(R1RT_REQ_STAND_HOLD);  // ABI 6: back from the freeze on the stand (S10)
    return d;
  }, "request names -> R1RT_REQ_* codes (the mode-request mailbox / Core.request)");
  m.def("modes", []() {
    nb::dict d;
    d["monitor"] = static_cast<int>(R1RT_MODE_MONITOR);
    d["takeover_mirror"] = static_cast<int>(R1RT_MODE_TAKEOVER_MIRROR);
    d["takeover_blend"] = static_cast<int>(R1RT_MODE_TAKEOVER_BLEND);
    d["balance"] = static_cast<int>(R1RT_MODE_BALANCE);
    d["guarded"] = static_cast<int>(R1RT_MODE_GUARDED);
    d["stand_freeze"] = static_cast<int>(R1RT_MODE_STAND_FREEZE);
    d["on_stand"] = static_cast<int>(R1RT_MODE_ON_STAND);
    d["overlay_arm_sdk"] = static_cast<int>(R1RT_MODE_OVERLAY_ARM_SDK);
    d["damp"] = static_cast<int>(R1RT_MODE_DAMP);
    return d;
  }, "mode names -> r1rt_mode values");
  m.def("requesters", []() {
    nb::dict d;
    d["none"] = static_cast<int>(R1RT_REQUESTER_NONE);
    d["r1rt"] = static_cast<int>(R1RT_REQUESTER_R1RT);
    d["r1guard"] = static_cast<int>(R1RT_REQUESTER_R1GUARD);
    d["r1d"] = static_cast<int>(R1RT_REQUESTER_R1D);
    d["operator"] = static_cast<int>(R1RT_REQUESTER_OPERATOR);
    d["sim"] = static_cast<int>(R1RT_REQUESTER_SIM);
    return d;
  }, "requester names -> ids (stop channel + mailbox; DAMP through the mailbox only from 'operator')");
  m.def("status_codes", []() {
    nb::dict d;
    d["ok"] = R1RT_OK;
    d["invalid_arg"] = R1RT_E_INVALID_ARG;
    d["bad_segment"] = R1RT_E_BAD_SEGMENT;
    d["not_owner"] = R1RT_E_NOT_OWNER;
    d["refused"] = R1RT_E_REFUSED;
    d["again"] = R1RT_E_AGAIN;
    d["abandoned"] = R1RT_E_ABANDONED;
    return d;
  }, "status names -> R1RT_OK / R1RT_E_* (mailbox replies)");
  m.def("refusals", []() {
    nb::dict d;
    d["ok"] = static_cast<int>(R1RT_TAKEOVER_OK);
    d["not_monitor"] = static_cast<int>(R1RT_TAKEOVER_NOT_MONITOR);
    d["no_policy"] = static_cast<int>(R1RT_TAKEOVER_NO_POLICY);
    d["state"] = static_cast<int>(R1RT_TAKEOVER_STATE);
    d["tilt"] = static_cast<int>(R1RT_TAKEOVER_TILT);
    d["moving"] = static_cast<int>(R1RT_TAKEOVER_MOVING);
    d["motor_error"] = static_cast<int>(R1RT_TAKEOVER_MOTOR_ERROR);
    d["hot"] = static_cast<int>(R1RT_TAKEOVER_HOT);
    d["guard"] = static_cast<int>(R1RT_TAKEOVER_GUARD);
    d["builtin"] = static_cast<int>(R1RT_TAKEOVER_BUILTIN);
    d["token"] = static_cast<int>(R1RT_TAKEOVER_TOKEN);
    d["rt"] = static_cast<int>(R1RT_TAKEOVER_RT);
    d["stand_session"] = static_cast<int>(R1RT_TAKEOVER_STAND_SESSION);  // ABI 6: r1rt --stand-session
    d["not_operator"] = static_cast<int>(R1RT_REFUSE_NOT_OPERATOR);
    d["mode"] = static_cast<int>(R1RT_REFUSE_MODE);
    d["unknown"] = static_cast<int>(R1RT_REFUSE_UNKNOWN);
    d["reclaim_token"] = static_cast<int>(R1RT_REFUSE_TOKEN);  // ABI 6: RECLAIM without the guard holding the token
    d["reclaim_state"] = static_cast<int>(R1RT_REFUSE_STATE);  // ABI 6: RECLAIM without fresh state / executors
    d["far"] = static_cast<int>(R1RT_REFUSE_FAR);  // ABI 6: the freeze on the stand from too far off the stand pose
    return d;
  }, "refusal names -> codes (R1RT_TAKEOVER_* gates and R1RT_REFUSE_*)");
  nb::class_<PyCore>(m, "Core")
      .def(nb::init<const nb::dict&>(), nb::arg("config"),
           "The core (r1rt_core_create) with its own segment; config = r1sim.v3_core.core_config(...)")
      .def("step", &PyCore::step, nb::arg("t_ns"), nb::arg("state").none(),
           "One writer tick: state dict (or None = no new lowstate) -> (publish, q, dq, kp, kd, tau_ff)")
      .def("policy_step", &PyCore::policy_step, nb::arg("t_ns"), "One iteration of the 50 Hz policy thread")
      .def(
          "request", [](PyCore& c, int req, std::int64_t t_ns) { return r1rt_core_request(c.core, req, t_ns); },
          nb::arg("request"), nb::arg("t_ns"))
      .def(
          "set_base_cmd",
          [](PyCore& c, std::uint64_t seq, std::int64_t t_issue_ns, std::int64_t ttl_ns, double vx, double vy,
             double wz, double lean) {
            r1rt_base_cmd b{};
            b.seq = seq;
            b.t_issue_ns = t_issue_ns;
            b.ttl_ns = ttl_ns;
            b.vx = vx;
            b.vy = vy;
            b.wz = wz;
            b.lean = lean;
            return r1rt_core_set_base_cmd(c.core, &b);
          },
          nb::arg("seq"), nb::arg("t_issue_ns"), nb::arg("ttl_ns"), nb::arg("vx"), nb::arg("vy"), nb::arg("wz"),
          nb::arg("lean"))
      .def(
          "inject",
          [](PyCore& c, int kind, std::int64_t t0, std::int64_t t1, double value) {
            return r1rt_core_inject(c.core, kind, t0, t1, value);
          },
          nb::arg("kind"), nb::arg("t_start_ns"), nb::arg("t_end_ns"), nb::arg("value") = 0.0)
      .def(
          "set_builtin_cmd",
          [](PyCore& c, const nb::handle& q, const nb::handle& dq, const nb::handle& kp, const nb::handle& kd,
             const nb::handle& tau_ff, const nb::handle& enable, std::int64_t t_rx_ns) {
            r1rt_lowcmd_out b{};
            b.publish = 1;
            b.mode_pr = R1RT_MODE_PR;
            b.mode_machine = R1RT_MODE_MACHINE_R1;
            read_vec(q, b.q, R1RT_NUM_JOINTS, "q");
            read_vec(dq, b.dq, R1RT_NUM_JOINTS, "dq");
            read_vec(kp, b.kp, R1RT_NUM_JOINTS, "kp");
            read_vec(kd, b.kd, R1RT_NUM_JOINTS, "kd");
            read_vec(tau_ff, b.tau_ff, R1RT_NUM_JOINTS, "tau_ff");
            if (enable.is_none()) {
              for (int j = 0; j < R1RT_NUM_JOINTS; ++j) b.enable[j] = 1;
            } else {
              double en[R1RT_NUM_JOINTS];
              read_vec(enable, en, R1RT_NUM_JOINTS, "enable");
              for (int j = 0; j < R1RT_NUM_JOINTS; ++j) b.enable[j] = en[j] != 0.0 ? 1 : 0;
            }
            return r1rt_core_set_builtin_cmd(c.core, &b, t_rx_ns);
          },
          nb::arg("q"), nb::arg("dq"), nb::arg("kp"), nb::arg("kd"), nb::arg("tau_ff"), nb::arg("enable").none(),
          nb::arg("t_rx_ns"),
          "the built-in controller's last rt/lowcmd (26 joints each; enable None = all enabled) with its receive time")
      .def_prop_ro(
          "segment_address", [](const PyCore& c) { return reinterpret_cast<std::uintptr_t>(c.seg); },
          "the address of this core's shared segment (ordinary memory, owned by this object): another binding (e.g. "
          "r1rt_py.Guard) attaches to the same segment in lockstep; valid while this Core lives")
      .def_prop_ro("segment_size", [](const PyCore& c) { return c.seg_size; })
      .def(
          "set_rt_granted", [](PyCore& c, bool granted) { return r1rt_core_set_rt_granted(c.core, granted ? 1 : 0); },
          nb::arg("granted"), "the process's real-time grants (session 5; tests)")
      .def(
          "set_stand_session",
          [](PyCore& c, bool on) { return r1rt_core_set_stand_session(c.core, on ? 1 : 0); }, nb::arg("on"),
          "session 6: a stand session - no BALANCE for this core's lifetime (latched: clearing is R1RT_E_REFUSED)")
      .def(
          "set_loop_timing",
          [](PyCore& c, std::int32_t wake_lateness_ns, std::int32_t prev_iteration_ns) {
            return r1rt_core_set_loop_timing(c.core, wake_lateness_ns, prev_iteration_ns);
          },
          nb::arg("wake_lateness_ns"), nb::arg("prev_iteration_ns"), "the writer loop's timing for the next record")
      .def("status", &PyCore::status)
      .def_prop_ro("mode", [](const PyCore& c) { return static_cast<int>(r1rt_core_mode(c.core)); })
      .def("hg_lowcmd", &PyCore::hg_lowcmd, "the hg LowCmd_ image (1004 bytes, CRC last) of the last command")
      .def(
          "stop_channel",
          [](PyCore& c, int level, std::uint32_t requester, std::int64_t t_ns) {
            static_cast<r1rt::shm::Segment*>(c.seg)->stop.request(level, requester, t_ns);
          },
          nb::arg("level"), nb::arg("requester"), nb::arg("t_ns"), "a client's request on the segment's stop channel")
      .def(
          "push_upper",
          [](PyCore& c, const nb::dict& rec) {
            const r1rt_upper_cmd cmd = read_upper(rec, r1rt_segment_stop_epoch(c.seg));
            return r1rt_segment_push_upper_cmd(c.seg, &cmd);
          },
          nb::arg("record"), "r1d's side: one upper-body record into the segment's upper_cmd ring (R1RT_OK / -6 full)")
      .def(
          "upper_keepalive",
          [](PyCore& c, int group, std::uint64_t cmd_id, std::int64_t t_ns) {
            return r1rt_segment_upper_keepalive(c.seg, static_cast<std::uint32_t>(group), cmd_id, t_ns);
          },
          nb::arg("group"), nb::arg("cmd_id"), nb::arg("t_ns"), "r1d's side: renew a TRAJECTORY's keepalive")
      .def(
          "stop_epoch", [](const PyCore& c) { return r1rt_segment_stop_epoch(c.seg); },
          "the stop channel's epoch (stamp it on records)")
      .def(
          "steal_token",
          [](PyCore& c) {
            auto& tok = static_cast<r1rt::shm::Segment*>(c.seg)->token;
            std::uint64_t w = tok.load();
            return tok.try_transfer(w, R1RT_OWNER_R1GUARD);
          },
          "tests: the ownership token goes to r1guard (CAS), as on a guard takeover");
  {  // r1guard (session 4): r1rt_py.Guard on a Core's segment (rt/bindings/r1rt_py_guard.cpp)
    void bind_guard(nb::module_&, void* (*)(nb::handle));
    bind_guard(m, [](nb::handle h) -> void* { return nb::cast<PyCore&>(h).seg; });
  }

  nb::class_<PyRunner>(m, "PolicyRunner")
      .def(nb::init<const std::string&, const nb::dict&>(), nb::arg("weights"), nb::arg("params"))
      .def(
          "step",
          [](PyRunner& r, ObsIn26 q, ObsIn26 dq, nb::ndarray<const double, nb::shape<4>, nb::c_contig> quat,
             nb::ndarray<const double, nb::shape<3>, nb::c_contig> gyro,
             nb::ndarray<const double, nb::shape<3>, nb::c_contig> cmd, double t) {
            double q26[R1RT_NUM_JOINTS], kp26[R1RT_NUM_JOINTS], kd26[R1RT_NUM_JOINTS];
            bool ran;
            {
              nb::gil_scoped_release release;
              ran = r.runner.step(q.data(), dq.data(), quat.data(), gyro.data(), cmd.data(), t, q26, kp26, kd26);
            }
            return std::make_tuple(ran, vec26(q26), vec26(kp26), vec26(kd26));
          },
          nb::arg("q26"), nb::arg("dq26"), nb::arg("quat"), nb::arg("gyro"), nb::arg("cmd"), nb::arg("t"))
      .def("reset", [](PyRunner& r) { r.runner.reset(); })
      .def("set_lean_target", [](PyRunner& r, double rad) { return r.runner.set_lean_target(rad); }, nb::arg("rad"))
      .def_prop_ro("lean", [](const PyRunner& r) { return r.runner.lean(); })
      .def_prop_ro("phase_t", [](const PyRunner& r) { return r.runner.phase_t(); })
      .def_prop_ro("steps", [](const PyRunner& r) { return r.runner.steps(); })
      .def_prop_ro("n_obs", [](const PyRunner& r) { return r.runner.n_obs(); })
      .def_prop_ro(
          "target",
          [](const PyRunner& r) {
            return vecn<R1RT_NUM_POLICY_ACTIONS>(r.runner.target(), static_cast<std::size_t>(r.runner.n_act()));
          },
          nb::rv_policy::automatic)
      .def_prop_ro(
          "last_action",
          [](const PyRunner& r) {
            return vecn<R1RT_NUM_POLICY_ACTIONS>(r.runner.last_action(), static_cast<std::size_t>(r.runner.n_act()));
          },
          nb::rv_policy::automatic);
  { void r1rt_bind_arm_model(nb::module_&); r1rt_bind_arm_model(m); }  // the arm model (r1rt_py_arm.cpp)
  void r1rt_bind_shm(nb::module_&, void* (*)(nb::handle)); r1rt_bind_shm(m, [](nb::handle h) -> void* { return nb::isinstance<PyCore>(h) ? nb::inst_ptr<PyCore>(h)->seg : nullptr; });  // s4 logger: r1rt_py_shm.cpp
}
