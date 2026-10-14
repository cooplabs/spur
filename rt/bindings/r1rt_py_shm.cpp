// rt/bindings/r1rt_py_shm.cpp - the shared segment's reader side for Python (session 4): what the logger `r1log`
// (python/r1mc3/log, DESIGN.md §9, §2.1) needs to drain r1rt's telemetry and policy rings and to sample the seqlocks.
//
// Why a binding and not a Python mmap view: the SPSC rings and seqlocks are correct only with their acquire / release
// order (spsc_ring.hpp, seqlock.hpp). A plain memoryview read has no barriers, so on the Orin (aarch64, weakly
// ordered) a reader could see a published head before the slot's bytes. The binding uses the same C++ code as r1rt.
//
// Three ways to get a SegmentView:
//   SegmentView.attach(name)   the robot: r1log attaches to r1rt's POSIX segment (R1RT_SHM_NAME), header checked
//   SegmentView.create(name)   tests / a named simulator segment (the creator unlinks the name on close)
//   SegmentView.of_core(core)  lockstep: the private segment of an r1rt_py.Core (kept alive by the view)
// The consumer side (pop_*) is single-consumer by construction: one view per ring consumer; a mutex in the view
// serialises Python threads that share it. push_* exist for tests only (the producer on the robot is r1rt alone).
//
// Records cross as raw bytes in the C layout (types.h); `layout()` gives the sizes and every field's offsetof so
// the Python side (r1mc3.log.records) can check its numpy dtypes against this build and fail on any drift.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

#include "r1rt/shm.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"
#include "r1rt/types.h"

namespace nb = nanobind;

r1rt_upper_cmd r1rt_py_read_upper(const nb::dict& d, std::uint64_t epoch_now);  // r1rt_py.cpp (Core.push_upper's)

namespace {

nb::ndarray<nb::numpy, double, nb::ndim<1>> arr(const double* src, std::size_t n) {
  double* out = new double[n];
  std::memcpy(out, src, n * sizeof(double));
  nb::capsule owner(out, [](void* q) noexcept { delete[] static_cast<double*>(q); });
  return nb::ndarray<nb::numpy, double, nb::ndim<1>>(out, {n}, owner);
}

nb::object state_dict(const r1rt::Seqlock<r1rt_state>& sl) {
  r1rt_state v{};
  if (!sl.try_read(v) || v.seq == 0) return nb::none();
  nb::dict d;
  d["seq"] = v.seq;
  d["t_core_ns"] = v.t_core_ns;
  d["t_rx_ns"] = v.in.t_rx_ns;
  d["board_tick"] = v.in.board_tick;
  d["q"] = arr(v.in.q, R1RT_NUM_JOINTS);
  d["dq"] = arr(v.in.dq, R1RT_NUM_JOINTS);
  d["tau_est"] = arr(v.in.tau_est, R1RT_NUM_JOINTS);
  d["imu_quat"] = arr(v.in.imu_quat, 4);
  d["imu_gyro"] = arr(v.in.imu_gyro, 3);
  d["imu_acc"] = arr(v.in.imu_acc, 3);
  d["imu_rpy"] = arr(v.in.imu_rpy, 3);
  d["projected_gravity"] = arr(v.projected_gravity, 3);
  d["tilt_rad"] = v.tilt_rad;
  d["lowstate_age_ns"] = v.lowstate_age_ns;
  d["faults"] = v.faults;
  d["mode"] = static_cast<int>(v.mode);
  d["token_owner"] = static_cast<int>(v.token_owner);
  nb::list temp, motor;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    temp.append(static_cast<int>(v.in.temperature[j][0] > v.in.temperature[j][1] ? v.in.temperature[j][0]
                                                                                    : v.in.temperature[j][1]));
    motor.append(v.in.motor_state[j]);
  }
  d["temperature_max"] = temp;
  d["motor_state"] = motor;
  return d;
}

nb::object mirror_dict(const r1rt::Seqlock<r1rt_mirror>& sl) {
  r1rt_mirror v{};
  if (!sl.try_read(v) || v.seq == 0) return nb::none();
  nb::dict d;
  d["seq"] = v.seq;
  d["t_ns"] = v.t_ns;
  d["last_action_t_ns"] = v.last_action_t_ns;
  d["gait_phase"] = v.gait_phase;
  d["last_action"] = arr(v.last_action, R1RT_NUM_POLICY_ACTIONS);
  d["base_cmd"] = arr(v.base_cmd, 4);
  d["q_target"] = arr(v.q_target, R1RT_NUM_JOINTS);
  d["kp"] = arr(v.kp, R1RT_NUM_JOINTS);
  d["kd"] = arr(v.kd, R1RT_NUM_JOINTS);
  d["blend_progress"] = static_cast<double>(v.blend_progress);
  d["mode"] = static_cast<int>(v.mode);
  d["policy_valid"] = static_cast<int>(v.policy_valid);
  d["freeze_invalid"] = static_cast<int>(v.freeze_invalid);
  d["blend_t0_ns"] = v.blend_t0_ns;
  d["blend_ns"] = v.blend_ns;
  d["freeze_t0_ns"] = v.freeze_t0_ns;
  d["freeze_ramp_ns"] = v.freeze_ramp_ns;
  d["waist"] = arr(v.waist, 2);
  d["waist_target"] = arr(v.waist_target, 2);
  d["takeover_target"] = static_cast<int>(v.takeover_target);
  d["tau_ff"] = arr(v.tau_ff, R1RT_NUM_JOINTS);  // layout 3 (session 8, fix #28)
  d["stops"] = v.stops;                           // layout 3 (session 8, #31 / #29)
  d["stop_ns_max"] = v.stop_ns_max;
  d["resume_blends"] = v.resume_blends;
  d["resume_waiting"] = static_cast<int>(v.resume_waiting);
  return d;
}


// layout 4 (session 10): the executors' per-group status r1rt publishes every tick - r1d's goal ends on real processes
// (the lockstep Core.status() has the same fields under upper_* / lower_*); None = never written / torn
nb::object exec_status_dict(const r1rt::Seqlock<r1rt_exec_status>& sl) {
  r1rt_exec_status v{};
  if (!sl.try_read(v) || v.seq == 0) return nb::none();
  nb::dict d;
  d["seq"] = v.seq;
  d["t_ns"] = v.t_ns;
  d["upper_accepted"] = v.upper_accepted;
  d["upper_rejected"] = v.upper_rejected;
  d["stop_epoch"] = v.stop_epoch;
  d["lower_accepted"] = v.lower_accepted;
  d["lower_rejected"] = v.lower_rejected;
  nb::list cid, src, ev, stp;
  for (int g = 0; g < R1RT_NUM_CMD_GROUPS; ++g) {
    cid.append(v.cmd_id[g]);
    src.append(static_cast<int>(v.source[g]));
    ev.append(static_cast<int>(v.event[g]));
    stp.append(static_cast<int>(v.stopping[g]));
  }
  d["cmd_id"] = cid;
  d["source"] = src;
  d["event"] = ev;
  d["stopping"] = stp;
  d["last_reject"] = static_cast<int>(v.last_reject);
  d["tick_event"] = static_cast<int>(v.tick_event);
  d["motor_error_groups"] = static_cast<int>(v.motor_error_groups);
  d["content_age_valid"] = v.content_age_valid != 0;
  d["stale_freeze"] = v.stale_freeze != 0;
  d["stale_rung"] = static_cast<int>(v.stale_rung);
  d["state_content_age_ns"] = v.state_content_age_ns;
  d["state_content_age_max_ns"] = v.state_content_age_max_ns;
  d["state_old_ticks"] = v.state_old_ticks;
  d["state_old_episodes"] = v.state_old_episodes;
  d["state_old_ns_max"] = v.state_old_ns_max;
  d["stale_rung_entries"] = v.stale_rung_entries;
  d["content_age_relearns"] = v.content_age_relearns;
  return d;
}

struct SegmentView {
  r1rt::ShmSegment shm;                 // attach / create
  r1rt::shm::Segment* seg = nullptr;    // the mapped (or the core's) segment
  nb::object keep;                      // of_core: the Core that owns the memory
  std::string kind;                     // "attach" | "create" | "core"
  std::mutex mu;                        // one consumer per ring: serialise Python threads

  r1rt::shm::Segment& s() {
    if (seg == nullptr) throw std::runtime_error("SegmentView is closed");
    return *seg;
  }
};

template <typename Ring, typename T>
nb::bytes pop_all(SegmentView& v, Ring& ring, std::size_t max_n) {
  std::vector<std::uint8_t> buf;
  {
    std::lock_guard<std::mutex> lk(v.mu);
    nb::gil_scoped_release rel;
    const std::size_t avail = ring.size();
    const std::size_t n = avail < max_n ? avail : max_n;
    buf.resize(n * sizeof(T));
    T rec;
    std::size_t got = 0;
    while (got < n && ring.try_pop(rec)) {
      std::memcpy(buf.data() + got * sizeof(T), &rec, sizeof(T));
      ++got;
    }
    buf.resize(got * sizeof(T));
  }
  return nb::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
}

template <typename Ring, typename T>
std::size_t push_all(Ring& ring, const nb::bytes& data) {
  const std::size_t n = data.size() / sizeof(T);
  if (n * sizeof(T) != data.size()) throw std::invalid_argument("push: not a whole number of records");
  const char* p = data.c_str();
  std::size_t pushed = 0;
  for (std::size_t i = 0; i < n; ++i) {
    T rec;
    std::memcpy(&rec, p + i * sizeof(T), sizeof(T));
    if (ring.try_push(rec)) ++pushed;
  }
  return pushed;
}

template <typename Ring>
nb::dict ring_stats(const Ring& ring) {
  nb::dict d;
  d["capacity"] = Ring::kCapacity;
  d["pushed"] = ring.pushed();
  d["popped"] = ring.popped();
  d["size"] = ring.size();
  d["drops"] = ring.drops();
  return d;
}

template <typename T>
nb::object read_seqlock(const r1rt::Seqlock<T>& sl) {
  T v;
  std::uint64_t seq = 0;
  if (!sl.try_read(v, 64, &seq) || seq == 0) return nb::none();
  return nb::make_tuple(seq / 2, nb::bytes(reinterpret_cast<const char*>(&v), sizeof(T)));
}

#define R1_FIELD(d, T, f) d[#f] = offsetof(T, f)

nb::dict layout() {
  nb::dict out;
  out["layout_version"] = R1RT_LAYOUT_VERSION;
  out["abi_version"] = R1RT_ABI_VERSION;
  out["shm_name"] = R1RT_SHM_NAME;
  out["telemetry_capacity"] = r1rt::shm::kTelemetryCapacity;
  out["policy_capacity"] = r1rt::shm::kPolicyCapacity;
  out["guard_telemetry_capacity"] = r1rt::shm::kGuardTelemetryCapacity;
  out["num_joints"] = R1RT_NUM_JOINTS;
  out["max_policy_obs"] = R1RT_MAX_POLICY_OBS;
  out["num_policy_actions"] = R1RT_NUM_POLICY_ACTIONS;

  nb::dict sizes;
  sizes["r1rt_telemetry_record"] = sizeof(r1rt_telemetry_record);
  sizes["r1rt_policy_record"] = sizeof(r1rt_policy_record);
  sizes["r1rt_hands_contact"] = sizeof(r1rt_hands_contact);
  sizes["r1rt_heartbeat"] = sizeof(r1rt_heartbeat);
  sizes["r1rt_guard_record"] = sizeof(r1rt_guard_record);
  sizes["r1rt_hands_pose"] = sizeof(r1rt_hands_pose);
  sizes["r1rt_exec_status"] = sizeof(r1rt_exec_status);  // layout 4 (session 10)
  out["sizes"] = sizes;

  nb::dict tel;
  using TR = r1rt_telemetry_record;
  R1_FIELD(tel, TR, seq); R1_FIELD(tel, TR, t_wake_ns); R1_FIELD(tel, TR, t_send_ns);
  R1_FIELD(tel, TR, t_state_rx_ns); R1_FIELD(tel, TR, board_tick); R1_FIELD(tel, TR, wake_lateness_ns);
  R1_FIELD(tel, TR, iteration_ns); R1_FIELD(tel, TR, faults); R1_FIELD(tel, TR, clamp_count);
  R1_FIELD(tel, TR, mode); R1_FIELD(tel, TR, token_owner); R1_FIELD(tel, TR, guard_event);
  R1_FIELD(tel, TR, q); R1_FIELD(tel, TR, dq); R1_FIELD(tel, TR, tau_est); R1_FIELD(tel, TR, q_cmd);
  R1_FIELD(tel, TR, dq_cmd); R1_FIELD(tel, TR, kp); R1_FIELD(tel, TR, kd); R1_FIELD(tel, TR, tau_ff);
  R1_FIELD(tel, TR, imu_quat); R1_FIELD(tel, TR, imu_gyro); R1_FIELD(tel, TR, imu_acc);
  R1_FIELD(tel, TR, temperature); R1_FIELD(tel, TR, motor_state);
  out["r1rt_telemetry_record"] = tel;

  nb::dict pol;
  using PR = r1rt_policy_record;
  R1_FIELD(pol, PR, seq); R1_FIELD(pol, PR, t_obs_ns); R1_FIELD(pol, PR, t_action_ns); R1_FIELD(pol, PR, n_obs);
  R1_FIELD(pol, PR, inline_run); R1_FIELD(pol, PR, obs); R1_FIELD(pol, PR, action);
  out["r1rt_policy_record"] = pol;

  nb::dict hc;
  using HC = r1rt_hands_contact;
  R1_FIELD(hc, HC, seq); R1_FIELD(hc, HC, t_ns); R1_FIELD(hc, HC, pad_contact); R1_FIELD(hc, HC, finger_stalled);
  R1_FIELD(hc, HC, finger_blocked); R1_FIELD(hc, HC, grip_state); R1_FIELD(hc, HC, valid);
  out["r1rt_hands_contact"] = hc;

  nb::dict hb;
  using HB = r1rt_heartbeat;
  R1_FIELD(hb, HB, seq); R1_FIELD(hb, HB, t_ns); R1_FIELD(hb, HB, lowstate_tick); R1_FIELD(hb, HB, rate_hz);
  R1_FIELD(hb, HB, faults); R1_FIELD(hb, HB, blend_progress); R1_FIELD(hb, HB, mode); R1_FIELD(hb, HB, rt_granted);
  R1_FIELD(hb, HB, guard_flags);  // ABI 6 (the former padding byte)
  out["r1rt_heartbeat"] = hb;

  nb::dict gr;  // session 5: r1guard's cycle record
  using GR = r1rt_guard_record;
  R1_FIELD(gr, GR, seq); R1_FIELD(gr, GR, t_ns); R1_FIELD(gr, GR, t_state_rx_ns); R1_FIELD(gr, GR, rt_beat_age_ns);
  R1_FIELD(gr, GR, rt_beat_seq); R1_FIELD(gr, GR, board_tick); R1_FIELD(gr, GR, faults); R1_FIELD(gr, GR, mode);
  R1_FIELD(gr, GR, mode_from); R1_FIELD(gr, GR, act_mode); R1_FIELD(gr, GR, token_owner); R1_FIELD(gr, GR, event);
  R1_FIELD(gr, GR, active); R1_FIELD(gr, GR, published); R1_FIELD(gr, GR, shadow_valid);
  R1_FIELD(gr, GR, shadow_dev_rad); R1_FIELD(gr, GR, tilt_rad); R1_FIELD(gr, GR, q_cmd); R1_FIELD(gr, GR, kp);
  R1_FIELD(gr, GR, kd);
  out["r1rt_guard_record"] = gr;
  return out;
}

#undef R1_FIELD

void check(r1rt::ShmError e, const std::string& what) {
  if (e != r1rt::ShmError::kOk) throw std::runtime_error(what + ": " + r1rt::shm_error_str(e));
}

}  // namespace

// Called once from r1rt_py.cpp's module init with an accessor for r1rt_py.Core's private segment (PyCore lives in
// that file's anonymous namespace; nullptr = not a Core).
void r1rt_bind_shm(nb::module_& m, void* (*core_segment)(nb::handle)) {
  static void* (*core_seg)(nb::handle) = nullptr;
  core_seg = core_segment;

  m.def("shm_layout", &layout, "sizes + offsetof of the records r1log reads (types.h / shm_layout.h of this build)");

  nb::class_<SegmentView>(m, "SegmentView",
                          "The reader side of r1rt's shared segment for r1log (rings drained, seqlocks sampled)")
      .def_static(
          "attach",
          [](const std::string& name) {
            auto* v = new SegmentView();
            const r1rt::ShmError e = r1rt::ShmSegment::attach(name.c_str(), v->shm);
            if (e != r1rt::ShmError::kOk) {
              delete v;
              check(e, "attach " + name);
            }
            v->seg = v->shm.segment();
            v->kind = "attach";
            return v;
          },
          nb::arg("name"), nb::rv_policy::take_ownership, "attach to r1rt's POSIX segment (header checked)")
      .def_static(
          "create",
          [](const std::string& name, bool replace_stale) {
            auto* v = new SegmentView();
            const unsigned flags = r1rt::kNoMlock | (replace_stale ? r1rt::kReplaceStale : 0u);
            const r1rt::ShmError e = r1rt::ShmSegment::create(name.c_str(), v->shm, flags);
            if (e != r1rt::ShmError::kOk) {
              delete v;
              check(e, "create " + name);
            }
            v->seg = v->shm.segment();
            v->kind = "create";
            return v;
          },
          nb::arg("name"), nb::arg("replace_stale") = false, nb::rv_policy::take_ownership,
          "tests / a named sim segment: create + initialise (the name is unlinked on close)")
      .def_static(
          "of_core",
          [](nb::handle core) {
            void* p = core_seg != nullptr ? core_seg(core) : nullptr;
            if (p == nullptr) throw std::invalid_argument("of_core: not an r1rt_py.Core");
            auto* v = new SegmentView();
            v->seg = static_cast<r1rt::shm::Segment*>(p);
            v->keep = nb::borrow(core);
            v->kind = "core";
            return v;
          },
          nb::arg("core"), nb::rv_policy::take_ownership, "lockstep: the private segment of an r1rt_py.Core")
      .def_prop_ro("kind", [](const SegmentView& v) { return v.kind; })
      .def_prop_ro("name", [](const SegmentView& v) { return std::string(v.shm.is_open() ? v.shm.name() : ""); })
      .def_prop_ro("closed", [](const SegmentView& v) { return v.seg == nullptr; })
      .def(
          "close",
          [](SegmentView& v) {
            std::lock_guard<std::mutex> lk(v.mu);
            v.seg = nullptr;
            v.shm.close();
            v.keep = nb::object();
          },
          "unmap (the creator also unlinks); safe twice")
      .def(
          "pop_telemetry",
          [](SegmentView& v, std::size_t max_n) {
            return pop_all<decltype(v.s().telemetry), r1rt_telemetry_record>(v, v.s().telemetry, max_n);
          },
          nb::arg("max_n") = r1rt::shm::kTelemetryCapacity, "drain up to max_n telemetry records (raw bytes)")
      .def(
          "pop_policy",
          [](SegmentView& v, std::size_t max_n) {
            return pop_all<decltype(v.s().policy), r1rt_policy_record>(v, v.s().policy, max_n);
          },
          nb::arg("max_n") = r1rt::shm::kPolicyCapacity, "drain up to max_n policy records (raw bytes)")
      .def(
          "push_telemetry",
          [](SegmentView& v, const nb::bytes& data) {
            return push_all<decltype(v.s().telemetry), r1rt_telemetry_record>(v.s().telemetry, data);
          },
          nb::arg("data"), "TESTS ONLY (the producer is r1rt): push records; returns how many fitted")
      .def(
          "push_policy",
          [](SegmentView& v, const nb::bytes& data) {
            return push_all<decltype(v.s().policy), r1rt_policy_record>(v.s().policy, data);
          },
          nb::arg("data"), "TESTS ONLY (the producer is r1rt): push records; returns how many fitted")
      .def(
          "pop_guard_telemetry",
          [](SegmentView& v, std::size_t max_n) {
            return pop_all<decltype(v.s().guard_telemetry), r1rt_guard_record>(v, v.s().guard_telemetry, max_n);
          },
          nb::arg("max_n") = r1rt::shm::kGuardTelemetryCapacity,
          "drain up to max_n r1guard cycle records (raw bytes; layout 2)")
      .def(
          "push_guard_telemetry",
          [](SegmentView& v, const nb::bytes& data) {
            return push_all<decltype(v.s().guard_telemetry), r1rt_guard_record>(v.s().guard_telemetry, data);
          },
          nb::arg("data"), "TESTS ONLY (the producer is r1guard): push records; returns how many fitted")
      .def("ring_stats",
           [](SegmentView& v) {
             nb::dict d;
             d["telemetry"] = ring_stats(v.s().telemetry);
             d["policy"] = ring_stats(v.s().policy);
             d["guard_telemetry"] = ring_stats(v.s().guard_telemetry);
             return d;
           })
      .def(
          "request_mode",
          [](SegmentView& v, std::uint32_t req, std::uint32_t requester, std::int64_t t_ns, double timeout_s) {
            // the mailbox's client side (r1rt_segment_request_mode); BUSY -> retry with the GIL released until
            // timeout_s (0 = one attempt). Raises BlockingIOError when still busy, RuntimeError when abandoned.
            void* base = &v.s();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
            for (;;) {
              std::uint64_t seq = 0;
              std::int32_t rc;
              {
                nb::gil_scoped_release rel;
                rc = r1rt_segment_request_mode(base, req, requester, t_ns, &seq);
              }
              if (rc == R1RT_OK) return seq;
              if (rc == R1RT_E_AGAIN && std::chrono::steady_clock::now() < deadline) {
                nb::gil_scoped_release rel;
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
              }
              if (rc == R1RT_E_AGAIN) {
                PyErr_SetString(PyExc_BlockingIOError, "request_mode: the mailbox is busy (another request pending)");
                throw nb::python_error();
              }
              throw std::runtime_error("request_mode: status " + std::to_string(rc));
            }
          },
          nb::arg("req"), nb::arg("requester"), nb::arg("t_ns"), nb::arg("timeout_s") = 0.0,
          "a request into the mode-request mailbox (R1RT_REQ_*; requester id): returns its sequence number")
      .def(
          "mode_reply",
          [](SegmentView& v, std::uint64_t seq) -> nb::object {
            std::int32_t status = 0, refusal = 0;
            std::uint32_t mode_after = 0;
            const std::int32_t rc = r1rt_segment_mode_reply(&v.s(), seq, &status, &refusal, &mode_after);
            if (rc == R1RT_OK) return nb::make_tuple(status, refusal, mode_after);
            if (rc == R1RT_E_AGAIN) return nb::none();
            throw std::invalid_argument("mode_reply: unknown sequence " + std::to_string(seq) +
                                        " (never issued, or overwritten by 64 later requests)");
          },
          nb::arg("seq"), "(status, refusal, mode after) of request seq, or None while pending")
      .def(
          "hands_pose", [](SegmentView& v) { return read_seqlock(v.s().hands_pose); },
          "(writes, bytes) of the last consistent r1rt_hands_pose, or None")
      // ---- session 5: r1d's client side on r1rt's real segment (the lockstep Core has the same on itself) ----
      .def(
          "push_upper",
          [](SegmentView& v, const nb::dict& rec) {
            void* base = &v.s();
            const r1rt_upper_cmd cmd = r1rt_py_read_upper(rec, r1rt_segment_stop_epoch(base));
            nb::gil_scoped_release rel;
            return r1rt_segment_push_upper_cmd(base, &cmd);
          },
          nb::arg("record"),
          "r1d's side: one upper_cmd record (Core.push_upper's dict; stop_epoch defaults to the current one) - "
          "R1RT_OK / -6 ring full. SINGLE producer: only r1d pushes (the CLI goes through r1d)")
      .def(
          "upper_keepalive",
          [](SegmentView& v, int group, std::uint64_t cmd_id, std::int64_t t_ns) {
            return r1rt_segment_upper_keepalive(&v.s(), static_cast<std::uint32_t>(group), cmd_id, t_ns);
          },
          nb::arg("group"), nb::arg("cmd_id"), nb::arg("t_ns"), "r1d's side: renew a TRAJECTORY's keepalive (groups 0-5)")
      .def(
          "stop_epoch", [](SegmentView& v) { return r1rt_segment_stop_epoch(&v.s()); },
          "the stop channel's epoch (stamp it on records)")
      .def(
          "write_hands_pose",
          [](SegmentView& v, const nb::handle& q_left, const nb::handle& q_right, const nb::handle& payload_kg,
             std::int64_t t_ns, bool valid) {
            // r1hands' side (its single writer): the Dex3 joint positions + the declared payloads for the core's arm
            // model (DESIGN §3.4)
            r1rt_hands_pose hp{};
            r1rt_hands_pose prev{};
            if (v.s().hands_pose.try_read(prev)) hp.seq = prev.seq;
            hp.seq += 1;
            hp.t_ns = t_ns;
            auto vec = [](const nb::handle& h, double* out, std::size_t n, const char* what) {
              std::size_t i = 0;
              for (nb::handle x : h) {
                if (i >= n) throw nb::value_error((std::string(what) + ": too many values").c_str());
                out[i++] = nb::cast<double>(x);
              }
              if (i != n) throw nb::value_error((std::string(what) + ": wrong length").c_str());
            };
            vec(q_left, hp.q[0], 7, "q_left");
            vec(q_right, hp.q[1], 7, "q_right");
            vec(payload_kg, hp.payload_kg, 2, "payload_kg");
            hp.valid = valid ? 1 : 0;
            v.s().hands_pose.write(hp);
            return hp.seq;
          },
          nb::arg("q_left"), nb::arg("q_right"), nb::arg("payload_kg"), nb::arg("t_ns"), nb::arg("valid") = true,
          "r1hands (single writer): the hands' 7 joints each + the payloads (kg) for the core's arm model")
      .def(
          "write_hands_contact",
          [](SegmentView& v, std::uint32_t pad_left, std::uint32_t pad_right, int stalled_left, int stalled_right,
             int blocked_left, int blocked_right, int grip_left, int grip_right, std::int64_t t_ns, bool valid) {
            // r1hands' side (its single writer): the contact summary the core's trajectory guards read
            r1rt_hands_contact hc{};
            r1rt_hands_contact prev{};
            if (v.s().hands_contact.try_read(prev)) hc.seq = prev.seq;
            hc.seq += 1;
            hc.t_ns = t_ns;
            hc.pad_contact[0] = pad_left;
            hc.pad_contact[1] = pad_right;
            hc.finger_stalled[0] = static_cast<std::uint8_t>(stalled_left);
            hc.finger_stalled[1] = static_cast<std::uint8_t>(stalled_right);
            hc.finger_blocked[0] = static_cast<std::uint8_t>(blocked_left);
            hc.finger_blocked[1] = static_cast<std::uint8_t>(blocked_right);
            hc.grip_state[0] = static_cast<std::uint8_t>(grip_left);
            hc.grip_state[1] = static_cast<std::uint8_t>(grip_right);
            hc.valid = valid ? 1 : 0;
            v.s().hands_contact.write(hc);
            return hc.seq;
          },
          nb::arg("pad_left"), nb::arg("pad_right"), nb::arg("stalled_left") = 0, nb::arg("stalled_right") = 0,
          nb::arg("blocked_left") = 0, nb::arg("blocked_right") = 0, nb::arg("grip_left") = 0,
          nb::arg("grip_right") = 0, nb::arg("t_ns") = 0, nb::arg("valid") = true,
          "r1hands (single writer): the pads in contact (bit k = pad k), stalled / blocked fingers, grip state")
      .def(
          "state", [](SegmentView& v) { return state_dict(v.s().state); },
          "the core's r1rt_state as a dict (seq, t_core_ns, q / dq / tau_est (26), imu_*, tilt_rad, faults, mode, "
          "token_owner, ...) or None (never written / torn: bounded seqlock retries)")
      .def(
          "mirror", [](SegmentView& v) { return mirror_dict(v.s().mirror); },
          "the core's guard mirror as a dict (q_target / kp / kd (26), last_action (14), mode, blend / freeze fields, "
          "...) or None")
      .def(
          "exec_status", [](SegmentView& v) { return exec_status_dict(v.s().exec_status); },
          "layout 4 (session 10): r1rt's executor status as a dict - per group (6, r1rt_group order) cmd_id / source / "
          "event / stopping, upper / lower accepted / rejected, last_reject, tick_event, the state-age counters - or "
          "None (never written / torn); r1d judges r1rt alive from t_ns")
      .def(
          "set_base_cmd",
          [](SegmentView& v, std::uint64_t seq, std::int64_t t_ns, std::int64_t ttl_ns, double vx, double vy, double wz,
             double lean, std::uint32_t requester) {
            // session 10: r1d's side of the base command (its single writer on the robot; the lockstep Core has
            // Core.set_base_cmd on the same seqlock). The core reads it every tick, zeroes it when the TTL ran out or a
            // value is not finite (core.cpp current_cmd) and keeps the policy's own limits (the lean ramp / range) -
            // nothing here bypasses them. TTL bounded here too: types.h "TTL <= 0.5 s".
            if (ttl_ns <= 0 || ttl_ns > 500'000'000LL) throw std::invalid_argument("set_base_cmd: ttl_ns must be in (0, 0.5 s]");
            r1rt_base_cmd b{};
            b.seq = seq;
            b.t_issue_ns = t_ns;
            b.ttl_ns = ttl_ns;
            b.vx = vx;
            b.vy = vy;
            b.wz = wz;
            b.lean = lean;
            b.requester = requester;
            v.s().base_cmd.write(b);
            return R1RT_OK;
          },
          nb::arg("seq"), nb::arg("t_ns"), nb::arg("ttl_ns"), nb::arg("vx"), nb::arg("vy"), nb::arg("wz"),
          nb::arg("lean"), nb::arg("requester") = 0,
          "r1d's side (SINGLE writer): the policy's base command (vx, vy, wz, lean; TTL <= 0.5 s, expiry = zero)")
      .def(
          "base_cmd", [](SegmentView& v) { return read_seqlock(v.s().base_cmd); },
          "(writes, bytes) of the last consistent r1rt_base_cmd, or None (tests: what r1d wrote)")
      .def(
          "hands_contact", [](SegmentView& v) { return read_seqlock(v.s().hands_contact); },
          "(writes, bytes) of the last consistent r1rt_hands_contact, or None (never written / torn)")
      .def(
          "rt_heartbeat", [](SegmentView& v) { return read_seqlock(v.s().rt_heartbeat); },
          "(writes, bytes) of r1rt's heartbeat, or None")
      .def(
          "guard_heartbeat", [](SegmentView& v) { return read_seqlock(v.s().guard_heartbeat); },
          "(writes, bytes) of r1guard's heartbeat, or None")
      .def(
          "guard_info",
          [](SegmentView& v) -> nb::object {
            // session 6 (r1d's status): r1guard's heartbeat decoded - alive is judged by the caller from t_ns
            r1rt_heartbeat hb{};
            if (!v.s().guard_heartbeat.try_read(hb) || hb.seq == 0) return nb::none();
            nb::dict d;
            d["seq"] = hb.seq;
            d["t_ns"] = hb.t_ns;
            d["mode"] = static_cast<int>(hb.mode);
            d["faults"] = hb.faults;
            d["rt_granted"] = static_cast<int>(hb.rt_granted);
            d["armed"] = (hb.guard_flags & R1RT_GUARD_FLAG_ARMED) != 0;
            d["active"] = (hb.guard_flags & R1RT_GUARD_FLAG_ACTIVE) != 0;
            d["damp"] = (hb.guard_flags & R1RT_GUARD_FLAG_DAMP) != 0;
            d["not_owner"] = (hb.guard_flags & R1RT_GUARD_FLAG_NOT_OWNER) != 0;
            d["mode_from"] = static_cast<int>((hb.guard_flags >> R1RT_GUARD_FLAG_MODE_FROM_SHIFT) & 0x0Fu);
            d["flags"] = static_cast<int>(hb.guard_flags);
            return d;
          },
          "r1guard's heartbeat as a dict (seq, t_ns, mode, faults, rt_granted, armed, active, damp, not_owner, "
          "mode_from, flags) or None (never written / torn)")
      .def(
          "rt_info",
          [](SegmentView& v) -> nb::object {
            // session 6: r1rt's heartbeat decoded (r1d's status: the stand session) - alive: judge t_ns yourself
            r1rt_heartbeat hb{};
            if (!v.s().rt_heartbeat.try_read(hb) || hb.seq == 0) return nb::none();
            nb::dict d;
            d["seq"] = hb.seq;
            d["t_ns"] = hb.t_ns;
            d["mode"] = static_cast<int>(hb.mode);
            d["faults"] = hb.faults;
            d["rate_hz"] = hb.rate_hz;
            d["rt_granted"] = static_cast<int>(hb.rt_granted);
            d["stand_session"] = (hb.guard_flags & R1RT_RT_FLAG_STAND_SESSION) != 0;
            d["stop_active"] = (hb.guard_flags & R1RT_RT_FLAG_STOP_ACTIVE) != 0;  // session 13: the latched stop
            d["flags"] = static_cast<int>(hb.guard_flags);
            // session 8 (HW1 fix #31): the core's STOPS (writer silences > resume_gap_ns: test SIGSTOPs, stalls)
            // apart from overruns, from the guard mirror (layout 3; the heartbeat itself is unchanged); None when the
            // mirror was never written / torn
            r1rt_mirror mi{};
            const bool mok = v.s().mirror.try_read(mi) && mi.seq != 0;
            d["stops"] = mok ? nb::cast(mi.stops) : nb::none();
            d["stop_ms_max"] = mok ? nb::cast(static_cast<double>(mi.stop_ns_max) * 1e-6) : nb::none();
            d["resume_blends"] = mok ? nb::cast(mi.resume_blends) : nb::none();
            d["resume_waiting"] = mok ? nb::cast(mi.resume_waiting != 0) : nb::none();
            // session 10 (layout 4; the stale agent's names for r1d's `r1 status` words): the state-age counters from
            // r1rt's executor status, None when it was never written / torn
            r1rt_exec_status es{};
            const bool eok = v.s().exec_status.try_read(es) && es.seq != 0;
            auto ms = [](std::int64_t ns) { return static_cast<double>(ns) * 1e-6; };
            d["state_content_age_ms"] =
                eok ? (es.state_content_age_ns < 0 ? nb::none() : nb::cast(ms(es.state_content_age_ns))) : nb::none();
            d["state_content_age_max_ms"] = eok ? nb::cast(ms(es.state_content_age_max_ns)) : nb::none();
            d["content_age_valid"] = eok ? nb::cast(es.content_age_valid != 0) : nb::none();
            d["state_old_episodes"] = eok ? nb::cast(es.state_old_episodes) : nb::none();
            d["state_old_ticks"] = eok ? nb::cast(es.state_old_ticks) : nb::none();
            d["state_old_ms_max"] = eok ? nb::cast(ms(es.state_old_ns_max)) : nb::none();
            d["stale_rung_entries"] = eok ? nb::cast(es.stale_rung_entries) : nb::none();
            d["stale_freeze"] = eok ? nb::cast(es.stale_freeze != 0) : nb::none();
            d["stale_rung"] = eok ? nb::cast(static_cast<int>(es.stale_rung)) : nb::none();
            d["content_age_relearns"] = eok ? nb::cast(es.content_age_relearns) : nb::none();
            return d;
          },
          "r1rt's heartbeat as a dict (seq, t_ns, mode, faults, rate_hz, rt_granted, stand_session, flags; session 8: "
          "stops, stop_ms_max, resume_blends, resume_waiting from the mirror, None if absent) or None")
      .def(
          "creator_pid", [](SegmentView& v) { return static_cast<int>(v.s().header.creator_pid); },
          "the pid of the segment's creator (r1rt on the robot; the stand tests' SIGSTOP target)")
      .def(
          "token",
          [](SegmentView& v) {
            const std::uint64_t w = v.s().token.load();
            return nb::make_tuple(static_cast<int>(r1rt::shm::OwnershipToken::owner_of(w)),
                                  r1rt::shm::OwnershipToken::generation_of(w));
          },
          "(owner, generation) of the ownership token (0 none, 1 r1rt, 2 r1guard)")
      .def(
          "stop_requests",
          [](SegmentView& v) {
            nb::list out;
            for (const auto& lvl : v.s().stop.level) {
              out.append(nb::make_tuple(lvl.requests.load(std::memory_order_acquire),
                                        lvl.last_request_ns.load(std::memory_order_relaxed),
                                        lvl.last_requester.load(std::memory_order_relaxed)));
            }
            return out;
          },
          "per stop level (stop, halt, damp): (requests, last_request_ns, last_requester)")
      .def(
          "request_stop",
          [](SegmentView& v, int level, std::uint32_t requester, std::int64_t t_ns) {
            if (level < 0 || level >= R1RT_NUM_STOP_LEVELS) throw std::invalid_argument("stop level 0..2");
            v.s().stop.request(level, requester, t_ns);
          },
          nb::arg("level"), nb::arg("requester"), nb::arg("t_ns"),
          "a client's request on the stop channel (r1d / the CLI / tests; r1log itself never calls it)");
  // Session 5 (the contract): Core.segment_view() = SegmentView.of_core(core) - the lockstep core's segment as a client
  // sees it (the mode-request mailbox, the stop channel, the rings)
  if (nb::hasattr(m, "Core")) {
    nb::object of_core = m.attr("SegmentView").attr("of_core");
    nb::setattr(m.attr("Core"), "segment_view",
                nb::cpp_function([of_core](nb::handle self) { return of_core(self); }, nb::is_method(),
                                 nb::name("segment_view"),
                                 "a SegmentView of this core's own segment (= SegmentView.of_core(core))"));
  }
}
