"""The v3 real-time core (r1rt, C++) driven by the simulator in lockstep (DESIGN.md §2 principle 2, §12.4).

    from r1sim.v3_core import V3CoreController
    ctl = V3CoreController(w.model)                    # the core: writer 500 Hz + its policy thread 50 Hz
    sim.add_controller(ctl, ctl.period_s, "v3")        # every writer tick: LowState -> r1rt_lowstate_in -> step()

The core is the C++ library behind the step C ABI (rt/include/r1rt/step_api.h, rt/src/core.cpp), through its Python
binding `r1rt_py.Core` (the module the C++ stage of scripts/ci.sh builds; r1sim.cpp_policy finds it). The same code
runs on the robot as the r1rt process (session 5); here the simulator calls it in a fixed order with simulated time:

* every writer tick (2 ms at 500 Hz): the sim state is converted to r1rt_lowstate_in (the DDS receive thread's job)
  and `step()` runs the writer (checks, policy action / inline rung / STAND_FREEZE, upper-body HOLD, limits, LowCmd);
* every 10th tick after BALANCE was entered: `policy_step()` = one iteration of the 50 Hz policy thread (its action
  reaches the writer at the next tick, as with a real thread: ~2 ms);
* the first call enters BALANCE (`R1RT_REQ_ENTER_BALANCE`: the policy's first action computed at entry) - the end of a
  takeover - unless a `TakeoverPlan` is given: then the core starts in MONITOR next to an `EmulatedBuiltin` (Unitree's
  built-in controller as the takeover sees it: the S1 stand fixture publishing on rt/lowcmd until ReleaseMode) and the
  controller plays r1d's part of the takeover (DESIGN §3.5, session 4): it feeds the built-in's last command into the
  core (`set_builtin_cmd`, the DDS layer's job), requests TAKEOVER (the core's gates; MIRROR republishes the
  built-in's command), "calls ReleaseMode" (the emulated built-in stops publishing, or refuses) and requests RELEASED
  (BLEND into BALANCE) or ABORT (back to MONITOR).

Arms + head: the core's upper-body executor (r1rt::UpperBodyExecutor, DESIGN §3.3) holds the pose measured at entry
until a record arrives. The controller plays r1d's part: `send_trajectory` / `send_stream` / `send_hold` push records
into the segment's upper_cmd ring (stamped with the stop channel's epoch), and a TRAJECTORY's keepalive is renewed
every KEEPALIVE_RENEW_S until `keepalive_until_s` (withhold it to test the deadman). `at(t, fn)` schedules such a call
at a simulated time (it runs before the writer tick of that time).

Faults (r1sim.faults, simulated time): `controller_stall` with `name="v3_policy"` stalls ONLY the policy thread (the
A7 stall moved into v3: the writer must run the policy inline, rung 1); `name="v3_lowstate"` stalls the DDS receive
thread (no new state reaches the writer: late / stale / lost); `policy_nan` / `policy_out_of_range` are injected into
the core's network output (`r1rt_core_inject`, both the thread and the inline rung see them). Each application is
recorded in the schedule's log.

The core's configuration comes from the policy's metadata (`policy_params`, the same values v2's WalkPolicy reads),
the robot model (`stand_q` = `r1sim.model.centred_stand_pose`: POSE_STAND itself falls backwards under a plain PD, S1)
and the safety numbers of DESIGN §4.2 (`core_config`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import dex3
from . import joints as J
from .cpp_policy import import_r1rt_py, weights_for
from .engine import HandCmd, LowCmd, LowState
from .faults import FaultSchedule

FloatArray = npt.NDArray[np.float64]

OBS_TERMS: dict[str, int] = {  # step_api.h R1RT_OBS_*
    "base_ang_vel": 1,
    "projected_gravity": 2,
    "command": 3,
    "lean": 4,
    "waist_cmd": 5,
    "phase": 6,
    "joint_pos": 7,
    "joint_vel": 8,
    "actions": 9,
}
MAX_JOINTS = 26
MAX_ACT = 14
MAX_TERMS = 9
# The flat layout of r1rt_policy_params used by the golden file (tools/policy_runner_golden.py) and the C++ test.
PARAMS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("n_joints", 1),
    ("n_act", 1),
    ("n_terms", 1),
    ("has_lean", 1),
    ("has_waist", 1),
    ("obs_terms", MAX_TERMS),
    ("joint_index", MAX_JOINTS),
    ("act_pos", MAX_ACT),
    ("default_pos", MAX_JOINTS),
    ("kp", MAX_JOINTS),
    ("kd", MAX_JOINTS),
    ("action_scale", MAX_ACT),
    ("kp_other", J.N),
    ("kd_other", J.N),
    ("period_s", 1),
    ("step_dt_s", 1),
    ("cmd_lo", 3),
    ("cmd_hi", 3),
    ("lean_lo", 1),
    ("lean_hi", 1),
    ("lean_rate", 1),
    ("waist_lo", 2),
    ("waist_hi", 2),
    ("waist_rate", 2),
)

# step_api.h / types.h constants (also exported by r1rt_py; duplicated here so the module imports without the build)
MODE_MONITOR, MODE_BALANCE, MODE_STAND_FREEZE, MODE_DAMP = 0, 3, 5, 8
MODE_TAKEOVER_MIRROR, MODE_TAKEOVER_BLEND = 1, 2
MODE_GUARDED, MODE_ON_STAND = 4, 6
REQ_ENTER_BALANCE, REQ_STOP, REQ_RESUME, REQ_DAMP, REQ_STAND_FREEZE = 1, 2, 3, 4, 5
REQ_TAKEOVER, REQ_TAKEOVER_RELEASED, REQ_TAKEOVER_ABORT = 6, 7, 8  # ABI 4 (session 4)
REQ_ON_STAND = 9  # ABI 5 (session 5)
REQ_RECLAIM, REQ_STAND_HOLD = (
    10,
    11,
)  # ABI 6 (session 6): the operator's hand-back from r1guard; back from the stand freeze
# types.h requester ids (session 5: the mode-request mailbox; DAMP through it only from OPERATOR)
REQUESTER_R1D, REQUESTER_OPERATOR, REQUESTER_SIM = 3, 4, 5
TAKEOVER_REFUSALS: dict[int, str] = {  # step_api.h R1RT_TAKEOVER_*
    0: "ok",
    1: "not_monitor",
    2: "no_policy",
    3: "state",
    4: "tilt",
    5: "moving",
    6: "motor_error",
    7: "hot",
    8: "guard",
    9: "builtin",
    10: "token",
    11: "rt",  # ABI 5: the process's real-time grants missing (takeover_require_rt)
}
INJECT_POLICY_NAN, INJECT_POLICY_OUT_OF_RANGE = 1, 2
FAULT_BITS: dict[str, int] = {
    "lowstate_late": 1 << 0,
    "lowstate_stale": 1 << 1,
    "lowstate_lost": 1 << 2,
    "crc": 1 << 3,
    "tilt_arm_stop": 1 << 4,
    "tilt_fall": 1 << 5,
    "joint_clamp": 1 << 6,
    "velocity_clamp": 1 << 7,
    "torque_clamp": 1 << 8,
    "motor_error": 1 << 9,
    "policy_late": 1 << 10,
    "policy_invalid": 1 << 11,
    "loop_gap": 1 << 12,
    "not_owner": 1 << 13,
    "guard_absent": 1 << 14,
    "rt_not_granted": 1 << 15,
    "takeover_alarm": 1 << 16,
    "builtin_active": 1 << 17,
    "over_temp": 1 << 18,  # session 6: a warning only (temperature >= the takeover gate)
}

# DESIGN §4.2 start values (v2's, revalidated in sim) and the hard limits.
LOWSTATE_LATE_S = 0.010
LOWSTATE_STALE_S = 0.100
LOWSTATE_LOST_S = 1.0
POLICY_MAX_AGE_S = 0.040
FALL_TILT_DEG = 30.0
ARM_TILT_STOP_DEG = 4.0
STAND_RAMP_S = 0.4  # DESIGN §3.5 STAND_FREEZE: 0.3-0.5 s
ACTION_ABS_MAX = 50.0  # the raw network output never left +-40 on the golden sets (S2: |action| max 39.7, random)
TAU_FF_FRACTION = 0.6  # v2 _cmd clips tau to 0.6 TAU_MAX
KP_MAX = 500.0
KD_MAX = 20.0
DQ_MAX = 30.0
DAMP_KD = np.array([3.0] * 24 + [1.0, 1.0])  # DESIGN §3.5 DAMP: legs kd 3, arms 1-3 (v2 joints.KD_DAMP)
# The takeover (DESIGN §3.5; v2 r1mc2/runtime.py:335-412, server.py:136-137): v2's gates and its default blend (1.0 s,
# `takeover into=stand`; 0.3 s was its `into=walk`). 1.0 s also because rung 3 needs it: an invalid policy action in
# the policy's first ~0.3 s after the takeover froze into a fall with a 0.3 s blend or an instant switch in this
# simulator, not with 1.0 s (tests/integration/test_v3_core_takeover.py; the sweep in the S4 notes).
TAKEOVER_BLEND_S = 1.0
TAKEOVER_TILT_MAX_DEG = 12.0
TAKEOVER_DQ_MAX = 0.5  # rad/s
TAKEOVER_TEMP_MAX_C = 60.0
TAKEOVER_STATE_MAX_AGE_S = 0.05
BUILTIN_MAX_AGE_S = 0.1
BUILTIN_MIN_LEG_KP = 10.0  # stand lock 200, the built-in's walk 84 / 31
MIRROR_MAX_DEV_RAD = 0.6
TAKEOVER_MIRROR_ALARM_S = 5.0  # v2 waited <= 4 s for ReleaseMode
GUARD_MAX_AGE_S = 0.05


# Upper body (types.h / executor.hpp)
GROUP_ARM_LEFT, GROUP_ARM_RIGHT, GROUP_HEAD = 0, 1, 2
GROUP_LEG_LEFT, GROUP_LEG_RIGHT, GROUP_WAIST = 3, 4, 5  # session 5: ON_STAND only (types.h R1RT_GROUP_*)
GROUP_JOINTS: dict[int, tuple[int, ...]] = {
    0: (14, 15, 16, 17, 18),
    1: (19, 20, 21, 22, 23),
    2: (24, 25),
    3: (0, 1, 2, 3, 4, 5),
    4: (6, 7, 8, 9, 10, 11),
    5: (12, 13),
}
SOURCE_HOLD, SOURCE_TRAJECTORY, SOURCE_STREAM, SOURCE_FREEDRIVE = 0, 1, 2, 3
STOP_STOP, STOP_HALT, STOP_DAMP = 0, 1, 2
EVENTS: dict[int, str] = {
    0: "none",
    1: "guard_lag",
    2: "guard_torque",
    3: "guard_contact",
    4: "guard_surface",
    5: "guard_tilt",
    6: "guard_contact_stale",
    16: "deadman",
    17: "stream_ttl",
    18: "operator_stop",
    19: "starved",
    20: "freedrive_end",
    21: "bad_state",
    22: "motor_error",
    23: "resume_after_silence",
    32: "done",
}
MAX_KNOTS = 8  # R1RT_MAX_KNOTS per record
KEEPALIVE_RENEW_S = 0.02  # r1d renews well inside the 50 ms deadman


def min_jerk_knots(
    q0: npt.ArrayLike, q1: npt.ArrayLike, duration_s: float, n_knots: int = MAX_KNOTS
) -> list[tuple[int, FloatArray, FloatArray, FloatArray]]:
    """Knots (t_ns from the start, q, dq, ddq) of a minimum-jerk move q0 -> q1: exact samples of the quintic
    q0 + (q1 - q0)(10 s^3 - 15 s^4 + 6 s^5), so the executor's quintic Hermite between them reproduces it exactly
    (the same polynomial on one interval; between knots of a multi-knot sampling, a C2 Hermite of it)."""
    a = np.asarray(q0, dtype=np.float64)
    d = np.asarray(q1, dtype=np.float64) - a
    out = []
    for i in range(n_knots):
        s = i / (n_knots - 1)
        t = s * duration_s
        p = 10 * s**3 - 15 * s**4 + 6 * s**5
        v = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration_s
        acc = (60 * s - 180 * s**2 + 120 * s**3) / duration_s**2
        out.append((round(t * 1e9), a + d * p, d * v, d * acc))
    return out


def min_jerk_path(
    q0: npt.ArrayLike, points: Sequence[npt.ArrayLike], durations_s: Sequence[float], n_knots: int = MAX_KNOTS
) -> list[tuple[int, FloatArray, FloatArray, FloatArray]]:
    """Knots of consecutive minimum-jerk legs q0 -> points[0] -> points[1] ... (each leg at rest at its ends, so the
    path is C2); the legs' shared knots appear once."""
    knots: list[tuple[int, FloatArray, FloatArray, FloatArray]] = []
    start, t0 = np.asarray(q0, dtype=np.float64), 0
    for p, dur in zip(points, durations_s, strict=True):
        leg = min_jerk_knots(start, p, dur, n_knots)
        knots += [(t0 + t, q, dq, ddq) for t, q, dq, ddq in (leg if not knots else leg[1:])]
        start, t0 = np.asarray(p, dtype=np.float64), t0 + round(dur * 1e9)
    return knots


def min_jerk_path_sample(
    q0: npt.ArrayLike, points: Sequence[npt.ArrayLike], durations_s: Sequence[float], t: float
) -> FloatArray:
    """The position of `min_jerk_path` at time t from its start."""
    start = np.asarray(q0, dtype=np.float64)
    for p, dur in zip(points, durations_s, strict=True):
        if t <= dur:
            return min_jerk_sample(start, p, dur, t)
        t -= dur
        start = np.asarray(p, dtype=np.float64)
    return start


def min_jerk_sample(q0: npt.ArrayLike, q1: npt.ArrayLike, duration_s: float, t: float) -> FloatArray:
    """The minimum-jerk move's position at time t (clamped to [0, duration])."""
    s = min(max(t / duration_s, 0.0), 1.0)
    a = np.asarray(q0, dtype=np.float64)
    return a + (np.asarray(q1, dtype=np.float64) - a) * (10 * s**3 - 15 * s**4 + 6 * s**5)


def trajectory_records(
    cmd_id: int,
    group: int,
    knots: Sequence[tuple[int, Any, Any, Any]],
    *,
    ttl_ns: int = 50_000_000,
    t_issue_ns: int = 0,
    guards: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """A TRAJECTORY as upper_cmd records of <= MAX_KNOTS knots (segment_index 0, 1, ..., last_segment on the last)."""
    n = len(GROUP_JOINTS[group])
    chunks = [knots[i : i + MAX_KNOTS] for i in range(0, len(knots), MAX_KNOTS)]
    recs = []
    for k, ch in enumerate(chunks):
        rec: dict[str, Any] = {
            "cmd_id": cmd_id,
            "group": group,
            "source": SOURCE_TRAJECTORY,
            "n_joints": n,
            "t_issue_ns": t_issue_ns,
            "ttl_ns": ttl_ns,
            "segment_index": k,
            "last_segment": int(k == len(chunks) - 1),
            "knots": [
                (int(t), list(map(float, q)), list(map(float, dq)), list(map(float, ddq))) for t, q, dq, ddq in ch
            ],
        }
        if guards is not None:
            rec["guards"] = dict(guards)
        recs.append(rec)
    return recs


def policy_params(pol: Any) -> dict[str, Any]:
    """The r1rt_policy_params of a r1mc3.policy.ReferencePolicy (its ONNX metadata + r1mc_policy.json)."""
    lr = pol.lean_range or (0.0, 0.0)
    wr = pol.waist_range or ((0.0, 0.0), (0.0, 0.0))
    return {
        "n_joints": int(pol.n),
        "n_act": int(pol.n_act),
        "n_terms": len(pol.obs_names),
        "has_lean": int(pol.lean_range is not None),
        "has_waist": int(pol.waist_range is not None),
        "obs_terms": [OBS_TERMS[n] for n in pol.obs_names],
        "joint_index": [int(i) for i in pol.idx],
        "act_pos": [int(i) for i in pol.act_pos],
        "default_pos": [float(x) for x in pol.default],
        "kp": [float(x) for x in pol.kp],
        "kd": [float(x) for x in pol.kd],
        "action_scale": [float(x) for x in pol.scale],
        "kp_other": [float(x) for x in J.KP_STIFF],
        "kd_other": [float(x) for x in J.KD_STIFF],
        "period_s": float(pol.period),
        "step_dt_s": float(pol.step_dt),
        "cmd_lo": [float(x) for x in pol.cmd_lo],
        "cmd_hi": [float(x) for x in pol.cmd_hi],
        "lean_lo": float(lr[0]),
        "lean_hi": float(lr[1]),
        "lean_rate": float(pol.lean_rate),
        "waist_lo": [float(wr[0][0]), float(wr[1][0])],
        "waist_hi": [float(wr[0][1]), float(wr[1][1])],
        "waist_rate": [float(x) for x in pol.waist_rate],
    }


def pack_params(p: Mapping[str, Any]) -> FloatArray:
    """`policy_params` as one float64 vector in PARAMS_LAYOUT order (unused tail entries 0)."""
    out: list[float] = []
    for name, n in PARAMS_LAYOUT:
        v = np.atleast_1d(np.asarray(p[name], dtype=np.float64))
        if v.size > n:
            raise ValueError(f"{name}: {v.size} values > {n}")
        out.extend(v.tolist() + [0.0] * (n - v.size))
    return np.array(out, dtype=np.float64)


def core_config(
    pol: Any,
    *,
    weights: str | Path | None,
    stand_q: npt.ArrayLike,
    writer_hz: int = 500,
    policy_hz: int = 50,
    upper_kp: npt.ArrayLike | None = None,
    upper_kd: npt.ArrayLike | None = None,
) -> dict[str, Any]:
    """The dict r1rt_py.Core takes (= r1rt_config). Arms + head HOLD gains: the policy metadata's arm gains (60/60/
    45/45/30, kd 2/2/2/2/1.5, as v2 drives the arms under this policy) and KP_STIFF / KD_STIFF for the head."""
    ukp = J.KP_STIFF.copy()
    ukd = J.KD_STIFF.copy()
    pos = {n: i for i, n in enumerate(pol.names)}
    for j in J.ARMS:
        ukp[j] = pol.kp[pos[J.JOINTS[j]]]
        ukd[j] = pol.kd[pos[J.JOINTS[j]]]
    if upper_kp is not None:
        ukp = np.asarray(upper_kp, dtype=np.float64)
    if upper_kd is not None:
        ukd = np.asarray(upper_kd, dtype=np.float64)
    return {
        "writer_hz": int(writer_hz),
        "policy_hz": int(policy_hz),
        "weights": None if weights is None else str(weights),
        "policy": policy_params(pol),
        "q_min": J.Q_MIN.tolist(),
        "q_max": J.Q_MAX.tolist(),
        "dq_max": [DQ_MAX] * J.N,
        "tau_max": (TAU_FF_FRACTION * J.TAU_MAX).tolist(),
        "kp_max": [KP_MAX] * J.N,
        "kd_max": [KD_MAX] * J.N,
        "stand_q": np.asarray(stand_q, dtype=np.float64).tolist(),
        "stand_kp": J.KP_STIFF.tolist(),
        "stand_kd": J.KD_STIFF.tolist(),
        "stand_ramp_s": STAND_RAMP_S,
        "upper_kp": np.asarray(ukp, dtype=np.float64).tolist(),
        "upper_kd": np.asarray(ukd, dtype=np.float64).tolist(),
        "damp_kd": DAMP_KD.tolist(),
        "action_abs_max": ACTION_ABS_MAX,
        "lowstate_late_ns": round(LOWSTATE_LATE_S * 1e9),
        "lowstate_stale_ns": round(LOWSTATE_STALE_S * 1e9),
        "lowstate_lost_ns": round(LOWSTATE_LOST_S * 1e9),
        "policy_max_age_ns": round(POLICY_MAX_AGE_S * 1e9),
        "fall_tilt_rad": float(np.radians(FALL_TILT_DEG)),
        "arm_tilt_stop_rad": float(np.radians(ARM_TILT_STOP_DEG)),
        "stop_time_s": 0.15,
        "loop_gap_ns": round(2e9 / writer_hz),
        "takeover_blend_s": TAKEOVER_BLEND_S,
        "takeover_tilt_max_rad": float(np.radians(TAKEOVER_TILT_MAX_DEG)),
        "takeover_dq_max": TAKEOVER_DQ_MAX,
        "takeover_temp_max_c": TAKEOVER_TEMP_MAX_C,
        "takeover_state_max_age_ns": round(TAKEOVER_STATE_MAX_AGE_S * 1e9),
        "builtin_max_age_ns": round(BUILTIN_MAX_AGE_S * 1e9),
        "builtin_min_leg_kp": BUILTIN_MIN_LEG_KP,
        "mirror_max_dev_rad": MIRROR_MAX_DEV_RAD,
        "takeover_mirror_alarm_ns": round(TAKEOVER_MIRROR_ALARM_S * 1e9),
        "takeover_require_guard": 0,  # no r1guard in this single-core controller (r1sim.v3_guard runs one)
        "guard_max_age_ns": round(GUARD_MAX_AGE_S * 1e9),
    }


def faults_of(word: int) -> list[str]:
    return [k for k, b in FAULT_BITS.items() if word & b]


class EmulatedBuiltin:
    """Unitree's built-in controller as the takeover sees it in the simulator: the S1 stand fixture (a stiff PD at
    the CoM-centred stand, `r1sim.control.StandHold` - Unitree's own balancer is not modelled, S2 "Not solid" 1)
    publishing on rt/lowcmd every call until ReleaseMode succeeds; `release_ok=False` emulates a ReleaseMode that does
    not take (the built-in keeps publishing). Register it as a lockstep controller BEFORE the core (both write the
    topic; the later write of a tick wins, as on DDS). `last` / `t_last` = its last command and when it was sent."""

    def __init__(self, model: Any, *, release_ok: bool = True, leg_kp: float | None = None) -> None:
        from .control import StandHold

        kp = J.KP_STIFF.copy()
        if leg_kp is not None:
            kp[:12] = leg_kp
        self.fixture = StandHold.for_model(model, kp=kp)
        self.release_ok = release_ok
        self.released = False
        self.release_calls = 0
        self.last: LowCmd | None = None
        self.t_last: float | None = None
        self.calls = 0

    def __call__(self, st: LowState) -> LowCmd | None:
        self.calls += 1
        if self.released:
            return None
        cmd = self.fixture(st)
        self.last = cmd
        self.t_last = float(st.t)
        return cmd

    def release(self) -> bool:
        """The motion-switcher's ReleaseMode (r1d's RPC): True = the built-in stops publishing."""
        self.release_calls += 1
        if self.release_ok:
            self.released = True
        return self.release_ok


class TakeoverPlan:
    """r1d's part of the takeover in lockstep (DESIGN §3.5): request TAKEOVER at `t_request_s`; ReleaseMode answers
    after `release_s` (v2 waited up to 4 s); then RELEASED (it succeeded) or ABORT (it did not).

    Session 5: `target="on_stand"` runs the same sequence into ON_STAND (R1RT_REQ_ON_STAND instead of TAKEOVER: the
    robot hanging on its stand); `via_mailbox=True` posts every request through the segment's mode-request mailbox
    (as r1d / the CLI do: requester R1D, processed inside the core's next writer tick) instead of the direct call."""

    def __init__(
        self,
        builtin: EmulatedBuiltin,
        *,
        t_request_s: float = 1.0,
        release_s: float = 0.3,
        target: str = "balance",
        via_mailbox: bool = False,
    ) -> None:
        if target not in ("balance", "on_stand"):
            raise ValueError(f"TakeoverPlan target {target!r}: 'balance' or 'on_stand'")
        self.builtin = builtin
        self.t_request_s = t_request_s
        self.release_s = release_s
        self.target = target
        self.via_mailbox = via_mailbox
        self.seqs: list[int] = []  # mailbox sequence numbers of the requests (via_mailbox)
        self._pending: tuple[str, int] | None = None  # (which, seq) of a mailbox request awaiting its reply
        self.request_status: int | None = None  # R1RT_OK / R1RT_E_REFUSED of TAKEOVER
        self.refusal: str | None = None
        self.t_requested: float | None = None
        self.release_ok: bool | None = None
        self.decision_status: int | None = None  # of RELEASED / ABORT
        self.t_decided: float | None = None


class V3CoreController:
    """The v3 core as a lockstep simulator controller `(LowState) -> LowCmd | None` (see the module docstring).

    `period_s` is the writer period (register it with that period). Hands: held open (Dex3 gains), as the policy
    controllers do - the hands are r1hands' business, not the core's. Records per call: `self.trace` (t, mode, faults,
    inline_active) when `trace=True`."""

    def __init__(
        self,
        model: Any,
        policy_dir: str | Path | None = None,
        *,
        writer_hz: int = 500,
        policy_hz: int = 50,
        faults: FaultSchedule | None = None,
        cmd: Sequence[float] = (0.0, 0.0, 0.0),
        lean_deg: float = 0.0,
        stand_q: npt.ArrayLike | None = None,
        enter_balance: bool = True,
        hands: tuple[str, ...] = dex3.SIDES,
        trace: bool = False,
        trace_upper: bool = False,
        takeover: TakeoverPlan | None = None,
        config: Mapping[str, Any] | None = None,
        trace_cmd: bool = False,
    ) -> None:
        from r1mc3.policy import POLICY_DIR, ReferencePolicy

        from .model import centred_stand_pose

        pdir = Path(policy_dir) if policy_dir is not None else POLICY_DIR
        self.policy = ReferencePolicy.load(pdir)  # metadata only: the core runs its own C++ runner
        onnx = pdir / "policy.onnx" if pdir.is_dir() else pdir
        self.stand_q: FloatArray = (
            np.asarray(stand_q, dtype=np.float64) if stand_q is not None else centred_stand_pose(model)
        )
        mod = import_r1rt_py()
        self.mod = mod
        self.config = core_config(
            self.policy, weights=weights_for(onnx), stand_q=self.stand_q, writer_hz=writer_hz, policy_hz=policy_hz
        )
        if config is not None:  # overrides (tests: e.g. a takeover gate)
            self.config.update(config)
        self.core: Any = mod.Core(self.config)
        self.view: Any = mod.SegmentView.of_core(self.core)  # session 5: the segment as a client sees it (mailbox)
        self.on_stand_entered = False
        self.period_s = 1.0 / writer_hz
        self.ratio = writer_hz // policy_hz
        self.faults = faults
        self.cmd: FloatArray = np.asarray(cmd, dtype=np.float64).copy()
        self.lean_rad = float(np.radians(lean_deg))
        self.takeover = takeover
        self.enter_balance = enter_balance and takeover is None  # a takeover enters BALANCE through MIRROR / BLEND
        self.entered = False
        self.enter_status: int | None = None
        self._hands = hands
        self._k_entry = 0
        self.calls = 0
        self.trace_on = trace
        self.trace: list[tuple[float, int, int, int]] = []
        # per writer tick when trace_upper: (t, mode, stop_active, source x3, event x3, stopping x3, q_cmd[14:26])
        self.trace_upper_on = trace_upper
        self.upper_trace: list[tuple[float, ...]] = []
        # per writer tick when trace_cmd: (t, publish, mode, faults, q_cmd[26], kp[26])
        self.trace_cmd_on = trace_cmd
        self.cmd_trace: list[tuple[float, int, int, int, FloatArray, FloatArray]] = []
        self.last_publish = 0
        self.last_q: FloatArray = np.zeros(J.N)  # the last published joint targets (the executor's upper targets)
        self._base_seq = 0
        self._plan: list[tuple[float, Callable[[V3CoreController, int], object]]] = []
        self._keepalive: dict[int, tuple[int, float]] = {}  # group -> (cmd_id, renew until t)
        self._keepalive_last: dict[int, float] = {}
        self._next_cmd_id = 1
        self.push_results: list[int] = []
        if faults is not None:  # network-output faults go into the core (both the thread and the inline rung)
            for f in faults.faults:
                if f.kind in ("policy_nan", "policy_out_of_range"):
                    kind = INJECT_POLICY_NAN if f.kind == "policy_nan" else INJECT_POLICY_OUT_OF_RANGE
                    val = float(f.params.get("value", 0.0))
                    t0 = round(f.t_start * 1e9)
                    self.core.inject(kind, t0, t0 + round(f.duration * 1e9), val)

    def _stalled(self, name: str, t: float) -> bool:
        if self.faults is None:
            return False
        for f in self.faults.active("controller_stall", t):
            if f.params.get("name") == name:
                self.faults.record(t, "controller_stall", "skip_call", name=name)
                return True
        return False

    def _base_cmd(self, t_ns: int) -> None:
        if not np.any(self.cmd) and self.lean_rad == 0.0:
            return
        self._base_seq += 1
        self.core.set_base_cmd(
            self._base_seq, t_ns, 500_000_000, float(self.cmd[0]), float(self.cmd[1]), float(self.cmd[2]), self.lean_rad
        )

    def _step(self, t_ns: int, st: LowState | None) -> tuple[int, Any, Any, Any, Any, Any]:
        if st is None:
            out: tuple[int, Any, Any, Any, Any, Any] = self.core.step(t_ns, None)
            return out
        state = {
            "t_rx_ns": t_ns,
            "board_tick": int(st.tick),
            "q": st.q,
            "dq": st.dq,
            "tau_est": st.tau_est,
            "quat": st.quat,
            "gyro": st.gyro,
            "acc": st.acc,
            "motor_state": np.asarray(st.motor_err, dtype=np.int64),
            "temperature": np.asarray(st.motor_temp, dtype=np.float64),
        }
        res: tuple[int, Any, Any, Any, Any, Any] = self.core.step(t_ns, state)
        return res

    # ---- r1d's part for the upper body (records into the segment, keepalives) ----
    def at(self, t_s: float, fn: Callable[[V3CoreController, int], object]) -> None:
        """Run fn(self, t_ns) before the first writer tick at or after simulated time t_s."""
        self._plan.append((t_s, fn))
        self._plan.sort(key=lambda e: e[0])

    def _push(self, rec: Mapping[str, Any]) -> int:
        rc = int(self.core.push_upper(dict(rec)))
        self.push_results.append(rc)
        return rc

    def new_cmd_id(self) -> int:
        cid = self._next_cmd_id
        self._next_cmd_id += 1
        return cid

    def send_trajectory(
        self,
        t_ns: int,
        group: int,
        q_goal: npt.ArrayLike,
        duration_s: float,
        *,
        n_knots: int = MAX_KNOTS,
        keepalive_until_s: float = math.inf,
        guards: Mapping[str, Any] | None = None,
        q_start: npt.ArrayLike | None = None,
    ) -> int:
        """A minimum-jerk TRAJECTORY of `group` from its current target (the last published command) to q_goal;
        its keepalive is renewed every KEEPALIVE_RENEW_S until keepalive_until_s (simulated time). Returns cmd_id."""
        return self.send_path(
            t_ns,
            group,
            [q_goal],
            [duration_s],
            n_knots=n_knots,
            keepalive_until_s=keepalive_until_s,
            guards=guards,
            q_start=q_start,
        )

    def send_path(
        self,
        t_ns: int,
        group: int,
        points: Sequence[npt.ArrayLike],
        durations_s: Sequence[float],
        *,
        n_knots: int = MAX_KNOTS,
        keepalive_until_s: float = math.inf,
        guards: Mapping[str, Any] | None = None,
        q_start: npt.ArrayLike | None = None,
    ) -> int:
        """One TRAJECTORY through `points` (minimum-jerk legs, `min_jerk_path`), sent as consecutive records of <=
        MAX_KNOTS knots (multi-segment); keepalive as send_trajectory. Returns cmd_id."""
        joints = list(GROUP_JOINTS[group])
        q0 = np.asarray(q_start, dtype=np.float64) if q_start is not None else self.last_q[joints].copy()
        cid = self.new_cmd_id()
        knots = min_jerk_path(q0, points, durations_s, n_knots)
        for rec in trajectory_records(cid, group, knots, t_issue_ns=t_ns, guards=guards):
            self._push(rec)
        self._keepalive[group] = (cid, keepalive_until_s)
        self._keepalive_last[group] = t_ns * 1e-9
        self.core.upper_keepalive(group, cid, t_ns)
        return cid

    def send_stream(self, t_ns: int, group: int, q: npt.ArrayLike, ttl_s: float, *, cmd_id: int | None = None) -> int:
        """One STREAM setpoint (TTL ttl_s from t_ns). Returns cmd_id (pass it again to continue the same stream)."""
        cid = cmd_id if cmd_id is not None else self.new_cmd_id()
        n = len(GROUP_JOINTS[group])
        self._push(
            {
                "cmd_id": cid,
                "group": group,
                "source": SOURCE_STREAM,
                "n_joints": n,
                "t_issue_ns": t_ns,
                "ttl_ns": round(ttl_s * 1e9),
                "q": list(map(float, np.asarray(q, dtype=np.float64))),
                "dq": [0.0] * n,
            }
        )
        return cid

    def send_hold(self, t_ns: int, group: int, q: npt.ArrayLike) -> int:
        cid = self.new_cmd_id()
        n = len(GROUP_JOINTS[group])
        self._push(
            {
                "cmd_id": cid,
                "group": group,
                "source": SOURCE_HOLD,
                "n_joints": n,
                "t_issue_ns": t_ns,
                "q": list(map(float, np.asarray(q, dtype=np.float64))),
            }
        )
        return cid

    def _upper_inputs(self, t: float, t_ns: int) -> None:
        while self._plan and self._plan[0][0] <= t + 1e-12:
            _, fn = self._plan.pop(0)
            fn(self, t_ns)
        for g, (cid, until) in self._keepalive.items():
            if t < until and t - self._keepalive_last.get(g, -1.0) >= KEEPALIVE_RENEW_S - 1e-9:
                self.core.upper_keepalive(g, cid, t_ns)
                self._keepalive_last[g] = t

    def _takeover_inputs(self, t: float, t_ns: int) -> None:
        """r1d's part of the takeover + the DDS layer's built-in command, before the writer tick of time t."""
        plan = self.takeover
        assert plan is not None
        b = plan.builtin
        if b.last is not None and b.t_last is not None and not b.released:  # foreign samples only (not ours)
            c = b.last
            self.core.set_builtin_cmd(c.q, c.dq, c.kp, c.kd, c.tau_ff, None, round(b.t_last * 1e9))
        first_req = REQ_ON_STAND if plan.target == "on_stand" else REQ_TAKEOVER
        if plan.via_mailbox:
            self._takeover_mailbox(plan, t, t_ns, first_req)
            return
        if plan.t_requested is None and t >= plan.t_request_s - 1e-12:
            plan.request_status = int(self.core.request(first_req, t_ns))
            key = "on_stand_refusal" if plan.target == "on_stand" else "takeover_refusal"
            plan.refusal = TAKEOVER_REFUSALS.get(int(self.core.status()[key]), "?")
            plan.t_requested = t
        elif (
            plan.t_requested is not None
            and plan.request_status == 0
            and plan.t_decided is None
            and t >= plan.t_requested + plan.release_s - 1e-12
        ):
            plan.release_ok = b.release()  # the motion-switcher's ReleaseMode (an RPC in r1d, never in the RT thread)
            req = REQ_TAKEOVER_RELEASED if plan.release_ok else REQ_TAKEOVER_ABORT
            plan.decision_status = int(self.core.request(req, t_ns))
            plan.t_decided = t
            self._after_decision(plan)

    def _after_decision(self, plan: TakeoverPlan) -> None:
        if plan.release_ok and int(self.core.mode) in (MODE_TAKEOVER_BLEND, MODE_STAND_FREEZE):
            self.entered = plan.target == "balance"  # the policy thread runs from here (BLEND, then BALANCE)
            self._k_entry = self.calls
        if plan.release_ok and plan.target == "on_stand":
            self.on_stand_entered = True  # no policy on the stand; the upper / lower records flow from here

    def _takeover_mailbox(self, plan: TakeoverPlan, t: float, t_ns: int, first_req: int) -> None:
        """The same sequence through the segment's mode-request mailbox (requester R1D): a request posted here is
        processed inside this tick's writer step; its reply is read before the next request (`_mailbox_replies`)."""
        if plan._pending is not None:
            return  # the previous request's reply has not been read yet
        if plan.t_requested is None and t >= plan.t_request_s - 1e-12:
            seq = int(self.view.request_mode(first_req, REQUESTER_R1D, t_ns))
            plan.seqs.append(seq)
            plan._pending = ("request", seq)
            plan.t_requested = t
        elif (
            plan.t_requested is not None
            and plan.request_status == 0
            and plan.t_decided is None
            and t >= plan.t_requested + plan.release_s - 1e-12
        ):
            plan.release_ok = plan.builtin.release()
            req = REQ_TAKEOVER_RELEASED if plan.release_ok else REQ_TAKEOVER_ABORT
            seq = int(self.view.request_mode(req, REQUESTER_R1D, t_ns))
            plan.seqs.append(seq)
            plan._pending = ("decision", seq)
            plan.t_decided = t

    def _mailbox_replies(self) -> None:
        plan = self.takeover
        if plan is None or plan._pending is None:
            return
        which, seq = plan._pending
        rep = self.view.mode_reply(seq)
        if rep is None:
            return
        status, refusal, _mode = (int(x) for x in rep)
        plan._pending = None
        if which == "request":
            plan.request_status = status
            plan.refusal = TAKEOVER_REFUSALS.get(refusal, "?")
        else:
            plan.decision_status = status
            self._after_decision(plan)

    def __call__(self, st: LowState) -> LowCmd | None:
        t = float(st.t)
        t_ns = round(t * 1e9)
        deliver = not self._stalled("v3_lowstate", t)
        self._base_cmd(t_ns)
        if self.takeover is not None:
            self._takeover_inputs(t, t_ns)
        if self.entered or self.on_stand_entered:
            self._upper_inputs(t, t_ns)
        res = self._step(t_ns, st if deliver else None)
        if self.takeover is not None and self.takeover.via_mailbox:
            self._mailbox_replies()  # the mailbox requests posted above were processed inside this step
        if self.enter_balance and not self.entered:
            self.enter_status = int(self.core.request(REQ_ENTER_BALANCE, t_ns))
            self.entered = self.enter_status == 0
            self._k_entry = self.calls
            res = self._step(t_ns, None)  # the same instant, now in BALANCE (the entry action is in the mailbox)
        k = self.calls - self._k_entry
        if self.entered and k > 0 and k % self.ratio == 0 and not self._stalled("v3_policy", t):
            self.core.policy_step(t_ns)
        self.calls += 1
        publish, q, dq, kp, kd, tau = res
        self.last_publish = int(publish)
        if publish:
            self.last_q = np.asarray(q, dtype=np.float64).copy()
        if self.trace_on or self.trace_upper_on or self.trace_cmd_on:
            s = self.core.status()
            if self.trace_cmd_on:
                self.cmd_trace.append(
                    (
                        t,
                        int(publish),
                        int(s["mode"]),
                        int(s["faults"]),
                        np.asarray(q, dtype=np.float64).copy(),
                        np.asarray(kp, dtype=np.float64).copy(),
                    )
                )
            if self.trace_on:
                self.trace.append((t, int(s["mode"]), int(s["faults"]), int(s["inline_active"])))
            if self.trace_upper_on:
                self.upper_trace.append(
                    (
                        t,
                        float(s["mode"]),
                        float(s["stop_active"]),
                        *map(float, s["upper_source"]),
                        *map(float, s["upper_event"]),
                        *map(float, s["upper_stopping"]),
                        *np.asarray(q, dtype=np.float64)[14:26].tolist(),
                    )
                )
        if not publish:
            return None
        cmd = LowCmd()
        cmd.q[:] = q
        cmd.dq[:] = dq
        cmd.kp[:] = kp
        cmd.kd[:] = kd
        cmd.tau_ff[:] = tau
        for side in self._hands:
            h = HandCmd()
            h.q[:] = dex3.preset("open", side)
            h.kp[:] = dex3.KP
            h.kd[:] = dex3.KD
            cmd.hands[side] = h
        return cmd

    def status(self) -> dict[str, Any]:
        s: dict[str, Any] = self.core.status()
        return s
