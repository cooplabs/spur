"""HW1 Tier B recording plans (DESIGN.md §13.2: B1 joint dynamics, B3 masses, B4 hands, the IMU statics; §14.1).

Every plan is a pure function of (kind, params, start pose): `plan(...)` returns the setpoints as a function of time,
its duration, the joints that move and the checks that refuse it BEFORE anything moves (DESIGN R-f): limits with a
margin, the executor's speed cap (dq_max 2 rad/s), one joint at a time for steps / chirps. Defaults are small on
purpose (the robot hangs on its stand; the user OKs each block on the day); amplitudes and frequencies are ESTIMATES
until the first recording says what the joints tolerate.

| kind | moves | what it is for |
|---|---|---|
| step | one joint: hold, +A, hold, base, hold, -A, ... | B1 step response; the stiction fit (ramps are `ramp`) |
| ramp | one joint, SOFT servo: settle, then +A, -A, 0 at 0.1 and 0.3 rad/s | U4: leg stiction (HW1_FITS.md) |
| chirp | one joint: A sin(2 pi (f0 t + (f1 - f0) t^2 / 2T)), 0.5 s raised-cosine fades | B1 phase lag 2 / 5 / 10 Hz |
| static_pose | one group to q, hold (+ the declared payload) | B3 static torques |
| imu_static | nothing | IMU statics (bias, the mount offset: HW1_IMU_LEVEL.md) |
| pad_press | nothing (the operator presses known weights; the hand holds open) | B4 pad pressure vs weight |
| finger_hold | one finger closes with a constant torque (kp 0, tau = N·m) | A10 hand heat (0.9 / 0.45 N·m) |

Session 9 (r1cam, R1CAM_PLAN.md; HW1_RUNBOOK_ADDITIONS S6 / S7 = HW2's S11 / S12): the CAMERA kinds - nothing moves
(the head pose comes before, with the ordinary head words); r1d asks r1cam (127.0.0.1:8781) for a recording for the
plan's duration and the manifest carries r1cam's:

| kind | moves | what it is for |
|---|---|---|
| flat_field | nothing (a white sheet fills the view; per light level, then turned 90 deg) | vignetting (B5) |
| charuco | nothing (the board held still per distance x place x tilt: 2 s still, 2 s move) | the lens model (B5) |
| depth_targets | nothing (a target at a tape-measured distance, still) | depth residuals (B5 / A6) |
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from r1mc3.service.codes import GROUP_JOINTS, JOINT_NAMES, LOWER_GROUPS, group_of_joint

FloatArray = npt.NDArray[np.float64]

DQ_MAX = 2.0  # rad/s: the executor's cap (EXECUTOR.md; an ESTIMATE until HW1)
LIMIT_MARGIN = 0.05  # rad inside the profile's position limits
FADE_S = 0.5  # chirp fade in / out
FINGERS = {"thumb": (0, 1, 2), "middle": (3, 4), "index": (5, 6)}  # Dex3 DDS order
FINGER_TAU_MAX = 1.4  # N m (URDF effort, fingers; thumb_0 2.45)
KINDS = ("step", "ramp", "chirp", "static_pose", "imu_static", "pad_press", "finger_hold")
CAMERA_KINDS = ("flat_field", "charuco", "depth_targets")  # session 9 (r1cam): nothing moves
CHARUCO_PLACES = ("top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right")
CHARUCO_TILTS = ("flat", "turned ~30 deg left-right", "tipped ~30 deg")
# Words the HW1 runbook drafts used (docs/hardware/HW1_RUNBOOK_ADDITIONS.md S1 / S4) -> the kinds that do it:
# `statics` = the read-only joints + IMU recording (imu_static: nothing moves; e.g. label=com_board for the CoM board);
# `torque_ramp` = the stiction protocol (ramp: the soft-servo ramp the fit expects, r1sim.sysid.stiction).
ALIASES = {"statics": "imu_static", "torque_ramp": "ramp"}
# The stiction protocol's defaults: the fit's own (r1sim.sysid.stiction: SPEEDS, HOLD_S, SETTLE_S, KP_TEST, KD_TEST,
# AMPLITUDE; the hwfit agent, session 5) - copied because r1mc3 must not import the simulator; a test keeps them equal.
RAMP_SPEEDS = (0.1, 0.3)
RAMP_HOLD_S = 1.0
RAMP_SETTLE_S = 1.5
RAMP_KP = {"hip": 40.0, "knee": 40.0, "ankle": 25.0, "waist": 40.0}
RAMP_KD = {"hip": 3.0, "knee": 1.0, "ankle": 1.0, "waist": 3.0}
RAMP_AMPLITUDE = {"hip_pitch": 0.25, "hip_roll": 0.15, "hip_yaw": 0.2, "knee": 0.25, "ankle_pitch": 0.2,
                  "ankle_roll": 0.12, "waist_roll": 0.1, "waist_yaw": 0.2}  # fmt: skip


def _leg_kind(name: str) -> str:
    for k in ("ankle_pitch", "ankle_roll", "hip_pitch", "hip_roll", "hip_yaw", "knee", "waist_roll", "waist_yaw"):
        if name.endswith(k):
            return k
    return name


def _family(name: str) -> str:
    k = _leg_kind(name)
    return "ankle" if k.startswith("ankle") else "waist" if k.startswith("waist") else k.split("_")[0]


def ramp_kp(name: str) -> float:
    """The soft servo's kp for the joint under test (waist roll carries the torso above it: much stiffer)."""
    if _leg_kind(name) == "waist_roll":
        return 150.0
    return RAMP_KP.get(_family(name), 20.0)


def ramp_kd(name: str) -> float:
    if _leg_kind(name) == "waist_roll":
        return 8.0
    return RAMP_KD.get(_family(name), 1.0)


class PlanError(ValueError):
    """The plan is refused before anything moves (details in `.details`)."""

    def __init__(self, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.details = {"reason": reason, **details}


@dataclass
class Plan:
    kind: str
    params: dict[str, Any]
    duration_s: float
    group: str | None  # the body group whose setpoints it streams (None: no body motion)
    joints: tuple[int, ...]  # the joints that move (SDK order)
    base: FloatArray | None  # the group's start pose
    setpoint: Callable[[float], tuple[FloatArray, FloatArray]] | None  # t -> (q, dq) of the group
    hand: dict[str, Any] | None = None  # finger_hold / pad_press: the hand command
    camera: dict[str, Any] | None = None  # session 9: the r1cam recording (streams, depth_every, schedule)
    gains: tuple[FloatArray, FloatArray] | None = None  # (kp, kd) of the group's records; 0 / 0 = its hold gains
    needs_mode: str | None = "on_stand"
    operator: list[str] = field(default_factory=list)  # what the operator does / watches
    summary: dict[str, Any] = field(default_factory=dict)

    def at(self, t: float) -> tuple[FloatArray, FloatArray]:
        """The group's (q, dq) at t seconds into the plan (body plans only)."""
        if self.setpoint is None:
            raise ValueError(f"{self.kind} moves no body joint")
        return self.setpoint(t)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "params": self.params,
            "duration_s": round(self.duration_s, 3),
            "group": self.group,
            "moves": [JOINT_NAMES[j] for j in self.joints],
            "hand": self.hand,
            "needs_mode": self.needs_mode,
            "operator": self.operator,
            "how_to_stop": "`r1 stop` (or the remote): the stream ends, the joint decelerates to HOLD (TTL 0.1 s)",
            **self.summary,
        }


def _joint(params: Mapping[str, Any]) -> int:
    j = params.get("joint")
    if isinstance(j, str) and j in JOINT_NAMES:
        return JOINT_NAMES.index(j)
    if isinstance(j, int) and 0 <= j < len(JOINT_NAMES):
        return j
    raise PlanError("joint: a joint name (e.g. left_knee) or an index 0-25", joint=j)


def _num(params: Mapping[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    v = float(params.get(key, default))
    if not (lo <= v <= hi) or not math.isfinite(v):
        raise PlanError(f"{key}: within [{lo}, {hi}]", **{key: v})
    return v


def _check_range(
    group: str, base: FloatArray, lo_q: FloatArray, hi_q: FloatArray, limits: Mapping[str, tuple[float, float]] | None
) -> None:
    if limits is None:
        return
    for k, j in enumerate(GROUP_JOINTS[group]):
        lo, hi = limits[JOINT_NAMES[j]]
        if lo_q[k] < lo + LIMIT_MARGIN or hi_q[k] > hi - LIMIT_MARGIN:
            raise PlanError(
                "the plan leaves the joint's limits (with margin)",
                joint=JOINT_NAMES[j],
                range=[float(lo_q[k]), float(hi_q[k])],
                limits=[lo + LIMIT_MARGIN, hi - LIMIT_MARGIN],
            )


def plan(
    kind: str,
    params: Mapping[str, Any],
    start_q: FloatArray | None,
    limits: Mapping[str, tuple[float, float]] | None = None,
) -> Plan:
    """Build and check one plan. `start_q`: the 26 current joint targets (None for a dry run without the robot: the
    body plans then use zeros as the base and say so)."""
    p = dict(params)
    if kind in ALIASES:
        p.setdefault("asked_as", kind)
        kind = ALIASES[kind]
    if "label" in p and not (isinstance(p["label"], str) and p["label"].replace("_", "").replace("-", "").isalnum()):
        raise PlanError("label: letters, digits, - and _ only (it goes into the file names)", label=p["label"])
    q26 = np.zeros(len(JOINT_NAMES)) if start_q is None else np.asarray(start_q, dtype=np.float64)
    note = {} if start_q is not None else {"base": "zeros (dry run without the robot's pose)"}
    if kind in ("step", "ramp", "chirp"):
        j = _joint(p)
        group = group_of_joint(j)
        cols = GROUP_JOINTS[group]
        k = cols.index(j)
        base = q26[list(cols)].copy()
        unit = np.zeros(len(cols))
        unit[k] = 1.0
        gains: tuple[FloatArray, FloatArray] | None = None
        if "kp" in p and kind != "ramp":  # the joint under test at these gains; the rest of the group at hold gains
            kp_t, kd_t = _num(p, "kp", 0.0, 1.0, 300.0), _num(p, "kd", 1.0, 0.0, 20.0)
            gains = (unit * kp_t, unit * kd_t)
        if kind == "step":
            a = _num(p, "amplitude_rad", 0.05, 0.005, 0.3)
            hold = _num(p, "hold_s", 2.0, 0.5, 10.0)
            reps = int(_num(p, "repeats", 2, 1, 10))
            seq = [0.0] + [a, 0.0, -a, 0.0] * reps
            dur = hold * len(seq)
            p.update(amplitude_rad=a, hold_s=hold, repeats=reps)

            def sp_step(t: float) -> tuple[FloatArray, FloatArray]:
                i = min(int(t / hold), len(seq) - 1)
                return base + seq[i] * unit, np.zeros(len(cols))

            sp = sp_step
            lo_q, hi_q, peak = base - a * unit, base + a * unit, DQ_MAX  # the executor caps the step's speed
        elif kind == "ramp":
            # The stiction protocol of docs/hardware/HW1_FITS.md (= r1sim.sysid.stiction.ramp_plan, the fit's own;
            # tests/unit/test_record_ramp_matches_fit.py checks the knots): settle, then for each speed: to +A, hold,
            # to -A, hold, back to 0, hold - a SOFT servo (kp_test / kd_test on the joint under test; the rest of the
            # group at its hold gains) turns the slow target ramp into a bounded torque ramp.
            kind_ = _leg_kind(JOINT_NAMES[j])
            a = _num(p, "amplitude_rad", RAMP_AMPLITUDE.get(kind_, 0.1), 0.01, 0.6)
            speeds = [float(v) for v in p.get("speeds", RAMP_SPEEDS)]
            if not speeds or any(not 0.005 <= v <= 0.5 for v in speeds):
                raise PlanError("speeds: 0.005..0.5 rad/s each", speeds=speeds)
            hold = _num(p, "hold_s", RAMP_HOLD_S, 0.0, 10.0)
            settle = _num(p, "settle_s", RAMP_SETTLE_S, 0.0, 10.0)
            ts, xs = [0.0, settle], [0.0, 0.0]
            for v in speeds:
                for target in (a, -a, 0.0):
                    ts.append(ts[-1] + abs(target - xs[-1]) / v)
                    xs.append(target)
                    ts.append(ts[-1] + hold)
                    xs.append(target)
            kt = np.asarray(ts)
            kx = np.asarray(xs)
            dur = float(kt[-1])
            kp_t = _num(p, "kp", ramp_kp(JOINT_NAMES[j]), 1.0, 300.0)
            kd_t = _num(p, "kd", ramp_kd(JOINT_NAMES[j]), 0.0, 20.0)
            p.update(amplitude_rad=a, speeds=speeds, hold_s=hold, settle_s=settle, kp=kp_t, kd=kd_t)
            gains = (unit * kp_t, unit * kd_t)  # 0 / 0 elsewhere = the group's hold gains (the executor's rule)

            def sp_ramp(t: float) -> tuple[FloatArray, FloatArray]:
                off = float(np.interp(t, kt, kx))
                i = int(np.searchsorted(kt, t, side="right")) - 1
                v = 0.0 if i < 0 or i >= len(kt) - 1 else (kx[i + 1] - kx[i]) / max(kt[i + 1] - kt[i], 1e-12)
                return base + off * unit, v * unit

            sp = sp_ramp
            lo_q, hi_q, peak = base - a * unit, base + a * unit, max(speeds)
        else:
            a = _num(p, "amplitude_rad", 0.03, 0.005, 0.2)
            f0 = _num(p, "f0_hz", 0.5, 0.05, 20.0)
            f1 = _num(p, "f1_hz", 5.0, 0.1, 20.0)
            dur = _num(p, "duration_s", 20.0, 2.0, 120.0)
            p.update(amplitude_rad=a, f0_hz=f0, f1_hz=f1, duration_s=dur)

            def sp_chirp(t: float) -> tuple[FloatArray, FloatArray]:
                t = min(max(t, 0.0), dur)
                ph = 2 * math.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))
                w = 2 * math.pi * (f0 + (f1 - f0) * t / dur)
                env, denv = 1.0, 0.0  # raised-cosine fades: q, dq and ddq continuous (no velocity step)
                if t < FADE_S:
                    env, denv = (
                        0.5 * (1 - math.cos(math.pi * t / FADE_S)),
                        0.5 * math.pi / FADE_S * math.sin(math.pi * t / FADE_S),
                    )
                elif t > dur - FADE_S:
                    u = dur - t
                    env, denv = (
                        0.5 * (1 - math.cos(math.pi * u / FADE_S)),
                        -0.5 * math.pi / FADE_S * math.sin(math.pi * u / FADE_S),
                    )
                q = a * env * math.sin(ph)
                dq = a * (denv * math.sin(ph) + env * w * math.cos(ph))
                return base + q * unit, dq * unit

            sp = sp_chirp
            lo_q, hi_q = base - a * unit, base + a * unit
            peak = a * (2 * math.pi * max(f0, f1) + 0.5 * math.pi / FADE_S)  # a bound on |dq|
        if peak > DQ_MAX + 1e-9 and kind != "step":
            raise PlanError("faster than the executor's dq_max", peak_dq=peak, dq_max=DQ_MAX)
        _check_range(group, base, lo_q, hi_q, limits)
        return Plan(
            kind=kind,
            params=p,
            duration_s=dur,
            group=group,
            joints=(j,),
            base=base,
            setpoint=sp,
            gains=gains,
            needs_mode="on_stand",
            operator=["robot on its stand, feet off the floor", f"watch {JOINT_NAMES[j]}"],
            summary={"peak_dq_rad_s": round(peak, 4), "range_rad": [float(lo_q[k]), float(hi_q[k])], **note},
        )
    if kind == "static_pose":
        group = str(p.get("group", ""))
        if group not in GROUP_JOINTS:
            raise PlanError("group: arm_left / arm_right / head / leg_left / leg_right / waist", group=group)
        cols = GROUP_JOINTS[group]
        goal = np.asarray(p.get("q", []), dtype=np.float64)
        if goal.shape != (len(cols),) or not np.all(np.isfinite(goal)):
            raise PlanError(f"q: {len(cols)} finite values for {group}")
        move = _num(p, "move_s", 3.0, 1.0, 20.0)
        hold = _num(p, "hold_s", 10.0, 1.0, 120.0)
        payload = _num(p, "payload_kg", 0.0, 0.0, 2.0)
        base = q26[list(cols)].copy()
        p.update(move_s=move, hold_s=hold, payload_kg=payload, q=goal.tolist())
        d = goal - base
        peak = float(np.max(np.abs(d))) * 1.875 / move  # minimum jerk peak speed

        def sp_pose(t: float) -> tuple[FloatArray, FloatArray]:
            s = min(max(t / move, 0.0), 1.0)
            h = 10 * s**3 - 15 * s**4 + 6 * s**5
            dh = (30 * s**2 - 60 * s**3 + 30 * s**4) / move
            return base + h * d, dh * d

        if peak > DQ_MAX:
            raise PlanError("faster than the executor's dq_max", peak_dq=peak, dq_max=DQ_MAX)
        _check_range(group, base, np.minimum(base, goal), np.maximum(base, goal), limits)
        return Plan(
            kind=kind,
            params=p,
            duration_s=move + hold,
            group=group,
            joints=tuple(cols),
            base=base,
            setpoint=sp_pose,
            needs_mode="on_stand",
            operator=["robot on its stand", f"payload {payload} kg declared (hang it before the run)"],
            summary={"peak_dq_rad_s": round(peak, 4), **note},
        )
    if kind == "imu_static":
        dur = _num(p, "duration_s", 30.0, 5.0, 600.0)
        p.update(duration_s=dur)
        return Plan(
            kind=kind,
            params=p,
            duration_s=dur,
            group=None,
            joints=(),
            base=None,
            setpoint=None,
            needs_mode=None,
            operator=["nothing moves: keep the robot still (and level, see HW1_IMU_LEVEL.md)"],
        )
    if kind == "pad_press":
        # docs/hardware/HW1_FITS.md (r1sim.sysid.pads): a weighed mass on one pad at a measured place x (mm along the
        # finger), `press_s` loaded then `rest_s` unloaded (the fit's baseline is the 1 s before each press). Either the
        # GAIN series (weights_g at one x_mm) or the FOOTPRINT series (one weight_g stepped along x_mm: a list).
        side = str(p.get("side", ""))
        if side not in ("left", "right"):
            raise PlanError("side: left / right", side=side)
        pad = int(_num(p, "pad", 0, 0, 8))
        xs = p.get("x_mm", 0.0)
        if isinstance(xs, list):
            w = _num(p, "weight_g", 200.0, 1.0, 2000.0)
            presses = [(w, float(x)) for x in xs]
        else:
            weights = [float(v) for v in p.get("weights_g", [50, 100, 200, 500])]
            if not weights or any(v <= 0 or v > 2000 for v in weights):
                raise PlanError("weights_g: 1..2000 g each", weights_g=weights)
            presses = [(v, float(xs)) for v in weights]
        if not presses:
            raise PlanError("no presses")
        per = _num(p, "press_s", 4.0, 2.0, 30.0)
        rest = _num(p, "rest_s", 4.0, 2.0, 30.0)
        quiet = 2.0
        sched = [
            {"weight_g": w, "x_mm": x, "t_start": quiet + k * (per + rest), "t_end": quiet + k * (per + rest) + per}
            for k, (w, x) in enumerate(presses)
        ]
        p.update(side=side, pad=pad, press_s=per, rest_s=rest, presses=[[w, x] for w, x in presses])
        return Plan(
            kind=kind,
            params=p,
            duration_s=quiet + len(presses) * (per + rest),
            group=None,
            joints=(),
            base=None,
            setpoint=None,
            hand={"side": side, "preset": "open", "pad": pad, "schedule": sched},
            needs_mode=None,
            operator=[
                f"t {e['t_start']:g}-{e['t_end']:g} s: rest {e['weight_g']:g} g on pad {pad} at x {e['x_mm']:g} mm"
                for e in sched
            ],
        )
    if kind == "finger_hold":
        side = str(p.get("side", ""))
        finger = str(p.get("finger", "index"))
        if side not in ("left", "right") or finger not in FINGERS:
            raise PlanError("side: left / right; finger: thumb / middle / index", side=side, finger=finger)
        tau = _num(p, "torque_nm", 0.9, 0.05, FINGER_TAU_MAX)
        dur = _num(p, "duration_s", 60.0, 5.0, 300.0)
        p.update(side=side, finger=finger, torque_nm=tau, duration_s=dur)
        return Plan(
            kind=kind,
            params=p,
            duration_s=dur,
            group=None,
            joints=(),
            base=None,
            setpoint=None,
            hand={"side": side, "finger": finger, "motors": list(FINGERS[finger]), "torque_nm": tau},
            needs_mode=None,
            operator=[
                f"the {side} {finger} closes on a held object (or a stop) with {tau} N m for {dur:g} s",
                "watch the hand temperature (r1 hand-state); `r1 stop` ends it",
            ],
        )
    if kind in CAMERA_KINDS:
        return _camera_plan(kind, p)
    raise PlanError(f"kind: one of {', '.join(KINDS + CAMERA_KINDS)}", kind=kind)


def _camera_plan(kind: str, p: dict[str, Any]) -> Plan:
    """The r1cam kinds (session 9): a schedule of operator prompts and the recording's streams; nothing moves."""
    sched: list[dict[str, Any]] = []
    if kind == "flat_field":
        levels = int(_num(p, "levels", 3, 1, 6))
        frames = int(_num(p, "frames_per_level", 10, 3, 60))
        turn = bool(p.get("turn", True))
        hold = 2.0 + frames / 7.5  # settle + the frames at the image rate (~7.5 FPS)
        t = 0.0
        for k in range(levels):
            for turned in (False, True) if turn else (False,):
                say = f"light level {k + 1}/{levels}: the white sheet fills the view" + (", turned" if turned else "")
                sched.append({"t_start": round(t, 2), "t_end": round(t + hold, 2), "say": say})
                t += hold + 3.0  # 3 s to change the light / turn the sheet
        cam = {"streams": ["left", "right"], "depth_every": 1, "video": False}
        p.update(levels=levels, frames_per_level=frames, turn=turn)
    elif kind == "charuco":
        dists = [float(v) for v in p.get("distances_m", [0.3, 0.45, 0.6])]
        if not dists or any(not 0.15 <= v <= 1.5 for v in dists):
            raise PlanError("distances_m: 0.15..1.5 m each", distances_m=dists)
        hold = _num(p, "hold_s", 2.0, 1.0, 10.0)
        move = _num(p, "move_s", 2.0, 1.0, 10.0)
        t = 0.0
        for z in dists:
            for place in CHARUCO_PLACES:
                for tilt in CHARUCO_TILTS:
                    t += move
                    sched.append({"t_start": round(t, 2), "t_end": round(t + hold, 2),
                                  "say": f"{z:.2f} m, {place}, {tilt}: hold still"})  # fmt: skip
                    t += hold
        depth_every = int(_num(p, "depth_every", 5, 1, 20))  # ~2 FPS of the 10 FPS depth (a raw frame is 487 KB)
        cam = {"streams": ["left", "right", "depth"], "depth_every": depth_every, "video": False}
        p.update(distances_m=dists, hold_s=hold, move_s=move, depth_every=depth_every,
                 board=str(p.get("board", "poster_6x4_90")))  # fmt: skip
    else:  # depth_targets
        if "label" not in p:
            raise PlanError("label: what the target is (e.g. board_flat_0p45, box_step, board_upright)")
        dist = _num(p, "distance_m", 0.0, 0.0, 3.0)
        dur = _num(p, "duration_s", 5.0, 2.0, 60.0)
        sched.append({"t_start": 0.0, "t_end": dur, "say": f"{p['label']} at {dist:g} m (tape): keep it still"})
        cam = {"streams": ["left", "depth"], "depth_every": 1, "video": False}
        p.update(distance_m=dist, duration_s=dur)
    duration = max(e["t_end"] for e in sched) + 1.0
    return Plan(
        kind=kind,
        params=p,
        duration_s=duration,
        group=None,
        joints=(),
        base=None,
        setpoint=None,
        camera={**cam, "schedule": sched},
        needs_mode=None,
        operator=[f"t {e['t_start']:g}-{e['t_end']:g} s: {e['say']}" for e in sched],
        summary={"camera_streams": cam["streams"], "prompts": len(sched)},
    )


def lower_body(plan_: Plan) -> bool:
    return plan_.group in LOWER_GROUPS
