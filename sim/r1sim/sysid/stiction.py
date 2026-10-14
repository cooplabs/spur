"""Leg-joint stiction from a slow ramp per joint on the stand (A7 nominal, the user's decision U4) - PREPARED for HW1.

**Not fixed in sim.** The nominal model keeps issue #51's MuJoCo frictionloss (2.5 N·m hips / knees / waist, 1.5
ankles: a community identification on a MOVING, unloaded R1); every stall / fall safety test runs without it
(`a7_fall.SAFETY_LEGS_FRICTIONLOSS`, U4). This module is the measurement that decides it: the protocol, a synthetic
recording from the simulator with KNOWN friction (the test), the loader-independent fit, and the provenance snippet.

**Protocol (one joint at a time; the robot hanging on its stand in ON_STAND, every other joint held stiff).** The joint
under test gets a SOFT position servo (kp_test, a small kd) whose target ramps slowly: hold - up to +A - down to -A -
back, at a slow speed and again at a faster one. A soft servo turns the target ramp into a torque ramp: the joint
sticks while kp (q_cmd - q) grows through the friction band, then follows the ramp with a lag set by friction + gravity
+ damping. It is bounded (the target never leaves hold +- A; the torque never exceeds kp_test x the lag) and never limp
(kp_test > 0) - unlike a pure tau_ff ramp, which a stuck-then-free joint would turn into a run-away.

**Fit.** The applied torque is the board's law from the logged command and encoders, tau = kp (q_cmd - q) + kd
(dq_cmd - dq) + tau_ff (`tau_est` is the cross-check). On the steady, moving part of each sweep, quasi-static:
tau(q) = g(q) + sign(v) Fc + b v (inertia negligible at <= 0.3 rad/s), so the up and down sweeps of one speed at the
same q give h = (tau_up - tau_down) / 2 = Fc + b |v| (gravity cancels) and g(q) = (tau_up + tau_down) / 2; two speeds
separate Fc (MuJoCo `frictionloss`) from b (viscous: MuJoCo joint `damping`). The reversals give the static
breakaway: while stuck the torque swings from +(Fc + b v) to -Fs before the joint moves the other way.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from .. import joints as J
from .records import BODY_FIELDS as BODY_FIELDS_
from .records import BodyRecording, body_from_arrays

FloatArray = npt.NDArray[np.float64]

#: the protocol's defaults (rad, rad/s, s, N·m/rad, N·m·s/rad); A per joint from the hang pose, inside the soft limits
SPEEDS = (0.1, 0.3)
HOLD_S = 1.0
SETTLE_S = 1.5
KP_TEST = {"hip": 40.0, "knee": 40.0, "ankle": 25.0, "waist": 40.0}
#: kd of the soft servo: the hips / waist carry a whole leg (or the torso) - kd 1 leaves them under-damped at kp 40
#: (zeta ~0.1: the leg rings and the fit's residual doubles); 3 settles them. Knee / ankle: 1.
KD_TEST = {"hip": 3.0, "knee": 1.0, "ankle": 1.0, "waist": 3.0}
AMPLITUDE = {"hip_pitch": 0.25, "hip_roll": 0.15, "hip_yaw": 0.2, "knee": 0.25, "ankle_pitch": 0.2,
             "ankle_roll": 0.12, "waist_roll": 0.1, "waist_yaw": 0.2}  # fmt: skip
#: where the hanging robot's pelvis is held in the synthetic recordings (feet ~0.26 m off the floor)
HANG_Z = 1.0
RATE_HZ = 500.0  # r1rt's writer rate = r1log's telemetry rate
#: every joint the protocol covers (legs + waist, SDK order)
LEG_JOINTS_FOR_TEST: tuple[str, ...] = J.JOINTS[:14]


def _kind(name: str) -> str:
    for k in ("ankle_pitch", "ankle_roll", "hip_pitch", "hip_roll", "hip_yaw", "knee", "waist_roll", "waist_yaw"):
        if name.endswith(k):
            return k
    raise ValueError(f"{name}: not a leg / waist joint")


def _family(name: str) -> str:
    k = _kind(name)
    return "ankle" if k.startswith("ankle") else "waist" if k.startswith("waist") else k.split("_")[0]


def kp_test(name: str) -> float:
    # waist roll carries the torso ABOVE the joint (an inverted pendulum, m g h ~ 45 N m/rad): the soft servo must stay
    # well above that or the torso tips to the joint limit
    return 150.0 if _kind(name) == "waist_roll" else KP_TEST[_family(name)]


def kd_test(name: str) -> float:
    return 8.0 if _kind(name) == "waist_roll" else KD_TEST[_family(name)]


@dataclass(frozen=True)
class RampPlan:
    """The target profile of one joint: piecewise-linear knots (t, offset from the hold pose)."""

    joint: str
    kp: float
    kd: float
    amplitude: float
    speeds: tuple[float, ...]
    t: tuple[float, ...]
    offset: tuple[float, ...]

    @property
    def duration_s(self) -> float:
        return self.t[-1]

    def q_cmd(self, t: float | FloatArray, q_hold: float) -> FloatArray:
        return np.asarray(q_hold + np.interp(t, self.t, self.offset), dtype=np.float64)


def ramp_plan(
    joint: str,
    *,
    amplitude: float | None = None,
    speeds: Sequence[float] = SPEEDS,
    hold_s: float = HOLD_S,
    settle_s: float = SETTLE_S,
    kp: float | None = None,
    kd: float | None = None,
) -> RampPlan:
    """settle - for each speed: up to +A, hold, down to -A, hold, back to 0, hold (both directions at every q)."""
    a = AMPLITUDE[_kind(joint)] if amplitude is None else float(amplitude)
    ts, xs = [0.0, settle_s], [0.0, 0.0]
    for v in speeds:
        for target in (a, -a, 0.0):
            ts.append(ts[-1] + abs(target - xs[-1]) / v)
            xs.append(target)
            ts.append(ts[-1] + hold_s)
            xs.append(target)
    return RampPlan(joint, kp_test(joint) if kp is None else kp, kd_test(joint) if kd is None else kd, a,
                    tuple(float(v) for v in speeds), tuple(ts), tuple(xs))  # fmt: skip


# ------------------------------------------------------------------------------------------------ synthetic data
def hanging_world(*, seed: int | None = None) -> Any:
    """The real-profile robot hanging: the pelvis welded to the world at HANG_Z (the stand), feet off the floor.

    Session 9 (armature): the weld's `data` is set explicitly. MjSpec's `add_equality` defaults `data` to a JOINT
    equality's polycoef (0, 1, 0, ...), which a WELD reads as its ANCHOR: the pelvis hung from a point 1 m to its left,
    its rotation held only by the soft weld torque against ~300 N m of gravity moment - the "hanging" robot rolled ~5-10
    deg, dropped 0.18-0.26 m and stood on its right foot (contacts from 0.44 s); the rig's other joints swung (waist
    roll 0.53 rad) and, with the legs' identified armature 0.008, the tested joints could not follow their ramps.
    Anchor 0, relpose computed from qpos0 (pelvis upright at HANG_Z), torquescale 1 = MJCF's `<weld body1="pelvis"/>`.
    """
    from .. import world

    def hang(spec: mujoco.MjSpec, _info: Any) -> None:
        pelvis = spec.body("pelvis")
        pelvis.pos = [0.0, 0.0, HANG_Z]
        data = np.zeros(11)
        data[10] = 1.0  # torquescale; anchor (0, 0, 0); relpose all 0 = from qpos0
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="pelvis", objtype=mujoco.mjtObj.mjOBJ_BODY, data=data)

    return world.build_world("real", with_scene=False, visual=False, seed=seed, extra=hang)


@dataclass
class SyntheticTruth:
    frictionloss: dict[str, float]
    damping: dict[str, float]
    sensors: str
    seed: int


def synthetic_ramp(
    joint: str,
    *,
    frictionloss: float,
    damping: float,
    plan: RampPlan | None = None,
    sensors: bool = True,
    seed: int = 0,
    stribeck: float = 0.0,
    stribeck_vs: float = 0.15,
) -> tuple[BodyRecording, SyntheticTruth]:
    """Run the protocol on the simulated hanging robot with the joint's friction set to KNOWN values; returns what r1log
    would have recorded (the sensor model's readings, the commands) and the truth.

    `stribeck` > 0 (session 8) adds a friction that FALLS with speed, as HW1's joints showed (L knee: Fs 0.94 > h(0.1)
    0.64 > h(0.3) 0.56 N m): a hidden torque -stribeck exp(-(v / stribeck_vs)^2) tanh(v / 0.01) on the joint's TRUE
    velocity, sent through the command's tau_ff but NOT logged (the recording shows the soft servo's command only - the
    joint just behaves as a Stribeck joint). With frictionloss 0.6, stribeck 0.3: h(0) ~ 0.9, h(0.1) ~ 0.8, h(0.3) ~
    0.62 N m."""
    from .. import sensors as S
    from ..driver import DriverConfig
    from ..engine import LowCmd, LowState, Simulation

    plan = plan or ramp_plan(joint)
    w = hanging_world()
    m = w.model
    ix = w.index
    j = J.IDX[joint]
    v = int(ix.vadr[j])
    m.dof_frictionloss[v] = frictionloss
    m.dof_damping[v] = damping
    sm = S.SensorModel(S.SensorConfig.from_yaml() if sensors else S.SensorConfig.off(), seed=seed)
    sim = Simulation(m, ix, seed=seed, driver=DriverConfig.real(), sensors=sm)
    qpos = m.qpos0.copy()
    hold = J.POSE_STAND.copy()
    for i in range(J.N):
        if ix.present[i]:
            qpos[ix.qadr[i]] = hold[i]
    sim.reset(qpos)

    n = round(plan.duration_s * RATE_HZ) + 1
    rows: dict[str, list[FloatArray]] = {k: [] for k in ("q", "dq", "tau_est", "q_cmd", "dq_cmd", "kp", "kd", "tau_ff")}
    times: list[float] = []
    kp = J.KP_STIFF.copy()
    kd = J.KD_STIFF.copy()
    kp[j], kd[j] = plan.kp, plan.kd

    def ctl(st: LowState) -> LowCmd:
        q_cmd = hold.copy()
        q_cmd[j] = float(plan.q_cmd(st.t, hold[j]))
        cmd = LowCmd(q=q_cmd, kp=kp.copy(), kd=kd.copy())
        logged_ff = cmd.tau_ff.copy()
        if stribeck:
            vt = float(sim.d.qvel[v])
            cmd.tau_ff[j] = -stribeck * np.exp(-((vt / stribeck_vs) ** 2)) * np.tanh(vt / 0.01)
        times.append(st.t)
        rows["q"].append(st.q.copy())
        rows["dq"].append(st.dq.copy())
        rows["tau_est"].append(st.tau_est.copy())
        rows["q_cmd"].append(q_cmd)
        rows["dq_cmd"].append(cmd.dq.copy())
        rows["kp"].append(cmd.kp.copy())
        rows["kd"].append(cmd.kd.copy())
        rows["tau_ff"].append(logged_ff)
        return cmd

    sim.add_controller(ctl, 1.0 / RATE_HZ, "ramp")
    sim.run_for(n / RATE_HZ)
    rec = body_from_arrays(np.asarray(times), {k: np.asarray(v_) for k, v_ in rows.items()})
    truth = SyntheticTruth({joint: frictionloss}, {joint: damping}, "S2 nominal" if sensors else "off", seed)
    return rec, truth


# ------------------------------------------------------------------------------------------------ the fit
#: the steady-sample rules (session 8, fix #27 - the HW1 wrap-up analysis C). On the real joints the static friction is
#: well above the sliding friction (L knee: Fs 0.94 > h(0.1) 0.64 > h(0.3) 0.56 N m), so every reversal ends in a
#: release transient: the joint breaks free late and overshoots the commanded speed. Session 5's filter counted from the
#: SEGMENT start with |v - vc| < 0.5 vc and kept that transient (more of it at 0.3 rad/s, where the sweeps are short:
#: the hips' fast sweeps never reach a steady state) -> h(0.3) came out low and the two-speed fit returned NEGATIVE
#: damping on 12 of the 14 HW1 joints. Now: the window starts at the BREAKAWAY + settle_s, |v - vc| < VEL_GATE vc, and a
#: speed counts only with MIN_STEADY samples per direction whose median |v| is >= MIN_SPEED_RATIO x the quasi-static
#: speed vc / (1 + g'(q) / kp) (the joint following the ramp with the gravity stiffness g' of its own limb).
VEL_GATE = 0.2
MIN_STEADY = 150
#: below this many steady samples per direction a speed gives no h at all (a quartic gravity + sign fit on fewer is
#: noise: the right ankle roll's 21 / 35 samples at 0.3 rad/s read 5.2 N m)
MIN_CURVE = 50
MIN_SPEED_RATIO = 0.9
#: the model's joint damping (r1.xml `<default><joint damping="0.05">`): kept when the data cannot resolve b (one speed,
#: or a friction that FALLS with speed - a Stribeck curve, which a viscous b cannot describe)
DAMPING_PRIOR = 0.05


@dataclass
class SpeedFit:
    """The friction at one ramp speed: h = (tau_up - tau_down) / 2 from tau = P(q) + h sign(v) (gravity cancels)."""

    speed: float  # the commanded |v|, rad/s
    h: float  # N·m (= Fc + b |v| for a Coulomb + viscous joint)
    h_se: float
    n_up: int  # steady samples moving up / down
    n_down: int
    v_ratio: float  # median |v| of the steady samples / the quasi-static speed vc / (1 + g'(q) / kp)
    valid: bool  # passes MIN_STEADY / MIN_SPEED_RATIO


@dataclass
class StictionFit:
    joint: str
    kp: float
    frictionloss: float  # Fc, N·m (MuJoCo frictionloss)
    frictionloss_se: float
    damping: float  # b, N·m·s/rad (MuJoCo joint damping); DAMPING_PRIOR when damping_source == "prior"
    damping_se: float
    static: float  # Fs from the reversals (median), N·m; nan if none
    n_reversals: int
    speeds_measured: tuple[float, ...]
    half_width: dict[float, float]  # per speed: median (tau_up - tau_down) / 2 (the binned cross-check)
    gravity_q: list[float] = field(default_factory=list)
    gravity_tau: list[float] = field(default_factory=list)
    residual_rms: float = float("nan")  # of the quasi-static model on the moving samples
    torque_source: str = "pd"
    notes: list[str] = field(default_factory=list)
    #: session 8: the friction curve - per commanded speed the steady h and whether that speed passed the rules
    curve: dict[float, SpeedFit] = field(default_factory=dict)
    #: "fit" (h rises with speed: Fc and b from one least-squares fit over the valid speeds) or "prior" (b =
    #: DAMPING_PRIOR, Fc = h at the slowest valid speed - b |v|)
    damping_source: str = "fit"
    #: False when no speed passed the steady-sample rules (Fc then comes from the speed with the most samples: a
    #: weak estimate, said in the notes)
    reliable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def applied_torque(rec: BodyRecording, j: int, source: str = "pd") -> FloatArray:
    """The joint's applied torque: the board's PD law from the logged command and encoders, or the motor's tau_est."""
    if source == "tau_est":
        return rec.joint("tau_est", j)
    if source != "pd":
        raise ValueError(f"source must be 'pd' or 'tau_est', not {source!r}")
    return np.asarray(
        rec.joint("kp", j) * (rec.joint("q_cmd", j) - rec.joint("q", j))
        + rec.joint("kd", j) * (rec.joint("dq_cmd", j) - rec.joint("dq", j))
        + rec.joint("tau_ff", j),
        dtype=np.float64,
    )


def _velocity(t: FloatArray, q: FloatArray, half_window_s: float) -> FloatArray:
    """Central difference over +-half_window (the encoders' dq is too noisy at 0.1 rad/s: 0.02 rad/s per sample)."""
    dt = float(np.median(np.diff(t)))
    w = max(1, round(half_window_s / dt))
    v = np.full_like(q, np.nan)
    v[w:-w] = (q[2 * w :] - q[: -2 * w]) / (t[2 * w :] - t[: -2 * w])
    return v


def _segments(t: FloatArray, q_cmd: FloatArray) -> list[tuple[int, int, float]]:
    """Runs of constant non-zero commanded velocity: (start, end exclusive, commanded velocity)."""
    vc = np.gradient(q_cmd, t)
    vr = np.round(vc, 3)
    out: list[tuple[int, int, float]] = []
    i = 0
    n = len(t)
    while i < n:
        k = i
        while k + 1 < n and abs(vr[k + 1] - vr[i]) < 1e-3:
            k += 1
        if abs(vr[i]) > 1e-3 and k - i > 10:
            out.append((i, k + 1, float(np.median(vc[i : k + 1]))))
        i = k + 1
    return out


def _breakaway(v: FloatArray, a: int, b: int, vc: float) -> int | None:
    """The first sample of segment [a, b) moving at > 0.5 |vc| in the commanded direction (the joint broke free)."""
    seg = v[a:b]
    mv = np.flatnonzero(np.isfinite(seg) & (np.sign(vc) * seg > 0.5 * abs(vc)))
    return None if mv.size == 0 else a + int(mv[0])


def _ls(
    q: FloatArray, tau: FloatArray, v: FloatArray, *, degree: int, viscous: bool
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    """tau = P(q) + c sign(v) [+ b v] by least squares; returns (coef, design matrix, residual, qc, qw). P is a
    polynomial in q (gravity: a binned up / down difference is biased where g(q) is steep - the hip roll lifting the
    whole leg)."""
    qc, qw = 0.5 * (q.max() + q.min()), max(1e-6, 0.5 * (q.max() - q.min()))
    x = (q - qc) / qw
    cols = [x**k for k in range(degree + 1)] + [np.sign(v)] + ([v] if viscous else [])
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
    return coef, A, tau - A @ coef, float(qc), float(qw)


def _cov(A: FloatArray, res: FloatArray) -> FloatArray:
    """Parameter covariance with the samples decimated to ~25 Hz (the encoder / servo noise is correlated over tens
    of ms, so 500 Hz samples are not independent)."""
    dec = max(1, round(RATE_HZ / 25.0))
    Ad, rd = A[::dec], res[::dec]
    s2 = float(rd @ rd) / max(1, len(rd) - A.shape[1])
    return np.asarray(s2 * np.linalg.pinv(Ad.T @ Ad), dtype=np.float64)


def fit_joint(
    rec: BodyRecording,
    joint: str | int,
    *,
    source: str = "pd",
    settle_s: float = 0.25,
    vel_gate: float = VEL_GATE,
    min_steady: int = MIN_STEADY,
    min_speed_ratio: float = MIN_SPEED_RATIO,
    damping_prior: float = DAMPING_PRIOR,
    bin_width: float = 0.01,
    min_per_bin: int = 5,
    poly_degree: int = 4,
) -> StictionFit:
    """Fc (frictionloss), b (damping), Fs (static), the friction curve h(v) and g(q) of one joint from a ramp
    recording (see the module doc and the steady-sample rules above). b is estimated only when the friction RISES
    with speed over >= 2 valid speeds; otherwise b = `damping_prior` and Fc = h(slowest valid speed) - b |v| (never a
    negative damping: a real friction that falls with speed is a Stribeck curve, reported in `curve`)."""
    j = J.IDX[joint] if isinstance(joint, str) else int(joint)
    name = J.JOINTS[j]
    t = rec.t
    q = rec.joint("q", j)
    q_cmd = rec.joint("q_cmd", j)
    tau = applied_torque(rec, j, source)
    v = _velocity(t, q, 0.05)
    # the segments from the PLAN's reference when the recording carries it (`ramp_recording`: r1d streams the
    # setpoint at its ~100 Hz tick, so the logged 500 Hz q_cmd is a staircase without constant-gradient runs: R6);
    # the torque above stays the logged command's (what the motors were really sent)
    segs = _segments(t, rec.joint("q_cmd_ref", j) if "q_cmd_ref" in rec.fields else q_cmd)
    if not segs:
        raise ValueError(f"{name}: no ramp segments in the recording (q_cmd constant)")
    speeds = sorted({round(abs(s[2]), 3) for s in segs})
    notes: list[str] = []

    def steady_masks(gate: float, settle: float) -> dict[float, npt.NDArray[np.bool_]]:
        m = {sp: np.zeros(len(t), dtype=bool) for sp in speeds}
        for a, b_, vc in segs:
            k0 = _breakaway(v, a, b_, vc)
            if k0 is None:
                continue
            idx = np.arange(k0, b_)
            ok = (t[idx] - t[k0] >= settle) & np.isfinite(v[idx]) & (np.abs(v[idx] - vc) < gate * abs(vc))
            m[round(abs(vc), 3)][idx[ok]] = True
        return m

    def counts(m: npt.NDArray[np.bool_]) -> tuple[int, int]:
        sel = np.flatnonzero(m)
        return int((v[sel] > 0).sum()), int((v[sel] < 0).sum())

    for a, b_, vc in segs:
        if _breakaway(v, a, b_, vc) is None:
            notes.append(f"segment at {t[a]:.2f} s ({vc:+.3f} rad/s): never moved at half the commanded speed")
    steady = steady_masks(vel_gate, settle_s)
    reliable = True
    if all(min(counts(m)) < MIN_CURVE for m in steady.values()):
        # no speed has a usable steady run at all (the HW1 ankle rolls: kp 25 against ~1.4 N m leaves a 0.06 rad lag
        # of a 0.12 rad sweep - the joint moves mostly in release transients): session 5's looser gate, from the
        # breakaway, marked unreliable - a bound, not an identification
        steady = steady_masks(0.5, 0.0)
        reliable = False
        notes.append("no steady run at any speed under the rules: the loose gate (|v - vc| < 0.5 vc from the "
                     "breakaway) - an UNRELIABLE estimate")  # fmt: skip

    curve: dict[float, SpeedFit] = {}
    half: dict[float, float] = {}
    for sp in speeds:
        sel = np.flatnonzero(steady[sp])
        n_up, n_dn = counts(steady[sp])
        if min(n_up, n_dn) < MIN_CURVE:
            notes.append(f"speed {sp}: {n_up} / {n_dn} steady samples up / down - no friction at this speed")
            continue
        coef, A, res, qc_s, qw_s = _ls(q[sel], tau[sel], v[sel], degree=poly_degree, viscous=False)
        # the speed a joint following the ramp quasi-statically really has: kp (vc - v) = g'(q) v, so v = vc / (1 +
        # g'(q) / kp) with g' the slope of the fitted holding torque - the hanging knee's gravity stiffness alone makes
        # a perfectly steady sweep run at ~0.89 vc (the plain |v| / vc dropped every synthetic sweep)
        x = (q[sel] - qc_s) / qw_s
        g_slope = sum(k * coef[k] * x ** (k - 1) for k in range(1, poly_degree + 1)) / qw_s
        denom = 1.0 + g_slope / np.maximum(rec.joint("kp", j)[sel], 1e-6)
        ok_d = denom > 0.2
        ratio = float(np.median(np.abs(v[sel][ok_d]) * denom[ok_d])) / sp if ok_d.any() else 0.0
        passes = reliable and n_up >= min_steady and n_dn >= min_steady and ratio >= min_speed_ratio
        curve[sp] = SpeedFit(sp, float(coef[-1]), float(np.sqrt(_cov(A, res)[-1, -1])), n_up, n_dn, ratio, passes)
        if not passes:
            notes.append(
                f"speed {sp}: dropped ({n_up} / {n_dn} steady samples, median |v| / vc {ratio:.2f}; the rule: >= "
                f"{min_steady} per direction and >= {min_speed_ratio})"
            )
        # the binned cross-check (median up - down at the same q)
        qu, tu = q[sel][v[sel] > 0], tau[sel][v[sel] > 0]
        qd, td = q[sel][v[sel] < 0], tau[sel][v[sel] < 0]
        lo, hi = max(qu.min(), qd.min()), min(qu.max(), qd.max())
        if hi - lo < 3 * bin_width:
            notes.append(f"speed {sp}: up / down sweeps overlap only {hi - lo:.3f} rad")
            continue
        hh: list[float] = []
        for e0, e1 in itertools.pairwise(np.arange(lo, hi + 1e-12, bin_width)):
            mu, md = (qu >= e0) & (qu < e1), (qd >= e0) & (qd < e1)
            if mu.sum() >= min_per_bin and md.sum() >= min_per_bin:
                hh.append(0.5 * (float(np.median(tu[mu])) - float(np.median(td[md]))))
        half[sp] = float(np.median(hh)) if hh else float("nan")
    if not curve:
        raise ValueError(f"{name}: too few steady moving samples at every speed ({', '.join(notes)})")

    valid = [sp for sp in sorted(curve) if curve[sp].valid]
    reliable = reliable and bool(valid)
    if not valid:  # nothing passes the rules: the speed with the most samples, said plainly
        best = max(curve, key=lambda sp: min(curve[sp].n_up, curve[sp].n_down))
        valid = [best]
        notes.append(f"no speed passes the steady-sample rules: Fc from speed {best} (weak)")
    rising = len(valid) >= 2 and curve[valid[-1]].h > curve[valid[0]].h
    sel = np.flatnonzero(np.logical_or.reduce([steady[sp] for sp in valid]))
    fc = b = float("nan")
    fc_se = b_se = float("nan")
    damping_source = "prior"
    if rising:  # Coulomb + viscous describes the curve: one fit over every valid steady sample
        coef, A, res, qc, qw = _ls(q[sel], tau[sel], v[sel], degree=poly_degree, viscous=True)
        cov = _cov(A, res)
        if coef[-1] >= 0.0:
            fc, b = float(coef[-2]), float(coef[-1])
            fc_se, b_se = float(np.sqrt(cov[-2, -2])), float(np.sqrt(cov[-1, -1]))
            damping_source = "fit"
        else:
            notes.append(f"h rises with speed but the joint fit's b = {coef[-1]:.3f} < 0: prior kept")
    if damping_source == "prior":
        slow = curve[valid[0]]
        b, b_se = float(damping_prior), float("nan")
        v_slow = float(np.median(np.abs(v[steady[valid[0]]])))
        fc, fc_se = slow.h - b * v_slow, slow.h_se
        if len(valid) >= 2:
            notes.append(
                f"friction FALLS with speed (h {', '.join(f'{curve[s].h:.3f} at {s}' for s in valid)}: Stribeck): "
                f"b not identifiable, prior {damping_prior} kept, Fc = h({valid[0]}) - b |v|"
            )
        else:
            notes.append(f"one valid speed ({valid[0]}): b = prior {damping_prior}, Fc = h - b |v|")
        coef, A, res, qc, qw = _ls(q[sel], tau[sel], v[sel], degree=poly_degree, viscous=False)
    gq = np.linspace(q[sel].min(), q[sel].max(), 25)
    g_tau = np.column_stack([((gq - qc) / qw) ** k for k in range(poly_degree + 1)]) @ coef[: poly_degree + 1]

    # Static breakaway at the reversals between two segments of the same speed, opposite direction.
    any_steady = np.logical_or.reduce(list(steady.values()))
    statics: list[float] = []
    for (a0, b0, v0), (a1, _b1, v1) in itertools.pairwise(segs):
        if np.sign(v0) == np.sign(v1) or round(abs(v0), 3) != round(abs(v1), 3) or round(abs(v0), 3) not in curve:
            continue
        before = np.arange(max(a0, b0 - int(0.2 * RATE_HZ)), b0)
        before = before[any_steady[before]]
        if before.size < 5:
            continue
        tau_before = float(np.median(tau[before]))
        k = _breakaway(v, a1, len(t), v1)
        if k is None:
            continue
        tau_break = float(np.median(tau[max(a1, k - 5) : k + 1]))
        # tau_before = g + sign(v0) h(|v0|); breakaway at g - sign(v0) Fs  ->  Fs = |delta| - h(|v0|)
        sp0 = round(abs(v0), 3)
        h0 = curve[sp0].h
        statics.append(abs(tau_before - tau_break) - h0)

    return StictionFit(
        joint=name,
        kp=float(np.median(rec.joint("kp", j))),
        frictionloss=fc,
        frictionloss_se=fc_se,
        damping=b,
        damping_se=b_se,
        static=float(np.median(statics)) if statics else float("nan"),
        n_reversals=len(statics),
        speeds_measured=tuple(float(np.median(np.abs(v[steady[sp]]))) for sp in valid),
        half_width=half,
        gravity_q=gq.tolist(),
        gravity_tau=np.asarray(g_tau).tolist(),
        residual_rms=float(np.sqrt(np.mean(res**2))) if res.size else float("nan"),
        torque_source=source,
        notes=notes,
        curve=curve,
        damping_source=damping_source,
        reliable=reliable,
    )


def plan_reference(manifest: Mapping[str, Any], t_abs_ns: npt.ArrayLike, q_cmd: npt.ArrayLike) -> FloatArray:
    """The commanded q (n, 26) the recording's PLAN asked for at each sample's absolute CLOCK_MONOTONIC time: the
    manifest's plan (`r1 record`'s kind + params + the group's base) rebuilt with r1mc3.record.plans and sampled at
    t - t_start_mono_ns; outside the plan's window and on the other joints the logged q_cmd."""
    from r1mc3.record.plans import plan as make_plan

    p = manifest["plan"]
    cols = [int(c) for c in manifest["group_joints"]]
    q26 = np.zeros(26)
    q26[cols] = np.asarray(manifest["base"], dtype=np.float64)
    pl = make_plan(str(p["kind"]), dict(p["params"]), q26)
    t_rel = (np.asarray(t_abs_ns, dtype=np.int64) - int(manifest["t_start_mono_ns"])) * 1e-9
    ref = np.array(q_cmd, dtype=np.float64, copy=True)
    for i in np.flatnonzero((t_rel >= 0.0) & (t_rel <= pl.duration_s)):
        ref[i, cols] = pl.at(float(t_rel[i]))[0]
    return ref


def ramp_recording(
    manifest: Mapping[str, Any] | str | Path, t_abs_ns: npt.ArrayLike, fields: Mapping[str, npt.ArrayLike]
) -> BodyRecording:
    """A ramp recording AS RECORDED through r1d (the manifest of `r1 record run ramp ...` + r1log's /r1rt/telemetry at
    their absolute t_wake_ns) cut to the plan's window, with the plan's reference as `q_cmd_ref` for the segments."""
    import json

    man = json.loads(Path(manifest).read_text()) if isinstance(manifest, str | Path) else dict(manifest)
    t = np.asarray(t_abs_ns, dtype=np.int64)
    t0, t1 = int(man["t_start_mono_ns"]), int(man.get("t_end_mono_ns") or t[-1])
    sel = (t >= t0) & (t <= t1)
    f = {k: np.asarray(fields[k], dtype=np.float64)[sel] for k in BODY_FIELDS_}
    rec = body_from_arrays(t[sel] * 1e-9, f)
    rec.fields["q_cmd_ref"] = plan_reference(man, t[sel], f["q_cmd"])
    return rec


def load_ramp(manifest_path: str | Path, mcap_paths: Sequence[str | Path] | None = None) -> BodyRecording:
    """`ramp_recording` from the files themselves: the manifest + its r1log MCAP files (the manifest's
    `r1log_files`, or `mcap_paths`)."""
    import json

    from r1mc3.log import reader

    from .records import BODY_TOPIC

    man = json.loads(Path(manifest_path).read_text())
    paths = list(mcap_paths) if mcap_paths is not None else list(man.get("r1log_files") or [])
    ts: list[FloatArray] = []
    cols: dict[str, list[FloatArray]] = {k: [] for k in BODY_FIELDS_}
    for path in paths:
        rec = reader.read(Path(path), log_time_order=True).topic(BODY_TOPIC)
        if rec.size == 0:
            continue
        ts.append(rec["t_wake_ns"].astype(np.int64))
        for k in BODY_FIELDS_:
            cols[k].append(np.asarray(rec[k], dtype=np.float64))
    if not ts:
        raise ValueError(f"{manifest_path}: no {BODY_TOPIC} records in {paths}")
    t = np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    return ramp_recording(man, t[order], {k: np.concatenate(v)[order] for k, v in cols.items()})


def fit_joints(rec: BodyRecording, joints: Sequence[str], **kw: Any) -> dict[str, StictionFit]:
    """Fit each joint over the part of the recording where it was the one ramping (its q_cmd moved)."""
    out: dict[str, StictionFit] = {}
    for name in joints:
        j = J.IDX[name]
        qc = rec.joint("q_cmd_ref" if "q_cmd_ref" in rec.fields else "q_cmd", j)
        moving = np.flatnonzero(np.abs(np.gradient(qc, rec.t)) > 1e-4)
        if moving.size == 0:
            continue
        a, b = max(0, int(moving[0]) - 50), min(len(rec.t), int(moving[-1]) + 50)
        sub = BodyRecording(rec.t[a:b] - rec.t[a], {k: v[a:b] for k, v in rec.fields.items()})
        out[name] = fit_joint(sub, j, **kw)
    return out


def provenance_yaml(fits: Mapping[str, StictionFit], *, source: str, date: str) -> str:
    """A models/provenance.yaml snippet for the fitted values (the model change itself is a reviewed edit). Refuses a
    negative damping (session 8, fix #27: session 5's fit wrote b < 0 for 12 of the 14 HW1 joints - not a physical
    joint damping, a symptom of the fit) and a non-finite Fc."""
    for name, f in fits.items():
        if not (f.damping >= 0.0):
            raise ValueError(f"{name}: damping {f.damping} - a negative / NaN joint damping is not a physical value")
        if not np.isfinite(f.frictionloss) or f.frictionloss < 0.0:
            raise ValueError(f"{name}: frictionloss {f.frictionloss}")
    lines = ["  leg_joint_stiction_hw1:", "    unit: {frictionloss: N m, damping: N m s/rad}", "    status: measured",
             f'    source: "{source}: slow ramp per joint on the stand (sim/r1sim/sysid/stiction.py)"',
             f"    date: {date}", "    profiles: [real]", "    value:"]  # fmt: skip
    for name, f in fits.items():
        curve = ", ".join(f"{sp}: {c.h:.3f}" for sp, c in sorted(f.curve.items()))
        lines.append(
            f"      {name}: {{frictionloss: {f.frictionloss:.3f}, frictionloss_se: {f.frictionloss_se:.3f}, "
            f"damping: {f.damping:.3f}, damping_source: {f.damping_source}, static: {f.static:.3f}, "
            f"h_per_speed: {{{curve}}}, residual_rms: {f.residual_rms:.3f}}}"
        )
    return "\n".join(lines) + "\n"
