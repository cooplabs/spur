"""B1 / B2 (DESIGN §13.2): replay the HW1 step and chirp recordings in the HANGING simulator and compare its joints
with the real ones.

The real robot hung on its stand (the harness's top bar in the cradle, the lower loop against the pole: the bridge's
`StandSpec`, attach="cradle") and r1rt commanded every joint at 500 Hz. The replay builds the same hanging robot
(real profile, lockstep), poses it at the MEASURED joint angles of the window's first sample, lets it settle for
`settle_s` under the command r1rt was sending then, and from there feeds the simulated board the LOGGED command of
every tick (q_cmd, dq_cmd, kp, kd, tau_ff for all 26 joints - exactly what the motors were sent). The sim's joint
state is sampled at the same ticks, so the sim and the real traces line up sample for sample (the real `q` of a tick =
the lowstate r1rt held then, ~0.5 ms old: HW1 T5).

What is compared (per window, the moving joint):

* **normalised RMS** = RMS(sim - real) / RMS(real - mean(real)) over the plan's window, after removing each trace's
  own mean (the steady offsets - gravity / friction at the stand pose - are reported apart as `offset_mm_rad`);
  target <= 10 %;
* **delay to 10 %** per step event: from the command's first change to the joint having moved 10 % of the step (the
  HW1 analysis' definition; the steps are rate limited by the executor's 2 rad/s cap, ~25 ms for 0.05 rad); B2's
  target: the sim's median within +-1 ms of the real one;
* **gain / phase lag** of q against the logged q_cmd at chosen frequencies of a chirp (a lock-in on the plan's chirp
  phase over +-0.25 Hz of instantaneous frequency); target: the sim's lag within +-5 deg of the real one at 2 / 5 Hz
  (the 5 Hz window ends where the 0.5 s fade-out starts: its centre is reported; 10 Hz was not excited at HW1);
* **torque**: the real tau_est (N m, + = +q; HW1 confirmed it equals the PD law to corr 0.999) vs the torque the
  simulated driver applied: the ratio of their RMS about the mean (target 1 +- 0.15) and their correlation.

The stand: the bridge's StandSpec geometry (two photos), but its pole COMPLIANT (`pole_k` / `pole_c`, a slide joint
under the harness's lower loop): with the bridge's rigid pole the hanging robot cannot rock, and at HW1 it did (every
hip step rocked it at ~2-3 Hz, +-0.11-0.14 rad/s pitch rate) - the rigid stand makes the hips look 3x too heavy.

Not modelled here (said plainly): the hands' state at HW1 (not in r1log: they hold open in the sim), the stand's ROLL
compliance (the cradle's pad: the waist roll rows), the board's PD rate (the sim's every 1 ms physics step).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cache
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from .. import dex3, world
from .. import joints as J
from ..driver import DriverConfig
from ..engine import HandCmd, LowCmd, LowState, Simulation
from .data import Window

FloatArray = npt.NDArray[np.float64]

SETTLE_S = 3.0
HW1_POLE_K = 35000.0  # N/m: the stand's effective compliance at the harness's lower loop (fitted, TIER_B.md)
HW1_POLE_C = 150.0  # N s/m
POLE_BODY = "r1sim_tierb_pole"
POLE_JOINT = "r1sim_tierb_pole_x"
POLE_GEOM = "r1sim_tierb_pole_geom"
TICK_S = 0.002
#: The leg / waist friction (N m) from the HW1 wrap-up re-fit (the HW1 wrap-up analysis, C;
#: s4_fits_v2.json) - the contract's provisional values until the fit agent's refit lands (fit_FRICTION_READY).
PROVISIONAL_FRICTION: dict[str, float] = {
    "hip_pitch": 0.6, "hip_roll": 0.4, "hip_yaw": 0.6, "knee": 0.55, "ankle_pitch": 2.9, "ankle_roll": 1.4,
    "waist_roll": 0.5, "waist_yaw": 0.4,
}  # fmt: skip


@dataclass(frozen=True)
class ReplayConfig:
    """The simulated robot of a replay: the command / state delay of the board path, and per-joint friction
    overrides (joint name without side, e.g. "knee" -> both knees; or a full SDK name) on top of the real profile."""

    cmd_delay_s: float = 0.0
    state_delay_s: float = 0.0
    friction: Mapping[str, float] = field(default_factory=dict)
    damping: Mapping[str, float] = field(default_factory=dict)
    armature: Mapping[str, float] = field(default_factory=dict)
    #: the stand's compliance at the harness's lower loop (N/m, N s/m of the pole's slide along x); None = the
    #: bridge's rigid pole. HW1: at every hip step the real robot rocks on its stand (+-0.11-0.14 rad/s pitch rate)
    #: and the other hip moves ~7-10 mrad; the rigid pole gives 0.007 rad/s / 0.5 mrad. Fitted on the hip steps and the
    #: knee chirps (docs/sim/TIER_B.md): 35 000 / 150.
    pole_k: float | None = HW1_POLE_K
    pole_c: float = HW1_POLE_C
    settle_s: float = SETTLE_S
    #: session 9 (B1's open rows, an EVALUATION option - the model has no such term): a friction that FALLS with speed
    #: on top of MuJoCo's constant frictionloss, per joint key -> (extra static torque N m, Stribeck speed rad/s): the
    #: hidden torque -s exp(-(v / vs)^2) tanh(v / 0.005) on the joint's TRUE velocity, added to the command's tau_ff at
    #: the replay tick (as `sysid.stiction.synthetic_ramp`'s Stribeck joint). HW1: Fs > h(0.1) > h(0.3) everywhere.
    stribeck: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    def driver(self) -> DriverConfig:
        return DriverConfig(kind="board", cmd_delay_s=self.cmd_delay_s, state_delay_s=self.state_delay_s)

    def label(self) -> str:
        fr = "model" if not self.friction else "custom"
        st = "rigid pole" if self.pole_k is None else f"compliant pole {self.pole_k:g} N/m"
        return f"delay {1e3 * self.cmd_delay_s:g} ms, state {1e3 * self.state_delay_s:g} ms, friction {fr}, {st}"


def _matches(key: str, name: str) -> bool:
    return name == key or name.endswith("_" + key) or name == key


def _apply_overrides(m: Any, values: Mapping[str, float], attr: str) -> dict[str, float]:
    arr = getattr(m, attr)
    done: dict[str, float] = {}
    for name in J.JOINTS:
        for key, v in values.items():
            if _matches(key, name):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
                if jid >= 0:
                    arr[int(m.jnt_dofadr[jid])] = float(v)
                    done[name] = float(v)
    return done


@cache
def _base_world() -> world.World:
    """The bridge's stand (StandSpec(): the cradle + the rigid pole) + a COMPLIANT copy of the pole: a child body of
    the stand on a slide joint along x, its stiffness / damping set per replay (`pole_k`, `pole_c`); each replay
    switches one of the two pole geoms' contact off."""
    from ..dds_bridge import STAND_BODY, STAND_COLLIDE_BIT, StandSpec, stand_extra

    stand = StandSpec()
    base = stand_extra(stand)

    def extra(spec: mujoco.MjSpec, info: Any) -> None:
        base(spec, info)
        pb = spec.body(STAND_BODY).add_body()
        pb.name = POLE_BODY
        pb.mass = 1.0
        pb.inertia = [0.01, 0.01, 0.01]
        jt = pb.add_joint()
        jt.name = POLE_JOINT
        jt.type = mujoco.mjtJoint.mjJNT_SLIDE
        jt.axis = [1.0, 0.0, 0.0]
        g = pb.add_geom()
        g.name = POLE_GEOM
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.fromto = [0.0, 0.0, 0.30, 0.0, 0.0, stand.bar_z_m - 0.04]  # the bridge's pole exactly
        g.size = [stand.pole_radius_m, 0.0, 0.0]
        # on at compile: MuJoCo filters body pairs by the compiled body_contype / body_conaffinity (a geom switched
        # on only after the compile is never collided); a replay without the compliant pole switches the GEOM off
        g.contype, g.conaffinity = 0, STAND_COLLIDE_BIT
        g.density = 0.0

    return world.build_world("real", with_scene=False, visual=False, extra=extra)


@dataclass
class ReplayResult:
    """The sim's traces at the window's ticks (n, 26) + the hang's numbers."""

    t: FloatArray
    q: FloatArray
    dq: FloatArray
    tau: FloatArray
    gyro: FloatArray
    quat: FloatArray
    hang: dict[str, float]


def replay(win: Window, cfg: ReplayConfig, *, t_from: float | None = None, t_to: float | None = None) -> ReplayResult:
    """Replay the logged commands of `win` between t_from and t_to (s, the window's clock; default: the whole
    window) on the hanging sim robot (module docstring)."""
    from ..dds_bridge import STAND_BODY, StandSpec, foot_geoms, hand_hold, lowest_point_z, place_stand

    stand = StandSpec()
    w0 = _base_world()
    m = _copy_model(w0.model)  # the overrides never change the cached world
    _apply_overrides(m, cfg.friction, "dof_frictionloss")
    _apply_overrides(m, cfg.damping, "dof_damping")
    _apply_overrides(m, cfg.armature, "dof_armature")
    jp = m.joint(POLE_JOINT).id
    if cfg.pole_k is not None:  # the compliant pole carries the loop; the bridge's rigid one is switched off
        m.geom_conaffinity[m.geom("r1sim_stand_pole").id] = 0
        m.jnt_stiffness[jp] = cfg.pole_k
        m.dof_damping[int(m.jnt_dofadr[jp])] = cfg.pole_c
    else:
        m.geom_conaffinity[m.geom(POLE_GEOM).id] = 0
        m.jnt_stiffness[jp] = 1e6
        m.dof_damping[int(m.jnt_dofadr[jp])] = 1e3
    i0 = 0 if t_from is None else int(np.searchsorted(win.t, t_from))
    i1 = len(win.t) if t_to is None else int(np.searchsorted(win.t, t_to))
    f = win.fields
    q0 = np.asarray(f["q"][i0], dtype=np.float64)
    sim = Simulation(m, w0.index, driver=cfg.driver())
    m = sim.m  # the engine works on its own copy of the model: the stand's anchors go there (as DdsBridge._hang)
    free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    b = int(m.jnt_qposadr[free[0]])
    qpos = w0.initial_qpos(q0)  # the measured joint angles exactly (no CoM centring: the robot hangs)
    qpos[b + 2] += stand.raise0_m
    place_stand(m, stand, qpos)
    sim.reset(qpos)
    hands = {s: hand_hold(dex3.preset("open", s)) for s in dex3.SIDES} if w0.index.hand_qadr else {}

    def cmd_at(i: int) -> LowCmd:
        c = LowCmd(
            q=np.asarray(f["q_cmd"][i], dtype=np.float64).copy(),
            dq=np.asarray(f["dq_cmd"][i], dtype=np.float64).copy(),
            kp=np.asarray(f["kp"][i], dtype=np.float64).copy(),
            kd=np.asarray(f["kd"][i], dtype=np.float64).copy(),
            tau_ff=np.asarray(f["tau_ff"][i], dtype=np.float64).copy(),
        )
        c.hands = {s: HandCmd(q=h.q.copy(), kp=h.kp.copy(), kd=h.kd.copy()) for s, h in hands.items()}
        return c

    # the bridge's hang (DdsBridge._hang): settle hanging, raise robot + stand together so the lowest foot point is
    # stand.clearance_m above the floor, restart from the settled pose at rest, settle again briefly
    sim.write_cmd(cmd_at(i0))
    sim.run_for(cfg.settle_s)
    kin = mujoco.MjData(m)
    kin.qpos[:] = sim.d.qpos
    mujoco.mj_kinematics(m, kin)
    qs = sim.d.qpos.copy()
    shift = stand.clearance_m - lowest_point_z(m, kin, foot_geoms(m))
    qs[b + 2] += shift
    m.body_pos[m.body(STAND_BODY).id, 2] += shift
    sim.reset(qs)
    if cfg.pole_k is not None:  # pre-load the pole's spring: the hang settles with the pole where the rigid one was
        qa = int(m.jnt_qposadr[jp])
        for _ in range(3):
            sim.reset(qs)
            sim.write_cmd(cmd_at(i0))
            sim.run_for(1.5)
            m.qpos_spring[qa] -= sim.d.qpos[qa]
        sim.reset(qs)
    sim.write_cmd(cmd_at(i0))
    sim.run_for(min(1.0, cfg.settle_s))
    kin.qpos[:] = sim.d.qpos
    mujoco.mj_kinematics(m, kin)
    hang = dict(
        foot_z_min_m=round(lowest_point_z(m, kin, foot_geoms(m)), 4),
        settle_err_max_rad=round(float(np.max(np.abs(sim.read_state().q - q0))), 4),
    )
    n = i1 - i0
    out_q = np.zeros((n, J.N))
    out_dq = np.zeros((n, J.N))
    out_tau = np.zeros((n, J.N))
    out_gyro = np.zeros((n, 3))
    out_quat = np.zeros((n, 4))
    k = [0]
    strib = {J.IDX[n]: v for key, v in cfg.stribeck.items() for n in J.JOINTS if _matches(key, n)}

    def ctl(st: LowState) -> LowCmd | None:
        i = k[0]
        if i >= n:
            return None
        out_q[i] = st.q
        out_dq[i] = st.dq
        out_tau[i] = st.tau_est
        out_gyro[i] = st.gyro
        out_quat[i] = st.quat
        k[0] += 1
        c = cmd_at(i0 + i)
        for jj, (fs, vs) in strib.items():
            v = float(sim.d.qvel[int(w0.index.vadr[jj])])
            c.tau_ff[jj] -= fs * np.exp(-((v / vs) ** 2)) * np.tanh(v / 0.005)
        return c

    sim.add_controller(ctl, TICK_S, "replay")
    sim.run_for(n * TICK_S)
    return ReplayResult(np.asarray(win.t[i0:i1], dtype=np.float64), out_q, out_dq, out_tau, out_gyro, out_quat, hang)


def _copy_model(m: Any) -> Any:
    import copy

    return copy.deepcopy(m)


# ------------------------------------------------------------------------------------------------ metrics
def nrms(sim: FloatArray, real: FloatArray) -> float:
    """RMS(sim - real) / RMS(real - mean(real)), each trace's mean removed first."""
    s = sim - np.mean(sim)
    r = real - np.mean(real)
    den = float(np.sqrt(np.mean(r**2)))
    return float(np.sqrt(np.mean((s - r) ** 2))) / den if den > 0 else float("nan")


def step_events(t: FloatArray, q_cmd: FloatArray, *, min_step: float = 0.01) -> list[tuple[float, float, float]]:
    """The steps of a rate-limited step plan: (t_start, q_before, q_after) per event - a run of command changes
    (gaps < 0.1 s) whose total change exceeds min_step."""
    d = np.flatnonzero(np.abs(np.diff(q_cmd)) > 1e-6)
    if d.size == 0:
        return []
    runs: list[list[int]] = [[int(d[0])]]
    for i in d[1:]:
        if t[i + 1] - t[runs[-1][-1] + 1] < 0.1:
            runs[-1].append(int(i))
        else:
            runs.append([int(i)])
    out = []
    for r in runs:
        a, z = r[0], r[-1] + 1
        before, after = float(q_cmd[a]), float(q_cmd[min(z, len(q_cmd) - 1)])
        if abs(after - before) >= min_step:
            out.append((float(t[a]), before, after))
    return out


def step_timing(t: FloatArray, q: FloatArray, events: Sequence[tuple[float, float, float]], *, frac: float = 0.1,
                hold_s: float = 1.5) -> list[float]:  # fmt: skip
    """Per step event: the time (s) from the command's first change until q has moved `frac` of the step from its
    value at the change (NaN if it never does within hold_s)."""
    out = []
    for t0, before, after in events:
        i0 = int(np.searchsorted(t, t0))
        sel = (t >= t0) & (t <= t0 + hold_s)
        idx = np.flatnonzero(sel)
        step = after - before
        moved = (q[idx] - q[i0]) * np.sign(step)
        hit = np.flatnonzero(moved >= frac * abs(step))
        if hit.size == 0:
            out.append(float("nan"))
            continue
        k = int(idx[hit[0]])
        # linear interpolation between the samples around the crossing
        if k > i0:
            m0, m1 = (q[k - 1] - q[i0]) * np.sign(step), (q[k] - q[i0]) * np.sign(step)
            a = (frac * abs(step) - m0) / (m1 - m0) if m1 != m0 else 1.0
            out.append(float(t[k - 1] + a * (t[k] - t[k - 1]) - t0))
        else:
            out.append(float(t[k] - t0))
    return out


def chirp_phase(t: FloatArray, f0: float, f1: float, duration_s: float) -> FloatArray:
    return np.asarray(2.0 * math.pi * (f0 * t + (f1 - f0) * t**2 / (2.0 * duration_s)), dtype=np.float64)


def lockin(t: FloatArray, y: FloatArray, phase: FloatArray, sel: npt.NDArray[np.bool_]) -> complex:
    """The complex amplitude of y at the chirp's phase over the selected samples (+ a constant and a drift)."""
    tt = t[sel]
    a = np.column_stack([np.sin(phase[sel]), np.cos(phase[sel]), np.ones(tt.size), tt - tt.mean()])
    c, *_ = np.linalg.lstsq(a, y[sel], rcond=None)
    return complex(c[0], c[1])  # y ~ c0 sin + c1 cos = Im((c0 + i c1) e^{i phase}) ... as a phasor


def chirp_response(t: FloatArray, q_cmd: FloatArray, q: FloatArray, params: Mapping[str, Any], freqs: Sequence[float],
                   *, half_band_hz: float = 0.25, fade_s: float = 0.5) -> dict[float, dict[str, float]]:  # fmt: skip
    """Gain and phase lag (deg, + = q lags q_cmd) of q against the logged command at each frequency."""
    f0, f1, dur = float(params["f0_hz"]), float(params["f1_hz"]), float(params["duration_s"])
    ph = chirp_phase(t, f0, f1, dur)
    finst = f0 + (f1 - f0) * t / dur
    out: dict[float, dict[str, float]] = {}
    for f in freqs:
        sel = (np.abs(finst - f) <= half_band_hz) & (t >= fade_s) & (t <= dur - fade_s)
        if np.count_nonzero(sel) < 50:
            continue
        zc = lockin(t, q_cmd, ph, sel)
        zq = lockin(t, q, ph, sel)
        # y = s sin(ph) + c cos(ph) = A sin(ph + phi) with A e^{i phi} = s + i c
        lag = math.degrees(np.angle(zc) - np.angle(zq))
        lag = (lag + 180.0) % 360.0 - 180.0
        out[float(f)] = dict(gain=abs(zq) / abs(zc), lag_deg=lag, f_centre=float(np.mean(finst[sel])))
    return out


def torque_compare(sim_tau: FloatArray, real_tau: FloatArray) -> dict[str, float]:
    s = sim_tau - np.mean(sim_tau)
    r = real_tau - np.mean(real_tau)
    rs, rr = float(np.sqrt(np.mean(s**2))), float(np.sqrt(np.mean(r**2)))
    corr = float(np.corrcoef(s, r)[0, 1]) if rs > 0 and rr > 0 else float("nan")
    return dict(rms_ratio=rs / rr if rr > 0 else float("nan"), corr=corr, real_rms=rr, sim_rms=rs)


@dataclass
class WindowScore:
    name: str
    kind: str
    joint: str
    nrms: float
    offset_rad: float
    delay10_real_ms: list[float] = field(default_factory=list)
    delay10_sim_ms: list[float] = field(default_factory=list)
    chirp_real: dict[float, dict[str, float]] = field(default_factory=dict)
    chirp_sim: dict[float, dict[str, float]] = field(default_factory=dict)
    torque: dict[str, float] = field(default_factory=dict)
    hang: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def delay_median_diff_ms(self) -> float:
        r = np.nanmedian(self.delay10_real_ms) if self.delay10_real_ms else float("nan")
        s = np.nanmedian(self.delay10_sim_ms) if self.delay10_sim_ms else float("nan")
        return float(s - r)


CHIRP_FREQS = (1.0, 2.0, 3.0, 4.0, 5.0)


def score(win: Window, res: ReplayResult, *, t_from: float = 0.0) -> WindowScore:
    """B1 numbers of one replayed window for its moving joint (inside the plan's window only)."""
    assert win.joint is not None
    j = J.IDX[win.joint]
    i0 = int(np.searchsorted(win.t, res.t[0]))
    n = len(res.t)
    t = res.t
    qr = win.j("q", j)[i0 : i0 + n]
    qc = win.j("q_cmd", j)[i0 : i0 + n]
    tr = win.j("tau_est", j)[i0 : i0 + n]
    qs = res.q[:, j]
    plan = (t >= max(t_from, 0.0)) & (t <= win.duration_s)
    sc = WindowScore(win.name, win.kind, win.joint, nrms(qs[plan], qr[plan]),
                     float(np.mean(qs[plan]) - np.mean(qr[plan])), hang=res.hang)  # fmt: skip
    if win.kind == "step":
        ev = step_events(t, qc)
        sc.delay10_real_ms = [1e3 * v for v in step_timing(t, qr, ev)]
        sc.delay10_sim_ms = [1e3 * v for v in step_timing(t, qs, ev)]
    elif win.kind == "chirp":
        sc.chirp_real = chirp_response(t, qc, qr, win.params, CHIRP_FREQS)
        sc.chirp_sim = chirp_response(t, qc, qs, win.params, CHIRP_FREQS)
    sc.torque = torque_compare(res.tau[plan, j], tr[plan])
    return sc


# ------------------------------------------------------------------------------------------------ the B1 / B2 table
B1_NRMS_MAX = 0.10
B1_PHASE_DEG = 5.0
B1_TORQUE_TOL = 0.15
B2_DELAY_MS = 1.0


def verdicts(sc: WindowScore) -> dict[str, bool | None]:
    """Each B1 / B2 target for one window: True / False, None = not applicable to this kind."""
    out: dict[str, bool | None] = {"nrms": bool(sc.nrms <= B1_NRMS_MAX)}
    out["delay"] = bool(abs(sc.delay_median_diff_ms) <= B2_DELAY_MS) if sc.kind == "step" else None
    if sc.kind == "chirp":
        ok = [abs(sc.chirp_sim[f]["lag_deg"] - sc.chirp_real[f]["lag_deg"]) <= B1_PHASE_DEG
              for f in (2.0, 5.0) if f in sc.chirp_sim and f in sc.chirp_real]  # fmt: skip
        out["phase"] = bool(all(ok)) if ok else None
    else:
        out["phase"] = None
    out["torque"] = bool(abs(sc.torque["rms_ratio"] - 1.0) <= B1_TORQUE_TOL)
    return out


def table(scores: Sequence[WindowScore]) -> str:
    """The markdown table of a replay run (one row per window)."""
    head = (
        "| window | joint | nRMS (<= 10 %) | delay10 real / sim ms (+-1) | lag 2 Hz real / sim "
        "| lag 5 Hz real / sim | torque RMS sim/real (1 +- 0.15), corr |"
    )
    rows = [head, "|---|---|---|---|---|---|---|"]
    for sc in scores:
        v = verdicts(sc)

        def mark(k: str, v: dict[str, bool | None] = v) -> str:
            return "" if v[k] is None else (" ok" if v[k] else " **X**")

        dl = (f"{np.nanmedian(sc.delay10_real_ms):.1f} / {np.nanmedian(sc.delay10_sim_ms):.1f}{mark('delay')}"
              if sc.kind == "step" else "-")  # fmt: skip
        lag = []
        for f in (2.0, 5.0):
            if f in sc.chirp_real and f in sc.chirp_sim:
                lag.append(f"{sc.chirp_real[f]['lag_deg']:.0f} / {sc.chirp_sim[f]['lag_deg']:.0f}")
            else:
                lag.append("-")
        rows.append(
            f"| {sc.name.split('_', 1)[1]} | {sc.joint} | {100 * sc.nrms:.1f} %{mark('nrms')} | {dl} | {lag[0]} | "
            f"{lag[1]}{mark('phase')} | {sc.torque['rms_ratio']:.2f}, {sc.torque['corr']:.2f}{mark('torque')} |"
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m r1sim.tier_b.replay --session <HW1 session copy> --out DIR [--config cfg.json] [--quick]`."""
    import argparse
    import json
    from pathlib import Path

    from .data import load_session

    ap = argparse.ArgumentParser(prog="python -m r1sim.tier_b.replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", required=True, help="a verified HW1 session copy (records/ + r1log/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="JSON: cmd_delay_s, state_delay_s, friction, damping, armature")
    ap.add_argument("--provisional-friction", action="store_true", help="the HW1 re-fit's leg / waist friction")
    ap.add_argument("--kinds", default="step,chirp")
    ap.add_argument("--quick", action="store_true", help="3 windows (left knee step + chirp, left elbow step)")
    a = ap.parse_args(argv)
    cfgd: dict[str, Any] = json.loads(Path(a.config).read_text()) if a.config else {}
    if a.provisional_friction and "friction" not in cfgd:
        cfgd["friction"] = PROVISIONAL_FRICTION
    cfg = ReplayConfig(**cfgd)
    wins = load_session(a.session, tuple(a.kinds.split(",")))
    if a.quick:
        wins = [w for w in wins if w.joint in ("left_knee", "left_elbow")][:3]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    for w in wins:
        sc = score(w, replay(w, cfg, t_from=-0.5))
        scores.append(sc)
        print(f"{w.name}: nrms {100 * sc.nrms:.1f} %, torque {sc.torque['rms_ratio']:.2f}", flush=True)
    (out / "b1_replay.json").write_text(
        json.dumps(
            dict(
                config=asdict(cfg),
                windows=[s.as_dict() for s in scores],
                verdicts={s.name: verdicts(s) for s in scores},
            ),
            indent=1,
            default=float,
        )
    )
    md = f"# B1 / B2 replay ({cfg.label()})\n\n{table(scores)}\n"
    (out / "b1_replay.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
