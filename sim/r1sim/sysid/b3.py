"""B3 (DESIGN.md §13.2: the static torques within +-10 % per joint over >= 20 poses, 0 / 0.5 kg) from HW2's slow arm
SWEEPS on the stand - session 11 (core), the analysis HW1_FITS.md "Session 8" item 3 asked for.

Why sweeps (HW1_FITS.md): a joint at rest under a position servo sits anywhere inside its friction band, and on the
shoulders that band (~1 N m) is as large as a 0.5 kg payload's effect - so HW1's holds could not show +-10 %. A slow
up-and-down sweep of one joint under a soft servo (`r1 record run ramp joint=<arm joint> ... payload_kg=<kg> --yes`:
the S4 stiction protocol, r1mc3.record.plans "ramp") passes every q once moving up and once moving down: the friction
has the opposite sign on the two passes, so their mean is the holding torque g(q) (`r1sim.sysid.stiction.fit_joint`'s
`gravity_q` / `gravity_tau`, the quartic over the steady samples with a sign(v) friction term). The same sweep with a
known payload gives the torque scale independently of the model's masses.

Per joint, from its sweeps (each a manifest + its r1log files; the payload from the manifest's `payload_kg` or given):

* the model's holding torque at the measured poses: MuJoCo's real-profile model (`world.build_world("real")`), the
  pelvis oriented by the recording's IMU quaternion (the IMU site's mount undone), every other joint at its median,
  the hands at the model's rest; the payload a point mass at the swept hand's palm site (+ an offset);
* **mass scale** s: measured g vs the model's at 0 kg (least squares with a constant offset c - a cable / a tau_est
  zero; HW1: c -0.13..-0.76 N m on the arms);
* **payload scale** k: the measured 0.5 kg increment (the payload sweep minus the empty one at the same q) vs the
  model's (J^T m g at the payload point) - the torque scale, whatever the arm's masses;
* **the B3 check** with the model's masses (s = 1): every point with |g_model| >= min_g within tol |g_model| after
  the joint's constant offset; shown when >= min_poses points and >= min_fraction of them pass AND both scales are
  within tol of 1 (a free offset over a short torque range would otherwise hide a mass error).

    python -m r1sim.sysid.b3 --sweep MANIFEST[@KG] [--sweep ...] [--source pd|tau_est] [--json out.json]

The synthetic check (tests/unit/test_sysid_b3.py): sweeps run in the hanging simulator with a payload body ARE what
the model says (s ~ 1, k ~ 1, shown); the same sweeps on a robot whose arm links are 20 % heavier than the model read
s ~ 1.2 and are NOT shown.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from .. import joints as J
from .records import BODY_FIELDS, BodyRecording
from .stiction import fit_joint, ramp_recording

FloatArray = npt.NDArray[np.float64]

ARM_JOINTS: tuple[str, ...] = tuple(J.JOINTS[14:24])
TOL = 0.10  # DESIGN §13.2 B3
MIN_POSES = 20
MIN_FRACTION = 0.9
MIN_G = 0.5  # N m: below this the relative check is noise (HW1_FITS: 1.0 for the shoulders' holds, 0.5 for yaw)
GRAVITY = 9.81


@dataclass
class Sweep:
    """One joint's sweep: the recording, the IMU quaternion (w, x, y, z; the median), the payload."""

    joint: str
    payload_kg: float
    rec: BodyRecording
    imu_quat: FloatArray
    source: str = ""  # where it came from (a manifest path / "synthetic")


@dataclass
class SweepGravity:
    joint: str
    payload_kg: float
    q: list[float]  # the joint's q at the fitted points
    tau_meas: list[float]  # the measured holding torque (up / down mean), N m
    tau_model: list[float]  # the model's at the same poses (incl. the payload), N m
    tau_payload_model: list[float]  # the payload's share of tau_model, N m
    frictionloss: float  # the sweep's own Fc (the half width), N m - a by-product
    notes: list[str] = field(default_factory=list)


@dataclass
class B3Joint:
    joint: str
    n_points: int  # points with |g_model| >= min_g over all its sweeps
    payloads_kg: list[float]
    mass_scale: float  # s from the 0 kg sweeps (nan without one)
    mass_scale_se: float
    offset_nm: float  # c
    payload_scale: float  # k (nan without a 0 kg + a payload sweep)
    median_rel_error: float  # with the model's masses after the offset
    within_tol: float
    shown: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ the model's side
class GravityModel:
    """MuJoCo's holding torques (qfrc_bias at rest) on the real-profile model for given joint poses and pelvis
    orientations, plus a point payload's share (J^T m g at a point in a site's frame)."""

    def __init__(self, model: mujoco.MjModel | None = None, index: Any = None) -> None:
        if model is None:
            from .. import world

            w = world.build_world("real", with_scene=False, visual=False)
            model, index = w.model, w.index
        self.m = model
        self.ix = index
        self.d = mujoco.MjData(model)
        sid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu"))
        self.q_site_imu = model.site_quat[sid].copy() if sid >= 0 else np.array([1.0, 0.0, 0.0, 0.0])

    def _set(self, q26: FloatArray, imu_quat: FloatArray) -> None:
        d, ix = self.d, self.ix
        d.qpos[:] = self.m.qpos0
        d.qvel[:] = 0.0
        # the pelvis: R_world_pelvis = R_world_imu R_pelvis_imu^T (the IMU site's mount, session 4's offset)
        inv = np.array([self.q_site_imu[0], *(-self.q_site_imu[1:])])
        qp = np.zeros(4)
        mujoco.mju_mulQuat(qp, np.asarray(imu_quat, dtype=np.float64), inv)
        d.qpos[3:7] = qp / np.linalg.norm(qp)
        for i in range(J.N):
            if ix.present[i]:
                d.qpos[ix.qadr[i]] = q26[i]
        mujoco.mj_forward(self.m, d)

    def holding(
        self,
        q26: npt.ArrayLike,
        imu_quat: npt.ArrayLike,
        joint: str,
        *,
        payload_kg: float = 0.0,
        payload_site: str | None = None,
        payload_offset: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> tuple[float, float]:
        """(the holding torque of `joint` incl. the payload, the payload's share), N m (+ = in the +q direction)."""
        self._set(np.asarray(q26, dtype=np.float64), np.asarray(imu_quat, dtype=np.float64))
        j = J.IDX[joint]
        v = int(self.ix.vadr[j])
        tau = float(self.d.qfrc_bias[v])
        share = 0.0
        if payload_kg > 0.0:
            site = payload_site or ("left_hand_palm" if joint.startswith("left") else "right_hand_palm")
            sid = int(mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, site))
            if sid < 0:
                raise ValueError(f"no site {site!r} in the model")
            R = self.d.site_xmat[sid].reshape(3, 3)
            point = self.d.site_xpos[sid] + R @ np.asarray(payload_offset, dtype=np.float64)
            jacp = np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, jacp, None, point, int(self.m.site_bodyid[sid]))
            share = float(jacp[:, v] @ np.array([0.0, 0.0, payload_kg * GRAVITY]))
        return tau + share, share


# ------------------------------------------------------------------------------------------------ one sweep
def sweep_gravity(
    sw: Sweep,
    gm: GravityModel,
    *,
    source: str = "pd",
    payload_site: str | None = None,
    payload_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> SweepGravity:
    """The measured holding torque along the sweep (stiction.fit_joint's g(q)) and the model's at the same poses."""
    fit = fit_joint(sw.rec, sw.joint, source=source)
    j = J.IDX[sw.joint]
    q_med = np.median(np.asarray(sw.rec["q"], dtype=np.float64), axis=0)
    tm, tp = [], []
    for qj in fit.gravity_q:
        q26 = q_med.copy()
        q26[j] = qj
        a, b = gm.holding(
            q26, sw.imu_quat, sw.joint, payload_kg=sw.payload_kg, payload_site=payload_site,
            payload_offset=payload_offset,
        )  # fmt: skip
        tm.append(a)
        tp.append(b)
    return SweepGravity(sw.joint, sw.payload_kg, list(fit.gravity_q), list(fit.gravity_tau), tm, tp,
                        float(fit.frictionloss), list(fit.notes))  # fmt: skip


def b3_joint(
    sweeps: Sequence[SweepGravity],
    *,
    tol: float = TOL,
    min_g: float = MIN_G,
    min_poses: int = MIN_POSES,
    min_fraction: float = MIN_FRACTION,
) -> B3Joint:
    """One joint's B3 verdict from its sweeps (see the module doc)."""
    if not sweeps:
        raise ValueError("no sweeps")
    name = sweeps[0].joint
    if any(s.joint != name for s in sweeps):
        raise ValueError("b3_joint: sweeps of one joint only")
    notes: list[str] = []
    meas = np.concatenate([np.asarray(s.tau_meas) for s in sweeps])
    model = np.concatenate([np.asarray(s.tau_model) for s in sweeps])
    big = np.abs(model) >= min_g
    n = int(big.sum())
    nan = float("nan")
    zero = [s for s in sweeps if s.payload_kg == 0.0]
    s_, s_se, c = nan, nan, nan
    if zero:
        gm0 = np.concatenate([np.asarray(s.tau_model) for s in zero])
        tm0 = np.concatenate([np.asarray(s.tau_meas) for s in zero])
        A = np.column_stack([gm0, np.ones(gm0.size)])
        coef, *_ = np.linalg.lstsq(A, tm0, rcond=None)
        res = tm0 - A @ coef
        s2 = float(res @ res) / max(1, gm0.size - 2)
        se = np.sqrt(np.clip(np.diag(s2 * np.linalg.pinv(A.T @ A)), 0.0, None))
        s_, s_se = float(coef[0]), float(se[0])
        if float(np.ptp(gm0)) < 2.0 * min_g:
            notes.append(f"the empty sweep spans only {np.ptp(gm0):.2f} N m of model torque: the mass scale is weak")
    else:
        notes.append("no 0 kg sweep: no mass scale")
    off = float(np.median(meas - model)) if meas.size else nan  # the joint's constant offset with the model's masses
    c = off
    k = nan
    loaded = [s for s in sweeps if s.payload_kg > 0.0]
    if zero and loaded:
        z = zero[0]
        num = den = 0.0
        for s in loaded:
            lo, hi = max(min(s.q), min(z.q)), min(max(s.q), max(z.q))
            if hi <= lo:
                notes.append(f"the {s.payload_kg} kg sweep and the empty one share no q range")
                continue
            qq = np.linspace(lo, hi, 15)
            d_meas = np.interp(qq, s.q, s.tau_meas) - np.interp(qq, z.q, z.tau_meas)
            d_model = np.interp(qq, s.q, s.tau_payload_model)
            num += float(d_meas @ d_model)
            den += float(d_model @ d_model)
        k = num / den if den > 0 else nan
    elif not loaded:
        notes.append("no payload sweep: no torque scale")
    e = meas - model - off
    rel = np.abs(e[big]) / np.abs(model[big]) if n else np.zeros(0)
    within = float(np.mean(rel <= tol)) if n else 0.0
    if n < min_poses:
        notes.append(f"{n} points with |g| >= {min_g} N m < {min_poses}: +-{tol:.0%} cannot be shown")
    # shown: enough points, the per-point check AND both scales within tol - a free offset over a short torque range
    # absorbs most of a mass error (the synthetic arm 20 % heavier: s 1.19, yet 90 % of the points within 10 % after the
    # offset), so the scales decide too; both need the 0 kg AND a payload sweep (DESIGN §13.2: "0 / 0.5 kg")
    scales_ok = bool(np.isfinite(s_) and abs(s_ - 1.0) <= tol and np.isfinite(k) and abs(k - 1.0) <= tol)
    if np.isfinite(s_) and abs(s_ - 1.0) > tol:
        notes.append(f"mass scale {s_:.3f}: the model's masses are off by more than {tol:.0%} for this joint")
    if np.isfinite(k) and abs(k - 1.0) > tol:
        notes.append(f"payload scale {k:.3f}: the torque scale (per N m of known load) is off by more than {tol:.0%}")
    return B3Joint(
        joint=name,
        n_points=n,
        payloads_kg=sorted({float(s.payload_kg) for s in sweeps}),
        mass_scale=s_,
        mass_scale_se=s_se,
        offset_nm=c,
        payload_scale=k,
        median_rel_error=float(np.median(rel)) if n else nan,
        within_tol=within,
        shown=n >= min_poses and within >= min_fraction and scales_ok,
        notes=notes,
    )


# ------------------------------------------------------------------------------------------------ synthetic check
ARM_PARTS = ("shoulder", "elbow", "wrist", "hand")


def synthetic_sweep(
    joint: str,
    *,
    payload_kg: float = 0.0,
    arm_mass_scale: float = 1.0,
    amplitude: float = 0.3,
    speeds: Sequence[float] = (0.1, 0.2),
    kp: float = 20.0,
    kd: float = 1.0,
    seed: int = 0,
    sensors: bool = True,
) -> Sweep:
    """The B3 sweep run on the simulated HANGING robot (the pelvis welded upright, as stiction.hanging_world) whose
    swept arm carries a point payload at its palm site and whose arm links are `arm_mass_scale` x the model's: what
    r1log would record (the sensor model's readings, the soft servo's command; the other joints held stiff at the
    policy's stand pose)."""
    from .. import sensors as S
    from .. import world
    from ..driver import DriverConfig
    from ..engine import LowCmd, LowState, Simulation
    from .records import body_from_arrays
    from .stiction import HANG_Z, RATE_HZ, ramp_plan

    side = "left" if joint.startswith("left") else "right"

    def extra(spec: mujoco.MjSpec, _info: Any) -> None:
        pelvis = spec.body("pelvis")
        pelvis.pos = [0.0, 0.0, HANG_Z]
        data = np.zeros(11)
        data[10] = 1.0
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1="pelvis", objtype=mujoco.mjtObj.mjOBJ_BODY, data=data)
        if arm_mass_scale != 1.0:  # every link of the swept arm, the hand's too (the forearm's load is mostly the hand)
            for body in spec.bodies:
                if body.name.startswith(f"{side}_") and any(k in body.name for k in ARM_PARTS):
                    body.mass = float(body.mass) * arm_mass_scale
        if payload_kg > 0.0:
            site = spec.site(f"{side}_hand_palm")
            parent = spec.body(f"{side}_hand_palm_link")
            pl = parent.add_body(name="b3_payload", pos=list(site.pos))
            pl.mass = payload_kg
            pl.ipos = [0.0, 0.0, 0.0]  # a point AT the site (MjSpec's default class would put it at 5 / -3 cm)
            pl.inertia = [1e-5, 1e-5, 1e-5]
            pl.explicitinertial = True

    w = world.build_world("real", with_scene=False, visual=False, extra=extra)
    m, ix = w.model, w.index
    plan = ramp_plan(joint, amplitude=amplitude, speeds=speeds, kp=kp, kd=kd)
    j = J.IDX[joint]
    sm = S.SensorModel(S.SensorConfig.from_yaml() if sensors else S.SensorConfig.off(), seed=seed)
    sim = Simulation(m, ix, seed=seed, driver=DriverConfig.real(), sensors=sm)
    qpos = m.qpos0.copy()
    hold = J.POSE_STAND.copy()
    for i in range(J.N):
        if ix.present[i]:
            qpos[ix.qadr[i]] = hold[i]
    sim.reset(qpos)
    kps, kds = J.KP_STIFF.copy(), J.KD_STIFF.copy()
    kps[j], kds[j] = plan.kp, plan.kd
    rows: dict[str, list[FloatArray]] = {k: [] for k in BODY_FIELDS}
    times: list[float] = []

    def ctl(st: LowState) -> LowCmd:
        q_cmd = hold.copy()
        q_cmd[j] = float(plan.q_cmd(st.t, hold[j]))
        cmd = LowCmd(q=q_cmd, kp=kps.copy(), kd=kds.copy())
        times.append(st.t)
        for k, v in (("q", st.q), ("dq", st.dq), ("tau_est", st.tau_est), ("q_cmd", q_cmd), ("dq_cmd", cmd.dq),
                     ("kp", cmd.kp), ("kd", cmd.kd), ("tau_ff", cmd.tau_ff)):  # fmt: skip
            rows[k].append(np.array(v, dtype=np.float64, copy=True))
        return cmd

    sim.add_controller(ctl, 1.0 / RATE_HZ, "b3")
    sim.run_for(plan.duration_s + 0.5)
    rec = body_from_arrays(np.asarray(times), {k: np.asarray(v) for k, v in rows.items()})
    return Sweep(joint, payload_kg, rec, np.array([1.0, 0.0, 0.0, 0.0]), source="synthetic")


# ------------------------------------------------------------------------------------------------ the files
def load_sweep(spec: str) -> Sweep:
    """`MANIFEST[@KG]`: a `r1 record run ramp ...` manifest + its r1log files (/r1rt/telemetry incl. imu_quat); the
    joint from the manifest's plan, the payload from @KG, else the manifest's `payload_kg`, else 0."""
    from r1mc3.log import reader

    from .records import BODY_TOPIC

    path, _, kg = spec.partition("@")
    man = json.loads(Path(path).read_text())
    params = dict(man["plan"]["params"])
    joint = str(params["joint"])
    payload = float(kg) if kg else float(params.get("payload_kg", 0.0))
    ts: list[npt.NDArray[np.int64]] = []
    cols: dict[str, list[FloatArray]] = {k: [] for k in (*BODY_FIELDS, "imu_quat")}
    for p in man.get("r1log_files") or []:
        rec = reader.read(Path(p), log_time_order=True).topic(BODY_TOPIC)
        if rec.size == 0:
            continue
        ts.append(rec["t_wake_ns"].astype(np.int64))
        for k in cols:
            cols[k].append(np.asarray(rec[k], dtype=np.float64))
    if not ts:
        raise ValueError(f"{path}: no {BODY_TOPIC} records in its r1log files")
    t = np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    fields = {k: np.concatenate(v)[order] for k, v in cols.items()}
    body = ramp_recording(man, t[order], {k: fields[k] for k in BODY_FIELDS})
    t0, t1 = int(man["t_start_mono_ns"]), int(man.get("t_end_mono_ns") or t[order][-1])
    sel = (t[order] >= t0) & (t[order] <= t1)
    quat = np.median(fields["imu_quat"][sel], axis=0) if sel.any() else np.array([1.0, 0.0, 0.0, 0.0])
    return Sweep(joint, payload, body, quat / np.linalg.norm(quat), source=path)


def analyse(
    sweeps: Sequence[Sweep],
    *,
    source: str = "pd",
    payload_site: str | None = None,
    payload_offset: Sequence[float] = (0.0, 0.0, 0.0),
    gm: GravityModel | None = None,
) -> dict[str, Any]:
    """Every joint's B3 verdict + the per-sweep numbers (JSON-ready)."""
    gm = gm or GravityModel()
    by_joint: dict[str, list[SweepGravity]] = {}
    for sw in sweeps:
        by_joint.setdefault(sw.joint, []).append(
            sweep_gravity(sw, gm, source=source, payload_site=payload_site, payload_offset=payload_offset)
        )
    joints = {jn: b3_joint(sg).as_dict() for jn, sg in by_joint.items()}
    return {
        "target": f"static torques within +-{TOL:.0%} per joint over >= {MIN_POSES} poses, 0 / 0.5 kg (DESIGN §13.2)",
        "torque_source": source,
        "joints": joints,
        "sweeps": [asdict(s) for v in by_joint.values() for s in v],
        "shown_all": bool(joints) and all(j["shown"] for j in joints.values()),
    }


def table(result: Mapping[str, Any]) -> str:
    """The per-joint verdict as a markdown table (what the runbook's B3 row and HW2.md quote)."""
    head = "| joint | points | payloads kg | mass scale s | torque scale k | offset N m | median err | in 10 % | B3 |"
    rows = [head, "|---|---|---|---|---|---|---|---|---|"]
    for jn, r in result["joints"].items():
        pay = ", ".join(f"{p:g}" for p in r["payloads_kg"])
        verdict = "shown" if r["shown"] else "NOT shown"
        rows.append(
            f"| {jn} | {r['n_points']} | {pay} | {r['mass_scale']:.3f} +- {r['mass_scale_se']:.3f} | "
            f"{r['payload_scale']:.3f} | {r['offset_nm']:+.3f} | {100 * r['median_rel_error']:.1f} % | "
            f"{100 * r['within_tol']:.0f} % | {verdict} |"
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1sim.sysid.b3", description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", action="append", required=True, help="MANIFEST[@KG] of one `r1 record run ramp` sweep")
    ap.add_argument("--source", choices=("pd", "tau_est"), default="pd", help="the board's PD law (default) or tau_est")
    ap.add_argument("--payload-site", default=None, help="the payload's site (default: the swept hand's palm)")
    ap.add_argument("--payload-offset", default="0,0,0", help="X,Y,Z m in the site's frame (the measured place)")
    ap.add_argument("--json", default=None, help="write the full result here")
    a = ap.parse_args(argv)
    off = [float(x) for x in a.payload_offset.split(",")]
    res = analyse([load_sweep(s) for s in a.sweep], source=a.source, payload_site=a.payload_site, payload_offset=off)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))
    print(table(res))
    print("B3:", "SHOWN for every swept joint" if res["shown_all"] else "NOT shown (see the rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
