"""H1 (the user's decision after HW1, HW2_PREP.md "Decisions"): the IMU-to-pelvis rotation from the robot's OWN waist
axes, out of the HW1 waist recordings - no reference face, no level.

The R1 hung on its stand by the torso (the harness's top bar); the IMU sits on the PELVIS (r1.xml site `imu`). When
the waist joints move, the pelvis + legs rotate relative to the held torso about the waist ROLL axis (x of
waist_roll_link, the pelvis's child) and the waist YAW axis (z of torso_link, waist_roll_link's child). The gyro then
measures those two axes in the IMU's own frame, scaled by the joint rates:

    gyro(t) = -k_y qd_yaw(t) R^T a_yaw(q_roll(t)) - k_r qd_roll(t) R^T a_roll + b

with a_yaw / a_roll the axes in the PELVIS frame from the model's kinematics (a_yaw tilts with the waist roll angle),
R the IMU's mounting rotation in the pelvis (what this estimates), b the gyro bias and k_y, k_r ~ 1 the share of the
relative rotation the pelvis takes (the torso is held by a compliant cradle: it counter-rotates a little - this
scales an axis, it does not turn it). Two non-parallel axes fix a rotation: the scaled axes are fitted by linear least
squares (one 3-vector per joint rate), normalised, and the rotation that maps the model's axes onto them is the
Wahba / Kabsch solution. Reported as (roll, pitch, yaw) of R in degrees - the model's `IMU_MOUNT_OFFSET_DEG` is
(roll, pitch) = (0.50, 0.96) from the 9 Oct legs-vs-IMU comparison (session 4).

What can bias it (said plainly): a counter-rotation of the torso about a DIFFERENT axis than the joint's (the
cradle's yaw / roll compliance is not the joint axis), gyro scale-factor errors per axis, a time offset between the
gyro and the encoders (both from the same lowstate here). The same estimator run on the simulator's replay of the
same commands (whose IMU offset is known) measures how much of this the method itself carries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from .. import joints as J

FloatArray = npt.NDArray[np.float64]
Segment = tuple[FloatArray, FloatArray, FloatArray, FloatArray]  # t, q_roll, q_yaw, gyro (n, 3)
J_ROLL = J.IDX["waist_roll"]
J_YAW = J.IDX["waist_yaw"]


def waist_axes(m: Any, q_roll: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The waist roll axis (3,) and the waist yaw axis per sample (n, 3) in the PELVIS frame, from the model."""
    jr, jy = m.joint("waist_roll_joint"), m.joint("waist_yaw_joint")
    br, by = int(m.jnt_bodyid[jr.id]), int(m.jnt_bodyid[jy.id])
    r_pw = np.zeros(9)
    mujoco.mju_quat2Mat(r_pw, m.body_quat[br])  # waist_roll_link in the pelvis (at q = 0)
    r_wt = np.zeros(9)
    mujoco.mju_quat2Mat(r_wt, m.body_quat[by])  # torso_link in waist_roll_link
    rpw, rwt = r_pw.reshape(3, 3), r_wt.reshape(3, 3)
    a_roll = rpw @ np.asarray(m.jnt_axis[jr.id], dtype=np.float64)
    ax_y = np.asarray(m.jnt_axis[jy.id], dtype=np.float64)
    out = np.zeros((len(q_roll), 3))
    for i, qr in enumerate(q_roll):
        c, s = math.cos(qr), math.sin(qr)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])  # about the roll axis (x of waist_roll_link)
        out[i] = rpw @ rx @ rwt @ ax_y
    return a_roll / np.linalg.norm(a_roll), out


def _rate(t: FloatArray, q: FloatArray, half_s: float = 0.01) -> FloatArray:
    """dq/dt by a centred difference over +-half_s (the encoder dq is noisier)."""
    dt = float(np.median(np.diff(t)))
    k = max(1, round(half_s / dt))
    v = np.full_like(q, np.nan)
    v[k:-k] = (q[2 * k :] - q[: -2 * k]) / (t[2 * k :] - t[: -2 * k])
    return v


def kabsch(src: FloatArray, dst: FloatArray, w: FloatArray | None = None) -> FloatArray:
    """The rotation R minimising sum w |R src_i - dst_i|^2 (rows = vectors)."""
    ww = np.ones(len(src)) if w is None else np.asarray(w, dtype=np.float64)
    h = (src * ww[:, None]).T @ dst
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return np.asarray(vt.T @ np.diag([1.0, 1.0, d]) @ u.T, dtype=np.float64)


def rpy_deg(r: FloatArray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) in degrees of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


@dataclass
class H1Result:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    k_roll: float  # |fitted roll vector|: the pelvis's share of the waist roll rate
    k_yaw: float
    axis_angle_deg: float  # the angle between the fitted axes (the model's: ~90)
    residual_rms: float  # rad/s
    n: int
    se_deg: tuple[float, float, float] = (float("nan"), float("nan"), float("nan"))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate(
    m: Any,
    segments: Sequence[Segment],
    *,
    min_rate: float = 0.02,
    n_boot: int = 50,
    seed: int = 0,
) -> H1Result:
    """segments: (t, q_roll, q_yaw, gyro (n, 3)) per recording. Fit gyro = C_r qd_roll + C_y qd_yaw + b (one 3-vector
    per rate: the axes as the IMU sees them, scaled by -k), Kabsch onto the model's axes, bootstrap over 0.5 s
    blocks for the standard error."""
    rates: list[FloatArray] = []
    gyros: list[FloatArray] = []
    yaw_axes: list[FloatArray] = []
    a_roll = np.zeros(3)
    for t, qr, qy, gyro in segments:
        vr, vy = _rate(t, qr), _rate(t, qy)
        a_roll, a_yaw = waist_axes(m, qr)
        ok = np.isfinite(vr) & np.isfinite(vy)
        rates.append(np.column_stack([vr, vy])[ok])
        gyros.append(np.asarray(gyro, dtype=np.float64)[ok])
        yaw_axes.append(a_yaw[ok & (np.abs(vy) > min_rate)])
    v_all = np.vstack(rates)
    g_all = np.vstack(gyros)
    for k, name in ((0, "roll"), (1, "yaw")):
        if np.count_nonzero(np.abs(v_all[:, k]) > min_rate) < 100:
            raise ValueError(f"H1 needs both waist axes moving: the waist {name} rate never exceeds {min_rate} rad/s")
    a_y = np.mean(np.vstack(yaw_axes), axis=0)
    src = np.vstack([a_roll, a_y / np.linalg.norm(a_y)])

    def fit(v: FloatArray, g: FloatArray) -> tuple[FloatArray, float, float, float, float]:
        a = np.column_stack([v, np.ones(len(v))])
        coef, *_ = np.linalg.lstsq(a, g, rcond=None)
        u_r, u_y = -coef[0], -coef[1]
        kr, ky = float(np.linalg.norm(u_r)), float(np.linalg.norm(u_y))
        dst = np.vstack([u_r / kr, u_y / ky])
        rt = kabsch(src, dst)  # pelvis-frame axes -> IMU-frame axes = R^T
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(dst[0] @ dst[1])))))
        return rt.T, kr, ky, ang, float(np.sqrt(np.mean((g - a @ coef) ** 2)))

    r, kr, ky, ang, rms = fit(v_all, g_all)
    roll, pitch, yaw = rpy_deg(r)
    rng = np.random.default_rng(seed)
    nb = 250  # 0.5 s at 500 Hz
    nblk = len(v_all) // nb
    boots = []
    for _ in range(n_boot if nblk > 4 else 0):
        idx = np.concatenate([np.arange(p * nb, (p + 1) * nb) for p in rng.integers(0, nblk, nblk)])
        boots.append(rpy_deg(fit(v_all[idx], g_all[idx])[0]))
    se = tuple(float(x) for x in np.std(np.asarray(boots), axis=0)) if len(boots) > 4 else (float("nan"),) * 3
    return H1Result(roll, pitch, yaw, kr, ky, ang, rms, len(v_all), (se[0], se[1], se[2]))


def segments_of(windows: Sequence[Any]) -> list[Segment]:
    """(t, q_roll, q_yaw, gyro) of recorded windows (r1sim.tier_b.data.Window with the IMU)."""
    return [(w.t, w.j("q", J_ROLL), w.j("q", J_YAW), w.imu["imu_gyro"]) for w in windows]


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m r1sim.tier_b.h1_imu --session <HW1 session copy> [--sim-config cfg.json]`: the estimate from the
    waist recordings (S2 steps, S3 chirp, S4 ramps of waist roll / yaw); with --sim-config also from the simulator's
    replay of the same commands (its IMU offset known: the method's own bias)."""
    import argparse
    import json
    from pathlib import Path

    from . import replay as R
    from .data import load_session

    ap = argparse.ArgumentParser(prog="python -m r1sim.tier_b.h1_imu")
    ap.add_argument("--session", required=True)
    ap.add_argument("--sim-config", help="a replay config JSON (r1sim.tier_b.replay.ReplayConfig fields)")
    ap.add_argument("--out", help="write the result JSON here")
    a = ap.parse_args(argv)
    wins = [w for w in load_session(a.session, ("step", "chirp", "ramp")) if (w.joint or "").startswith("waist")]
    m = R._base_world().model
    out: dict[str, Any] = {"windows": [w.name for w in wins], "real": estimate(m, segments_of(wins)).as_dict()}
    if a.sim_config:
        cfg = R.ReplayConfig(**json.loads(Path(a.sim_config).read_text()))
        segs: list[Segment] = []
        for w in wins:
            res = R.replay(w, cfg, t_from=-0.5)
            segs.append((res.t, res.q[:, J_ROLL], res.q[:, J_YAW], res.gyro))
        out["sim"] = estimate(m, segs).as_dict()
        out["sim_truth_deg"] = _site_offset_deg(m)
    print(json.dumps(out, indent=1, default=float))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    return 0


def _site_offset_deg(m: Any) -> tuple[float, float, float]:
    """The model's IMU mounting rotation (the `imu` site's quaternion in the pelvis) as (roll, pitch, yaw) deg."""
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, m.site_quat[m.site("imu").id])
    return rpy_deg(r.reshape(3, 3))


if __name__ == "__main__":
    raise SystemExit(main())
