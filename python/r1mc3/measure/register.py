"""An entity's pose from its tags (DESIGN §8.2: the table / board / container poses relative to the robot).

Every corner of every tag of the entity, seen in one or more views (each view = a frame + the camera's pose in the
reference frame AT CAPTURE, e.g. the head camera from the joints' forward kinematics), enters one least-squares fit of
the entity's pose: the corners' predicted normalised image coordinates vs the detected ones (Levenberg-Marquardt,
numerical Jacobian; numpy only). One rigid fit over several tags is far better than a tag alone: a single 4-12 cm tag's
free PnP pose is ~8-9 mm off ALONG the ray (the session-3 render test), the spread of several tags pins it down.

`level=True` (default) fits x, y, z, yaw with the entity's face horizontal in the reference frame (a table / board /
container on a level floor, the reference = a gravity-aligned base frame); `level=False` fits all 6 degrees of freedom.
The result says how well it fits (reprojection RMS in px), how sure it is (1-sigma per coordinate from the fit's
covariance scaled by the residual) and refuses (ok False + the reason) with too few tags or a bad fit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .lenses import Lens
from .tags import TagObs, TagPlacement, rot_z

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class View:
    """One image's detections + its camera: `T_ref_cam` = the camera (OpenCV frame) in the reference frame at capture
    (the head camera: the forward kinematics of the joints at that frame @ the mount)."""

    tags: Mapping[int, TagObs]
    lens: Lens
    T_ref_cam: FloatArray
    name: str = ""


@dataclass(frozen=True)
class Registration:
    entity: str
    ok: bool
    reason: str
    T_ref_entity: FloatArray | None = None
    used: tuple[int, ...] = ()  # tag ids that entered the fit
    missing: tuple[int, ...] = ()  # tag ids of the plan not seen in any view
    n_views: int = 0
    rms_px: float = math.nan
    sigma: dict[str, float] = field(default_factory=dict)  # 1-sigma: x_mm, y_mm, z_mm, yaw_deg (+ roll / pitch)
    tag_err_mm: dict[int, float] = field(default_factory=dict)  # each tag centre's ray -> its fitted plane, vs the fit
    level: bool = True

    @property
    def xyz(self) -> FloatArray:
        assert self.T_ref_entity is not None
        return np.asarray(self.T_ref_entity[:3, 3])

    @property
    def yaw_deg(self) -> float:
        assert self.T_ref_entity is not None
        R = self.T_ref_entity
        return math.degrees(math.atan2(R[1, 0], R[0, 0]))

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "entity": self.entity,
            "ok": self.ok,
            "reason": self.reason,
            "used": list(self.used),
            "missing": list(self.missing),
            "n_views": self.n_views,
            "rms_px": None if math.isnan(self.rms_px) else round(self.rms_px, 3),
            "sigma": {k: round(v, 3) for k, v in self.sigma.items()},
            "tag_err_mm": {str(k): round(v, 2) for k, v in self.tag_err_mm.items()},
            "level": self.level,
        }
        if self.T_ref_entity is not None:
            d["xyz_m"] = [round(float(v), 5) for v in self.xyz]
            d["yaw_deg"] = round(self.yaw_deg, 3)
            d["T_ref_entity"] = np.round(self.T_ref_entity, 6).tolist()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Registration:
        """The inverse of `as_dict` (r1d's `measure.setup` / `measure.phone` replies -> a client's checks); the pose to
        the 6 decimals as_dict keeps (1 um / 1e-6 rad)."""
        T = d.get("T_ref_entity")
        rms = d.get("rms_px")
        return cls(
            entity=str(d["entity"]),
            ok=bool(d["ok"]),
            reason=str(d.get("reason", "")),
            T_ref_entity=None if T is None else np.asarray(T, dtype=np.float64),
            used=tuple(int(i) for i in d.get("used", ())),
            missing=tuple(int(i) for i in d.get("missing", ())),
            n_views=int(d.get("n_views", 0)),
            rms_px=math.nan if rms is None else float(rms),
            sigma={str(k): float(v) for k, v in dict(d.get("sigma", {})).items()},
            tag_err_mm={int(k): float(v) for k, v in dict(d.get("tag_err_mm", {})).items()},
            level=bool(d.get("level", True)),
        )


def rodrigues(rv: npt.ArrayLike) -> FloatArray:
    """Rotation vector -> rotation matrix."""
    v = np.asarray(rv, dtype=np.float64)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.asarray(np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K))


def rotvec(R: FloatArray) -> FloatArray:
    """Rotation matrix -> rotation vector."""
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if math.pi - th < 1e-6:  # near 180 deg: from the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(k))
        k = A[:, i] / max(k[i], 1e-12)
        return np.asarray(th * k / np.linalg.norm(k))
    return np.asarray(th * w / (2.0 * math.sin(th)))


def _pose(p: FloatArray, level: bool) -> FloatArray:
    T = np.eye(4)
    if level:
        T[:3, :3] = rot_z(float(p[3]))
    else:
        T[:3, :3] = rodrigues(p[3:6])
    T[:3, 3] = p[:3]
    return T


def _params(T: FloatArray, level: bool) -> FloatArray:
    if level:
        return np.array([T[0, 3], T[1, 3], T[2, 3], math.atan2(T[1, 0], T[0, 0])])
    return np.concatenate([T[:3, 3], rotvec(T[:3, :3])])


@dataclass
class _Obs:
    P_e: FloatArray  # N x 3 corners in the entity frame
    uv: FloatArray  # N x 2 normalised, detected
    T_cam_ref: FloatArray
    f_px: float
    tag_id: int
    view: int


def _residuals(p: FloatArray, obs: Sequence[_Obs], level: bool) -> FloatArray:
    T = _pose(p, level)
    out = []
    for o in obs:
        M = o.T_cam_ref @ T
        P = (M[:3, :3] @ o.P_e.T).T + M[:3, 3]
        z = np.where(P[:, 2] > 1e-6, P[:, 2], 1e-6)
        out.append(o.f_px * (P[:, :2] / z[:, None] - o.uv).ravel())
    return np.concatenate(out)


def _lm(p0: FloatArray, obs: Sequence[_Obs], level: bool, iters: int = 60) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Levenberg-Marquardt on the pixel-scaled normalised residuals -> (params, residuals, Jacobian at the end)."""
    p = p0.astype(np.float64).copy()
    lam = 1e-3
    r = _residuals(p, obs, level)
    cost = float(r @ r)
    steps = np.array([1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7])[: len(p)]
    J = np.zeros((len(r), len(p)))
    for _ in range(iters):
        for k in range(len(p)):
            dp = np.zeros_like(p)
            dp[k] = steps[k]
            J[:, k] = (_residuals(p + dp, obs, level) - _residuals(p - dp, obs, level)) / (2 * steps[k])
        A = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(10):
            try:
                delta = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            pn = p + delta
            rn = _residuals(pn, obs, level)
            cn = float(rn @ rn)
            if cn < cost:
                p, r, cost = pn, rn, cn
                lam = max(lam / 10, 1e-9)
                improved = True
                break
            lam *= 10
        if not improved or float(np.max(np.abs(delta))) < 1e-10:
            break
    for k in range(len(p)):
        dp = np.zeros_like(p)
        dp[k] = steps[k]
        J[:, k] = (_residuals(p + dp, obs, level) - _residuals(p - dp, obs, level)) / (2 * steps[k])
    return p, r, J


def _initial(obs: Sequence[_Obs], placements: Mapping[int, TagPlacement], views: Sequence[View]) -> FloatArray:
    """The entity pose from the largest tag seen (IPPE_SQUARE on normalised corners, K = I)."""
    import cv2

    best = max(obs, key=lambda o: float(np.ptp(o.uv[:, 0]) * o.f_px))
    pl = placements[best.tag_id]
    h = pl.size_m / 2.0
    sq = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0.0]])
    uv = np.ascontiguousarray(best.uv, dtype=np.float64).reshape(-1, 1, 2)
    ok, rv, tv = cv2.solvePnP(sq, uv, np.eye(3), np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("IPPE failed on the largest tag")
    T_cam_tag = np.eye(4)
    T_cam_tag[:3, :3] = cv2.Rodrigues(rv)[0]
    T_cam_tag[:3, 3] = np.asarray(tv).ravel()
    T_ref_cam = np.asarray(views[best.view].T_ref_cam, dtype=np.float64)
    return np.asarray(T_ref_cam @ T_cam_tag @ np.linalg.inv(pl.T_entity_tag))


def _ray_plane(o: FloatArray, d: FloatArray, p0: FloatArray, n: FloatArray) -> FloatArray:
    t = float((p0 - o) @ n) / float(d @ n)
    return np.asarray(o + t * d)


def locate_entity(
    entity: str,
    placements: Sequence[TagPlacement],
    views: Sequence[View],
    *,
    level: bool = True,
    min_tags: int = 2,
    max_rms_px: float = 2.0,
    prior: FloatArray | None = None,
) -> Registration:
    """The entity's pose in the reference frame from the tags of `placements` seen in `views`. Refuses (ok False, the
    reason in plain words) with fewer than `min_tags` distinct tags or a reprojection RMS above `max_rms_px`.
    `prior`: a starting pose (e.g. the nominal layout) instead of the largest tag's own pose."""
    pl = {p.tag_id: p for p in placements}
    obs: list[_Obs] = []
    for vi, v in enumerate(views):
        T_cam_ref = np.linalg.inv(np.asarray(v.T_ref_cam, dtype=np.float64))
        for tid, t in v.tags.items():
            if tid in pl:
                uv = v.lens.normalise(t.corners_px)
                obs.append(_Obs(pl[tid].corners_entity(), uv, T_cam_ref, float(v.lens.f_px), tid, vi))
    seen = tuple(sorted({o.tag_id for o in obs}))
    missing = tuple(sorted(set(pl) - set(seen)))
    if len(seen) < min_tags:
        return Registration(
            entity,
            False,
            f"only {len(seen)} of {len(pl)} tags seen on the {entity} (need {min_tags}); not seen: "
            + (", ".join(str(i) for i in missing) or "-")
            + " - uncover / flatten the tags or check the head pose",
            used=seen,
            missing=missing,
            n_views=len(views),
            level=level,
        )
    T0 = np.asarray(prior, dtype=np.float64) if prior is not None else _initial(obs, pl, views)
    p, r, J = _lm(_params(T0, level), obs, level)
    T = _pose(p, level)
    m, n = len(r), len(p)
    rms = float(np.sqrt(np.mean(r**2)))
    s2 = float(r @ r) / max(m - n, 1)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        sd = np.full(n, math.inf)
    sigma = {"x_mm": 1000 * sd[0], "y_mm": 1000 * sd[1], "z_mm": 1000 * sd[2]}
    if level:
        sigma["yaw_deg"] = math.degrees(sd[3])
    else:
        sigma.update(roll_deg=math.degrees(sd[3]), pitch_deg=math.degrees(sd[4]), yaw_deg=math.degrees(sd[5]))
    # each tag's centre: its detected ray meets its fitted plane - how far from where the fit puts it (mm)
    terr: dict[int, list[float]] = {}
    for o in obs:
        v = views[o.view]
        T_rt = T @ pl[o.tag_id].T_entity_tag
        c_px = TagObs(o.tag_id, v.tags[o.tag_id].corners_px).centre_px
        ray_c = np.r_[v.lens.normalise(c_px[None, :])[0], 1.0]
        Trc = np.asarray(v.T_ref_cam, dtype=np.float64)
        hit = _ray_plane(Trc[:3, 3], Trc[:3, :3] @ ray_c, T_rt[:3, 3], T_rt[:3, 2])
        terr.setdefault(o.tag_id, []).append(1000 * float(np.linalg.norm(hit - T_rt[:3, 3])))
    tag_err = {k: float(np.mean(vv)) for k, vv in terr.items()}
    ok = rms <= max_rms_px
    reason = "ok" if ok else f"the {entity}'s tags do not fit one rigid pose (rms {rms:.2f} px > {max_rms_px} px)"
    return Registration(entity, ok, reason, T, seen, missing, len(views), rms, sigma, tag_err, level)
