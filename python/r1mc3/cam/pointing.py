"""Pointing: a head-image pixel -> a point in `base` (DESIGN §8.1; ported from v2's headcam.py, git r21_try c3702ed).

v2's method, which v2's table-task numbers rest on (INV App. C §2): the calibrated line of sight of the clicked pixel
meets the DECLARED surface it looks at (4 Oct 2026, the user's decision: the R1's stereo depth is off 1-8 cm across a
table, the calibrated ray a few mm); the depth only says how high above that surface the clicked thing stands (a box
top), judged against the surface right around the click (7 Oct: the native depth's bias is smooth). Depth never finds
objects: it only converts the client's own choice of pixel.

What changed from v2 (generic, D32): v2 knew one table + raised "surfaces" on it; here the supports are whatever the
client declared in the world model (`r1mc3.world`): every Surface's top face and every Container's floor (a hand may
be inside; a click inside it means its floor, not the rim / far wall the depth hits - v2's 8 Oct rule), each with its
own outline and yaw. Heights are reported above the support the line of sight meets ("height_above_surface_m"); a
client that wants v2's "above the table" subtracts its own table top.

Pure functions (no robot, no simulator): r1d's `camera.point` / `camera.object_extent` (service/perception.py) and the
tests call them. The camera model dict = v2's / calib_head.py's (`lens` pinhole | equidistant, `f`, `cx`, `cy`, `k1`,
`depth` z | range); pixel centres at integer indices (OpenCV); the optical frame x right, y down, z forward.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

DEPTH_MIN_M, DEPTH_MAX_M = 0.10, 3.0
ON_SURFACE_M = 0.015  # v2 ON_TABLE_M: a depth height above the surface below this = on the surface
REL_RAISED_M = 0.04  # v2: raised when this far above the surface right around the click ...
ABS_RAISED_M = 0.08  # ... or this far above the surface outright (a large object filling the ring)
RING_PX = (18, 60)  # v2: the ring around the click that stands for "the surface around it"
BELOW_SURFACE_M = -0.03  # v2 BELOW_TABLE_M: depth clearly below the surface = off it (the floor)
BELOW_INSIDE_MARGIN_M = 0.02  # v2 (9 Oct, F9): ...unless the line of sight meets a surface this far inside its outline
BELOW_INSIDE_LABEL = "on the surface (the depth read below it here - glare?): where the line of sight meets it"
OUTLINE_MARGIN_M = 0.02  # v2: a pointed point may lie this far outside the declared outline
MAX_HEIGHT_M = 0.45  # v2: a pointed height is capped here
# session 13 (HW2 day 2's teleop runs: the click -> object layer 2-6 cm off): a click whose depth patch straddles an
# object's edge (part of the patch on the object, part on the surface behind it) took the patch MEDIAN - the surface
# behind when more of the patch looked past the object. Above this spread the patch holds two surfaces: it is split
# at the middle of its 15th / 85th percentiles and the side the CLICKED pixel itself shows (its 3 x 3 median) is used
# - the object for a click on its edge, the table for a click on the table beside an object (the median's job, kept).
# On a plain surface the real native depth's patch spread is 2.7-8.7 mm (runs_calib s4_objects), so a 7 x 7 patch on
# the table keeps v2's median.
EDGE_SPREAD_M = 0.015
CONTAINER_NO_DEPTH_LABEL = "the container's floor, by geometry (no depth there: a shiny container gives none)"
NO_DEPTH_WORDS = (
    "no depth reading at that pixel (a clear, shiny or dark object, or outside the depth view): the robot cannot tell "
    "how high the thing there is, and taking it as the surface would put the hand beyond it - click the object's foot "
    "(where it meets the table or the board) or give its height"
)


# ---------------------------------------------------------------- the camera model (v2 headcam.py:72-123)
def _undistort(model: Mapping[str, Any], xd: FloatArray, yd: FloatArray) -> tuple[FloatArray, FloatArray]:
    k1 = float(model.get("k1") or 0.0)
    if not k1:
        return xd, yd
    x, y = xd, yd
    for _ in range(20):
        s = 1.0 + k1 * (x * x + y * y)
        x, y = xd / s, yd / s
    return x, y


def unproject(model: Mapping[str, Any], px: npt.ArrayLike, py: npt.ArrayLike, d: npt.ArrayLike) -> FloatArray:
    """Pixel(s) + depth value(s) -> camera-frame point(s) (N x 3)."""
    pxa, pya, da = (np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in (px, py, d))
    x, y = (pxa - float(model["cx"])) / float(model["f"]), (pya - float(model["cy"])) / float(model["f"])
    lens = model.get("lens", "pinhole")
    if lens != "equidistant":
        x, y = _undistort(model, x, y)
        ray = np.stack([x, y, np.ones_like(x)], 1)
    else:
        th = np.hypot(x, y)
        s = np.where(th > 1e-9, np.sin(th) / np.maximum(th, 1e-9), 1.0)
        ray = np.stack([x * s, y * s, np.cos(th)], 1)
    if model.get("depth", "z") == "range":
        ray = ray / np.linalg.norm(ray, axis=1, keepdims=True)
    else:
        ray = ray / ray[:, 2:3]
    return np.asarray(ray * da[:, None], dtype=np.float64)


def project(model: Mapping[str, Any], P: npt.ArrayLike) -> FloatArray:
    """Camera-frame point(s) -> pixels (N x 2); NaN behind the camera."""
    Pa = np.atleast_2d(np.asarray(P, dtype=np.float64))
    out = np.full((len(Pa), 2), np.nan)
    lens = model.get("lens", "pinhole")
    ok = Pa[:, 2] > 1e-6 if lens == "pinhole" else np.linalg.norm(Pa, axis=1) > 1e-9
    f, c = float(model["f"]), np.array([float(model["cx"]), float(model["cy"])])
    if lens == "equidistant":
        r = np.hypot(Pa[:, 0], Pa[:, 1])
        th = np.arctan2(r, Pa[:, 2])
        s = np.where(r > 1e-12, th / np.maximum(r, 1e-12), 1.0 / np.maximum(Pa[:, 2], 1e-12))
        out[ok] = np.c_[Pa[ok, 0] * s[ok], Pa[ok, 1] * s[ok]] * f + c
    else:
        xy = Pa[ok, :2] / Pa[ok, 2:3]
        k1 = float(model.get("k1") or 0.0)
        if k1:
            xy = xy * (1.0 + k1 * np.sum(xy * xy, axis=1, keepdims=True))
        out[ok] = xy * f + c
    return out


def ray_base(model: Mapping[str, Any], T_base_cam: FloatArray, px: float, py: float) -> tuple[FloatArray, FloatArray]:
    """The pixel's line of sight in `base`: (origin, unit direction)."""
    T = np.asarray(T_base_cam, dtype=np.float64)
    u = T[:3, :3] @ unproject({**model, "depth": "range"}, px, py, 1.0)[0]
    return T[:3, 3].copy(), u / float(np.linalg.norm(u))


def check_model(model: Mapping[str, Any]) -> str | None:
    """None when `model` is a usable camera model, else why not."""
    try:
        if model.get("lens", "pinhole") not in ("pinhole", "equidistant"):
            return f"lens {model.get('lens')!r}: pinhole or equidistant"
        if not (float(model["f"]) > 0 and math.isfinite(float(model["cx"])) and math.isfinite(float(model["cy"]))):
            return "f must be > 0, cx / cy finite"
        if model.get("depth", "z") not in ("z", "range"):
            return "depth: z or range"
    except (KeyError, TypeError, ValueError) as e:
        return f"camera model needs f, cx, cy (+ lens, depth, k1): {e}"
    return None


# ---------------------------------------------------------------- the declared supports (generic: the world model)
@dataclass(frozen=True)
class Support:
    """A horizontal face a pointed pixel can lie on: a Surface's top or a Container's floor (`inside` = the container's
    inner outline; its rim height is kept for the "inside a container" rule)."""

    id: str
    kind: str  # "surface" | "container"
    z: float  # the face's height in base (m)
    centre: tuple[float, float]
    yaw: float  # rad, the outline's axes
    half: tuple[float, float]  # half extents along the outline's axes (m)
    rim_z: float | None = None  # containers: the rim's height

    def local(self, x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx = np.asarray(x, dtype=np.float64) - self.centre[0]
        dy = np.asarray(y, dtype=np.float64) - self.centre[1]
        return c * dx + s * dy, -s * dx + c * dy

    def contains(self, x: npt.ArrayLike, y: npt.ArrayLike, grow: float = 0.0) -> npt.NDArray[np.bool_]:
        u, v = self.local(x, y)
        return np.asarray((np.abs(u) <= self.half[0] + grow) & (np.abs(v) <= self.half[1] + grow), dtype=np.bool_)


def _yaw_of_quat(q: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def supports_from_entities(entities: Iterable[Any]) -> list[Support]:
    """The supports of a world snapshot's entities (r1mc3.world: Surface pose = the top face's centre, size = (x, y,
    thickness); Container pose = the bottom face's centre, size_outer = (x, y, height), wall_m, floor_m). Tilted
    entities are taken by their yaw (pointing assumes horizontal faces, as v2's tables did)."""
    out: list[Support] = []
    for e in entities:
        kind = getattr(e, "kind", None)
        pos = tuple(float(v) for v in e.pose.pos)
        yaw = _yaw_of_quat(e.pose.quat)
        if kind == "surface":
            sx, sy, _ = (float(v) for v in e.size)
            out.append(Support(e.id, "surface", pos[2], (pos[0], pos[1]), yaw, (sx / 2, sy / 2)))
        elif kind == "container":
            sx, sy, sz = (float(v) for v in e.size_outer)
            w = float(e.wall_m)
            out.append(
                Support(e.id, "container", pos[2] + float(e.floor_m), (pos[0], pos[1]), yaw,
                        (max(1e-3, sx / 2 - w), max(1e-3, sy / 2 - w)), rim_z=pos[2] + sz)
            )  # fmt: skip
    return out


def top_at(supports: Sequence[Support], X: npt.ArrayLike, Y: npt.ArrayLike, grow: float = 0.0) -> FloatArray:
    """Vectorised: the highest support face under each (X, Y) (an outline grown by `grow`); NaN where none."""
    Xa, Ya = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    z = np.full(Xa.shape, np.nan)
    for s in supports:
        m = s.contains(Xa, Ya, grow)
        z = np.where(m & ~(z > s.z), s.z, z)  # (NaN > x is False: the first support fills, a higher one wins)
    return z


def ray_hit(
    supports: Sequence[Support], origin: FloatArray, ray: FloatArray, margin: float = 0.0
) -> tuple[Support, FloatArray] | None:
    """The support face the line of sight meets first (the highest face whose outline, shrunk by `margin`, holds the
    intersection), with the point; None when it meets none (a downward ray only)."""
    if ray[2] > -1e-6:
        return None
    for s in sorted(supports, key=lambda s: -s.z):
        t = (s.z - origin[2]) / ray[2]
        if t <= 0:
            continue
        p = origin + t * ray
        if bool(s.contains(p[0], p[1], -margin)):
            return s, p
    return None


# ---------------------------------------------------------------- depth helpers (v2 headcam.py)
def depth_at(depth: FloatArray, px: float, py: float, patch: int = 7) -> dict[str, Any]:
    """The depth (m) in a patch around the pixel: v2 HeadCam.depth_at's median - or, when the patch straddles an edge
    (its spread above EDGE_SPREAD_M: two surfaces), the median of the surface the clicked pixel itself shows (session
    13). `depth_m` = the one used, `depth_median_m` the patch's median, `depth_pick` "median" | "nearer of two" |
    "farther of two"."""
    h, w = depth.shape
    if not (0 <= px < w and 0 <= py < h):
        return {"ok": False, "error": f"pixel ({px:.0f}, {py:.0f}) is outside the head image ({w} x {h})"}
    u, v, r = round(px), round(py), patch // 2
    win = depth[max(0, v - r) : v + r + 1, max(0, u - r) : u + r + 1]
    ok = np.isfinite(win) & (win > DEPTH_MIN_M) & (win < DEPTH_MAX_M)
    if ok.mean() < 0.3:
        return {
            "ok": False,
            "error": "no depth reading at that pixel (shiny, dark or see-through surface, or outside the depth view?)",
            "valid_fraction": round(float(ok.mean()), 2),
        }
    vals = win[ok]
    med, spread = float(np.median(vals)), float(np.std(vals))
    use, pick = med, "median"
    if spread > EDGE_SPREAD_M:
        lo, hi = (float(v) for v in np.percentile(vals, [15.0, 85.0]))
        cut = 0.5 * (lo + hi)
        c = depth[max(0, v - 1) : v + 2, max(0, u - 1) : u + 2]
        c = c[np.isfinite(c) & (c > DEPTH_MIN_M) & (c < DEPTH_MAX_M)]
        centre = float(np.median(c)) if c.size else med
        near = centre <= cut
        side = vals[vals <= cut] if near else vals[vals > cut]
        if side.size:
            use, pick = float(np.median(side)), "nearer of two" if near else "farther of two"
    return {
        "ok": True,
        "depth_m": round(use, 4),
        "depth_median_m": round(med, 4),
        "depth_pick": pick,
        "valid_fraction": round(float(ok.mean()), 2),
        "spread_mm": round(1e3 * spread, 1),
    }


def object_height(hd: float, lvl: float | None) -> tuple[bool, float, str]:
    """(raised?, height above the surface m, rule) from the depth's height at the click `hd` and the surface around it
    `lvl` (None: no ring) - v2 headcam.object_height (7 Oct): relative to the surface around it first; the height used
    is halfway between the absolute and the relative reading (on saved real frames this halved the object errors)."""
    if lvl is None or not math.isfinite(lvl):
        return (hd > ON_SURFACE_M), (hd if hd > ON_SURFACE_M else 0.0), "absolute (no surface around to compare with)"
    rel = hd - lvl
    if rel > REL_RAISED_M:
        return True, (0.5 * (hd + rel) if abs(lvl) < ABS_RAISED_M else hd), "relative to the surface around it"
    if hd > ABS_RAISED_M:
        return True, hd, "absolute (a large object)"
    return False, 0.0, "on the surface (relative to the surface around it)"


def local_level(
    depth: FloatArray, model: Mapping[str, Any], T: FloatArray, px: float, py: float, z_ref: float
) -> float | None:
    """Height above `z_ref` (m) of "the surface around the click": the 30th percentile of the depth's heights in a ring
    RING_PX around it (objects in the ring stand higher) - v2 HeadCam._local_table_height. None without enough depth."""
    h_img, w_img = depth.shape
    r1, r2 = RING_PX
    u0, u1 = max(0, int(px) - r2), min(w_img, int(px) + r2 + 1)
    v0, v1 = max(0, int(py) - r2), min(h_img, int(py) + r2 + 1)
    v, u = np.mgrid[v0:v1, u0:u1]
    rr = np.hypot(u - px, v - py)
    d = depth[v0:v1, u0:u1]
    sel = (rr >= r1) & (rr <= r2) & np.isfinite(d) & (d > DEPTH_MIN_M) & (d < DEPTH_MAX_M)
    if sel.sum() < 30:
        return None
    P = unproject(model, u[sel].astype(float), v[sel].astype(float), d[sel]) @ T[:3, :3].T + T[:3, 3]
    return float(np.percentile(P[:, 2] - z_ref, 30))


# ---------------------------------------------------------------- point (v2 HeadCam.point, generic supports)
def point(
    depth: FloatArray | None,
    model: Mapping[str, Any],
    T_base_cam: FloatArray,
    px: float,
    py: float,
    supports: Sequence[Support],
    *,
    patch: int = 7,
    use_depth: bool = True,
) -> dict[str, Any]:
    """The pixel -> `base` point. With no declared support: the depth alone (v2's fallback). Otherwise the line of
    sight meets the support it looks at, raised by the depth's height of the clicked thing above it (v2's rules:
    relative to the surface around the click, glare inside the outline -> the surface, capped at 0.45 m).

    -> {ok, point_base, surface_id, surface_z, height_above_surface_m, via ("plane" | "plane+depth" | "depth"),
        height_from (words), surface_point_base (where the line of sight meets the face itself), ray_origin,
        ray_direction, depth_* diagnostics} or {ok: False, error, code_hint}."""
    T = np.asarray(T_base_cam, dtype=np.float64)
    h_img, w_img = (int(model.get("height", 448)), int(model.get("width", 544))) if depth is None else depth.shape
    if not (0 <= px < w_img and 0 <= py < h_img):
        return {"ok": False, "error": f"pixel ({px:.0f}, {py:.0f}) is outside the head image ({w_img} x {h_img})",
                "code_hint": "INVALID_ARGS"}  # fmt: skip
    o, ray = ray_base(model, T, px, py)
    dep: dict[str, Any] = {}
    pd: FloatArray | None = None
    if depth is not None and use_depth:
        d = depth_at(depth, px, py, patch)
        dep = {f"depth_{k}" if not k.startswith("depth") else k: d[k] for k in ("depth_m", "valid_fraction",
               "spread_mm", "depth_median_m", "depth_pick") if k in d}  # fmt: skip
        if d["ok"]:
            pd = (T @ np.r_[unproject(model, px, py, d["depth_m"])[0], 1.0])[:3]
    base = {"ray_origin": np.round(o, 4).tolist(), "ray_direction": np.round(ray, 5).tolist()}
    if not supports:  # nothing declared: depth alone (v2: "no table declared")
        if pd is None:
            why = "no declared surface and no depth at that pixel"
            return {"ok": False, "error": why, "code_hint": "UNKNOWN_FRAME", **dep, **base}
        return {"ok": True, "point_base": np.round(pd, 4).tolist(), "surface_id": None, "surface_z": None,
                "height_above_surface_m": None, "via": "depth", "height_from": "depth alone (no surface declared)",
                **dep, **base}  # fmt: skip
    hit = ray_hit(supports, o, ray)
    if hit is None:
        return {"ok": False, "error": "that pixel's line of sight meets no declared surface (off the table?)",
                "code_hint": "UNREACHABLE", **dep, **base}  # fmt: skip
    sup, p_s = hit
    h, height_from, via = 0.0, "on the surface", "plane"
    if depth is not None and use_depth and pd is None:
        if sup.kind != "container":
            # session 13: no usable depth at a click whose line of sight meets a declared surface, in a frame that HAS
            # depth -> refused in words (before: "taken as on the surface" = the line of sight's hit on the table /
            # board BEHIND whatever stands there; on day 2 a clear plastic object read the board and the hover went ~5
            # cm beyond it). `use_depth=False` (the click on an object's FOOT, where it meets its support) never needs
            # depth.
            return {"ok": False, "error": NO_DEPTH_WORDS, "code_hint": "NO_DEPTH", "surface_id": sup.id, **dep, **base}
        # a click INTO a container (its floor - v2's rule: never the rim or wall the depth sees) needs no depth: a
        # shiny metal container gives none, and refusing it would make every item undeliverable (the carry's click)
        height_from = CONTAINER_NO_DEPTH_LABEL
    if pd is not None:
        hd = float(pd[2] - sup.z)
        dep["depth_height_m"] = round(hd, 4)
        lvl = local_level(depth, model, T, px, py, sup.z) if depth is not None else None
        if lvl is not None:
            dep["local_surface_m"] = round(lvl, 4)
            dep["relative_height_m"] = round(hd - lvl, 4)
        below_inside = False
        if (hd - lvl if lvl is not None else hd) < BELOW_SURFACE_M:
            if ray_hit(supports, o, ray, margin=BELOW_INSIDE_MARGIN_M) is None:
                return {"ok": False, "error": "that point is below the surface (off the table?)",
                        "code_hint": "UNREACHABLE", **dep, **base}  # fmt: skip
            below_inside, height_from = True, BELOW_INSIDE_LABEL  # v2 (9 Oct, F9): glare - the surface hit is used
        raised, hz, rule = object_height(hd, lvl) if not below_inside else (False, 0.0, "below the surface")
        dep["height_rule"] = rule
        if raised:
            h, height_from, via = min(hz, MAX_HEIGHT_M), f"above the surface, height from depth ({rule})", "plane+depth"
    elif depth is None and use_depth:
        height_from = "no depth in that frame: taken as on the surface"
    elif not use_depth:
        height_from = "on the surface (asked without depth: where the line of sight meets it)"
    p = o + (sup.z + h - o[2]) / ray[2] * ray
    if np.isnan(top_at(supports, p[0], p[1], OUTLINE_MARGIN_M)):
        return {"ok": False, "error": "that point is not on a declared surface", "code_hint": "UNREACHABLE",
                **dep, **base}  # fmt: skip
    return {
        "ok": True,
        "point_base": np.round(p, 4).tolist(),
        "surface_id": sup.id,
        "surface_kind": sup.kind,
        "surface_z": round(sup.z, 4),
        "height_above_surface_m": round(h, 4),
        "via": via,
        "height_from": height_from,
        "surface_point_base": np.round(p_s, 4).tolist(),
        **dep,
        **base,
    }


def project_points(model: Mapping[str, Any], T_base_cam: FloatArray, points_base: npt.ArrayLike, width: int,
                   height: int) -> list[list[float] | None]:  # fmt: skip
    """Base points -> pixels of that frame (None when outside / behind) - v2 HeadCam.project."""
    P = np.atleast_2d(np.asarray(points_base, dtype=np.float64))
    Tinv = np.linalg.inv(np.asarray(T_base_cam, dtype=np.float64))
    Pc = (np.c_[P, np.ones(len(P))] @ Tinv.T)[:, :3]
    uv = project(model, Pc)
    return [
        None
        if not np.all(np.isfinite(q)) or not (0 <= q[0] < width and 0 <= q[1] < height)
        else [round(float(q[0]), 1), round(float(q[1]), 1)]
        for q in uv
    ]


# ---------------------------------------------------------------- object extent (v2 headcam.py:142-383)
EXT_WIN_M = 0.16  # the window round the click (half size, m at the click's depth)
EXT_RAISED_M = 0.012  # above the local surface: part of an object
EXT_STEP_M = 0.015  # neighbours further apart than this in 3-D: not the same surface
EXT_MAX_R_M = 0.12  # an object's points stay within this of the click (horizontally)
EXT_MAX_H_M = 0.30  # higher above the surface than this: not on the surface (an arm)
EXT_EDGE_M = 0.01  # a declared face counts this far round its outline (its edge face)
EXT_DOME_MIN = 0.12  # (top height at the middle - at 0.55-0.85 of the half width) / half width: round above
# session 13 (HW2 day 2: a low object measured 14.7 x 7.1 x 4.6 cm for 9.8 x 5.4 x 3.9, a round one 9.2 x 6.2 x 5.3 for
# 6 x 6 x 6.2 - the click -> auto rule then chose the wrong style). Two causes, both reproduced in the simulator with
# the fitted native-depth model (r1sim.depth, A6: the outline low-pass + the disparity field; ground truth gives the
# true sizes): (1) the surface under the window was taken as ONE level (the ring's median) while the depth's bias tilts
# across a 30-70 px window by ~1 cm - the size of EXT_RAISED_M - so a strip of the surface on the low side counted as
# "raised"; now a plane is fitted to the ring (robust: two passes); (2) the low-passed outline is a skirt of points
# BETWEEN the object's top and the surface behind it (every height from 0 up), which the 3-D connectivity keeps and
# which lies beyond the object along the line of sight; the footprint is now fitted on the points above
# EXT_FOOT_FRAC of the object's height (a box's top + its face projects onto the same outline; a dome's lower sides
# narrow it a little: the round object's body at 0.35 of its height is 0.96 of its widest)
EXT_PLANE_FIT = True
EXT_PLANE_MAX_SLOPE = 0.15  # m / m: a steeper fitted "surface" is not the table / board (the fit falls back to a level)
EXT_FOOT_FRAC = 0.35
EXT_MIRROR_NEAR_ONLY = True


def _hull(P: FloatArray) -> FloatArray:
    """Convex hull (monotone chain) of 2-D points, counter-clockwise."""
    Q: FloatArray = np.asarray(np.unique(np.round(np.asarray(P, dtype=np.float64), 4), axis=0), dtype=np.float64)
    if len(Q) < 3:
        return Q
    Q = Q[np.lexsort((Q[:, 1], Q[:, 0]))]

    def half(pts: Iterable[FloatArray]) -> list[FloatArray]:
        h: list[FloatArray] = []
        for q in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:  # fmt: skip
                h.pop()
            h.append(q)
        return h

    lo, up = half(Q), half(Q[::-1])
    return np.array(lo[:-1] + up[:-1])


def _ring_plane(X: FloatArray, Y: FloatArray, hs: FloatArray, ring: npt.NDArray[np.bool_]) -> FloatArray | None:
    """The surface's level under the window as a plane in (x, y) fitted to the ring's heights (two passes, dropping
    what lies > 1 cm off the first fit: an object's edge in the ring), clipped to +-3 cm like the level it replaces;
    None when the fit is not a table-like plane (too steep)."""
    xr, yr, hr = X[ring], Y[ring], hs[ring]
    keep = np.ones(len(hr), dtype=bool)
    coef = None
    for _ in range(2):
        A = np.c_[np.ones(int(keep.sum())), xr[keep] - xr.mean(), yr[keep] - yr.mean()]
        if len(A) < 30:
            return None
        coef, *_ = np.linalg.lstsq(A, hr[keep], rcond=None)
        res = hr - (coef[0] + coef[1] * (xr - xr.mean()) + coef[2] * (yr - yr.mean()))
        keep = np.abs(res) < 0.01
    assert coef is not None
    if math.hypot(float(coef[1]), float(coef[2])) > EXT_PLANE_MAX_SLOPE:
        return None
    pl = coef[0] + coef[1] * (X - xr.mean()) + coef[2] * (Y - yr.mean())
    return np.asarray(np.clip(pl, -0.03, 0.03), dtype=np.float64)


def object_extent(
    depth: FloatArray,
    model: Mapping[str, Any],
    T_base_cam: FloatArray,
    px: float,
    py: float,
    surface_z: Callable[[FloatArray, FloatArray, float], FloatArray] | None,
    exclude: Sequence[tuple[float, float, float, float]] = (),
) -> dict[str, Any]:
    """What stands at head pixel (px, py): footprint (oriented box, base frame), heights above its surface, roundness,
    a coarse class and a confidence - v2's object_extent_from_depth (8 Oct), the declared supports as `surface_z(X, Y,
    grow)` (NaN where none; None = nothing declared: the window's low part stands for it). `exclude`: base-frame balls
    (x, y, z, r) whose points are dropped (the robot's own hands)."""
    depth = np.asarray(depth, dtype=np.float64)
    T = np.asarray(T_base_cam, dtype=np.float64)
    h_img, w_img = depth.shape
    if not (0 <= px < w_img and 0 <= py < h_img):
        return {"ok": False, "error": f"pixel ({px:.0f}, {py:.0f}) is outside the head image ({w_img} x {h_img})"}
    notes: list[str] = []
    u0, v0 = round(px), round(py)
    pr = 5
    pat = depth[max(0, v0 - pr) : v0 + pr + 1, max(0, u0 - pr) : u0 + pr + 1]
    okp = np.isfinite(pat) & (pat > DEPTH_MIN_M) & (pat < DEPTH_MAX_M)
    if okp.any():
        d0 = float(np.median(pat[okp]))
    else:
        allv = np.where((depth > DEPTH_MIN_M) & (depth < DEPTH_MAX_M), depth, np.nan)
        d0 = float(np.nanmedian(allv)) if np.isfinite(allv).any() else float("nan")
    if not math.isfinite(d0):
        return {"ok": False, "error": "no depth around that pixel"}
    R = int(np.clip(float(model["f"]) * EXT_WIN_M / d0, 20, 220))
    ua, ub, va, vb = max(0, u0 - R), min(w_img, u0 + R + 1), max(0, v0 - R), min(h_img, v0 + R + 1)
    v, u = np.mgrid[va:vb, ua:ub]
    d = depth[va:vb, ua:ub].copy()
    valid = np.isfinite(d) & (d > DEPTH_MIN_M) & (d < DEPTH_MAX_M)
    d = np.where(valid, d, 1.0)
    P = unproject(model, u.ravel().astype(float), v.ravel().astype(float), d.ravel()) @ T[:3, :3].T + T[:3, 3]
    P = P.reshape((*d.shape, 3))
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    zs: FloatArray
    if surface_z is not None:
        zs = np.asarray(surface_z(X, Y, EXT_EDGE_M), dtype=np.float64)
        has_support = bool(np.isfinite(zs[valid]).any())
        if has_support:
            fill = float(np.nanmin(zs[np.isfinite(zs)]))
            zs = np.where(np.isfinite(zs), zs, fill)  # beyond every outline: the lowest declared face in view
        else:
            notes.append("no declared surface in view: heights from the window's lowest part")
    else:
        has_support = False
        notes.append("no surface declared: heights from the window's lowest part")
    if not has_support:
        zs = np.full(d.shape, float(np.percentile(Z[valid], 20))) if valid.sum() > 30 else np.zeros(d.shape)
    hs = Z - zs
    rr = np.hypot(u - px, v - py)
    ring = valid & (rr > 0.6 * R) & (np.abs(hs) < 0.03)
    bias = float(np.clip(np.median(hs[ring]), -0.03, 0.03)) if ring.sum() > 30 else 0.0
    plane = _ring_plane(X, Y, hs, ring) if EXT_PLANE_FIT and ring.sum() > 60 else None
    if plane is not None:
        hs = hs - plane
        bias = float(np.clip(np.median(plane[ring]), -0.03, 0.03))
    else:
        hs = hs - bias
    keep = valid & (hs < EXT_MAX_H_M)
    for ex, ey, ez, er in exclude:
        keep &= np.sqrt((X - ex) ** 2 + (Y - ey) ** 2 + (Z - ez) ** 2) > er
    raised = keep & (hs > EXT_RAISED_M)
    cu, cv = u0 - ua, v0 - va
    seed_on_click = bool(raised[cv, cu])
    if not seed_on_click:
        near = raised & (rr <= 12)
        if not near.any():
            why = "nothing stands above the surface there (a flat thing, or the surface itself)"
            return {"ok": False, "error": why, "flat_or_empty": True}
        k = int(np.argmin(np.where(near, rr, np.inf)))
        cv, cu = (int(x) for x in np.unravel_index(k, rr.shape))
        notes.append("the click is beside the object: measured from its nearest raised part")
    p_seed = P[cv, cu]
    raised &= np.hypot(X - p_seed[0], Y - p_seed[1]) <= EXT_MAX_R_M
    comp = np.zeros_like(raised)
    comp[cv, cu] = True
    edges = []
    for dv, du in ((0, 1), (1, 0)):
        a = (slice(0, raised.shape[0] - dv), slice(0, raised.shape[1] - du))
        b = (slice(dv, None), slice(du, None))
        okn = raised[a] & raised[b] & (np.linalg.norm(P[a] - P[b], axis=-1) < EXT_STEP_M)
        edges.append((a, b, okn))
    for _ in range(4 * R):
        prev = comp.sum()
        for a, b, okn in edges:
            comp[b] |= comp[a] & okn
            comp[a] |= comp[b] & okn
        if comp.sum() == prev:
            break
    n = int(comp.sum())
    if n < 12:
        return {"ok": False, "error": "too little of it in the depth (a thin or see-through thing?)", "n_points": n}
    border = np.zeros_like(comp)
    occ = np.zeros_like(comp)
    dep = np.where(valid, depth[va:vb, ua:ub], 0.0)
    for dv, du in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        ins = comp & ~np.roll(comp, (-dv, -du), axis=(0, 1))
        border |= ins
        nd = np.roll(dep, (-dv, -du), axis=(0, 1))
        nval = np.roll(valid & keep, (-dv, -du), axis=(0, 1))
        occ |= ins & (~nval | (nd < dep - 0.02))
    occ_frac = float(occ.sum() / max(1, border.sum()))
    truncated = bool(
        (comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any())
        or (comp & (np.hypot(X - p_seed[0], Y - p_seed[1]) > EXT_MAX_R_M - 0.01)).any()
    )
    hmax = float(np.percentile(hs[comp], 98))
    foot = comp & (hs > max(EXT_RAISED_M, EXT_FOOT_FRAC * hmax))
    if foot.sum() >= 12:
        comp = foot
    xy = np.c_[X[comp], Y[comp]]
    hh = hs[comp]
    top = hh > 0.85 * hmax
    c_top = xy[top].mean(0) if top.sum() >= 5 else xy.mean(0)
    e_v = c_top - T[:2, 3]
    e_v = e_v / max(1e-9, float(np.linalg.norm(e_v)))
    if top.sum() >= 5:
        e_p = np.array([-e_v[1], e_v[0]])
        a_v, a_p = xy[top] @ e_v, xy[top] @ e_p
        c_top = e_v * 0.5 * (a_v.min() + a_v.max()) + e_p * 0.5 * (a_p.min() + a_p.max())
    nearh = (xy - c_top) @ e_v < 0
    if EXT_MIRROR_NEAR_ONLY and top.sum() >= 5 and nearh.sum() >= 12:
        # session 13: the far half's points include the outline's low-passed skirt (beyond the object along the line
        # of sight); the near half (the front face + the top's near part: no depth step there) mirrored about the
        # top's centre stands for the far half too
        xy_fit = np.vstack([xy[nearh], 2 * c_top - xy[nearh], xy[top]])
    else:
        xy_fit = np.vstack([xy, 2 * c_top - xy[nearh]])
    best: tuple[float, float, float, float, float, float] | None = None
    for ang in np.radians(np.arange(0.0, 180.0, 2.0)):
        e = np.array([np.cos(ang), np.sin(ang)])
        s1, s2 = xy_fit @ e, xy_fit @ np.array([-e[1], e[0]])
        l1, h1 = np.percentile(s1, [0.5, 99.5])
        l2, h2 = np.percentile(s2, [0.5, 99.5])
        A = float((h1 - l1) * (h2 - l2))
        if best is None or A < best[0]:
            best = (A, float(ang), float(l1), float(h1), float(l2), float(h2))
    assert best is not None
    _, a_, l1, h1, l2, h2 = best
    if (h2 - l2) > (h1 - l1):  # the length axis is the longer side
        a_ = a_ + math.pi / 2
    a_ = float((a_ + math.pi / 2) % math.pi - math.pi / 2)  # yaw of the length axis in [-90, 90) deg
    e1 = np.array([math.cos(a_), math.sin(a_)])
    e2 = np.array([-e1[1], e1[0]])
    f1, f2 = xy_fit @ e1, xy_fit @ e2
    l1, h1 = (float(x) for x in np.percentile(f1, [0.5, 99.5]))
    l2, h2 = (float(x) for x in np.percentile(f2, [0.5, 99.5]))
    L, W = h1 - l1, h2 - l2
    ctr = e1 * 0.5 * (l1 + h1) + e2 * 0.5 * (l2 + h2)
    inner = (f1 >= l1 - 1e-4) & (f1 <= h1 + 1e-4) & (f2 >= l2 - 1e-4) & (f2 <= h2 + 1e-4)
    H = _hull(xy_fit[inner])
    if len(H) > 24:
        H = H[np.linspace(0, len(H) - 1, 24).astype(int)]
    s1, s2 = xy @ e1, xy @ e2
    q1 = (s1 - 0.5 * (l1 + h1)) / max(1e-3, 0.5 * L)
    q2 = (s2 - 0.5 * (l2 + h2)) / max(1e-3, 0.5 * W)
    band = np.abs(q1) < 0.5
    mid, side_ = band & (np.abs(q2) < 0.25), band & (np.abs(q2) > 0.55) & (np.abs(q2) < 0.85)
    dome = None
    if mid.sum() >= 4 and side_.sum() >= 4:
        dome = float((np.percentile(hh[mid], 90) - np.median(hh[side_])) / max(1e-3, 0.5 * W))
    top_c = float(np.median(hh[mid])) if mid.sum() >= 4 else None
    aspect = W / max(1e-3, L)
    if hmax < 0.035 or hmax < 0.35 * W:
        shape = "flat"
    elif hmax > 1.4 * L and hmax > 0.10:
        shape = "tall"
    elif dome is not None and dome > EXT_DOME_MIN:
        shape = "round"
    else:
        shape = "box"
    conf = min(1.0, n / 150.0) * (1.0 - 0.6 * occ_frac)
    if truncated:
        conf *= 0.4
        notes.append(
            "it runs into something bigger or leaves the window (touching a container?): the size may be too big"
        )
    if occ_frac > 0.25:
        notes.append("partly hidden (by the hand or missing depth): the size may be too small")
    if not seed_on_click:
        conf *= 0.7
    if dome is None:
        conf *= 0.8
    if float(valid.mean()) < 0.6:
        conf *= 0.7
        notes.append("patchy depth around it")
    sz = float(np.median(zs[comp])) + bias
    return {
        "ok": True,
        "centre_base": np.round(ctr, 4).tolist(),
        "surface_z": round(sz, 4),
        "surface": "a declared surface" if has_support else "the window's lowest part",
        "length_m": round(L, 4),
        "width_m": round(W, 4),
        "yaw_rad": round(a_, 4),
        "yaw_deg": round(math.degrees(a_), 1),
        "max_height_m": round(hmax, 4),
        "top_height_m": None if top_c is None else round(top_c, 4),
        "size_cm": [round(100 * L, 1), round(100 * W, 1), round(100 * hmax, 1)],
        "aspect": round(aspect, 3),
        "dome": None if dome is None else round(dome, 3),
        "shape": shape,
        "confidence": round(float(conf), 2),
        "n_points": n,
        "occluded_fraction": round(occ_frac, 3),
        "truncated": truncated,
        "depth_bias_m": round(bias, 4),
        "hull_xy": np.round(H, 4).tolist(),
        "notes": notes,
    }
