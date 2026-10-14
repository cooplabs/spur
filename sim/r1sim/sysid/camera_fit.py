"""The head-camera fits for B5 (DESIGN §13.2; R1CAM_PLAN.md §3), as tools run on the Mac after HW2's S11 / S12 captures,
each with a synthetic check (the HW1_FITS pattern: the fit must recover a known model first).

* `lens` - S12: an r1cam charuco recording (MCAP, /r1cam/left JPEGs) or detections -> per lens model (pinhole + k1 =
  v2's, pinhole + k1 k2, equidistant): the full-field reprojection RMS (B5: <= 1 px), the leave-one-pose-out RMS, BIC;
  f / cx / cy / the distortion terms.
* `vignetting` - S11: flat-field frames (a white sheet filling the view) -> gain(r) = 1 + a2 x^2 + a4 x^4 + a6 x^6
  (models/head_camera/vignetting.yaml's form).
* `depth` - S12 step 3: depth frames of the board held flat at tape-measured distances -> the valid fraction, the median
  error vs the tape, sigma(z) of the plane-fit residuals per distance band (B5 / A6: within +-20 %).
* `latency` - a head-yaw step while the board is in view: the board's image x per frame (receive times) vs the head yaw
  (r1log) -> the exposure -> receive latency (ms) by a least-squares lag scan (r1d's `latency_s`).
* `stereo` - S12 with the right stream: the same frames' left + right detections -> the right camera's intrinsics +
  the left-right extrinsics (v2's stereo.py `right` calibration: its T_site_cam = the left's @ T_left_right).

    bash scripts/run.sh python -m r1sim.sysid.camera_fit lens --mcap REC.mcap [--board poster_6x4_90]
    bash scripts/run.sh python -m r1sim.sysid.camera_fit vignetting --mcap FLAT.r1cam.mcap          # S11 (C2)
    bash scripts/run.sh python -m r1sim.sysid.camera_fit depth --mcap A.r1cam.mcap [B ...]        # S12 step 3 (C4)
    bash scripts/run.sh python -m r1sim.sysid.camera_fit latency --mcap REC.r1cam.mcap --r1log LOG.mcap [...] (C5)
    bash scripts/run.sh python -m r1sim.sysid.camera_fit synthetic        # every fit's synthetic check

Session 11: `vignetting` / `depth` / `latency` read r1cam's recordings (`r1 record run flat_field` / `depth_targets` /
any recording with the board in view during a head-yaw step) and r1log's file (the head joint), so the HW2 runbook's
C2 / C4 / C5 each have one command; each has a synthetic check that goes through the SAME MCAP readers.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
W, H = 544, 448
LENS_MODELS = ("pinhole_k1", "pinhole_k1k2", "equidistant")
N_PARAMS = {"pinhole_k1": 4, "pinhole_k1k2": 5, "equidistant": 4}  # f (aspect fixed), cx, cy, distortion terms


# ---------------------------------------------------------------- detections
def detections_from_mcap(
    path: str | Path, board: Any, topic: str = "/r1cam/left"
) -> list[tuple[FloatArray, FloatArray]]:
    """(object points m (N x 3), pixels (N x 2)) per frame of an r1cam recording with >= 6 corners."""
    from mcap.reader import make_reader
    from PIL import Image

    from r1sim import charuco_render as cr

    C = np.c_[cr.corners_mm(board) * cr.MM, np.zeros(board.n_corners)]
    out: list[tuple[FloatArray, FloatArray]] = []
    with Path(path).open("rb") as f:
        for _s, _ch, msg in make_reader(f).iter_messages(topics=[topic]):
            gray = np.asarray(Image.open(io.BytesIO(msg.data)).convert("L"))
            ids, px = cr.detect(gray, board)
            if len(ids) >= cr.MIN_CORNERS_FOR_FIT:
                out.append((C[ids].astype(np.float64), px.astype(np.float64)))
    return out


# ---------------------------------------------------------------- lens
VIEW_MIN_CORNERS = 8  # session 12: fewer corners than this in a view -> not used for the lens fit
VIEW_MIN_SPREAD_PX = 5.0  # ... nor a view whose corners lie along a line (the minor axis' std of the pixels, px)
K0_F = 150.0  # the fit's starting focal length (px; the image centre as the starting principal point)


def view_usable(px: FloatArray) -> bool:
    """A view the lens fit can use: >= VIEW_MIN_CORNERS corners NOT along a line. Why (HW2 day 1's C3): ONE view of 6
    corners in a 77 x 22 px strip (minor-axis std 4.2 px) made OpenCV's calibrateCamera return its STARTING guess
    (f 150.0, the image centre, RMS 1.14 px over 1935 views) - its per-view initial pose from a homography is
    garbage for collinear points and the Levenberg-Marquardt never left the start. Without that view the same
    recording converges to f ~125 px, RMS ~0.34 px, from any start."""
    p = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    if len(p) < VIEW_MIN_CORNERS:
        return False
    sv = np.linalg.svd(p - p.mean(axis=0), compute_uv=False)
    return bool(sv[-1] / math.sqrt(len(p)) >= VIEW_MIN_SPREAD_PX)


def _calibrate(
    views: Sequence[tuple[FloatArray, FloatArray]], kind: str, size: tuple[int, int] = (W, H)
) -> tuple[float, FloatArray, FloatArray, list[Any], list[Any]]:
    import cv2

    obj = [np.asarray(o, np.float32).reshape(-1, 1, 3) for o, _ in views]
    img = [np.asarray(p, np.float32).reshape(-1, 1, 2) for _, p in views]
    K0 = np.array([[K0_F, 0, size[0] / 2], [0, K0_F, size[1] / 2], [0, 0, 1]])
    if kind == "equidistant":
        return _calibrate_equidistant(views, size)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if kind == "pinhole_k1":
        flags |= cv2.CALIB_FIX_K2
    rms, K, D, rv, tv = cv2.calibrateCamera(obj, img, size, K0, np.zeros(5), flags=flags)
    return float(rms), np.asarray(K, dtype=np.float64), np.asarray(D, dtype=np.float64).ravel(), list(rv), list(tv)


def _project(kind: str, K: FloatArray, D: FloatArray, obj: FloatArray, rv: Any, tv: Any) -> FloatArray:
    import cv2

    if kind == "equidistant":
        return _eq_project(K, obj, np.asarray(rv, np.float64).ravel(), np.asarray(tv, np.float64).ravel())
    p, _ = cv2.projectPoints(np.asarray(obj, np.float64), rv, tv, K, D)
    return np.asarray(p, dtype=np.float64).reshape(-1, 2)


def _eq_project(K: FloatArray, obj: FloatArray, rv: FloatArray, tv: FloatArray) -> FloatArray:
    """v2's equidistant model (no distortion terms): r = f theta."""
    import cv2
    from r1mc3.cam.pointing import project

    R, _ = cv2.Rodrigues(rv)
    Pc = np.asarray(obj, np.float64) @ R.T + tv
    return project({"lens": "equidistant", "f": float(K[0, 0]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}, Pc)


def _eq_pose(K: FloatArray, obj: FloatArray, px: FloatArray) -> tuple[FloatArray, FloatArray] | None:
    import cv2
    from r1mc3.cam.pointing import unproject

    m = {"lens": "equidistant", "f": float(K[0, 0]), "cx": float(K[0, 2]), "cy": float(K[1, 2]), "depth": "z"}
    with np.errstate(all="ignore"):
        und = unproject(m, px[:, 0], px[:, 1], np.ones(len(px)))[:, :2]  # tan-space (pinhole normalised) coordinates
    good = np.isfinite(und).all(axis=1) & (np.hypot(und[:, 0], und[:, 1]) < 8.0)  # within ~83 deg of the axis
    if good.sum() < 6:
        return None
    ok, rv, tv = cv2.solvePnP(np.asarray(obj, np.float64)[good], und[good].reshape(-1, 1, 2), np.eye(3), None)
    return (np.asarray(rv, dtype=np.float64).ravel(), np.asarray(tv, dtype=np.float64).ravel()) if ok else None


def _calibrate_equidistant(
    views: Sequence[tuple[FloatArray, FloatArray]], size: tuple[int, int]
) -> tuple[float, FloatArray, FloatArray, list[Any], list[Any]]:
    """v2's equidistant lens (f, cx, cy; no distortion terms) + every view's pose by Levenberg-Marquardt with the
    Schur complement (OpenCV's fisheye.calibrate cannot initialise some planar views; the poses start from PnP)."""
    K = np.array([[140.0, 0, size[0] / 2], [0, 140.0, size[1] / 2], [0, 0, 1]])
    init = [(v, _eq_pose(K, *v)) for v in views]
    data = [(o, p) for (o, p), ps in init if ps is not None]
    P = [np.r_[ps[0], ps[1]] for _v, ps in init if ps is not None]
    c = np.array([K[0, 0], K[0, 2], K[1, 2]])

    def kmat(cv: FloatArray) -> FloatArray:
        return np.array([[cv[0], 0, cv[1]], [0, cv[0], cv[2]], [0, 0, 1]])

    def view_res(cv: FloatArray, pv: FloatArray, o: FloatArray, p: FloatArray) -> FloatArray:
        r = (_eq_project(kmat(cv), o, pv[:3], pv[3:]) - p).ravel()
        return np.nan_to_num(r, nan=1e3)

    def total(cv: FloatArray, Ps: list[FloatArray]) -> float:
        return float(sum(float(np.sum(view_res(cv, pv, o, p) ** 2)) for pv, (o, p) in zip(Ps, data, strict=True)))

    lam, cost = 1e-3, total(c, P)
    for _ in range(60):
        A = np.zeros((3, 3))
        b = np.zeros(3)
        blocks = []
        for pv, (o, p) in zip(P, data, strict=True):
            r = view_res(c, pv, o, p)
            Jc = np.empty((r.size, 3))
            for k in range(3):
                d = np.zeros(3)
                d[k] = 1e-4
                Jc[:, k] = (view_res(c + d, pv, o, p) - r) / 1e-4
            Jp = np.empty((r.size, 6))
            for k in range(6):
                d = np.zeros(6)
                d[k] = 1e-6
                Jp[:, k] = (view_res(c, pv + d, o, p) - r) / 1e-6
            U = Jp.T @ Jp
            U = U + lam * np.diag(np.diag(U) + 1e-9)
            Ui = np.linalg.inv(U)
            W_ = Jc.T @ Jp
            A += Jc.T @ Jc - W_ @ Ui @ W_.T
            b += Jc.T @ r - W_ @ Ui @ (Jp.T @ r)
            blocks.append((Ui, W_, Jp, r))
        A = A + lam * np.diag(np.diag(A) + 1e-9)
        dc = -np.linalg.solve(A, b)
        P_new = [pv - Ui @ (Jp.T @ r + W_.T @ dc) for pv, (Ui, W_, Jp, r) in zip(P, blocks, strict=True)]
        c_new = c + dc
        cost_new = total(c_new, P_new)
        if cost_new < cost:
            done = (cost - cost_new) < 1e-10 * max(cost, 1e-12)
            c, P, cost, lam = c_new, P_new, cost_new, max(lam / 10, 1e-12)
            if done:
                break
        else:
            lam *= 10
            if lam > 1e10:
                break
    n = sum(len(p) for _o, p in data)
    rms = math.sqrt(cost / max(n, 1))
    return rms, kmat(c), np.zeros(4), [pv[:3] for pv in P], [pv[3:] for pv in P]


def _pose(kind: str, K: FloatArray, D: FloatArray, obj: FloatArray, px: FloatArray) -> tuple[Any, Any] | None:
    import cv2

    if kind == "equidistant":
        return _eq_pose(K, obj, px)
    ok, rv, tv = cv2.solvePnP(np.asarray(obj, np.float64), np.asarray(px, np.float64), K, D)
    return (rv, tv) if ok else None


def fit_lens(views: Sequence[tuple[FloatArray, FloatArray]], kinds: Iterable[str] = LENS_MODELS,
             loo: bool = True) -> dict[str, Any]:  # fmt: skip
    """Every lens model on the same detections: the full-field RMS, the leave-one-pose-out RMS (refit without the
    pose, its corners predicted from its own PnP pose under the left-out intrinsics), BIC; the best by BIC.
    Session 12: degenerate views are dropped first (`view_usable`; `views_dropped`), and a fit that ends AT its
    starting guess is marked `converged: false` (never a B5 pass)."""
    n_in = len(views)
    views = [v for v in views if view_usable(v[1])]
    n_pts = sum(len(p) for _, p in views)
    out: dict[str, Any] = {"views": len(views), "views_dropped": n_in - len(views), "corners": n_pts, "models": {}}
    for kind in kinds:
        rms, K, D, _rv, _tv = _calibrate(views, kind)
        at_start = (
            abs(float(K[0, 0]) - K0_F) < 0.1 and abs(float(K[0, 2]) - W / 2) < 0.1 and abs(float(K[1, 2]) - H / 2) < 0.1
        )
        row: dict[str, Any] = {
            "converged": not at_start,
            "rms_px": round(rms, 4),
            "f": round(float(K[0, 0]), 3),
            "cx": round(float(K[0, 2]), 3),
            "cy": round(float(K[1, 2]), 3),
            "dist": [round(float(x), 6) for x in D[: 2 if kind == "pinhole_k1k2" else 1]],
        }
        rss = (rms**2) * 2 * n_pts
        k = N_PARAMS[kind] + 6 * len(views)
        row["bic"] = round(2 * n_pts * math.log(max(rss, 1e-12) / (2 * n_pts)) + k * math.log(2 * n_pts), 2)
        if loo and len(views) >= 6:
            errs = []
            for i in range(len(views)):
                rest = [v for j, v in enumerate(views) if j != i]
                _r, Ki, Di, _a, _b = _calibrate(rest, kind)
                pose = _pose(kind, Ki, Di, *views[i])
                if pose is None:
                    continue
                e = _project(kind, Ki, Di, views[i][0], *pose) - views[i][1]
                errs.append(float(np.sqrt(np.mean(np.sum(e**2, axis=1)))))
            row["loo_rms_px_median"] = round(float(np.median(errs)), 4) if errs else None
        out["models"][kind] = row
    best = min(out["models"], key=lambda k: out["models"][k]["bic"])
    out["best"] = best
    out["b5_pass"] = bool(out["models"][best]["rms_px"] <= 1.0 and out["models"][best].get("converged", True))
    return out


def synthetic_views(
    kind: str = "pinhole_k1", *, f: float = 126.94, cx: float = 259.0, cy: float = 203.9, k1: float = 0.0249,
    noise_px: float = 0.2, seed: int = 0,
) -> list[tuple[FloatArray, FloatArray]]:  # fmt: skip
    """Synthetic S12 detections: the chosen board at the 3 distances x 9 places x 3 tilts, projected with a known lens
    (v2's model dict) + pixel noise - the fit's check without rendering."""
    from r1mc3.cam.pointing import project, unproject

    from r1sim import charuco_render as cr

    b = cr.CHOSEN
    rng = np.random.default_rng(seed)
    model = {"lens": "pinhole" if kind.startswith("pinhole") else "equidistant", "f": f, "cx": cx, "cy": cy,
             "k1": k1 if kind.startswith("pinhole") else 0.0, "depth": "z"}  # fmt: skip
    C = np.c_[cr.corners_mm(b) * cr.MM, np.zeros(b.n_corners)]
    views = []
    for z in cr.DISTANCES_M:
        for u, v in cr.PLACES:
            for _n, ax, deg in cr.TILTS:
                ray = unproject(model, [u * (W - 1)], [v * (H - 1)], [1.0])[0]
                centre = ray / np.linalg.norm(ray) * z
                R = cr._rot(ax, deg)
                Pc = (R @ (C - np.array([b.width_mm * cr.MM / 2, b.height_mm * cr.MM / 2, 0.0])).T).T + centre
                px = project(model, Pc)
                ok = np.isfinite(px).all(axis=1) & (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
                if ok.sum() >= 6:
                    views.append((C[ok], px[ok] + rng.normal(0.0, noise_px, (int(ok.sum()), 2))))
    return views


# ---------------------------------------------------------------- stereo (S12, both images)
def fit_stereo(pairs: Sequence[tuple[FloatArray, FloatArray, FloatArray]], kind: str = "pinhole_k1") -> dict[str, Any]:
    """`pairs` = (object points, left pixels, right pixels) of the corners both images found in the same frame: each
    camera's lens (fit_lens's model `kind`), then the right camera in the left one's frame (cv2.stereoCalibrate with
    the intrinsics fixed) -> T_left_right (4 x 4: a right-camera point -> the left frame), the baseline, RMS."""
    import cv2

    rl, Kl, Dl, _a, _b = _calibrate([(o, pl) for o, pl, _ in pairs], kind)
    rr, Kr, Dr, _c, _d = _calibrate([(o, pr) for o, _, pr in pairs], kind)
    obj = [np.asarray(o, np.float32).reshape(-1, 1, 3) for o, _, _ in pairs]
    il = [np.asarray(pl, np.float32).reshape(-1, 1, 2) for _, pl, _ in pairs]
    ir = [np.asarray(pr, np.float32).reshape(-1, 1, 2) for _, _, pr in pairs]
    rms, *_k, R, T, _E, _F = cv2.stereoCalibrate(obj, il, ir, Kl, Dl, Kr, Dr, (W, H), flags=cv2.CALIB_FIX_INTRINSIC)
    T_rl = np.eye(4)  # stereoCalibrate: x_right = R x_left + T
    T_rl[:3, :3], T_rl[:3, 3] = np.asarray(R), np.asarray(T).ravel()
    T_lr = np.linalg.inv(T_rl)
    return {
        "rms_px": round(float(rms), 4),
        "left": {"rms_px": round(rl, 4), "f": round(float(Kl[0, 0]), 3)},
        "right": {
            "rms_px": round(rr, 4),
            "f": round(float(Kr[0, 0]), 3),
            "cx": round(float(Kr[0, 2]), 3),
            "cy": round(float(Kr[1, 2]), 3),
        },
        "T_left_right": np.round(T_lr, 6).tolist(),
        "baseline_m": round(float(np.linalg.norm(T_lr[:3, 3])), 5),
    }


def synthetic_pairs(
    baseline_m: float = 0.06, noise_px: float = 0.2, seed: int = 0
) -> list[tuple[FloatArray, FloatArray, FloatArray]]:
    """Left views of synthetic_views + the same board corners seen by a right camera `baseline_m` to the right (x)."""
    from r1mc3.cam.pointing import project

    rng = np.random.default_rng(seed)
    left = synthetic_views("pinhole_k1", noise_px=0.0, seed=seed)
    model = {"lens": "pinhole", "f": 126.94, "cx": 259.0, "cy": 203.9, "k1": 0.0249, "depth": "z"}
    import cv2

    out = []
    for o, pl in left:
        ok, rv, tv = cv2.solvePnP(o, pl, np.array([[126.94, 0, 259.0], [0, 126.94, 203.9], [0, 0, 1]]),
                                  np.array([0.0249, 0, 0, 0, 0]))  # fmt: skip
        R, _ = cv2.Rodrigues(rv)
        Pc = o @ R.T + np.asarray(tv).ravel() - np.array([baseline_m, 0.0, 0.0])  # the right camera's frame
        pr = project(model, Pc)
        good = np.isfinite(pr).all(axis=1) & (pr[:, 0] >= 0) & (pr[:, 0] < W) & (pr[:, 1] >= 0) & (pr[:, 1] < H)
        if ok and good.sum() >= 6:
            n = int(good.sum())
            out.append(
                (o[good], pl[good] + rng.normal(0, noise_px, (n, 2)), pr[good] + rng.normal(0, noise_px, (n, 2)))
            )
    return out


# ---------------------------------------------------------------- vignetting (S11)
def fit_vignetting(frames: Sequence[npt.NDArray[Any]], cx: float, cy: float, r_max: float) -> dict[str, Any]:
    """gain(x) = 1 + a2 x^2 + a4 x^4 + a6 x^6 (x = r / r_max) from flat-field frames (grey or RGB, the mean of the
    frames), normalised to the centre (x < 0.25)."""
    img = np.mean([np.asarray(f, np.float64).mean(axis=2) if np.ndim(f) == 3 else np.asarray(f, np.float64)
                   for f in frames], axis=0)  # fmt: skip
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]].astype(np.float64)
    x = np.hypot(xx - cx, yy - cy) / r_max
    centre = float(np.median(img[x < 0.25]))
    g = img / max(centre, 1e-9)
    m = (x < 1.0) & (img > 2.0)
    A = np.c_[x[m] ** 2, x[m] ** 4, x[m] ** 6]
    coef, *_ = np.linalg.lstsq(A, g[m] - 1.0, rcond=None)
    resid = g[m] - 1.0 - A @ coef
    return {"a2": round(float(coef[0]), 4), "a4": round(float(coef[1]), 4), "a6": round(float(coef[2]), 4),
            "r_max_px": r_max, "rms": round(float(np.sqrt(np.mean(resid**2))), 4), "frames": len(frames)}  # fmt: skip


# ---------------------------------------------------------------- depth (S12 step 3)
def depth_residuals(
    depth_m: FloatArray, truth_m: float, bands: Sequence[float] = (0.3, 0.45, 0.6, 0.8)
) -> dict[str, Any]:
    """A depth frame of a flat board at a tape-measured distance: the valid fraction, the median error vs the tape and
    sigma(z) of the plane-fit residuals - per frame; the band is the nearest of `bands` (B5 / A6: within +-20 % of the
    depth model's per-band numbers)."""
    d = np.asarray(depth_m, np.float64)
    v = np.isfinite(d) & (d > 0.1) & (d < 3.0)
    yy, xx = np.mgrid[0 : d.shape[0], 0 : d.shape[1]].astype(np.float64)
    out: dict[str, Any] = {
        "band_m": min(bands, key=lambda b: abs(b - truth_m)),
        "valid_fraction": round(float(v.mean()), 3),
    }
    if v.sum() < 50:
        return {**out, "error": "too little depth"}
    A = np.c_[xx[v], yy[v], np.ones(int(v.sum()))]
    c, *_ = np.linalg.lstsq(A, d[v], rcond=None)
    res = d[v] - A @ c
    out.update(median_err_m=round(float(np.median(d[v]) - truth_m), 4), sigma_z_m=round(float(np.std(res)), 4))
    return out


# ---------------------------------------------------------------- latency
def latency_ms(t_img_s: FloatArray, img_x: FloatArray, t_joint_s: FloatArray, yaw: FloatArray,
               max_lag_s: float = 0.3, step_s: float = 0.001) -> dict[str, Any]:  # fmt: skip
    """The camera's latency: the board's image x (receive times) lags the head yaw (r1log) by L; L = the shift that
    best fits img_x(t) ~ a * yaw(t - L) + b (a least-squares scan; the image's own sign / scale fitted)."""
    t_img_s, img_x, t_joint_s, yaw = (np.asarray(v, np.float64) for v in (t_img_s, img_x, t_joint_s, yaw))
    best = (math.inf, 0.0, 0.0)
    for lag in np.arange(0.0, max_lag_s + 1e-12, step_s):
        y = np.interp(t_img_s - lag, t_joint_s, yaw)
        A = np.c_[y, np.ones_like(y)]
        coef, *_ = np.linalg.lstsq(A, img_x, rcond=None)
        r = float(np.sum((img_x - A @ coef) ** 2))
        if r < best[0]:
            best = (r, float(lag), float(coef[0]))
    return {"latency_ms": round(1e3 * best[1], 1), "px_per_rad": round(best[2], 2),
            "rms_px": round(math.sqrt(best[0] / len(img_x)), 3)}  # fmt: skip


# ---------------------------------------------------------------- the synthetic checks
def synthetic(seed: int = 0) -> dict[str, Any]:
    """Every fit on data made from a known model: what each must recover before it is trusted on HW2's captures."""
    out: dict[str, Any] = {}
    lens_p = fit_lens(synthetic_views("pinhole_k1", seed=seed), loo=False)
    out["lens_on_pinhole_k1"] = {"best": lens_p["best"], **lens_p["models"]["pinhole_k1"]}
    lens_e = fit_lens(synthetic_views("equidistant", seed=seed), kinds=("pinhole_k1", "equidistant"), loo=False)
    out["lens_on_equidistant"] = {"best": lens_e["best"], **lens_e["models"]["equidistant"]}
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    true = (-0.0266, 0.0053, -0.4905)
    x = np.hypot(xx - 259.0, yy - 203.9) / 375.23
    g = 1 + true[0] * x**2 + true[1] * x**4 + true[2] * x**6
    frames = [np.clip(200.0 * g + rng.normal(0, 2.0, g.shape), 0, 255) for _ in range(10)]
    vg = fit_vignetting(frames, 259.0, 203.9, 375.23)
    xs = np.linspace(0.0, 1.0, 101)
    g_fit = 1 + vg["a2"] * xs**2 + vg["a4"] * xs**4 + vg["a6"] * xs**6
    g_true = 1 + true[0] * xs**2 + true[1] * xs**4 + true[2] * xs**6
    out["vignetting"] = {**vg, "true": list(true), "gain_max_err": round(float(np.max(np.abs(g_fit - g_true))), 4)}
    t_j = np.arange(0.0, 3.0, 0.002)
    yaw = np.where(t_j > 1.0, 0.2, 0.0) + 0.05 * np.sin(2 * math.pi * 1.3 * t_j)
    t_i = np.arange(0.0, 3.0, 1 / 7.5) + 0.01
    img_x = 300.0 - 120.0 * np.interp(t_i - 0.085, t_j, yaw) + rng.normal(0, 0.3, t_i.shape)
    out["latency"] = {**latency_ms(t_i, img_x, t_j, yaw), "true_ms": 85.0}
    plane = 0.45 + 0.0002 * (xx - 272) + rng.normal(0, 0.004, xx.shape)
    plane[rng.random(plane.shape) < 0.1] = np.nan
    out["depth"] = {**depth_residuals(plane, 0.45), "true_sigma_m": 0.004}
    st = fit_stereo(synthetic_pairs(0.06, seed=seed))
    out["stereo"] = {"rms_px": st["rms_px"], "baseline_m": st["baseline_m"], "true_baseline_m": 0.06,
                     "right_f": st["right"]["f"]}  # fmt: skip
    return out


# ---------------------------------------------------------------- the r1cam recordings (session 11)
def mcap_frames(path: str | Path, topic: str) -> list[tuple[int, bytes]]:
    """(log_time ns = r1cam's receive time, payload) of every message on `topic` of an r1cam recording."""
    from mcap.reader import make_reader

    out: list[tuple[int, bytes]] = []
    with Path(path).open("rb") as f:
        for _s, _ch, msg in make_reader(f).iter_messages(topics=[topic]):
            out.append((int(msg.log_time), bytes(msg.data)))
    return out


def manifest_of(mcap_path: str | Path) -> dict[str, Any]:
    """The manifest r1cam writes beside a recording (`<stem>.r1cam.json`), {} if absent."""
    p = Path(mcap_path)
    m = p.with_name(p.name.replace(".r1cam.mcap", ".r1cam.json"))
    return dict(json.loads(m.read_text())) if m.exists() else {}


def _gray(jpeg: bytes) -> npt.NDArray[np.float64]:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"), dtype=np.float64)


def _calibration_model(calib: str | Path | None) -> Any:
    from r1sim import lens

    return (lens.load_calibration(calib) if calib else lens.load_calibration()).model


def vignetting_from_mcap(paths: Sequence[str | Path], *, calib: str | Path | None = None,
                         r_max: float | None = None) -> dict[str, Any]:  # fmt: skip
    """S11 (C2): the left frames of `r1 record run flat_field` recordings -> fit_vignetting about the calibration's
    principal point (default: the robot's head-camera file), r_max = models/head_camera/vignetting.yaml's (the model's
    own normalisation). Each frame is scaled to its own centre first (the light levels differ by design)."""
    import yaml

    from r1sim import lens

    M = _calibration_model(calib)
    if r_max is None:
        r_max = float(yaml.safe_load(lens.VIGNETTING_FILE.read_text())["r_max_px"])
    frames = []
    for p in paths:
        for _t, data in mcap_frames(p, "/r1cam/left"):
            g = _gray(data)
            yy, xx = np.mgrid[0 : g.shape[0], 0 : g.shape[1]]
            c = float(np.median(g[np.hypot(xx - M.cx, yy - M.cy) < 0.25 * r_max]))
            if c > 20.0:  # a frame with light (a covered lens / a dark level says nothing)
                frames.append(g * (200.0 / c))
    if not frames:
        return {"error": "no usable left frames (is the sheet lit and filling the view?)"}
    out = fit_vignetting(frames, float(M.cx), float(M.cy), float(r_max))
    v = yaml.safe_load(lens.VIGNETTING_FILE.read_text())
    xs = np.linspace(0.0, 1.0, 51)
    g_fit = 1 + out["a2"] * xs**2 + out["a4"] * xs**4 + out["a6"] * xs**6
    g_mod = 1 + float(v["a2"]) * xs**2 + float(v["a4"]) * xs**4 + float(v["a6"]) * xs**6
    return {**out, "model_now": {k: float(v[k]) for k in ("a2", "a4", "a6")},
            "gain_max_diff_vs_model": round(float(np.max(np.abs(g_fit - g_mod))), 4)}  # fmt: skip


def depth_from_mcap(paths: Sequence[str | Path], distances_m: Sequence[float] | None = None,
                    every: int = 1) -> dict[str, Any]:  # fmt: skip
    """S12 step 3 (C4): `r1 record run depth_targets label=... distance_m=...` recordings -> per recording the depth
    frames' depth_residuals (the median over frames: valid fraction, median error vs the tape, sigma z) and per band
    (B5 / A6: compare with the depth model's per-band numbers, +-20 %). The distance comes from the recording's manifest
    (the plan's distance_m) unless given (one per recording, same order)."""
    rows = []
    for k, p in enumerate(paths):
        man = manifest_of(p)
        z = float(distances_m[k]) if distances_m else float(man.get("params", {}).get("distance_m", 0.0))
        if z <= 0.0:
            rows.append(
                {"recording": str(p), "error": "no tape distance (distance_m) in the manifest - pass --distance-m"}
            )
            continue
        per = []
        for i, (_t, data) in enumerate(mcap_frames(p, "/r1cam/depth")):
            if i % max(1, every):
                continue
            d = np.frombuffer(data, dtype="<u2").reshape(H, W).astype(np.float64) / 1000.0
            d[d <= 0] = np.nan
            per.append(depth_residuals(d, z))
        ok = [r for r in per if "error" not in r]
        row: dict[str, Any] = {"recording": str(p), "label": man.get("label", ""), "distance_m": z,
                               "frames": len(per), "usable": len(ok)}  # fmt: skip
        if ok:
            row.update(band_m=ok[0]["band_m"],
                       valid_fraction=round(float(np.median([r["valid_fraction"] for r in ok])), 3),
                       median_err_m=round(float(np.median([r["median_err_m"] for r in ok])), 4),
                       sigma_z_m=round(float(np.median([r["sigma_z_m"] for r in ok])), 4))  # fmt: skip
        rows.append(row)
    bands: dict[str, dict[str, float]] = {}
    for r in rows:
        if "sigma_z_m" in r:
            b = bands.setdefault(str(r["band_m"]), {"n": 0, "sigma_z_m": 0.0, "median_err_m": 0.0})
            b["n"] += 1
            b["sigma_z_m"] += float(r["sigma_z_m"])
            b["median_err_m"] += float(r["median_err_m"])
    for b in bands.values():
        b["sigma_z_m"] = round(b["sigma_z_m"] / b["n"], 4)
        b["median_err_m"] = round(b["median_err_m"] / b["n"], 4)
    return {"recordings": rows, "bands": bands}


def board_x_from_mcap(path: str | Path, board: Any) -> tuple[FloatArray, FloatArray]:
    """(receive times s, the detected board corners' mean image x px) per left frame with >= MIN_CORNERS_FOR_FIT."""
    from r1sim import charuco_render as cr

    t, x = [], []
    for t_ns, data in mcap_frames(path, "/r1cam/left"):
        ids, px = cr.detect(np.asarray(_gray(data), dtype=np.uint8), board)
        if len(ids) >= cr.MIN_CORNERS_FOR_FIT:
            t.append(t_ns * 1e-9)
            x.append(float(np.mean(px[:, 0])))
    return np.asarray(t), np.asarray(x)


def head_yaw_from_r1log(paths: Sequence[str | Path]) -> tuple[FloatArray, FloatArray]:
    """(t_wake s, the measured head yaw rad) from r1log files (topic /r1rt/telemetry), time-ordered."""
    from r1mc3.log import reader

    from r1sim import joints as J

    ts, qs = [], []
    for p in paths:
        rec = reader.read(Path(p), log_time_order=True).topic("/r1rt/telemetry")
        if rec.size:
            ts.append(rec["t_wake_ns"].astype(np.float64) * 1e-9)
            qs.append(rec["q"][:, J.IDX["head_yaw"]].astype(np.float64))
    if not ts:
        return np.zeros(0), np.zeros(0)
    t, q = np.concatenate(ts), np.concatenate(qs)
    o = np.argsort(t, kind="stable")
    return t[o], q[o]


def latency_from_mcap(cam: str | Path, r1log: Sequence[str | Path], board: Any, *, offset_ns: int = 0,
                      max_lag_s: float = 0.3) -> dict[str, Any]:  # fmt: skip
    """C5: the board's image x (r1cam's receive times) vs the head yaw (r1log) during head-yaw steps -> latency_ms.
    Both clocks are CLOCK_MONOTONIC on the backpack (offset 0); `offset_ns` = r1cam's clock minus r1log's where they
    differ (macOS: Python's monotonic vs CLOCK_MONOTONIC - the sim rehearsal passes it)."""
    t_img, x = board_x_from_mcap(cam, board)
    t_j, yaw = head_yaw_from_r1log(r1log)
    if t_img.size < 10 or t_j.size < 100:
        return {"error": f"too little data: {t_img.size} frames with the board, {t_j.size} head samples"}
    t_img = t_img - offset_ns * 1e-9
    inside = (t_img > t_j[0] + max_lag_s) & (t_img < t_j[-1])
    if inside.sum() < 10:
        return {"error": "the camera frames and the head samples do not overlap in time (clock offset?)"}
    span = float(np.ptp(np.interp(t_img[inside], t_j, yaw)))
    out = latency_ms(t_img[inside], x[inside], t_j, yaw, max_lag_s=max_lag_s)
    return {**out, "frames": int(inside.sum()), "head_yaw_span_rad": round(span, 3)}


# ---------------------------------------------------------------- session 12: the hold-still lens fit
# HW2 day 1's C3 (12 Oct 2026, 325 s, 2428 left frames): the fit over ALL 1935 frames with a board gave RMS 1.14 px
# (B5 <= 1 px not met) - it includes the 2 s moves between the prompted holds (motion blur, rolling shutter). The plan's
# own timeline (`r1 record plan charuco`: 81 holds of 2 s, each after a 2 s move) says which frames are meant to be
# still; the corners' own motion between neighbouring frames confirms it (the person may be early / late).
STILL_SETTLE_S = 0.4  # the first 0.4 s of each prompted hold is skipped (the board is still settling)
STILL_MOTION_PX = 1.0  # still = the corners moved <= 1 px (median over the shared corners) from each neighbour frame
#                         (a hand-held board shakes ~0.5-1 px between frames at 7.5 FPS; a move is tens of px)
STILL_NEIGHBOUR_S = 0.4  # a neighbour farther away than this (a dropped / undetected frame) does not count
STILL_PER_WINDOW = 3  # views kept per hold (near-duplicates beyond a few add weight, not information)


def detections_timed(
    path: str | Path, board: Any, topic: str = "/r1cam/left"
) -> list[tuple[int, npt.NDArray[np.int64], FloatArray]]:
    """(receive time ns, corner ids, pixels N x 2) per frame of an r1cam recording with >= 6 corners (time-ordered)."""
    from r1sim import charuco_render as cr

    out: list[tuple[int, npt.NDArray[np.int64], FloatArray]] = []
    for t, data in mcap_frames(path, topic):
        ids, px = cr.detect(_gray(data).astype(np.uint8), board)
        if len(ids) >= cr.MIN_CORNERS_FOR_FIT:
            out.append((t, ids, px))
    out.sort(key=lambda d: d[0])
    return out


def save_detections(path: str | Path, dets: Sequence[tuple[int, npt.NDArray[np.int64], FloatArray]]) -> None:
    """A small cache of the detections (the 402 MB recording takes ~30 s to detect)."""
    n = np.array([len(i) for _, i, _ in dets], dtype=np.int64)
    np.savez_compressed(
        path,
        t=np.array([t for t, _, _ in dets], dtype=np.int64),
        n=n,
        ids=np.concatenate([i for _, i, _ in dets]) if dets else np.zeros(0, np.int64),
        px=np.concatenate([p for _, _, p in dets]) if dets else np.zeros((0, 2)),
    )


def load_detections(path: str | Path) -> list[tuple[int, npt.NDArray[np.int64], FloatArray]]:
    z = np.load(path)
    out: list[tuple[int, npt.NDArray[np.int64], FloatArray]] = []
    k = 0
    for t, n in zip(z["t"], z["n"], strict=True):
        out.append((int(t), z["ids"][k : k + n].astype(np.int64), z["px"][k : k + n].astype(np.float64)))
        k += int(n)
    return out


def hold_windows(
    mcap_path: str | Path, run_manifest: str | Path | None = None
) -> tuple[int, list[tuple[float, float]]]:
    """(t0 ns, [(t_start s, t_end s)] of every prompted hold) - from the record run's manifest (`<stamp>_<kind>.json`
    beside it: its t_start_mono_ns + camera.plan.schedule), else re-planned from r1cam's manifest (the same params:
    `r1mc3.record.plans.plan`) with r1cam's own start (0.4 ms apart on day 1). Both clocks are the backpack's
    CLOCK_MONOTONIC = the recording's log times."""
    if run_manifest is not None:
        m = json.loads(Path(run_manifest).read_text())
        sched = m["camera"]["plan"]["schedule"]
        t0 = int(m["t_start_mono_ns"])
    else:
        from r1mc3.record import plans

        cam = manifest_of(mcap_path)
        if not cam:
            raise ValueError(f"no r1cam manifest beside {mcap_path} and no --plan manifest given")
        camp = plans.plan(str(cam["kind"]), dict(cam.get("params", {})), None).camera
        assert camp is not None, "not a camera recording"
        sched = camp["schedule"]
        t0 = int(cam["t_start_mono_ns"])
    return t0, [(float(e["t_start"]), float(e["t_end"])) for e in sched]


def _motion_px(
    a: tuple[int, npt.NDArray[np.int64], FloatArray], b: tuple[int, npt.NDArray[np.int64], FloatArray]
) -> float:
    common, ia, ib = np.intersect1d(a[1], b[1], return_indices=True)
    if len(common) < 4:
        return math.inf
    return float(np.median(np.linalg.norm(a[2][ia] - b[2][ib], axis=1)))


def still_views(
    dets: Sequence[tuple[int, npt.NDArray[np.int64], FloatArray]],
    t0_ns: int,
    windows: Sequence[tuple[float, float]],
    board: Any,
    *,
    settle_s: float = STILL_SETTLE_S,
    max_motion_px: float = STILL_MOTION_PX,
    per_window: int = STILL_PER_WINDOW,
) -> tuple[list[tuple[FloatArray, FloatArray]], dict[str, Any]]:
    """The views for the lens fit: frames inside a prompted hold (after `settle_s`) whose corners moved <=
    max_motion_px from each neighbouring frame (within STILL_NEIGHBOUR_S), `per_window` evenly spaced per hold.
    Returns (views, a selection report)."""
    from r1sim import charuco_render as cr

    C = np.c_[cr.corners_mm(board) * cr.MM, np.zeros(board.n_corners)]
    ts = [(t - t0_ns) * 1e-9 for t, _, _ in dets]
    still: list[bool] = []
    for i, d in enumerate(dets):
        ok = True
        for j in (i - 1, i + 1):
            if (
                0 <= j < len(dets)
                and abs(ts[j] - ts[i]) <= STILL_NEIGHBOUR_S
                and _motion_px(d, dets[j]) > max_motion_px
            ):
                ok = False
        still.append(ok)
    views: list[tuple[FloatArray, FloatArray]] = []
    per: list[int] = []
    in_win = 0
    for t_a, t_b in windows:
        idx = [i for i, t in enumerate(ts) if t_a + settle_s <= t <= t_b]
        in_win += len(idx)
        good = [i for i in idx if still[i]]
        if not good:
            per.append(0)
            continue
        pick = (
            sorted({good[round(k * (len(good) - 1) / max(1, per_window - 1))] for k in range(per_window)})
            if per_window > 1
            else [good[len(good) // 2]]
        )
        per.append(len(pick))
        for i in pick:
            _t, ids, px = dets[i]
            views.append((C[ids].astype(np.float64), px.astype(np.float64)))
    report = {
        "frames_with_board": len(dets),
        "frames_in_holds": in_win,
        "frames_still": int(
            sum(1 for i, t in enumerate(ts) if still[i] and any(a + settle_s <= t <= b for a, b in windows))
        ),
        "holds": len(windows),
        "holds_used": int(sum(1 for n in per if n > 0)),
        "views": len(views),
        "rule": f"inside a prompted hold after {settle_s:g} s, corners moved <= {max_motion_px:g} px from each "
        f"neighbour frame, {per_window} per hold",
    }
    return views, report


def lens_still_from_mcap(
    path: str | Path,
    board: Any,
    *,
    run_manifest: str | Path | None = None,
    detections: str | Path | None = None,
    per_window: int = STILL_PER_WINDOW,
    max_motion_px: float = STILL_MOTION_PX,
    loo: bool = True,
) -> dict[str, Any]:
    """The lens fit on the hold-still frames only (B5 <= 1 px) + the all-frames fit for comparison (no LOO).
    `detections`: an .npz cache (written when missing)."""
    if detections is not None and Path(detections).exists():
        dets = load_detections(detections)
    else:
        dets = detections_timed(path, board)
        if detections is not None:
            save_detections(detections, dets)
    t0, wins = hold_windows(path, run_manifest)
    views, sel = still_views(dets, t0, wins, board, per_window=per_window, max_motion_px=max_motion_px)
    out = fit_lens(views, loo=loo)
    out["selection"] = sel
    from r1sim import charuco_render as cr

    C = np.c_[cr.corners_mm(board) * cr.MM, np.zeros(board.n_corners)]
    every = [(C[i].astype(np.float64), p) for _t, i, p in dets]
    allf = fit_lens(every, kinds=("pinhole_k1",), loo=False)["models"]["pinhole_k1"]
    out["all_frames_pinhole_k1"] = {"views": len(every), **allf}
    return out


def _board(name: str) -> Any:
    from r1sim import charuco_render as cr

    return next(b for b in cr.CANDIDATES if b.name == name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("lens")
    p.add_argument("--mcap", required=True)
    p.add_argument("--board", default="poster_6x4_90")
    p.add_argument("--no-loo", action="store_true")
    p.add_argument("--still", action="store_true", help="session 12: only the hold-still frames of the plan's timeline")
    p.add_argument("--plan", help="the record run's manifest (<stamp>_charuco_<label>.json; default: re-planned)")
    p.add_argument("--detections", help="an .npz cache of the detections (written when missing)")
    p.add_argument("--per-window", type=int, default=STILL_PER_WINDOW)
    p.add_argument("--max-motion-px", type=float, default=STILL_MOTION_PX)
    p = sub.add_parser("vignetting", help="S11 (C2): flat_field recordings -> the vignetting gain")
    p.add_argument("--mcap", nargs="+", required=True)
    p.add_argument("--calib", help="the head-camera calibration (default: the robot's file in models/head_camera)")
    p = sub.add_parser("depth", help="S12 step 3 (C4): depth_targets recordings -> depth residuals per band")
    p.add_argument("--mcap", nargs="+", required=True)
    p.add_argument("--distance-m", type=float, nargs="+", help="the tape distances (default: each manifest's)")
    p = sub.add_parser("latency", help="C5: a recording with the board in view during head-yaw steps + r1log")
    p.add_argument("--mcap", required=True)
    p.add_argument("--r1log", nargs="+", required=True)
    p.add_argument("--board", default="poster_6x4_90")
    p.add_argument("--offset-ns", type=int, default=0, help="r1cam's clock minus r1log's (0 on the robot)")
    sub.add_parser("synthetic")
    a = ap.parse_args(argv)
    if a.what == "synthetic":
        print(json.dumps(synthetic(), indent=1))
        return 0
    if a.what == "vignetting":
        print(json.dumps(vignetting_from_mcap(a.mcap, calib=a.calib), indent=1))
        return 0
    if a.what == "depth":
        print(json.dumps(depth_from_mcap(a.mcap, a.distance_m), indent=1))
        return 0
    if a.what == "latency":
        print(json.dumps(latency_from_mcap(a.mcap, a.r1log, _board(a.board), offset_ns=a.offset_ns), indent=1))
        return 0
    from r1sim import charuco_render as cr

    board = next(b for b in cr.CANDIDATES if b.name == a.board)
    if a.still:
        res = lens_still_from_mcap(
            a.mcap, board, run_manifest=a.plan, detections=a.detections, per_window=a.per_window,
            max_motion_px=a.max_motion_px, loo=not a.no_loo,
        )  # fmt: skip
        print(json.dumps(res, indent=1))
        return 0
    views = detections_from_mcap(a.mcap, board)
    print(json.dumps(fit_lens(views, loo=not a.no_loo), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
