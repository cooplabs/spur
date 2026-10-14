"""The head camera's MOUNT re-fitted from a STILL ChArUco board seen at several head poses (session 12; HW2 day 1's C3).

Why: the camera's pose on the head (`T_site_cam`, the calibration file's mount) was fitted on 4 Oct from 16 taped marks
in ONE image - together with the lens, so the two partly stand in for each other (the 4 Oct lens f 126.94 px; HW2's
full-field ChArUco recording C3: f 124.97 px, RMS 0.34 px over 1718 views). A new lens needs its own mount, and one
image of coplanar marks determines the mount's distance along the line of sight poorly (with C3's lens the 4 Oct marks
move the mount 29 mm along x). Here the head itself moves (pitch x yaw, known from the joints) while the board lies
still anywhere in view - the classic eye-in-hand calibration: per frame the board's pose in the camera (PnP with the
given lens), then the mount AND the board's pose in `base` by least squares on every corner's reprojection. No tape
measure: the board's pose is a by-product (and a check: lying flat on the table, its plane must be level at the
table's height - `--table-z`).

    # on the robot (r1d running, r1cam running, the board lying still on the table in front of the robot; the head
    # moves through a grid of poses - the operator's OK; nothing else moves):
    python -m r1mc3.measure.head_mount capture --r1d-port 8780 --out DIR [--pitch 0.30 0.55 --n-pitch 3]
        [--yaw -0.35 0.35 --n-yaw 4] --yes
    # the fit (the Mac or the robot; OpenCV = the perception extra):
    python -m r1mc3.measure.head_mount fit DIR --calib /etc/r1mc3/head_camera.json [--lens LENS.json]
        [--table-z 0.752] [--board-pose X Y Z YAW_DEG] [--out NEW_CALIBRATION.json]

**The honest limit (sim, rendered frames through r1d, tests/integration/test_head_mount_sim.py):** over the head's own
small rotations (+-0.35 rad) a mount shift and a board shift look almost alike - the free-board fit's mount came back
0.26 deg / 8.6 mm from the truth over the full 12-pose grid with an RMS barely below the truth's (0.40 vs 0.43 px). So
the FREE fit is a check (`file_as_is_rms_px`: how well the calibration in force explains the capture), not a
calibration; the accurate variant lays the board flat at a TAPE-MEASURED pose (`--board-pose X Y Z YAW_DEG`: its origin
corner in base) and fits only the mount (synthetic: the tape's own error, ~3 mm).

`fit` writes a calibration file only with `--out` (the robot's `/etc/r1mc3/head_camera.json` is replaced by hand,
with the user's OK - never by this tool). Generic (D32): a printed ChArUco board (CHARUCO.md) and the head's joints.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..cam.pointing import project
from .phone_calib import BOARD, board_and_detector

FloatArray = npt.NDArray[np.float64]

PITCH_RANGE = (0.30, 0.55)  # rad (head pitch + = nod DOWN; r1d keeps 0.05 inside the 0.6283 limit: max 0.5783)
YAW_RANGE = (-0.35, 0.35)  # rad (head yaw + = look LEFT)
SETTLE_S = 1.5  # the head's move + settling before the frames
FRAMES_PER_POSE = 2
MIN_CORNERS = 8
MIN_FRAMES = 6
GOOD_RMS_PX = 0.8  # the fit's reprojection RMS above this: not a calibration (blur, the board moved, a wrong lens).
#                    Synthetic (test_head_mount): 0.3 px corner noise -> 0.42 px; the board nudged 2 cm half-way -> 0.97
#                    px (the fit splits the difference, so the bar sits between); C3's real corners fit a full lens at
#                    0.34-0.38 px. Day 2's real number decides whether 0.8 is right (said in the report)


@dataclass(frozen=True)
class MountFrame:
    """One frame: the head_camera site in `base` at capture (r1d's observe) + the board's corners."""

    name: str
    T_base_site: FloatArray
    obj: FloatArray  # N x 3 board points (m, the board's own frame)
    px: FloatArray  # N x 2 pixels
    q_head: tuple[float, float] | None = None


def head_grid(
    pitch: Sequence[float] = PITCH_RANGE, yaw: Sequence[float] = YAW_RANGE, n_pitch: int = 3, n_yaw: int = 4
) -> list[tuple[float, float]]:
    """The head poses (pitch, yaw), a serpentine so each move is short."""
    ps = np.linspace(float(pitch[0]), float(pitch[1]), max(1, n_pitch))
    ys = np.linspace(float(yaw[0]), float(yaw[1]), max(1, n_yaw))
    out: list[tuple[float, float]] = []
    for i, p in enumerate(ps):
        for y in ys if i % 2 == 0 else ys[::-1]:
            out.append((round(float(p), 4), round(float(y), 4)))
    return out


# ---------------------------------------------------------------- geometry
def _rot(rv: FloatArray) -> FloatArray:
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.asarray(np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K), dtype=np.float64)


def _rv(R: FloatArray) -> FloatArray:
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return np.asarray(w * th, dtype=np.float64)


def _T(R: FloatArray, t: FloatArray) -> FloatArray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _apply(T0: FloatArray, x: FloatArray) -> FloatArray:
    return _T(_rot(x[:3]) @ T0[:3, :3], T0[:3, 3] + x[3:6])


def angle_deg(Ta: FloatArray, Tb: FloatArray) -> float:
    R = Ta[:3, :3].T @ Tb[:3, :3]
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def reproject(model: Mapping[str, Any], T_site_cam: FloatArray, T_base_board: FloatArray, f: MountFrame) -> FloatArray:
    T_cam_board = np.linalg.inv(f.T_base_site @ T_site_cam) @ T_base_board
    Pc = (T_cam_board[:3, :3] @ f.obj.T).T + T_cam_board[:3, 3]
    return project(model, Pc)


def _pnp(model: Mapping[str, Any], f: MountFrame) -> FloatArray | None:
    """The board in the camera (4 x 4) from one frame (OpenCV's k1 = the model's k1: the same radial form)."""
    import cv2

    K = np.array([[model["f"], 0, model["cx"]], [0, model["f"], model["cy"]], [0, 0, 1]], dtype=np.float64)
    D = np.array([float(model.get("k1") or 0.0), 0.0, 0.0, 0.0, 0.0])
    ok, rv, tv = cv2.solvePnP(f.obj, f.px, K, D)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rv)
    return _T(np.asarray(R, dtype=np.float64), np.asarray(tv, dtype=np.float64).ravel())


def initial_board(model: Mapping[str, Any], frames: Sequence[MountFrame], T_site_cam: FloatArray) -> FloatArray:
    """The board's pose in `base` under a mount: the frames' PnP poses through it, averaged (rotations via SVD)."""
    Ts = [f.T_base_site @ T_site_cam @ Tc for f in frames if (Tc := _pnp(model, f)) is not None]
    if not Ts:
        raise ValueError("no frame gives a board pose (PnP failed everywhere)")
    U, _s, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return _T(R, np.mean([T[:3, 3] for T in Ts], axis=0))


def fit_mount(
    model: Mapping[str, Any],
    frames: Sequence[MountFrame],
    T0: FloatArray,
    *,
    iters: int = 60,
    translation: bool = True,
    board: FloatArray | None = None,
) -> dict[str, Any]:
    """The mount (and the board's pose in base) that best reproject every corner: Levenberg-Marquardt on 12
    parameters (a rotation vector + a translation each), numeric Jacobian (no scipy in the environment).
    `translation=False`: the mount's ROTATION only (its translation kept from T0) - over the head's small rotations a
    mount shift and a board shift look almost alike (sim, 6 frames over +-0.2 rad: the translation 14.5 mm off).
    `board`: the board's KNOWN pose in base (tape-measured, lying flat on the table) - then only the mount is fitted
    (absolute correspondences: the accurate variant; the free-board fit's mount came back 0.26 deg / 8.6 mm off in sim
    over the full 12-pose grid, its RMS barely above the truth's)."""
    if len(frames) < MIN_FRAMES:
        raise ValueError(f"{len(frames)} usable frames (need {MIN_FRAMES}: more head poses with the board in view)")
    B0 = initial_board(model, frames, T0) if board is None else np.asarray(board, dtype=np.float64)
    obs = np.concatenate([f.px for f in frames])

    free = list(range(12)) if translation else [0, 1, 2, 6, 7, 8, 9, 10, 11]
    if board is not None:
        free = [k for k in free if k < 6]

    def full(xf: FloatArray) -> FloatArray:
        x12 = np.zeros(12)
        x12[free] = xf
        return x12

    def res(xf: FloatArray) -> FloatArray:
        x12 = full(xf)
        Tm, Tb = _apply(T0, x12[:6]), _apply(B0, x12[6:])
        r = np.concatenate([reproject(model, Tm, Tb, f) for f in frames]) - obs
        return np.asarray(np.nan_to_num(r, nan=1e3).ravel(), dtype=np.float64)

    n = len(free)
    x = np.zeros(n)
    r = res(x)
    lam = 1e-3
    for _ in range(iters):
        J = np.empty((r.size, n))
        for k in range(n):
            dx = np.zeros(n)
            dx[k] = 1e-6
            J[:, k] = (res(x + dx) - r) / 1e-6
        A, g = J.T @ J, J.T @ r
        step: FloatArray = np.zeros(n)
        while True:
            step = np.asarray(np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g), dtype=np.float64)
            r2 = res(x + step)
            if float(r2 @ r2) < float(r @ r):
                x, r, lam = x + step, r2, max(lam / 10.0, 1e-9)
                break
            lam *= 10.0
            if lam > 1e8:
                break
        if float(np.linalg.norm(step)) < 1e-10 or lam > 1e8:
            break
    e = np.linalg.norm(r.reshape(-1, 2), axis=1)
    per = []
    k = 0
    for f in frames:
        n = len(f.px)
        per.append({"frame": f.name, "rms_px": round(float(np.sqrt(np.mean(e[k : k + n] ** 2))), 3)})
        k += n
    x12 = full(x)
    Tm, Tb = _apply(T0, x12[:6]), _apply(B0, x12[6:])
    return {
        "translation_fitted": translation,
        "board_known": board is not None,
        "T_site_cam": Tm,
        "T_base_board": Tb,
        "rms_px": float(np.sqrt(np.mean(e**2))),
        "max_px": float(e.max()),
        "frames": len(frames),
        "corners": len(e),
        "per_frame": per,
    }


def board_only_rms(model: Mapping[str, Any], frames: Sequence[MountFrame], T_site_cam: FloatArray) -> float:
    """How well a calibration AS IT IS explains the capture: its lens + its mount fixed, only the board's pose fitted
    (6 parameters) - the day's direct check of the installed file (no refit of the camera at all)."""
    B0 = initial_board(model, frames, T_site_cam)
    obs = np.concatenate([f.px for f in frames])

    def res(x: FloatArray) -> FloatArray:
        Tb = _apply(B0, x)
        r = np.concatenate([reproject(model, T_site_cam, Tb, f) for f in frames]) - obs
        return np.asarray(np.nan_to_num(r, nan=1e3).ravel(), dtype=np.float64)

    x = np.zeros(6)
    r = res(x)
    lam = 1e-3
    for _ in range(40):
        J = np.empty((r.size, 6))
        for k in range(6):
            dx = np.zeros(6)
            dx[k] = 1e-6
            J[:, k] = (res(x + dx) - r) / 1e-6
        A, g = J.T @ J, J.T @ r
        step: FloatArray = np.zeros(6)
        while True:
            step = np.asarray(np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g), dtype=np.float64)
            r2 = res(x + step)
            if float(r2 @ r2) < float(r @ r):
                x, r, lam = x + step, r2, max(lam / 10.0, 1e-9)
                break
            lam *= 10.0
            if lam > 1e8:
                break
        if float(np.linalg.norm(step)) < 1e-10 or lam > 1e8:
            break
    return float(np.sqrt(np.mean(np.sum(r.reshape(-1, 2) ** 2, axis=1))))


def leave_one_pose_out(model: Mapping[str, Any], frames: Sequence[MountFrame], T0: FloatArray) -> list[float]:
    """Per head pose: refit without its frames, then its corners' RMS (px) under that mount + board - how well the
    mount predicts a head pose it has not seen."""
    poses = sorted({f.q_head for f in frames if f.q_head is not None})
    out: list[float] = []
    for q in poses:
        rest = [f for f in frames if f.q_head != q]
        if len(rest) < MIN_FRAMES:
            continue
        fit = fit_mount(model, rest, T0, iters=30)
        held = [f for f in frames if f.q_head == q]
        e = np.concatenate([reproject(model, fit["T_site_cam"], fit["T_base_board"], f) - f.px for f in held])
        out.append(float(np.sqrt(np.nanmean(np.sum(e**2, axis=1)))))
    return out


def board_pose_flat(x: float, y: float, z: float, yaw_deg: float) -> FloatArray:
    """The board lying flat on the table, face up: its corner-origin at (x, y, z) in base, its x axis turned yaw_deg
    from base x (the tape: the board's first inner corner = OpenCV's ChArUco origin)."""
    a = math.radians(yaw_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return _T(R, np.array([x, y, z]))


def board_check(T_base_board: FloatArray, table_z: float | None) -> dict[str, Any]:
    """The board lying flat on the table: its plane's tilt from level and its height vs the table's (a check of the
    mount's pitch / roll and height that needs no tape)."""
    n = T_base_board[:3, 2]
    tilt = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
    out: dict[str, Any] = {"tilt_from_level_deg": round(tilt, 2), "origin_m": np.round(T_base_board[:3, 3], 4).tolist()}
    if table_z is not None:
        out["height_vs_table_mm"] = round(1000.0 * (float(T_base_board[2, 3]) - float(table_z)), 1)
    return out


# ---------------------------------------------------------------- the robot side
def capture(
    port: int,
    out_dir: str | Path,
    poses: Sequence[tuple[float, float]],
    *,
    settle_s: float = SETTLE_S,
    frames_per_pose: int = FRAMES_PER_POSE,
    say: Any = print,
) -> list[dict[str, Any]]:
    """Through r1d's public ops only: per pose `stream.head` (TTL 0.3 s, re-sent every 0.1 s while settling: the head
    HOLDs where it is after the last one) then `camera.observe` (no depth) x frames_per_pose -> DIR/<k>_<n>.jpg +
    .json (r1d's T_base_site at capture, the commanded head q, the reply's pose source / frame age)."""
    from ..service.client import Client

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with Client(port=port) as c:
        for k, (p, y) in enumerate(poses):
            say(f"pose {k + 1}/{len(poses)}: head pitch {p:+.3f}, yaw {y:+.3f} rad")
            t_end = time.monotonic() + settle_s
            while time.monotonic() < t_end:
                r = c.call("stream.head", q=[p, y], ttl=0.3)
                if not r.get("ok"):
                    raise RuntimeError(f"the head was refused at ({p}, {y}): {r.get('code')} {r.get('details')}")
                time.sleep(0.1)
            time.sleep(0.4)  # the last command's TTL runs out: the head holds
            for n in range(frames_per_pose):
                ob = c.call("camera.observe", depth=False, jpeg=True, max_age_s=0.5)
                if not ob.get("ok"):
                    say(f"  observe refused: {ob.get('code')} {ob.get('details')}")
                    continue
                d = ob["data"]
                name = f"{k:02d}_{n}"
                (out / f"{name}.jpg").write_bytes(base64.b64decode(d["jpeg_b64"]))
                meta = {"name": name, "q_head": [p, y], "T_base_site": d["T_base_site"],
                        "pose_source": d.get("pose_source"), "frame_age_ms": d.get("frame_age_ms"),
                        "calibration": d.get("calibration"), "camera_model": d.get("camera_model")}  # fmt: skip
                (out / f"{name}.json").write_text(json.dumps(meta, indent=1))
                rows.append(meta)
                time.sleep(0.15)
    return rows


def load_frames(out_dir: str | Path, board_spec: Mapping[str, Any] = BOARD) -> tuple[list[MountFrame], list[str]]:
    """The captured frames with >= MIN_CORNERS board corners (and the names of those skipped)."""
    import cv2

    board, det = board_and_detector(board_spec)
    frames: list[MountFrame] = []
    skipped: list[str] = []
    for js in sorted(Path(out_dir).glob("*.json")):
        meta = json.loads(js.read_text())
        img = cv2.imread(str(js.with_suffix(".jpg")), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append(js.stem)
            continue
        corners, ids, _a, _b = det.detectBoard(img)
        if ids is None or len(ids) < MIN_CORNERS:
            skipped.append(js.stem)
            continue
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        obj = np.asarray(board.getChessboardCorners(), dtype=np.float64)[ids]
        q = meta.get("q_head")
        frames.append(MountFrame(meta["name"], np.asarray(meta["T_base_site"], dtype=np.float64), obj,
                                 np.asarray(corners, dtype=np.float64).reshape(-1, 2),
                                 (float(q[0]), float(q[1])) if q else None))  # fmt: skip
    return frames, skipped


def run_fit(
    frames: Sequence[MountFrame],
    calib: Mapping[str, Any],
    lens: Mapping[str, Any] | None = None,
    *,
    table_z: float | None = None,
    loo: bool = True,
    board: FloatArray | None = None,
) -> dict[str, Any]:
    """The mount under `lens` (default: the calibration file's own), started from the file's mount; the report.
    `board`: its known pose in base (`board_pose_flat`) - only the mount is fitted."""
    model = {**dict(calib["model"]), **(dict(lens) if lens else {})}
    model.setdefault("depth", "z")
    T0 = np.asarray(calib["T_site_cam"], dtype=np.float64)
    fit = fit_mount(model, frames, T0, board=board)
    file_model = {**dict(calib["model"])}
    file_model.setdefault("depth", "z")
    rep: dict[str, Any] = {
        # the calibration file AS IT IS (its lens + its mount; only the board's pose free): the day's direct check
        "file_as_is_rms_px": round(board_only_rms(file_model, frames, T0), 3),
        "frames": fit["frames"],
        "corners": fit["corners"],
        "rms_px": round(fit["rms_px"], 3),
        "max_px": round(fit["max_px"], 3),
        "ok": bool(fit["rms_px"] <= GOOD_RMS_PX),
        "mount_change": {
            "dpos_mm": np.round(1000.0 * (fit["T_site_cam"][:3, 3] - T0[:3, 3]), 2).tolist(),
            "dang_deg": round(angle_deg(T0, fit["T_site_cam"]), 3),
        },
        "board": board_check(fit["T_base_board"], table_z),
        "lens": {k: model[k] for k in ("lens", "f", "cx", "cy", "k1") if k in model},
    }
    if loo and board is None:
        lo = leave_one_pose_out(model, frames, T0)
        rep["left_out_pose_rms_px"] = {
            "median": round(float(np.median(lo)), 3) if lo else None,
            "max": round(float(np.max(lo)), 3) if lo else None,
            "poses": len(lo),
        }
    rep["T_site_cam"] = np.round(fit["T_site_cam"], 9).tolist()
    rep["model"] = model
    return rep


def calibration_file(rep: Mapping[str, Any], *, note: str, source: str, stream: str | None = None) -> dict[str, Any]:
    out = {
        "T_site_cam": rep["T_site_cam"],
        "model": rep["model"],
        "width": 544,
        "height": 448,
        "frame": "v3 (r1mc3.world.Geometry.set_stance: the IMU mount rotation undone)",
        "note": note,
        "source": source,
    }
    if stream:
        out["stream"] = stream
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="what", required=True)
    c = sub.add_parser("capture", help="the robot: the head through a grid of poses, frames of the still board")
    c.add_argument("--r1d-port", type=int, default=8780)
    c.add_argument("--out", required=True)
    c.add_argument("--pitch", type=float, nargs=2, default=list(PITCH_RANGE))
    c.add_argument("--yaw", type=float, nargs=2, default=list(YAW_RANGE))
    c.add_argument("--n-pitch", type=int, default=3)
    c.add_argument("--n-yaw", type=int, default=4)
    c.add_argument("--yes", action="store_true", help="move the head (without it: the plan only)")
    f = sub.add_parser("fit", help="the mount from a capture folder")
    f.add_argument("dir")
    f.add_argument("--calib", required=True, help="the calibration file in force (its mount = the start)")
    f.add_argument("--lens", help="the lens to fit the mount under: a JSON {lens, f, cx, cy, k1} or a calibration file")
    f.add_argument("--table-z", type=float, help="the table top (m, base) when the board lies flat on it")
    f.add_argument("--no-loo", action="store_true")
    f.add_argument("--board-pose", type=float, nargs=4, metavar=("X", "Y", "Z", "YAW_DEG"),
                   help="the board's KNOWN pose (flat, face up; its origin corner tape-measured in base): the "
                   "accurate variant - only the mount is fitted")  # fmt: skip
    f.add_argument("--out", help="write a NEW calibration file here (never the robot's file)")
    a = ap.parse_args(argv)
    if a.what == "capture":
        poses = head_grid(a.pitch, a.yaw, a.n_pitch, a.n_yaw)
        print(json.dumps({"poses": poses, "frames_per_pose": FRAMES_PER_POSE, "settle_s": SETTLE_S}))
        if not a.yes:
            print("plan only: add --yes to move the head (the operator's OK)")
            return 0
        rows = capture(a.r1d_port, a.out, poses)
        print(f"{len(rows)} frames in {a.out}")
        return 0
    calib = json.loads(Path(a.calib).read_text())
    lens = json.loads(Path(a.lens).read_text()) if a.lens else None
    if lens is not None and "model" in lens:  # a calibration file: its lens
        lens = dict(lens["model"])
    frames, skipped = load_frames(a.dir)
    bp = board_pose_flat(*a.board_pose) if a.board_pose else None
    rep = run_fit(frames, calib, lens, table_z=a.table_z, loo=not a.no_loo, board=bp)
    rep["skipped_frames"] = skipped
    show = {k: v for k, v in rep.items() if k not in ("T_site_cam", "model")}
    print(json.dumps(show, indent=1))
    if a.out:
        if not rep["ok"]:
            print(f"NOT written: RMS {rep['rms_px']} px > {GOOD_RMS_PX} (blur, the board moved, or a wrong lens)")
            return 1
        cal = calibration_file(rep, note=f"head_mount fit: {rep['frames']} frames, RMS {rep['rms_px']} px",
                               source=str(a.dir), stream=calib.get("stream"))  # fmt: skip
        Path(a.out).write_text(json.dumps(cal, indent=1) + "\n")
        print(f"wrote {a.out} (NOT installed: /etc/r1mc3/head_camera.json changes only by hand, with the user's OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
