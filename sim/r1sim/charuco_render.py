"""The ChArUco render test (R1CAM_PLAN.md §5 session 9 item 1; DESIGN §13.2 B5): which calibration board does the R1
head camera detect over its WHOLE field at the S12 distances (0.3 / 0.45 / 0.6 m), so the board the user prints works?

The head camera is wide (pinhole + k1, f 126.9 px for a 544 x 448 frame: ~130 deg across), so a board's squares are
small in the image: at 0.6 m a 40 mm square spans 8.5 px and its 30 mm marker 6.3 px = ~1 px per marker cell - below
what ArUco detection needs (~2-3 px per cell). The drafted board (7 x 5 squares of 40 mm, 30 mm markers, DICT_4X4_50 on
A3, HW1_RUNBOOK_ADDITIONS S7) is therefore compared with bigger ones that still fit a print-shop paper size.

Each board is built from geometry, not a texture (as r1sim.tag_render): a white backing plate and one thin black box per
black square and per run of black marker cells, read from OpenCV's own `CharucoBoard.generateImage` (so the board in the
sim IS OpenCV's board, whatever its id / orientation conventions). The board is a mocap body placed in the head camera's
frame: its centre on the ray through a target pixel (3 x 3 places over the image) at a depth z (0.3 / 0.45 / 0.6 m),
facing the camera, tilted 0 / +30 deg (about its vertical axis) / -30 deg (about its horizontal axis) - the S12 sweep
(R1CAM_PLAN.md: 9 places x 3 tilts per distance). Each frame goes through the full head chain (wide render -> remap ->
544 x 448 -> vignetting -> JPEG q90 -> decode); cv2.aruco.CharucoDetector finds the corners; they are compared with the
truth (the corners projected with the calibrated lens model). A "soft" variant adds a Gaussian blur (sigma 1 px: focus /
H.264 / motion) before the detection - the margin. Last, the lens is re-fitted from all detections of a board
(cv2.calibrateCamera: f, cx, cy, k1 from a generic start) - the B5 fit's synthetic check: the board is good when the fit
recovers the model and the full-field RMS sits at the render's own floor.

The robot stands at the 4 Oct capture's joints (tests/assets/render/s3_cal/head_pose.json); no scene (the board fills
the view; a generic floor behind). Not modelled: print quality, paper sheen / glare, the hand-held board's motion.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from . import joints, lens, model, world

FloatArray = npt.NDArray[np.float64]
MM = 1e-3
DISTANCES_M = (0.3, 0.45, 0.6)  # S12 (HW1_RUNBOOK_ADDITIONS S7 step 2)
PLACES = tuple((u, v) for v in (0.18, 0.5, 0.82) for u in (0.15, 0.5, 0.85))  # target pixel as a fraction of W / H
TILTS = (("flat", "", 0.0), ("yaw+30", "y", 30.0), ("pitch-30", "x", -30.0))
SOFT_SIGMA_PX = 1.0
MIN_CORNERS_FOR_FIT = 6


@dataclass(frozen=True)
class Paper:
    label: str
    w_mm: float  # landscape: w >= h
    h_mm: float


TABLOID = Paper("11 x 17 in (tabloid / ledger)", 431.8, 279.4)
A3 = Paper("A3", 420.0, 297.0)
POSTER_18X24 = Paper("18 x 24 in (poster)", 609.6, 457.2)
POSTER_20X30 = Paper("20 x 30 in (poster)", 762.0, 508.0)  # the user's print service: its only size (11 Oct ~23:15)


@dataclass(frozen=True)
class BoardSpec:
    """A ChArUco board: `cols` x `rows` squares of `square_mm`, ArUco markers of `marker_mm` from `dict_name` in the
    white squares (OpenCV 4.7+ layout: the top-left square black, marker ids 0.. row-major)."""

    name: str
    cols: int
    rows: int
    square_mm: float
    marker_mm: float
    dict_name: str
    paper: Paper

    @property
    def width_mm(self) -> float:
        return self.cols * self.square_mm

    @property
    def height_mm(self) -> float:
        return self.rows * self.square_mm

    @property
    def n_corners(self) -> int:
        return (self.cols - 1) * (self.rows - 1)

    def margins_mm(self) -> tuple[float, float]:
        return (self.paper.w_mm - self.width_mm) / 2.0, (self.paper.h_mm - self.height_mm) / 2.0


# The drafted board and the alternatives that still fit a print-shop paper with >= 15 mm of white around the grid.
CANDIDATES: tuple[BoardSpec, ...] = (
    BoardSpec("draft_7x5_40", 7, 5, 40.0, 30.0, "DICT_4X4_50", TABLOID),
    BoardSpec("tabloid_6x4_60", 6, 4, 60.0, 45.0, "DICT_4X4_50", TABLOID),
    BoardSpec("poster_8x6_60", 8, 6, 60.0, 45.0, "DICT_4X4_50", POSTER_18X24),
    BoardSpec("poster_7x5_75", 7, 5, 75.0, 56.0, "DICT_4X4_50", POSTER_18X24),
    BoardSpec("poster_7x5_80", 7, 5, 80.0, 60.0, "DICT_4X4_50", POSTER_18X24),
    # the chosen grid on 20 x 30 in (the user, 11 Oct ~23:15: the print service's only size; "you can add white space"):
    # the grid is the one the render test chose on 18 x 24 in - the paper only adds white around it
    BoardSpec("poster_6x4_90", 6, 4, 90.0, 68.0, "DICT_4X4_50", POSTER_20X30),
)

# The board to print (session 9, this render test: docs/hardware/CHARUCO.md has the table): the one that detects at
# all three S12 distances and keeps detecting at 0.3 / 0.45 m under a 1 px blur, the best lens re-fit.
CHOSEN = next(b for b in CANDIDATES if b.name == "poster_6x4_90")
DRAFT = CANDIDATES[0]


def opencv_board(b: BoardSpec) -> Any:
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b.dict_name))
    return cv2.aruco.CharucoBoard((b.cols, b.rows), b.square_mm * MM, b.marker_mm * MM, d)


@dataclass(frozen=True)
class Rect:
    """A black rectangle on the board (mm, board frame: x right, y DOWN from the grid's top-left corner)."""

    x: float
    y: float
    w: float
    h: float


def board_rects(b: BoardSpec) -> list[Rect]:
    """Every black area of the board as rectangles (black squares + horizontal runs of black marker cells), read from
    OpenCV's own drawing so the geometry is exactly OpenCV's board."""
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b.dict_name))
    n_cells = d.markerSize + 2  # the marker's bits + its one-cell black border
    k = 20  # image px per marker cell when sampling
    px_per_mm = k * n_cells / b.marker_mm
    img = opencv_board(b).generateImage(
        (round(b.width_mm * px_per_mm), round(b.height_mm * px_per_mm)), marginSize=0, borderBits=1
    )
    sx, sy = img.shape[1] / b.width_mm, img.shape[0] / b.height_mm
    out: list[Rect] = []
    cell = b.marker_mm / n_cells
    off = (b.square_mm - b.marker_mm) / 2.0
    for r in range(b.rows):
        for c in range(b.cols):
            x0, y0 = c * b.square_mm, r * b.square_mm
            mid = img[int((y0 + b.square_mm / 2) * sy), int((x0 + b.square_mm / 2) * sx)]
            corner = img[int((y0 + off / 2) * sy), int((x0 + off / 2) * sx)]
            if corner < 128 and mid < 128:  # a black square
                out.append(Rect(x0, y0, b.square_mm, b.square_mm))
                continue
            # a white square with a marker: its cells, row by row, black runs merged
            for i in range(n_cells):
                cy = y0 + off + (i + 0.5) * cell
                run0: int | None = None
                for j in range(n_cells + 1):
                    black = j < n_cells and img[int(cy * sy), int((x0 + off + (j + 0.5) * cell) * sx)] < 128
                    if black and run0 is None:
                        run0 = j
                    elif not black and run0 is not None:
                        out.append(Rect(x0 + off + run0 * cell, y0 + off + i * cell, (j - run0) * cell, cell))
                        run0 = None
    return out


def rasterize(b: BoardSpec, rects: Sequence[Rect], px_per_mm: float, margin_mm: float = 20.0) -> npt.NDArray[np.uint8]:
    """The rectangles drawn on white (an independent check of `board_rects` / the PDF: OpenCV must find the board)."""
    w = round((b.width_mm + 2 * margin_mm) * px_per_mm)
    h = round((b.height_mm + 2 * margin_mm) * px_per_mm)
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in rects:
        x0 = round((r.x + margin_mm) * px_per_mm)
        y0 = round((r.y + margin_mm) * px_per_mm)
        x1 = round((r.x + r.w + margin_mm) * px_per_mm)
        y1 = round((r.y + r.h + margin_mm) * px_per_mm)
        img[y0:y1, x0:x1] = 0
    return img


def corners_mm(b: BoardSpec) -> FloatArray:
    """The ChArUco corners (id order) in the board frame (mm): ((j + 1) s, (i + 1) s) row-major = OpenCV's ids."""
    return np.array(
        [((j + 1) * b.square_mm, (i + 1) * b.square_mm) for i in range(b.rows - 1) for j in range(b.cols - 1)]
    )


def add_board(spec: Any, b: BoardSpec, name: str = "charuco") -> None:
    """The board as a mocap body whose frame = the board frame (x right, y down, z INTO the board: the camera sees
    the -z face). Visual only (no contact, no mass)."""
    body = spec.worldbody.add_body(name=name, mocap=True)
    t = 0.0006
    m_white = 15.0  # mm of white around the grid (the print's margin)
    W, H = b.width_mm * MM, b.height_mm * MM
    body.add_geom(
        name=f"{name}_white",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[W / 2 + m_white * MM, H / 2 + m_white * MM, t / 2],
        pos=[W / 2, H / 2, t / 2],
        rgba=[0.97, 0.97, 0.97, 1],
        contype=0,
        conaffinity=0,
        mass=0.0,
    )
    for k, r in enumerate(board_rects(b)):
        body.add_geom(
            name=f"{name}_b{k}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[r.w * MM / 2, r.h * MM / 2, t / 2],
            pos=[(r.x + r.w / 2) * MM, (r.y + r.h / 2) * MM, -t / 2],
            rgba=[0.03, 0.03, 0.03, 1],
            contype=0,
            conaffinity=0,
            mass=0.0,
        )


def _rot(axis: str, deg: float) -> FloatArray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.eye(3)


def board_in_camera(
    b: BoardSpec, cal: lens.HeadCalibration, u_frac: float, v_frac: float, z_m: float, axis: str, deg: float
) -> FloatArray:
    """T_cam_board (OpenCV camera frame): the board's centre on the ray through pixel (u, v) at depth z, its axes along
    the camera's (upright, facing it), then tilted about its own centre."""
    M = cal.model
    ray = lens.normalised_rays(M, [u_frac * (M.width - 1)], [v_frac * (M.height - 1)])[0]
    centre = ray * z_m
    R = _rot(axis, deg)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centre - R @ np.array([b.width_mm * MM / 2, b.height_mm * MM / 2, 0.0])
    return T


def _mat_to_quat(R: FloatArray) -> FloatArray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).ravel())
    return q


def detect(gray: npt.NDArray[np.uint8], b: BoardSpec) -> tuple[npt.NDArray[np.int64], FloatArray]:
    """ChArUco corners: (ids, pixels N x 2); empty when the board is not found."""
    import cv2

    det = cv2.aruco.CharucoDetector(opencv_board(b))
    cc, ci, _, _ = det.detectBoard(gray)
    ids: Any = ci
    if ids is None or len(ids) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 2))
    return np.asarray(ids, dtype=np.int64).ravel(), np.asarray(cc, dtype=np.float64).reshape(-1, 2)


CAPTURE_POSE = Path(__file__).resolve().parents[2] / "tests" / "assets" / "render" / "s3_cal" / "head_pose.json"


def capture_qpos(m: Any) -> npt.NDArray[np.float64]:
    """The robot at the head-camera calibration capture's logged joints (4 Oct 2026, the head pitched down to a table;
    tests/assets/render/s3_cal/head_pose.json) on the floor - where the sweep's camera looks from."""
    hp = json.loads(CAPTURE_POSE.read_text())
    pose = np.array([hp["joints"][n] for n in joints.JOINTS])
    return np.asarray(model.stand_qpos(m, pose), dtype=np.float64).copy()


def run_board(b: BoardSpec, *, soft_sigma: float = SOFT_SIGMA_PX) -> dict[str, Any]:
    """The S12 sweep for one board: per distance the detection / coverage / corner error (sharp and soft), and the lens
    re-fit from all sharp detections."""
    import cv2
    from PIL import Image

    cal = lens.load_calibration()
    M = cal.model
    w = world.build_world("real", with_scene=False, head_camera=True, extra=lambda spec, _i: add_board(spec, b))
    m = w.model
    d = mujoco.MjData(m)
    d.qpos[:] = capture_qpos(m)
    mujoco.mj_forward(m, d)
    mocap = int(m.body_mocapid[m.body("charuco").id])
    C = np.c_[corners_mm(b) * MM, np.zeros(b.n_corners)]
    grid = (8, 6)  # image cells for the coverage count
    per_dist: list[dict[str, Any]] = []
    all_cells: dict[str, set[tuple[int, int]]] = {"sharp": set(), "soft": set()}
    fit_obj: list[npt.NDArray[np.float32]] = []
    fit_img: list[npt.NDArray[np.float32]] = []
    with lens.HeadCamera(m, cal) as hc:
        T_wc = lens.T_world_cam(m, d, hc.name)
        for z in DISTANCES_M:
            stats: dict[str, Any] = {"z_m": z, "poses": 0}
            for variant in ("sharp", "soft"):
                stats[variant] = {"in_view": 0, "found": 0, "err": [], "cells": set(), "boards": 0}
            for u, v in PLACES:
                for _tname, axis, deg in TILTS:
                    T_cb = board_in_camera(b, cal, u, v, z, axis, deg)
                    T_wb = T_wc @ T_cb
                    d.mocap_pos[mocap] = T_wb[:3, 3]
                    d.mocap_quat[mocap] = _mat_to_quat(T_wb[:3, :3])
                    mujoco.mj_forward(m, d)
                    gray = np.asarray(Image.open(io.BytesIO(lens.jpeg(hc.rgb(d)))).convert("L"))
                    P = (T_cb @ np.c_[C, np.ones(len(C))].T).T[:, :3]
                    truth = lens.project(M, P)
                    inside = (
                        np.isfinite(truth).all(axis=1)
                        & (truth[:, 0] > 2)
                        & (truth[:, 0] < M.width - 3)
                        & (truth[:, 1] > 2)
                        & (truth[:, 1] < M.height - 3)
                    )
                    stats["poses"] += 1
                    for variant in ("sharp", "soft"):
                        g = gray
                        if variant == "soft":
                            g = np.asarray(cv2.GaussianBlur(gray, (0, 0), soft_sigma), dtype=np.uint8)
                        ids, px = detect(g, b)
                        s = stats[variant]
                        s["in_view"] += int(inside.sum())
                        ok = ids < b.n_corners
                        ids, px = ids[ok], px[ok]
                        s["found"] += len(ids)
                        if len(ids):
                            s["boards"] += 1
                            e = np.linalg.norm(px - truth[ids], axis=1)
                            s["err"].extend(float(x) for x in e)
                            for p in px:
                                s["cells"].add((int(p[0] * grid[0] / M.width), int(p[1] * grid[1] / M.height)))
                            if variant == "sharp" and len(ids) >= MIN_CORNERS_FOR_FIT:
                                fit_obj.append(C[ids].astype(np.float32))
                                fit_img.append(px.astype(np.float32))
            row: dict[str, Any] = {"z_m": z, "poses": stats["poses"]}
            for variant in ("sharp", "soft"):
                s = stats[variant]
                all_cells[variant] |= s["cells"]
                err = np.asarray(s["err"])
                row[variant] = {
                    "found_frac": round(s["found"] / max(1, s["in_view"]), 3),
                    "boards_found": s["boards"],
                    "corners_found": s["found"],
                    "coverage": round(len(s["cells"]) / (grid[0] * grid[1]), 3),
                    "err_px_p50": round(float(np.median(err)), 3) if len(err) else None,
                    "err_px_p95": round(float(np.percentile(err, 95)), 3) if len(err) else None,
                }
            per_dist.append(row)
    fit: dict[str, Any] = {"views": len(fit_obj)}
    if len(fit_obj) >= 3:
        K0 = np.array([[150.0, 0, M.width / 2], [0, 150.0, M.height / 2], [0, 0, 1]])
        flags = (
            cv2.CALIB_USE_INTRINSIC_GUESS
            | cv2.CALIB_FIX_ASPECT_RATIO
            | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
        )
        rms, K, dist, _, _ = cv2.calibrateCamera(fit_obj, fit_img, (M.width, M.height), K0, np.zeros(5), flags=flags)
        fit.update(
            rms_px=round(float(rms), 3),
            f=round(float(K[0, 0]), 2),
            cx=round(float(K[0, 2]), 2),
            cy=round(float(K[1, 2]), 2),
            k1=round(float(np.ravel(dist)[0]), 4),
            truth={"f": M.f, "cx": M.cx, "cy": M.cy, "k1": M.k1},
        )
    return {
        "board": b.name,
        "layout": f"{b.cols} x {b.rows} squares of {b.square_mm:g} mm, markers {b.marker_mm:g} mm, {b.dict_name}",
        "paper": b.paper.label,
        "grid_mm": [b.width_mm, b.height_mm],
        "per_distance": per_dist,
        "coverage_all": {v: round(len(c) / (grid[0] * grid[1]), 3) for v, c in all_cells.items()},
        "fit": fit,
    }


def run(boards: Sequence[BoardSpec] = CANDIDATES) -> list[dict[str, Any]]:
    return [run_board(b) for b in boards]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="the ChArUco render test (B5's board choice)")
    ap.add_argument("--boards", default=",".join(b.name for b in CANDIDATES))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    want = set(a.boards.split(","))
    res = run(tuple(b for b in CANDIDATES if b.name in want))
    for r in res:
        print(f"{r['board']:16s} {r['layout']} on {r['paper']}")
        for p in r["per_distance"]:
            sh, so = p["sharp"], p["soft"]
            print(
                f"   z {p['z_m']:.2f} m: sharp found {sh['found_frac']:.2f} cover {sh['coverage']:.2f} "
                f"err p50/p95 {sh['err_px_p50']}/{sh['err_px_p95']} px | soft found {so['found_frac']:.2f} "
                f"cover {so['coverage']:.2f}"
            )
        print(f"   coverage over all distances: {r['coverage_all']}; fit: {r['fit']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
