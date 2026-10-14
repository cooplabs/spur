"""The head-camera mount fit from a still ChArUco board at several head poses (session 12, `r1mc3.measure.head_mount`):
on synthetic frames through the REAL head chain (r1mc3.world.Geometry: the measured robot, the head pitch / yaw) and
the lens of HW2's C3 recording, a known mount error (1.5 deg, ~16 mm) and an unknown board pose are recovered from 12
head poses with 0.3 px corner noise; a moved board / a wrong lens is caught (RMS); the board's level / height check.
New module (the base 1cf3afd has no head_mount: every test here fails there)."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from r1mc3.cam.pointing import project
from r1mc3.measure import head_mount as hm
from r1mc3.measure.phone_calib import BOARD
from r1mc3.world import Geometry
from r1sim import joints as J

ROOT = Path(__file__).resolve().parents[2]
CAL = json.loads((ROOT / "models" / "head_camera" / "calibration_s3_cal_20261004_v3frame.json").read_text())
C3_LENS: dict[str, Any] = {"lens": "pinhole", "f": 124.971, "cx": 257.902, "cy": 205.168, "k1": -0.002203, "depth": "z"}


def _board_points() -> np.ndarray:
    pytest.importorskip("cv2")
    from r1mc3.measure.phone_calib import board_and_detector

    board, _ = board_and_detector(BOARD)
    return np.asarray(board.getChessboardCorners(), dtype=np.float64)


def _site_poses(poses: list[tuple[float, float]]) -> list[np.ndarray]:
    geo = Geometry(None)
    sid = mujoco.mj_name2id(geo.m, mujoco.mjtObj.mjOBJ_SITE, "head_camera")
    out = []
    for p, y in poses:
        q = np.asarray(J.POSE_STAND, dtype=np.float64).copy()
        q[J.IDX["head_pitch"]], q[J.IDX["head_yaw"]] = p, y
        geo.set_stance(q, np.array([1.0, 0.0, 0.0, 0.0]))
        T = np.eye(4)
        T[:3, :3] = np.asarray(geo.d.site_xmat[sid]).reshape(3, 3)
        T[:3, 3] = np.asarray(geo.d.site_xpos[sid])
        out.append(T)
    return out


def _frames(T_mount: np.ndarray, T_board: np.ndarray, model: dict[str, Any], *, noise: float = 0.3,
            seed: int = 0, move_board_after: int | None = None) -> list[hm.MountFrame]:  # fmt: skip
    rng = np.random.default_rng(seed)
    obj = _board_points()
    poses = hm.head_grid()
    frames = []
    for k, ((p, y), T_site) in enumerate(zip(poses, _site_poses(poses), strict=True)):
        Tb = T_board.copy()
        if move_board_after is not None and k >= move_board_after:
            Tb[:3, 3] += np.array([0.02, 0.0, 0.0])  # someone nudged the board 2 cm
        T_cb = np.linalg.inv(T_site @ T_mount) @ Tb
        px = project(model, (T_cb[:3, :3] @ obj.T).T + T_cb[:3, 3])
        ok = np.isfinite(px).all(axis=1) & (px[:, 0] > 2) & (px[:, 0] < 541) & (px[:, 1] > 2) & (px[:, 1] < 445)
        if ok.sum() >= hm.MIN_CORNERS:
            frames.append(hm.MountFrame(f"{k:02d}", T_site, obj[ok], px[ok] + rng.normal(0, noise, (int(ok.sum()), 2)),
                                        (p, y)))  # fmt: skip
    return frames


def _truth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T0 = np.asarray(CAL["T_site_cam"], dtype=np.float64)
    T_true = hm._apply(T0, np.array([math.radians(1.0), math.radians(-0.8), math.radians(0.6), 0.012, -0.008, 0.006]))
    yaw = math.radians(10.0)
    T_board = np.eye(4)  # flat on a 0.752 m table, ~0.45 m ahead (the board's z up)
    T_board[:3, :3] = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    T_board[:3, 3] = [0.33, -0.12, 0.752]
    return T0, T_true, T_board


def test_a_known_mount_error_is_recovered_through_the_real_head_chain() -> None:
    T0, T_true, T_board = _truth()
    frames = _frames(T_true, T_board, C3_LENS)
    assert len(frames) >= 10  # the board stays in view over the 12-pose grid
    rep = hm.run_fit(frames, {**CAL, "model": C3_LENS}, table_z=0.752, loo=True)
    T_fit = np.asarray(rep["T_site_cam"])
    assert rep["ok"] and rep["rms_px"] < 0.45
    # ~0.1 deg (= ~1 mm at the table) with a 15-corner board and 0.3 px noise
    assert hm.angle_deg(T_fit, T_true) < 0.2 and 1000 * np.linalg.norm(T_fit[:3, 3] - T_true[:3, 3]) < 2.0
    assert hm.angle_deg(T0, T_true) > 1.0  # the start was really off
    assert rep["board"]["tilt_from_level_deg"] < 0.1 and abs(rep["board"]["height_vs_table_mm"]) < 1.0
    assert rep["left_out_pose_rms_px"]["median"] < 0.6
    # the start file as it is (the mount 1.5 deg / 16 mm off, the 4 Oct lens vs the truth's C3 lens) explains the
    # capture clearly worse than the fit (1.12 vs 0.42 px here: the free board pose absorbs part of a mount error over
    # +-0.35 rad of head motion - a sensitivity of ~1 px per ~1.5 deg, said in the runbook words)
    assert rep["file_as_is_rms_px"] > 2 * rep["rms_px"]


def test_a_board_that_moved_or_a_wrong_lens_is_not_a_calibration() -> None:
    _T0, T_true, T_board = _truth()
    moved = _frames(T_true, T_board, C3_LENS, move_board_after=6)
    rep = hm.run_fit(moved, {**CAL, "model": C3_LENS}, loo=False)
    assert not rep["ok"] and rep["rms_px"] > hm.GOOD_RMS_PX
    wrong = _frames(T_true, T_board, C3_LENS)
    rep2 = hm.run_fit(wrong, {**CAL, "model": {**C3_LENS, "f": 150.05, "cx": 271.9, "cy": 223.9}}, loo=False)
    assert not rep2["ok"] and rep2["rms_px"] > 1.0  # day 1's stuck lens (f 150.05, the image centre) does not fit


def test_the_head_grid_stays_inside_r1ds_head_limit() -> None:
    g = hm.head_grid()
    assert len(g) == 12 and max(p for p, _ in g) <= 0.5783 and min(p for p, _ in g) >= 0.0
    assert all(abs(y) <= 0.35 for _, y in g)
    # serpentine: consecutive poses differ in one joint by at most one grid step
    steps = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in itertools.pairwise(g)]
    assert max(steps) <= 0.25


def test_a_tape_measured_board_pose_fixes_the_mount_to_the_tapes_accuracy() -> None:
    """The accurate variant: the board's pose known (flat on the table, tape-measured) -> only the mount is fitted."""
    _T0, T_true, T_board = _truth()
    frames = _frames(T_true, T_board, C3_LENS)
    exact = hm.fit_mount(C3_LENS, frames, np.asarray(CAL["T_site_cam"]), board=T_board)
    assert exact["board_known"] and hm.angle_deg(exact["T_site_cam"], T_true) < 0.25  # 0.3 px noise: ~0.14 deg
    assert 1000 * np.linalg.norm(exact["T_site_cam"][:3, 3] - T_true[:3, 3]) < 2.0
    taped = T_board.copy()
    taped[:3, 3] += np.array([0.003, -0.002, 0.0])  # the tape 3 / 2 mm off
    t = hm.fit_mount(C3_LENS, frames, np.asarray(CAL["T_site_cam"]), board=taped)
    assert 1000 * np.linalg.norm(t["T_site_cam"][:3, 3] - T_true[:3, 3]) < 6.0  # ~ the tape's error, not more
    bp = hm.board_pose_flat(0.33, -0.12, 0.752, 10.0)
    assert np.allclose(bp, T_board, atol=1e-9)
