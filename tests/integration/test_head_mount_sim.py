"""The head-camera mount capture + fit end to end on the DDS-bridge PROCESS stack with the simulated camera (session
12): the robot hanging in ON_STAND, the printed board hung still in front of the head, `r1mc3.measure.head_mount
capture` moves the head through its 3 x 4 grid with r1d's public `stream.head` and takes frames with `camera.observe`;
`fit` then explains the frames with the camera mount the simulator renders through (the robot's v3-frame file).
THE HONEST LIMIT (found here): over the head's own small rotations a mount shift and a board shift look almost alike -
the free-board fit's mount comes back ~0.3-0.5 deg / ~9-15 mm from the truth with an RMS barely below the truth's
(the full 12-pose grid: 0.26 deg / 8.6 mm, RMS 0.40 vs 0.43 px) - so a mount for the robot needs the board at a
tape-measured pose (`fit --board-pose`, test_head_mount.py) or known marks; the free fit is a check, not a calibration.
New module (absent on the base 1cf3afd)."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
from r1sim import stack

pytestmark = [pytest.mark.slow, pytest.mark.needs_dds]
if not stack.available():
    pytest.skip("the DDS build of r1rt / r1guard or CycloneDDS's Python binding is missing", allow_module_level=True)
pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "models" / "head_camera" / "calibration_s3_cal_20261004_v3frame.json"  # what the sim renders through


def _hang_board(st: stack.ProcStack, dist_m: float = 0.55) -> None:
    from r1sim import charuco_render as cr

    T = st.camera.T_world_cam()
    b = cr.CHOSEN
    R = T[:3, :3]
    centre = T[:3, 3] + dist_m * R[:, 2]
    origin = centre - R @ np.array([b.width_mm * cr.MM / 2, b.height_mm * cr.MM / 2, 0.0])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    m, d = st.bridge.sim.m, st.bridge.sim.d
    k = int(m.body_mocapid[m.body("charuco").id])
    with st.bridge.sim._lock:
        d.mocap_pos[k], d.mocap_quat[k] = origin, q


def test_the_mount_capture_and_fit_run_through_r1d_and_recover_the_sims_mount(tmp_path: Path) -> None:
    from r1mc3.measure import head_mount as hm
    from r1sim import charuco_render as cr

    bridge = {"stance": "hanging", "scene": "none", "builtin": "stand_lock"}
    cam = {"extra": lambda spec, _i: cr.add_board(spec, cr.CHOSEN)}
    poses = hm.head_grid()  # the default 3 x 4 grid
    with stack.ProcStack(
        domain=107, workdir=tmp_path / "stack", bridge=bridge, camera=cam, rt_args=["--stand-session"],
        shm_name=f"/r1mc3.cam.{os.getpid()}.7",
    ) as st:  # fmt: skip
        assert st.cli("on-stand")["ok"]
        st.wait_mode("on_stand", 20)
        assert st.cli("head", "0.42,0.0", "--ttl", "0.3")["ok"]
        time.sleep(1.0)
        _hang_board(st)
        rows = hm.capture(st.port, tmp_path / "cap", poses, settle_s=1.2, frames_per_pose=1, say=lambda _s: None)
    assert len(rows) == len(poses) and all("measured state" in str(r["pose_source"]) for r in rows)
    frames, skipped = hm.load_frames(tmp_path / "cap")
    assert len(frames) >= 10, skipped
    calib = json.loads(FILE.read_text())
    rep = hm.run_fit(frames, calib, loo=False)
    assert rep["file_as_is_rms_px"] < 1.0, rep  # the sim renders through this very file: it explains the frames
    assert rep["rms_px"] <= rep["file_as_is_rms_px"] + 1e-6 and rep["ok"], rep
    T_file = np.asarray(calib["T_site_cam"])
    off = hm._apply(T_file, np.array([math.radians(1.0), math.radians(-0.8), math.radians(0.6), 0.012, -0.008, 0.006]))
    rep2 = hm.run_fit(frames, {**calib, "T_site_cam": off.tolist()}, loo=False)
    T_back = np.asarray(rep2["T_site_cam"])
    # from 1.5 deg / 16 mm off: back within the free fit's honest limit (rotation well, translation loosely)
    assert hm.angle_deg(T_back, T_file) < 0.7 and 1000 * np.linalg.norm(T_back[:3, 3] - T_file[:3, 3]) < 16.0, rep2
