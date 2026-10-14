"""HW2's camera block (C2 / C4 / C5) on the DDS-bridge PROCESS stack with the simulated camera (session 11): the
robot hanging in ON_STAND, `r1 record run flat_field` / `depth_targets` through r1d -> r1cam's recordings (left +
right + depth, the robot's MCAP format) -> `camera_fit vignetting / depth / latency` read them; the latency step = head
yaw steps (`r1 head ... --ttl 0.3`) while the printed board hangs in view, the board's image motion vs the head yaw in
r1log -> the camera's modelled latency comes back. Base (1e55bdd): no camera on the process stack (ProcStack has no
`camera`), no MCAP readers in camera_fit for these fits."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from r1sim import stack

pytestmark = [pytest.mark.slow, pytest.mark.needs_dds]
if not stack.available():
    pytest.skip("the DDS build of r1rt / r1guard or CycloneDDS's Python binding is missing", allow_module_level=True)
pytest.importorskip("cv2")

LATENCY_S = 0.05  # the modelled exposure -> receive latency


def _place_board(st: Any, dist_m: float = 0.6) -> None:
    """The ChArUco board (a mocap body) squarely in front of the head camera, dist_m along its axis."""
    from r1sim import charuco_render as cr

    cam = st.camera
    T = cam.T_world_cam()
    b = cr.CHOSEN
    R = T[:3, :3]  # board frame = the camera's (x right, y down, z into the board)
    centre = T[:3, 3] + dist_m * R[:, 2]
    origin = centre - R @ np.array([b.width_mm * cr.MM / 2, b.height_mm * cr.MM / 2, 0.0])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    m, d = st.bridge.sim.m, st.bridge.sim.d
    k = int(m.body_mocapid[m.body("charuco").id])
    with st.bridge.sim._lock:  # the physics reads mocap each step; a 7-number write
        d.mocap_pos[k], d.mocap_quat[k] = origin, q


def test_the_camera_block_records_and_fits_on_real_processes(tmp_path: Path) -> None:
    from r1sim import charuco_render as cr
    from r1sim.sysid import camera_fit as CF

    bridge = {"stance": "hanging", "scene": "none", "builtin": "stand_lock"}
    cam = {"latency_s": LATENCY_S, "extra": lambda spec, _i: cr.add_board(spec, cr.CHOSEN)}
    with stack.ProcStack(
        domain=108, workdir=tmp_path, bridge=bridge, camera=cam, r1log=True, rt_args=["--stand-session"],
        shm_name=f"/r1mc3.cam.{os.getpid()}.8",
    ) as st:  # fmt: skip
        assert st.cli("on-stand")["ok"]
        st.wait_mode("on_stand", 20)
        assert st.cli("head", "0.30,0.0", "--ttl", "0.3")["ok"]
        time.sleep(1.0)
        _place_board(st)
        # C2: the white sheet's recording (here the board scene: the frames and the fit's plumbing, not the gain)
        r = st.cli("record", "run", "flat_field", "levels=1", "frames_per_level=4", "turn=false", "--yes")
        assert r["ok"], r
        time.sleep(float(r["data"]["started"]["duration_s"]) + 1.5)  # one recording at a time (RESOURCE_BUSY)
        # C5: head yaw steps while a depth_targets recording runs and the board is in view
        out: dict[str, Any] = {}
        th = threading.Thread(target=lambda: out.update(r=st.cli(
            "record", "run", "depth_targets", "label=latency_board", "distance_m=0.6", "duration_s=9", "--yes",
            timeout_s=60)))  # fmt: skip
        th.start()
        time.sleep(1.5)
        t0 = time.time()
        while time.time() - t0 < 6.0:
            yaw = 0.12 if int((time.time() - t0) / 1.5) % 2 else -0.04
            st.cli("head", f"0.30,{yaw}", "--ttl", "0.3")
        th.join()
        assert out["r"]["ok"], out["r"]
        time.sleep(3.0)  # both recordings stopped and written
        recs = sorted((tmp_path / "records").glob("*.r1cam.mcap"))
        flat = [p for p in recs if "_cam_flat_field" in p.name]
        dep = [p for p in recs if "_cam_depth_targets_latency_board" in p.name]
        assert flat and dep, recs
        man_f, man_d = CF.manifest_of(flat[0]), CF.manifest_of(dep[0])
        assert man_f["frames"]["left"] >= 4 and man_f["frames"]["right"] >= 4  # the simulated right camera too
        assert man_d["frames"]["left"] >= 20 and man_d["frames"]["depth"] >= 20
        assert "error" not in CF.vignetting_from_mcap(flat)
        d = CF.depth_from_mcap(dep)["recordings"][0]
        assert d["distance_m"] == 0.6 and d["usable"] >= 20
        logs = sorted((tmp_path / "logs").glob("r1log_*.mcap"))
        st.stop()  # r1log's final drain before reading its file
        offset = time.monotonic_ns() - time.clock_gettime_ns(time.CLOCK_MONOTONIC)  # r1cam's clock - r1log's (macOS)
        lat = CF.latency_from_mcap(dep[0], logs, cr.CHOSEN, offset_ns=offset)
        assert "error" not in lat, lat
        assert lat["head_yaw_span_rad"] > 0.1 and lat["frames"] >= 20, lat
        # the modelled 50 ms comes back (+ r1log's state age, a few ms; 7.5 fps frames)
        assert abs(lat["latency_ms"] - 1000 * LATENCY_S) <= 25.0, lat
