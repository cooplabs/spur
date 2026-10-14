"""The robot stands under the minimal stand hold (no policy) in both profiles; the render path works headless."""

from __future__ import annotations

import zlib

import numpy as np
import pytest
from r1sim import joints as J
from r1sim import model as M
from r1sim.control import StandHold
from r1sim.engine import Simulation
from r1sim.render import VIEW_FRONT, png_bytes, render_views

from tests.integration.test_engine import _report, _tilt_deg


@pytest.mark.parametrize("profile", ["train", "real"])
def test_robot_stands_10_s_under_the_stand_hold(profile: str) -> None:
    m = M.build_model(profile, visual=False)
    ctrl = StandHold.for_model(m)
    sim = Simulation(m, M.RobotIndex(m))
    sim.reset(M.stand_qpos(m, ctrl.pose))
    sim.add_controller(ctrl, 0.002, "stand")
    s0 = sim.read_state()
    feet0 = np.array([sim.d.site_xpos[f][:2] for f in M.RobotIndex(sim.m).foot_sites])
    sim.run_for(1.0)  # settle: the stance sags into the joint PD (train: the soles slip ~2.4 cm forward here)
    s1 = sim.read_state()
    sim.run_for(9.0)
    s = sim.read_state()
    feet = np.array([sim.d.site_xpos[f][:2] for f in M.RobotIndex(sim.m).foot_sites])
    tilt = _tilt_deg(s.base_quat) - _tilt_deg(s0.base_quat)
    dz = float(s.base_pos[2] - s0.base_pos[2])
    settle = float(np.linalg.norm(s1.base_pos[:2] - s0.base_pos[:2]))
    drift = float(np.linalg.norm(s.base_pos[:2] - s1.base_pos[:2]))
    _report(
        f"stand_10s_{profile}",
        {
            "tilt_start_deg": round(_tilt_deg(s0.base_quat), 3),
            "tilt_change_deg": round(tilt, 3),
            "pelvis_z_start_m": round(float(s0.base_pos[2]), 4),
            "pelvis_dz_m": round(dz, 4),
            "pelvis_xy_settle_first_1s_m": round(settle, 4),
            "pelvis_xy_drift_1_to_10s_m": round(drift, 4),
            "feet_slip_m": round(float(np.abs(feet - feet0).max()), 4),
            "ankle_pitch_used": round(float(ctrl.pose[J.IDX["left_ankle_pitch"]]), 4),
        },
    )
    assert abs(tilt) < 5.0
    assert abs(dz) < 0.03
    assert drift < 0.02


@pytest.mark.parametrize("profile", ["train", "real"])
def test_plain_pose_stand_falls_backwards_without_balance(profile: str) -> None:
    """Why StandHold centres the CoM: the policy's default pose (Unitree's stand lock, which needs an active balance
    controller) under a plain stiff PD tips over backwards - the mechanism of the 4 Oct 2026 fall on `lock`."""
    m = M.build_model(profile, visual=False)
    ctrl = StandHold.for_model(m, centred=False)
    sim = Simulation(m, M.RobotIndex(m))
    sim.reset(M.stand_qpos(m, ctrl.pose))
    sim.add_controller(ctrl, 0.002, "stand")
    sim.run_for(5.0)
    s = sim.read_state()
    assert _tilt_deg(s.base_quat) > 30.0


def _png_size_and_pixels(data: bytes) -> tuple[int, int, np.ndarray]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(data):
        n = int.from_bytes(data[pos : pos + 4], "big")
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + n]
        if tag == b"IHDR":
            w, h = int.from_bytes(body[:4], "big"), int.from_bytes(body[4:8], "big")
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + 3 * w)
    assert np.all(raw[:, 0] == 0)
    return w, h, raw[:, 1:].reshape(h, w, 3)


def test_png_writer_round_trip() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(7, 5, 3), dtype=np.uint8)
    w, h, px = _png_size_and_pixels(png_bytes(img))
    assert (w, h) == (5, 7)
    assert np.array_equal(px, img)


@pytest.mark.needs_gl
def test_headless_render_shows_the_robot() -> None:
    m = M.build_model("real", visual=False)
    ctrl = StandHold.for_model(m)
    sim = Simulation(m, M.RobotIndex(m))
    sim.reset(M.stand_qpos(m, ctrl.pose))
    img = render_views(sim.m, sim.d, (VIEW_FRONT,), width=160, height=120)
    assert img.shape == (120, 160, 3)
    centre = img[30:110, 60:100].astype(float).mean(axis=-1)
    assert centre.std() > 5.0  # the robot is in the middle of the front view, not an empty floor
