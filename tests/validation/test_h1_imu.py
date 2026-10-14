"""H1 (HW2_PREP.md decision H1): the IMU-to-pelvis rotation from the waist axes (r1sim.tier_b.h1_imu) recovers a KNOWN
mounting rotation from synthetic gyro data made with the model's own waist kinematics (noise at HW1's level: the
quiet gyro's std 0.007 rad/s; a bias; the pelvis taking 0.8 / 0.95 of the joint rates as on the cradle)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from r1sim import world
from r1sim.tier_b import h1_imu as H


def _rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = (math.radians(v) for v in (roll, pitch, yaw))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return np.asarray(rz @ ry @ rx)


@pytest.fixture(scope="module")
def m() -> object:
    return world.build_world("real", with_scene=False, visual=False).model


def _segment(m: object, r_true: np.ndarray, rng: np.random.Generator, which: str) -> H.Segment:
    t = np.arange(0.0, 12.0, 0.002)
    s = 0.1 * np.sin(2 * math.pi * 0.4 * t) + 0.05 * np.sin(2 * math.pi * 2.3 * t)
    qr = s if which == "roll" else np.full_like(t, 0.01)
    qy = 2.0 * s if which == "yaw" else np.zeros_like(t)
    vr, vy = np.gradient(qr, t), np.gradient(qy, t)
    a_roll, a_yaw = H.waist_axes(m, qr)
    w_pel = -(0.8 * vr[:, None] * a_roll[None, :] + 0.95 * vy[:, None] * a_yaw)
    gyro = w_pel @ r_true + np.array([0.003, -0.002, 0.005]) + rng.normal(0.0, 0.007, (len(t), 3))  # R^T w (rows)
    return (np.asarray(t, dtype=np.float64), np.asarray(qr, dtype=np.float64), np.asarray(qy, dtype=np.float64),
            np.asarray(gyro, dtype=np.float64))  # fmt: skip


@pytest.mark.parametrize("truth", [(0.50, 0.96, 0.0), (-1.2, 2.0, 0.7)])
def test_h1_recovers_a_known_mounting_rotation(m: object, truth: tuple[float, float, float]) -> None:
    rng = np.random.default_rng(3)
    r_true = _rot(*truth)
    segs = [_segment(m, r_true, rng, "roll"), _segment(m, r_true, rng, "yaw")]
    res = H.estimate(m, segs, n_boot=20)
    for got, want in zip((res.roll_deg, res.pitch_deg, res.yaw_deg), truth, strict=True):
        assert got == pytest.approx(want, abs=0.1), res
    assert res.k_roll == pytest.approx(0.8, abs=0.02) and res.k_yaw == pytest.approx(0.95, abs=0.02)
    assert 88.0 < res.axis_angle_deg < 92.0
    assert all(0.0 < s < 0.1 for s in res.se_deg)


def test_one_axis_alone_does_not_fix_the_rotation(m: object) -> None:
    """Both waist axes are needed (two axes fix a rotation; why H1 pools the roll AND the yaw recordings): one axis
    alone is refused, not answered."""
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="both waist axes"):
        H.estimate(m, [_segment(m, _rot(0.5, 0.96, 0.0), rng, "yaw")], n_boot=0)
