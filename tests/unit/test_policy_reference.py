"""r1mc3.policy.reference.ReferencePolicy: the metadata the runner reads (gains, default pose, scales, observation
layout) and its schedule rules (the lean ramp, the phase clock, the command composition), deterministic.
(Its bit-for-bit equality with the previous controller's runner is an internal validation.)"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.policy import POLICY_DIR, ReferencePolicy
from r1sim import joints as J


@pytest.fixture(scope="module")
def pol() -> ReferencePolicy:
    return ReferencePolicy.load(POLICY_DIR)


def test_metadata(pol: ReferencePolicy) -> None:
    # DESIGN §3.2 / V2_INVENTORY App. E §3.4-3.5: legs/waist 84.302/5.367, ankles 31.385/1.998, arms 60/60/45/45/30
    assert pol.names == [J.JOINTS[i] for i in range(24)]
    assert pol.act_names == list(J.JOINTS[:14])
    assert list(pol.idx) == list(range(24)) and list(pol.act_idx26) == list(range(14))
    legs = [84.302, 84.302, 84.302, 84.302, 31.385, 31.385]
    arm = [60.0, 60.0, 45.0, 45.0, 30.0]
    assert np.array_equal(pol.kp, legs + legs + [84.302, 84.302] + arm + arm)
    assert np.array_equal(pol.kd[:6], [5.367] * 4 + [1.998] * 2) and np.array_equal(pol.kd[14:19], [2, 2, 2, 2, 1.5])
    assert np.allclose(pol.default, J.POSE_STAND[:24], atol=5e-4)  # = R1MC_MANIP2_HOME, the policy's stance
    assert np.array_equal(pol.scale, [0.178] * 4 + [0.398] * 2 + [0.178] * 4 + [0.398] * 2 + [0.178] * 2)
    assert pol.obs_names == [
        "base_ang_vel", "projected_gravity", "command", "lean", "phase", "joint_pos", "joint_vel", "actions",
    ]  # fmt: skip
    assert pol.period == 0.86 and pol.step_dt == 0.02 and pol.arms_external
    assert pol.lean_range == (0.0, 0.087) and pol.waist_range is None
    assert np.array_equal(pol.cmd_lo, [-0.5, -0.4, -1.0]) and np.array_equal(pol.cmd_hi, [0.8, 0.4, 1.0])


def _state(k: int) -> dict[str, Any]:
    rng = np.random.default_rng(k)
    quat = np.array([1.0, 0.02 * np.sin(k / 30), 0.03 * np.cos(k / 40), 0.01])
    return {
        "q": J.POSE_STAND + 0.05 * rng.normal(size=26),
        "dq": 0.5 * rng.normal(size=26),
        "quat": quat / np.linalg.norm(quat),
        "gyro": 0.2 * rng.normal(size=3),
    }


def test_observation_layout_and_phase_zeroing(pol: ReferencePolicy) -> None:
    pol.reset()
    s = _state(1)
    q, dq = s["q"][pol.idx], s["dq"][pol.idx]
    pol.phase_t = 0.2
    o = pol.observe(q, dq, s["quat"], s["gyro"], np.array([0.05, 0.0, 0.0]))
    assert o.shape == (74,)
    assert np.array_equal(o[0:3], s["gyro"]) and np.array_equal(o[6:9], [0.05, 0.0, 0.0])
    assert o[9] == 0.0  # lean
    assert np.array_equal(o[10:12], [0.0, 0.0])  # |cmd| < 0.1: phase zeroed
    assert np.array_equal(o[12:36], q - pol.default) and np.array_equal(o[36:60], dq)
    o2 = pol.observe(q, dq, s["quat"], s["gyro"], np.array([0.3, 0.0, 0.0]))
    ph = 0.2 / 0.86
    assert np.allclose(o2[10:12], [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
    g = o2[3:6]
    assert abs(np.linalg.norm(g) - 1.0) < 1e-12 and g[2] < -0.99


def test_lean_ramp_and_schedule(pol: ReferencePolicy) -> None:
    pol.reset()
    assert pol.set_body_targets(lean_deg=4.0) is None
    assert pol.set_body_targets(lean_deg=9.0) is not None  # outside 0..5 deg: refused, nothing changed
    zero = np.zeros(3)
    for k in range(500):  # 1 s at 500 Hz -> 50 policy steps
        pol.step(_state(0), zero, k * 0.002)
    assert pol.policy_steps == 50
    assert abs(pol.lean - np.radians(4.0)) < 1e-12  # 50 steps of 4 deg/s x 20 ms reach the 4 deg target
    for k in range(500, 525):  # walking: ramps back toward 0
        pol.step(_state(0), np.array([0.4, 0.0, 0.0]), k * 0.002)
    assert pol.lean < np.radians(4.0)


def test_step_command_composition(pol: ReferencePolicy) -> None:
    pol.reset()
    s = _state(5)
    q26, kp26, kd26 = pol.step(s, np.zeros(3), 0.0)
    assert np.array_equal(q26[:14], pol.target) and np.array_equal(q26[14:], s["q"][14:])
    assert np.array_equal(kp26[:24], pol.kp) and np.array_equal(kp26[24:], J.KP_STIFF[24:])
    assert np.array_equal(kd26[:24], pol.kd) and np.array_equal(kd26[24:], J.KD_STIFF[24:])


def test_deterministic() -> None:
    a, b = ReferencePolicy.load(POLICY_DIR), ReferencePolicy.load(POLICY_DIR)
    for k in range(200):
        ra = a.step(_state(k), np.array([0.2, 0.0, 0.1]), k * 0.002)
        rb = b.step(_state(k), np.array([0.2, 0.0, 0.1]), k * 0.002)
        assert all(np.array_equal(x, y) for x, y in zip(ra, rb, strict=True))


def _cmd_script(k: int) -> np.ndarray:
    """0 -> below the 0.1 phase threshold -> walking (also beyond the limits, clipped) -> turning -> 0."""
    t = k * 0.002
    if t < 0.6:
        return np.zeros(3)
    if t < 1.0:
        return np.array([0.05, 0.03, 0.0])
    if t < 1.8:
        return np.array([0.9, -0.5, 0.0])
    if t < 2.4:
        return np.array([0.0, 0.0, 0.6])
    return np.zeros(3)
