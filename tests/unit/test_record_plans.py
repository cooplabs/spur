"""The HW1 recording plans (python/r1mc3/record/plans.py): each kind's setpoints, its checks before anything moves
(limits with a margin, the executor's speed cap, one joint at a time) and its description."""

from __future__ import annotations

import math

import numpy as np
import pytest
from r1mc3.record.plans import DQ_MAX, KINDS, PlanError, plan
from r1mc3.service.codes import JOINT_NAMES

LIMITS = {n: (-1.0, 1.0) for n in JOINT_NAMES}
Q26 = np.zeros(26)


def test_step_holds_then_steps_one_joint() -> None:
    p = plan("step", {"joint": "left_knee", "amplitude_rad": 0.1, "hold_s": 1.0, "repeats": 1}, Q26, LIMITS)
    assert p.group == "leg_left" and p.joints == (3,) and p.duration_s == pytest.approx(5.0)
    k = 3  # left_knee within leg_left (0..5)
    assert [p.at(t)[0][k] for t in (0.5, 1.5, 2.5, 3.5, 4.5)] == pytest.approx([0, 0.1, 0, -0.1, 0])
    q = p.at(1.5)[0]
    assert np.count_nonzero(q) == 1  # one joint at a time


def test_chirp_is_continuous_and_within_dq_max() -> None:
    p = plan("chirp", {"joint": "left_elbow", "amplitude_rad": 0.03, "f0_hz": 0.5, "f1_hz": 5.0, "duration_s": 10}, Q26)
    ts = np.arange(0.0, 10.0, 0.002)
    k = 3  # left_elbow within arm_left
    q = np.array([p.at(t)[0][k] for t in ts])
    dq = np.array([p.at(t)[1][k] for t in ts])
    assert abs(q[0]) < 1e-12 and abs(p.at(10.0)[0][k]) < 1e-9  # faded in and out
    assert np.max(np.abs(dq)) <= p.summary["peak_dq_rad_s"] + 1e-9 <= DQ_MAX
    np.testing.assert_allclose(np.gradient(q, ts)[5:-5], dq[5:-5], atol=2e-3)  # dq is q's derivative
    assert p.describe()["moves"] == ["left_elbow"]


def test_too_fast_or_out_of_limits_is_refused_before_moving() -> None:
    with pytest.raises(PlanError, match="dq_max"):
        plan("chirp", {"joint": "left_elbow", "amplitude_rad": 0.2, "f1_hz": 10.0}, Q26)
    with pytest.raises(PlanError, match="limits") as e:
        plan("step", {"joint": "left_knee", "amplitude_rad": 0.1}, np.full(26, 0.9), LIMITS)
    assert e.value.details["joint"] == "left_knee"
    with pytest.raises(PlanError, match="joint"):
        plan("step", {"joint": "elbow"}, Q26)
    with pytest.raises(PlanError, match="kind"):
        plan("dance", {}, Q26)


def test_ramp_is_the_stiction_protocol() -> None:
    p = plan("ramp", {"joint": "left_knee", "amplitude_rad": 0.1, "speeds": [0.1]}, Q26, LIMITS)
    # settle 1.5, to +A (1 s), hold 1, to -A (2 s), hold 1, to 0 (1 s), hold 1
    assert p.duration_s == pytest.approx(1.5 + 1 + 1 + 2 + 1 + 1 + 1)
    k = 3
    assert p.at(1.0)[0][k] == 0.0 and p.at(2.0)[0][k] == pytest.approx(0.05) and p.at(2.0)[1][k] == pytest.approx(0.1)
    assert p.at(3.0)[0][k] == pytest.approx(0.1) and p.at(3.0)[1][k] == 0.0  # holding at +A
    assert p.at(4.5)[1][k] == pytest.approx(-0.1) and p.at(5.75)[0][k] == pytest.approx(-0.1)
    assert p.gains is not None and p.gains[0][k] == 40.0 and p.gains[1][k] == 1.0  # the knee's soft servo
    assert np.count_nonzero(p.gains[0]) == 1  # the rest of the group at its hold gains (0 / 0)
    waist = plan("ramp", {"joint": "waist_roll"}, Q26)
    assert waist.gains is not None and waist.gains[0][0] == 150.0  # waist roll carries the torso


def test_static_pose_minimum_jerk() -> None:
    goal = [0.2, 0.3, 0.0, 0.5, 0.0]
    p = plan("static_pose", {"group": "arm_left", "q": goal, "move_s": 2.0, "hold_s": 5.0, "payload_kg": 0.5}, Q26)
    assert p.at(0.0)[0] == pytest.approx(np.zeros(5)) and p.at(2.0)[0] == pytest.approx(goal)
    assert p.at(1.0)[1][3] == pytest.approx(0.5 * 1.875 / 2.0)  # the minimum-jerk peak speed at mid-move
    assert p.params["payload_kg"] == 0.5 and p.duration_s == 7.0


def test_hand_plans_and_imu_static_move_no_body_joint() -> None:
    fh = plan("finger_hold", {"side": "left", "finger": "index", "torque_nm": 0.45, "duration_s": 30}, None)
    assert fh.group is None and fh.hand == {"side": "left", "finger": "index", "motors": [5, 6], "torque_nm": 0.45}
    with pytest.raises(PlanError):
        plan("finger_hold", {"side": "left", "torque_nm": 2.0}, None)  # above the finger's 1.4 N m
    pp = plan("pad_press", {"side": "right", "pad": 4, "weights_g": [50, 100]}, None)
    assert pp.needs_mode is None and len(pp.operator) == 2
    im = plan("imu_static", {"duration_s": 10}, None)
    assert im.joints == () and im.needs_mode is None
    assert set(KINDS) == {"step", "ramp", "chirp", "static_pose", "imu_static", "pad_press", "finger_hold"}
    peak = plan("chirp", {"joint": "head_yaw"}, None).summary["peak_dq_rad_s"]
    assert math.isclose(peak, 0.03 * (2 * math.pi * 5 + 0.5 * math.pi / 0.5), rel_tol=1e-4)
