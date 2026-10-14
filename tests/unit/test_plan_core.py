"""The planner pipeline (DESIGN.md §7.2): whole goals planned before anything moves, refusals with codes, the
executor's knot rules, timing limits met, the result's predicted landing."""

from __future__ import annotations

import numpy as np
import pytest
from r1mc3.plan import PlanGoal, PlanResult, Segment, plan
from r1mc3.plan.core import START_TOL_RAD
from r1mc3.plan.timing import A_MAX, V_MAX, Timed, sample
from r1mc3.world import Container, Object, Pose, Surface, WorldModel, WorldSnapshot
from r1sim import joints as J


@pytest.fixture(scope="module")
def snap() -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.55, 0.0, 0.76), size=(0.6, 1.2, 0.04)))
    w.declare(Container("bin", Pose.xyz_yaw(0.45, 0.30, 0.76), size_outer=(0.35, 0.26, 0.045)))
    w.declare(Object("box", Pose.xyz_yaw(0.40, -0.15, 0.785), size=(0.025, 0.025, 0.025)))
    return w.snapshot()


def _check_segments(res: PlanResult, q_now: np.ndarray) -> None:
    side = res.segments[0].arm
    idx = J.LEFT_ARM if side == "left" else J.RIGHT_ARM
    assert np.max(np.abs(res.segments[0].q[0] - q_now[idx])) <= START_TOL_RAD
    for a, b in zip(res.segments, res.segments[1:], strict=False):
        assert np.max(np.abs(b.q[0] - a.q[-1])) < 1e-9  # legs chain exactly
    for s in res.segments:
        assert np.all(np.diff(s.t) > 0) and s.t[0] == 0.0
        assert np.allclose(s.dq[0], 0) and np.allclose(s.dq[-1], 0)  # from rest, to rest
        _, q, dq = sample(_timed(s), 0.002)
        assert np.max(np.abs(dq)) <= V_MAX * 1.001
        acc = np.diff(dq, axis=0) / 0.002
        assert np.max(np.abs(acc)) <= A_MAX * 1.05
        assert np.all(q >= J.SOFT_MIN[idx] - 1e-6) and np.all(q <= J.SOFT_MAX[idx] + 1e-6)


def _timed(s: Segment) -> Timed:
    return Timed(s.t, s.q, s.dq, s.ddq, 0.0, 0.0, 0.0)


def test_move_joint_plans_a_clear_joint_path(snap: WorldSnapshot) -> None:
    q = J.POSE_STAND.copy()
    res = plan(snap, PlanGoal("move_joint", "left", q=(0.0, 0.4, 0.0, 0.5, 0.0)), q)
    assert res.ok and res.code == "OK" and res.plan_id and len(res.segments) == 1
    _check_segments(res, q)
    assert np.allclose(res.segments[0].q[-1], [0.0, 0.4, 0.0, 0.5, 0.0])


def test_move_joint_refusals_say_why(snap: WorldSnapshot) -> None:
    q = J.POSE_STAND.copy()
    r = plan(snap, PlanGoal("move_joint", "left", q=(-0.5, 0.3, 0.0, 0.9, 0.0)), q)  # hand into the table
    assert not r.ok and r.code == "COLLISION" and "table" in r.reason and r.segments == ()
    r = plan(snap, PlanGoal("move_joint", "left", q=(5.0, 0.3, 0.0, 0.9, 0.0)), q)
    assert not r.ok and r.code == "LIMIT"
    r = plan(snap, PlanGoal("move_joint", "left", q=(0.0, 0.3)), q)
    assert not r.ok and r.code == "INVALID_ARGS"
    r = plan(snap, PlanGoal("reach", "left", position=(0.3, 0.2, 0.9), frame="table"), q)
    assert not r.ok and r.code == "UNKNOWN_FRAME"


def test_reach_lands_the_pinch_point_on_the_target(snap: WorldSnapshot) -> None:
    q = J.POSE_STAND.copy()
    tgt = (0.35, 0.10, 0.90)
    res = plan(snap, PlanGoal("reach", "left", position=tgt), q)
    assert res.ok, res.reason
    _check_segments(res, q)
    assert np.linalg.norm(np.array(res.predicted["point_m"]) - tgt) < 0.003
    assert res.clearance_m is not None and res.clearance_m >= 0.0


def test_reach_out_of_range_is_unreachable(snap: WorldSnapshot) -> None:
    r = plan(snap, PlanGoal("reach", "left", position=(1.2, 0.2, 0.85)), J.POSE_STAND.copy())
    assert not r.ok and r.code == "UNREACHABLE" and r.details["error_mm"] > 500


def test_the_path_starts_at_the_executors_target(snap: WorldSnapshot) -> None:
    """The executor refuses a first knot > 0.05 rad from its target: with the arm sagging 0.04 rad below its target
    (measured != target), the plan starts at the TARGET r1d passes."""
    q = J.POSE_STAND.copy()
    target = q[J.LEFT_ARM].copy()
    q[J.LEFT_ARM] = target + np.array([0.04, 0.0, 0.0, -0.04, 0.0])  # the measured pose (sag)
    r = plan(snap, PlanGoal("move_joint", "left", q=(0.0, 0.4, 0.0, 0.5, 0.0)), q, arm_target=target)
    assert r.ok and np.allclose(r.segments[0].q[0], target)
    bad = plan(snap, PlanGoal("move_joint", "left", q=(0.0, 0.4, 0.0, 0.5, 0.0)), q, arm_target=np.zeros(3))
    assert not bad.ok and bad.code == "INVALID_ARGS"
