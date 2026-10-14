"""Paths (DESIGN.md §7.2 step 2) and whole goals: RRT-Connect around an obstacle with dense checks, the straight line
only when asked, hover / lower / carry planned end to end before anything moves, the executor-ready segments."""

from __future__ import annotations

import numpy as np
import pytest
from r1mc3.plan import PlanGoal, ik, plan
from r1mc3.plan import path as P
from r1mc3.plan.core import GUARD_CONTACT, LOWER_SPEED
from r1mc3.world import Container, Geometry, Object, Obstacle, Pose, Surface, WorldModel, WorldSnapshot
from r1sim import joints as J

TABLE_Z = 0.76


@pytest.fixture(scope="module")
def snap() -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.55, 0.0, TABLE_Z), size=(0.6, 1.2, 0.04)))
    w.declare(Container("bin", Pose.xyz_yaw(0.45, 0.30, TABLE_Z), size_outer=(0.35, 0.26, 0.045)))
    w.declare(Object("box", Pose.xyz_yaw(0.40, -0.15, TABLE_Z + 0.025), size=(0.025, 0.025, 0.025)))
    return w.snapshot()


def test_rrt_goes_around_and_every_step_is_clear() -> None:
    w = WorldModel()
    # a post between the hanging left arm and a pose forward: the straight joint line is blocked
    w.declare(Obstacle("post", Pose.xyz_yaw(0.24, 0.20, 0.70), shape="box", size=(0.03, 0.06, 0.12)))
    geo = Geometry(w.snapshot())
    geo.set_stance(J.POSE_STAND)
    chk = P.Checker(geo, "left", ik.arm_limits("left"))
    q0 = J.POSE_STAND[J.LEFT_ARM].copy()
    q1 = np.array([-1.0, 0.25, 0.0, 0.4, 0.0])
    assert chk.valid(q0) and chk.valid(q1)
    assert not P.joint_line(chk, q0, q1).ok
    r = P.rrt_connect(chk, q0, q1, seed=3)
    assert r.ok and r.method == "rrt" and len(r.path) >= 3
    assert np.allclose(r.path[0], q0) and np.allclose(r.path[-1], q1)
    fresh = P.Checker(geo, "left", ik.arm_limits("left"))
    assert fresh.path(r.path)  # re-checked densely by a new checker
    # reproducible from its seed
    assert np.allclose(P.rrt_connect(chk, q0, q1, seed=3).path, r.path)


def test_move_linear_is_refused_when_the_line_is_blocked(snap: WorldSnapshot) -> None:
    q = J.POSE_STAND.copy()
    r = plan(snap, PlanGoal("move_linear", "left", delta=(0.0, 0.0, 0.05)), q)
    assert r.ok and r.details["methods"] == ["line"]
    # straight forward from under the table's edge: into the table
    r = plan(snap, PlanGoal("move_linear", "left", delta=(0.25, 0.0, 0.20)), q)
    assert not r.ok and r.code in ("COLLISION", "UNREACHABLE") and r.segments == ()


def test_hover_plans_lift_transit_descend_first(snap: WorldSnapshot) -> None:
    q = J.POSE_STAND.copy()
    r = plan(snap, PlanGoal("hover", "right", position=(0.40, -0.15, 0.0), height_m=0.05, allow_hand=("box",)), q)
    assert r.ok, r.reason
    assert [s.leg for s in r.segments][-1] in ("descend", "transit") and r.segments[0].leg in ("lift", "transit")
    tips = r.predicted["fingertips_m"]
    low = min(v[2] for v in tips.values())
    # height_m counts from what is under the point apart from the goal's target (allow_hand): the table here
    assert abs(low - (TABLE_Z + 0.05)) < 0.004
    assert np.hypot(r.predicted["point_m"][0] - 0.40, r.predicted["point_m"][1] + 0.15) < 0.004
    assert r.clearance_m is not None and r.clearance_m > 0.0  # the table / bin (the box is the allowed target)
    # without the target declared, the box is what is under the point: 5 cm above ITS top
    r2 = plan(snap, PlanGoal("hover", "right", position=(0.40, -0.15, 0.0), height_m=0.05), q)
    assert r2.ok, r2.reason
    low2 = min(v[2] for v in r2.predicted["fingertips_m"].values())
    assert abs(low2 - (TABLE_Z + 0.05 + 0.05)) < 0.004


def test_hover_refused_whole_when_one_leg_cannot_be_planned(snap: WorldSnapshot) -> None:
    q = J.POSE_STAND.copy()
    r = plan(snap, PlanGoal("hover", "left", position=(0.95, 0.10, 0.0), height_m=0.05), q)
    assert not r.ok and r.segments == () and r.code in ("UNREACHABLE", "COLLISION")


def test_lower_is_a_guarded_straight_descent_onto_the_surface(snap: WorldSnapshot) -> None:
    q = J.POSE_STAND.copy()
    h = plan(snap, PlanGoal("hover", "right", position=(0.35, -0.05, 0.0), height_m=0.04), q)
    assert h.ok, h.reason
    q2 = q.copy()
    q2[J.RIGHT_ARM] = h.segments[-1].q[-1]
    r = plan(snap, PlanGoal("lower", "right", max_depth_m=0.10), q2)
    assert r.ok, r.reason
    s = r.segments[0]
    assert s.leg == "descend" and s.guards & GUARD_CONTACT
    geo = Geometry(snap)
    geo.set_stance(q2)
    xy0 = None
    for qk in s.q:  # the hand point stays on a vertical line
        geo.set_arm("right", qk)
        p = geo.point("right", "pinch")
        xy0 = p[:2] if xy0 is None else xy0
        assert np.linalg.norm(p[:2] - xy0) < 0.002
    # it goes 1 cm past the surface (the contact guard ends it there)
    assert r.predicted["landing_z_m"] == pytest.approx(TABLE_Z - 0.01, abs=0.004)
    assert s.duration_s > 0.04 / LOWER_SPEED * 0.5


def test_carry_rises_travels_lowers(snap: WorldSnapshot) -> None:
    q = J.POSE_STAND.copy()
    h = plan(snap, PlanGoal("hover", "left", position=(0.33, 0.05, 0.0), height_m=0.02), q)  # "holding" there
    assert h.ok, h.reason
    q[J.LEFT_ARM] = h.segments[-1].q[-1]
    r = plan(snap, PlanGoal("carry", "left", position=(0.40, 0.30, 0.85), lift_m=0.05), q)
    assert r.ok, r.reason
    assert [s.leg for s in r.segments] == ["rise", "travel", "lower"]
    assert np.linalg.norm(np.array(r.predicted["point_m"]) - (0.40, 0.30, 0.85)) < 0.004


def test_plans_under_the_balance_policy_stay_in_its_envelope(snap: WorldSnapshot) -> None:
    from r1mc3.plan import policy_envelope
    from r1mc3.plan.timing import Timed, sample

    meta = {"shoulder_pitch": (-2.4, 0.75), "shoulder_roll": (-0.1, 1.85), "shoulder_yaw": (-1.87, 0.9),
            "elbow": (-0.93, 2.05), "wrist_roll": (-1.87, 1.87)}  # fmt: skip
    env = policy_envelope(meta, "right")
    q = J.POSE_STAND.copy()
    r = plan(snap, PlanGoal("hover", "right", position=(0.36, -0.05, 0.0), height_m=0.06), q, envelope=env)
    assert r.ok, r.reason
    for s in r.segments:
        _, qs, _ = sample(Timed(s.t, s.q, s.dq, s.ddq, 0, 0, 0), 0.002)
        assert np.all(qs >= env[0] - 1e-3) and np.all(qs <= env[1] + 1e-3)
    # a joint target outside it is refused (LIMIT), not clipped
    bad = plan(snap, PlanGoal("move_joint", "right", q=(1.0, -0.2, 0.0, 0.5, 0.0)), q, envelope=env)
    assert not bad.ok and bad.code == "LIMIT"


def test_hover_with_a_grip_aims_where_that_grip_closes(snap: WorldSnapshot) -> None:
    """HG-2: with grip + width the reference point is the grip's closing point (palm-fixed), not the open pinch."""
    q = J.POSE_STAND.copy()
    g = PlanGoal("hover", "right", position=(0.40, -0.15, 0.0), height_m=0.06, grip="wrap", width_m=0.05,
                 allow_hand=("box",))  # fmt: skip
    r = plan(snap, g, q)
    assert r.ok, r.reason
    geo = Geometry(snap)
    geo.set_stance(q)
    geo.set_arm("right", r.segments[-1].q[-1])
    aim = geo.point("right", "grasp:wrap@0.0500")
    assert np.hypot(aim[0] - 0.40, aim[1] + 0.15) < 0.004
    assert np.allclose(r.predicted["point_m"], aim, atol=1e-4)
    assert np.linalg.norm(np.array(r.predicted["pinch_m"][:2]) - aim[:2]) > 0.005  # not the open pinch point
    bad = plan(snap, PlanGoal("hover", "right", position=(0.40, -0.15, 0.0), grip="fist"), q)
    assert not bad.ok and bad.code == "INVALID_ARGS"
