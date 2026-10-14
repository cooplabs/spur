"""Session 10 (planner, FIRST 5): the planner's rule for a start already inside its clearance + v2's out-and-around
way for a hand under a declared table (`skill.table_ready` from the resting arms at a table).

Session 9 refused every goal whose start was inside the 1 cm clearance ("at the start the arm already touches",
COLLISION). The rule now: the arm may LEAVE - every pose of a sequentially checked path touches nothing new and no pair
(arm body, entity) gets closer - until it is clear; the clearance itself is unchanged. A hand under a declared surface
going above it goes out to the side and around (v2 outaround.py: the person's way on the real robot) with every part
2 cm from the world (no hand exception), at <= 0.5 rad/s. Each test names what a041731 did."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from r1mc3.plan import PlanGoal, ik, plan
from r1mc3.plan import core as C
from r1mc3.plan import path as PP
from r1mc3.skills import posture
from r1mc3.world import Geometry, Obstacle, Pose, Surface, WorldModel, WorldSnapshot
from r1sim import joints as J

TOP_Z = 0.7498  # a table top 75 cm above the soles (a start under a table: the resting hands hang under its near edge)


def _table(edge_x: float) -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(edge_x + 0.3555, 0.0, TOP_Z), size=(0.711, 1.219, 0.025)))
    return w.snapshot()


def _contacts(geo: Geometry, side: str, eid: str) -> float:
    """The smallest contact distance between `side`'s arm and entity `eid` within the geometry's margin (1.0: none)."""
    mujoco.mj_collision(geo.m, geo.d)
    best = 1.0
    arm = geo._geom_arm[side]
    for i in range(geo.d.ncon):
        c = geo.d.contact[i]
        a, b = int(c.geom1), int(c.geom2)
        if arm[a] != arm[b] and eid in (geo.geom_entity.get(a), geo.geom_entity.get(b)):
            best = min(best, float(c.dist))
    return best


@pytest.fixture(scope="module")
def elbow_box() -> WorldSnapshot:
    """A box beside the resting left elbow (on its outer, +y side), 2-8 mm away: inside the 1 cm clearance."""
    geo = Geometry(None)
    geo.set_stance(J.POSE_STAND)
    for dy in np.arange(0.075, 0.05, -0.0005):
        w = WorldModel()
        w.declare(Obstacle("box", Pose.xyz_yaw(-0.032, 0.191 + float(dy), 0.733), size=(0.03, 0.03, 0.03)))
        snap = w.snapshot()
        g = Geometry(snap)
        g.set_stance(J.POSE_STAND)
        d = _contacts(g, "left", "box")
        if 0.002 < d < 0.008:
            return snap
    raise AssertionError("no placement 2-8 mm from the elbow")


def test_a_start_inside_the_clearance_leaves_never_coming_closer(elbow_box: WorldSnapshot) -> None:
    """a041731: COLLISION 'at the start the arm already touches: left_elbow_link -> entity:box'."""
    q = J.POSE_STAND.copy()
    tgt = q[J.LEFT_ARM].copy()
    tgt[1] = 0.0  # shoulder roll toward the body (positive roll = toward the robot's LEFT: the box side)
    r = plan(elbow_box, PlanGoal("move_joint", "left", q=tuple(float(v) for v in tgt)), q)
    assert r.ok, r.reason
    assert r.details["legs"][0] == "leave" and r.details["left_inside_clearance"]
    geo = Geometry(elbow_box)
    geo.set_stance(q)
    prev = 1.0
    geo.set_arm("left", q[J.LEFT_ARM])
    prev = _contacts(geo, "left", "box")
    assert prev < 0.01
    for qq in PP.densify(r.segments[0].q, 0.005):
        geo.set_arm("left", qq)
        d = _contacts(geo, "left", "box")
        if d >= 1.0:
            break  # clear of the clearance: from here the normal checks
        assert d >= prev - PP.ESCAPE_TOL_M, (prev, d)
        prev = d
    geo.set_arm("left", r.segments[-1].q[-1])
    assert _contacts(geo, "left", "box") >= 1.0  # the goal pose is clear


def test_leaving_never_allows_coming_closer_or_a_new_contact(elbow_box: WorldSnapshot) -> None:
    geo = Geometry(elbow_box)
    q = J.POSE_STAND.copy()
    geo.set_stance(q)
    lim = ik.arm_limits("left")
    chk = PP.Checker(geo, "left", lim)
    q0 = q[J.LEFT_ARM].copy()
    pairs, bad = PP.start_pairs(chk, q0)
    assert pairs and not bad and min(pairs.values()) < 0.01
    toward = q0.copy()
    toward[1] += 0.3  # roll toward the LEFT = into the box
    assert PP.escape_segment(chk, pairs, q0, toward) is None
    assert "closer" in chk.last_hit or "touch" in chk.last_hit
    away = q0.copy()
    away[1] -= 0.25
    assert PP.escape_segment(chk, pairs, q0, away) == {}  # leaves and ends clear


@pytest.mark.parametrize("side", ["left", "right"])
def test_hands_under_a_table_go_out_and_around(side: str) -> None:
    """A start under a table (the resting arms, the hands 3 cm under the near edge, 21 cm below the top) -> the
    table_ready posture. a041731: one RRT joint path ('move') whose hand may pass 2 mm from the top (the planner's
    hand-over-a-support exception). Now: leave out to the side, over, on - the out / over legs keep every part 2 cm from
    the table INCLUDING the hand, at <= 0.5 rad/s, inside v2's demonstrated envelope."""
    snap = _table(0.081)
    q = J.POSE_STAND.copy()
    r = plan(snap, PlanGoal("move_joint", side, q=tuple(posture("table_ready", side)), speed=1.0), q)
    assert r.ok, r.reason
    assert r.details["legs"] == ["out", "over", "on"], r.details["legs"]
    oa = r.details["out_around"]
    assert oa["under"] == "table" and oa["clearance_m"] in (0.03, 0.02)  # the widest margin with a way (3 cm here)
    m = float(oa["clearance_m"])
    # the planner checks every 0.02 rad (CHECK_STEP_RAD); between two checks the margin can be grazed by a fraction of
    # a mm (1.98 of 2 cm seen at 0.01 rad): judged here at 5 mrad steps against 0.9 x the margin, the hand included
    geo2 = Geometry(snap, clearance_m=0.9 * m)
    geo2.set_stance(q)
    chk2 = PP.Checker(geo2, side, ik.arm_limits(side), hand_support_m=0.9 * m)
    for s in r.segments[:2]:
        assert all(chk2.valid(qq) for qq in PP.densify(s.q, 0.005)), (s.leg, chk2.last_hit)
        assert float(np.max(np.abs(s.dq))) <= 0.5 * 1.0 + 1e-6  # V_MAX x the out-and-around's 0.5
    # out to the robot's own side (directions from the verified table: positive roll = toward the LEFT on both arms)
    q_out = r.segments[0].q[-1]
    assert (q_out[1] > 0.8) if side == "left" else (q_out[1] < -0.8)
    geo2.set_arm(side, q_out)
    y = geo2.point(side, "pinch")[1]
    assert (y > 0.3) if side == "left" else (y < -0.3), y
    # the over pose: the whole hand >= 5 cm above the top; the end = the posture
    geo2.set_arm(side, r.segments[1].q[-1])
    assert geo2.lowest(side) >= TOP_Z + C.OVER_ABOVE_M
    assert np.max(np.abs(r.segments[-1].q[-1] - np.asarray(posture("table_ready", side)))) < 1e-6
    lo, hi = C.demo_envelope(side)
    for s in r.segments[:2]:
        assert np.all(s.q[:, :4] >= lo[:4] - 1e-6) and np.all(s.q[:, :4] <= hi[:4] + 1e-6)


def test_a_deeper_stance_with_the_elbow_inside_the_clearance() -> None:
    """The table 6 cm nearer (the real robot's 'deeper stance of the night', edge gap -99 mm): the resting elbow is 8
    mm from the top's edge. a041731: COLLISION 'at the start the arm already touches'. Now: out and around, the elbow
    leaving the edge (reported), never closer."""
    snap = _table(0.02)
    q = J.POSE_STAND.copy()
    for side in ("left", "right"):
        r = plan(snap, PlanGoal("move_joint", side, q=tuple(posture("table_ready", side)), speed=1.0), q)
        assert r.ok, r.reason
        assert r.details["legs"] == ["out", "over", "on"]
        assert any("elbow" in s for s in r.details["left_inside_clearance"]), r.details


def test_not_under_a_surface_plans_as_before() -> None:
    """A move_joint whose hand starts ABOVE the table (table_ready -> a nearby pose) is not routed out and around."""
    snap = _table(0.081)
    q = J.POSE_STAND.copy()
    q[J.LEFT_ARM] = posture("table_ready", "left")
    tgt = np.asarray(posture("table_ready", "left")) + np.array([0.0, -0.1, 0.0, 0.0, 0.0])
    r = plan(snap, PlanGoal("move_joint", "left", q=tuple(float(v) for v in tgt)), q)
    assert r.ok and r.details["legs"] == ["move"] and "out_around" not in r.details
