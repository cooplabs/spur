"""Session 10 (grasp_TO_PLANNER_3.md, the integrator): arm.hover's hand orientation. grasp's closed task grasps in sim
found the thumb -> index line ALONG x for every `prefer`: box-shaped objects were gripped across their length / a
diagonal (a standing thin box: tips meeting over its top; a wide box: slipping at the lift). The fixes, each failing on
a041731: (1) `grasp_yaw_deg` - the axis the fingers close along, seen from above (base frame, a line: mod 180; v2's
close_across_deg from the object's footprint, orient.py) ranks the clear poses; (2) `prefer` = v2's heading family of
the FINGERS (front / inward / outward / across, deg from above, + toward the hand's own side) - it already ranked the
search, but a family whose pose could not be held at the transit height was refused (across: 8 mm short at the
table_ready hand's height): the transit now comes down in 2 cm steps (never below 6 cm over the declared tops);
(3) a requested `approach` (a finger direction) ranks the hover's search too (it was ignored by hover);
(4) r1d passes the hover's grip (the HG-2 aim: the reference point where that grip closes), grasp_yaw_deg and approach
to the planner (it passed none of them)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.plan import PlanGoal, plan
from r1mc3.service.l2 import L2
from r1mc3.service.schemas import REQUEST
from r1mc3.skills import posture
from r1mc3.world import Geometry, Object, Pose, Surface, WorldModel, WorldSnapshot
from r1sim import joints as J

TOP = 0.7498
POS = (0.25, -0.1, TOP + 0.012)  # the right hand over a board-like surface in front of the robot's right side


@pytest.fixture(scope="module")
def table() -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.436, 0.0, TOP), size=(0.711, 1.219, 0.025)))
    w.declare(Surface("board", Pose.xyz_yaw(0.25, 0.0, TOP + 0.012), size=(0.356, 0.356, 0.012)))
    return w.snapshot()


def _q() -> np.ndarray:
    q = J.POSE_STAND.copy()
    q[J.LEFT_ARM] = posture("table_ready", "left")
    q[J.RIGHT_ARM] = posture("table_ready", "right")
    return q


def _close_yaw(snap: WorldSnapshot, q26: np.ndarray, side: str, q5: np.ndarray, grip: str) -> float:
    geo = Geometry(snap)
    geo.set_stance(q26)
    geo.set_arm(side, q5)
    t, f = geo.point(side, "thumb_tip"), geo.point(side, "index_tip")
    if grip == "wrap":
        f = 0.5 * (f + geo.point(side, "middle_tip"))
    v = f - t
    a = float(np.degrees(np.arctan2(v[1], v[0])))
    return (a + 90.0) % 180.0 - 90.0


def _axis_err(a: float, b: float) -> float:
    return abs((a - b + 90.0) % 180.0 - 90.0)


@pytest.mark.parametrize("grip", ["pinch", "wrap"])
def test_grasp_yaw_turns_the_closing_line(table: WorldSnapshot, grip: str) -> None:
    q = _q()
    got = {}
    for yaw in (0.0, 90.0):
        r = plan(table, PlanGoal("hover", "right", position=POS, height_m=0.10, grip=grip, grasp_yaw_deg=yaw), q)
        assert r.ok, r.reason
        cy = _close_yaw(table, q, "right", r.segments[-1].q[-1], grip)
        assert _axis_err(cy, yaw) <= 15.0, (grip, yaw, cy)
        assert _axis_err(r.details["orientation"]["close_yaw_deg"], cy) < 5.0  # the search vs the executed end
        got[yaw] = cy
    assert _axis_err(got[0.0], got[90.0]) >= 60.0


def test_a_thin_box_can_be_pinched_across_its_narrow_side(table: WorldSnapshot) -> None:
    """A standing thin box (9.4 cm along x, 2.1 cm along y, 6.6 cm tall): the hover with grasp_yaw 90 deg closes across
    its 2.1 cm (the line along y), centred over it."""
    w = WorldModel()
    for e in table.entities:
        w.declare(e)
    w.declare(Object("box", Pose.xyz_yaw(0.25, -0.1, TOP + 0.012 + 0.033), shape="box", size=(0.047, 0.0105, 0.033)))
    snap = w.snapshot()
    q = _q()
    g = PlanGoal("hover", "right", position=POS, height_m=0.08, grip="pinch", width_m=0.021, grasp_yaw_deg=90.0,
                 allow_hand=("box",))  # fmt: skip
    r = plan(snap, g, q)
    assert r.ok, r.reason
    cy = _close_yaw(snap, q, "right", r.segments[-1].q[-1], "pinch")
    assert _axis_err(cy, 90.0) <= 15.0, cy
    assert np.linalg.norm(np.array(r.predicted["point_m"][:2]) - np.array(POS[:2])) < 0.01


def test_prefer_across_is_reachable_and_turns_the_hand(table: WorldSnapshot) -> None:
    """a041731: UNREACHABLE 'the right hand cannot reach [0.25, -0.1, 0.978] (8 mm short)' (the transit height)."""
    q = _q()
    r0 = plan(table, PlanGoal("hover", "right", position=POS, height_m=0.10, grip="pinch"), q)
    r1 = plan(table, PlanGoal("hover", "right", position=POS, height_m=0.10, grip="pinch", prefer="across"), q)
    assert r0.ok and r1.ok, (r0.reason, r1.reason)
    o0, o1 = r0.details["orientation"], r1.details["orientation"]
    assert o1["angle_used"] == "across" and -100.0 <= o1["heading_deg"] <= -60.0, o1
    assert _axis_err(o0["close_yaw_deg"], o1["close_yaw_deg"]) >= 45.0, (o0, o1)


def test_a_requested_approach_ranks_the_hover(table: WorldSnapshot) -> None:
    """The right hand asked to come from its own side (fingers toward +y, 'from the right') vs from the left: the
    first is far closer (the 5-DoF arm cannot point the fingers back across the body at table height)."""
    q = _q()
    errs = {}
    for name, a in (("from_right", (0.0, 1.0, -0.3)), ("from_left", (0.0, -1.0, -0.3))):
        r = plan(table, PlanGoal("hover", "right", position=POS, height_m=0.08, grip="wrap", approach=a), q)
        assert r.ok, r.reason
        errs[name] = r.details["orientation"]["approach_err_deg"]
    assert errs["from_right"] < errs["from_left"] - 30.0, errs


def test_r1d_passes_the_hovers_grip_yaw_and_approach() -> None:
    req: Any = REQUEST.validate_python({"op": "arm.hover", "arm": "right", "position": list(POS), "height_m": 0.1,
                                        "grip": "wrap", "prefer": "across", "grasp_yaw_deg": 90.0,
                                        "approach": [0.0, 1.0, -0.3]})  # fmt: skip
    d = L2.plan_goal_dict(None, req)  # type: ignore[arg-type]  # (uses nothing of the instance)
    assert d["grip"] == "wrap" and d["grasp_yaw_deg"] == 90.0 and d["approach"] == [0.0, 1.0, -0.3]
    assert d["prefer"] == "across" and d["kind"] == "hover"
    g = PlanGoal.from_dict(d)
    assert g.grip == "wrap" and g.grasp_yaw_deg == 90.0 and g.approach == (0.0, 1.0, -0.3)
