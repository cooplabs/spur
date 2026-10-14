"""Session 10 (planner THEN): a held object in the planning model. The planner modelled the hand only; a carried
object could be planned through a container's rim or the table (the grasp grid's small round object dropped at a
container's rim). Now the object travels with the hand as a collision shape on the palm, centred where the hand closes
(the grasp centre at the measured finger pose), checked like the hand (the world clearance, the 2 mm support rule, the
body, the other arm), and r1d adds it to every arm goal while a grip holds (a sphere of half the grasp's width). On
a041731: PlanGoal has no `held`, r1d no held_object - every test here fails."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from r1mc3.plan import PlanGoal, ik, plan
from r1mc3.plan import core as C
from r1mc3.plan import path as PP
from r1mc3.service.l2 import L2
from r1mc3.service.schemas import REQUEST
from r1mc3.skills import posture
from r1mc3.world import Geometry, Pose, Surface, WorldModel, WorldSnapshot
from r1sim import joints as J

TOP = 0.7498
HELD = {"shape": "sphere", "size": [0.035]}


@pytest.fixture(scope="module")
def table() -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.436, 0.0, TOP), size=(0.711, 1.219, 0.025)))
    return w.snapshot()


def _held_start(snap: WorldSnapshot) -> np.ndarray:
    """The right hand 12 cm over the table (a hover from table_ready): where a carry starts."""
    q = J.POSE_STAND.copy()
    q[J.LEFT_ARM] = posture("table_ready", "left")
    q[J.RIGHT_ARM] = posture("table_ready", "right")
    r = plan(snap, PlanGoal("hover", "right", position=(0.28, -0.12, TOP), height_m=0.12), q)
    assert r.ok, r.reason
    q[J.RIGHT_ARM] = r.segments[-1].q[-1]
    return q


def test_a_held_object_is_checked_like_the_hand(table: WorldSnapshot) -> None:
    """A carry that ends with the grasp point 2 cm over the table: fine for the hand alone (it may work 2 mm above a
    support), a COLLISION with a 3.5 cm object in it (its bottom would be 1.5 cm in the table)."""
    q = _held_start(table)
    g = PlanGoal("carry", "right", position=(0.30, -0.20, TOP + 0.02), lift_m=0.05)
    r0 = plan(table, g, q)
    r1 = plan(table, PlanGoal("carry", "right", position=(0.30, -0.20, TOP + 0.02), lift_m=0.05, held=HELD), q)
    assert r0.ok, r0.reason
    assert not r1.ok and r1.code == "COLLISION" and "entity:table" in r1.reason, r1  # (the held shape is on the palm)


def test_the_held_object_keeps_its_clearance_along_the_carry(table: WorldSnapshot) -> None:
    q = _held_start(table)
    r = plan(table, PlanGoal("carry", "right", position=(0.30, -0.20, TOP + 0.08), lift_m=0.05, held=HELD), q)
    assert r.ok, r.reason
    geo = Geometry(table)
    geo.set_stance(q)
    h = C.held_in_palm(geo, "right", HELD)
    gh = Geometry(table, held=(h,))
    gh.set_stance(q)
    chk = PP.Checker(gh, "right", ik.arm_limits("right"))
    for s in r.segments:
        assert all(chk.valid(qq) for qq in PP.densify(s.q, 0.01)), (s.leg, chk.last_hit)
    # the object sits where the hand closes: its centre = the grasp centre
    gh.set_arm("right", r.segments[-1].q[-1])
    gid = gh.m.geom("held_right").id
    assert np.linalg.norm(gh.d.geom_xpos[gid] - gh.point("right", "grasp_centre")) < 1e-3


def test_an_object_resting_on_the_table_leaves_it_on_the_first_leg(table: WorldSnapshot) -> None:
    """Just grasped, the object still touches the table (inside its clearance at the start): it counts as the hand,
    so the first leg (the rise) may leave the table it rests on - nothing else may be touched."""
    q = J.POSE_STAND.copy()
    q[J.LEFT_ARM] = posture("table_ready", "left")
    q[J.RIGHT_ARM] = posture("table_ready", "right")
    r = plan(table, PlanGoal("hover", "right", position=(0.28, -0.12, TOP), height_m=0.02, grip="wrap"), q)
    assert r.ok, r.reason
    q[J.RIGHT_ARM] = r.segments[-1].q[-1]
    big = {"shape": "sphere", "size": [0.06]}  # its bottom below the fingertips: on the table at the start
    geo = Geometry(table)
    geo.set_stance(q)
    gh = Geometry(table, held=(C.held_in_palm(geo, "right", big),))
    gh.set_stance(q)
    assert any(h.other == "entity:table" for h in gh.hits("right", hand_support_m=0.01)), "the object is not down"
    rc = plan(table, PlanGoal("carry", "right", position=(0.30, -0.25, TOP + 0.10), lift_m=0.08, held=big), q)
    assert rc.ok, rc.reason
    assert rc.details["legs"][0] == "rise"


def test_r1d_adds_the_held_object_while_a_grip_holds() -> None:
    req: Any = REQUEST.validate_python({"op": "arm.carry", "arm": "right", "position": [0.3, -0.2, 0.83]})
    l2: Any = SimpleNamespace(grips={})
    l2.held_object = lambda side: L2.held_object(l2, side)
    assert "held" not in L2.plan_goal_dict(l2, req)
    l2.grips["right"] = SimpleNamespace(width_m=0.065, state="HOLDING")
    d = L2.plan_goal_dict(l2, req)
    assert d["held"] == {"shape": "sphere", "size": [0.0325]}
    l2.grips["right"] = SimpleNamespace(width_m=0.065, state="NOT_HOLDING")
    assert "held" not in L2.plan_goal_dict(l2, req)
