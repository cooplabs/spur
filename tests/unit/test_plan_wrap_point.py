"""Session 10 (grasp_TO_PLANNER_4.md, the integrator): the wrap's aim without the object's width. r1d's arm.hover passes
the grip (84b2318) but no width, and HG-2's closing point for a wrap without a width is the FULLY closed fist's
(r1mc3.world.Geometry.grasp_offset: 3.6 cm back along the fingers and 3 cm into the palm from the open pinch point) -
in the physical-hands rig every 6-8 cm wrap aimed there closed too low / too deep (0 / 9 placed; 3 / 9 at the default
point). The choice: a wrap WITHOUT a width keeps the default reference point (the open pinch point, v2's / session 9's
aim); WITH a width HG-2's point where the tips are that far apart (planner-level; not validated in the rig); pinch /
tip keep HG-2 (their closed point = the tips meeting: 1 -> 4 / 6 pinches placed with it). On a041731 / 84b2318 the
first test fails (the wrap aimed at the fist's closed point)."""

from __future__ import annotations

import numpy as np
import pytest
from r1mc3.plan import PlanGoal, plan
from r1mc3.skills import posture
from r1mc3.world import Geometry, Pose, Surface, WorldModel, WorldSnapshot
from r1sim import joints as J

TOP = 0.7498
POS = (0.25, -0.1, TOP)


@pytest.fixture(scope="module")
def table() -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.436, 0.0, TOP), size=(0.711, 1.219, 0.025)))
    return w.snapshot()


def _q() -> np.ndarray:
    q = J.POSE_STAND.copy()
    q[J.LEFT_ARM] = posture("table_ready", "left")
    q[J.RIGHT_ARM] = posture("table_ready", "right")
    return q


def _aim(snap: WorldSnapshot, q: np.ndarray, q5: np.ndarray, name: str) -> np.ndarray:
    geo = Geometry(snap)
    geo.set_stance(q)
    geo.set_arm("right", q5)
    return geo.point("right", name)


def test_a_wrap_without_a_width_aims_at_the_default_point(table: WorldSnapshot) -> None:
    q = _q()
    r = plan(table, PlanGoal("hover", "right", position=POS, height_m=0.06, grip="wrap"), q)
    assert r.ok, r.reason
    p = _aim(table, q, r.segments[-1].q[-1], "pinch")  # the default reference point is over the target
    assert np.linalg.norm(p[:2] - np.array(POS[:2])) < 0.003, p
    fist = _aim(table, q, r.segments[-1].q[-1], "grasp:wrap")
    assert np.linalg.norm(fist[:2] - np.array(POS[:2])) > 0.02  # (the fist's closed point is not what is aimed)


def test_a_pinch_keeps_hg2_and_a_wrap_with_a_width_too(table: WorldSnapshot) -> None:
    q = _q()
    r = plan(table, PlanGoal("hover", "right", position=POS, height_m=0.06, grip="pinch"), q)
    assert r.ok, r.reason
    p = _aim(table, q, r.segments[-1].q[-1], "grasp:pinch")
    assert np.linalg.norm(p[:2] - np.array(POS[:2])) < 0.003
    r2 = plan(table, PlanGoal("hover", "right", position=POS, height_m=0.06, grip="wrap", width_m=0.065), q)
    assert r2.ok, r2.reason
    p2 = _aim(table, q, r2.segments[-1].q[-1], "grasp:wrap@0.0650")
    assert np.linalg.norm(p2[:2] - np.array(POS[:2])) < 0.003
