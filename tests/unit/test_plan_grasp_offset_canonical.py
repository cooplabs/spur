"""Session 12 (fixes; the S11 CI leak: the whole-episode exact replay 130 / 135 after other tests declared the same
world): a plan does not depend on which earlier goal first filled the planner's geometry cache.

`Geometry.grasp_offset` (where a grip's fingers close, in the PALM frame) is cached per geometry, and the planner caches
geometries per world snapshot (`plan.core.geometry_for`). It was computed at the CALLER's stance: the palm-frame floats
then varied ~1e-16 m with that stance, so the value a geometry kept depended on the first goal that asked - and the
lockstep simulation grew that into a different episode (S12 fixes.md: the full-suite order reproduced S11's 135 / 135
/ 130 / 109 exactly; the stale geometry came from an earlier task test with the same table / board / room). Now it is
computed at one canonical stance (the model's qpos0).

On the base (1cf3afd): the offsets differ bitwise between the two stances (max 1.1e-16 m) and so do the plan's knots.
"""

from __future__ import annotations

import numpy as np
from r1mc3 import plan as P
from r1mc3.plan import core as PC
from r1mc3.world import Geometry, Pose, Surface, WorldModel
from r1sim import joints as J

_Q_OTHER = J.POSE_STAND.copy()
_Q_OTHER[19:24] = [-0.4, -0.3, 0.2, -0.6, 0.3]  # the right arm elsewhere (an earlier goal's stance)
_Q_OTHER[12] = 0.1
_IMU_OTHER = np.array([0.999, 0.02, 0.03, 0.0]) / np.linalg.norm([0.999, 0.02, 0.03, 0.0])


def test_the_grasp_offset_is_the_same_floats_whichever_stance_asked_first() -> None:
    snap = WorldModel().snapshot()
    ga, gb = Geometry(snap), Geometry(snap)
    ga.set_stance(J.POSE_STAND.copy())
    gb.set_stance(_Q_OTHER, _IMU_OTHER)
    for grip, w in (("wrap", 0.065), ("pinch", 0.02), ("wrap", None), ("tip", 0.004)):
        a, b = ga.grasp_offset("right", grip, w), gb.grasp_offset("right", grip, w)
        assert np.array_equal(a, b), (grip, w, float(np.max(np.abs(a - b))))


def test_a_plan_does_not_depend_on_which_goal_filled_the_geometry_cache_first() -> None:
    wm = WorldModel()
    wm.declare(Surface("table", Pose((0.45, 0.0, 0.75)), size=(0.30, 0.60, 0.02)))
    snap = wm.snapshot()
    q = J.POSE_STAND.copy()
    goal = P.PlanGoal("hover", "right", position=(0.30, -0.15, 0.75), height_m=0.06, grip="wrap", width_m=0.065)
    PC._GEO.clear()
    clean = P.plan(snap, goal, q)
    PC._GEO.clear()
    g = PC.geometry_for(snap)
    g.set_stance(_Q_OTHER, _IMU_OTHER)
    g.grasp_offset("right", "wrap", 0.065)  # an earlier goal, from another stance, filled the cache
    again = P.plan(snap, goal, q)
    assert clean.ok and again.ok, (clean.reason, again.reason)
    assert len(clean.segments) == len(again.segments)
    for a, b in zip(clean.segments, again.segments, strict=True):
        assert np.array_equal(a.q, b.q) and np.array_equal(a.t, b.t), a.leg
