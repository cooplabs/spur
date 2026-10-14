"""Session 12 (fixes; the CI-leak question "does any planner cache make plans depend on earlier goals?"): a plan
depends only on its inputs. The planner caches its geometry per world snapshot (`core.geometry_for`: re-declaring a
table is free), and `Geometry.set_stance` kept a hand's finger pose from the PREVIOUS call when the new call gave none
for that side - r1d passes no hand pose when the hand state is not valid (`l2.plan_inputs`: the hand driver starting,
a stale hand). On the robot r1d's planner worker is long-lived, so such a plan used whatever fingers the previous plan
had: the same reach landed its index tip 11 cm elsewhere after a plan with the fist (0.342 / -0.263 / 1.004 m fresh
vs 0.272 / -0.176 / 0.943 m after). Now a missing hand is posed open, as in a fresh geometry.

On the base (1cf3afd) the second plan below reproduces the fist's fingertips, not the fresh plan's.
"""

from __future__ import annotations

from r1mc3 import plan as P
from r1mc3.plan import core as PC
from r1mc3.world import WorldModel
from r1sim import dex3
from r1sim import joints as J


def test_a_plan_without_a_hand_state_does_not_inherit_the_previous_plans_fingers() -> None:
    snap = WorldModel().snapshot()
    q = J.POSE_STAND.copy()
    goal = P.PlanGoal("reach", "right", position=(0.30, -0.20, 0.95))
    fist = {s: dex3.preset("fist", s) for s in ("left", "right")}
    PC._GEO.clear()
    fresh = P.plan(snap, goal, q, hands=None)
    after_fist = P.plan(snap, goal, q, hands=fist)
    again = P.plan(snap, goal, q, hands=None)
    assert fresh.ok and after_fist.ok and again.ok
    assert again.predicted["fingertips_m"] == fresh.predicted["fingertips_m"], again.predicted["fingertips_m"]
    # the hand pose given matters (the test would be vacuous otherwise)
    assert after_fist.predicted["fingertips_m"] != fresh.predicted["fingertips_m"]


def test_one_side_given_the_other_is_open_whatever_came_before() -> None:
    snap = WorldModel().snapshot()
    q = J.POSE_STAND.copy()
    goal = P.PlanGoal("reach", "right", position=(0.30, -0.20, 0.95))
    PC._GEO.clear()
    ref = P.plan(snap, goal, q, hands={"left": dex3.preset("fist", "left")})
    P.plan(snap, goal, q, hands={s: dex3.preset("fist", s) for s in ("left", "right")})
    got = P.plan(snap, goal, q, hands={"left": dex3.preset("fist", "left")})  # the right hand: no state
    assert got.predicted["fingertips_m"] == ref.predicted["fingertips_m"]
