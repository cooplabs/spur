"""Geometry.distances() vs MuJoCo 3.5's box-box distance (session 11): mj_geomDistance answers an exact 0 for some
box-box pairs FAR apart (with a fromto, so the old "empty fromto" rule missed it) - the planner's reported clearance
then said 0.0 m with the palm 0.3 m from a table. On the base (1e55bdd): the pose below reads table 0.0 with no
contact; 5 of 400 random left-arm poses did. The contact checks (Geometry.hits: mj_collision) never read this value.
"""

from __future__ import annotations

import numpy as np
from r1mc3.plan import geometry_for
from r1mc3.plan.ik import arm_limits
from r1mc3.world import Pose, Surface, WorldModel
from r1mc3.world.geometry import Geometry
from r1sim import joints as J

ZERO_POSE = np.array([1.128, 1.347, -0.087, -0.141, -1.596])  # left arm: the palm box ~0.17 m from the table box


def _geo() -> Geometry:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.55, 0.0, 0.76), size=(0.6, 1.2, 0.04)))
    geo = geometry_for(w.snapshot())
    geo.set_stance(J.POSE_STAND.copy(), None, None)
    return geo


def _point_to_box(geo: Geometry, pts: np.ndarray, g: int) -> float:
    R = geo.d.geom_xmat[g].reshape(3, 3)
    loc = (pts - geo.d.geom_xpos[g]) @ R
    return float(np.linalg.norm(np.maximum(np.abs(loc) - geo.m.geom_size[g], 0.0), axis=1).min())


def _reference(geo: Geometry, side: str, eid: str) -> float:
    """The arm's distance to the entity box, brute force: every arm geom's AABB corners + centre (a lower bound is not
    needed here: a check that the reported value is not ABOVE the truth by more than the sampling's slack)."""
    gb = geo.entity_geoms[eid][0]
    best = np.inf
    for ga in geo._arm_geoms[side]:
        c, r = geo.d.geom_xpos[ga], float(geo.m.geom_rbound[ga])
        best = min(best, _point_to_box(geo, c[None, :], gb) - r)  # the bounding sphere: <= the true distance
    return best


def test_a_far_box_box_pair_is_not_read_as_touching() -> None:
    geo = _geo()
    geo.set_arm("left", ZERO_POSE)
    assert not geo.hits("left")  # no contact within the 1 cm clearance
    d = geo.distances("left")["table"]
    assert d > 0.01, d  # base: 0.0
    assert d >= _reference(geo, "left", "table") - 1e-6  # the bound is the bounding sphere's: conservative


def test_no_exact_zero_without_a_contact() -> None:
    geo = _geo()
    lo, hi = arm_limits("left")
    rng = np.random.default_rng(0)
    zeros = 0
    for _ in range(200):
        geo.set_arm("left", np.round(lo + rng.random(5) * (hi - lo), 3))
        if not geo.hits("left") and geo.distances("left").get("table", 1.0) == 0.0:
            zeros += 1
    assert zeros == 0  # base: 3 of these 200
