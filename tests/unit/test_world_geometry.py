"""The one geometry engine (DESIGN.md §7.1): the robot posed in its base frame + the entities, contacts, distances."""

from __future__ import annotations

import numpy as np
import pytest
from r1mc3.world import Container, Geometry, Object, Pose, Surface, WorldModel
from r1sim import joints as J


@pytest.fixture(scope="module")
def geo() -> Geometry:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.55, 0.0, 0.76), size=(0.6, 1.2, 0.04)))
    w.declare(Container("bin", Pose.xyz_yaw(0.45, 0.30, 0.76), size_outer=(0.35, 0.26, 0.045)))
    w.declare(Object("box", Pose.xyz_yaw(0.40, -0.15, 0.785), size=(0.025, 0.025, 0.025)))
    return Geometry(w.snapshot())


def test_the_base_frame_is_the_model_world(geo: Geometry) -> None:
    geo.set_stance(J.POSE_STAND)
    fl, fr = geo.d.site_xpos[geo.ix.foot_sites[0]], geo.d.site_xpos[geo.ix.foot_sites[1]]
    mid = 0.5 * (fl + fr)
    assert abs(mid[0]) < 1e-9 and abs(mid[1]) < 1e-9 and min(fl[2], fr[2]) == pytest.approx(0.0, abs=1e-9)
    assert fl[1] > 0.05 and fr[1] < -0.05  # left foot on +y (REP-103)
    # the stance's hands hang below a 0.76 m table, the pinch point between the thumb and index tips
    for s, sign in (("left", 1.0), ("right", -1.0)):
        p = geo.point(s, "pinch")
        assert 0.4 < p[2] < 0.7 and sign * p[1] > 0.1
        assert np.allclose(p, 0.5 * (geo.point(s, "thumb_tip") + geo.point(s, "index_tip")))
        assert geo.hits(s) == []


def _into_table(geo: Geometry) -> np.ndarray:
    """Swing the left shoulder pitch forward (negative = forward: JOINT_DIRECTIONS.md "pitch + = limb BACK") until
    the arm meets the table's near edge."""
    q = J.POSE_STAND[J.LEFT_ARM].copy()
    for p in np.linspace(0.2, -1.5, 70):
        q[0] = p
        geo.set_arm("left", q)
        if any(h.other == "entity:table" for h in geo.hits("left")):
            return q
    raise AssertionError("never met the table")


def test_contacts_name_what_the_arm_touches(geo: Geometry) -> None:
    geo.set_stance(J.POSE_STAND)
    geo.set_arm("left", _into_table(geo))
    hits = geo.hits("left")
    assert any(h.other == "entity:table" and h.kind == "world" for h in hits), hits
    assert all(h.other != "entity:table" for h in geo.hits("left", allow=("table",)))
    assert 0.0 < geo.distance("left", "table") + 1e-9 <= geo.clearance_m + 1e-6 or geo.distance("left", "table") < 0


def test_allow_hand_only_lets_the_hand_touch(geo: Geometry) -> None:
    geo.set_stance(J.POSE_STAND)
    geo.set_arm("left", _into_table(geo))
    hand = {geo.m.body(b).name for b in geo.hand_bodies["left"]}
    left = geo.hits("left", allow_hand=("table",))
    assert all(h.arm_body not in hand for h in left if h.other == "entity:table")


def test_point_jacobian_matches_finite_differences(geo: Geometry) -> None:
    geo.set_stance(J.POSE_STAND)
    q = np.array([-0.6, 0.3, 0.2, 0.7, -0.3])
    geo.set_arm("left", q)
    Jp = geo.point_jac("left", "pinch")
    eps = 1e-6
    for k in range(5):
        dq = np.zeros(5)
        dq[k] = eps
        geo.set_arm("left", q + dq)
        a = geo.point("left", "pinch")
        geo.set_arm("left", q - dq)
        b = geo.point("left", "pinch")
        assert np.allclose((a - b) / (2 * eps), Jp[:, k], atol=1e-5)


def test_distance_and_surfaces_under(geo: Geometry) -> None:
    geo.set_stance(J.POSE_STAND)
    d = geo.distance("left", "table")
    assert 0.05 < d < 0.5
    tops = geo.surfaces_under(np.array([0.45, 0.30]))
    assert tops[0][0] == "bin" and tops[0][1] == pytest.approx(0.763)  # the container's floor
    sid, top = geo.surfaces_under(np.array([0.40, -0.15]))[0]
    assert sid == "box" and top == pytest.approx(0.81)
    assert geo.surfaces_under(np.array([2.0, 0.0])) == []
