"""The hand-orientation search (ported from v2 orient.py, tests/test_orient.py's contracts on a generic scene):
the swivel x roll family at a point, the touch rule, the preferred heading family, refusals that say why."""

from __future__ import annotations

import time

import numpy as np
import pytest
from r1mc3.plan import orient as O
from r1mc3.world import Geometry, Pose, Surface, WorldModel
from r1sim import joints as J

TABLE_Z = 0.76


@pytest.fixture(scope="module")
def geo() -> Geometry:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.55, 0.0, TABLE_Z), size=(0.6, 1.2, 0.04)))
    g = Geometry(w.snapshot())
    g.set_stance(J.POSE_STAND)
    return g


def test_contract_and_speed(geo: Geometry) -> None:
    """v2 test_best_orientation_contract_and_speed: the keys, a clear pose that meets the point, < 2 s."""
    t0 = time.monotonic()
    r = O.best_orientation(geo, "right", np.array([0.33, -0.12]), TABLE_Z + 0.05)
    dt = time.monotonic() - t0
    assert r.ok and r.clear > 0 and r.candidates >= r.clear
    geo.set_arm("right", r.q)
    p = geo.point("right", "pinch")
    low = min(float(geo.point("right", k)[2]) for k in ("thumb_tip", "index_tip", "middle_tip"))
    assert np.hypot(p[0] - 0.33, p[1] + 0.12) < 0.003 and abs(low - (TABLE_Z + 0.05)) < 0.003
    assert geo.hits("right") == []
    assert r.angle_name in O.FAMILIES and r.pitch_down_deg >= O.PITCH_MIN_DEG - 1e-6
    assert dt < 2.0  # macOS under load (11 agents); v2: 0.05-0.3 s with its numpy chain


def test_preferred_family_used_when_feasible(geo: Geometry) -> None:
    """v2 test_preferred_family_used_when_feasible: a feasible family is honoured (angle error 0)."""
    free = O.best_orientation(geo, "left", np.array([0.33, 0.12]), TABLE_Z + 0.06)
    assert free.ok
    fam = free.angle_name
    r = O.best_orientation(geo, "left", np.array([0.33, 0.12]), TABLE_Z + 0.06, prefer=fam)
    assert r.ok and r.angle_err_deg == 0.0 and r.angle_name == fam


def test_touch_rule_fingertips_land_first(geo: Geometry) -> None:
    """v2 touch_rule (AUDIT R6): need_touch -> the fingertips are the lowest part of the hand by >= 3 mm."""
    r = O.best_orientation(geo, "right", np.array([0.33, -0.12]), TABLE_Z + 0.05, need_touch=True)
    assert r.ok and r.touch_margin_mm >= 0.0
    geo.set_arm("right", r.q)
    tm, tips_z = O.touch_margin(geo, "right")
    assert tm >= 0.0 and abs(tips_z - (TABLE_Z + 0.05)) < 0.003


def test_out_of_reach_says_so(geo: Geometry) -> None:
    """v2 test_refusal_names_what_would_work (the reason part): far away -> out of reach, not 'blocked'."""
    r = O.best_orientation(geo, "left", np.array([0.9, 0.2]), TABLE_Z + 0.05)
    assert not r.ok and "reach" in r.why


def test_heading_convention_mirrors_left_and_right() -> None:
    a_left_out = np.array([0.5, 0.5, -0.7])  # fingers toward +y = the left hand's own side
    assert O.heading_deg("left", a_left_out) > 0 and O.angle_name(O.heading_deg("left", a_left_out)) == "outward"
    a_right_out = np.array([0.5, -0.5, -0.7])
    assert O.heading_deg("right", a_right_out) == pytest.approx(O.heading_deg("left", a_left_out))
    assert O.angle_err(-30.0, "front") == pytest.approx(10.0) and O.angle_err(0.0, None) == 0.0
