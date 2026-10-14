"""5-DoF arm IK (DESIGN.md §7.2 step 1; v2 arms.py ArmKin.solve semantics): position first, approach second."""

from __future__ import annotations

import numpy as np
import pytest
from r1mc3.plan import ik
from r1mc3.world import Geometry
from r1sim import joints as J


@pytest.fixture(scope="module")
def geo() -> Geometry:
    g = Geometry()
    g.set_stance(J.POSE_STAND)
    return g


@pytest.mark.parametrize("side", ["left", "right"])
def test_reaches_points_in_front_exactly(geo: Geometry, side: str) -> None:
    g = 1.0 if side == "left" else -1.0
    for tgt in ((0.35, 0.15, 0.90), (0.40, 0.30, 0.85), (0.30, 0.05, 0.80)):
        t = np.array([tgt[0], g * tgt[1], tgt[2]])
        r = ik.solve(geo, side, t)
        assert r.ok and r.pos_err_m < 0.002, (side, tgt, r)
        lo, hi = ik.arm_limits(side)
        assert np.all(r.q >= lo - 1e-9) and np.all(r.q <= hi + 1e-9)


def test_approach_is_secondary_never_costs_position(geo: Geometry) -> None:
    down = np.array([0.0, 0.0, -1.0])
    q_stand = J.POSE_STAND[J.LEFT_ARM]
    free = ik.solve(geo, "left", np.array([0.35, 0.15, 0.90]), q_init=q_stand)
    geo.set_arm("left", free.q)
    a_free, _ = geo.hand_axes("left")
    free_err = float(np.degrees(np.arccos(np.clip(-a_free[2], -1, 1))))
    r = ik.solve(geo, "left", np.array([0.35, 0.15, 0.90]), q_init=q_stand, approach=down)
    assert r.ok and r.pos_err_m < 0.002
    # the requested approach comes much closer to straight down than the position-only solve (110 -> ~64 deg here:
    # straight down is not reachable at this point with this hand)
    assert r.approach_err_deg < free_err - 20.0, (r.approach_err_deg, free_err)


def test_out_of_reach_reports_the_shortfall(geo: Geometry) -> None:
    r = ik.solve(geo, "left", np.array([1.2, 0.2, 0.85]))
    assert not r.ok and r.pos_err_m > 0.5


def test_envelope_limits_are_kept(geo: Geometry) -> None:
    env = (np.array([-1.0, 0.0, -0.5, 0.2, -1.0]), np.array([0.3, 0.8, 0.5, 1.5, 1.0]))
    lo, hi = ik.arm_limits("left", env)
    r = ik.solve(geo, "left", np.array([0.35, 0.15, 0.80]), limits=(lo, hi))
    assert np.all(r.q >= lo - 1e-9) and np.all(r.q <= hi + 1e-9)


def test_policy_envelope_mirrors_the_right_arm() -> None:
    meta = (
        '{"shoulder_pitch": [-2.4, 0.75], "shoulder_roll": [-0.1, 1.85], "shoulder_yaw": [-1.87, 0.9], '
        '"elbow": [-0.93, 2.05], "wrist_roll": [-1.87, 1.87]}'
    )  # the live policy's ONNX metadata (policies/r1mc2_manip2_lean_20261006T0421Z)
    lo, hi = ik.policy_envelope(meta, "left")
    assert lo.tolist() == [-2.4, -0.1, -1.87, -0.93, -1.87] and hi.tolist() == [0.75, 1.85, 0.9, 2.05, 1.87]
    lo_r, hi_r = ik.policy_envelope(meta, "right")
    assert lo_r.tolist() == [-2.4, -1.85, -0.9, -0.93, -1.87] and hi_r.tolist() == [0.75, 0.1, 1.87, 2.05, 1.87]
    a, b = ik.arm_limits("right", (lo_r, hi_r))
    assert np.all(a >= lo_r) and np.all(b <= hi_r)
