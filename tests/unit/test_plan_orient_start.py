"""The orientation search must not depend on where the arm comes from (session 11, episode's finding).

orient.best_orientation sweeps a shoulder-yaw x wrist-roll grid; Newton started from the arm's CURRENT pose first. A
grasp hover close in front of the robot (16 cm ahead, the lowest fingertip 12.4 cm over a table, a wrap closing along
y) from an arm that had just placed something in a bin on its right missed the requested closing line by 21-51 deg -
or was refused "out of the arm's reach" - while from other starts it was within 10 deg. On the base (1e55bdd): from the
bin-side start below close_err 51.1 deg; from the stand pose UNREACHABLE. Generic fixture: a table, a plate on it, a
bin beside it (no task objects).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.plan import PlanGoal, geometry_for, ik, plan
from r1mc3.plan import orient as O
from r1mc3.world import Container, Pose, Surface, WorldModel, WorldSnapshot
from r1mc3.world.geometry import ARM_JOINTS
from r1sim import joints as J

R = list(ARM_JOINTS["right"])
HOVER = {"kind": "hover", "arm": "right", "position": [0.162, -0.014, 0.751], "height_m": 0.124, "grip": "wrap",
         "grasp_yaw_deg": 90.0}  # fmt: skip


@pytest.fixture(scope="module")
def snap() -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.381, 0.0, 0.736), size=(0.6, 1.2, 0.03)))
    w.declare(Surface("plate", Pose.xyz_yaw(0.38, 0.0, 0.757), size=(0.45, 0.6, 0.012)))
    w.declare(Container("bin", Pose.xyz_yaw(0.281, -0.45, 0.7735), size_outer=(0.35, 0.26, 0.045)))
    return w.snapshot()


def _after(snap: WorldSnapshot, goal: dict[str, Any]) -> np.ndarray:
    """The 26 joints with the right arm where `goal` (planned from the stand pose) ends."""
    q = J.POSE_STAND.copy()
    r = plan(snap, PlanGoal.from_dict(goal), q)
    assert r.ok, r.reason
    q[R] = r.segments[-1].q[-1]
    return q


@pytest.fixture(scope="module")
def starts(snap: WorldSnapshot) -> dict[str, np.ndarray]:
    over_bin = {"kind": "hover", "arm": "right", "position": [0.20, -0.47, 0.751], "height_m": 0.04,
                "grasp_yaw_deg": 90.0}  # fmt: skip
    return {
        "stand": J.POSE_STAND.copy(),  # base: UNREACHABLE ("the point is out of the arm's reach")
        "over_bin": _after(snap, over_bin),  # base: close_err 51.1 deg
        "over_bin_2": _after(snap, {**over_bin, "position": [0.28, -0.40, 0.751], "height_m": 0.08}),  # base 21.4
        "front": _after(snap, {"kind": "hover", "arm": "right", "position": [0.25, -0.15, 0.751], "height_m": 0.15}),
    }


def test_the_grasp_hover_meets_its_closing_line_from_every_start(
    snap: WorldSnapshot, starts: dict[str, np.ndarray]
) -> None:
    got: dict[str, float | str] = {}
    for name, q in starts.items():
        r = plan(snap, PlanGoal.from_dict(HOVER), q)
        got[name] = r.details["orientation"]["close_err_deg"] if r.ok else r.code
    # base: {'stand': 'UNREACHABLE', 'over_bin': 51.1, 'over_bin_2': 21.4, 'front': 9.1}
    assert all(isinstance(v, float) and v <= O.CLOSE_TOL_DEG for v in got.values()), got


def test_a_met_request_costs_one_sweep_a_missed_one_four(snap: WorldSnapshot, starts: dict[str, np.ndarray]) -> None:
    geo = geometry_for(snap)
    xy, z = np.array([0.162, -0.014]), 0.751 + 0.124  # HOVER: the datum + height_m
    geo.set_stance(starts["over_bin"], None, None)
    missed = O.best_orientation(geo, "right", xy, z, grasp_yaw_deg=90.0, grip="wrap")
    assert missed.ok and missed.sweeps == 1 + len(ik.SEEDS_LEFT) and missed.close_err_deg <= O.CLOSE_TOL_DEG
    geo.set_stance(J.POSE_STAND.copy(), None, None)
    easy = O.best_orientation(geo, "right", np.array([0.30, -0.15]), 0.90)  # no request: nothing to miss
    assert easy.ok and easy.sweeps == 1
    # the travel still counts from the CURRENT pose: the same request from two starts may pick different poses, both
    # within the tolerance
    geo.set_stance(starts["front"], None, None)
    other = O.best_orientation(geo, "right", xy, z, grasp_yaw_deg=90.0, grip="wrap")
    assert other.ok and other.close_err_deg <= O.CLOSE_TOL_DEG
