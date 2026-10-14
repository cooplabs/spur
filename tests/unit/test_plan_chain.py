"""Session 10 (grasp's finding, grasp_TO_PLANNER.md): a multi-leg goal's legs must chain exactly - every segment's
first knot = the previous segment's last - or the executor refuses the next leg ("the path's first point is more than
0.05 rad from the group's current target": LEG_REFUSED). `go_line` replaced a line's last pose by the planned target
pose ("end EXACTLY at the planned pose") but returned the line IK's own end, so the next leg was planned from a pose the
executed leg never reached. On a041731 this grid has two hovers whose descend starts 3.2 / 3.5 mrad off (grasp's
lockstep case: 0.09 rad of wrist roll)."""

from __future__ import annotations

import numpy as np
import pytest
from r1mc3.plan import PlanGoal, plan
from r1mc3.skills import posture
from r1mc3.world import Pose, Surface, WorldModel, WorldSnapshot
from r1sim import joints as J


@pytest.fixture(scope="module")
def table_and_board() -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.436, 0.0, 0.7495), size=(0.711, 1.219, 0.025)))
    w.declare(Surface("board", Pose.xyz_yaw(0.25, 0.0, 0.7615), size=(0.356, 0.356, 0.012)))
    return w.snapshot()


def test_hover_legs_chain_exactly(table_and_board: WorldSnapshot) -> None:
    q = J.POSE_STAND.copy()
    q[J.LEFT_ARM] = posture("table_ready", "left")
    q[J.RIGHT_ARM] = posture("table_ready", "right")
    n_multi = 0
    for side, sgn in (("right", -1.0), ("left", 1.0)):
        for x in (0.2, 0.25):
            for y in (0.0083, 0.05, 0.1):
                r = plan(table_and_board, PlanGoal("hover", side, position=(x, sgn * y, 0.763), height_m=0.102,
                                                   grip="wrap"), q)  # fmt: skip
                if not r.ok:
                    continue
                for a, b in zip(r.segments[:-1], r.segments[1:], strict=True):
                    n_multi += 1
                    gap = float(np.max(np.abs(b.q[0] - a.q[-1])))
                    assert gap < 1e-9, (side, x, y, a.leg, b.leg, gap)
    assert n_multi >= 6
