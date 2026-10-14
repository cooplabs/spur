"""r1d's `target_id` on the arm goals (session 10 round B): the declared world entity a goal is FOR - the hand may
touch it (the planner's allow_hand), the arm keeps its clearance, every other entity blocks as before. Why: a client
that declares the objects on its table (so planned paths go around them) must still be able to reach INTO the one it
grasps (a task client declares each object it measures).

On the base (3fc1d31) r1d refuses the field (INVALID_ARGS: extra fields are forbidden), so the hand cannot reach a
declared object at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import r1mc3.plan as planner
from r1mc3.service.codes import Code
from r1mc3.service.goals import hand_points
from r1sim import cpp_policy, world

from tests.integration.test_service_lockstep import Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def test_a_goal_may_reach_into_its_own_target_and_only_that(w: world.World) -> None:
    rig = Rig(w)
    rig.svc.planner = planner
    res: dict[str, Any] = {}

    def go() -> None:
        pts = hand_points(rig.port, None)
        assert pts is not None
        c = np.asarray(pts[0, 4], dtype=np.float64)  # the left grasp centre now
        p = [float(v) for v in c + np.array([0.12, 0.0, 0.06])]  # clear of the start (a start inside an
        # entity's clearance may leave it: the planner allows what the start touches)
        box = {"kind": "object", "id": "box_flat", "shape": "box", "size": [0.05, 0.05, 0.02],  # 10 x 10 x 4 cm
               "pose": {"pos": p}}  # fmt: skip
        other = {"kind": "object", "id": "other", "shape": "box", "size": [0.02, 0.02, 0.02],
                 "pose": {"pos": [p[0], p[1] + 0.40, p[2]]}}  # fmt: skip
        for e in (box, other):
            r = rig.svc.submit({"op": "world.declare", "entity": e}).result
            assert r is not None and r.ok, r
        reach = {"op": "arm.reach", "arm": "left", "position": p}
        res["blocked"] = rig.svc.submit({"op": "arm.plan", "goal": reach}).result
        res["target"] = rig.svc.submit({"op": "arm.plan", "goal": {**reach, "target_id": "box_flat"}}).result
        res["wrong"] = rig.svc.submit({"op": "arm.plan", "goal": {**reach, "target_id": "other"}}).result
        res["empty"] = rig.svc.submit({"op": "arm.plan", "goal": {**reach, "target_id": ""}}).result

    rig.at(1.0, go)
    rig.run(1.5)
    b, t, wr, e = res["blocked"], res["target"], res["wrong"], res["empty"]
    assert b is not None and not b.ok and b.code == Code.COLLISION and "box_flat" in str(b.details), b
    assert t is not None and t.ok, t  # base: INVALID_ARGS (target_id unknown)
    assert wr is not None and not wr.ok and wr.code == Code.COLLISION and "box_flat" in str(wr.details), wr
    assert e is not None and not e.ok and e.code == Code.INVALID_ARGS, e
