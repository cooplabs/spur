"""The goal table's parent rule (l2.py `accept` / `tick`, session 10 fix): a composite goal (seq.run, a skill,
par.run) holds its steps' resources for them, so

* two goal steps on the SAME arm back to back: the second is deferred until the limb is still (the recent-release
  rule) and then must start - on the base (a041731) it waited for its own parent forever (the sequence never ended);
* a nested composite (a skill inside a seq.run): the grandchild (the skill's arm goal) runs on what its grandparent
  holds - on the base it was refused RESOURCE_BUSY by the grandparent (the skill failed, the sequence STEP_FAILED).

The real v3 core in lockstep, no planner, no world (arm.move_joint runs its local minimum-jerk path).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1sim import cpp_policy, world
from r1sim.v3_core import GROUP_ARM_LEFT, GROUP_ARM_RIGHT, GROUP_JOINTS

from tests.integration.test_service_lockstep import Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
LEFT = list(GROUP_JOINTS[GROUP_ARM_LEFT])
RIGHT = list(GROUP_JOINTS[GROUP_ARM_RIGHT])


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def test_two_goal_steps_on_one_arm_back_to_back(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        q0 = rig.ctl.last_q[LEFT].copy()
        q1, q2 = q0 + np.array([0.0, 0.0, 0.0, -0.15, 0.0]), q0 + np.array([0.0, 0.1, 0.0, -0.15, 0.0])
        res["q2"] = q2
        steps = [{"op": "arm.move_joint", "arm": "left", "q": q1.tolist(), "speed": 0.5},
                 {"op": "arm.move_joint", "arm": "left", "q": q2.tolist(), "speed": 0.5}]  # fmt: skip
        res["seq"] = rig.svc.submit({"op": "seq.run", "steps": steps, "timeout_s": 8.0, "wait": True})

    rig.at(1.0, go)
    rig.run(6.0)
    out = res["seq"].result
    assert out is not None and out.ok and out.data["state"] == "succeeded", out
    assert [s["state"] for s in out.data["steps"]] == ["succeeded", "succeeded"]
    assert np.max(np.abs(rig.ctl.last_q[LEFT] - res["q2"])) < 1e-3


def test_a_skill_inside_a_sequence_runs_on_its_grandparents_resources(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        # the first step on the OTHER arm: the sequence is in the goal table when the skill starts its own arm goal
        qr = rig.ctl.last_q[RIGHT] + np.array([0.0, 0.0, 0.0, -0.1, 0.0])
        first = {"op": "arm.move_joint", "arm": "right", "q": qr.tolist(), "speed": 0.5}
        inner = {"op": "skill.table_ready", "arms": "left", "speed": 0.6}
        res["seq"] = rig.svc.submit({"op": "seq.run", "steps": [first, inner], "timeout_s": 10.0, "wait": True})

    rig.at(1.0, go)
    rig.run(9.0)
    out = res["seq"].result
    assert out is not None and out.ok and out.data["state"] == "succeeded", out
    inner_res = out.data["steps"][1]
    assert inner_res["state"] == "succeeded" and inner_res["data"]["steps"][0]["state"] == "succeeded", inner_res
