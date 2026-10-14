"""Session 13 (HW2 day 2's F12, "the lean does nothing on the robot"): the policy's lean after a stop, r1d + the REAL
v3 core in lockstep (the Rig of test_service_lockstep.py).

What happened on the robot (results/hw2_20261013/floor/r1d_journal.jsonl): the task harness ended teleop run 2 with a
client `stop` at 02:47:08 UTC; nobody resumed; at 02:53:00 `r1 lean 3` was ACCEPTED ("moved": true) and `r1 status`
said `lean_deg [3.0, 0.0]` - the core's stop LATCHES until a resume and zeroes the base command (core.cpp
current_cmd), the arms meanwhile took new goals (two skill.table_ready at 02:52). The sim's F12 (2.85 deg) had no stop
before it. On the base (dcc01a2) the lean after the stop is accepted and the core's lean stays 0 (this test fails at
the STOPPED assertion); r1d now refuses it in words, a stop ends a running lean, and after `r1 resume` the lean works.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.service.codes import Code
from r1sim import cpp_policy, world
from r1sim.v3_core import MODE_BALANCE

from tests.integration.test_service_lockstep import Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
LEAN3 = float(np.radians(3.0))


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _core_lean(rig: Rig) -> float:
    return float(rig.ctl.core.status()["lean"])


def test_a_lean_after_a_stop_is_refused_in_words_and_works_after_resume(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def put(key: str, req: dict[str, Any]) -> None:
        res[key] = rig.svc.submit(req)

    def sample(key: str) -> None:
        res[key] = (_core_lean(rig), rig.svc.status(), rig.port.rt_info())

    rig.at(1.0, lambda: put("stop", {"op": "stop"}))  # a client's stop (a task client's end of a run)
    rig.at(1.1, lambda: sample("after_stop"))
    rig.at(1.2, lambda: put("lean_stopped", {"op": "body.pose", "lean_deg": 3.0}))
    rig.at(2.5, lambda: sample("stopped_later"))
    rig.at(2.6, lambda: put("resume", {"op": "resume"}))
    rig.at(2.9, lambda: sample("after_resume"))
    rig.at(3.0, lambda: put("lean", {"op": "body.pose", "lean_deg": 3.0}))
    rig.at(4.5, lambda: sample("leaning"))
    rig.at(4.6, lambda: put("stop2", {"op": "stop"}))  # a stop while leaning ends the lean
    rig.at(6.0, lambda: sample("stopped_again"))
    rig.at(6.1, lambda: put("resume2", {"op": "resume"}))
    rig.at(7.5, lambda: sample("resumed_upright"))
    rig.run(7.6)

    assert res["stop"].result.ok
    _, st, info = res["after_stop"]
    assert info is not None and info["stop_active"] is True  # r1rt's heartbeat flag (R1RT_RT_FLAG_STOP_ACTIVE)
    assert st["stop"]["latched"] is True and "r1 resume" in st["stop"]["words"]
    r = res["lean_stopped"].result
    assert r is not None and r.code == Code.STOPPED and not r.moved, r  # base: OK, moved True - and nothing happened
    assert "r1 resume" in r.details["reason"] and r.details["needs"] == "r1 resume"
    lean, st, _ = res["stopped_later"]
    assert abs(lean) < 1e-9 and st["body"]["lean_deg"][0] is None  # nothing asked, nothing used
    assert res["resume"].result is not None and res["resume"].result.ok, res["resume"].result
    _, st, info = res["after_resume"]
    assert info is not None and info["stop_active"] is False and st["stop"]["latched"] is False
    assert res["lean"].result.ok and res["lean"].result.moved
    lean, st, _ = res["leaning"]
    assert abs(lean - LEAN3) < 0.005, np.degrees(lean)  # the ramp (4 deg/s) reached the 3 deg
    assert st["body"]["lean_deg"][0] == pytest.approx(3.0) and st["body"]["lean_deg"][1] == pytest.approx(3.0, abs=0.3)
    lean, st, _ = res["stopped_again"]
    assert abs(lean) < 1e-9, np.degrees(lean)  # the core ramped it back to 0 within ~0.75 s of the stop
    assert st["body"]["lean_deg"][0] is None and "r1 resume" in st["body"]["lean_note"]
    lean, st, _ = res["resumed_upright"]
    assert abs(lean) < 1e-9  # no lean coming back by itself after the resume
    assert st["body"]["lean_note"] is not None and "ended with a 'stop'" in st["body"]["lean_note"]
    assert int(rig.ctl.core.mode) == MODE_BALANCE
