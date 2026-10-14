"""r1d's resource leases (DESIGN.md §5 "one lease model", session 10; service/leases.py), with the REAL v3 core in
lockstep (test_service_lockstep.Rig). OFF until a client takes one: every other test in the suite runs without a
lease and passes unchanged (the default-off proof), and the first test here shows it directly.

On the base (a041731): `lease.*` answered UNKNOWN_COMMAND and a `lease` field was refused (INVALID_ARGS: extra field).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.service.codes import Code
from r1sim import cpp_policy, world
from r1sim.v3_core import GROUP_ARM_LEFT, GROUP_ARM_RIGHT, GROUP_JOINTS

from tests.integration.test_service_lockstep import Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
LEFT = list(GROUP_JOINTS[GROUP_ARM_LEFT])
RIGHT = list(GROUP_JOINTS[GROUP_ARM_RIGHT])


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _mj(rig: Rig, arm: str, d_elbow: float, **kw: Any) -> dict[str, Any]:
    q = rig.ctl.last_q[LEFT if arm == "left" else RIGHT] + np.array([0.0, 0.0, 0.0, d_elbow, 0.0])
    return {"op": "arm.move_joint", "arm": arm, "q": q.tolist(), "speed": 0.5, **kw}


def test_no_lease_nothing_changes(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: res.__setitem__("g", rig.svc.submit({**_mj(rig, "left", -0.1), "wait": True})))
    rig.run(3.0)
    out = res["g"].result
    assert out is not None and out.ok and out.data["state"] == "succeeded", out
    lst = rig.svc.submit({"op": "lease.list"}).result
    assert lst is not None and lst.data["leases"] == []


def test_a_lease_gates_its_resources_only(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        sub = rig.svc.submit
        a = sub({"op": "lease.acquire", "resources": ["arm_left"], "client": "teleop", "ttl_s": 30}).result
        res["a"] = a
        assert a is not None
        lid = a.data["lease_id"]
        res["other_client"] = sub({"op": "lease.acquire", "resources": ["arm_left", "head"], "client": "x"}).result
        res["no_lease"] = sub(_mj(rig, "left", -0.1)).result
        res["wrong"] = sub(_mj(rig, "left", -0.1, lease="L999")).result
        res["right_free"] = sub(_mj(rig, "right", -0.1)).result
        res["with"] = sub(_mj(rig, "left", -0.1, lease=lid)).result
        gid = res["with"].data["goal_id"]
        res["cancel_no"] = sub({"op": "goal.cancel", "goal_id": gid}).result
        res["status"] = sub({"op": "status"}).result
        res["seq"] = sub({"op": "seq.run", "steps": [_mj(rig, "left", 0.0)]}).result  # no lease: refused
        res["cancel"] = sub({"op": "goal.cancel", "goal_id": gid, "lease": lid}).result
        res["seq_ok"] = sub({"op": "seq.run", "steps": [_mj(rig, "left", -0.05)], "lease": lid, "wait": True})
        res["arm_stop"] = None

    def stops() -> None:
        res["arm_stop"] = rig.svc.submit({"op": "arm.stop", "arm": "left"}).result  # never gated
        res["stop"] = rig.svc.submit({"op": "stop"}).result  # never gated

    rig.at(1.0, go)
    rig.at(4.0, stops)
    rig.run(4.5)
    assert res["a"].ok and res["a"].data["resources"] == ["arm_left"], res["a"]
    assert res["other_client"].code == Code.RESOURCE_BUSY
    for k in ("no_lease", "wrong"):
        r = res[k]
        assert r.code == Code.NOT_OWNER and not r.moved and r.details["held_by"]["client"] == "teleop", (k, r)
    assert res["right_free"].ok and res["with"].ok
    assert res["cancel_no"].code == Code.NOT_OWNER and res["status"].ok
    assert [le["client"] for le in res["status"].data["leases"]] == ["teleop"]  # status shows the live leases
    assert res["seq"].code == Code.NOT_OWNER
    assert res["cancel"].ok
    s = res["seq_ok"].result
    assert s is not None and s.ok and s.data["state"] == "succeeded", s  # the sequence's own steps are not gated
    assert res["arm_stop"].ok and res["stop"].ok


def test_a_lease_expires_renews_and_releases(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        sub = rig.svc.submit
        res["a"] = sub({"op": "lease.acquire", "resources": ["arm_left"], "client": "c", "ttl_s": 0.5}).result
        res["b"] = sub({"op": "lease.acquire", "resources": ["arm_right"], "client": "c", "ttl_s": 0.5}).result
        res["blocked"] = sub(_mj(rig, "left", -0.05)).result

    def later() -> None:  # 0.4 s: renew b; 0.7 s: a has expired
        res["renew"] = rig.svc.submit({"op": "lease.renew", "lease_id": res["b"].data["lease_id"], "ttl_s": 5}).result

    def after() -> None:
        res["free"] = rig.svc.submit(_mj(rig, "left", -0.05)).result
        res["right_held"] = rig.svc.submit(_mj(rig, "right", -0.05)).result
        res["rel"] = rig.svc.submit({"op": "lease.release", "lease_id": res["b"].data["lease_id"]}).result
        res["right_free"] = rig.svc.submit(_mj(rig, "right", -0.05)).result
        res["gone"] = rig.svc.submit({"op": "lease.release", "lease_id": res["b"].data["lease_id"]}).result
        res["bad"] = rig.svc.submit({"op": "lease.acquire", "resources": ["tail"], "client": "c"}).result

    rig.at(1.0, go)
    rig.at(1.4, later)
    rig.at(1.7, after)
    rig.run(2.0)
    assert res["blocked"].code == Code.NOT_OWNER
    assert res["renew"].ok and res["renew"].data["expires_in_s"] > 4.0
    assert res["free"].ok, res["free"]  # the left lease expired by itself
    assert res["right_held"].code == Code.NOT_OWNER  # the renewed one still holds
    assert res["rel"].ok and res["right_free"].ok
    assert res["gone"].code == Code.NOT_OWNER and res["bad"].code == Code.INVALID_ARGS
