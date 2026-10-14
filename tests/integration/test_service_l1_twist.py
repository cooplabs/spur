"""r1d's L1 `stream.arm_twist` (DESIGN.md §5.1, session 10; service/stream_l1.py): the grasp centre at a Cartesian
velocity, re-linearised at every request, with the REAL v3 core in lockstep (test_service_lockstep.Rig).

On the base (a041731) `stream.arm_twist` answered UNKNOWN_COMMAND.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.service.codes import Code
from r1mc3.service.goals import hand_points
from r1sim import cpp_policy, world

from tests.integration.test_service_lockstep import Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _centre(rig: Rig) -> np.ndarray:
    pts = hand_points(rig.port, None)
    assert pts is not None
    return np.asarray(pts[0, 4], dtype=np.float64)


def test_a_streamed_twist_moves_the_hand_along_it(w: world.World) -> None:
    """3 cm/s forward for 1 s (a request every 0.1 s, TTL 0.3 s), then silence: the grasp centre moves ~3 cm along
    +x, little sideways / up, and stops (the TTL)."""
    rig = Rig(w)
    res: dict[str, Any] = {"replies": []}

    def start() -> None:
        res["c0"] = _centre(rig)

    def twist() -> None:
        res["replies"].append(rig.svc.submit({"op": "stream.arm_twist", "arm": "left", "v": [0.03, 0.0, 0.0],
                                              "ttl": 0.3}).result)  # fmt: skip

    def end() -> None:
        res["c1"] = _centre(rig)

    rig.at(1.0, start)
    for k in range(10):
        rig.at(1.0 + 0.1 * k, twist)
    rig.at(3.0, end)
    rig.run(3.5)
    assert all(r is not None and r.ok for r in res["replies"]), res["replies"][:2]
    d = res["c1"] - res["c0"]
    assert 0.020 < d[0] < 0.040, d  # ~1 s x 3 cm/s (+ the last TTL's tail, minus the arm's lag)
    assert abs(d[1]) < 0.006 and abs(d[2]) < 0.006, d
    assert not rig.svc._jogs  # the stream ended with its TTL


def test_the_refusals(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        sub = rig.svc.submit
        res["fast"] = sub({"op": "stream.arm_twist", "arm": "left", "v": [0.2, 0.0, 0.0]}).result
        sub({"op": "world.declare", "entity": {"kind": "surface", "id": "t", "pose": {"pos": [0.3, 0.0, 0.38]},
                                                "size": [0.4, 1.0, 0.04]}})  # fmt: skip
        res["world"] = sub({"op": "stream.arm_twist", "arm": "left", "v": [0.02, 0.0, 0.0]}).result

    rig.at(1.0, go)
    rig.run(1.2)
    assert res["fast"].code == Code.INVALID_ARGS
    assert res["world"].code == Code.UNSUPPORTED and not res["world"].moved
