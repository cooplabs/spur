"""Session 12: r1d writes every grasp's trace (r1mc3.hands.trace) into its records dir - the REAL v3 core in lockstep +
the grip tests' geometric fake hand (tests/integration/test_service_l2_hands.py's rig). On the base (1cf3afd) no file:
HW2 day 1's B4 grasps left nothing of what the verdict saw (the tape pinch's NOT_HOLDING could not be explained)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from r1mc3.hands import trace as T
from r1mc3.service.codes import Code
from r1sim import cpp_policy, world

from tests.integration.test_service_l2_hands import HandsRig, _object

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def test_r1d_writes_each_grasps_trace_and_it_replays(
    w: world.World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("R1_RECORD_DIR", str(tmp_path))
    rig = HandsRig(w)
    _object(rig, "left", 0.040)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: res.__setitem__("g", rig.svc.submit({"op": "hand.grasp", "side": "left", "grip": "wrap",
                                                             "width_m": 0.04, "wait": True})))  # fmt: skip
    rig.at(4.0, lambda: res.__setitem__("r", rig.svc.submit({"op": "hand.release", "side": "left", "wait": True})))
    rig.at(5.0, lambda: res.__setitem__("e", rig.svc.submit({"op": "hand.grasp", "side": "left", "grip": "wrap",
                                                             "wait": True})))  # fmt: skip
    rig.run(9.0)
    assert res["g"].result is not None and res["g"].result.ok, res["g"].result
    assert res["r"].result is not None and res["r"].result.ok
    e = res["e"].result
    assert e is not None and e.code in (Code.OK, Code.NOT_HOLDING)
    rig.svc.submit({"op": "hand.release", "side": "left", "wait": True})
    rig.run(1.0)
    files = sorted(tmp_path.glob("grasp_*_left.npz"))
    assert files, list(tmp_path.iterdir())
    first = T.load(files[0])
    assert first["meta"]["params"]["grip"] == "wrap" and first["meta"]["params"]["width_m"] == 0.04
    assert first["meta"]["verdict"]["state"] == "TENTATIVE"  # the last verdict before the release (no lift here)
    out = T.replay(first)
    assert out["same_as_recorded"], (out["verdict"]["state"], out["verdict"]["reason"])
