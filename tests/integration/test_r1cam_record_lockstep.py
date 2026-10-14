"""The camera record kinds through r1d against the REAL core in lockstep (session 9): `record.run depth_targets` moves
nothing, r1cam records for the plan's duration, the record manifest carries r1cam's manifest; without r1cam the camera
kinds refuse with a code (nothing else depends on the camera)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from r1mc3.cam.fake import FakeRobot
from r1mc3.cam.robot import RobotSource
from r1mc3.cam.service import CamService
from r1mc3.record.runner import CameraClient, Recorder
from r1sim import cpp_policy, world

from tests.integration.test_service_lockstep import Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


class InProc:
    def __init__(self, svc: CamService) -> None:
        self.svc = svc

    def request(self, req: dict[str, Any]) -> dict[str, Any]:
        return self.svc.handle(req)


def test_depth_targets_records_through_r1cam_and_moves_nothing(w: world.World, tmp_path: Path) -> None:
    fake = FakeRobot(depth_hz=20.0, image_hz=15.0, image_bytes=5_000).start()
    cam = CamService(RobotSource(depth_addr=fake.depth_addr, image_command=fake.image_command), tmp_path / "cam")
    rig = Rig(w)
    rig.svc.recorder = Recorder(rig.svc, out_dir=tmp_path, camera=InProc(cam))
    res: dict[str, Any] = {}
    params = {"label": "board_flat_0p45", "distance_m": 0.45, "duration_s": 2.0}

    def go() -> None:
        res["q0"] = rig.ctl.last_q.copy()
        res["run"] = rig.svc.submit({"op": "record.run", "kind": "depth_targets", "params": params,
                                     "dry_run": False}).result  # fmt: skip

    rig.at(0.5, go)
    rig.run(4.5)  # the plan: 2 s + 1 s from t = 0.5 s
    cam.close()
    fake.close()
    assert res["run"].ok and not res["run"].moved
    assert float(abs(rig.ctl.last_q[14:] - res["q0"][14:]).max()) < 1e-9  # no arm / head target moved
    (m,) = list(tmp_path.glob("*_depth_targets_*.json"))
    doc = json.loads(m.read_text())
    assert doc["status"] == "done" and doc["plan"]["moves"] == []
    r1cam = doc["camera"]["r1cam"]
    assert r1cam["kind"] == "depth_targets" and r1cam["label"] == "board_flat_0p45"
    assert r1cam["frames"]["depth"] >= 1 and Path(r1cam["files"][0]["path"]).exists()


def test_without_r1cam_the_camera_kinds_refuse(w: world.World, tmp_path: Path) -> None:
    rig = Rig(w)
    rig.svc.recorder = Recorder(rig.svc, out_dir=tmp_path, camera=CameraClient(port=1))  # nothing listens on port 1
    out = rig.svc.submit({"op": "record.run", "kind": "flat_field", "params": {}, "dry_run": False}).result
    assert out is not None and out.code == "UNSUPPORTED" and not out.moved and "r1cam" in out.details["reason"]
