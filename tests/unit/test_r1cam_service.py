"""r1cam's requests over its TCP port (127.0.0.1 only): status, observe (base64 frames + receive times; starts idle
streams), record_start / record_stop, and the camera-services switch as an EXPLICIT operator word only (the user's
answer Q2): refused without operator + confirm, never reached by any other request."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.cam.fake import FakeRobot
from r1mc3.cam.robot import RobotSource
from r1mc3.cam.service import CamServer, CamService, request
from r1mc3.cam.services import API_SERVICE_LIST, API_SERVICE_SWITCH, CameraServices, ServiceRefused


class FakeRobotState:
    """unitree's robot_state service as the R1 answers it (v2's 26 Sep records: 0 running, 1 stopped)."""

    def __init__(self) -> None:
        self.state = {"video_hub": 0, "stereo_patch_pc1": 1, "other": 0}
        self.calls: list[tuple[int, str]] = []

    def call(self, api_id: int, parameter: str = "", *, timeout_s: float = 2.0) -> tuple[int, str]:
        self.calls.append((api_id, parameter))
        if api_id == API_SERVICE_LIST:
            return 0, json.dumps([{"name": k, "status": v, "protect": False} for k, v in self.state.items()])
        if api_id == API_SERVICE_SWITCH:
            p = json.loads(parameter)
            self.state[p["name"]] = 0 if p["switch"] else 1
            return 0, json.dumps({"name": p["name"], "status": self.state[p["name"]]})
        return 3203, ""

    def switches(self) -> list[tuple[int, str]]:
        return [c for c in self.calls if c[0] == API_SERVICE_SWITCH]


@pytest.fixture()
def stack(tmp_path: Path) -> Any:
    fake = FakeRobot(depth_hz=20.0, image_hz=15.0, image_bytes=8_000).start()
    src = RobotSource(depth_addr=fake.depth_addr, image_command=fake.image_command, idle_s=1.0)
    rs = FakeRobotState()
    svc = CamService(src, tmp_path, services=CameraServices(rs, poll_s=0.01, sleep=lambda s: None))
    srv = CamServer(svc, port=0).start()
    yield srv, rs, fake
    srv.close()
    fake.close()


def test_observe_returns_the_newest_frames_and_record_writes_a_manifest(stack: Any) -> None:
    srv, rs, _fake = stack
    st = request(srv.port, {"op": "status", "id": 1})
    assert st["ok"] and st["data"]["source"] == "robot" and not st["data"]["streams"]["left"]["active"]
    ob = request(srv.port, {"op": "observe", "id": 2, "depth": True})
    assert ob["ok"], ob
    left = base64.b64decode(ob["data"]["left"]["b64"])
    assert left[:2] == b"\xff\xd8" and ob["data"]["left"]["width"] == 544
    depth = np.frombuffer(base64.b64decode(ob["data"]["depth"]["b64"]), dtype="<u2")
    assert depth.size == 544 * 448 and ob["data"]["depth"]["src_counter"] >= 1
    r = request(srv.port, {"op": "record_start", "kind": "depth_targets", "label": "board_0p45",
                           "streams": ["left", "depth"]})  # fmt: skip
    assert r["ok"], r
    stop = request(srv.port, {"op": "record_stop", "rec_id": r["data"]["rec_id"]})
    assert stop["ok"] and Path(stop["data"]["files"][0]["path"]).exists()
    assert request(srv.port, {"op": "nope"})["code"] == "UNKNOWN_COMMAND"
    assert rs.switches() == []  # status / observe / recording never touch the camera services


def test_the_services_switch_is_an_explicit_operator_word_only(stack: Any) -> None:
    srv, rs, _fake = stack
    for req in (
        {"op": "services", "action": "stereo"},  # r1d, no confirm
        {"op": "services", "action": "stereo", "requester": "operator"},  # no confirm
        {"op": "services", "action": "stereo", "requester": "r1d", "confirm": True},  # not the operator
        {"op": "services", "action": "stereo", "requester": "operator", "confirm": "yes"},  # confirm must be true
    ):
        out = request(srv.port, req)
        assert out["code"] == "NOT_PERMITTED", (req, out)
    assert rs.switches() == []
    st = request(srv.port, {"op": "services", "action": "status"})
    assert st["data"]["status"] == {"video_hub": "running", "stereo_patch_pc1": "stopped"}
    out = request(srv.port, {"op": "services", "action": "stereo", "requester": "operator", "confirm": True})
    assert out["ok"], out
    assert [json.loads(p) for _, p in rs.switches()] == [
        {"name": "video_hub", "switch": 0},
        {"name": "stereo_patch_pc1", "switch": 1},
    ]  # v2's order: video_hub off first, then the stereo module on
    assert out["data"]["after"] == {"video_hub": "stopped", "stereo_patch_pc1": "running"}


def test_a_switch_that_does_not_take_stops_at_that_step() -> None:
    rs = FakeRobotState()
    orig = rs.call

    def stuck(api_id: int, parameter: str = "", *, timeout_s: float = 2.0) -> tuple[int, str]:
        if api_id == API_SERVICE_SWITCH and json.loads(parameter)["name"] == "video_hub":
            rs.calls.append((api_id, parameter))
            return 0, ""  # accepted, but the service never stops
        return orig(api_id, parameter, timeout_s=timeout_s)

    rs.call = stuck  # type: ignore[method-assign]
    cs = CameraServices(rs, wait_s=0.05, poll_s=0.01, sleep=lambda s: None)
    out = cs.switch("stereo", requester="operator", confirm=True)
    assert not out["ok"] and len(out["steps"]) == 1 and not out["steps"][0]["reached"]
    assert [json.loads(p)["name"] for _, p in rs.switches()] == ["video_hub"]  # the stereo module was not started
    with pytest.raises(ServiceRefused):
        cs.switch("video_hub", requester="r1d", confirm=True)
