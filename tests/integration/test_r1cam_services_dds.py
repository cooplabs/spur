"""The camera-services switch over the real DDS RPC wire (our unitree_api types, loopback, domain 100): r1cam's
CameraServices against a robot_state server that answers as the R1 does (v2 camera_services.py: ServiceList ->
status 0 running / 1 stopped; ServiceSwitch {"name", "switch"}). Proven here before its first robot use (Q2)."""

from __future__ import annotations

import json

import pytest
from r1mc3 import dds

pytestmark = pytest.mark.needs_dds
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1mc3.cam import services  # noqa: E402
from r1mc3.dds import rpc  # noqa: E402

DOMAIN = 100


def test_the_operator_switch_runs_over_dds_and_nothing_else_reaches_it() -> None:
    rpc.use_loopback()
    state = {"video_hub": 0, "stereo_patch_pc1": 1}

    def lst(_: str) -> tuple[int, str]:
        return 0, json.dumps([{"name": k, "status": v, "protect": False} for k, v in state.items()])

    def sw(p: str) -> tuple[int, str]:
        d = json.loads(p)
        state[d["name"]] = 0 if d["switch"] else 1
        return 0, json.dumps({"name": d["name"], "status": state[d["name"]]})

    handlers = {services.API_SERVICE_LIST: lst, services.API_SERVICE_SWITCH: sw}
    with rpc.RpcServer(services.SERVICE_ROBOT_STATE, handlers, domain=DOMAIN) as srv:
        cs = services.CameraServices(lambda: services.dds_rpc(DOMAIN), poll_s=0.05)
        assert cs.status() == {"video_hub": "running", "stereo_patch_pc1": "stopped"}
        with pytest.raises(services.ServiceRefused):
            cs.switch("stereo", requester="r1d", confirm=True)
        assert not any(e["api_id"] == services.API_SERVICE_SWITCH for e in srv.log)
        out = cs.switch("stereo", requester="operator", confirm=True)
        assert out["ok"] and out["after"] == {"video_hub": "stopped", "stereo_patch_pc1": "running"}
        sw_log = [json.loads(e["parameter"]) for e in srv.log if e["api_id"] == services.API_SERVICE_SWITCH]
        assert sw_log == [{"name": "video_hub", "switch": 0}, {"name": "stereo_patch_pc1", "switch": 1}]
