"""r1d's camera ops (session 9, cam's lines wired by the integrator): `camera.status` / `camera.services` reach r1cam
through its client; the services switch carries the requester + confirm (r1cam refuses it otherwise, the user's Q2);
a silent r1cam is a code, nothing else; the CLI words build exactly these requests."""

from __future__ import annotations

from typing import Any

from r1mc3.cli.main import build_request
from r1mc3.service.schemas import REQUEST, CameraReq
from r1mc3.service.service import Service


class _Cam:
    def __init__(self, reply: dict[str, Any] | None = None, fail: bool = False) -> None:
        self.reply = reply or {"ok": True, "data": {"sources": {"head": "sim"}}}
        self.fail = fail
        self.seen: list[dict[str, Any]] = []

    def request(self, req: dict[str, Any]) -> dict[str, Any]:
        self.seen.append(dict(req))
        if self.fail:
            raise ConnectionRefusedError("no r1cam")
        return self.reply


def _svc(cam: _Cam) -> Service:
    return Service(None, camera=cam)  # type: ignore[arg-type]  # the camera ops never touch the port


def test_camera_status_and_the_services_switch_reach_r1cam_with_requester_and_confirm() -> None:
    cam = _Cam()
    out = _svc(cam).handle({"op": "camera.status", "id": 1})
    assert out["ok"] and out["data"]["sources"] == {"head": "sim"} and out["moved"] is False
    _svc(cam).handle({"op": "camera.services", "action": "stereo", "requester": "operator", "confirm": True})
    assert cam.seen == [
        {"op": "status"},
        {"op": "services", "action": "stereo", "requester": "operator", "confirm": True},
    ]


def test_a_refusal_keeps_r1cams_code_and_a_silent_r1cam_is_unsupported() -> None:
    cam = _Cam({"ok": False, "code": "NOT_OWNER", "details": {"reason": "operator + confirm only"}})
    out = _svc(cam).handle({"op": "camera.services", "action": "stereo"})
    assert not out["ok"] and out["code"] == "NOT_OWNER" and out["details"]["reason"] == "operator + confirm only"
    out = _svc(_Cam(fail=True)).handle({"op": "camera.status"})
    assert not out["ok"] and out["code"] == "UNSUPPORTED" and "r1cam does not answer" in out["details"]["reason"]


def test_the_cli_words_build_the_camera_requests() -> None:
    req, _ = build_request(["camera", "status"])
    assert req == {"requester": "operator", "op": "camera.status"}
    req, _ = build_request(["camera", "services", "--stereo", "--yes"])
    assert req == {"requester": "operator", "op": "camera.services", "action": "stereo", "confirm": True}
    req, _ = build_request(["camera", "services", "--stereo"])
    assert req["confirm"] is False
    assert isinstance(REQUEST.validate_python(req), CameraReq)
