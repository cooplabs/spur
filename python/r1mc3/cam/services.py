"""Unitree's camera services on the robot (moves nothing): the stereo module (`stereo_patch_pc1`: left / right images +
native depth) and `video_hub` (the main head camera) exclude each other; after every reboot video_hub is on and the
stereo module off (v2 robot_side/camera_services.py, 26 Sep records: status 0 = running, 1 = stopped).

The user (11 Oct, R1CAM_PLAN.md Q2): switching them is allowed as an EXPLICIT OPERATOR word only - never automatic.
`CameraServices.switch` refuses unless the requester is the operator AND the request confirms; r1cam never calls it on
its own (no start-up, observe or recording path reaches it; a test checks). Proven in the sim / on the bench first and
confirmed with the user before its first use on the robot (HW2).

The RPC is unitree_sdk2's robot_state client (go2/robot_state: service "robot_state", ServiceSwitch 1001 with
{"name", "switch"}, ServiceList 1003 -> [{"name", "status", "protect"}]) over our DDS RPC (r1mc3.dds.rpc; the
optional `dds` extra, imported only when a real client is made).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

SERVICE_ROBOT_STATE = "robot_state"
API_SERVICE_SWITCH = 1001
API_SERVICE_LIST = 1003
CAMERA_SERVICES = ("video_hub", "stereo_patch_pc1")
TARGETS: dict[str, tuple[tuple[str, bool], ...]] = {
    "stereo": (("video_hub", False), ("stereo_patch_pc1", True)),  # what r1cam's RobotSource needs
    "video_hub": (("stereo_patch_pc1", False), ("video_hub", True)),  # back to Unitree's default
}
RUNNING, STOPPED = 0, 1


@runtime_checkable
class Rpc(Protocol):
    def call(self, api_id: int, parameter: str = "", *, timeout_s: float = 2.0) -> tuple[int, str]: ...


class ServiceRefused(PermissionError):
    pass


def dds_rpc(domain: int = 0) -> Rpc:
    from r1mc3.dds.rpc import RpcClient

    return RpcClient(SERVICE_ROBOT_STATE, domain=domain)


class CameraServices:
    def __init__(
        self,
        rpc: Rpc | Callable[[], Rpc],
        *,
        wait_s: float = 12.0,
        poll_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rpc_or_factory = rpc
        self._rpc: Rpc | None = rpc if isinstance(rpc, Rpc) else None
        self.wait_s, self.poll_s, self.sleep = float(wait_s), float(poll_s), sleep
        self.switches: list[dict[str, Any]] = []  # every switch request sent (for the status / the tests)

    def _client(self) -> Rpc:
        if self._rpc is None:
            factory: Any = self._rpc_or_factory
            self._rpc = factory()
        assert self._rpc is not None
        return self._rpc

    def status(self, timeout_s: float = 3.0) -> dict[str, str]:
        """{"video_hub": "running" | "stopped", "stereo_patch_pc1": ...} (read-only)."""
        code, data = self._client().call(API_SERVICE_LIST, "", timeout_s=timeout_s)
        if code != 0:
            raise RuntimeError(f"ServiceList code {code}")
        lst = json.loads(data or "[]")
        st = {str(s["name"]): int(s["status"]) for s in lst}
        return {n: ("running" if st[n] == RUNNING else "stopped") for n in CAMERA_SERVICES if n in st}

    def switch(self, target: str, *, requester: str, confirm: bool) -> dict[str, Any]:
        """The operator's word: `target` "stereo" (video_hub off, stereo_patch_pc1 on) or "video_hub" (the reverse).
        Refused (ServiceRefused) unless requester == "operator" and confirm is True; moves nothing."""
        if target not in TARGETS:
            raise ValueError(f"target: one of {', '.join(TARGETS)}")
        if requester != "operator" or confirm is not True:
            raise ServiceRefused("the camera services switch is an explicit operator word (operator + confirm)")
        before = self.status()
        steps = []
        for name, on in TARGETS[target]:
            code, _ = self._client().call(API_SERVICE_SWITCH, json.dumps({"name": name, "switch": int(on)}),
                                          timeout_s=5.0)  # fmt: skip
            self.switches.append({"name": name, "on": on, "code": code})
            want = "running" if on else "stopped"
            t0 = time.monotonic()
            st = self.status().get(name)
            while st != want and time.monotonic() - t0 < self.wait_s:  # the status follows a few seconds later
                self.sleep(self.poll_s)
                st = self.status().get(name)
            steps.append({"name": name, "on": on, "code": code, "status": st, "reached": st == want})
            if st != want:
                break
        return {"target": target, "before": before, "steps": steps, "after": self.status(),
                "ok": all(s["reached"] for s in steps) and len(steps) == len(TARGETS[target])}  # fmt: skip
