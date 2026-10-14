"""Unitree's RPC over DDS (the request / response pattern of unitree_sdk2's `robot::Client` / `robot::Server`) and the
motion switcher (CheckMode / SelectMode / ReleaseMode) on top of it (session 5, the bridge agent).

Wire format (rt/idl/src/unitree/idl/api/SOURCE.md): a client writes `unitree_api::msg::dds_::Request_` on
`rt/api/<service>/request` (header.identity = a unique request id + the api id, lease 0, policy priority 0 / noreply,
`parameter` = the api's JSON); the service answers `Response_` on `rt/api/<service>/response` with the same identity,
`status.code` (0 = success, else the service's error code) and `data` (JSON). Every client reads every response on
the topic and keeps the ones carrying its own request ids.

Motion switcher (unitree_sdk2 include/unitree/robot/b2/motion_switcher/motion_switcher_api.hpp): service
`motion_switcher`, CheckMode 1001 -> data `{"form": ..., "name": ...}` (v2 reads `name` "ai" while Unitree's built-in
controller runs, "" after a release), SelectMode 1002 (parameter `{"name": <mode>}`), ReleaseMode 1003. r1d runs the
release between the core's TAKEOVER and RELEASED / ABORT requests (DESIGN §3.5: never in an RT thread); the simulator's
DDS bridge (`r1sim.dds_bridge`) serves the same service for its emulated built-in.

QoS (one rule for every topic we touch, so we match whatever the peer chose): writers RELIABLE + KEEP_LAST (a reliable
writer matches reliable and best-effort readers), readers BEST_EFFORT + KEEP_LAST (a best-effort reader matches both
kinds of writers). A request is re-sent with the SAME id every `resend_s` until its answer arrives (a request written
before discovery matched the server is otherwise lost); the server answers a repeated id from its cache without
calling the handler twice.

Client-side codes (ours, not Unitree's): RPC_TIMEOUT (-1) = no answer within the timeout.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

from .api_types import Request_, Response_, request, response

SERVICE_MOTION_SWITCHER = "motion_switcher"
MOTION_SWITCHER_API_VERSION = "1.0.0.1"
API_CHECK_MODE = 1001
API_SELECT_MODE = 1002
API_RELEASE_MODE = 1003
API_SET_SILENT = 1004
API_GET_SILENT = 1005
# motion_switcher_error.hpp
ERR_SWITCH_PARAMETER = 7001
ERR_SWITCH_BUSY = 7002
ERR_SWITCH_NAME = 7004
ERR_SWITCH_EXEC_RELEASE = 7008
RPC_TIMEOUT = -1  # ours: no answer in time (the client's view; nothing is known about the server)
ERR_UNKNOWN_API = 3203  # ours (the server's answer to an api id it does not serve; Unitree's own code is not known)

WRITER_QOS = Qos(Policy.Reliability.Reliable(duration(milliseconds=100)), Policy.History.KeepLast(16))
READER_QOS = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(64))

_ids = itertools.count(1)


def request_topic(service: str) -> str:
    return f"rt/api/{service}/request"


def response_topic(service: str) -> str:
    return f"rt/api/{service}/response"


def new_request_id() -> int:
    """Unique per process and over time (the SDK uses a time-based counter): pid, wall-clock ms, a counter."""
    return ((os.getpid() & 0xFFFF) << 47) | ((time.time_ns() // 1_000_000 & 0x3FFFFFFF) << 17) | (next(_ids) & 0x1FFFF)


class RpcClient:
    """Call a Unitree-style DDS service. Thread-safe (one call at a time per client)."""

    def __init__(self, service: str, *, domain: int = 0, participant: DomainParticipant | None = None) -> None:
        self.service = service
        self.dp = participant if participant is not None else DomainParticipant(domain)
        self._req_topic = Topic(self.dp, request_topic(service), Request_)
        self._rsp_topic = Topic(self.dp, response_topic(service), Response_)
        self._writer = DataWriter(self.dp, self._req_topic, qos=WRITER_QOS)
        self._reader = DataReader(self.dp, self._rsp_topic, qos=READER_QOS)
        self._lock = threading.Lock()
        self.calls = 0
        self.resends = 0

    def call(
        self, api_id: int, parameter: str = "", *, timeout_s: float = 2.0, resend_s: float = 0.25
    ) -> tuple[int, str]:
        """(status code, data): 0 = success; RPC_TIMEOUT when no answer arrived within `timeout_s`."""
        with self._lock:
            self.calls += 1
            rid = new_request_id()
            req = request(rid, api_id, parameter)
            t_end = time.monotonic() + timeout_s
            t_send = 0.0
            while True:
                now = time.monotonic()
                if now >= t_end:
                    return RPC_TIMEOUT, ""
                if now >= t_send:
                    if t_send > 0.0:
                        self.resends += 1
                    self._writer.write(req)
                    t_send = now + resend_s
                taken: list[Any] = self._reader.take(N=64)
                for rsp in taken:
                    if isinstance(rsp, Response_) and rsp.header.identity.id == rid:
                        return int(rsp.header.status.code), str(rsp.data)
                time.sleep(0.001)


Handler = Callable[[str], tuple[int, str]]


class RpcServer:
    """Serve a Unitree-style DDS service: `handlers[api_id](parameter) -> (code, data)`, called in the server's own
    thread (poll every `poll_s`); a request id seen before is answered from the cache (a client's re-send)."""

    def __init__(
        self,
        service: str,
        handlers: Mapping[int, Handler],
        *,
        domain: int = 0,
        participant: DomainParticipant | None = None,
        poll_s: float = 0.001,
    ) -> None:
        self.service = service
        self.handlers = dict(handlers)
        self.poll_s = poll_s
        self.dp = participant if participant is not None else DomainParticipant(domain)
        self._req_topic = Topic(self.dp, request_topic(service), Request_)
        self._rsp_topic = Topic(self.dp, response_topic(service), Response_)
        self._reader = DataReader(self.dp, self._req_topic, qos=READER_QOS)
        self._writer = DataWriter(self.dp, self._rsp_topic, qos=WRITER_QOS)
        self._answered: OrderedDict[int, tuple[int, str]] = OrderedDict()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.log: list[dict[str, Any]] = []  # every distinct request: wall time, api id, parameter, code

    def start(self) -> RpcServer:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"rpc-{self.service}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
            self._thread = None

    def __enter__(self) -> RpcServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def poll_once(self) -> int:
        """Serve every pending request; returns how many were taken."""
        n = 0
        taken: list[Any] = self._reader.take(N=64)
        for req in taken:
            if not isinstance(req, Request_):
                continue
            n += 1
            ident = req.header.identity
            rid = int(ident.id)
            if rid in self._answered:
                code, data = self._answered[rid]
            else:
                h = self.handlers.get(int(ident.api_id))
                code, data = (ERR_UNKNOWN_API, "") if h is None else h(str(req.parameter))
                self._answered[rid] = (code, data)
                while len(self._answered) > 256:
                    self._answered.popitem(last=False)
                self.log.append(dict(wall=time.time(), api_id=int(ident.api_id), parameter=req.parameter, code=code))
            if not req.header.policy.noreply:
                self._writer.write(response(ident, code, data))
        return n

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            time.sleep(self.poll_s)


class MotionSwitcherClient:
    """unitree_sdk2's b2::MotionSwitcherClient over our RPC client: check_mode / select_mode / release_mode."""

    def __init__(self, domain: int = 0, *, participant: DomainParticipant | None = None) -> None:
        self.rpc = RpcClient(SERVICE_MOTION_SWITCHER, domain=domain, participant=participant)

    def check_mode(self, timeout_s: float = 2.0) -> tuple[int, dict[str, str]]:
        """(code, {"form", "name"}): name "" = no built-in mode active (released)."""
        code, data = self.rpc.call(API_CHECK_MODE, "", timeout_s=timeout_s)
        if code != 0 or not data:
            return code, {}
        d = json.loads(data)
        return code, {"form": str(d.get("form", "")), "name": str(d.get("name", ""))}

    def select_mode(self, name: str, timeout_s: float = 5.0) -> int:
        return self.rpc.call(API_SELECT_MODE, json.dumps({"name": name}), timeout_s=timeout_s)[0]

    def release_mode(self, timeout_s: float = 5.0) -> int:
        return self.rpc.call(API_RELEASE_MODE, "", timeout_s=timeout_s)[0]


class MotionSwitcherServer(RpcServer):
    """The motion switcher's service side for an emulated built-in: `mode()` -> the active mode name ("" = none),
    `release()` -> True when the built-in let go, `select(name)` -> True when it came back."""

    def __init__(
        self,
        *,
        mode: Callable[[], str],
        release: Callable[[], bool],
        select: Callable[[str], bool] | None = None,
        form: str = "0",
        domain: int = 0,
        participant: DomainParticipant | None = None,
    ) -> None:
        def check(_: str) -> tuple[int, str]:
            name = mode()
            return 0, json.dumps({"form": form, "name": name} if name else {"name": ""})

        def rel(_: str) -> tuple[int, str]:
            return (0, "") if release() else (ERR_SWITCH_EXEC_RELEASE, "")

        def sel(param: str) -> tuple[int, str]:
            try:
                name = str(json.loads(param)["name"]) if param else ""
            except (ValueError, KeyError, TypeError):
                return ERR_SWITCH_PARAMETER, ""
            if select is None or not name:
                return ERR_SWITCH_NAME, ""
            return (0, "") if select(name) else (ERR_SWITCH_BUSY, "")

        super().__init__(
            SERVICE_MOTION_SWITCHER,
            {API_CHECK_MODE: check, API_RELEASE_MODE: rel, API_SELECT_MODE: sel},
            domain=domain,
            participant=participant,
        )


# CycloneDDS on the loopback interface only, unicast discovery (the contract's test configuration: every test and
# every simulator run stays on the loopback; the robot uses its own config on the motor NIC). The loopback is "lo0" on
# macOS and "lo" on Linux (session 6, the Linux build check: CycloneDDS refuses a domain whose named interface does
# not exist).
LOOPBACK_IFACE = "lo0" if sys.platform == "darwin" else "lo"
LOOPBACK_URI = (
    f'<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="{LOOPBACK_IFACE}"/></Interfaces>'
    "<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>"
    '<Peers><Peer address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'
)


def use_loopback(env: dict[str, str] | None = None) -> dict[str, str]:
    """Set CYCLONEDDS_URI to the loopback configuration in `env` (default: this process's environment; CycloneDDS reads
    it when a process creates its first participant of a domain) and return the environment."""
    e: Any = os.environ if env is None else env
    e["CYCLONEDDS_URI"] = LOOPBACK_URI
    return dict(e)
