"""r1d's transport and the `r1` CLI against a fake core port (no C++ needed): JSON lines over localhost TCP, typed
codes for bad input, damp only from an operator, the mode sequences' refusals, the CLI's word -> request mapping and
exit codes; plus the Port surface a binding must offer (ShmPort.missing names what is absent)."""

from __future__ import annotations

import json
import socket
from typing import Any

import numpy as np
import pytest
from r1mc3.cli.main import build_request
from r1mc3.cli.main import main as cli_main
from r1mc3.service.codes import CoreTables
from r1mc3.service.port import Heartbeat, ShmPort
from r1mc3.service.server import R1dServer
from r1mc3.service.service import Service


class FakePort:
    """The core's segment as r1d sees it: a mode, a stop channel, a ring, a mailbox answering at the next tick."""

    def __init__(self, mode: int = 3) -> None:
        self.tables = CoreTables(None)
        self.mode = mode
        self.t = 1_000_000_000
        self.stops: list[tuple[int, int]] = []
        self.epoch = 0
        self.records: list[dict[str, Any]] = []
        self.requests: list[tuple[int, int]] = []

    def missing(self) -> list[str]:
        return []

    def now_ns(self) -> int:
        self.t += 1_000_000
        return self.t

    def heartbeat(self) -> Heartbeat:
        return Heartbeat(1, self.t, 0, 500, 0, 1.0, self.mode, False)

    def guard_heartbeat(self) -> None:
        return None

    def token(self) -> tuple[int, int]:
        return (1, 1)

    def request_stop(self, level: int, requester: int, t_ns: int) -> None:
        self.stops.append((level, requester))
        self.epoch += 1

    def stop_epoch(self) -> int:
        return self.epoch

    def push_upper(self, rec: Any) -> int:
        self.records.append(dict(rec))
        return 0

    def upper_keepalive(self, group: int, cmd_id: int, t_ns: int) -> int:
        return 0

    def request_mode(self, req: int, requester: int, t_ns: int) -> int:
        self.requests.append((req, requester))
        return len(self.requests)

    def mode_reply(self, seq: int) -> tuple[int, int, int] | None:
        req = self.requests[seq - 1][0]
        if req == self.tables.requests["takeover"]:
            return (-3, 8, 0)  # refused: no guard (R1RT_TAKEOVER_GUARD)
        return (0, 0, self.mode)

    def targets(self) -> np.ndarray:
        return np.zeros(26)

    def measured(self) -> np.ndarray:
        return np.zeros(26)


class _NoRelease:
    synchronous = True

    def release(self, timeout_s: float) -> tuple[bool, dict[str, Any]]:
        raise AssertionError("must not be called when the gates refuse")


@pytest.fixture
def server() -> Any:
    port = FakePort()
    svc = Service(port, releaser=_NoRelease())
    with R1dServer(svc, port=0) as srv:
        yield srv, port


def _call(port: int, lines: list[bytes]) -> list[dict[str, Any]]:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        f = s.makefile("rb")
        out = []
        for ln in lines:
            s.sendall(ln + b"\n")
            out.append(json.loads(f.readline()))
        return out


def test_json_lines_codes(server: Any) -> None:
    srv, fake = server
    st, bad, unk, damp, damp_op, cmd = _call(
        srv.port,
        [
            b'{"op": "status", "id": 7}',
            b"not json",
            b'{"op": "fly"}',
            b'{"op": "damp"}',
            b'{"op": "damp", "requester": "operator"}',
            b'{"op": "joint.command", "group": "arm_left", "q": [0, 0.1, 0, 0, 0], "ttl": 0.1}',
        ],
    )
    assert st["ok"] and st["id"] == 7 and st["data"]["core"]["mode"] == "balance"
    assert bad["code"] == "INVALID_ARGS" and unk["code"] == "UNKNOWN_COMMAND"
    assert damp["code"] == "NOT_PERMITTED" and not damp["moved"]
    assert damp_op["ok"] and fake.stops[-1] == (2, 4)  # the damp level, the operator's requester id (fallback)
    assert cmd["ok"] and fake.records[-1]["source"] == 2 and fake.records[-1]["ttl_ns"] == 100_000_000
    assert fake.records[-1]["stop_epoch"] == 1  # stamped with the epoch after the damp request


def test_takeover_refused_by_the_gates_never_releases(server: Any) -> None:
    srv, fake = server
    fake.mode = 0  # MONITOR
    (r,) = _call(srv.port, [b'{"op": "takeover"}'])
    assert r["code"] == "CONTROLLER_REJECTED" and r["details"]["refusal"] == 8 and not r["moved"]
    fake.mode = 3
    (r2,) = _call(srv.port, [b'{"op": "takeover"}'])
    assert r2["code"] == "WRONG_MODE" and r2["details"]["needs"] == "monitor"


def test_cli_words_and_exit_codes(server: Any, capsys: Any) -> None:
    srv, _ = server
    req, _ = build_request(["joint", "arm_right", "0,0.1,0,0.2,0", "--ttl", "0.15", "--kp", "50,50,40,40,30"])
    assert req == {
        "requester": "operator",
        "op": "joint.command",
        "group": "arm_right",
        "q": [0.0, 0.1, 0.0, 0.2, 0.0],
        "ttl": 0.15,
        "kp": [50.0, 50.0, 40.0, 40.0, 30.0],
    }
    rec, _ = build_request(["record", "run", "chirp", "joint=left_knee", "f1_hz=4"])
    assert rec["dry_run"] is True and rec["params"] == {"joint": "left_knee", "f1_hz": 4}  # no --yes: a dry run
    assert build_request(["record", "run", "chirp", "--yes"])[0]["dry_run"] is False
    assert cli_main(["--port", str(srv.port), "status"]) == 0
    assert cli_main(["--port", str(srv.port), "joint", "leg_left", ",".join(["0"] * 6)]) == 1  # WRONG_MODE in balance
    out = capsys.readouterr().out.strip().splitlines()
    assert json.loads(out[-1])["code"] == "WRONG_MODE"
    assert cli_main(["--port", "1", "status"]) == 2  # nothing listens on port 1


def test_shmport_names_what_a_binding_lacks() -> None:
    class OldView:  # an ABI-4 SegmentView: no client-side ring / mailbox functions
        def rt_heartbeat(self) -> None:
            return None

    p = ShmPort(OldView(), None)
    assert set(p.missing()) == {"push_upper", "upper_keepalive", "stop_epoch", "request_mode", "mode_reply"}
    with pytest.raises(NotImplementedError, match="push_upper"):
        p.push_upper({})
