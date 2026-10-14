"""Session 12 (S11's stand rehearsal finding 2 / not-solid 8): `r1 measure setup` must answer in words within a bounded
time, and the CLI must wait for that time.

The finding: typed with r1cam not answering properly, `r1 measure setup` gave no answer within the CLI's 20 s (the CLI
printed "r1d not reachable" while r1d was still measuring). The causes: (1) r1d's measurement had no overall budget -
every r1cam call could wait its socket's 15 s (r1mc3.cam.service.request) and each of `frames` head frames up to
`wait_s` on top; (2) the CLI waited 20 s for a word whose own waits add up to more (3 frames x 5 s + the fits; ~6 s a
frame under load in the simulator). Now: one budget per measurement (`measure_ops.setup_budget_s` = frames x wait_s +
the fits' margin), every r1cam call bounded by what is left of it (the socket too), a refusal in words when it is
spent; the CLI waits for the budget + a margin.

On the base (1cf3afd): the hung r1cam below answers only after the socket's 15 s (the first test fails its 6 s bound);
the CLI's timeout for `measure setup` is its default 20 s < 3 x 5 + 10 (the second test fails).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from r1mc3.cli.main import build_request
from r1mc3.service.service import Service

from tests.unit.test_measure_ops import StatePort, _setup_req, _svc, _wait


class HungCam:
    """A TCP 'r1cam' that accepts connections and reads the request but never answers (a stuck camera service)."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = int(self.sock.getsockname()[1])
        self.conns: list[socket.socket] = []
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                c, _ = self.sock.accept()
            except OSError:
                continue
            self.conns.append(c)  # held open, never answered

    def close(self) -> None:
        self._stop.set()
        for c in self.conns:
            c.close()
        self.sock.close()
        self.thread.join(timeout=2.0)


def test_a_hung_r1cam_gets_a_refusal_in_words_within_the_measurements_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from r1mc3.record.runner import CameraClient

    cam = HungCam()
    try:
        svc: Service = _svc(StatePort(), None, monkeypatch)
        svc.camera = CameraClient(port=cam.port)
        t0 = time.monotonic()
        out = _wait(svc, {**_setup_req(frames=1), "wait_s": 1.0}, timeout_s=30.0)
        took = time.monotonic() - t0
    finally:
        cam.close()
    assert not out["ok"] and out["code"] in ("UNSUPPORTED", "TIMEOUT"), out
    assert "r1cam" in out["details"]["reason"], out
    assert took < 6.0, took  # budget 1 x 1 s (+ the socket's 1 s grace): the base waited the socket's 15 s


def test_the_cli_waits_for_the_measurement_budget(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"entities": [{"id": "slab", "tags": [{"id": 1, "size_m": 0.08, "xyz": [0, 0, 0]}]}]}))
    _, a = build_request(["measure", "setup", str(plan)])
    assert a.timeout >= 3 * 5.0 + 10.0, a.timeout  # frames x r1d's wait_s default + the fits' margin
    _, a = build_request(["measure", "setup", str(plan), "--frames", "6"])
    assert a.timeout >= 6 * 5.0 + 10.0, a.timeout


def test_frames_that_arrive_in_time_are_still_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The budget changes nothing for a camera that delivers (each frame within its wait)."""
    from tests.unit.test_measure_ops import FakeCam, _head_jpeg

    svc = _svc(StatePort(), FakeCam(_head_jpeg(), delay_s=0.2), monkeypatch)
    out: dict[str, Any] = _wait(svc, {**_setup_req(frames=3), "wait_s": 1.0})
    assert out["ok"], out
