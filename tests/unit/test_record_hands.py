"""The hand recordings (r1mc3.record: finger_hold, pad_press) write what the HW1 fits read: the /r1hands/state MCAP
(r1sim.sysid.records.load_hand) with the state next to the command on the wire, and for pad presses the operator's
notes in the fit's format (load_presses) with the scheduled windows in the recording's clock."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.record.runner import Recorder
from r1mc3.service.service import Service

from tests.unit.test_service_server import FakePort

records = pytest.importorskip("r1sim.sysid.records")


class FakeHands:
    def __init__(self) -> None:
        self.cmds: list[dict[str, Any]] = []
        self.q = np.linspace(0.0, 0.3, 7)

    def command(self, side: str, q: Any, **kw: Any) -> dict[str, Any]:
        self.cmds.append({"side": side, "q": np.asarray(q, dtype=float), **kw})
        return {"ok": True}

    def state(self, side: str) -> dict[str, Any]:
        last = self.cmds[-1] if self.cmds else {}
        kp = last.get("kp")
        return {
            "valid": True,
            "q": self.q,
            "dq": np.zeros(7),
            "tau_est": np.full(7, 0.1),
            "temperature": np.full((7, 2), 41.0),
            "pressure": np.full((9, 12), 30000.0),
            "q_cmd": np.asarray(last.get("q", np.zeros(7)), dtype=float),
            "kp": np.full(7, 1.5) if kp is None else np.asarray(kp, dtype=float),
            "kd": np.full(7, 0.2),
            "tau_ff": np.zeros(7) if last.get("tau") is None else np.asarray(last["tau"], dtype=float),
        }


def _run(tmp_path: Path, kind: str, params: dict[str, Any], seconds: float) -> tuple[FakeHands, dict[str, Any]]:
    port = FakePort(mode=0)
    hands = FakeHands()
    svc = Service(port, hands=hands, recorder_factory=lambda s: Recorder(s, out_dir=tmp_path))
    r = svc.submit({"op": "record.run", "kind": kind, "params": params, "dry_run": False}).result
    assert r is not None and r.ok, r
    t = port.t
    while svc.recorder.run is not None:  # type: ignore[union-attr]
        t += 10_000_000
        port.t = t
        svc.tick(t)
        assert t - port.t < seconds * 1e9 + 1e9
    (m,) = list(tmp_path.glob(f"*_{kind}_*.json"))
    return hands, json.loads(m.read_text())


def test_finger_hold_logs_the_fits_format(tmp_path: Path) -> None:
    hands, doc = _run(
        tmp_path, "finger_hold", {"side": "left", "finger": "index", "torque_nm": 0.45, "duration_s": 5}, 5
    )
    assert doc["status"] == "done" and doc["hand_samples"] >= 490
    hold = hands.cmds[1]
    assert hold["kp"][5] == 0.0 and hold["kp"][6] == 0.0 and hold["kp"][0] == 1.5  # index: torque only
    np.testing.assert_allclose(hold["tau"][5:7], [-0.45, -0.45])  # the left index closes negative
    assert hands.cmds[-1]["ttl_s"] == 1.0  # at the end: the finger lets go (open preset)
    rec = records.load_hand(doc["hands_mcap"], "left")
    assert rec.t.size == doc["hand_samples"] and rec.t[-1] == pytest.approx(5.0, abs=0.05)
    assert rec["pressure"].shape == (rec.t.size, 9, 12) and rec["temperature"].shape == (rec.t.size, 7, 2)
    np.testing.assert_allclose(rec["tau_ff"][-1][5:7], [-0.45, -0.45])


def test_pad_press_notes_in_the_fits_format(tmp_path: Path) -> None:
    params = {"side": "right", "pad": 4, "weights_g": [50, 200], "x_mm": 12.5, "press_s": 2, "rest_s": 2}
    _, doc = _run(tmp_path, "pad_press", params, 10)
    assert doc["status"] == "done"
    presses = records.load_presses(doc["press_notes"])
    assert [(p.weight_g, p.x_mm, p.pad, p.side) for p in presses] == [
        (50.0, 12.5, 4, "right"),
        (200.0, 12.5, 4, "right"),
    ]
    assert presses[0].t_start == pytest.approx(2.0, abs=0.02) and presses[1].t_end == pytest.approx(8.0, abs=0.02)
    rec = records.load_hand(doc["hands_mcap"], "right")
    assert rec.t[-1] == pytest.approx(10.0, abs=0.05)


def test_a_hand_recording_ends_the_sides_kept_grip(tmp_path: Path) -> None:
    """Session 11 (contact decides): a NOT_HOLDING grip keeps re-sending its grip until told to open - a hand recording
    (pad_press / finger_hold) must take the hand like the operator's `r1 hand`, so the grip ends when the recording
    starts. Fails on 1e55bdd + grip's rule: the grip stays in r1d's table and keeps ticking over the recording."""
    from unittest import mock

    port = FakePort(mode=0)
    hands = FakeHands()
    svc = Service(port, hands=hands, recorder_factory=lambda s: Recorder(s, out_dir=tmp_path))
    svc.l2.grips["left"] = mock.MagicMock()
    svc.l2.grips["right"] = mock.MagicMock()
    r = svc.submit({"op": "record.run", "kind": "pad_press", "params": {"side": "left", "weights_g": [50]},
                    "dry_run": False}).result  # fmt: skip
    assert r is not None and r.ok, r
    assert "left" not in svc.l2.grips  # the recorded hand's grip ended
    assert "right" in svc.l2.grips  # the other hand's grip is not touched
