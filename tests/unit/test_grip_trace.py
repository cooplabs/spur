"""Session 12: the grasp trace (r1mc3.hands.trace: what the grip saw, recorded on the robot, replayed offline) and the
narrow pinch's double-speed close (r1mc3.hands.grip.GraspRunner._fast_until).

HW2 day 1 (12 Oct 2026): a thin object's pinch (2.1 cm wide) read NOT_HOLDING 5 / 5 while held, and nothing on the
robot had kept the hand data the verdict judged (tests/assets/hands_hw2/README.md) - r1d now writes each grasp's trace;
and the only B4 object narrower than ~2.6 cm was the only one whose close ran at double speed all the way into it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.hands import grip as G
from r1mc3.hands import trace as T

from tests.unit.grip_fake import FakeHand, aperture_stop


def _object(h: FakeHand, aperture_m: float) -> None:
    h.stops = {g: aperture_stop(h.side, f, aperture_m) for g, f in (("thumb", "index"), ("index", "index"),
                                                                   ("middle", "middle"))}  # fmt: skip


def _traced_grasp(h: FakeHand, grip: str, width: float | None, lift_mm: float, t_s: float = 6.0) -> tuple[Any, Any]:
    params = {"grip": grip, "object_class": "rigid", "width_m": width, "squeeze": None, "surface_z_m": None,
              "contact_height": False}  # fmt: skip
    tr = T.TracingHands(h, h.side, params)
    r = G.GraspRunner(tr, h.side, grip, "rigid", width_m=width)
    t = 0.0
    tr.now_ns = 0
    r.start(0)
    dz = 0.0
    while t < t_s:
        h.step()
        t += 0.01
        now = round(t * 1e9)
        tr.now_ns = now
        st = r.tick(now)
        if st == "TENTATIVE" and lift_mm > 0:
            dz = min(lift_mm / 1000.0, dz + 0.0005)  # the arm lifts 5 cm/s
            r.notify_lift(dz)
            tr.note_lift(dz)
        if st != "OPEN":
            lv = tr.last_verdict
            if lv is None or lv.get("state") != st or lv.get("reason") != r.reason:
                tr.last_verdict = r.verdict().as_dict()
    return r, tr


@pytest.mark.parametrize(("grip", "aperture", "width", "lift", "want"), [
    ("wrap", 0.050, 0.050, 30.0, "HOLDING"),  # a 5 cm wrap, lifted 3 cm
    ("pinch", 0.021, 0.021, 30.0, "HOLDING"),  # a 2.1 cm pinch (the tape's width)
    ("wrap", None, None, 0.0, "NOT_HOLDING"),  # an empty close
])  # fmt: skip
def test_a_recorded_grasp_replays_to_the_same_verdict(
    tmp_path: Path, grip: str, aperture: float | None, width: float | None, lift: float, want: str
) -> None:
    """The runner is deterministic: the recorded states fed back in order at the recorded times give the robot's
    verdict again (same state, reason, stops) - so a changed rule can be judged on the robot's own data."""
    h = FakeHand()
    if aperture is not None:
        _object(h, aperture)
    r, tr = _traced_grasp(h, grip, width, lift)
    assert r.state == want, r.verdict()
    path = tr.save(tmp_path, r, background=False)
    assert path.exists() and path.stat().st_size < 1_000_000
    loaded = T.load(path)
    assert loaded["meta"]["schema"] == T.SCHEMA and loaded["meta"]["verdict"]["state"] == want
    assert loaded["pressure"].shape[1:] == (9, 12) and len(loaded["t_ns"]) == loaded["meta"]["samples"] > 100
    out = T.replay(loaded)
    assert out["same_as_recorded"], (out["verdict"], loaded["meta"]["verdict"])
    assert out["stops"] == r.stops()


def test_the_replay_judges_a_changed_rule_on_the_recorded_data(tmp_path: Path) -> None:
    """The point of the trace: the same recorded close judged with another parameter (here the gap rule for a width
    the hand did not have) - the replay says what the robot would have said."""
    h = FakeHand()
    _object(h, 0.021)
    r, tr = _traced_grasp(h, "pinch", 0.021, 30.0)
    loaded = T.load(tr.save(tmp_path, r, background=False))
    # told 9 cm: the 2.1 cm aperture fails the gap term (>= 0.5 x 9 cm - the pinch's 1.3 cm model allowance, session
    # 13; session 12 used 6 cm here: 3.0 cm needed then, 1.7 cm now - the 2.1 cm passes)
    out = T.replay(loaded, width_m=0.090)
    assert out["verdict"]["state"] == "NOT_HOLDING" and out["verdict"]["reason"].startswith("gap")


def test_the_trace_cli_prints_the_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    h = FakeHand()
    _object(h, 0.05)
    r, tr = _traced_grasp(h, "wrap", 0.05, 30.0)
    p = tr.save(tmp_path, r, background=False)
    assert T.main(["info", str(p)]) == 0 and "HOLDING" in capsys.readouterr().out
    assert T.main(["replay", str(p)]) == 0
    out = capsys.readouterr().out
    assert "same as the robot's: True" in out and "opposition" in out


def test_trace_dir_follows_r1ds_records_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Svc:
        _journal_path = tmp_path / "r1d_journal_x.jsonl"

    assert T.trace_dir(Svc()) == tmp_path
    monkeypatch.setenv("R1_GRASP_TRACE", "0")
    assert T.trace_dir(Svc()) is None
    monkeypatch.delenv("R1_GRASP_TRACE")
    monkeypatch.delenv("R1_RECORD_DIR", raising=False)
    assert T.trace_dir(object()) is None


@pytest.mark.parametrize("side", ["right", "left"])
def test_a_narrow_pinch_slows_down_before_the_object(side: str) -> None:
    """The close runs PREOPEN_FAST x faster only until the tips are PREOPEN_MARGIN_M wider than the known width (v2).
    v3's pinches close PINCH_PAST beyond the preset, where the tips pass each other and the gap grows again: v2's
    'last point that wide' was then the END of the path for any width under ~2.6 cm - the tape's 2.1 cm pinch closed
    at double speed all the way into the object (HW2 day 1: the only B4 object that narrow, and the only one read
    NOT_HOLDING while held). Fails on the base: s_fast 1.0 for 2.1 cm."""
    for width, lo, hi in ((0.021, 0.35, 0.45), (0.054, 0.20, 0.26)):
        r = G.GraspRunner(FakeHand(side), side, "pinch", "rigid", width_m=width)
        r.start(0)
        assert lo <= r._s_fast <= hi, (width, r._s_fast)
        tp = G.tips(side, r._q_pre + r._s_fast * (r._q_closed - r._q_pre))
        gap = float(np.linalg.norm(tp["thumb"] - tp["index"])) - r.aperture_zero["index"]
        assert gap >= width + G.PREOPEN_MARGIN_M - 1e-9  # still the margin wider where the fast part ends
