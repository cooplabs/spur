"""The air-pinch calibration on HW2 day 1's REAL replies (tests/assets/hands_hw2/h3_pinch_*_fix.json; session 12).

H3 (12 Oct 2026, release 5e0130b): both hands' tip pinches on nothing RAN - the thumb and the index both "stall", the
user saw the tips touch - and were refused CALIBRATION_FAILED: the real tip pads read 0.00-0.02 kPa (a ~10 N switch,
H4) vs the 1.5 kPa rule, and the capsule model put the touching tips 6.42 / 9.30 mm inside each other (the old floor
-2 mm). Cloth grasps need the reference. Fails on the base: accept_air_pinch does not exist; hand_cal refused both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from r1mc3.hands import calibration as C

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "hands_hw2"


def _verdict(name: str) -> dict[str, Any]:
    v: dict[str, Any] = json.loads((ASSETS / name).read_text())["data"]["verdict"]
    return v


@pytest.mark.parametrize(("name", "layer_mm"), [("h3_pinch_right_fix.json", -6.42), ("h3_pinch_left_fix.json", -9.30)])
def test_hw2s_real_air_pinches_are_accepted_from_the_stall(name: str, layer_mm: float) -> None:
    v = _verdict(name)
    assert v["evidence"]["terms"]["tip_pressure"]["ok"] is False  # what refused it on the robot
    z, basis, why = C.accept_air_pinch(v)
    assert why is None and basis == "stall"
    assert z is not None and z == pytest.approx(layer_mm / 1000.0)
    hc = C.HandCalibration(None)
    assert hc.set_air_pinch("right", z).layer_zero_m == pytest.approx(layer_mm / 1000.0)  # inside the plausible range


def _edit(v: dict[str, Any], **ev: Any) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(v))
    out["evidence"].update(ev)
    return out


def test_a_close_that_did_not_end_tip_on_tip_is_still_refused() -> None:
    v = _verdict("h3_pinch_right_fix.json")
    # the thumb closed on nothing (no stall): not a tip-on-tip meeting
    z, _, why = C.accept_air_pinch(_edit(v, stops={"thumb": "closed", "index": "stall"}))
    assert z is None and why is not None and "did not press" in why
    # both stalled, but the tips' points 25 mm apart: something between them (an object, a finger's side)
    z, _, why = C.accept_air_pinch(_edit(v, aperture_mm={"index": 25.0}))
    assert z is None and why is not None
    # met by the stall, but the layer implausible (-15 mm: beyond the real hands' -9.3 + margin)
    w = json.loads(json.dumps(v))
    w["evidence"]["terms"]["layer"]["layer_mm"] = -15.0
    z, basis, why = C.accept_air_pinch(w)
    assert z is None and basis == "stall" and why is not None and "outside" in why


def test_the_pads_still_count_when_they_press() -> None:
    """The simulated hand's air pinch (its tip pads register): accepted by tip pressure as before (2.2 mm)."""
    v = _verdict("h3_pinch_right_fix.json")
    w = json.loads(json.dumps(v))
    w["evidence"]["terms"]["tip_pressure"]["ok"] = True
    w["evidence"]["stops"] = {"thumb": "pad", "index": "pad"}
    w["evidence"]["terms"]["layer"]["layer_mm"] = 2.19
    z, basis, why = C.accept_air_pinch(w)
    assert why is None and basis == "tip pressure" and z == pytest.approx(0.00219)


def test_not_a_thin_verdict() -> None:
    z, _, why = C.accept_air_pinch({"evidence": {"terms": {"gap": {"ok": True}}}})
    assert z is None and why is not None and "no layer term" in why
