"""r1mc3.hands.calibration (session 10): the per-hand air-pinch reference r1d keeps (memory + R1_HAND_CAL_FILE), and
the `r1 calibrate-pinch` word. On the base (a041731) the module and the word do not exist."""

from __future__ import annotations

from pathlib import Path

import pytest
from r1mc3.cli.main import build_request
from r1mc3.hands.calibration import AIR_PINCH_MAX_M, HandCalibration


def test_kept_in_memory_and_in_the_file(tmp_path: Path) -> None:
    f = tmp_path / "sub" / "cal.json"
    hc = HandCalibration(f)
    assert hc.layer_zero("right") is None and hc.summary() == {"left": None, "right": None}
    hc.set_air_pinch("right", 0.0022)
    assert hc.layer_zero("right") == pytest.approx(0.0022) and f.exists()
    again = HandCalibration(f)
    assert again.layer_zero("right") == pytest.approx(0.0022) and again.air_pinch["right"].source == "file"
    assert again.layer_zero("left") is None


def test_an_implausible_reference_is_refused_and_the_old_kept(tmp_path: Path) -> None:
    hc = HandCalibration(tmp_path / "cal.json")
    hc.set_air_pinch("left", 0.002)
    with pytest.raises(ValueError, match="outside"):
        hc.set_air_pinch("left", AIR_PINCH_MAX_M + 0.001)  # e.g. something between the tips
    assert hc.layer_zero("left") == pytest.approx(0.002)
    with pytest.raises(ValueError):
        hc.set_air_pinch("middle", 0.002)


def test_no_file_without_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("R1_HAND_CAL_FILE", raising=False)
    hc = HandCalibration()
    hc.set_air_pinch("right", 0.001)
    assert hc.path is None and hc.layer_zero("right") == pytest.approx(0.001)


def test_the_cli_word() -> None:
    req, _ = build_request(["calibrate-pinch", "left"])
    assert req["op"] == "hand.calibrate" and req["side"] == "left" and req["kind"] == "air_pinch" and req["wait"]
    req, _ = build_request(["calibrate-pinch", "right", "--no-wait"])
    assert req["wait"] is False
