"""The pad fit (r1sim.sysid.pads) recovers a KNOWN gain and footprint from synthetic presses through the simulator's
pad model (bilinear cells, detection floor, saturation, per-cell offsets, noise, glitches); the press notes and the
hands' state topic round-trip. Prepared for HW1 (A10 rows 2-3); the simulator's pads are unchanged."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from r1sim.sysid import pads, records


@pytest.fixture(scope="module")
def geom() -> dict[int, pads.PadGeom]:
    return pads.pad_geometry("left")


def test_the_geometry_is_the_simulators(geom: dict[int, pads.PadGeom]) -> None:
    assert sorted(geom) == list(range(9))
    for k, g in geom.items():
        assert g.half_long_mm > 0 and g.x_center_mm > 0  # distal of the joint axis
        assert len(g.slots) in (3, 4)
        lo, hi = g.footprint_mm
        assert hi - lo == pytest.approx(2 * (g.half_long_mm + g.margin_mm))
        assert k in (1, 3, 5) or len(g.slots) == 4  # the tips are the 3x1 pads


@pytest.mark.parametrize(("k", "gain", "shrink"), [(2, 0.6, 0.7), (3, 1.4, 0.8), (7, 0.9, 1.0)])
def test_gain_and_footprint_are_recovered(geom: dict[int, pads.PadGeom], k: int, gain: float, shrink: float) -> None:
    g = geom[k]
    plan = pads.press_plan(g, centre_weights_g=(5.0, 20.0, 100.0, 200.0, 500.0))
    truth = pads.PadTruth(gain={k: gain}, half_long_mm={k: g.half_long_mm * shrink})
    rec = pads.synthetic_presses("left", plan, truth, seed=7)
    fit = pads.fit_hand(rec, plan, geom)[k]
    h = g.half_long_mm * shrink + g.margin_mm
    lo, hi = g.x_center_mm - h, g.x_center_mm + h
    step = 2.0  # the protocol's spacing: the ends are known to +-step / 2
    assert fit.sim_pad_gain == pytest.approx(gain, rel=0.03)
    assert abs(fit.x0_mm - lo) <= step / 2 + 0.25 and abs(fit.x1_mm - hi) <= step / 2 + 0.25
    assert fit.centre_slope_raw_per_n > 0
    # the detection floor (10 g per taxel in the model, split over the pad's cells): 5 g never reads
    assert fit.floor_g >= 5.0


def test_press_notes_and_the_hands_topic_round_trip(tmp_path: Path, geom: dict[int, pads.PadGeom]) -> None:
    k = 4
    plan = pads.press_plan(geom[k], step_mm=4.0)
    notes = tmp_path / "presses.yaml"
    records.save_presses(notes, plan)
    assert records.load_presses(notes) == plan
    rec = pads.synthetic_presses("left", plan, pads.PadTruth(gain={k: 0.8}, half_long_mm={}), seed=8, rate_hz=50.0)
    path = tmp_path / "hands.mcap"
    records.write_hands_mcap(path, [rec])
    back = records.load_hand(path, "left")
    assert np.array_equal(back["pressure"], rec["pressure"])
    fit = pads.fit_hand(back, records.load_presses(notes), geom)[k]
    assert fit.sim_pad_gain == pytest.approx(0.8, rel=0.03)


def test_the_pinch_check_agrees_with_the_finger_torque(geom: dict[int, pads.PadGeom]) -> None:
    """A thin-block contact of known force at a measured place: the fitted pad reads the force the finger's torque
    implies (F = tau / lever) - and a contact past the footprint end is flagged."""
    k = 2
    g = geom[k]
    plan = pads.press_plan(g)
    truth = pads.PadTruth(gain={k: 0.6}, half_long_mm={})
    fit = pads.fit_hand(pads.synthetic_presses("left", plan, truth, seed=9), plan, geom)[k]
    x = g.x_center_mm + 3.0
    pinch = records.Press("left", k, x, 520.0, 2.0, 7.0)  # ~5.1 N, a thin block's grip force on 9 Oct (5.19 N)
    rec = pads.synthetic_presses("left", [pinch], truth, seed=10)
    chk = pads.pinch_check(rec, fit, g.slots, x, (2.5, 7.0), baseline=(0.5, 1.9),
                           torque_nm=pinch.force_n * x * 1e-3, lever_mm=x)  # fmt: skip
    assert chk.inside_footprint
    assert chk.ratio == pytest.approx(1.0, abs=0.05)
    outside = pads.pinch_check(rec, fit, g.slots, fit.x1_mm + 3.0, (2.5, 7.0), baseline=(0.5, 1.9))
    assert not outside.inside_footprint
