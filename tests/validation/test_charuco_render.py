"""The ChArUco render test (session 9, B5's board choice): the chosen board detects over the head camera's field at all
three S12 distances and the lens re-fit from its detections recovers the model; the drafted 40 mm board does not reach
0.45 / 0.6 m (why it was replaced). Numbers: docs/hardware/CHARUCO.md."""

from __future__ import annotations

from typing import Any

import pytest
from r1sim import charuco_render as cr

pytestmark = pytest.mark.needs_gl  # every test renders the board through the head camera


@pytest.fixture(scope="module")
def chosen() -> dict[str, Any]:
    return cr.run_board(cr.CHOSEN)


def test_the_chosen_board_detects_at_every_s12_distance(chosen: dict[str, Any]) -> None:
    by_z = {p["z_m"]: p for p in chosen["per_distance"]}
    for z in cr.DISTANCES_M:
        assert by_z[z]["sharp"]["found_frac"] >= 0.4, by_z[z]  # 0.46 / 0.51 / 0.51 at 0.3 / 0.45 / 0.6 m
        assert by_z[z]["sharp"]["err_px_p95"] < 1.0
    for z in (0.3, 0.45):  # a 1 px blur (focus / H.264 / motion): still found at 0.3 / 0.45 m (0.38 / 0.37)
        assert by_z[z]["soft"]["found_frac"] >= 0.25, by_z[z]
    assert chosen["coverage_all"]["sharp"] >= 0.85  # 0.875 of the 8 x 6 image cells over the sweep


def test_the_lens_refit_from_the_chosen_board_meets_b5(chosen: dict[str, Any]) -> None:
    fit = chosen["fit"]
    t = fit["truth"]
    assert fit["rms_px"] <= 1.0  # B5's full-field target (0.19 px here: the render's floor)
    assert abs(fit["f"] - t["f"]) / t["f"] < 0.01  # 127.15 vs 126.94
    assert abs(fit["cx"] - t["cx"]) < 1.5 and abs(fit["cy"] - t["cy"]) < 1.5
    assert abs(fit["k1"] - t["k1"]) < 0.003  # 0.0250 vs 0.0249


def test_the_drafted_40mm_board_is_lost_beyond_0p3_m() -> None:
    r = cr.run_board(cr.DRAFT)
    by_z = {p["z_m"]: p for p in r["per_distance"]}
    assert by_z[0.45]["sharp"]["found_frac"] < 0.1  # 0.04
    assert by_z[0.6]["sharp"]["found_frac"] == 0.0
