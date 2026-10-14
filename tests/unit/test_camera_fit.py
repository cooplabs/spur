"""The head-camera fits as tools (r1sim.sysid.camera_fit, session 10; R1CAM_PLAN.md §3): each fit recovers a known
model from synthetic data before it is trusted on HW2's S11 / S12 captures (B5). New on the base (a041731: no tool)."""

from __future__ import annotations

from typing import Any

import pytest
from r1sim.sysid import camera_fit as CF


@pytest.fixture(scope="module")
def syn() -> dict[str, Any]:
    out: dict[str, Any] = CF.synthetic()
    return out


def test_the_lens_fit_recovers_the_model_and_picks_it(syn: dict[str, Any]) -> None:
    p = syn["lens_on_pinhole_k1"]
    # the 4 Oct calibration's values, 0.2 px noise per axis: f 126.87 / cx 259.08 / cy 203.83 / k1 0.02489
    assert p["best"] == "pinhole_k1" and p["rms_px"] < 0.35
    assert abs(p["f"] - 126.94) < 0.5 and abs(p["cx"] - 259.0) < 0.5 and abs(p["cy"] - 203.9) < 0.5
    assert abs(p["dist"][0] - 0.0249) < 0.002
    e = syn["lens_on_equidistant"]
    assert e["best"] == "equidistant" and e["rms_px"] < 0.35 and abs(e["f"] - 126.94) < 0.6


def test_vignetting_latency_depth_are_recovered(syn: dict[str, Any]) -> None:
    assert syn["vignetting"]["gain_max_err"] < 0.01
    assert abs(syn["latency"]["latency_ms"] - syn["latency"]["true_ms"]) <= 5.0  # 7.5 fps frames
    d = syn["depth"]
    assert abs(d["sigma_z_m"] - d["true_sigma_m"]) < 0.0005 and abs(d["median_err_m"]) < 0.001
    st = syn["stereo"]
    assert (
        abs(st["baseline_m"] - st["true_baseline_m"]) < 0.001
        and st["rms_px"] < 0.4
        and abs(st["right_f"] - 126.94) < 0.6
    )


def test_leave_one_pose_out_runs_on_a_subset() -> None:
    views = CF.synthetic_views("pinhole_k1")[::7]
    res = CF.fit_lens(views, kinds=("pinhole_k1",))
    row = res["models"]["pinhole_k1"]
    assert res["views"] == len(views) >= 8 and row["loo_rms_px_median"] is not None
    assert row["loo_rms_px_median"] < 1.0 and res["b5_pass"]
