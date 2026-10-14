"""The camera impact test's harness (tools/cam/impact.py) runs end to end on the DDS-bridge stack: five windows (off /
work / work_fast / calib / off again; session 12: day 1's stream / record windows ran all three streams continuously),
the camera streams only in its windows (the depth ~39 Mbit/s while connected; on demand in `work`), r1cam's
calibration recording drops nothing, and every RT number the HW2 verdict needs is present. The NUMBERS on this Mac are
relative (no RT scheduling, "macOS, under load"); the verdict is the robot's (HW2, before S11 / S12)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
from r1sim import stack

pytestmark = [pytest.mark.slow, pytest.mark.realtime, pytest.mark.needs_dds]
if not stack.available():
    pytest.skip("the DDS build of r1rt / r1guard or CycloneDDS's Python binding is missing", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cam_impact", ROOT / "tools" / "cam" / "impact.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_impact_harness_measures_the_windows(tmp_path: Path) -> None:
    tool = _tool()
    os.environ.setdefault("PYTHONPATH", str(ROOT / "python"))
    a = tool.argparse.Namespace(where="sim", domain=104, burn_ms=0.0, scratch=str(tmp_path), seconds=4.0, out="")
    res = tool.run_sim(a)
    w = res["windows"]
    assert list(w) == list(tool.CONDITIONS)
    for k in tool.CONDITIONS:
        r = w[k]["r1rt"]
        assert r["mode"] == "ON_STAND" and r["ticks"] > 1500 and r["lowstate_rate_hz"] > 50
        assert w[k]["lowstate_age_at_r1rt"]["n"] > 100 and "r1rt" in w[k]["cpu"]["proc_pct"]
        assert w[k]["r1guard"]["lateness"]["n"] > 0
    assert w["off"]["link_mbit_s"] == 0.0 and w["off_again"]["link_mbit_s"] == 0.0
    assert 20.0 < w["work_fast"]["link_mbit_s"] < 50.0 and 25.0 < w["calib"]["link_mbit_s"] < 50.0  # the ~39 Mbit/s
    # session 12: an observe every 3 s holds the depth only until its frame + 0.5 s (day 1: 3 s after each request)
    assert w["work"]["link_mbit_s"] < 0.75 * w["calib"]["link_mbit_s"], (w["work"], w["calib"])
    rec = w["calib"]["r1cam_recording"]
    assert (
        rec["dropped_queue"] == 0
        and min(rec["frames"].values()) > 5
        and set(rec["frames"]) == {"left", "right", "depth"}
    )
    for k in ("work", "work_fast"):
        assert w[k]["r1cam"]["streams"]["right"]["frames"] == 0  # the right image never streamed while working
    assert "r1cam" in w["work"]["cpu"]["proc_pct"]
    assert len(tool.table(res)) == 2 + 5
    assert tool.verdict(w["off"])["checks"].keys() == {
        "no_skipped_deadlines", "lateness_max_le_0p4ms", "age_p99_le_2ms", "rate_ge_1000"
    }  # fmt: skip


def test_the_impact_harness_with_the_simulated_head_camera(tmp_path: Path) -> None:
    """Session 11 (`impact.py sim --camera rendered`): the same four windows with r1cam's SIM process rendering the
    bridge robot's left + right + depth (r1sim.bridge_camera) - frames only in the camera windows, nothing dropped by
    the recording, every RT number present. Base (1e55bdd): no --camera option (the bench stand-in only)."""
    tool = _tool()
    os.environ.setdefault("PYTHONPATH", str(ROOT / "python"))
    a = tool.argparse.Namespace(where="sim", camera="rendered", domain=109, scratch=str(tmp_path), seconds=4.0, out="")
    res = tool.run_sim(a)
    w = res["windows"]
    assert res["mode"] == "sim-rendered" and list(w) == list(tool.CONDITIONS)
    for k in tool.CONDITIONS:
        r = w[k]["r1rt"]
        assert r["mode"] == "ON_STAND" and r["ticks"] > 1500 and r["lowstate_rate_hz"] > 50
        assert w[k]["lowstate_age_at_r1rt"]["n"] > 100 and "r1cam" in w[k]["cpu"]["proc_pct"]
    assert w["work_fast"]["frames_rendered"] > 10 and w["calib"]["frames_rendered"] > 10
    rec = w["calib"]["r1cam_recording"]
    assert rec["dropped_queue"] == 0 and min(rec["frames"].values()) > 5
    assert len(tool.table(res)) == 2 + 5
