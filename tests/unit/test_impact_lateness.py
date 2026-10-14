"""The camera impact tool's verdict reads each window's OWN lateness (session 12): HW2 day 1's C1 (`cam_impact.json`,
12 Oct 2026, copied verbatim into tests/assets/cam_hw2/) failed the lateness check in EVERY window - the OFF baseline
too - because the check read r1rt's cumulative `max_us` (`max_ms_so_far` 3001 ms = T6's deliberate 3 s pause minutes
earlier). The real C1 failure (the state rate and age while all three streams ran) must stay visible."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from r1mc3.cam import streams

ROOT = Path(__file__).resolve().parents[2]
DAY1 = ROOT / "tests" / "assets" / "cam_hw2" / "cam_impact_20261013.json"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cam_impact_unit", ROOT / "tools" / "cam" / "impact.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_day1_off_window_passes_once_lateness_is_the_windows_own() -> None:
    """Base: lateness_max_le_0p4ms False in all four windows (3001.111 ms 'so far'), so even OFF failed."""
    tool = _tool()
    w = json.loads(DAY1.read_text())["windows"]
    assert w["off"]["r1rt"]["lateness"]["max_ms_so_far"] > 3000  # the stale cumulative max is in the file
    for k in ("off", "stream", "record", "off_again"):
        assert tool.verdict(w[k])["checks"]["lateness_max_le_0p4ms"] is True, k
    assert tool.verdict(w["off"])["pass"] and tool.verdict(w["off_again"])["pass"]
    # the real finding stays: while all three streams ran, the rate and the age failed
    for k in ("stream", "record"):
        c = tool.verdict(w[k])["checks"]
        assert c["rate_ge_1000"] is False and c["age_p99_le_2ms"] is False and not tool.verdict(w[k])["pass"]


def test_the_window_bound_is_the_new_max_or_the_highest_bucket_reached() -> None:
    tool = _tool()
    h0: dict[str, Any] = {
        "n": 100,
        "max_us": 3_001_111.0,
        "hist_us": {"<50": 90, "<100": 9, "<250": 1, "<500": 0, ">=20000": 0},
    }
    h1: dict[str, Any] = {
        "n": 300,
        "max_us": 3_001_111.0,
        "hist_us": {"<50": 280, "<100": 15, "<250": 1, "<500": 4, ">=20000": 0},
    }
    late = tool._late_counts(h0, h1)
    assert late["max_ms"] is None and late["window_max_bound_ms"] == 0.5 and late["over_0p5ms"] == 0
    assert tool.window_lateness_ok(late)
    h2 = {**h1, "n": 400, "max_us": 3_100_000.0, "hist_us": {**h1["hist_us"], ">=20000": 1}}
    late2 = tool._late_counts(h1, h2)
    assert late2["max_ms"] == 3100.0 and late2["window_max_bound_ms"] == 3100.0 and not tool.window_lateness_ok(late2)
    h3 = {**h1, "n": 400, "max_us": 3_001_111.0, "hist_us": {**h1["hist_us"], "<1000": 1}}
    assert not tool.window_lateness_ok(tool._late_counts(h1, h3))  # a tick past 0.5 ms in the window


def test_the_windows_follow_the_fewer_streams_rule() -> None:
    """Base: windows `stream` / `record` with left + right + depth continuously (observe right=True)."""
    tool = _tool()
    assert tool.CONDITIONS == ("off", "work", "work_fast", "calib", "off_again")
    assert tool.JUDGED == ("work", "work_fast")
    assert not tool.WORK_OBSERVE.get("right") and tool.WORK_OBSERVE["depth"] is True
    assert streams.right_allowed(tool.CALIB_RECORD["kind"]) and "right" in tool.CALIB_RECORD["streams"]
    assert streams.record_refusal(tool.CALIB_RECORD["kind"], tool.CALIB_RECORD["streams"]) is None
