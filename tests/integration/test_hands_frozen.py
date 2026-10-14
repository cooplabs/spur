"""A FROZEN Dex3 state stream (DESIGN §12.6 "hand state freeze"; the bridge's `inject("hand_freeze", side=...)`: the
hand's state keeps arriving but repeats one sample bit for bit) must read as NOT valid - otherwise the pad / heat
recordings (HW1 B3.2 / B3.3) would store a frozen sample as data. On 8a3807e r1hands kept reporting it valid.

The threshold's margin, measured on the bridge (hanging, the sensor model's noise, 10 s at rest + 3 s closing):
0 identical consecutive samples in 1300 per hand (`repeats_max` 0); a freeze needs > 0.1 s of identical samples
(FROZEN_S, ~10 samples at 100 Hz) and was flagged 0.11 s after it began. Real-robot noise: checked at HW1 B3.1
(`r1 hand-state` shows repeats_max)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]

from r1sim import stack  # noqa: E402

if not stack.available():
    pytest.skip("needs the dds extra and the DDS build of r1rt / r1guard (scripts/ci.sh dds)", allow_module_level=True)


def _until(pred: object, timeout_s: float) -> float | None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if pred():  # type: ignore[operator]
            return time.monotonic() - t0
        time.sleep(0.005)
    return None


def test_frozen_hand_state_is_not_valid() -> None:
    os.environ["CYCLONEDDS_URI"] = stack.loopback_uri()
    from r1mc3.hands.dex3 import FROZEN_S, Dex3Hands
    from r1sim.dds_bridge import DdsBridge

    with DdsBridge(domain=66, scene="none", stance="hanging") as br:
        h = Dex3Hands(66).start()
        try:
            assert _until(lambda: h.state("left")["valid"] and h.state("right")["valid"], 10.0) is not None
            time.sleep(2.0)
            for s in ("left", "right"):  # the noise floor: real (sensor-model) samples never repeat
                st = h.state(s)
                assert st["valid"] and not st["frozen"] and st["repeats_max"] == 0 and st["states"] > 100, st
            br.inject("hand_freeze", side="left", duration_s=1.0)
            dt = _until(lambda: h.state("left")["frozen"], 1.0)
            assert dt is not None and FROZEN_S <= dt < 0.4, dt
            st = h.state("left")
            assert not st["valid"] and st["invalid_reason"] == "frozen" and st["age_ms"] < 100.0  # samples still arrive
            assert h.state("right")["valid"]  # the other hand untouched
            assert _until(lambda: h.state("left")["valid"], 2.0) is not None  # valid again after the window
        finally:
            h.close()


def test_frozen_hand_through_r1d(tmp_path: Path) -> None:
    """r1 hand-state / r1 status say FROZEN; a pad-press recording on that hand aborts with the reason."""
    with stack.ProcStack(domain=67, workdir=tmp_path, bridge={"scene": "none", "stance": "hanging"}) as st:
        st.wait_mode("monitor", 15)
        assert _until(lambda: st.cli("hand-state", "left")["data"]["left"]["valid"], 10.0) is not None
        rec = st.cli("record", "run", "pad_press", "side=left", "pad=0", "weights_g=[20,50]", "--yes")
        assert rec["ok"], rec
        time.sleep(0.5)
        st.bridge.inject("hand_freeze", side="left", duration_s=3.0)
        time.sleep(0.6)
        hs = st.cli("hand-state", "left")["data"]["left"]
        assert not hs["valid"] and hs["frozen"] and hs["invalid_reason"] == "frozen", hs
        s = st.status()["data"]
        assert s["hands"]["left"]["invalid_reason"] == "frozen"
        assert any("left state FROZEN" in w for w in s["summary"]), s["summary"]
        done = st.cli("record", "list")["data"]["done"]
        assert done and done[-1]["status"] == "aborted" and "frozen" in done[-1]["reason"], done
