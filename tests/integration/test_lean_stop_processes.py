"""HW2 day 2's F12 sequence with REAL processes (session 13): the DDS-bridge sim + r1rt + r1guard + r1d (over the shared
segment, as on the robot) + the `r1` CLI, one word per process as the operator types them:

    r1 takeover -> BALANCE;  r1 stop (= the harness ending a run)  ->  r1 lean 3: STOPPED (r1 status: stop.latched)
    r1 resume  ->  r1 lean 3: the body leans ~2.5-3 deg forward (the sim's F12: 2.85)  ->  r1 lean 0

On the robot (results/hw2_20261013/floor/r1d_journal.jsonl) the stop at 02:47:08 UTC was never resumed and `r1 lean 3`
at 02:53:00 was ACCEPTED while the core used 0 - the latch was invisible to r1d (base dcc01a2: `lean` answers ok here
and the pitch stays). macOS without real-time scheduling: the bridge's lowstate gaps are reported next to a failure.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]

from r1sim import stack  # noqa: E402

if not stack.available():
    pytest.skip("needs the dds extra and the DDS build of r1rt / r1guard (scripts/ci.sh dds)", allow_module_level=True)

DOMAIN = 95
BRIDGE: dict[str, Any] = {"scene": "none", "builtin": "stand", "board_tick": "wall"}


def _pitch_deg(st: stack.ProcStack) -> float:
    w, x, y, z = (float(v) for v in st.bridge.sim.read_state().base_quat)
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))


def _health(st: stack.ProcStack) -> str:
    lg = st.bridge.log()
    gaps = f"{lg['lowstate_gap_p99_ms']} / {lg['lowstate_gap_max_ms']} ms"
    return f"bridge physics {lg['physics']}, lowstate gap p99 / max {gaps}"


def test_the_lean_after_a_stop_is_refused_and_works_after_resume(tmp_path: Path) -> None:
    with stack.ProcStack(domain=DOMAIN, workdir=tmp_path, bridge=BRIDGE) as st:
        st.wait_mode("monitor", 10)
        assert st.cli("takeover")["ok"]
        st.wait_mode("balance", 10)
        time.sleep(1.0)
        assert st.cli("stop")["ok"]
        time.sleep(0.3)
        stopped = st.cli("lean", "3")
        status = st.status()["data"]
        assert not stopped["ok"] and stopped["code"] == "STOPPED", (stopped, _health(st))
        assert "r1 resume" in stopped["details"]["reason"]
        assert status["stop"]["latched"] is True, status.get("stop")
        assert st.cli("resume")["ok"]
        time.sleep(0.5)
        assert st.status()["data"]["stop"]["latched"] is False
        p0 = _pitch_deg(st)
        lean = st.cli("lean", "3")
        assert lean["ok"], lean
        time.sleep(3.0)
        p1 = _pitch_deg(st)
        body = st.status()["data"]["body"]
        assert st.cli("lean", "0")["ok"]
        print(f"pitch {p0:+.2f} -> {p1:+.2f} deg (lean {p1 - p0:+.2f}); status lean_deg {body['lean_deg']}")
        assert 2.0 <= p1 - p0 <= 4.5, (p0, p1, _health(st))
        assert body["lean_deg"][0] == pytest.approx(3.0) and body["lean_deg"][1] == pytest.approx(3.0, abs=0.3)
        assert st.mode() == "balance"
