"""`r1 balance` with REAL processes (session 11; HW2's F5): the DDS-bridge sim (the S1 stand fixture as Unitree's
built-in, the board tick in real time as on the robot) + r1rt (the shipped profile: stale_rung 0) + r1guard + r1d + the
`r1` CLI, one word per process as the operator types them:

    r1 takeover -> BALANCE;  r1 balance -> WRONG_MODE ("already balancing")
    r1 freeze -> STAND_FREEZE;  r1 balance -> BALANCE;  r1 resume
    r1 freeze;  the bridge publishes rt/lowstate 40 ms late for 3.5 s  ->  r1 balance 2.5 s into it: REFUSED (the
        robot state older than the limit - the board-tick clock no longer re-bases during the freeze), still frozen;
        once the state is live again: r1 balance -> BALANCE

Never limp (every applied command keeps the legs stiff), one writer. Fails on 1e55bdd (`balance` is not a word).
macOS without real-time scheduling: the bridge's lowstate gaps are reported next to any failure.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]

from r1sim import stack  # noqa: E402

if not stack.available():
    pytest.skip("needs the dds extra and the DDS build of r1rt / r1guard (scripts/ci.sh dds)", allow_module_level=True)

DOMAIN = 72
BRIDGE: dict[str, Any] = {"scene": "none", "builtin": "stand", "board_tick": "wall"}


def _health(st: stack.ProcStack) -> str:
    lg = st.bridge.log()
    gaps = f"{lg['lowstate_gap_p99_ms']} / {lg['lowstate_gap_max_ms']} ms"
    return f"bridge physics {lg['physics']}, lowstate gap p99 / max {gaps}"


def _legs_stiff_since(st: stack.ProcStack, n0: int) -> bool:
    applied = list(st.bridge.applied)[n0:]
    return all(a.min_leg_kp > 1.0 for a in applied if a.crc_ok) if applied else True


def test_balance_from_the_operators_freeze_and_never_on_late_state(tmp_path: Path) -> None:
    with stack.ProcStack(domain=DOMAIN, workdir=tmp_path, bridge=BRIDGE) as st:
        st.wait_mode("monitor", 10)
        assert st.cli("takeover")["ok"]
        st.wait_mode("balance", 10)
        time.sleep(1.0)
        n0 = len(st.bridge.applied)
        busy = st.cli("balance")
        assert not busy["ok"] and busy["code"] == "WRONG_MODE", busy

        fz = st.cli("freeze")
        assert fz["ok"], fz
        st.wait_mode("stand_freeze", 2)
        time.sleep(1.0)  # the 0.4 s ramp + a hold
        bal = st.cli("balance")
        assert bal["ok"] and bal["data"]["mode"] == "balance", (bal, _health(st))
        assert st.cli("resume")["ok"]
        time.sleep(1.0)
        assert st.mode() == "balance" and st.bridge.tilt_deg() < 5.0, _health(st)

        assert st.cli("freeze")["ok"]
        st.wait_mode("stand_freeze", 2)
        time.sleep(0.6)
        st.bridge.inject("lowstate_delay", duration_s=3.5, delay_ms=40)
        time.sleep(2.5)  # past the 2 s after which session 10's clock re-based a steady lateness
        late = st.cli("balance")
        print("late:", late)
        assert not late["ok"] and late["code"] == "CONTROLLER_REJECTED", (late, _health(st))
        assert "older than the state-age limit" in late["details"]["reason"], late
        assert st.mode() == "stand_freeze"
        time.sleep(2.0)  # the delay is over; the state is live again
        live = st.cli("balance")
        assert live["ok"] and live["data"]["mode"] == "balance", (live, _health(st))
        assert st.cli("resume")["ok"]
        time.sleep(1.0)
        assert st.mode() == "balance" and st.bridge.tilt_deg() < 5.0, _health(st)
        assert _legs_stiff_since(st, n0), "a leg went limp"
        writers = {a.writer for a in list(st.bridge.applied)[n0:] if a.crc_ok}
        assert len(writers) == 1, writers  # one writer (r1rt) throughout
