"""`r1sim.stack.ProcStack` (session 6 foundation): the bridge + r1rt + r1guard + r1d + r1log + the CLI, started and
stopped as one; the takeover into BALANCE through `r1 takeover`; r1guard stopped on standby and started again (the
board-hold test's sequence's process part) with the core's guard heartbeat fresh again afterwards. Wall-clock on a Mac
without real-time scheduling: the asserted facts are the sequence and the end states."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]

from r1sim import stack  # noqa: E402

if not stack.available():
    pytest.skip("needs the dds extra and the DDS build of r1rt / r1guard (scripts/ci.sh dds)", allow_module_level=True)

DOMAIN = 33


def test_stack_takeover_and_guard_restart(tmp_path: Path) -> None:
    with stack.ProcStack(
        domain=DOMAIN, workdir=tmp_path, bridge={"scene": "none", "builtin": "stand"}, r1log=True
    ) as st:
        s = st.wait_mode("monitor", 10)
        assert s["data"]["token"]["owner"] == "none"
        assert not [w for w in st.bridge.writers() if w != st.bridge.builtin_handle]  # nothing of ours on rt/lowcmd
        tk = st.cli("takeover")
        assert tk["ok"] and tk["_exit"] == 0, tk
        st.wait_mode("balance", 10)
        time.sleep(1.0)
        assert st.bridge.tilt_deg() < 5.0
        assert st.stop_guard() == 0  # on standby r1guard exits on SIGTERM
        st.start_guard()
        time.sleep(0.5)
        g = st.status()["data"]["guard"]
        assert g is not None and g["age_ms"] < 500, g
        assert st.mode() == "balance"
        assert set(st.pids()) == {"r1rt", "r1guard", "r1d", "r1log"}
