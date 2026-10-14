"""The stand session with REAL processes (session 6; DESIGN §3.5's start flag, the user's OK 10 Oct): r1rt
--stand-session on the HANGING DDS-bridge sim with Unitree in STAND LOCK (the case where a takeover's gates pass and
the balance policy would run in the air): the mailbox TAKEOVER and ENTER_BALANCE are refused with
R1RT_TAKEOVER_STAND_SESSION, r1rt stays in MONITOR publishing nothing, `r1 takeover` is refused, r1rt's heartbeat says
stand_session (SegmentView.rt_info()); ON_STAND is accepted as before. Fails on 934666c (no flag there: the takeover
was accepted). Wall-clock on a Mac without real-time scheduling: states asserted, not timings."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]

from r1sim import stack  # noqa: E402

if not stack.available():
    pytest.skip("needs the dds extra and the DDS build of r1rt / r1guard (scripts/ci.sh dds)", allow_module_level=True)

DOMAIN = 74


def test_a_stand_session_refuses_the_takeover_on_the_hanging_robot_and_allows_on_stand(tmp_path: Path) -> None:
    from r1sim.cpp_policy import import_r1rt_py

    mod = import_r1rt_py()
    bridge = {"scene": "none", "stance": "hanging", "builtin": "stand_lock"}
    with stack.ProcStack(domain=DOMAIN, workdir=tmp_path, bridge=bridge, rt_args=("--stand-session",)) as st:
        assert st.rt is not None
        ready = st.rt.wait_for(lambda d: d.get("ready") is True, 5)
        assert ready["stand_session"] is True
        st.wait_mode("monitor", 10)
        v = mod.SegmentView.attach(st.shm)
        try:
            info = v.rt_info()
            assert info is not None and info["stand_session"] is True
            code = mod.refusals()["stand_session"]
            for req in ("takeover", "enter_balance"):
                t_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                seq = v.request_mode(mod.requests()[req], mod.requesters()["r1d"], t_ns, timeout_s=1.0)
                rep = None
                end = time.monotonic() + 5.0
                while rep is None and time.monotonic() < end:
                    rep = v.mode_reply(seq)
                    time.sleep(0.005)
                assert rep is not None and rep[:2] == (-6, code), (req, rep)
                assert rep[2] == 0  # MONITOR
        finally:
            v.close()
        tk = st.cli("takeover")
        assert not tk["ok"] and tk["_exit"] == 1, tk
        time.sleep(0.3)
        assert st.mode() == "monitor"
        assert not [w for w in st.bridge.writers().values() if w["label"] in ("r1rt", "r1guard")]  # nothing of ours
        on = st.cli("on-stand")
        assert on["ok"], on
        st.wait_mode("on_stand", 10)
        assert [w for w in st.bridge.writers().values() if w["label"] == "r1rt"]
