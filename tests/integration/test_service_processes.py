"""r1d + the `r1` CLI with REAL processes (DESIGN §12.4 "DDS bridge"): the DDS-bridge simulator (real time, the
emulated built-in standing the robot), r1rt + r1guard as processes (rt/apps), r1d as a process attached to r1rt's
segment (`python -m r1mc3.service`), and the CLI as a process (`python -m r1mc3.cli`). The operator's sequence:

    r1 status            -> MONITOR, the guard's heartbeat fresh, nothing of ours published
    r1 takeover          -> the core's gates (mailbox) -> ReleaseMode over the motion-switcher RPC (the bridge's
                            built-in lets go) -> RELEASED -> BLEND -> BALANCE; the simulated robot stays up
    r1 stop              -> accepted (arms HOLD, balance continues)
    r1 hand-state left   -> the hand's state over DDS (r1hands in r1d)

Wall-clock on a loaded Mac without real-time scheduling: generous timeouts; the asserted facts are the sequence and
the end states, not the timing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from tests.integration.test_rt_processes import APPS, LOOPBACK_URI, Proc, _unlink  # noqa: E402

DOMAIN = 64


def _cli(port: int, *words: str) -> dict[str, Any]:
    env = dict(os.environ, CYCLONEDDS_URI=LOOPBACK_URI)
    out = subprocess.run(
        [sys.executable, "-m", "r1mc3.cli", "--port", str(port), *words],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert out.stdout.strip(), out.stderr[-800:]
    res: dict[str, Any] = json.loads(out.stdout.strip().splitlines()[-1])
    res["_exit"] = out.returncode
    return res


def _wait_mode(port: int, mode: str, timeout: float) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = _cli(port, "status")
        if (st.get("data", {}).get("core") or {}).get("mode") == mode:
            return st
        time.sleep(0.3)
    raise AssertionError(f"mode {mode} not reached in {timeout} s: {st}")


def test_operator_sequence_with_real_processes(tmp_path: Path) -> None:
    from r1mc3 import rtconfig
    from r1sim.dds_bridge import DdsBridge

    shm = f"/r1mc3.svc.{os.getpid()}"
    cfg = rtconfig.write(tmp_path / "r1rt.cfg", rtconfig.core_dict())
    procs: list[Proc] = []
    with DdsBridge(domain=DOMAIN, scene="none", builtin="stand") as br:
        try:
            rt = Proc([str(APPS / "r1rt"), "--config", str(cfg), "--shm-name", shm, "--domain", str(DOMAIN)])
            procs.append(rt)
            rt.wait_for(lambda d: d.get("ready") is True, 20)
            gd = Proc([str(APPS / "r1guard"), "--config", str(cfg), "--shm-name", shm, "--domain", str(DOMAIN)])
            procs.append(gd)
            gd.wait_for(lambda d: d.get("ready") is True, 20)
            r1d = Proc(
                [sys.executable, "-m", "r1mc3.service", "--shm-name", shm, "--domain", str(DOMAIN), "--port", "0"]
            )
            procs.append(r1d)
            ready = r1d.wait_for(lambda d: d.get("ready") is True, 30)
            port = int(ready["port"])

            st = _wait_mode(port, "monitor", 10)
            assert st["ok"] and st["data"]["guard"] is not None and st["data"]["guard"]["age_ms"] < 500
            assert st["data"]["token"]["owner"] == "none"
            assert not [w for w in br.writers() if w != br.builtin_handle]  # nothing of ours on rt/lowcmd

            tk = _cli(port, "takeover")
            assert tk["ok"] and tk["moved"] and tk["_exit"] == 0, tk
            assert [e["stage"] for e in tk["data"]["log"]] == ["gates", "release", "decide"]
            assert not br.builtin_active  # the emulated built-in let go on ReleaseMode
            st = _wait_mode(port, "balance", 10)
            assert st["data"]["token"]["owner"] == "r1rt"
            time.sleep(2.0)
            assert br.tilt_deg() < 5.0  # balancing under our policy (the fixture's centred stand leans ~2-3 deg)

            again = _cli(port, "takeover")
            assert again["code"] == "WRONG_MODE" and again["_exit"] == 1

            stop = _cli(port, "stop")
            assert stop["ok"]
            assert ready.get("missing") == []  # the binding has every client-side function r1d uses
            hold = _cli(port, "hold", "arm_left")  # HOLD where it is (r1d reads the core's targets: SegmentView.mirror)
            assert hold["ok"] and len(hold["data"]["q"]) == 5, hold
            q = hold["data"]["q"]
            q[1] += 0.05  # shoulder roll out by 0.05 rad, one L0 setpoint
            joint = _cli(port, "joint", "arm_left", ",".join(f"{v:.4f}" for v in q), "--ttl", "0.3")
            assert joint["ok"] and joint["moved"], joint

            hs = _cli(port, "hand-state", "left")
            assert hs["ok"] and hs["data"]["left"]["valid"], hs
            assert len(hs["data"]["left"]["pressure"]) == 9
            assert _wait_mode(port, "balance", 2)["ok"]  # still balancing after the stop
            assert br.tilt_deg() < 5.0
        finally:
            for p in reversed(procs):
                p.stop()
            _unlink(shm)
