"""r1d on the ROBOT path (session 10, core): the DDS-bridge simulator with a plain table (real time), r1rt + r1guard
(rt/apps), r1d as a process on r1rt's segment and the `r1` CLI. Before session 10 r1rt's segment published no per-group
executor events and no base-command writer existed in the binding, so on real processes r1d

* judged a goal's end from the mirror's commanded target and called every guard stop "STOPPED" (no guard name), and
* answered `body.pose` UNSUPPORTED ("SegmentView.set_base_cmd is not in this build").

Layout 4 publishes the executor's status in the segment (`SegmentView.exec_status()`, read by `ShmPort.upper_groups()`)
and the binding writes the base command (`SegmentView.set_base_cmd`):

    r1 lean 3                    -> the policy's lean (the core's ramped lean in the mirror) reaches 3 deg; lean 0 back
    r1 move-joint left <into the table edge> -> the forearm meets the table, the executor's lag guard stops it: the goal
                                    ends aborted with stopped_by GUARD_LAG (a named executor guard), not STOPPED
    SIGSTOP r1rt during a goal   -> r1guard takes the motors (it holds, never limp): the goal ends aborted with
                                    stopped_by GUARD_TAKEOVER (the base: STOPPED from the frozen mirror)

On the base (a041731): the lean answered UNSUPPORTED; the blocked goal ended {"code": "STOPPED"}; the takeover goal
ended {"code": "STOPPED"} (numbers in the session-10 core notes).
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from tests.integration.test_rt_processes import APPS, Proc, _unlink  # noqa: E402
from tests.integration.test_service_processes import _cli, _wait_mode  # noqa: E402

DOMAIN = 41
# the left arm's table_ready posture (r1mc3.skills.postures): from the resting arms at the table the forearm swings
# up and forward through the table's near edge (session 9, compat: the lag guard stopped skill.table_ready there)
INTO_THE_TABLE = [0.0689, 0.685, 0.0192, -0.5269, -1.5534]


def _goal_end(port: int, gid: int, timeout: float) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = _cli(port, "goal", "status", str(gid))
        if st["data"].get("state") not in ("accepted", "active"):
            return st
        time.sleep(0.1)
    raise AssertionError(f"goal {gid} still running after {timeout} s: {st}")


def _lean_used(port: int) -> float | None:
    body = _cli(port, "status")["data"].get("body") or {}
    lean = body.get("lean_deg") or [None, None]
    return None if lean[1] is None else float(lean[1])


def test_the_robot_path_names_the_guard_and_writes_the_base_command(tmp_path: Path) -> None:
    from r1mc3 import rtconfig
    from r1sim.dds_bridge import DdsBridge

    shm = f"/r1mc3.core.{os.getpid()}"
    cfg = rtconfig.write(tmp_path / "r1rt.cfg", rtconfig.core_dict())
    procs: list[Proc] = []
    with DdsBridge(domain=DOMAIN, scene="table", builtin="stand") as br:
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
            port = int(r1d.wait_for(lambda d: d.get("ready") is True, 30)["port"])
            _wait_mode(port, "monitor", 10)
            assert _cli(port, "takeover")["ok"]
            _wait_mode(port, "balance", 10)
            time.sleep(1.0)

            # r1rt's 1 Hz status line carries the state-age counters (session 10)
            line = rt.wait_for(lambda d: d.get("status") == "r1rt" and "state_age" in d, 5)
            assert line["state_age"] is not None and "old_episodes" in line["state_age"], line
            assert line["state_age"]["stale_rung"] == 0  # D34: the rung OFF in the robot profile

            # (1) body.pose through the segment's base command
            lean = _cli(port, "lean", "3")
            assert lean["ok"] and lean["moved"], lean  # base: UNSUPPORTED
            t_end = time.monotonic() + 6.0
            used = _lean_used(port)
            while time.monotonic() < t_end and (used is None or abs(used - 3.0) > 0.3):
                time.sleep(0.3)
                used = _lean_used(port)
            assert used is not None and abs(used - 3.0) <= 0.3, used
            assert _cli(port, "lean", "0")["ok"]
            t_end = time.monotonic() + 6.0
            while time.monotonic() < t_end and abs(_lean_used(port) or 0.0) > 0.3:
                time.sleep(0.3)
            assert abs(_lean_used(port) or 0.0) <= 0.3
            assert br.tilt_deg() < 5.0

            # (2) a goal the table blocks: the executor's guard named, not "STOPPED"
            q0 = _cli(port, "hold", "arm_left")["data"]["q"]
            mv = _cli(port, "move-joint", "left", ",".join(f"{v:.4f}" for v in INTO_THE_TABLE), "--speed", "0.6")
            assert mv["data"]["state"] == "aborted", mv
            sb = mv["data"]["stopped_by"]
            assert sb["code"].startswith("GUARD_") and sb["code"] != "GUARD_TAKEOVER", sb  # base: STOPPED
            assert "executor event" in sb["detail"], sb
            assert mv["code"] == "GUARD_STOP"
            assert _wait_mode(port, "balance", 2)["ok"] and br.tilt_deg() < 5.0

            # (3) r1rt silent during a goal: r1guard takes over and the goal says so
            back = _cli(port, "move-joint", "left", ",".join(f"{v:.4f}" for v in q0), "--speed", "0.2", "--no-wait")
            assert back["ok"] and back["data"]["state"] == "accepted", back
            time.sleep(0.5)
            os.kill(rt.p.pid, signal.SIGSTOP)
            try:
                end = _goal_end(port, int(back["data"]["goal_id"]), 10.0)
                st = _cli(port, "status")["data"]
            finally:
                os.kill(rt.p.pid, signal.SIGCONT)
            assert end["data"]["state"] == "aborted", end
            assert end["data"]["result"]["data"]["stopped_by"]["code"] == "GUARD_TAKEOVER", end  # base: STOPPED
            assert st["guard"]["active"] is True, st["guard"]  # the guard holds (never limp)
            assert br.tilt_deg() < 5.0
        finally:
            for p in reversed(procs):
                p.stop()
            _unlink(shm)
