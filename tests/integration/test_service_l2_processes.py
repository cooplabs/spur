"""r1d's L2 goals with REAL processes (session 9): the DDS-bridge simulator (real time), r1rt + r1guard (rt/apps), r1d
as a process on r1rt's segment (its default PlannerWorker + WorldModel) and the `r1` CLI - the robot path, where r1rt's
segment publishes no per-group executor events and r1d judges a goal's end from the mirror's commanded target.

    r1 takeover                       -> BALANCE (the bridge's built-in lets go)
    r1 move-joint left Q              -> planned by r1d's worker, TRAJECTORY records, succeeded (measured keys)
    r1 move-joint left Q0 --no-wait   -> accepted; r1 stop -> the goal aborted (OPERATOR_STOP), the arm HOLDs
    r1 goals                          -> nothing active
    r1 grasp left / r1 release left   -> the grip state machine on the bridge's simulated Dex3 over DDS: an empty wrap
                                         ends NOT_HOLDING; the release opens the hand

On the base (6449aee) `move-joint` was not a word and arm.move_joint answered UNKNOWN_COMMAND.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from tests.integration.test_rt_processes import APPS, Proc, _unlink  # noqa: E402
from tests.integration.test_service_processes import _cli, _wait_mode  # noqa: E402

DOMAIN = 81


def test_a_planned_joint_goal_on_the_robot_path(tmp_path: Path) -> None:
    from r1mc3 import rtconfig
    from r1sim.dds_bridge import DdsBridge

    shm = f"/r1mc3.api.{os.getpid()}"
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
            port = int(r1d.wait_for(lambda d: d.get("ready") is True, 30)["port"])
            _wait_mode(port, "monitor", 10)
            assert _cli(port, "takeover")["ok"]
            _wait_mode(port, "balance", 10)
            time.sleep(1.0)

            hold = _cli(port, "hold", "arm_left")
            q0 = hold["data"]["q"]
            q1 = [q0[0], q0[1] + 0.15, q0[2], q0[3] - 0.15, q0[4]]
            mv = _cli(port, "move-joint", "left", ",".join(f"{v:.4f}" for v in q1), "--speed", "0.5")
            assert mv["ok"] and mv["_exit"] == 0 and mv["data"]["state"] == "succeeded", mv
            assert mv["data"]["planned_on"] == "worker" and mv["data"]["stopped_by"] is None
            assert max(abs(a - b) for a, b in zip(mv["data"]["target_q"], q1, strict=True)) < 1e-3
            assert mv["data"]["error_mm"] < 40.0 and len(mv["data"]["fingertips_m"]) == 3, mv["data"]

            acc = _cli(port, "move-joint", "left", ",".join(f"{v:.4f}" for v in q0), "--speed", "0.2", "--no-wait")
            assert acc["ok"] and acc["data"]["state"] == "accepted", acc
            time.sleep(0.6)
            assert _cli(port, "stop")["ok"]
            st = _cli(port, "goal", "status", str(acc["data"]["goal_id"]))
            assert st["data"]["state"] == "aborted", st
            assert st["data"]["result"]["data"]["stopped_by"]["code"] == "OPERATOR_STOP"
            assert _cli(port, "goals")["data"]["active"] == []

            # the hands over DDS (r1hands in r1d, the bridge's simulated Dex3): an empty wrap must END with a verdict
            assert _cli(port, "resume")["ok"]
            gr = _cli(port, "grasp", "left", "--grip", "wrap")
            assert gr["data"]["state"] in ("succeeded", "aborted") and gr["data"]["verdict"]["state"], gr
            assert gr["data"]["verdict"]["state"] == "NOT_HOLDING", gr["data"]["verdict"]  # nothing in the hand
            rel = _cli(port, "release", "left")
            assert rel["ok"] and rel["data"]["state"] == "succeeded", rel
            assert _wait_mode(port, "balance", 2)["ok"] and br.tilt_deg() < 5.0
        finally:
            for p in reversed(procs):
                p.stop()
            _unlink(shm)
