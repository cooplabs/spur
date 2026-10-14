"""The planner's worker process ends when r1d (its parent) dies (session 11). Fails on 1e55bdd: a ProcessPoolExecutor
worker blocks on its call queue forever after its parent is SIGKILLed (~550 such orphans had piled up on the Mac; on
the robot r1d.service's KillMode=process would keep them across r1d crashes)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

PARENT = """
import json, os, sys, time
from r1mc3.plan.worker import PlannerWorker
w = PlannerWorker(warm=False)
pid = w.call(os.getpid).result(timeout=60)
print(json.dumps({"worker": pid}), flush=True)
time.sleep(120)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


def test_worker_exits_when_its_parent_is_killed() -> None:
    p = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout is not None
        worker = int(json.loads(p.stdout.readline())["worker"])
        assert worker != p.pid and _alive(worker)
        os.kill(p.pid, signal.SIGKILL)
        p.wait(timeout=10)
        end = time.monotonic() + 5.0
        while _alive(worker) and time.monotonic() < end:
            time.sleep(0.1)
        alive = _alive(worker)
        if alive:
            os.kill(worker, signal.SIGKILL)
        assert not alive, "the planner worker outlived its parent"
    finally:
        if p.poll() is None:
            p.kill()
