"""`r1sim.stack.Proc.kill()` takes the process's descendants with it (session 11: r1d's multiprocessing workers were
orphaned by ProcStack.stop()'s SIGKILL - ~29 left behind by one agent's rehearsal runs). Fails on 1e55bdd: the
grandchild survives its parent's kill."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from r1sim.stack import Proc

CHILD = """
import json, subprocess, sys, time
c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(json.dumps({"ready": True, "child": c.pid}), flush=True)
time.sleep(120)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers kill(0) until reaped by init: check its state
    st = os.popen(f"ps -o stat= -p {pid}").read().strip()
    return bool(st) and not st.startswith("Z")


def test_kill_takes_the_descendants(tmp_path: Path) -> None:
    p = Proc("parent", [sys.executable, "-c", CHILD], dict(os.environ), log_dir=tmp_path)
    child = int(p.wait_for(lambda d: d.get("ready") is True, 30.0)["child"])
    kids = os.popen(f"pgrep -P {p.pid}").read().split()
    assert str(child) in kids
    p.kill()
    end = time.monotonic() + 5.0
    while _alive(child) and time.monotonic() < end:
        time.sleep(0.05)
    assert not _alive(child), "the grandchild outlived Proc.kill()"
    assert not p.alive()
