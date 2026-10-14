"""Fault injection into real processes (r1sim.proc_faults): a child that prints a timestamp every 5 ms is frozen by
SIGSTOP for the planned time (its own clock shows the gap) and resumes; kill -9 ends it with SIGKILL."""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import numpy as np
import pytest
from r1sim import proc_faults

TICKER = "import time, sys\nwhile True:\n    print(time.monotonic(), flush=True)\n    time.sleep(0.005)\n"


def _child() -> subprocess.Popen[str]:
    p = subprocess.Popen([sys.executable, "-c", TICKER], stdout=subprocess.PIPE, text=True)
    assert p.stdout is not None
    p.stdout.readline()
    return p


@pytest.mark.realtime
def test_sigstop_freezes_the_whole_process_for_the_planned_time() -> None:
    p = _child()
    try:
        f = proc_faults.sigstop_for(p.pid, proc_faults.in_s(0.2), 0.5).wait(5.0)
        time.sleep(0.2)
        p.kill()
        out, _ = p.communicate(timeout=5)
    finally:
        if p.poll() is None:
            p.kill()
    ts = np.array([float(x) for x in out.split()])
    gaps = np.diff(ts)
    assert f.error is None and f.sent is not None and f.resumed is not None
    d = f.as_dict()
    assert 0.49 < d["stopped_s"] < 0.6, d
    assert abs(d["late_s"]) < 0.05, d
    assert 0.45 < gaps.max() < 0.7, gaps.max()  # the child's own clock: one gap of the stop's length
    assert int(np.sum(gaps > 0.1)) == 1


@pytest.mark.realtime
def test_kill9_is_sigkill() -> None:
    p = _child()
    f = proc_faults.kill9_at(p.pid, proc_faults.in_s(0.1)).wait(5.0)
    assert p.wait(timeout=5) == -signal.SIGKILL
    assert f.error is None and f.sent is not None


def test_a_gone_process_is_recorded_not_raised() -> None:
    p = _child()
    p.kill()
    p.wait(timeout=5)
    f = proc_faults.kill9_at(p.pid, proc_faults.in_s(0.0)).wait(5.0)
    # the pid may have been reused in theory; on this test's time scale it is gone
    assert f.error is not None and "SIGKILL" in f.error
