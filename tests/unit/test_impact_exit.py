"""tools/cam/impact.py's exit (session 13; HW2 day 2's C1 run printed nanobind "leaked" warnings at every exit): the
age sampler's stop Event was named `_stop` - threading.Thread's own method, which join() calls - so join() raised
TypeError, the sampler was never joined, and the daemon thread's SegmentView outlived the interpreter. Now the sampler
joins, lets go of the view, and the tool closes its segment view before it returns."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import importlib.util, os, sys, time
from pathlib import Path
from r1mc3.log._native import native
from r1mc3.service.port import ShmPort
spec = importlib.util.spec_from_file_location("impact", Path(sys.argv[1]))
impact = importlib.util.module_from_spec(spec); spec.loader.exec_module(impact)
m = native()
name = "/r1mc3.ops.%d" % os.getpid()
seg = m.SegmentView.create(name)
try:
    port = ShmPort.attach(name)
    ages = impact.AgeSampler(port.view)
    ages.start(); time.sleep(0.15)
    ages.stop()                      # base: TypeError ('Event' object is not callable) only if joined; never joined
    assert not ages.is_alive() and ages.view is None
    impact.close_port(port)
    assert port.view.closed
finally:
    seg.close()
print("clean")
"""


def _native_ok() -> bool:
    try:
        from r1mc3.log._native import native

        native()
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _native_ok(), reason="r1rt_py (the C++ binding) is not built")
def test_the_impact_tool_exits_without_nanobind_leaks(tmp_path: Path) -> None:
    p = subprocess.run([sys.executable, "-c", SCRIPT, str(REPO / "tools" / "cam" / "impact.py")], capture_output=True,
                       text=True, timeout=60, env={**os.environ})  # fmt: skip
    assert p.returncode == 0, p.stderr[-2000:]
    assert "clean" in p.stdout
    assert "nanobind: leaked" not in p.stderr, p.stderr[:500]
