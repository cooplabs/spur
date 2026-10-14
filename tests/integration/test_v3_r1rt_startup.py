"""Fix #2 (session 8, HW1 part 1 B1.1): r1rt leaves no shared-memory segment behind when its start fails after the
segment was created.

At HW1 the DDS participant could not be created (cyclonedds.xml asks for a >= 4 MB socket receive buffer, the robot's
net.core.rmem_max was 416 KB): r1rt exited with "DDS: ..." and left /dev/shm/r1mc3.rt - the next start refused
"segment exists" until it was removed by hand (NOTES.md fix list #2). app::die's std::exit does not run main()'s
destructors, so the segment's owner never unlinked it. Here the DDS start is made to fail on purpose (a CycloneDDS
configuration naming a network interface that does not exist), then r1rt is started again on the SAME segment name
with a working configuration: it must come up (base bad5a5c: "segment exists").
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1sim import stack  # noqa: E402

BAD_DDS = (
    "<CycloneDDS><Domain id='any'><General><Interfaces><NetworkInterface name='r1mc3nosuchif0' /></Interfaces>"
    "</General></Domain></CycloneDDS>"
)


def test_a_failed_dds_start_leaves_no_segment_behind(tmp_path: Path) -> None:
    if not stack.available():
        pytest.skip("needs the DDS build of r1rt / r1guard (scripts/ci.sh dds)")
    from r1mc3 import rtconfig

    apps = stack.apps_dir()
    assert apps is not None
    cfg = rtconfig.write(tmp_path / "r1rt.cfg", rtconfig.core_dict())
    shm = f"/r1mc3.core.{os.getpid()}.2"
    argv = [str(apps / "r1rt"), "--config", str(cfg), "--shm-name", shm, "--domain", "53"]
    try:
        bad = stack.Proc("r1rt-bad", argv, {**os.environ, "CYCLONEDDS_URI": BAD_DDS}, log_dir=tmp_path)
        bad.p.wait(30)
        assert bad.p.returncode != 0
        err = [d for d in bad.json_lines() if "error" in d]
        assert err and str(err[-1]["error"]).startswith("DDS"), bad.raw
        good = stack.Proc("r1rt", argv, {**os.environ, "CYCLONEDDS_URI": stack.loopback_uri()}, log_dir=tmp_path)
        try:
            ready = good.wait_for(lambda d: d.get("ready") is True or "error" in d, 30)
            assert ready.get("ready") is True, ready  # base: {"error": "segment ...: segment exists ..."}
        finally:
            good.terminate_and_wait(10)
    finally:
        stack.unlink_shm(shm)
