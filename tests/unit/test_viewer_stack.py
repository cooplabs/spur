"""r1sim.viewer's stack mode (`--stack`, session R): the window on the DDS-bridge simulator while the REAL r1rt,
r1guard and r1d processes run against it - the takeover through r1d, the core stopped (SIGSTOP) and r1guard taking the
motors, a jog through r1d's `stream.joint_jog`, an object dropped into the running physics, the headless frames.

Needs CycloneDDS's Python binding and the DDS build of rt/ (skipped otherwise); real time, so `slow`. Base 3f87b85: the
module r1sim.viewer.stack did not exist.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import pytest
from r1sim.viewer import stack as vstack

pytestmark = [pytest.mark.slow, pytest.mark.needs_dds]
if not vstack.available():
    pytest.skip(f"--stack cannot run here: {vstack.why_not()}", allow_module_level=True)

DOMAIN = 73  # the viewer's DDS range 70-79


def _wait(pred: Callable[[], bool], timeout_s: float) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_the_real_processes_in_the_window(tmp_path: Path) -> None:
    app = vstack.StackApp(vstack.StackConfig(scene="floor", domain=DOMAIN, workdir=str(tmp_path / "stack")))
    try:

        def core() -> Any:
            return (app.status().get("core") or {}).get("mode")

        def owner() -> Any:
            return (app.status().get("token") or {}).get("owner")

        assert _wait(lambda: core() == "balance", 30.0), app.status()
        assert owner() == "r1rt" and any(e.what == "takeover: accepted" for e in app.events)
        app.key("J")  # the right arm
        assert app.key("=").endswith(": ok")  # stream.joint_jog accepted by r1d
        assert app.key("O").startswith("dropped a box")
        assert app.key("S").startswith("r1rt stopped (SIGSTOP)")
        assert _wait(lambda: owner() == "r1guard", 5.0), app.status()
        t = app.t
        app.advance(t + 1.5)
        b = mujoco.mj_name2id(app.m, mujoco.mjtObj.mjOBJ_BODY, "viewer_drop_0_box")
        assert float(app.d.xpos[b][2]) == pytest.approx(0.025, abs=0.005)  # the box lies on the floor
        assert app.tilt_deg() < 5.0  # r1guard keeps the robot up
        ov = dict(app.overlay())
        assert ov["motors owned by"] == "r1guard" and ov["r1guard"].endswith("ACTIVE")
        from r1sim.viewer.__main__ import _parser, headless

        a = _parser().parse_args(["--headless", "--frames", "2", "--every", "0.2", "--start", str(app.t + 0.1),
                                  "--out", str(tmp_path / "frames"), "--width", "160", "--height", "120"])  # fmt: skip
        res = headless(a, app)
        assert len(res["frames"]) == 2 and all(f["std"] > 5.0 for f in res["frames"])
        assert json.loads((tmp_path / "frames" / "viewer.json").read_text())["controller"].startswith("r1rt")
    finally:
        app.close()
