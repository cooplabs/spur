"""Session 12 (the user's decision after HW2 day 1's C1, 12 Oct ~21:00: "fewer streams - the left image + depth while
working, all three only for calibration recordings"; r1mc3.cam.streams has the numbers): the right image streams only
during a calibration recording, the depth stream closes shortly after the observe that needed it (the depth = ~87 % of
the camera's traffic on the robot's link), and the depth socket bounds its TCP receive window. Every test here fails
on the base 1cf3afd (noted per test)."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any

import pytest
from r1mc3.cam import streams
from r1mc3.cam.fake import FakeRobot
from r1mc3.cam.robot import DepthReader, RobotSource
from r1mc3.cam.service import CamServer, CamService, request
from r1mc3.cam.source import IDLE_S


@pytest.fixture()
def stack(tmp_path: Path) -> Any:
    fake = FakeRobot(depth_hz=10.0, image_hz=15.0, image_bytes=8_000, repeat_every=0).start()
    src = RobotSource(depth_addr=fake.depth_addr, image_command=fake.image_command)  # the default idle (3 s)
    svc = CamService(src, tmp_path)
    srv = CamServer(svc, port=0).start()
    yield srv, src, fake
    srv.close()
    fake.close()


def test_an_observe_gets_no_right_image_outside_a_calibration_recording(stack: Any) -> None:
    """Base: the right image streamed and came back with every observe that asked for it."""
    srv, src, _fake = stack
    ob = request(srv.port, {"op": "observe", "right": True, "depth": False})
    assert ob["ok"], ob
    assert "right" not in ob["data"] and "calibration" in ob["data"]["right_refused"]
    assert src.slots["right"].puts == 0 and not src.wanted("right")  # never started
    st = request(srv.port, {"op": "status"})["data"]
    assert st["policy"]["work_streams"] == ["left", "depth"] and "calibration" in st["policy"]["right"]


def test_a_charuco_recording_streams_the_right_image_and_observes_may_use_it(stack: Any) -> None:
    srv, src, _fake = stack
    r = request(srv.port, {"op": "record_start", "kind": "charuco", "streams": ["left", "right", "depth"],
                           "depth_every": 5})  # fmt: skip
    assert r["ok"], r
    try:
        assert src.wanted("right")
        ob = request(srv.port, {"op": "observe", "right": True})
        assert ob["ok"] and "right" in ob["data"] and "right_refused" not in ob["data"]
    finally:
        assert request(srv.port, {"op": "record_stop", "rec_id": r["data"]["rec_id"]})["ok"]


def test_a_recording_with_the_right_image_needs_a_calibration_kind(stack: Any) -> None:
    """Base: any kind could record the right image, and a recording without `streams` took left + right."""
    srv, src, _fake = stack
    bad = request(srv.port, {"op": "record_start", "kind": "impact", "streams": ["left", "right", "depth"]})
    assert bad["code"] == "NOT_PERMITTED" and "calibration" in bad["details"]["reason"]
    r = request(srv.port, {"op": "record_start", "kind": "trial"})  # the default: the left image only
    assert r["ok"], r
    try:
        assert src.wanted("left") and not src.wanted("right")
    finally:
        man = request(srv.port, {"op": "record_stop", "rec_id": r["data"]["rec_id"]})["data"]
    assert list(man["frames"]) == ["left"]
    assert streams.record_refusal("flat_field", ["left", "right"]) is None  # the white-sheet recording keeps both


def test_the_depth_stream_closes_shortly_after_the_observe_that_needed_it(stack: Any) -> None:
    """The robot's depth sends ~10 frames / s (487 KB each) for as long as r1cam stays connected. Base: an observe kept
    the depth connected IDLE_S = 3 s after it -> ~30 frames (~14.6 MB on the robot's link) per observe; now the depth
    is held only until the observe's own frame, then closes DEPTH_KEEP_S (0.5 s) later."""
    srv, src, fake = stack
    assert IDLE_S >= 3.0 and streams.DEPTH_KEEP_S <= 0.5
    t0 = time.monotonic()
    ob = request(srv.port, {"op": "observe", "depth": True})
    assert ob["ok"] and "depth" in ob["data"], ob
    t_obs = time.monotonic() - t0
    time.sleep(streams.DEPTH_KEEP_S + 0.4)
    assert not src.wanted("depth")  # released; the left image keeps its 3 s (an H.264 restart needs a key frame)
    assert src.wanted("left")
    n_after = fake.frames_sent
    time.sleep(1.5)
    assert fake.frames_sent - n_after <= 1  # nothing more streams (at most one frame in flight at the close)
    # the whole observe cost: the frames until the observe's own + DEPTH_KEEP_S x 10 /s (+ one in flight)
    assert n_after <= int((t_obs + streams.DEPTH_KEEP_S + 0.4) * 10) + 2, (n_after, t_obs)
    assert n_after < 0.5 * IDLE_S * 10  # base: ~30


def test_the_depth_socket_bounds_its_receive_window() -> None:
    """The receive buffer is set BEFORE connect = the TCP window the robot may fill (what can queue ahead of the
    lowstate packets on the shared link). Base: no setting (the OS default, autotuned up to MBs on Linux)."""
    fake = FakeRobot(depth_hz=10.0, repeat_every=0).start()
    src = RobotSource(depth_addr=fake.depth_addr, image_command=fake.image_command, streams=("depth",))
    try:
        src.demand(["depth"], keep_s=2.0)
        assert src.slots["depth"].get(timeout_s=3.0) is not None
        rd = next(t for t in src.threads if isinstance(t, DepthReader))
        assert rd.rcvbuf == streams.DEPTH_RCVBUF_BYTES == 16384
        assert rd.rcvbuf_effective is not None and rd.rcvbuf_effective <= 2 * streams.DEPTH_RCVBUF_BYTES
        default = socket.socket().getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        assert default > rd.rcvbuf_effective or default <= 2 * streams.DEPTH_RCVBUF_BYTES
        assert src.stats()["depth_reader"]["connects"] == 1
    finally:
        src.close()
        fake.close()


def test_r1cam_takes_the_depth_window_and_keep_from_its_environment() -> None:
    """Day 2's C1 may compare settings without a code change (`systemctl edit r1cam`: Environment=...)."""
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root / "python"), "R1CAM_DEPTH_RCVBUF": "0", "R1CAM_DEPTH_KEEP_S": "1.5"}
    for extra, want in (({}, (0, 1.5)), ({"R1CAM_DEPTH_RCVBUF": "32768"}, (32768, 1.5))):
        p = subprocess.Popen([sys.executable, "-m", "r1mc3.cam", "--source", "fake", "--port", "0", "--records",
                              "/tmp"], stdout=subprocess.PIPE, text=True, env={**env, **extra})  # fmt: skip
        try:
            assert p.stdout is not None
            ready = json.loads(p.stdout.readline())
            assert (ready["depth_rcvbuf"], ready["depth_keep_s"]) == want
        finally:
            p.terminate()
            p.wait(10)
