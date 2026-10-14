"""RobotSource against a fake robot (the depth TCP format of v2 native_cam.py; the image path through a stand-in for
gst-launch-1.0, which the Mac does not have): frames arrive with our receive times, nothing connects while idle, the
streams stop `idle_s` after the last request, repeated depth frames are skipped."""

from __future__ import annotations

import shlex
import time

import numpy as np
import pytest
from r1mc3.cam import robot
from r1mc3.cam.fake import FakeRobot, synthetic_depth

# The robot's 36-byte header for counter 1, timestamp 2, written out byte by byte from the documented layout
# (v2 native_cam.py: <IQQIIII = magic 0x59313620, counter, timestamp, 544, 448, magic, 544 * 448 * 2 bytes).
GOLDEN_HEADER = bytes.fromhex("203631590100000000000000020000000000000020020000c00100002036315900700700")


def test_the_depth_header_matches_the_robots_layout() -> None:
    raw = np.zeros((448, 544), dtype=np.uint16)
    frame = robot.pack_depth(1, 2, raw)
    assert frame[:36] == GOLDEN_HEADER and len(frame) == 36 + 544 * 448 * 2
    assert robot.parse_depth_header(GOLDEN_HEADER) == (1, 2)
    bad = GOLDEN_HEADER[:12] + bytes(8) + GOLDEN_HEADER[20:]
    assert robot.parse_depth_header(bad) == (1, 0)
    with pytest.raises(robot.DepthHeaderError):
        robot.parse_depth_header(GOLDEN_HEADER[:20] + (640).to_bytes(4, "little") + GOLDEN_HEADER[24:])


def test_depth_metres_drop_the_no_depth_codes() -> None:
    raw = np.full((448, 544), 1234, dtype=np.uint16)
    raw[0, 0], raw[0, 1] = 0, 65535
    d = robot.depth_m(raw.tobytes())
    assert d[0, 0] == 0.0 and d[0, 1] == 0.0 and abs(float(d[1, 1]) - 1.234) < 1e-6


def test_split_jpegs_finds_whole_frames_across_chunks() -> None:
    a, b = b"\xff\xd8AAA\xff\xd9", b"\xff\xd8BB\xff\xd9"
    jpegs, rest = robot.split_jpegs(b"junk" + a + b[:3])
    assert jpegs == [a] and rest == b[:3]
    jpegs, rest = robot.split_jpegs(rest + b[3:])
    assert jpegs == [b] and rest == b""


def test_the_pipelines_are_v2s_decode_chain_and_a_remux_without_reencoding() -> None:
    dec = robot.decode_pipeline(5002)
    assert "udpsrc port=5002" in dec and "avdec_h264" in dec and "jpegenc" in dec and "decodebin" not in dec
    rec = robot.record_pipeline(5003, "/tmp/x y.mkv")
    assert "matroskamux" in rec and "tee name=t" in rec and "'/tmp/x y.mkv'" in rec
    assert "x264enc" not in rec and "nvv4l2h264enc" not in rec and rec.count("avdec_h264") == 1  # no encoder
    argv = robot.RobotSource.gst_command("left", None)
    assert argv[:3] == ["gst-launch-1.0", "-q", "-e"] and shlex.join(argv[3:]).startswith("udpsrc port=5002")


def test_robot_source_streams_on_demand_from_a_fake_robot() -> None:
    fake = FakeRobot(depth_hz=20.0, image_hz=15.0, image_bytes=20_000, repeat_every=3).start()
    src = robot.RobotSource(depth_addr=fake.depth_addr, image_command=fake.image_command, idle_s=0.6)
    try:
        time.sleep(0.4)
        assert fake.connections == 0 and src.slots["left"].puts == 0  # idle: nothing connected, no child
        t0 = time.monotonic_ns()
        src.demand(["left", "depth"])
        d = src.slots["depth"].get(timeout_s=3.0)
        left = src.slots["left"].get(timeout_s=5.0)
        assert d is not None and left is not None and fake.connections == 1
        assert d.t_rx_ns >= t0 and left.t_rx_ns >= t0 and len(left.data) == 20_000
        assert left.data[:2] == b"\xff\xd8" and left.data[-2:] == b"\xff\xd9"
        assert np.array_equal(
            np.frombuffer(d.data, dtype="<u2").reshape(448, 544)[1:3, 1:3],
            synthetic_depth(int(d.src_counter or 0))[1:3, 1:3],
        )
        assert src.slots["right"].puts == 0  # not asked for
        time.sleep(1.5)  # idle_s 0.6 passed: both streams stopped
        n_d, n_l = src.slots["depth"].puts, src.slots["left"].puts
        time.sleep(0.5)
        assert src.slots["depth"].puts == n_d and src.slots["left"].puts == n_l
        assert not src.wanted("depth") and not src.wanted("left")
        counters = []  # the repeats the fake sends (every 3rd frame twice) never reach the slot twice
        src.demand(["depth"], keep_s=1.0)
        old = src.slots["depth"].peek()
        last = 0 if old is None else old.seq  # (a new connection: the fake's counter starts again at 1)
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            f = src.slots["depth"].get(after_seq=last, timeout_s=0.3)
            if f is not None:
                last = f.seq
                counters.append(f.src_counter)
        assert len(counters) >= 5 and len(set(counters)) == len(counters)
        reader = next(t for t in src.threads if isinstance(t, robot.DepthReader))
        assert reader.repeats >= 1
    finally:
        src.close()
        fake.close()
