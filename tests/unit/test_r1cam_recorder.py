"""r1cam recordings: an MCAP file with one channel per stream (log_time = our receive clock, depth sequence = the
header's counter) + a manifest with counts, drops and sha256; a slow disk drops frames, it never blocks the camera."""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
import time
from pathlib import Path

from mcap.reader import make_reader
from r1mc3.cam.fake import FakeRobot
from r1mc3.cam.frames import Frame, monotonic_ns
from r1mc3.cam.recorder import CamRecorder
from r1mc3.cam.robot import RobotSource
from r1mc3.cam.source import SourceBase


def test_a_recording_writes_every_stream_with_its_times_and_a_manifest(tmp_path: Path) -> None:
    fake = FakeRobot(depth_hz=20.0, image_hz=15.0, image_bytes=10_000, repeat_every=0).start()
    src = RobotSource(depth_addr=fake.depth_addr, image_command=fake.image_command, idle_s=0.3)
    try:
        rec = CamRecorder(src, tmp_path, kind="charuco", label="t1", streams=("left", "right", "depth"),
                          depth_every=2, params={"board": "poster_6x4_90"}).start()  # fmt: skip
        time.sleep(1.5)
        doc = rec.stop()
    finally:
        src.close()
        fake.close()
    assert doc["status"] == "done" and doc["frames"]["left"] >= 10 and doc["frames"]["right"] >= 10
    assert 5 <= doc["frames"]["depth"] and doc["skipped_depth"] >= doc["frames"]["depth"] - 1  # every 2nd kept
    assert doc["dropped_queue"] == 0 and doc["write_errors"] == 0
    mcap = Path(doc["files"][0]["path"])
    assert doc["files"][0]["sha256"] == hashlib.sha256(mcap.read_bytes()).hexdigest()
    assert json.loads(Path(rec.manifest_path).read_text())["schema"] == "r1mc3.r1cam_manifest/1"
    by_topic: dict[str, list[tuple[int, int, int]]] = {}
    with mcap.open("rb") as f:
        for _schema, ch, msg in make_reader(f).iter_messages():
            by_topic.setdefault(ch.topic, []).append((msg.log_time, msg.sequence, len(msg.data)))
    assert {"/r1cam/left", "/r1cam/right", "/r1cam/depth", "/r1cam/depth_meta", "/r1cam/meta"} <= set(by_topic)
    assert len(by_topic["/r1cam/left"]) == doc["frames"]["left"]
    depth = by_topic["/r1cam/depth"]
    assert all(n == 544 * 448 * 2 for _, _, n in depth)
    seqs = [s for _, s, _ in depth]
    assert all(b - a == 2 for a, b in itertools.pairwise(seqs))  # the header counters, every 2nd frame
    times = [t for t, _, _ in by_topic["/r1cam/left"]]
    assert times == sorted(times) and doc["t_start_mono_ns"] <= times[0] <= doc["t_end_mono_ns"]


class _Burst(SourceBase):
    name = "burst"


def test_a_slow_writer_drops_the_oldest_frames_and_never_blocks_the_source(tmp_path: Path) -> None:
    src = _Burst(streams=("left",))
    rec = CamRecorder(src, tmp_path, kind="t", streams=("left",), queue_len=8).start()
    gate = threading.Event()
    orig = rec._write

    def slow(f: Frame) -> None:
        gate.wait(5.0)
        orig(f)

    rec._write = slow  # type: ignore[method-assign]
    t0 = time.monotonic()
    for i in range(1, 201):  # 200 frames while the writer is stuck
        src.publish(Frame("left", b"\xff\xd8x\xff\xd9", monotonic_ns(), i))
    publish_s = time.monotonic() - t0
    gate.set()
    doc = rec.stop()
    assert publish_s < 0.5  # the camera side never waited for the disk
    assert doc["dropped_queue"] >= 190 and doc["frames"]["left"] + doc["dropped_queue"] == 200
