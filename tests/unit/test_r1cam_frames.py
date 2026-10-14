"""r1cam's frame plumbing never blocks a producer: the newest-frame slot replaces, the recorder's queue drops the
oldest (counted)."""

from __future__ import annotations

import threading
import time

from r1mc3.cam.frames import DropQueue, Frame, NewestSlot, monotonic_ns


def _f(seq: int, stream: str = "left") -> Frame:
    return Frame(stream, b"x", monotonic_ns(), seq)


def test_the_slot_keeps_only_the_newest_and_counts_unread_overwrites() -> None:
    s = NewestSlot("left")
    for i in range(1, 6):
        s.put(_f(i))
    got = s.get()
    assert got is not None and got.seq == 5
    assert s.overwritten == 4 and s.puts == 5
    assert s.get(after_seq=5) is None  # nothing newer (no wait)


def test_the_slot_waits_for_a_newer_frame_and_honours_max_age() -> None:
    s = NewestSlot("depth")
    s.put(Frame("depth", b"x", monotonic_ns() - 2_000_000_000, 1))
    assert s.get(max_age_s=1.0) is None  # 2 s old
    threading.Timer(0.05, lambda: s.put(_f(2, "depth"))).start()
    t0 = time.monotonic()
    got = s.get(after_seq=1, max_age_s=1.0, timeout_s=2.0)
    assert got is not None and got.seq == 2 and time.monotonic() - t0 < 1.0


def test_the_drop_queue_never_blocks_and_drops_the_oldest() -> None:
    q = DropQueue(maxlen=3)
    t0 = time.monotonic()
    for i in range(1, 101):
        q.put(_f(i))
    assert time.monotonic() - t0 < 0.5
    assert q.dropped == 97 and len(q) == 3
    got = [q.get(0.0) for _ in range(3)]
    assert [f.seq for f in got if f is not None] == [98, 99, 100]
    assert q.get(0.0) is None
