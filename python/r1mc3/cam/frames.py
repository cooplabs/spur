"""Frames and the queues between r1cam's threads: nothing here ever blocks a producer.

A `Frame` carries our CLOCK_MONOTONIC receive time (`t_rx_ns`: the clock r1rt / r1log / r1d use on the backpack, so a
frame can be matched with the head joints in r1log) next to the source's own counter / timestamp (the depth header's;
the RTP 90 kHz stamp for images when known). `NewestSlot` keeps only the newest frame of a stream (an `observe` wants
the latest picture, never a backlog); `DropQueue` feeds a slow consumer (the recorder) and drops the OLDEST frame when
it is full, counting it.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

STREAMS = ("left", "right", "depth")
WIDTH, HEIGHT = 544, 448  # the head stereo module's frames (v2 native_cam.py W, H)


def monotonic_ns() -> int:
    """CLOCK_MONOTONIC on Linux (Python's time.monotonic_ns) - the backpack's control clock."""
    return time.monotonic_ns()


@dataclass(frozen=True)
class Frame:
    stream: str  # "left" | "right" (JPEG bytes) | "depth" (raw uint16 little-endian, mm, H x W)
    data: bytes
    t_rx_ns: int  # our receive time (CLOCK_MONOTONIC)
    seq: int  # our count for this stream (1, 2, ...)
    width: int = WIDTH
    height: int = HEIGHT
    src_counter: int | None = None  # the depth header's frame counter
    src_stamp: int | None = None  # the depth header's timestamp / an image's RTP stamp
    meta: dict[str, object] = field(default_factory=dict)


class NewestSlot:
    """One stream's newest frame. `put` never blocks (it replaces); `get` waits for a frame newer than `after_seq`."""

    def __init__(self, stream: str) -> None:
        self.stream = stream
        self._cv = threading.Condition()
        self._frame: Frame | None = None
        self.puts = 0
        self.overwritten = 0  # frames replaced before anyone read them (normal: readers want only the newest)
        self._read_seq = 0

    def put(self, f: Frame) -> None:
        with self._cv:
            if self._frame is not None and self._frame.seq > self._read_seq:
                self.overwritten += 1
            self._frame = f
            self.puts += 1
            self._cv.notify_all()

    def peek(self) -> Frame | None:
        with self._cv:
            return self._frame

    def get(self, *, after_seq: int = 0, max_age_s: float | None = None, timeout_s: float = 0.0) -> Frame | None:
        """The newest frame with seq > after_seq (and younger than max_age_s), waiting up to timeout_s; None if none."""
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while True:
                f = self._frame
                if f is not None and f.seq > after_seq:
                    age = (monotonic_ns() - f.t_rx_ns) / 1e9
                    if max_age_s is None or age <= max_age_s:
                        self._read_seq = max(self._read_seq, f.seq)
                        return f
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(min(left, 0.05))


class DropQueue:
    """A bounded FIFO for a slower consumer: `put` drops the oldest frame when full (counted), never blocks."""

    def __init__(self, maxlen: int = 64) -> None:
        self._q: deque[Frame] = deque()
        self.maxlen = int(maxlen)
        self._cv = threading.Condition()
        self.dropped = 0
        self.puts = 0

    def put(self, f: Frame) -> None:
        with self._cv:
            if len(self._q) >= self.maxlen:
                self._q.popleft()
                self.dropped += 1
            self._q.append(f)
            self.puts += 1
            self._cv.notify()

    def get(self, timeout_s: float = 0.1) -> Frame | None:
        with self._cv:
            if not self._q:
                self._cv.wait(timeout_s)
            return self._q.popleft() if self._q else None

    def __len__(self) -> int:
        with self._cv:
            return len(self._q)
