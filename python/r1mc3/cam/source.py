"""What every frame source gives r1cam (RobotSource on the backpack, SimSource in the simulator - DESIGN §12.4 "r1cam
has a sim backend"): one `NewestSlot` per stream, on-demand streaming, listeners (the recorder) fed without blocking."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from .frames import STREAMS, Frame, NewestSlot

IDLE_S = 3.0  # a stream stops this long after its last request (v2 native_cam.py IDLE_S, after the 5 Oct overruns)

Listener = Callable[[Frame], None]


class SourceBase:
    """Slots, the demand clock per stream, listeners. Subclasses run the streams while `wanted(stream)`."""

    name = "base"

    def __init__(self, *, idle_s: float = IDLE_S, streams: Iterable[str] = STREAMS) -> None:
        self.idle_s = float(idle_s)
        self.streams = tuple(streams)
        self.slots = {s: NewestSlot(s) for s in self.streams}
        self._want_until = dict.fromkeys(self.streams, 0.0)
        self._hold: dict[str, int] = dict.fromkeys(self.streams, 0)  # recordings keep a stream up while they run
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()
        self._seq = dict.fromkeys(self.streams, 0)
        self.errors: dict[str, str | None] = dict.fromkeys(self.streams, None)
        self.closed = threading.Event()

    # ---- demand
    def demand(self, streams: Iterable[str], keep_s: float | None = None) -> None:
        """Start (or keep) these streams for the next keep_s (default idle_s) seconds (an on-demand source also
        produces a frame now)."""
        self.keep(streams, keep_s)

    def keep(self, streams: Iterable[str], keep_s: float | None = None) -> None:
        """Keep these streams up for the next keep_s seconds - the bookkeeping only (session 12: the depth's short
        keep after an observe must not make an on-demand source render another frame)."""
        t = time.monotonic() + (self.idle_s if keep_s is None else float(keep_s))
        with self._lock:
            for s in streams:
                if s not in self._want_until:
                    raise ValueError(f"unknown stream {s!r} (this source has {', '.join(self.streams)})")
                self._want_until[s] = max(self._want_until[s], t)

    def hold(self, streams: Iterable[str], on: bool) -> None:
        """A recording keeps its streams up until it ends (counted: two recordings may share one)."""
        with self._lock:
            for s in streams:
                self._hold[s] = max(0, self._hold[s] + (1 if on else -1))

    def stream_id(self, stream: str) -> str:
        """Which image stream a frame came from (session 12: a calibration file names the one it was made on)."""
        return f"{self.name}/{stream}"

    def wanted(self, stream: str) -> bool:
        with self._lock:
            return self._hold[stream] > 0 or time.monotonic() < self._want_until[stream]

    def active(self) -> dict[str, bool]:
        return {s: self.wanted(s) for s in self.streams}

    # ---- frames
    def add_listener(self, fn: Listener) -> None:
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        with self._lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def next_seq(self, stream: str) -> int:
        with self._lock:
            self._seq[stream] += 1
            return self._seq[stream]

    def publish(self, f: Frame) -> None:
        """A new frame: into its slot, then to every listener (each must not block: the recorder's DropQueue)."""
        self.slots[f.stream].put(f)
        with self._lock:
            ls = list(self._listeners)
        for fn in ls:
            fn(f)

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.name, "streams": {}}
        now = time.monotonic_ns()
        for s in self.streams:
            slot = self.slots[s]
            f = slot.peek()
            out["streams"][s] = {
                "active": self.wanted(s),
                "frames": slot.puts,
                "age_s": None if f is None else round((now - f.t_rx_ns) / 1e9, 3),
                "error": self.errors.get(s),
            }
        return out

    def close(self) -> None:
        self.closed.set()
