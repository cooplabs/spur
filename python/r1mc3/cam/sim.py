"""SimSource: r1cam's frames from the simulator (DESIGN §12.4: r1cam has a sim backend), at the real rates and sizes.

Generic: `render(stream)` returns the frame - JPEG bytes for "left" / "right", a uint16 mm H x W array (or its raw
bytes) for "depth". The simulator supplies it (r1sim.lens.HeadCamera -> lens.jpeg; r1sim.depth for the noise model;
the D13 render worker for the photo look); r1mc3 never imports the simulator. Each stream runs in its own thread at its
rate while wanted (the same on-demand rule as the robot); `latency_s` back-dates nothing - it delays the delivery, so
a frame's receive time is `latency_s` after its render (the camera's exposure -> receive latency, measured at HW2).
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .frames import HEIGHT, WIDTH, Frame, monotonic_ns
from .source import IDLE_S, SourceBase

RATES_HZ = {"left": 7.5, "right": 7.5, "depth": 10.0}  # the robot's (v2 native_cam.py)


class SimSource(SourceBase):
    name = "sim"

    def __init__(
        self,
        render: Callable[[str], Any],
        *,
        rates_hz: dict[str, float] | None = None,
        latency_s: float = 0.0,
        idle_s: float = IDLE_S,
        streams: Sequence[str] = ("left", "right", "depth"),
    ) -> None:
        super().__init__(idle_s=idle_s, streams=streams)
        self.render = render
        self.rates = {**RATES_HZ, **(rates_hz or {})}
        self.latency_s = float(latency_s)
        self.render_s: dict[str, float] = dict.fromkeys(self.streams, 0.0)  # the last render's duration
        self.threads = [threading.Thread(target=self._run, args=(s,), daemon=True, name=f"r1cam-sim-{s}")
                        for s in self.streams]  # fmt: skip
        for t in self.threads:
            t.start()

    def _run(self, stream: str) -> None:
        period = 1.0 / self.rates[stream]
        nxt = time.monotonic()
        counter = 0
        while not self.closed.is_set():
            if not self.wanted(stream):
                time.sleep(0.02)
                nxt = time.monotonic()
                continue
            t0 = time.monotonic()
            try:
                out = self.render(stream)
            except Exception as ex:  # a render failure is reported, the stream retries
                self.errors[stream] = f"{type(ex).__name__}: {ex}"
                time.sleep(0.2)
                continue
            self.render_s[stream] = time.monotonic() - t0
            if self.latency_s > 0:
                time.sleep(self.latency_s)
            counter += 1
            if stream == "depth":
                raw = out if isinstance(out, bytes) else np.asarray(out, dtype="<u2").reshape(HEIGHT, WIDTH).tobytes()
                f = Frame(stream, raw, monotonic_ns(), self.next_seq(stream), src_counter=counter,
                          src_stamp=time.time_ns() // 1000)  # fmt: skip
            else:
                f = Frame(stream, bytes(out), monotonic_ns(), self.next_seq(stream))
            self.publish(f)
            nxt += period
            delay = nxt - time.monotonic()
            if delay < 0:  # a slow render: the next frame now, never a backlog
                nxt = time.monotonic()
            else:
                time.sleep(delay)


class OnDemandSource(SourceBase):
    """Frames rendered at the moment they are asked for, in the caller's thread (session 10): a LOCKSTEP simulation's
    r1cam - each `observe` gets the frame of the current simulated instant, and no render thread reads the physics
    while it steps. `render(stream)` as SimSource's. The receive stamps are our CLOCK_MONOTONIC like every source's
    (r1d turns a frame's age into its capture time); depth repeats are not an issue (one render per request)."""

    name = "sim-on-demand"

    def __init__(
        self,
        render: Callable[[str], Any],
        *,
        streams: Sequence[str] = ("left", "right", "depth"),
        run_in_owner: Callable[[Callable[[], Any]], Any] | None = None,
    ) -> None:
        super().__init__(idle_s=0.0, streams=streams)
        self.render = render
        self.counter = 0
        # an OpenGL renderer belongs to the thread that made it (macOS: another thread hangs): `run_in_owner(fn)` runs
        # fn in the simulation's own thread and returns its result (OwnerThreadCalls); None = the caller's thread
        self.run_in_owner = run_in_owner

    def demand(self, streams: Any, keep_s: float | None = None) -> None:
        streams = list(streams)
        super().demand(streams, keep_s)
        for stream in streams:
            try:
                out = (
                    self.run_in_owner(functools.partial(self.render, stream))
                    if self.run_in_owner
                    else self.render(stream)
                )
            except Exception as ex:  # reported like SimSource's; the observe then times out with the stream's error
                self.errors[stream] = f"{type(ex).__name__}: {ex}"
                continue
            self.errors[stream] = None
            self.counter += 1
            if stream == "depth":
                raw = out if isinstance(out, bytes) else np.asarray(out, dtype="<u2").reshape(HEIGHT, WIDTH).tobytes()
                f = Frame(stream, raw, monotonic_ns(), self.next_seq(stream), src_counter=self.counter,
                          src_stamp=time.time_ns() // 1000)  # fmt: skip
            else:
                f = Frame(stream, bytes(out), monotonic_ns(), self.next_seq(stream))
            self.publish(f)


class OwnerThreadCalls:
    """Run callables in the thread that owns a resource (the simulation's thread, which made the OpenGL renderer):
    `call(fn)` from that thread runs fn at once; from any other thread it queues fn and waits until the owner's
    `service()` (called between simulation steps) has run it. A lockstep simulation thus renders between steps - never
    while the physics moves - whichever thread asked."""

    def __init__(self, timeout_s: float = 30.0) -> None:
        self.owner = threading.get_ident()
        self.timeout_s = float(timeout_s)
        self._q: list[tuple[Callable[[], Any], threading.Event, list[Any]]] = []
        self._lock = threading.Lock()

    def call(self, fn: Callable[[], Any]) -> Any:
        if threading.get_ident() == self.owner:
            return fn()
        done = threading.Event()
        box: list[Any] = []
        with self._lock:
            self._q.append((fn, done, box))
        if not done.wait(self.timeout_s):
            raise TimeoutError(f"the owner thread did not run the call within {self.timeout_s} s")
        if box and isinstance(box[0], BaseException):
            raise box[0]
        return box[0] if box else None

    def service(self) -> int:
        with self._lock:
            q, self._q = self._q, []
        for fn, done, box in q:
            try:
                box.append(fn())
            except Exception as ex:
                box.append(ex)
            done.set()
        return len(q)
