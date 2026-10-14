"""Fault injection into REAL processes (DESIGN.md §12.6, session 5): a process frozen for a while (SIGSTOP ... SIGCONT -
the 9 Oct stall stand-in for a whole process: every thread of it stops, as a process blocked in the kernel would look
from outside) and a process killed outright (SIGKILL = kill -9: no handler, no clean-up, no last words).

Each fault runs in its own timer thread, scheduled at an absolute `time.monotonic()` instant, and records when the
signals were ACTUALLY sent (the last ~2 ms before an instant are spun, not slept, so the error is ~0.1 ms on this Mac
when idle; under load it is recorded, not assumed). POSIX only (macOS / Linux).
"""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcFault:
    """One scheduled fault and what really happened (monotonic seconds)."""

    kind: str  # "sigstop" | "kill9"
    pid: int
    at: float  # the planned instant (time.monotonic())
    duration_s: float = 0.0  # sigstop: how long the process stays stopped
    sent: float | None = None  # when SIGSTOP / SIGKILL was sent
    resumed: float | None = None  # sigstop: when SIGCONT was sent
    error: str | None = None
    _thread: threading.Thread | None = field(default=None, repr=False)

    def wait(self, timeout: float | None = None) -> ProcFault:
        if self._thread is not None:
            self._thread.join(timeout)
        return self

    @property
    def done(self) -> bool:
        return self._thread is None or not self._thread.is_alive()

    def as_dict(self) -> dict[str, Any]:
        stopped = None if self.sent is None or self.resumed is None else self.resumed - self.sent
        return dict(kind=self.kind, pid=self.pid, planned=self.at, sent=self.sent, resumed=self.resumed,
                    late_s=None if self.sent is None else self.sent - self.at, stopped_s=stopped,
                    error=self.error)  # fmt: skip


def _until(t: float) -> float:
    """Sleep until monotonic instant t (spin the last 2 ms); returns the instant reached."""
    while True:
        now = time.monotonic()
        rem = t - now
        if rem <= 0.0:
            return now
        if rem > 0.002:
            time.sleep(rem - 0.002)


def _signal(pid: int, sig: int) -> str | None:
    try:
        os.kill(pid, sig)
    except OSError as e:  # gone already, not ours
        return f"{signal.Signals(sig).name}: {e}"
    return None


def sigstop_for(pid: int, at: float, duration_s: float) -> ProcFault:
    """Freeze process `pid` (SIGSTOP) at monotonic instant `at` for `duration_s`, then SIGCONT. Returns at once."""
    f = ProcFault("sigstop", int(pid), float(at), float(duration_s))

    def run() -> None:
        _until(f.at)
        f.sent = time.monotonic()
        f.error = _signal(f.pid, signal.SIGSTOP)
        if f.error is not None:
            return
        _until(f.sent + f.duration_s)
        f.resumed = time.monotonic()
        f.error = _signal(f.pid, signal.SIGCONT)

    f._thread = threading.Thread(target=run, name=f"sigstop-{pid}", daemon=True)
    f._thread.start()
    return f


def kill9_at(pid: int, at: float) -> ProcFault:
    """kill -9 process `pid` at monotonic instant `at`. Returns at once."""
    f = ProcFault("kill9", int(pid), float(at))

    def run() -> None:
        _until(f.at)
        f.sent = time.monotonic()
        f.error = _signal(f.pid, signal.SIGKILL)

    f._thread = threading.Thread(target=run, name=f"kill9-{pid}", daemon=True)
    f._thread.start()
    return f


def in_s(delay_s: float) -> float:
    """The monotonic instant `delay_s` from now (for the `at` arguments)."""
    return time.monotonic() + float(delay_s)
