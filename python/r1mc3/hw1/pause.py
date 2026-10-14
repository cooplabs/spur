"""The board-hold and the guard-takeover tests of the stand session (DESIGN.md §14.1 items 6-7): r1rt paused by
SIGSTOP / SIGCONT from r1d.

r1d runs on the backpack next to r1rt (the systemd units, deploy/units), so it can stop the whole r1rt process for a
measured time - the same fault the 9 Oct stall was (a control process that stops), with no new code path inside r1rt:

* **the board-hold test** (`r1 board-hold --pause-ms N`): LowCmd stops for N ms (20 / 100 / 300 / 400 / 500: #34)
  with the guard OFF (stopped or observe-only: otherwise it takes over at ~10 ms and the test measures the guard, not
  the board). What the board does without commands is recorded by r1d's OWN rt/lowstate reader from 1 s before to 2 s
  after (r1rt's receive thread is stopped too, so r1log has nothing of that window). Robot on the stand, arms down;
  the explicit-OK step on the day.
* **the guard-takeover test** (`r1 guard-takeover --stop-s S`): r1rt stopped S s in ON_STAND with the guard ARMED:
  the guard must take the token (and hold the core's last command); after SIGCONT r1rt stays silent (NOT_OWNER) until
  the operator's `r1 reclaim`. If the guard has not taken the token within `TAKEOVER_DEADLINE_S`, r1d sends SIGCONT at
  once (r1rt still owns the token: it simply resumes) and reports FAULT - the robot is never left without a writer for
  S seconds by this tool.

SIGCONT is sent in a `finally` AND by a detached backstop process (`sh -c "sleep D; kill -CONT pid"`, its own session)
started before the SIGSTOP, so r1rt is continued even if r1d dies mid-test (SIGCONT to a running process is harmless).
If both fail, the robot shell's `kill -CONT $(pidof r1rt)` continues it (the runbook says so).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# DESIGN §14.1 item 6, session 8 (HW1 fix #34): the board holds the last LowCmd ~400 ms, then releases (the board-hold
# test, 11 Oct 2026: the user stopped at 500 ms) -> the escalation is 20 / 100 / 300 / 400 / 500 ms; 1000 / 2000 are
# REFUSED until the resume blend (#29) is proven on the stand and the user OKs them again
BOARD_HOLD_PAUSES_MS = (20, 100, 300, 400, 500)
TAKEOVER_STOP_MIN_S = 0.5
TAKEOVER_STOP_MAX_S = 8.0
TAKEOVER_DEADLINE_S = 0.2  # the guard takes over at ~10-12 ms (S5, real processes, macOS); 0.2 s = it did not
RECORD_PRE_S = 1.0
RECORD_POST_S = 2.0
BACKSTOP_MARGIN_S = 0.5  # the detached SIGCONT fires this long after the planned end


def monotonic_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def _sleep_until(t_ns: int) -> None:
    """Sleep to an absolute CLOCK_MONOTONIC time: sleeps of half the time left, the last 2 ms spun (yielding) - macOS
    stretches a long sleep by ~5 % (timer leeway; measured 100 ms -> 104.7), halving bounds that to ~0.1 ms."""
    while True:
        left = (t_ns - monotonic_ns()) * 1e-9
        if left <= 0:
            return
        time.sleep(left / 2 if left > 0.002 else 0)


@dataclass
class Paused:
    """What a pause did, in CLOCK_MONOTONIC ns (the clock of r1log's records and of the core's heartbeat)."""

    pid: int
    requested_s: float
    t_stop_ns: int = 0
    t_cont_ns: int = 0
    early: str | None = None  # why it was continued before the planned end (T6's net), else None
    backstop_pid: int | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def actual_ms(self) -> float:
        return (self.t_cont_ns - self.t_stop_ns) * 1e-6

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "requested_ms": round(self.requested_s * 1e3, 3),
            "actual_ms": round(self.actual_ms, 3),
            "t_stop_ns": self.t_stop_ns,
            "t_cont_ns": self.t_cont_ns,
            "continued_early": self.early,
            "backstop_pid": self.backstop_pid,
            "errors": self.errors,
        }


def _readline(proc: subprocess.Popen[str], timeout_s: float) -> str | None:
    """One line from a child's stdout within `timeout_s` (None on timeout / EOF)."""
    import select

    assert proc.stdout is not None
    r, _, _ = select.select([proc.stdout], [], [], timeout_s)
    if not r:
        return None
    line = proc.stdout.readline()
    return line or None


def sigstop_for(
    pid: int,
    dur_s: float,
    *,
    watch: Callable[[float], str | None] | None = None,
    watch_period_s: float = 0.001,
    in_process: bool = False,
) -> Paused:
    """SIGSTOP `pid` for `dur_s`, then SIGCONT (always: `finally` + the detached backstop).

    The stop / sleep / continue runs in a one-thread child (`r1mc3.hw1.pausechild`, started BEFORE the pause so its
    start-up is not in the timing): r1d's own threads cannot delay the SIGCONT through the GIL. `in_process=True`
    does it in this thread (the fallback if the child cannot start). `watch(elapsed_s)` (T6) is polled every
    `watch_period_s` while stopped; a non-None return continues the process AT ONCE from here (`early` = the reason;
    the child's later SIGCONT is harmless)."""
    out = Paused(pid=pid, requested_s=dur_s)
    os.kill(pid, 0)  # ProcessLookupError / PermissionError before anything else happens
    child: subprocess.Popen[str] | None = None
    if not in_process:
        try:
            child = subprocess.Popen(
                [sys.executable, "-m", "r1mc3.hw1.pausechild", str(int(pid)), f"{dur_s:.6f}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if _readline(child, 10.0) is None:
                raise OSError("the pause child did not start")
        except OSError as e:
            out.errors.append(f"pause child: {e!r} - paused in-process instead")
            if child is not None:
                child.kill()
            child = None
    backstop = subprocess.Popen(
        ["/bin/sh", "-c", f"sleep {dur_s + BACKSTOP_MARGIN_S:.3f}; kill -CONT {int(pid)} 2>/dev/null"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # survives r1d (another session: not killed with r1d's process group)
    )
    out.backstop_pid = backstop.pid
    threading.Thread(target=backstop.wait, name="hw1-backstop-reaper", daemon=True).start()
    try:
        out.t_stop_ns = monotonic_ns()
        if child is not None:
            assert child.stdin is not None
            child.stdin.write("go\n")
            child.stdin.flush()
        else:
            os.kill(pid, signal.SIGSTOP)
        end = out.t_stop_ns + round(dur_s * 1e9)
        if watch is None and child is None:
            _sleep_until(end)
        elif watch is not None:
            while (now := monotonic_ns()) < end:
                why = watch((now - out.t_stop_ns) * 1e-9)
                if why is not None:
                    out.early = why
                    os.kill(pid, signal.SIGCONT)
                    out.t_cont_ns = monotonic_ns()
                    if child is not None:  # it would only SIGCONT again at the planned end: not needed now
                        child.kill()
                    break
                _sleep_until(min(end, now + round(watch_period_s * 1e9)))
        if child is not None and out.early is None:
            line = _readline(child, dur_s + 5.0)
            if line:
                times = json.loads(line)
                out.t_stop_ns = int(times["t_stop_ns"])  # the child's own times: the pause r1rt got
                if out.early is None:
                    out.t_cont_ns = int(times["t_cont_ns"])
            else:
                out.errors.append("the pause child gave no times (its SIGCONT and the backstop still apply)")
    finally:
        try:
            os.kill(pid, signal.SIGCONT)  # always (harmless when it already runs)
        except OSError as e:  # gone (killed meanwhile): the backstop is harmless then
            out.errors.append(f"SIGCONT: {e!r}")
        if out.t_cont_ns == 0:
            out.t_cont_ns = monotonic_ns()
        if child is not None:
            with contextlib.suppress(Exception):
                child.wait(timeout=2.0)
            if child.poll() is None:
                child.kill()
    return out


class ChildLowstateRecorder:
    """`LowstateRecorder` in its own process (`r1mc3.hw1.lowstate_rec`): the same interface (`stop()`, `arrays()`,
    `close()`); `ready` = it had seen a sample before the timeout. The file it writes is removed by `arrays()`."""

    def __init__(self, domain: int, tmp_path: str | os.PathLike[str], *, ready_timeout_s: float = 5.0) -> None:
        from pathlib import Path

        self.path = Path(tmp_path)
        self.p = subprocess.Popen(
            [sys.executable, "-m", "r1mc3.hw1.lowstate_rec", "--domain", str(domain), "--out", str(self.path),
             "--ready-timeout-s", str(ready_timeout_s)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )  # fmt: skip
        line = _readline(self.p, ready_timeout_s + 10.0)
        info = json.loads(line) if line else {}
        self.ready = bool(info.get("ready")) and int(info.get("samples", 0)) > 0
        self.done: dict[str, Any] = {}

    def stop(self) -> None:
        if self.p.poll() is not None:
            return
        assert self.p.stdin is not None
        with contextlib.suppress(BrokenPipeError):
            self.p.stdin.write("stop\n")
            self.p.stdin.flush()
        line = _readline(self.p, 15.0)
        self.done = json.loads(line) if line else {}
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.p.wait(timeout=5.0)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.path.exists():
            return {"t_rx_ns": np.zeros(0, np.int64), "tick": np.zeros(0, np.uint32)}
        with np.load(self.path) as z:
            out = {k: z[k] for k in z.files}
        self.path.unlink()
        return out

    def close(self) -> None:
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait()


class LowstateRecorder:
    """r1d's own rt/lowstate reader (its own DDS participant): every sample with its receive time, saved as .npz.

    The board's tick (1 kHz) in each sample is the robot-side clock; the receive time is r1d's CLOCK_MONOTONIC (taken
    when a batch is read: a few ms coarse). All 35 hg slots are kept (the R1 uses 26: R1RT_HG_SLOT_MAP_INIT)."""

    def __init__(self, domain: int) -> None:
        from cyclonedds.core import Policy, Qos
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        from r1mc3.dds import hg

        self.dp = DomainParticipant(domain)
        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(256))
        self.reader = DataReader(self.dp, Topic(self.dp, "rt/lowstate", hg.LowState_), qos=qos)
        self.rows: list[tuple[Any, ...]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, name="r1d-lowstate-rec", daemon=True)

    def start(self) -> LowstateRecorder:
        self._t.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            batch: list[Any] = list(self.reader.take(N=256))
            now = monotonic_ns()
            for s in batch:
                if not hasattr(s, "motor_state"):
                    continue  # an invalid sample (instance state only)
                ms = s.motor_state
                self.rows.append(
                    (
                        now,
                        int(s.tick),
                        [m.q for m in ms],
                        [m.dq for m in ms],
                        [m.tau_est for m in ms],
                        [list(m.temperature) for m in ms],
                        [m.motorstate for m in ms],
                        list(s.imu_state.rpy),
                        list(s.imu_state.gyroscope),
                        list(s.imu_state.accelerometer),
                        int(s.crc),
                    )
                )
            self._stop.wait(0.002)

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2.0)

    def arrays(self) -> dict[str, np.ndarray]:
        rows = list(self.rows)
        if not rows:
            return {"t_rx_ns": np.zeros(0, np.int64), "tick": np.zeros(0, np.uint32)}
        cols = list(zip(*rows, strict=True))
        return {
            "t_rx_ns": np.asarray(cols[0], np.int64),
            "tick": np.asarray(cols[1], np.uint32),
            "q": np.asarray(cols[2], np.float32),
            "dq": np.asarray(cols[3], np.float32),
            "tau_est": np.asarray(cols[4], np.float32),
            "temperature": np.asarray(cols[5], np.int16),
            "motorstate": np.asarray(cols[6], np.uint32),
            "rpy": np.asarray(cols[7], np.float32),
            "gyroscope": np.asarray(cols[8], np.float32),
            "accelerometer": np.asarray(cols[9], np.float32),
            "crc": np.asarray(cols[10], np.uint32),
        }

    def close(self) -> None:
        self.stop()
        with contextlib.suppress(Exception):
            self.reader = None  # type: ignore[assignment]
            self.dp = None  # type: ignore[assignment]


def window_stats(a: dict[str, np.ndarray], t0_ns: int, t1_ns: int) -> dict[str, Any]:
    """The lowstate seen in [t0, t1] (receive times): samples, the largest receive gap, the board-tick span, and the
    largest joint motion relative to the first sample of the window (rad, the 35 slots)."""
    t = a["t_rx_ns"]
    sel = (t >= t0_ns) & (t <= t1_ns)
    n = int(sel.sum())
    out: dict[str, Any] = {"samples": n}
    if n == 0:
        return out
    tt = t[sel]
    ticks = a["tick"][sel].astype(np.int64)
    out["max_rx_gap_ms"] = round(float(np.diff(tt).max()) * 1e-6, 3) if n > 1 else None
    out["tick_span"] = int(ticks[-1] - ticks[0])
    if "q" in a:
        q = a["q"][sel]
        out["max_joint_motion_rad"] = round(float(np.abs(q - q[0]).max()), 5)
    return out
