"""`python -m r1mc3.hw1.pausechild PID DUR_S` - the SIGSTOP / SIGCONT of r1rt in a process of its own.

r1d is a multi-threaded Python process (the tick, the server, the hands, the recorder): a thread that sleeps for a
20 ms pause can wait milliseconds more for the GIL before it sends SIGCONT (measured: a 1.5 s T6 pause ended 228 ms
late under load). This child has one thread and nothing else to do, so the pause it times is the pause r1rt gets
(within the OS's scheduling). It prints {"ready": true} once started, waits for "go" on stdin, stops PID, sleeps to the
absolute end time, continues PID and prints {"t_stop_ns": ..., "t_cont_ns": ...} (CLOCK_MONOTONIC, the clock r1d and
r1log use). Any other line (or stdin closing) before "go" exits without touching PID.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

SPIN_S = 0.002


def _now() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    pid, dur_s = int(args[0]), float(args[1])
    print(json.dumps({"ready": True, "pid": os.getpid()}), flush=True)
    if sys.stdin.readline().strip() != "go":
        return 1
    t_stop = _now()
    os.kill(pid, signal.SIGSTOP)
    end = t_stop + round(dur_s * 1e9)
    try:
        while True:  # sleeps of half the time left, the last ms spun (macOS stretches a long sleep by ~5 %)
            left = (end - _now()) * 1e-9
            if left <= 0:
                break
            time.sleep(left / 2 if left > SPIN_S else 0)
    finally:
        os.kill(pid, signal.SIGCONT)
        t_cont = _now()
    print(json.dumps({"t_stop_ns": t_stop, "t_cont_ns": t_cont}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
