"""`python -m r1mc3.log` - r1log, the logger process (DESIGN §2.1: Python, nice 10, drains the rings).

    python -m r1mc3.log run --out DIR [--segment /r1mc3.rt] [--period-ms 20] [--max-mb 512] [--max-min 10]
                            [--flight-s 30] [--post-s 5] [--tilt-deg 10] [--nice 10] [--wait-s 30] [--duration-s S]
    python -m r1mc3.log info FILE.mcap [...]

`run` attaches to r1rt's segment (waiting up to --wait-s for r1rt to create it), then polls until SIGINT / SIGTERM
(final drain, pending flight dump, files closed). SIGUSR1 = an operator flight-recorder dump. It prints its pid and
one status line per 10 s. It never writes into the segment except the rings' consumer index.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from ._native import native
from .logger import Logger, LoggerConfig, run_forever
from .reader import info


def _attach(name: str, wait_s: float) -> Any:
    mod = native()
    t_end = time.monotonic() + wait_s
    while True:
        try:
            return mod.SegmentView.attach(name)
        except RuntimeError as e:
            if time.monotonic() >= t_end:
                raise SystemExit(f"r1log: cannot attach to {name}: {e}") from None
            time.sleep(0.5)


def _run(a: argparse.Namespace) -> int:
    if a.nice:
        try:
            os.nice(a.nice)
        except OSError as e:
            print(f"r1log: nice {a.nice} refused ({e}); continuing", file=sys.stderr)
    view = _attach(a.segment, a.wait_s)
    cfg = LoggerConfig(
        out_dir=Path(a.out),
        max_bytes=int(a.max_mb * (1 << 20)),
        max_span_s=a.max_min * 60.0,
        flight=not a.no_flight,
        flight_window_s=a.flight_s,
        flight_post_s=a.post_s,
        tilt_trigger_deg=a.tilt_deg,
    )
    lg = Logger(view, cfg, source=f"attach {a.segment}")
    stopping = {"stop": False}

    def _stop(*_: Any) -> None:
        stopping["stop"] = True

    def _dump(*_: Any) -> None:
        lg.request_dump("operator")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _dump)
    print(f"r1log: pid {os.getpid()} attached to {a.segment}, writing {a.out} (SIGUSR1 = flight dump)", flush=True)
    last = [time.monotonic()]

    def _status() -> bool:
        if time.monotonic() - last[0] >= 10.0:
            last[0] = time.monotonic()
            print("r1log:", json.dumps(lg.stats()), flush=True)
        return stopping["stop"]

    out = run_forever(lg, period_s=a.period_ms * 1e-3, duration_s=a.duration_s, stop=_status)
    view.close()
    print("r1log: closed", json.dumps(out), flush=True)
    return 0


def _info(a: argparse.Namespace) -> int:
    for f in a.files:
        print(json.dumps(info(f), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m r1mc3.log", description=__doc__.split("\n\n")[0] if __doc__ else "")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="attach to r1rt's segment and log until SIGINT / SIGTERM")
    r.add_argument("--out", required=True, help="output directory")
    r.add_argument("--segment", default="/r1mc3.rt", help="POSIX shm name (R1RT_SHM_NAME)")
    r.add_argument("--period-ms", type=float, default=20.0, help="drain period (the rings hold ~8 s at 500 Hz)")
    r.add_argument("--max-mb", type=float, default=512.0, help="rotate after this many MB on disk")
    r.add_argument("--max-min", type=float, default=10.0, help="rotate after this many minutes of records")
    r.add_argument("--flight-s", type=float, default=30.0, help="flight recorder: seconds before a trigger")
    r.add_argument("--post-s", type=float, default=5.0, help="flight recorder: seconds after a trigger")
    r.add_argument("--tilt-deg", type=float, default=10.0, help="flight trigger: absolute IMU tilt")
    r.add_argument("--no-flight", action="store_true")
    r.add_argument("--nice", type=int, default=10)
    r.add_argument("--wait-s", type=float, default=30.0, help="wait this long for r1rt to create the segment")
    r.add_argument("--duration-s", type=float, default=None, help="stop after this long (tests)")
    r.set_defaults(fn=_run)
    i = sub.add_parser("info", help="summarise MCAP files")
    i.add_argument("files", nargs="+")
    i.set_defaults(fn=_info)
    a = p.parse_args(argv)
    rc: int = a.fn(a)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
