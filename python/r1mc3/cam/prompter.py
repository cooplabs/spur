"""The operator's timed prompts for a camera recording (session 12; HW2 day 1's C3 used an ad-hoc prompter because the
live count read 0 frames - its JPEG decode needed Pillow, absent on the robot; fixed in coverage.py): the plan's own
timeline (`r1mc3.record.plans`: the same schedule `r1 record run <kind>` records against), printed - and spoken with
macOS `say` if asked - at its times, with r1cam's live count beside it when r1d answers.

    python -m r1mc3.cam.prompter charuco [distances_m=0.3,0.45,0.6] [--r1d-port 8780] [--say] [--wait-recording]

Start it, then type `r1 record run charuco --yes` in the other tab and press Enter here at the same moment - or give
`--wait-recording`: it starts when r1cam's status shows a recording of that kind (polled through r1d every 0.2 s).
Nothing moves; it only reads `camera.status`. Generic: any camera record kind (charuco, flat_field, depth_targets).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

POLL_S = 0.2


def schedule(kind: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    from ..record import plans

    cam = plans.plan(kind, params, None).camera
    if cam is None:
        raise ValueError(f"{kind!r} is not a camera recording")
    return list(cam["schedule"])


def parse_params(items: Sequence[str]) -> dict[str, Any]:
    """`key=value` words (the CLI's own form): numbers, comma lists of numbers, else strings."""
    out: dict[str, Any] = {}
    for it in items:
        k, _, v = it.partition("=")
        if "," in v:
            out[k] = [float(x) for x in v.split(",") if x]
            continue
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


def live_count(port: int | None) -> Callable[[], str]:
    """r1cam's live count through r1d ('' when r1d / r1cam / the count is not there)."""
    if port is None:
        return lambda: ""
    from ..service.client import Client, R1dError

    c = Client(port=port, timeout_s=2.0)

    def f() -> str:
        try:
            r = c.call("camera.status")
        except (R1dError, OSError):
            c.close()
            return ""
        recs = (r.get("data") or {}).get("recordings") or {}
        for rec in recs.values():
            cov = rec.get("coverage")
            if cov:
                return f"[live: {cov.get('say')} - {cov.get('cells_done')}/{cov.get('cells')} cells]"
        return ""

    return f


def recording_started(port: int, kind: str) -> Callable[[], bool]:
    from ..service.client import Client, R1dError

    c = Client(port=port, timeout_s=2.0)

    def f() -> bool:
        try:
            r = c.call("camera.status")
        except (R1dError, OSError):
            c.close()
            return False
        recs = (r.get("data") or {}).get("recordings") or {}
        return any(rec.get("kind") == kind for rec in recs.values())

    return f


def run(
    sched: Sequence[dict[str, Any]],
    *,
    out: Callable[[str], None] = print,
    speak: Callable[[str], None] | None = None,
    count: Callable[[], str] = lambda: "",
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Each prompt at its t_start (from now), a 'move' line at its t_end; returns the prompts shown."""
    t0 = clock()
    n = 0
    for k, e in enumerate(sched):
        while clock() - t0 < float(e["t_start"]):
            sleep(min(POLL_S, max(0.0, float(e["t_start"]) - (clock() - t0))))
        line = f"t {float(e['t_start']):6.1f} s  [{k + 1}/{len(sched)}]  {e['say']}"
        extra = count()
        out(f"{line}  {extra}".rstrip())
        if speak is not None:
            speak(str(e["say"]).split(":")[0])
        n += 1
        while clock() - t0 < float(e["t_end"]):
            sleep(min(POLL_S, max(0.0, float(e["t_end"]) - (clock() - t0))))
        nxt = sched[k + 1]["say"].split(":")[0] if k + 1 < len(sched) else "done: the recording ends by itself"
        out(f"t {float(e['t_end']):6.1f} s  move -> {nxt}")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", help="the camera record kind (charuco, flat_field, depth_targets)")
    ap.add_argument("params", nargs="*", help="key=value, as for `r1 record run`")
    ap.add_argument("--r1d-port", type=int, default=None, help="show r1cam's live count (camera.status via r1d)")
    ap.add_argument("--say", action="store_true", help="speak each prompt (macOS `say`)")
    ap.add_argument("--wait-recording", action="store_true", help="start when r1cam shows that recording (needs r1d)")
    a = ap.parse_args(argv)
    sched = schedule(a.kind, parse_params(a.params))
    print(f"{a.kind}: {len(sched)} prompts over {sched[-1]['t_end']:.0f} s")
    if a.wait_recording:
        if a.r1d_port is None:
            print("--wait-recording needs --r1d-port")
            return 2
        started = recording_started(a.r1d_port, a.kind)
        print("waiting for the recording to start (type the `r1 record run` word now) ...")
        while not started():
            time.sleep(POLL_S)
    else:
        input("press Enter at the moment you start the recording ")
    speak = None
    if a.say and shutil.which("say"):

        def speak(s: str) -> None:
            subprocess.Popen(["say", s])

    run(sched, speak=speak, count=live_count(a.r1d_port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
