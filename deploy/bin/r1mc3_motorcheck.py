#!/usr/bin/env python3
"""Read-only: may a service stop / restart / deploy switch happen now? (DESIGN.md §2.1, §4.4)

"Stopping a service never damps. `r1 shutdown` is accepted only from DAMP or MONITOR. While we own the motors (any
publishing mode), a systemd stop / restart, a reboot request or a deploy is refused." This tool answers that one
question from r1rt's shared-memory segment (`/r1mc3.rt`): the ownership token and the two heartbeats. It never
writes to the segment, never publishes anything, never damps.

Exit status: 0 = safe (no segment, nobody owns the token, or the owner r1rt is in MONITOR / DAMP with a fresh
heartbeat); 3 = REFUSED (r1rt or r1guard owns the motors in a publishing mode, or the owner's heartbeat is stale - a
stalled r1rt is exactly when the guard must keep running); 2 = cannot tell (the segment exists but cannot be read) -
callers treat 2 as refused. One JSON line on stdout says why.

Sources: the real segment through `r1rt_py.SegmentView.attach` (the same seqlock code r1log uses, header and layout
checked; it maps the segment read-only in effect: only `token()` / `rt_heartbeat()` / `guard_heartbeat()` are called).
With `--root DIR` other than "/" (tests, rehearsals) the state comes from `DIR/run/r1mc3/motorcheck_state.json`
instead, and the segment's presence from `DIR/dev/shm/<name>`; the fake source is never consulted on the real root.

Layouts: rt/include/r1rt/types.h (`r1rt_heartbeat`, `r1rt_mode`, `r1rt_owner`), shm_layout.h (the token word: owner in
bits 0-7). tests/unit/test_deploy_motorcheck.py checks the heartbeat layout against python/r1mc3/log/records.py.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.util
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

MODES = {
    0: "MONITOR",
    1: "TAKEOVER_MIRROR",
    2: "TAKEOVER_BLEND",
    3: "BALANCE",
    4: "GUARDED",
    5: "STAND_FREEZE",
    6: "ON_STAND",
    7: "OVERLAY_ARM_SDK",
    8: "DAMP",
}
SAFE_MODES = {"MONITOR", "DAMP"}  # DESIGN §2.1: shutdown only from DAMP or MONITOR
OWNERS = {0: "none", 1: "r1rt", 2: "r1guard"}
HEARTBEAT_FMT = "<QqQIIfHBB"  # seq, t_ns, lowstate_tick, rate_hz, faults, blend_progress, mode, rt_granted, _pad0
HEARTBEAT_SIZE = 40
STALE_NS = 100_000_000  # a heartbeat older than 100 ms is stale (the guard's own threshold is 10 ms)

EXIT_OK, EXIT_CANNOT_TELL, EXIT_REFUSED = 0, 2, 3


def decode_heartbeat(raw: bytes) -> dict[str, Any]:
    if len(raw) != HEARTBEAT_SIZE:
        raise ValueError(f"heartbeat is {len(raw)} bytes, expected {HEARTBEAT_SIZE}")
    seq, t_ns, tick, rate, faults, blend, mode, granted, _ = struct.unpack(HEARTBEAT_FMT, raw)
    return {
        "seq": seq,
        "t_ns": t_ns,
        "lowstate_tick": tick,
        "rate_hz": rate,
        "faults": faults,
        "blend_progress": blend,
        "mode": MODES.get(mode, f"UNKNOWN({mode})"),
        "rt_granted": granted,
    }


def decide(state: dict[str, Any]) -> tuple[int, str]:
    """state: {"segment": bool, "token_owner": "none"|"r1rt"|"r1guard"|..., "rt": hb or None, "guard": hb or None,
    "now_ns": int}. Returns (exit status, reason)."""
    if not state.get("segment", False):
        return EXIT_OK, "no r1rt segment: nothing of ours runs"
    if state.get("processes_alive") is False:  # unknown (None) is treated as alive
        return EXIT_OK, "a stale segment: no r1rt / r1guard process is running, so nothing of ours publishes"
    owner = state.get("token_owner")
    now = int(state["now_ns"])
    if owner == "none":
        return EXIT_OK, "the ownership token is free: nothing of ours publishes rt/lowcmd"
    if owner == "r1rt":
        hb = state.get("rt")
        if hb is None:
            return EXIT_REFUSED, "r1rt owns the token but its heartbeat cannot be read"
        age = now - int(hb["t_ns"])
        if age > STALE_NS:
            return (
                EXIT_REFUSED,
                f"r1rt owns the token but its heartbeat is {age / 1e6:.0f} ms old (stalled?): "
                "the guard must keep running",
            )
        if hb["mode"] in SAFE_MODES:
            return EXIT_OK, f"r1rt is in {hb['mode']}"
        return EXIT_REFUSED, f"r1rt owns the motors in {hb['mode']}"
    if owner == "r1guard":
        return (
            EXIT_REFUSED,
            "r1guard owns the motors (after a takeover): end the session with the operator's damp + power off",
        )
    return EXIT_REFUSED, f"unknown token owner {owner!r}"


def read_fake(root: Path, name: str) -> dict[str, Any]:
    shm = root / "dev" / "shm" / name.lstrip("/")
    st: dict[str, Any] = {"segment": shm.exists(), "now_ns": time.monotonic_ns()}
    f = root / "run" / "r1mc3" / "motorcheck_state.json"
    if st["segment"] and f.exists():
        d = json.loads(f.read_text())
        st["token_owner"] = d.get("token_owner", "none")
        st["processes_alive"] = d.get("processes_alive")
        for key in ("rt", "guard"):
            hb = d.get(key)
            if hb is not None:  # ages are given relative to now
                st[key] = {"mode": hb["mode"], "t_ns": st["now_ns"] - int(float(hb.get("age_ms", 0.0)) * 1e6)}
    elif st["segment"]:
        raise RuntimeError(f"fake segment {shm} without {f}")
    return st


def processes_alive() -> bool | None:
    """Whether an r1rt or r1guard process runs (pgrep -x); None when pgrep is unavailable (= assume alive)."""
    found = False
    for name in ("r1rt", "r1guard"):
        try:
            r = subprocess.run(["pgrep", "-x", name], capture_output=True, check=False)
        except OSError:
            return None
        found = found or r.returncode == 0
    return found


def owner_of(token: Any) -> int:
    """The token's owner from `r1rt_py.SegmentView.token()`: an (owner, generation) tuple (the binding since session
    4; r1d reads it so too), or - an older binding - the raw 64-bit word with the owner in bits 0-7. (Session 6: this
    tool did int() on the tuple, so on a LIVE segment it always answered "cannot tell" = refused.)"""
    if isinstance(token, tuple | list):
        return int(token[0])
    return int(token) & 0xFF


def state_from_view(view: Any, *, now_ns: int | None = None, alive: bool | None = None) -> dict[str, Any]:
    """The decision's input from an attached segment view: the token's owner and both heartbeats (read only).
    `now_ns` defaults to CLOCK_MONOTONIC (the processes' clock); a lockstep core's view passes its own clock."""
    owner = owner_of(view.token())
    st: dict[str, Any] = {
        "segment": True,
        "token_owner": OWNERS.get(owner, f"unknown({owner})"),
        "now_ns": time.monotonic_ns() if now_ns is None else int(now_ns),
        "processes_alive": alive,
    }
    for key, fn in (("rt", view.rt_heartbeat), ("guard", view.guard_heartbeat)):
        r = fn()
        st[key] = decode_heartbeat(bytes(r[1])) if r is not None else None
    return st


BINDING = "r1rt_py"


def binding_dirs(script: Path | None = None) -> list[Path]:
    """Where the release's own build of the binding is, in search order: <release>/build/rt next to this script's
    deploy/bin (install.sh builds it there; r1d / r1log find it the same way through r1mc3.log._native), then
    $R1MC3_BUILD_DIR/rt (a checkout's build tree)."""
    here = (script or Path(__file__)).resolve().parent
    out = [here.parents[1] / "build" / "rt"]
    env = os.environ.get("R1MC3_BUILD_DIR")
    if env:
        out.append(Path(env) / "rt")
    return out


def import_binding() -> ModuleType:
    """The segment reader `r1rt_py`. HW1 (fix #1): the units run this check as `python -I .../r1mc3_motorcheck.py`,
    and a bare `import r1rt_py` found nothing in the installed release (the module lives in <release>/build/rt, on no
    sys.path) -> "cannot tell" -> every stop refused and the inhibitor blocked the reboot; the session's fix was a .pth
    in the release venv. Now the check loads the release's build itself (ImportError if there is none)."""
    if BINDING in sys.modules:
        return sys.modules[BINDING]
    for d in binding_dirs():
        hits = sorted(d.glob(f"{BINDING}*.so")) + sorted(d.glob(f"{BINDING}*.pyd"))
        if hits:
            loader = importlib.machinery.ExtensionFileLoader(BINDING, str(hits[0]))
            spec = importlib.util.spec_from_file_location(BINDING, hits[0], loader=loader)
            if spec is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[BINDING] = mod
            loader.exec_module(mod)
            return mod
    return importlib.import_module(BINDING)  # an installed module (or ImportError: cannot tell)


def read_segment(name: str) -> dict[str, Any]:
    linux_shm = Path("/dev/shm") / name.lstrip("/")
    if sys.platform.startswith("linux") and not linux_shm.exists():
        return {"segment": False, "now_ns": time.monotonic_ns()}
    r1rt_py = import_binding()  # the release's build (rt/bindings); absent -> cannot tell

    try:
        view = r1rt_py.SegmentView.attach(name)
    except Exception as e:
        if not sys.platform.startswith("linux"):
            return {"segment": False, "now_ns": time.monotonic_ns(), "note": f"attach: {e}"}
        raise
    try:
        return state_from_view(view, alive=processes_alive())
    finally:
        view.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/")
    ap.add_argument("--shm-name", default=os.environ.get("R1_SHM_NAME", "/r1mc3.rt"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    try:
        state = read_fake(root, a.shm_name) if str(root) != "/" else read_segment(a.shm_name)
    except Exception as e:
        out = {"ok": False, "status": EXIT_CANNOT_TELL, "reason": f"cannot read the segment: {e}"}
        print(json.dumps(out))
        return EXIT_CANNOT_TELL
    status, reason = decide(state)
    if not a.quiet:
        rt = state.get("rt") or {}
        out = {
            "ok": status == EXIT_OK,
            "status": status,
            "reason": reason,
            "owner": state.get("token_owner"),
            "rt_mode": rt.get("mode"),
            "rt_age_ms": round((state["now_ns"] - rt["t_ns"]) / 1e6, 1) if "t_ns" in rt else None,
        }
        print(json.dumps(out))
    return status


if __name__ == "__main__":
    sys.exit(main())
