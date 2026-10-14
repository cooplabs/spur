"""Cut a grasp trace to its first seconds - small enough for a test asset, still replayed exactly (session 13).

    python tools/hands/cut_trace.py IN.npz OUT.npz --seconds 7.0 [--lean]

Keeps the reads (and the commands, lift notifications, contact-height answers) up to `--seconds` after the first read;
the meta's `samples` follows. `--lean` (test assets): the joint velocities zeroed (the grip never reads dq) and r1d's
read timing (wall_ns / age_ms: for `info` only) dropped - the replay is unchanged. The recorded verdict is kept as it
was (the full trace's last verdict) - a replay of the cut trace is compared with what the labels say, not with that
verdict. Generic (D32).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from r1mc3.hands import trace as T

PER_READ = ("t_ns", "q", "dq", "tau_est", "pressure", "temperature", "valid", "age_ms", "wall_ns")


def cut(tr: dict[str, Any], seconds: float, *, lean: bool = False) -> dict[str, Any]:
    t = np.asarray(tr["t_ns"], dtype=np.int64)
    n = int(np.searchsorted(t, t[0] + int(seconds * 1e9), side="right"))
    out: dict[str, Any] = {k: np.asarray(tr[k])[:n] for k in PER_READ if k in tr}
    if lean:
        out["dq"] = np.zeros_like(out["dq"])
        out.pop("wall_ns", None)
        out.pop("age_ms", None)
    m = int(np.searchsorted(np.asarray(tr["cmd_t_ns"], dtype=np.int64), t[0] + int(seconds * 1e9), side="right"))
    out["cmd_t_ns"] = np.asarray(tr["cmd_t_ns"])[:m]
    out["cmd_q"] = np.asarray(tr["cmd_q"])[:m]
    meta = dict(tr["meta"])
    meta["lifts"] = [lf for lf in meta.get("lifts") or [] if int(lf[0]) < n]
    meta["samples"] = n
    meta["cut_s"] = seconds
    meta["lean"] = lean
    out["meta"] = np.array(json.dumps(meta, default=T._json_default))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--lean", action="store_true", help="dq zeroed, the read timing dropped (test assets)")
    a = ap.parse_args(argv)
    arrays = cut(T.load(a.src), a.seconds, lean=a.lean)
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.dst, **arrays)
    print(f"{a.dst}: {int(json.loads(str(arrays['meta']))['samples'])} reads, {Path(a.dst).stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
