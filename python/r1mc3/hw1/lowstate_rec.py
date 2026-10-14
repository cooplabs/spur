"""`python -m r1mc3.hw1.lowstate_rec --domain N --out FILE.npz` - r1d's rt/lowstate recorder as a process of its own.

The board-hold and guard-takeover tests record what the board does while r1rt is stopped (r1rt's own receive thread is
stopped then, so r1log has nothing of that window). The recorder runs in its own process so its Python work (500
samples/s x 35 motors) never holds r1d's GIL while r1d times the pause. Prints {"ready": true} at its first sample (DDS
matched), records until a "stop" line on stdin (or stdin closes), writes FILE.npz (`LowstateRecorder.arrays()`), prints
{"done": true, ...}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .pause import LowstateRecorder


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1mc3.hw1.lowstate_rec")
    ap.add_argument("--domain", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ready-timeout-s", type=float, default=5.0)
    a = ap.parse_args(argv)
    rec = LowstateRecorder(a.domain).start()
    end = time.monotonic() + a.ready_timeout_s
    while not rec.rows and time.monotonic() < end:
        time.sleep(0.01)
    print(json.dumps({"ready": True, "samples": len(rec.rows)}), flush=True)
    sys.stdin.readline()  # "stop" (or EOF: the parent went away)
    rec.stop()
    arrays = rec.arrays()
    np.savez(a.out, **arrays)  # type: ignore[arg-type]
    print(json.dumps({"done": True, "samples": int(arrays["t_rx_ns"].size)}), flush=True)
    rec.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
