"""The Dex3 pads AT REST, read out against r1hands' contact threshold (HW2 runbook H2; HW2_PREP.md "Session 10").

    python -m r1mc3.record.pads_rest [--side left|right|both] [--seconds 10] [--hz 25] [--port 8780] [--json out.json]

Why: r1hands calls a pad "in contact" when its largest taxel rises more than `CONTACT_THRESHOLD` (raw units,
r1mc3.hands.contact) above that pad's TARE (the median of its largest taxel over the first `TARE_SAMPLES` streaming
states, the hand open at start). HW2 day 1 (12 Oct 2026, H2) read the real pads with this tool: noise sd 5-8, at most
+16 above the tare in 10 s, vs the 5.0 estimate of the time (0.4-45 % of the samples "in contact" per pad) -> the user
raised the threshold to 40 raw (session 12). The simulator's pad model carries ~100 units of noise (15 x the real
pads): in sim the pads still read "contact" at rest. This tool decides nothing and changes nothing: it reads.

What it does (moves nothing; read-only through r1d, like `r1 hand-state <side>` typed again and again): polls r1d's
`hand.state` for `seconds` per hand, takes each pad's largest taxel per sample (the same reduction as the contact
code), uses the first `TARE_SAMPLES / 100 Hz` = 0.5 s as the tare (its median, as r1hands does), and reports per pad
the mean / sd / max of (peak - tare) over the rest, the share of samples above the threshold, and how many of the 9
pads would read "contact" at rest. It also reads r1d's own contact bits (`status.hands_contact`) once at the end - what
the guards actually see. Hands open, touching nothing, nobody near the fingertips.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from r1mc3.hands.contact import CONTACT_THRESHOLD, N_PADS, TARE_SAMPLES

FloatArray = npt.NDArray[np.float64]
TARE_S = TARE_SAMPLES / 100.0  # r1hands tares over its first 50 states at r1d's 100 Hz tick = 0.5 s
PAD_NAMES = ("thumb pad", "thumb tip", "finger A pad", "finger A tip", "finger B pad", "finger B tip", "palm 1",
             "palm 2", "palm 3")  # fmt: skip  # HW1_RUNBOOK.md B3.2a's pad numbering


def pad_peaks(pressure: Any) -> FloatArray | None:
    """The largest taxel per pad (9,) of one hand state's `pressure` (9 x 12), None if absent / malformed."""
    if pressure is None:
        return None
    p = np.asarray(pressure, dtype=np.float64)
    if p.size % N_PADS != 0:
        return None
    out: FloatArray = np.max(p.reshape(N_PADS, -1), axis=1)
    return out


def summarize(t: Sequence[float], peaks: Sequence[FloatArray], threshold: float = CONTACT_THRESHOLD,
              tare_s: float = TARE_S) -> dict[str, Any]:  # fmt: skip
    """The at-rest numbers of one hand: samples (t in s from the first, the (9,) peaks each). The tare = the median of
    each pad's peak over the first `tare_s` (at least one sample); the statistics over the samples after it."""
    tt = np.asarray(t, dtype=np.float64)
    pk = np.stack([np.asarray(p, dtype=np.float64) for p in peaks]) if len(peaks) else np.zeros((0, N_PADS))
    if pk.shape[0] < 3:
        return {"samples": int(pk.shape[0]), "error": "fewer than 3 valid hand states"}
    tt = tt - tt[0]
    n_tare = max(1, int(np.count_nonzero(tt < tare_s)))
    if n_tare >= pk.shape[0] - 1:
        n_tare = max(1, pk.shape[0] // 4)
    tare = np.median(pk[:n_tare], axis=0)
    above = pk[n_tare:] - tare
    over = above > threshold
    pads = []
    for k in range(N_PADS):
        pads.append({
            "pad": k,
            "name": PAD_NAMES[k],
            "tare": round(float(tare[k]), 1),
            "mean": round(float(above[:, k].mean()), 2),
            "sd": round(float(above[:, k].std()), 2),
            "max": round(float(above[:, k].max()), 2),
            "share_over_threshold": round(float(over[:, k].mean()), 3),
        })  # fmt: skip
    n_contact = int(np.count_nonzero(over.mean(axis=0) > 0.5))
    noise = float(np.median(above.std(axis=0)))
    return {
        "samples": int(pk.shape[0]),
        "tare_samples": n_tare,
        "duration_s": round(float(tt[-1]), 2),
        "rate_hz": round(float((pk.shape[0] - 1) / max(float(tt[-1]), 1e-9)), 1),
        "threshold": threshold,
        "pads": pads,
        "pads_over_threshold_most_of_the_time": n_contact,
        "pads_ever_over_threshold": int(np.count_nonzero(over.any(axis=0))),
        "noise_sd_median": round(noise, 2),
        "noise_vs_threshold": round(noise / threshold, 1) if threshold > 0 else None,
    }


def words(side: str, s: dict[str, Any]) -> str:
    """One line in plain words for the operator."""
    if "error" in s:
        return f"{side}: {s['error']}"
    worst = max(s["pads"], key=lambda p: p["max"])
    return (
        f"{side}: {s['samples']} states over {s['duration_s']} s; at rest {s['pads_over_threshold_most_of_the_time']} "
        f"of {N_PADS} pads read 'contact' most of the time ({s['pads_ever_over_threshold']} at least once) with the "
        f"threshold {s['threshold']:g}; the pads' noise sd (median) {s['noise_sd_median']:g} = "
        f"{s['noise_vs_threshold']:g} x the threshold; the noisiest: {worst['name']} (max {worst['max']:g} above its "
        "tare). This tool reads the threshold, it never changes it (40 raw: the user's decision on HW2's H2, 12 Oct)."
    )


def read_hand(call: Callable[..., dict[str, Any]], side: str, seconds: float, hz: float,
              clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep
              ) -> tuple[list[float], list[FloatArray], int]:  # fmt: skip
    """Poll `hand.state` of one side for `seconds` at ~`hz`: (t, peaks, invalid states)."""
    ts: list[float] = []
    pk: list[FloatArray] = []
    bad = 0
    t_end = clock() + seconds
    period = 1.0 / hz
    nxt = clock()
    while clock() < t_end:
        res = call("hand.state", side=side)
        st = (res.get("data") or {}).get(side) or {}
        p = pad_peaks(st.get("pressure")) if res.get("ok") and st.get("valid") else None
        if p is None:
            bad += 1
        else:
            ts.append(clock())
            pk.append(p)
        nxt += period
        d = nxt - clock()
        if d > 0:
            sleep(d)
    return ts, pk, bad


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1mc3.record.pads_rest", description=__doc__.splitlines()[0])
    ap.add_argument("--side", choices=("left", "right", "both"), default="both")
    ap.add_argument("--seconds", type=float, default=10.0, help="per hand (the runbook: 10 s)")
    ap.add_argument("--hz", type=float, default=25.0, help="hand.state reads per second")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--json", default="", help="also write the numbers here")
    a = ap.parse_args(argv)
    from r1mc3.service.client import Client, R1dError

    cl = Client(a.host, a.port)
    out: dict[str, Any] = {"tool": "r1mc3.record.pads_rest", "threshold": CONTACT_THRESHOLD, "hands": {}}
    try:
        for side in ("left", "right") if a.side == "both" else (a.side,):
            ts, pk, bad = read_hand(cl.call, side, a.seconds, a.hz)
            s = summarize(ts, pk)
            s["invalid_states"] = bad
            out["hands"][side] = s
            out.setdefault("words", []).append(words(side, s))
        st = cl.call("status")
        out["r1d_contact_bits"] = (st.get("data") or {}).get("hands_contact")
    except R1dError as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        return 2
    finally:
        cl.close()
    bits = out.get("r1d_contact_bits") or {}
    out["r1d_contact_words"] = {
        s: f"r1d's guards see {bin(int(bits.get('pad_' + s, 0))).count('1')} of {N_PADS} {s} pads in contact now"
        for s in ("left", "right")
        if "pad_" + s in bits
    }
    out["ok"] = all("error" not in h for h in out["hands"].values())
    text = json.dumps(out, indent=1)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text)
    print(text)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
