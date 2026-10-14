"""A grasp's hand data as the grip saw it - recorded on the robot, replayed offline (session 12).

    python -m r1mc3.hands.trace replay FILE.npz [--json]     # the grip's verdict again, from the recorded states
    python -m r1mc3.hands.trace info FILE.npz

Why (HW2 day 1, 12 Oct 2026): the tape pinch read NOT_HOLDING 5 / 5 while the user saw it held, and nothing on the robot
had kept what the verdict saw - r1log carries the pads' contact bits (and its grip fields stayed 0: r1d never
registered its grasps with the contact publisher), r1d's journal the request and the code, the B4 log the state word.
The failing term could not be named afterwards. From session 12 on r1d wraps the hand interface each grasp uses
(`TracingHands`): every state the GraspRunner reads (q, dq, tau_est, the 9 x 12 pads, temperature, valid), every
command it sends, every lift notification and contact-height answer, its events and its final verdict - one compressed
file per grasp in r1d's records dir (`grasp_<UTC>_<side>.npz`, ~0.1-0.4 MB; saved by `r1 shutdown` with the other
records). `replay()` feeds the recorded states back to a fresh GraspRunner in the same order at the same times: the
runner is deterministic, so the replay reproduces the robot's verdict (a self-check: `same_as_recorded`) - and a changed
rule can then be judged on the robot's own data.

Generic (D32): grips, object classes, widths - nothing task-specific.
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "r1mc3.grasp_trace/1"
MAX_SAMPLES = 12_000  # 120 s at 100 Hz: the close, the lift and the hold are in the first ~15 s; later ticks dropped


class TracingHands:
    """The hand interface a GraspRunner gets in r1d: passes everything through to the real hands and keeps what the
    runner read / sent for `side` (only `side`; another side's reads pass through unrecorded)."""

    def __init__(
        self, hands: Any, side: str, params: Mapping[str, Any], *, clock_ns: Callable[[], int] | None = None
    ) -> None:
        self.hands, self.side = hands, side
        self.params = dict(params)
        self.now_ns = 0  # r1d sets it before each call into the runner (its tick's time = the runner's clock)
        self.clock_ns: Callable[[], int] = clock_ns if clock_ns is not None else (lambda: self.now_ns)
        self.last_verdict: dict[str, Any] | None = None  # r1d keeps the runner's last verdict before a release
        self.t: list[int] = []
        self.q: list[Any] = []
        self.dq: list[Any] = []
        self.tau: list[Any] = []
        self.press: list[Any] = []
        self.temp: list[float] = []
        self.valid: list[bool] = []
        self.age_ms: list[float] = []  # the hand state's age at the read (dex3: > 100 ms = stale)
        self.wall_ns: list[int] = []  # the read's wall-clock time: r1d's real tick spacing (a starved tick shows here)
        self.why_invalid: list[str] = []
        self.cmd_t: list[int] = []
        self.cmd_q: list[Any] = []
        self.lifts: list[tuple[int, int, float]] = []  # (the read count before it, t_ns, dz_m)
        self.heights: list[dict[str, float]] = []
        t_wall = time.time()  # to the millisecond: two grasps of one side within a second must not share a file name
        self.utc = time.strftime("%Y%m%dT%H%M%S", time.gmtime(t_wall)) + f".{int(t_wall * 1000) % 1000:03d}Z"
        self.closed = False

    # the hands interface -------------------------------------------------------------------------------------
    def state(self, side: str) -> dict[str, Any]:
        st: dict[str, Any] = self.hands.state(side)
        if side == self.side and not self.closed and len(self.t) < MAX_SAMPLES:
            n = 7

            def arr(key: str, shape: tuple[int, ...]) -> Any:
                v = st.get(key)
                if v is None:
                    return np.full(shape, np.nan)
                a = np.asarray(v, dtype=np.float64)
                return a.reshape(shape) if a.size == int(np.prod(shape)) else np.full(shape, np.nan)

            self.t.append(int(self.clock_ns()))
            self.q.append(arr("q", (n,)))
            self.dq.append(arr("dq", (n,)))
            self.tau.append(arr("tau_est", (n,)) if st.get("tau_est") is not None else arr("tau", (n,)))
            self.press.append(arr("pressure", (9, 12)))
            temp = st.get("temperature", st.get("temp"))
            self.temp.append(float(np.nanmax(np.asarray(temp, dtype=np.float64))) if temp is not None else np.nan)
            self.valid.append(bool(st.get("valid", True)) and st.get("q") is not None)
            age = st.get("age_ms")
            self.age_ms.append(float(age) if isinstance(age, (int, float)) else np.nan)
            self.wall_ns.append(time.monotonic_ns())
            self.why_invalid.append(str(st.get("invalid_reason") or ""))
        return st

    def command(self, side: str, q: Any, **kw: Any) -> Any:
        if side == self.side and not self.closed and len(self.cmd_t) < MAX_SAMPLES:
            self.cmd_t.append(int(self.clock_ns()))
            self.cmd_q.append(np.asarray(q, dtype=np.float64).copy())
        return self.hands.command(side, q, **kw)

    def __getattr__(self, name: str) -> Any:  # anything else of the real hands (e.g. Dex3Hands.start / close)
        return getattr(self.hands, name)

    # the runner's other inputs -------------------------------------------------------------------------------
    def wrap_contact_height(self, f: Callable[[Any], Mapping[str, float]] | None) -> Any:
        """The runner's contact_height(q), its answers kept (the replay gives them back in order)."""
        if f is None:
            return None

        def g(q: Any) -> Mapping[str, float]:
            out = f(q)
            self.heights.append({k: float(v) for k, v in out.items()})
            return out

        return g

    def note_lift(self, dz_m: float) -> None:
        """Call next to the runner's notify_lift (r1d's _tick_grips)."""
        if not self.closed:
            self.lifts.append((len(self.t), int(self.clock_ns()), float(dz_m)))

    # the file ------------------------------------------------------------------------------------------------
    def arrays(self, runner: Any = None) -> dict[str, Any]:
        verdict = self.last_verdict
        if verdict is None and runner is not None:
            verdict = runner.verdict().as_dict()
        meta = {
            "schema": SCHEMA,
            "side": self.side,
            "utc": self.utc,
            "params": self.params,
            "verdict": verdict,
            "events": list(getattr(runner, "events", [])) if runner is not None else [],
            "heights": self.heights,
            "lifts": self.lifts,
            "samples": len(self.t),
            "invalid_reasons": sorted({w for w in self.why_invalid if w}),
            "truncated": len(self.t) >= MAX_SAMPLES,
        }
        return {
            "meta": np.array(json.dumps(meta, default=_json_default)),
            "t_ns": np.asarray(self.t, dtype=np.int64),
            "q": np.asarray(self.q, dtype=np.float64).reshape(-1, 7),
            "dq": np.asarray(self.dq, dtype=np.float64).reshape(-1, 7),
            "tau_est": np.asarray(self.tau, dtype=np.float64).reshape(-1, 7),  # float64: the replay is exact
            "pressure": np.asarray(self.press, dtype=np.float32).reshape(-1, 9, 12),  # raw ints < 2^24: exact
            "temperature": np.asarray(self.temp, dtype=np.float32),
            "valid": np.asarray(self.valid, dtype=bool),
            "age_ms": np.asarray(self.age_ms, dtype=np.float32),
            "wall_ns": np.asarray(self.wall_ns, dtype=np.int64),
            "cmd_t_ns": np.asarray(self.cmd_t, dtype=np.int64),
            "cmd_q": np.asarray(self.cmd_q, dtype=np.float32).reshape(-1, 7),
        }

    def save(self, directory: str | Path, runner: Any = None, *, background: bool = True) -> Path:
        """Write the trace (compressed npz) and stop recording; in a background thread by default - r1d's tick must
        never wait on a disk (session 12's lesson from the hotfix: a slow r1d tick starves the hands' reader)."""
        self.closed = True
        arrays = self.arrays(runner)
        path = _claim(Path(directory) / f"grasp_{self.utc}_{self.side}.npz")

        def write() -> None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **arrays)
                tmp.replace(path)
            except OSError as e:  # a full disk never stops r1d
                print(f"r1d: grasp trace not saved ({e})", file=sys.stderr)

        if background:
            threading.Thread(target=write, name="grasp-trace", daemon=True).start()
        else:
            write()
        return path


_CLAIMED: set[Path] = set()
_CLAIM_LOCK = threading.Lock()


def _claim(path: Path) -> Path:
    """A trace path no other trace of this process holds and no file has: `-1`, `-2` ... before the side when needed
    (session 12, the integrator: the Linux CI's lockstep test ran two grasps within one wall second and the second
    trace REPLACED the first - the name had whole seconds only; the background writer makes a race possible too)."""
    with _CLAIM_LOCK:
        cand, n = path, 0
        while cand in _CLAIMED or cand.exists():
            n += 1
            stem = path.name[: -len(".npz")]
            head, _, side = stem.rpartition("_")
            cand = path.with_name(f"{head}-{n}_{side}.npz")
        _CLAIMED.add(cand)
        return cand


def trace_dir(svc: Any) -> Path | None:
    """Where r1d keeps its records (its journal's directory, else $R1_RECORD_DIR); None = no traces (tests, a r1d
    without a records dir) or `R1_GRASP_TRACE=0`."""
    import os

    if os.environ.get("R1_GRASP_TRACE", "1") == "0":
        return None
    jp = getattr(svc, "_journal_path", None)
    if jp is not None:
        return Path(jp).parent
    d = os.environ.get("R1_RECORD_DIR")
    return Path(d) if d else None


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def load(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out: dict[str, Any] = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


class _ReplayHands:
    """Gives the recorded states back in order (one per state() call of the traced side); commands are ignored."""

    def __init__(self, tr: Mapping[str, Any]) -> None:
        self.tr, self.k = tr, 0
        self.side = str(tr["meta"]["side"])

    def state(self, side: str) -> dict[str, Any]:
        k = min(self.k, len(self.tr["t_ns"]) - 1)
        self.k += 1
        return {
            "q": np.asarray(self.tr["q"][k], dtype=np.float64),
            "dq": np.asarray(self.tr["dq"][k], dtype=np.float64),
            "tau_est": np.asarray(self.tr["tau_est"][k], dtype=np.float64),
            "pressure": np.asarray(self.tr["pressure"][k], dtype=np.float64),
            "temperature": np.full((7, 2), float(self.tr["temperature"][k])),
            "valid": bool(self.tr["valid"][k]),
        }

    def command(self, side: str, q: Any, **kw: Any) -> dict[str, Any]:
        return {"ok": True}


def trace_wall_s(tr: Mapping[str, Any]) -> float:
    """The trace's first read in wall-clock seconds (Unix): its file stamp (TracingHands.utc, taken when the grasp
    request arrived - r1d's journal logs the same request at the same millisecond)."""
    u = str(tr["meta"]["utc"])
    head, _, ms = u.rstrip("Z").partition(".")
    return float(calendar.timegm(time.strptime(head, "%Y%m%dT%H%M%S"))) + (int(ms) / 1000.0 if ms else 0.0)


# Session 13: the arm's rise after a grasp as r1d measured it on the robot (HW2 day 2, the 20 traces whose grip stayed
# TENTATIVE into the lift - r1d notifies the grip only then): the grasp centre starts rising 0.30-0.67 s after the lift
# request (median ~0.38: the plan + the start) and passes 1 cm 0.1-0.65 s later (the arm's ~2-8 cm/s).
LIFT_DELAY_S = 0.40
LIFT_RATE_M_S = 0.04
LIFT_OPS = ("arm.move_relative", "arm.move_linear")


def lifts_from_journal(
    tr: Mapping[str, Any], journal: Sequence[Mapping[str, Any]], *, within_s: float = 15.0
) -> list[tuple[int, int, float]]:
    """The lift notifications a trace WOULD have carried had its grip stayed TENTATIVE: r1d notifies a grip of the
    arm's rise only while it is TENTATIVE (L2._tick_grips), so a trace whose verdict was NOT_HOLDING at the close has
    none - and a changed rule replayed on it needs them. From r1d's journal (its request lines, `utc` + `request`): the
    first arm move UP (delta z > 0) of the trace's side within `within_s` after the grasp, as a ramp from LIFT_DELAY_S
    after the request at LIFT_RATE_M_S up to the move's delta - one notification per recorded read (as r1d's tick).
    A RECONSTRUCTION (the rise's timing from the robot's other grasps): say so wherever it is used. [] = no lift."""
    side = str(tr["meta"]["side"])
    w0 = trace_wall_s(tr)
    t = [int(x) for x in tr["t_ns"]]
    if not t:
        return []
    for e in journal:
        req = e.get("request") or {}
        if req.get("op") not in LIFT_OPS or req.get("arm") != side:
            continue
        w = journal_wall_s(str(e.get("utc", "")))
        d = req.get("delta") or [0.0, 0.0, 0.0]
        if w is None or not (w0 < w <= w0 + within_s) or float(d[2]) <= 0.0:
            continue
        t_req = t[0] + int((w - w0) * 1e9)
        return [
            (k, tk, min(float(d[2]), max(0.0, (tk - t_req) / 1e9 - LIFT_DELAY_S) * LIFT_RATE_M_S))
            for k, tk in enumerate(t)
            if tk >= t_req
        ]
    return []


def journal_wall_s(utc: str) -> float | None:
    """r1d's journal stamp ("2026-10-14T01:27:25.222Z") in Unix seconds; None if it is not one."""
    try:
        head, _, frac = utc.rstrip("Z").partition(".")
        return float(calendar.timegm(time.strptime(head, "%Y-%m-%dT%H:%M:%S"))) + (float("0." + frac) if frac else 0.0)
    except ValueError:
        return None


def replay(
    tr: Mapping[str, Any],
    *,
    runner_factory: Any = None,
    lifts: Sequence[tuple[int, int, float]] | None = None,
    until_s: float | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """The grip's verdict again from a recorded trace: a fresh GraspRunner (the recorded parameters + `overrides`)
    started at the first recorded read and ticked at each later one, the lift notifications and contact-height answers
    given back where they came (`lifts` replaces the recorded notifications - `lifts_from_journal`; `until_s` stops
    the replay that long after the first read: the verdict a caller read then). Returns {"verdict", "stops",
    "events", "same_as_recorded"} (same_as_recorded compares with the trace's LAST verdict)."""
    from .grip import GraspRunner

    meta = tr["meta"]
    p = dict(meta["params"])
    p.update(overrides)
    hands = _ReplayHands(tr)
    heights = list(meta.get("heights") or [])
    ch: Any = None
    if p.pop("contact_height", False):

        def ch(q: Any) -> Mapping[str, float]:
            return heights.pop(0) if heights else {}

    factory = runner_factory or GraspRunner
    r = factory(hands, meta["side"], p.pop("grip"), p.pop("object_class"), contact_height=ch, **p)
    t = [int(x) for x in tr["t_ns"]]
    src = (meta.get("lifts") or []) if lifts is None else lifts
    lifts = [(int(a), int(b), float(c)) for a, b, c in src]
    if not t:
        raise ValueError("an empty trace")
    r.start(t[0])
    for k in range(1, len(t)):
        if until_s is not None and t[k] - t[0] > until_s * 1e9:
            break
        while lifts and lifts[0][0] <= k:
            r.notify_lift(lifts.pop(0)[2])
        r.tick(t[k])
    v = r.verdict().as_dict()
    rec = meta.get("verdict") or {}
    same = (v.get("state"), v.get("reason")) == (rec.get("state"), rec.get("reason"))
    return {"verdict": v, "stops": r.stops(), "events": r.events, "same_as_recorded": bool(same)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1mc3.hands.trace", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("replay", help="the grip's verdict again from the recorded states")
    a1.add_argument("file")
    a1.add_argument("--json", action="store_true")
    a2 = sub.add_parser("info", help="what the trace holds")
    a2.add_argument("file")
    a = ap.parse_args(argv)
    tr = load(a.file)
    meta = tr["meta"]
    if a.cmd == "info":
        v, prm = meta.get("verdict") or {}, meta["params"]
        dur = (tr["t_ns"][-1] - tr["t_ns"][0]) / 1e9 if len(tr["t_ns"]) else 0.0
        print(f"{a.file}: {meta['side']} {prm.get('grip')} {prm.get('object_class')} width {prm.get('width_m')}; "
              f"{meta['samples']} states over {dur:.2f} s; verdict {v.get('state')} ({v.get('reason')})")  # fmt: skip
        if len(tr.get("wall_ns", [])) > 1:
            gaps = np.diff(tr["wall_ns"]) / 1e6
            ages = tr["age_ms"][np.isfinite(tr["age_ms"])]
            age_max = float(np.max(ages)) if ages.size else float("nan")
            print(f"  r1d's reads: wall-clock gap max {float(np.max(gaps)):.1f} ms (p99 "
                  f"{float(np.percentile(gaps, 99)):.1f}); state age max {age_max:.1f} ms; invalid reads "
                  f"{int(np.sum(~tr['valid']))} {meta.get('invalid_reasons') or ''}")  # fmt: skip
        return 0
    out = replay(tr)
    if a.json:
        print(json.dumps(out, default=_json_default))
    else:
        v = out["verdict"]
        print(f"replayed: {v['state']} ({v['reason']}); stops {out['stops']}; same as the robot's: "
              f"{out['same_as_recorded']}")  # fmt: skip
        for k, term in (v.get("evidence") or {}).get("terms", {}).items():
            print(f"  {k}: {json.dumps(term, default=_json_default)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
