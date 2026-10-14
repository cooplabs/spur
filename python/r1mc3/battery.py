"""The R1's battery in r1d's status - READ-ONLY (session 13; HW2 day 2: the percentage came from a helper in v2's
environment over SSH, `results/hw2_20261013/tools/r1battery.py`, and the swaps were planned from it).

Unitree's battery management publishes `unitree_hg BmsState_` on `rt/lf/bmsstate` (the main pack) and, when a second
pack is fitted, on `rt/lf/secondary_bmsstate` (v2's battery probe, 30 Sep: `r1mc2/backends/robot.py:138-146`). This
module only SUBSCRIBES (no writer, nothing published, no mode touched): safe while anything else runs.

The message layout (r1mc3.dds.bms) = unitree_sdk2py's `unitree_hg.msg.dds_.BmsState_` as v2 read it on the R1,
decoded on day 2 by the helper (64 %, 34.59 V, -3.06 A, cells 3.847-3.854 V, 32 C). The units, as v2 read them
(`bms_to_dict`): `bmsvoltage` mV, `current` mA (NEGATIVE = discharging), `cell_vol` mV, `temperature` deg C.
`python -m r1mc3.battery --check` on the robot prints what arrives (and the remote's type name from discovery) -
the first look after a firmware change.

The drain: a straight-line fit of the percentage over the last `window_s` (default 10 min) once the samples span
>= 2 min - day 2: ~0.7 %/min hanging, ~1 %/min balancing (~115 W). The minutes left are to `swap_at_pct` (30 %
by default: day 2 swapped at 62 % and 33 %, the 4 Oct trouble began near 19 %) - words for the operator, nothing
automatic ever follows from them.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

TOPIC_MAIN = "rt/lf/bmsstate"  # = r1mc3.dds.bms.TOPIC_BMS (that module needs the dds extra)
TOPIC_SECONDARY = "rt/lf/secondary_bmsstate"
SWAP_AT_PCT = 30.0
WINDOW_S = 600.0  # the drain fit's window
MIN_SPAN_S = 120.0  # the drain is reported once the samples span this long
STALE_S = 10.0  # no message for this long -> the reading is called old
POLL_S = 1.0


@dataclass(frozen=True)
class Sample:
    """One battery reading (v2's bms_to_dict, + the power)."""

    soc_pct: int
    voltage_v: float | None
    current_a: float
    power_w: float | None  # positive while discharging
    cell_min_v: float | None
    cell_max_v: float | None
    temp_max_c: int | None
    soh_pct: int
    cycles: int


def decode(msg: Any) -> Sample:
    volts = [int(v) for v in list(getattr(msg, "bmsvoltage", []) or []) if v]
    cells = [int(c) for c in list(getattr(msg, "cell_vol", []) or []) if c]
    temps = [int(t) for t in list(getattr(msg, "temperature", []) or []) if t]
    v = round(volts[0] / 1000.0, 2) if volts else None
    i = round(float(msg.current) / 1000.0, 2)
    return Sample(
        soc_pct=int(msg.soc),
        voltage_v=v,
        current_a=i,
        power_w=None if v is None else round(-v * i, 1),
        cell_min_v=round(min(cells) / 1000.0, 3) if cells else None,
        cell_max_v=round(max(cells) / 1000.0, 3) if cells else None,
        temp_max_c=max(temps) if temps else None,
        soh_pct=int(getattr(msg, "soh", 0) or 0),
        cycles=int(getattr(msg, "cycle", 0) or 0),
    )


def drain_pct_per_min(hist: Sequence[tuple[float, float]], *, min_span_s: float = MIN_SPAN_S) -> float | None:
    """The percentage's fall per minute (a least-squares line through (t_s, soc) samples; positive = draining), or None
    while the samples span less than min_span_s."""
    if len(hist) < 2 or hist[-1][0] - hist[0][0] < min_span_s:
        return None
    n = float(len(hist))
    mt = sum(t for t, _ in hist) / n
    ms = sum(s for _, s in hist) / n
    den = sum((t - mt) ** 2 for t, _ in hist)
    if den <= 0.0:
        return None
    slope = sum((t - mt) * (s - ms) for t, s in hist) / den  # % per second
    return round(-60.0 * slope, 2)


Source = Callable[[], list[tuple[str, Any]]]  # -> [(topic, message), ...] newly arrived since the last call


class BatteryReader:
    """The newest battery reading + the drain, polled every POLL_S in a daemon thread (`start`) or by hand (`poll`).

    `source` (tests): returns the messages that arrived since the last call; default = DDS readers on the two topics
    (CycloneDDS, read-only) in `participant` or a new one on `domain`."""

    def __init__(
        self,
        domain: int = 0,
        *,
        participant: Any = None,
        source: Source | None = None,
        clock: Callable[[], float] = time.monotonic,
        swap_at_pct: float = SWAP_AT_PCT,
        window_s: float = WINDOW_S,
    ) -> None:
        self.domain = domain
        self.participant = participant
        self.source = source
        self.clock = clock
        self.swap_at_pct = float(swap_at_pct)
        self.window_s = float(window_s)
        self.latest: dict[str, tuple[float, Sample]] = {}
        self.hist: deque[tuple[float, float]] = deque()
        self.error: str | None = None
        self.messages = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._readers: list[tuple[str, Any]] = []

    # ------------------------------------------------------------------ the wire
    def _dds_source(self) -> Source:
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        from r1mc3.dds.bms import BmsState_

        dp = self.participant if self.participant is not None else DomainParticipant(self.domain)
        self.participant = dp
        t = BmsState_
        self._readers = [(name, DataReader(dp, Topic(dp, name, t))) for name in (TOPIC_MAIN, TOPIC_SECONDARY)]

        def take() -> list[tuple[str, Any]]:
            out: list[tuple[str, Any]] = []
            for name, rd in self._readers:
                for s in rd.take(N=16):
                    if type(s).__name__ != "InvalidSample":
                        out.append((name, s))
            return out

        return take

    def start(self) -> BatteryReader:
        if self.source is None:
            try:
                self.source = self._dds_source()
            except Exception as e:  # no dds extra / no network: the status says so, r1d runs on
                self.error = f"no battery reader: {type(e).__name__}: {e}"
                return self
        self._thread = threading.Thread(target=self._loop, name="r1d-battery", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(POLL_S):
            self.poll()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def poll(self) -> None:
        if self.source is None:
            return
        try:
            got = self.source()
        except Exception as e:  # a read error is reported, never raised into r1d
            self.error = f"battery read failed: {type(e).__name__}: {e}"
            return
        now = self.clock()
        with self._lock:
            for name, msg in got:
                try:
                    s = decode(msg)
                except (AttributeError, TypeError, ValueError) as e:
                    self.error = f"battery message not decoded: {e}"
                    continue
                self.messages += 1
                self.latest[name] = (now, s)
                if name == TOPIC_MAIN:
                    self.hist.append((now, float(s.soc_pct)))
            while self.hist and now - self.hist[0][0] > self.window_s:
                self.hist.popleft()

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        """r1d status `battery`: `available`, the newest main-pack reading, its age, the drain, the minutes to the swap
        level, `words` (empty until the first message: a robot without the topic, or the sim, says nothing)."""
        now = self.clock()
        with self._lock:
            main = self.latest.get(TOPIC_MAIN)
            second = self.latest.get(TOPIC_SECONDARY)
            hist = list(self.hist)
        if main is None:
            why = self.error or f"no message on {TOPIC_MAIN} yet"
            return {"available": False, "reason": why, "words": []}
        t, s = main
        age = round(now - t, 1)
        drain = drain_pct_per_min(hist)
        out: dict[str, Any] = {"available": True, **asdict(s), "age_s": age, "drain_pct_per_min": drain,
                               "swap_at_pct": self.swap_at_pct, "minutes_to_swap": None}  # fmt: skip
        if second is not None:
            out["secondary_soc_pct"] = second[1].soc_pct
        if drain is not None and drain > 0.05 and s.soc_pct > self.swap_at_pct:
            out["minutes_to_swap"] = round((s.soc_pct - self.swap_at_pct) / drain)
        out["words"] = battery_words(out, stale_s=STALE_S)
        return out


def battery_words(b: dict[str, Any], *, stale_s: float = STALE_S) -> list[str]:
    """The operator's line(s) (`r1 status` summary)."""
    if not b.get("available"):
        return []
    parts = [f"battery {b['soc_pct']} %"]
    det = []
    if b.get("voltage_v") is not None:
        det.append(f"{b['voltage_v']:.1f} V")
    if b.get("power_w") is not None:
        det.append(f"{b['power_w']:.0f} W")
    if b.get("cell_min_v") is not None:
        det.append(f"cells >= {b['cell_min_v']:.2f} V")
    if det:
        parts.append("(" + ", ".join(det) + ")")
    drain = b.get("drain_pct_per_min")
    if drain is None:
        parts.append("- the drain: measuring (2 min)")
    elif b.get("minutes_to_swap") is not None:
        parts.append(f"- ~{drain:.1f} %/min: ~{b['minutes_to_swap']} min to {b['swap_at_pct']:.0f} %")
    else:
        parts.append(f"- {drain:+.1f} %/min")
    words = [" ".join(parts)]
    if b["soc_pct"] <= b["swap_at_pct"]:
        words.append(f"battery at or below the swap level ({b['swap_at_pct']:.0f} %): plan the battery swap now")
    if float(b.get("age_s") or 0.0) > stale_s:
        words.append(f"battery: the last message is {b['age_s']:.0f} s old ({TOPIC_MAIN} stopped?)")
    return words


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m r1mc3.battery [--check] [--seconds 4]` - one reading from the robot (read-only; on the backpack)."""
    ap = argparse.ArgumentParser(prog="python -m r1mc3.battery", description=main.__doc__)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--check", action="store_true", help="also list the BMS topics' remote type names (discovery)")
    a = ap.parse_args(argv)
    rd = BatteryReader(a.domain).start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.seconds and TOPIC_MAIN not in rd.latest:
        time.sleep(0.1)
        rd.poll()
    st = rd.status()
    if a.check and rd.participant is not None:
        from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication

        pubs = BuiltinDataReader(rd.participant, BuiltinTopicDcpsPublication)
        seen: dict[str, set[str]] = {}
        for p in pubs.take(N=500):
            if str(getattr(p, "topic_name", "")).startswith("rt/lf/"):
                seen.setdefault(str(p.topic_name), set()).add(str(p.type_name))
        st["remote_types"] = {k: sorted(v) for k, v in seen.items()}
    rd.close()
    print("\n".join(st.get("words") or [st.get("reason", "")]))
    print(json.dumps(st))
    return 0 if st.get("available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
