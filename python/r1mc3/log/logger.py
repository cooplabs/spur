"""r1log's core: drain r1rt's rings, write the session log, feed the flight recorder, spot the triggers.

`Logger.poll()` is one drain cycle (the CLI calls it every 20 ms; a lockstep test can call it every writer tick). It
only POPS from the single-consumer rings and READS seqlocks: the RT core never waits on r1log (DESIGN §9 - a logger
that falls behind loses records, counted twice: the ring's own drop counter on the producer side and the gaps r1log
sees in the writer's tick numbers).

Streams: every telemetry record (500 Hz) and policy record (50 Hz) from the rings, and (layout 2, session 5) every
r1guard cycle record (1 kHz) from the guard's own ring; the hands-contact seqlock and r1guard's heartbeat are SAMPLED
once per poll (written when their write count moved); events (mode / token owner /
fault-bit changes in the telemetry, stop-channel requests) and 1 Hz stats as JSON.

Triggers for the flight recorder (all in record time, rising edges only): a fault bit in `trigger_faults` appears
(default: not the 10 ms lowstate-late bit and not the clamp counters, which fire in normal operation), the mode
enters GUARDED / STAND_FREEZE / DAMP, the token goes to r1guard ("takeover"), the absolute IMU tilt passes
`tilt_trigger_deg` (re-armed below `tilt_rearm_deg`), an operator stop / halt / damp request, `request_dump()`.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import records as R
from ._native import native
from .flight import FlightRecorder
from .mcapfile import RotatingLog, utc_stamp

DEFAULT_TRIGGER_FAULTS = (
    R.FAULT_BITS["lowstate_stale"]
    | R.FAULT_BITS["lowstate_lost"]
    | R.FAULT_BITS["crc"]
    | R.FAULT_BITS["tilt_arm_stop"]
    | R.FAULT_BITS["tilt_fall"]
    | R.FAULT_BITS["motor_error"]
    | R.FAULT_BITS["policy_late"]
    | R.FAULT_BITS["policy_invalid"]
    | R.FAULT_BITS["loop_gap"]
    | R.FAULT_BITS["not_owner"]
)
TRIGGER_MODES = (4, 5, 8)  # GUARDED, STAND_FREEZE, DAMP
# Session 8 (HW1 fix #33): r1rt raises lowstate_stale + lowstate_lost from its first tick until the first lowstate
# arrives - at BOTH HW1 power-ups that wrote a flight file at every start (flight_..._000_fault-lowstate_stale+
# lowstate_lost.mcap, before any state). Until the core has received its first lowstate (telemetry t_state_rx_ns
# != 0) these two bits are logged as events but do not trigger a dump; after it, they trigger as before.
STARTUP_MASKED_FAULTS = R.FAULT_BITS["lowstate_stale"] | R.FAULT_BITS["lowstate_lost"]


@dataclass
class LoggerConfig:
    out_dir: Path
    prefix: str = "r1log"
    max_bytes: int = 512 << 20
    max_span_s: float = 600.0
    chunk_size: int = 1 << 20
    flight: bool = True
    flight_window_s: float = 30.0
    flight_post_s: float = 5.0
    flight_holdoff_s: float = 10.0
    trigger_faults: int = DEFAULT_TRIGGER_FAULTS
    trigger_modes: tuple[int, ...] = TRIGGER_MODES
    tilt_trigger_deg: float = 10.0
    tilt_rearm_deg: float = 5.0
    stats_period_s: float = 1.0
    stamp: str = field(default_factory=utc_stamp)


@dataclass
class StreamCount:
    records: int = 0
    seq_gaps: int = 0  # records missing between consecutive seq numbers seen by r1log
    seq_backwards: int = 0
    last_seq: int | None = None

    def update(self, seq: np.ndarray[Any, Any]) -> None:
        if len(seq) == 0:
            return
        s = seq.astype(np.int64)
        d = np.diff(s) if self.last_seq is None else np.diff(np.concatenate([[self.last_seq], s]))
        self.seq_gaps += int(np.sum(np.maximum(d - 1, 0)))
        self.seq_backwards += int(np.sum(d <= 0))
        self.records += len(s)
        self.last_seq = int(s[-1])

    def as_dict(self) -> dict[str, int]:
        return {"records": self.records, "seq_gaps": self.seq_gaps, "seq_backwards": self.seq_backwards}


class Logger:
    """One r1log instance on one segment view (r1rt_py.SegmentView: attach / create / of_core)."""

    def __init__(self, view: Any, cfg: LoggerConfig, *, source: str = "") -> None:
        mod = native()
        layout = mod.shm_layout()
        R.check_layout(layout)
        self.view = view
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        abi = int(layout["abi_version"])
        meta = {
            "source": source or str(getattr(view, "kind", "")),
            "host": socket.gethostname(),
            "layout_version": str(layout["layout_version"]),
            "abi_version": str(abi),
            "clock": "record times are CLOCK_MONOTONIC ns of the writer (lockstep: simulated time)",
            "started_utc": utc_stamp(),
        }
        self.log = RotatingLog(self.out_dir, meta, prefix=cfg.prefix, max_bytes=cfg.max_bytes,
                               max_span_s=cfg.max_span_s, chunk_size=cfg.chunk_size, stamp=cfg.stamp,
                               abi_version=abi)  # fmt: skip
        self.flight = (
            FlightRecorder(
                self.out_dir,
                window_s=cfg.flight_window_s,
                post_s=cfg.flight_post_s,
                holdoff_s=cfg.flight_holdoff_s,
                metadata=meta,
                abi_version=abi,
            )
            if cfg.flight
            else None
        )
        st = view.ring_stats()
        self._drops0 = {k: int(v["drops"]) for k, v in st.items()}
        self.ring_drops = {k: 0 for k in st}
        self.counts = {"telemetry": StreamCount(), "policy": StreamCount(), "guard": StreamCount()}
        self.sampled = {"hands_contact": 0, "guard_heartbeat": 0}
        self._hands_writes = 0
        self._guard_writes = 0
        self._last_faults = 0
        self._state_seen = False  # the core has received a lowstate (fix #33)
        self._last_mode: int | None = None
        self._last_owner: int | None = None
        self._tilt_armed = True
        self._stops = [int(r[0]) for r in view.stop_requests()]
        self._t_last = 0
        self._t_stats: int | None = None
        self.events = 0
        self.polls = 0
        self.closed = False

    # ---------------------------------------------------------------- one drain cycle
    def poll(self) -> int:
        """Drain everything available; returns the number of ring records written."""
        if self.closed:
            raise RuntimeError("logger closed")
        self.polls += 1
        n = self._drain_telemetry() + self._drain_policy() + self._drain_guard()
        self._sample_seqlocks()
        self._check_stops()
        st = self.view.ring_stats()
        for k, v in st.items():
            self.ring_drops[k] = int(v["drops"]) - self._drops0[k]
        if self._t_stats is None or self._t_last - self._t_stats >= round(self.cfg.stats_period_s * 1e9):
            self._t_stats = self._t_last
            self._write_json("stats", self._t_last, self.stats())
        if self.flight is not None:
            self.flight.poll()
        self.log.maybe_rotate(self._close_meta())  # the next file opens with the next record
        return n

    def _drain_telemetry(self) -> int:
        raw = self.view.pop_telemetry()
        if not raw:
            return 0
        rec = np.frombuffer(raw, dtype=R.TELEMETRY)
        t = rec["t_wake_ns"].astype(np.int64)
        seq = rec["seq"]
        f = self.log.current()
        size = R.TELEMETRY.itemsize
        for i in range(len(rec)):
            f.write("telemetry", int(t[i]), raw[i * size : (i + 1) * size], int(seq[i]))
        self.counts["telemetry"].update(seq)
        if self.flight is not None:
            self.flight.add("telemetry", t, seq, raw, size)
        self._t_last = max(self._t_last, int(t[-1]))
        self._scan_telemetry(rec, t)
        return len(rec)

    def _drain_policy(self) -> int:
        raw = self.view.pop_policy()
        if not raw:
            return 0
        rec = np.frombuffer(raw, dtype=R.POLICY)
        t = np.where(rec["t_action_ns"] != 0, rec["t_action_ns"], rec["t_obs_ns"]).astype(np.int64)
        seq = rec["seq"]
        f = self.log.current()
        size = R.POLICY.itemsize
        for i in range(len(rec)):
            f.write("policy", int(t[i]), raw[i * size : (i + 1) * size], int(seq[i]))
        self.counts["policy"].update(seq)
        if self.flight is not None:
            self.flight.add("policy", t, seq, raw, size)
        self._t_last = max(self._t_last, int(t[-1]))
        return len(rec)

    def _drain_guard(self) -> int:
        """r1guard's own cycle records (session 5): written as they are; the flight recorder keeps them too."""
        raw = self.view.pop_guard_telemetry()
        if not raw:
            return 0
        rec = np.frombuffer(raw, dtype=R.GUARD)
        t = rec["t_ns"].astype(np.int64)
        seq = rec["seq"]
        f = self.log.current()
        size = R.GUARD.itemsize
        for i in range(len(rec)):
            f.write("guard", int(t[i]), raw[i * size : (i + 1) * size], int(seq[i]))
        self.counts["guard"].update(seq)
        if self.flight is not None:
            self.flight.add("guard", t, seq, raw, size)
        self._t_last = max(self._t_last, int(t[-1]))
        return len(rec)

    def _sample_seqlocks(self) -> None:
        for stream, getter, attr, dt in (
            ("hands_contact", self.view.hands_contact, "_hands_writes", R.HANDS_CONTACT),
            ("guard_heartbeat", self.view.guard_heartbeat, "_guard_writes", R.HEARTBEAT),
        ):
            got = getter()
            if got is None:
                continue
            writes, data = int(got[0]), bytes(got[1])
            if writes == getattr(self, attr):
                continue
            setattr(self, attr, writes)
            rec = np.frombuffer(data, dtype=dt)
            t = int(rec["t_ns"][0])
            self.log.current().write(stream, t, data, int(rec["seq"][0]))
            self.sampled[stream] += 1
            if self.flight is not None:
                self.flight.add(stream, np.array([t], np.int64), rec["seq"].copy(), data, dt.itemsize)

    # ---------------------------------------------------------------- events and triggers
    def _scan_telemetry(self, rec: np.ndarray[Any, Any], t: np.ndarray[Any, Any]) -> None:
        faults = rec["faults"].astype(np.int64)
        prev = np.concatenate([[self._last_faults], faults[:-1]])
        rising = faults & ~prev
        got_state = np.asarray(rec["t_state_rx_ns"]) != 0
        first_state = int(np.argmax(got_state)) if bool(got_state.any()) else len(rec)
        if self._state_seen:
            first_state = 0
        for i in np.nonzero(rising)[0]:
            names = R.fault_names(int(rising[i]))
            self._event(int(t[i]), {"kind": "fault", "rising": names, "faults": R.fault_names(int(faults[i])),
                                    "seq": int(rec["seq"][i])})  # fmt: skip
            hit = int(rising[i]) & self.cfg.trigger_faults
            if i < first_state:  # no lowstate received yet: the start-up state, not a fault worth a dump (#33)
                hit &= ~STARTUP_MASKED_FAULTS
            if hit:
                self._trigger("fault:" + "+".join(R.fault_names(hit)), int(t[i]), {"seq": int(rec["seq"][i])})
        falling = prev & ~faults
        for i in np.nonzero(falling)[0]:
            self._event(int(t[i]), {"kind": "fault_clear", "cleared": R.fault_names(int(falling[i])),
                                    "seq": int(rec["seq"][i])})  # fmt: skip
        self._last_faults = int(faults[-1])
        self._state_seen = self._state_seen or first_state < len(rec)

        for key, last_attr, labels in (("mode", "_last_mode", R.MODES), ("token_owner", "_last_owner", R.OWNERS)):
            vals = rec[key].astype(np.int64)
            last = getattr(self, last_attr)
            prevv = np.concatenate([[vals[0] if last is None else last], vals[:-1]])
            for i in np.nonzero(vals != prevv)[0]:
                old, new = int(prevv[i]), int(vals[i])
                self._event(int(t[i]), {"kind": key, "from": labels.get(old, old), "to": labels.get(new, new),
                                        "seq": int(rec["seq"][i])})  # fmt: skip
                if key == "mode" and new in self.cfg.trigger_modes:
                    self._trigger(f"mode:{labels.get(new, new)}", int(t[i]), {"from": labels.get(old, old)})
                if key == "token_owner" and new == 2:
                    self._trigger("takeover", int(t[i]), {"from": labels.get(old, old), "to": "r1guard"})
            setattr(self, last_attr, int(vals[-1]))

        tilt = R.tilt_deg(rec["imu_quat"])
        if np.any(tilt > self.cfg.tilt_trigger_deg) or not self._tilt_armed:
            for j in range(len(tilt)):
                if self._tilt_armed and tilt[j] > self.cfg.tilt_trigger_deg:
                    self._tilt_armed = False
                    self._trigger("tilt", int(t[j]), {"tilt_deg": round(float(tilt[j]), 2), "seq": int(rec["seq"][j])})
                elif not self._tilt_armed and tilt[j] < self.cfg.tilt_rearm_deg:
                    self._tilt_armed = True

    def _check_stops(self) -> None:
        for lvl, (req, t_req, who) in enumerate(self.view.stop_requests()):
            req = int(req)
            if req > self._stops[lvl]:
                self._stops[lvl] = req
                name = R.STOP_LEVELS[lvl]
                t = int(t_req) if int(t_req) > 0 else self._t_last
                self._event(t, {"kind": "stop_request", "level": name, "requests": req, "requester": int(who)})
                self._trigger(f"operator:{name}", t, {"requester": int(who)}, category="operator")

    def request_dump(self, reason: str = "operator", detail: dict[str, Any] | None = None) -> bool:
        """An explicit dump (the CLI's SIGUSR1, r1d's request): around the newest record time."""
        self._event(self._t_last, {"kind": "dump_request", "reason": reason})
        return self._trigger(f"request:{reason}", self._t_last, detail, category="request")

    def _trigger(
        self, reason: str, t_ns: int, detail: dict[str, Any] | None = None, category: str | None = None
    ) -> bool:
        if self.flight is None:
            return False
        return self.flight.trigger(reason, t_ns, detail, category=category)

    def _event(self, t_ns: int, obj: dict[str, Any]) -> None:
        obj = {"t_ns": int(t_ns), **obj}
        self.events += 1
        self._write_json("event", t_ns, obj)

    def _write_json(self, stream: str, t_ns: int, obj: dict[str, Any]) -> None:
        self.log.current().write_json(stream, t_ns, obj)
        if self.flight is not None:
            self.flight.add_json(stream, t_ns, obj)

    # ---------------------------------------------------------------- status and shutdown
    def stats(self) -> dict[str, Any]:
        cur = self.log.open_file
        return {
            "t_ns": self._t_last,
            "telemetry": {**self.counts["telemetry"].as_dict(), "ring_drops": self.ring_drops.get("telemetry", 0)},
            "policy": {**self.counts["policy"].as_dict(), "ring_drops": self.ring_drops.get("policy", 0)},
            "guard": {**self.counts["guard"].as_dict(), "ring_drops": self.ring_drops.get("guard_telemetry", 0)},
            "sampled": dict(self.sampled),
            "events": self.events,
            "file": None if cur is None else cur.path.name,
            "file_bytes": 0 if cur is None else cur.size(),
            "files": len(self.log.files),
            "flight_dumps": 0 if self.flight is None else len(self.flight.dumps),
            "flight_suppressed": 0 if self.flight is None else self.flight.suppressed,
            "polls": self.polls,
        }

    def _close_meta(self) -> dict[str, str]:
        s = self.stats()
        return {
            "telemetry_records": str(s["telemetry"]["records"]),
            "telemetry_ring_drops": str(s["telemetry"]["ring_drops"]),
            "telemetry_seq_gaps": str(s["telemetry"]["seq_gaps"]),
            "policy_records": str(s["policy"]["records"]),
            "policy_ring_drops": str(s["policy"]["ring_drops"]),
            "policy_seq_gaps": str(s["policy"]["seq_gaps"]),
            "guard_records": str(s["guard"]["records"]),
            "guard_ring_drops": str(s["guard"]["ring_drops"]),
            "guard_seq_gaps": str(s["guard"]["seq_gaps"]),
            "closed_utc": utc_stamp(),
        }

    def close(self) -> dict[str, Any]:
        """Final drain, the pending flight dump, close the files. Returns the final stats."""
        if self.closed:
            return self.stats()
        self.poll()
        if self.flight is not None:
            self.flight.flush()
        out = self.stats()
        out["files"] = [str(p) for p in self.log.files]
        out["flight_files"] = [] if self.flight is None else [str(p) for p in self.flight.dumps]
        self.log.close(self._close_meta())
        self.closed = True
        return out


def run_forever(logger: Logger, *, period_s: float = 0.02, duration_s: float | None = None,
                stop: Any = None) -> dict[str, Any]:  # fmt: skip
    """The process loop (CLI): poll every period_s of wall time until duration_s or stop() is true."""
    t_end = None if duration_s is None else time.monotonic() + duration_s
    while True:
        logger.poll()
        if stop is not None and stop():
            break
        if t_end is not None and time.monotonic() >= t_end:
            break
        time.sleep(period_s)
    return logger.close()
