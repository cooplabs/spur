"""The flight recorder (DESIGN §9): the last 30 s of every stream kept in memory, written to its own MCAP file when
something happens - a fault bit rises, a takeover / mode change to GUARDED, STAND_FREEZE or DAMP, a large tilt, an
operator stop or an explicit request.

Why in memory and separate from the session log: the session log can be rotated, lost with the backpack or simply
too long to find the moment in; a dump is a small self-contained file around the event, written at once (9 Oct: the
fall's evidence had to be pieced together from 8 Hz rows). A dump holds [trigger - window_s, trigger + post_s] of
every stream (the seconds after a trigger matter: a fall develops after its first fault), plus every trigger that
arrived meanwhile on the "/r1log/trigger" channel. A reason that fired is held off for holdoff_s (a fault bit that
stays set must not dump every tick); suppressed triggers are counted.

Records are buffered as the chunks r1log drained (raw bytes + their times), so buffering costs one reference per
drain, not one object per record.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .mcapfile import McapFile, utc_stamp


@dataclass
class Trigger:
    reason: str
    t_ns: int
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "t_ns": self.t_ns, **self.detail}


@dataclass
class _Chunk:
    t: npt.NDArray[np.int64]
    seq: npt.NDArray[np.uint64]
    data: bytes
    itemsize: int


@dataclass
class _Pending:
    t_trigger: int
    triggers: list[Trigger]


class FlightRecorder:
    def __init__(
        self,
        out_dir: Path,
        *,
        window_s: float = 30.0,
        post_s: float = 5.0,
        holdoff_s: float = 10.0,
        prefix: str = "flight",
        metadata: dict[str, str] | None = None,
        abi_version: int | None = None,
    ) -> None:
        if window_s <= 0 or post_s < 0 or holdoff_s < 0:
            raise ValueError("window_s > 0, post_s >= 0, holdoff_s >= 0")
        self.out_dir = Path(out_dir)
        self.window_ns = round(window_s * 1e9)
        self.post_ns = round(post_s * 1e9)
        self.holdoff_ns = round(holdoff_s * 1e9)
        self.prefix = prefix
        self.metadata = dict(metadata or {})
        self.abi_version = abi_version
        self._struct: dict[str, deque[_Chunk]] = {}
        self._json: dict[str, deque[tuple[int, dict[str, Any]]]] = {}
        self._newest = -(1 << 62)
        self._pending: _Pending | None = None
        self._last_fired: dict[str, int] = {}
        self.dumps: list[Path] = []
        self.suppressed = 0
        self.triggers: list[Trigger] = []

    # ---------------------------------------------------------------- buffering
    @property
    def keep_ns(self) -> int:
        return self.window_ns + self.post_ns

    def add(
        self, stream: str, t: npt.NDArray[np.int64], seq: npt.NDArray[np.uint64], data: bytes, itemsize: int
    ) -> None:
        """A drained block of struct records (n = len(t), data = n * itemsize bytes in record order)."""
        if len(t) == 0:
            return
        if len(data) != len(t) * itemsize:
            raise ValueError("flight: data length does not match the record count")
        self._struct.setdefault(stream, deque()).append(_Chunk(np.asarray(t, np.int64), np.asarray(seq, np.uint64),
                                                               data, itemsize))  # fmt: skip
        self._advance(int(np.max(t)))

    def add_json(self, stream: str, t_ns: int, obj: dict[str, Any]) -> None:
        self._json.setdefault(stream, deque()).append((int(t_ns), obj))
        self._advance(int(t_ns))

    def _advance(self, t_ns: int) -> None:
        if t_ns <= self._newest:
            return
        self._newest = t_ns
        horizon = t_ns - self.keep_ns
        if self._pending is not None:  # never drop what a pending dump still needs
            horizon = min(horizon, self._pending.t_trigger - self.window_ns)
        for dq in self._struct.values():
            while dq and int(dq[0].t[-1]) < horizon:
                dq.popleft()
        for jq in self._json.values():
            while jq and jq[0][0] < horizon:
                jq.popleft()

    def buffered_s(self) -> float:
        """Record time covered by the buffer (diagnostics)."""
        firsts = [int(dq[0].t[0]) for dq in self._struct.values() if dq]
        return 0.0 if not firsts else (self._newest - min(firsts)) * 1e-9

    # ---------------------------------------------------------------- triggers
    def trigger(
        self, reason: str, t_ns: int, detail: dict[str, Any] | None = None, *, category: str | None = None
    ) -> bool:
        """Ask for a dump around t_ns. Joins a pending dump; otherwise held off when the same category fired within
        holdoff_s. True = recorded in a dump."""
        cat = category or reason.split(":", 1)[0]
        trig = Trigger(reason, int(t_ns), dict(detail or {}))
        if self._pending is not None:
            self._pending.triggers.append(trig)
            self.triggers.append(trig)
            self._last_fired[cat] = int(t_ns)
            return True
        last = self._last_fired.get(cat)
        if last is not None and int(t_ns) - last < self.holdoff_ns:
            self.suppressed += 1
            return False
        self._last_fired[cat] = int(t_ns)
        self._pending = _Pending(int(t_ns), [trig])
        self.triggers.append(trig)
        return True

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def poll(self, now_ns: int | None = None) -> list[Path]:
        """Write the pending dump once its post window has passed in record time (now_ns defaults to the newest
        record seen)."""
        now = self._newest if now_ns is None else int(now_ns)
        p = self._pending
        if p is not None and now >= p.t_trigger + self.post_ns:
            return [self._dump(p, closed_by="post window")]
        return []

    def flush(self) -> list[Path]:
        """Write a pending dump now (at shutdown: whatever of the post window exists)."""
        p = self._pending
        return [self._dump(p, closed_by="flush")] if p is not None else []

    def _dump(self, p: _Pending, *, closed_by: str) -> Path:
        self._pending = None
        t0, t1 = p.t_trigger - self.window_ns, p.t_trigger + self.post_ns
        reason = p.triggers[0].reason.replace(":", "-").replace("/", "-")
        path = self.out_dir / f"{self.prefix}_{utc_stamp()}_{len(self.dumps):03d}_{reason}.mcap"
        meta = dict(self.metadata)
        meta.update(t_trigger_ns=str(p.t_trigger), window_s=str(self.window_ns * 1e-9), post_s=str(self.post_ns * 1e-9),
                    reasons=",".join(tr.reason for tr in p.triggers))  # fmt: skip
        f = McapFile(path, metadata=meta, abi_version=self.abi_version)
        n_rec = 0
        for stream, dq in self._struct.items():
            for ch in dq:
                sel = np.nonzero((ch.t >= t0) & (ch.t <= t1))[0]
                for i in sel:
                    f.write(stream, int(ch.t[i]), ch.data[i * ch.itemsize : (i + 1) * ch.itemsize], int(ch.seq[i]))
                n_rec += len(sel)
        for stream, jq in self._json.items():
            for t, obj in jq:
                if t0 <= t <= t1:
                    f.write_json(stream, t, obj)
        for tr in p.triggers:
            f.write_json("trigger", tr.t_ns, tr.as_dict())
        f.close({"records": str(n_rec), "triggers": str(len(p.triggers)), "closed_by": closed_by})
        self.dumps.append(path)
        return path
