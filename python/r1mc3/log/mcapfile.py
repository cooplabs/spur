"""MCAP files with r1log's channels, and the rotation of the session log (DESIGN §9: MCAP with zstd chunks).

One channel per stream. Struct streams carry the raw C record (schema + message encoding "r1rt-struct", see
records.py); r1log's own streams (events, stats, flight-recorder triggers) are JSON with a JSON schema, which MCAP
viewers show as they are. Every file starts with a metadata record "r1log" (the clock: the record times are
CLOCK_MONOTONIC ns - in lockstep the simulated time - with the wall clock read at file start next to it, the layout /
ABI versions, the source segment) and ends with "r1log_close" (counts, drops, the reason it closed).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from mcap.writer import CompressionType, Writer

from . import records as R

JSON_ENCODING = "json"
SCHEMA_JSONSCHEMA = "jsonschema"


@dataclass(frozen=True)
class Stream:
    """One MCAP channel. struct = a types.h record name (raw C bytes), or None for a JSON stream."""

    topic: str
    struct: str | None = None
    title: str = ""


STREAMS: dict[str, Stream] = {
    "telemetry": Stream("/r1rt/telemetry", "r1rt_telemetry_record", "r1rt writer tick records (500 Hz)"),
    "policy": Stream("/r1rt/policy", "r1rt_policy_record", "policy steps (50 Hz, thread or inline)"),
    "hands_contact": Stream("/r1hands/contact", "r1rt_hands_contact", "r1hands contact summary (sampled)"),
    "guard_heartbeat": Stream("/r1guard/heartbeat", "r1rt_heartbeat", "r1guard heartbeat (sampled)"),
    # layout 2 (session 5): every r1guard cycle from its own ring (1 kHz, standby and active)
    "guard": Stream("/r1guard/telemetry", "r1rt_guard_record", "r1guard cycle records (1 kHz)"),
    "event": Stream("/r1log/event", None, "mode / token / fault / stop changes seen by r1log"),
    "stats": Stream("/r1log/stats", None, "r1log counters (1 Hz)"),
    "trigger": Stream("/r1log/trigger", None, "flight-recorder triggers (dump files only)"),
}

_JSON_SCHEMA = json.dumps({"type": "object", "additionalProperties": True}).encode()


def utc_stamp(t: float | None = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if t is None else t))


class McapFile:
    """One open MCAP file with every stream's channel registered (registration is cheap; empty channels cost bytes,
    not records)."""

    def __init__(
        self,
        path: Path,
        *,
        metadata: dict[str, str],
        chunk_size: int = 1 << 20,
        compression: CompressionType = CompressionType.ZSTD,
        abi_version: int | None = None,
    ) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f: IO[bytes] = open(path, "wb")  # closed in close()
        self._w = Writer(self._f, chunk_size=chunk_size, compression=compression)
        self._w.start(profile="", library="r1mc3.log (r1log)")
        self._ch: dict[str, int] = {}
        for name, s in STREAMS.items():
            if s.struct is not None:
                sid = self._w.register_schema(
                    s.struct, R.STRUCT_ENCODING, R.schema_json(s.struct, abi_version=abi_version)
                )
                enc = R.STRUCT_ENCODING
            else:
                sid = self._w.register_schema(f"r1log.{name}", SCHEMA_JSONSCHEMA, _JSON_SCHEMA)
                enc = JSON_ENCODING
            self._ch[name] = self._w.register_channel(s.topic, enc, sid, {"title": s.title})
        self._w.add_metadata("r1log", metadata)
        self.counts: dict[str, int] = {k: 0 for k in STREAMS}
        self.t_first: int | None = None
        self.t_last: int | None = None
        self._json_seq = 0
        self.closed = False

    def write(self, stream: str, t_ns: int, data: bytes, seq: int = 0) -> None:
        t = max(0, int(t_ns))
        self._w.add_message(self._ch[stream], log_time=t, data=data, publish_time=t, sequence=int(seq) & 0xFFFFFFFF)
        self.counts[stream] += 1
        if self.t_first is None or t < self.t_first:
            self.t_first = t
        if self.t_last is None or t > self.t_last:
            self.t_last = t

    def write_json(self, stream: str, t_ns: int, obj: dict[str, Any]) -> None:
        self._json_seq += 1
        self.write(stream, t_ns, json.dumps(obj, separators=(",", ":"), default=_json_default).encode(), self._json_seq)

    def size(self) -> int:
        """Bytes on disk so far (closed chunks; the open chunk is in memory)."""
        return self._f.tell()

    def close(self, metadata: dict[str, str] | None = None) -> None:
        if self.closed:
            return
        if metadata:
            self._w.add_metadata("r1log_close", metadata)
        self._w.finish()  # type: ignore[no-untyped-call]
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()
        self.closed = True


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON-able: {type(o)}")


@dataclass
class RotatingLog:
    """The session log: <out_dir>/<prefix>_<UTC start>_<NNN>.mcap, a new file when the current one passes max_bytes
    on disk or spans max_span_s of record time (so a crash loses at most one file's tail, and files stay copyable)."""

    out_dir: Path
    metadata: dict[str, str]
    prefix: str = "r1log"
    max_bytes: int = 512 << 20
    max_span_s: float = 600.0
    chunk_size: int = 1 << 20
    stamp: str = field(default_factory=utc_stamp)
    abi_version: int | None = None
    files: list[Path] = field(default_factory=list)
    _cur: McapFile | None = None
    _index: int = 0

    def current(self) -> McapFile:
        if self._cur is None:
            path = self.out_dir / f"{self.prefix}_{self.stamp}_{self._index:03d}.mcap"
            meta = dict(self.metadata)
            meta.update(
                file_index=str(self._index), wall_utc_at_open=utc_stamp(), monotonic_ns_at_open=str(time.monotonic_ns())
            )
            self._cur = McapFile(path, metadata=meta, chunk_size=self.chunk_size, abi_version=self.abi_version)
            self.files.append(path)
            self._index += 1
        return self._cur

    @property
    def open_file(self) -> McapFile | None:
        return self._cur

    def maybe_rotate(self, close_meta: dict[str, str] | None = None) -> bool:
        cur = self._cur
        if cur is None:
            return False
        span = 0.0 if cur.t_first is None or cur.t_last is None else (cur.t_last - cur.t_first) * 1e-9
        if cur.size() >= self.max_bytes or span >= self.max_span_s:
            meta = dict(close_meta or {})
            meta["closed_by"] = "rotation"
            cur.close(meta)
            self._cur = None
            return True
        return False

    def close(self, close_meta: dict[str, str] | None = None) -> None:
        if self._cur is not None:
            meta = dict(close_meta or {})
            meta.setdefault("closed_by", "close")
            self._cur.close(meta)
            self._cur = None
