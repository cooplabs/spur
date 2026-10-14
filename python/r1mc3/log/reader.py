"""Read r1log's MCAP files back: struct streams as numpy structured arrays (dtype rebuilt from each file's own schema),
JSON streams as lists of dicts, metadata as dicts. Used by the tests, by `python -m r1mc3.log info` and for analysis.

A file cut short (r1log killed, power lost: no summary, no footer, maybe a partial last chunk) is read sequentially up
to its last complete record and flagged `truncated` - the indexed reader cannot open it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from mcap.exceptions import McapError
from mcap.reader import make_reader
from mcap.records import Channel, Message, Metadata, Schema
from mcap.stream_reader import StreamReader

from . import records as R


@dataclass
class LogData:
    structs: dict[str, np.ndarray[Any, Any]] = field(default_factory=dict)  # topic -> records (file order)
    log_time: dict[str, np.ndarray[Any, Any]] = field(default_factory=dict)  # topic -> MCAP log_time (ns)
    json: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # topic -> objects
    metadata: dict[str, dict[str, str]] = field(default_factory=dict)  # name -> data (last of each name)
    schemas: dict[str, str] = field(default_factory=dict)  # topic -> schema name
    truncated: bool = False  # no summary / footer: read sequentially up to the last complete record

    def topic(self, name: str) -> np.ndarray[Any, Any]:
        return self.structs.get(name, np.zeros(0))


def read(path: str | Path, *, log_time_order: bool = False) -> LogData:
    """Every message of one file. log_time_order=False keeps the order r1log wrote them in (= the ring order)."""
    out = LogData()
    raw: dict[str, list[bytes]] = {}
    times: dict[str, list[int]] = {}
    dtypes: dict[str, np.dtype[Any]] = {}

    def take(schema: Schema | None, channel: Channel, msg: Message) -> None:
        topic = channel.topic
        if schema is not None:
            out.schemas[topic] = schema.name
        if schema is not None and schema.encoding == R.STRUCT_ENCODING:
            if topic not in dtypes:
                dtypes[topic] = R.dtype_from_schema(schema.data)
            raw.setdefault(topic, []).append(msg.data)
            times.setdefault(topic, []).append(msg.log_time)
        else:
            out.json.setdefault(topic, []).append(json.loads(msg.data))

    try:
        with open(path, "rb") as f:
            reader = make_reader(f)
            for schema, channel, msg in reader.iter_messages(log_time_order=log_time_order):
                take(schema, channel, msg)
            f.seek(0)
            reader = make_reader(f)
            for md in reader.iter_metadata():
                out.metadata[md.name] = dict(md.metadata)
    except McapError:
        out = LogData(truncated=True)
        raw.clear()
        times.clear()
        _read_sequential(path, out, take)
    for topic, chunks in raw.items():
        out.structs[topic] = np.frombuffer(b"".join(chunks), dtype=dtypes[topic])
        out.log_time[topic] = np.asarray(times[topic], dtype=np.int64)
    return out


def _read_sequential(path: str | Path, out: LogData, take: Any) -> None:
    """A file without its summary: walk the records (chunks decompressed) until the data ends."""
    schemas: dict[int, Schema] = {}
    channels: dict[int, Channel] = {}
    with open(path, "rb") as f:
        try:
            for rec in StreamReader(f, skip_magic=False).records:
                if isinstance(rec, Schema):
                    schemas[rec.id] = rec
                elif isinstance(rec, Channel):
                    channels[rec.id] = rec
                elif isinstance(rec, Message):
                    ch = channels[rec.channel_id]
                    take(schemas.get(ch.schema_id), ch, rec)
                elif isinstance(rec, Metadata):
                    out.metadata[rec.name] = dict(rec.metadata)
        except McapError:
            pass  # the cut: everything before it has been read


def info(path: str | Path) -> dict[str, Any]:
    """A summary: per topic the count and record-time span; the metadata; the summary statistics."""
    d = read(path)
    topics: dict[str, Any] = {}
    for topic, rec in d.structs.items():
        lt = d.log_time[topic]
        topics[topic] = {"records": len(rec), "t0_s": float(lt.min()) * 1e-9, "t1_s": float(lt.max()) * 1e-9}
        if "seq" in (rec.dtype.names or ()):
            s = rec["seq"].astype(np.int64)
            topics[topic]["seq_gaps"] = int(np.sum(np.maximum(np.diff(s) - 1, 0))) if len(s) > 1 else 0
    for topic, objs in d.json.items():
        topics[topic] = {"messages": len(objs)}
    return {"path": str(path), "bytes": Path(path).stat().st_size, "truncated": d.truncated, "topics": topics,
            "metadata": d.metadata}  # fmt: skip
