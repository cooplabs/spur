"""The recordings the HW1 fits read, in ONE in-memory form, and the loaders from the formats the robot side writes.

Formats (docs/hardware/HW1_FITS.md has the protocols that produce them):

* **Body** (the stiction ramps): r1log's MCAP, topic `/r1rt/telemetry` (500 Hz, the r1rt telemetry record:
  docs/architecture/LOGGER.md; `python/r1mc3/log/records.py`): t = `t_wake_ns`, and per joint (26, SDK order) q, dq,
  tau_est, q_cmd, dq_cmd, kp, kd, tau_ff. Or an .npz with the same names (t in seconds) - what the synthetic
  generators and tests write.
* **Hands** (pad presses, finger holds): MCAP topic `/r1hands/state` (JSON encoding, one message per hand per sample,
  100 Hz) with the fields of `HAND_FIELDS` - r1hands' full-rate ring does not exist yet (LOGGER.md "Not solid" 2), so
  the HW1 recording tool writes this topic itself; or an .npz with keys `<side>/<field>`.
* **Press notes** (what the operator did): a YAML list, one entry per press: pad, x_mm (along the finger from the pad's
  reference end, measured), weight_g (kitchen scale), t_start / t_end (s, the recording's clock).

Everything is converted to float64 numpy arrays; times start at the first sample (t[0] = 0).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

FloatArray = npt.NDArray[np.float64]

BODY_TOPIC = "/r1rt/telemetry"
HANDS_TOPIC = "/r1hands/state"
BODY_FIELDS: tuple[str, ...] = ("q", "dq", "tau_est", "q_cmd", "dq_cmd", "kp", "kd", "tau_ff")
#: per hand sample: t_ns (int, monotonic), side ("left" / "right"), and these arrays (7 motors in Dex3 DDS order;
#: temperature = the motor's two sensors (7, 2); pressure = the 9 pads x 12 slots, raw absolute Pa as HandState_ gives;
#: the command the hand was sent at that sample: q_cmd, kp, kd, tau_ff)
HAND_FIELDS: tuple[str, ...] = ("q", "dq", "tau_est", "temperature", "pressure", "q_cmd", "kp", "kd", "tau_ff")


@dataclass
class BodyRecording:
    """Full-rate body telemetry: t (n,), every field (n, 26)."""

    t: FloatArray
    fields: dict[str, FloatArray]

    def __getitem__(self, name: str) -> FloatArray:
        return self.fields[name]

    def joint(self, name: str, j: int) -> FloatArray:
        return np.asarray(self.fields[name][:, j], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, t=self.t, **self.fields)  # type: ignore[arg-type]


@dataclass
class HandRecording:
    """One hand's samples: t (n,), q / dq / tau_est / q_cmd / kp / kd / tau_ff (n, 7), temperature (n, 7, 2),
    pressure (n, 9, 12)."""

    side: str
    t: FloatArray
    fields: dict[str, FloatArray] = field(default_factory=dict)

    def __getitem__(self, name: str) -> FloatArray:
        return self.fields[name]


@dataclass(frozen=True)
class Press:
    """One press from the operator's notes: a known weight resting on `pad` at x_mm along it in [t_start, t_end]."""

    side: str
    pad: int
    x_mm: float
    weight_g: float
    t_start: float
    t_end: float

    @property
    def force_n(self) -> float:
        return self.weight_g * 1e-3 * 9.80665


def _t0(t: FloatArray) -> FloatArray:
    return np.asarray(t - t[0], dtype=np.float64) if t.size else t


# ------------------------------------------------------------------------------------------------- body
def body_from_arrays(t: npt.ArrayLike, fields: Mapping[str, npt.ArrayLike]) -> BodyRecording:
    tt = np.asarray(t, dtype=np.float64)
    out = {k: np.asarray(fields[k], dtype=np.float64) for k in BODY_FIELDS}
    for k, v in out.items():
        if v.shape != (tt.size, 26):
            raise ValueError(f"body field {k}: shape {v.shape}, expected ({tt.size}, 26)")
    return BodyRecording(_t0(tt), out)


def load_body(path: str | Path) -> BodyRecording:
    """An r1log MCAP file (topic /r1rt/telemetry) or an .npz (t in seconds + BODY_FIELDS)."""
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p) as z:
            return body_from_arrays(z["t"], {k: z[k] for k in BODY_FIELDS})
    from r1mc3.log import reader

    data = reader.read(p, log_time_order=True)
    rec = data.topic(BODY_TOPIC)
    if rec.size == 0:
        raise ValueError(f"{p}: no {BODY_TOPIC} records")
    t = rec["t_wake_ns"].astype(np.float64) * 1e-9
    return body_from_arrays(t, {k: rec[k] for k in BODY_FIELDS})


def write_body_mcap(path: str | Path, rec: BodyRecording, *, t0_ns: int = 1_000_000_000) -> None:
    """The recording as r1log would have written it (r1mc3.log.mcapfile, topic /r1rt/telemetry, the r1rt telemetry
    record of this build): what the synthetic tests feed load_body, so the loader is exercised on r1log's format."""
    from r1mc3.log import mcapfile
    from r1mc3.log import records as R

    recs = np.zeros(rec.t.size, dtype=R.TELEMETRY)
    recs["seq"] = np.arange(rec.t.size)
    recs["t_wake_ns"] = t0_ns + np.round(rec.t * 1e9).astype(np.int64)
    for k in BODY_FIELDS:
        recs[k] = rec.fields[k]
    f = mcapfile.McapFile(Path(path), metadata={"source": "r1sim.sysid synthetic"})
    for r in recs:
        f.write("telemetry", int(r["t_wake_ns"]), r.tobytes(), int(r["seq"]))
    f.close()


# ------------------------------------------------------------------------------------------------- hands
def hand_from_samples(side: str, samples: Sequence[Mapping[str, Any]]) -> HandRecording:
    rows = [s for s in samples if s.get("side", side) == side]
    if not rows:
        raise ValueError(f"no samples for the {side} hand")
    t = np.array([float(r["t_ns"]) * 1e-9 for r in rows])
    order = np.argsort(t, kind="stable")
    fields = {k: np.asarray([rows[i][k] for i in order], dtype=np.float64) for k in HAND_FIELDS if k in rows[0]}
    return HandRecording(side, _t0(t[order]), fields)


def load_hand(path: str | Path, side: str) -> HandRecording:
    """MCAP (topic /r1hands/state, JSON) or .npz (keys `t`, `<side>/<field>`; t in seconds)."""
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p) as z:
            fields = {k: np.asarray(z[f"{side}/{k}"], dtype=np.float64) for k in HAND_FIELDS if f"{side}/{k}" in z}
            return HandRecording(side, _t0(np.asarray(z["t"], dtype=np.float64)), fields)
    from mcap.reader import make_reader

    samples: list[dict[str, Any]] = []
    with open(p, "rb") as f:
        for _schema, _channel, msg in make_reader(f).iter_messages(topics=[HANDS_TOPIC], log_time_order=True):
            samples.append(json.loads(msg.data))
    return hand_from_samples(side, samples)


def save_hand_npz(path: str | Path, recs: Sequence[HandRecording]) -> None:
    if not recs:
        raise ValueError("no hand recordings")
    out: dict[str, FloatArray] = {"t": recs[0].t}
    for r in recs:
        if r.t.shape != recs[0].t.shape:
            raise ValueError("the hands' recordings must share one clock in an npz")
        out.update({f"{r.side}/{k}": v for k, v in r.fields.items()})
    np.savez_compressed(path, **out)  # type: ignore[arg-type]


def write_hands_mcap(path: str | Path, recs: Sequence[HandRecording]) -> None:
    """The reference writer of the /r1hands/state topic (the HW1 recording tool writes the same JSON)."""
    from mcap.writer import Writer

    with open(path, "wb") as f:
        w = Writer(f)
        w.start(profile="", library="r1sim.sysid")
        sid = w.register_schema(name="r1hands_state", encoding="jsonschema", data=json.dumps(hand_schema()).encode())
        cid = w.register_channel(topic=HANDS_TOPIC, message_encoding="json", schema_id=sid)
        events = sorted((float(t), r, i) for r in recs for i, t in enumerate(r.t))
        for seq, (t, r, i) in enumerate(events):
            msg = {"t_ns": round(t * 1e9), "side": r.side}
            msg.update({k: v[i].tolist() for k, v in r.fields.items()})
            ns = round(t * 1e9)
            w.add_message(channel_id=cid, log_time=ns, publish_time=ns, data=json.dumps(msg).encode(), sequence=seq)
        w.finish()  # type: ignore[no-untyped-call]


def hand_schema() -> dict[str, Any]:
    arr = {"type": "array"}
    props: dict[str, Any] = {"t_ns": {"type": "integer"}, "side": {"enum": ["left", "right"]}}
    props.update({k: arr for k in HAND_FIELDS})
    return {"title": "r1hands_state", "type": "object", "properties": props, "required": ["t_ns", "side"]}


# ------------------------------------------------------------------------------------------------- press notes
def load_presses(path: str | Path) -> list[Press]:
    rows = yaml.safe_load(Path(path).read_text())
    return [
        Press(str(r["side"]), int(r["pad"]), float(r["x_mm"]), float(r["weight_g"]), float(r["t_start"]),
              float(r["t_end"]))
        for r in rows
    ]  # fmt: skip


def save_presses(path: str | Path, presses: Sequence[Press]) -> None:
    rows = [
        {"side": p.side, "pad": p.pad, "x_mm": p.x_mm, "weight_g": p.weight_g, "t_start": p.t_start, "t_end": p.t_end}
        for p in presses
    ]
    Path(path).write_text(yaml.safe_dump(rows, sort_keys=False))
