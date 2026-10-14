"""HW1 recordings for the Tier B checks (DESIGN §13.2): one record window = a `r1 record` manifest + the r1log
telemetry around it, in seconds from the manifest's start, every body field (n, 26) plus the IMU.

The manifests (`records/*.json`, schema r1mc3.record_manifest/1) name their r1log files by the BACKPACK's path
(/var/lib/r1mc3/logs/...); the Mac copies keep the file names (results/hw1_20261010/end/session_part3/r1log), so the
loader looks the names up in the folder it is given. A window's samples are r1rt's telemetry at its 500 Hz writer
tick: `q` etc. = the lowstate r1rt held at that tick (age p50 0.49 ms, HW1 T5), `q_cmd` / `kp` / ... = the command it
sent at that tick.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

BODY_TOPIC = "/r1rt/telemetry"
FIELDS: tuple[str, ...] = ("q", "dq", "tau_est", "q_cmd", "dq_cmd", "kp", "kd", "tau_ff")
IMU_FIELDS: tuple[str, ...] = ("imu_quat", "imu_gyro", "imu_acc")


@dataclass
class Window:
    """One recording window: t (n,) in s from the manifest's t_start_mono_ns; fields (n, 26); imu fields (n, 4 / 3)."""

    name: str
    kind: str  # step / chirp / ramp / static_pose ...
    joint: str | None  # the plan's joint (step / chirp / ramp), else None
    params: dict[str, Any]
    t: FloatArray
    fields: dict[str, FloatArray]
    imu: dict[str, FloatArray] = field(default_factory=dict)
    t_start_ns: int = 0
    duration_s: float = 0.0

    def j(self, name: str, joint: int) -> FloatArray:
        return np.asarray(self.fields[name][:, joint], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        meta = dict(name=self.name, kind=self.kind, joint=self.joint, params=self.params, t_start_ns=self.t_start_ns,
                    duration_s=self.duration_s)  # fmt: skip
        np.savez_compressed(path, t=self.t, meta=json.dumps(meta), **self.fields, **self.imu)  # type: ignore[arg-type]

    @classmethod
    def load(cls, path: str | Path) -> Window:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            fields = {k: np.asarray(z[k], dtype=np.float64) for k in FIELDS if k in z}
            imu = {k: np.asarray(z[k], dtype=np.float64) for k in IMU_FIELDS if k in z}
            return cls(meta["name"], meta["kind"], meta["joint"], meta["params"], np.asarray(z["t"], dtype=np.float64),
                       fields, imu, int(meta["t_start_ns"]), float(meta["duration_s"]))  # fmt: skip


def read_telemetry(paths: list[Path]) -> tuple[npt.NDArray[np.int64], dict[str, FloatArray]]:
    """t_wake_ns + every body / IMU field of these r1log files, merged in time order."""
    from r1mc3.log import reader

    ts: list[npt.NDArray[np.int64]] = []
    cols: dict[str, list[FloatArray]] = {k: [] for k in FIELDS + IMU_FIELDS}
    for p in paths:
        rec = reader.read(p, log_time_order=True).topic(BODY_TOPIC)
        if rec.size == 0:
            continue
        ts.append(rec["t_wake_ns"].astype(np.int64))
        for k in cols:
            cols[k].append(np.asarray(rec[k], dtype=np.float64))
    if not ts:
        raise ValueError(f"no {BODY_TOPIC} records in {paths}")
    t = np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    return t[order], {k: np.concatenate(v)[order] for k, v in cols.items()}


def cut(manifest: dict[str, Any], name: str, t_ns: npt.NDArray[np.int64], cols: dict[str, FloatArray], *,
        pad_s: float = 1.0) -> Window:  # fmt: skip
    """The manifest's window (+- pad_s) out of merged telemetry."""
    t0 = int(manifest["t_start_mono_ns"])
    t1 = int(manifest.get("t_end_mono_ns") or t0 + float(manifest["plan"]["duration_s"]) * 1e9)
    sel = (t_ns >= t0 - pad_s * 1e9) & (t_ns <= t1 + pad_s * 1e9)
    if not np.any(sel):
        raise ValueError(f"{name}: no telemetry inside its window")
    plan = manifest["plan"]
    params = dict(plan.get("params") or {})
    return Window(
        name=name,
        kind=str(plan["kind"]),
        joint=params.get("joint"),
        params=params,
        t=np.asarray((t_ns[sel] - t0) * 1e-9, dtype=np.float64),
        fields={k: cols[k][sel] for k in FIELDS},
        imu={k: cols[k][sel] for k in IMU_FIELDS},
        t_start_ns=t0,
        duration_s=(t1 - t0) * 1e-9,
    )


def load_session(session_dir: str | Path, kinds: tuple[str, ...] = ("step", "chirp"), *, pad_s: float = 1.0
                 ) -> list[Window]:  # fmt: skip
    """Every record of these kinds in a verified HW1 session copy (records/*.json + r1log/*.mcap); each MCAP is read
    once."""
    d = Path(session_dir)
    mans: list[tuple[str, dict[str, Any]]] = []
    for p in sorted((d / "records").glob("*.json")):
        m = json.loads(p.read_text())
        if m.get("schema") == "r1mc3.record_manifest/1" and m["plan"]["kind"] in kinds and m.get("status") == "done":
            mans.append((p.stem, m))
    files = sorted({Path(f).name for _, m in mans for f in (m.get("r1log_files") or [])})
    t_ns, cols = read_telemetry([d / "r1log" / f for f in files])
    return [cut(m, n, t_ns, cols, pad_s=pad_s) for n, m in mans]
