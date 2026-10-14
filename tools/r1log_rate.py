"""Measure r1log's file rate and its own CPU cost on the v3 core in lockstep (real profile, the default scene).

    uv run python tools/r1log_rate.py [--seconds 20] [--out DIR]

Two cases: ideal sensors (the simulator's default: smooth signals, compresses too well) and the S2 sensor model on
(encoder / IMU / tau noise, closer to a robot). Reports per case: records, raw bytes, file bytes, the compression
ratio, MB per minute and GB per hour of robot time, and r1log's own CPU per second of robot time (the poll calls
only; the Mac's numbers, not the Orin's). The numbers go to docs/architecture/LOGGER.md (DESIGN §9 estimated
~0.5 GB/h).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from r1mc3.log import Logger, LoggerConfig, read
from r1mc3.log._native import native
from r1sim import world
from r1sim.engine import LowState, Simulation
from r1sim.sensors import SensorConfig, SensorModel
from r1sim.v3_core import V3CoreController


def measure(seconds: float, out_dir: Path, *, sensors: bool) -> dict[str, Any]:
    w = world.build_world("real", visual=False)
    sim = Simulation(w.model, w.index, sensors=SensorModel(SensorConfig.from_yaml(), seed=3) if sensors else None)
    sim.reset(w.initial_qpos())
    ctl = V3CoreController(w.model)
    sim.add_controller(ctl, ctl.period_s, "v3")
    lg = Logger(native().SegmentView.of_core(ctl.core), LoggerConfig(out_dir=out_dir, stamp="rate", flight=True))
    cpu = [0.0]

    def drain(st: LowState) -> None:
        if ctl.calls % 10 == 0:  # every 20 ms of robot time, as the CLI's default period
            t0 = time.process_time()
            lg.poll()
            cpu[0] += time.process_time() - t0

    sim.add_controller(drain, ctl.period_s, "r1log")
    sim.run_for(seconds)
    t0 = time.process_time()
    out = lg.close()
    cpu[0] += time.process_time() - t0
    files = [Path(f) for f in out["files"]]
    size = sum(f.stat().st_size for f in files)
    d = read(files[0])
    n_tel = sum(len(read(f).structs.get("/r1rt/telemetry", [])) for f in files)
    n_pol = sum(len(read(f).structs.get("/r1rt/policy", [])) for f in files)
    raw = n_tel * 1136 + n_pol * 600
    return {
        "sensors": "S2 sensor model (seed 3)" if sensors else "ideal",
        "robot_s": seconds,
        "telemetry_records": n_tel,
        "policy_records": n_pol,
        "raw_bytes": raw,
        "file_bytes": size,
        "compression": round(raw / size, 2),
        "MB_per_min": round(size / seconds * 60 / 1e6, 2),
        "GB_per_h": round(size / seconds * 3600 / 1e9, 3),
        "raw_GB_per_h": round(raw / seconds * 3600 / 1e9, 3),
        "r1log_cpu_per_robot_s_ms": round(cpu[0] / seconds * 1e3, 1),
        "drops": out["telemetry"]["ring_drops"] + out["policy"]["ring_drops"],
        "metadata_abi": d.metadata["r1log"]["abi_version"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    base = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="r1log_rate_"))
    rows = [measure(a.seconds, base / name, sensors=s) for name, s in (("ideal", False), ("sensors", True))]
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
