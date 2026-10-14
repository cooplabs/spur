"""A robustness batch for the balance controller: many simulated robots, pushed and given faults, the real C++ core
balancing each one; a table of falls and tilts.

    bash scripts/run.sh python examples/robustness_batch.py          # 8 robots x 3 pushes x 3 fault cases (~4 min)
    bash scripts/run.sh python examples/robustness_batch.py --quick  # 2 robots, one push size, no fault (~10 s)
    bash scripts/run.sh python examples/robustness_batch.py --seeds 50 --impulses 4 8 12 --faults none cmd_gap \\
        --workers 4 --json batch.json

Each run, in simulated time (lockstep: reproducible bit for bit for a seed):

* **the robot** = one draw of the simulator's robot model within its parameter ranges (`r1sim.variants`: masses and
  centres of mass, the IMU's mounting, the hands' mounting, foot / floor friction, and each joint's friction, armature
  and damping around the values identified on a real R1 on its stand); the motor board as identified (it holds the
  last command ~0.4 s, then damps) and the sensor-noise model, both of the same seed;
* **the controller** = `r1rt`'s own C++ core (`r1rt_py`, the same code as the robot's real-time process) running the
  balance policy at 500 Hz, from the stand pose, no walking command;
* **the fault** (optional, at t = 2 s): `cmd_gap` - the board receives no commands for 0.3 s; `state_delay` - the
  robot's state arrives 40 ms old for 2 s (the first push lands inside that window); `policy_nan` - the policy's
  output is NaN for 0.2 s (the core's answer: STAND_FREEZE, a stiff hold of the stand pose - it does not balance any
  more, and the pushes test that);
* **the pushes**: four horizontal pushes on the torso, forward at 3 s, back at 5 s, left at 7 s, right at 9 s, each an
  impulse of `--impulses` N s over 0.2 s (5 N s changes the 32 kg robot's speed by ~0.16 m/s);
* **the outcome**: fallen (the pelvis tilted > 30 deg or below 0.6 m), the largest tilt after the first second, the
  tilt at the end (11 s), how far the feet moved (a recovery step), and the core's mode at the end.

A batch like this is how we checked that the policy still stands on the robot identified from hardware data before
it first balanced the real robot (100 seeds per case there; the pass rule: >= 99 / 100). The numbers are the
simulator's: it is a model of one R1, and a push in the simulator is not a push on your robot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from r1sim import joints as J
from r1sim import model, variants, world
from r1sim.driver import DriverConfig
from r1sim.engine import Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.sensors import SensorConfig, SensorModel

DURATION_S = 11.0
FAULT_AT_S = 2.0
PUSHES = ((3.0, (1.0, 0.0)), (5.0, (-1.0, 0.0)), (7.0, (0.0, 1.0)), (9.0, (0.0, -1.0)))  # (t, direction in base x/y)
PUSH_S = 0.2
FAULTS: dict[str, tuple[str, float, dict[str, Any]] | None] = {
    "none": None,
    "cmd_gap": ("cmd_gap", 0.3, {}),
    "state_delay": ("state_delay", 2.0, {"delay_s": 0.04}),
    "policy_nan": ("policy_nan", 0.2, {}),
}
FALL_TILT_DEG = 30.0
FALL_Z_M = 0.6
SAMPLE_S = 0.02
MODES = {0: "MONITOR", 3: "BALANCE", 4: "GUARDED", 5: "STAND_FREEZE", 6: "ON_STAND", 8: "DAMP"}


def _tilt_deg(quat_wxyz: Any) -> float:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat_wxyz, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(r[8], -1.0, 1.0))))


def run_one(seed: int, impulse_ns: float, fault: str) -> dict[str, Any]:
    """One robot, one push size, one fault case (module docstring)."""
    from r1sim.v3_core import V3CoreController

    t_wall = time.perf_counter()
    params = variants.sample_params("real", seed)
    spec = model.robot_spec("real", visual=False, params=params)
    m = spec.compile()
    w = world.World(profile="real", model=m, index=model.RobotIndex(m), scene=None, seed=seed, spec=spec)
    sched = FaultSchedule(seed=seed)
    f = FAULTS[fault]
    if f is not None:
        sched.add(Fault(f[0], FAULT_AT_S, f[1], f[2]))  # type: ignore[arg-type]
    sensors = SensorModel(SensorConfig.from_yaml(), seed=seed)
    sim = Simulation(w.model, w.index, driver=DriverConfig.real(), sensors=sensors, faults=sched)
    sim.reset(w.initial_qpos(J.POSE_STAND.copy()))
    ctl = V3CoreController(w.model, faults=sched)
    sim.add_controller(ctl, ctl.period_s, "core")
    ix = w.index
    torso = ix.torso_body
    feet0 = np.array([sim.d.site_xpos[s][:2] for s in ix.foot_sites])
    force = impulse_ns / PUSH_S
    tilts: list[tuple[float, float]] = []
    fallen_t: float | None = None
    for _ in range(round(DURATION_S / SAMPLE_S)):
        t = sim.time
        sim.d.xfrc_applied[torso, :3] = 0.0
        for tp, (dx, dy) in PUSHES:
            if tp - 1e-9 <= t < tp + PUSH_S - 1e-9:
                sim.d.xfrc_applied[torso, :3] = (force * dx, force * dy, 0.0)
        sim.run_for(SAMPLE_S)
        tilt = _tilt_deg(sim.d.xquat[ix.pelvis_body])
        tilts.append((sim.time, tilt))
        if tilt > FALL_TILT_DEG or float(sim.d.xpos[ix.pelvis_body][2]) < FALL_Z_M:
            fallen_t = sim.time
            break
    sim.d.xfrc_applied[torso, :3] = 0.0
    feet = np.array([sim.d.site_xpos[s][:2] for s in ix.foot_sites])
    after = [tl for t, tl in tilts if t >= 1.0]
    return {
        "seed": seed,
        "impulse_ns": impulse_ns,
        "fault": fault,
        "fallen": fallen_t is not None,
        "fallen_t_s": None if fallen_t is None else round(fallen_t, 2),
        "tilt_max_deg": round(max(after), 2) if after else None,
        "tilt_end_deg": round(tilts[-1][1], 2),
        "feet_moved_mm": round(1000.0 * float(np.linalg.norm(feet - feet0, axis=1).max()), 1),
        "mode_end": MODES.get(int(ctl.core.mode), str(int(ctl.core.mode))),
        "mass_kg": round(ix.total_mass, 2),
        "wall_s": round(time.perf_counter() - t_wall, 1),
    }


def _task(args: tuple[int, float, str]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        return run_one(*args)
    except Exception as e:  # a crash is a result, reported - never a silent pass
        return {"seed": args[0], "impulse_ns": args[1], "fault": args[2], "fallen": True, "error": repr(e)}


def table(rows: Sequence[dict[str, Any]]) -> str:
    """One line per (push size, fault): falls / runs, the largest and the median max tilt, the end mode(s)."""
    out = [f"{'push N s':>8}  {'fault':12}  {'falls':>7}  {'max tilt (deg): median / worst':>32}  {'feet moved mm':>13}"
           f"  end mode"]  # fmt: skip
    keys = sorted({(r["impulse_ns"], r["fault"]) for r in rows}, key=lambda k: (k[0], list(FAULTS).index(k[1])))
    for imp, fault in keys:
        rs = [r for r in rows if r["impulse_ns"] == imp and r["fault"] == fault]
        falls = sum(bool(r["fallen"]) for r in rs)
        tl = [float(r["tilt_max_deg"]) for r in rs if r.get("tilt_max_deg") is not None]
        fm = [float(r["feet_moved_mm"]) for r in rs if r.get("feet_moved_mm") is not None]
        modes = sorted({str(r.get("mode_end", "error")) for r in rs})
        med = f"{float(np.median(tl)):5.1f} / {max(tl):5.1f}" if tl else "-"
        feet = f"{max(fm):13.0f}" if fm else f"{'-':>13}"
        out.append(f"{imp:8.1f}  {fault:12}  {falls:3d} / {len(rs):<3d}  {med:>32}  {feet}  {', '.join(modes)}")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8, help="robots (model draws) per case")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--impulses", type=float, nargs="+", default=[4.0, 8.0, 12.0], help="push sizes (N s)")
    ap.add_argument("--faults", nargs="+", default=["none", "cmd_gap", "state_delay"], choices=list(FAULTS))
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--quick", action="store_true", help="2 robots, one 5 N s push size, no fault")
    ap.add_argument("--json", type=Path, default=None, help="every run's numbers to this file")
    a = ap.parse_args(argv)
    from r1sim import cpp_policy

    if cpp_policy.find_module() is None:
        sys.exit("the C++ core's Python binding (r1rt_py) is not built: bash scripts/ci.sh cpp (README.md)")
    if a.quick:
        a.seeds, a.impulses, a.faults = 2, [5.0], ["none"]
    jobs = [(s, float(i), f) for i in a.impulses for f in a.faults for s in range(a.first_seed, a.first_seed + a.seeds)]
    t0 = time.monotonic()
    print(f"{len(jobs)} runs ({a.seeds} robots x {len(a.impulses)} push sizes x {len(a.faults)} fault cases), "
          f"{DURATION_S:.0f} s simulated each, {a.workers} worker(s)", flush=True)  # fmt: skip
    rows: list[dict[str, Any]] = []
    if a.workers <= 1:
        for k, j in enumerate(jobs, 1):
            rows.append(_task(j))
            print(f"  {k} / {len(jobs)} done", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for k, r in enumerate(ex.map(_task, jobs), 1):
                rows.append(r)
                if k % max(1, len(jobs) // 10) == 0 or k == len(jobs):
                    print(f"  {k} / {len(jobs)} done", flush=True)
    print()
    print(table(rows))
    errors = [r for r in rows if "error" in r]
    for r in errors:
        print(f"ERROR seed {r['seed']} {r['impulse_ns']} N s {r['fault']}: {r['error']}")
    falls = sum(bool(r["fallen"]) for r in rows)
    print(f"\n{falls} of {len(rows)} runs fell; {time.monotonic() - t0:.0f} s wall")
    if a.json is not None:
        a.json.write_text(json.dumps(rows, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
