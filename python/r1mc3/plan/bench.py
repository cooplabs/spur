"""The planner's numbers (DESIGN.md §7.2 performance target: a hover or carry plan < 1 s on the Orin NX; v2 took
28 s / 4 min on the backpack, 12-18 s / 13 s with the Mac planner).

    python -m r1mc3.plan.bench [--n 40] [--seed 1] [--json out.json]

A generic scene (a 0.76 m table 0.6 x 1.2 m, an open bin, a few boxes - nothing task-specific) and a set of goals per
kind drawn on a grid of the table: reachable ones, ones out of reach, ones inside an obstacle. Per kind: how many
planned / were refused with the expected code, the plan time p50 / p90 / max (wall clock) and the planning thread's
CPU time p50 / max (the number that does not depend on what else the machine runs), the motion's duration. Runs anywhere
the package runs (the Mac now; the backpack at HW2 for the Orin number: the same command, nothing moves).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from typing import Any

import numpy as np
from r1sim import joints as J

from ..world import Container, Object, Obstacle, Pose, Surface, WorldModel, WorldSnapshot
from .core import PlanGoal, geometry_for, plan

TABLE_Z = 0.76


def scene() -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.55, 0.0, TABLE_Z), size=(0.6, 1.2, 0.04)))
    w.declare(Container("bin", Pose.xyz_yaw(0.45, 0.30, TABLE_Z), size_outer=(0.35, 0.26, 0.045)))
    w.declare(Object("box_a", Pose.xyz_yaw(0.40, -0.15, TABLE_Z + 0.025), size=(0.025, 0.025, 0.025)))
    w.declare(Object("box_b", Pose.xyz_yaw(0.33, 0.05, TABLE_Z + 0.04), shape="cylinder", size=(0.03, 0.04)))
    w.declare(Obstacle("block_l", Pose.xyz_yaw(0.30, 0.12, TABLE_Z + 0.06), size=(0.03, 0.03, 0.06)))
    w.declare(Obstacle("block_r", Pose.xyz_yaw(0.30, -0.08, TABLE_Z + 0.06), size=(0.03, 0.03, 0.06)))
    return w.snapshot()


def goals(n: int, seed: int) -> list[tuple[str, PlanGoal, str]]:
    """(label, goal, expected) with expected 'ok' or a refusal code family ('refuse')."""
    rng = np.random.default_rng(seed)
    out: list[tuple[str, PlanGoal, str]] = []
    for i in range(n):
        side = "left" if i % 2 == 0 else "right"
        g = 1.0 if side == "left" else -1.0
        x, y = float(rng.uniform(0.30, 0.42)), g * float(rng.uniform(0.02, 0.25))
        out.append(("hover", PlanGoal("hover", side, position=(x, y, 0.0), height_m=float(rng.uniform(0.03, 0.10)),
                                      seed=i), "ok"))  # fmt: skip
        out.append(("reach", PlanGoal("reach", side, position=(x, y, TABLE_Z + float(rng.uniform(0.08, 0.2))),
                                      seed=i), "ok"))  # fmt: skip
    for i in range(max(2, n // 4)):
        side = "left" if i % 2 == 0 else "right"
        g = 1.0 if side == "left" else -1.0
        out.append(("far", PlanGoal("reach", side, position=(1.1, g * 0.2, 0.9), seed=i), "refuse"))
        y_block = 0.12 if side == "left" else -0.08
        out.append(("inside", PlanGoal("reach", side, position=(0.30, y_block, TABLE_Z + 0.05), seed=i), "refuse"))
    return out


def carry_goals(snap: WorldSnapshot, n: int, seed: int) -> list[tuple[str, PlanGoal, np.ndarray]]:
    """Carries from a held pose (a hover 2 cm over the table) to a place 10-25 cm away."""
    rng = np.random.default_rng(seed + 7)
    out: list[tuple[str, PlanGoal, np.ndarray]] = []
    for i in range(n):
        side = "left" if i % 2 == 0 else "right"
        g = 1.0 if side == "left" else -1.0
        q = J.POSE_STAND.copy()
        h = plan(snap, PlanGoal("hover", side, position=(0.44, 0.0, 0.0), height_m=0.02), q)
        if not h.ok:
            continue
        idx = J.LEFT_ARM if side == "left" else J.RIGHT_ARM
        q[idx] = h.segments[-1].q[-1]
        tgt = (
            float(rng.uniform(0.32, 0.42)),
            g * float(rng.uniform(0.0, 0.2)),
            TABLE_Z + float(rng.uniform(0.06, 0.12)),
        )
        out.append(("carry", PlanGoal("carry", side, position=tgt, lift_m=0.05, seed=i), q))
    return out


def style_goals(n: int) -> list[tuple[str, PlanGoal, str]]:
    """Session 12: hovers with a requested approach over the scene's two declared objects (the grasp pose chosen
    where the grip closes, the open hand clear of the object) - the Orin's number for a styled hover (HW2 day 2: R5)."""
    out: list[tuple[str, PlanGoal, str]] = []
    targets = (("box_b", "left", (0.33, 0.05), "wrap"), ("box_a", "right", (0.40, -0.15), "pinch"))
    styles = ("around", "low", "front", "inward")
    for i in range(max(2, n // 4)):
        tid, side, xy, grip = targets[i % 2]
        st = styles[(i // 2) % len(styles)]
        g = PlanGoal("hover", side, position=(xy[0], xy[1], TABLE_Z), height_m=0.08, grip=grip, grasp_yaw_deg=90.0,
                     allow_hand=(tid,), approach_style=st, seed=i)  # fmt: skip
        out.append(("styled", g, "any"))  # a refusal (out of reach, no clear way) is a valid answer
    return out


def run(n: int = 40, seed: int = 1) -> dict[str, Any]:
    snap = scene()
    t0 = time.perf_counter()
    geometry_for(snap)
    build_s = time.perf_counter() - t0
    rows: dict[str, dict[str, Any]] = {}

    def add(label: str, ok: bool, expected: str, dt: float, dur: float, code: str, cpu: float) -> None:
        r = rows.setdefault(
            label, {"n": 0, "as_expected": 0, "planned": 0, "times": [], "cpu": [], "durations": [], "codes": {}}
        )
        r["cpu"].append(cpu)
        r["n"] += 1
        r["planned"] += int(ok)
        r["as_expected"] += int(expected == "any" or ok == (expected == "ok"))
        r["times"].append(dt)
        if ok:
            r["durations"].append(dur)
        r["codes"][code] = r["codes"].get(code, 0) + 1

    q0 = J.POSE_STAND.copy()
    for label, g, exp in goals(n, seed):
        res = plan(snap, g, q0)
        add(label, res.ok, exp, res.planning_s, res.duration_s, res.code, res.details["cpu_s"])
    for label, g, exp in style_goals(n):
        res = plan(snap, g, q0)
        add(label, res.ok, exp, res.planning_s, res.duration_s, res.code, res.details["cpu_s"])
    for label, g, q in carry_goals(snap, n // 2, seed):
        res = plan(snap, g, q)  # carries may meet the scene's blocks: a refusal is a valid answer
        add(label, res.ok, "any", res.planning_s, res.duration_s, res.code, res.details["cpu_s"])
    out: dict[str, Any] = {"machine": platform.platform(), "python": platform.python_version(), "build_s": build_s,
                           "kinds": {}}  # fmt: skip
    for label, r in rows.items():
        t = np.array(r["times"])
        d = np.array(r["durations"]) if r["durations"] else np.zeros(1)
        out["kinds"][label] = {
            "n": r["n"], "planned": r["planned"], "as_expected": r["as_expected"], "codes": r["codes"],
            "plan_s_p50": round(float(np.median(t)), 3), "plan_s_p90": round(float(np.percentile(t, 90)), 3),
            "plan_s_max": round(float(t.max()), 3), "motion_s_p50": round(float(np.median(d)), 2),
            "cpu_s_p50": round(float(np.median(r["cpu"])), 3), "cpu_s_max": round(float(np.max(r["cpu"])), 3),
        }  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1mc3.plan.bench", description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    res = run(a.n, a.seed)
    print(f"{res['machine']}  model build {res['build_s']:.3f} s")
    hdr = ("kind", "n", "planned", "expected", "p50 s", "p90 s", "max s", "cpu p50", "cpu max", "motion s")
    print("{:8} {:>4} {:>8} {:>9} {:>7} {:>7} {:>7} {:>8} {:>8} {:>9}  codes".format(*hdr))
    for k, r in res["kinds"].items():
        vals = (k, r["n"], r["planned"], r["as_expected"], r["plan_s_p50"], r["plan_s_p90"], r["plan_s_max"])
        cpu = (r["cpu_s_p50"], r["cpu_s_max"], r["motion_s_p50"])
        print("{:8} {:4d} {:8d} {:9d} {:7.3f} {:7.3f} {:7.3f}".format(*vals), "{:8.3f} {:8.3f} {:9.2f}".format(*cpu),
              f" {r['codes']}")  # fmt: skip
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
