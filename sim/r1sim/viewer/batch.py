"""Headless batch runs with seeds (`python -m r1sim.viewer.batch`): a robustness check the way a researcher runs one.

For every seed: one robot VARIANT within the model's parameter ranges (`r1sim.variants`, drawn from the seed: masses,
CoM, friction, delays, ... - `world.build_world(seed=...)`), the viewer's in-process controller (default `guarded` =
the C++ core + r1guard in lockstep), a schedule of the viewer's keys at fixed times (`--keys 'S@4.0,N@6.0'`) and / or
random pushes drawn from the same seed (how many, how hard, how long; any direction), for `--duration` simulated
seconds. Per seed it records whether the robot fell (the pelvis below FALL_Z_M or the base tilted beyond FALL_TILT_DEG,
and when), the largest tilt, the core's modes in order, r1guard's takeovers, every event; optionally an MP4 per seed.
One JSON summary for the batch (stdout and `--out`).

    python -m r1sim.viewer.batch --seeds 0-9 --duration 10 --pushes 3 --push-n 40,90
    python -m r1sim.viewer.batch --seeds nominal,0-4 --keys 'P@2.0,S@4.0,P@6.0' --mp4-dir out/batch
    python -m r1sim.viewer.batch --seeds 0-19 --scene table --controller core --pushes 2 --out out/batch.json

`run_batch(BatchConfig(...))` is the same from Python. Lockstep, so the numbers do not depend on the machine's load:
the same seeds give the same result. What is NOT varied: the scene, the floor, the policy's weights; what is NOT
modelled is in the model's provenance (models/provenance.yaml) - a pass here is the simulator's verdict, not the
robot's.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .app import CONTROLLERS, SCENES, App, Event, ViewerConfig, valid_scene

FALL_Z_M = 0.45  # the pelvis below this (standing: ~0.73 m) = fallen
FALL_TILT_DEG = 30.0  # the base tilted beyond this = fallen (the core's own fall event is at 30 deg)
SAMPLE_S = 0.01  # the run is checked this often (simulated s)


@dataclass
class BatchConfig:
    seeds: list[int | None] = field(default_factory=lambda: list(range(10)))  # None = the nominal robot
    duration_s: float = 10.0
    scene: str = "floor"
    controller: str = "guarded"
    keys: str = ""  # the viewer's keys at times: 'P@2.0,S@4.0' (P push, S stall, N policy NaN, G, D, M, ...)
    pushes: int = 0  # random pushes per seed (drawn from the seed)
    push_n: tuple[float, float] = (40.0, 90.0)  # their force range (N, horizontal, on the pelvis)
    push_s: float = 0.15  # their duration (s)
    push_window: tuple[float, float] | None = None  # when they may come (s); default (1.0, duration - 1.0)
    mp4_dir: str | None = None  # an MP4 per seed (needs ffmpeg), every mp4_every_s simulated seconds
    mp4_every_s: float = 0.05
    width: int = 480
    height: int = 360


def parse_seeds(spec: str) -> list[int | None]:
    """'0-9', '0,3,5', 'nominal,0-4' -> the seed list (None = the nominal robot)."""
    out: list[int | None] = []
    for tok in filter(None, (t.strip() for t in spec.split(","))):
        if tok == "nominal":
            out.append(None)
        elif "-" in tok[1:]:
            a, _, b = tok.partition("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    if not out:
        raise ValueError("--seeds: e.g. 0-9, 0,3,5 or nominal,0-4")
    return out


def parse_keys(spec: str) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        k, _, t = item.partition("@")
        if len(k) != 1 or not t:
            raise ValueError(f"keys: 'K@t' items (one character, a time), not {item!r}")
        out.append((float(t), k))
    return sorted(out)


def random_pushes(cfg: BatchConfig, seed: int | None) -> list[tuple[float, tuple[float, float]]]:
    """The seed's random pushes: (time, (fx, fy)) - uniform times in the window, uniform directions, uniform forces."""
    if cfg.pushes <= 0:
        return []
    rng = np.random.default_rng([7, 0 if seed is None else seed + 1])
    lo, hi = cfg.push_window if cfg.push_window is not None else (1.0, max(1.0, cfg.duration_s - 1.0))
    times = np.sort(rng.uniform(lo, hi, cfg.pushes))
    out: list[tuple[float, tuple[float, float]]] = []
    for t in times:
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        f = float(rng.uniform(*cfg.push_n))
        out.append((float(t), (f * math.cos(ang), f * math.sin(ang))))
    return out


def run_seed(cfg: BatchConfig, seed: int | None) -> dict[str, Any]:
    """One seed's run (see the module docstring); the result row."""
    t_wall = time.perf_counter()
    app = App(ViewerConfig(scene=cfg.scene, controller=cfg.controller, seed=seed, camera="follow",
                           width=cfg.width, height=cfg.height))  # fmt: skip
    plan: list[tuple[float, str, Any]] = [(t, "key", k) for t, k in parse_keys(cfg.keys)]
    plan += [(t, "push", f) for t, f in random_pushes(cfg, seed)]
    plan.sort(key=lambda e: e[0])
    rec: Any = None
    frames_dir: Path | None = None
    if cfg.mp4_dir is not None:
        from .__main__ import Recorder

        frames_dir = Path(cfg.mp4_dir) / f"seed_{'nominal' if seed is None else seed}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        rec = Recorder(app, frames_dir, cfg.width, cfg.height)
    max_tilt, t_fall, fall_why = 0.0, None, ""
    modes: list[tuple[float, str]] = []
    next_frame = 0.0
    n_frame = 0
    t = 0.0
    try:
        while t < cfg.duration_s - 1e-9:
            t = min(cfg.duration_s, t + SAMPLE_S)
            while plan and plan[0][0] <= t + 1e-9:
                tp, kind, arg = plan.pop(0)
                app.advance(tp, max_steps=10**7)
                if kind == "key":
                    app.key(arg)
                else:
                    app.events.append(Event(app.t, app.push(arg, cfg.push_s)))
            app.advance(t, max_steps=10**7)
            tilt, z = app.tilt_deg(), app.pelvis_z()
            max_tilt = max(max_tilt, tilt)
            if t_fall is None and (z < FALL_Z_M or tilt > FALL_TILT_DEG):
                t_fall, fall_why = round(app.t, 3), (f"pelvis {z:.2f} m" if z < FALL_Z_M else f"tilt {tilt:.0f} deg")
            mode = app.mode_name() or app.controller
            if app.token_owner() == 2:
                mode = "r1guard"
            if not modes or modes[-1][1] != mode:
                modes.append((round(app.t, 3), mode))
            if rec is not None and frames_dir is not None and app.t + 1e-9 >= next_frame:
                rec.frame(frames_dir / f"frame_{n_frame:05d}.png")
                n_frame += 1
                next_frame += cfg.mp4_every_s
        movie = None
        if rec is not None and frames_dir is not None:
            movie = rec.mp4(frames_dir.parent / f"{frames_dir.name}.mp4", 1.0 / cfg.mp4_every_s)
            if movie is not None:
                shutil.rmtree(frames_dir)
    finally:
        if rec is not None:
            rec.close()
    return {
        "seed": seed,
        "fell": t_fall is not None,
        "t_fall": t_fall,
        "fall": fall_why,
        "max_tilt_deg": round(max_tilt, 3),
        "final_tilt_deg": round(app.tilt_deg(), 3),
        "modes": modes,
        "guard_takeovers": app.guard_takeovers(),
        "events": [{"t": round(e.t, 3), "what": e.what} for e in app.events],
        "mp4": None if movie is None else str(movie),
        "wall_s": round(time.perf_counter() - t_wall, 3),
    }


def run_batch(cfg: BatchConfig, *, progress: bool = False) -> dict[str, Any]:
    """Every seed of `cfg` -> {"config", "runs" (run_seed rows), "n", "falls", "fall_seeds", "max_tilt_deg",
    "wall_s"}."""
    if not valid_scene(cfg.scene) or cfg.scene == "stand":
        raise ValueError(f"scene: one of {tuple(s for s in SCENES if s != 'stand')} or 'package.module:attr'")
    if cfg.controller not in CONTROLLERS:
        raise ValueError(f"controller: one of {CONTROLLERS}")
    t0 = time.perf_counter()
    runs = []
    for s in cfg.seeds:
        r = run_seed(cfg, s)
        runs.append(r)
        if progress:
            print(f"seed {s}: {'FELL at ' + str(r['t_fall']) + ' s (' + r['fall'] + ')' if r['fell'] else 'stood'}; "
                  f"max tilt {r['max_tilt_deg']:.2f} deg; {r['wall_s']:.1f} s", file=sys.stderr)  # fmt: skip
    falls = [r["seed"] for r in runs if r["fell"]]
    return {
        "config": asdict(cfg),
        "n": len(runs),
        "falls": len(falls),
        "fall_seeds": falls,
        "max_tilt_deg": round(max((r["max_tilt_deg"] for r in runs), default=0.0), 3),
        "runs": runs,
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m r1sim.viewer.batch", description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--seeds", default="0-9", help="'0-9', '0,3,5', 'nominal,0-4' (nominal = no variant)")
    ap.add_argument("--duration", type=float, default=10.0, help="simulated seconds per seed")
    ap.add_argument("--scene", default="floor", help="floor | table | 'package.module:attr'")
    ap.add_argument("--controller", choices=CONTROLLERS, default="guarded")
    ap.add_argument("--keys", default="", help="the viewer's keys at times, e.g. 'P@2.0,S@4.0,N@6.0'")
    ap.add_argument("--pushes", type=int, default=0, help="random pushes per seed (drawn from the seed)")
    ap.add_argument("--push-n", default="40,90", help="their force range, N (lo,hi)")
    ap.add_argument("--push-s", type=float, default=0.15, help="their duration, s")
    ap.add_argument("--mp4-dir", default=None, help="an MP4 per seed in this folder (needs ffmpeg)")
    ap.add_argument("--out", default=None, help="the JSON summary (also printed)")
    ap.add_argument("--fail-on-fall", action="store_true", help="exit code 1 when any seed fell")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    lo, _, hi = a.push_n.partition(",")
    cfg = BatchConfig(seeds=parse_seeds(a.seeds), duration_s=a.duration, scene=a.scene, controller=a.controller,
                      keys=a.keys, pushes=a.pushes, push_n=(float(lo), float(hi or lo)), push_s=a.push_s,
                      mp4_dir=a.mp4_dir)  # fmt: skip
    res = run_batch(cfg, progress=True)
    text = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(json.dumps({k: res[k] for k in ("n", "falls", "fall_seeds", "max_tilt_deg", "wall_s")}))
    return 1 if a.fail_on_fall and res["falls"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
