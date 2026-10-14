"""`python -m r1sim.sysid` - the HW1 fits (see r1sim.sysid and docs/hardware/HW1_FITS.md).

python -m r1sim.sysid stiction --log r1log_XXX.mcap [--joints left_knee,...] [--source pd|tau_est] [--json OUT]
python -m r1sim.sysid pads --hands hands.mcap --presses presses.yaml --side left [--json OUT]
python -m r1sim.sysid heat --hands hands.mcap --side left --motor 3 --windows 10:190,200:380 [--json OUT]
python -m r1sim.sysid demo {stiction,pads,heat}      # the same fits on synthetic recordings (known truth)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import joints as J

LEG_WAIST = [n for n in J.JOINTS[:14]]


def _dump(obj: Any, path: str | None) -> None:
    txt = json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if path:
        Path(path).write_text(txt)
    else:
        print(txt)


def _stiction(a: argparse.Namespace) -> int:
    from . import records, stiction

    rec = records.load_body(a.log)
    joints = a.joints.split(",") if a.joints else LEG_WAIST
    fits = stiction.fit_joints(rec, joints, source=a.source)
    for name, f in fits.items():
        print(
            f"{name:18s} Fc {f.frictionloss:6.3f} +- {f.frictionloss_se:.3f} N m  b {f.damping:6.3f} +- "
            f"{f.damping_se:.3f} N m s/rad  Fs {f.static:6.3f}  rms {f.residual_rms:.3f}  {'; '.join(f.notes)}"
        )
    if a.provenance:
        print(stiction.provenance_yaml(fits, source=str(a.log), date=_dt.date.today().isoformat()))
    if a.json:
        _dump({k: v.as_dict() for k, v in fits.items()}, a.json)
    return 0


def _pads(a: argparse.Namespace) -> int:
    from . import pads, records

    rec = records.load_hand(a.hands, a.side)
    presses = records.load_presses(a.presses)
    fits = pads.fit_hand(rec, presses)
    for k, f in fits.items():
        print(
            f"pad {k}: gain {f.gain_raw_per_n:.0f} raw/N (sim pad_gain {f.sim_pad_gain:.3f}), footprint "
            f"{f.x0_mm:.1f}-{f.x1_mm:.1f} mm (mesh {f.mesh_footprint_mm}), edge {f.edge_mm} mm, floor {f.floor_g} g"
        )
    if a.json:
        _dump({k: v.as_dict() for k, v in fits.items()}, a.json)
    return 0


def _heat(a: argparse.Namespace) -> int:
    from . import heat, records

    rec = records.load_hand(a.hands, a.side)
    win = [tuple(float(x) for x in w.split(":")) for w in a.windows.split(",")]
    holds = heat.holds_from_recording(rec, a.motor, [(w[0], w[1]) for w in win])
    rep = heat.fit_holds(holds)
    print(rep.supported)
    print(rep.table())
    if a.json:
        _dump(heat.summary(rep, holds), a.json)
    return 0


def _demo(a: argparse.Namespace) -> int:
    if a.what == "stiction":
        from . import stiction

        for j, fl, b in (("left_knee", 1.8, 0.5), ("left_ankle_pitch", 0.9, 0.2), ("left_hip_pitch", 0.0, 0.05)):
            rec, _ = stiction.synthetic_ramp(j, frictionloss=fl, damping=b, seed=a.seed)
            f = stiction.fit_joint(rec, j)
            print(f"{j:18s} true Fc {fl:.3f} b {b:.3f} -> Fc {f.frictionloss:.3f} +- {f.frictionloss_se:.3f}, "
                  f"b {f.damping:.3f} +- {f.damping_se:.3f}, Fs {f.static:.3f}")  # fmt: skip
    elif a.what == "pads":
        from . import pads

        g = pads.pad_geometry("left")
        for k in (2, 3):
            pl = pads.press_plan(g[k])
            truth = pads.PadTruth(gain={k: 0.6}, half_long_mm={k: g[k].half_long_mm * 0.7})
            hrec = pads.synthetic_presses("left", pl, truth, seed=a.seed)
            pf = pads.fit_hand(hrec, pl, g)[k]
            h = g[k].half_long_mm * 0.7 + g[k].margin_mm
            print(
                f"pad {k}: true gain 0.600, footprint {g[k].x_center_mm - h:.1f}-{g[k].x_center_mm + h:.1f} mm -> "
                f"gain {pf.sim_pad_gain:.3f}, footprint {pf.x0_mm:.1f}-{pf.x1_mm:.1f} mm, floor {pf.floor_g} g"
            )
    else:
        from . import heat

        for name, tr in (
            ("sim (two poles, |tau|^1.5, no derating)", heat.TruthModel.sim()),
            ("Joule + derating", heat.TruthModel(g_fast=30.0, tau_fast_s=2.0, g_slow=150.0, tau_slow_s=400.0, p=2.0,
                                                 derate_from_c=60.0, derate_per_c=0.03, derate_floor=0.3)),
        ):  # fmt: skip
            holds = [heat.synthetic_hold(tr, 0.9, cool_s=600.0, seed=a.seed),
                     heat.synthetic_hold(tr, 0.45, cool_s=600.0, seed=a.seed + 1)]  # fmt: skip
            rep = heat.fit_holds(holds)
            print(f"truth: {name}\n  -> {rep.supported}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1sim.sysid", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stiction", help="leg / waist friction + damping from the stand ramps (r1log MCAP or .npz)")
    s.add_argument("--log", required=True)
    s.add_argument("--joints", default="")
    s.add_argument("--source", choices=("pd", "tau_est"), default="pd")
    s.add_argument("--provenance", action="store_true", help="print a models/provenance.yaml snippet")
    s.add_argument("--json")
    p = sub.add_parser("pads", help="pad gain + footprint from presses with known weights")
    p.add_argument("--hands", required=True)
    p.add_argument("--presses", required=True)
    p.add_argument("--side", choices=("left", "right"), required=True)
    p.add_argument("--json")
    h = sub.add_parser("heat", help="the finger motors' thermal structure from full-rate holds")
    h.add_argument("--hands", required=True)
    h.add_argument("--side", choices=("left", "right"), required=True)
    h.add_argument("--motor", type=int, required=True)
    h.add_argument("--windows", required=True, help="a:b,c:d (s): each hold + its cooling")
    h.add_argument("--json")
    d = sub.add_parser("demo", help="run a fit on a synthetic recording with known truth")
    d.add_argument("what", choices=("stiction", "pads", "heat"))
    d.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    fn = {"stiction": _stiction, "pads": _pads, "heat": _heat, "demo": _demo}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
