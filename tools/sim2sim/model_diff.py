"""Field-by-field diff of mjlab's compiled training model against r1sim's train-equivalent profile (A1 groundwork),
robot + floor, elements matched by name (r1sim.train_equiv.model_fields / diff_fields).

    uv run python tools/sim2sim/model_diff.py /tmp/x/mjlab_r1.mjb        # a model from mjlab_reference.py model
    uv run python tools/sim2sim/model_diff.py tests/assets/a1/mjlab_model_fields.npz
    uv run python tools/sim2sim/model_diff.py /tmp/x/mjlab_r1.mjb --save tests/assets/a1/mjlab_model_fields.npz

Prints one line per differing field and whether it is explained (r1sim.train_equiv.EXPLAINED_DIFFS); exit 1 if any
unexplained difference remains. Default tolerance 0: bit-exact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "sim"), str(REPO / "python")]


def main(argv: list[str] | None = None) -> int:
    from r1sim import model as r1model
    from r1sim import train_equiv as te

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("reference", help="mjlab model (.mjb) or its saved fingerprint (.npz)")
    ap.add_argument("--tol", type=float, default=0.0, help="relative tolerance (0 = bit-exact)")
    ap.add_argument("--collision-only", action="store_true", help="skip geoms that collide in neither model")
    ap.add_argument("--save", help="write the reference's fingerprint to this .npz")
    a = ap.parse_args(argv)
    if a.reference.endswith(".npz"):
        ref = te.load_fields(a.reference)
    else:
        ref = te.model_fields(mujoco.MjModel.from_binary_path(a.reference))
    if a.save:
        te.save_fields(a.save, ref)
        print(f"saved {a.save} ({len(ref)} fields)")
    ours = te.model_fields(r1model.build_model("train"))
    lines = te.diff_fields(ref, ours, a.tol, collision_geoms_only=a.collision_only)
    bad = 0
    for ln in lines:
        key = ln.split(":", 1)[0]
        why = te.EXPLAINED_DIFFS.get(key)
        bad += why is None
        print(f"{ln}   [{'explained: ' + why if why else 'UNEXPLAINED'}]")
    print(f"{len(lines)} differences, {bad} unexplained (tol {a.tol:g})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
