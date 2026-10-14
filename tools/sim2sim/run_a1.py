"""A1 sim2sim numbers (DESIGN.md §13.1 A1): the live policy on r1sim's train-equivalent profile vs mjlab's
reference rollouts (tests/assets/a1, from tools/sim2sim/mjlab_reference.py), same policy executor, observation code,
initial state (training's reset) and command scripts (S0 stand 10 s; S1 stand 2 s, vx 0.3 m/s 2-6 s, stand to 10 s).

    uv run python tools/sim2sim/run_a1.py            # table for both scripts, both reference engines

Rows: ours (training interface) vs mjlab on MuJoCo CPU (same engine: isolates the model / step structure) and vs
mjlab on MuJoCo Warp (the training engine, float32 on Warp's CPU device), the two mjlab engines against each other
(risk K3), and ours with the DEPLOYED interface (the ONNX metadata's 3-decimal action scale and gains, as the robot
runs it) vs the training reference. Targets (A1): per-joint RMS <= 0.02 rad, tilt RMS <= 0.5 deg; a report, not a
gate, in session 2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "sim"), str(REPO / "python")]


def main(argv: list[str] | None = None) -> int:
    from r1mc3.policy import POLICY_DIR
    from r1mc3.policy.onnx_np import OnnxModel
    from r1sim import model as M
    from r1sim import train_equiv as te

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--assets", default=str(te.A1_ASSETS))
    a = ap.parse_args(argv)
    net = OnnxModel.load(POLICY_DIR / "policy.onnx")
    pi_t, pi_d = te.PolicyInterface.training(net.meta), te.PolicyInterface.deployed(net.meta)
    m = M.build_model("train", visual=False)
    print(f"training reset: soles {1000 * te.foot_clearance(m, te.training_reset_qpos(m)):.1f} mm above the floor")
    print("| script | comparison | joint RMS max (rad) | worst joint | max abs error (rad) | tilt RMS diff (deg) | "
          "first > 0.02 rad | fell (ours/ref) |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|")
    for sc in ("S0", "S1"):
        ref = {e: dict(np.load(Path(a.assets) / f"mjlab_{e}_{sc}.npz")) for e in ("cpu", "warp")}
        ours = te.run_policy(net, pi_t, script=sc, m=m).arrays()
        dep = te.run_policy(net, pi_d, script=sc, m=m).arrays()
        rows = (("ours vs mjlab MuJoCo-CPU", ours, ref["cpu"]), ("ours vs mjlab MuJoCo-Warp", ours, ref["warp"]),
                ("mjlab CPU vs mjlab Warp", ref["cpu"], ref["warp"]),
                ("ours, deployed interface, vs mjlab CPU", dep, ref["cpu"]))  # fmt: skip
        for label, x, y in rows:
            n = te.a1_numbers(x, y, pi_t.names)
            w = int(np.argmax(n.joint_rms))
            fd = "-" if n.first_divergence_s is None else f"{n.first_divergence_s:.2f} s"
            print(f"| {sc} | {label} | {n.joint_rms_max:.3g} | {n.names[w]} | {n.joint_err_max:.3g} | "
                  f"{n.tilt_rms_diff_deg:.3g} | {fd} | {n.fell_ours}/{n.fell_ref} |")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
