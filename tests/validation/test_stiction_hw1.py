"""Session 8, fix #27: the stiction fit on the friction HW1 really showed (a Stribeck curve), not only on MuJoCo's.

HW1's 14 stand ramps (11 Oct 2026, results/hw1_20261010/end/session_part3) gave plausible Fc but NEGATIVE damping on 12
of the 14 joints with session 5's fit: the real friction is LOWER at 0.3 than at 0.1 rad/s and both are below the static
breakaway (L knee: Fs 0.94 > h(0.1) 0.64 > h(0.3) 0.56 N m), and the steady-sample filter counted from the segment
start, keeping each reversal's release transient. These tests fail on bad5a5c (b < 0 on both recordings; no friction
curve; provenance_yaml wrote the negative damping) and pass with the breakaway-anchored window (the HW1 wrap-up
analysis, C)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from r1sim import joints as J
from r1sim.sysid import stiction
from r1sim.sysid.records import BodyRecording

ASSET = Path(__file__).resolve().parents[1] / "assets" / "hw1" / "s4_ramp_left_knee.npz"


def _left_knee_excerpt() -> BodyRecording:
    """The real left-knee ramp (manifest 20261011T195110Z_ramp_left_knee.json + r1log_20261011T192459Z_002.mcap, cut
    like fit_joints; tests/assets/hw1/README.md): only the knee's columns were kept, the others are zero."""
    z = np.load(ASSET)
    n = len(z["t"])
    fields = {}
    for k in ("q", "dq", "tau_est", "q_cmd", "dq_cmd", "kp", "kd", "tau_ff", "q_cmd_ref"):
        a = np.zeros((n, J.N))
        a[:, J.IDX["left_knee"]] = z[k]
        fields[k] = a
    return BodyRecording(np.asarray(z["t"], dtype=np.float64), fields)


def test_the_real_left_knee_ramp_gives_its_friction_curve_and_no_negative_damping() -> None:
    fit = stiction.fit_joint(_left_knee_excerpt(), "left_knee")
    assert fit.damping >= 0.0, fit  # bad5a5c: -0.340
    assert fit.damping_source == "prior" and fit.damping == stiction.DAMPING_PRIOR
    assert fit.reliable
    assert fit.curve[0.1].valid and fit.curve[0.3].valid
    assert fit.curve[0.1].h == pytest.approx(0.64, abs=0.05)
    assert fit.curve[0.3].h == pytest.approx(0.56, abs=0.05)
    assert fit.curve[0.3].h < fit.curve[0.1].h < fit.static  # Stribeck: Fs > h(0.1) > h(0.3)
    assert fit.frictionloss == pytest.approx(fit.curve[0.1].h - 0.05 * 0.1, abs=0.01)  # Fc = h(slowest) - b |v|
    assert any("FALLS with speed" in n for n in fit.notes)
    stiction.provenance_yaml({"left_knee": fit}, source="hw1_20261011", date="2026-10-11")  # accepted


def test_a_stribeck_joint_gives_no_negative_damping() -> None:
    """The protocol on the simulated hanging knee whose friction FALLS with speed (frictionloss 0.6 + a hidden 0.3 N m
    Stribeck excess: h(0.1) ~ 0.80, h(0.3) ~ 0.62): bad5a5c's fit returned b = -1.107 here."""
    rec, _ = stiction.synthetic_ramp("left_knee", frictionloss=0.6, damping=0.05, stribeck=0.3, seed=11)
    fit = stiction.fit_joint(rec, "left_knee")
    assert fit.damping >= 0.0, fit
    assert fit.damping_source == "prior"
    h1, h3 = fit.curve[0.1], fit.curve[0.3]
    assert h1.valid and h3.valid
    want = {sp: 0.6 + 0.05 * sp + 0.3 * np.exp(-((sp / 0.15) ** 2)) for sp in (0.1, 0.3)}
    assert h1.h == pytest.approx(want[0.1], abs=0.06), fit.curve
    assert h3.h == pytest.approx(want[0.3], abs=0.06), fit.curve
    assert fit.frictionloss == pytest.approx(want[0.1] - 0.05 * 0.1, abs=0.07)


def test_provenance_refuses_a_negative_damping() -> None:
    fit = stiction.fit_joint(_left_knee_excerpt(), "left_knee")
    bad = dataclasses.replace(fit, damping=-0.34)
    with pytest.raises(ValueError, match="negative"):
        stiction.provenance_yaml({"left_knee": bad}, source="x", date="2026-10-11")


def test_a_rising_friction_still_gives_its_damping() -> None:
    """Coulomb + viscous (MuJoCo's own joint): h rises with speed, so b is fitted, not the prior."""
    rec, _ = stiction.synthetic_ramp("left_knee", frictionloss=1.0, damping=0.6, seed=12)
    fit = stiction.fit_joint(rec, "left_knee")
    assert fit.damping_source == "fit"
    assert fit.frictionloss == pytest.approx(1.0, abs=0.08)
    assert fit.damping == pytest.approx(0.6, abs=0.15)


def test_the_real_profile_carries_the_hw1_friction() -> None:
    """Session 8: the legs' + waist's frictionloss in the real profile = the HW1 values (models/provenance.yaml
    "leg_joint_stiction_hw1" = model.REAL_JOINT_FRICTIONLOSS_HW1's legs / waist), the seeded variants draw inside its
    per-joint range. bad5a5c: the left knee 2.5 (issue #51), no such provenance entry."""
    import yaml
    from r1sim import model, variants

    entry = yaml.safe_load((model.models_dir() / "provenance.yaml").read_text())["parameters"]["leg_joint_stiction_hw1"]
    assert entry["value"] == {n: model.REAL_JOINT_FRICTIONLOSS_HW1[n] for n in J.JOINTS[:14]}
    assert set(entry["value"]) == set(J.JOINTS[:14]) and entry["status"] == "measured"
    for n, (lo, hi) in entry["range"].items():
        assert lo < entry["value"][n] < hi, n
    m = model.build_model("real", visual=False)
    fr = {n: float(m.dof_frictionloss[m.joint(f"{n}_joint").dofadr[0]]) for n in J.JOINTS}
    assert fr["left_knee"] == pytest.approx(0.631) and fr["right_ankle_pitch"] == pytest.approx(2.074)
    # (the arms / head: refit2's B1 values, test_the_real_profile_carries_the_b1_identifications)
    for seed in range(5):
        p = variants.sample_params("real", seed)
        for n, (lo, hi) in entry["range"].items():
            assert lo <= p.joint_frictionloss[n] <= hi, (seed, n)
    # the train profile is unchanged (the training's own values)
    t = model.build_model("train", visual=False)
    assert float(t.dof_frictionloss[t.joint("left_knee_joint").dofadr[0]]) == model.TRAIN_JOINT["leg"][1]


def test_the_real_profile_carries_the_b1_identifications() -> None:
    """Session 8 refit2: in the real profile = models/provenance.yaml real_joint_armature_hw1 / _frictionloss_arms_hw1 /
    _damping_hw1, drawn by the seeded variants inside their ranges: the armature of the arms / head 0.0025 (session 9:
    the legs' / waist's 0.008 applied; ankles 0.10), the arms' / head's friction (shoulder pitch / roll 0.1, the
    others 0.03, head pitch 0.1), the damping of the elbows / head pitch 1.0. Before: armature 0.01 on the arms,
    shoulder friction 2.5, damping 0.05 everywhere."""
    import yaml
    from r1sim import model, variants

    prov = yaml.safe_load((model.models_dir() / "provenance.yaml").read_text())["parameters"]
    e = {k: prov[f"real_joint_{k}_hw1"] for k in ("armature", "damping")}
    e["frictionloss"] = prov["real_joint_frictionloss_arms_hw1"]
    assert e["armature"]["value"] == model.REAL_JOINT_ARMATURE_HW1
    assert e["damping"]["value"] == model.REAL_JOINT_DAMPING_HW1
    arms = {k: v for k, v in model.REAL_JOINT_FRICTIONLOSS_HW1.items() if k not in J.JOINTS[:14]}
    assert e["frictionloss"]["value"] == arms and len(arms) == 12
    for key, ent in e.items():
        assert ent["status"] == "measured"
        for n, (lo, hi) in ent["range"].items():
            assert lo < ent["value"][n] < hi, (key, n)
    m = model.build_model("real", visual=False)

    def dof(n: str, attr: str) -> float:
        return float(getattr(m, attr)[m.joint(f"{n}_joint").dofadr[0]])

    # session 9 (armature): the legs' / waist's identified 0.008 applied (session 8: issue #51's 0.05); ankles 0.10
    assert dof("left_knee", "dof_armature") == 0.008 and dof("left_ankle_pitch", "dof_armature") == 0.10
    assert dof("right_shoulder_pitch", "dof_frictionloss") == 0.1 and dof("right_elbow", "dof_frictionloss") == 0.03
    assert dof("left_elbow", "dof_damping") == 1.0 and dof("left_knee", "dof_damping") == 0.05
    for seed in range(4):
        p = variants.sample_params("real", seed)
        for key, field in (("armature", p.joint_armature), ("damping", p.joint_damping)):
            for n, (lo, hi) in e[key]["range"].items():
                assert lo <= field[n] <= hi, (seed, key, n)
        pm = model.build_model("real", visual=False, params=p)
        got = float(pm.dof_damping[pm.joint("left_elbow_joint").dofadr[0]])
        assert got == pytest.approx(p.joint_damping["left_elbow"], rel=1e-5)  # the XML carries 6 digits
