"""The leg-stiction fit (r1sim.sysid.stiction) recovers KNOWN friction from a synthetic stand recording.

The recording is the HW1 protocol run on the simulated robot hanging on its stand (pelvis welded, feet off the floor),
through the S2 sensor model (encoder noise 2e-4 rad, dq 0.02 rad/s, tau_est 0.05 N m) and written / read in r1log's
own MCAP format, with the joint's MuJoCo frictionloss / damping set to values that are NOT the model's nominal ones -
the fit must find them, not the nominal. Prepared for HW1: the model itself is unchanged (S5, U4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from r1sim.sysid import records, stiction

# (joint, true frictionloss N m, true damping N m s/rad, tolerance on Fc, tolerance on b)
CASES = [
    ("left_knee", 1.8, 0.5, 0.10, 0.15),  # issue #51's 2.5 would be wrong by 0.7
    ("right_ankle_pitch", 0.9, 0.2, 0.08, 0.12),  # nominal 1.5
    ("left_hip_pitch", 2.5, 0.05, 0.12, 0.40),  # the leg's gravity changes along the sweep: b poorly resolved
    ("right_hip_roll", 0.0, 0.05, 0.15, None),  # the SAFETY configuration (no stiction): Fc must come out ~0
]


@pytest.mark.slow
@pytest.mark.parametrize(("joint", "fc", "b", "tol_fc", "tol_b"), CASES)
def test_stiction_fit_recovers_known_friction(tmp_path: Path, joint: str, fc: float, b: float, tol_fc: float,
                                              tol_b: float | None) -> None:  # fmt: skip
    rec, truth = stiction.synthetic_ramp(joint, frictionloss=fc, damping=b, seed=3)
    assert truth.frictionloss[joint] == fc
    path = tmp_path / "r1log_synthetic.mcap"
    records.write_body_mcap(path, rec)
    back = records.load_body(path)  # r1log's format end to end (float32 records)
    fit = stiction.fit_joint(back, joint)
    assert abs(fit.frictionloss - fc) <= tol_fc, fit
    if tol_b is not None:
        assert abs(fit.damping - b) <= tol_b, fit
    assert fit.residual_rms < 0.35
    # the cross-check with the motor's own torque estimate agrees
    fit_est = stiction.fit_joint(back, joint, source="tau_est")
    assert abs(fit_est.frictionloss - fit.frictionloss) < 0.1


def test_the_fit_does_not_return_the_nominal_value() -> None:
    """A joint simulated at 1.0 N m must not read as issue #51's 2.5 (nominal): the fit sees the data, not the model."""
    rec, _ = stiction.synthetic_ramp("left_knee", frictionloss=1.0, damping=0.05, seed=4)
    fit = stiction.fit_joint(rec, "left_knee")
    assert abs(fit.frictionloss - 1.0) < 0.1
    assert abs(fit.frictionloss - 2.5) > 1.0


def test_the_ramp_plan_is_bounded_and_never_limp() -> None:
    for name in stiction.LEG_JOINTS_FOR_TEST:
        p = stiction.ramp_plan(name)
        assert p.kp > 0 and p.kd > 0  # a soft servo, never kp 0
        assert max(abs(x) for x in p.offset) == pytest.approx(p.amplitude)
        assert p.duration_s < 25.0
        # both directions at both speeds
        assert len(p.speeds) == 2


@pytest.mark.slow
def test_fit_joints_cuts_each_joints_part_of_a_session(tmp_path: Path) -> None:
    import numpy as np

    a, _ = stiction.synthetic_ramp("left_knee", frictionloss=1.5, damping=0.3, seed=5)
    b, _ = stiction.synthetic_ramp("left_ankle_pitch", frictionloss=0.7, damping=0.1, seed=6)
    t = np.concatenate([a.t, b.t + a.t[-1] + 0.002])
    both = records.body_from_arrays(t, {k: np.concatenate([a[k], b[k]]) for k in records.BODY_FIELDS})
    fits = stiction.fit_joints(both, ["left_knee", "left_ankle_pitch", "left_hip_pitch"])
    assert set(fits) == {"left_knee", "left_ankle_pitch"}  # the hip never ramped
    assert abs(fits["left_knee"].frictionloss - 1.5) < 0.1
    assert abs(fits["left_ankle_pitch"].frictionloss - 0.7) < 0.08
    snippet = stiction.provenance_yaml(fits, source="synthetic", date="2026-10-10")
    assert "leg_joint_stiction_hw1" in snippet and "left_knee" in snippet
