"""The recording tool's stiction ramp (r1mc3.record.plans "ramp") is exactly the protocol the HW1 fit assumes
(r1sim.sysid.stiction.ramp_plan): the same knots, amplitude and soft-servo gains for every leg / waist joint - a
recording made with r1d is what the fit was written (and tested on synthetic data) for."""

from __future__ import annotations

import numpy as np
import pytest
from r1mc3.record.plans import plan
from r1mc3.service.codes import GROUP_JOINTS, JOINT_NAMES, group_of_joint

stiction = pytest.importorskip("r1sim.sysid.stiction")


@pytest.mark.parametrize("joint", JOINT_NAMES[:14])
def test_ramp_knots_and_gains_equal_the_fits(joint: str) -> None:
    ours = plan("ramp", {"joint": joint}, np.zeros(26))
    ref = stiction.ramp_plan(joint)
    (k,) = ours.joints
    assert ours.duration_s == pytest.approx(ref.duration_s)
    ts = np.linspace(0.0, ref.duration_s, 2001)
    want = np.array([float(ref.q_cmd(t, 0.0)) for t in ts])
    col = GROUP_JOINTS[group_of_joint(k)].index(k)
    got = np.array([ours.at(t)[0][col] for t in ts])
    np.testing.assert_allclose(got, want, atol=1e-12)
    assert ours.gains is not None
    assert ours.gains[0][col] == ref.kp and ours.gains[1][col] == ref.kd
    assert ours.params["amplitude_rad"] == ref.amplitude
