"""r1mc3.measure.preflight: measured poses vs nominal -> pass, or refuse with the words that say what to move (always in
the robot's frame: away / toward the robot, the robot's left / right, counter-clockwise seen from above)."""

from __future__ import annotations

import math

import numpy as np
from r1mc3.measure import PoseCheck, Registration, ValueCheck, run_checks
from r1mc3.measure.tags import rot_z


def reg(x: float, y: float, z: float, yaw_deg: float, sigma_mm: float = 0.5, ok: bool = True) -> Registration:
    T = np.eye(4)
    T[:3, :3] = rot_z(math.radians(yaw_deg))
    T[:3, 3] = [x, y, z]
    s = {"x_mm": sigma_mm, "y_mm": sigma_mm, "z_mm": sigma_mm, "yaw_deg": 0.05}
    return Registration("bin", ok, "ok" if ok else "only 1 of 3 tags seen on the bin (need 2)", T if ok else None,
                        sigma=s)  # fmt: skip


def test_an_in_tolerance_layout_passes() -> None:
    rep = run_checks([PoseCheck("bin", reg(0.281, -0.450, 0.751, 0.4), (0.281, -0.450, 0.751, 0.0))])
    assert rep.ok and rep.refusals == []
    assert rep.summary().startswith("PASS")


def test_out_of_tolerance_says_what_to_move_in_the_robot_frame() -> None:
    rep = run_checks(
        [
            PoseCheck("bin", reg(0.281, -0.471, 0.751, 0.0), (0.281, -0.450, 0.751, 0.0)),
            PoseCheck("board", reg(0.075, 0.0, 0.763, -2.5), (0.091, 0.0, 0.763, 0.0), tol_yaw_deg=1.0),
        ]
    )
    assert not rep.ok
    assert "bin 2.1 cm too far to the robot's right - move it 2.1 cm to the robot's left" in rep.refusals
    assert "board 1.6 cm too close to the robot - move it 1.6 cm away from the robot" in rep.refusals
    assert "board turned 2.5 deg clockwise (seen from above) - turn it 2.5 deg counter-clockwise (seen from above)" in (
        rep.refusals
    )
    assert rep.summary().startswith("REFUSED: ")
    d = rep.as_dict()
    assert d["ok"] is False and len(d["results"]) == 8


def test_an_unmeasured_or_imprecise_entity_refuses_instead_of_judging() -> None:
    bad = run_checks([PoseCheck("bin", reg(0, 0, 0, 0, ok=False), (0.281, -0.45, 0.751, 0.0))])
    assert not bad.ok and "could not measure the bin: only 1 of 3 tags seen" in bad.refusals[0]
    vague = run_checks([PoseCheck("bin", reg(0.281, -0.45, 0.751, 0.0, sigma_mm=8.0), (0.281, -0.45, 0.751, 0.0))])
    assert not vague.ok and "not measured precisely enough" in vague.refusals[0]


def test_value_checks() -> None:
    rep = run_checks([ValueCheck("battery", 18.0, lo=30.0, unit="%", what="battery"), ValueCheck("rate", 1050.0, 900)])
    assert not rep.ok and rep.refusals == ["battery 18 % below 30 - raise the battery above 30 %"]
