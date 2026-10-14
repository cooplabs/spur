"""The robot profile's `processes` section (version 2, session 6: the source of the robot's r1rt / r1guard config):
present and complete, every joint in damp_kd, never tunable, only the 'stiff' ON_STAND hold."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from r1mc3.config import RobotProfile, load_profile

PROFILE = Path(__file__).resolve().parents[2] / "config" / "robot_profile.r1_dex3.yaml"


def _data() -> dict[str, Any]:
    d: dict[str, Any] = yaml.safe_load(PROFILE.read_text())
    return d


def test_version_2_has_the_processes_section() -> None:
    p = load_profile(PROFILE)
    assert p.version == 2 and p.processes is not None
    assert p.processes.stand_freeze_ramp_s.value == 0.4  # U3 / D15
    assert p.processes.takeover_blend_stand_s.value == 1.0  # DESIGN §3.5 (stand), D22
    assert set(p.processes.damp_kd.value) == set(p.joint_names)
    assert "processes.guard_max_age_ms" in dict(p.scalars())


def test_damp_kd_must_name_every_joint() -> None:
    d = _data()
    d["processes"]["damp_kd"]["value"].pop("head_yaw")
    with pytest.raises(ValidationError, match="damp_kd"):
        RobotProfile.model_validate(d)


def test_processes_values_are_never_tunable() -> None:
    d = _data()
    d["processes"]["stand_freeze_ramp_s"].update({"tunable": True, "bounds": {"min": 0.05, "max": 0.5}})
    with pytest.raises(ValidationError, match="cannot be tunable"):
        RobotProfile.model_validate(d)


def test_only_the_stiff_on_stand_hold() -> None:
    d = _data()
    d["processes"]["on_stand_hold_gains"]["value"] = "soft"
    with pytest.raises(ValidationError):
        RobotProfile.model_validate(d)


def test_a_version_1_profile_without_processes_still_loads() -> None:
    d = _data()
    d.pop("processes")
    d["version"] = 1
    assert RobotProfile.model_validate(d).processes is None
