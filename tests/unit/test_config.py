"""The configuration schema: provenance entries and the robot profile (DESIGN.md §3.6, R-j)."""

from __future__ import annotations

import copy
import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError
from r1mc3.config import (
    ParameterEntry,
    Provenanced,
    ProvenanceFile,
    RobotProfile,
    TuningRefused,
    check_upgrade,
    load_profile,
    load_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "robot_profile.r1_dex3.yaml"
MODEL_PROVENANCE = ROOT / "models" / "provenance.yaml"

# The R1 SDK body order and hg slots (v2 r1mc2/joints.py:18-29, confirmed on our robot from the 28 Sep capture).
SDK_JOINTS = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_roll", "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
    "head_pitch", "head_yaw",
)  # fmt: skip
SDK_SLOTS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30)

EXAMPLE = """
schema: r1mc3.provenance/1
parameters:
  hand_mount_pos_m:
    value: [0.075, 0.0, 0.0]
    unit: m
    status: estimate
    source: "r1mc2 dex3.MOUNT_POS (v2 dex3.py:117-130): ..."
    date: 2026-10-01
    profiles: [train, real]
    until: "hand-eye capture on HW1"
"""


def _raw_profile() -> dict[str, Any]:
    with open(PROFILE, encoding="utf-8") as f:
        d: dict[str, Any] = yaml.safe_load(f)
    return d


@pytest.fixture(scope="module")
def profile() -> RobotProfile:
    return load_profile(PROFILE)


# ------------------------------------------------------------------ provenance entries


def test_provenance_example_entry_validates() -> None:
    f = ProvenanceFile.model_validate(yaml.safe_load(EXAMPLE))
    e = f.parameters["hand_mount_pos_m"]
    assert e.value == [0.075, 0.0, 0.0]
    assert e.status == "estimate" and e.until == "hand-eye capture on HW1"
    assert e.date == datetime.date(2026, 10, 1)
    assert e.profiles == ("train", "real")
    assert set(f.by_status("estimate")) == {"hand_mount_pos_m"}


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"until": None}, "an estimate without `until`"),
        ({"status": "reported", "until": "  "}, "a report with a blank `until`"),
        ({"status": "guess"}, "an unknown status"),
        ({"source": ""}, "an empty source"),
        ({"date": "yesterday"}, "a date that is not a date"),
        ({"profiles": ["sim"]}, "an unknown profile"),
        ({"profiles": []}, "no profile"),
        ({"untill": "x"}, "a misspelt field (extra fields are forbidden)"),
    ],
)
def test_provenance_rejects_bad_entries(change: dict[str, Any], why: str) -> None:
    d = yaml.safe_load(EXAMPLE)
    d["parameters"]["hand_mount_pos_m"].update(change)
    with pytest.raises(ValidationError):
        ProvenanceFile.model_validate(d)
        pytest.fail(f"accepted {why}")


def test_provenance_rejects_bad_file_shapes() -> None:
    d = yaml.safe_load(EXAMPLE)
    with pytest.raises(ValidationError):
        ProvenanceFile.model_validate({**d, "schema": "r1mc3.provenance/2"})
    with pytest.raises(ValidationError):
        ProvenanceFile.model_validate({**d, "parameters": {"Bad-Id": d["parameters"]["hand_mount_pos_m"]}})
    with pytest.raises(ValidationError):
        ProvenanceFile.model_validate({**d, "parameters": {}})


def test_official_and_modelling_choice_need_no_until() -> None:
    for status in ("official", "measured", "derived", "modelling_choice"):
        p = Provenanced[float](value=1.0, status=status, source="s", date=datetime.date(2026, 10, 10))
        assert p.until is None
    ParameterEntry(value={"a": 1}, status="official", source="s", date=datetime.date(2026, 10, 10))


def test_model_provenance_file_validates() -> None:
    """models/provenance.yaml (the simulator's parameters) uses the same schema; it lands with the model agent."""
    if not MODEL_PROVENANCE.exists():
        pytest.skip("models/provenance.yaml not in this tree yet")
    f = load_provenance(MODEL_PROVENANCE)
    assert f.parameters
    for k, e in f.parameters.items():
        if e.status in ("estimate", "reported"):
            assert e.until, k


# ------------------------------------------------------------------ the robot profile


def test_profile_loads(profile: RobotProfile) -> None:
    assert profile.name == "r1_dex3" and profile.version >= 1
    assert profile.rates.writer_hz.value == 500 and profile.rates.writer_fallback_hz.value == 200
    assert profile.rates.policy_hz.value == 50
    assert profile.limits.torque_clip_fraction.value == 0.6


def test_joint_map_is_the_sdk_slot_map(profile: RobotProfile) -> None:
    jm = profile.joint_map
    assert jm.names == SDK_JOINTS
    assert len(set(jm.names)) == 26
    assert jm.slots == SDK_SLOTS
    assert jm.num_slots == 35 and jm.mode_machine == 1 and jm.mode_pr == 0


def test_every_value_has_provenance(profile: RobotProfile) -> None:
    for path, s in profile.scalars():
        assert s.source and s.date, path
        if s.status in ("estimate", "reported"):
            assert s.until, path
    for name, g in profile.gains.items():
        assert g.source, name


def test_policy_gains_are_the_trained_ones(profile: RobotProfile) -> None:
    g = profile.gains["policy_legs_waist"].value
    assert g["left_knee"] == (84.302, 5.367) and g["waist_yaw"] == (84.302, 5.367)
    assert g["right_ankle_roll"] == (31.385, 1.998)
    assert len(g) == 14


def test_profile_is_frozen(profile: RobotProfile) -> None:
    with pytest.raises(ValidationError):
        profile.version = 2
    with pytest.raises(ValidationError):
        profile.safety.fall_detect_deg.value = 60.0
    with pytest.raises(TypeError):
        profile.limits.effort.value["left_knee"] = 999.0  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.gains["stiff"] = profile.gains["arms_manip"]  # type: ignore[index]


def test_tuning_a_tunable_field_within_bounds(profile: RobotProfile) -> None:
    assert set(profile.tunables()) == {"executor.speed_scale"}
    q = profile.with_tuned("executor.speed_scale", 0.5)
    assert q.executor.speed_scale.value == 0.5
    assert profile.executor.speed_scale.value == 1.0  # the original is unchanged
    check_upgrade(profile, q)  # a tunable change needs no version bump


@pytest.mark.parametrize(
    "path",
    ["safety.fall_detect_deg", "rates.writer_hz", "limits.torque_clip_fraction", "executor.keepalive_ms", "nope.x"],
)
def test_tuning_a_non_tunable_field_is_refused(profile: RobotProfile, path: str) -> None:
    with pytest.raises(TuningRefused):
        profile.with_tuned(path, 1.0)


@pytest.mark.parametrize("v", [0.0, 0.09, 1.01, 5.0])
def test_out_of_bounds_tuning_is_refused(profile: RobotProfile, v: float) -> None:
    with pytest.raises(TuningRefused):
        profile.with_tuned("executor.speed_scale", v)


def test_safety_change_needs_a_version_bump(profile: RobotProfile) -> None:
    d = _raw_profile()
    d["safety"]["fall_detect_deg"]["value"] = 35
    changed = RobotProfile.model_validate(d)
    with pytest.raises(ValueError, match="version bump"):
        check_upgrade(profile, changed)
    d["version"] = profile.version + 1
    check_upgrade(profile, RobotProfile.model_validate(d))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("safety", "fall_detect_deg", "until"), None),  # an estimate without `until`
        (("safety", "fall_detect_deg", "tunable"), True),  # safety cannot be tunable
        (("executor", "speed_scale", "bounds"), None),  # tunable without bounds
        (("executor", "speed_scale", "value"), 2.0),  # outside its bounds
        (("rates", "writer_hz", "value"), 1200),  # outside 200-1000
        (("rates", "policy_hz", "value"), 60),  # the writer is not a multiple of the policy rate
        (("safety", "lowstate_age_resend_ms", "value"), 5),  # count < resend < fault broken
        (("joint_map", "num_slots"), 30),  # slot 30 does not fit
        (("schema",), "r1mc3.robot_profile/9"),
    ],
)
def test_invalid_profiles_are_refused(path: tuple[str, ...], value: Any) -> None:
    d = copy.deepcopy(_raw_profile())
    node = d
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(ValidationError):
        RobotProfile.model_validate(d)


def test_joint_sets_must_match_the_joint_map() -> None:
    d = _raw_profile()
    del d["limits"]["effort"]["value"]["head_yaw"]
    with pytest.raises(ValidationError, match=r"limits\.effort"):
        RobotProfile.model_validate(d)
    d = _raw_profile()
    d["joint_map"]["value"][0]["name"] = "left_hip_pitch_x"
    with pytest.raises(ValidationError):
        RobotProfile.model_validate(d)
    d = _raw_profile()
    d["gains"]["stiff"]["value"]["left_hip_pitch"] = [400.0, 3.0]  # above kp_bounds
    with pytest.raises(ValidationError):
        RobotProfile.model_validate(d)


def test_profile_round_trips(profile: RobotProfile) -> None:
    again = RobotProfile.model_validate(profile.model_dump(mode="json", by_alias=True))
    assert again == profile


# ---------------------------------------------------------------- ranges (session 2: parameters as ranges)
def _entry(**kw: Any) -> dict[str, Any]:
    e: dict[str, Any] = {
        "value": 0.454,
        "unit": "kg",
        "status": "measured",
        "source": "kitchen scale",
        "date": "2026-10-09",
    }
    e.update(kw)
    return e


def test_range_scalar_accepts_value_inside() -> None:
    e = ParameterEntry.model_validate(_entry(range=[0.35, 0.6], range_source="weighed ~1 lb, scale resolution"))
    assert e.range == (0.35, 0.6)


def test_range_vector_per_component() -> None:
    e = ParameterEntry.model_validate(
        _entry(value=[0.075, 0.0, 0.0], unit="m", range=[[0.06, 0.09], [-0.01, 0.01], [-0.01, 0.01]], range_source="x")
    )
    assert e.range == ((0.06, 0.09), (-0.01, 0.01), (-0.01, 0.01))


@pytest.mark.parametrize(
    "kw",
    [
        {"range": [0.5, 0.6], "range_source": "x"},  # nominal outside
        {"range": [0.6, 0.35], "range_source": "x"},  # min > max
        {"range": [0.35, 0.6]},  # no source
        {"range_source": "x"},  # source without range
        {"value": [0.1, 0.2], "range": [[0.0, 1.0]], "range_source": "x"},  # wrong component count
        {"value": "a string", "range": [0.0, 1.0], "range_source": "x"},  # non-numeric value
    ],
)
def test_range_rejects(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ParameterEntry.model_validate(_entry(**kw))
