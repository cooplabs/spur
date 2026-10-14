"""models/provenance.yaml: the format the config schema validates, and the values equal the code's constants."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest
import yaml
from r1sim import dex3, model
from r1sim import joints as J

STATUSES = {"official", "measured", "derived", "reported", "estimate", "modelling_choice"}
KEYS = {"value", "unit", "status", "source", "date", "profiles", "until", "range", "range_source"}


@pytest.fixture(scope="module")
def params() -> dict[str, dict[str, Any]]:
    d = yaml.safe_load((model.models_dir() / "provenance.yaml").read_text())
    assert d["schema"] == "r1mc3.provenance/1"
    p: dict[str, dict[str, Any]] = d["parameters"]
    return p


def test_format(params: dict[str, dict[str, Any]]) -> None:
    for k, v in params.items():
        assert k == k.lower() and " " not in k, k
        assert set(v) <= KEYS and {"value", "status", "source", "date", "profiles"} <= set(v), k
        assert v["status"] in STATUSES, k
        assert isinstance(v["date"], dt.date), k
        assert set(v["profiles"]) <= set(model.PROFILES) and v["profiles"], k
        assert len(v["source"]) > 10, k
        if v["status"] in ("estimate", "reported"):
            assert v.get("until"), f"{k}: an estimate needs `until`"


def test_values_match_code(params: dict[str, dict[str, Any]]) -> None:
    v = {k: e["value"] for k, e in params.items()}
    assert model.R1_XML_SHA256 in v["r1_mjcf_file"]
    assert v["r1_effort_limits_nm"] == J.TAU_MAX.tolist()
    assert v["sdk_joint_order_and_slots"] == J.SLOT.tolist()
    assert v["fist_mass_kg"] == dex3.FIST_MASS and v["forearm_com_x_m"] == dex3.FOREARM_COM_X
    assert tuple(v["hand_mount_pos_m"]) == dex3.MOUNT_POS and tuple(v["hand_mount_rpy_rad"]) == dex3.MOUNT_RPY
    lump = v["train_hand_lump"]
    assert lump["mass_kg"] == model.HAND_LUMP_MASS and lump["mount_x_m"] == model.HAND_LUMP_MOUNT_X
    assert tuple(lump["inertia_diag"]) == model.HAND_LUMP_INERTIA
    tj = v["train_joint_armature_frictionloss"]
    assert tuple(tj["legs_waist"]) == model.TRAIN_JOINT["leg"] == model.TRAIN_JOINT["waist"]
    assert tuple(tj["ankles"]) == model.TRAIN_JOINT["ankle"] and tuple(tj["arms"]) == model.TRAIN_JOINT["arm"]
    rj = v["real_joint_armature_frictionloss"]
    assert tuple(rj["hips_knee_waist"]) == model.REAL_JOINT["leg"] == model.REAL_JOINT["waist"]
    assert tuple(rj["ankles"]) == model.REAL_JOINT["ankle"]
    assert tuple(rj["shoulder_pitch_roll"]) == model.REAL_JOINT["shoulder_pr"]
    assert tuple(rj["shoulder_yaw_elbow_wrist_head"]) == model.REAL_JOINT["other"]
    assert v["real_torso_inertial"]["mass_kg"] == model.TORSO_URDF_MASS
    assert tuple(v["real_torso_inertial"]["com_m"]) == model.TORSO_URDF_POS
    assert tuple(v["imu_site_m"]) == model.IMU_URDF_POS
    assert tuple(v["head_camera_site_m"]) == model.HEAD_CAMERA_POS
    assert v["backpack_mass_kg"] == model.BACKPACK_MASS and tuple(v["backpack_pos_m"]) == model.BACKPACK_POS
    assert tuple(v["backpack_box_half_m"]) == model.BACKPACK_HALF
    assert tuple(v["train_floor_friction"]) == model.FLOOR_FRICTION["train"]
    assert tuple(v["real_floor_friction"]) == model.FLOOR_FRICTION["real"]
    for prof in model.PROFILES:
        opts = v[f"{prof}_physics_options"]
        assert {k: str(x) for k, x in opts.items()} == model.OPTIONS[prof]
    hj = v["dex3_joint_params"]
    assert (hj["armature"], hj["damping"], hj["frictionloss"]) == (
        dex3.JOINT_ARMATURE,
        dex3.JOINT_DAMPING,
        dex3.JOINT_FRICTIONLOSS,
    )
    assert v["dex3_fingertip_offset_m"] == dex3.TIP
    assert v["kp_kd_stiff"]["kp"] == J.KP_STIFF.tolist() and v["kp_kd_stiff"]["kd"] == J.KD_STIFF.tolist()
    g = v["dex3_gains"]
    assert np.all(dex3.KP == g["kp"]) and np.all(dex3.KD == g["kd"]) and dex3.RATE_HZ == g["rate_hz"]
    ps = v["pose_stand"]
    assert J.POSE_STAND[J.IDX["left_knee"]] == ps["knee"] == J.POSE_STAND[J.IDX["right_knee"]]
    assert J.POSE_STAND[J.IDX["left_hip_pitch"]] == ps["hip_pitch"]
    assert J.POSE_STAND[J.IDX["left_ankle_pitch"]] == ps["ankle_pitch"]
    assert J.POSE_STAND[J.IDX["left_elbow"]] == ps["elbow"] and J.POSE_STAND[J.IDX["left_shoulder_pitch"]] == 0.22


def test_centred_ankle_values(params: dict[str, dict[str, Any]]) -> None:
    rec = params["centred_stand_ankle_pitch_rad"]["value"]
    for prof in model.PROFILES:
        m = model.build_model(prof, visual=False)
        a = model.centred_stand_pose(m)[J.IDX["left_ankle_pitch"]]
        assert a == pytest.approx(rec[prof], abs=5e-4), prof
