"""Parameters as ranges (session 2, item 1b): every estimate carries a range with a source, and the builders draw
seeded variants within them; no seed = today's model, unchanged."""

from __future__ import annotations

import dataclasses
import zlib
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from r1mc3.config import ParameterEntry, load_provenance
from r1sim import dex3, model, variants
from r1sim import joints as J

ROOT = Path(__file__).resolve().parents[2]
ENV_YAML = ROOT / "policies" / "r1mc2_manip2_lean_20261006T0421Z" / "params" / "env.yaml"

# Estimates / reported values that carry no range, and why (anything else with these statuses must have one).
NO_RANGE = {
    "dex3_collision_shapes": "a text description of primitive shapes; varying them is a geometry fit (scans/meshes)",
    "dex3_pad_geometry": "a text description of the pad boxes; the 33 per-cell sensors replace it (sensors model)",
    "dex3_gains": "controller command gains (the sim takes kp/kd from each command), not a model parameter",
}


def _pairs(r: Any) -> list[tuple[float, float]]:
    return variants._pairs(r)


def _inside(v: Any, r: Any) -> bool:
    vals = [v] if not isinstance(v, (list, tuple)) else list(v)
    return all(lo - 1e-12 <= float(x) <= hi + 1e-12 for x, (lo, hi) in zip(vals, _pairs(r), strict=True))


# ---------------------------------------------------------------- the files
def test_every_estimate_has_a_range_with_a_source() -> None:
    f = load_provenance(model.models_dir() / "provenance.yaml")  # validates: nominal inside its range, source given
    missing = [
        k
        for k, e in f.parameters.items()
        if e.status in ("estimate", "reported") and e.range is None and k not in NO_RANGE
    ]
    assert not missing, f"estimates without a range: {missing}"
    for k in NO_RANGE:
        assert f.parameters[k].range is None, k
    assert f.parameters["backpack_mass_kg"].range == (0.35, 0.6)  # the weighed ~1 lb (the user, 9 Oct)


def test_mapping_ranges_in_the_schema() -> None:
    base: dict[str, Any] = {
        "value": {"condim": 4, "friction": [1.2, 0.02, 0.002]},
        "status": "estimate",
        "source": "v2 dex3.hand_xml",
        "date": "2026-10-01",
        "until": "Tier B4",
        "range_source": "x",
    }
    ok = ParameterEntry.model_validate({**base, "range": {"friction": [[0.8, 1.6], [0.01, 0.04], [0.001, 0.004]]}})
    assert isinstance(ok.range, dict)
    for bad in (
        {"friction": [[1.3, 1.6], [0.01, 0.04], [0.001, 0.004]]},  # nominal outside
        {"mu": [0.0, 1.0]},  # not a key of the value
        {"friction": [[1.6, 0.8], [0.01, 0.04], [0.001, 0.004]]},  # min > max
    ):
        with pytest.raises(ValidationError):
            ParameterEntry.model_validate({**base, "range": bad})


def _env_events() -> dict[str, Any]:
    class Loader(yaml.SafeLoader):
        pass

    def any_python(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> Any:
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node)  # type: ignore[arg-type]

    Loader.add_multi_constructor("tag:yaml.org,2002:python/", any_python)
    d = yaml.load(ENV_YAML.read_text(), Loader=Loader)  # a SafeLoader subclass: python tags read as plain data
    ev: dict[str, Any] = d["events"]
    return ev


def test_training_dr_constants_equal_env_yaml() -> None:
    ev = _env_events()
    assert tuple(ev["foot_friction"]["params"]["ranges"]) == variants.TRAIN_DR_FOOT_FRICTION
    assert ev["foot_friction"]["params"]["operation"] == "abs" and ev["foot_friction"]["params"]["shared_random"]
    assert tuple(ev["joint_friction"]["params"]["ranges"]) == variants.TRAIN_DR_JOINT_FRICTION_SCALE
    assert tuple(ev["joint_armature"]["params"]["ranges"]) == variants.TRAIN_DR_JOINT_ARMATURE_SCALE
    assert tuple(ev["torso_mass"]["params"]["ranges"]) == variants.TRAIN_DR_TORSO_MASS_ADD
    assert ev["torso_mass"]["params"]["asset_cfg"]["body_names"] == ["torso_link"]
    assert tuple(ev["hand_payload"]["params"]["ranges"]) == variants.TRAIN_DR_HAND_PAYLOAD_ADD
    assert ev["hand_payload"]["params"]["asset_cfg"]["body_names"] == ["left_dex3_hand", "right_dex3_hand"]
    com = ev["base_com"]["params"]["ranges"]
    assert all(tuple(com[k]) == variants.TRAIN_DR_TORSO_COM_ADD for k in (0, 1, 2))
    assert ev["base_com"]["params"]["asset_cfg"]["body_names"] == ["torso_link"]


# ---------------------------------------------------------------- nominal = today's model
@pytest.mark.parametrize("profile", model.PROFILES)
def test_nominal_params_are_the_constants(profile: str) -> None:
    p = variants.nominal_params(profile)
    assert p.backpack_mass_kg == model.BACKPACK_MASS and p.backpack_pos_m == model.BACKPACK_POS
    assert p.backpack_half_m == model.BACKPACK_HALF and p.head_camera_pos_m == model.HEAD_CAMERA_POS
    assert (p.fist_mass_kg, p.forearm_com_x_m) == (dex3.FIST_MASS, dex3.FOREARM_COM_X)
    assert (p.hand_mount_pos_m, p.hand_mount_rpy_rad) == (dex3.MOUNT_POS, dex3.MOUNT_RPY)
    assert (p.hand_joint_armature, p.hand_joint_damping, p.hand_joint_frictionloss) == (
        dex3.JOINT_ARMATURE,
        dex3.JOINT_DAMPING,
        dex3.JOINT_FRICTIONLOSS,
    )
    assert p.floor_friction == model.FLOOR_FRICTION[profile]
    table = model.TRAIN_JOINT if profile == "train" else model.REAL_JOINT
    assert p.joint_armature["left_ankle_pitch"] == table["ankle"][0]
    knee = table["leg"][1] if profile == "train" else model.REAL_JOINT_FRICTIONLOSS_HW1["left_knee"]  # session 8
    assert p.joint_frictionloss["left_knee"] == knee
    assert ("head_yaw" in p.joint_armature) == (profile == "real")
    assert p.torso_mass_add_kg == 0.0 and p.hand_payload_kg == (0.0, 0.0)


@pytest.mark.parametrize("profile", model.PROFILES)
def test_no_seed_builds_the_nominal_model(profile: str) -> None:
    """seed None goes through the same code as explicit nominal params, and the varied-only branches (foot / hand
    friction, torso DR) add nothing. (Checked by hand on 10 Oct 2026: the nominal XML of both profiles, visual on
    and off, was byte-identical to s2 c8c6938, before ranges existed.)"""
    xml = model.robot_xml(profile, visual=False)
    assert xml == model.robot_xml(profile, visual=False, seed=None)
    assert xml == model.robot_xml(profile, visual=False, params=variants.nominal_params(profile))
    if profile == "real":
        assert 'friction="1.2 0.02 0.002"' in xml  # hand_xml's own string, not rewritten
    with pytest.raises(ValueError):
        model.robot_xml(profile, seed=1, params=variants.nominal_params(profile))
    other = "train" if profile == "real" else "real"
    with pytest.raises(ValueError):
        model.robot_xml(profile, params=variants.nominal_params(other))


# ---------------------------------------------------------------- seeded variants
def _real_range_checks(p: variants.ModelParams) -> None:
    r = variants.provenance_ranges()
    assert _inside(p.backpack_mass_kg, r["backpack_mass_kg"])
    assert _inside(p.backpack_pos_m, r["backpack_pos_m"]) and _inside(p.backpack_half_m, r["backpack_box_half_m"])
    assert _inside(p.fist_mass_kg, r["fist_mass_kg"]) and _inside(p.forearm_com_x_m, r["forearm_com_x_m"])
    assert _inside(p.hand_mount_pos_m, r["hand_mount_pos_m"])
    assert _inside(p.hand_mount_rpy_rad, r["hand_mount_rpy_rad"])
    assert _inside(p.head_camera_pos_m, r["head_camera_site_m"])
    rj = r["real_joint_armature_frictionloss"]
    for n in p.joint_armature:
        g = variants._REAL_GROUP_KEY[model._joint_group(f"{n}_joint")]
        (alo, ahi), (flo, fhi) = _pairs(rj[g])
        dyn = {"armature": r["real_joint_armature_hw1"], "frictionloss": r["real_joint_frictionloss_arms_hw1"]}  # s8
        if n in r["leg_joint_stiction_hw1"]:
            flo, fhi = r["leg_joint_stiction_hw1"][n]
        elif n in dyn["frictionloss"]:
            flo, fhi = dyn["frictionloss"][n]
        if n in dyn["armature"]:
            alo, ahi = dyn["armature"][n]
        assert alo <= p.joint_armature[n] <= ahi and flo <= p.joint_frictionloss[n] <= fhi, n
    for n, v in p.joint_damping.items():
        assert _inside(v, r["real_joint_damping_hw1"][n]), n
    hj = r["dex3_joint_params"]
    assert _inside(p.hand_joint_armature, hj["armature"]) and _inside(p.hand_joint_damping, hj["damping"])
    assert _inside(p.hand_joint_frictionloss, hj["frictionloss"])
    assert _inside(p.hand_friction, r["dex3_contact_params"]["friction"])
    assert _inside(p.foot_friction, r["real_foot_contact"]["friction"])
    assert _inside(p.floor_friction, r["real_floor_friction"])


def _train_range_checks(p: variants.ModelParams) -> None:
    nom = variants.nominal_params("train")
    for n, a in p.joint_armature.items():
        s = a / nom.joint_armature[n]
        assert variants.TRAIN_DR_JOINT_ARMATURE_SCALE[0] <= s <= variants.TRAIN_DR_JOINT_ARMATURE_SCALE[1], n
        s = p.joint_frictionloss[n] / nom.joint_frictionloss[n]
        assert variants.TRAIN_DR_JOINT_FRICTION_SCALE[0] <= s <= variants.TRAIN_DR_JOINT_FRICTION_SCALE[1], n
    assert _inside(p.foot_friction[0], variants.TRAIN_DR_FOOT_FRICTION)
    assert _inside(p.torso_mass_add_kg, variants.TRAIN_DR_TORSO_MASS_ADD)
    assert all(_inside(c, variants.TRAIN_DR_TORSO_COM_ADD) for c in p.torso_com_offset_m)
    assert all(_inside(h, variants.TRAIN_DR_HAND_PAYLOAD_ADD) for h in p.hand_payload_kg)
    # the estimate ranges are about the real robot: the train model's other fields stay as trained
    assert (
        dataclasses.replace(
            p,
            seed=None,
            joint_armature=nom.joint_armature,
            joint_frictionloss=nom.joint_frictionloss,
            foot_friction=nom.foot_friction,
            torso_mass_add_kg=0.0,
            torso_com_offset_m=(0.0, 0.0, 0.0),
            hand_payload_kg=(0.0, 0.0),
        )
        == nom
    )


@pytest.mark.parametrize("profile", model.PROFILES)
def test_seeded_variants_are_reproducible_inside_the_ranges(profile: str) -> None:
    check = _real_range_checks if profile == "real" else _train_range_checks
    drawn = [variants.sample_params(profile, s) for s in range(20)]
    for p in drawn:
        check(p)
    assert variants.sample_params(profile, 7) == drawn[7]
    assert model.robot_xml(profile, visual=False, seed=7) == model.robot_xml(profile, visual=False, seed=7)
    assert model.robot_xml(profile, visual=False, seed=7) != model.robot_xml(profile, visual=False, seed=8)
    assert len({p.foot_friction for p in drawn}) == 20
    with pytest.raises(ValueError):
        variants.sample_params(profile, -1)


def test_each_field_draws_from_its_own_stream() -> None:
    """Adding a field later must not change the existing draws: field f of seed s = the first uniform of the
    generator seeded with (s, crc32("robot." + f))."""
    lo, hi = variants.provenance_ranges()["backpack_mass_kg"]
    for s in (0, 5, 123):
        g = np.random.default_rng([s, zlib.crc32(b"robot.backpack_mass_kg")])
        assert variants.sample_params("real", s).backpack_mass_kg == lo + (hi - lo) * float(g.random(1)[0])


@pytest.mark.parametrize("profile", model.PROFILES)
def test_variants_reach_the_compiled_model(profile: str) -> None:
    p = variants.corner_params(profile, "max")
    m = model.build_model(profile, visual=False, params=p)
    m0 = model.build_model(profile, visual=False)

    def jid(mm: mujoco.MjModel, n: str) -> int:
        return int(mm.jnt_dofadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, n)])

    d = jid(m, "left_ankle_pitch_joint")
    assert m.dof_armature[d] == pytest.approx(p.joint_armature["left_ankle_pitch"], rel=1e-5)
    assert m.dof_frictionloss[d] == pytest.approx(p.joint_frictionloss["left_ankle_pitch"], rel=1e-5)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot1_collision")
    assert m.geom_friction[g][0] == pytest.approx(p.foot_friction[0], rel=1e-5)
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if profile == "real":
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "backpack")
        assert m.body_mass[b] == pytest.approx(p.backpack_mass_kg)
        palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_hand_palm_link")
        assert m.body_pos[palm][0] == pytest.approx(p.hand_mount_pos_m[0], abs=1e-6)
        hg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_pad_6")
        assert m.geom_friction[hg][0] == pytest.approx(p.hand_friction[0], rel=1e-5)
        dm = m.body_subtreemass[pel] - m0.body_subtreemass[pel]
        exp = (
            (p.backpack_mass_kg - model.BACKPACK_MASS)
            - 2 * (p.fist_mass_kg - dex3.FIST_MASS)
            + (p.rear_mass_kg - model.REAR_MASS_KG)  # session 5: the CoM fit's rear mass (model.REAR_MASS_KG)
        )
        assert dm == pytest.approx(exp, abs=1e-6)
    else:
        dm = m.body_subtreemass[pel] - m0.body_subtreemass[pel]
        assert dm == pytest.approx(p.torso_mass_add_kg + sum(p.hand_payload_kg), abs=1e-6)


def _com_x_at_stand(m: mujoco.MjModel) -> float:
    d = mujoco.MjData(m)
    d.qpos[:] = model.stand_qpos(m, J.POSE_STAND)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    return float(d.subtree_com[model.RobotIndex(m).pelvis_body][0])


@pytest.mark.parametrize("profile", model.PROFILES)
def test_corners(profile: str) -> None:
    nom = variants.nominal_params(profile)
    lo, hi = variants.corner_params(profile, "min"), variants.corner_params(profile, "max")
    for f in dataclasses.fields(nom):
        v0, a, b = getattr(nom, f.name), getattr(lo, f.name), getattr(hi, f.name)
        if isinstance(v0, float):
            assert a <= v0 <= b, f.name
    x0 = _com_x_at_stand(model.build_model(profile, visual=False))
    xb = _com_x_at_stand(model.build_model(profile, visual=False, params=variants.corner_params(profile, "heavy_back")))
    xf = _com_x_at_stand(
        model.build_model(profile, visual=False, params=variants.corner_params(profile, "heavy_front"))
    )
    assert xb < x0 < xf  # the named corners move the CoM the way their names say
    with pytest.raises(ValueError):
        variants.corner_params(profile, "sideways")
