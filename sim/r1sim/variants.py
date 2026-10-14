"""Seeded model and scene variants within the parameter ranges (session 2; the user, 9 Oct 2026: "can we build it as
a range").

Every estimate of OUR robot in models/provenance.yaml carries a `range` (with its source) next to its nominal value
(a scene's own estimates may do the same: a scene builder draws its variants from the same seed). The builders take
an optional seed:

* no seed -> the nominal model: today's constants, bit-identical (all deterministic tests);
* a seed  -> one variant drawn uniformly within the ranges, reproducible.

What varies per profile:

* **real** (our robot as best known): the estimates of models/provenance.yaml - rear module mass / placement / box,
  fist mass and forearm COM, hand mount (position, rpy), head camera site, body joint armature + frictionloss (per
  joint, within its group's range), Dex3-1 joint armature / damping / frictionloss, hand contact friction, sole and
  floor friction, the IMU's mounting rotation (session 4).
* **train** (as trained): the training's own domain randomisation for the fields the model builder owns
  (policy params/env.yaml `events`, V2_INVENTORY App. E §3.6): foot friction (absolute, shared), joint friction
  (x0.5-3.0) and armature (x0.7-2.5) per joint, torso mass (+-0.5..+3.0 kg), torso COM (+-5 cm per axis), hand
  payload (-0.08..+0.5 kg per hand lump). The estimate ranges are about the real robot, not about the training
  model, so they do not apply here. (PD gains and encoder bias are randomised by the actuators / sensors, not here.)

Reproducibility: each field draws from its own generator, seeded with (seed, crc32(field name)) - so adding a field
later, or reordering the dataclass, does not change the draws of the existing fields. Robot fields are named
"robot.<field>" (a scene's fields "scene.<field>"): one world seed gives independent robot and scene draws.

Ranges are read from the YAML files (one source of truth); the nominal values are the code's constants (checked equal
to the YAML values by tests/unit/test_provenance_model.py and tests/unit/test_ranges.py).
"""

from __future__ import annotations

import dataclasses
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import yaml

from . import dex3, model
from . import joints as J

Vec3 = tuple[float, float, float]
Pair = tuple[float, float]

CORNERS: tuple[str, ...] = ("min", "max", "heavy_back", "heavy_front")

# Training domain randomisation (policy params/env.yaml `events`; V2_INVENTORY App. E §3.6). Checked against env.yaml by
# tests/unit/test_ranges.py.
TRAIN_DR_FOOT_FRICTION: Pair = (0.3, 1.6)  # events.foot_friction: geom_friction abs, axis 0, shared_random
TRAIN_DR_JOINT_FRICTION_SCALE: Pair = (0.5, 3.0)  # events.joint_friction: dof_frictionloss scale, joints .*
TRAIN_DR_JOINT_ARMATURE_SCALE: Pair = (0.7, 2.5)  # events.joint_armature: dof_armature scale, joints .*
TRAIN_DR_TORSO_MASS_ADD: Pair = (-0.5, 3.0)  # events.torso_mass: body_mass add, torso_link (mass only)
TRAIN_DR_HAND_PAYLOAD_ADD: Pair = (-0.08, 0.5)  # events.hand_payload: body_mass add, left/right_dex3_hand
TRAIN_DR_TORSO_COM_ADD: Pair = (-0.05, 0.05)  # events.base_com: body_ipos add per axis, torso_link

# Nominal frictions written only when a variant changes them (r1.xml's collision class / MuJoCo's default).
REAL_FOOT_FRICTION: Vec3 = (1.0, 0.005, 0.0001)
HAND_FRICTION: Vec3 = (1.2, 0.02, 0.002)  # r1sim.dex3.hand_xml geoms and pads
TRAIN_FOOT_FRICTION3: Vec3 = (model.TRAIN_FOOT_FRICTION, 0.005, 0.0001)  # MuJoCo fills torsional / rolling defaults

# Real-profile joint groups: model._joint_group name -> the provenance key of real_joint_armature_frictionloss.
_REAL_GROUP_KEY = {
    "leg": "hips_knee_waist",
    "waist": "hips_knee_waist",
    "ankle": "ankles",
    "shoulder_pr": "shoulder_pitch_roll",
    "other": "shoulder_yaw_elbow_wrist_head",
}


@dataclass(frozen=True)
class ModelParams:
    """Every quantity a robot model variant can change. Nominal = today's constants (r1sim.model / r1sim.dex3)."""

    profile: str
    seed: int | None
    backpack_mass_kg: float
    backpack_pos_m: Vec3
    backpack_half_m: Vec3
    fist_mass_kg: float
    forearm_com_x_m: float
    hand_mount_pos_m: Vec3
    hand_mount_rpy_rad: Vec3
    head_camera_pos_m: Vec3
    joint_armature: Mapping[str, float]  # body joint (SDK name) -> armature
    joint_frictionloss: Mapping[str, float]  # body joint (SDK name) -> frictionloss
    hand_joint_armature: float
    hand_joint_damping: float
    hand_joint_frictionloss: float
    hand_friction: Vec3
    foot_friction: Vec3
    floor_friction: Vec3
    torso_mass_add_kg: float  # train: training DR (0 nominal)
    torso_com_offset_m: Vec3  # train: training DR (0 nominal)
    hand_payload_kg: Pair  # train: (left, right) added to the hand lumps (0 nominal)
    imu_offset_deg: Pair = (0.0, 0.0)  # real: the IMU mounting rotation (roll, pitch), deg; train: none (as trained)
    rear_mass_kg: float = 0.0  # real: the CoM correction behind the torso (model.REAR_MASS_KG, session 5); train: none
    #: real (session 8): body joint (SDK name) -> joint damping, only where HW1 identified one (else r1.xml's 0.05)
    joint_damping: Mapping[str, float] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------- the ranges (read from the YAML files)
@cache
def provenance_ranges() -> dict[str, Any]:
    """The `range` of every models/provenance.yaml entry that has one (entry id -> range as in the file)."""
    d = yaml.safe_load((model.models_dir() / "provenance.yaml").read_text())
    return {k: e["range"] for k, e in d["parameters"].items() if e.get("range") is not None}


# ---------------------------------------------------------------- nominal values
def _body_joints(profile: str) -> list[str]:
    """The commanded body joints a profile's model has (SDK order): train has no head joints (welded)."""
    return [n for n in J.JOINTS if profile == "real" or J.IDX[n] not in J.HEAD]


def _nominal_joint(profile: str, name: str) -> Pair:
    grp = model._joint_group(f"{name}_joint")
    if profile == "train":
        return model.TRAIN_JOINT[grp if grp in ("leg", "ankle", "waist") else "arm"]
    arm, fr = model.REAL_JOINT[grp]
    # session 8: the friction / armature measured at HW1 (legs / waist S4; the rest the B1 replay)
    return model.REAL_JOINT_ARMATURE_HW1.get(name, arm), model.REAL_JOINT_FRICTIONLOSS_HW1.get(name, fr)


def nominal_params(profile: str) -> ModelParams:
    """Today's model: every field = the constant the builder used before ranges existed."""
    model._check_profile(profile)
    names = _body_joints(profile)
    return ModelParams(
        profile=profile,
        seed=None,
        backpack_mass_kg=model.BACKPACK_MASS,
        backpack_pos_m=model.BACKPACK_POS,
        backpack_half_m=model.BACKPACK_HALF,
        fist_mass_kg=dex3.FIST_MASS,
        forearm_com_x_m=dex3.FOREARM_COM_X,
        hand_mount_pos_m=dex3.MOUNT_POS,
        hand_mount_rpy_rad=dex3.MOUNT_RPY,
        head_camera_pos_m=model.HEAD_CAMERA_POS,
        joint_armature={n: _nominal_joint(profile, n)[0] for n in names},
        joint_frictionloss={n: _nominal_joint(profile, n)[1] for n in names},
        hand_joint_armature=dex3.JOINT_ARMATURE,
        hand_joint_damping=dex3.JOINT_DAMPING,
        hand_joint_frictionloss=dex3.JOINT_FRICTIONLOSS,
        hand_friction=HAND_FRICTION,
        foot_friction=TRAIN_FOOT_FRICTION3 if profile == "train" else REAL_FOOT_FRICTION,
        floor_friction=model.FLOOR_FRICTION[profile],
        torso_mass_add_kg=0.0,
        torso_com_offset_m=(0.0, 0.0, 0.0),
        hand_payload_kg=(0.0, 0.0),
        imu_offset_deg=model.IMU_MOUNT_OFFSET_DEG if profile == "real" else (0.0, 0.0),
        rear_mass_kg=model.REAR_MASS_KG if profile == "real" else 0.0,
        joint_damping=dict(model.REAL_JOINT_DAMPING_HW1) if profile == "real" else {},
    )


# ---------------------------------------------------------------- drawing
def _rng(seed: int, field: str) -> np.random.Generator:
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return np.random.default_rng([int(seed), zlib.crc32(field.encode())])


def _pairs(rng: Any) -> list[Pair]:
    """[min, max] -> [(min, max)]; [[min, max], ...] -> one pair per component."""
    items = list(rng)
    if items and not isinstance(items[0], (list, tuple)):
        return [(float(items[0]), float(items[1]))]
    return [(float(a), float(b)) for a, b in items]


class _Draw:
    """Uniform draws (seed) or range corners (pick: "min" / "max" per field) for one variant."""

    def __init__(self, prefix: str, seed: int | None, pick: Mapping[str, str] | None = None) -> None:
        self.prefix, self.seed, self.pick = prefix, seed, pick or {}

    def u(self, field: str, pairs: Sequence[Pair]) -> list[float]:
        if self.seed is None:  # a corner
            side = self.pick.get(field, self.pick.get("*", "min"))
            if side == "nominal":
                raise KeyError(field)
            return [lo if side == "min" else hi for lo, hi in pairs]
        g = _rng(self.seed, f"{self.prefix}.{field}")
        return [lo + (hi - lo) * float(x) for (lo, hi), x in zip(pairs, g.random(len(pairs)), strict=True)]

    def scalar(self, field: str, rng: Any) -> float:
        return self.u(field, _pairs(rng))[0]

    def vec3(self, field: str, rng: Any) -> Vec3:
        v = self.u(field, _pairs(rng))
        return (v[0], v[1], v[2])


def _vary(nom: ModelParams, d: _Draw) -> ModelParams:
    """A variant of `nom` with every field the profile varies drawn by `d` (fields d skips keep the nominal)."""
    upd: dict[str, Any] = {}

    def put(name: str, fn: Any) -> None:
        try:
            upd[name] = fn()
        except KeyError:  # a corner that keeps this field nominal
            pass

    if nom.profile == "real":
        r = provenance_ranges()
        put("backpack_mass_kg", lambda: d.scalar("backpack_mass_kg", r["backpack_mass_kg"]))
        put("backpack_pos_m", lambda: d.vec3("backpack_pos_m", r["backpack_pos_m"]))
        put("backpack_half_m", lambda: d.vec3("backpack_half_m", r["backpack_box_half_m"]))
        put("fist_mass_kg", lambda: d.scalar("fist_mass_kg", r["fist_mass_kg"]))
        put("forearm_com_x_m", lambda: d.scalar("forearm_com_x_m", r["forearm_com_x_m"]))
        put("hand_mount_pos_m", lambda: d.vec3("hand_mount_pos_m", r["hand_mount_pos_m"]))
        put("hand_mount_rpy_rad", lambda: d.vec3("hand_mount_rpy_rad", r["hand_mount_rpy_rad"]))
        put("head_camera_pos_m", lambda: d.vec3("head_camera_pos_m", r["head_camera_site_m"]))
        rj = r["real_joint_armature_frictionloss"]
        names = list(nom.joint_armature)
        grp = [_REAL_GROUP_KEY[model._joint_group(f"{n}_joint")] for n in names]
        # session 8: every joint HW1 identified is drawn around ITS value (per joint: the legs' / waist's friction from
        # provenance "leg_joint_stiction_hw1", the rest "real_joint_armature_hw1" / "real_joint_frictionloss_arms_hw1" /
        # "real_joint_damping_hw1");
        # the others within their group's range as before (the same draw per joint: only the bounds changed)
        legs = r["leg_joint_stiction_hw1"]
        dyn = {k: r[f"real_joint_{k}_hw1"] for k in ("armature", "damping")}
        dyn["frictionloss"] = r["real_joint_frictionloss_arms_hw1"]
        pairs = list(zip(names, grp, strict=True))
        arm_rng = [_pairs(dyn["armature"][n])[0] if n in dyn["armature"] else _pairs(rj[g])[0] for n, g in pairs]
        fr_rng = [
            _pairs(legs[n])[0]
            if n in legs
            else _pairs(dyn["frictionloss"][n])[0]
            if n in dyn["frictionloss"]
            else _pairs(rj[g])[1]
            for n, g in pairs
        ]
        put("joint_armature", lambda: dict(zip(names, d.u("joint_armature", arm_rng), strict=True)))
        put("joint_frictionloss", lambda: dict(zip(names, d.u("joint_frictionloss", fr_rng), strict=True)))
        damped = [n for n in names if n in dyn["damping"]]
        dm_rng = [_pairs(dyn["damping"][n])[0] for n in damped]
        put("joint_damping", lambda: dict(zip(damped, d.u("joint_damping", dm_rng), strict=True)))
        hj = r["dex3_joint_params"]
        put("hand_joint_armature", lambda: d.scalar("hand_joint_armature", hj["armature"]))
        put("hand_joint_damping", lambda: d.scalar("hand_joint_damping", hj["damping"]))
        put("hand_joint_frictionloss", lambda: d.scalar("hand_joint_frictionloss", hj["frictionloss"]))
        put("hand_friction", lambda: d.vec3("hand_friction", r["dex3_contact_params"]["friction"]))
        put("foot_friction", lambda: d.vec3("foot_friction", r["real_foot_contact"]["friction"]))
        put("floor_friction", lambda: d.vec3("floor_friction", r["real_floor_friction"]))
        put("imu_offset_deg", lambda: tuple(d.u("imu_offset_deg", _pairs(r["imu_mount_offset_deg"]))))
        put("rear_mass_kg", lambda: d.scalar("rear_mass_kg", r["rear_mass_kg"]))
    else:
        names = list(nom.joint_armature)
        a_lo, a_hi = TRAIN_DR_JOINT_ARMATURE_SCALE
        f_lo, f_hi = TRAIN_DR_JOINT_FRICTION_SCALE
        put(
            "joint_armature",
            lambda: {
                n: nom.joint_armature[n] * s
                for n, s in zip(names, d.u("joint_armature", [(a_lo, a_hi)] * len(names)), strict=True)
            },
        )
        put(
            "joint_frictionloss",
            lambda: {
                n: nom.joint_frictionloss[n] * s
                for n, s in zip(names, d.u("joint_frictionloss", [(f_lo, f_hi)] * len(names)), strict=True)
            },
        )
        put(
            "foot_friction",
            lambda: (d.scalar("foot_friction", TRAIN_DR_FOOT_FRICTION), *TRAIN_FOOT_FRICTION3[1:]),
        )
        put("torso_mass_add_kg", lambda: d.scalar("torso_mass_add_kg", TRAIN_DR_TORSO_MASS_ADD))
        put("torso_com_offset_m", lambda: d.vec3("torso_com_offset_m", [TRAIN_DR_TORSO_COM_ADD] * 3))
        put(
            "hand_payload_kg",
            lambda: tuple(d.u("hand_payload_kg", [TRAIN_DR_HAND_PAYLOAD_ADD] * 2)),
        )
    return dataclasses.replace(nom, **upd)


def sample_params(profile: str, seed: int) -> ModelParams:
    """One robot variant: every varied field drawn uniformly within its range (reproducible per seed)."""
    nom = nominal_params(profile)
    return dataclasses.replace(_vary(nom, _Draw("robot", int(seed))), seed=int(seed))


# Corner definitions: field -> "min" | "max" | "nominal" ("*" = every other field). "heavy_back" moves mass back /
# hands in (the direction the S1 stand fixture and the 4 Oct fall leaned); "heavy_front" the opposite.
_CORNER_PICKS: dict[str, dict[str, dict[str, str]]] = {
    "real": {
        "min": {"*": "min"},
        "max": {"*": "max"},
        "heavy_back": {
            "*": "nominal",
            "backpack_mass_kg": "max",
            "backpack_pos_m": "min",  # x most backward (y, z also at their minimum)
            "rear_mass_kg": "max",
            "fist_mass_kg": "max",  # removed from the wrist: a heavier fist removed = less forward mass
            "hand_mount_pos_m": "min",
        },
        "heavy_front": {
            "*": "nominal",
            "backpack_mass_kg": "min",
            "backpack_pos_m": "max",
            "rear_mass_kg": "min",
            "fist_mass_kg": "min",
            "hand_mount_pos_m": "max",
        },
    },
    "train": {
        "min": {"*": "min"},
        "max": {"*": "max"},
        "heavy_back": {"*": "nominal", "torso_mass_add_kg": "max", "torso_com_offset_m": "min"},
        "heavy_front": {
            "*": "nominal",
            "torso_mass_add_kg": "max",
            "torso_com_offset_m": "max",
            "hand_payload_kg": "max",
        },
    },
}


def corner_params(profile: str, which: str) -> ModelParams:
    """A deterministic corner of the ranges: "min" / "max" (every varied field at its bound), "heavy_back" /
    "heavy_front" (the mass-placement fields that move the CoM back / forward at their bounds, the rest nominal)."""
    if which not in CORNERS:
        raise ValueError(f"unknown corner {which!r}; use one of {CORNERS}")
    nom = nominal_params(profile)
    return _vary(nom, _Draw("robot", None, _CORNER_PICKS[profile][which]))
