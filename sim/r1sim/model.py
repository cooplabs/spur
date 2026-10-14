"""Model assembly: the R1 + two Dex3-1 hands, in two profiles (DESIGN.md §12.3).

Ported from r1mc2 (v2) r1mc2/model.py at git r21_try 2a4a165 (the "real" construction) and from the training kit
r1mc2/training/r1mc_task/assets/r1mc_manip_constants.py get_spec_with_hands (the "train" construction); changes:
two profiles; the IMU site at the URDF IMU position and the backpack mass ("real"); per-profile physics options;
no planning-only forearm capsule; the result is an `mujoco.MjSpec` so the scene can be added before compiling.
Every parameter set here is listed with its source in models/provenance.yaml.

Profiles:
  train  - as trained (mjlab 1.2.0 / MuJoCo 3.5.0): Unitree's r1.xml as is (head welded into the torso), the stock
           fist removed and a rigid 0.697 kg Dex3 lump per hand, FULL_COLLISION, the training armature/friction,
           5 ms implicitfast. (The training actuators are MuJoCo <position> actuators; S1 drives every profile with
           torque motors from the simulator's driver block - the actuator profiles are session 2.)
  real   - our robot as best known: head articulated, the articulated Dex3-1 hands with pads, the IMU site at the
           URDF IMU position, the rear computing module (weighed, 0.454 kg) inside the rear handle frame, joint
           armature/friction from unitree_rl_mjlab issue #51 (community-reported), 1 ms implicitfast.

World frame = v2's base frame: on the floor under the middle of the feet, x forward, y left, z up (v2 arms.py:5).
"""

from __future__ import annotations

import functools
import math
import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import numpy.typing as npt

from . import dex3
from . import joints as J

if TYPE_CHECKING:
    from .variants import ModelParams

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

PROFILES: tuple[str, str] = ("train", "real")


def models_dir() -> Path:
    """The repository's models/ folder (override with $R1SIM_MODELS_DIR)."""
    env = os.environ.get("R1SIM_MODELS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "models"


R1_DIR = models_dir() / "r1"
BASE_XML = R1_DIR / "r1_mjlab.xml"
R1_XML_SHA256 = "19c2291ce675217cc193092e59726bbd8d21c53bd5b2599912b60fe538f2de5f"

FOOT_SITES: tuple[str, str] = ("left_foot", "right_foot")
HEAD_CAMERA_SITE = "head_camera"

# ---------------------------------------------------------------- parameters (sources in models/provenance.yaml)
# URDF waist_yaw_link inertial (the torso without the head), used when the head is articulated ("real").
TORSO_URDF_POS = (-0.000648, -0.000052, 0.144497)
TORSO_URDF_MASS = 6.024198
TORSO_URDF_FULLINERTIA = (0.060949, 0.050645, 0.024602, -0.000060, 0.001044, -0.000164)
HEAD_PITCH_POS = (-0.006, 0.03155, 0.255)
HEAD_PITCH_INERTIAL = ((-0.00006956, -0.03077972, 0.08058589), 0.50391253,
                       (0.00095995, 0.00091344, 0.00045356, -0.00000045, 0.00000624, 0.00000588))  # fmt: skip
HEAD_YAW_POS = (0.0010126, -0.03155, 0.1165)
HEAD_YAW_INERTIAL = ((0.02211657, -0.00004725, 0.00199411), 0.71432240,
                     (0.00168182, 0.00202452, 0.00181448, 0.0000001, 0.00029553, 0.0000001))  # fmt: skip
HEAD_CAMERA_POS = (0.08, 0.0, 0.0)  # on head_yaw_link (v2 model.py:70)
# URDF imu_in_pelvis_joint origin (the IMU mesh is placed there in r1.xml:72); r1.xml puts the site at (0, 0, 0).
IMU_URDF_POS = (0.0762025, 2.19914e-05, -0.0884315)
# The IMU's mounting rotation in the pelvis, (roll, pitch) in deg (session 4; models/provenance.yaml
# "imu_mount_offset_deg"). The robot's IMU reads more forward pitch and more roll than its leg encoders imply with the
# soles flat (S3, the A9 finding): per session 0.53-1.12 deg pitch / 0.40-0.63 deg roll in the quiet stance of six
# sessions, 1.17 / 0.52 in the session-15 picks. Nominal = the median of the six sessions' quiet-stance means. The URDF
# mounts the IMU without rotation; r1.xml's site has none (the train profile keeps it so: A1). The site carries the
# rotation, so the orientation, the gyro and the accelerometer all read in the rotated frame, with or without the
# sensor model (it is the mount's geometry, not noise). Set per session with `set_imu_offset` (data of that session).
IMU_MOUNT_OFFSET_DEG = (0.50, 0.96)
# Training's rigid hand lump (r1mc_manip_constants.py: HAND_MASS, HAND_COM, MOUNT_X, inertia, capsule).
# Hand self-contact (session 3, A8 finding; models/provenance.yaml "hand_self_contact"). S1 excluded EVERY pair of a
# hand's links ("the real hand is built not to collide with itself", after v2's sim), so the simulated thumb passed
# through the curled fingers. The real Dex3 does stop on itself: in the 9 Oct false holds (two logged steps) the
# logged thumb_1 stayed at -0.56..-0.63 of its -0.9 command while thumb_2 ran on and the palm pads read nothing (the
# thumb blocked by the curled fingers, not by the board), and v2's own verdict has a rule for pinch tips that close on
# each other (hands.py `_escape` (2), TIP_MEET_M). Only the palm-thumb pairs stay excluded: the palm's collision shape
# over-approximates the real palm exactly where the thumb folds (the primitive box's wrist-end corners stick out up to
# 16 mm past the real taper; the mesh variant's convex hull fills the thumb slot), so palm-thumb_1 overlaps from
# thumb_1 = -0.12..-0.15 rad (primitives; the real thumb closes to -0.56 and beyond freely) and in EVERY pose with the
# hull (7.5 mm), and palm-thumb_2 touches only in the fist preset (1.3-1.7 mm) - artefacts of the shapes, not
# contacts of the real hand (tests/unit/test_hand_self_contact.py). Parent-child pairs are filtered by MuJoCo.
HAND_SELF_EXCLUDES: tuple[tuple[str, str], ...] = (("palm", "thumb_1"), ("palm", "thumb_2"))
HAND_LUMP_MASS = 0.697
HAND_LUMP_COM = (0.0667, 0.008, 0.0)  # y is multiplied by g = -1 (left) / +1 (right)
HAND_LUMP_INERTIA = (6.0e-4, 1.4e-3, 1.2e-3)
HAND_LUMP_MOUNT_X = 0.075
HAND_LUMP_CAPSULE_SIZE = 0.035
# The rear computing module ("backpack": our EDU unit's Jetson Orin NX) sits INSIDE the rear handle frame - the black
# bars - against the back plate (the user, 9 Oct 2026). The frame is in Unitree's torso mesh (waist_yaw_link, torso_link
# frame): bars at x -0.098..-0.090, y +-0.061, z 0.015..0.220; the back plate at x ~-0.068 (centre). The box fills that
# space (test_backpack_sits_inside_the_rear_frame). Mass: weighed by the user on the kitchen scale (9 Oct 2026: "around
# 1 pound"). It replaces the 2 kg of the v2 4 Oct sag fit, which this module does not explain (open item for A2).
BACKPACK_MASS = 0.454
BACKPACK_POS = (-0.083, 0.0, 0.12)  # torso_link frame: the centre of the space inside the rear frame
BACKPACK_HALF = (0.015, 0.055, 0.09)  # 3 x 11 x 18 cm
# The rear mass the CoM fit needs (session 5, A2; models/provenance.yaml "rear_mass_kg"). Three sessions' commanded
# joint loads say our model's CoM sits too far forward: 9 Oct (the quiet stance under the policy: the ankles carry ~2x
# the logged torque), 4 Oct (v2's STAND rocked back on the heels at 5.1 deg for 18 min; the model rocks forward onto
# flat feet) and 6 Oct (the hip-roll load stance: the logged ankle torque even has the opposite sign of the model's).
# r1sim.tier_a.a2_com fits ONE correction to the three together. The physical hypothesis: mass behind the torso that
# the model lacks or misplaces - the expansion module's dock and rails around the weighed computing module, the hands' /
# head's cables routed to it, the quick-release battery's real place in the back (Unitree's "approx. 29 kg with
# battery" matches the URDF sum, but no file says where the battery sits; RESEARCH.md §1.8 mass provenance). Lumped
# at the rear module's place (BACKPACK_POS): a moment, not a weighing - the reaction-board CoM (U6) and a whole-robot
# weighing replace it (added mass if the robot weighs more than the model's 30.4 kg, misplaced mass if not).
REAR_MASS_KG = 1.6
REAR_MASS_POS = BACKPACK_POS

# The stall / fall SAFETY configuration (decision U4, 10 Oct 2026, session 4): the legs' MuJoCo stiction OFF
# (frictionloss 0 on hips / knees / waist / ankles); the nominal model keeps issue #51's 2.5 / 1.5 N·m (identified on
# another, moving, unloaded R1). Why: the real 9 Oct robot fell at the no-stiction model's speed (85.78 deg ~3.2 s
# after the freeze; the frozen stance's growth 2.85 /s vs 2.91 by the rough estimate),
# and the least stable plausible robot is the conservative case for every stall / fall safety test (session 5's "v2
# falls, v3 does not" included). Until HW1's slow torque ramp per leg joint measures the stiction.
SAFETY_LEGS_FRICTIONLOSS: tuple[float, float] = (0.0, 0.0)
# Joint armature / frictionloss per group.
TRAIN_JOINT = {"leg": (0.021354, 1.0), "ankle": (0.00795, 0.8), "waist": (0.021354, 1.0), "arm": (0.01, 0.2)}
REAL_JOINT = {"leg": (0.05, 2.5), "ankle": (0.10, 1.5), "waist": (0.05, 2.5), "shoulder_pr": (0.01, 2.5),
              "other": (0.01, 0.2)}  # fmt: skip
# Session 8 (fix #27): the legs' + waist's frictionloss MEASURED on the robot - HW1's 14 stiction ramps on the stand
# (11 Oct 2026, results/hw1_20261010/end/session_part3/records/*_ramp_*.json; r1sim.sysid.stiction.fit_joint after the
# fix; models/provenance.yaml "leg_joint_stiction_hw1"). They REPLACE issue #51's 2.5 / 1.5 N m for these joints in the
# real profile (the arms keep REAL_JOINT). Fc = the steady sliding friction at the slowest valid ramp speed (0.1 rad/s;
# the ankle pitches 0.3: their 0.1 sweeps never slid steadily) minus the prior damping 0.05 x |v|; the real friction
# FALLS with speed (Stribeck: Fs 0.7-1.1 > h(0.1) > h(0.3)), which MuJoCo's Coulomb frictionloss cannot express - Fc at
# the slow end is the value nearest to the near-standstill joints of a standing robot. L / R averaged where they agree
# within 2 combined standard errors (hip pitch, hip roll), per side where not. Joint damping stays 0.05 (r1.xml; not
# identifiable from the protocol). The ankle rolls never slid steadily (kp 25 vs ~1.4 N m): a loose-gate estimate,
# UNRELIABLE (the reversals' breakaway 1.95 / 1.29 bounds them above). Measured HANGING (~0 bearing load): the friction
# of a standing robot may be higher - the ranges are +-30 % (and the L / R spread), not the loaded case.
REAL_JOINT_FRICTIONLOSS_HW1: dict[str, float] = {
    "left_hip_pitch": 0.585, "left_hip_roll": 0.373, "left_hip_yaw": 0.720, "left_knee": 0.631,
    "left_ankle_pitch": 2.516, "left_ankle_roll": 1.445,
    "right_hip_pitch": 0.585, "right_hip_roll": 0.373, "right_hip_yaw": 0.489, "right_knee": 0.425,
    "right_ankle_pitch": 2.074, "right_ankle_roll": 1.181,
    "waist_roll": 0.516, "waist_yaw": 0.410,
    # session 8 refit2: the arms / head from HW1's S2 steps replayed (r1sim.tier_b.replay, docs/sim/TIER_B.md) - far
    # below issue #51's 2.5 (shoulder pitch / roll) and r1.xml's 0.2: a 60-kp shoulder step that overshoots 8-24 %
    # cannot carry 2.5 N m of friction. The shoulders' nominal: provenance real_joint_dynamics_hw1 "shoulder_note"
    "left_shoulder_pitch": 0.1, "left_shoulder_roll": 0.1, "right_shoulder_pitch": 0.1, "right_shoulder_roll": 0.1,
    "left_shoulder_yaw": 0.03, "left_elbow": 0.03, "left_wrist_roll": 0.03,
    "right_shoulder_yaw": 0.03, "right_elbow": 0.03, "right_wrist_roll": 0.03,
    "head_pitch": 0.1, "head_yaw": 0.03,
}  # fmt: skip
# Session 8 (refit2): the joints' ARMATURE from HW1's step / chirp replay (B1; replay's coarse grid per family): arms /
# head <= 0.0025 (r1.xml / Unitree's placeholder 0.01); the hips / knees / waist ~0.008 (0.005-0.011; issue #51's 0.05
# gives 1.6-1.9 x the real knee step's torque swing). Session 9 (armature): the legs' / waist's 0.008 APPLIED. Session 8
# held it back because two rigs failed at start-up with it; both were rig artefacts, not the armature: the stiction
# fit's synthetic hanging ramp had a broken weld (MjSpec's default equality data = a 1 m anchor: the "hanging" robot
# rolled ~10 deg and stood on one foot; sysid/stiction.py hanging_world), and A2-oct4 posed the robot 0.5 mm ABOVE the
# floor under full stance torques (the unloaded legs whipped for ~10 ms; with the lighter real joints that kick tipped
# the heel-edge stance - posed on the floor it settles as at 0.05: tier_a/a2_oct4.py). The ankles are not identified:
# 0.10.
# Provenance "real_joint_armature_hw1".
REAL_JOINT_ARMATURE_HW1: dict[str, float] = {
    **{f"{s}_{k}": 0.0025 for s in ("left", "right")
       for k in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")},
    "head_pitch": 0.0025, "head_yaw": 0.0025,
    **{f"{s}_{k}": 0.008 for s in ("left", "right") for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee")},
    "waist_roll": 0.008, "waist_yaw": 0.008,
}  # fmt: skip
# Session 8 (refit2): joint DAMPING where the replay identified one (the elbows' and the head pitch's step settling:
# ~1.0 vs r1.xml's 0.05); every other joint keeps r1.xml's 0.05.
REAL_JOINT_DAMPING_HW1: dict[str, float] = {"left_elbow": 1.0, "right_elbow": 1.0, "head_pitch": 1.0}
# The floating base (session 4, A7 finding; models/provenance.yaml "real_floating_base_joint"). r1.xml's global
# `<default><joint damping="0.05" armature="0.01" frictionloss="0.2"/>` is meant for the hinge joints, but MuJoCo
# applies it to `<joint type="free">` too: the pelvis had 0.2 N / 0.2 N·m of dry friction AGAINST THE WORLD on all six
# base DOFs (+ 0.05 damping, 0.01 armature). A free body has no such friction. In the frozen 9 Oct stance it acted on
# exactly the whole-body pitch mode: the frozen-PD stance's unstable rate 0.27/s with it vs 0.83/s without (A7,
# r1sim.tier_a.a7_stability). The train profile keeps it: mjlab built the training robot from the same XML (A1).
FLOATING_BASE_JOINT_REAL = {"damping": 0.0, "armature": 0.0, "frictionloss": 0.0}
TRAIN_FOOT_FRICTION = 0.6
FLOOR_FRICTION = {"train": (1.0, 0.005, 0.0001), "real": (0.8, 0.02, 0.01)}
OPTIONS = {
    "train": dict(timestep="0.005", integrator="implicitfast", solver="Newton", iterations="10",
                  ls_iterations="20", cone="elliptic", impratio="10", tolerance="1e-08", ls_tolerance="0.01",
                  ccd_iterations="50", jacobian="auto", gravity="0 0 -9.81"),
    "real": dict(timestep="0.001", integrator="implicitfast", solver="Newton", iterations="20",
                 ls_iterations="20", cone="elliptic", impratio="10"),
}  # fmt: skip


def _check_profile(profile: str) -> None:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; use one of {PROFILES}")


def _f(v: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{x:.9g}" for x in v)


def _body(root: ET.Element, name: str) -> ET.Element:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(f"body {name} not in r1.xml")


def _joint_group(name: str) -> str:
    if "ankle" in name:
        return "ankle"
    if "hip" in name or "knee" in name:
        return "leg"
    if "waist" in name:
        return "waist"
    if "shoulder_pitch" in name or "shoulder_roll" in name:
        return "shoulder_pr"
    return "other"


def _remove_fist(wr: ET.Element, fist_mass: float = dex3.FIST_MASS, forearm_com_x: float = dex3.FOREARM_COM_X) -> None:
    """Take the stock fist (FIST_MASS) out of the wrist_roll_link inertial: COM x -> FOREARM_COM_X, inertia scaled."""
    iw = wr.find("inertial")
    assert iw is not None
    m0 = float(iw.get("mass", "0"))
    m1 = m0 - fist_mass
    pos = [float(v) for v in iw.get("pos", "0 0 0").split()]
    pos[0] = forearm_com_x
    # Full precision (repr), as training's MjSpec edit sets these floats: with 9 digits the wrist mass and inertia
    # differ from mjlab's model by 1 ulp, enough to break the bit-identical A1 rollouts (s2).
    iw.set("mass", repr(m1))
    iw.set("pos", _f(pos))
    di = [float(v) * m1 / m0 for v in iw.get("diaginertia", "0 0 0").split()]
    iw.set("diaginertia", " ".join(repr(x) for x in di))


@functools.lru_cache(maxsize=8)
def _wrist_mesh_cut(side: str, cut_x: float) -> tuple[str, str]:
    """The stock <side>_wrist_roll_link mesh WITHOUT its fixed fist, as inline MJCF (vertex, face) strings: the
    triangles whose centroid lies beyond the Dex3 mount (x >= cut_x in the link frame) are dropped. The stock mesh
    carries the R1's fist from x ~0.08 m (the curled fingers widen y from -0.029..+0.015 to -0.020..+0.034); with a
    Dex3 attached it poked through the palm in every render. Visual only (the fist's MASS is removed by _remove_fist);
    the file on disk is unchanged."""
    b = (R1_DIR / "meshes" / f"{side}_wrist_roll_link.STL").read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(b[84 : 84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    tri = rec["v"].astype(np.float64)
    tri = tri[tri[:, :, 0].mean(axis=1) < cut_x]
    verts, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    return " ".join(f"{v:.6g}" for v in verts.ravel()), " ".join(str(int(i)) for i in faces.ravel())


def _hand_visuals(root: ET.Element) -> None:
    """Real profile: draw the Dex3-1 hands with Unitree's link meshes and cut the stock fist off the wrist meshes
    (renders only). hand_xml(meshes=True) already put a mesh geom on every hand link; this adds the assets + the
    material and swaps the wrist meshes. Visual geoms never collide and carry no mass: the compiled physics is the
    same (tests/unit/test_dex3_visual.py rolls the model out with and without them, bit for bit).

    NOT in the train profile: its model must stay field-for-field mjlab's (tests/validation/test_a1_sim2sim.py
    compares every geom at tolerance 0, visual ones included), so the train renders keep training's look (the stock
    fist, an invisible hand lump) - session 3 finding, said in S3.md."""
    asset = root.find("asset")
    assert asset is not None
    ET.SubElement(asset, "material", dex3.VISUAL_MATERIAL_ATTRS)
    for side in dex3.SIDES:
        for a in dex3.mesh_assets(side):
            ET.SubElement(asset, "mesh", a)
        wr = _body(root, f"{side}_wrist_roll_link")
        palm = _body(wr, f"{side}_hand_palm_link")
        cut_x = float(palm.get("pos", "0 0 0").split()[0])  # the Dex3 mount (a seeded variant moves it)
        vert, face = _wrist_mesh_cut(side, cut_x)
        ET.SubElement(asset, "mesh", name=f"{side}_wrist_roll_link_nofist", vertex=vert, face=face)
        for g in wr.findall("geom"):
            if g.get("mesh") == f"{side}_wrist_roll_link":
                g.set("mesh", f"{side}_wrist_roll_link_nofist")


def _params(profile: str, seed: int | None, params: ModelParams | None) -> ModelParams:
    """The parameters of the model to build: `params` if given, else the seed's variant, else the nominal model."""
    from . import variants

    if params is not None:
        if seed is not None:
            raise ValueError("give a seed or params, not both")
        if params.profile != profile:
            raise ValueError(f"params are for profile {params.profile!r}, not {profile!r}")
        return params
    return variants.nominal_params(profile) if seed is None else variants.sample_params(profile, seed)


def robot_xml(
    profile: str = "real", *, visual: bool = True, seed: int | None = None, params: ModelParams | None = None
) -> str:
    """The robot (+ floor and light) as an MJCF string for `profile`. seed None (and no params) = the nominal model;
    a seed = one variant within the parameter ranges (r1sim.variants); params = exactly these parameters."""
    _check_profile(profile)
    from . import variants

    p = _params(profile, seed, params)
    tree = ET.parse(BASE_XML)
    root = tree.getroot()
    root.set("model", f"r1mc3_{profile}")
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str(R1_DIR / "meshes"))

    contact = root.find("contact")
    assert contact is not None
    for ex in list(contact):  # the two excludes that name bodies r1.xml does not have
        if "wrist_pitch" in ex.get("body1", "") + ex.get("body2", ""):
            contact.remove(ex)

    torso = _body(root, "torso_link")
    if profile == "real":
        ti = torso.find("inertial")
        assert ti is not None
        for attr in ("quat", "diaginertia"):
            ti.attrib.pop(attr, None)
        ti.set("pos", _f(TORSO_URDF_POS))
        ti.set("mass", f"{TORSO_URDF_MASS}")
        ti.set("fullinertia", _f(TORSO_URDF_FULLINERTIA))
        for g in list(torso.findall("geom")):
            if g.get("mesh") in ("head_pitch_link", "head_yaw_link") or g.get("name") == "head_collision":
                torso.remove(g)
        hp = ET.SubElement(torso, "body", name="head_pitch_link", pos=_f(HEAD_PITCH_POS))
        ET.SubElement(hp, "inertial", pos=_f(HEAD_PITCH_INERTIAL[0]), mass=f"{HEAD_PITCH_INERTIAL[1]}",
                      fullinertia=_f(HEAD_PITCH_INERTIAL[2]))  # fmt: skip
        ET.SubElement(hp, "joint", name="head_pitch_joint", pos="0 0 0", axis="0 1 0", range="-0.62832 0.62832")
        ET.SubElement(hp, "geom", {"class": "visual", "rgba": "0.79216 0.81961 0.93333 1", "mesh": "head_pitch_link"})
        hy = ET.SubElement(hp, "body", name="head_yaw_link", pos=_f(HEAD_YAW_POS))
        ET.SubElement(hy, "inertial", pos=_f(HEAD_YAW_INERTIAL[0]), mass=f"{HEAD_YAW_INERTIAL[1]}",
                      fullinertia=_f(HEAD_YAW_INERTIAL[2]))  # fmt: skip
        ET.SubElement(hy, "joint", name="head_yaw_joint", pos="0 0 0", axis="0 0 1", range="-2.0071 2.0071")
        ET.SubElement(hy, "geom", {"class": "visual", "rgba": "0.79216 0.81961 0.93333 1", "mesh": "head_yaw_link"})
        ET.SubElement(hy, "geom", {"name": "head_collision", "class": "collision", "type": "sphere",
                                   "size": "0.055", "pos": "0.02 0 -0.0215"})  # fmt: skip
        ET.SubElement(hy, "site", name=HEAD_CAMERA_SITE, pos=_f(p.head_camera_pos_m), size="0.01")
        # The rear computing module inside the rear handle frame: a mass the URDF lacks (estimate until weighed).
        bp_pos = list(p.backpack_pos_m)
        hx, hy_, hz = p.backpack_half_m
        m = p.backpack_mass_kg
        inertia = (m / 3 * (hy_**2 + hz**2), m / 3 * (hx**2 + hz**2), m / 3 * (hx**2 + hy_**2))
        bp = ET.SubElement(torso, "body", name="backpack", pos=_f(bp_pos))
        ET.SubElement(bp, "inertial", pos="0 0 0", mass=f"{m}", diaginertia=_f(inertia))
        ET.SubElement(bp, "geom", {"name": "backpack_visual", "type": "box", "size": _f(p.backpack_half_m),
                                   "contype": "0", "conaffinity": "0", "density": "0", "group": "1",
                                   "rgba": "0.25 0.25 0.28 1"})  # fmt: skip
        if p.rear_mass_kg > 0.0:  # the CoM correction (REAR_MASS_KG): an inertial only, the rear module's box size
            mr = p.rear_mass_kg
            ri = (mr / 3 * (hy_**2 + hz**2), mr / 3 * (hx**2 + hz**2), mr / 3 * (hx**2 + hy_**2))
            rb = ET.SubElement(torso, "body", name="rear_mass", pos=_f(REAR_MASS_POS))
            ET.SubElement(rb, "inertial", pos="0 0 0", mass=f"{mr}", diaginertia=_f(ri))
        # The IMU site at the URDF IMU position (the sensors are attached to the site), rotated by the IMU's mounting
        # offset (IMU_MOUNT_OFFSET_DEG: what the robot's IMU reads beyond the body's lean).
        pelvis = _body(root, "pelvis")
        for st in pelvis.findall("site"):
            if st.get("name") == "imu":
                st.set("pos", _f(IMU_URDF_POS))
                if any(p.imu_offset_deg):
                    st.set("quat", _f(imu_offset_quat(*p.imu_offset_deg).tolist()))
    else:
        # Head welded (as trained): the camera site on the torso where the articulated model has it at q = 0.
        cam = [HEAD_PITCH_POS[i] + HEAD_YAW_POS[i] + p.head_camera_pos_m[i] for i in range(3)]
        ET.SubElement(torso, "site", name=HEAD_CAMERA_SITE, pos=_f(cam), size="0.01")
        if p.torso_mass_add_kg != 0.0 or any(p.torso_com_offset_m):  # training DR (mass only, as mjlab body_mass)
            ti = torso.find("inertial")
            assert ti is not None
            ti.set("mass", f"{float(ti.get('mass', '0')) + p.torso_mass_add_kg:.9g}")
            tpos = [float(v) for v in ti.get("pos", "0 0 0").split()]
            ti.set("pos", _f([tpos[i] + p.torso_com_offset_m[i] for i in range(3)]))

    for side in dex3.SIDES:
        wr = _body(root, f"{side}_wrist_roll_link")
        _remove_fist(wr, p.fist_mass_kg, p.forearm_com_x_m)
        sgn = -1.0 if side == "left" else 1.0
        if profile == "real":
            for geom in list(wr.findall("geom")):
                if geom.get("name") == f"{side}_hand_collision":
                    wr.remove(geom)  # the dummy hand's collision capsule
            for st in list(wr.findall("site")):
                if st.get("name") == f"{side}_palm":
                    wr.remove(st)
            hand = ET.fromstring(
                dex3.hand_xml(
                    side,
                    mount_pos=p.hand_mount_pos_m,
                    mount_rpy=p.hand_mount_rpy_rad,
                    armature=p.hand_joint_armature,
                    damping=p.hand_joint_damping,
                    frictionloss=p.hand_joint_frictionloss,
                    meshes=visual,
                )
            )
            if p.hand_friction != variants.HAND_FRICTION:
                for g in hand.iter("geom"):
                    if g.get("friction") is not None:
                        g.set("friction", _f(p.hand_friction))
            wr.append(hand)
            if any(g.get("mesh", "").endswith("_cmesh") for g in hand.iter("geom")):  # dex3 collision "mesh" variant
                asset_el = root.find("asset")
                assert asset_el is not None
                for cma in dex3.collision_mesh_assets(side):
                    ET.SubElement(asset_el, "mesh", cma)
            ET.SubElement(contact, "exclude", body1=f"{side}_elbow_link", body2=f"{side}_hand_palm_link")
            for a, b in HAND_SELF_EXCLUDES:  # the thumb and the fingers DO touch each other (session 3, A8)
                ET.SubElement(contact, "exclude", body1=f"{side}_hand_{a}_link", body2=f"{side}_hand_{b}_link")
        else:
            hand = ET.SubElement(wr, "body", name=f"{side}_dex3_hand", pos=_f((HAND_LUMP_MOUNT_X, 0.0, 0.0)))
            com = (HAND_LUMP_COM[0], sgn * HAND_LUMP_COM[1], HAND_LUMP_COM[2])
            lump_mass = HAND_LUMP_MASS + p.hand_payload_kg[0 if side == "left" else 1]
            ET.SubElement(hand, "inertial", pos=_f(com), mass=f"{lump_mass}", diaginertia=_f(HAND_LUMP_INERTIA))
            ET.SubElement(hand, "geom", {"name": f"{side}_dex3_collision", "class": "collision", "type": "capsule",
                                         "fromto": _f((0.015, 0.0, 0.0, 0.15, sgn * 0.01, 0.0)),
                                         "size": f"{HAND_LUMP_CAPSULE_SIZE}"})  # fmt: skip

    # Joint armature / frictionloss per group and profile (body joints only; the hand joints carry their own).
    for j in root.iter("joint"):
        n = j.get("name", "")
        if not n.endswith("_joint") or n == "floating_base_joint" or "_hand_" in n:
            continue
        grp = _joint_group(n)
        if profile == "train":
            arm, fr = TRAIN_JOINT[grp if grp in ("leg", "ankle", "waist") else "arm"]
        else:
            arm, fr = REAL_JOINT[grp]
            arm = REAL_JOINT_ARMATURE_HW1.get(n[: -len("_joint")], arm)
            fr = REAL_JOINT_FRICTIONLOSS_HW1.get(n[: -len("_joint")], fr)
        arm = p.joint_armature.get(n[: -len("_joint")], arm)
        fr = p.joint_frictionloss.get(n[: -len("_joint")], fr)
        j.set("armature", f"{arm:g}")
        j.set("frictionloss", f"{fr:g}")
        if n[: -len("_joint")] in p.joint_damping:  # session 8: only the identified ones (else r1.xml's 0.05)
            j.set("damping", f"{p.joint_damping[n[: -len('_joint')]]:g}")
    if profile == "real":  # session 4 (A7): no joint friction / damping / armature on the free-floating base
        fb = next(j for j in root.iter("joint") if j.get("name") == "floating_base_joint")
        for attr, v in FLOATING_BASE_JOINT_REAL.items():
            fb.set(attr, f"{v:g}")

    if profile == "train":  # mjlab FULL_COLLISION (unitree_rl_mjlab r1_constants.py)
        for parent in root.iter():
            for geom in parent.findall("geom"):
                gn = geom.get("name", "")
                if not gn.endswith("_collision"):
                    continue
                if "_foot" in gn:
                    geom.set("condim", "3")
                    geom.set("priority", "1")
                    tf = p.foot_friction
                    geom.set("friction", f"{TRAIN_FOOT_FRICTION:g}" if tf == variants.TRAIN_FOOT_FRICTION3 else _f(tf))
                else:
                    geom.set("condim", "1")
                    geom.set("priority", "0")  # mjlab CollisionCfg priority default (r1.xml's class has 1)

    if profile == "real" and p.foot_friction != variants.REAL_FOOT_FRICTION:  # a variant's sole friction
        for parent in root.iter():
            for geom in parent.findall("geom"):
                gn = geom.get("name", "")
                if gn.endswith("_collision") and "_foot" in gn:
                    geom.set("friction", _f(p.foot_friction))

    if visual and profile == "real":
        _hand_visuals(root)
    if not visual:
        asset = root.find("asset")
        assert asset is not None
        for mesh in list(asset.findall("mesh")):
            if not mesh.get("name", "").endswith("_cmesh"):  # collision meshes (dex3 "mesh" variant) stay
                asset.remove(mesh)
        for parent in root.iter():
            for geom in list(parent.findall("geom")):
                if geom.get("class") == "visual" or (geom.get("mesh") and not geom.get("mesh", "").endswith("_cmesh")):
                    parent.remove(geom)

    opt = ET.Element("option", OPTIONS[profile])
    root.insert(1, opt)
    world = root.find("worldbody")
    assert world is not None
    ET.SubElement(world, "light", name="sun", pos="0 0 3", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", rgba="0.35 0.4 0.45 1",
                  friction=_f(p.floor_friction))  # fmt: skip

    act = ET.SubElement(root, "actuator")
    present = {j.get("name") for j in root.iter("joint")}
    for i, n in enumerate(J.JOINTS):
        if f"{n}_joint" not in present:
            continue
        t = J.TAU_MAX[i]
        if profile == "train":  # mjlab's <position> actuators (train_position_actuator, end of this file)
            act.append(train_position_actuator(root, n))
            continue
        ET.SubElement(act, "motor", name=n, joint=f"{n}_joint", ctrlrange=f"{-t:g} {t:g}", ctrllimited="true")
    if profile == "real":
        for side in dex3.SIDES:
            for k, (jn, an) in enumerate(zip(dex3.joint_names(side), dex3.actuator_names(side), strict=True)):
                t = dex3.TAU_MAX[k]
                ET.SubElement(act, "motor", name=an, joint=jn, ctrlrange=f"{-t:g} {t:g}", ctrllimited="true")
    return ET.tostring(root, encoding="unicode")


def imu_offset_quat(roll_deg: float, pitch_deg: float) -> FloatArray:
    """The IMU's mounting rotation in the pelvis (w, x, y, z): roll about x, then pitch about the new y. With the site
    rotated by it the IMU reports orientation = the pelvis's x this rotation, so + pitch = the IMU reads more FORWARD
    lean than the body has (v2's sign; = the session-3 a7_fall / a9 `imu_bias_quat`)."""
    hr, hp = math.radians(roll_deg) / 2.0, math.radians(pitch_deg) / 2.0
    qr = np.array([math.cos(hr), math.sin(hr), 0.0, 0.0])
    qp = np.array([math.cos(hp), 0.0, math.sin(hp), 0.0])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qr, qp)
    return out


def set_imu_offset(m: mujoco.MjModel, roll_deg: float, pitch_deg: float) -> None:
    """Set the IMU's mounting offset of a compiled real-profile model in place (a session's own measured offset, e.g.
    the 9 Oct fall session's (0.63, 0.53) deg; the nominal is built in). Same rotation as the XML builder writes."""
    sid = _id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    if sid < 0:
        raise ValueError("the model has no imu site")
    m.site_quat[sid] = imu_offset_quat(roll_deg, pitch_deg)


def imu_offset_of(m: mujoco.MjModel) -> tuple[float, float]:
    """The (roll, pitch) in deg of a compiled model's IMU mounting rotation (the inverse of `imu_offset_quat`)."""
    sid = _id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(m.site_quat[sid], dtype=np.float64))
    R = r.reshape(3, 3)
    return math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(-math.asin(float(np.clip(R[2, 0], -1, 1))))


def robot_spec(
    profile: str = "real", *, visual: bool = True, seed: int | None = None, params: ModelParams | None = None
) -> mujoco.MjSpec:
    """The robot + floor + light as an editable MjSpec (add the scene to `spec.worldbody`, then compile)."""
    return mujoco.MjSpec.from_string(robot_xml(profile, visual=visual, seed=seed, params=params))


def build_model(
    profile: str = "real", *, visual: bool = True, seed: int | None = None, params: ModelParams | None = None
) -> mujoco.MjModel:
    return robot_spec(profile, visual=visual, seed=seed, params=params).compile()


def _id(m: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    return int(mujoco.mj_name2id(m, kind, name))


class RobotIndex:
    """Index bookkeeping between the 26-joint SDK vector (and the 2 x 7 hand vectors) and a compiled model."""

    def __init__(self, m: mujoco.MjModel) -> None:
        self.m = m
        jid = [_id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint") for n in J.JOINTS]
        self.present: npt.NDArray[np.bool_] = np.array([j >= 0 for j in jid], dtype=bool)
        self.qadr: IntArray = np.array([int(m.jnt_qposadr[j]) if j >= 0 else -1 for j in jid], dtype=np.int64)
        self.vadr: IntArray = np.array([int(m.jnt_dofadr[j]) if j >= 0 else -1 for j in jid], dtype=np.int64)
        self.act: IntArray = np.array([_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in J.JOINTS], dtype=np.int64)
        self.tau_max: FloatArray = J.TAU_MAX.copy()
        self.hand_qadr: dict[str, IntArray] = {}
        self.hand_vadr: dict[str, IntArray] = {}
        self.hand_act: dict[str, IntArray] = {}
        self.hand_tau_max: FloatArray = dex3.TAU_MAX.copy()
        if _id(m, mujoco.mjtObj.mjOBJ_BODY, "left_hand_palm_link") >= 0:
            for side in dex3.SIDES:
                hj = [_id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in dex3.joint_names(side)]
                self.hand_qadr[side] = np.array([int(m.jnt_qposadr[j]) for j in hj], dtype=np.int64)
                self.hand_vadr[side] = np.array([int(m.jnt_dofadr[j]) for j in hj], dtype=np.int64)
                self.hand_act[side] = np.array(
                    [_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in dex3.actuator_names(side)], dtype=np.int64
                )
        self.pelvis_body = _id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_body = _id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.imu_site = _id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.foot_sites: tuple[int, int] = (
            _id(m, mujoco.mjtObj.mjOBJ_SITE, FOOT_SITES[0]),
            _id(m, mujoco.mjtObj.mjOBJ_SITE, FOOT_SITES[1]),
        )
        self.head_camera_site = _id(m, mujoco.mjtObj.mjOBJ_SITE, HEAD_CAMERA_SITE)
        self.base_qadr = int(m.jnt_qposadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")])
        self.total_mass = float(m.body_subtreemass[self.pelvis_body])
        self.foot_geoms: dict[str, list[int]] = {
            side: [_id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot{k}_collision") for k in range(1, 8)]
            for side in ("left", "right")
        }
        self.profile: str | None
        if _id(m, mujoco.mjtObj.mjOBJ_BODY, "backpack") >= 0:
            self.profile = "real"
        elif _id(m, mujoco.mjtObj.mjOBJ_BODY, "left_dex3_hand") >= 0:
            self.profile = "train"
        else:
            self.profile = None


def _lowest_foot_z(m: mujoco.MjModel, d: mujoco.MjData, ix: RobotIndex) -> float:
    """The lowest point of the foot capsules (world z) for the current kinematics."""
    lo = np.inf
    for gids in ix.foot_geoms.values():
        for g in gids:
            r, half = float(m.geom_size[g][0]), float(m.geom_size[g][1])
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
            c = d.geom_xpos[g]
            lo = min(lo, float(c[2] - abs(axis[2]) * half - r))
    return float(lo)


def stand_qpos(
    m: mujoco.MjModel, pose: FloatArray = J.POSE_STAND, *, clearance: float = 0.001, level_feet: bool = True
) -> FloatArray:
    """qpos of the robot standing at `pose`: base yaw 0, the soles level in pitch (the base pitched by the feet's mean
    pitch; at POSE_STAND with the base upright the toes point 2.7 deg down), the lowest foot point `clearance` above
    z = 0, the midpoint of the two foot sites at world x = y = 0, hands open. Other free bodies keep qpos0."""
    ix = RobotIndex(m)
    d = mujoco.MjData(m)
    q = m.qpos0.copy()
    b = ix.base_qadr
    q[b : b + 3] = (0.0, 0.0, 0.8)
    q[b + 3 : b + 7] = (1.0, 0.0, 0.0, 0.0)
    for i in range(J.N):
        if ix.present[i]:
            q[ix.qadr[i]] = pose[i]
    for side, adr in ix.hand_qadr.items():
        q[adr] = dex3.preset("open", side)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    if level_feet:
        pitch = 0.0
        for s in ix.foot_sites:
            r = d.xmat[int(m.site_bodyid[s])].reshape(3, 3)
            pitch += float(np.arctan2(r[2, 0], r[2, 2])) / 2.0  # foot x-axis elevation (+ = toes up)
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 1.0, 0.0]), pitch)  # +y rotation tips the toes down by pitch
        q[b + 3 : b + 7] = quat
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
    feet = 0.5 * (d.site_xpos[ix.foot_sites[0]] + d.site_xpos[ix.foot_sites[1]])
    q[b] -= feet[0]
    q[b + 1] -= feet[1]
    q[b + 2] += clearance - _lowest_foot_z(m, d, ix)
    return np.asarray(q, dtype=np.float64)


def centred_stand_pose(m: mujoco.MjModel, pose: FloatArray = J.POSE_STAND, *, tol: float = 1e-5) -> FloatArray:
    """`pose` with both ankle pitches shifted by the same amount so that, with level soles (stand_qpos), the
    whole-body CoM lies over the midpoint of the foot sites (x).

    Why: POSE_STAND is the policy's default pose, i.e. Unitree's stand lock under an ACTIVE balance controller. With
    level soles its CoM is 3.8 cm (train) / 3.9 cm (real) behind the foot centre, and a plain stiff joint
    PD lets it sag and fall backwards (S1 check; the 4 Oct real-robot fall on `lock`, v2 README.md:310-318, is the
    same mechanism). A balance-free stand hold therefore uses this pose (S1: ankle pitch -0.204 train, -0.206 real).
    """
    ix = RobotIndex(m)
    d = mujoco.MjData(m)
    ankles = [J.IDX["left_ankle_pitch"], J.IDX["right_ankle_pitch"]]

    def com_x(delta: float) -> tuple[float, FloatArray]:
        p = pose.copy()
        p[ankles] += delta
        d.qpos[:] = stand_qpos(m, p)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return float(d.subtree_com[ix.pelvis_body][0]), p

    lo, hi = -0.4, 0.4  # com_x(lo) > 0 > com_x(hi): more ankle dorsiflexion (negative) leans the body forward
    if not com_x(lo)[0] > 0.0 > com_x(hi)[0]:
        raise RuntimeError("centred_stand_pose: the CoM cannot be centred with the ankle pitch alone")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if com_x(mid)[0] > 0.0:
            lo = mid
        else:
            hi = mid
    return com_x(0.5 * (lo + hi))[1]


# ---------------------------------------------------------------- train-equivalent profile (session 2)
# mjlab 1.2.0 BuiltinPositionActuatorCfg -> mjlab/utils/spec.py create_position_actuator: a <position> actuator per
# joint, gainprm[0] = kp, biasprm = [0, -kp, -kv], forcerange +-effort, ctrllimited False with inheritrange 0
# (setpoints beyond the joint range allowed), an informational ctrlrange [lo - effort / kp, hi + effort / kp]. Gains
# as the policy's params/env.yaml scene.entities.robot.articulation.actuators (lines 119-187): legs / waist / ankles
# by mjlab's armature rule kp = a * wn^2, kv = 2 zeta a wn (wn = 2 pi 10, zeta = 2; v2 training
# r1mc_constants.py:20-30, the same float expression, so the values are bit-identical to env.yaml's), the arms the
# hardware KP_ARM / KD_ARM
# (r1mc_manip_constants.py:38-40). The armature / frictionloss the same configs set are TRAIN_JOINT.
_WN = 2.0 * math.pi * 10.0
_ZETA = 2.0


def _armature_rule(a: float) -> tuple[float, float]:
    return a * _WN**2, 2.0 * _ZETA * a * _WN


TRAIN_PD: dict[str, tuple[float, float, float]] = {  # group -> (kp, kv, effort limit N m)
    "leg": (*_armature_rule(0.021354), 60.0),
    "ankle": (*_armature_rule(0.00795), 50.0),
    "waist": (*_armature_rule(0.021354), 60.0),
    "shoulder_pr": (60.0, 2.0, 60.0),
    "shoulder_yaw_elbow": (45.0, 2.0, 33.0),
    "wrist": (30.0, 1.5, 33.0),
}


def train_pd_group(name: str) -> str:
    """TRAIN_PD group of an SDK joint name (env.yaml's target_names_expr)."""
    if "ankle" in name:
        return "ankle"
    if "hip" in name or "knee" in name:
        return "leg"
    if "waist" in name:
        return "waist"
    if "shoulder_pitch" in name or "shoulder_roll" in name:
        return "shoulder_pr"
    if "shoulder_yaw" in name or "elbow" in name:
        return "shoulder_yaw_elbow"
    if "wrist" in name:
        return "wrist"
    raise KeyError(f"{name}: not a joint the training actuates")


def train_position_actuator(root: ET.Element, name: str) -> ET.Element:
    """The <position> actuator mjlab builds for SDK joint `name` (see TRAIN_PD); actuator name = the SDK name."""
    kp, kv, eff = TRAIN_PD[train_pd_group(name)]
    jn = f"{name}_joint"
    lo, hi = 0.0, 0.0
    for j in root.iter("joint"):
        if j.get("name") == jn:
            lo, hi = (float(v) for v in j.get("range", "0 0").split())
            break
    delta = eff / kp
    return ET.Element("position", {
        "name": name, "joint": jn, "kp": repr(kp), "kv": repr(kv), "inheritrange": "0", "ctrllimited": "false",
        "ctrlrange": f"{lo - delta!r} {hi + delta!r}", "forcelimited": "true", "forcerange": f"{-eff!r} {eff!r}",
    })  # fmt: skip
