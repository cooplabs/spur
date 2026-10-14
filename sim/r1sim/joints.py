"""R1 joint table: names, HG LowCmd slots, limits, gains and the nominal stand pose.

Ported from r1mc2 (v2) r1mc2/joints.py at git r21_try 2a4a165; changes: typed, no controller poses other than
the ones the simulator needs; POSE_STAND added (the live policy's default pose); the policy gain table and the RL
default pose dropped (the policy's gains come from its ONNX metadata, session 2).

Order is the SDK body order (unitree_sdk2_python example/r1/low_level): 12 legs, waist roll/yaw, 2 x 5 arm,
head pitch/yaw. The slot map was confirmed on our robot from the 28 Sep 2026 read-only lowstate capture
(slots 14, 20, 21, 27, 28, 31-34 report mode 0). All angles are radians, joint side (ankles in PR mode).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

NUM_SLOTS = 35  # unitree_hg LowCmd_.motor_cmd length
MODE_MACHINE = 1  # R1 26-DoF (observed on our robot)
MODE_PR = 0  # ankles commanded as pitch/roll (observed)

JOINTS: tuple[str, ...] = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_roll", "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
    "head_pitch", "head_yaw",
)  # fmt: skip
N = len(JOINTS)
IDX: dict[str, int] = {n: i for i, n in enumerate(JOINTS)}
SLOT: IntArray = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30]
)

LEFT_LEG: IntArray = np.arange(0, 6)
RIGHT_LEG: IntArray = np.arange(6, 12)
LEGS: IntArray = np.arange(0, 12)
WAIST: IntArray = np.array([12, 13])
LEFT_ARM: IntArray = np.arange(14, 19)
RIGHT_ARM: IntArray = np.arange(19, 24)
ARMS: IntArray = np.arange(14, 24)
HEAD: IntArray = np.array([24, 25])
UPPER: IntArray = np.arange(12, 26)
GROUPS: dict[str, IntArray] = {
    "legs": LEGS, "left_leg": LEFT_LEG, "right_leg": RIGHT_LEG, "waist": WAIST,
    "left_arm": LEFT_ARM, "right_arm": RIGHT_ARM, "arms": ARMS, "head": HEAD,
    "upper": UPPER, "all": np.arange(N),
}  # fmt: skip

# Mechanical limits from the official R1 URDF (unitree_ros r1_description).
Q_MIN: FloatArray = np.array([
    -2.9322, -1.0472, -2.7402, -0.1745, -0.8727, -0.2618,
    -2.9322, -1.7453, -2.7402, -0.1745, -0.8727, -0.2618,
    -0.5236, -2.618,
    -3.1416, -0.2269, -1.9199, -0.9756, -1.9199,
    -3.1416, -2.4785, -1.9199, -0.9756, -1.9199,
    -0.6283, -2.0071,
])  # fmt: skip
Q_MAX: FloatArray = np.array([
    2.5482, 1.7453, 2.7402, 2.4260, 0.5760, 0.2618,
    2.5482, 1.0472, 2.7402, 2.4260, 0.5760, 0.2618,
    0.5236, 2.618,
    2.0944, 2.4784, 1.9199, 2.1852, 1.9199,
    2.0944, 0.2268, 1.9199, 2.1852, 1.9199,
    0.6283, 2.0071,
])  # fmt: skip
# URDF effort limits [N m]: hips/knee/waist/shoulder pitch+roll 60, ankles 50, shoulder yaw/elbow/wrist/head 33.
TAU_MAX: FloatArray = np.array(
    [60, 60, 60, 60, 50, 50, 60, 60, 60, 60, 50, 50, 60, 60, 60, 60, 33, 33, 33, 60, 60, 33, 33, 33, 33, 33],
    dtype=float,
)

# Commands are clamped to the mechanical range shrunk by this margin on each side
# (the waist yaw / head yaw ranges are further limited for cable safety).
LIMIT_MARGIN = 0.05
SOFT_MIN: FloatArray = Q_MIN + LIMIT_MARGIN
SOFT_MAX: FloatArray = Q_MAX - LIMIT_MARGIN
SOFT_MIN[IDX["waist_yaw"]], SOFT_MAX[IDX["waist_yaw"]] = -1.5, 1.5
SOFT_MIN[IDX["head_yaw"]], SOFT_MAX[IDX["head_yaw"]] = -1.5, 1.5

# Drive-side PD gains, STIFF set: official R1 low-level example (unitree_sdk2_python), used for stand / hold.
KP_STIFF: FloatArray = np.array(
    [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 300, 300,
     100, 100, 100, 100, 50, 100, 100, 100, 100, 50, 50, 10],
    dtype=float,
)  # fmt: skip
KD_STIFF: FloatArray = np.array(
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0.1], dtype=float
)

# The nominal double-support stand: the live policy's default pose (R1MC-Manip2 HOME = Unitree's stand lock as
# measured on our robot before every take-over, feet 0.22 m apart; training r1mc_manip_constants.py
# R1MC_MANIP2_HOME, = the ONNX default_joint_pos), head 0. Initial base height in training 0.745 m.
POSE_STAND: FloatArray = np.zeros(N)
for _n, _v in {
    "left_hip_pitch": -0.300, "right_hip_pitch": -0.300,
    "left_hip_roll": 0.025, "right_hip_roll": -0.025,
    "left_hip_yaw": -0.130, "right_hip_yaw": 0.130,
    "left_knee": 0.455, "right_knee": 0.455,
    "left_ankle_pitch": -0.140, "right_ankle_pitch": -0.140,
    "left_ankle_roll": -0.050, "right_ankle_roll": 0.050,
    "left_shoulder_pitch": 0.22, "right_shoulder_pitch": 0.22,
    "left_shoulder_roll": 0.193, "right_shoulder_roll": -0.194,
    "left_shoulder_yaw": 0.039, "right_shoulder_yaw": -0.033,
    "left_elbow": 0.76, "right_elbow": 0.76,
}.items():  # fmt: skip
    POSE_STAND[IDX[_n]] = _v


def clamp(q: FloatArray) -> FloatArray:
    """Clip a 26-vector to the soft limits."""
    return np.clip(q, SOFT_MIN, SOFT_MAX)


def resolve(names: str | Iterable[str]) -> IntArray:
    """Group name, joint name or list of either -> sorted joint indices."""
    if isinstance(names, str):
        names = [names]
    out: list[int] = []
    for n in names:
        if n in GROUPS:
            out.extend(int(i) for i in GROUPS[n])
        elif n in IDX:
            out.append(IDX[n])
        else:
            raise KeyError(f"unknown joint or group: {n}")
    return np.array(sorted(set(out)), dtype=np.int64)
