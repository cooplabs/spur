"""Unitree Dex3-1 hand: joint table, limits, presets, tactile layout and the MuJoCo hand model.

Ported from r1mc2 (v2) r1mc2/dex3.py at git r21_try 2a4a165; changes: typed; the hand body XML is unchanged
(geometry, pads, joint parameters) except that masses are written at 9 significant digits (v2: 6); `hand_xml`
takes the joint parameters as arguments (named in models/provenance.yaml); the tests check this file against
Unitree's URDFs vendored in models/dex3_1/.

Sources (1 Oct 2026): unitree_ros robots/dexterous_hand_description/dex3_1/dex3_1_{l,r}.urdf (kinematics, limits,
inertias), xr_teleoperate assets/unitree_hand (tip links 5 cm past the last joints), Dex3-1 User Manual (DDS
order, mode byte, 9 pressure pads x 12 slots).

DDS motor order (both hands): thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1.
Hand frame (palm link): x out of the wrist towards the fingers; the fingers sit at z = +-0.0285 ("index" +z,
"middle" -z); the thumb chain lies on the grasp side (left -y, right +y). The two hands are mirror images
(y -> -y): joints about z flip sign, thumb_0 (about y) does not.
  left : fingers close NEGATIVE, thumb_1 / thumb_2 close POSITIVE
  right: fingers close POSITIVE, thumb_1 / thumb_2 close NEGATIVE
The mount on the R1 wrist (MOUNT_*) is an ESTIMATE from the R1 model and the install video (never calibrated).

Visual geometry (session 3, the user's note 10 Oct: "the hands are not super accurate"): Unitree's 16 link meshes
(models/dex3_1/meshes, same commit as the URDFs, sha256 in SOURCE.md) are drawn in the link frames (every URDF
<visual> origin is identity), in the real hand's matte black (photos sim_capture/20261010_hand_photos). They are
visual only (no contact, no mass); the collision shapes stay the primitives below (group 3, hidden in renders).
tools/hands/compare_collision.py measures primitives vs meshes (S3.md).
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import struct
from collections.abc import Iterator, Sequence
from pathlib import Path

import mujoco
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Vec3 = tuple[float, float, float]

SIDES: tuple[str, str] = ("left", "right")
NAMES: tuple[str, ...] = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
N = 7
THUMB: IntArray = np.array([0, 1, 2])
MIDDLE: IntArray = np.array([3, 4])
INDEX: IntArray = np.array([5, 6])
FINGERS: IntArray = np.array([3, 4, 5, 6])

# Limits [rad], DDS order (URDF).
Q_MIN: dict[str, FloatArray] = {
    "left": np.array([-1.0472, -0.6109, 0.0, -1.5708, -1.7453, -1.5708, -1.7453]),
    "right": np.array([-1.0472, -1.0472, -1.7453, 0.0, 0.0, 0.0, 0.0]),
}
Q_MAX: dict[str, FloatArray] = {
    "left": np.array([1.0472, 1.0472, 1.7453, 0.0, 0.0, 0.0, 0.0]),
    "right": np.array([1.0472, 0.6109, 0.0, 1.5708, 1.7453, 1.5708, 1.7453]),
}
MARGIN = 0.03  # commands stay this far inside the limits
TAU_MAX: FloatArray = np.array([2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])  # URDF effort [N m]
# Mirror: right = MIRROR * left (thumb_0 rotates about y, which a y-mirror leaves unchanged).
MIRROR: FloatArray = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
CLOSE_DIR: dict[str, FloatArray] = {"left": np.array([0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])}
CLOSE_DIR["right"] = MIRROR * CLOSE_DIR["left"]

# Gains used by Unitree's own clients (xr_teleoperate Dex3_1_Controller: kp 1.5, kd 0.2 at 100 Hz).
KP: FloatArray = np.array([1.5] * N)
KD: FloatArray = np.array([0.2] * N)
RATE_HZ = 100.0

# Joint parameters of the MuJoCo hand (no source: ESTIMATES, models/provenance.yaml).
JOINT_ARMATURE = 0.0005
JOINT_DAMPING = 0.02
JOINT_FRICTIONLOSS = 0.01

# Presets in LEFT-hand angles (right = MIRROR * left).
PRESETS_LEFT: dict[str, FloatArray] = {
    "open": np.array([0.0, 0.0, 0.05, -0.05, -0.05, -0.05, -0.05]),
    "relax": np.array([0.0, 0.25, 0.35, -0.35, -0.35, -0.35, -0.35]),
    # power grasp target (closing stops at contact in close-until-contact)
    "fist": np.array([0.0, 0.9, 1.5, -1.45, -1.6, -1.45, -1.6]),
    # Tip pinch: the fingers have no sideways joint and sit 57 mm apart, so the thumb (rotated by thumb_0) opposes
    # ONE finger. Solved on the model for tip-to-tip contact (gap < 1 mm); the other finger is curled away.
    "pinch_index": np.array([-0.438, 0.736, 0.155, -1.54, -1.71, -1.015, -0.872]),
    "pinch_middle": np.array([0.453, 0.626, 0.326, -1.051, -0.854, -1.54, -1.71]),
    # wide open for large objects / approach
    "spread": np.array([-0.6, -0.4, 0.05, -0.05, -0.05, -0.05, -0.05]),
    # open with the thumb swung back (community r1-dex3-manipulation-sdk, left hand, rounded)
    "open_wide": np.array([-0.077, -0.594, 0.05, -0.05, -0.05, -0.05, -0.05]),
}
PRESETS_LEFT["pinch"] = PRESETS_LEFT["pinch_index"]
GRASPS: dict[str, tuple[str, float, IntArray]] = {  # grasp kind -> (closed preset, pre-shape fraction, closing joints)
    "power": ("fist", 0.0, np.arange(7)),
    "pinch": ("pinch_index", 0.35, np.array([0, 1, 2, 5, 6])),
    "pinch_index": ("pinch_index", 0.35, np.array([0, 1, 2, 5, 6])),
    "pinch_middle": ("pinch_middle", 0.35, np.array([0, 1, 2, 3, 4])),
}


def clamp(q: FloatArray, side: str) -> FloatArray:
    lo = Q_MIN[side] + MARGIN * (Q_MIN[side] < 0)
    hi = Q_MAX[side] - MARGIN * (Q_MAX[side] > 0)
    return np.asarray(np.clip(q, lo, hi), dtype=np.float64)


def preset(name: str, side: str) -> FloatArray:
    q = PRESETS_LEFT[name]
    return clamp(q.copy() if side == "left" else MIRROR * q, side)


def closure(q: FloatArray, side: str) -> FloatArray:
    """0 = open .. 1 = fully closed per joint, along the closing direction (thumb_0 -> 0)."""
    d = CLOSE_DIR[side]
    lo, hi = Q_MIN[side], Q_MAX[side]
    span = np.where(d > 0, hi, np.where(d < 0, -lo, 1.0))
    c = np.where(d != 0, d * np.asarray(q) / np.maximum(span, 1e-6), 0.0)
    return np.asarray(np.clip(c, 0.0, 1.0), dtype=np.float64)


def mode_byte(motor_id: int, status: int = 0x01, timeout: int = 1) -> int:
    """Unitree RIS_Mode_t: id (4 bits) | status (3 bits) << 4 | timeout (1 bit) << 7.
    status 1 = FOC (closed loop). timeout 1 = the motor goes passive ~1 s after commands stop (the dead-man)."""
    return (motor_id & 0x0F) | ((status & 0x07) << 4) | ((timeout & 0x01) << 7)


# ---------------------------------------------------------------- tactile layout
# 9 pads per hand, each a PressSensorState_ with 12 slots (3 x 4 grid). Manual p.19: 0 thumb pad, 1 thumb tip,
# 2/3 one finger pad/tip, 4/5 the other finger pad/tip, 6-8 palm. 2x2 pads report slots 0, 2, 9, 11; 3x1 fingertip
# pads report 3, 6, 8; 30000 = no data.
PADS: tuple[str, ...] = (
    "thumb_pad", "thumb_tip", "finger_a_pad", "finger_a_tip", "finger_b_pad", "finger_b_tip",
    "palm_0", "palm_1", "palm_2",
)  # fmt: skip
PAD_SLOTS: dict[int, tuple[int, ...]] = {
    0: (0, 2, 9, 11), 1: (3, 6, 8), 2: (0, 2, 9, 11), 3: (3, 6, 8), 4: (0, 2, 9, 11),
    5: (3, 6, 8), 6: (0, 2, 9, 11), 7: (0, 2, 9, 11), 8: (0, 2, 9, 11),
}  # fmt: skip
NO_DATA = 30000.0
# Measured 5 Oct 2026 by pressing each tip on the real hands: index -> pads 4/5 ("finger_b"), middle -> 2/3.
PAD_FINGER: dict[str, str] = {"finger_a": "middle", "finger_b": "index"}
FINGER_PADS: dict[str, tuple[int, int]] = {
    "thumb": (0, 1),
    PAD_FINGER["finger_a"]: (2, 3),
    PAD_FINGER["finger_b"]: (4, 5),
}
TAXELS = sum(len(v) for v in PAD_SLOTS.values())  # 33

# ---------------------------------------------------------------- MuJoCo hand model
# Mount on the R1 wrist_roll_link: palm-link origin along the forearm axis, no extra rotation (palms face each other
# with wrist_roll = 0 and the forearms forward, like G1 + Dex3, whose palm joint is xyz 0.0415 0.003 0, rpy 0 on
# wrist_yaw_link). The stock R1 wrist_roll_link mesh narrows at x ~0.07-0.08 m, so the flange is taken at 0.075 m.
# ESTIMATE until the hand-eye capture on HW1.
MOUNT_POS: Vec3 = (0.075, 0.0, 0.0)
MOUNT_RPY: Vec3 = (0.0, 0.0, 0.0)
# The stock wrist_roll_link inertial (0.3248 kg, COM x 0.068) includes the fixed fist. Solving forearm-part COM ~0.035
# / fist COM ~0.11 for the combined COM gives a fist of ~0.143 kg (ESTIMATE). Removed when a Dex3 is attached.
FIST_MASS = 0.143
FOREARM_COM_X = 0.035
TIP = 0.05  # fingertip point past the last joint (xr_teleoperate)

MESH_DIR = Path(__file__).resolve().parents[2] / "models" / "dex3_1" / "meshes"
# The real hand's shell: matte black plastic (photos IMG_8768-8779, 10 Oct); a dull, slightly specular material.
VISUAL_MATERIAL = "dex3_matte_black"
VISUAL_MATERIAL_ATTRS: dict[str, str] = {
    "name": VISUAL_MATERIAL, "rgba": "0.10 0.10 0.11 1", "specular": "0.25", "shininess": "0.3", "reflectance": "0",
}  # fmt: skip
VISUAL_GROUP = 2  # the R1's visual meshes (r1_mjlab.xml class "visual")
COLLISION_GROUP = 3  # the R1's collision shapes (class "collision"): hidden in renders by default

# Collision geometry of the hand links (session 3). "primitives" = the S1 / v2 shapes (palm box, capsules, sphere):
# the DEFAULT. "mesh" = each link collides with the convex hull of Unitree's link mesh (the outer envelope the CAD
# shell presents), the pads moved to that surface - an OPT-IN variant until A8 / A9 / A10 are re-run on it (the
# primitives differ from the real shapes by up to ~4 mm per side in finger width, +3.6 / -2.3 mm at the pads, 3.5 mm in
# fingertip reach: tools/hands/compare_collision.py, S3.md). Pass hand_xml(collision="mesh") or set this constant.
COLLISION = "primitives"
COLLISIONS = ("primitives", "mesh")
# "mesh": where each pad's outer face sits relative to the link's convex hull along the pad normal at the pad centre
# (+ = the hull lies beyond the S1 pad face), measured by tools/hands/compare_collision.py on both hands (identical);
# the pad is shifted by this + PAD_PROUD_M so it still takes the first touch, 0.5 mm proud of the hull
# (modelling_choice: the real pads sit under the shell's surface skin; tests/unit/test_dex3_collision.py re-measures).
MESH_PAD_SHIFT_M: dict[int, float] = {
    0: 0.00365, 1: -0.00232, 2: 0.00365, 3: -0.00232, 4: 0.00365, 5: -0.00232, 6: 0.00114, 7: 0.00218, 8: 0.00114,
}  # fmt: skip
PAD_PROUD_M = 0.0005

# Pad layout (session 4; models/provenance.yaml "dex3_finger_pads"). "real" = the finger pads where the real Dex3-1's
# pads are (an OPT-IN variant); "s1" = the S1 / v2 layout (a 1.5 mm box on the grasp face of every phalanx), still the
# DEFAULT: with the real pads the A8 thin-block pinch (a constructed placement) slips out of the rounded noses and the
# sim pads' sensitivity is unknown (the A8 / A10 table in models/provenance.yaml "dex3_finger_pads"). Unitree's link
# meshes (identical for both fingers and the thumb, both hands; tools/hands/pad_geometry.py measures them) and the
# hand photos (IMG_8775 / IMG_8779, 10 Oct) say:
# * PROXIMAL pads (0 thumb_1, 2 middle_0, 4 index_0) are a raised rectangular block on the proximal phalanx: 20.6..43.3
#   mm along it, 19.8 mm wide, its face 14.65 mm from the joint axis. The S1 pad (10..38 mm, 16 mm wide, face 11.0 mm)
#   sat 3.65 mm INSIDE the real surface and 8 mm too far back. The block is the outermost surface there, so it stays a
#   colliding box, now AT the block (FINGER_PAD_BLOCK).
# * DISTAL pads (1 thumb_2, 3 middle_1, 5 index_1) sit UNDER the smooth bullet-shaped shell of the distal phalanx: the
#   mesh has no raised pad there; its grasp face at the S1 pad centre (34 mm) is 7.54 mm from the axis where the S1 box
#   face stood at 10.0 (2.46 mm proud) and the box reached the rounded nose (to 44 mm, its corner 2.5 mm proud of the
#   capsule's end cap), so in the sim every light tip touch landed on the box. On the robot "the very fingertip is
#   beyond the tip pads" (the 8 Oct grip failure log, row 2: S9B t~1021 the index pushed into the board with
#   every pad <= 0.2 kPa; S10 the thumb on the board <= 0.07 kPa). Each distal pad is now a SENSING region only
#   (contype / conaffinity 0) whose face lies on the real shell surface (FINGER_TIP_PAD_FACE_M), with the S1 in-plane
#   extent (24..44 mm, 14 mm wide: the 3 cells' real extent needs the HW1 pad presses): the link's own shape takes
#   the contact and the sensor model counts it for the pad only inside that footprint (+ pad_face_margin_m;
#   sensors.SensorModel._which_pad) - the nose past ~46 mm is felt by no pad.
# Palm pads 6-8: unchanged (the real palm face lies 0.2-1.7 mm from the S1 pad faces: within the 2 mm margin).
PAD_LAYOUT = "s1"
PAD_LAYOUTS = ("s1", "real")
FINGER_PAD_BLOCK: dict[str, float] = {  # proximal phalanx, link frame (fingers: along x, grasp +-y; thumb: y / x)
    "along_lo": 0.0206, "along_hi": 0.0433, "half_width": 0.0099, "face": 0.01465,
}  # fmt: skip
FINGER_TIP_PAD_FACE_M = 0.00754  # distal phalanx: the shell's grasp face at the S1 pad centre (along 0.034 m)
PAD_HALF_THICK = 0.0015  # every pad box is 3 mm thick (S1)
TIP_PAD_SENSING_ONLY = True  # the distal pads do not collide (the shell takes the touch); False: ablation only


@contextlib.contextmanager
def hand_variant(collision: str | None = None, pads: str | None = None) -> Iterator[None]:
    """Build models with this hand collision / pad layout (None = keep the current default), e.g. to compare A8-A10
    on the variants; restores the module defaults afterwards."""
    global COLLISION, PAD_LAYOUT
    if collision is not None and collision not in COLLISIONS:
        raise ValueError(f"collision {collision!r} not in {COLLISIONS}")
    if pads is not None and pads not in PAD_LAYOUTS:
        raise ValueError(f"pads {pads!r} not in {PAD_LAYOUTS}")
    old = COLLISION, PAD_LAYOUT
    COLLISION = old[0] if collision is None else collision
    PAD_LAYOUT = old[1] if pads is None else pads
    try:
        yield
    finally:
        COLLISION, PAD_LAYOUT = old


def link_names(side: str) -> list[str]:
    """The 8 link (body) names of one hand, palm first (URDF names)."""
    return [f"{side}_hand_{n}_link" for n in ("palm", *NAMES)]


def mesh_file(side: str, link: str) -> Path:
    """The vendored STL of a link ("palm", "thumb_0", ... ) of one hand."""
    return MESH_DIR / f"{side}_hand_{link}_link.STL"


def collision_mesh_assets(side: str) -> list[dict[str, str]]:
    """<mesh> asset attributes of the "mesh" collision variant (separate names, so a visual=False model keeps them)."""
    return [{"name": f"{side}_hand_{n}_cmesh", "file": str(mesh_file(side, n))} for n in ("palm", *NAMES)]


# The drawn copies are decimated (vertex clustering on a VISUAL_CELL_M grid: 325,668 -> ~84,000 triangles for both
# hands): Unitree's meshes are CAD-fine (up to 43,000 triangles a link) and at full resolution they made a 970 x 825
# render 35 % slower (18.5 -> 25.5 ms, this Mac). A cell of 1 mm moves no drawn vertex more than ~0.9 mm (the cell's
# half diagonal); collision (the "mesh" variant) uses the full-resolution files.
VISUAL_CELL_M = 0.001
_DECIMATE_VERSION = 1


def _stl_triangles(path: Path) -> npt.NDArray[np.float64]:
    b = path.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(b[84 : 84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return np.asarray(rec["v"], dtype=np.float64)


def decimate(tri: npt.NDArray[np.float64], cell: float) -> tuple[FloatArray, IntArray]:
    """Vertex clustering: every vertex snaps to the mean of the vertices in its cell; degenerate and duplicate
    triangles are dropped. Deterministic. Returns (vertices (n, 3), faces (m, 3))."""
    v = tri.reshape(-1, 3)
    k = np.floor(v / cell).astype(np.int64)
    k -= k.min(axis=0)
    span = k.max(axis=0) + 1
    key = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    sums = np.zeros((len(uk), 3))
    np.add.at(sums, inv, v)
    verts = sums / np.bincount(inv, minlength=len(uk))[:, None]
    f = inv.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    return verts, f[np.sort(first)].astype(np.int64)


def _write_stl(path: Path, verts: FloatArray, faces: IntArray) -> None:
    t = verts[faces].astype(np.float32)
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(faces), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = nrm, t
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(b"r1sim dex3 visual (decimated)".ljust(80, b" ") + struct.pack("<I", len(faces)) + rec.tobytes())
    os.replace(tmp, path)  # atomic: concurrent builders write identical bytes


def cache_dir() -> Path:
    """Where the decimated drawing copies live: $R1MC3_CACHE_DIR/dex3_visual (default ~/.cache/r1mc3/dex3_visual),
    outside the repository (a synced checkout stays clean)."""
    env = os.environ.get("R1MC3_CACHE_DIR")
    return (Path(env) if env else Path.home() / ".cache" / "r1mc3") / "dex3_visual"


@functools.lru_cache(maxsize=32)
def visual_mesh_file(side: str, link: str) -> Path:
    """The decimated drawing copy of a link mesh (made once, cached by the source file's sha256 and the cell; the
    path is remembered per process)."""
    src = mesh_file(side, link)
    sha = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    out = cache_dir() / f"{sha}_c{VISUAL_CELL_M * 1e4:.0f}_v{_DECIMATE_VERSION}.stl"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_stl(out, *decimate(_stl_triangles(src), VISUAL_CELL_M))
    return out


def mesh_assets(side: str) -> list[dict[str, str]]:
    """<mesh> asset attributes for one hand's 8 visual meshes (the decimated drawing copies; absolute paths, so
    independent of the model's meshdir)."""
    return [{"name": f"{side}_hand_{n}_link", "file": str(visual_mesh_file(side, n))} for n in ("palm", *NAMES)]


def _axis_quat(axis: Sequence[float], angle: float) -> FloatArray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=np.float64), float(angle))
    return q


def link_poses(side: str, q: npt.ArrayLike | None = None) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Forward kinematics of one hand in the PALM frame: link short name ("palm", "thumb_0", ...) -> (pos (3,),
    quat (4,) w x y z) of the link frame at joint angles q (DDS order; default the open preset). Same chain as
    hand_xml (URDF joint origins, identity rotations, one revolute joint per link)."""
    qq = preset("open", side) if q is None else np.asarray(q, dtype=np.float64)
    out: dict[str, tuple[FloatArray, FloatArray]] = {"palm": (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))}
    for r in _L[1:]:
        name, parent, origin, axis = r[0], r[1], r[2], r[3]
        assert parent is not None and axis is not None
        pp, pq = out[parent]
        off = np.zeros(3)
        mujoco.mju_rotVecQuat(off, np.asarray(_mir(origin, side), dtype=np.float64), pq)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, pq, _axis_quat(axis, qq[NAMES.index(name)]))
        out[name] = (pp + off, quat)
    return out


# (name, parent, joint origin xyz, axis, com, mass, (ixx, iyy, izz, ixy, ixz, iyz)) for the LEFT hand (URDF).
_Row = tuple[str, str | None, Vec3, Vec3 | None, Vec3, float, tuple[float, float, float, float, float, float]]
_L: list[_Row] = [
    ("palm", None, (0, 0, 0), None, (0.06214635, -0.00050870, -0.00058171), 0.37283854,
     (2.7535e-4, 5.3952e-4, 3.9623e-4, -1.5955e-5, -2.4216e-6, -4.228e-7)),
    ("thumb_0", "palm", (0.0255, 0, 0), (0, 1, 0), (-0.00088425, -0.00863407, 0.00094429), 0.08623657,
     (1.6029e-5, 1.4518e-5, 1.6379e-5, 1.068e-7, 1.673e-7, -5.109e-7)),
    ("thumb_1", "thumb_0", (-0.0025, -0.0193, 0), (0, 0, 1), (-0.00082789, -0.03547436, -0.00038090), 0.0588507,
     (1.2747e-5, 6.016e-6, 1.2345e-5, -5.077e-7, 1.609e-7, -2.784e-7)),
    ("thumb_2", "thumb_1", (0, -0.0458, 0), (0, 0, 1), (-0.00171735, -0.02628193, 0.00010779), 0.02030626,
     (4.613e-6, 1.536e-6, 3.866e-6, -3.42e-8, -8.2e-9, -2.55e-8)),
    ("middle_0", "palm", (0.0777, 0.0016, -0.0285), (0, 0, 1), (0.03547436, 0.00082789, 0.00038090), 0.0588507,
     (6.016e-6, 1.2747e-5, 1.2345e-5, -5.077e-7, -2.784e-7, 1.609e-7)),
    ("middle_1", "middle_0", (0.0458, 0, 0), (0, 0, 1), (0.02628193, 0.00171735, -0.00010779), 0.02030626,
     (1.536e-6, 4.613e-6, 3.866e-6, -3.42e-8, -2.55e-8, -8.2e-9)),
    ("index_0", "palm", (0.0777, 0.0016, 0.0285), (0, 0, 1), (0.03547436, 0.00082789, 0.00038090), 0.0588507,
     (6.016e-6, 1.2747e-5, 1.2345e-5, -5.077e-7, -2.784e-7, 1.609e-7)),
    ("index_1", "index_0", (0.0458, 0, 0), (0, 0, 1), (0.02628193, 0.00171735, -0.00010779), 0.02030626,
     (1.536e-6, 4.613e-6, 3.866e-6, -3.42e-8, -2.55e-8, -8.2e-9)),
]  # fmt: skip
BODIES: dict[str, _Row] = {r[0]: r for r in _L}
MASS = sum(r[5] for r in _L)  # ~0.6965 kg (product page: 710 g)


def _mir(v: Sequence[float], side: str) -> Vec3:
    """Mirror a LEFT-hand vector to the right hand (y -> -y)."""
    return (v[0], v[1], v[2]) if side == "left" else (v[0], -v[1], v[2])


def _fmt(v: Sequence[float]) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def hand_xml(
    side: str,
    mount_pos: Vec3 = MOUNT_POS,
    mount_rpy: Vec3 = MOUNT_RPY,
    visual_rgba: str = "0.15 0.15 0.15 1",
    armature: float = JOINT_ARMATURE,
    damping: float = JOINT_DAMPING,
    frictionloss: float = JOINT_FRICTIONLOSS,
    meshes: bool = False,
    collision: str | None = None,
    pads: str | None = None,
) -> str:
    """MJCF body subtree for one hand, to be placed inside <side>_wrist_roll_link.

    Collision geoms are simple shapes: palm box, finger / thumb capsules and thin tactile pad boxes named
    <side>_pad_<k> (k = Dex3 pad id) on the grasp surfaces. Joints are named <side>_hand_<name>_joint (DDS order).
    meshes=True adds Unitree's link mesh per body as a visual-only geom <side>_hand_<name>_visual (group 2, no
    contact, no mass; the model must declare the mesh_assets(side) and the VISUAL_MATERIAL) and moves the primitives
    to group 3, so renders show the real hand's shape while the physics is unchanged.
    collision="mesh" (default COLLISION) gives each link the convex hull of its mesh as the collision shape (same
    geom names, so the sensor model's pad mapping holds; the model must declare collision_mesh_assets(side)).
    pads (default PAD_LAYOUT): "real" = the proximal finger pads on the real raised blocks (colliding), the distal
    pads sensing regions on the real shell surface (not colliding); "s1" = the S1 / v2 boxes. With collision="mesh"
    the colliding pads stand PAD_PROUD_M beyond the hull (palm / "s1" pads: shifted by MESH_PAD_SHIFT_M)."""
    collision = COLLISION if collision is None else collision
    if collision not in COLLISIONS:
        raise ValueError(f"collision {collision!r} not in {COLLISIONS}")
    pads = PAD_LAYOUT if pads is None else pads
    if pads not in PAD_LAYOUTS:
        raise ValueError(f"pads {pads!r} not in {PAD_LAYOUTS}")
    real_pads = pads == "real"
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array(mount_rpy, float), "xyz")
    g = -1.0 if side == "left" else 1.0  # grasp side along palm y
    p = f"{side}_hand"
    out = [f'<body name="{p}_palm_link" pos="{_fmt(mount_pos)}" quat="{_fmt(q.tolist())}">']

    def inertial(r: _Row) -> str:
        com, mass, inertia = _mir(r[4], side), r[5], r[6]
        ixy, ixz, iyz = (
            (inertia[3], inertia[4], inertia[5]) if side == "left" else (-inertia[3], inertia[4], -inertia[5])
        )
        return (
            f'<inertial pos="{_fmt(com)}" mass="{mass:.9g}" '
            f'fullinertia="{inertia[0]:.6g} {inertia[1]:.6g} {inertia[2]:.6g} {ixy:.6g} {ixz:.6g} {iyz:.6g}"/>'
        )

    def geoms(name: str) -> list[str]:
        s: list[str] = []
        grp = f' group="{COLLISION_GROUP}"' if meshes else ""
        cl = f'contype="1" conaffinity="1" condim="4" friction="1.2 0.02 0.002" rgba="{visual_rgba}"{grp}'
        pad = f'contype="1" conaffinity="1" condim="4" friction="1.2 0.02 0.002" rgba="0.9 0.3 0.2 1"{grp}'
        hull = collision == "mesh"

        def link_shape(prim: str) -> str:
            """The link's collision geom: the S1 primitive, or (variant "mesh") the hull of its mesh, same name."""
            if not hull:
                return prim
            return f'<geom name="{p}_{name}_geom" type="mesh" mesh="{p}_{name}_cmesh" {cl}/>'

        def pad_geom(k: int, pos: Vec3, size: str, normal: Vec3) -> str:
            """Pad k: at its S1 place (pos strings as v2 wrote them), or moved along its normal to the hull."""
            if not hull:  # exactly the S1 / v2 strings (the palm pads print z as Python floats, the others 0)
                xyz = f"{pos[0]} {pos[1]:.4f} {pos[2] if name == 'palm' else 0}"
            else:
                sh = MESH_PAD_SHIFT_M[k] + PAD_PROUD_M
                xyz = _fmt((pos[0] + sh * normal[0], pos[1] + sh * normal[1], pos[2] + sh * normal[2]))
            return f'<geom name="{side}_pad_{k}" type="box" pos="{xyz}" size="{size}" {pad}/>'

        def real_finger_pad(k: int, thumb: bool, distal: bool) -> str:
            """Pad k in the "real" layout (PAD_LAYOUT docs): a finger runs along x with its pads facing g*y, the
            thumb along g*y with its pads facing +x."""
            if distal:  # a sensing region on the shell (the S1 footprint), not a contact surface
                c_along, h_along, h_w = 0.034, 0.010, 0.007
                c_out = FINGER_TIP_PAD_FACE_M - PAD_HALF_THICK
                sensing = 'contype="0" conaffinity="0"'
                attrs = pad.replace('contype="1" conaffinity="1"', sensing) if TIP_PAD_SENSING_ONLY else pad
            else:  # the raised block: the outermost surface there
                b = FINGER_PAD_BLOCK
                c_along = (b["along_lo"] + b["along_hi"]) / 2
                h_along, h_w = (b["along_hi"] - b["along_lo"]) / 2, b["half_width"]
                c_out = b["face"] - PAD_HALF_THICK + (PAD_PROUD_M if hull else 0.0)
                attrs = pad
            if thumb:
                xyz, size = (c_out, c_along * g, 0.0), (PAD_HALF_THICK, h_along, h_w)
            else:
                xyz, size = (c_along, c_out * g, 0.0), (h_along, PAD_HALF_THICK, h_w)
            return f'<geom name="{side}_pad_{k}" type="box" pos="{_fmt(xyz)}" size="{_fmt(size)}" {attrs}/>'

        if meshes:
            s.append(
                f'<geom name="{p}_{name}_visual" type="mesh" mesh="{p}_{name}_link" material="{VISUAL_MATERIAL}" '
                f'contype="0" conaffinity="0" density="0" group="{VISUAL_GROUP}"/>'
            )
        if name == "palm":
            s.append(
                link_shape(
                    f'<geom name="{p}_palm_geom" type="box" pos="0.042 {-0.002 * g:.4f} 0" '
                    f'size="0.036 0.016 0.044" {cl}/>'
                )
            )
            for k, z in ((6, 0.026), (7, 0.0), (8, -0.026)):
                s.append(pad_geom(k, (0.05, 0.0165 * g, z), "0.02 0.0015 0.011", (0.0, g, 0.0)))
            s.append(f'<site name="{p}_palm" pos="0.05 {0.03 * g:.4f} 0" size="0.006"/>')
        elif name in ("middle_0", "index_0"):
            s.append(
                link_shape(
                    f'<geom name="{p}_{name}_geom" type="capsule" fromto="0.006 0 0 0.040 0 0" size="0.0095" {cl}/>'
                )
            )
            k = FINGER_PADS["middle" if name == "middle_0" else "index"][0]
            if real_pads:
                s.append(real_finger_pad(k, thumb=False, distal=False))
            else:
                s.append(pad_geom(k, (0.024, 0.0095 * g, 0.0), "0.014 0.0015 0.008", (0.0, g, 0.0)))
        elif name in ("middle_1", "index_1"):
            s.append(
                link_shape(
                    f'<geom name="{p}_{name}_geom" type="capsule" fromto="0.004 0 0 0.040 0 0" size="0.0085" {cl}/>'
                )
            )
            k = FINGER_PADS["middle" if name == "middle_1" else "index"][1]
            if real_pads:
                s.append(real_finger_pad(k, thumb=False, distal=True))
            else:
                s.append(pad_geom(k, (0.034, 0.0085 * g, 0.0), "0.010 0.0015 0.007", (0.0, g, 0.0)))
            s.append(f'<site name="{p}_{name[:-2]}_tip" pos="{TIP} 0 0" size="0.005"/>')
        elif name == "thumb_0":
            s.append(
                link_shape(f'<geom name="{p}_thumb_0_geom" type="sphere" pos="0 {0.009 * g:.4f} 0" size="0.011" {cl}/>')
            )
        elif name == "thumb_1":
            s.append(
                link_shape(
                    f'<geom name="{p}_thumb_1_geom" type="capsule" fromto="0 {0.006 * g:.4f} 0 0 {0.040 * g:.4f} 0" '
                    f'size="0.0095" {cl}/>'
                )
            )
            if real_pads:
                s.append(real_finger_pad(0, thumb=True, distal=False))
            else:
                s.append(pad_geom(0, (0.0095, 0.024 * g, 0.0), "0.0015 0.014 0.008", (1.0, 0.0, 0.0)))
        elif name == "thumb_2":
            s.append(
                link_shape(
                    f'<geom name="{p}_thumb_2_geom" type="capsule" fromto="0 {0.004 * g:.4f} 0 0 {0.040 * g:.4f} 0" '
                    f'size="0.0085" {cl}/>'
                )
            )
            if real_pads:
                s.append(real_finger_pad(1, thumb=True, distal=True))
            else:
                s.append(pad_geom(1, (0.0085, 0.034 * g, 0.0), "0.0015 0.010 0.007", (1.0, 0.0, 0.0)))
            s.append(f'<site name="{p}_thumb_tip" pos="0 {TIP * g:.4f} 0" size="0.005"/>')
        return s

    def emit(name: str, indent: int) -> None:
        r = BODIES[name]
        lo = Q_MIN[side][NAMES.index(name)]
        hi = Q_MAX[side][NAMES.index(name)]
        axis = r[3]
        assert axis is not None
        out.append(" " * indent + f'<body name="{p}_{name}_link" pos="{_fmt(_mir(r[2], side))}">')
        out.append(" " * indent + "  " + inertial(r))
        out.append(
            " " * indent + f'  <joint name="{p}_{name}_joint" axis="{_fmt(axis)}" range="{lo:.5f} {hi:.5f}" '
            f'armature="{armature:g}" damping="{damping:g}" frictionloss="{frictionloss:g}"/>'
        )
        for gm in geoms(name):
            out.append(" " * indent + "  " + gm)
        for child in [c[0] for c in _L if c[1] == name]:
            emit(child, indent + 2)
        out.append(" " * indent + "</body>")

    out.append("  " + inertial(BODIES["palm"]))
    for gm in geoms("palm"):
        out.append("  " + gm)
    for child in [c[0] for c in _L if c[1] == "palm"]:
        emit(child, 2)
    out.append("</body>")
    return "\n".join(out)


def joint_names(side: str) -> list[str]:
    return [f"{side}_hand_{n}_joint" for n in NAMES]


def actuator_names(side: str) -> list[str]:
    return [f"{side}_hand_{n}" for n in NAMES]


def preshape(kind: str, side: str) -> FloatArray:
    """Hand shape before a grasp closes: the closing joints part-way open (GRASPS fraction), the thumb already
    rotated, fingers that take no part already curled."""
    closed_name, pre, joints = GRASPS[kind]
    q_closed = preset(closed_name, side)
    q_open = preset("open", side)
    q = q_open + pre * (q_closed - q_open)
    q[0] = q_closed[0]
    others = np.setdiff1d(np.arange(N), joints)
    q[others] = q_closed[others]
    return clamp(q, side)


PAD_SITE: dict[str, str] = {"thumb": "thumb_tip", "palm": "palm"}  # where a touching pad is
