"""The grasp and the "holding" verdict of one Dex3-1 hand (DESIGN.md §6.2; fixes v2's false hold HG-1; D27).

    g = GraspRunner(hands, "right", "wrap", "rigid", width_m=0.05, contact_height=heights)
    g.start(now_ns)
    while g.tick(now_ns) in ("OPEN", "CLOSING", "CONTACT"): ...   # r1hands' 100 Hz tick
    g.verdict()            # TENTATIVE: the hand's evidence holds - the lift check decides
    g.notify_lift(0.015)   # the arm lifted 1.5 cm (called while the arm moves; cumulative since the verdict)
    g.verdict()            # HOLDING | NOT_HOLDING(reason), with the evidence table and a confidence

The sequence is v2's (r1mc2/hands.py at git r21_try 2a4a165, `Hand._start_grasp` / `_tick_grasp`, the parts that held
real objects on hardware): pre-shape -> close along a path at a speed (faster until the tips are just wider than a
known object) -> each finger group stops on its own contact evidence (a pad rising above its reading at the close's
start for PAD_PERSIST_S, a stall, a creep far behind the command, a calibrated torque with the joints stopped) -> the
first side rests on the object until the other side touches, then the squeeze ramps in (wrap 0.3, pinch 0.4 rad past
contact, kp x error capped at MAX_ERR_TAU) -> the verdict -> the lift check.

What v3 changes is the verdict: v2 decided "holding" on the grip FORCE (a calibrated torque through the finger levers
+ the pads), and on 9 Oct an EMPTY wrap on a table surface read 5.24 / 5.5 N - as much as a real hold of a small
object (5.19 N; A8, V2_INVENTORY App. C HG-1): the thumb stalled on the curled index finger (the real thumb_1 at
-0.56..-0.63 of its -0.9 command) and the fingertips pressed the surface. Force alone cannot tell those apart. Here
the verdict is a table of evidence, every term computed from the hand's state, its kinematics and (when the caller
gives it) where the support surface is; the result reports every term:

The terms, by object class (rigid = the default; soft = rigid with the soft allowances):
* opposition - rigid / soft: the thumb side AND a finger side stopped on contact evidence that is NOT the hand
  touching itself (the thumb's capsules within SELF_TOUCH_M of a finger's, from the hand's kinematics);
  thin_flexible ("tip_pressure"): the thumb TIP pad and the opposing finger's TIP pad both pressed (firm fingertip
  contact: HW1 B3.2, the only contact the real pads feel).
* gap - rigid: the aperture (tip distance minus the tips-met distance) >= GAP_FRAC x the object's width, or >=
  GAP_MIN_M without one (tips that met hold nothing); soft: >= SOFT_GAP_FRAC x width; thin_flexible ("layer"): the
  clearance between the two tip shells in [LAYER_MIN_M, LAYER_MAX_M] - a thin layer between the tips, not tips that
  met (0), not an open hand.
* contact height - rigid / soft, only when `contact_height` is given: touching fingertips ALL at the support surface
  (their lowest points within SURFACE_MARGIN_M of it) are a hold only when the aperture matches the object's width
  (the tips closed on its sides: v2's lower-then-close grasps put the tips at the table) - otherwise "pressing the
  surface"; without a width the lift decides; thin_flexible: reported only (a layer is pinched AT the surface).
* force - rigid / soft, with a trusted torque or pads: the grip force >= HOLD_MIN_N (v2's rule: necessary, no longer
  sufficient).
* lift - after a lift of >= LIFT_CHECK_M, for LIFT_OBS_S: rigid / soft: the contact persists (pads still risen or the
  fingers still held back from their commands) and no group runs on more than RUN_ON_RAD (soft SOFT_RUN_ON_RAD) past
  the verdict's pose; thin_flexible: the tip pressure persists and the layer stays >= LAYER_MIN_M.

States: OPEN -> CLOSING -> CONTACT -> TENTATIVE -> HOLDING | NOT_HOLDING (reason = the first failing term).

**Contact decides (the user's rule, 12 Oct 2026 ~02:00, session 10 round B -> session 11):** (1) after HOLDING, fingers
closing further while the contact persists is NOT a slip - only closing on nothing (no touching group keeps any
evidence: no pad, not held back by its squeeze) is; the run-on is reported, it no longer decides once the hold is
confirmed (the lift window TENTATIVE -> HOLDING stays strict: every group's evidence AND no run-on past RUN_ON_RAD);
(2) a NOT_HOLDING verdict never relaxes the fingers by itself: the hand keeps its grip (the last grip command, re-sent
every tick with the squeeze's heat easing) until the caller opens it (`release`). Round B found the old rule dropping
held objects: the simulated tall cylinder was HOLDING (8.0 N, every group touching, ~7 cm up) when its fingers settled
0.101 rad past the verdict pose (> RUN_ON_RAD) -> NOT_HOLDING -> v2's relax (the joints backed off) -> it fell; the
round object the same at 0.103 rad. The one exception is the heat abort (TEMP_ABORT_C: the finger motors' protection,
HW1 B3.3): it still backs off (powered, never limp).

The object class is GENERIC (what the hand must judge), never a task's object name (DESIGN D32). The hand facts used
(HW1 part 1, docs/sessions/HW1.md B3.1-B3.4): the pads feel only firm fingertip contact; the inner pads go negative with
bending; thumb_1 stops ~0.06 rad short of 0; the hands' tau_est is raw (1 raw = 1e-6 N m); a finger at 0.9 N m heats
~6 C/s (41 -> 71 C in ~5 s) -> the squeeze eases from TEMP_EASE_C and the grasp ends (relaxed) at TEMP_ABORT_C.
Every threshold below is an ESTIMATE until HW2's 10 scripted grasps per object (B4) and its thin-layer pinches (D29).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .dex3 import KD_DEFAULT, KP_DEFAULT, N_PADS, Q_MAX, Q_MIN, SIDES, N

FloatArray = npt.NDArray[np.float64]
GripState = Literal["OPEN", "CLOSING", "CONTACT", "TENTATIVE", "HOLDING", "NOT_HOLDING"]
GripName = Literal["wrap", "pinch", "tip"]
ObjectClass = Literal["rigid", "thin_flexible", "soft"]
OBJECT_CLASSES: tuple[str, ...] = ("rigid", "thin_flexible", "soft")
STATES: tuple[str, ...] = ("OPEN", "CLOSING", "CONTACT", "TENTATIVE", "HOLDING", "NOT_HOLDING")

# ------------------------------------------------------------------------------------------------ the hand (model)
# Presets in LEFT-hand angles, Dex3 DDS order (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1);
# right = MIRROR * left. v2 r1mc2/dex3.py PRESETS_LEFT (= r1sim.dex3.PRESETS_LEFT; tests/unit/test_grip.py checks).
MIRROR: FloatArray = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
PRESETS_LEFT: dict[str, FloatArray] = {
    "open": np.array([0.0, 0.0, 0.05, -0.05, -0.05, -0.05, -0.05]),
    "relax": np.array([0.0, 0.25, 0.35, -0.35, -0.35, -0.35, -0.35]),
    "fist": np.array([0.0, 0.9, 1.5, -1.45, -1.6, -1.45, -1.6]),
    "pinch_index": np.array([-0.438, 0.736, 0.155, -1.54, -1.71, -1.015, -0.872]),
    "pinch_middle": np.array([0.453, 0.626, 0.326, -1.051, -0.854, -1.54, -1.71]),
    "spread": np.array([-0.6, -0.4, 0.05, -0.05, -0.05, -0.05, -0.05]),
    "open_wide": np.array([-0.077, -0.594, 0.05, -0.05, -0.05, -0.05, -0.05]),
}
CLOSE_DIR_LEFT: FloatArray = np.array([0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
MARGIN = 0.03  # rad inside the limits (r1mc3.hands.dex3.MARGIN)
GROUPS: dict[str, tuple[int, ...]] = {"thumb": (1, 2), "middle": (3, 4), "index": (5, 6)}  # the CLOSING joints
GROUP_PADS: dict[str, tuple[int, int]] = {"thumb": (0, 1), "middle": (2, 3), "index": (4, 5)}  # (proximal, tip) pad
# (the finger -> pad mapping measured 5 Oct 2026 on the real hands: index -> pads 4 / 5, middle -> 2 / 3; v2 dex3.py)


@dataclass(frozen=True)
class GripSpec:
    closed: str  # the preset the close runs toward
    pre: float  # the pre-shape: this fraction of the way from open to closed (closing joints)
    joints: tuple[int, ...]  # the joints that close (thumb_0 = the thumb's rotation, set by the pre-shape)
    squeeze: float  # rad past contact (DESIGN §6.2: wrap 0.3, pinch 0.4)


GRIPS: dict[str, GripSpec] = {
    "wrap": GripSpec("fist", 0.0, (0, 1, 2, 3, 4, 5, 6), 0.3),
    "pinch": GripSpec("pinch_index", 0.35, (0, 1, 2, 5, 6), 0.4),
    # a fingertip pinch for thin layers: the same tip-to-tip close, a lighter squeeze (the tips press each other
    # through the layer; kp 1.5 x 0.25 = 0.375 N m per joint - the heat stays low, HW1 B3.3)
    "tip": GripSpec("pinch_index", 0.35, (0, 1, 2, 5, 6), 0.25),
}


def close_dir(side: str) -> FloatArray:
    return CLOSE_DIR_LEFT.copy() if side == "left" else MIRROR * CLOSE_DIR_LEFT


def clamp(q: npt.ArrayLike, side: str) -> FloatArray:
    lo = Q_MIN[side] + MARGIN * (Q_MIN[side] < 0)
    hi = Q_MAX[side] - MARGIN * (Q_MAX[side] > 0)
    return np.asarray(np.clip(np.asarray(q, dtype=np.float64), lo, hi), dtype=np.float64)


def preset(name: str, side: str) -> FloatArray:
    q = PRESETS_LEFT[name]
    return clamp(q.copy() if side == "left" else MIRROR * q, side)


def preshape(grip: str, side: str) -> FloatArray:
    """The hand before a grip closes (v2 dex3.preshape): the closing joints part-way, the thumb rotated, the fingers
    that take no part curled to the closed preset."""
    spec = GRIPS[grip]
    qc, qo = preset(spec.closed, side), preset("open", side)
    q = qo + spec.pre * (qc - qo)
    q[0] = qc[0]
    others = [i for i in range(N) if i not in spec.joints]
    q[others] = qc[others]
    return clamp(q, side)


# Kinematics (palm frame; Unitree's dex3_1 URDF joint origins, one revolute joint per link; the same chain as
# r1sim.dex3.link_poses, tests check it). LEFT-hand vectors; the right hand mirrors y.
_CHAIN: tuple[tuple[str, str, tuple[float, float, float], tuple[float, float, float]], ...] = (
    ("thumb_0", "palm", (0.0255, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("thumb_1", "thumb_0", (-0.0025, -0.0193, 0.0), (0.0, 0.0, 1.0)),
    ("thumb_2", "thumb_1", (0.0, -0.0458, 0.0), (0.0, 0.0, 1.0)),
    ("middle_0", "palm", (0.0777, 0.0016, -0.0285), (0.0, 0.0, 1.0)),
    ("middle_1", "middle_0", (0.0458, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("index_0", "palm", (0.0777, 0.0016, 0.0285), (0.0, 0.0, 1.0)),
    ("index_1", "index_0", (0.0458, 0.0, 0.0), (0.0, 0.0, 1.0)),
)
_JOINT_INDEX = {"thumb_0": 0, "thumb_1": 1, "thumb_2": 2, "middle_0": 3, "middle_1": 4, "index_0": 5, "index_1": 6}
TIP_M = 0.05  # the fingertip point past the last joint (xr_teleoperate; r1sim.dex3.TIP)
# The collision capsules (link frame, LEFT: fingers along +x, the thumb along -y; r1sim.dex3.hand_xml "primitives")
_CAPSULES: dict[str, tuple[float, float, float]] = {  # link -> (from, to along the link's axis, radius)
    "thumb_1": (0.006, 0.040, 0.0095),
    "thumb_2": (0.004, 0.040, 0.0085),
    "middle_0": (0.006, 0.040, 0.0095),
    "middle_1": (0.004, 0.040, 0.0085),
    "index_0": (0.006, 0.040, 0.0095),
    "index_1": (0.004, 0.040, 0.0085),
}
TIP_RADIUS_M = 0.0085  # the distal capsule: a tip touching a surface has its point about this far above it


def _rot(axis: tuple[float, float, float], a: float) -> FloatArray:
    x, y, z = axis
    c, s = np.cos(a), np.sin(a)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def link_frames(side: str, q: npt.ArrayLike) -> dict[str, tuple[FloatArray, FloatArray]]:
    """Link name -> (position (3,), rotation (3, 3)) in the palm frame at joint angles q (DDS order)."""
    qq = np.asarray(q, dtype=np.float64)
    m = 1.0 if side == "left" else -1.0
    out: dict[str, tuple[FloatArray, FloatArray]] = {"palm": (np.zeros(3), np.eye(3))}
    for name, parent, origin, axis in _CHAIN:
        pp, pr = out[parent]
        o = np.array([origin[0], m * origin[1], origin[2]])
        out[name] = (pp + pr @ o, pr @ _rot(axis, float(qq[_JOINT_INDEX[name]])))
    return out


def _axis_dir(side: str, link: str) -> FloatArray:
    """A link's own axis direction (the thumb links run along -y on the left hand, +y on the right; fingers +x)."""
    if link.startswith("thumb"):
        return np.array([0.0, -1.0 if side == "left" else 1.0, 0.0])
    return np.array([1.0, 0.0, 0.0])


def tips(side: str, q: npt.ArrayLike) -> dict[str, FloatArray]:
    """The three fingertip points (palm frame, m): r1sim's `<side>_hand_{thumb,index,middle}_tip` sites."""
    f = link_frames(side, q)
    out = {}
    for name, link in (("thumb", "thumb_2"), ("index", "index_1"), ("middle", "middle_1")):
        p, r = f[link]
        out[name] = p + r @ (TIP_M * _axis_dir(side, link))
    return out


def _seg_dist(p1: FloatArray, q1: FloatArray, p2: FloatArray, q2: FloatArray) -> float:
    """The shortest distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection §5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    c, b = float(d1 @ r), float(d1 @ d2)
    den = a * e - b * b
    s = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if den > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
    elif t > 1.0:
        t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def thumb_finger_clearance(side: str, q: npt.ArrayLike, *, tips_pair: bool = True) -> dict[str, float]:
    """Per finger: the smallest surface distance (m, < 0 = overlapping) between the thumb's capsules (thumb_1, thumb_2)
    and that finger's (proximal + distal). The thumb resting on its own curled finger = the 9 Oct false hold (a wrap
    with the thumb rotated runs the thumb into the index with the tips still ~15 mm apart). `tips_pair` False leaves
    out thumb_2 vs the distal link (a pinch closes the tips onto each other by design: the gap term judges that)."""
    f = link_frames(side, q)

    def seg(link: str) -> tuple[FloatArray, FloatArray, float]:
        a, b, rad = _CAPSULES[link]
        p, r = f[link]
        u = r @ _axis_dir(side, link)
        return p + a * u, p + b * u, rad

    out = {}
    for finger in ("index", "middle"):
        best = np.inf
        for tl in ("thumb_1", "thumb_2"):
            p1, q1, r1 = seg(tl)
            for fl in (f"{finger}_0", f"{finger}_1"):
                if not tips_pair and tl == "thumb_2" and fl.endswith("_1"):
                    continue
                p2, q2, r2 = seg(fl)
                best = min(best, _seg_dist(p1, q1, p2, q2) - r1 - r2)
        out[finger] = float(best)
    return out


def tip_clearance(side: str, q: npt.ArrayLike, finger: str = "index") -> float:
    """The surface distance (m, <= 0 = touching) between the thumb's distal capsule and the finger's distal capsule:
    the layer between the two tip shells in a tip pinch (the real shells are not capsules: HW2's air pinch gives each
    hand's offset, `layer_zero_m`)."""
    f = link_frames(side, q)
    a, b, rt = _CAPSULES["thumb_2"]
    pt, rr = f["thumb_2"]
    ut = rr @ _axis_dir(side, "thumb_2")
    c, d, rf = _CAPSULES[f"{finger}_1"]
    pf, rf_ = f[f"{finger}_1"]
    uf = rf_ @ _axis_dir(side, f"{finger}_1")
    return _seg_dist(pt + a * ut, pt + b * ut, pf + c * uf, pf + d * uf) - rt - rf


def empty_aperture(side: str, finger: str = "index", grip: str = "pinch") -> float:
    """The thumb-tip to finger-tip distance (m) at which the two distal capsules first touch along the grip's close
    with nothing between them - the 'tips met' reference the gap / layer terms subtract (the presets put the tip
    POINTS together: the real tips meet earlier). Calibrated per hand at HW2 (an air pinch at tare).

    Cached (HW2, 12 Oct 2026): a constant of the hand model, but a 401-step capsule search - ~50 ms per call on the
    Orin. GraspRunner computed it twice at every grasp's start; meanwhile r1hands' reader thread could not run, the
    hand state aged past its 100 ms freshness limit and the grasp refused "no valid hand state" before any finger
    closed (hand.grasp / hand.calibrate ~2 of 3 times on the robot). r1d warms it at start (`warm_grip_caches`)."""
    return _empty_aperture(str(side), str(finger), str(grip))


@cache
def _empty_aperture(side: str, finger: str, grip: str) -> float:
    spec = GRIPS[grip]
    q0, q1 = preshape(grip, side), preset(spec.closed, side)
    for s in np.linspace(0.0, 1.0, 401):
        q = q0 + s * (q1 - q0)
        if tip_clearance(side, q, finger) <= 0.0:
            tp = tips(side, q)
            return float(np.linalg.norm(tp["thumb"] - tp[finger]))
    tp = tips(side, q1)
    return float(np.linalg.norm(tp["thumb"] - tp[finger]))


def warm_grip_caches() -> None:
    """Compute the model-only references a GraspRunner reads at its start (`empty_aperture` per side / finger, the
    pinch close it is defined on) once, ahead of the first grasp - r1d calls it at start-up when it has hands."""
    for side in SIDES:
        for finger in ("index", "middle"):
            empty_aperture(side, finger, "pinch")


# ------------------------------------------------------------------------------------------------ thresholds
KP_GRIP = float(KP_DEFAULT[1])  # 1.5 (Unitree's clients; v2 Hand kp)
KD_GRIP = float(KD_DEFAULT[1])  # 0.2
CLOSE_SPEED = 1.2  # rad/s along the closing path's largest joint (v2 Hand._start_grasp speed default)
PREOPEN_FAST = 2.0  # x the speed until the tips are PREOPEN_MARGIN_M wider than a known object (v2)
PREOPEN_MARGIN_M = 0.02
PREOPEN_MIN_GAP_M = 0.04
TIMEOUT_S = 6.0  # the close ends here at the latest (v2)
STALL_ERR = 0.15  # rad behind the closing path ... (v2 Hand.STALL_ERR)
STALL_WIN_S = 0.20  # ... with the joints (nearly) stopped this long (v2 STALL_WIN)
STALL_MOVE = 0.02  # rad they may still move within STALL_WIN_S
BLOCK_WIN_S, BLOCK_RATE = 0.40, 0.15  # s, rad/s: far behind and creeping slower than this = blocked (v2)
TAU_CONTACT_NM = 0.30  # a calibrated torque with the joints stopped = touching (v2)
CLOSED_TOL = 0.10  # rad from the closed target: the group closed (on nothing) (v2)
PINCH_PAST = 0.25  # rad: a pinch / tip close runs this far past its preset (the tips meet before the preset)
PAD_MIN_KPA = 1.5  # a pad's contact level above its reading at the close's start (v2 Tactile min_kpa) ...
PAD_NOISE_K = 6.0  # ... or this many standard deviations of its noise before the close
PAD_PERSIST_S = 0.03  # the rise must last this long (DESIGN §6.2: 30 ms persistence)
CONTACT_HOLD = 0.05  # rad past contact while only one side touches (v2 CONTACT_HOLD; sync close)
SQUEEZE_RAMP_S = 0.3  # contact -> the full squeeze once both sides touch (v2)
MAX_ERR_TAU = 0.9  # N m: kp x closing error never above this (v2 Hand.MAX_ERR_TAU)
SETTLE_S = 0.4  # after the squeeze ramp, before the verdict (the force builds; v2 verdict at 0.25-0.6 s)
TAU_SCALE = 1e-6  # N m per raw tau_est unit (HW1 B3.1: v2's 7 Oct calibration confirmed as kp x error)
TAU_PLAUSIBLE_NM = 5.0  # a reading above this is junk (v2: the hand service's raw read 840+ "N m" at rest)
LEVER_PROX, LEVER_DIST = 0.072, 0.028  # m: proximal / distal joint to a contact on the distal link (v2)
TAXEL_AREA_M2 = np.array([1.2e-5 if k not in (1, 3, 5) else 0.9e-5 for k in range(N_PADS)])  # (v2 TAXEL_AREA_M2)
NO_DATA = 30000.0  # Dex3-1 manual: 30000 = no data in a pressure slot
SELF_TOUCH_M = 0.002  # the thumb's capsules within this of a finger's: the hand touches itself
GAP_FRAC = 0.5  # rigid: the aperture >= this x the object's width (v2's opt-in F2 rule, made required) ...
GAP_MIN_M = 0.006  # ... or, without a width, at least this (tips that met hold nothing)
SOFT_GAP_FRAC = 0.3  # soft objects squash
# Session 13 (HW2 day 2, 13 Oct 2026; GRIP.md "Session 13"): the hand model reads a CLOSED PINCH narrower than the
# object between the tips - the aperture is the distance of the model's tip POINTS, and a pinch holds an object on the
# distal PADS with the distal links converging, so the tip points sit closer together than the object's sides. The
# real held pinches of day 2 (grasp traces, the user saw every one held): a 21 mm object -> 8.37-8.89 mm (12.1-12.6 mm
# under), a 48 mm object -> 42.3-43.5 mm (4.5-5.7 mm under); the real air pinch (tips on tips, day 1's H3) -> +1.55 /
# +5.48 mm (right / left). With 0.5 x width the held 21 mm object failed 4 / 4 (day 1: 5 / 5) and the low pinches of a
# 5.4 cm-footprint object (held by its narrower body) 4 / 4. A pinch's width rule allows the largest measured
# under-read (+ 0.4 mm) ...
PINCH_MODEL_ERR_M = 0.013
# ... and, because the width rule alone then lets a pinch whose tips met pass at GAP_MIN_M (the real LEFT air pinch
# read 5.48 mm, 0.5 mm under it), a pinch must also show the tips did NOT meet: the distal shells' clearance (the
# capsule model, `tip_clearance`) at least this far above the hand's air-pinch reference (layer_zero_m; none = the
# model's own 0, conservative: the real tips meet at -6.42 / -9.30 mm). Day 2: the held 21 mm object +8.4-8.7 mm, the
# 5.4 cm object's low pinches +17.1-17.2, the 48 mm object +38.2-39.0; the real air pinches -6.42 / -9.30 (H3) - empty
# pinches fail by 9-12 mm.
PINCH_MET_MARGIN_M = 0.003
PINCH_GRIPS = ("pinch", "tip")  # the grips that close the thumb tip onto the index tip (GRIPS: pinch_index)
LAYER_MIN_M = 0.0005  # thin_flexible: the layer between the tips (single cotton jersey ~0.5-1 mm; HW2 calibrates) ...
LAYER_MAX_M = 0.012  # ... and at most this (more = the tips did not close on a thin layer)
SURFACE_MARGIN_M = 0.004  # a fingertip whose lowest point is within this of the support surface touches the surface
WIDTH_TOL_M = 0.015  # tips at the surface: the aperture at most the width + this (closed on the object's sides)
HOLD_MIN_N = 1.5  # N, the grip force a hold needs at least (v2 HOLD_MIN_N: necessary, not sufficient here)
LIFT_CHECK_M = 0.01  # the lift that starts the check (DESIGN §6.2: 1-2 cm)
LIFT_OBS_S = 0.3  # the contact must persist this long after the lift began
RUN_ON_RAD = 0.10  # a group closing this far past the verdict's pose ran on (its object left; v2 HOLD_ESCAPE 0.15)
SOFT_RUN_ON_RAD = 0.25  # soft objects creep (v2 HOLD_FOLLOW)
HELD_BACK_FRAC = 0.5  # of the squeeze: a group this far short of its command is held back by something (v2)
RELAX_BACKOFF = 0.03  # rad: the heat abort backs off to this short of where the fingers ARE (v2; session 11: only it)
TEMP_EASE_C = 60.0  # the squeeze halves from here (v2 TEMP_EASE; HW1: 0.45 N m plateaus ~60 C)
TEMP_ABORT_C = 72.0  # the grasp ends (relaxed, still powered) - HW1's watchdog stopped the 0.9 N m hold at 68-69 C
CMD_TTL_S = 0.5  # every tick re-sends the command with this TTL (r1hands holds where it is if the ticks stop)


@dataclass
class GripVerdict:
    state: str
    reason: str | None
    evidence: dict[str, Any]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "reason": self.reason, "evidence": self.evidence, "confidence": self.confidence}


@dataclass
class _Group:
    name: str
    idx: list[int]
    why: str | None = None  # why it stopped: pad / stall / blocked / torque / self / closed / timeout
    q_at: FloatArray | None = None  # where it stopped
    hist: list[tuple[float, FloatArray]] = field(default_factory=list)
    pad_ticks: int = 0
    pad_felt: bool = False


class _Pads:
    """The 9 pads relative to their reading at the close's start: risen per taxel by max(PAD_MIN_KPA, PAD_NOISE_K x
    the noise seen before the close). Raw values are absolute pressure (manual: >= 1e5 valid, 30000 = no data); the
    reference cancels any offset (v2 Tactile.rise_contact: the pads' zero moves with the finger shape and heat)."""

    def __init__(self) -> None:
        self.hist: list[FloatArray] = []
        self.ref: FloatArray | None = None
        self.noise: FloatArray | None = None
        self.valid: npt.NDArray[np.bool_] | None = None
        self.rise_kpa = np.zeros((N_PADS, 12))
        self.ok = False

    def update(self, raw: Any) -> None:
        if raw is None:
            self.ok = False
            return
        p = np.asarray(raw, dtype=np.float64).reshape(N_PADS, -1)[:, :12]
        valid = (np.abs(p - NO_DATA) > 0.5) & (p > 0) & np.isfinite(p)
        self.ok = bool(valid.any())
        self.hist.append(np.where(valid, p, np.nan))
        del self.hist[:-20]
        if self.ref is not None and self.valid is not None:
            self.rise_kpa = np.where(self.valid & valid, p - self.ref, 0.0) / 1000.0

    def reference(self) -> None:
        if not self.hist:
            return
        h = np.array(self.hist[-10:])
        valid = np.all(np.isfinite(h), axis=0)
        hv = np.where(valid, h, 0.0)
        self.ref = np.median(hv, axis=0)
        self.noise = np.std(hv, axis=0)
        self.valid = valid

    def contact(self) -> FloatArray:
        """(9,) bool: a taxel of the pad risen above its contact level."""
        if self.ref is None or self.valid is None or self.noise is None:
            return np.zeros(N_PADS, dtype=bool)
        thr = np.maximum(PAD_MIN_KPA, PAD_NOISE_K * self.noise / 1000.0)
        return np.asarray(np.any((self.rise_kpa > thr) & self.valid, axis=1))

    def kpa(self) -> FloatArray:
        return np.asarray(np.max(self.rise_kpa, axis=1))

    def force_n(self) -> FloatArray:
        return np.asarray(np.sum(np.maximum(self.rise_kpa, 0.0), axis=1) * 1000.0 * TAXEL_AREA_M2)


def air_pinch_layer(verdict: GripVerdict) -> float:
    """The hand's air-pinch reference (m) from the verdict of a `tip` pinch closed on NOTHING with layer_zero_m=0.0:
    the tip-shell clearance where the tips stopped on each other - pass it as `layer_zero_m` to every thin pinch."""
    layer = verdict.evidence.get("terms", {}).get("layer", {}).get("layer_mm")
    if layer is None:
        raise ValueError("not a thin_flexible verdict (no layer term): calibrate with a tip pinch on nothing")
    return float(layer) / 1000.0


ContactHeight = Callable[[FloatArray], Mapping[str, float]]


def make_contact_height(
    palm_pose: Callable[[], tuple[npt.ArrayLike, npt.ArrayLike]], surface_z_m: float, side: str
) -> ContactHeight:
    """`contact_height` from the palm's pose in a frame whose z is up (the arm's forward kinematics: position (3,),
    rotation (3, 3)) and the declared support surface's height in that frame: the fingertip points' heights above it."""

    def f(q: FloatArray) -> dict[str, float]:
        p, r = palm_pose()
        pa, ra = np.asarray(p, dtype=np.float64), np.asarray(r, dtype=np.float64).reshape(3, 3)
        return {k: float((pa + ra @ v)[2] - surface_z_m) for k, v in tips(side, q).items()}

    return f


class GraspRunner:
    """One grasp of one hand (see the module docstring). `hands` = r1mc3.hands.dex3.Dex3Hands or anything with its
    `command(side, q, *, dq, kp, kd, tau, ttl_s)` / `state(side)` (q, dq, tau_est raw, pressure 9 x 12, temperature,
    valid). Times are monotonic ns (the caller's clock: r1d's tick, the simulator's time in lockstep).

    `width_m`: the object's measured width across the grip (perception / the world model), None = unknown.
    `squeeze`: rad past contact (default per grip). `surface_z_m` is informational (the caller's `contact_height`
    already measures from the surface). `contact_height(q_hand)` -> {thumb, index, middle: tip point height above the
    support surface, m} (make_contact_height builds one from the arm's forward kinematics); None = the term is not
    judged (said in the evidence). `aperture_zero_m`: the tips-met distance (calibrated per hand; default the model's);
    `layer_zero_m`: the tip-shell clearance an air pinch reads (`air_pinch_layer`; REQUIRED for thin_flexible:
    calibrated per hand - at HW2 on the real hands).
    """

    def __init__(
        self,
        hands: Any,
        side: str,
        grip: str = "wrap",
        object_class: str = "rigid",
        width_m: float | None = None,
        squeeze: float | None = None,
        surface_z_m: float | None = None,
        contact_height: ContactHeight | None = None,
        *,
        aperture_zero_m: float | None = None,
        layer_zero_m: float | None = None,
        tau_scale: float = TAU_SCALE,
        speed: float = CLOSE_SPEED,
        sync: bool = True,
    ) -> None:
        if side not in SIDES:
            raise ValueError(f"side: one of {SIDES}")
        if grip not in GRIPS:
            raise ValueError(f"grip: one of {tuple(GRIPS)}")
        if object_class not in OBJECT_CLASSES:
            raise ValueError(f"object_class: one of {OBJECT_CLASSES}")
        if width_m is not None and not (0.0 < float(width_m) < 0.3):
            raise ValueError("width_m: (0, 0.3) m")
        if object_class == "thin_flexible" and layer_zero_m is None:
            # an uncalibrated layer term passes an AIR pinch (the sim's hand: its tip pads read a 2.2 mm "layer" with
            # nothing between them; the real shells are not the model's capsules either) - so it is required
            raise ValueError(
                "thin_flexible needs the hand's air-pinch reference layer_zero_m (calibrate: a tip pinch on nothing "
                "with layer_zero_m=0.0, then air_pinch_layer(verdict))"
            )
        self.hands, self.side, self.grip, self.object_class = hands, side, grip, object_class
        self.spec = GRIPS[grip]
        self.width_m = None if width_m is None else float(width_m)
        self.squeeze = float(self.spec.squeeze if squeeze is None else squeeze)
        if not 0.0 <= self.squeeze <= 0.6:
            raise ValueError("squeeze: 0 .. 0.6 rad past contact")
        self.surface_z_m = surface_z_m
        self.contact_height = contact_height
        self.tau_scale = float(tau_scale)
        self.speed = float(speed)
        self.sync = bool(sync)
        self.d = close_dir(side)
        self.finger_groups = [g for g in ("index", "middle") if set(GROUPS[g]) & set(self.spec.joints)]
        self.aperture_zero = {
            f: float(aperture_zero_m) if aperture_zero_m is not None else empty_aperture(side, f, "pinch")
            for f in ("index", "middle")
        }
        self.layer_zero_m = 0.0 if layer_zero_m is None else float(layer_zero_m)
        self.state: str = "OPEN"
        self.reason: str | None = None
        self.evidence: dict[str, Any] = {}
        self.confidence = 0.0
        self.pads = _Pads()
        self.events: list[dict[str, Any]] = []
        self._q: FloatArray | None = None
        self._tau = np.zeros(N)
        self._tau_ok = False
        self._temp = 0.0
        self._cmd: FloatArray | None = None
        self._t = 0.0
        self._phase = "idle"
        self._groups: dict[str, _Group] = {}
        self._lift_dz = 0.0
        self._lift_t0: float | None = None
        self._lift_ok = True
        self._q_verdict: FloatArray | None = None
        self._layer_at_verdict: float | None = None
        self._layer_finger: str | None = None
        self._t_opp: float | None = None
        self._external = False  # judging a close another controller made (observe_external / judge_external)
        self._kept = False  # NOT_HOLDING with the grip kept (contact decides: only `release` opens the hand)

    # ------------------------------------------------------------------ io
    def _read(self) -> bool:
        st = self.hands.state(self.side)
        if not st.get("valid", True) or st.get("q") is None:
            return False
        self._q = np.asarray(st["q"], dtype=np.float64).copy()
        raw = st.get("tau_est", st.get("tau"))
        if raw is not None:
            t = np.asarray(raw, dtype=np.float64) * self.tau_scale
            self._tau_ok = bool(np.all(np.isfinite(t[1:])) and np.all(np.abs(t[1:]) <= TAU_PLAUSIBLE_NM))
            self._tau = np.where(np.isfinite(t), t, 0.0) if self._tau_ok else np.zeros(N)
        temp = st.get("temperature", st.get("temp"))
        if temp is not None:
            self._temp = float(np.nanmax(np.asarray(temp, dtype=np.float64)))
        self.pads.update(st.get("pressure"))
        return True

    def _send(self, q: FloatArray) -> None:
        assert self._q is not None
        if self._external:  # another controller drives the hand: only remember what the verdict assumes
            self._cmd = np.asarray(q, dtype=np.float64).copy()
            return
        qc = clamp(q, self.side)
        # kp x error never above MAX_ERR_TAU (finger safety, v2): the target stays within that of where the joint is
        lim = MAX_ERR_TAU / KP_GRIP
        qc = np.clip(qc, self._q - lim, self._q + lim)
        self._cmd = qc
        self.hands.command(
            self.side, qc, kp=np.full(N, KP_GRIP), kd=np.full(N, KD_GRIP), tau=np.zeros(N), ttl_s=CMD_TTL_S
        )

    def _event(self, kind: str, **kw: Any) -> None:
        self.events.append({"t": round(self._t, 3), "event": kind, **kw})

    # ------------------------------------------------------------------ public
    def start(self, now_ns: int) -> None:
        self._t = now_ns * 1e-9
        if not self._read() or self._q is None:
            self._finish_not("no valid hand state", {})
            return
        if self._temp >= TEMP_ABORT_C:
            self._finish_not(f"hand hot ({self._temp:.0f} C)", {}, relax=True)
            return
        q0 = self._q.copy()
        self._q_pre = self._pre_from(q0)
        self._q_closed = preset(self.spec.closed, self.side)
        if self.grip == "wrap":
            self._q_closed[0] = self._q_pre[0]
        else:  # the pinch presets put the tip POINTS together: the shells meet earlier - close past, so they press
            ix = [i for i in self.spec.joints if i != 0]
            self._q_closed[ix] += self.d[ix] * PINCH_PAST
            self._q_closed = clamp(self._q_closed, self.side)
        dist = float(np.max(np.abs(self._q_pre - q0)))
        self._pre_q0, self._pre_T, self._pre_t0 = q0, max(0.25, dist / 2.5), self._t
        self._s = 0.0
        self._s_fast = self._fast_until()
        self._groups = {
            g: _Group(g, [i for i in GROUPS[g] if i in self.spec.joints]) for g in ("thumb", *self.finger_groups)
        }
        self._t_opp = None
        self._phase = "preshape"
        self.state = "CLOSING"
        self._event("start", grip=self.grip, object_class=self.object_class, squeeze=self.squeeze, width_m=self.width_m)

    def tick(self, now_ns: int) -> str:
        self._t = now_ns * 1e-9
        if self._phase == "done_not" and self._kept and self.state == "NOT_HOLDING":
            # contact decides: the hand keeps its grip until it is told to open (r1hands holds the MEASURED pose when
            # the command's TTL runs out, i.e. the squeeze would fade - so the grip is re-sent every tick)
            if self._read() and self._q is not None:
                if self._temp >= TEMP_ABORT_C:
                    self._event("relaxed", why=f"hand hot ({self._temp:.0f} C >= {TEMP_ABORT_C:.0f})")
                    self._relax()
                else:
                    self._send(self._keep_cmd())
            return self.state
        if self._phase in ("idle", "done_not") or self.state == "OPEN":
            return self.state
        if not self._read() or self._q is None:
            if self.state in ("CLOSING", "CONTACT"):
                self._finish_not("hand state stale / invalid during the close", {})
            return self.state
        if self._temp >= TEMP_ABORT_C and self.state != "NOT_HOLDING":
            self._finish_not(f"hand hot ({self._temp:.0f} C >= {TEMP_ABORT_C:.0f})", self.evidence, relax=True)
            return self.state
        if self._phase == "external":
            return self.state
        if self._phase == "preshape":
            a = min(1.0, (self._t - self._pre_t0) / self._pre_T)
            a = a * a * a * (10 - 15 * a + 6 * a * a)
            self._send(self._pre_q0 + a * (self._q_pre - self._pre_q0))
            if a >= 1.0:
                self.pads.reference()
                self._phase, self._t_close, self._t_last = "close", self._t, self._t
            return self.state
        if self._phase == "close":
            self._tick_close()
        elif self._phase == "squeeze":
            self._send(self._hold_cmd())
            if self._t - self._t_sq >= self._squeeze_wait():
                self._judge()
        elif self._phase == "hold":
            self._send(self._hold_cmd())
            if self._lift_t0 is not None:
                self._tick_lift()
        return self.state

    def observe_external(self, now_ns: int) -> None:
        """Judge a close ANOTHER controller makes (validation: v3's verdict on the exact hand state of v2's false holds,
        A8): reads the state and takes the pads' reference now (call it when that close starts); then `tick` only reads
        (nothing is commanded) until `judge_external`."""
        self._t = now_ns * 1e-9
        self._external = True
        if not self._read():
            self._finish_not("no valid hand state", {})
            return
        self.pads.reference()
        self._phase, self.state = "external", "CLOSING"

    def judge_external(self, now_ns: int, stops: Mapping[str, str], cmd_q: npt.ArrayLike) -> GripVerdict:
        """The verdict on the hand as the other controller left it: `stops` = why each finger group stopped (pad /
        stall / blocked / torque / closed; a group missing = it did not stop), `cmd_q` = that controller's last hand
        command (the 'held back' measure of the lift check)."""
        self._t = now_ns * 1e-9
        if not self._read() or self._q is None:
            self._finish_not("no valid hand state", {})
            return self.verdict()
        self._cmd = np.asarray(cmd_q, dtype=np.float64).copy()
        clear = thumb_finger_clearance(self.side, self._q, tips_pair=self.grip == "wrap")
        self._groups = {
            g: _Group(g, [i for i in GROUPS[g] if i in self.spec.joints]) for g in ("thumb", *self.finger_groups)
        }
        for g in self._groups.values():
            why = stops.get(g.name)
            if why in ("pad", "tactile", "stall", "blocked", "torque") and self.object_class != "thin_flexible":
                why = "pad" if why == "tactile" else why
                if (g.name == "thumb" and min(clear.values()) < SELF_TOUCH_M) or (
                    g.name != "thumb" and clear.get(g.name, np.inf) < SELF_TOUCH_M
                ):
                    why = "self"
            g.why, g.q_at = (why or "timeout"), self._q.copy()
        self._judge()
        return self.verdict()

    def external_command(self, q: npt.ArrayLike) -> None:
        """External mode: the other controller's current hand command (the lift check's 'held back' measure)."""
        self._cmd = np.asarray(q, dtype=np.float64).copy()

    def notify_lift(self, dz_m: float) -> None:
        """The arm lifted the hand by dz_m (m, cumulative since the verdict; call while the arm moves). The lift check
        starts at LIFT_CHECK_M and decides TENTATIVE -> HOLDING | NOT_HOLDING within LIFT_OBS_S."""
        self._lift_dz = max(self._lift_dz, float(dz_m))
        if self.state == "TENTATIVE" and self._lift_t0 is None and self._lift_dz >= LIFT_CHECK_M:
            self._lift_t0 = self._t
            self._lift_ok = True
            self._event("lift_check", dz_mm=round(1000 * self._lift_dz, 1))

    def stops(self) -> dict[str, str | None]:
        """Why each finger group stopped so far (pad / stall / blocked / torque / self / closed / timeout; None =
        still closing or not started)."""
        return {g.name: g.why for g in self._groups.values()}

    @property
    def phase(self) -> str:
        """The sequence's phase: idle, preshape, close, squeeze, hold, done_not (finer than `state`; a NOT_HOLDING hand
        that keeps its grip stays in done_not - `keeps_grip`)."""
        return self._phase

    def verdict(self) -> GripVerdict:
        return GripVerdict(self.state, self.reason, dict(self.evidence), round(float(self.confidence), 3))

    def release(self, now_ns: int) -> None:
        """Open the hand (the open preset; the caller keeps ticking `tick` is not needed: one command, TTL held)."""
        self._t = now_ns * 1e-9
        if self._read() and self._q is not None:
            q = preset("open", self.side)
            q[0] = self._q[0]
            self.hands.command(self.side, q, kp=np.full(N, KP_GRIP), kd=np.full(N, KD_GRIP), ttl_s=1.0)
        self.state, self._phase = "OPEN", "idle"
        self._kept = False
        self._event("release")

    # ------------------------------------------------------------------ the close
    def _pre_from(self, q0: FloatArray) -> FloatArray:
        """v2 Hand._pre_grasp: no closing joint opens (each starts from where it is or the pre-shape, whichever is
        more closed, short of the closed target); a wrap keeps the thumb's current rotation."""
        q_pre = preshape(self.grip, self.side)
        q_closed = preset(self.spec.closed, self.side)
        if self.grip == "wrap":
            q_pre[0] = q0[0]
        ix = [i for i in self.spec.joints if i != 0]
        d = self.d
        for i in ix:
            cur = q0[i] if d[i] * (q0[i] - q_pre[i]) > 0 else q_pre[i]
            q_pre[i] = cur if d[i] * (q_closed[i] - cur) > 0.05 else q_closed[i] - d[i] * 0.05
        return clamp(q_pre, self.side)

    def _fast_until(self) -> float:
        """The closing path's fraction up to which the close runs PREOPEN_FAST x faster: the tips still at least
        PREOPEN_MARGIN_M wider than the known width (v2 hands.py:713-726).

        Session 12 (HW2 day 1's tape pinch, 2.1 cm): v2 took the LAST point with the gap that wide, which is the first
        one only while the gap shrinks all the way; v3's pinches close PINCH_PAST beyond the preset, where the tips
        pass each other and the gap grows again (the right pinch: 119 mm -> -1 mm at 65 % -> 46 mm at the end) - so
        for any object narrower than ~2.6 cm the whole close ran at double speed into it. Now: up to the first point
        where the gap falls below that width."""
        if self.width_m is None:
            return 0.0
        f = self.finger_groups[0] if self.finger_groups else "index"
        best = 0.0
        for s in np.linspace(0.0, 1.0, 41):
            tp = tips(self.side, self._q_pre + s * (self._q_closed - self._q_pre))
            gap = float(np.linalg.norm(tp["thumb"] - tp[f])) - self.aperture_zero[f]
            if gap < max(self.width_m + PREOPEN_MARGIN_M, PREOPEN_MIN_GAP_M):
                break
            best = float(s)
        return best

    def _still(self, g: _Group) -> bool:
        if not g.hist or self._t - g.hist[0][0] < STALL_WIN_S * 0.9:
            return False
        recent = [h for h in g.hist if self._t - h[0] <= STALL_WIN_S]
        last = g.hist[-1][1]
        return float(np.max(np.abs(np.asarray([h[1] for h in recent]) - last))) < STALL_MOVE

    def _creeping(self, g: _Group) -> bool:
        if self._t - g.hist[0][0] < BLOCK_WIN_S * 0.9:
            return False
        moved = float(np.max(np.abs(g.hist[-1][1] - g.hist[0][1])))
        return moved < BLOCK_RATE * (self._t - g.hist[0][0])

    def _tick_close(self) -> None:
        assert self._q is not None
        q_meas = self._q
        dt = max(self._t - self._t_last, 0.0)
        self._t_last = self._t
        delta = self._q_closed - self._q_pre
        span = float(np.max(np.abs(delta)))
        fast = PREOPEN_FAST if self._s < self._s_fast else 1.0
        if self._s < 1.0:
            self._s = min(1.0, self._s + fast * self.speed * dt / max(span, 1e-6))
        q_path = self._q_pre + self._s * delta
        q = q_path.copy()
        contact = self.pads.contact()
        clear = thumb_finger_clearance(self.side, q_meas, tips_pair=self.grip == "wrap")
        for g in self._groups.values():
            idx = g.idx if g.name != "thumb" else [i for i in g.idx if i != 0]
            if g.why is not None:
                continue
            g.hist.append((self._t, q_meas[idx].copy()))
            while len(g.hist) > 2 and self._t - g.hist[1][0] >= max(STALL_WIN_S, BLOCK_WIN_S):
                g.hist.pop(0)
            lag = self.d[idx] * (q_path[idx] - q_meas[idx])
            pads_on = bool(np.any(contact[list(GROUP_PADS[g.name])]))
            g.pad_ticks = g.pad_ticks + 1 if pads_on else 0
            g.pad_felt = g.pad_felt or pads_on
            still = self._still(g)
            why = None
            if g.pad_ticks * 0.01 >= PAD_PERSIST_S - 1e-9:
                why = "pad"
            elif np.any(lag > STALL_ERR) and self._t - self._t_close > 0.15 and still:
                why = "stall"
            elif np.any(lag > 2 * STALL_ERR) and self._t - self._t_close > 0.4 and self._creeping(g):
                why = "blocked"
            elif self._tau_ok and np.any(np.abs(self._tau[idx]) > TAU_CONTACT_NM) and self._s > 0.05 and still:
                why = "torque"
            elif self._s >= 1.0 and still and float(np.max(np.abs(q_meas[idx] - self._q_closed[idx]))) < CLOSED_TOL:
                why = "closed"
            if why is None:
                continue
            # the hand touching itself: the thumb on a curled finger (or a finger on the thumb) is no object
            if why in ("pad", "stall", "blocked", "torque") and self.object_class != "thin_flexible":
                if g.name == "thumb" and min(clear.values()) < SELF_TOUCH_M:
                    why = "self"
                elif g.name != "thumb" and clear.get(g.name, np.inf) < SELF_TOUCH_M:
                    why = "self"
            g.why, g.q_at = why, q_meas.copy()
            self._event("finger_stop", group=g.name, why=why, s=round(self._s, 3))
        for g in self._groups.values():
            if g.why is not None:
                q[g.idx if g.name != "thumb" else [i for i in g.idx if i != 0]] = self._group_hold(g)
        if self.state == "CLOSING" and any(
            g.why in ("pad", "stall", "blocked", "torque") for g in self._groups.values()
        ):
            self.state = "CONTACT"
        self._send(q)
        done = all(g.why is not None for g in self._groups.values())
        if done or self._t - self._pre_t0 > TIMEOUT_S:
            for g in self._groups.values():
                if g.why is None:
                    g.why, g.q_at = "timeout", q_meas.copy()
            self._phase, self._t_sq = "squeeze", self._t

    def _opposed(self) -> bool:
        ev = ("pad", "stall", "blocked", "torque")
        th = self._groups["thumb"].why in ev
        return th and any(self._groups[f].why in ev for f in self.finger_groups)

    def _squeeze_now(self) -> float:
        sq = self.squeeze
        if self._temp >= TEMP_EASE_C:
            sq *= 0.5
        return sq

    def _group_hold(self, g: _Group) -> FloatArray:
        idx = g.idx if g.name != "thumb" else [i for i in g.idx if i != 0]
        assert g.q_at is not None
        if g.why in ("closed", "timeout", "self"):
            return np.asarray(g.q_at[idx], dtype=np.float64)  # nothing to squeeze (or the hand itself): no push
        sq = self._squeeze_now()
        if self.sync and not self._opposed():
            push = min(sq, CONTACT_HOLD)
        elif self.sync:
            if self._t_opp is None:
                self._t_opp = self._t
            a = min(1.0, (self._t - self._t_opp) / SQUEEZE_RAMP_S)
            push = min(sq, CONTACT_HOLD) + a * (sq - min(sq, CONTACT_HOLD))
        else:
            push = sq
        return np.asarray(g.q_at[idx] + self.d[idx] * push, dtype=np.float64)

    def _hold_cmd(self) -> FloatArray:
        assert self._cmd is not None
        q = self._cmd.copy()
        for g in self._groups.values():
            if g.q_at is not None:
                q[g.idx if g.name != "thumb" else [i for i in g.idx if i != 0]] = self._group_hold(g)
        return q

    def _squeeze_wait(self) -> float:
        return (SQUEEZE_RAMP_S if self._opposed() and self.sync else 0.0) + SETTLE_S

    # ------------------------------------------------------------------ the verdict
    def _aperture(self, finger: str) -> float:
        assert self._q is not None
        tp = tips(self.side, self._q)
        return float(np.linalg.norm(tp["thumb"] - tp[finger])) - self.aperture_zero[finger]

    def _held_back(self, g: _Group) -> float:
        """How far (rad) the group is short of its command along the closing direction (something resists)."""
        assert self._q is not None and self._cmd is not None
        idx = g.idx if g.name != "thumb" else [i for i in g.idx if i != 0]
        return float(np.max(self.d[idx] * (self._cmd[idx] - self._q[idx])))

    def _group_force(self, g: _Group) -> float:
        pads = self.pads.force_n()[list(GROUP_PADS[g.name])].sum()
        idx = g.idx if g.name != "thumb" else [i for i in g.idx if i != 0]
        f_t = 0.0
        if self._tau_ok and len(idx) == 2:
            tp, td = (max(0.0, float(self.d[i] * self._tau[i])) for i in idx)
            f_t = (tp * LEVER_PROX + td * LEVER_DIST) / (LEVER_PROX**2 + LEVER_DIST**2)
        return float(min(50.0, max(pads, f_t)))

    def _judge(self) -> None:
        assert self._q is not None
        ev_ok = ("pad", "stall", "blocked", "torque")
        th = self._groups["thumb"]
        touching = [f for f in self.finger_groups if self._groups[f].why in ev_ok]
        clear = thumb_finger_clearance(self.side, self._q, tips_pair=self.grip == "wrap")
        e: dict[str, Any] = {"object_class": self.object_class, "grip": self.grip}
        e["stops"] = {g.name: g.why for g in self._groups.values()}
        e["self_clearance_mm"] = {k: round(1000 * v, 2) for k, v in clear.items()}
        pk = self.pads.kpa()
        e["pads_kpa"] = [round(float(x), 2) for x in pk]
        e["pads_contact"] = [bool(x) for x in self.pads.contact()]
        e["closure_q"] = [round(float(x), 4) for x in self._q]
        fingers = touching or self.finger_groups
        ap = {f: self._aperture(f) for f in fingers}
        best_f = max(ap, key=lambda k: ap[k]) if ap else None
        aperture = ap[best_f] if best_f else 0.0
        e["aperture_mm"] = {k: round(1000 * v, 2) for k, v in ap.items()}
        # contact height (when the caller gave the surface)
        h_low: dict[str, float] | None = None
        if self.contact_height is not None:
            hh = self.contact_height(self._q.copy())
            h_low = {k: float(hh[k]) - TIP_RADIUS_M for k in ("thumb", "index", "middle") if k in hh}
        terms: dict[str, Any] = {}
        if self.object_class == "thin_flexible":
            contact = self.pads.contact()
            fs = [f for f in self.finger_groups if contact[GROUP_PADS[f][1]]]
            tip_ok = bool(contact[GROUP_PADS["thumb"][1]] and fs)
            if not tip_ok and not self.pads.ok:  # no pads at all: both groups stopped on a stall / torque
                tip_ok = th.why in ("stall", "torque") and bool(touching)
            terms["tip_pressure"] = {
                "ok": tip_ok,
                "thumb_tip_kpa": round(float(pk[GROUP_PADS["thumb"][1]]), 2),
                "finger_tip_kpa": {f: round(float(pk[GROUP_PADS[f][1]]), 2) for f in self.finger_groups},
            }
            f_layer = fs[0] if fs else (best_f or self.finger_groups[0])
            layer = tip_clearance(self.side, self._q, f_layer) - self.layer_zero_m
            terms["layer"] = {
                "ok": bool(LAYER_MIN_M <= layer <= LAYER_MAX_M),
                "layer_mm": round(1000 * layer, 2),
                "range_mm": [1000 * LAYER_MIN_M, 1000 * LAYER_MAX_M],
                "rule": "tips closed on a thin layer: not met (0), not open",
            }
            self._layer_at_verdict = layer
            self._layer_finger = f_layer
            if h_low is not None:
                terms["contact_height"] = {
                    "ok": None,
                    "lowest_mm": {k: round(1000 * v, 1) for k, v in h_low.items()},
                    "rule": "reported only: a thin layer is pinched at the surface",
                }
        else:
            self_th = min(clear.values()) < SELF_TOUCH_M
            opp = th.why in ev_ok and bool(touching) and not self_th
            terms["opposition"] = {
                "ok": opp,
                "thumb": th.why,
                "fingers": {f: self._groups[f].why for f in self.finger_groups},
                "thumb_on_own_finger": bool(self_th),
                "rule": "thumb side AND a finger side on contact evidence, not the hand touching itself",
            }
            frac = SOFT_GAP_FRAC if self.object_class == "soft" else GAP_FRAC
            need = frac * self.width_m if self.width_m is not None else GAP_MIN_M
            rule = f"aperture >= {frac} x width" if self.width_m is not None else "aperture >= 6 mm (no width)"
            gap_ok = bool(aperture >= need)
            gap: dict[str, Any] = {}
            if self.grip in PINCH_GRIPS:
                # session 13: the pinch's width rule allows the hand model's measured under-read, and the tips must
                # not have met (the distal shells' clearance over the hand's air-pinch reference) - see the constants
                if self.width_m is not None:
                    need = max(GAP_MIN_M, need - PINCH_MODEL_ERR_M)
                    rule = (f"aperture >= {frac} x width - {1000 * PINCH_MODEL_ERR_M:.0f} mm (the model's pinch "
                            f"under-read), >= 6 mm")  # fmt: skip
                f_met = best_f or self.finger_groups[0]
                met = tip_clearance(self.side, self._q, f_met) - self.layer_zero_m
                met_ok = bool(met >= PINCH_MET_MARGIN_M)
                gap_ok = bool(aperture >= need) and met_ok
                gap = {"tips_clearance_mm": round(1000 * met, 2), "tips_met": not met_ok,
                       "tips_clearance_need_mm": 1000 * PINCH_MET_MARGIN_M}  # fmt: skip
                rule += f"; the tips did not meet (shell clearance >= {1000 * PINCH_MET_MARGIN_M:.0f} mm)"
            terms["gap"] = {
                "ok": gap_ok,
                "aperture_mm": round(1000 * aperture, 2),
                "need_mm": round(1000 * need, 2),
                **gap,
                "rule": rule,
            }
            if h_low is not None:
                side_groups = ["thumb", *touching] if touching else ["thumb", *self.finger_groups]
                lows = {k: h_low[k] for k in side_groups if k in h_low}
                at_surface = bool(lows) and all(v < SURFACE_MARGIN_M for v in lows.values())
                ch: dict[str, Any] = {
                    "lowest_mm": {k: round(1000 * v, 1) for k, v in h_low.items()},
                    "margin_mm": 1000 * SURFACE_MARGIN_M,
                    "at_surface": at_surface,
                }
                if not at_surface:
                    ch.update(ok=True, rule="the touching fingertips above the support surface")
                elif self.width_m is not None:
                    # a low grasp (tips at the table, v2's lower-then-close) is real when the tips closed on the
                    # object's sides; tips stopped by the surface stop wherever they hit it
                    hi = self.width_m + WIDTH_TOL_M
                    ch.update(
                        ok=bool(need <= aperture <= hi),
                        rule=f"tips at the surface: the aperture must match the width ({1000 * need:.0f}-"
                        f"{1000 * hi:.0f} mm), else pressing the surface",
                    )
                else:
                    ch.update(ok=None, rule="tips at the surface, no width: the lift check decides")
                terms["contact_height"] = ch
            else:
                terms["contact_height"] = {"ok": None, "rule": "not judged: no support surface given"}
            fth = self._group_force(th)
            ffi = max((self._group_force(self._groups[f]) for f in touching), default=0.0)
            grip_n = min(fth, ffi)
            judged = self._tau_ok or self.pads.ok
            terms["force"] = {
                "ok": bool(grip_n >= HOLD_MIN_N) if judged else None,
                "grip_force_n": round(grip_n, 2),
                "need_n": HOLD_MIN_N,
                "torque_trusted": self._tau_ok,
            }
        e["terms"] = terms
        e["temperature_c"] = round(self._temp, 1)
        self.evidence = e
        failed = [k for k, v in terms.items() if v.get("ok") is False]
        self._q_verdict = self._q.copy()
        if failed:
            self._finish_not(f"{failed[0]}: {terms[failed[0]].get('rule', '')}".strip(), e)
            return
        self.state, self._phase = "TENTATIVE", "hold"
        self.confidence = self._confidence(terms, lift=None)
        self._event("tentative", aperture_mm=round(1000 * aperture, 2))
        if self._lift_dz >= LIFT_CHECK_M:
            self.notify_lift(self._lift_dz)

    def _tick_lift(self) -> None:
        assert self._q is not None and self._q_verdict is not None and self._lift_t0 is not None
        e = self.evidence
        terms = e["terms"]
        if self.object_class == "thin_flexible":
            c = self.pads.contact()
            f = self._layer_finger or (self.finger_groups[0] if self.finger_groups else "index")
            persist = bool(c[GROUP_PADS["thumb"][1]] and c[GROUP_PADS[f][1]])
            if not self.pads.ok:
                persist = True
            layer = tip_clearance(self.side, self._q, f) - self.layer_zero_m
            ok_now = persist and layer >= LAYER_MIN_M
            detail = {"tip_pressure": persist, "layer_mm": round(1000 * layer, 2)}
        else:
            allow = SOFT_RUN_ON_RAD if self.object_class == "soft" else RUN_ON_RAD
            if self._external:  # the other controller moves its own fingers (v2 re-squeezes in the lift): not judged
                allow = float("inf")
            c = self.pads.contact()
            run_on = 0.0
            persist = False
            touching = [g for g in self._groups.values() if g.why in ("pad", "stall", "blocked", "torque")]
            for g in touching:
                idx = g.idx if g.name != "thumb" else [i for i in g.idx if i != 0]
                run_on = max(run_on, float(np.max(self.d[idx] * (self._q[idx] - self._q_verdict[idx]))))
            held = [self._held_back(g) >= HELD_BACK_FRAC * self._squeeze_now() for g in touching]
            pads = [bool(np.any(c[list(GROUP_PADS[g.name])])) for g in touching]
            each = [h or p for h, p in zip(held, pads, strict=True)]
            # CONTACT DECIDES (the user, session 11) - in the lift check (TENTATIVE) AND the hold (HOLDING), the user's
            # answer at the end of session 11 for the lift check: the object has LEFT only when NO touching group keeps
            # any evidence (closing on nothing). One group's pad dropping out while the others hold (session 10: the
            # round object's load shifted onto thumb + index) is not a slip, and neither is a finger settling further
            # while the contact persists (round B: the tall cylinder at 0.101 rad, 8 N, every group touching; grip's
            # grid: 12 of 71 picks that really carried the object were called NOT_HOLDING by the strict lift check as
            # the fingers settled under the weight) - the run-on is reported, not judged. An empty close never reaches
            # the lift (no opposition: refused before it), so A8's empty hand stays NOT_HOLDING.
            persist = bool(touching) and any(each)
            detail = {"contact_persists": persist, "run_on_rad": round(run_on, 3), "allow_rad": allow}
            detail["run_on_judged"] = False
            if self.grip in PINCH_GRIPS and not self._external:
                # session 13 (HW2 day 2's two pinch slips): a pinch whose object slips out closes until the TIPS press
                # each other - held back by the other tip, which "contact decides" read as the object's contact. The
                # traces: the two slips' tip clearance fell from +18 / +20 mm to -6 mm (= the real air pinch, -6.42)
                # 1.9 / 2.9 s after the lift began, while the holds settled at +17-18 mm and stayed
                f_met = next((g.name for g in touching if g.name != "thumb"), self.finger_groups[0])
                met = tip_clearance(self.side, self._q, f_met) - self.layer_zero_m
                detail["tips_clearance_mm"] = round(1000 * met, 2)
                if met < PINCH_MET_MARGIN_M:
                    persist = False
                    detail["contact_persists"] = False
                    detail["tips_met"] = True
            ok_now = persist
        self._lift_ok = self._lift_ok and ok_now
        if not ok_now:
            terms["lift"] = {"ok": False, "dz_mm": round(1000 * self._lift_dz, 1), **detail}
            why = ("lift: the tips met (the object left the pinch)" if detail.get("tips_met")
                   else "lift: the contact did not persist / the fingers ran on")  # fmt: skip
            self._finish_not(why, e)
            return
        if self._t - self._lift_t0 >= LIFT_OBS_S and self.state != "HOLDING":  # (once: the hold phase keeps watching)
            terms["lift"] = {"ok": True, "dz_mm": round(1000 * self._lift_dz, 1), **detail}
            self.state = "HOLDING"
            self.reason = None
            self.confidence = self._confidence(terms, lift=True)
            self._event("holding")

    def _confidence(self, terms: Mapping[str, Any], lift: bool | None) -> float:
        judged = [v for v in terms.values() if v.get("ok") is not None]
        if not judged:
            return 0.0
        base = sum(1.0 for v in judged if v["ok"]) / len(judged)
        return float(base * (1.0 if lift else 0.6))

    def _finish_not(self, reason: str, evidence: dict[str, Any], *, relax: bool = False) -> None:
        """NOT_HOLDING. The hand KEEPS its grip (contact decides, the user's rule: a NOT_HOLDING verdict never relaxes
        the fingers by itself - `tick` keeps re-sending the grip's command until `release`); only `relax` (the heat
        abort: the motors' protection) backs off to just short of where the fingers ARE (powered, never limp)."""
        self.state, self.reason, self._phase = "NOT_HOLDING", reason, "done_not"
        self.evidence = evidence
        self.confidence = 0.9 if evidence.get("terms") else 0.5
        self._kept = (not relax) and self._q is not None and self._cmd is not None and not self._external
        self._event("not_holding", reason=reason, kept=self._kept)
        if relax and self._q is not None and not self._external:
            self._relax()

    def _relax(self) -> None:
        """Back off to just short of where the fingers ARE (powered, not limp: v2's relax) - the heat abort only."""
        assert self._q is not None
        self._kept = False
        q = self._q - self.d * RELAX_BACKOFF
        q[0] = self._q[0]
        self.hands.command(self.side, clamp(q, self.side), kp=np.full(N, KP_GRIP), kd=np.full(N, KD_GRIP), ttl_s=1.0)

    def _keep_cmd(self) -> FloatArray:
        """The grip's command while a NOT_HOLDING hand keeps its grip: the squeeze hold of every stopped group (no push
        for a group that closed on nothing / on the hand itself), the last command for the rest."""
        assert self._cmd is not None
        if any(g.q_at is not None for g in self._groups.values()):
            return self._hold_cmd()
        return self._cmd.copy()

    @property
    def temperature_c(self) -> float | None:
        """The hand's hottest sensor at the last read (session 13: r1d's heat words, r1mc3.hands.heat); None before."""
        return float(self._temp) if self._q is not None else None

    @property
    def keeps_grip(self) -> bool:
        """NOT_HOLDING with the hand still gripping (it opens only on `release`)."""
        return self.state == "NOT_HOLDING" and self._kept
