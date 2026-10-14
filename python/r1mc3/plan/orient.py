"""The hand-orientation search for the 5-DoF arm (DESIGN.md §7.2 step 1): ported from v2's orient.py (git r21_try
2a4a165, r1mc2/orient.py, 7 Oct 2026, AUDIT P1 / R1 / R6 / P6), which worked on hardware.

Why (v2's docstring): an R1 arm has 5 joints, so at a fixed hand position 2 DoF stay free - the elbow swinging round
the shoulder-hand line (swivel) and the wrist roll. Pinning ONE finger direction (v2 before 7 Oct: 45 deg down) asks
for poses the arm cannot take at many places; the family of poses at a point has hundreds of clear ones.

How (as v2): grid the shoulder yaw (q3: the swivel) x the wrist roll (q5) inside the limits (the profile's soft
limits intersected with the balance policy's arm envelope); at each node Newton solves shoulder pitch / roll / elbow
so the reference point (pinch) is over (x, y) and the LOWEST fingertip is at z; every solution is scored with the
same rules v2 used, through v3's one geometry engine instead of v2's numpy chain + ArmKin replica:
  - clear: no blocking contact (self, the other arm, the world within the clearance; a goal's allowed touches apart);
  - the touch rule (v2 touch_rule, AUDIT R6): lowered straight down, the fingertips arrive >= TOUCH_GAP before the
    knuckles / palm / forearm (margin = the rest's lowest point - the lowest tip - TOUCH_GAP);
  - the fingers point down at least PITCH_MIN_DEG (v2 GRASP_PITCH_MIN);
  - the preferred heading family (v2 handposes.FAMILIES, deg seen from above, + = toward the hand's own side) is a
    preference, never a refusal; then the touch margin and the joint travel.
Not ported (v2 controller-specific): the cloth-pinch maps, the held-object bottom, the elbow-above-shoulder rule (a
v2 policy constraint: here it is the policy envelope), the per-posture numpy chain (MuJoCo FK is fast enough: ~30 us
per Newton step here).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import numpy.typing as npt

from ..world.geometry import TIPS, Geometry
from . import ik

FloatArray = npt.NDArray[np.float64]

FAMILIES: dict[str, tuple[float, float]] = {
    "front": (-20.0, 20.0),
    "inward": (-60.0, -20.0),
    "outward": (20.0, 60.0),
    "across": (-100.0, -60.0),
}  # v2 handposes.py:23
TOUCH_GAP = 0.003  # m (v2 orient.TOUCH_GAP)
PITCH_MIN_DEG = 5.0  # v2 GRASP_PITCH_MIN
POS_TOL = 0.003
N_YAW = 9
N_ROLL = 9


@dataclass(frozen=True)
class OrientResult:
    ok: bool
    q: FloatArray
    approach: FloatArray
    heading_deg: float = 0.0
    pitch_down_deg: float = 0.0
    angle_name: str = ""
    angle_err_deg: float = 0.0
    touch_margin_mm: float = 0.0
    why: str = ""
    candidates: int = 0
    clear: int = 0
    alternatives: list[dict[str, float | str]] = field(default_factory=list)
    close_yaw_deg: float = 0.0  # the closing line (thumb tip -> the opposing tip(s)) seen from above, base frame
    close_err_deg: float = 0.0  # its distance from the requested grasp_yaw_deg (mod 180; 0 without a request)
    sweeps: int = 1  # session 11: 1 + the sweeps from the fixed seeds after a miss (the search's cost)
    # session 12: the requested approach
    style: str | None = None
    style_err_deg: float = 0.0
    palm_down_deg: float = 0.0
    tilt_deg: float = 0.0
    ranked: list[FloatArray] = field(default_factory=list)  # the next-best clear poses, in rank order (a caller retry)
    nearest_blocked: dict[str, object] = field(default_factory=dict)  # the closest-to-the-style pose that is NOT clear
    tips_out_m: float = 0.0  # session 13: the grip's closing tips outside the object's height band (sum, m)


def heading_deg(side: str, approach: FloatArray) -> float:
    """The fingers' heading seen from above (deg; 0 = forward, + = toward the hand's own side)."""
    g = 1.0 if side == "left" else -1.0
    return float(np.degrees(np.arctan2(g * approach[1], approach[0])))


def angle_err(h: float, prefer: str | None) -> float:
    if prefer is None:
        return 0.0
    lo, hi = FAMILIES[prefer]
    return float(max(lo - h, h - hi, 0.0))


CLOSE_TOL_DEG = 10.0  # the closing line within this of the requested axis costs nothing (v2 orient.py close_across)
CLOSE_WEIGHT = 1.0  # per degree beyond it - as much as a degree outside the preferred family (v2: 0.3, then the box
#                     objects failed in sim for want of it: session 10, the grasp grid's short and tall boxes)


def close_yaw_deg(geo: Geometry, side: str, grip: str | None) -> float:
    """The closing line seen from above (deg, base frame, mod 180 folded to (-90, 90]): the thumb tip -> the index tip
    (pinch / tip / no grip) or -> the index + middle tips' midpoint (wrap): what squeezes the object (v2 grasp_metrics
    close_heading)."""
    t = geo.point(side, "thumb_tip")
    f = geo.point(side, "index_tip")
    if grip == "wrap":
        f = 0.5 * (f + geo.point(side, "middle_tip"))
    v = f - t
    return fold_deg(float(np.degrees(np.arctan2(v[1], v[0]))))


def fold_deg(a: float) -> float:
    """An axis direction (deg) folded to (-90, 90] (a line has no sign)."""
    return float((a + 90.0) % 180.0 - 90.0) if (a + 90.0) % 180.0 != 0.0 else 90.0


def axis_err_deg(a: float, b: float) -> float:
    return abs(float((a - b + 90.0) % 180.0 - 90.0))


def angle_name(h: float) -> str:
    return min(FAMILIES, key=lambda n: max(FAMILIES[n][0] - h, h - FAMILIES[n][1], 0.0))


# ---------------------------------------------------------------------------------- session 12: the requested approach
# HW2 day 1 (12 Oct 2026, the robot hanging at a table, scripted grasps): a tall cylinder grazed on the way down and
# slipped, a tall box was pushed, a flat bar got ~40 of 56-72 mm of its descent; the user: "the approach angle is the
# problem, nothing else"; "it most definitely can turn - the forearm just needs to rotate more and the shoulder can come
# more down" ("from around, not on top"); "the approach is technically a MODEL decision but the controller has to be
# very good about following that model decision". The ranking above never looked at the hand's TILT - the palm facing
# down ("on top") vs facing the object ("around"), the closing line's tilt from level ("the thumb 6-7 cm below the
# fingertips", S11) - and without a request the touch margin (fingertips first when lowered straight down: a palm-down
# preference) and then the travel ("the fastest one", the user on a low object) decided. A requested family was a cost
# of 1 per degree against the closing line's 1 per degree: with a level closing line asked for a round object every
# family came out the same ~-57 deg heading (measured on the HW2 cases: front / inward / outward = one pose).
# A style is GENERIC (D32): `front` / `inward` = the fingers' heading family DELIVERED; `around` = for a TALL object:
# the palm turned to face it (palm_down small), the closing line level, ranked at the GRASP pose with the open hand
# clear of the object (the hand comes down beside it, not on top); `low` = for a FLAT object: the grip's point down at
# the grasp height with the hand clear of the support and the closing line level (both tips reach the sides).
STYLES = ("front", "inward", "around", "low")
STYLE_FAMILY = {"front": "front", "inward": "inward"}
STYLE_WEIGHT = 3.0  # per degree of the style's own error: a requested approach ranks above every other quality term
# v2's real session-14 wraps that HELD (tests/assets/tier_a/grasps/s14_picks.json.gz, the right hand at (20, -10) cm,
# this model's FK on the logged joints): a tall cylinder: palm 14.6 deg down, fingers 6.3 deg down, heading -57.6, the
# closing line -87.2 deg, the thumb 3.5 deg below the fingers; a tall box (held): palm 8.3 / fingers -1.8 (up) / thumb
# 6.4 below; a small round object: palm 33.4 / fingers 14.4 / thumb 7.4 below. 'around' asks for the tall ones' kind.
AROUND_PALM_DOWN_MAX_DEG = 25.0  # the palm at most this far below vertical ("turned"); 90 = palm down, "on top"
AROUND_PALM_TARGET_DEG = 15.0  # 'around' ranks the palm toward this (the real held tall cylinder: 14.6)
QUALITY_WEIGHT = 0.2  # per degree of a style's continuous quality terms (the palm toward its target, the closing line
#                       level, the heading toward its family's centre): among the poses that DELIVER the style the best
#                       one wins, not the nearest to where the arm is ("the fastest one", the user on a low object)
FAMILY_CENTRE = {"front": 0.0, "inward": -40.0}
AROUND_PITCH_UP_MAX_DEG = 15.0  # 'around': the fingers may point up to this much above level (the real tall box: 2-4)
TILT_FREE_DEG = 10.0  # the closing line within this of level costs nothing (the real held wraps: 3.5-7.4 deg)
TILT_WEIGHT = 0.5  # per degree beyond it, when a style is requested (the rolled hand of S11 on a round object: 40+ deg)
ROLLED_FREE_DEG: float | None = 12.0  # session 12 (a round object), OFF by default: a GRASP hover WITHOUT a style -
# the closing line within this of level costs nothing, beyond it TILT_WEIGHT per degree (20.0 measured: a small round
# object in whole simulated episodes 3 / 9 -> 5 / 9 held upright; poses within 20 deg - the robot's held round object /
# short box / low object - unchanged). OFF
# because skill.pick assumes its hover leaves the grasp centre ABOVE its stop height: with the level hand it came out
# 1.65 cm below and the guarded move refused (tests/integration/test_service_l3_pick.py) - the skill first.
# Session 13 (D48): ON at 12 deg - skill.pick now chooses its hover where the grip closes (skills/pick.py
# PICK_MIN_DESCENT_M); HW2 day 2 showed the 14-16 deg default lines were the problem (standing grasps: a short box
# 14.4-14.8 deg, "the index finger comes down first"; a low object 16.3 deg, "keeps hitting the table first", 2 / 3
# slipped while its 4.5 deg grasp held) - 20 left exactly those poses. The session-10 grasp grid (5 objects x 3 slots,
# undeclared hovers): off 11 / 15, 15 deg 12 / 15, **12 deg 13 / 15** (+ both tall cylinders at (20, +-10)), 10 deg
# 12 / 15 (the round object at (20, -10) lost); the robot's day-1 5 / 5 holders (no request) all still hold at 12
# session 13 (HW2 day 2's small round object, standing grasps: default 2 / 4, around 2 / 4; the trace replays: the index
# tip 5-13 mm over the board, the thumb 42-60 mm, the MIDDLE tip 70-80 mm - above the 62 mm object's top: a two-tip grip
# "with the fingertips"): at
# a grasp onto a DECLARED object the grip's closing tips (a wrap: thumb, index, middle; a pinch: thumb, index) should
# all start on the object's side - between FIT_LO_M over its bottom and FIT_HI_M under its top. Out of that band costs
# FIT_WEIGHT per mm (as much as a degree of the closing line); core._orient_grasp also lowers a short wrap's grasp
# height (no client grasp height) until the band fits.
FIT_LO_M = 0.006
FIT_HI_M = 0.006
FIT_WEIGHT = 1.0
CLOSING_TIPS = {"wrap": ("thumb_tip", "index_tip", "middle_tip"), "pinch": ("thumb_tip", "index_tip"),
                "tip": ("thumb_tip", "index_tip")}  # fmt: skip
TARGET_CLEAR_M = 0.004  # 'around' / 'low' at the grasp pose: the OPEN hand keeps this from the target object
# session 13 (HW2 day 2's standing grasps: `around` on a tall box "very close to / grazing the box sometimes, pushed
# down on a corner once" - a real hover ends 5-10 mm off its plan): 'around' keeps the open hand this far from a TALL
# target (taller than AROUND_CLEAR_MIN_H_M: on flat / short ones 10 mm cost holds in the session-12 grid - a flat bar /
# a low object asked `around` 3 / 3 -> 1-2 / 3 - and gained nothing; they keep TARGET_CLEAR_M)
AROUND_CLEAR_M = 0.010
AROUND_CLEAR_MIN_H_M = 0.08


def target_clear_m(style: str | None, height_m: float | None = None) -> float:
    """How far the OPEN hand keeps from the declared target (`height_m` tall) at the grasp pose and on the way for
    `style`."""
    if style == "around" and (height_m is None or height_m > AROUND_CLEAR_MIN_H_M):
        return AROUND_CLEAR_M
    return TARGET_CLEAR_M


RANKED_KEEP = 400  # the clear poses returned in rank order (a caller whose next leg fails tries the next)
N_YAW_STYLE = 13  # the grid for a requested style (the turned poses sit near the wrist roll's limit, -1.87 rad)
N_ROLL_STYLE = 13


@dataclass(frozen=True)
class HandPose:
    """The hand's orientation in words a person can check (deg): heading = the fingers seen from above (0 forward,
    + toward the hand's own side); pitch_down = the fingers below level; palm_down = the palm's normal below level (90 =
    palm down, 0 = palm vertical facing sideways / forward); tilt = the closing line's tilt from level (+ = the thumb
    tip HIGHER than the opposing tip(s)); close_yaw = the closing line seen from above (mod 180)."""

    heading_deg: float
    pitch_down_deg: float
    palm_down_deg: float
    tilt_deg: float
    close_yaw_deg: float
    roll_deg: float = 0.0  # session 13: the hand rolled about the fingers' axis (0 = the index and middle side by side)


def hand_pose(geo: Geometry, side: str, grip: str | None) -> HandPose:
    a, n = geo.hand_axes(side)
    t = geo.point(side, "thumb_tip")
    f = geo.point(side, "index_tip")
    if grip == "wrap":
        f = 0.5 * (f + geo.point(side, "middle_tip"))
    v = f - t
    tilt = float(np.degrees(np.arctan2(t[2] - f[2], max(float(np.hypot(v[0], v[1])), 1e-9))))
    lat = np.cross(a, n)
    return HandPose(
        heading_deg(side, a),
        float(np.degrees(np.arcsin(np.clip(-a[2], -1.0, 1.0)))),
        float(np.degrees(np.arcsin(np.clip(-n[2], -1.0, 1.0)))),
        tilt,
        fold_deg(float(np.degrees(np.arctan2(v[1], v[0])))),
        float(np.degrees(np.arcsin(np.clip(float(lat[2]), -1.0, 1.0)))),
    )


def style_error(style: str | None, hp: HandPose) -> float:
    """How far a pose is from the requested style, in degrees of the style's defining angle (0 = delivered): front /
    inward - the heading outside the family; around - the palm more than AROUND_PALM_DOWN_MAX_DEG below vertical; low -
    the closing line more than TILT_FREE_DEG off level (both tips must reach the object's sides near the support)."""
    if style in STYLE_FAMILY:
        return angle_err(hp.heading_deg, STYLE_FAMILY[style])
    if style == "around":
        return max(hp.palm_down_deg - AROUND_PALM_DOWN_MAX_DEG, 0.0)
    if style == "low":
        return max(abs(hp.tilt_deg) - TILT_FREE_DEG, 0.0)
    return 0.0


def tips_out_m(geo: Geometry, side: str, grip: str | None, band: tuple[float, float], width_m: float | None = None,
               ) -> float:  # fmt: skip
    """How far (m, summed) the grip's closing tips lie outside the height band (z_lo, z_hi) at the current arm pose
    with the hand where its fingers meet an object `width_m` wide (Geometry.closing_hand_q: the open hand's tips are
    not where they close - the simulated round object: the middle tip rose 55 -> 95 mm while the wrap closed)."""
    tips = CLOSING_TIPS.get(grip or "", ())
    if not tips or grip is None:
        return 0.0
    lo, hi = band
    h_save = geo.hands[side].copy()
    geo.set_hand(side, geo.closing_hand_q(side, grip, width_m))
    out = 0.0
    try:
        for k in tips:
            z = float(geo.point(side, k)[2])
            out += max(lo - z, 0.0) + max(z - hi, 0.0)
    finally:
        geo.set_hand(side, h_save)
    return out


def touch_margin(geo: Geometry, side: str) -> tuple[float, float]:
    """(margin m: >= 0 = the fingertips land first; the lowest tip z). The rest = the hand's other geoms + the
    forearm."""
    tips_z = min(float(geo.point(side, k)[2]) for k in TIPS)
    m = geo.m
    tip_bodies = {int(m.site_bodyid[geo.site[side][k]]) for k in TIPS}
    rest = (geo.hand_bodies[side] | {geo.wrist_body[side]}) - tip_bodies
    low_rest = geo.lowest(side, rest)
    return low_rest - tips_z - TOUCH_GAP, tips_z


def _z_of(geo: Geometry, side: str, point: str, z_mode: str) -> tuple[float, str]:
    if z_mode == "point":
        return float(geo.point(side, point)[2]), point
    tz = [float(geo.point(side, k)[2]) for k in TIPS]
    i = int(np.argmin(tz))
    return tz[i], TIPS[i]


def _solve_node(
    geo: Geometry,
    side: str,
    xy: FloatArray,
    z: float,
    q: FloatArray,
    point: str,
    lo: FloatArray,
    hi: FloatArray,
    z_mode: str = "lowest_tip",
) -> tuple[FloatArray, float]:
    """Newton on (q1, q2, q4) with q3, q5 fixed: the point over (x, y), the lowest fingertip (or the point) at z.
    One mj_kinematics + one mj_comPos per iteration; the site Jacobians straight from MuJoCo (session 9: this is the
    planner's hot loop - ~60 % of a hover plan)."""
    m, d = geo.m, geo.d
    free = [0, 1, 3]
    cols = geo.arm_vadr[side][free]
    S = geo.site[side]
    jp, jr = geo._jp, geo._jr
    tip_ids = [S[k] for k in TIPS]
    if point == "pinch":
        ref_ids = [S["thumb_tip"], S["index_tip"]]
    elif point == "fingertips":
        ref_ids = tip_ids
    elif point in S:
        ref_ids = [S[point]]
    else:
        ref_ids = []  # a palm-fixed or body point: the generic path
    err = np.inf
    for it in range(12):
        geo.set_arm(side, q)
        if ref_ids:
            p = sum(d.site_xpos[k] for k in ref_ids) / len(ref_ids)
        else:
            p = geo.point(side, point)
        if z_mode == "point":
            zc, k_low = float(p[2]), -1
        else:
            tz = [float(d.site_xpos[k][2]) for k in tip_ids]
            i_low = int(np.argmin(tz))
            zc, k_low = tz[i_low], tip_ids[i_low]
        e = np.array([xy[0] - p[0], xy[1] - p[1], z - zc])
        prev, err = err, float(np.linalg.norm(e))
        if err < 5e-4 or (it >= 4 and (err > 0.04 or err > 0.9 * prev)):  # converged, or not going to (a far node)
            break
        mujoco.mj_comPos(m, d)
        if ref_ids:
            Jp = np.zeros((3, len(cols)))
            for k in ref_ids:
                mujoco.mj_jacSite(m, d, jp, jr, k)
                Jp += jp[:, cols]
            Jp /= len(ref_ids)
        else:
            Jp = geo.point_jac(side, point)[:, free]
        if k_low >= 0:
            mujoco.mj_jacSite(m, d, jp, jr, k_low)
            Jz = jp[2, cols]
        else:
            Jz = Jp[2]
        Jm = np.vstack([Jp[0], Jp[1], Jz])
        dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(3), e)
        st = float(np.max(np.abs(dq)))
        if st > 0.25:
            dq *= 0.25 / st
        q = q.copy()
        q[free] = np.clip(q[free] + dq, lo[free], hi[free])
    else:
        geo.set_arm(side, q)
        p = geo.point(side, point)
        zc = float(p[2]) if z_mode == "point" else min(float(d.site_xpos[k][2]) for k in tip_ids)
        err = float(np.linalg.norm([xy[0] - p[0], xy[1] - p[1], z - zc]))
    return q, err


SEARCH_AGAIN_APPROACH_DEG = 20.0  # a requested approach missed by more than this also searches from the fixed seeds


def _grid(
    geo: Geometry,
    side: str,
    xy: FloatArray,
    z: float,
    seeds: list[FloatArray],
    yaws: FloatArray,
    rolls: FloatArray,
    point: str,
    lo: FloatArray,
    hi: FloatArray,
    z_mode: str,
    *,
    across_rows: bool = False,
) -> list[FloatArray]:
    """The converged node solutions of one sweep: per shoulder-yaw row the first node tries `seeds` (stops at the
    first that converges), then a continuation along the roll sweep. across_rows: a row whose first node fails also
    starts from the previous row's first solution (the second pass's chains follow ONE branch across the grid)."""
    out: list[FloatArray] = []
    row_start: FloatArray | None = None
    for q3 in yaws:
        warm: FloatArray | None = None  # the previous node's solution (a continuation along the roll sweep)
        first = True
        for q5 in rolls:
            best_node: tuple[FloatArray, float] | None = None
            starts = [warm] if warm is not None else list(seeds)
            if warm is None and across_rows and row_start is not None:
                starts.append(row_start)
            for st in starts:
                q = st.copy()
                q[2], q[4] = q3, q5
                qs, err = _solve_node(geo, side, xy, z, q, point, lo, hi, z_mode)
                if best_node is None or err < best_node[1]:
                    best_node = (qs, err)
                if err < POS_TOL:
                    break
            assert best_node is not None
            qs, err = best_node
            if err >= POS_TOL:
                warm = None
                continue
            warm = qs
            if first:
                row_start, first = qs, False
            out.append(qs)
    return out


def best_orientation(
    geo: Geometry,
    side: str,
    xy: FloatArray,
    z_lowest_tip: float,
    *,
    point: str = "pinch",
    prefer: str | None = None,
    need_touch: bool = False,
    limits: tuple[FloatArray, FloatArray] | None = None,
    allow: tuple[str, ...] = (),
    allow_hand: tuple[str, ...] = (),
    n_yaw: int = N_YAW,
    n_roll: int = N_ROLL,
    z_mode: str = "lowest_tip",
    approach: FloatArray | None = None,
    grasp_yaw_deg: float | None = None,
    grip: str | None = None,
    style: str | None = None,
    hand_clear: dict[str, float] | None = None,
    tips_band: tuple[float, float] | None = None,
    close_width_m: float | None = None,
) -> OrientResult:
    """The best clear arm pose with `point` over xy and the lowest fingertip at z_lowest_tip (v2 best_orientation).
    z_mode "point": the point itself at that z (a reach); `approach`: a preferred finger direction (ranks only);
    grasp_yaw_deg: the axis the fingers should close along, seen from above (base frame, deg, a line: mod 180) - e.g.
    across a box's narrow side (v2 orient.py close_across_deg); ranked with the preferred family.

    Session 11 (episode's finding): the sweep's Newton starts from the arm's CURRENT pose first, so the answer
    depended on where the arm came from - the same grasp hover from an arm over a container missed its closing line by
    21-51 deg (or found the point "out of reach") while from the table_ready arms it was within 10. So when the first
    sweep misses what was asked (no clear pose; the closing line > CLOSE_TOL_DEG off; outside the preferred family; a
    requested approach > SEARCH_AGAIN_APPROACH_DEG off; no pose with the fingertips first when they must be), the grid
    is swept again from each fixed seed (ik.SEEDS_LEFT, mirrored: one branch per chain, continued across the rows) and
    all clear poses are ranked together - the same key, the travel still from the CURRENT pose. Bounded: at most
    1 + len(SEEDS_LEFT) sweeps, the extra ones only on a miss.

    Session 12 (the requested approach, the module's section above): `style` (STYLES) ranks first (STYLE_WEIGHT per
    degree of style_error), with a style the closing line's tilt ranks too (TILT_WEIGHT beyond TILT_FREE_DEG), and for
    'around' neither the touch margin (a palm-down preference) nor PITCH_MIN_DEG applies (the fingers may point up to
    AROUND_PITCH_UP_MAX_DEG); a style uses the finer N_YAW_STYLE x N_ROLL_STYLE grid and, when its best clear pose
    still misses, a local grid around the two best; hand_clear (Geometry.hits): entities the hand must keep a distance
    from (the target object at the grasp pose). The result carries the style's error, the hand's palm / tilt, the next
    clear poses (`ranked`: a caller whose next leg fails tries them) and the closest-to-the-style pose that is not
    clear with what it touches (`nearest_blocked`: the refusal's words)."""
    if style is not None and style not in STYLES:
        raise ValueError(f"style {style!r}: one of {', '.join(STYLES)}")
    lo, hi = limits if limits is not None else ik.arm_limits(side)
    q_now = geo.arm_q(side)
    g = np.ones(5) if side == "left" else ik.MIRROR
    fixed = [np.clip(g * s, lo, hi) for s in ik.SEEDS_LEFT]
    if style is not None:
        n_yaw, n_roll = max(n_yaw, N_YAW_STYLE), max(n_roll, N_ROLL_STYLE)
    yaws = np.linspace(lo[2], hi[2], n_yaw)
    rolls = np.linspace(lo[4], hi[4], n_roll)
    fam = STYLE_FAMILY.get(style or "", prefer)
    blocked: list[tuple[float, dict[str, object]]] = []
    xy_a = np.asarray(xy, dtype=np.float64)
    zf = float(z_lowest_tip)
    cands: list[tuple[tuple[float, ...], dict[str, object]]] = []
    counts = {"solved": 0, "clear": 0, "sweeps": 0}

    def score(sols: list[FloatArray]) -> None:
        counts["sweeps"] += 1
        for qs in sols:
            counts["solved"] += 1
            geo.set_arm(side, qs)
            hit = geo.hits(side, allow=allow, allow_hand=allow_hand, stop_at_first=True, hand_clear=hand_clear)
            if hit:
                if style is not None:  # the refusal's words: the closest-to-the-style pose that is not clear
                    hb = hand_pose(geo, side, grip)
                    blocked.append((style_error(style, hb), {"q": qs, "hp": hb, "hit": f"{hit[0].arm_body} -> "
                                    f"{hit[0].other}"}))  # fmt: skip
                continue
            counts["clear"] += 1
            a, _ = geo.hand_axes(side)
            hp = hand_pose(geo, side, grip)
            h, pitch = hp.heading_deg, hp.pitch_down_deg
            tm, _ = touch_margin(geo, side)
            ferr = angle_err(h, fam)
            serr = style_error(style, hp)
            aerr = ferr
            if style == "around":
                down_short = max(-AROUND_PITCH_UP_MAX_DEG - pitch, 0.0)
            else:
                down_short = max(PITCH_MIN_DEG - pitch, 0.0)
            touch_bad = 1.0 if (need_touch and tm < 0.0) else 0.0
            travel = float(np.linalg.norm(qs - q_now))
            apperr = 0.0
            if approach is not None:  # a requested direction ranks first among the clear poses
                apperr = float(np.degrees(np.arccos(np.clip(np.dot(a, approach) / np.linalg.norm(approach), -1, 1))))
                aerr += apperr
            cy = hp.close_yaw_deg  # = close_yaw_deg(geo, side, grip)
            cerr = 0.0 if grasp_yaw_deg is None else axis_err_deg(cy, float(grasp_yaw_deg))
            # the tilt ranks every REQUESTED style (a hover without a style keeps session 11's ranking exactly: the
            # round object / short box / low object that held 5 / 5 on the robot keep their poses - measured: ranking
            # it for every hover moved the default poses, e.g. the pick skill's grasp centre came out below its stop
            # height)
            tilt_cost = TILT_WEIGHT * max(abs(hp.tilt_deg) - TILT_FREE_DEG, 0.0) if style is not None else 0.0
            if style is None and grip is not None and ROLLED_FREE_DEG is not None:  # a ROLLED grasp hand (session 12)
                tilt_cost = TILT_WEIGHT * max(abs(hp.tilt_deg) - ROLLED_FREE_DEG, 0.0)
            style_cost = (STYLE_WEIGHT - 1.0) * ferr if style in STYLE_FAMILY else STYLE_WEIGHT * serr
            if style is not None:  # the quality among the delivering poses (continuous: start-independent)
                q_terms = abs(hp.tilt_deg)
                if style == "around":
                    q_terms += abs(hp.palm_down_deg - AROUND_PALM_TARGET_DEG)
                elif style in FAMILY_CENTRE:
                    q_terms += 0.5 * abs(h - FAMILY_CENTRE[style])
                style_cost += QUALITY_WEIGHT * q_terms
            fit = 0.0 if tips_band is None else tips_out_m(geo, side, grip, tips_band, close_width_m)
            cost = (
                aerr
                + style_cost
                + 2.0 * down_short
                + CLOSE_WEIGHT * max(cerr - CLOSE_TOL_DEG, 0.0)
                + tilt_cost
                + FIT_WEIGHT * 1000.0 * fit
            )
            tm_key = 0.0 if style == "around" else -min(tm, 0.02)
            key = (touch_bad, cost, tm_key, travel)
            cands.append((key, {"q": qs, "a": a, "h": h, "pitch": pitch, "tm": tm, "aerr": aerr, "cy": cy,
                                "cerr": cerr, "ferr": ferr, "apperr": apperr, "serr": serr, "hp": hp,
                                "fit": fit}))  # fmt: skip

    def missed() -> bool:
        if not cands:
            return True
        k, b = min(cands, key=lambda c: c[0])
        return bool(
            k[0] > 0
            or float(b["cerr"]) > CLOSE_TOL_DEG  # type: ignore[arg-type]
            or float(b["ferr"]) > 0.0  # type: ignore[arg-type]
            or float(b["apperr"]) > SEARCH_AGAIN_APPROACH_DEG  # type: ignore[arg-type]
            or float(b["serr"]) > 0.0  # type: ignore[arg-type]
        )

    score(_grid(geo, side, xy_a, zf, [np.clip(q_now, lo, hi), *fixed[:2]], yaws, rolls, point, lo, hi, z_mode))
    if style is not None or missed():  # a style: always every seed's branch (the choice must not depend on the start)
        for s in fixed:
            score(_grid(geo, side, xy_a, zf, [s], yaws, rolls, point, lo, hi, z_mode, across_rows=True))
    if style is not None and missed():
        # a local grid (half a step each way) around the two best clear poses and the two closest blocked ones: a
        # narrow clear window between the nodes (PLANNER.md "Not solid") is where a turned pose is often found
        dy, dr = (hi[2] - lo[2]) / (n_yaw - 1), (hi[4] - lo[4]) / (n_roll - 1)
        near = [c[1]["q"] for c in sorted(cands, key=lambda c: c[0])[:2]]
        near += [b[1]["q"] for b in sorted(blocked, key=lambda b: b[0])[:2]]
        for qc in near:
            qa = np.asarray(qc, dtype=np.float64)
            ly = np.clip(np.linspace(qa[2] - dy / 2, qa[2] + dy / 2, 5), lo[2], hi[2])
            lr = np.clip(np.linspace(qa[4] - dr / 2, qa[4] + dr / 2, 5), lo[4], hi[4])
            score(_grid(geo, side, xy_a, zf, [qa], ly, lr, point, lo, hi, z_mode))
    n_solved, n_clear = counts["solved"], counts["clear"]
    nb: dict[str, object] = {}
    if blocked:
        be, bd = min(blocked, key=lambda b: b[0])
        bh = bd["hp"]
        assert isinstance(bh, HandPose)
        nb = {"style_err_deg": round(be, 1), "palm_down_deg": round(bh.palm_down_deg, 1),
              "heading_deg": round(bh.heading_deg, 1), "tilt_deg": round(bh.tilt_deg, 1), "touches": bd["hit"],
              "q": np.round(np.asarray(bd["q"]), 3).tolist()}  # fmt: skip
    if not cands:
        sw = int(counts["sweeps"])  # (several sweeps of the grid: the count of solutions can exceed one grid's nodes)
        why = (f"{n_solved} of {n_yaw * n_roll} orientations reach the point"
               + (f" ({sw} sweeps of the grid)" if sw > 1 else "") + ", none is clear of the robot / world"
               if n_solved else "the point is out of the arm's reach at that height")  # fmt: skip
        if nb:
            why += f" (the closest to '{style}' touches {nb['touches']})"
        geo.set_arm(side, q_now)
        return OrientResult(False, q_now, np.zeros(3), why=why, candidates=n_solved, clear=0, style=style,
                            nearest_blocked=nb)  # fmt: skip
    cands.sort(key=lambda c: c[0])
    k0, b = cands[0]
    if need_touch and k0[0] > 0:
        geo.set_arm(side, q_now)
        return OrientResult(False, q_now, np.zeros(3), why="no clear orientation lets the fingertips land first",
                            candidates=n_solved, clear=n_clear)  # fmt: skip
    alts: list[dict[str, float | str]] = []
    for _, c in cands[1:4]:
        h_c, tm_c = float(c["h"]), float(c["tm"])  # type: ignore[arg-type]
        alts.append(
            {"angle_name": angle_name(h_c), "heading_deg": round(h_c, 1), "touch_margin_mm": round(1000 * tm_c, 1)}
        )
    qb = np.asarray(b["q"], dtype=np.float64)
    hpb = b["hp"]
    assert isinstance(hpb, HandPose)
    ranked = [np.asarray(c["q"], dtype=np.float64) for _, c in cands[1:RANKED_KEEP]]
    fit_b = float(b.get("fit", 0.0))  # type: ignore[arg-type]
    geo.set_arm(side, qb)
    return OrientResult(
        True,
        qb,
        np.asarray(b["a"], dtype=np.float64),
        float(b["h"]),  # type: ignore[arg-type]
        float(b["pitch"]),  # type: ignore[arg-type]
        angle_name(float(b["h"])),  # type: ignore[arg-type]
        float(b["aerr"]),  # type: ignore[arg-type]
        1000 * float(b["tm"]),  # type: ignore[arg-type]
        "",
        n_solved,
        n_clear,
        alts,
        float(b["cy"]),  # type: ignore[arg-type]
        float(b["cerr"]),  # type: ignore[arg-type]
        int(counts["sweeps"]),
        style,
        float(b["serr"]),  # type: ignore[arg-type]
        hpb.palm_down_deg,
        hpb.tilt_deg,
        ranked,
        nb,
        fit_b,
    )
