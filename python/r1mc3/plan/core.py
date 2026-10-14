"""The planner pipeline (DESIGN.md §7.2): goal -> target set -> path -> timing -> WHOLE-GOAL validation -> result.

`plan(snapshot, goal, q_now)` is a pure function of a world snapshot, the measured joints and the goal: nothing moves
until the whole goal is planned (AP-3: v2 moved at five places before checking). Multi-leg goals are planned end to
end first: hover = lift + transit + descend, carry = rise + travel + lower; one Segment per leg (each leg is one
executor TRAJECTORY command; a leg can carry its own guards, e.g. the guarded descent's contact guard).

Goal kinds (r1d's L2 arm ops, CONTRACT "Interfaces"):
  move_joint     q[5]                      a joint-space path (straight if clear, else RRT-Connect)
  reach          position[3] (+approach)   IK (or the orientation search) -> a straight hand line if valid, else RRT
  move_linear    position[3] | delta[3]    the straight hand line ONLY (refused when blocked: says by what)
  move_relative  delta[3]                  reach (current point + delta), any path
  hover          position[3] + height_m: the LOWEST FINGERTIP (the three tip sites = the RT core's arm model
                 lowest_tip_z; v2's datum) height_m above the DATUM: position[2] when > 0 (the support top the client
                 names - compat sends the table top as v2 did), else the highest declared top under (x, y) - a surface,
                 a container's floor, an object - apart from the goal's target (allow_hand). predicted.lowest_tip_z_m
                 = that fingertip; predicted.landing_z_m = the lowest COLLISION point of the hand (a sideways thumb /
                 palm / curled finger can hang below the tips: api's 3.6 vs 6 cm reading was this)
  lower          max_depth_m (+ until)     a vertical descent of the hand point with the contact guard (the
                                           executor stops at the first pad contact); the surface below may be
                                           touched by the hand only (allow_hand)
  carry          position[3] + lift_m      rise lift_m, travel above the target, lower to the target point
Frames: `base` only (r1mc3.world.FRAMES). Units: m, rad, s.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from r1sim import joints as J

from ..world.entities import FRAMES, WorldSnapshot
from ..world.geometry import ARM_JOINTS, TIPS, Geometry, Held
from . import ik
from . import orient as O
from . import path as P
from .timing import Timed, time_path

FloatArray = npt.NDArray[np.float64]

KINDS = ("move_joint", "reach", "move_linear", "move_relative", "hover", "lower", "carry")
TRANSIT_CLEAR_M = 0.06  # hover / carry: the transit height clears what is under the path by this (v2: 6 cm)
LIFT_DEFAULT_M = 0.05
LOWER_SPEED = 0.25  # the guarded descent at a quarter of the planner's speed
START_TOL_RAD = 0.05  # the executor: the first knot within 0.05 rad of the current target (EXECUTOR.md)
MAX_KNOTS = 512
GUARD_LAG, GUARD_TORQUE, GUARD_CONTACT, GUARD_SURFACE = 1, 2, 4, 8  # r1rt_traj_guards bits (EXECUTOR.md)
_ids = itertools.count(1)


@dataclass(frozen=True)
class PlanGoal:
    kind: str
    arm: str
    q: tuple[float, ...] | None = None
    position: tuple[float, float, float] | None = None
    delta: tuple[float, float, float] | None = None
    frame: str = "base"
    approach: tuple[float, float, float] | None = None
    normal: tuple[float, float, float] | None = None
    point: str = "pinch"
    grip: str | None = None  # wrap | pinch | tip: the reference point = where that grip's fingers close (HG-2)
    width_m: float | None = None  # the object's width across the grip (the closing point depends on it)
    orientation: tuple[float, float, float, float] | None = None  # a palm quaternion (w, x, y, z) -> approach + normal
    height_m: float | None = None
    lift_m: float | None = None
    max_depth_m: float = 0.10
    until: str = "contact"
    speed: float = 1.0
    prefer: str | None = None  # orientation family (orient.FAMILIES) when no approach is given
    allow: tuple[str, ...] = ()  # entity ids the whole arm may touch
    allow_hand: tuple[str, ...] = ()  # entity ids only the hand may touch (the grasp target)
    seed: int = 0
    grasp_yaw_deg: float | None = None  # hover: the axis the fingers close along, seen from above (base, deg, mod 180)
    held: dict[str, Any] | None = None  # an object in the goal arm's hand: {shape, size} at its grasp centre
    hand: str | None = None  # plan with the goal arm's hand in this preset (r1sim.dex3: "fist" ...) - the caller
    #                          puts the hand there BEFORE the motion (skill.table_ready closes the fingers first, v2)
    # session 12 (HW2 day 1: "the approach angle is the problem"; orient.py's section): the requested approach
    approach_style: str | None = None  # hover: front | inward | around | low (orient.STYLES); None = as before
    approach_tol_deg: float | None = None  # None = the NEAREST deliverable pose, its error reported; a number = STRICT
    grasp_z_m: float | None = None  # hover: where the grip's point will close (base z): the pose is chosen THERE
    # session 13 (D48 / skill.pick): a grasp-pose hover ends at least this far above its grasp pose (the descent the
    # client's next straight-down step needs: skill.pick's stop height was refused when the hover left the grasp
    # centre AT or BELOW it - "0 < the depth below the grasp centre")
    min_descent_m: float | None = None
    end_q: tuple[float, ...] | None = None  # lower: end EXACTLY at this pose when the line's end is within END_Q_TOL_M
    #                                         of its point (r1d passes the last hover's grasp pose: the 5-DoF arm's
    #                                         straight descent cannot hold the palm's roll, the hover's line up did)
    end_point: str | None = None  # lower: the point the hover's line held plumb (its grip's point) - the retrace's

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PlanGoal:
        d = dict(d)
        for k in ("q", "position", "delta", "approach", "normal", "orientation", "allow", "allow_hand", "end_q"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        known = set(PlanGoal.__dataclass_fields__)
        return PlanGoal(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class Segment:
    """One executor TRAJECTORY command for one arm: knots (t, q, dq, ddq) + the guard bits (r1rt_traj_guards)."""

    arm: str
    t: FloatArray
    q: FloatArray  # (K, 5)
    dq: FloatArray
    ddq: FloatArray
    guards: int = GUARD_LAG
    leg: str = ""

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])

    def to_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "leg": self.leg, "guards": self.guards, "t": self.t.tolist(), "q": self.q.tolist(),
                "dq": self.dq.tolist(), "ddq": self.ddq.tolist()}  # fmt: skip


@dataclass(frozen=True)
class PlanResult:
    ok: bool
    code: str  # service.codes.Code name: OK / INVALID_ARGS / UNKNOWN_FRAME / UNREACHABLE / COLLISION / LIMIT
    plan_id: str = ""
    segments: tuple[Segment, ...] = ()
    predicted: dict[str, Any] = field(default_factory=dict)
    clearance_m: float | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    planning_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return float(sum(s.duration_s for s in self.segments))

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "code": self.code, "plan_id": self.plan_id, "reason": self.reason,
                "segments": [s.to_dict() for s in self.segments], "predicted": self.predicted,
                "clearance_m": self.clearance_m, "details": self.details, "planning_s": self.planning_s,
                "duration_s": self.duration_s}  # fmt: skip


class _Refuse(Exception):
    def __init__(self, code: str, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.code, self.reason, self.details = code, reason, details


_GEO: dict[tuple[object, ...], Geometry] = {}


def geometry_for(snapshot: WorldSnapshot, clearance_m: float | None = None, held: tuple[Held, ...] = ()) -> Geometry:
    """The compiled geometry for a snapshot's entities (cached per geometry key: re-declaring a table is free);
    clearance_m: a geometry with a wider margin than the default 1 cm (the out-and-around way's 2 cm); held: objects
    in the hands (their shapes on the palms)."""
    k = (snapshot.key(), clearance_m, held)
    g = _GEO.get(k)
    if g is None:
        if len(_GEO) > 8:
            _GEO.clear()
        kw: dict[str, Any] = {"held": held} if held else {}
        if clearance_m is not None:
            kw["clearance_m"] = clearance_m
        g = _GEO[k] = Geometry(snapshot, **kw)
    return g


HELD_SHAPES = {"sphere": 1, "cylinder": 2, "box": 3}


def held_in_palm(geo: Geometry, side: str, held: dict[str, Any]) -> Held:
    """A held object (r1d / the client: {shape, size (MuJoCo half-sizes), at: "grasp_centre"}) centred where the hand
    closes (the RT core's grasp centre at the measured finger pose), in the palm body's frame (upright in the palm)."""
    shape = str(held.get("shape", "sphere"))
    size = tuple(float(v) for v in held.get("size", ()))
    if shape not in HELD_SHAPES or len(size) != HELD_SHAPES[shape] or not all(0.0 < v <= 0.3 for v in size):
        raise _Refuse("INVALID_ARGS", "held: {shape: sphere | cylinder | box, size: half-sizes (m) - 1 / 2 / 3 values}")
    b = geo.palm_body[side]
    R = geo.d.xmat[b].reshape(3, 3)
    local = R.T @ (geo.point(side, "grasp_centre") - geo.d.xpos[b])
    pos = (round(float(local[0]), 4), round(float(local[1]), 4), round(float(local[2]), 4))
    return Held(side, shape, tuple(round(v, 4) for v in size), pos)


def _vec3(v: tuple[float, ...] | None, name: str) -> FloatArray:
    if v is None or len(v) != 3 or not all(math.isfinite(x) for x in v):
        raise _Refuse("INVALID_ARGS", f"{name}: 3 finite numbers (m, base frame)")
    return np.asarray(v, dtype=np.float64)


def _timed_segment(arm: str, path: FloatArray, speed: float, guards: int, leg: str) -> Segment:
    tm: Timed = time_path(P.densify(path), speed=speed)
    if len(tm.t) > MAX_KNOTS:
        raise _Refuse("LIMIT", f"{leg}: {len(tm.t)} knots > the executor's {MAX_KNOTS}")
    return Segment(arm, tm.t, tm.q, tm.dq, tm.ddq, guards, leg)


def _resolved(goal: PlanGoal) -> PlanGoal:
    """The goal with its reference point from `grip` and its approach / normal from `orientation`."""
    upd: dict[str, Any] = {}
    if goal.grip is not None:
        if goal.grip not in ("wrap", "pinch", "tip"):
            raise _Refuse("INVALID_ARGS", f"grip {goal.grip!r}: wrap | pinch | tip")
        if goal.width_m is not None and not (0.0 < goal.width_m <= 0.2):
            raise _Refuse("INVALID_ARGS", "width_m: (0, 0.2] m")
        # session 10 (grasp_TO_PLANNER_4.md): without the object's width the wrap's "closing point" is the FULLY closed
        # fist's - 3.6 cm back along the fingers and 3 cm into the palm from the open pinch point: in the physical-hands
        # rig every 6-8 cm wrap aimed there closed too low / too deep (0 / 9 vs 3 / 9 at the default point). A wrap
        # without a width keeps the default reference point; with a width, HG-2's point where the tips are that far
        # apart (planner-level only: r1d's arm.hover has no width - not validated in the rig yet)
        if not (goal.grip == "wrap" and goal.width_m is None):
            upd["point"] = f"grasp:{goal.grip}" + ("" if goal.width_m is None else f"@{goal.width_m:.4f}")
    if goal.approach_style is not None and goal.approach_style not in O.STYLES:
        raise _Refuse("INVALID_ARGS", f"approach_style {goal.approach_style!r}: {' | '.join(O.STYLES)}")
    if goal.approach_tol_deg is not None and not (0.0 <= goal.approach_tol_deg <= 90.0):
        raise _Refuse("INVALID_ARGS", "approach_tol_deg: 0-90 deg")
    if goal.orientation is not None:
        q = goal.orientation
        if len(q) != 4 or not all(math.isfinite(v) for v in q) or abs(math.sqrt(sum(v * v for v in q)) - 1) > 0.01:
            raise _Refuse("INVALID_ARGS", "orientation: a unit quaternion (w, x, y, z)")
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64))
        Rm = R.reshape(3, 3)
        g = -1.0 if goal.arm == "left" else 1.0
        upd["approach"] = tuple(float(v) for v in Rm[:, 0])
        upd["normal"] = tuple(float(v) for v in g * Rm[:, 1])
    if not upd:
        return goal
    return replace(goal, **upd)


class _Ctx:
    def __init__(
        self, geo: Geometry, goal: PlanGoal, q_now: FloatArray, envelope: tuple[FloatArray, FloatArray] | None
    ):
        self.geo, self.goal, self.side = geo, goal, goal.arm
        soft = ik.arm_limits(goal.arm)
        q0 = np.asarray(q_now, dtype=np.float64)[ARM_JOINTS[goal.arm]].copy()
        q0c = np.clip(q0, soft[0], soft[1])  # a start AT a soft limit moves 2 mm-rad inside (the first-knot rule)
        if float(np.max(np.abs(q0c - q0))) > START_TOL_RAD:
            raise _Refuse("LIMIT", "the arm's current target is outside the joint limits", q=q0.round(4).tolist())
        lo, hi = ik.arm_limits(goal.arm, envelope)
        # the envelope never excludes where the arm already is (v2 controller.py:2072: min / max with q_cmd)
        self.limits = (np.minimum(lo, q0c), np.maximum(hi, q0c))
        self.chk = P.Checker(geo, goal.arm, self.limits, allow=goal.allow, allow_hand=goal.allow_hand)
        self.q0 = q0c
        self.legs: list[tuple[str, FloatArray, int, float]] = []  # (leg, path, guards, speed)
        self.first_allow: tuple[str, ...] = ()
        self.methods: list[str] = []
        self.escape: P.Pairs = {}  # start_touching: the pairs (arm body, entity) the start is already inside
        self.escaped_from: list[str] = []
        self.snapshot: WorldSnapshot | None = None  # plan(): for the out-and-around's wider-margin geometry
        self.stance: tuple[FloatArray, FloatArray | None, dict[str, FloatArray] | None] | None = None
        self.info: dict[str, Any] = {}  # extra result details (the out-and-around's keyframes, the escape)

    def start_touching(self) -> None:
        """A hand that starts ON an entity (resting on the table, holding at a wall) may touch it during the FIRST
        leg only (it has to leave it). Another part of the arm that starts inside an entity's clearance (session 10:
        the resting elbow at a table's edge) may LEAVE it under path.escape_step's rule (`self.escape`: those pairs;
        the first leg leaves them, never coming closer). A start touching the body, the other arm or the floor
        refuses."""
        self.geo.set_arm(self.side, self.q0)
        hits = self.geo.hits(self.side, allow=self.goal.allow, allow_hand=self.goal.allow_hand)
        hand = {self.geo.m.body(b).name for b in self.geo.hand_bodies[self.side]}
        ents = {h.other.split(":", 1)[1] for h in hits if h.other.startswith("entity:") and h.arm_body in hand}
        others = [h for h in hits if not h.other.startswith("entity:")]
        if others:
            h = others[0]
            raise _Refuse("COLLISION", f"at the start the arm already touches: {h.arm_body} -> {h.other}", hit=h.other)
        self.first_allow = tuple(sorted(ents))
        self.chk.allow_hand = tuple(self.goal.allow_hand) + self.first_allow
        pairs, bad = P.start_pairs(self.chk, self.q0)
        if bad:  # (cannot happen after the check above: kept for a changed hits() contract)
            raise _Refuse("COLLISION", f"at the start the arm already touches: {bad}", hit=bad)
        self.escape = pairs
        self.escaped_from = sorted({f"{b} -> {o} ({1000 * d:.0f} mm)" for (b, o), d in pairs.items()})

    def leave_start(self, leg: str = "leave", speed: float = 1.0) -> FloatArray:
        """The generic way out of a start inside the clearance: steepest ascent of the smallest pair distance until
        clear (path.escape_ascent) as the first leg; the goal is then planned from there with the normal checks."""
        r = P.escape_ascent(self.chk, self.q0, self.escape)
        if not r.ok:
            raise _Refuse("COLLISION", f"the arm starts inside the clearance of {', '.join(self.escaped_from)} and "
                          f"cannot leave it without coming closer: {r.reason}", hit=r.blocked_by)  # fmt: skip
        self.legs.append((leg, r.path, GUARD_LAG, speed))
        self.methods.append("escape")
        self.escape = {}
        self._leg_done()
        self.q0 = np.asarray(r.path[-1])
        return self.q0

    def _leg_done(self) -> None:
        self.chk.allow_hand = tuple(self.goal.allow_hand)

    def point(self, q: FloatArray, name: str | None = None) -> FloatArray:
        self.geo.set_arm(self.side, q)
        return self.geo.point(self.side, name or self.goal.point)

    def solve(self, target: FloatArray, q_init: FloatArray, approach: FloatArray | None) -> FloatArray:
        nrm = None if self.goal.normal is None else np.asarray(self.goal.normal)
        r = ik.solve(self.geo, self.side, target, q_init=q_init, point=self.goal.point, approach=approach,
                     normal=nrm, limits=self.limits)  # fmt: skip
        if not r.ok:
            raise _Refuse("UNREACHABLE", f"the {self.side} hand cannot reach {np.round(target, 3).tolist()} "
                          f"({1000 * r.pos_err_m:.0f} mm short)", error_mm=round(1000 * r.pos_err_m, 1))  # fmt: skip
        if not self.chk.valid(r.q):
            hit = self.chk.last_hit
            # 5 joints, 3 position constraints: the swivel x roll family may hold a clear pose (the orientation search)
            o = O.best_orientation(self.geo, self.side, target[:2], float(target[2]), point=self.goal.point,
                                   prefer=self.goal.prefer, limits=self.limits, allow=self.goal.allow,
                                   allow_hand=self.chk.allow_hand, z_mode="point", approach=approach)  # fmt: skip
            if not o.ok:
                raise _Refuse("COLLISION", f"at the target the arm would touch: {hit} (and no other hand orientation "
                              f"there is clear: {o.why})", hit=hit)  # fmt: skip
            return o.q
        return r.q

    def go_joint(self, leg: str, qa: FloatArray, qb: FloatArray, speed: float = 1.0) -> FloatArray:
        r = P.rrt_connect(self.chk, qa, qb, seed=self.goal.seed)
        if not r.ok:
            raise _Refuse("COLLISION" if r.blocked_by else "UNREACHABLE", f"{leg}: {r.reason}", hit=r.blocked_by)
        self.legs.append((leg, r.path, GUARD_LAG, speed))
        self.methods.append(r.method)
        self._leg_done()
        return qb

    def go_line(self, leg: str, qa: FloatArray, p1: FloatArray, *, strict: bool, speed: float = 1.0,
                guards: int = GUARD_LAG, approach: FloatArray | None = None,
                target_q: FloatArray | None = None) -> FloatArray:  # fmt: skip
        r = P.cartesian_line(self.chk, qa, p1, point=self.goal.point, approach=approach)
        if r.ok:
            path = r.path
            if target_q is not None and float(np.max(np.abs(path[-1] - target_q))) < 0.1 and len(path) >= 2:
                # end EXACTLY at the planned pose (the line's IK keeps the point, not the orientation the target
                # set chose: the fingertips' height drifted 1-4 mm, api's hover datum check)
                if self.chk.segment(path[-2], target_q):
                    path = np.vstack([path[:-1], target_q])
            tol = 0.001 if guards & GUARD_CONTACT else 0.003  # a guarded descent stays straighter
            self.legs.append((leg, P.decimate(self.chk, path, point=self.goal.point, tol_m=tol), guards, speed))
            self.methods.append("line")
            self._leg_done()
            # the leg's STORED end (target_q when it replaced the line's last pose): the next leg starts where this
            # one really ends (grasp's finding, session 10: returning the line IK's own end made the descend start
            # 0.09 rad of wrist roll away from the executed transit -> the executor refused it, LEG_REFUSED)
            return np.asarray(path[-1])
        if strict:
            code = "COLLISION" if r.blocked_by else "UNREACHABLE"
            raise _Refuse(code, f"{leg}: the straight line is not possible ({r.reason}"
                          f"{': ' + r.blocked_by if r.blocked_by else ''})", hit=r.blocked_by)  # fmt: skip
        qb = target_q if target_q is not None else self.solve(p1, qa, approach)
        return self.go_joint(leg, qa, qb, speed)


def _surface_top(geo: Geometry, xy: FloatArray, ignore: tuple[str, ...] = ()) -> tuple[str, float]:
    tops = [t for t in geo.surfaces_under(xy) if t[0] not in ignore]
    return tops[0] if tops else ("floor", 0.0)


def _transit_z(geo: Geometry, a: FloatArray, b: FloatArray, extra: float) -> float:
    """The highest top under the straight line a -> b (sampled every 2 cm) + extra."""
    n = max(2, int(np.ceil(np.linalg.norm(b[:2] - a[:2]) / 0.02)))
    top = 0.0
    for k in range(n + 1):
        xy = a[:2] + (b[:2] - a[:2]) * (k / n)
        top = max(top, _surface_top(geo, xy)[1])
    return top + extra


def _orient_info(c: _Ctx, r: O.OrientResult, a: FloatArray | None) -> None:
    g = c.goal
    c.info["orientation"] = {"angle_used": r.angle_name, "heading_deg": round(r.heading_deg, 1),
                             "angle_err_deg": round(r.angle_err_deg, 1), "pitch_down_deg": round(r.pitch_down_deg, 1),
                             "close_yaw_deg": round(r.close_yaw_deg, 1), "close_err_deg": round(r.close_err_deg, 1),
                             "touch_margin_mm": round(r.touch_margin_mm, 1),
                             # session 12: the hand's tilt, in every hover's result
                             "palm_down_deg": round(r.palm_down_deg, 1), "tilt_deg": round(r.tilt_deg, 1),
                             "approach_style": g.approach_style}  # fmt: skip
    vec_err = None
    if a is not None:
        cosv = float(np.dot(r.approach, a) / max(float(np.linalg.norm(a)), 1e-9))
        vec_err = round(float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))), 1)
    if g.approach_style is not None:  # approach_err_deg = the requested style's own error (orient.style_error)
        c.info["orientation"]["approach_err_deg"] = round(r.style_err_deg, 1)
        if vec_err is not None:
            c.info["orientation"]["approach_vec_err_deg"] = vec_err
        c.info["orientation"]["approach_delivered"] = bool(
            r.style_err_deg <= (g.approach_tol_deg if g.approach_tol_deg is not None else 0.0) + 1e-9
        )
    elif vec_err is not None:  # session 10's meaning: the angle to the requested finger direction
        c.info["orientation"]["approach_err_deg"] = vec_err


STYLE_WORDS = {
    "front": "the fingers pointing away from the robot (heading -20..20 deg)",
    "inward": "the fingers pointing toward the robot's centre line (heading -60..-20 deg)",
    "around": f"the palm turned to face the object (at most {O.AROUND_PALM_DOWN_MAX_DEG:.0f} deg below vertical)",
    "low": f"the closing line level within {O.TILT_FREE_DEG:.0f} deg at the grasp height",
}


def _check_style(c: _Ctx, r: O.OrientResult) -> None:
    """A STRICT request (approach_tol_deg given) refused beyond its tolerance - in words: what the style needs, how
    far the nearest clear pose is, what the closest turned pose would touch."""
    g = c.goal
    if g.approach_style is None or g.approach_tol_deg is None or r.style_err_deg <= g.approach_tol_deg + 1e-9:
        return
    nb = r.nearest_blocked
    more = ""
    if nb and float(nb.get("style_err_deg", 1e9)) < r.style_err_deg:  # type: ignore[arg-type]
        more = (
            f"; a pose {nb['style_err_deg']} deg off (palm {nb['palm_down_deg']} deg down, heading "
            f"{nb['heading_deg']}) exists but would touch {nb['touches']}"
        )
    raise _Refuse("UNREACHABLE", f"the '{g.approach_style}' approach ({STYLE_WORDS[g.approach_style]}) is not possible "
                  f"here within {g.approach_tol_deg:g} deg: the nearest clear pose is {r.style_err_deg:.1f} deg off "
                  f"(palm {r.palm_down_deg:.1f} deg down, heading {r.heading_deg:.1f}, fingers {r.pitch_down_deg:.1f} "
                  f"deg down, the closing line {r.tilt_deg:.1f} deg from level){more}",
                  **c.info.get("orientation", {}), nearest=nb)  # fmt: skip


def _orient_target(c: _Ctx, xy: FloatArray, z_lowest_tip: float) -> tuple[FloatArray, FloatArray]:
    """The arm pose at (x, y) with the lowest fingertip at z (the orientation search: the preferred family, a requested
    approach and closing axis rank the clear poses), -> (q, approach). The choice goes into the result (c.info)."""
    g = c.goal
    a = None if g.approach is None else _vec3(g.approach, "approach")
    r = O.best_orientation(c.geo, c.side, xy, z_lowest_tip, point=g.point, prefer=g.prefer, limits=c.limits,
                           allow=g.allow, allow_hand=g.allow_hand, approach=a, grasp_yaw_deg=g.grasp_yaw_deg,
                           grip=g.grip, style=g.approach_style)  # fmt: skip
    if not r.ok:
        raise _Refuse("UNREACHABLE", f"no clear hand orientation at {np.round(xy, 3).tolist()}: {r.why}", why=r.why,
                      nearest=r.nearest_blocked)  # fmt: skip
    _orient_info(c, r, a)
    _check_style(c, r)
    return r.q, r.approach


# ------------------------------------------------------------------------------- session 12: the grasp pose first
GRASP_SAME_RAD = 0.05  # ranked grasp poses this close (every joint) to one whose way up failed are skipped
GRASP_LINE_BUDGET_S = 0.6  # the clear grasp poses tried in rank order (the nearest first) until one has a straight
#                             way up to the hover - within this much planning time (the rest: refused, saying why)
END_Q_TOL_M = 0.015  # a lower ends at the hover's grasp pose when the line's end is this close to its point (z) ...
# (session 12, a round object: was 3 mm - an executed hover ends 5-12 mm from its plan (lockstep hand-over re-picks:
# 10.3 / 12.3 mm; r1d's hover error_mm median 5.9 on the robot's stack) and the lower then fell back to the plain
# descent)
END_Q_XY_M = 0.02  # ... and this close sideways (the lower's pinch point is not the grip's: it drifts on the line)
LOWER_POINT = "pinch"  # arm.lower's reference point (r1d's lower goal names no grip: PlanGoal's default)


@dataclass(frozen=True)
class _Grasp:
    z: float  # where the grip's reference point closes (base z)
    target: str | None  # the declared Object the hand must keep TARGET_CLEAR_M from at the grasp pose
    width_m: float | None


def _target_object(c: _Ctx) -> Any:
    if c.snapshot is None:
        return None
    ids = set(c.goal.allow_hand)
    for e in c.snapshot.entities:
        if e.id in ids and getattr(e, "kind", "") == "object":
            return e
    return None


def _width_across(e: Any, yaw_deg: float | None) -> float:
    """The object's width across the closing line (a cylinder / sphere: its diameter; a box: its extent along the
    closing line seen from above - the narrow side when no line is asked)."""
    if e.shape in ("cylinder", "sphere"):
        return 2.0 * float(e.size[0])
    hx, hy = float(e.size[0]), float(e.size[1])
    if yaw_deg is None:
        return 2.0 * min(hx, hy)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(e.pose.quat, dtype=np.float64))
    ex, ey = R.reshape(3, 3)[:2, 0], R.reshape(3, 3)[:2, 1]
    u = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))])
    return 2.0 * (hx * abs(float(np.dot(ex, u))) + hy * abs(float(np.dot(ey, u))))


def _grasp_of(c: _Ctx) -> _Grasp | None:
    """A hover whose pose is chosen at the GRASP height: `grasp_z_m` given, or a style with the goal's target
    declared as an Object (its middle = its centre: the r1d recipe's datum, half its height)."""
    g = c.goal
    e = _target_object(c)
    # session 13: a GRASP hover (a grip) onto a declared target object chooses its pose where the grip closes too, with
    # or without a style (before: only a style did) - the caller that declares what it grasps (a task client / a
    # scripted grasp tool given the object's size) gets the same checked grasp pose + the lower that retraces it as a
    # styled one
    if g.grasp_z_m is None and not (e is not None and (g.approach_style is not None or g.grip is not None)):
        return None
    z = float(g.grasp_z_m) if g.grasp_z_m is not None else float(e.pose.pos[2])
    w = g.width_m
    if w is None and e is not None and g.grip is not None:
        w = round(min(max(_width_across(e, g.grasp_yaw_deg), 0.005), 0.2), 4)
        tb = None if e is None else _band_of(e)
        if g.grip == "wrap" and tb is not None and tb[2] <= SHORT_WRAP_MAX_H_M:
            # session 13 (a small round object): a SHORT object is wrapped DEEP - the grip's closing point taken where
            # the tips are WRAP_DEEP_FRAC of its width apart, so the hand comes that much further over it and the
            # fingers wrap round it instead of closing on it with their tips (the user on a standing `around` grasp:
            # "grab with the fingertips rather than from the palm, which lets the object slip out")
            w = round(w * WRAP_DEEP_FRAC, 4)
    return _Grasp(z, None if e is None else str(e.id), w)


FIT_Z_FRACS = (0.42, 0.35, 0.28)  # session 13: a short wrap's grasp heights tried (of the object's height, over its
#                                   bottom) when its closing tips do not fit the object at its middle (orient.FIT_*)
# session 13 (a small round object, the simulated twin of day 2's measured layout, 6 jittered seeds: `around` 1 / 3
# held on the base, 4 / 6 with the tips' fit, 5 / 6 at 0.8, 6 / 6 at 0.6; on the robot: 2 / 4 - "with the fingertips"):
# a wrap onto an object no taller than SHORT_WRAP_MAX_H_M closes at the point where the tips are WRAP_DEEP_FRAC of its
# width apart. Taller objects (the robot's 8 / 8 tall cylinder + tall box `around`) keep their grasp point.
SHORT_WRAP_MAX_H_M = 0.08
WRAP_DEEP_FRAC = 0.6
LOW_TIP_MARGIN_M = 0.006  # session 13: 'low' keeps the closing tips this far over the object's bottom (its support)


def _band_of(e: Any) -> tuple[float, float, float]:
    """A declared object's (bottom z, top z, height)."""
    hz = float(e.size[0]) if e.shape == "sphere" else float(e.size[1]) if e.shape == "cylinder" else float(e.size[2])
    zc = float(e.pose.pos[2])
    return zc - hz, zc + hz, 2.0 * hz


def _target_band(c: _Ctx) -> tuple[float, float, float] | None:
    """The declared target object's (bottom z, top z, height) - None without one."""
    e = _target_object(c)
    return None if e is None else _band_of(e)


def _orient_grasp(c: _Ctx, xy: FloatArray, tip_z_hover: float, gr: _Grasp) -> tuple[FloatArray, FloatArray]:
    """Choose the hand's orientation where the grip will CLOSE (the reference point at gr.z over xy, the open hand
    TARGET_CLEAR_M clear of the target object, the style ranked there), then the hover = that pose straight above
    it with the lowest fingertip at tip_z_hover: the client's descent (`descent_m` in the result) comes down beside /
    around the object instead of onto it. HW2 day 1: the hover chose the pose at the HOVER height and the descent
    found what it found (a tall cylinder: the palm on its top 36 mm down; a tall box: the knuckle 18 mm down; a flat
    bar: 8 of 52 mm allowed - the planner's own numbers on the robot's layout)."""
    g = c.goal
    if gr.width_m is not None and g.grip is not None:
        c.goal = g = replace(g, width_m=gr.width_m, point=f"grasp:{g.grip}@{gr.width_m:.4f}")
    a = None if g.approach is None else _vec3(g.approach, "approach")
    tb0 = _target_band(c)
    clear_m = O.target_clear_m(g.approach_style, None if tb0 is None else tb0[2])
    hc = None if gr.target is None else {gr.target: clear_m}
    # session 13 (a small round object): a WRAP onto a declared object ranks its closing tips onto the object's side
    # (orient.FIT_*); 'low' (HW2 day 2: the fingertips went below a flat bar's bottom with the NOMINAL table) keeps them
    # over its bottom
    tb = _target_band(c) if g.grip == "wrap" or g.approach_style == "low" else None
    band: tuple[float, float] | None = None
    if tb is not None:
        band = (tb[0] + O.FIT_LO_M, tb[1] - O.FIT_HI_M) if g.grip == "wrap" else (tb[0] + LOW_TIP_MARGIN_M, math.inf)

    def search(z: float) -> O.OrientResult:
        return O.best_orientation(c.geo, c.side, xy, z, point=g.point, prefer=g.prefer, limits=c.limits,
                                  allow=g.allow, allow_hand=g.allow_hand, approach=a, grasp_yaw_deg=g.grasp_yaw_deg,
                                  grip=g.grip, z_mode="point", style=g.approach_style,
                                  need_touch=g.approach_style == "low", hand_clear=hc, tips_band=band,
                                  close_width_m=gr.width_m)  # fmt: skip

    r = search(gr.z)
    z_from = "client" if g.grasp_z_m is not None else "middle"
    lower_ok = r.ok and tb is not None and band is not None and g.grip == "wrap" and g.grasp_z_m is None
    if lower_ok and tb is not None and r.tips_out_m > 1e-4:
        # no grasp height from the client and the tips do not all fit at the middle: lower it (a short object's
        # middle finger above its top - day 2's round object); the first height whose best pose fits (or the best fit)
        # wins
        for frac in FIT_Z_FRACS:
            z2 = tb[0] + frac * tb[2]
            if z2 >= gr.z - 1e-4:
                continue
            r2 = search(z2)
            if r2.ok and r2.tips_out_m < r.tips_out_m - 1e-4:
                r, gr, z_from = r2, replace(gr, z=z2), f"lowered to fit ({frac:.2f} of the height)"
            if r.tips_out_m <= 1e-4:
                break
        c.geo.set_arm(c.side, r.q)
    if not r.ok:
        raise _Refuse("UNREACHABLE", f"no clear hand orientation to close at {np.round([*xy, gr.z], 3).tolist()}"
                      f"{' around ' + gr.target if gr.target else ''}: {r.why}", why=r.why,
                      nearest=r.nearest_blocked)  # fmt: skip
    _orient_info(c, r, a)
    _check_style(c, r)
    why = ""
    # the way up (= the client's way down, arm.lower retraces it) keeps the open hand clear of the target too
    chk_up = P.Checker(c.geo, c.side, c.limits, allow=g.allow, allow_hand=c.chk.allow_hand, hand_clear=hc)
    tol = g.approach_tol_deg
    t_end = time.perf_counter() + GRASP_LINE_BUDGET_S
    tried = 0
    failed: list[FloatArray] = []
    for k, qg in enumerate([r.q, *r.ranked]):
        if k > 0 and time.perf_counter() > t_end:
            why += f"; {tried} grasp poses tried in {GRASP_LINE_BUDGET_S:g} s"
            break
        if any(float(np.max(np.abs(qg - qf))) < GRASP_SAME_RAD for qf in failed):
            continue  # a near-duplicate of a pose whose way up was blocked (the seeds' branches meet)
        c.geo.set_arm(c.side, qg)
        if k > 0 and tol is not None and O.style_error(g.approach_style, O.hand_pose(c.geo, c.side, g.grip)) > tol:
            continue  # STRICT: only the poses within the tolerance
        tip_z = min(float(c.geo.point(c.side, t)[2]) for t in TIPS)
        pg = c.geo.point(c.side, g.point)
        ag, _ = c.geo.hand_axes(c.side)
        dz = max(float(g.min_descent_m or 0.0), tip_z_hover - tip_z, 0.0)
        tried += 1
        if dz < 1e-4:
            qh = qg
        else:
            line = P.cartesian_line(chk_up, qg, pg + np.array([0.0, 0.0, dz]), point=g.point, approach=ag)
            if not line.ok:
                why = line.reason + (f": {line.blocked_by}" if line.blocked_by else "")
                failed.append(np.asarray(qg))
                continue
            qh = np.asarray(line.path[-1])
        if k > 0:  # a lower-ranked grasp pose: its own numbers
            c.geo.set_arm(c.side, qg)
            hp = O.hand_pose(c.geo, c.side, g.grip)
            c.info["orientation"].update({"heading_deg": round(hp.heading_deg, 1), "palm_down_deg":
                                          round(hp.palm_down_deg, 1), "tilt_deg": round(hp.tilt_deg, 1),
                                          "pitch_down_deg": round(hp.pitch_down_deg, 1), "close_yaw_deg":
                                          round(hp.close_yaw_deg, 1), "approach_err_deg":
                                          round(O.style_error(g.approach_style, hp), 1), "grasp_rank": k})  # fmt: skip
        c.geo.set_arm(c.side, qh)
        z_top = float(c.geo.point(c.side, LOWER_POINT)[2])
        c.geo.set_arm(c.side, qg)
        # the descent the client asks arm.lower for = the drop of ITS reference point (pinch) between the two poses
        # (the hand turns on the way: the grip's point dropped dz, the pinch point up to ~5 mm more)
        dz = max(0.0, z_top - float(c.geo.point(c.side, LOWER_POINT)[2])) if dz >= 1e-4 else 0.0
        if band is not None:
            c.info["orientation"].update({"grasp_z_from": z_from, "tips_out_mm": round(1000.0 * O.tips_out_m(
                c.geo, c.side, g.grip, band, gr.width_m), 1)})  # fmt: skip
        c.info["orientation"].update({"grasp_z_m": round(gr.z, 4), "descent_m": round(dz, 4),
                                      "grasp_q": np.round(qg, 4).tolist(), "grasp_point": g.point,
                                      "target_clear_m": clear_m if gr.target else None,
                                      "width_m": gr.width_m})  # fmt: skip
        c.geo.set_arm(c.side, qh)
        return qh, ag
    raise _Refuse("UNREACHABLE", f"the hand can close at {np.round([*xy, gr.z], 3).tolist()} but has no straight way "
                  f"up from there to the hover height ({why})", **c.info.get("orientation", {}))  # fmt: skip


def _try_up(c: _Ctx, leg: str, q0: FloatArray, p_up: FloatArray, approach: FloatArray | None) -> FloatArray:
    """The straight rise to the transit height; when it is not possible (out of reach straight up, blocked), no rise
    leg: the transit then plans from where the arm is (a joint path that clears the world)."""
    n = len(c.legs)
    try:
        return c.go_line(leg, q0, p_up, strict=True, approach=approach)
    except _Refuse:
        del c.legs[n:]
        return q0


# ---------------------------------------------------------------------------------------------- out and around
# A hand under a declared surface (the resting arms at a table: the hands hang under its near edge) going to a pose
# above it: v2's way out (r1mc2/outaround.py, git r21_try 2a4a165 outaround.py:1-60), the way a person moved the arms
# on the real robot under the balance policy (5-6 Oct lead-throughs, tilt < 1.8 deg): the shoulder rolls OUT to the
# side and yaws, the elbow kept and the shoulder pitched back only a little ("out": the hand swings to the side, y ~0.5
# m, and up behind the edge), then pitch forward, the elbow opens ("over": the hand up and over the table), then on to
# the goal pose. Keyframes = the two demonstrated poses adapted to the start and the world: candidates nearest the
# demonstration first (v2's grids), straight joint-space legs checked densely with a WIDER margin than the planner's
# 1 cm - OUT_AROUND_CLEARANCE_M (v2's 2 cm) for EVERY part incl. the hand (no "hand works just above a support"
# exception: v2's lead-throughs passed the edge at 0-1 cm, which the planner must not copy) - and within v2's
# demonstrated envelope as well as the policy's. Left-arm angles (pitch, roll, yaw, elbow); the right arm mirrored
# (roll / yaw / wrist roll negated: JOINT_DIRECTIONS.md - positive roll is toward the robot's LEFT on both arms, so
# the right arm goes out with a NEGATIVE roll). The wrist roll stays as it is until the last leg.
OUT_AROUND_CLEARANCE_M = 0.02  # v2 CLEARANCE["outaround"] (arms.py:197): every part 2 cm from the declared entities
# tried first: 3 cm (session 10: the lockstep run measured the arm 10.5 mm from the top on a 2 cm plan - the body's
# sway between plan and execution + the executor's curve between the checked straight legs); then v2's 2 cm
OUT_AROUND_MARGINS: tuple[tuple[float, float], ...] = ((0.03, 3.0), (OUT_AROUND_CLEARANCE_M, 8.0))  # (m, budget s)
OVER_ABOVE_M = 0.05  # v2 outaround_over: at the "over" pose the whole hand >= 5 cm above the surface it came from under
OUT_AROUND_SPEED = 0.5  # v2 outaround.JOINT_VMAX 0.5 rad/s peak ("slowly, like the person"; V_MAX x speed)
UNDER_DEPTH_M = 0.01  # a hand point more than this below a surface's top ...
UNDER_REACH_M = 0.10  # ... and within this of its footprint (under its near edge or just behind it) = under it
_MIRROR = np.array([1.0, -1.0, -1.0, 1.0, -1.0])
DEMO_OUT = np.array([0.431, 0.958, 0.471, 0.831])  # v2 outaround.py:44 (session 1 right arm at mono 1218.9, mirrored)
DEMO_OVER = np.array([0.091, 0.961, 0.485, 0.579])  # v2 outaround.py:45 (mono 1219.9: the hand up and over)
# v2's demonstrated envelope (outaround.py ENVELOPE_LO / _HI: R1MC-Manip's trained range, the roll widened to 1.06
# where the person took it); v3 intersects it with the live policy's trained range and the soft limits
DEMO_ENV_LO = np.array([-1.7, -0.1, -1.0, -0.6, -1.6])
DEMO_ENV_HI = np.array([0.5, 1.06, 1.0, 2.0, 1.6])


def demo_envelope(side: str) -> tuple[FloatArray, FloatArray]:
    """v2's demonstrated out-and-around envelope for `side` (the right arm mirrored)."""
    if side == "left":
        return DEMO_ENV_LO.copy(), DEMO_ENV_HI.copy()
    a, b = _MIRROR * DEMO_ENV_LO, _MIRROR * DEMO_ENV_HI
    return np.minimum(a, b), np.maximum(a, b)


def _left(q: FloatArray, side: str) -> FloatArray:
    return np.asarray(q, dtype=np.float64) if side == "left" else _MIRROR * np.asarray(q, dtype=np.float64)


def surface_above_hand(geo: Geometry, side: str) -> tuple[str, float] | None:
    """The declared surface (or container floor / object top) a hand point is UNDER at the arm's current pose: the
    point > UNDER_DEPTH_M below its top and within UNDER_REACH_M of its footprint -> (id, top z) or None."""
    best: tuple[str, float] | None = None
    for name in ("pinch", "palm", "wrist", "thumb_tip", "index_tip", "middle_tip"):
        p = geo.point(side, name)
        for dx in (0.0, UNDER_REACH_M):  # under the footprint, or within UNDER_REACH_M in front of its near edge
            for sid, top in geo.surfaces_under(np.array([p[0] + dx, p[1]])):
                if p[2] < top - UNDER_DEPTH_M and (best is None or top > best[1]):
                    best = (sid, top)
    return best


def _over_ok(geo2: Geometry, chk2: P.Checker, side: str, q: FloatArray, top: float) -> bool:
    if not chk2.valid(q):
        return False
    return geo2.lowest(side) >= top + OVER_ABOVE_M


def _out_around(c: _Ctx, q1: FloatArray, speed: float, under: tuple[str, float]) -> None:
    """start -> "out" -> "over" -> q1 (module comment above), at the widest margin of OUT_AROUND_MARGINS that has a
    way. Raises _Refuse with the reason (at the last margin) when no keyframes clear."""
    why = ""
    for margin, budget in OUT_AROUND_MARGINS:
        why = _out_around_at(c, q1, speed, under, margin, budget)
        if not why:
            return
    raise _Refuse("COLLISION", f"out and around from under {under[0]}: {why}", out_around_under=under[0])


def _out_around_at(
    c: _Ctx, q1: FloatArray, speed: float, under: tuple[str, float], margin: float, budget_s: float
) -> str:
    """One keyframe search at `margin` -> "" (the legs appended) or why not."""
    assert c.snapshot is not None and c.stance is not None
    side, q0 = c.side, c.q0
    geo2 = geometry_for(c.snapshot, margin)
    geo2.set_stance(*c.stance)
    lo_e, hi_e = demo_envelope(side)
    lo, hi = np.maximum(c.limits[0], lo_e), np.minimum(c.limits[1], hi_e)
    if np.any(q0 < lo - 0.05) or np.any(q0 > hi + 0.05):
        return "the arm does not start inside the demonstrated range"
    lo, hi = np.minimum(lo, q0), np.maximum(hi, q0)  # never exclude where the arm already is
    chk2 = P.Checker(geo2, side, (lo, hi), allow=c.goal.allow, allow_hand=c.goal.allow_hand,
                     hand_support_m=margin)  # fmt: skip
    pairs, bad = P.start_pairs(chk2, q0)
    if bad:
        return f"at the start the arm touches {bad}"
    q0l = _left(q0, side)

    def key(k4: tuple[float, ...]) -> FloatArray:
        return np.clip(_left(np.r_[k4, q0l[4]], side), lo, hi)

    def near(cands: set[tuple[float, ...]], demo: FloatArray) -> list[tuple[float, ...]]:
        return sorted(cands, key=lambda k: (float(np.linalg.norm(np.array(k) - demo)), k))

    outs = near({(round(min(q0l[0] + dp, DEMO_ENV_HI[0]), 3), r, y, round(q0l[3] + de, 3))
                 for dp, r, y, de in itertools.product((0.2, 0.28, 0.1), (0.96, 1.06, 0.85),
                                                       (0.47, 0.65, 0.85, 0.3, 1.0),
                                                       (0.0, 0.15, -0.15, 0.3, 0.45))}, DEMO_OUT)  # fmt: skip
    overs = near(set(itertools.product((0.1, 0.0, 0.2, 0.3, 0.4, -0.1, 0.5), (0.96, 1.06, 0.85),
                                       (0.48, 0.6, 0.8, 0.3, 1.0), (0.58, 0.5, 0.7, 0.4, 0.9))), DEMO_OVER)  # fmt: skip
    t0 = time.monotonic()
    over_cache: dict[tuple[float, ...], bool] = {}
    tried = 0
    why = f"no way out to the side and around keeps {100 * margin:.0f} cm from the world"
    for k1 in outs:
        if time.monotonic() - t0 > budget_s:
            why += " (search time out)"
            break
        q_out = key(k1)
        tried += 1
        left = P.escape_segment(chk2, pairs, q0, q_out)
        if left is None:
            continue
        for k2 in overs:
            if time.monotonic() - t0 > budget_s:
                break
            q_over = key(k2)
            if k2 not in over_cache:
                over_cache[k2] = _over_ok(geo2, chk2, side, q_over, under[1])
            if not over_cache[k2]:
                continue
            if P.escape_segment(chk2, left, q_out, q_over) is None:
                continue
            # on to the goal pose with the planner's own checks (a straight joint leg, else RRT-Connect) - session 13
            # (HW2 day 2: this leg passed a container 1.19 cm away and touched it; session 13's patch): at the SAME
            # margin as out / over (the planner's own limits, not the demonstrated envelope)
            chk_on = P.Checker(geo2, side, c.limits, allow=c.goal.allow, allow_hand=c.goal.allow_hand,
                               hand_support_m=margin)  # fmt: skip
            if chk_on.segment(q_over, q1):
                last = np.array([q_over, q1])
            else:
                r = P.rrt_connect(chk_on, q_over, q1, seed=c.goal.seed)
                if not r.ok:
                    continue
                last = r.path
            sp = min(speed, OUT_AROUND_SPEED)
            c.legs.append(("out", np.array([q0, q_out]), GUARD_LAG, sp))
            c.legs.append(("over", np.array([q_out, q_over]), GUARD_LAG, sp))
            c.legs.append(("on", last, GUARD_LAG, sp))
            c.methods += ["out_around", "out_around", "joint" if len(last) == 2 else "rrt"]
            c._leg_done()
            c.escape = {}
            c.info["out_around"] = {
                "under": under[0],
                "out_left": [round(float(v), 3) for v in k1],
                "over_left": [round(float(v), 3) for v in k2],
                "tried": tried,
                "search_s": round(time.monotonic() - t0, 3),
                "clearance_m": margin,
                "left_inside": c.escaped_from,
            }
            return ""
    return f"{why} ({tried} 'out' poses tried)"


def _joint_target(c: _Ctx, g: PlanGoal) -> FloatArray:
    """move_joint's target: 5 finite angles inside the limits / the policy envelope (a target AT a soft limit moves
    the 2 mrad knot margin inside: the executor refuses knots past it) and clear of everything."""
    if g.q is None or len(g.q) != 5 or not all(math.isfinite(x) for x in g.q):
        raise _Refuse("INVALID_ARGS", "q: 5 finite joint angles (rad)")
    q1 = np.asarray(g.q, dtype=np.float64)
    near = (q1 >= c.limits[0] - ik.KNOT_MARGIN_RAD - 1e-9) & (q1 <= c.limits[1] + ik.KNOT_MARGIN_RAD + 1e-9)
    q1 = np.where(near, np.clip(q1, c.limits[0], c.limits[1]), q1)
    if np.any(q1 < c.limits[0]) or np.any(q1 > c.limits[1]):
        bad = [int(i) for i in np.nonzero((q1 < c.limits[0]) | (q1 > c.limits[1]))[0]]
        raise _Refuse("LIMIT", f"joints {bad} outside the limits / the policy envelope", joints=bad)
    if not c.chk.valid(q1):
        raise _Refuse("COLLISION", f"at the target the arm would touch: {c.chk.last_hit}", hit=c.chk.last_hit)
    return q1


def _retrace_to_grasp(c: _Ctx, q0: FloatArray, p0: FloatArray, p1: FloatArray, guards: int, speed: float) -> bool:
    """Session 12: a lower that ends where the last hover's grasp pose is (`end_q`: its point within END_Q_TOL_M of
    the line's end) comes down along the hover's own vertical line - planned UP from the grasp pose (as the hover did)
    and run in reverse - so it ENDS at the grasp pose whose style / target clearance the hover checked. The 5-DoF arm
    cannot hold the palm's roll on a straight descent (approach + point = 5 constraints): a line planned DOWN from the
    hover turned the forearm differently and ended 0.25 rad away (a tall cylinder's palm 31 instead of <= 25 deg down).
    False = not applicable (no end_q, elsewhere, the line not possible): the plain descent."""
    g = c.goal
    if g.end_q is None or len(g.end_q) != 5 or not all(math.isfinite(v) for v in g.end_q):
        return False
    qe = np.asarray(g.end_q, dtype=np.float64)
    d = c.point(qe) - p1  # the hand turns on the hover's line: its point moves sideways a little (only the grip's
    if abs(float(d[2])) > END_Q_TOL_M or float(np.hypot(d[0], d[1])) > END_Q_XY_M or not c.chk.valid(qe):  # is plumb)
        return False
    ae, _ = c.geo.hand_axes(c.side)
    pt = g.end_point or g.point  # the hover's own line: the same start, point, approach and height (deterministic)
    pe = c.point(qe, pt)
    top = np.array([pe[0], pe[1], float(c.point(q0, pt)[2])])
    up = P.cartesian_line(c.chk, qe, top, point=pt, approach=ae)
    if not up.ok:
        return False
    q_top = np.asarray(up.path[-1])
    align: FloatArray | None = None
    gap = float(np.max(np.abs(q_top - q0)))
    if gap > RETRACE_START_RAD:
        # session 12 (a round object): the line re-solved from the grasp pose need not come back to where the hover left
        # the arm
        # - near a joint limit (the shoulder yaw at 1.868 in simulated hand-over re-picks) a few mm of execution error
        # in the line's end height sent the clamped IK to another wrist roll (0.71-0.74 rad away) and the lower fell
        # back to the plain descent, refused at the board (the item never picked). Turn the hand at the hover height
        # first: one straight joint leg q0 -> the line's top, the OPEN hand kept TARGET_CLEAR_M from the target
        # meanwhile.
        if gap > RETRACE_ALIGN_MAX_RAD:
            return False
        e_t = _target_object(c)
        chk_a = P.Checker(c.geo, c.side, c.limits, allow=g.allow, allow_hand=c.chk.allow_hand,
                          hand_clear=None if e_t is None else {str(e_t.id): O.TARGET_CLEAR_M})  # fmt: skip
        if not chk_a.segment(q0, q_top):
            return False
        align = q_top
    start = q_top if align is not None else q0
    down = np.vstack([start[None, :], np.asarray(up.path[::-1])])
    for k in range(len(down) - 1):
        if not c.chk.segment(down[k], down[k + 1]):
            return False
    if align is not None:
        c.legs.append(("align", np.vstack([q0[None, :], align[None, :]]), GUARD_LAG, speed))
        c.methods.append("joint")
        c._leg_done()
        c.info["retrace_aligned_rad"] = round(gap, 3)
    c.legs.append(("descend", P.decimate(c.chk, down, point=pt, tol_m=0.001), guards, speed))
    c.methods.append("line")
    c._leg_done()
    c.info["ended_at_grasp_pose"] = True
    return True


LEAVE_TARGET_M = 0.03  # a hover whose hand starts at its target lifts at least this much first
RETRACE_START_RAD = 0.05  # the retraced line's top within this of where the arm is (= the executor's first-knot rule)
RETRACE_ALIGN_MAX_RAD = 1.0  # session 12: beyond RETRACE_START_RAD (to this) a checked joint 'align' first


def _plan_legs(c: _Ctx) -> None:
    g, geo, side = c.goal, c.geo, c.side
    if g.kind == "move_joint":
        q1 = _joint_target(c, g)
        geo.set_arm(side, c.q0)
        under = surface_above_hand(geo, side)
        if under is not None:
            geo.set_arm(side, q1)
            if geo.lowest(side) > under[1]:  # the goal's hand is above that surface: out and around (v2's way)
                _out_around(c, q1, g.speed, under)
                return
    if c.escape:  # a start inside the clearance: leave it first (the steepest way out), then the goal as usual
        c.leave_start(speed=g.speed)
    q0 = c.q0
    if g.kind == "move_joint":
        c.go_joint("move", q0, q1, g.speed)
    elif g.kind in ("reach", "move_relative"):
        p0 = c.point(q0)
        p1 = p0 + _vec3(g.delta, "delta") if g.kind == "move_relative" else _vec3(g.position, "position")
        a = None if g.approach is None else _vec3(g.approach, "approach")
        q1 = c.solve(p1, q0, a)
        c.go_line("move", q0, p1, strict=False, speed=g.speed, approach=a, target_q=q1)
    elif g.kind == "move_linear":
        p0 = c.point(q0)
        p1 = p0 + _vec3(g.delta, "delta") if g.position is None else _vec3(g.position, "position")
        a = None if g.approach is None else _vec3(g.approach, "approach")
        c.go_line("move", q0, p1, strict=True, speed=g.speed, approach=a)
    elif g.kind == "hover":
        tgt = _vec3(g.position, "position")
        h = 0.05 if g.height_m is None else float(g.height_m)
        if not (0.0 <= h <= 0.5):
            raise _Refuse("INVALID_ARGS", "height_m: 0-0.5 m above what is under the point")
        _, top = _surface_top(geo, tgt[:2], ignore=g.allow_hand)
        if tgt[2] > 0.0:  # the client's datum (the support top), v2's: the lowest fingertip above the TABLE top
            top = float(tgt[2])
        gr = _grasp_of(c)  # session 12: a pose chosen where the grip closes (grasp_z_m / around / low + a target)
        qh, ah = _orient_target(c, tgt[:2], top + h) if gr is None else _orient_grasp(c, tgt[:2], top + h, gr)
        g = c.goal  # (_orient_grasp may set the grip's width point)
        ph = c.point(qh)
        p0 = c.point(q0)
        zt = max(_transit_z(geo, p0, ph, TRANSIT_CLEAR_M), p0[2], ph[2])
        # the transit height: the start hand's / the target's height or 6 cm over what is under the path; when the
        # chosen orientation cannot be held up there (session 10: "across" 8 mm short at the table_ready hand's
        # height), lower in 2 cm steps - never below the 6 cm over the declared tops nor below the hover itself.
        # The descend is planned UPWARD from the chosen pose (a vertical hand line holding its approach) and run in
        # reverse: it ENDS exactly at the orientation the search chose (session 10: a line planned down from an IK
        # pose above could end 0.1+ rad away - the wrap asked to close along y closed along x)
        z_min = max(_transit_z(geo, p0, ph, TRANSIT_CLEAR_M), ph[2])
        # session 12 (b4's trace of the base on HW2's cases): the hand may touch the goal's target (allow_hand) for
        # the WHOLE goal, so the transit swept through a tall declared target (the middle finger dragged across its
        # top, 468 contacts); now the lift / transit keep the open hand TARGET_CLEAR_M from it and only the final
        # descend may touch it - a grasp-pose hover (a style) not even that (its way down was checked clear). A hand
        # that STARTS at the target (the table_ready hand's open fingers 3 mm over a 13 cm object) lifts LEAVE_TARGET_M
        # first (the lift may touch it: it moves away).
        e_t = _target_object(c)
        # session 13: a style's own clearance (an `around` tall box: the transit's 4 mm + a 5 mm placement error = the
        # thumb on its top edge in the simulated twin, the robot's "grazing ... pushed down on a corner once")
        hc_t = (
            None
            if e_t is None or str(e_t.id) in c.first_allow
            else {str(e_t.id): O.target_clear_m(g.approach_style, _band_of(e_t)[2])}
        )

        def near_target(qq: FloatArray) -> bool:  # the hand within TARGET_CLEAR_M of the target
            if not hc_t:
                return False
            c.geo.set_arm(side, qq)
            hs = c.geo.hits(side, hand_clear=hc_t, allow=g.allow, allow_hand=c.chk.allow_hand)
            return any(h.other.split(":", 1)[-1] in hc_t for h in hs)

        near0 = near_target(q0)
        if near0:
            zt = max(zt, p0[2] + LEAVE_TARGET_M)
        c.chk.hand_clear = hc_t if gr is not None else None
        up: FloatArray | None = None
        q_above: FloatArray | None = None
        r_up = P.cartesian_line(c.chk, qh, np.array([ph[0], ph[1], zt]), point=g.point, approach=ah)
        if r_up.ok:
            up = r_up.path
        else:  # session 9's way at the full transit height: an IK pose above (the approach secondary) + a line down
            try:
                q_above = c.solve(np.array([ph[0], ph[1], zt]), qh, ah)
            except _Refuse:
                for zt_k in np.arange(zt - 0.02, z_min - 1e-9, -0.02).tolist():
                    r_up = P.cartesian_line(c.chk, qh, np.array([ph[0], ph[1], zt_k]), point=g.point, approach=ah)
                    if r_up.ok:
                        up, zt = r_up.path, float(zt_k)
                        break
        if up is None and q_above is None:
            raise _Refuse("UNREACHABLE", f"the chosen hand orientation cannot be held above the target at any transit "
                          f"height {z_min:.3f}-{zt:.3f} m", **c.info.get("orientation", {}))  # fmt: skip
        c.chk.hand_clear = None if near0 else hc_t  # a hand starting at the target lifts away from it
        q = _try_up(c, "lift", q0, np.array([p0[0], p0[1], zt]), None)
        c.chk.hand_clear = None if near_target(q) else hc_t  # (still at it after the lift: the transit as before)
        if hc_t and c.chk.hand_clear is None:
            c.info["transit_may_touch_target"] = True
        p_above = np.array([ph[0], ph[1], zt])
        if up is None:
            assert q_above is not None
            q = c.go_line("transit", q, p_above, strict=False, target_q=q_above, approach=ah)
            c.chk.hand_clear = hc_t if gr is not None else None
            c.go_line("descend", q, ph, strict=False, target_q=qh, approach=ah)
        else:
            q_up = np.asarray(up[-1])
            q = c.go_line("transit", q, p_above, strict=False, target_q=q_up, approach=ah)
            if float(np.max(np.abs(q - q_up))) > 1e-9:  # the transit ended elsewhere: a checked joint step onto it
                c.go_joint("align", q, q_up, g.speed)
            c.chk.hand_clear = hc_t if gr is not None else None
            down = np.asarray(up[::-1])
            if len(down) >= 2 and float(np.max(np.abs(down[-1] - down[0]))) > 1e-6:
                c.legs.append(("descend", P.decimate(c.chk, down, point=g.point, tol_m=0.003), GUARD_LAG, g.speed))
                c.methods.append("line")
                c._leg_done()
    elif g.kind == "lower":
        d = float(g.max_depth_m)
        if not (0.0 < d <= 0.3):
            raise _Refuse("INVALID_ARGS", "max_depth_m: (0, 0.3] m")
        p0 = c.point(q0)
        a0, _ = geo.hand_axes(side)
        sid, top = _surface_top(geo, p0[:2])
        c.geo.set_arm(side, q0)
        low = geo.lowest(side)
        depth = d if g.until == "depth" else min(d, max(low - top, 0.0) + 0.01)  # 1 cm past the surface: the guard
        c.chk.allow_hand = tuple(c.chk.allow_hand) + ((sid,) if sid != "floor" else ())
        guards = GUARD_LAG | GUARD_CONTACT if g.until in ("contact", "surface") else GUARD_LAG
        p1 = p0 - np.array([0.0, 0.0, depth])
        if not _retrace_to_grasp(c, q0, p0, p1, guards, g.speed * LOWER_SPEED):
            c.go_line("descend", q0, p1, strict=True, speed=g.speed * LOWER_SPEED, guards=guards, approach=a0)
    elif g.kind == "carry":
        tgt = _vec3(g.position, "position")
        lift = LIFT_DEFAULT_M if g.lift_m is None else float(g.lift_m)
        if not (0.0 <= lift <= 0.3):
            raise _Refuse("INVALID_ARGS", "lift_m: 0-0.3 m")
        p0 = c.point(q0)
        a0, _ = geo.hand_axes(side)
        q_end = c.solve(tgt, q0, a0)
        zt = max(_transit_z(geo, p0, tgt, TRANSIT_CLEAR_M), p0[2] + lift, tgt[2] + lift)
        q = _try_up(c, "rise", q0, np.array([p0[0], p0[1], zt]), a0)
        p_above = np.array([tgt[0], tgt[1], zt])
        q_above = c.solve(p_above, q_end, a0)
        q = c.go_line("travel", q, p_above, strict=False, target_q=q_above, approach=a0)
        c.go_line("lower", q, tgt, strict=False, target_q=q_end, approach=a0)
    else:
        raise _Refuse("INVALID_ARGS", f"unknown goal kind {g.kind!r} (known: {', '.join(KINDS)})")


def plan(
    snapshot: WorldSnapshot,
    goal: PlanGoal,
    q_now: FloatArray,
    *,
    imu_quat: FloatArray | None = None,
    hands: dict[str, FloatArray] | None = None,
    envelope: tuple[FloatArray, FloatArray] | None = None,
    geometry: Geometry | None = None,
    arm_target: FloatArray | None = None,
) -> PlanResult:
    """Plan a whole arm goal on a world snapshot from the measured joints (26, SDK order). Pure: moves nothing.

    arm_target: the executor's CURRENT TARGET for the goal's arm (5; what it holds - the measured pose sags below it
    by the arm's static error, A3 / A9). The executor refuses a trajectory whose first knot is > 0.05 rad from its
    target (EXECUTOR.md), so r1d passes it: the path starts there. Default: the measured arm joints."""
    t0 = time.perf_counter()
    c0 = time.thread_time()
    pid = f"p{next(_ids)}-{snapshot.version}"
    try:
        if goal.arm not in ARM_JOINTS:
            raise _Refuse("INVALID_ARGS", "arm: left | right")
        if goal.frame not in FRAMES:
            raise _Refuse("UNKNOWN_FRAME", f"frame {goal.frame!r}: known {', '.join(FRAMES)}")
        q26 = np.asarray(q_now, dtype=np.float64)
        if q26.shape != (J.N,) or not np.all(np.isfinite(q26)):
            raise _Refuse("INVALID_ARGS", "q_now: the 26 measured joints")
        goal = _resolved(goal)
        if goal.hand is not None:
            from r1sim import dex3

            try:
                hq = dex3.preset(goal.hand, goal.arm)
            except KeyError:
                raise _Refuse("INVALID_ARGS", f"hand: a hand preset ({', '.join(sorted(dex3.PRESETS_LEFT))})") from None
            hands = {**(hands or {}), goal.arm: hq}
        geo = geometry if geometry is not None else geometry_for(snapshot)
        geo.set_stance(q26, imu_quat, hands)
        if goal.held is not None and geometry is None:  # the object travels with the hand (session 10)
            geo = geometry_for(snapshot, None, (held_in_palm(geo, goal.arm, dict(goal.held)),))
            geo.set_stance(q26, imu_quat, hands)
        q_plan = q26.copy()
        if arm_target is not None:
            at = np.asarray(arm_target, dtype=np.float64)
            if at.shape != (5,) or not np.all(np.isfinite(at)):
                raise _Refuse("INVALID_ARGS", "arm_target: the arm's 5 target joints")
            q_plan[ARM_JOINTS[goal.arm]] = at
        c = _Ctx(geo, goal, q_plan, envelope)
        c.snapshot, c.stance = snapshot, (q26, imu_quat, hands)
        c.start_touching()
        _plan_legs(c)
        legs = [lg for lg in c.legs if float(np.max(np.abs(lg[1][-1] - lg[1][0]))) > 1e-4 or len(lg[1]) > 2]
        if not legs:  # already there: one still segment (the executor holds; the result reports it)
            legs = [c.legs[-1]] if c.legs else [("hold", np.array([c.q0, c.q0]), GUARD_LAG, 1.0)]
        segs = tuple(_timed_segment(goal.arm, p, sp, gd, leg) for leg, p, gd, sp in legs)
        q_end = segs[-1].q[-1]
        geo.set_arm(goal.arm, q_end)
        tips = {k: geo.point(goal.arm, k).round(4).tolist() for k in ("thumb_tip", "index_tip", "middle_tip")}
        pred = {
            "fingertips_m": tips,
            # = the RT core's arm model (r1rt_py.fingertips_base) / v2 api._grasp_centre: what r1d measures results with
            "grasp_centre_m": geo.point(goal.arm, "grasp_centre").round(4).tolist(),
            "pinch_m": geo.point(goal.arm, "pinch").round(4).tolist(),
            "point_m": geo.point(goal.arm, goal.point).round(4).tolist(),
            "lowest_tip_z_m": round(min(v[2] for v in tips.values()), 4),
            "landing_z_m": round(geo.lowest(goal.arm), 4),  # the lowest point of the hand's collision shapes
            "q_end": q_end.round(4).tolist(),
        }
        clr = _min_clearance(geo, goal.arm, segs, snapshot, skip=set(goal.allow) | set(c.chk.allow_hand))
        det = {"legs": [s.leg for s in segs], "methods": c.methods, "checks": c.chk.checks,
               "cpu_s": round(time.thread_time() - c0, 4), **c.info}  # fmt: skip
        if c.escaped_from:
            det["left_inside_clearance"] = c.escaped_from
        return PlanResult(True, "OK", pid, segs, pred, clr, "", det, time.perf_counter() - t0)
    except _Refuse as r:
        det = {**r.details, "cpu_s": round(time.thread_time() - c0, 4)}
        return PlanResult(False, r.code, pid, reason=r.reason, details=det, planning_s=time.perf_counter() - t0)


def _min_clearance(
    geo: Geometry, side: str, segs: tuple[Segment, ...], snap: WorldSnapshot, skip: set[str]
) -> float | None:
    """The smallest arm-to-entity distance at the segments' knots, the allowed touches apart (None: no entity)."""
    ents = [e for e in snap.entities if e.id not in skip]
    if not ents:
        return None
    keep = {e.id for e in ents}
    best = 0.5
    for s in segs:
        for q in s.q[:: max(1, len(s.q) // 8)]:
            geo.set_arm(side, q)
            for eid, dist in geo.distances(side, distmax=best + 1e-6).items():
                if eid in keep:
                    best = min(best, dist)
    return round(best, 4)
