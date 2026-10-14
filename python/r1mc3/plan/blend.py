"""Blending a planned multi-waypoint arm path (arm.move_path's `blend_m`, session 11).

arm.move_path plans its waypoints as chained planner goals (r1d `service/l2_paths.py`: leg k+1 planned from leg k's
planned end, the whole path checked before anything moves). Unblended, each leg is one executor TRAJECTORY that ends
AT REST, so the hand stops at every waypoint (S10: final error 2.56 mm, but a stop-and-go motion). Blended, the SAME
planned joint path becomes ONE trajectory that does not stop at the interior waypoints:

  1. the legs' knots are joined into one joint path (the waypoints are knots of it: the planner's own ends);
  2. at each interior waypoint the corner is cut inside a ball of `radius_m` around the waypoint's hand point: the
     knots inside the ball are dropped and the path's entry into the ball (A) joins its exit (B) - the executor's
     quintic through A and B rounds the corner (a chord of the ball passes the waypoint at ~0.7 r on a right angle);
     then the knots are thinned (>= THIN_STEP_RAD apart: the planner's 1 cm line steps and the ball edges make short
     segments that force a high jerk on the timing - a first try kept them and was SLOWER than the stops);
  3. the whole joint path is timed once (`timing.time_path`: from rest to rest, every sampled velocity /
     acceleration / jerk limit met; the knot velocities stay non-zero through a corner unless every joint reverses);
  4. the executed curve (the executor's quintic Hermite, sampled every SAMPLE_DT) is CHECKED before it is used: every
     sample is a valid pose for the planner's checker (the joint limits / the policy's envelope + every declared
     entity, the body, the other arm, the floor - the SAME 1 cm clearance; until the first waypoint the hand may touch
     what it started on, the planner's first-leg rule; a first leg that LEAVES a start inside the clearance is kept
     knot for knot and not re-checked: the planner's escape rule checked it); the hand stays within the CLIENT's
     tolerance (blend_m + TOL_SLACK_M) of the unblended hand path everywhere and of every interior waypoint. The end is
     the last waypoint's planned end exactly (the last knot, at rest): the plan's prediction for it is unchanged.

A blend that fails a check is retried at half the radius, then as a pass-through (radius 0: the curve through the
waypoints' exact joints, still not stopping); if that fails too the UNBLENDED plan is returned (the result says
`blend.applied` false and why) - a blend never trades clearance for smoothness. A path whose legs carry other guards
than the lag guard (a contact descent) is never blended.

Pure (a function of the world snapshot + the planned legs): it runs in the planner's worker process like `plan`
(`PlannerWorker.call`), or inline where r1d plans inline (lockstep tests).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
from r1sim import joints as J

from ..world.entities import WorldSnapshot
from ..world.geometry import ARM_JOINTS
from . import ik
from . import path as P
from .core import GUARD_LAG, MAX_KNOTS, PlanResult, Segment, geometry_for
from .timing import Timed, sample, time_path

FloatArray = npt.NDArray[np.float64]

BLEND_MAX_M = 0.05  # the largest corner radius a client may ask for (r1d's schema bounds it the same)
SAMPLE_DT = 0.01  # s: the check's sampling of the executed curve (<= 0.01 rad of joint travel at the planner's 1 rad/s)
TOL_SLACK_M = 0.001  # the waypoint tolerance's numeric slack (the closest sample vs the true closest point)
THIN_STEP_RAD = 0.10  # the thinned knots' spacing (path.decimate keeps >= 0.12 on a hand line)
_SEARCH_STEPS = 24  # bisection steps for the ball's entry / exit on a joint segment (< 0.01 mm)


def _junctions(segments: tuple[Segment, ...]) -> list[int]:
    """Segment indices that END an interior waypoint (legs are named `w<k>:<leg>` by r1d's path chain)."""
    out: list[int] = []
    names = [s.leg.split(":", 1)[0] for s in segments]
    for i in range(len(segments) - 1):
        if names[i] != names[i + 1]:
            out.append(i)
    return out


def _joined_knots(segments: tuple[Segment, ...]) -> tuple[FloatArray, list[int]]:
    """The legs' knots as one joint path + the knot index of each interior waypoint."""
    rows: list[FloatArray] = []
    ends: list[int] = []
    for s in segments:
        q = np.asarray(s.q, dtype=np.float64)
        if rows and float(np.max(np.abs(q[0] - rows[-1]))) <= 1e-9:
            q = q[1:]
        rows.extend(q)
        ends.append(len(rows) - 1)
    jn = _junctions(segments)
    return np.asarray(rows), [ends[i] for i in jn]


class _Hand:
    """The goal's hand point at a joint pose (the planner's geometry with the stance set)."""

    def __init__(self, geo: Any, side: str, point: str) -> None:
        self.geo, self.side, self.point = geo, side, point
        self.evals = 0

    def __call__(self, q: FloatArray) -> FloatArray:
        self.evals += 1
        self.geo.set_arm(self.side, q)
        return np.asarray(self.geo.point(self.side, self.point), dtype=np.float64)


def _ball_edge(hand: _Hand, w: FloatArray, r: float, inside: FloatArray, outside: FloatArray) -> FloatArray:
    """The pose on the straight joint segment inside -> outside where the hand point leaves the ball (|p - w| = r)."""
    lo, hi = 0.0, 1.0
    for _ in range(_SEARCH_STEPS):
        mid = 0.5 * (lo + hi)
        if float(np.linalg.norm(hand(inside + mid * (outside - inside)) - w)) < r:
            lo = mid
        else:
            hi = mid
    return np.asarray(inside + hi * (outside - inside))


def _cut_corners(Q: FloatArray, junctions: list[int], r: float, hand: _Hand) -> FloatArray:
    """Q with each interior waypoint's knots inside the ball of radius r (around its hand point) replaced by the ball's
    entry A and exit B (r 0: Q itself, a pass-through of the waypoints' exact joints)."""
    if r <= 0.0:
        return Q
    out: list[FloatArray] = []
    cursor = 0
    for n, j in enumerate(junctions):
        # each corner stays in its own half of the legs around it (the windows never overlap; the start / end stay)
        lo_bound = 0 if n == 0 else (junctions[n - 1] + j) // 2 + 1
        hi_bound = len(Q) - 1 if n == len(junctions) - 1 else (j + junctions[n + 1]) // 2
        lo_bound = max(lo_bound, cursor)
        w = hand(Q[j])
        a = j  # the first knot inside the ball, going back
        while a - 1 > lo_bound and float(np.linalg.norm(hand(Q[a - 1]) - w)) < r:
            a -= 1
        b = j  # the last knot inside the ball, going forward
        while b + 1 < hi_bound and float(np.linalg.norm(hand(Q[b + 1]) - w)) < r:
            b += 1
        out.extend(Q[cursor:a])  # everything before the ball (knot a itself is inside it: replaced)
        out.append(_ball_edge(hand, w, r, Q[a], Q[a - 1]) if a - 1 >= lo_bound else Q[a])
        out.append(_ball_edge(hand, w, r, Q[b], Q[b + 1]))
        cursor = b + 1
    out.extend(Q[cursor:])
    return np.asarray(out)


def _thin(path: FloatArray, keep_until: int, min_step: float) -> FloatArray:
    """Fewer knots after index keep_until (each kept knot >= min_step rad from the previous one; the last kept): the
    planner's 1 cm line steps and the corners' ball edges make short segments that force a high jerk on the timing
    (path.decimate's note) - a blended path would then be SLOWER than the stops. The executed curve is checked after
    (`_check`), so nothing here has to be proven on its own."""
    keep = list(range(keep_until + 1))
    for i in range(keep_until + 1, len(path) - 1):
        if float(np.max(np.abs(path[i] - path[keep[-1]]))) >= min_step:
            keep.append(i)
    if keep[-1] != len(path) - 1:
        if len(keep) > keep_until + 1 and float(np.max(np.abs(path[-1] - path[keep[-1]]))) < 0.5 * min_step:
            keep[-1] = len(path) - 1  # no short last segment either
        else:
            keep.append(len(path) - 1)
    return np.asarray(path[keep])


def _reference(segments: tuple[Segment, ...], hand: _Hand) -> FloatArray:
    """The UNBLENDED hand path as the executor would run it (every leg's quintic sampled every SAMPLE_DT)."""
    pts: list[FloatArray] = []
    for sg in segments:
        tm = Timed(sg.t, sg.q, sg.dq, sg.ddq, 0.0, 0.0, 0.0)
        _, qs, _ = sample(tm, SAMPLE_DT)
        pts.extend(hand(q) for q in np.vstack([qs, sg.q[-1]]))
    return np.asarray(pts)


def _deviation(pts: FloatArray, ref: FloatArray) -> FloatArray:
    """Per point the distance to the nearest reference point (chunked: no (n, m) array larger than ~4 M)."""
    out = np.empty(len(pts))
    step = max(1, 4_000_000 // max(1, len(ref)))
    for i in range(0, len(pts), step):
        d = np.linalg.norm(pts[i : i + step, None, :] - ref[None, :, :], axis=2)
        out[i : i + step] = d.min(axis=1)
    return out


def _check(
    tm: Timed,
    chk: P.Checker,
    chk_first: P.Checker,
    hand: _Hand,
    waypoints: list[FloatArray],
    ref: FloatArray,
    tol: float,
    t_skip: float,
) -> tuple[bool, str, list[float], float]:
    """The executed curve, sampled every SAMPLE_DT: the hand within tol (+ slack) of the unblended hand path everywhere
    and of each interior waypoint; every sample valid - before the first waypoint for `chk_first` (the planner's
    first-leg rule: the hand may touch what it started on), after it for `chk` (the normal checks); samples before
    t_skip (a kept "leave" leg) are the planner's own -> (ok, why not, the closest approach per waypoint, the largest
    deviation) in m."""
    ts, qs, _ = sample(tm, SAMPLE_DT)
    ts = np.append(ts, tm.duration)
    qs = np.vstack([qs, tm.q[-1]])
    pts = np.array([hand(q) for q in qs])
    allp = np.vstack([pts, np.array([hand(q) for q in tm.q])])  # + the knots: a pass-through's waypoint IS a knot
    closest = [float(np.min(np.linalg.norm(allp - w, axis=1))) for w in waypoints]
    dev = _deviation(pts, ref)
    worst = float(dev.max())
    for k, c in enumerate(closest):
        if c > tol + TOL_SLACK_M:
            return False, f"waypoint {k}: passed {1000 * c:.1f} mm away (tolerance {1000 * tol:.1f} mm)", closest, worst
    if worst > tol + TOL_SLACK_M:
        at = float(ts[int(np.argmax(dev))])
        why = f"at t = {at:.2f} s {1000 * worst:.1f} mm off the planned path (tolerance {1000 * tol:.1f} mm)"
        return False, why, closest, worst
    t_first = float(ts[int(np.argmin(np.linalg.norm(pts - waypoints[0], axis=1)))])
    for t, q in zip(ts, qs, strict=True):
        if t + 1e-9 < t_skip:
            continue
        cc = chk_first if t < t_first else chk
        if not cc.valid(q):
            return False, f"at t = {t:.2f} s: {cc.last_hit}", closest, worst
    return True, "", closest, worst


def _window_clearance(geo: Any, side: str, tm: Timed, snapshot: WorldSnapshot, t_from: float) -> float | None:
    """The smallest arm-to-entity distance on the checked part of the curve (every 4th sample). (An exact 0.0 next to
    a passed contact check was MuJoCo 3.5's box-box mj_geomDistance answering 0 with a fromto - Geometry.distances
    takes the conservative bound for it since session 11; skipped here as well.)"""
    if not snapshot.entities:
        return None
    ts, qs, _ = sample(tm, SAMPLE_DT)
    best = 0.5
    for t, q in list(zip(ts, qs, strict=True))[::4]:
        if t + 1e-9 < t_from:
            continue
        geo.set_arm(side, q)
        for dist in geo.distances(side, distmax=best + 1e-6).values():
            if dist > 0.0:
                best = min(best, dist)
    return round(best, 4)


def blend_path(
    snapshot: WorldSnapshot,
    joined: PlanResult,
    q_now: FloatArray,
    *,
    arm: str,
    radius_m: float,
    speed: float = 1.0,
    point: str = "pinch",
    imu_quat: FloatArray | None = None,
    hands: dict[str, FloatArray] | None = None,
    envelope: tuple[FloatArray, FloatArray] | None = None,
    arm_target: FloatArray | None = None,
) -> PlanResult:
    """The joined, checked multi-waypoint plan -> one blended TRAJECTORY (or `joined` itself with the reason).

    joined: r1d's path chain result (segments named `w<k>:<leg>`, every leg planned + checked); q_now: the 26
    measured joints; arm_target: the arm's executor target the path starts from (as `plan`)."""
    t0 = time.perf_counter()
    c0 = time.thread_time()
    r_req = float(radius_m)

    def unblended(why: str, **extra: Any) -> PlanResult:
        info = {"applied": False, "requested_m": r_req, "why": why, "cpu_s": round(time.thread_time() - c0, 4),
                **extra}  # fmt: skip
        return replace(joined, details={**dict(joined.details), "blend": info})

    if not joined.ok:
        return joined
    if not 0.0 < r_req <= BLEND_MAX_M:
        return unblended(f"radius {r_req} m: (0, {BLEND_MAX_M}]")
    segs = tuple(joined.segments)
    if any(s.arm != arm for s in segs):
        return unblended("a leg for the other arm")
    if any(int(s.guards) & ~GUARD_LAG for s in segs):
        return unblended("a leg with another guard than the lag guard (a guarded descent) is never blended")
    Q, junction_knots = _joined_knots(segs)
    if not junction_knots:
        return unblended("one waypoint: nothing to blend")

    q26 = np.asarray(q_now, dtype=np.float64).copy()
    if q26.shape != (J.N,):
        return unblended("q_now: the 26 measured joints")
    joints = list(ARM_JOINTS[arm])
    q26[joints] = Q[0] if arm_target is None else np.asarray(arm_target, dtype=np.float64)
    geo = geometry_for(snapshot)
    geo.set_stance(q26, imu_quat, hands)
    lo, hi = ik.arm_limits(arm, envelope)
    limits = (np.minimum(lo, Q.min(axis=0)), np.maximum(hi, Q.max(axis=0)))  # the planned path is inside them
    chk = P.Checker(geo, arm, limits)
    # the planner's first-leg rule (core._Ctx.start_touching): a hand that starts ON an entity may touch it until it
    # has left it - here until the first waypoint
    geo.set_arm(arm, Q[0])
    hand_bodies = {geo.m.body(b).name for b in geo.hand_bodies[arm]}
    first_allow = tuple(sorted({h.other.split(":", 1)[1] for h in geo.hits(arm) if h.other.startswith("entity:")
                                and h.arm_body in hand_bodies}))  # fmt: skip
    chk_first = P.Checker(geo, arm, limits, allow_hand=first_allow)
    hand = _Hand(geo, arm, point)
    waypoints = [hand(Q[j]) for j in junction_knots]

    ref = _reference(segs, hand)
    # the knots of a first leg that LEAVES a start inside the clearance (the planner's "leave" leg, checked with its own
    # rule, not the normal one) are kept as planned; every other knot may be thinned (the curve is checked after)
    keep_until = 0
    if segs[0].leg.split(":", 1)[-1] == "leave":
        keep_until = len(segs[0].q) - 1
    tries: list[dict[str, Any]] = []
    for r, thin in ((r_req, True), (0.5 * r_req, True), (0.0, False)):
        path = _cut_corners(Q, junction_knots, r, hand)
        if thin:
            path = _thin(path, keep_until, THIN_STEP_RAD)
        dense = P.densify(path)
        tm = time_path(dense, speed=speed)
        if len(tm.t) > MAX_KNOTS:
            tries.append({"radius_m": r, "why": f"{len(tm.t)} knots > {MAX_KNOTS}"})
            continue
        # a kept "leave" leg runs the planner's own knots (its escape rule): the checks start at its end
        t_skip = 0.0
        if keep_until > 0:
            t_skip = float(tm.t[int(np.argmin(np.max(np.abs(tm.q - Q[keep_until]), axis=1)))])
        ok, why, closest, worst = _check(tm, chk, chk_first, hand, waypoints, ref, r_req, t_skip)
        tries.append({"radius_m": r, "ok": ok, "why": why, "closest_mm": [round(1000 * c, 1) for c in closest],
                      "deviation_mm": round(1000 * worst, 1), "duration_s": round(tm.duration, 3)})  # fmt: skip
        if not ok:
            continue
        seg = Segment(arm, tm.t, tm.q, tm.dq, tm.ddq, GUARD_LAG, f"w0-w{len(junction_knots)}:blend")
        ts, qs, _ = sample(tm, SAMPLE_DT)
        pts = np.array([hand(qq) for qq in qs])
        pass_t = [round(float(ts[int(np.argmin(np.linalg.norm(pts - wp, axis=1)))]), 3) for wp in waypoints]
        clr = _window_clearance(geo, arm, tm, snapshot, t_skip)
        info = {
            "applied": True,
            "requested_m": r_req,
            "radius_m": r,
            "closest_mm": [round(1000 * c, 1) for c in closest],
            "pass_t_s": pass_t,  # when the curve passes each interior waypoint (r1d: waypoints_done after a stop)
            "deviation_mm": round(1000 * worst, 1),
            "duration_s": round(tm.duration, 3),
            "unblended_s": round(joined.duration_s, 3),
            "v_peak": round(tm.v_peak, 3),
            "tries": tries,
            "checks": chk.checks,
            "cpu_s": round(time.thread_time() - c0, 4),
        }
        if r < r_req:
            info["why_smaller"] = "; ".join(str(t["why"]) for t in tries[:-1])
        clr_all = [c for c in (clr, joined.clearance_m) if c is not None]
        details = {**dict(joined.details), "legs": [seg.leg], "blend": info}
        return PlanResult(True, "OK", f"{joined.plan_id}+blend", (seg,), dict(joined.predicted),
                          min(clr_all) if clr_all else None, "", details,
                          joined.planning_s + time.perf_counter() - t0)  # fmt: skip
    return unblended("; ".join(f"r {1000 * t['radius_m']:.0f} mm: {t['why']}" for t in tries), tries=tries)
