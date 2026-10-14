"""Paths for one arm (DESIGN.md §7.2 step 2): a straight Cartesian line when it is valid, otherwise RRT-Connect in
joint space with shortcut smoothing; every candidate checked DENSELY (joint steps <= CHECK_STEP_RAD) against the
robot itself, the other arm and the world through the one geometry engine.

The straight line (v2 arms.py line_path, git r21_try 2a4a165 arms.py:719): the hand point moves on the segment in
1 cm steps, each step solved by IK seeded with the previous one (so the arm follows continuously; a jump > JUMP_RAD
between steps = the line leaves the arm's reach / passes a singularity -> invalid), the approach held or interpolated.
RRT-Connect (Kuffner & LaValle 2000) with a seeded RNG (a plan is reproducible from its seed); the shortcut pass
replaces sub-paths by straight joint segments that are collision-free (Geraerts & Overmars 2007's random shortcuts).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..world.geometry import HAND_SUPPORT_CLEARANCE_M, Geometry
from . import ik

FloatArray = npt.NDArray[np.float64]

CHECK_STEP_RAD = 0.02  # dense collision checking: no joint moves more than this between two checks
LINE_STEP_M = 0.01
JUMP_RAD = 0.25  # a line step needing more joint motion than this is not continuous (after halving the step)
MIN_STEP_M = 0.002  # ... down to this hand step (0.25 rad per 2 mm = a branch flip or a singularity, not a line)
RRT_STEP_RAD = 0.15
RRT_MAX_ITERS = 4000
GOAL_BIAS = 0.1


@dataclass(frozen=True)
class PathResult:
    ok: bool
    path: FloatArray  # (K, 5) waypoints, the first = the start
    method: str  # "joint", "line", "rrt"
    reason: str = ""
    checks: int = 0
    blocked_by: str = ""


class Checker:
    """Dense validity for one arm: a joint pose is valid when inside the limits and without blocking contacts."""

    def __init__(
        self,
        geo: Geometry,
        side: str,
        limits: tuple[FloatArray, FloatArray],
        *,
        allow: tuple[str, ...] = (),
        allow_hand: tuple[str, ...] = (),
        hand_support_m: float = HAND_SUPPORT_CLEARANCE_M,
        hand_clear: dict[str, float] | None = None,
    ) -> None:
        self.geo, self.side, self.lo, self.hi = geo, side, limits[0], limits[1]
        self.allow, self.allow_hand = allow, allow_hand
        self.hand_support_m = hand_support_m
        self.hand_clear = hand_clear  # session 12: entities the hand keeps a distance from (Geometry.hits)
        self.checks = 0
        self.last_hit = ""

    def valid(self, q: FloatArray) -> bool:
        self.checks += 1
        if np.any(q < self.lo - 1e-9) or np.any(q > self.hi + 1e-9):
            self.last_hit = "joint limits"
            return False
        self.geo.set_arm(self.side, q)
        h = self.geo.hits(self.side, allow=self.allow, allow_hand=self.allow_hand, stop_at_first=True,
                          hand_support_m=self.hand_support_m, hand_clear=self.hand_clear)  # fmt: skip
        if h:
            self.last_hit = f"{h[0].arm_body} touches {h[0].other}"
            return False
        return True

    def segment(self, a: FloatArray, b: FloatArray) -> bool:
        n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / CHECK_STEP_RAD)))
        return all(self.valid(a + (b - a) * (k / n)) for k in range(1, n + 1))

    def path(self, P: FloatArray) -> bool:
        return self.valid(P[0]) and all(self.segment(P[i], P[i + 1]) for i in range(len(P) - 1))


# ---------------------------------------------------------------------------------------------- leaving a start
# A start can already be inside the clearance of a declared entity: the table-side start (the resting arms at a table:
# the model's elbow 3 cm into the table top's edge, the hands under it - the real elbow is not inside the real table;
# the declared top and the padded collision shapes overlap by a few cm). Session 9 refused every goal from there
# (api / compat: skill.table_ready from the resting arms). The rule (session 10): from such a start the arm may LEAVE -
# along a sequentially checked path every pose (a) touches nothing the start did not already touch (no new pair (arm
# body, entity), no self / other-arm / floor contact) and (b) no pair gets closer than at the previous pose (signed
# distance never decreasing, ESCAPE_TOL_M of numeric noise) - until the arm is clear; from there the normal checks
# apply. The 1 cm clearance itself is unchanged: nothing is allowed to come closer, only to move away.
ESCAPE_TOL_M = 0.0005  # the contact distance's numeric noise between two checks 0.02 rad apart (< 0.1 mm seen)
ESCAPE_MAX_STEPS = 120  # steepest ascent: at most 120 x CHECK_STEP_RAD (2.4 rad of joint travel) to get clear
Pairs = dict[tuple[str, str], float]


def start_pairs(chk: Checker, q: FloatArray) -> tuple[Pairs, str]:
    """The pairs (arm body, `entity:<id>`) within the clearance at q -> their smallest signed distance; plus the first
    contact that is NOT leavable (self / the other arm / the floor: "" when none)."""
    geo = chk.geo
    geo.set_arm(chk.side, q)
    out: Pairs = {}
    for h in geo.hits(chk.side, allow=chk.allow, allow_hand=chk.allow_hand, hand_support_m=chk.hand_support_m):
        if h.kind != "world" or not h.other.startswith("entity:"):
            return out, f"{h.arm_body} touches {h.other}"
        k = (h.arm_body, h.other)
        out[k] = min(out.get(k, np.inf), h.dist)
    return out, ""


def escape_step(chk: Checker, prev: Pairs, q: FloatArray) -> Pairs | None:
    """The leaving rule at q given the previous pose's pairs: the pairs now (empty = clear), None = not allowed."""
    chk.checks += 1
    if np.any(q < chk.lo - 1e-9) or np.any(q > chk.hi + 1e-9):
        chk.last_hit = "joint limits"
        return None
    now, bad = start_pairs(chk, q)
    if bad:
        chk.last_hit = bad
        return None
    for k, d in now.items():
        if k not in prev:
            chk.last_hit = f"{k[0]} comes within the clearance of {k[1]}"
            return None
        if d < prev[k] - ESCAPE_TOL_M:
            chk.last_hit = f"{k[0]} gets closer to {k[1]} ({1000 * prev[k]:.1f} -> {1000 * d:.1f} mm)"
            return None
    return now


def escape_segment(chk: Checker, pairs: Pairs, a: FloatArray, b: FloatArray) -> Pairs | None:
    """The straight joint segment a -> b checked densely: the leaving rule while `pairs` (the start's) is not empty,
    then the normal validity. -> the pairs still inside the clearance at b (empty = clear), None = not allowed."""
    n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / CHECK_STEP_RAD)))
    prev = dict(pairs)
    for k in range(1, n + 1):
        q = a + (b - a) * (k / n)
        if prev:
            nxt = escape_step(chk, prev, q)
            if nxt is None:
                return None
            prev = nxt
        elif not chk.valid(q):
            return None
    return prev


def escape_ascent(chk: Checker, q0: FloatArray, pairs: Pairs, *, max_steps: int = ESCAPE_MAX_STEPS) -> PathResult:
    """Leave the start by steepest ascent of the smallest pair distance (finite differences, steps of CHECK_STEP_RAD
    in joint space, every pose under the leaving rule) until clear -> the path [q0, ..., q_clear] (decimated to its
    corners) or a refusal with what blocked it. Generic: no direction is assumed."""
    q = np.asarray(q0, dtype=np.float64).copy()
    prev = dict(pairs)
    path = [q.copy()]
    for _ in range(max_steps):
        if not prev:
            return PathResult(True, _corners(np.array(path)), "escape", "", chk.checks)
        f0 = min(prev.values())
        g = np.zeros(5)
        for j in range(5):
            dq = np.zeros(5)
            dq[j] = 0.01
            qj = np.clip(q + dq, chk.lo, chk.hi)
            pj, bad = start_pairs(chk, qj)
            if bad or float(qj[j] - q[j]) <= 0.0:
                continue
            # a pair that leaves the clearance reads as the clearance (it is at least that far)
            fj = min((pj.get(k, chk.geo.clearance_m) for k in prev), default=chk.geo.clearance_m)
            g[j] = (fj - f0) / float(qj[j] - q[j])
        if float(np.max(np.abs(g))) < 1e-6:
            return PathResult(False, np.array(path), "escape", "no joint direction moves the arm away", chk.checks,
                              chk.last_hit)  # fmt: skip
        step = g / float(np.max(np.abs(g))) * CHECK_STEP_RAD
        qn = np.clip(q + step, chk.lo, chk.hi)
        nxt = escape_step(chk, prev, qn)
        if nxt is None:
            return PathResult(False, np.array(path), "escape", f"leaving the start: {chk.last_hit}", chk.checks,
                              chk.last_hit)  # fmt: skip
        q, prev = qn, nxt
        path.append(q.copy())
    return PathResult(False, np.array(path), "escape", f"not clear after {max_steps} steps", chk.checks, chk.last_hit)


def _corners(P: FloatArray, tol: float = 1e-3) -> FloatArray:
    """Drop the waypoints of a dense path that lie on the straight joint segment of their neighbours (the escape's
    steps keep their directions for long stretches); the result only feeds the timing (the path was checked dense)."""
    if len(P) <= 2:
        return P
    keep = [0]
    for i in range(1, len(P) - 1):
        a, b = P[keep[-1]], P[i + 1]
        s = float(np.dot(P[i] - a, b - a) / max(float(np.dot(b - a, b - a)), 1e-12))
        if float(np.max(np.abs(a + s * (b - a) - P[i]))) > tol:
            keep.append(i)
    keep.append(len(P) - 1)
    return P[keep]


def joint_line(chk: Checker, q0: FloatArray, q1: FloatArray) -> PathResult:
    ok = chk.segment(q0, q1)
    return PathResult(ok, np.array([q0, q1]), "joint", "" if ok else "blocked", chk.checks, "" if ok else chk.last_hit)


def cartesian_line(
    chk: Checker,
    q0: FloatArray,
    p1: FloatArray,
    *,
    point: str = "pinch",
    approach: FloatArray | None = None,
    hold_approach: bool = True,
    step_m: float = LINE_STEP_M,
) -> PathResult:
    """The hand point on the straight segment from where it is (at q0) to p1 (the approach held / interpolated; when
    that fails for want of reach, the same line once more with the approach free)."""
    r = _cartesian_line(chk, q0, p1, point=point, approach=approach, hold_approach=hold_approach, step_m=step_m)
    if not r.ok and approach is None and hold_approach:
        r2 = _cartesian_line(chk, q0, p1, point=point, approach=None, hold_approach=False, step_m=step_m)
        if r2.ok:
            return r2
    return r


def _cartesian_line(
    chk: Checker,
    q0: FloatArray,
    p1: FloatArray,
    *,
    point: str,
    approach: FloatArray | None,
    hold_approach: bool,
    step_m: float,
) -> PathResult:
    geo, side = chk.geo, chk.side
    geo.set_arm(side, q0)
    p0 = geo.point(side, point)
    a0, _ = geo.hand_axes(side)
    a_goal = a0 if approach is None else np.asarray(approach, dtype=np.float64)
    L = float(np.linalg.norm(p1 - p0))
    qs = [np.asarray(q0, dtype=np.float64)]
    q = qs[0]
    s = 0.0
    ds = 1.0 if L <= 0.0 else min(1.0, step_m / L)
    ds_min = 1.0 if L <= 0.0 else min(1.0, MIN_STEP_M / L)
    while s < 1.0 - 1e-12:
        s1 = min(1.0, s + ds)
        a = None
        if approach is not None or hold_approach:
            v = (1 - s1) * a0 + s1 * a_goal
            a = v / np.linalg.norm(v)
        r = ik.solve_from(geo, side, p0 + s1 * (p1 - p0), q, point=point, approach=a, limits=(chk.lo, chk.hi), iters=25)
        if not r.ok:
            return PathResult(False, np.array(qs), "line", f"the line leaves the arm's reach at {s1:.0%}", chk.checks)
        if float(np.max(np.abs(r.q - q))) > JUMP_RAD:
            # near the edge of the workspace (the arm hanging nearly straight: ~0.15 rad per cm at the elbow) a
            # 1 cm step needs a big joint step without being a jump: halve the step down to MIN_STEP_M first
            if s1 - s > ds_min * 1.0001:
                ds = max(ds / 2.0, ds_min)
                continue
            return PathResult(False, np.array(qs), "line", f"the line needs a joint jump at {s1:.0%}", chk.checks)
        if not chk.segment(q, r.q):
            return PathResult(False, np.array(qs), "line", f"blocked at {s1:.0%}", chk.checks, chk.last_hit)
        q = r.q
        qs.append(q)
        s = s1
        ds = 1.0 if L <= 0.0 else min(1.0, step_m / L)
    return PathResult(True, np.array(qs), "line", "", chk.checks)


def rrt_connect(
    chk: Checker,
    q0: FloatArray,
    q1: FloatArray,
    *,
    seed: int = 0,
    max_iters: int = RRT_MAX_ITERS,
    step: float = RRT_STEP_RAD,
) -> PathResult:
    """Bidirectional RRT in the 5-D joint box; then shortcut smoothing. Start and goal must be valid."""
    if not chk.valid(q0):
        return PathResult(False, np.array([q0]), "rrt", "the start pose is blocked", chk.checks, chk.last_hit)
    if not chk.valid(q1):
        return PathResult(False, np.array([q0]), "rrt", "the goal pose is blocked", chk.checks, chk.last_hit)
    if chk.segment(q0, q1):
        return PathResult(True, np.array([q0, q1]), "joint", "", chk.checks)
    rng = np.random.default_rng(seed)
    lo, hi = chk.lo, chk.hi
    Ta: list[tuple[FloatArray, int]] = [(np.asarray(q0, dtype=np.float64), -1)]
    Tb: list[tuple[FloatArray, int]] = [(np.asarray(q1, dtype=np.float64), -1)]
    a_is_start = True

    def nearest(T: list[tuple[FloatArray, int]], q: FloatArray) -> int:
        X = np.array([n[0] for n in T])
        return int(np.argmin(np.max(np.abs(X - q), axis=1)))

    def extend(T: list[tuple[FloatArray, int]], q: FloatArray) -> tuple[str, int]:
        i = nearest(T, q)
        qn = T[i][0]
        d = q - qn
        L = float(np.max(np.abs(d)))
        qnew = q if L <= step else qn + d * (step / L)
        if chk.segment(qn, qnew):
            T.append((qnew, i))
            return ("reached" if L <= step else "advanced"), len(T) - 1
        return "trapped", -1

    def connect(T: list[tuple[FloatArray, int]], q: FloatArray) -> tuple[str, int]:
        while True:
            s, k = extend(T, q)
            if s != "advanced":
                return s, k

    def branch(T: list[tuple[FloatArray, int]], k: int) -> list[FloatArray]:
        out = []
        while k >= 0:
            out.append(T[k][0])
            k = T[k][1]
        return out[::-1]

    for _ in range(max_iters):
        q_rand = (Tb[0][0] if rng.random() < GOAL_BIAS else rng.uniform(lo, hi)).copy()
        s, ka = extend(Ta, q_rand)
        if s != "trapped":
            s2, kb = connect(Tb, Ta[ka][0])
            if s2 == "reached":
                pa, pb = branch(Ta, ka), branch(Tb, kb)
                path = pa + pb[::-1][1:]
                if not a_is_start:
                    path = path[::-1]
                P = shortcut(chk, np.array(path), seed=seed)
                return PathResult(True, P, "rrt", "", chk.checks)
        Ta, Tb = Tb, Ta
        a_is_start = not a_is_start
    return PathResult(False, np.array([q0]), "rrt", f"no path found in {max_iters} iterations", chk.checks)


def shortcut(chk: Checker, P: FloatArray, *, seed: int = 0, rounds: int = 60) -> FloatArray:
    """Random shortcuts: try to replace P[i..j] by the straight joint segment P[i] -> P[j]."""
    rng = np.random.default_rng(seed + 1)
    Q = [p for p in P]
    for _ in range(rounds):
        if len(Q) <= 2:
            break
        i, j = sorted(rng.choice(len(Q), size=2, replace=False))
        if j - i < 2:
            continue
        if chk.segment(Q[i], Q[j]):
            Q = Q[: i + 1] + Q[j:]
    return np.array(Q)


def decimate(
    chk: Checker, P: FloatArray, min_step: float = 0.12, point: str = "pinch", tol_m: float = 0.003
) -> FloatArray:
    """Fewer knots for a hand line (1 cm IK steps -> knots >= min_step rad apart): closely spaced knots force a high
    jerk on the timing (session 9 probe: 6.4 s -> ~3 s for a 30 cm lift). Kept only when the joint interpolation
    between the kept knots stays clear (dense checks) and the hand stays within tol_m of the dropped waypoints."""
    if len(P) <= 2:
        return P
    keep = [0]
    for i in range(1, len(P) - 1):
        if float(np.max(np.abs(P[i] - P[keep[-1]]))) >= min_step:
            keep.append(i)
    keep.append(len(P) - 1)
    Q = P[keep]
    geo, side = chk.geo, chk.side
    for a, b in itertools.pairwise(keep):
        for i in range(a + 1, b):  # the dropped waypoint vs the joint interpolation at the same fraction
            s = (i - a) / (b - a)
            geo.set_arm(side, P[i])
            p_line = geo.point(side, point)
            geo.set_arm(side, P[a] + s * (P[b] - P[a]))
            if float(np.linalg.norm(geo.point(side, point) - p_line)) > tol_m:
                return P
    return Q if chk.path(Q) else P


def densify(P: FloatArray, max_step: float = 0.3) -> FloatArray:
    """Insert waypoints so no joint moves more than max_step between two (smoother timing of long joint segments)."""
    out = [P[0]]
    for i in range(len(P) - 1):
        n = max(1, int(np.ceil(float(np.max(np.abs(P[i + 1] - P[i]))) / max_step)))
        out += [P[i] + (P[i + 1] - P[i]) * (k / n) for k in range(1, n + 1)]
    return np.array(out)


Validity = Callable[[FloatArray], bool]
