"""Inverse kinematics for one 5-DoF R1 arm (DESIGN.md §7.2 step 1).

Ported from v2 arms.py ArmKin.solve / _solve_from (git r21_try 2a4a165, arms.py:371-470): damped least squares on the
arm's 5 joints, the hand POINT first (pinch / palm / a tip / the fingertips' mean / the wrist), the approach (the
fingers' direction) as a secondary task weighed against it (with 5 joints position + a full orientation are 6
constraints: v2 7 Oct "both exact are 6 constraints on 5 joints"), an optional palm normal as a third, weaker task;
joint limits = the profile's soft limits intersected with the balance policy's arm envelope (v2 controller
_arm_envelope: the arm angles the policy was trained with). Several seeds (v2's three standard ones, mirrored for the
right arm) after the current pose; the best by position error, then the orientation error, then the travel.
Uses the one geometry engine (r1mc3.world.Geometry) for every forward-kinematics evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from r1sim import joints as J

from ..world.geometry import ARM_JOINTS, Geometry

FloatArray = npt.NDArray[np.float64]

SEEDS_LEFT = (
    np.array([-0.9, 0.15, 0.0, 0.6, 0.0]),
    np.array([-0.5, 0.3, 0.3, 1.2, 0.0]),
    np.array([-1.3, 0.1, -0.2, 0.3, 0.0]),
)  # v2 arms.py:371-372
MIRROR = np.array([1.0, -1.0, -1.0, 1.0, -1.0])  # left arm joints -> right arm joints (v2 handposes.MIRROR_Q)
POS_TOL_M = 0.002
APPROACH_W = 1.0  # the orientation tasks are secondary (null space): their weights only rank them
NORMAL_W = 0.3
LAMBDA = 0.02


@dataclass(frozen=True)
class IKResult:
    ok: bool
    q: FloatArray
    pos_err_m: float
    approach_err_deg: float
    normal_err_deg: float
    iters: int


ARM_KEYS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
MIRRORED = ("shoulder_roll", "shoulder_yaw", "wrist_roll")


def policy_envelope(arm_ranges_left: dict[str, tuple[float, float]] | str, side: str) -> tuple[FloatArray, FloatArray]:
    """The arm angles the balance policy was trained with (its ONNX metadata `arm_ranges_left`, left-arm convention)
    -> (lo, hi) for `side` (the right arm mirrored: roll / yaw / wrist roll negated, v2 controller._arm_envelope).
    Planned paths under the balance policy stay inside it (v2 5 Oct, sim: a hand taken out from under the table with
    the arm swung back to ~1.7 rad - never trained - leaned the robot 16 deg and stepped into the table)."""
    rng = json.loads(arm_ranges_left) if isinstance(arm_ranges_left, str) else arm_ranges_left
    lo, hi = [], []
    for k in ARM_KEYS:
        a, b = (float(v) for v in rng[k])
        if side == "right" and k in MIRRORED:
            a, b = -b, -a
        lo.append(a)
        hi.append(b)
    return np.array(lo), np.array(hi)


KNOT_MARGIN_RAD = 0.002  # planned knots stay this far INSIDE the soft limits: the executor refuses a knot past its
#                          q_min / q_max (= the soft limits) and a knot computed AT the limit can land a hair outside
#                          (api's lockstep test, 11 Oct: a transit knot at shoulder roll -0.177 -> kRejectKnots)


def arm_limits(side: str, envelope: tuple[FloatArray, FloatArray] | None = None) -> tuple[FloatArray, FloatArray]:
    """The soft joint limits of `side`'s arm (KNOT_MARGIN_RAD inside), intersected with the policy envelope."""
    idx = ARM_JOINTS[side]
    lo, hi = J.SOFT_MIN[idx] + KNOT_MARGIN_RAD, J.SOFT_MAX[idx] - KNOT_MARGIN_RAD
    if envelope is not None:
        lo, hi = np.maximum(lo, envelope[0]), np.minimum(hi, envelope[1])
    return lo, hi


def _angle_deg(a: FloatArray, b: FloatArray) -> float:
    c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _cross(a: FloatArray, b: FloatArray) -> FloatArray:
    """3-vector cross product without np.cross's axis handling (session 10: np.cross was ~20 % of a hover's IK time;
    the same arithmetic)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def solve_from(
    geo: Geometry,
    side: str,
    target: FloatArray,
    q0: FloatArray,
    *,
    point: str = "pinch",
    approach: FloatArray | None = None,
    normal: FloatArray | None = None,
    limits: tuple[FloatArray, FloatArray] | None = None,
    iters: int = 60,
) -> IKResult:
    lo, hi = limits if limits is not None else arm_limits(side)
    q = np.clip(np.asarray(q0, dtype=np.float64), lo, hi)
    tgt = np.asarray(target, dtype=np.float64)
    a_des = None if approach is None else np.asarray(approach, dtype=np.float64) / np.linalg.norm(approach)
    n_des = None if normal is None else np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    k = 0
    for _ in range(iters):
        k += 1
        geo.set_arm(side, q)
        e = tgt - geo.point(side, point)
        rows = [geo.point_jac(side, point)]
        errs = [e]
        if a_des is not None or n_des is not None:
            Jr = geo.palm_rot_jac(side)
            a, n = geo.hand_axes(side)
            if a_des is not None:
                rows.append(APPROACH_W * Jr)
                errs.append(APPROACH_W * _cross(a, a_des))
            if n_des is not None:
                rows.append(NORMAL_W * Jr)
                errs.append(NORMAL_W * _cross(n, n_des))
        # task priority (position first, v2's rule): the orientation tasks act in the position task's null space
        J1 = rows[0]
        J1p = J1.T @ np.linalg.solve(J1 @ J1.T + LAMBDA**2 * np.eye(3), np.eye(3))
        dq = J1p @ errs[0]
        if len(rows) > 1:
            N1 = np.eye(5) - J1p @ J1
            J2 = np.vstack(rows[1:])
            e2 = np.concatenate(errs[1:]) - J2 @ dq
            n2 = float(np.linalg.norm(e2))
            if n2 > 0.2:  # small orientation steps: the position task stays converged
                e2 *= 0.2 / n2
            J2n = J2 @ N1
            dq2 = N1 @ (J2n.T @ np.linalg.solve(J2n @ J2n.T + (5 * LAMBDA) ** 2 * np.eye(J2n.shape[0]), e2))
            s2 = float(np.max(np.abs(dq2)))
            if s2 > 0.1:
                dq2 *= 0.1 / s2
            dq = dq + dq2
        step = float(np.max(np.abs(dq)))
        if step > 0.3:
            dq *= 0.3 / step
        q = np.clip(q + dq, lo, hi)
        if np.linalg.norm(e) < POS_TOL_M * 0.25 and step < 1e-4:
            break
    if a_des is not None or n_des is not None:  # polish: the position task alone (the orientation steps' residue)
        for _ in range(15):
            geo.set_arm(side, q)
            e = tgt - geo.point(side, point)
            if float(np.linalg.norm(e)) < POS_TOL_M * 0.1:
                break
            J1 = geo.point_jac(side, point)
            dq = J1.T @ np.linalg.solve(J1 @ J1.T + LAMBDA**2 * np.eye(3), e)
            q = np.clip(q + dq, lo, hi)
    geo.set_arm(side, q)
    pe = float(np.linalg.norm(tgt - geo.point(side, point)))
    if pe > POS_TOL_M and (a_des is not None or n_des is not None):
        # the orientation preference pulled it away: position first (v2's rule), the orientation as it comes
        r = solve_from(geo, side, target, q0, point=point, limits=(lo, hi), iters=iters)
        a, n = geo.hand_axes(side)
        ae = _angle_deg(a, a_des) if a_des is not None else 0.0
        ne = _angle_deg(n, n_des) if n_des is not None else 0.0
        return IKResult(r.ok, r.q, r.pos_err_m, ae, ne, k + r.iters)
    a, n = geo.hand_axes(side)
    ae = _angle_deg(a, a_des) if a_des is not None else 0.0
    ne = _angle_deg(n, n_des) if n_des is not None else 0.0
    return IKResult(pe <= POS_TOL_M, q, pe, ae, ne, k)


def solve(
    geo: Geometry,
    side: str,
    target: FloatArray,
    *,
    q_init: FloatArray | None = None,
    point: str = "pinch",
    approach: FloatArray | None = None,
    normal: FloatArray | None = None,
    limits: tuple[FloatArray, FloatArray] | None = None,
    seeds: bool = True,
    near_weight: float = 0.01,
) -> IKResult:
    """Arm joints that put `point` of `side`'s hand at `target` (base frame). Tries q_init (default: the current arm
    pose of the geometry), then v2's seeds; returns the best (it may be ok=False: the caller refuses with the error)."""
    start = geo.arm_q(side) if q_init is None else np.asarray(q_init, dtype=np.float64)
    starts = [start]
    if seeds:
        g = np.ones(5) if side == "left" else MIRROR
        starts += [g * s0 for s0 in SEEDS_LEFT]
    best: tuple[float, IKResult] | None = None
    for q0 in starts:
        r = solve_from(geo, side, target, q0, point=point, approach=approach, normal=normal, limits=limits)
        score = r.pos_err_m + 0.002 * (r.approach_err_deg + 0.3 * r.normal_err_deg)
        score += near_weight * float(np.linalg.norm(r.q - start))
        if best is None or score < best[0]:
            best = (score, r)
        if r.ok and r.approach_err_deg < 5.0 and r.normal_err_deg < 10.0:
            break
    assert best is not None
    geo.set_arm(side, best[1].q)
    return best[1]
