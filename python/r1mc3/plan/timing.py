"""Time parameterisation of a joint path (DESIGN.md §7.2 step 3) into the executor's TRAJECTORY knots.

The executor (EXECUTOR.md) plays a quintic Hermite through knots (t, q, dq, ddq): position, velocity and acceleration
match at every knot. So the planner's job is to pick knot times and velocities such that the quintic segments stay
inside the joint velocity / acceleration limits - and to PROVE it on the result:
  1. knot times from the path geometry: each segment's duration = max over the joints of a trapezoidal minimum time
     (|dq_j| with v_max and a_max), the first / last segments stretched for the start / stop from rest;
  2. knot velocities = the central difference bounded Fritsch-Carlson style (zero at the ends, zero where the
     direction reverses: no overshoot past a waypoint), knot accelerations = the central difference of those (zero at
     the ends);
  3. the quintic segments are sampled densely; if any |dq| > v_max or |ddq| > a_max or |dddq| > j_max, every time is
     scaled up and the check repeats (velocities / accelerations scale as 1/s, 1/s^2).
No new dependency (Ruckig would be the jerk-optimal alternative; this keeps the robot's wheel set unchanged).
The result: `Timed(t, q, dq, ddq, v_peak, a_peak, j_peak)` - the peaks as sampled, for the plan's report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

V_MAX = 1.0  # rad/s: v2 planned at <= 1 rad/s (EXECUTOR.md; the executor's own cap dq_max 2 rad/s)
A_MAX = 4.0  # rad/s^2: well inside the executor's ddq_stop 20 rad/s^2
J_MAX = 60.0  # rad/s^3
SAMPLES_PER_SEGMENT = 24


@dataclass(frozen=True)
class Timed:
    t: FloatArray  # (K,)
    q: FloatArray  # (K, n)
    dq: FloatArray
    ddq: FloatArray
    v_peak: float
    a_peak: float
    j_peak: float

    @property
    def duration(self) -> float:
        return float(self.t[-1])


def _seg_time(dq: FloatArray, v: float, a: float) -> float:
    d = float(np.max(np.abs(dq)))
    if d <= 0.0:
        return 0.0
    if d <= v * v / a:  # triangular
        return 2.0 * float(np.sqrt(d / a))
    return d / v + v / a


def _hermite5(
    p0: FloatArray,
    v0: FloatArray,
    a0: FloatArray,
    p1: FloatArray,
    v1: FloatArray,
    a1: FloatArray,
    T: float,
    s: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Quintic Hermite on [0, T] at times s (S,) -> q, dq, ddq, dddq (S, n)."""
    Tp2, Tp3, Tp4, Tp5 = T * T, T**3, T**4, T**5
    c0, c1, c2 = p0, v0, a0 / 2.0
    c3 = (20 * (p1 - p0) - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * Tp2) / (2 * Tp3)
    c4 = (30 * (p0 - p1) + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * Tp2) / (2 * Tp4)
    c5 = (12 * (p1 - p0) - 6 * (v1 + v0) * T - (a0 - a1) * Tp2) / (2 * Tp5)
    t = s[:, None]
    q = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    dq = c1 + 2 * c2 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
    ddq = 2 * c2 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
    dddq = 6 * c3 + 24 * c4 * t + 60 * c5 * t**2
    return q, dq, ddq, dddq


def sample(tm: Timed, dt: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The executor's curve sampled every dt: (times, q, dq) - for tests and the result's prediction."""
    ts = np.arange(0.0, tm.duration + 1e-12, dt)
    out_q, out_dq = [], []
    for t in ts:
        i = int(np.clip(np.searchsorted(tm.t, t, side="right") - 1, 0, len(tm.t) - 2))
        T = tm.t[i + 1] - tm.t[i]
        q, dq, _, _ = _hermite5(tm.q[i], tm.dq[i], tm.ddq[i], tm.q[i + 1], tm.dq[i + 1], tm.ddq[i + 1], T,
                                np.array([t - tm.t[i]]))  # fmt: skip
        out_q.append(q[0])
        out_dq.append(dq[0])
    return ts, np.array(out_q), np.array(out_dq)


def _seg_peaks(t: FloatArray, q: FloatArray, dq: FloatArray, ddq: FloatArray) -> FloatArray:
    """(K-1, 3): per segment the sampled peak |dq|, |ddq|, |dddq| over all joints."""
    out = np.zeros((len(t) - 1, 3))
    s = np.linspace(0.0, 1.0, SAMPLES_PER_SEGMENT)
    for i in range(len(t) - 1):
        T = float(t[i + 1] - t[i])
        _, v, a, j = _hermite5(q[i], dq[i], ddq[i], q[i + 1], dq[i + 1], ddq[i + 1], T, s * T)
        out[i] = (np.max(np.abs(v)), np.max(np.abs(a)), np.max(np.abs(j)))
    return out


def _knot_accels(t: FloatArray, dq: FloatArray) -> FloatArray:
    """The central difference of the knot velocities (0 at the ends: from / to rest): a smooth acceleration through
    closely spaced knots (with 0 at every knot the acceleration would swing up and down inside each short segment -
    a jerk ~10x higher on a 1 cm hand line, session 9 probe)."""
    ddq = np.zeros_like(dq)
    if len(t) > 2:
        ddq[1:-1] = (dq[2:] - dq[:-2]) / (t[2:] - t[:-2])[:, None]
    return ddq


def _knot_velocities(P: FloatArray, dur: FloatArray, v: float) -> FloatArray:
    K, n = P.shape
    dq = np.zeros((K, n))
    for i in range(1, K - 1):
        sl = (P[i] - P[i - 1]) / dur[i - 1]
        sr = (P[i + 1] - P[i]) / dur[i]
        same = (np.sign(sl) == np.sign(sr)) & (sl != 0.0)
        # the central difference, bounded by 3x the smaller slope (Fritsch-Carlson: a monotone segment stays
        # monotone, no overshoot past a waypoint) and v; 0 where the direction reverses
        c = (P[i + 1] - P[i - 1]) / (dur[i - 1] + dur[i])
        m = np.minimum(np.minimum(np.abs(c), 3.0 * np.minimum(np.abs(sl), np.abs(sr))), v)
        dq[i] = np.where(same, np.sign(sl) * m, 0.0)
    return dq


def time_path(
    path: FloatArray,
    *,
    v_max: float = V_MAX,
    a_max: float = A_MAX,
    j_max: float = J_MAX,
    speed: float = 1.0,
    min_segment_s: float = 0.02,
) -> Timed:
    """Waypoints (K, n), K >= 2 -> knots that start and end at rest, every sampled limit met (speed in (0, 1])."""
    P = np.asarray(path, dtype=np.float64)
    # drop repeated waypoints (zero-length segments)
    keep = [0] + [i for i in range(1, len(P)) if np.max(np.abs(P[i] - P[i - 1])) > 1e-9]
    P = P[keep]
    if len(P) == 1:
        P = np.vstack([P, P])
    sp = float(np.clip(speed, 0.05, 1.0))
    v, a, jm = v_max * sp, a_max * sp, j_max * sp
    K = P.shape[0]
    # cruise times inside the path, a ramp from / to rest at the ends (refined per segment below)
    dur = np.array([max(float(np.max(np.abs(P[i + 1] - P[i]))) / v, min_segment_s) for i in range(K - 1)])
    dur[0] = max(dur[0], _seg_time(P[1] - P[0], v, a))
    dur[-1] = max(dur[-1], _seg_time(P[-1] - P[-2], v, a))
    if K == 2:
        dur[0] = max(dur[0], 1.875 * float(np.max(np.abs(P[1] - P[0]))) / v)  # a quintic's peak speed = 1.875 d / T
    for _ in range(80):
        t = np.concatenate([[0.0], np.cumsum(dur)])
        dq = _knot_velocities(P, dur, v)
        ddq = _knot_accels(t, dq)
        pk = _seg_peaks(t, P, dq, ddq)
        f = np.maximum.reduce([pk[:, 0] / v, np.sqrt(pk[:, 1] / a), np.cbrt(pk[:, 2] / jm), np.ones(K - 1)])
        if float(np.max(f)) <= 1.0 + 1e-6:
            return Timed(t, P, dq, ddq, float(pk[:, 0].max()), float(pk[:, 1].max()), float(pk[:, 2].max()))
        # stretch the offending segments (and their neighbours a little: the knot velocities couple them)
        g = np.minimum(f * 1.02, 2.0)
        g[1:] = np.maximum(g[1:], 1.0 + 0.3 * (g[:-1] - 1.0))
        g[:-1] = np.maximum(g[:-1], 1.0 + 0.3 * (g[1:] - 1.0))
        dur = dur * g
    t = np.concatenate([[0.0], np.cumsum(dur)])
    dq = _knot_velocities(P, dur, v)
    ddq = _knot_accels(t, dq)
    pk = _seg_peaks(t, P, dq, ddq)
    return Timed(t, P, dq, ddq, float(pk[:, 0].max()), float(pk[:, 1].max()), float(pk[:, 2].max()))
