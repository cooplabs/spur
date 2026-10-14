"""Static joint torques vs the model's gravity torque with the friction band taken out (B3, session 8).

B3's target (DESIGN §13.2): the static torques within +-10 % per joint over >= 20 poses with 0 / 0.5 kg. HW1's wrap-up
compared tau_est at quasi-static holds with the model's gravity torque and found the arm masses broadly right (elbow
slope 1.06, shoulder pitch 1.06, roll 1.37) but could not show +-10 %: a joint at rest under a position servo sits
ANYWHERE inside its friction band, and on the shoulders that band (~1 N m) is as large as a 0.5 kg payload's effect.
The band has a sign that the data DO carry: a joint that came into the pose moving up stops short of its target, the
servo still pushing up - it rests on the upper edge of the band (tau = g + f, 0 < f <= ~Fc); approached from above, on
the lower edge. So with the approach direction d = +-1 of every hold (from the motion before it):

    tau = s g + F d + c

s = the model's mass scale for that joint (1 = right), F = the friction the holds rest on (a fraction of the sliding
friction), c = a constant offset (a cable, a tau_est zero). F and c separate only when both directions occur. The per
pose check uses the MODEL's masses (s = 1): e = tau - c - F d - g, within tol |g| for every pose with |g| >= min_g.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass
class StaticFit:
    n: int  # poses used (finite, |g| >= min_g)
    n_up: int  # approached moving + / -
    n_down: int
    slope: float  # s: the measured / model gravity scale (from the fit with F and c)
    slope_se: float
    friction: float  # F, N m (nan when no direction was known)
    friction_se: float
    offset: float  # c, N m (0 when not separable: one direction only)
    median_rel_error: float  # median |e| / |g| with the model's masses (s = 1)
    within_tol: float  # fraction of poses with |e| <= tol |g|
    slope_plain: float  # session-5-style slope with an offset only (tau = s g + c): the wrap-up's number
    shown: bool  # n >= min_poses and within_tol >= min_fraction
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_static(
    g: npt.ArrayLike,
    tau: npt.ArrayLike,
    direction: npt.ArrayLike,
    *,
    min_g: float = 1.0,
    tol: float = 0.10,
    min_poses: int = 20,
    min_fraction: float = 0.9,
) -> StaticFit:
    """One joint: the model's gravity torque g, the measured static torque tau and the approach direction (+1 / -1;
    0 = unknown) per pose. See the module doc."""
    g_, t_, d_ = (np.asarray(x, dtype=np.float64) for x in (g, tau, direction))
    ok = np.isfinite(g_) & np.isfinite(t_) & (np.abs(g_) >= min_g)
    g_, t_, d_ = g_[ok], t_[ok], np.sign(d_[ok])
    notes: list[str] = []
    n = int(g_.size)
    if n < 3:
        nan = float("nan")
        return StaticFit(n, 0, 0, nan, nan, nan, nan, 0.0, nan, 0.0, nan, False, [f"{n} poses with |g| >= {min_g}"])
    n_up, n_dn = int((d_ > 0).sum()), int((d_ < 0).sum())
    a_plain = np.column_stack([g_, np.ones(n)])
    s_plain = float(np.linalg.lstsq(a_plain, t_, rcond=None)[0][0])
    cols = [g_]
    if n_up + n_dn >= 3:
        cols.append(d_)
    both = n_up >= 2 and n_dn >= 2
    if both:
        cols.append(np.ones(n))
    elif n_up + n_dn:
        notes.append("one approach direction only: the friction and a constant offset are not separable (F holds both)")
    else:
        notes.append("no approach direction known: no friction term")
    a = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(a, t_, rcond=None)
    res = t_ - a @ coef
    s2 = float(res @ res) / max(1, n - a.shape[1])
    cov = s2 * np.linalg.pinv(a.T @ a)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    f = float(coef[1]) if len(cols) > 1 else float("nan")
    f_se = float(se[1]) if len(cols) > 1 else float("nan")
    c = float(coef[2]) if both else 0.0
    # the per-pose check with the MODEL's masses (s = 1), the friction / offset as fitted with them
    a1 = np.column_stack(cols[1:]) if len(cols) > 1 else np.zeros((n, 0))
    k1 = np.linalg.lstsq(a1, t_ - g_, rcond=None)[0] if a1.shape[1] else np.zeros(0)
    e = t_ - g_ - (a1 @ k1 if a1.shape[1] else 0.0)
    rel = np.abs(e) / np.abs(g_)
    within = float(np.mean(rel <= tol))
    shown = n >= min_poses and within >= min_fraction
    if n < min_poses:
        notes.append(f"{n} poses < {min_poses}: +-{tol:.0%} cannot be shown")
    return StaticFit(n, n_up, n_dn, float(coef[0]), float(se[0]), f, f_se, c, float(np.median(rel)), within, s_plain,
                     shown, notes)  # fmt: skip
