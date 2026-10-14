"""The Dex3 pads' gain and footprint from presses with known weights (A10 rows 2-3) - PREPARED for HW1, not in sim.

What is open (S3 / S4): the simulated pads feel light fingertip touches the real ones do not, and a small round object's
wrap touches the fingertips at the pad edges; the pads' positions came from Unitree's meshes
(`tools/hands/pad_geometry.py`: the proximal pads a raised block 20.6-43.3 mm along the finger), but WHERE along the
finger a pad stops feeling and how many raw units it reads per newton were never measured. The `pads="real"` layout
loses the thin-block pinch (0/10) because of exactly these two unknowns.

**Protocol (HW1, the hand in Unitree's / our low-level hold, palm up, fingers open; docs/hardware/HW1_FITS.md).**
A small probe (flat 4 mm tip) carrying a kitchen-scale-weighed mass rests on one pad at a measured position x (mm along
the finger from the finger link's proximal joint axis; marks on tape beside the finger), 4 s each with 4 s unloaded
between: (a) the footprint - one mass (200 g) stepped every 2 mm from ~8 mm before the pad's mesh start to ~8 mm past
its end; (b) the gain - 50 / 100 / 200 / 500 g at the pad's centre. Then (c) a thin-block pinch with the block's place
on each pad measured (photo + ruler), held 5 s - the check of (a)-(b) on a real grasp.

**Fit.** Per press: the pad's signal S = sum over its active cells of (reading - the cell's own baseline in the 1 s
before the press) [raw units]; y = S / F. Inside the footprint y = G (the gain, raw units per N), outside 0, the edges
smeared by the probe's width: y(x) = G s((x - x0) / w) s((x1 - x) / w) (s = the logistic). x0, x1, w on a grid, G by
least squares; the weights at the centre give the linearity (slope, intercept) and the detection floor. The
simulator's `pad_gain` = G x the cell area (its readings are Pa = N / area).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from .records import HandRecording, Press

FloatArray = npt.NDArray[np.float64]

N_PAD, N_SLOT = 9, 12
QUIET_S = 1.0  # baseline window before each press
SETTLE_S = 0.5  # skip after the weight is placed


@dataclass(frozen=True)
class PadGeom:
    """One pad along its finger: centre and half-length in mm from the finger link's origin (its proximal joint
    axis), along the finger (the pad's long axis, pointing distally)."""

    side: str
    k: int
    x_center_mm: float
    half_long_mm: float
    margin_mm: float
    slots: tuple[int, ...]
    cell_area_m2: float

    @property
    def footprint_mm(self) -> tuple[float, float]:
        e = self.half_long_mm + self.margin_mm
        return (self.x_center_mm - e, self.x_center_mm + e)


def _bound_sensor_model(seed: int = 0) -> tuple[Any, Any]:
    from .. import sensors as S
    from .stiction import hanging_world

    w = hanging_world()
    sm = S.SensorModel(S.SensorConfig.from_yaml(), seed=seed)
    sm.bind(w.model, w.index)
    sm.reset(seed)
    sm._start()
    return sm, w


def pad_geometry(side: str = "left", *, seed: int = 0) -> dict[int, PadGeom]:
    """The simulator's pads (the default layout) in the measurable frame: mm along each finger link from its joint."""
    import mujoco

    sm, w = _bound_sensor_model(seed)
    m = w.model
    out: dict[int, PadGeom] = {}
    c = sm.cfg
    for p in sm._hands[side].pads:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[p.geom])
        axis = R.reshape(3, 3)[:, p.long] * p.distal  # the pad's distal direction in its body frame
        x_c = float(np.dot(m.geom_pos[p.geom], axis)) * 1e3
        tip = len(p.slots) == 3
        out[p.k] = PadGeom(side, p.k, x_c, float(p.half[p.long]) * 1e3, c.pad_face_margin_m * 1e3, p.slots,
                           c.tip_cell_area_m2 if tip else c.pad_cell_area_m2)  # fmt: skip
    return out


# ------------------------------------------------------------------------------------------------ synthetic data
@dataclass(frozen=True)
class PadTruth:
    """Known pad parameters for synthetic presses: per pad a gain multiplier on the simulator's model and its
    footprint (half-length, mm)."""

    gain: Mapping[int, float]
    half_long_mm: Mapping[int, float]


def press_plan(geom: PadGeom, *, step_mm: float = 2.0, beyond_mm: float = 8.0, weight_g: float = 200.0,
               centre_weights_g: Sequence[float] = (50.0, 100.0, 200.0, 500.0), on_s: float = 4.0,
               off_s: float = 4.0, t0: float = 2.0) -> list[Press]:  # fmt: skip
    """The protocol's presses for one pad: the footprint sweep, then the weights at the centre."""
    lo, hi = geom.footprint_mm
    xs = np.arange(lo - beyond_mm, hi + beyond_mm + 1e-9, step_mm)
    out: list[Press] = []
    t = t0
    for x in xs:
        out.append(Press(geom.side, geom.k, float(round(x, 2)), weight_g, t, t + on_s))
        t += on_s + off_s
    for wg in centre_weights_g:
        out.append(Press(geom.side, geom.k, geom.x_center_mm, float(wg), t, t + on_s))
        t += on_s + off_s
    return out


def synthetic_presses(
    side: str,
    presses: Sequence[Press],
    truth: PadTruth,
    *,
    rate_hz: float = 100.0,
    seed: int = 0,
    tail_s: float = 2.0,
) -> HandRecording:
    """The hand's pressure readings through the simulator's pad model (r1sim.sensors: bilinear cell split, detection
    floor, saturation, per-cell offsets, noise, glitches) for point loads at the presses, with the pads' gain and
    footprint set to `truth`. The force placement is direct (a point load at x on the pad's mid-line), not a MuJoCo
    contact: the fit's inputs are (x, F, readings), which this reproduces exactly as the sensor model maps them."""
    from ..sensors import N_CELLS, SensorModel

    sm, _w = _bound_sensor_model(seed)
    geom = pad_geometry(side, seed=seed)
    pads = {p.k: p for p in sm._hands[side].pads}
    c = sm.cfg
    t_end = max(p.t_end for p in presses) + tail_s
    t = np.arange(0.0, t_end, 1.0 / rate_hz)
    rng = np.random.default_rng(seed)
    pressure = np.zeros((len(t), N_PAD, N_SLOT))
    q_open = np.zeros(7)
    for i, ti in enumerate(t):
        f_cells = np.zeros((N_PAD, N_SLOT))
        for pr in presses:
            if pr.side != side or not (pr.t_start <= ti < pr.t_end):
                continue
            g = geom[pr.pad]
            half = truth.half_long_mm.get(pr.pad, g.half_long_mm)
            pad = pads[pr.pad]
            u = pr.x_mm - g.x_center_mm  # mm along the finger from the pad centre
            if abs(u) > half + g.margin_mm:
                continue  # off the pad's face: the sensor model's _which_pad finds no pad
            p_t = replace(pad, half=pad.half.copy())
            p_t.half[pad.long] = half * 1e-3
            loc = np.zeros(3)
            loc[pad.long] = pad.distal * u * 1e-3
            f_cells[pr.pad] += pr.force_n * truth.gain.get(pr.pad, 1.0) * SensorModel._weights(p_t, loc)
        n_p = rng.normal(0.0, 1.0, N_CELLS)
        u_g, m_g = rng.random(N_PAD), rng.random(N_PAD)
        rep, _ideal = sm._pressure(side, f_cells, n_p, u_g, m_g, q_open)
        pressure[i] = rep
    _ = c
    fields = {"pressure": pressure}
    return HandRecording(side, t, fields)


# ------------------------------------------------------------------------------------------------ the fit
@dataclass
class PressSignal:
    press: Press
    signal: float  # raw units, summed over the pad's active cells
    per_n: float  # signal / F
    noise: float = float("nan")  # std of the summed cells over the baseline window (raw units)


@dataclass
class PadFit:
    side: str
    pad: int
    gain_raw_per_n: float  # G
    gain_se: float
    sim_pad_gain: float  # G x the cell area (the simulator's dimensionless pad_gain)
    x0_mm: float
    x1_mm: float
    edge_mm: float
    centre_slope_raw_per_n: float
    centre_intercept_raw: float
    floor_g: float  # the smallest weight that read above the noise (nan if every weight did)
    n_presses: int
    rms_per_n: float
    mesh_footprint_mm: tuple[float, float] | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _active(slots: Sequence[int]) -> FloatArray:
    return np.asarray(slots, dtype=np.int64)


def press_signals(
    rec: HandRecording, presses: Sequence[Press], slots: Mapping[int, Sequence[int]]
) -> list[PressSignal]:
    """S per press = sum over the pad's active cells of (median during the press - median in the QUIET_S before)."""
    P = rec["pressure"]
    out = []
    for pr in presses:
        if pr.side != rec.side:
            continue
        sl = _active(slots[pr.pad])
        before = (rec.t >= pr.t_start - QUIET_S) & (rec.t < pr.t_start)
        during = (rec.t >= pr.t_start + SETTLE_S) & (rec.t < pr.t_end)
        if before.sum() < 3 or during.sum() < 3:
            continue
        base = np.median(P[before][:, pr.pad, sl], axis=0)
        val = np.median(P[during][:, pr.pad, sl], axis=0)
        s = float(np.sum(val - base))
        noise = float(np.std(P[before][:, pr.pad, sl].sum(axis=1)))
        out.append(PressSignal(pr, s, s / pr.force_n, noise))
    return out


def _logistic(z: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))), dtype=np.float64)


def fit_pad(
    sigs: Sequence[PressSignal],
    *,
    cell_area_m2: float,
    mesh_footprint_mm: tuple[float, float] | None = None,
    grid_mm: float = 0.25,
    edges_mm: Sequence[float] = (0.2, 0.5, 1.0, 2.0, 3.0),
    noise_raw: float | None = None,
) -> PadFit:
    if not sigs:
        raise ValueError("no presses")
    pad, side = sigs[0].press.pad, sigs[0].press.side
    notes: list[str] = []
    # the footprint sweep = the most common weight; the centre series = the weights at one position
    weights = [s.press.weight_g for s in sigs]
    w_sweep = max(set(weights), key=weights.count)
    sweep = [s for s in sigs if s.press.weight_g == w_sweep]
    x = np.array([s.press.x_mm for s in sweep])
    y = np.array([s.per_n for s in sweep])
    best: tuple[float, float, float, float, float] | None = None
    lo, hi = x.min(), x.max()
    grid = np.arange(lo - 2.0, hi + 2.0 + 1e-9, grid_mm)
    for w in edges_mm:
        for i, x0 in enumerate(grid):
            s0 = _logistic((x - x0) / w)
            for x1 in grid[i + 4 :]:
                shape = s0 * _logistic((x1 - x) / w)
                den = float(shape @ shape)
                if den < 1e-9:
                    continue
                g = float(shape @ y) / den
                r = y - g * shape
                rss = float(r @ r)
                if best is None or rss < best[0]:
                    best = (rss, g, x0, float(x1), w)
    assert best is not None
    rss, g, x0, x1, w = best
    inside = (x > x0 + 2 * w) & (x < x1 - 2 * w)
    g_se = float(np.std(y[inside]) / np.sqrt(max(1, inside.sum()))) if inside.sum() > 1 else float("nan")
    if not inside.any():
        notes.append("no press fully inside the footprint")
    # centre series: linearity and floor
    xc = 0.5 * (x0 + x1)
    centre = [s for s in sigs if abs(s.press.x_mm - xc) <= 2.0]
    slope = intercept = float("nan")
    floor = float("nan")
    if len({s.press.weight_g for s in centre}) >= 2:
        F = np.array([s.press.force_n for s in centre])
        S = np.array([s.signal for s in centre])
        slope, intercept = (float(v) for v in np.polyfit(F, S, 1))
        # a weight "reads" when its signal clears 5 sigma of the summed cells' noise (a median over the press window
        # of ~350 samples has ~1/sqrt(350) of the per-sample noise; 5 per-sample sigma is conservative)
        nz = noise_raw if noise_raw is not None else float(np.nanmedian([s.noise for s in sigs]))
        thr = 5.0 * nz
        lows = sorted(s.press.weight_g for s in centre if s.signal < thr)
        if lows:
            floor = float(max(lows))
            notes.append(f"weights <= {floor:g} g read below the noise (the detection floor)")
    return PadFit(
        side=side, pad=pad, gain_raw_per_n=g, gain_se=g_se, sim_pad_gain=g * cell_area_m2, x0_mm=float(x0),
        x1_mm=float(x1), edge_mm=float(w), centre_slope_raw_per_n=slope, centre_intercept_raw=intercept,
        floor_g=floor, n_presses=len(sigs), rms_per_n=float(np.sqrt(rss / len(y))),
        mesh_footprint_mm=mesh_footprint_mm, notes=notes,
    )  # fmt: skip


def fit_hand(
    rec: HandRecording, presses: Sequence[Press], geom: Mapping[int, PadGeom] | None = None
) -> dict[int, PadFit]:
    """Every pad pressed in the recording."""
    geom = geom or pad_geometry(rec.side)
    slots = {k: g.slots for k, g in geom.items()}
    sigs = press_signals(rec, presses, slots)
    out: dict[int, PadFit] = {}
    for k in sorted({s.press.pad for s in sigs}):
        ks = [s for s in sigs if s.press.pad == k]
        out[k] = fit_pad(ks, cell_area_m2=geom[k].cell_area_m2, mesh_footprint_mm=geom[k].footprint_mm)
    return out


@dataclass
class PinchCheck:
    pad: int
    x_mm: float
    inside_footprint: bool
    force_from_pad_n: float  # S / G
    force_from_torque_n: float | None  # the finger's tau_est / lever arm, if given
    ratio: float | None


def pinch_check(
    rec: HandRecording, fit: PadFit, slots: Sequence[int], x_mm: float, window: tuple[float, float],
    *, baseline: tuple[float, float], torque_nm: float | None = None, lever_mm: float | None = None,
) -> PinchCheck:  # fmt: skip
    """The thin-block pinch: the force the fitted pad reads at the measured place vs the force the finger's torque
    implies (one contact on one link: F = tau / lever, lever = the contact's distance from the joint axis)."""
    P = rec["pressure"]
    sl = _active(slots)
    b = (rec.t >= baseline[0]) & (rec.t < baseline[1])
    d = (rec.t >= window[0]) & (rec.t < window[1])
    s = float(np.sum(np.median(P[d][:, fit.pad, sl], axis=0) - np.median(P[b][:, fit.pad, sl], axis=0)))
    f_pad = s / fit.gain_raw_per_n
    f_tau = None if torque_nm is None or not lever_mm else torque_nm / (lever_mm * 1e-3)
    return PinchCheck(fit.pad, x_mm, fit.x0_mm <= x_mm <= fit.x1_mm, f_pad, f_tau,
                      None if f_tau is None else f_pad / f_tau)  # fmt: skip
