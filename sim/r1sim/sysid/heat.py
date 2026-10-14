"""The Dex3 finger motors' heating from full-rate holds (A10's fast pole) - PREPARED for HW1, not fixed in sim.

What is known (S3 / S4, 8 Hz telemetry of S12 only): at a 0.9 N·m hold a finger motor's reported temperature rises
+22..+27 C in ~5 s and then creeps 0.12-0.14 C/s; at 0.45 N·m the creep is FASTER. The simulator's model
(r1sim.sensors: two parallel poles, input |tau|^1.5 of the APPLIED torque, ambient = the powered idle hand ~40 C)
cannot give both: no single-input two-pole model with fixed gains makes the smaller torque creep faster. Joule heating
(tau^2) plus a firmware derating of the current when hot would (S4: 58.1 C at 5 s vs 65 real) - but the 8 Hz logs
have no torque reading at the needed rate to tell. HW1 records holds at 0.9 and 0.45 N·m at full rate WITH the
motors' own torque estimate; this module fits several model STRUCTURES to both holds jointly and reports which one the
data supports (BIC), so the sim's thermal model is changed from data, not by choice.

Structures (per motor; T = ambient + D_fast + D_slow (+ the initial slow state decaying), tau_k dD_k/dt = -D_k + G_k u):
* `one_pole` - one pole, u = |tau|^p;
* `two_pole` - two poles, u = |tau|^p (p from a grid; the sim's current structure has p = 1.5);
* the INPUT: `cmd` = the commanded torque (what the hold asked for: kp (q_cmd - q) + kd (0 - dq) + tau_ff, or the
  set point), `meas` = the motor's tau_est (carries any derating: the current the firmware actually allowed).
The fit: for each (structure, input, p, time constants on a log grid) the response is linear in the gains, the
ambient offset and the initial slow-pole state -> exact least squares; the best per structure by RMS, the structures
compared by BIC over every sample of every hold (the temperatures are whole degrees: the quantisation noise
1/sqrt(12) C is the floor).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .records import HandRecording

FloatArray = npt.NDArray[np.float64]

FIT_RATE_HZ = 5.0  # the fit resamples to this (the fastest pole is seconds; temperatures are whole degrees)
TAU_FAST_GRID = np.geomspace(0.5, 40.0, 40)
TAU_SLOW_GRID = np.geomspace(40.0, 4000.0, 40)
TAU_ONE_GRID = np.geomspace(0.5, 4000.0, 80)
P_GRID = (1.0, 1.5, 2.0, 2.5)
STRUCTURES = ("one_pole", "two_pole")
INPUTS = ("cmd", "meas")


@dataclass(frozen=True)
class Hold:
    """One hold of one motor: time (s), the commanded and the measured torque (N·m), the reported temperature (C)."""

    t: FloatArray
    tau_cmd: FloatArray
    tau_meas: FloatArray
    temp: FloatArray
    label: str = ""


def holds_from_recording(
    rec: HandRecording, motor: int, windows: Sequence[tuple[float, float]], *, labels: Sequence[str] = ()
) -> list[Hold]:
    """Cut one motor's holds (each window = the hold AND the cooling after it) from a hands recording. The reported
    temperature = the hotter of the motor's two sensors (as the body's model reads it)."""
    temp2 = rec["temperature"]
    temp = temp2[:, motor].max(axis=1) if temp2.ndim == 3 else temp2[:, motor]
    tau_cmd = (
        rec["kp"][:, motor] * (rec["q_cmd"][:, motor] - rec["q"][:, motor])
        - rec["kd"][:, motor] * rec["dq"][:, motor]
        + rec["tau_ff"][:, motor]
    )
    out = []
    for i, (a, b) in enumerate(windows):
        m = (rec.t >= a) & (rec.t <= b)
        out.append(
            Hold(rec.t[m] - a, tau_cmd[m], rec["tau_est"][m, motor], temp[m], labels[i] if i < len(labels) else "")
        )
    return out


# ------------------------------------------------------------------------------------------------ synthetic truth
@dataclass(frozen=True)
class TruthModel:
    """A thermal truth for synthetic holds: two poles with input |tau_actual|^p; optional derating (the firmware
    scales the current down when hot: actual = cmd x clip(1 - k (T - T_d), floor, 1))."""

    ambient_c: float = 40.0
    g_fast: float = 20.5
    tau_fast_s: float = 3.0
    g_slow: float = 124.0
    tau_slow_s: float = 300.0
    p: float = 1.5
    derate_from_c: float | None = None
    derate_per_c: float = 0.0
    derate_floor: float = 0.0

    @classmethod
    def sim(cls) -> TruthModel:
        """The simulator's own hand model (r1sim.sensors.SensorConfig defaults, session 3)."""
        from ..sensors import SensorConfig

        c = SensorConfig()
        return cls(c.hand_ambient_c, c.hand_heat_gain_fast, c.hand_heat_tau_fast_s, c.hand_heat_gain_slow,
                   c.hand_heat_tau_slow_s, c.hand_heat_exponent)  # fmt: skip


def synthetic_hold(
    truth: TruthModel,
    tau_cmd: float,
    *,
    hold_s: float = 60.0,
    cool_s: float = 120.0,
    pre_s: float = 2.0,
    rate_hz: float = 100.0,
    tau_noise: float = 0.01,
    quant_c: float = 1.0,
    seed: int = 0,
) -> Hold:
    """One hold at `tau_cmd` (then released) through `truth`, sampled like the hand's state topic: the measured
    torque = the actual one + noise; the temperature reported in whole degrees (quant_c)."""
    rng = np.random.default_rng(seed)
    n = round((pre_s + hold_s + cool_s) * rate_hz) + 1
    t = np.arange(n) / rate_hz
    cmd = np.where((t >= pre_s) & (t < pre_s + hold_s), tau_cmd, 0.0)
    dt = 1.0 / rate_hz
    af, as_ = np.exp(-dt / truth.tau_fast_s), np.exp(-dt / truth.tau_slow_s)
    df = ds = 0.0
    act = np.zeros(n)
    temp = np.zeros(n)
    for i in range(n):
        T = truth.ambient_c + df + ds
        temp[i] = T
        scale = 1.0
        if truth.derate_from_c is not None and T > truth.derate_from_c:
            scale = float(np.clip(1.0 - truth.derate_per_c * (T - truth.derate_from_c), truth.derate_floor, 1.0))
        act[i] = cmd[i] * scale
        u = abs(act[i]) ** truth.p
        df = df * af + truth.g_fast * u * (1.0 - af)
        ds = ds * as_ + truth.g_slow * u * (1.0 - as_)
    meas = act + rng.normal(0.0, tau_noise, n)
    rep = np.round(temp / quant_c) * quant_c if quant_c > 0 else temp
    return Hold(t, cmd, meas, rep, f"{tau_cmd:g} N m")


# ------------------------------------------------------------------------------------------------ the fit
def _resample(h: Hold, rate: float) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    t = np.arange(0.0, h.t[-1] + 1e-9, 1.0 / rate)
    # torque: the mean over each output interval (a hold is piecewise constant at the hand's rate)
    edges = np.searchsorted(h.t, np.append(t, t[-1] + 1.0 / rate))
    cmd = np.array(
        [h.tau_cmd[a:b].mean() if b > a else h.tau_cmd[min(a, len(h.t) - 1)] for a, b in itertools.pairwise(edges)]
    )
    meas = np.array(
        [h.tau_meas[a:b].mean() if b > a else h.tau_meas[min(a, len(h.t) - 1)] for a, b in itertools.pairwise(edges)]
    )
    temp = np.interp(t, h.t, h.temp)
    return t, cmd, meas, temp


def _pole(u: FloatArray, tau: float, dt: float) -> FloatArray:
    """Unit-gain first-order response to a piecewise-constant input (exact), via an exponential-kernel convolution."""
    a = np.exp(-dt / tau)
    n = len(u)
    k = (1.0 - a) * a ** np.arange(n)
    y = np.convolve(u, k)[:n]  # sum_{j <= i} (1-a) a^(i-j) u[j]
    # the temperature at sample i has seen the input up to sample i - 1 (y[i] = a y[i-1] + (1-a) u[i-1])
    return np.asarray(np.concatenate([[0.0], y[:-1]]), dtype=np.float64)


@dataclass
class HeatFit:
    structure: str
    input: str
    p: float
    tau_fast_s: float
    tau_slow_s: float  # nan for one_pole
    g_fast: float
    g_slow: float
    ambient_c: list[float]  # per hold (the fitted offset)
    rms_c: float
    rms_per_hold: list[float]
    bic: float
    n: int
    k: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HeatReport:
    best: HeatFit
    candidates: list[HeatFit] = field(default_factory=list)
    holds: list[str] = field(default_factory=list)
    supported: str = ""  # plain words: which structure / input the data supports
    d_bic_input: float = float("inf")  # BIC of the best fit on the OTHER torque input minus the best (>= 10: decisive)

    def table(self) -> str:
        rows = ["| structure | input | p | tau_fast s | tau_slow s | G_fast | G_slow | RMS C | BIC |", "|" + "---|" * 9]
        for c in sorted(self.candidates, key=lambda c: c.bic):
            rows.append(
                f"| {c.structure} | {c.input} | {c.p:g} | {c.tau_fast_s:.2f} | {c.tau_slow_s:.0f} | {c.g_fast:.1f} | "
                f"{c.g_slow:.1f} | {c.rms_c:.3f} | {c.bic:.0f} |"
            )
        return "\n".join(rows)


def _fit_one(
    data: list[tuple[FloatArray, FloatArray, FloatArray]], structure: str, p: float, dt: float
) -> tuple[float, float, FloatArray, FloatArray, float]:
    """Best time constants for one (structure, p): returns tau_f, tau_s, coef, residuals, rss."""
    y = np.concatenate([d[2] for d in data])
    nh = len(data)
    sizes = [len(d[2]) for d in data]

    # per-hold regressors: ambient offset (1) and the initial slow state decaying exp(-t / tau_slow)
    def design(cols: list[FloatArray], tau_init: float) -> FloatArray:
        blocks = []
        off = 0
        A = np.zeros((len(y), len(cols) + 2 * nh))
        for i, (t, _u, _T) in enumerate(data):
            n = sizes[i]
            for c, col in enumerate(cols):
                A[off : off + n, c] = col[off : off + n]
            A[off : off + n, len(cols) + i] = 1.0
            A[off : off + n, len(cols) + nh + i] = np.exp(-t / tau_init)
            off += n
            blocks.append(n)
        return A

    us = [np.abs(d[1]) ** p for d in data]

    def resp(tau: float) -> FloatArray:
        return np.concatenate([_pole(u, tau, dt) for u in us])

    best: tuple[float, float, float, FloatArray, FloatArray] | None = None
    if structure == "one_pole":
        cache = {tau: resp(tau) for tau in TAU_ONE_GRID}
        for tau in TAU_ONE_GRID:
            A = design([cache[tau]], max(tau, 60.0))
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = y - A @ coef
            rss = float(r @ r)
            if best is None or rss < best[0]:
                best = (rss, tau, float("nan"), coef, r)
    else:
        cf = {tau: resp(tau) for tau in TAU_FAST_GRID}
        cs = {tau: resp(tau) for tau in TAU_SLOW_GRID}
        for tf in TAU_FAST_GRID:
            for ts in TAU_SLOW_GRID:
                A = design([cf[tf], cs[ts]], ts)
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                r = y - A @ coef
                rss = float(r @ r)
                if best is None or rss < best[0]:
                    best = (rss, tf, ts, coef, r)
    assert best is not None
    rss, tf, ts, coef, r = best
    return tf, ts, coef, r, rss


def fit_holds(
    holds: Sequence[Hold],
    *,
    structures: Sequence[str] = STRUCTURES,
    inputs: Sequence[str] = INPUTS,
    p_grid: Sequence[float] = P_GRID,
    rate_hz: float = FIT_RATE_HZ,
) -> HeatReport:
    """Fit every structure x input x p jointly over all holds (shared gains and time constants, a per-hold ambient
    offset and initial slow state); the best by BIC."""
    if not holds:
        raise ValueError("no holds")
    dt = 1.0 / rate_hz
    rs = [_resample(h, rate_hz) for h in holds]
    cands: list[HeatFit] = []
    for inp in inputs:
        data = [(t, cmd if inp == "cmd" else meas, temp) for t, cmd, meas, temp in rs]
        for st in structures:
            best: HeatFit | None = None
            for p in p_grid:
                tf, ts, coef, r, rss = _fit_one(data, st, p, dt)
                n = len(r)
                ng = 1 if st == "one_pole" else 2
                k = ng + (1 if st == "one_pole" else 2) + 1 + 2 * len(holds)  # gains, taus, p, per-hold offset + init
                bic = n * np.log(max(rss / n, 1e-12)) + k * np.log(n)
                off = 0
                per_hold = []
                for d in data:
                    m = len(d[2])
                    per_hold.append(float(np.sqrt(np.mean(r[off : off + m] ** 2))))
                    off += m
                fit = HeatFit(
                    st, inp, float(p), float(tf), float(ts),
                    float(coef[0]), float(coef[1]) if ng == 2 else 0.0,
                    [float(c) for c in coef[ng : ng + len(holds)]],
                    float(np.sqrt(rss / n)), per_hold, float(bic), n, k,
                )  # fmt: skip
                if best is None or fit.bic < best.bic:
                    best = fit
            assert best is not None
            cands.append(best)
    best_all = min(cands, key=lambda c: c.bic)
    other = [c for c in cands if c.input != best_all.input and c.structure == best_all.structure]
    d_input = (other[0].bic - best_all.bic) if other else float("inf")
    which = "MEASURED (derated) torque" if best_all.input == "meas" else "commanded torque"
    words = (
        f"{best_all.structure.replace('_', ' ')} on the {which}"
        f", u = |tau|^{best_all.p:g}: tau_fast {best_all.tau_fast_s:.2f} s"
        + (f", tau_slow {best_all.tau_slow_s:.0f} s" if best_all.structure == "two_pole" else "")
        + f"; RMS {best_all.rms_c:.2f} C over {len(holds)} holds"
        + (
            "; commanded vs measured torque indistinguishable (dBIC < 10: no derating in these holds)"
            if d_input < 10.0
            else f"; the other input is worse by dBIC {d_input:.0f}"
        )
    )
    return HeatReport(best_all, cands, [h.label for h in holds], words, float(d_input))


def predicted(fit: HeatFit, hold: Hold, *, ambient_c: float | None = None, rate_hz: float = FIT_RATE_HZ) -> FloatArray:
    """The fitted model's temperature for a hold (initial slow state 0), on the resampled clock."""
    _t, cmd, meas, _ = _resample(hold, rate_hz)
    u = np.abs(cmd if fit.input == "cmd" else meas) ** fit.p
    dt = 1.0 / rate_hz
    T = fit.g_fast * _pole(u, fit.tau_fast_s, dt)
    if fit.structure == "two_pole":
        T = T + fit.g_slow * _pole(u, fit.tau_slow_s, dt)
    return np.asarray(T + (fit.ambient_c[0] if ambient_c is None else ambient_c), dtype=np.float64)


def rise_at(hold: Hold, seconds: float) -> float:
    """Reported rise over the pre-hold temperature `seconds` after the torque comes on (the A10 numbers)."""
    on = np.flatnonzero(np.abs(hold.tau_cmd) > 1e-6)
    if on.size == 0:
        return float("nan")
    t0 = hold.t[on[0]]
    base = float(np.median(hold.temp[: max(1, on[0])]))
    return float(np.interp(t0 + seconds, hold.t, hold.temp) - base)


def summary(report: HeatReport, holds: Sequence[Hold]) -> dict[str, Any]:
    return {
        "supported": report.supported,
        "best": report.best.as_dict(),
        "candidates": [c.as_dict() for c in report.candidates],
        "rise_5s_c": {h.label: rise_at(h, 5.0) for h in holds},
        "rise_30s_c": {h.label: rise_at(h, 30.0) for h in holds},
    }


def describe(truth: TruthModel) -> Mapping[str, Any]:
    return asdict(truth)
