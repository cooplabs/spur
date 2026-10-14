"""The pre-flight check framework (DESIGN §8.2): measured values against their nominal values and tolerances ->
pass, or refuse with the words that say what to move ("container 2.1 cm too far to the robot's right - move it 2.1 cm to
the robot's left").

Directions are always in the ROBOT's frame (x forward = away from the robot, y = the robot's left, z up; yaw
counter-clockwise seen from above), so a word never depends on where the operator stands. The client builds the
checks (which entities, nominal poses, tolerances); `run_checks` evaluates them; the report is saved with the trial.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .register import Registration


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    measured: float | None
    nominal: float | None
    delta: float | None
    tol: float | None
    unit: str
    words: str  # the state, in the robot's frame ("board 1.2 cm too close to the robot")
    fix: str = ""  # what to do ("move it 1.2 cm away from the robot"); "" when ok

    def as_dict(self) -> dict[str, Any]:
        def r(v: float | None) -> float | None:
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)

        return {
            "name": self.name,
            "ok": self.ok,
            "measured": r(self.measured),
            "nominal": r(self.nominal),
            "delta": r(self.delta),
            "tol": r(self.tol),
            "unit": self.unit,
            "words": self.words,
            "fix": self.fix,
        }


class Check(Protocol):
    @property
    def name(self) -> str: ...

    def evaluate(self) -> list[CheckResult]: ...


def _cm(v: float) -> str:
    return f"{abs(v) * 100:.1f} cm"


@dataclass(frozen=True)
class PoseCheck:
    """An entity's measured pose (a Registration in the robot's frame) vs its nominal (x, y, z m, yaw deg). Axes with a
    tolerance of None are not checked. The measurement's own 1-sigma must be below half the tolerance, else it refuses
    ("not precise enough") instead of judging the placement."""

    name: str
    reg: Registration
    nominal: tuple[float, float, float, float]
    tol_xy_m: float | None = 0.01
    tol_z_m: float | None = 0.005
    tol_yaw_deg: float | None = 1.0
    precision_frac: float = 0.5

    def evaluate(self) -> list[CheckResult]:
        n = self.name
        if not self.reg.ok or self.reg.T_ref_entity is None:
            return [
                CheckResult(
                    f"{n}.measured",
                    False,
                    None,
                    None,
                    None,
                    None,
                    "",
                    f"could not measure the {n}: {self.reg.reason}",
                    f"fix the {n}'s tags and measure again",
                )
            ]
        x, y, z = (float(v) for v in self.reg.xyz)
        yaw = self.reg.yaw_deg
        nx, ny, nz, nyaw = self.nominal
        out: list[CheckResult] = []
        sig = self.reg.sigma
        axes: list[tuple[str, float, float, float | None, str]] = [
            ("x", x, nx, self.tol_xy_m, "x_mm"),
            ("y", y, ny, self.tol_xy_m, "y_mm"),
            ("z", z, nz, self.tol_z_m, "z_mm"),
        ]
        for ax, m, nom, tol, sk in axes:
            if tol is None:
                continue
            d = m - nom
            s = sig.get(sk, 0.0) / 1000.0
            if s > self.precision_frac * tol:
                out.append(
                    CheckResult(
                        f"{n}.{ax}",
                        False,
                        m,
                        nom,
                        d,
                        tol,
                        "m",
                        f"the {n}'s {ax} is not measured precisely enough (+-{s * 100:.1f} cm vs the "
                        f"+-{tol * 100:.1f} cm tolerance)",
                        f"add / uncover {n} tags and measure again",
                    )
                )
                continue
            ok = abs(d) <= tol
            words, fix = _axis_words(n, ax, d)
            out.append(
                CheckResult(
                    f"{n}.{ax}",
                    ok,
                    m,
                    nom,
                    d,
                    tol,
                    "m",
                    f"{n} {ax} within {_cm(tol)}" if ok else words,
                    "" if ok else fix,
                )
            )
        if self.tol_yaw_deg is not None:
            d = (yaw - nyaw + 180.0) % 360.0 - 180.0
            s = sig.get("yaw_deg", 0.0)
            tol = self.tol_yaw_deg
            if s > self.precision_frac * tol:
                out.append(
                    CheckResult(
                        f"{n}.yaw",
                        False,
                        yaw,
                        nyaw,
                        d,
                        tol,
                        "deg",
                        f"the {n}'s rotation is not measured precisely enough (+-{s:.2f} deg vs the "
                        f"+-{tol:.1f} deg tolerance)",
                        f"add / uncover {n} tags and measure again",
                    )
                )
            else:
                ok = abs(d) <= tol
                turn = "counter-clockwise" if d > 0 else "clockwise"
                back = "clockwise" if d > 0 else "counter-clockwise"
                out.append(
                    CheckResult(
                        f"{n}.yaw",
                        ok,
                        yaw,
                        nyaw,
                        d,
                        tol,
                        "deg",
                        f"{n} rotation within {tol:.1f} deg"
                        if ok
                        else f"{n} turned {abs(d):.1f} deg {turn} (seen from above)",
                        "" if ok else f"turn it {abs(d):.1f} deg {back} (seen from above)",
                    )
                )
        return out


def _axis_words(name: str, ax: str, d: float) -> tuple[str, str]:
    if ax == "x":
        if d > 0:
            return f"{name} {_cm(d)} too far from the robot", f"move it {_cm(d)} toward the robot"
        return f"{name} {_cm(d)} too close to the robot", f"move it {_cm(d)} away from the robot"
    if ax == "y":
        if d > 0:
            return f"{name} {_cm(d)} too far to the robot's left", f"move it {_cm(d)} to the robot's right"
        return f"{name} {_cm(d)} too far to the robot's right", f"move it {_cm(d)} to the robot's left"
    if d > 0:
        return f"{name} {_cm(d)} too high", f"check the {name}'s height (lower it {_cm(d)})"
    return f"{name} {_cm(d)} too low", f"check the {name}'s height (raise it {_cm(d)})"


@dataclass(frozen=True)
class ValueCheck:
    """A scalar within [lo, hi] (battery %, lowstate rate, a pointing error in mm, ...)."""

    name: str
    value: float | None
    lo: float | None = None
    hi: float | None = None
    unit: str = ""
    what: str = ""  # the words for the quantity ("battery", "pointing error at the tag corners")

    def evaluate(self) -> list[CheckResult]:
        w = self.what or self.name
        if self.value is None or (isinstance(self.value, float) and math.isnan(self.value)):
            return [CheckResult(self.name, False, None, None, None, None, self.unit, f"{w}: no value", f"read the {w}")]
        v = float(self.value)
        if self.lo is not None and v < self.lo:
            return [
                CheckResult(
                    self.name,
                    False,
                    v,
                    self.lo,
                    v - self.lo,
                    None,
                    self.unit,
                    f"{w} {v:g} {self.unit} below {self.lo:g}",
                    f"raise the {w} above {self.lo:g} {self.unit}",
                )
            ]
        if self.hi is not None and v > self.hi:
            return [
                CheckResult(
                    self.name,
                    False,
                    v,
                    self.hi,
                    v - self.hi,
                    None,
                    self.unit,
                    f"{w} {v:g} {self.unit} above {self.hi:g}",
                    f"bring the {w} below {self.hi:g} {self.unit}",
                )
            ]
        return [CheckResult(self.name, True, v, None, None, None, self.unit, f"{w} {v:g} {self.unit} ok")]


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    results: tuple[CheckResult, ...]
    measured: dict[str, Any] = field(default_factory=dict)  # the measured layout, saved with the trial

    @property
    def refusals(self) -> list[str]:
        return [f"{r.words} - {r.fix}" if r.fix else r.words for r in self.results if not r.ok]

    def summary(self) -> str:
        if self.ok:
            return f"PASS: {len(self.results)} checks within tolerance - cover the tags and start"
        return "REFUSED: " + "; ".join(self.refusals)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary(),
            "results": [r.as_dict() for r in self.results],
            "measured": self.measured,
        }


def run_checks(checks: Iterable[Check], measured: dict[str, Any] | None = None) -> PreflightReport:
    res: list[CheckResult] = []
    for c in checks:
        res += c.evaluate()
    return PreflightReport(all(r.ok for r in res) and bool(res), tuple(res), dict(measured or {}))


def checks_named(results: Sequence[CheckResult], prefix: str) -> list[CheckResult]:
    return [r for r in results if r.name.startswith(prefix)]
