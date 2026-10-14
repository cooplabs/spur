"""Fault injection schedule (DESIGN.md §12.6): every fault has a scheduled start (simulated time) and a duration, so
a failure is reproducible - in lockstep bit for bit.

This module only holds the schedule and its log. The components apply the faults that concern them:

Kinds, who applies them, and their params (all optional unless marked required):

* `controller_stall` - engine (lockstep: due controllers are not called; real time: `ControllerThread`);
  `name` = one controller (default all).
* `cmd_gap` - engine: every command written in the window is discarded (the board sees no commands).
* `cmd_drop` - engine: each command in the window is discarded with probability `p` (default 0.5; seeded).
* `state_delay` - engine: states read in the window are `delay_s` (required) old.
* `hand_state_freeze` - engine: the hand's state stops updating (q, dq, tau, pressure, temperature);
  `side` = "left" / "right" (default both).
* `motor_error` - engine: `LowState.motor_err[joints] = word` in the window; `joints` (SDK indices), `word` (int, 1).
* `sensor_spike` - sensors: a reading jumps by `magnitude` in the window; `field` (q, dq, tau_est, gyro, acc,
  hand_pressure), `index`, `magnitude`.
* `policy_nan` - policy runner: the action vector is NaN in the window; `index` (default all).
* `policy_out_of_range` - policy runner: the action vector is `value` (required) in the window; `index` (default all).

The log also gets the engine's reactions that are not faults themselves: kind "rejected_cmd" (a command with a
non-finite value, rejected whole by the simulated board). Commands are discarded (cmd_gap / cmd_drop) at the time they
are written, before the driver's command delay; state-side faults are logged once, at their first application
("apply"); lockstep stalls once per skipped controller call ("skip_call"), real-time stalls once per block ("block").
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np

FaultKind = Literal[
    "controller_stall",
    "cmd_gap",
    "cmd_drop",
    "state_delay",
    "hand_state_freeze",
    "motor_error",
    "sensor_spike",
    "policy_nan",
    "policy_out_of_range",
]
FAULT_KINDS: tuple[str, ...] = get_args(FaultKind)


@dataclass(frozen=True)
class Fault:
    """One scheduled fault: active for simulated times t with t_start <= t < t_start + duration."""

    kind: FaultKind
    t_start: float
    duration: float
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}; use one of {FAULT_KINDS}")
        if not (np.isfinite(self.t_start) and np.isfinite(self.duration)) or self.duration <= 0:
            raise ValueError(f"{self.kind}: t_start must be finite and duration > 0")

    @property
    def t_end(self) -> float:
        return self.t_start + self.duration

    def active(self, t: float) -> bool:
        return self.t_start <= t < self.t_end


@dataclass(frozen=True)
class FaultEvent:
    """What a component did because of a fault (the injector's log, for reports and tests)."""

    t: float
    kind: str
    what: str
    detail: Mapping[str, Any]


class FaultSchedule:
    """The faults of one run and their log. Random draws (e.g. `cmd_drop`) use `rng(kind)`: one generator per kind,
    seeded from (seed, kind), so adding a fault of another kind does not change the draws."""

    def __init__(self, faults: Iterable[Fault] = (), *, seed: int = 0) -> None:
        self.seed = int(seed)
        self.faults: list[Fault] = list(faults)
        self.log: list[FaultEvent] = []
        self._rngs: dict[str, np.random.Generator] = {}

    def add(self, fault: Fault) -> None:
        self.faults.append(fault)

    def active(self, kind: str, t: float) -> list[Fault]:
        """The faults of `kind` active at simulated time t (schedule order)."""
        return [f for f in self.faults if f.kind == kind and f.active(t)]

    def rng(self, kind: str) -> np.random.Generator:
        g = self._rngs.get(kind)
        if g is None:
            g = np.random.default_rng([self.seed, zlib.crc32(kind.encode())])
            self._rngs[kind] = g
        return g

    def record(self, t: float, kind: str, what: str, **detail: Any) -> None:
        self.log.append(FaultEvent(float(t), kind, what, dict(detail)))

    def count(self, kind: str, what: str | None = None) -> int:
        return sum(1 for e in self.log if e.kind == kind and (what is None or e.what == what))
