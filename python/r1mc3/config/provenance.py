"""Provenance for every physical or safety-relevant value (DESIGN.md §3.6, §12.3; requirement R-j).

A value never travels without where it came from: its source (a document section, a file:line, a measurement), the
date, a status, and - when it is only an estimate or a third-party report - what will replace it (`until`).

The same entry format is used by the simulator's model parameters (`models/provenance.yaml`, validated by
`load_provenance`) and by the controller's robot profile (`config/robot_profile.*.yaml`, see `profile.py`):

```yaml
schema: r1mc3.provenance/1
parameters:
  hand_mount_pos_m:
    value: [0.075, 0.0, 0.0]
    unit: m
    status: estimate
    source: "r1mc2 dex3.MOUNT_POS (v2 dex3.py:117-130): ..."
    date: 2026-10-01
    profiles: [train, real]
    until: "hand-eye capture on HW1"
```
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

__all__ = [
    "NEEDS_UNTIL",
    "PROFILES",
    "PROVENANCE_SCHEMA",
    "EntryRange",
    "ParameterEntry",
    "Provenance",
    "ProvenanceFile",
    "Provenanced",
    "RangeSpec",
    "Status",
    "load_provenance",
    "range_pairs",
]

#: How a value is known.
#: official = a manufacturer file or document; measured = measured by us on our hardware; derived = computed from
#: other entries; reported = a third-party claim not checked by us; estimate = our best guess; modelling_choice = a
#: deliberate choice of ours (a solver setting, a margin) rather than a fact about the robot.
Status = Literal["official", "measured", "derived", "reported", "estimate", "modelling_choice"]

#: Statuses that must say what will replace them.
NEEDS_UNTIL: frozenset[str] = frozenset({"estimate", "reported"})

#: Simulator model profiles (DESIGN.md §12.3).
PROFILES: tuple[str, ...] = ("train", "real")

PROVENANCE_SCHEMA = "r1mc3.provenance/1"

_ID = re.compile(r"^[a-z][a-z0-9_]*$")

T = TypeVar("T")


class Provenance(BaseModel):
    """Where a value comes from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Status
    source: str = Field(min_length=1)
    date: datetime.date
    until: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _until_when_unconfirmed(self) -> Provenance:
        if self.status in NEEDS_UNTIL and not (self.until and self.until.strip()):
            raise ValueError(f"status {self.status!r} needs `until` (what measurement or source replaces it)")
        return self


#: A parameter range: `[min, max]` for a scalar value, or one `[min, max]` per component for a vector value.
RangeSpec = tuple[float, float] | tuple[tuple[float, float], ...]
#: The range of an entry: a RangeSpec, or - for a mapping value (e.g. per joint group) - a RangeSpec per key (keys may
#: be a subset of the value's keys: the others are not ranged, e.g. a condim next to a friction).
EntryRange = RangeSpec | dict[str, RangeSpec]


def range_pairs(rng: Any) -> tuple[bool, list[tuple[float, float]]]:
    """(is_scalar, [(min, max), ...]) for a RangeSpec."""
    items = list(rng)
    if items and not isinstance(items[0], (list, tuple)):
        return True, [(float(items[0]), float(items[1]))]
    return False, [(float(a), float(b)) for a, b in items]


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _in_range(value: Any, rng: Any) -> str | None:
    """None if `value` lies inside `rng` (see RangeSpec / EntryRange), else why not."""
    if isinstance(rng, dict):
        if not isinstance(value, dict):
            return f"a per-key range needs a mapping value, got {value!r}"
        for k, r in rng.items():
            if k not in value:
                return f"range key {k!r} is not a key of the value"
            why = _in_range(value[k], r)
            if why:
                return f"{k}: {why}"
        return None
    scalar, pairs = range_pairs(rng)
    if scalar:
        lo, hi = pairs[0]
        if not _num(value):
            return f"a [min, max] range needs a numeric scalar value, got {value!r}"
        return None if lo <= float(value) <= hi else f"value {value} outside range [{lo}, {hi}]"
    if not isinstance(value, (list, tuple)) or len(value) != len(pairs):
        return f"a per-component range needs a vector value with {len(pairs)} components, got {value!r}"
    for k, (v, (lo, hi)) in enumerate(zip(value, pairs, strict=True)):
        if not _num(v) or not lo <= float(v) <= hi:
            return f"component {k}: value {v!r} outside range [{lo}, {hi}]"
    return None


class Provenanced(Provenance, Generic[T]):
    """A value with its unit and provenance, all fields inline (the YAML entry format above).

    `range` (optional, session 2): the interval the true value is believed to lie in - `[min, max]`, one pair per
    component of a vector value, or such a range per key of a mapping value - with `range_source` saying where the
    bounds come from. The simulator draws seeded
    model variants within these ranges (no seed = the nominal `value`)."""

    value: T
    unit: str | None = None
    range: EntryRange | None = None
    range_source: str | None = None

    @model_validator(mode="after")
    def _range_valid(self) -> Provenanced[T]:
        if self.range is None:
            if self.range_source is not None:
                raise ValueError("range_source without a range")
            return self
        if not (self.range_source and self.range_source.strip()):
            raise ValueError("a range needs `range_source` (where its bounds come from)")
        specs = list(self.range.values()) if isinstance(self.range, dict) else [self.range]
        for spec in specs:
            for lo, hi in range_pairs(spec)[1]:
                if not lo <= hi:
                    raise ValueError(f"range [{lo}, {hi}]: min > max")
        why = _in_range(self.value, self.range)
        if why:
            raise ValueError(f"the nominal value must lie in its range: {why}")
        return self


class ParameterEntry(Provenanced[JsonValue]):
    """One entry of `models/provenance.yaml`: any JSON-like value, plus the simulator profiles it applies to."""

    profiles: tuple[Literal["train", "real"], ...] = PROFILES  # type: ignore[assignment]
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("profiles")
    @classmethod
    def _profiles_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("profiles must not be empty")
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate profiles: {v}")
        return v


class ProvenanceFile(BaseModel):
    """The whole provenance file: a schema tag and the parameters by id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_: Literal["r1mc3.provenance/1"] = Field(alias="schema")
    parameters: dict[str, ParameterEntry]

    @field_validator("parameters")
    @classmethod
    def _ids(cls, v: dict[str, ParameterEntry]) -> dict[str, ParameterEntry]:
        if not v:
            raise ValueError("no parameters")
        bad = [k for k in v if not _ID.match(k)]
        if bad:
            raise ValueError(f"parameter ids must be snake_case: {bad}")
        return v

    def by_status(self, status: str) -> dict[str, ParameterEntry]:
        """The entries with this status (e.g. every `estimate`, for the 'not solid' list of a report)."""
        return {k: e for k, e in self.parameters.items() if e.status == status}


def _read_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_provenance(path: str | Path) -> ProvenanceFile:
    """Read and validate a provenance file (YAML). Raises pydantic.ValidationError on any problem."""
    return ProvenanceFile.model_validate(_read_yaml(path))
