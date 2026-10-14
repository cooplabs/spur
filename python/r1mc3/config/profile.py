"""The robot profile: one versioned YAML file -> validated -> frozen at start (DESIGN.md §3.6; requirement R-j).

What it holds: the joint map (R1 SDK slots), rates, limits, named gain sets with bounds, safety thresholds and executor
settings. Every value is a `Provenanced` entry (source, date, status, `until` for estimates).

Rules enforced here:
- the profile is immutable after loading (pydantic frozen models; per-joint maps are read-only mappings);
- only fields marked `tunable: true` (which must carry `bounds`) may change at runtime, within their bounds
  (`RobotProfile.with_tuned`); everything else - in particular every safety field - needs a new file with a higher
  `version` (`check_upgrade`).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    WrapSerializer,
    model_validator,
)

from .provenance import Provenanced

__all__ = [
    "PROFILE_SCHEMA",
    "Bounds",
    "Executor",
    "GainSet",
    "JointMap",
    "JointSlot",
    "Limits",
    "PerJoint",
    "PerJointRange",
    "Rates",
    "RobotProfile",
    "Safety",
    "Scalar",
    "TuningRefused",
    "check_upgrade",
    "load_profile",
]

PROFILE_SCHEMA = "r1mc3.robot_profile/1"

K = TypeVar("K")
V = TypeVar("V")


def _readonly(m: Mapping[K, V]) -> Mapping[K, V]:
    return MappingProxyType(dict(m))


def _plain(m: Mapping[Any, Any], handler: SerializerFunctionWrapHandler) -> Any:
    return handler(dict(m))


# A validated mapping that cannot be modified afterwards (read-only view), serialised as a plain dict.
_RO = (AfterValidator(_readonly), WrapSerializer(_plain))


_FROZEN = ConfigDict(extra="forbid", frozen=True)


class TuningRefused(ValueError):
    """A runtime change that the profile does not allow (not tunable, unknown field, or out of bounds)."""


class Bounds(BaseModel):
    """Closed interval [min, max]."""

    model_config = _FROZEN

    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> Bounds:
        if not self.min <= self.max:
            raise ValueError(f"bounds min {self.min} > max {self.max}")
        return self

    def __contains__(self, x: float) -> bool:
        return self.min <= x <= self.max


class Scalar(Provenanced[float]):
    """A number with provenance; optional bounds; `tunable` only with bounds."""

    bounds: Bounds | None = None
    tunable: bool = False

    @model_validator(mode="after")
    def _bounded(self) -> Scalar:
        if self.tunable and self.bounds is None:
            raise ValueError("a tunable value needs bounds")
        if self.bounds is not None and self.value not in self.bounds:
            raise ValueError(f"value {self.value} outside its bounds [{self.bounds.min}, {self.bounds.max}]")
        return self


class JointSlot(BaseModel):
    """One body joint and its slot in unitree_hg LowCmd_.motor_cmd."""

    model_config = _FROZEN

    name: str = Field(pattern=r"^[a-z][a-z_]*$")
    slot: int = Field(ge=0)


class JointMap(Provenanced[tuple[JointSlot, ...]]):
    """The 26 body joints in SDK order with their hg slots (R1: mode_machine 1, PR ankles)."""

    num_slots: int = Field(ge=1)
    mode_machine: int
    mode_pr: int

    @model_validator(mode="after")
    def _consistent(self) -> JointMap:
        names = [j.name for j in self.value]
        slots = [j.slot for j in self.value]
        if len(set(names)) != len(names):
            raise ValueError("duplicate joint names in joint_map")
        if len(set(slots)) != len(slots):
            raise ValueError("duplicate slots in joint_map")
        if any(s >= self.num_slots for s in slots):
            raise ValueError(f"slot >= num_slots ({self.num_slots})")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.value)

    @property
    def slots(self) -> tuple[int, ...]:
        return tuple(j.slot for j in self.value)


class PerJoint(Provenanced[Annotated[Mapping[str, float], *_RO]]):
    """One number per joint (e.g. effort or velocity limits)."""


class PerJointRange(Provenanced[Annotated[Mapping[str, tuple[float, float]], *_RO]]):
    """A (low, high) pair per joint (e.g. position limits)."""

    @model_validator(mode="after")
    def _ordered(self) -> PerJointRange:
        bad = [k for k, (lo, hi) in self.value.items() if not lo < hi]
        if bad:
            raise ValueError(f"low >= high for {bad}")
        return self


class GainSet(Provenanced[Annotated[Mapping[str, tuple[float, float]], *_RO]]):
    """A named set of (kp, kd) per joint, with the bounds every member must respect."""

    kp_bounds: Bounds
    kd_bounds: Bounds

    @model_validator(mode="after")
    def _within(self) -> GainSet:
        bad = [k for k, (kp, kd) in self.value.items() if kp not in self.kp_bounds or kd not in self.kd_bounds]
        if bad:
            raise ValueError(f"gains outside kp {self.kp_bounds} / kd {self.kd_bounds} for {bad}")
        return self


class Rates(BaseModel):
    model_config = _FROZEN

    writer_hz: Scalar
    writer_fallback_hz: Scalar
    policy_hz: Scalar
    hands_hz: Scalar
    executor_hz: Scalar

    @model_validator(mode="after")
    def _writer(self) -> Rates:
        if self.writer_hz.bounds is None:
            raise ValueError("rates.writer_hz needs bounds (the allowed range)")
        if self.writer_fallback_hz.value not in self.writer_hz.bounds:
            raise ValueError("rates.writer_fallback_hz outside the writer's allowed range")
        if self.writer_hz.value % self.policy_hz.value or self.writer_fallback_hz.value % self.policy_hz.value:
            raise ValueError("the writer rates must be integer multiples of the policy rate")
        return self


class Limits(BaseModel):
    model_config = _FROZEN

    position: PerJointRange
    effort: PerJoint
    velocity: PerJoint
    torque_clip_fraction: Scalar


class Safety(BaseModel):
    """RT-core thresholds (DESIGN.md §4.2-4.4). Never tunable at runtime."""

    model_config = _FROZEN

    lowstate_age_count_ms: Scalar
    lowstate_age_resend_ms: Scalar
    lowstate_age_fault_ms: Scalar
    arm_tilt_stop_deg: Scalar
    fall_detect_deg: Scalar
    policy_age_inline_ms: Scalar
    guard_heartbeat_writer_periods: Scalar
    takeover_max_tilt_deg: Scalar
    takeover_max_state_age_ms: Scalar
    takeover_max_dq_rad_s: Scalar
    takeover_max_temp_c: Scalar
    takeover_release_timeout_s: Scalar
    # session 9 (U19 / D25, docs/sim/STATE_AGE.md): the state-age limit + the "hold the stand" rung. Optional (absent =
    # the core's defaults: 10 ms / 60 ms / 0.5 s, the rung OFF); stale_rung 0 = off, 1 = hold the stand, 2 = + return
    state_age_limit_ms: Scalar | None = None
    state_age_sustain_ms: Scalar | None = None
    stale_rung: Scalar | None = None
    stale_return_live_ms: Scalar | None = None

    @model_validator(mode="after")
    def _not_tunable(self) -> Safety:
        tun = [n for n, s in self if isinstance(s, Scalar) and s.tunable]
        if tun:
            raise ValueError(f"safety fields cannot be tunable: {tun}")
        if not (
            self.lowstate_age_count_ms.value < self.lowstate_age_resend_ms.value < self.lowstate_age_fault_ms.value
        ):
            raise ValueError("lowstate age thresholds must increase: count < resend < fault")
        if self.stale_rung is not None and self.stale_rung.value not in (0.0, 1.0, 2.0):
            raise ValueError("stale_rung must be 0 (off), 1 (hold the stand) or 2 (hold the stand, then return)")
        return self


class Executor(BaseModel):
    """Upper-body executor settings (DESIGN.md §3.3)."""

    model_config = _FROZEN

    keepalive_ms: Scalar
    trajectory_stop_time_s: Scalar
    stream_ttl_max_arm_s: Scalar
    speed_scale: Scalar


class Processes(BaseModel):
    """The r1rt / r1guard values the other sections do not cover (version 2, session 6): with them the profile is the
    whole source of the robot's processes config file (`python -m r1mc3.rtconfig --profile robot`). Never tunable."""

    model_config = _FROZEN

    stand_freeze_ramp_s: Scalar
    takeover_blend_stand_s: Scalar
    damp_kd: PerJoint
    on_stand_hold_gains: Provenanced[Literal["stiff"]]
    action_abs_max: Scalar
    builtin_max_age_ms: Scalar
    builtin_min_leg_kp: Scalar
    mirror_max_dev_rad: Scalar
    takeover_mirror_alarm_s: Scalar
    guard_max_age_ms: Scalar
    guard_cycle_hz: Scalar
    guard_takeover_blend_s: Scalar

    @model_validator(mode="after")
    def _not_tunable(self) -> Processes:
        tun = [n for n, v in self if isinstance(v, Scalar) and v.tunable]
        if tun:
            raise ValueError(f"processes fields cannot be tunable: {tun}")
        return self


class RobotProfile(BaseModel):
    """The whole robot profile. Load with `load_profile`; change at runtime only with `with_tuned`."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["r1mc3.robot_profile/1"] = Field(alias="schema")
    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    description: str | None = None
    joint_map: JointMap
    rates: Rates
    limits: Limits
    gains: Annotated[Mapping[str, GainSet], *_RO]
    safety: Safety
    executor: Executor
    processes: Processes | None = None  # version 2 (session 6); absent in a version-1 profile

    @model_validator(mode="after")
    def _joint_keys(self) -> RobotProfile:
        names = set(self.joint_map.names)
        for what, m in (
            ("limits.position", self.limits.position.value),
            ("limits.effort", self.limits.effort.value),
            ("limits.velocity", self.limits.velocity.value),
        ):
            if set(m) != names:
                raise ValueError(
                    f"{what}: joints differ from joint_map (missing {sorted(names - set(m))}, "
                    f"unknown {sorted(set(m) - names)})"
                )
        for g, gs in self.gains.items():
            unknown = set(gs.value) - names
            if unknown:
                raise ValueError(f"gains.{g}: unknown joints {sorted(unknown)}")
        if self.processes is not None and set(self.processes.damp_kd.value) != names:
            dk = set(self.processes.damp_kd.value)
            raise ValueError(
                f"processes.damp_kd: joints differ from joint_map (missing {sorted(names - dk)}, "
                f"unknown {sorted(dk - names)})"
            )
        return self

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.joint_map.names

    def scalars(self) -> Iterator[tuple[str, Scalar]]:
        """Every Scalar field with its dotted path (e.g. 'executor.speed_scale')."""
        for section in ("rates", "limits", "safety", "executor", "processes"):
            sec = getattr(self, section)
            if sec is None:
                continue
            for name, v in sec:
                if isinstance(v, Scalar):
                    yield f"{section}.{name}", v

    def tunables(self) -> dict[str, Scalar]:
        return {p: s for p, s in self.scalars() if s.tunable}

    def with_tuned(self, path: str, value: float) -> RobotProfile:
        """A copy with one tunable field changed. Refuses unknown, non-tunable or out-of-bounds changes."""
        scalars = dict(self.scalars())
        if path not in scalars:
            raise TuningRefused(f"{path}: no such field")
        s = scalars[path]
        if not s.tunable or s.bounds is None:
            raise TuningRefused(f"{path}: not tunable (needs a new profile version)")
        v = float(value)
        if v not in s.bounds:
            raise TuningRefused(f"{path}: {v} outside [{s.bounds.min}, {s.bounds.max}]")
        section, name = path.split(".")
        sec = getattr(self, section)
        new_sec = sec.model_copy(update={name: s.model_copy(update={"value": v})})
        return self.model_copy(update={section: new_sec})


def _dump(p: RobotProfile) -> dict[str, Any]:
    d: dict[str, Any] = p.model_dump(mode="json", by_alias=True)
    return d


def check_upgrade(old: RobotProfile, new: RobotProfile) -> None:
    """Refuse a new profile that changes anything but tunable values without raising `version`."""
    if new.version > old.version:
        return
    a, b = _dump(old), _dump(new)
    for path in old.tunables():  # tunable values may differ freely
        sec, name = path.split(".")
        a[sec][name]["value"] = b[sec][name]["value"]
    if a != b:
        raise ValueError(f"profile {new.name!r} changed without a version bump (still {new.version})")


def load_profile(path: str | Path) -> RobotProfile:
    """YAML -> validated -> frozen. Raises pydantic.ValidationError on any problem."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return RobotProfile.model_validate(data)
