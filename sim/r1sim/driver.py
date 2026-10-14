"""Motor-driver profiles of the simulator (DESIGN.md §12.3 "Actuators"; V2_INVENTORY App. E §4.2 items 1 and 4).

The engine (`r1sim.engine.Simulation`) evaluates the joint PD inside MuJoCo. A `DriverConfig` says how:

* ``kind="board"`` - the motor board's law on the LowCmd fields, every physics step:
  ``tau = clip(kp * (q_des - q) + kd * (dq_des - dq) + tau_ff, -tau_max, tau_max)`` (an affine actuator: gain 1,
  bias ``[0, -kp, -kd]``, ctrl ``kp * q_des + kd * dq_des + tau_ff``). The S1 behaviour and the real-equivalent
  profile.
* ``kind="position"`` - MuJoCo ``<position>`` actuators exactly as mjlab 1.2.0 builds them for training
  (``BuiltinPositionActuatorCfg`` -> ``mjlab/utils/spec.py`` ``create_position_actuator``: gain ``kp``, bias
  ``[0, -kp, -kd]``, ``ctrl = q_des``, ``ctrllimited`` false, force range +-effort limit). Only position targets:
  a command with ``dq != 0`` or ``tau_ff != 0`` is refused (ValueError). For ``dq = tau_ff = 0`` both kinds give the
  same forces bit for bit (tests/unit/test_driver.py); the train-equivalent profile uses this kind so its actuators
  are the training ones, not merely equivalent.

On top of the law, the real board's input path and its behaviour when commands stop - all UNKNOWN for our robot until
the HW1 latency tests and the board-hold test on the stand, so the defaults (models/driver.yaml) are estimates with
ranges:

* ``cmd_delay_s``: a command written at sim time t takes effect at the first step boundary >= t + delay;
* ``state_delay_s``: `read_state()` returns the newest state published at or before t - delay;
* ``cmd_timeout_s`` / ``on_cmd_timeout``: after that long without a new command the board ``hold``s the last
  command (what it did for >= 4 s on 9 Oct 2026), ``damp``s (kp 0, kd ``damp_kd``) or applies ``zero_torque``,
  until the next command arrives. None = hold forever.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .model import models_dir

DriverKind = Literal["board", "position"]
TimeoutAction = Literal["hold", "damp", "zero_torque"]
DRIVER_KINDS: tuple[str, ...] = ("board", "position")
TIMEOUT_ACTIONS: tuple[str, ...] = ("hold", "damp", "zero_torque")
N_BODY_JOINTS = 26  # the body joints in SDK order (r1sim.joints.JOINTS); a per-joint damp_kd has one value each


def driver_yaml() -> Path:
    """The real-equivalent driver parameters with their provenance (schema r1mc3.provenance/1)."""
    return models_dir() / "driver.yaml"


@dataclass(frozen=True)
class DriverConfig:
    """How the simulated motor drivers turn commands into joint torques (see the module docstring)."""

    kind: DriverKind = "board"
    cmd_delay_s: float = 0.0
    state_delay_s: float = 0.0
    cmd_timeout_s: float | None = None
    on_cmd_timeout: TimeoutAction = "hold"
    damp_kd: float | tuple[float, ...] = 2.0  # one value, or one per body joint (SDK order; session 8: per group)

    def __post_init__(self) -> None:
        if self.kind not in DRIVER_KINDS:
            raise ValueError(f"driver kind {self.kind!r}: use one of {DRIVER_KINDS}")
        if self.on_cmd_timeout not in TIMEOUT_ACTIONS:
            raise ValueError(f"on_cmd_timeout {self.on_cmd_timeout!r}: use one of {TIMEOUT_ACTIONS}")
        for name in ("cmd_delay_s", "state_delay_s"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {v}")
        kds = (self.damp_kd,) if isinstance(self.damp_kd, (int, float)) else tuple(self.damp_kd)
        if not isinstance(self.damp_kd, (int, float)) and len(kds) != N_BODY_JOINTS:
            raise ValueError(f"damp_kd: one value or {N_BODY_JOINTS} (one per body joint), got {len(kds)}")
        for v in kds:
            if not math.isfinite(float(v)) or float(v) < 0.0:
                raise ValueError(f"damp_kd must be finite and >= 0, got {v}")
        if self.cmd_timeout_s is not None and not (math.isfinite(self.cmd_timeout_s) and self.cmd_timeout_s > 0.0):
            raise ValueError(f"cmd_timeout_s must be None or > 0, got {self.cmd_timeout_s}")

    @property
    def plain(self) -> bool:
        """No delay and no timeout: the engine can use its S1 fast path (only the PD law differs by kind)."""
        return self.cmd_delay_s == 0.0 and self.state_delay_s == 0.0 and self.cmd_timeout_s is None

    @classmethod
    def train(cls) -> DriverConfig:
        """The train-equivalent profile: mjlab's <position> actuators, no delay (training had none, App. E §3.5)."""
        return cls(kind="position")

    @classmethod
    def real(cls, path: str | Path | None = None) -> DriverConfig:
        """The real-equivalent profile: the board law with the estimated delays / timeout of models/driver.yaml."""
        from r1mc3.config import load_provenance

        p = load_provenance(path or driver_yaml()).parameters

        def val(key: str) -> Any:
            return p[key].value

        timeout = val("cmd_timeout_s")
        action = str(val("on_cmd_timeout"))
        if action not in TIMEOUT_ACTIONS:
            raise ValueError(f"{path or driver_yaml()}: on_cmd_timeout {action!r}: use one of {TIMEOUT_ACTIONS}")
        return cls(
            kind="board",
            cmd_delay_s=float(val("cmd_delay_s")),
            state_delay_s=float(val("state_delay_s")),
            cmd_timeout_s=None if timeout is None else float(timeout),
            on_cmd_timeout=cast(TimeoutAction, action),
            damp_kd=float(val("damp_kd")),
        )


def driver_for_profile(profile: str) -> DriverConfig:
    """The driver of a model profile: "train" -> DriverConfig.train(), "real" -> DriverConfig.real()."""
    if profile == "train":
        return DriverConfig.train()
    if profile == "real":
        return DriverConfig.real()
    raise ValueError(f"unknown profile {profile!r}; use 'train' or 'real'")
