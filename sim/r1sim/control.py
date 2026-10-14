"""Controllers for the simulator: the balance policy (session 2) and the session-1 stand hold (a fixture).

`PolicyController` runs the live balance policy (r1mc3.policy.ReferencePolicy = v2's WalkPolicy, bit-identical) at
50 Hz inside the caller's control rate: legs + waist from the policy with the gains of its ONNX metadata, the arms held
at a fixed pose with the metadata arm gains (the policy is `arms_external`: in training a scripted arm term moved
them; v2's controller drives them on the robot), the head at 0 with the stiff gains, the hands open.

`StandHold` is a stiff joint PD at a fixed stand pose, with the gains of the official R1 low-level example (v2
joints.KP_STIFF / KD_STIFF), and the hands held open (Dex3 gains kp 1.5, kd 0.2, xr_teleoperate). It balances
nothing: it only stands if the whole-body CoM lies over the feet. That is why the default pose is
`r1sim.model.centred_stand_pose` (POSE_STAND with both ankle pitches shifted until the CoM is over the foot centre):
POSE_STAND itself is Unitree's stand lock under an ACTIVE balance controller and falls backwards under a plain PD
(the 4 Oct 2026 real-robot fall on `lock` is the same mechanism, v2 README.md:310-318).

A controller is a callable `(LowState) -> LowCmd | None`; it never steps physics (r1sim.engine).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from . import dex3
from . import joints as J
from .engine import HandCmd, LowCmd, LowState

if TYPE_CHECKING:
    from r1mc3.policy import ReferencePolicy

    from .faults import FaultSchedule

FloatArray = npt.NDArray[np.float64]


class StandHold:
    """Stiff PD at a fixed pose for the 26 body joints (+ hands open). Returns the same command every tick."""

    def __init__(
        self,
        pose: npt.ArrayLike = J.POSE_STAND,
        *,
        kp: npt.ArrayLike = J.KP_STIFF,
        kd: npt.ArrayLike = J.KD_STIFF,
        hands: tuple[str, ...] = dex3.SIDES,
        hand_preset: str = "open",
    ) -> None:
        self.pose: FloatArray = np.asarray(pose, dtype=np.float64).copy()
        cmd = LowCmd()
        cmd.q[:] = self.pose
        cmd.kp[:] = np.asarray(kp, dtype=np.float64)
        cmd.kd[:] = np.asarray(kd, dtype=np.float64)
        for side in hands:
            h = HandCmd()
            h.q[:] = dex3.preset(hand_preset, side)
            h.kp[:] = dex3.KP
            h.kd[:] = dex3.KD
            cmd.hands[side] = h
        self._cmd = cmd
        self.calls = 0

    @classmethod
    def for_model(cls, m: Any, *, centred: bool = True, **kw: Any) -> StandHold:
        """A stand hold for this compiled robot model: the CoM-centred stand pose (default) or POSE_STAND."""
        from .model import centred_stand_pose

        pose = centred_stand_pose(m) if centred else J.POSE_STAND.copy()
        return cls(pose, **kw)

    @property
    def command(self) -> LowCmd:
        return self._cmd.copy()

    def __call__(self, state: LowState) -> LowCmd:
        self.calls += 1
        return self._cmd.copy()


def _open_hands(cmd: LowCmd, hands: tuple[str, ...], preset: str = "open") -> None:
    for side in hands:
        h = HandCmd()
        h.q[:] = dex3.preset(preset, side)
        h.kp[:] = dex3.KP
        h.kd[:] = dex3.KD
        cmd.hands[side] = h


class PolicyController:
    """The balance policy as a simulator controller: `(LowState) -> LowCmd` at the caller's rate (500 Hz real profile,
    every 5 ms step in the train profile), the network every `policy.step_dt` (50 Hz) with the target held between
    policy steps - v2's `WalkPolicy.step` schedule (v2 policy.py:324-348).

    The command: legs + waist = the policy's targets with its metadata kp/kd (84.302 / 5.367, ankles 31.385 / 1.998);
    arms = `arm_pose` (default: the policy's default arm angles) with the metadata arm gains (60/60/45/45/30, kd
    2/2/2/2/1.5); head = 0 with KP_STIFF/KD_STIFF; hands open (Dex3 kp 1.5, kd 0.2). `cmd` = (vx, vy, wz).

    Faults (r1sim.faults, at the simulated time of the policy step): `policy_nan` sets the network output to NaN,
    `policy_out_of_range` to `value` (both on `index`, default every action). The resulting targets are passed on
    unchanged - what the board / engine does with them is not this controller's business."""

    def __init__(
        self,
        policy: ReferencePolicy,
        *,
        cmd: Sequence[float] = (0.0, 0.0, 0.0),
        arm_pose: npt.ArrayLike | None = None,
        faults: FaultSchedule | None = None,
        hands: tuple[str, ...] = dex3.SIDES,
    ) -> None:
        self.policy = policy
        self.cmd: FloatArray = np.asarray(cmd, dtype=np.float64).copy()
        self.faults = faults
        arm_pos = [policy.names.index(J.JOINTS[i]) for i in J.ARMS]
        self.arm_pose: FloatArray = (
            policy.default[arm_pos].copy() if arm_pose is None else np.asarray(arm_pose, dtype=np.float64).copy()
        )
        if self.arm_pose.shape != (len(J.ARMS),):
            raise ValueError(f"arm_pose must have {len(J.ARMS)} values (SDK arm order), got {self.arm_pose.shape}")
        self._hands = hands
        self._t = 0.0
        self.calls = 0
        self.max_abs_action = 0.0
        policy.action_hook = self._filter

    def _filter(self, a: FloatArray) -> FloatArray:
        out = a
        if self.faults is not None:
            for kind in ("policy_nan", "policy_out_of_range"):
                for f in self.faults.active(kind, self._t):
                    out = out.copy()
                    idx = f.params.get("index")
                    sel = slice(None) if idx is None else np.asarray(idx, dtype=np.int64)
                    out[sel] = np.nan if kind == "policy_nan" else float(f.params["value"])
                    self.faults.record(self._t, kind, "applied", index=idx)
        if np.all(np.isfinite(out)):
            self.max_abs_action = max(self.max_abs_action, float(np.abs(out).max()))
        return out

    def reset(self) -> None:
        self.policy.reset()
        self.calls = 0
        self.max_abs_action = 0.0

    def __call__(self, state: LowState) -> LowCmd:
        self.calls += 1
        self._t = float(state.t)
        st = {"q": state.q, "dq": state.dq, "quat": state.quat, "gyro": state.gyro}
        q26, kp26, kd26 = self.policy.step(st, self.cmd, self._t)
        cmd = LowCmd()
        cmd.q[:] = q26
        cmd.q[J.ARMS] = self.arm_pose
        cmd.q[J.HEAD] = 0.0
        cmd.kp[:] = kp26
        cmd.kd[:] = kd26
        _open_hands(cmd, self._hands)
        return cmd
