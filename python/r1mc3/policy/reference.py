"""The Python reference policy runner: v2's WalkPolicy, the test oracle and the simulator's own policy until the C++
runner (DESIGN.md §3.2) is bound.

Ported from r1mc2 (v2) r1mc2/policy.py (WalkPolicy, cap_arm_ranges, MANIP3_ARM_CAP_LEFT, _csv; lines 140-353) at git
r21_try 2a4a165; changes: typed; the executor is r1mc3.policy.onnx_np.OnnxModel (bit-identical to v2's NumpyOnnx);
`load` reads the ONNX file only (no npz path); the SDK joint table comes from r1sim.joints (v2 joints.py has the same
order, KP_STIFF and KD_STIFF - checked). Every computation is v2's, in the same order, so the observations, actions
and (q26, kp26, kd26) are bit-identical to v2's for the same inputs (tests/unit/test_policy_reference.py).

Observation (actor), in metadata order (`observation_names`), each term as mjlab builds it in training:
  base_ang_vel      IMU gyro (pelvis frame)                         3
  projected_gravity R_pelvis^T [0, 0, -1]                           3
  command           (vx, vy, wz), clipped to the policy's limits     3
  lean              forward body lean command, rad (R1MC-Manip2)    1
  waist_cmd         waist roll, yaw command, rad (R1MC-Manip3)      2
  phase             sin/cos(2 pi t / period), zeroed if |cmd| < 0.1  2
  joint_pos         q - default (policy joint order)                n
  joint_vel         dq                                              n
  actions           previous raw action                             n_act
Action: q_target = default + action_scale * action at 50 Hz (`step_dt`); the PD gains come from the metadata.
The live policy (R1MC-Manip2-Lean) observes 24 joints (no head) and acts on 14 (legs + waist): `arms_external`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from r1sim import joints as J

from .onnx_np import OnnxModel

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# v2 policy.py:140-152 (7 Oct 2026, R1MC-Manip3 envelope sweep): the arm envelope a Manip3 policy is capped to.
MANIP3_ARM_CAP_LEFT: dict[str, tuple[float, float]] = {
    "shoulder_pitch": (-2.74, 1.39),
    "shoulder_roll": (-0.13, 2.13),
    "shoulder_yaw": (-1.86, 1.38),
    "elbow": (-0.92, 2.09),
    "wrist_roll": (-1.86, 1.86),
}


def cap_arm_ranges(
    ranges: Mapping[str, Sequence[float]], cap: Mapping[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    """Per joint, the overlap of the trained range and the cap (left-arm convention)."""
    return {
        k: (max(float(a), cap[k][0]), min(float(b), cap[k][1])) if k in cap else (float(a), float(b))
        for k, (a, b) in ranges.items()
    }


def _csv_f(s: str) -> list[float]:
    return [float(x) for x in s.split(",")] if s else []


def _csv_s(s: str) -> list[str]:
    return [str(x) for x in s.split(",")] if s else []


def _strip(n: str) -> str:
    return n[:-6] if n.endswith("_joint") else n


class ReferencePolicy:
    """v2's WalkPolicy: observation building, the lean / waist ramps, the 50 Hz schedule and the 26-joint command."""

    def __init__(
        self,
        net: OnnxModel,
        period: float = 0.6,
        step_dt: float = 0.02,
        cmd_limits: Sequence[Sequence[float]] = ((-0.5, 1.0), (-0.5, 0.5), (-1.0, 1.0)),
        joint_index: Sequence[int] | None = None,
    ) -> None:
        self.net = net
        meta = net.meta
        self.names: list[str] = [_strip(n) for n in _csv_s(meta["joint_names"])]
        self.kp: FloatArray = np.array(_csv_f(meta["joint_stiffness"]))
        self.kd: FloatArray = np.array(_csv_f(meta["joint_damping"]))
        self.default: FloatArray = np.array(_csv_f(meta["default_joint_pos"]))
        act = [_strip(n) for n in _csv_s(meta.get("action_joint_names", ""))] or list(self.names)
        self.act_names: list[str] = act
        self.act_pos: IntArray = np.array([self.names.index(n) for n in act])  # positions within names
        sc = meta.get("action_scale", "0.25")
        self.scale: FloatArray = np.array(_csv_f(sc)) if "," in sc else np.full(len(act), float(sc))
        self.obs_names: list[str] = _csv_s(meta["observation_names"])
        lr = _csv_f(meta.get("lean_range_rad", ""))
        self.lean_range: tuple[float, float] | None = (
            (lr[0], lr[1]) if "lean" in self.obs_names and len(lr) == 2 else None
        )
        self.arm_ranges_left: dict[str, tuple[float, float]] | None = (
            {k: (float(v[0]), float(v[1])) for k, v in json.loads(meta["arm_ranges_left"]).items()}
            if meta.get("arm_ranges_left")
            else None
        )
        self.arm_ranges_trained_left = self.arm_ranges_left
        if self.arm_ranges_left and meta.get("r1mc_task", "").startswith("R1MC-Manip3"):  # v2 7 Oct: see the cap
            self.arm_ranges_left = cap_arm_ranges(self.arm_ranges_left, MANIP3_ARM_CAP_LEFT)
        self.lean_target = 0.0  # rad, + = forward
        self.lean = 0.0  # what the policy sees: ramps to the target (0 while walking)
        wr, wy = _csv_f(meta.get("waist_roll_range_rad", "")), _csv_f(meta.get("waist_yaw_range_rad", ""))
        self.waist_range: tuple[tuple[float, float], tuple[float, float]] | None = (
            ((wr[0], wr[1]), (wy[0], wy[1]))
            if "waist_cmd" in self.obs_names and len(wr) == 2 and len(wy) == 2
            else None
        )
        rates = _csv_f(meta.get("body_rate_rad_s", ""))
        self.lean_rate = rates[0] if len(rates) == 3 else float(np.radians(4.0))
        self.waist_rate: FloatArray = np.array(rates[1:3]) if len(rates) == 3 else np.array([0.25, 0.40])
        self.lean_body = meta.get("lean_body", "torso_link")
        self.waist_target: FloatArray = np.zeros(2)  # rad (roll, yaw)
        self.waist: FloatArray = np.zeros(2)  # what the policy sees: ramps to the target (0 while walking)
        self.period = float(period)
        self.step_dt = float(step_dt)
        self.cmd_lo: FloatArray = np.array([c[0] for c in cmd_limits])
        self.cmd_hi: FloatArray = np.array([c[1] for c in cmd_limits])
        # Map policy joints onto the 26-joint SDK vector (the R1 policy has no head joints).
        self.idx: IntArray = np.array(joint_index if joint_index is not None else [J.IDX[n] for n in self.names])
        self.n = len(self.names)
        self.n_act = len(act)
        self.act_idx26: IntArray = self.idx[self.act_pos]  # 26-index of the action joints
        self.arms_external = not np.intersect1d(self.act_idx26, J.ARMS).size
        self.check_idx = self.act_idx26
        self.check_default: FloatArray = self.default[self.act_pos]
        self.sha256 = ""
        self.source = ""
        # Optional filter on the raw network output (r1sim.control.PolicyController injects the policy faults here);
        # None = the output as is (v2's behaviour).
        self.action_hook: Callable[[FloatArray], FloatArray] | None = None
        self.reset()

    @classmethod
    def load(cls, policy_dir: str | Path, **kw: Any) -> ReferencePolicy:
        """A policy folder (policy.onnx + r1mc_policy.json, e.g. r1mc3.policy.POLICY_DIR) or the .onnx path."""
        p = Path(policy_dir)
        path = p / "policy.onnx" if p.is_dir() else p
        side = path.parent / "r1mc_policy.json"
        if side.exists():
            extra = json.loads(side.read_text())
            kw.setdefault("period", extra.get("period", 0.6))
            if "cmd_limits" in extra:
                kw.setdefault("cmd_limits", extra["cmd_limits"])
        data = path.read_bytes()
        pol = cls(OnnxModel.load(path), **kw)
        pol.sha256 = hashlib.sha256(data).hexdigest()
        pol.source = path.resolve().parent.name + "/" + path.name
        return pol

    def reset(self, q26: npt.ArrayLike | None = None) -> None:
        self.last_action: FloatArray = np.zeros(self.n_act)
        self.phase_t = 0.0
        self.t_last: float | None = None
        self.t_policy: float | None = None
        self.target: FloatArray = self.default[self.act_pos].copy()
        self.zero_since: float | None = None
        self.last_dq: FloatArray | None = None
        self.lean = 0.0
        self.waist = np.zeros(2)
        self.policy_steps = 0

    def observe(
        self, q: FloatArray, dq: FloatArray, quat: npt.ArrayLike, gyro: npt.ArrayLike, cmd: FloatArray
    ) -> FloatArray:
        """The observation vector (policy-order joint vectors q, dq; IMU quat (w, x, y, z) and gyro)."""
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, float))
        g = R.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        ph = (self.phase_t % self.period) / self.period
        phase = np.array([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)])
        if np.linalg.norm(cmd) < 0.1:
            phase[:] = 0.0
        terms = dict(
            base_ang_vel=np.asarray(gyro, float),
            projected_gravity=g,
            command=cmd,
            phase=phase,
            joint_pos=q - self.default,
            joint_vel=dq,
            actions=self.last_action,
            lean=np.array([self.lean]),
            waist_cmd=self.waist.copy(),
        )
        return np.concatenate([terms[n] for n in self.obs_names])

    def network(self, obs: FloatArray) -> FloatArray:
        """The network output for one observation, through `action_hook` when one is set."""
        a = self.net(obs)
        return a if self.action_hook is None else self.action_hook(a)

    def act(
        self, q: FloatArray, dq: FloatArray, quat: npt.ArrayLike, gyro: npt.ArrayLike, cmd: npt.ArrayLike
    ) -> FloatArray:
        """One 50 Hz policy step on policy-order joint vectors; returns the 14 joint targets."""
        c = np.clip(np.asarray(cmd, float), self.cmd_lo, self.cmd_hi)
        walking = (np.linalg.norm(c[:2]) + abs(c[2])) > 0.1
        if self.lean_range is not None:  # as in training: 4 deg/s towards the target, 0 while walking
            goal = 0.0 if walking else float(self.lean_target)
            stp = self.lean_rate * self.step_dt
            self.lean += float(np.clip(goal - self.lean, -stp, stp))
        if self.waist_range is not None:  # R1MC-Manip3, as in training: ramp, 0 while walking
            wgoal = np.zeros(2) if walking else self.waist_target
            wstp = self.waist_rate * self.step_dt
            self.waist = self.waist + np.clip(wgoal - self.waist, -wstp, wstp)
        a = self.network(self.observe(q, dq, quat, gyro, c))
        self.last_action = a
        self.target = self.default[self.act_pos] + self.scale * a
        self.policy_steps += 1
        return self.target

    def set_body_targets(
        self, lean_deg: float | None = None, waist_roll: float | None = None, waist_yaw: float | None = None
    ) -> str | None:
        """Targets the policy holds while standing (None = keep). All or nothing: None when set, else why not."""
        if lean_deg is not None:
            lr = self.lean_range
            if lr is None and abs(float(lean_deg)) >= 1e-9:
                return "this policy has no lean command (R1MC-Manip2 has one)"
            if lr is not None and not (lr[0] - 1e-6 <= np.radians(float(lean_deg)) <= lr[1] + 1e-6):
                return (
                    f"lean {float(lean_deg):g} deg outside the policy's "
                    f"{np.degrees(lr[0]):g}..{np.degrees(lr[1]):g} deg"
                )
        w = self.waist_target.copy()
        for i, (nm, v) in enumerate((("roll", waist_roll), ("yaw", waist_yaw))):
            if v is None:
                continue
            fv = float(v)
            if not np.isfinite(fv):
                return f"waist {nm} {fv} is not a number"
            if self.waist_range is None:
                if abs(fv) < 1e-9:
                    continue
                return "this policy has no waist command (R1MC-Manip3 has one); the waist is the policy's"
            lo, hi = self.waist_range[i]
            if not (lo - 1e-6 <= fv <= hi + 1e-6):
                return f"waist {nm} {fv:+.3f} rad outside the policy's {lo:+.3f}..{hi:+.3f} rad"
            w[i] = fv
        if lean_deg is not None and self.lean_range is not None:
            self.lean_target = float(np.radians(float(lean_deg)))
        self.waist_target = w
        return None

    def body_settled(self, walking: bool = False) -> bool:
        """The ramped lean / waist the policy sees have reached their targets (0 while walking)."""
        lt = 0.0 if walking else self.lean_target
        wt = np.zeros(2) if walking else self.waist_target
        return abs(self.lean - lt) < 1e-6 and bool(np.all(np.abs(self.waist - wt) < 1e-6))

    def step(self, state: Mapping[str, Any], cmd: npt.ArrayLike, t: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Controller interface (called every tick): the 26-joint (q, kp, kd). The network runs every `step_dt`
        (50 Hz) inside the caller's faster loop; between policy steps the last target is held."""
        q = np.asarray(state["q"], float)[self.idx]
        dq = np.asarray(state["dq"], float)[self.idx]
        self.last_dq = dq
        if self.t_last is not None:
            self.phase_t += t - self.t_last
        self.t_last = t
        if self.t_policy is None or t - self.t_policy >= self.step_dt - 1e-4:
            self.t_policy = t if self.t_policy is None else self.t_policy + self.step_dt
            if t - self.t_policy > self.step_dt:
                self.t_policy = t
            self.act(q, dq, state["quat"], state["gyro"], cmd)
        if np.linalg.norm(np.asarray(cmd, float)) < 1e-6:
            self.zero_since = t if self.zero_since is None else self.zero_since
        else:
            self.zero_since = None
        q26 = np.asarray(state["q"], float).copy()
        full = len(q26) == J.N
        kp26 = J.KP_STIFF.copy() if full else np.zeros(len(q26))
        kd26 = J.KD_STIFF.copy() if full else np.zeros(len(q26))
        q26[self.act_idx26] = self.target
        kp26[self.idx] = self.kp
        kd26[self.idx] = self.kd
        return q26, kp26, kd26

    def settled(self, t: float | None = None) -> bool:
        if self.zero_since is None or self.t_last is None:
            return False
        return (self.t_last - self.zero_since) > 1.0 and (self.last_dq is None or np.abs(self.last_dq).max() < 0.5)
