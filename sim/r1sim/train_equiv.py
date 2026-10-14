"""The train-equivalent profile at run time: training's initial state, its policy interface and observations, and
the A1 sim2sim harness (DESIGN.md §12.3, §13.1 A1; V2_INVENTORY App. E §3.2-3.5, §4.2).

The MODEL side of the profile is `r1sim.model` profile "train" (mjlab's <position> actuators, FULL_COLLISION,
the sim options of the policy's params/env.yaml); tools/sim2sim/model_diff.py checks it field by field against the
model mjlab 1.2.0 itself compiles (tools/sim2sim/mjlab_reference.py). This module holds what training does at run
time, as mjlab's `ManagerBasedRlEnv.step` does it (mjlab/envs/manager_based_rl_env.py:340-416):

* one policy step = `DECIMATION` (4) physics steps of 5 ms with the actuator targets held (50 Hz policy);
* the observation is computed after `sim.forward()` (sensors from the current state, not one substep stale);
* actor observation = base_ang_vel (the `imu_ang_vel` gyro), projected_gravity (root body = pelvis frame), twist
  command, lean command, phase (sin, cos of 2 pi (k * 0.02 mod 0.86) / 0.86 with k the policy-step counter since the
  reset; zero while |command| < 0.1: unitree_rl_mjlab@1425b15 src/tasks/velocity/mdp/observations.py:47-53),
  joint_pos - default, joint_vel, last action (env.yaml observations.actor :257-399; noise off here);
* action -> targets: default + action_scale * action for the 14 legs + waist joints (env.yaml actions.joint_pos
  :670-694, use_default_offset); the arms are a scripted action term in training (`arms` :695-750) and are held at
  their default angles here;
* the training reset (env.yaml events reset_base / reset_robot_joints :751-834 with the ranges at 0): base at
  (0, 0, 0.745) upright, joints at R1MC_MANIP2_HOME (= `r1sim.joints.POSE_STAND`), zero velocity.

**Deployment differs from training in three rounded numbers.** The ONNX metadata that v2 (and our reference runner)
read rounds to 3 decimals: action_scale 0.178 / 0.398 (training 0.1779312722 / 0.3982750929: +0.04 % / -0.07 %),
kp 84.302 / 31.385 (training 84.30221295 / 31.38534200), kd 5.367 / 1.998 (5.36684556 / 1.99805293).
`PolicyInterface.training()` uses training's values (A1: the simulator alone); `.deployed()` the metadata's
(what runs on the robot).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from . import joints as J
from . import model as M

FloatArray = npt.NDArray[np.float64]

PHYSICS_DT = 0.005  # env.yaml sim.mujoco.timestep (:1497)
DECIMATION = 4  # env.yaml decimation (:1)
POLICY_DT = PHYSICS_DT * DECIMATION  # 0.02 s = ManagerBasedRlEnv.step_dt
GAIT_PERIOD = 0.86  # env.yaml observations.actor.phase.params.period; r1mc_policy.json "period"
TRAIN_BASE_POS = (0.0, 0.0, 0.745)  # env.yaml scene.entities.robot.init_state.pos (:89-94)
ARM_POLICY_JOINTS = tuple(J.JOINTS[i] for i in J.ARMS)

# A1 command scripts: (t0, t1, vx, vy, wz) over 10 s.
SCRIPTS: dict[str, tuple[tuple[float, float, float, float, float], ...]] = {
    "S0": ((0.0, 10.0, 0.0, 0.0, 0.0),),  # stand
    "S1": ((0.0, 2.0, 0.0, 0.0, 0.0), (2.0, 6.0, 0.3, 0.0, 0.0), (6.0, 10.0, 0.0, 0.0, 0.0)),  # stand, walk, stand
}


def command_at(script: str, t: float) -> FloatArray:
    """The twist command (vx, vy, wz) of `script` at time t."""
    for t0, t1, vx, vy, wz in SCRIPTS[script]:
        if t0 <= t < t1:
            return np.array([vx, vy, wz], dtype=np.float64)
    return np.zeros(3)


def _csv(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


@dataclass(frozen=True)
class PolicyInterface:
    """The policy's joints, default pose, action scale, gains and observation order (24 policy joints, no head)."""

    names: tuple[str, ...]  # SDK joint names in the policy's order (ONNX joint_names without "_joint")
    act_pos: npt.NDArray[np.int64]  # (14,) positions of the action joints within `names`
    default: FloatArray  # (24,)
    scale: FloatArray  # (14,)
    kp: FloatArray  # (24,)
    kd: FloatArray  # (24,)
    obs_names: tuple[str, ...]
    source: str  # "training" | "deployed"
    sdk_idx: npt.NDArray[np.int64] = field(init=False)  # (24,) SDK indices of `names`

    def __post_init__(self) -> None:
        object.__setattr__(self, "sdk_idx", np.array([J.IDX[n] for n in self.names], dtype=np.int64))

    @staticmethod
    def _parse(meta: Mapping[str, str]) -> tuple[tuple[str, ...], npt.NDArray[np.int64], tuple[str, ...]]:
        names = tuple(n.removesuffix("_joint") for n in _csv(meta["joint_names"]))
        acts = [n.removesuffix("_joint") for n in _csv(meta["action_joint_names"])]
        return names, np.array([names.index(a) for a in acts], dtype=np.int64), tuple(_csv(meta["observation_names"]))

    @classmethod
    def deployed(cls, meta: Mapping[str, str]) -> PolicyInterface:
        """Everything from the ONNX metadata (3-decimal rounding), as v2's WalkPolicy runs it on the robot."""
        names, act_pos, obs = cls._parse(meta)

        def f(key: str) -> FloatArray:
            return np.array([float(x) for x in _csv(meta[key])], dtype=np.float64)

        return cls(names, act_pos, f("default_joint_pos"), f("action_scale"), f("joint_stiffness"),
                   f("joint_damping"), obs, "deployed")  # fmt: skip

    @classmethod
    def training(cls, meta: Mapping[str, str]) -> PolicyInterface:
        """Training's exact values: the default pose (the metadata's 3 decimals are exact for it), gains from
        `model.TRAIN_PD` (= env.yaml) and action_scale = 0.25 * effort / kp (env.yaml actions.joint_pos.scale)."""
        names, act_pos, obs = cls._parse(meta)
        default = np.array([float(x) for x in _csv(meta["default_joint_pos"])], dtype=np.float64)
        pd = [M.TRAIN_PD[M.train_pd_group(n)] for n in names]
        kp = np.array([p[0] for p in pd])
        kd = np.array([p[1] for p in pd])
        scale = np.array([0.25 * pd[i][2] / pd[i][0] for i in act_pos])
        return cls(names, act_pos, default, scale, kp, kd, obs, "training")


def mjlab_observation(
    pi: PolicyInterface,
    *,
    gyro: FloatArray,
    quat: FloatArray,
    q: FloatArray,
    dq: FloatArray,
    cmd: FloatArray,
    lean: float,
    step: int,
    last_action: FloatArray,
) -> FloatArray:
    """The actor observation as mjlab computes it (module docstring); q, dq in the policy's joint order (24,),
    quat (w, x, y, z) of the root body, gyro of the imu site, `step` = policy steps since the reset."""
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat, dtype=np.float64))
    g = r.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    ph = (step * POLICY_DT) % GAIT_PERIOD / GAIT_PERIOD
    phase = np.array([np.sin(ph * np.pi * 2.0), np.cos(ph * np.pi * 2.0)])
    if np.linalg.norm(cmd) < 0.1:
        phase[:] = 0.0
    terms = {
        "base_ang_vel": np.asarray(gyro, dtype=np.float64),
        "projected_gravity": g,
        "command": np.asarray(cmd, dtype=np.float64),
        "lean": np.array([lean], dtype=np.float64),
        "phase": phase,
        "joint_pos": np.asarray(q, dtype=np.float64) - pi.default,
        "joint_vel": np.asarray(dq, dtype=np.float64),
        "actions": np.asarray(last_action, dtype=np.float64),
    }
    return np.concatenate([terms[n] for n in pi.obs_names])


def training_reset_qpos(m: mujoco.MjModel, *, touchdown: bool = False) -> FloatArray:
    """qpos of training's reset: base at TRAIN_BASE_POS upright, joints at POSE_STAND (head absent in "train" or 0),
    hands open (real profile), every other free body at its qpos0. No contact settling: in the train profile the soles
    start 10.9 mm ABOVE the floor (`foot_clearance`), so every training episode begins with a 1.1 cm drop; mjlab's
    first steps and ours are the same (A1). `touchdown=True` lowers the base until the lowest sole point is 1 mm
    above the floor (no drop)."""
    ix = M.RobotIndex(m)
    q = m.qpos0.copy()
    b = ix.base_qadr
    q[b : b + 3] = TRAIN_BASE_POS
    q[b + 3 : b + 7] = (1.0, 0.0, 0.0, 0.0)
    for i in range(J.N):
        if ix.present[i]:
            q[ix.qadr[i]] = J.POSE_STAND[i]
    from . import dex3

    for side, adr in ix.hand_qadr.items():
        q[adr] = dex3.preset("open", side)
    if touchdown:
        q[b + 2] -= foot_clearance(m, q) - 0.001
    return np.asarray(q, dtype=np.float64)


def foot_clearance(m: mujoco.MjModel, qpos: FloatArray) -> float:
    """Lowest foot-capsule point above z = 0 for this qpos (m; negative = inside the floor)."""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_kinematics(m, d)
    return M._lowest_foot_z(m, d, M.RobotIndex(m))


# ---------------------------------------------------------------- the A1 harness (our side)
Executor = Callable[[FloatArray], FloatArray]  # (74,) observation -> (14,) action


@dataclass
class Rollout:
    """Per policy step k (time k * 0.02 s, BEFORE the action of step k is applied)."""

    t: FloatArray
    q: FloatArray  # (n, 24) policy joint order
    dq: FloatArray
    base_pos: FloatArray  # (n, 3)
    base_quat: FloatArray  # (n, 4)
    action: FloatArray  # (n, 14)
    obs: FloatArray  # (n, 74)
    cmd: FloatArray  # (n, 3)

    def arrays(self) -> dict[str, FloatArray]:
        return {k: getattr(self, k) for k in ("t", "q", "dq", "base_pos", "base_quat", "action", "obs", "cmd")}


def default_sim_factory(m: mujoco.MjModel, index: M.RobotIndex) -> Any:
    """The engine's Simulation with the train-equivalent driver (mjlab's <position> law, no delays)."""
    from .driver import DriverConfig
    from .engine import Simulation

    return Simulation(m, index, driver=DriverConfig.train())


def run_policy(
    executor: Executor,
    pi: PolicyInterface,
    *,
    script: str = "S0",
    seconds: float = 10.0,
    m: mujoco.MjModel | None = None,
    sim_factory: Callable[[mujoco.MjModel, M.RobotIndex], Any] = default_sim_factory,
    qpos0: FloatArray | None = None,
) -> Rollout:
    """The policy on the train-equivalent profile (robot + floor) from training's reset, mjlab's step structure:
    the observation at each policy step (lockstep controller every DECIMATION physics steps), legs + waist targets
    from the action, arms at default, the PolicyInterface's gains."""
    from .engine import LowCmd, LowState

    if m is None:
        m = M.build_model("train", visual=False)
    ix = M.RobotIndex(m)
    sim = sim_factory(m, ix)
    if abs(sim.dt - PHYSICS_DT) > 1e-12:
        raise ValueError(f"the train-equivalent profile steps at {PHYSICS_DT} s, this model at {sim.dt}")
    sim.reset(training_reset_qpos(m) if qpos0 is None else qpos0)
    n = round(seconds / POLICY_DT)
    rec: dict[str, list[FloatArray]] = {
        k: [] for k in ("t", "q", "dq", "base_pos", "base_quat", "action", "obs", "cmd")
    }
    last = np.zeros(len(pi.act_pos))
    k = 0

    def policy_step(s: LowState) -> LowCmd:
        nonlocal last, k
        q = s.q[pi.sdk_idx]
        dq = s.dq[pi.sdk_idx]
        cmd = command_at(script, k * POLICY_DT)
        obs = mjlab_observation(pi, gyro=s.gyro, quat=s.base_quat, q=q, dq=dq, cmd=cmd, lean=0.0, step=k,
                                last_action=last)  # fmt: skip
        a = np.asarray(executor(obs), dtype=np.float64).reshape(-1)
        for key, v in (("t", np.array(k * POLICY_DT)), ("q", q), ("dq", dq), ("base_pos", s.base_pos),
                       ("base_quat", s.base_quat), ("action", a), ("obs", obs), ("cmd", cmd)):  # fmt: skip
            rec[key].append(np.array(v, dtype=np.float64))
        last = a
        target = pi.default.copy()
        target[pi.act_pos] = pi.default[pi.act_pos] + pi.scale * a
        c = LowCmd()
        c.q[pi.sdk_idx] = target
        c.kp[pi.sdk_idx] = pi.kp
        c.kd[pi.sdk_idx] = pi.kd
        k += 1
        return c

    sim.add_controller(policy_step, POLICY_DT, "policy")
    sim.step(n * DECIMATION)
    return Rollout(**{key: np.array(v) for key, v in rec.items()})


# ---------------------------------------------------------------- A1 metrics
def tilt_deg(quat: FloatArray) -> FloatArray:
    """Angle between the body z axis and world z for (n, 4) quaternions (w, x, y, z)."""
    qq = np.atleast_2d(quat)
    c = 1.0 - 2.0 * (qq[:, 1] ** 2 + qq[:, 2] ** 2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


@dataclass(frozen=True)
class A1Numbers:
    joint_rms: FloatArray  # (24,) rad, per joint over the run
    joint_rms_max: float
    joint_err_max: float  # rad, largest |dq| at any step
    tilt_rms_diff_deg: float  # RMS of the tilt difference over the run
    first_divergence_s: float | None  # first time any joint differs by > 0.02 rad
    fell_ours: bool
    fell_ref: bool
    names: tuple[str, ...]

    def table(self) -> str:
        worst = np.argsort(self.joint_rms)[::-1][:4]
        w = ", ".join(f"{self.names[i]} {self.joint_rms[i]:.4f}" for i in worst)
        fd = "none" if self.first_divergence_s is None else f"{self.first_divergence_s:.2f} s"
        return (f"joint RMS max {self.joint_rms_max:.4f} rad (target <= 0.02; worst: {w}); max |error| "
                f"{self.joint_err_max:.4f} rad; tilt RMS diff {self.tilt_rms_diff_deg:.3f} deg (target <= 0.5); "
                f"first > 0.02 rad: {fd}; fell: ours {self.fell_ours}, ref {self.fell_ref}")  # fmt: skip


def a1_numbers(ours: Mapping[str, Any], ref: Mapping[str, Any], names: Sequence[str]) -> A1Numbers:
    """Compare two rollouts (same length, policy-step samples)."""
    n = min(len(ours["t"]), len(ref["t"]))
    e = np.asarray(ours["q"])[:n] - np.asarray(ref["q"])[:n]
    rms = np.sqrt(np.mean(e**2, axis=0))
    ta, tb = tilt_deg(np.asarray(ours["base_quat"])[:n]), tilt_deg(np.asarray(ref["base_quat"])[:n])
    big = np.flatnonzero(np.abs(e).max(axis=1) > 0.02)
    return A1Numbers(
        joint_rms=rms,
        joint_rms_max=float(rms.max()),
        joint_err_max=float(np.abs(e).max()),
        tilt_rms_diff_deg=float(np.sqrt(np.mean((ta - tb) ** 2))),
        first_divergence_s=float(np.asarray(ours["t"])[big[0]]) if big.size else None,
        fell_ours=bool(ta.max() > 60.0),
        fell_ref=bool(tb.max() > 60.0),
        names=tuple(names),
    )


A1_ASSETS = Path(__file__).resolve().parents[2] / "tests" / "assets" / "a1"


def settled_initial_state(
    m: mujoco.MjModel | None = None, executor: Executor | None = None, *, seconds: float = 2.0
) -> tuple[FloatArray, FloatArray]:
    """(qpos, qvel) after the live policy has stood (command 0) for `seconds` from training's reset on the
    train-equivalent model: the 1.1 cm drop of the reset is over, the soles rest on the floor and stay put (they move
    9.6 mm sideways/forwards while landing in the first 0.1 s, then < 0.1 mm). "Training reset" = what every training
    episode starts from (`training_reset_qpos`); "settled" = this. The policy cannot be replaced by a plain PD hold
    here: POSE_STAND falls backwards under plain PD (S1), and the S1 stiff stand hold (KP_STIFF 200) makes the feet
    chatter at the train profile's 5 ms step (contacts flicker 28 <-> 11 per 40 ms) and creep 2.5 cm forwards in
    0.25 s - the "2.5 cm sole slip" of S1 was that numerical chatter, not an unsettled contact."""
    if executor is None:
        from r1mc3.policy import POLICY_DIR
        from r1mc3.policy.onnx_np import OnnxModel

        executor = OnnxModel.load(POLICY_DIR / "policy.onnx")
    meta = getattr(executor, "meta", None)
    if meta is None:
        raise ValueError("the executor needs the ONNX metadata (.meta) for the policy interface")
    if m is None:
        m = M.build_model("train", visual=False)
    ix = M.RobotIndex(m)
    holder: dict[str, Any] = {}

    def factory(mm: mujoco.MjModel, ii: M.RobotIndex) -> Any:
        holder["sim"] = default_sim_factory(mm, ii)
        return holder["sim"]

    run_policy(executor, PolicyInterface.training(meta), script="S0", seconds=seconds, m=m, sim_factory=factory)
    d = holder["sim"].d
    del ix
    return np.asarray(d.qpos, dtype=np.float64).copy(), np.asarray(d.qvel, dtype=np.float64).copy()


# ---------------------------------------------------------------- model fingerprint (tools/sim2sim/model_diff.py)
OPT_FIELDS = (
    "timestep", "integrator", "solver", "iterations", "ls_iterations", "tolerance", "ls_tolerance",
    "noslip_iterations", "noslip_tolerance", "ccd_iterations", "ccd_tolerance", "cone", "impratio", "jacobian",
    "gravity", "wind", "magnetic", "density", "viscosity", "o_margin", "o_solref", "o_solimp", "o_friction",
    "disableflags", "enableflags", "sdf_iterations", "sdf_initpoints",
)  # fmt: skip
_PER_OBJECT: dict[str, tuple[str, ...]] = {
    "body": ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "body_gravcomp"),
    "joint": ("jnt_type", "jnt_range", "jnt_limited", "jnt_stiffness", "jnt_margin", "jnt_solref", "jnt_solimp",
              "jnt_pos", "jnt_axis"),
    "dof": ("dof_armature", "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp"),
    "actuator": ("actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype",
                 "actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_ctrllimited",
                 "actuator_forcelimited", "actuator_ctrlrange", "actuator_forcerange", "actuator_gear",
                 "actuator_actlimited"),
    "geom": ("geom_type", "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority", "geom_friction",
             "geom_solref", "geom_solimp", "geom_solmix", "geom_margin", "geom_gap", "geom_size", "geom_pos",
             "geom_quat"),
    "sensor": ("sensor_type", "sensor_objtype", "sensor_noise", "sensor_cutoff"),
}  # fmt: skip
_OBJ = {"body": mujoco.mjtObj.mjOBJ_BODY, "joint": mujoco.mjtObj.mjOBJ_JOINT, "dof": mujoco.mjtObj.mjOBJ_JOINT,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR, "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "sensor": mujoco.mjtObj.mjOBJ_SENSOR}  # fmt: skip


def _norm_name(kind: str, n: str) -> str:
    """Names compared across models: mjlab prefixes the robot's elements with "robot/", names each actuator after its
    joint (r1sim: the SDK name) and calls its plane "terrain" (r1sim: "floor")."""
    n = n.removeprefix("robot/")
    if kind == "actuator":
        n = n.removesuffix("_joint")
    return "floor" if n == "terrain" else n


def model_fields(m: mujoco.MjModel) -> dict[str, FloatArray]:
    """A name-keyed fingerprint of every physics-relevant field ("opt.<f>", "<kind>/<name>/<field>", "size.<f>").
    mjlab's plane body ("terrain": static, massless) is skipped: r1sim's plane sits in the world body."""
    out: dict[str, FloatArray] = {}
    for f in OPT_FIELDS:
        out[f"opt.{f}"] = np.atleast_1d(np.asarray(getattr(m.opt, f), dtype=np.float64))
    counts = {"body": m.nbody, "joint": m.njnt, "dof": m.njnt, "actuator": m.nu, "geom": m.ngeom, "sensor": m.nsensor}
    for kind, fields in _PER_OBJECT.items():
        for i in range(counts[kind]):
            name = _norm_name(kind, mujoco.mj_id2name(m, _OBJ[kind], i) or "")
            if kind == "body" and (i == 0 or name == "floor"):
                continue
            j = int(m.jnt_dofadr[i]) if kind == "dof" else i
            for f in fields:
                out[f"{kind}/{name}/{f}"] = np.atleast_1d(np.asarray(getattr(m, f)[j], dtype=np.float64))
    for f in ("nq", "nv", "nu", "neq", "nkey"):
        out[f"size.{f}"] = np.array([float(getattr(m, f))])
    return out


def save_fields(path: str | Path, fields: Mapping[str, FloatArray]) -> None:
    """Compact .npz of a fingerprint: the keys, the lengths and one concatenated float64 vector."""
    keys = sorted(fields)
    vals = [np.asarray(fields[k], dtype=np.float64).ravel() for k in keys]
    np.savez_compressed(path, keys=np.array(keys), lengths=np.array([len(v) for v in vals]),
                        values=np.concatenate(vals))  # fmt: skip


def load_fields(path: str | Path) -> dict[str, FloatArray]:
    z = np.load(path)
    out: dict[str, FloatArray] = {}
    i = 0
    for k, n in zip(z["keys"].tolist(), z["lengths"].tolist(), strict=True):
        out[str(k)] = z["values"][i : i + int(n)].copy()
        i += int(n)
    return out


def diff_fields(ref: Mapping[str, FloatArray], ours: Mapping[str, FloatArray], tol: float = 0.0,
                *, collision_geoms_only: bool = False) -> list[str]:  # fmt: skip
    """Every differing entry as a line "key: ref -> ours" (relative tolerance `tol`, 0 = bit-exact)."""

    def collides(fp: Mapping[str, FloatArray], key: str) -> bool:
        if not key.startswith("geom/"):
            return True
        base = key.rsplit("/", 1)[0]
        return bool(
            fp.get(f"{base}/geom_contype", np.zeros(1))[0] or fp.get(f"{base}/geom_conaffinity", np.zeros(1))[0]
        )

    keep_r = {k for k in ref if not collision_geoms_only or collides(ref, k)}
    keep_o = {k for k in ours if not collision_geoms_only or collides(ours, k)}
    lines = [f"{k}: only in the reference" for k in sorted(keep_r - keep_o)]
    lines += [f"{k}: only in ours" for k in sorted(keep_o - keep_r)]
    for k in sorted(keep_r & keep_o):
        a, b = np.asarray(ref[k]), np.asarray(ours[k])
        if a.shape != b.shape or np.any(np.abs(a - b) > tol * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))):
            lines.append(f"{k}: {np.round(a, 12).tolist()} -> {np.round(b, 12).tolist()}")
    return lines


#: Differences between mjlab's compiled training model and the train profile that are understood (no physics effect).
EXPLAINED_DIFFS: dict[str, str] = {
    "geom/floor/geom_size": "plane size[2] is the rendering grid spacing (mjlab 0.01, r1sim 0.05); planes are infinite",
    "size.nkey": "mjlab stores the reset state as keyframe 'init_state'; r1sim gives it as training_reset_qpos()",
}
