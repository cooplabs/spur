"""mjlab reference for the A1 sim2sim check (DESIGN.md §13.1 A1): the training model as mjlab 1.2.0 builds it, and
reference rollouts of the live policy on it.

Run with the TRAINING venv's python (read-only use, nothing is installed; it has mjlab 1.2.0, mujoco + mujoco-warp
3.5.0, warp 1.12.0, torch, onnx):

    ~/.venvs/r1-train/bin/python tools/sim2sim/mjlab_reference.py model  --out /tmp/x/mjlab_r1.mjb
    ~/.venvs/r1-train/bin/python tools/sim2sim/mjlab_reference.py rollout --engine cpu|warp \
        --out tests/assets/a1/mjlab_cpu_S0.npz --script S0

How the model is built (no unitree_rl_mjlab code is executed - its checkout was gone and downloaded code is not run):
* the R1 spec = `mujoco.MjSpec.from_file(r1.xml)` exactly as unitree_rl_mjlab@1425b15
  src/assets/robots/unitree_r1/r1_constants.py:get_spec does (our models/r1/r1_mjlab.xml is that file, sha256
  19c2291c...), with the meshes from models/r1/meshes;
* + the rigid Dex3 lumps: a port of the training kit's get_spec_with_hands (v2 training/r1mc_task/assets/
  r1mc_manip_constants.py:51-71, git r21_try 2a4a165; our own code);
* then mjlab's OWN classes do the rest: `EntityCfg` with the actuators, collision config and initial state copied
  from the policy's params/env.yaml (scene.entities.robot), `TerrainEntityCfg(terrain_type="plane")`, `SceneCfg` /
  `Scene` (attach with the "robot/" prefix), `Scene.compile()`, and `MujocoCfg(**env.yaml sim.mujoco).apply(model)`.

Rollouts (`rollout`): mjlab's env step structure (ManagerBasedRlEnv: action -> decimation x (write ctrl, physics
step) -> observations), 4 physics steps of 5 ms per 20 ms policy step; legs + waist targets = default +
action_scale * action, arm actuators held at their default angles (the scripted arm term off); observations as
mjlab's mdp functions compute them (imu_ang_vel gyro sensor, the root body's projected gravity, joint_pos - default,
joint_vel, last action, twist command, lean command, phase from the policy-step counter, zeroed when |cmd| < 0.1);
noise, delays, domain randomisation, pushes and hand forces OFF. engine "cpu" = MuJoCo CPU 3.5.0 (mj_step) on that
model; engine "warp" = MuJoCo Warp 3.5.0 on Warp's CPU device through mjlab's own `Simulation` class (the training
engine). The observation code (r1sim.train_equiv.mjlab_observation) and the policy executor (r1mc3.policy.onnx_np,
bit-identical to v2's NumPy executor) are the SAME on both sides, so A1 measures the simulator alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "sim"), str(REPO / "python")]
POLICY_DIR = REPO / "policies" / "r1mc2_manip2_lean_20261006T0421Z"
R1_XML = REPO / "models" / "r1" / "r1_mjlab.xml"
MESH_DIR = REPO / "models" / "r1" / "meshes"

# Training kit get_spec_with_hands constants (v2 training r1mc_manip_constants.py:22-26).
HAND_MASS = 0.697
HAND_COM = (0.0667, 0.008, 0.0)
MOUNT_X = 0.075
FIST_MASS = 0.143
FOREARM_COM_X = 0.035


def _env_yaml() -> dict:
    class _Loader(yaml.SafeLoader):
        pass

    def _any(loader, suffix, node):  # python tags -> plain data
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node)

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _any)
    _Loader.add_multi_constructor("!!python/", _any)
    with open(POLICY_DIR / "params" / "env.yaml") as f:
        return yaml.load(f, Loader=_Loader)


def get_spec_with_hands() -> mujoco.MjSpec:
    """r1_constants.get_spec (MjSpec.from_file(r1.xml)) + the training kit's rigid Dex3 lumps."""
    spec = mujoco.MjSpec.from_file(str(R1_XML))
    spec.meshdir = str(MESH_DIR)
    for side, g in (("left", -1.0), ("right", 1.0)):
        wr = spec.body(f"{side}_wrist_roll_link")
        m0 = float(wr.mass)
        m1 = m0 - FIST_MASS
        ip = list(wr.ipos)
        wr.ipos = [FOREARM_COM_X, ip[1], ip[2]]
        wr.inertia = [float(v) * m1 / m0 for v in wr.inertia]
        wr.mass = m1
        hand = wr.add_body(name=f"{side}_dex3_hand", pos=[MOUNT_X, 0.0, 0.0])
        hand.mass = HAND_MASS
        hand.ipos = [HAND_COM[0], g * HAND_COM[1], HAND_COM[2]]
        hand.inertia = [6.0e-4, 1.4e-3, 1.2e-3]
        hand.explicitinertial = True
        hand.add_geom(
            name=f"{side}_dex3_collision",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.015, 0.0, 0.0, 0.15, g * 0.01, 0.0],
            size=[0.035, 0, 0],
        )
    return spec


def build_scene(device: str = "cpu"):  # -> (mjlab Scene, MjModel, SimulationCfg)
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains.terrain_entity import TerrainEntityCfg
    from mjlab.utils.spec_config import CollisionCfg

    env = _env_yaml()
    rob = env["scene"]["entities"]["robot"]
    acts = []
    for a in rob["articulation"]["actuators"]:
        acts.append(
            BuiltinPositionActuatorCfg(
                target_names_expr=tuple(a["target_names_expr"]),
                stiffness=float(a["stiffness"]),
                damping=float(a["damping"]),
                effort_limit=float(a["effort_limit"]),
                armature=float(a["armature"]),
                frictionloss=float(a["frictionloss"]),
            )
        )
    art = EntityArticulationInfoCfg(
        actuators=tuple(acts), soft_joint_pos_limit_factor=float(rob["articulation"]["soft_joint_pos_limit_factor"])
    )
    c = rob["collisions"][0]
    coll = CollisionCfg(
        geom_names_expr=tuple(c["geom_names_expr"]),
        contype=c["contype"],
        conaffinity=c["conaffinity"],
        condim=dict(c["condim"]),
        priority=dict(c["priority"]),
        friction={k: tuple(v) for k, v in c["friction"].items()},
        solref=c["solref"],
        solimp=c["solimp"],
        disable_other_geoms=bool(c["disable_other_geoms"]),
    )
    ist = rob["init_state"]
    init = EntityCfg.InitialStateCfg(
        pos=tuple(float(v) for v in ist["pos"]),
        rot=tuple(float(v) for v in ist["rot"]),
        joint_pos={k: float(v) for k, v in ist["joint_pos"].items()},
        joint_vel={k: float(v) for k, v in ist["joint_vel"].items()},
    )
    robot = EntityCfg(init_state=init, collisions=(coll,), spec_fn=get_spec_with_hands, articulation=art)
    scfg = SceneCfg(
        num_envs=1, env_spacing=2.0, terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": robot}
    )
    scene = Scene(scfg, device=device)
    m = scene.compile()
    mj = env["sim"]["mujoco"]
    mcfg = MujocoCfg(
        timestep=float(mj["timestep"]),
        integrator=mj["integrator"],
        impratio=float(mj["impratio"]),
        cone=mj["cone"],
        jacobian=mj["jacobian"],
        solver=mj["solver"],
        iterations=int(mj["iterations"]),
        tolerance=float(mj["tolerance"]),
        ls_iterations=int(mj["ls_iterations"]),
        ls_tolerance=float(mj["ls_tolerance"]),
        ccd_iterations=int(mj["ccd_iterations"]),
        gravity=tuple(float(v) for v in mj["gravity"]),
        multiccd=bool(mj["multiccd"]),
    )
    mcfg.apply(m)
    simcfg = SimulationCfg(njmax=env["sim"]["njmax"], nconmax=env["sim"]["nconmax"], mujoco=mcfg)
    return scene, m, simcfg


# ---------------------------------------------------------------- rollouts
def rollout(engine: str, script: str, seconds: float = 10.0, interface: str = "training") -> dict[str, np.ndarray]:
    """The live policy on mjlab's model from training's reset (the model's init_state keyframe)."""
    from r1mc3.policy.onnx_np import OnnxModel
    from r1sim import train_equiv as te

    _, m, simcfg = build_scene("cpu")
    net = OnnxModel.load(POLICY_DIR / "policy.onnx")
    pi = te.PolicyInterface.training(net.meta) if interface == "training" else te.PolicyInterface.deployed(net.meta)
    pre = "robot/"
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{pre}{n}_joint") for n in pi.names]
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    vadr = np.array([m.jnt_dofadr[j] for j in jid])
    act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{pre}{n}_joint") for n in pi.names])
    free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    bq = int(m.jnt_qposadr[free])
    gyro_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"{pre}imu_ang_vel")])
    # the deployed interface also sends its own (rounded) gains, as the robot's board would apply them
    m.actuator_gainprm[act, 0] = pi.kp
    m.actuator_biasprm[act, 1] = -pi.kp
    m.actuator_biasprm[act, 2] = -pi.kd
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("init_state").id)
    mujoco.mj_forward(m, d)
    sim = None
    if engine == "warp":
        import torch
        from mjlab.sim.sim import Simulation

        sim = Simulation(num_envs=1, cfg=simcfg, model=m, device="cpu")
        sim.data.qpos[:] = torch.tensor(d.qpos, dtype=sim.data.qpos.dtype).unsqueeze(0)
        sim.data.qvel[:] = 0.0

        def read() -> None:
            sim.forward()  # as ManagerBasedRlEnv.step does before the observations (sensors from the current state)
            d.qpos[:] = sim.data.qpos[0].cpu().numpy()
            d.qvel[:] = sim.data.qvel[0].cpu().numpy()
            d.sensordata[:] = sim.data.sensordata[0].cpu().numpy()

        read()
    n_steps = round(seconds / te.POLICY_DT)
    last = np.zeros(len(pi.act_pos))
    keys = ("t", "q", "dq", "base_pos", "base_quat", "action", "obs", "cmd")
    rec: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for k in range(n_steps):
        t = k * te.POLICY_DT
        cmd = te.command_at(script, t)
        q, dq = d.qpos[qadr].copy(), d.qvel[vadr].copy()
        quat = d.qpos[bq + 3 : bq + 7].copy()
        obs = te.mjlab_observation(pi, gyro=d.sensordata[gyro_adr : gyro_adr + 3].copy(), quat=quat, q=q, dq=dq,
                                   cmd=cmd, lean=0.0, step=k, last_action=last)  # fmt: skip
        a = np.asarray(net(obs), dtype=np.float64)
        for key, v in (("t", np.array(t)), ("q", q), ("dq", dq), ("base_pos", d.qpos[bq : bq + 3].copy()),
                       ("base_quat", quat), ("action", a), ("obs", obs), ("cmd", cmd)):  # fmt: skip
            rec[key].append(v)
        last = a
        target = pi.default.copy()
        target[pi.act_pos] = pi.default[pi.act_pos] + pi.scale * a
        if engine == "cpu":
            d.ctrl[act] = target
            for _ in range(te.DECIMATION):
                mujoco.mj_step(m, d)
            # ManagerBasedRlEnv.step calls sim.forward() before the observations: mj_step leaves the sensors one
            # substep stale. mj_forward does not change the trajectory (the integrator saves the warm start).
            mujoco.mj_forward(m, d)
        else:
            import torch

            assert sim is not None
            sim.data.ctrl[0, torch.as_tensor(act)] = torch.tensor(target, dtype=sim.data.ctrl.dtype)
            for _ in range(te.DECIMATION):
                sim.step()
            read()
    out = {k: np.array(v) for k, v in rec.items()}
    out["joint_names"] = np.array(pi.names)
    out["engine"] = np.array(engine)
    out["script"] = np.array(script)
    out["interface"] = np.array(interface)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["model", "rollout"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", choices=["cpu", "warp"], default="cpu")
    ap.add_argument("--script", choices=["S0", "S1"], default="S0")
    ap.add_argument("--interface", choices=["training", "deployed"], default="training")
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.what == "model":
        scene, m, _ = build_scene("cpu")
        mujoco.mj_saveModel(m, str(out))
        out.with_suffix(".xml").write_text(scene.spec.to_xml())
        print(f"saved {out} (nq {m.nq}, nu {m.nu}, ngeom {m.ngeom}, mass {m.body_subtreemass[1]:.4f})")
        return 0
    res = rollout(a.engine, a.script, a.seconds, a.interface)
    keep = ("t", "q", "base_pos", "base_quat", "action", "cmd", "joint_names", "engine", "script", "interface")
    compact = {k: res[k] for k in keep}
    compact["obs_head"] = res["obs"][:5]  # the first observations, for debugging an observation mismatch
    np.savez_compressed(out, **compact)
    print(f"saved {out}: {len(res['t'])} policy steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
