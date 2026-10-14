"""A1 sim2sim (DESIGN.md §13.1): the train-equivalent profile against mjlab 1.2.0's own training model and its
reference rollouts (tests/assets/a1, made by tools/sim2sim/mjlab_reference.py in the training venv).

What is asserted: the compiled model equals mjlab's field by field (explained differences only); on the same engine
(MuJoCo CPU 3.5.0) the policy's 10 s rollouts match mjlab's to 1e-9 rad; nothing falls. The A1 targets against
mjlab on MuJoCo Warp (per-joint RMS <= 0.02 rad, tilt RMS <= 0.5 deg) are REPORTED (printed), not asserted, in
session 2 (gate GA is session 4).
"""

from __future__ import annotations

import platform
import re
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
import yaml
from r1mc3.policy import POLICY_DIR
from r1mc3.policy.onnx_np import OnnxModel
from r1mc3.policy.reference import ReferencePolicy
from r1sim import model as M
from r1sim import train_equiv as te

ASSETS = te.A1_ASSETS
ENV_YAML = POLICY_DIR / "params" / "env.yaml"


@pytest.fixture(scope="module")
def net() -> OnnxModel:
    return OnnxModel.load(POLICY_DIR / "policy.onnx")


@pytest.fixture(scope="module")
def train_model() -> mujoco.MjModel:
    return M.build_model("train", visual=False)


def _ref(engine: str, script: str) -> dict[str, Any]:
    return dict(np.load(ASSETS / f"mjlab_{engine}_{script}.npz"))


# ---------------------------------------------------------------- the model
def test_model_equals_mjlab_up_to_explained_differences() -> None:
    """Fails on the S1 model (torque motors, collision priority 1, ccd_iterations 35)."""
    ref = te.load_fields(ASSETS / "mjlab_model_fields.npz")
    ours = te.model_fields(M.build_model("train"))
    lines = te.diff_fields(ref, ours, tol=1e-12)
    unexplained = [ln for ln in lines if ln.split(":", 1)[0] not in te.EXPLAINED_DIFFS]
    assert not unexplained, "\n".join(unexplained)


def test_model_bit_identical_to_mjlab() -> None:
    """Tolerance 0: failed (strict xfail) while model._remove_fist wrote the wrist mass / inertia with 9 significant
    digits (1 ulp off mjlab's float); passes since it writes them with repr() (session 2, integrator patch)."""
    ref = te.load_fields(ASSETS / "mjlab_model_fields.npz")
    # Bit-exact on arm64, where the reference was made (macOS / Linux, as the robot's Orin). On x86_64 MuJoCo's compiler
    # rounds a few derived fields (inertia frames, mesh-fitted geoms) differently in the last bits (the public CI's
    # ubuntu-latest, 14 Oct 2026): a 1e-9 relative tolerance there.
    tol = 0.0 if platform.machine() in ("arm64", "aarch64") else 1e-9
    lines = te.diff_fields(ref, te.model_fields(M.build_model("train")), tol=tol)
    assert all(ln.split(":", 1)[0] in te.EXPLAINED_DIFFS for ln in lines), "\n".join(lines)


_ACT_RE = re.compile(
    r"target_names_expr: !!python/tuple\n((?:\s+- .+\n)+)(?:.|\n)*?armature: (\S+)\n\s+frictionloss: (\S+)\n\s+"
    r"stiffness: (\S+)\n\s+damping: (\S+)\n\s+effort_limit: (\S+)"
)


def _env_yaml_actuators() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in _ACT_RE.findall(ENV_YAML.read_text()):
        vals = [float(x) for x in b[1:]]
        out.append(dict(zip(("armature", "frictionloss", "kp", "kv", "effort"), vals, strict=True)))
        out[-1]["names"] = re.findall(r"- (\S+)", b[0])
    return out


def test_train_actuators_are_mjlab_position_actuators(train_model: mujoco.MjModel) -> None:
    """Every actuated joint: <position> with env.yaml's exact gains, effort and joint armature / frictionloss."""
    acts = _env_yaml_actuators()
    assert len(acts) == 6
    m = train_model
    checked = 0
    for a in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        cfg = [c for c in acts if any(re.fullmatch(p, name + "_joint") for p in c["names"])]
        assert len(cfg) == 1, name
        c = cfg[0]
        assert m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED and m.actuator_gainprm[a, 0] == c["kp"], name
        assert m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE, name
        assert tuple(m.actuator_biasprm[a, :3]) == (0.0, -c["kp"], -c["kv"]), name
        assert not m.actuator_ctrllimited[a] and m.actuator_forcelimited[a], name
        assert tuple(m.actuator_forcerange[a]) == (-c["effort"], c["effort"]), name
        dof = m.jnt_dofadr[m.actuator_trnid[a, 0]]
        assert (m.dof_armature[dof], m.dof_frictionloss[dof]) == (c["armature"], c["frictionloss"]), name
        checked += 1
    assert checked == 24


def test_provenance_records_the_train_gains() -> None:
    p = yaml.safe_load((M.models_dir() / "provenance.yaml").read_text())["parameters"]
    g = p["train_position_actuator_gains"]["value"]
    assert tuple(g["legs_waist"]) == M.TRAIN_PD["leg"] == M.TRAIN_PD["waist"]
    assert tuple(g["ankles"]) == M.TRAIN_PD["ankle"]
    assert tuple(g["shoulder_pitch_roll"]) == M.TRAIN_PD["shoulder_pr"]
    assert tuple(g["shoulder_yaw_elbow"]) == M.TRAIN_PD["shoulder_yaw_elbow"]
    assert tuple(g["wrist_roll"]) == M.TRAIN_PD["wrist"]
    opts = p["train_physics_options"]["value"]
    assert {k: str(v) for k, v in opts.items()} == M.OPTIONS["train"]


# ---------------------------------------------------------------- the policy interface and observations
def test_training_vs_deployed_interface(net: OnnxModel) -> None:
    """Training's action scale (env.yaml) vs the ONNX metadata's 3-decimal one (what v2 and our runner use)."""
    tr, de = te.PolicyInterface.training(net.meta), te.PolicyInterface.deployed(net.meta)
    actions = ENV_YAML.read_text().split("\nactions:\n")[1]
    mleg = re.search(r"\.\*_hip_pitch\.\*: (\S+)\n", actions)
    mank = re.search(r"\.\*_ankle_pitch\.\*: (\S+)\n", actions)
    assert mleg and mank
    leg, ank = float(mleg.group(1)), float(mank.group(1))
    assert set(tr.scale.tolist()) == {leg, ank}
    assert np.array_equal(tr.default, de.default) and tr.names == de.names
    assert np.allclose(tr.scale, de.scale, rtol=1e-3) and not np.array_equal(tr.scale, de.scale)
    assert np.allclose(tr.kp, de.kp, rtol=1e-4) and np.allclose(tr.kd, de.kd, rtol=1e-3)


def test_reference_policy_builds_mjlabs_observation(net: OnnxModel) -> None:
    """v2's observation code (our ReferencePolicy port) = mjlab's terms for the same state, incl. the phase while
    walking and zeroed while standing - so the deployed runner observes what training observed."""
    pol = ReferencePolicy.load(POLICY_DIR)
    pi = te.PolicyInterface.deployed(net.meta)
    rng = np.random.default_rng(3)
    for k in (0, 7, 43, 200):
        q = pi.default + rng.uniform(-0.2, 0.2, 24)
        dq = rng.uniform(-2, 2, 24)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        gyro = rng.uniform(-1, 1, 3)
        last = rng.uniform(-1, 1, 14)
        for cmd in (np.zeros(3), np.array([0.3, 0.0, 0.0])):
            pol.phase_t = k * te.POLICY_DT
            pol.last_action = last
            ours = pol.observe(q, dq, quat, gyro, cmd)
            ref = te.mjlab_observation(pi, gyro=gyro, quat=quat, q=q, dq=dq, cmd=cmd, lean=0.0, step=k,
                                       last_action=last)  # fmt: skip
            assert np.allclose(ours, ref, rtol=0, atol=1e-12), (k, cmd, np.abs(ours - ref).max())


# ---------------------------------------------------------------- rollouts
@pytest.mark.parametrize("script", ["S0", "S1"])
def test_policy_rollout_matches_mjlab_on_mujoco_cpu(net: OnnxModel, train_model: mujoco.MjModel, script: str) -> None:
    """Same engine as the reference: the 10 s rollouts agree to rounding (1e-12 observed; 0 with the fist patch)."""
    ref = _ref("cpu", script)
    ours = te.run_policy(net, te.PolicyInterface.training(net.meta), script=script, m=train_model).arrays()
    n = te.a1_numbers(ours, ref, tuple(str(x) for x in ref["joint_names"]))
    print(f"\nA1 {script} ours vs mjlab MuJoCo-CPU: {n.table()}")
    assert len(ours["t"]) == len(ref["t"]) == 500
    assert n.joint_err_max < 1e-9 and n.tilt_rms_diff_deg < 1e-6
    assert not n.fell_ours and not n.fell_ref


@pytest.mark.parametrize("script", ["S0", "S1"])
def test_a1_numbers_vs_mjlab_warp_reported(net: OnnxModel, train_model: mujoco.MjModel, script: str) -> None:
    """Against the training engine (MuJoCo Warp, float32): the A1 numbers, printed; only "no fall" is asserted."""
    ref = _ref("warp", script)
    ours = te.run_policy(net, te.PolicyInterface.training(net.meta), script=script, m=train_model).arrays()
    n = te.a1_numbers(ours, ref, tuple(str(x) for x in ref["joint_names"]))
    print(f"\nA1 {script} ours vs mjlab MuJoCo-Warp: {n.table()}")
    assert np.isfinite(n.joint_rms).all()
    assert not n.fell_ours and not n.fell_ref


def test_training_reset_drops_onto_the_floor(train_model: mujoco.MjModel) -> None:
    q0 = te.training_reset_qpos(train_model)
    assert q0[2] == 0.745
    assert te.foot_clearance(train_model, q0) == pytest.approx(0.0109, abs=0.0005)
    assert te.foot_clearance(train_model, te.training_reset_qpos(train_model, touchdown=True)) == pytest.approx(0.001)


def test_settled_state_rests_on_the_floor(net: OnnxModel, train_model: mujoco.MjModel) -> None:
    """After 2 s of standing under the policy: soles on the floor, the body at rest, and it stays put."""
    qpos, qvel = te.settled_initial_state(train_model, net)
    assert abs(te.foot_clearance(train_model, qpos)) < 0.002
    assert np.abs(qvel).max() < 0.05
    ix = M.RobotIndex(train_model)
    feet: list[np.ndarray] = []

    def factory(m: mujoco.MjModel, i: M.RobotIndex) -> Any:
        sim = te.default_sim_factory(m, i)
        orig = sim.step

        def step(n: int = 1) -> None:  # record the foot sites around the run
            feet.append(np.array([sim.d.site_xpos[s][:2].copy() for s in ix.foot_sites]))
            orig(n)
            feet.append(np.array([sim.d.site_xpos[s][:2].copy() for s in ix.foot_sites]))

        sim.step = step
        return sim

    te.run_policy(net, te.PolicyInterface.training(net.meta), seconds=2.0, m=train_model, sim_factory=factory,
                  qpos0=qpos)  # fmt: skip
    slip_mm = 1000.0 * float(np.linalg.norm(feet[-1] - feet[0], axis=1).max())
    assert slip_mm < 1.0, slip_mm


def test_assets_are_listed_with_checksums() -> None:
    readme = (ASSETS / "README.md").read_text()
    import hashlib

    for f in sorted(Path(ASSETS).glob("*.npz")):
        assert hashlib.sha256(f.read_bytes()).hexdigest() in readme, f.name
