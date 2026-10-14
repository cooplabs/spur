"""The RT core's built-in arm model (DESIGN.md §3.4; rt/include/r1rt/arm_model.hpp; docs/architecture/ARM_MODEL.md)
against MuJoCo itself on the simulator's own model: gravity torques vs `mj_rne` (zero velocity and acceleration =
the holding torque, qfrc_bias) and every exported site vs `mj_kinematics`, on 100,000 random poses (the DESIGN's
test, error < 1e-6), plus the whole-body paths the core uses (gravity from the IMU + waist, v2's base frame for the
surface guard) and the check that the generated C++ table is the generator's current output (the MJCF and the core
cannot drift apart).

Fails on the base (70435dc): neither the generator, the table nor the binding functions exist there.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import mujoco
import numpy as np
import pytest
from r1sim import cpp_policy, dex3, joints
from r1sim import model as M

REPO = Path(__file__).resolve().parents[2]
TOL_TAU = 1e-6  # N·m (DESIGN §3.4)
TOL_POS = 1e-6  # m

needs_binding = pytest.mark.skipif(
    cpp_policy.find_module() is None,
    reason=f"r1rt_py not built (looked in {[str(d) for d in cpp_policy.build_dirs()]}): run scripts/ci.sh cpp",
)


def _gen() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_arm_model", REPO / "tools" / "gen_arm_model.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rt() -> Any:
    return cpp_policy.import_r1rt_py()


@pytest.fixture(scope="module")
def m() -> mujoco.MjModel:
    return M.build_model("real", visual=False)


def test_generated_file_is_current() -> None:
    gen = _gen()
    committed = (REPO / "rt" / "src" / "arm_model_gen.cpp").read_text()
    assert committed == gen.generate(), "the robot model changed: run `uv run python tools/gen_arm_model.py`"


@needs_binding
def test_binding_is_built_from_the_current_table(rt: Any) -> None:
    gen = _gen()
    digest = gen._digest(gen.export(M.build_model("real", visual=False)))
    assert rt.ARM_MODEL_DIGEST == digest, "r1rt_py was built from an older arm_model_gen.cpp: rebuild (ci.sh cpp)"
    assert tuple(rt.ARM_SITES) == gen.SITES


# ------------------------------------------------------------------------------------------------- random poses
def _random_poses(m: mujoco.MjModel, n: int, seed: int, leg_span: float | None = None) -> dict[str, Any]:
    """n random states: a uniformly random base orientation, every body / hand joint uniform in its range (or
    +-leg_span around 0 for the legs, when given)."""
    rng = np.random.default_rng(seed)
    ix = M.RobotIndex(m)
    q26 = np.zeros((n, joints.N))
    for i in range(joints.N):
        jid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{joints.JOINTS[i]}_joint"))
        lo, hi = m.jnt_range[jid]
        if i < 12 and leg_span is not None:
            lo, hi = max(lo, -leg_span), min(hi, leg_span)
        q26[:, i] = rng.uniform(lo, hi, n)
    hands = {}
    for side in dex3.SIDES:
        hands[side] = rng.uniform(dex3.Q_MIN[side], dex3.Q_MAX[side], (n, 7))
    quat = rng.normal(size=(n, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return {"q26": q26, "left": hands["left"], "right": hands["right"], "quat": quat, "ix": ix}


def _mujoco_reference(m: mujoco.MjModel, poses: dict[str, Any], quat_mode: str = "full") -> dict[str, np.ndarray]:
    """Pose the model per row (qvel = 0), run mj_kinematics + mj_comPos + mj_comVel + mj_rne: holding torques of the
    upper-body dofs, gravity in the torso frame, every exported site in the pelvis frame, the IMU quaternion."""
    gen = _gen()
    ix = poses["ix"]
    d = mujoco.MjData(m)
    n = len(poses["q26"])
    up = list(range(14, 26))
    vadr_up = ix.vadr[up]
    site_ids = [int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)) for s in gen.SITES]
    torso, pelvis = ix.torso_body, ix.pelvis_body
    tau = np.zeros((n, 12))
    g_torso = np.zeros((n, 3))
    sites = np.zeros((n, len(site_ids), 3))
    quat_imu = np.zeros((n, 4))
    qfull = np.zeros(m.nq)
    qfull[:] = m.qpos0
    res = np.zeros(m.nv)
    qimu = np.zeros(4)
    b = ix.base_qadr
    for r in range(n):
        qfull[b : b + 3] = (0.1, -0.2, 0.9)
        qfull[b + 3 : b + 7] = poses["quat"][r] if quat_mode == "full" else _rp_only(poses["quat"][r])
        qfull[ix.qadr] = poses["q26"][r]
        qfull[ix.hand_qadr["left"]] = poses["left"][r]
        qfull[ix.hand_qadr["right"]] = poses["right"][r]
        d.qpos[:] = qfull
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_rne(m, d, 0, res)
        tau[r] = res[vadr_up]
        Rt = d.xmat[torso].reshape(3, 3)
        g_torso[r] = Rt.T @ m.opt.gravity
        Rp = d.xmat[pelvis].reshape(3, 3)
        sites[r] = (d.site_xpos[site_ids] - d.xpos[pelvis]) @ Rp
        mujoco.mju_mat2Quat(qimu, d.site_xmat[ix.imu_site])
        quat_imu[r] = qimu
    return {"tau": tau, "g_torso": g_torso, "sites": sites, "quat_imu": quat_imu}


def _rp_only(quat: np.ndarray) -> np.ndarray:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, quat)
    R = r.reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (Rz @ R).reshape(-1))
    return q


def _check_against_mujoco(rt: Any, m: mujoco.MjModel, n: int, seed: int) -> dict[str, float]:
    poses = _random_poses(m, n, seed)
    ref = _mujoco_reference(m, poses)
    q26 = poses["q26"]
    err: dict[str, float] = {}
    # 1. each arm / the head alone, with gravity in the torso frame taken from MuJoCo's own torso pose
    for s, side in enumerate(dex3.SIDES):
        a0 = 14 + 5 * s
        tau = rt.arm_gravity(s, ref["g_torso"], np.ascontiguousarray(q26[:, a0 : a0 + 5]), poses[side], np.zeros(n))
        err[f"arm_{side}"] = float(np.max(np.abs(tau - ref["tau"][:, 5 * s : 5 * s + 5])))
    th = rt.head_gravity(ref["g_torso"], np.ascontiguousarray(q26[:, 24:26]))
    err["head"] = float(np.max(np.abs(th - ref["tau"][:, 10:12])))
    # 2. the core's path: gravity from the IMU quaternion + the waist joints, the 26-vector
    g = rt.gravity_in_torso(ref["quat_imu"], np.ascontiguousarray(q26[:, 12:14]))
    err["g_torso"] = float(np.max(np.abs(g - ref["g_torso"])))
    ff = rt.gravity_ff26(q26, ref["quat_imu"], poses["left"], poses["right"], None)
    err["ff26"] = float(np.max(np.abs(ff[:, 14:26] - ref["tau"])))
    err["ff26_legs_zero"] = float(np.max(np.abs(ff[:, :14])))
    # 3. every exported site in the pelvis frame
    q40 = np.ascontiguousarray(np.hstack([q26, poses["left"], poses["right"]]))
    sp = rt.sites_pelvis(q40)
    err["sites"] = float(np.max(np.abs(sp - ref["sites"])))
    return err


@needs_binding
def test_against_mujoco_100k(rt: Any, m: mujoco.MjModel) -> None:
    """DESIGN §3.4: 100,000 random poses, error < 1e-6 (N·m / m). Measured 10 Oct: <= 2.4e-14 N·m, 1.2e-15 m;
    ~3 s (one MuJoCo pass per pose in Python, the arm model batched in C++)."""
    err = _check_against_mujoco(rt, m, 100_000, seed=2)
    for k, v in err.items():
        assert v < (TOL_POS if k == "sites" else TOL_TAU), (k, v)


@needs_binding
def test_payload_is_a_point_mass_at_the_grasp_centre(rt: Any, m: mujoco.MjModel) -> None:
    """A declared payload = a point mass at v2's grasp centre: the same torque as MuJoCo with that mass added to the
    model at the grasp centre (a site-attached body on the palm would do the same; checked through the torque
    difference against the FK's grasp-centre Jacobian from MuJoCo: mj_jac at the point)."""
    poses = _random_poses(m, 200, seed=5)
    ix = poses["ix"]
    d = mujoco.MjData(m)
    gen = _gen()
    for s, side in enumerate(dex3.SIDES):
        palm = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_palm_link"))
        tips = [int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, gen.SITES[2 + 4 * s + k])) for k in (1, 2, 3)]
        dof = ix.vadr[14 + 5 * s : 19 + 5 * s]
        for r in range(len(poses["q26"])):
            qf = m.qpos0.copy()
            qf[ix.base_qadr + 3 : ix.base_qadr + 7] = poses["quat"][r]
            qf[ix.qadr] = poses["q26"][r]
            qf[ix.hand_qadr["left"]] = poses["left"][r]
            qf[ix.hand_qadr["right"]] = poses["right"][r]
            d.qpos[:] = qf
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            gc = 0.5 * d.site_xpos[tips[0]] + 0.25 * (d.site_xpos[tips[1]] + d.site_xpos[tips[2]])
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jacp, None, gc, palm)  # the grasp centre moves with the palm for the ARM joints
            want = -(jacp[:, dof].T @ m.opt.gravity) * 0.35
            g_t = d.xmat[ix.torso_body].reshape(3, 3).T @ m.opt.gravity
            qa = poses["q26"][r : r + 1, 14 + 5 * s : 19 + 5 * s]
            hq = poses[side][r : r + 1]
            t1 = rt.arm_gravity(s, g_t[None, :], np.ascontiguousarray(qa), hq, np.array([0.35]))
            t0 = rt.arm_gravity(s, g_t[None, :], np.ascontiguousarray(qa), hq, np.array([0.0]))
            assert np.max(np.abs((t1 - t0)[0] - want)) < TOL_TAU


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


@needs_binding
def test_open_hand_is_the_default(rt: Any) -> None:
    q = np.array([[-0.4, 0.3, 0.1, 0.8, 0.2]])
    for s, side in enumerate(dex3.SIDES):
        a = rt.hand_points_torso(s, q, None)
        b = rt.hand_points_torso(s, q, dex3.preset("open", side)[None, :])
        assert np.array_equal(a, b)


@needs_binding
def test_bad_rows_raise(rt: Any) -> None:
    q = np.array([[np.nan, 0.0, 0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="refused row 0"):
        rt.arm_gravity(0, np.array([[0.0, 0.0, -9.81]]), q, None, np.zeros(1))
    with pytest.raises(ValueError, match="shape"):
        rt.arm_gravity(0, np.zeros((1, 2)), np.zeros((1, 5)), None, np.zeros(1))
