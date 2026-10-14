"""Model assembly (r1sim.model): both profiles compile, masses, the SDK joint order, the IMU / backpack / head
differences between profiles, the stand placement and the training's hand lump."""

from __future__ import annotations

import copy
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from r1sim import dex3, model
from r1sim import joints as J

# v2 joints.py:18-29 (r1mc2 at r21_try 2a4a165), confirmed on our robot by the 28 Sep 2026 read-only lowstate
# capture (slots 14, 20, 21, 27, 28, 31-34 report mode 0). Hard-coded here on purpose: a change must be deliberate.
V2_JOINTS = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_roll", "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
    "head_pitch", "head_yaw",
)  # fmt: skip
V2_SLOT = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30)

# Component masses (kg): r1.xml (unitree_rl_mjlab@1425b15), the URDF torso/head inertials, the Dex3-1 URDF,
# training's hand lump, the fist and backpack estimates.
R1_XML_MASS = 28.932  # loaded total of r1.xml (V2_INVENTORY App. E §2.2), checked below to 1 g
R1_XML_TORSO = 7.3387


def _raw_r1() -> mujoco.MjModel:
    """Unitree's r1.xml as shipped, minus what stops it compiling alone (meshes, 2 excludes naming absent bodies)."""
    root = ET.parse(model.BASE_XML).getroot()
    contact = root.find("contact")
    asset = root.find("asset")
    assert contact is not None and asset is not None
    for ex in list(contact):
        if "wrist_pitch" in ex.get("body1", "") + ex.get("body2", ""):
            contact.remove(ex)
    for mesh in list(asset):
        asset.remove(mesh)
    for parent in root.iter():
        for g in list(parent.findall("geom")):
            if g.get("mesh"):
                parent.remove(g)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


@pytest.fixture(scope="module", params=model.PROFILES)
def built(request: pytest.FixtureRequest) -> tuple[str, mujoco.MjModel, model.RobotIndex]:
    m = model.build_model(request.param, visual=False)
    return request.param, m, model.RobotIndex(m)


def test_r1_xml_is_unitrees_file() -> None:
    assert hashlib.sha256(model.BASE_XML.read_bytes()).hexdigest() == model.R1_XML_SHA256


def test_r1_xml_total_mass() -> None:
    m = _raw_r1()
    assert abs(float(m.body_subtreemass[1]) - R1_XML_MASS) < 1e-3


def test_compiles_with_meshes() -> None:
    for p in model.PROFILES:
        m = model.build_model(p, visual=True)
        assert m.nmesh > 0


def test_joint_and_actuator_counts(built: tuple[str, mujoco.MjModel, model.RobotIndex]) -> None:
    profile, m, ix = built
    hinges = int(np.sum(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
    frees = int(np.sum(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
    assert frees == 1
    if profile == "train":
        assert hinges == 24 and m.nu == 24
        assert not ix.hand_qadr
    else:
        assert hinges == 26 + 14 and m.nu == 26 + 14
        assert set(ix.hand_qadr) == {"left", "right"}
    assert ix.profile == profile


def test_total_mass(built: tuple[str, mujoco.MjModel, model.RobotIndex]) -> None:
    profile, _, ix = built
    fist, lump = dex3.FIST_MASS, model.HAND_LUMP_MASS
    r1 = float(_raw_r1().body_subtreemass[1])
    if profile == "train":
        expected = r1 - 2 * fist + 2 * lump
    else:
        head = model.HEAD_PITCH_INERTIAL[1] + model.HEAD_YAW_INERTIAL[1]
        torso = model.TORSO_URDF_MASS + head - R1_XML_TORSO
        expected = r1 + torso - 2 * fist + 2 * dex3.MASS + model.BACKPACK_MASS + model.REAR_MASS_KG
    print(f"{profile}: total mass {ix.total_mass:.4f} kg (expected {expected:.4f})")
    assert abs(ix.total_mass - expected) < 1e-6  # (the brief asked for 1 g; the sum is exact)


def test_sdk_joint_order_and_names(built: tuple[str, mujoco.MjModel, model.RobotIndex]) -> None:
    profile, m, ix = built
    assert V2_JOINTS == J.JOINTS
    assert tuple(int(s) for s in J.SLOT) == V2_SLOT
    assert J.N == 26 and J.NUM_SLOTS == 35 and J.MODE_MACHINE == 1
    for i, n in enumerate(J.JOINTS):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint")
        if n.startswith("head_") and profile == "train":
            assert jid < 0 and not ix.present[i] and ix.act[i] == -1 and ix.qadr[i] == -1
            continue
        assert jid >= 0 and ix.present[i]
        assert ix.qadr[i] == m.jnt_qposadr[jid] and ix.vadr[i] == m.jnt_dofadr[jid]
        a = int(ix.act[i])
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) == n
        assert m.actuator_trnid[a][0] == jid
        if profile == "train":  # mjlab's <position> actuators (session 2): force-limited, ctrl not limited
            assert not m.actuator_ctrllimited[a] and m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
            assert np.allclose(m.actuator_forcerange[a], [-J.TAU_MAX[i], J.TAU_MAX[i]])
        else:
            assert np.allclose(m.actuator_ctrlrange[a], [-J.TAU_MAX[i], J.TAU_MAX[i]])
    # actuator order: the present body joints in SDK order, then (real) the hands in DDS order, left then right
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, k) for k in range(m.nu)]
    body = [n for i, n in enumerate(J.JOINTS) if ix.present[i]]
    hands = [] if profile == "train" else dex3.actuator_names("left") + dex3.actuator_names("right")
    assert names == body + hands
    for side, acts in ix.hand_act.items():
        for k, a in enumerate(acts):
            assert np.allclose(m.actuator_ctrlrange[a], [-dex3.TAU_MAX[k], dex3.TAU_MAX[k]])
            assert (
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a][0]) == (dex3.joint_names(side)[k])
            )


def test_profile_differences(built: tuple[str, mujoco.MjModel, model.RobotIndex]) -> None:
    profile, m, ix = built

    def has(kind: mujoco.mjtObj, name: str) -> bool:
        return bool(mujoco.mj_name2id(m, kind, name) >= 0)

    imu = m.site_pos[ix.imu_site]
    assert ix.head_camera_site >= 0 and ix.imu_site >= 0
    if profile == "real":
        assert np.allclose(imu, model.IMU_URDF_POS)
        assert has(mujoco.mjtObj.mjOBJ_BODY, "backpack")
        bp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "backpack")
        assert abs(m.body_mass[bp] - 0.454) < 1e-9  # weighed by the user, 9 Oct 2026 ("around 1 pound")
        assert np.allclose(m.body_pos[bp], model.BACKPACK_POS)
        assert has(mujoco.mjtObj.mjOBJ_JOINT, "head_pitch_joint") and has(mujoco.mjtObj.mjOBJ_JOINT, "head_yaw_joint")
        assert m.opt.timestep == 0.001
    else:
        assert np.allclose(imu, 0.0)
        assert not has(mujoco.mjtObj.mjOBJ_BODY, "backpack")
        assert not has(mujoco.mjtObj.mjOBJ_JOINT, "head_pitch_joint")
        assert m.opt.timestep == 0.005
    assert m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.impratio == 10
    # the IMU sensors stay attached to the (moved) site
    for s in ("imu_ang_vel", "imu_lin_acc"):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, s)
        assert sid >= 0 and m.sensor_objid[sid] == ix.imu_site


def test_train_hand_lump_and_fist(built: tuple[str, mujoco.MjModel, model.RobotIndex]) -> None:
    profile, m, _ = built
    base = _raw_r1()
    for side, g in (("left", -1.0), ("right", 1.0)):
        wr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_roll_link")
        wr0 = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_roll_link")
        assert abs(m.body_mass[wr] - (base.body_mass[wr0] - 0.143)) < 1e-6
        assert abs(m.body_ipos[wr][0] - 0.035) < 1e-9
        if profile == "train":
            h = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_dex3_hand")
            assert m.body_parentid[h] == wr
            assert abs(m.body_mass[h] - 0.697) < 1e-9
            assert np.allclose(m.body_pos[h], [0.075, 0, 0])
            assert np.allclose(m.body_ipos[h], [0.0667, g * 0.008, 0.0])
            assert np.allclose(np.sort(m.body_inertia[h]), np.sort([6e-4, 1.4e-3, 1.2e-3]))
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_hand_collision") >= 0  # stock capsule kept
        else:
            palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_palm_link")
            assert m.body_parentid[palm] == wr and np.allclose(m.body_pos[palm], dex3.MOUNT_POS)
            assert abs(m.body_subtreemass[palm] - dex3.MASS) < 1e-6


def test_train_collision_and_joint_params(built: tuple[str, mujoco.MjModel, model.RobotIndex]) -> None:
    profile, m, _ = built
    knee = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")
    ankle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_pitch_joint")
    elbow = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_elbow_joint")

    def arm(j: int) -> tuple[float, float]:
        return float(m.dof_armature[m.jnt_dofadr[j]]), float(m.dof_frictionloss[m.jnt_dofadr[j]])

    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_foot4_collision")
    if profile == "train":
        assert arm(knee) == pytest.approx((0.021354, 1.0))
        assert arm(ankle) == pytest.approx((0.00795, 0.8))
        assert arm(elbow) == pytest.approx((0.01, 0.2))
        assert m.geom_condim[foot] == 3 and m.geom_priority[foot] == 1 and m.geom_friction[foot][0] == 0.6
        shin = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_shin_collision")
        assert m.geom_condim[shin] == 1
    else:
        # session 8: friction / armature measured at HW1 (model.REAL_JOINT_*_HW1; the ankles' armature not identified)
        # session 9: the legs' / waist's identified 0.008 applied (base 6449aee: 0.05); the ankles keep 0.10
        assert arm(knee) == pytest.approx((0.008, model.REAL_JOINT_FRICTIONLOSS_HW1["left_knee"]))
        for n in ("left_hip_pitch", "right_hip_roll", "right_hip_yaw", "waist_roll", "waist_yaw"):
            assert arm(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_joint"))[0] == pytest.approx(0.008), n
        assert arm(ankle) == pytest.approx((0.10, model.REAL_JOINT_FRICTIONLOSS_HW1["left_ankle_pitch"]))
        assert arm(elbow) == pytest.approx((0.0025, 0.03))
        assert float(m.dof_damping[m.jnt_dofadr[elbow]]) == pytest.approx(1.0)  # identified (r1.xml: 0.05)
        assert float(m.dof_damping[m.jnt_dofadr[knee]]) == pytest.approx(0.05)


@pytest.mark.parametrize("centred", [False, True])
def test_stand_qpos(built: tuple[str, mujoco.MjModel, model.RobotIndex], centred: bool) -> None:
    _, m, ix = built
    pose = model.centred_stand_pose(m) if centred else J.POSE_STAND
    d = mujoco.MjData(m)
    d.qpos[:] = model.stand_qpos(m, pose)
    mujoco.mj_forward(m, d)
    lowest = model._lowest_foot_z(m, d, ix)
    assert 0.0 <= lowest <= 0.002
    mid = 0.5 * (d.site_xpos[ix.foot_sites[0]] + d.site_xpos[ix.foot_sites[1]])
    assert np.allclose(mid[:2], 0.0, atol=1e-3)
    for s in ix.foot_sites:  # soles level in pitch
        r = d.xmat[m.site_bodyid[s]].reshape(3, 3)
        assert abs(np.arctan2(r[2, 0], r[2, 2])) < 1e-6
    for i in range(J.N):
        if ix.present[i]:
            assert d.qpos[ix.qadr[i]] == pose[i]
    for side, adr in ix.hand_qadr.items():
        assert np.allclose(d.qpos[adr], dex3.preset("open", side))
    if centred:
        assert abs(d.subtree_com[ix.pelvis_body][0] - mid[0]) < 1e-4
    assert d.ncon == 0  # nothing touches anything at the start (feet 1 mm up)


def _pd_stand(m: mujoco.MjModel, ix: model.RobotIndex, pose: np.ndarray, seconds: float) -> mujoco.MjData:
    """A balance-free stiff joint PD (KP/KD_STIFF, torque-clipped) at `pose`, hands held open."""
    if np.any(m.actuator_biastype != mujoco.mjtBias.mjBIAS_NONE):  # train: mjlab <position> actuators -> torque
        m = copy.deepcopy(m)
        m.actuator_gainprm[:, 0] = 1.0
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_biasprm[:] = 0.0
        m.actuator_ctrllimited[:] = 0
    d = mujoco.MjData(m)
    d.qpos[:] = model.stand_qpos(m, pose)
    mujoco.mj_forward(m, d)
    pres = ix.present
    qa, va, ac = ix.qadr[pres], ix.vadr[pres], ix.act[pres]
    for _ in range(round(seconds / m.opt.timestep)):
        tau = J.KP_STIFF[pres] * (pose[pres] - d.qpos[qa]) - J.KD_STIFF[pres] * d.qvel[va]
        d.ctrl[ac] = np.clip(tau, -J.TAU_MAX[pres], J.TAU_MAX[pres])
        for side, a in ix.hand_act.items():
            q, v = d.qpos[ix.hand_qadr[side]], d.qvel[ix.hand_vadr[side]]
            d.ctrl[a] = dex3.KP * (dex3.preset("open", side) - q) - dex3.KD * v
        mujoco.mj_step(m, d)
    return d


def _pitch_deg(d: mujoco.MjData, body: int) -> float:
    r = d.xmat[body].reshape(3, 3)
    return float(np.degrees(np.arctan2(-r[2, 0], r[2, 2])))


def test_centred_pose_stands_under_stiff_pd(built: tuple[str, mujoco.MjModel, model.RobotIndex]) -> None:
    """Model sanity: with the CoM over the feet, a stiff joint PD keeps the robot up (3 s, feet still)."""
    profile, m, ix = built
    d = _pd_stand(m, ix, model.centred_stand_pose(m), 3.0)
    feet = d.site_xpos[list(ix.foot_sites)][:, :2]
    print(f"{profile}: pelvis z {d.qpos[ix.base_qadr + 2]:.4f} pitch {_pitch_deg(d, ix.pelvis_body):+.2f} deg")
    assert d.qpos[ix.base_qadr + 2] > 0.70
    assert abs(_pitch_deg(d, ix.pelvis_body)) < 8.0
    assert np.allclose(feet, [[0.0, 0.1094], [0.0, -0.1094]], atol=0.005)


def test_policy_pose_falls_without_balance(built: tuple[str, mujoco.MjModel, model.RobotIndex]) -> None:
    """Pins the S1 finding behind centred_stand_pose: POSE_STAND (an actively balanced stance, CoM 4-5 cm behind the
    foot centre) falls backwards under a balance-free stiff PD, like the real robot on 4 Oct (`lock`)."""
    _, m, ix = built
    d = _pd_stand(m, ix, J.POSE_STAND, 3.0)
    assert d.qpos[ix.base_qadr + 2] < 0.3 and _pitch_deg(d, ix.pelvis_body) < -45.0


def _torso_back_geometry(m: mujoco.MjModel) -> tuple[np.ndarray, float]:
    """From Unitree's torso mesh (waist_yaw_link, torso_link frame): the rear handle frame's vertices (the black bars,
    x < -0.088) and the back plate's x at the centre of the back."""
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != t or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) != "waist_yaw_link":
            continue
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, m.geom_quat[g])
        v = v @ r.reshape(3, 3).T + m.geom_pos[g]
        bars = v[v[:, 0] < -0.088]
        plate = v[(np.abs(v[:, 1]) < 0.03) & (v[:, 2] > 0.09) & (v[:, 2] < 0.18) & (v[:, 0] > -0.085)]
        return bars, float(plate[:, 0].min())
    raise AssertionError("no waist_yaw_link mesh on torso_link")


def test_backpack_sits_inside_the_rear_frame() -> None:
    """The rear computing module sits INSIDE the black rear handle frame, against the back plate (the user, 9 Oct
    2026; the frame is in Unitree's torso mesh). S1's first placement (a 10 x 25 x 20 cm box 15 cm behind the torso
    CoM, from the 4 Oct sag fit) stuck out far behind the bars."""
    m = model.build_model("real", visual=True)
    bars, plate_x = _torso_back_geometry(m)
    bp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "backpack")
    g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == bp)
    c, h = m.body_pos[bp] + m.geom_pos[g], m.geom_size[g]
    tol = 0.002
    assert c[0] - h[0] >= bars[:, 0].min() - tol, "sticks out behind the rear frame"
    assert c[0] + h[0] <= plate_x + tol, "inside the back plate"
    assert c[1] - h[1] >= bars[:, 1].min() - tol and c[1] + h[1] <= bars[:, 1].max() + tol, "wider than the frame"
    assert c[2] - h[2] >= bars[:, 2].min() - tol and c[2] + h[2] <= bars[:, 2].max() + tol, "taller than the frame"


def test_floating_base_has_no_joint_friction(built: tuple[str, mujoco.MjModel, model.RobotIndex]) -> None:
    """Session 4 (A7): r1.xml's global joint default (damping 0.05, armature 0.01, frictionloss 0.2) also lands on the
    pelvis's `type="free"` joint - dry friction against the world on all six base DOFs, which held the frozen 9 Oct
    stance (r1sim.tier_a.a7_stability). The real profile has none; the train profile keeps mjlab's (A1). Fails on the
    session-3 model (0.2 / 0.05 / 0.01 on the real profile)."""
    profile, m, _ = built
    a = int(m.joint("floating_base_joint").dofadr[0])
    fr, damp, arm = m.dof_frictionloss[a : a + 6], m.dof_damping[a : a + 6], m.dof_armature[a : a + 6]
    if profile == "real":
        assert np.all(fr == 0.0) and np.all(damp == 0.0) and np.all(arm == 0.0)
        assert model.FLOATING_BASE_JOINT_REAL == {"damping": 0.0, "armature": 0.0, "frictionloss": 0.0}
    else:
        assert np.allclose(fr, 0.2) and np.allclose(damp, 0.05) and np.allclose(arm, 0.01)
    # the hinge joints keep the default where the code does not set them (e.g. the joint damping 0.05)
    assert m.dof_damping[int(m.joint("left_knee_joint").dofadr[0])] == pytest.approx(0.05)


def test_rear_mass_is_the_com_fit(built: tuple[str, mujoco.MjModel, model.RobotIndex]) -> None:
    """Session 5 (A2, r1sim.tier_a.a2_com): ONE CoM correction - a rear mass at the rear module's place - fitted to the
    9 Oct, 4 Oct and 6 Oct joint loads together; real profile only (A1: the train profile is mjlab's). Fails on the
    session-4 model (no rear mass: the CoM sat ~1-3 cm too far forward of the logged loads)."""
    profile, m, ix = built
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rear_mass")
    if profile == "train":
        assert b < 0
        return
    assert b >= 0 and m.body_mass[b] == pytest.approx(model.REAR_MASS_KG) and 1.0 <= model.REAR_MASS_KG <= 2.5
    assert np.allclose(m.body_pos[b], model.REAR_MASS_POS) and model.REAR_MASS_POS == model.BACKPACK_POS
    assert m.body_parentid[b] == ix.torso_body
    assert not any(m.geom_bodyid[g] == b for g in range(m.ngeom))  # an inertial only: nothing to collide or render
