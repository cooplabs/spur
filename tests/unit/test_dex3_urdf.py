"""The simulator's Dex3-1 hands (r1sim.dex3, as compiled in the "real" profile) against Unitree's URDFs
(models/dex3_1, unitree_ros@5994d4f): link masses, COMs, inertias, joint origins, axes, limits and efforts."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from r1sim import dex3, model

URDF_DIR = model.models_dir() / "dex3_1"
SHA256 = {
    "dex3_1_l.urdf": "c55f54d6d27b16a62b78a3dbebe85a85a6a4a4b8851dceb5e42339605f9aff30",
    "dex3_1_r.urdf": "a388e69058ec6b84306ff05a7b2b613548c833c3f8455127ab3488324553da57",
}
TOL = 1e-4


def _vec(s: str | None) -> np.ndarray:
    assert s is not None
    return np.array([float(v) for v in s.split()])


def _urdf(side: str) -> ET.Element:
    f = URDF_DIR / f"dex3_1_{side[0]}.urdf"
    assert hashlib.sha256(f.read_bytes()).hexdigest() == SHA256[f.name]
    return ET.parse(f).getroot()


def _full_inertia(m: mujoco.MjModel, b: int) -> np.ndarray:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, m.body_iquat[b])
    rm = r.reshape(3, 3)
    return np.asarray(rm @ np.diag(m.body_inertia[b]) @ rm.T)


@pytest.fixture(scope="module")
def real() -> mujoco.MjModel:
    return model.build_model("real", visual=False)


@pytest.mark.parametrize("side", dex3.SIDES)
def test_links_match_urdf(real: mujoco.MjModel, side: str) -> None:
    root = _urdf(side)
    total = 0.0
    for link in root.findall("link"):
        name = link.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        b = mujoco.mj_name2id(real, mujoco.mjtObj.mjOBJ_BODY, name)
        assert b >= 0, f"{name} missing in the model"
        mass_el, org, inr = inertial.find("mass"), inertial.find("origin"), inertial.find("inertia")
        assert mass_el is not None and org is not None and inr is not None
        mass = float(mass_el.get("value", "nan"))
        total += mass
        assert real.body_mass[b] == pytest.approx(mass, abs=1e-6), name
        assert np.allclose(real.body_ipos[b], _vec(org.get("xyz")), atol=TOL), name
        assert _vec(org.get("rpy")).tolist() == [0.0, 0.0, 0.0]
        a = {k: float(inr.get(k, "nan")) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        full = np.array(
            [[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]]
        )
        assert np.allclose(_full_inertia(real, b), full, atol=1e-8), name
    assert total == pytest.approx(dex3.MASS, abs=1e-6)


@pytest.mark.parametrize("side", dex3.SIDES)
def test_joints_match_urdf(real: mujoco.MjModel, side: str) -> None:
    root = _urdf(side)
    seen = []
    for j in root.findall("joint"):
        if j.get("type") != "revolute":
            continue
        name = j.get("name", "")
        k = dex3.joint_names(side).index(name)
        seen.append(name)
        jid = mujoco.mj_name2id(real, mujoco.mjtObj.mjOBJ_JOINT, name)
        child = j.find("child")
        org, axis, lim = j.find("origin"), j.find("axis"), j.find("limit")
        assert child is not None and org is not None and axis is not None and lim is not None
        b = mujoco.mj_name2id(real, mujoco.mjtObj.mjOBJ_BODY, child.get("link", ""))
        assert real.jnt_bodyid[jid] == b
        assert np.allclose(real.body_pos[b], _vec(org.get("xyz")), atol=TOL), name
        assert _vec(org.get("rpy")).tolist() == [0.0, 0.0, 0.0]
        assert np.allclose(real.jnt_axis[jid], _vec(axis.get("xyz")), atol=1e-9), name
        lo, hi = float(lim.get("lower", "nan")), float(lim.get("upper", "nan"))
        assert dex3.Q_MIN[side][k] == pytest.approx(lo, abs=TOL) and dex3.Q_MAX[side][k] == pytest.approx(hi, abs=TOL)
        assert np.allclose(real.jnt_range[jid], [lo, hi], atol=TOL), name
        assert dex3.TAU_MAX[k] == pytest.approx(float(lim.get("effort", "nan")))
    assert sorted(seen) == sorted(dex3.joint_names(side))
