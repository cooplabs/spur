"""The Dex3-1 hands' visual meshes (session 3; the user, 10 Oct: "the hands are not super accurate").

Unitree's 16 link meshes (models/dex3_1/meshes, unitree_ros@5994d4f) are drawn in the link frames of the real
profile, and the stock R1 fist is cut off its wrist meshes. They must not change the physics: no contact, no mass,
and the compiled model rolls out bit-identically with and without them. The train profile is left exactly as
trained (the A1 model diff compares visual geoms too, at tolerance 0)."""

from __future__ import annotations

import hashlib
import re
import struct
import xml.etree.ElementTree as ET
from collections.abc import Iterator

import mujoco
import numpy as np
import pytest
from r1sim import dex3, model

SOURCE_MD = model.models_dir() / "dex3_1" / "SOURCE.md"


def _stl_vertices(path: object) -> np.ndarray:
    b = open(str(path), "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(b[84 : 84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return np.unique(rec["v"].reshape(-1, 3).astype(np.float64), axis=0)


def test_mesh_files_match_source_md() -> None:
    table = dict(re.findall(r"\| meshes/(\S+\.STL) \| ([0-9a-f]{64}) \|", SOURCE_MD.read_text()))
    assert len(table) == 16
    for side in dex3.SIDES:
        for link in ("palm", *dex3.NAMES):
            f = dex3.mesh_file(side, link)
            assert hashlib.sha256(f.read_bytes()).hexdigest() == table[f.name], f.name


def _geom_world_vertices(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> np.ndarray:
    mid = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[a : a + n].astype(np.float64)
    return np.asarray(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)


def _link_frame_vertices(d: mujoco.MjData, body: int, v_link: np.ndarray) -> np.ndarray:
    return np.asarray(d.xpos[body] + v_link @ d.xmat[body].reshape(3, 3).T)


@pytest.fixture(scope="module")
def real_visual() -> mujoco.MjModel:
    return model.build_model("real")


@pytest.mark.parametrize("side", dex3.SIDES)
def test_real_profile_draws_each_link_mesh_in_its_link_frame(real_visual: mujoco.MjModel, side: str) -> None:
    """Every hand link carries its STL as a visual geom placed exactly as the URDF's <visual> (identity origin):
    the compiled mesh vertices (MuJoCo re-centres meshes) land on the STL's vertices in the link frame."""
    m = real_visual
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    for k, jn in enumerate(dex3.joint_names(side)):  # a bent pose, so the joints are part of the check
        d.qpos[m.jnt_qposadr[m.joint(jn).id]] = dex3.preset("relax", side)[k]
    mujoco.mj_forward(m, d)
    for link in ("palm", *dex3.NAMES):
        g = m.geom(f"{side}_hand_{link}_visual").id
        b = m.body(f"{side}_hand_{link}_link").id
        assert m.geom_bodyid[g] == b
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        assert (m.geom_contype[g], m.geom_conaffinity[g], m.geom_group[g]) == (0, 0, dex3.VISUAL_GROUP)
        got = _geom_world_vertices(m, d, g)
        want = _link_frame_vertices(d, b, _stl_vertices(dex3.visual_mesh_file(side, link)))
        assert got.shape == want.shape, link
        err = np.abs(np.sort(got, axis=0) - np.sort(want, axis=0)).max()
        assert err < 1e-5, (link, err)


def test_drawn_copies_are_faithful_and_outside_the_repo() -> None:
    """The decimated drawing copies (vertex clustering, 1 mm cells) stay within the cell of Unitree's surface: every
    drawn vertex lies within the cell's diagonal of an original vertex (p99 within 1 mm), the outlines agree to 1 mm,
    the triangle count drops to < 35 %; the copies live in the cache, not in the repository; the clustering is
    deterministic."""
    root = model.models_dir().parent
    n0 = n1 = 0
    for side in dex3.SIDES:
        for link in ("palm", *dex3.NAMES):
            f = dex3.visual_mesh_file(side, link)
            assert root not in f.parents
            orig, drawn = _stl_vertices(dex3.mesh_file(side, link)), _stl_vertices(f)
            sub = drawn[:: max(1, len(drawn) // 400)]
            dist = np.min(np.linalg.norm(sub[:, None, :] - orig[None, :, :], axis=2), axis=1)
            assert dist.max() < dex3.VISUAL_CELL_M * np.sqrt(3) and np.percentile(dist, 99) < 0.001, (side, link)
            assert np.allclose(drawn.min(0), orig.min(0), atol=0.001) and np.allclose(
                drawn.max(0), orig.max(0), atol=0.001
            )
            n0 += struct.unpack("<I", dex3.mesh_file(side, link).read_bytes()[80:84])[0]
            n1 += struct.unpack("<I", f.read_bytes()[80:84])[0]
    assert n1 < 0.35 * n0
    tri = dex3._stl_triangles(dex3.mesh_file("right", "thumb_2"))
    a, b = dex3.decimate(tri, dex3.VISUAL_CELL_M), dex3.decimate(tri, dex3.VISUAL_CELL_M)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.mark.parametrize("side", dex3.SIDES)
def test_primitives_are_hidden_but_still_collide(real_visual: mujoco.MjModel, side: str) -> None:
    m = real_visual
    hand_bodies = {m.body(n).id for n in dex3.link_names(side)}
    prim = [g for g in range(m.ngeom) if m.geom_bodyid[g] in hand_bodies and m.geom_contype[g]]
    assert len(prim) == 8 + 9  # palm box + 7 link shapes + 9 pads
    assert all(m.geom_group[g] == dex3.COLLISION_GROUP for g in prim)


def test_visuals_add_no_mass(real_visual: mujoco.MjModel) -> None:
    m0 = model.build_model("real", visual=False)
    assert np.array_equal(real_visual.body_mass, m0.body_mass)
    assert np.array_equal(real_visual.body_inertia, m0.body_inertia)
    assert np.array_equal(real_visual.body_ipos, m0.body_ipos)


def test_wrist_mesh_has_no_fist(real_visual: mujoco.MjModel) -> None:
    """The stock wrist_roll_link mesh carries the R1's fist beyond x ~0.08 m; the drawn one stops at the mount."""
    m = real_visual
    for side in dex3.SIDES:
        g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{side}_wrist_roll_link").id
                 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)  # fmt: skip
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        v_link = (_geom_world_vertices(m, d, g) - d.xpos[m.geom_bodyid[g]]) @ d.xmat[m.geom_bodyid[g]].reshape(3, 3)
        stock = _stl_vertices(model.R1_DIR / "meshes" / f"{side}_wrist_roll_link.STL")
        assert stock[:, 0].max() > 0.14  # the fist is in the file
        assert v_link[:, 0].max() < dex3.MOUNT_POS[0] + 0.006  # and not in the drawing (triangle centroid cut)


def _without_hand_visuals(monkeypatch: pytest.MonkeyPatch) -> None:
    """The s2 drawing: primitives visible, stock wrist mesh, no hand meshes."""
    orig = dex3.hand_xml
    monkeypatch.setattr(dex3, "hand_xml", lambda *a, **kw: orig(*a, **{**kw, "meshes": False}))
    monkeypatch.setattr(model, "_hand_visuals", lambda *a, **kw: None)


def _collapse(m: mujoco.MjModel, seconds: float) -> Iterator[np.ndarray]:
    """No control: the robot falls; arms and hands hit the body and the floor (contacts on the hand primitives)."""
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qpos[2] = 0.74
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    yield d.qpos.copy()
    yield d.qvel.copy()
    yield np.array([d.ncon], dtype=np.float64)


def test_physics_bit_identical_with_and_without_hand_visuals(
    real_visual: mujoco.MjModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as mp:
        _without_hand_visuals(mp)
        without_v = model.build_model("real")
    assert real_visual.ngeom == without_v.ngeom + 16
    a = list(_collapse(real_visual, 1.5))
    b = list(_collapse(without_v, 1.5))
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)
    assert a[2][0] > 0


def test_train_profile_is_left_as_trained() -> None:
    """The train profile must stay field-for-field mjlab's model (A1, tolerance 0, visual geoms included): no hand
    meshes, the stock wrist mesh; its renders keep training's look."""
    root = ET.fromstring(model.robot_xml("train"))
    names = {g.get("name") for g in root.iter("geom")}
    assert not any(n and n.endswith("_visual") and "_hand_" in n for n in names)
    assert {g.get("mesh") for g in root.iter("geom")} >= {"left_wrist_roll_link", "right_wrist_roll_link"}
    assert not any(m.get("name", "").endswith("_nofist") for m in root.iter("mesh"))


def test_link_poses_match_the_compiled_real_hand(real_visual: mujoco.MjModel) -> None:
    """dex3.link_poses (used by the comparison tool) == MuJoCo's FK of the real hand."""
    m = real_visual
    d = mujoco.MjData(m)
    for side in dex3.SIDES:
        q = dex3.preset("pinch", side)
        for k, jn in enumerate(dex3.joint_names(side)):
            d.qpos[m.jnt_qposadr[m.joint(jn).id]] = q[k]
        mujoco.mj_forward(m, d)
        palm = m.body(f"{side}_hand_palm_link").id
        rp, pp = d.xmat[palm].reshape(3, 3), d.xpos[palm]
        for link, (pos, quat) in dex3.link_poses(side, q).items():
            b = m.body(f"{side}_hand_{link}_link").id
            assert np.allclose(rp.T @ (d.xpos[b] - pp), pos, atol=1e-9), (side, link)
            r = np.zeros(9)
            mujoco.mju_quat2Mat(r, quat)
            assert np.allclose(rp.T @ d.xmat[b].reshape(3, 3), r.reshape(3, 3), atol=1e-9), (side, link)
