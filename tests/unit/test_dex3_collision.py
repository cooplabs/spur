"""The Dex3-1 collision geometry: the S1 primitives (default, unchanged) and the opt-in "mesh" variant (session 3).

The primitives differ from Unitree's link meshes by up to ~4 mm per side in finger width, +3.6 / -2.3 mm at the
pads and 3.5 mm in fingertip reach (tools/hands/compare_collision.py). The "mesh" variant collides each link with
the convex hull of its mesh and moves the pads to that surface; it stays opt-in until A8 / A9 / A10 are re-run on
it (S3.md). These tests pin: the default hand is byte-identical to session 2's; the variant keeps the geom names the
sensor model needs; its pads sit PAD_PROUD_M beyond the hull (re-measured from MuJoCo's own hull); and the grasp-rig
numbers that motivate it (contact points with an object at v2's open-hand aim point)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import mujoco
import numpy as np
import pytest
from r1sim import dex3, model
from r1sim.sensors import SensorModel

ROOT = Path(__file__).resolve().parents[2]
# sha256 of dex3.hand_xml(side) on s3 base ea264ab (= session 2): the default must not move
HAND_XML_SHA256 = {
    "left": "8c0828f77674b225eeec3cd1a1a5b327ce93aee870a6cacf57d37b104078af92",
    "right": "8a15ee09a09031a6c5bd9e16f5c0fc22107ddd3e4a3c64f0a00e322e873fe11d",
}


def _tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"r1mc3_tools_hands_{name}", ROOT / "tools" / "hands" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("side", dex3.SIDES)
def test_default_hand_is_session2s(side: str) -> None:
    assert dex3.COLLISION == "primitives"
    assert hashlib.sha256(dex3.hand_xml(side).encode()).hexdigest() == HAND_XML_SHA256[side]
    assert dex3.hand_xml(side) == dex3.hand_xml(side, collision="primitives")
    with pytest.raises(ValueError):
        dex3.hand_xml(side, collision="boxes")


@pytest.fixture(scope="module")
def mesh_model() -> mujoco.MjModel:
    orig = dex3.COLLISION
    dex3.COLLISION = "mesh"
    try:
        return model.build_model("real", visual=False)
    finally:
        dex3.COLLISION = orig


def test_mesh_variant_keeps_the_names_and_binds_the_sensors(mesh_model: mujoco.MjModel) -> None:
    m = mesh_model
    for side in dex3.SIDES:
        for link in ("palm", *dex3.NAMES):
            g = m.geom(f"{side}_hand_{link}_geom").id
            assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] == 1
        for k in range(9):
            assert m.geom_type[m.geom(f"{side}_pad_{k}").id] == mujoco.mjtGeom.mjGEOM_BOX
    sm = SensorModel()
    sm.bind(m, model.RobotIndex(m))
    for side in dex3.SIDES:
        hand = sm._hands[side]
        assert len(hand.pads) == 9 and len(hand.link_pads) == 7  # thumb_0 carries no pad


def test_mesh_variant_pads_sit_just_beyond_the_hull(mesh_model: mujoco.MjModel) -> None:
    """Each pad's outer face centre lies PAD_PROUD_M beyond the link hull along the pad normal (MuJoCo's hull of the
    compiled collision mesh), so MESH_PAD_SHIFT_M is what the meshes say."""
    m = mesh_model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    checked = 0
    for side in dex3.SIDES:
        for link in ("palm", *dex3.NAMES):
            hg = m.geom(f"{side}_hand_{link}_geom").id
            mid = m.geom_dataid[hg]
            a = int(m.mesh_graphadr[mid])
            nv, nf = int(m.mesh_graph[a]), int(m.mesh_graph[a + 1])
            off = a + 2 + 3 * nv + 3 * nf
            faces = np.asarray(m.mesh_graph[off : off + 3 * nf]).reshape(nf, 3)
            va = int(m.mesh_vertadr[mid])
            v = np.asarray(m.mesh_vert[va : va + int(m.mesh_vertnum[mid])], dtype=np.float64)
            vw = d.geom_xpos[hg] + v @ d.geom_xmat[hg].reshape(3, 3).T
            t = vw[faces]
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            keep = np.linalg.norm(n, axis=1) > 2e-9
            n, t = n[keep], t[keep]
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            c = np.einsum("ij,ij->i", n, t[:, 0])
            flip = (n @ vw.mean(axis=0) - c) > 0
            n[flip] *= -1
            c[flip] *= -1
            for k in range(9):
                pg = m.geom(f"{side}_pad_{k}").id
                if m.geom_bodyid[pg] != m.geom_bodyid[hg]:
                    continue
                R = d.geom_xmat[pg].reshape(3, 3)
                ax = int(np.argmin(m.geom_size[pg]))
                nrm = R[:, ax] * np.sign(R[:, ax] @ (d.geom_xpos[pg] - d.xpos[m.geom_bodyid[pg]]) or 1.0)
                if link == "palm":  # palm pads face the grasp side (+g y of the palm)
                    g = -1.0 if side == "left" else 1.0
                    nrm = R[:, ax] * np.sign(R[:, ax] @ (d.xmat[m.geom_bodyid[pg]].reshape(3, 3)[:, 1] * g))
                face = d.geom_xpos[pg] + nrm * m.geom_size[pg][ax]
                dn = n @ nrm
                sel = dn > 1e-9
                beyond = -((c[sel] - n[sel] @ face) / dn[sel]).min()  # > 0: the pad face lies beyond the hull
                assert beyond == pytest.approx(dex3.PAD_PROUD_M, abs=5e-5), (side, k, beyond)
                checked += 1
    assert checked == 18


def test_grasp_rig_numbers() -> None:
    """The motivation, pinned (object welded at v2's open-hand aim point, the hand closing to the preset):
    a 6 cm round object is touched by the distal pads ~17 mm FORWARD of the aim point (along the
    fingers) with either geometry (they differ by < 2 mm there); the tall box's contact centroid moves ~6 mm and its
    fingers close further with the real (mesh) shapes; the flat bar's pinch is touched ~6 mm further across the fingers
    (the real fingers are wider)."""
    rig = _tool("grasp_rig")
    cases = (("cylinder60", "fist"), ("box_tall", "fist"), ("bar_flat", "pinch"))
    res = {(o, c): rig.run("right", c, o, p, fixed=True) for o, p in cases for c in dex3.COLLISIONS}
    aim = np.array(res[("cylinder60", "primitives")].aim_mm)
    assert aim == pytest.approx([99.3, 60.4, 0.0], abs=0.2)
    for c in dex3.COLLISIONS:
        cen = np.array(res[("cylinder60", c)].centroid_mm)
        assert 12.0 < cen[0] - aim[0] < 22.0, (c, cen)
    d_cyl = np.array(res[("cylinder60", "mesh")].centroid_mm) - np.array(res[("cylinder60", "primitives")].centroid_mm)
    assert np.linalg.norm(d_cyl) < 3.0
    d_tea = np.array(res[("box_tall", "mesh")].centroid_mm) - np.array(res[("box_tall", "primitives")].centroid_mm)
    assert 4.0 < np.linalg.norm(d_tea) < 12.0
    d_er = np.array(res[("bar_flat", "mesh")].centroid_mm) - np.array(res[("bar_flat", "primitives")].centroid_mm)
    assert 3.0 < abs(d_er[2]) < 10.0


def test_provenance_records_the_hand_geometry() -> None:
    import yaml

    v = {
        k: e["value"]
        for k, e in yaml.safe_load((model.models_dir() / "provenance.yaml").read_text())["parameters"].items()
    }
    assert {int(k): x for k, x in v["dex3_mesh_pad_shift_m"].items()} == dex3.MESH_PAD_SHIFT_M
    assert v["dex3_pad_proud_m"] == dex3.PAD_PROUD_M
    assert v["dex3_collision_geometry"]["default"] == dex3.COLLISION
    assert tuple(v["dex3_collision_geometry"]["variants"]) == dex3.COLLISIONS
    mat = v["dex3_visual_material"]
    assert " ".join(f"{x:g}" for x in mat["rgba"]) == "0.1 0.1 0.11 1"
    assert dex3.VISUAL_MATERIAL_ATTRS["rgba"] == "0.10 0.10 0.11 1"
