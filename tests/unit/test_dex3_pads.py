"""The Dex3-1 "real" pad layout (session 4; r1sim.dex3 PAD_LAYOUT docs, models/provenance.yaml "dex3_finger_pads").

The S1 / v2 layout put a 1.5 mm box on the grasp face of every phalanx. Unitree's meshes say: the PROXIMAL pads are a
raised block (20.6..43.3 mm along the phalanx, face 14.65 mm out: 3.65 mm beyond the S1 pad face, 8 mm further out
along it) and the DISTAL pads sit under the smooth bullet-shaped shell (face 7.54 mm out at the S1 pad centre, where
the S1 box stood 2.46 mm proud and reached the rounded nose). These tests pin the layout to the meshes (re-measured by
tools/hands/pad_geometry.py), the compiled pads (proximal = colliding boxes on the block, distal = sensing regions on
the shell that never collide), that the sensor model binds them, and that "s1" stays the default (unchanged physics).
Every test here fails on the session-3 code (no `pads` layout, no dex3.hand_variant)."""

from __future__ import annotations

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


def _tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"r1mc3_tools_hands_{name}", ROOT / "tools" / "hands" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def measured() -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = _tool("pad_geometry").measure()
    return out


@pytest.mark.parametrize("side", dex3.SIDES)
def test_the_layout_constants_are_what_the_meshes_say(
    measured: dict[str, dict[str, dict[str, float]]], side: str
) -> None:
    b = dex3.FINGER_PAD_BLOCK
    for link in ("index_0", "thumb_1"):  # the thumb's proximal phalanx is the finger's mesh turned
        m = measured[side][link]
        assert m["along_lo"] == pytest.approx(b["along_lo"], abs=1e-4)
        assert m["along_hi"] == pytest.approx(b["along_hi"], abs=1e-4)
        assert m["half_width"] == pytest.approx(b["half_width"], abs=1e-4)
        assert m["face"] == pytest.approx(b["face"], abs=5e-5)
    for link in ("index_1", "thumb_2"):
        m = measured[side][link]
        assert float(m["face_at_tip_pad_centre"]) == pytest.approx(dex3.FINGER_TIP_PAD_FACE_M, abs=5e-5)
        # the S1 tip pad face (the capsule radius 8.5 mm + its 1.5 mm box) stood ~2.5 mm proud of the real shell
        assert 0.010 - dex3.FINGER_TIP_PAD_FACE_M == pytest.approx(0.0025, abs=3e-4)


def test_s1_stays_the_default_and_bad_layouts_are_refused() -> None:
    assert dex3.PAD_LAYOUT == "s1"
    for side in dex3.SIDES:
        assert dex3.hand_xml(side) == dex3.hand_xml(side, pads="s1")
        assert dex3.hand_xml(side, pads="real") != dex3.hand_xml(side)
        with pytest.raises(ValueError):
            dex3.hand_xml(side, pads="v2")
    with pytest.raises(ValueError), dex3.hand_variant(pads="v2"):
        pass
    with dex3.hand_variant(collision="mesh", pads="real"):
        assert (dex3.COLLISION, dex3.PAD_LAYOUT) == ("mesh", "real")
    assert (dex3.COLLISION, dex3.PAD_LAYOUT) == ("primitives", "s1")


def _build(collision: str) -> mujoco.MjModel:
    with dex3.hand_variant(collision=collision, pads="real"):
        return model.build_model("real", visual=False)


@pytest.fixture(scope="module", params=dex3.COLLISIONS)
def real_pads(request: pytest.FixtureRequest) -> tuple[str, mujoco.MjModel]:
    return str(request.param), _build(str(request.param))


def _pad_frame(side: str, k: int) -> tuple[int, int, float, float]:
    """(along axis, out axis, along sign, out sign) of pad k in its link frame."""
    g = -1.0 if side == "left" else 1.0
    thumb = k in dex3.FINGER_PADS["thumb"]
    return (1, 0, g, 1.0) if thumb else (0, 1, 1.0, g)


def test_the_compiled_pads_sit_where_the_real_pads_are(real_pads: tuple[str, mujoco.MjModel]) -> None:
    collision, m = real_pads
    b = dex3.FINGER_PAD_BLOCK
    for side in dex3.SIDES:
        for finger, (k_prox, k_tip) in dex3.FINGER_PADS.items():
            for k, proximal in ((k_prox, True), (k_tip, False)):
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_pad_{k}")
                assert g >= 0, (side, finger, k)
                al, out, sa, so = _pad_frame(side, k)
                pos, half = np.asarray(m.geom_pos[g]), np.asarray(m.geom_size[g])
                face = so * pos[out] + half[out]
                lo, hi = sa * pos[al] - half[al], sa * pos[al] + half[al]
                if proximal:  # the raised block: a contact surface (+ PAD_PROUD_M over the hull in the mesh variant)
                    assert int(m.geom_contype[g]) == 1 and int(m.geom_conaffinity[g]) == 1
                    proud = dex3.PAD_PROUD_M if collision == "mesh" else 0.0
                    assert face == pytest.approx(b["face"] + proud, abs=1e-6)
                    assert (lo, hi) == pytest.approx((b["along_lo"], b["along_hi"]), abs=1e-6)
                else:  # under the shell: sensing only, never a contact
                    assert int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0
                    assert face == pytest.approx(dex3.FINGER_TIP_PAD_FACE_M, abs=1e-6)
                    assert (lo, hi) == pytest.approx((0.024, 0.044), abs=1e-6)  # the S1 footprint (HW1 maps it)


def test_the_sensor_model_binds_the_real_pads(real_pads: tuple[str, mujoco.MjModel]) -> None:
    _, m = real_pads
    sm = SensorModel()
    sm.bind(m, model.RobotIndex(m))
    for side in dex3.SIDES:
        hand = sm._hands[side]
        assert [p.k for p in hand.pads] == list(range(9))
        # the link shapes carry their pads' faces (a touch on the shell inside a distal pad's footprint is felt)
        assert sorted(k for ks in hand.link_pads.values() for k in ks) == list(range(9))
