"""The robot's own look in the simulator (visual only): the R1's white / navy / dark plastic on its link meshes.

Moved out of the photo-look module in session R (the scene's photo look stays with its scene); `world.build_world`
applies it with look="photo".
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

# ---------------------------------------------------------------- the robot's own look (session 4)
# r1_mjlab.xml draws every R1 link in one lavender grey (0.792, 0.820, 0.933); the real R1 is white plastic (photos and
# the real head camera's frames of our robot, 9 Oct 2026) with small navy
# details (discs on the shoulder caps, bands at the shoulders, the visor's "A") and black hands. The link meshes are not
# split per part, so one colour per link: white everywhere (a navy upper arm was tried and looked wrong next to the
# photos: the arms read white-silver in the head frames); the Dex3 hands keep their matte black
# (dex3.VISUAL_MATERIAL_ATTRS). Visual only: materials and rgba of mesh geoms that never collide.
R1_STOCK_RGBA = (0.79216, 0.81961, 0.93333, 1.0)
ROBOT_MATERIALS: dict[str, dict[str, Any]] = {
    "white": {"rgba": [0.93, 0.93, 0.92, 1.0], "specular": 0.5, "shininess": 0.6},
    "navy": {"rgba": [0.13, 0.19, 0.36, 1.0], "specular": 0.4, "shininess": 0.5},
    "dark": {"rgba": [0.10, 0.10, 0.11, 1.0], "specular": 0.3, "shininess": 0.4},
}
ROBOT_LINK_LOOK: dict[str, str] = {}  # link-name suffix -> a ROBOT_MATERIALS key; default "white"


def dress_robot(spec: Any) -> int:
    """The R1's white / navy / dark materials on its stock-coloured link meshes (visual only). Returns how many geoms
    were dressed."""
    for name, attrs in ROBOT_MATERIALS.items():
        spec.add_material(name=f"r1mat_{name}", **attrs)
    n = 0
    stock = np.asarray(R1_STOCK_RGBA)
    for g in spec.geoms:
        if g.type != mujoco.mjtGeom.mjGEOM_MESH or g.material or not np.allclose(np.asarray(g.rgba), stock, atol=1e-3):
            continue
        body = g.parent.name
        look = next((v for k, v in ROBOT_LINK_LOOK.items() if body.endswith(k)), "white")
        g.material = f"r1mat_{look}"
        g.rgba = [1.0, 1.0, 1.0, 1.0]
        n += 1
    return n
