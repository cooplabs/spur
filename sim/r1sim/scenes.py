"""The simulator's generic scenes: what stands around the robot when nothing else is asked for.

`r1sim.world.build_world(scene=...)` takes one of:

* ``"floor"`` (the default) - the robot on its own floor plane, nothing else (the robot spec of `r1sim.model` brings
  the floor and the light); ``"none"`` and ``None`` mean the same;
* ``"table"`` - a plain table in front of the robot (0.6 x 1.2 m, top 0.78 m above the floor, its near edge 0.25 m in
  front of the feet) with three loose objects on it: a 5 cm box, a can (6 cm across, 10 cm tall) and a 6 cm ball;
* a builder - any callable ``builder(spec, *, seed, visual, look) -> info`` that adds bodies to the MjSpec before the
  compile and returns what it built (its "scene info", kept as `World.scene`);
* a ``"package.module:attr"`` string naming such a builder - how a scene from another package (your own task scene)
  is named on a command line or in a process's settings: `resolve` imports it.

Frames: world = the robot's base frame (the floor under the middle of the feet, x forward, y left, z up), so a scene's
positions are also what a client declares to r1d's world model (`table_entities`).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mujoco

BUILTIN = ("floor", "table")
NO_SCENE = ("none", "floor")  # names that add nothing to the robot's own floor

# The plain table (estimates of an ordinary table, not a measured one): a fixed body, its top face TABLE_TOP_Z_M above
# the floor, its near edge TABLE_EDGE_X_M in front of the feet, centred on the robot (y = 0).
TABLE_SIZE_M = (0.60, 1.20)  # depth (x) x width (y)
TABLE_TOP_Z_M = 0.78
TABLE_THICKNESS_M = 0.04
TABLE_EDGE_X_M = 0.25
TABLE_LEG_HALF_M = 0.02  # the legs: 4 cm square posts at the corners (inset 3 cm)
TABLE_RGBA = (0.72, 0.62, 0.50, 1.0)
TABLE_LEG_RGBA = (0.40, 0.35, 0.30, 1.0)
OBJECT_MASS_KG = 0.1
OBJECT_FRICTION = (1.0, 0.005, 0.0001)  # MuJoCo's default sliding / torsional / rolling


@dataclass(frozen=True)
class SceneObject:
    """A loose object of a scene: its body, its shape and size (MuJoCo geom size: box half-sizes, cylinder (radius,
    half height), sphere (radius,)) and where it starts (world, the body origin = the shape's centre)."""

    name: str
    shape: str  # "box" | "cylinder" | "sphere"
    size: tuple[float, ...]
    pos: tuple[float, float, float]
    mass_kg: float
    rgba: tuple[float, float, float, float]

    @property
    def body(self) -> str:
        """The body's name (= `name`; the attribute other scenes' objects use too)."""
        return self.name

    @property
    def half_height(self) -> float:
        return float(self.size[0] if self.shape == "sphere" else self.size[-1])


@dataclass(frozen=True)
class SceneInfo:
    """What a built-in scene built (world frame, m)."""

    name: str
    objects: tuple[SceneObject, ...] = ()
    table_body: str | None = None
    table_top_z: float | None = None
    table_edge_x: float | None = None
    table_size: tuple[float, float] | None = None  # depth (x) x width (y)

    def object(self, name: str) -> SceneObject:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)


# (name, shape, size, (x, y) on the table) - the three objects of the "table" scene
TABLE_OBJECTS: tuple[
    tuple[str, str, tuple[float, ...], tuple[float, float], tuple[float, float, float, float]], ...
] = (
    ("box", "box", (0.025, 0.025, 0.025), (0.42, 0.20), (0.85, 0.25, 0.20, 1.0)),
    ("can", "cylinder", (0.03, 0.05), (0.45, -0.15), (0.20, 0.45, 0.85, 1.0)),
    ("ball", "sphere", (0.03,), (0.55, 0.05), (0.95, 0.75, 0.20, 1.0)),
)

_GEOM = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
         "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}  # fmt: skip


def add_table(spec: mujoco.MjSpec, *, seed: int | None = None, visual: bool = True, look: str = "photo") -> SceneInfo:
    """The "table" scene: the plain table + the box, the can and the ball resting on it (free bodies). `seed`,
    `visual` and `look` are accepted for the builder interface; this scene has no variants and one look."""
    del seed, visual, look
    dx, dy = TABLE_SIZE_M
    cx = TABLE_EDGE_X_M + dx / 2.0
    tb = spec.worldbody.add_body()
    tb.name = "table"
    tb.pos = [cx, 0.0, TABLE_TOP_Z_M - TABLE_THICKNESS_M / 2.0]
    top = tb.add_geom()
    top.name = "table_top"
    top.type = mujoco.mjtGeom.mjGEOM_BOX
    top.size = [dx / 2.0, dy / 2.0, TABLE_THICKNESS_M / 2.0]
    top.rgba = list(TABLE_RGBA)
    leg_h = (TABLE_TOP_Z_M - TABLE_THICKNESS_M) / 2.0
    inset = 0.03
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        lg = tb.add_geom()
        lg.name = f"table_leg{k}"
        lg.type = mujoco.mjtGeom.mjGEOM_BOX
        lg.size = [TABLE_LEG_HALF_M, TABLE_LEG_HALF_M, leg_h]
        lg.pos = [sx * (dx / 2.0 - inset), sy * (dy / 2.0 - inset), -TABLE_THICKNESS_M / 2.0 - leg_h]
        lg.rgba = list(TABLE_LEG_RGBA)
    objs: list[SceneObject] = []
    for name, shape, size, (x, y), rgba in TABLE_OBJECTS:
        half_h = size[0] if shape == "sphere" else size[-1]
        pos = (x, y, TABLE_TOP_Z_M + half_h + 0.001)
        b = spec.worldbody.add_body()
        b.name = name
        b.pos = list(pos)
        b.add_freejoint()
        g = b.add_geom()
        g.name = f"{name}_geom"
        g.type = _GEOM[shape]
        g.size = list(size) + [0.0] * (3 - len(size))
        g.mass = OBJECT_MASS_KG
        g.friction = list(OBJECT_FRICTION)
        g.rgba = list(rgba)
        objs.append(SceneObject(name, shape, tuple(size), pos, OBJECT_MASS_KG, rgba))
    return SceneInfo(
        name="table",
        objects=tuple(objs),
        table_body="table",
        table_top_z=TABLE_TOP_Z_M,
        table_edge_x=TABLE_EDGE_X_M,
        table_size=TABLE_SIZE_M,
    )


def add_floor(spec: mujoco.MjSpec, *, seed: int | None = None, visual: bool = True, look: str = "photo") -> SceneInfo:
    """The "floor" scene: nothing beyond the robot's own floor plane."""
    del spec, seed, visual, look
    return SceneInfo(name="floor")


SceneBuilder = Callable[..., Any]
_BUILTIN_BUILDERS: dict[str, SceneBuilder] = {"floor": add_floor, "table": add_table}


def resolve(scene: str | SceneBuilder | None) -> SceneBuilder | None:
    """A scene value -> its builder, or None for no scene. Built-in names ("floor", "table"; "none" = no scene), a
    callable (returned as is), or a "package.module:attr" string (imported here: how another package's scene is named
    without this package ever importing it)."""
    if scene is None or scene == "none":
        return None
    if callable(scene):
        return scene
    if not isinstance(scene, str):
        raise TypeError(f"scene: a name, a builder or 'package.module:attr', not {type(scene).__name__}")
    if scene in _BUILTIN_BUILDERS:
        return _BUILTIN_BUILDERS[scene]
    if ":" in scene:
        mod, _, attr = scene.partition(":")
        obj: Any = importlib.import_module(mod)
        for part in attr.split("."):
            obj = getattr(obj, part)
        if not callable(obj):
            raise TypeError(f"scene {scene!r}: {attr} is not callable")
        return obj  # type: ignore[no-any-return]
    raise ValueError(f"scene: one of {BUILTIN}, 'none', or 'package.module:attr' - not {scene!r}")


def table_entities(info: SceneInfo | None = None) -> list[dict[str, Any]]:
    """r1d `world.declare` entities for the "table" scene (base frame = the sim's world frame): the table top as a
    surface and each object as an object, at the positions the scene starts them at (`info` = a built scene, else the
    scene's constants). A client declares these so the planner knows what the simulator holds."""
    dx, dy = TABLE_SIZE_M
    ents: list[dict[str, Any]] = [
        {
            "kind": "surface",
            "id": "table",
            "pose": {"pos": [TABLE_EDGE_X_M + dx / 2.0, 0.0, TABLE_TOP_Z_M]},
            "size": [dx, dy, TABLE_THICKNESS_M],
        }
    ]
    if info is not None:
        objs = [(o.name, o.shape, o.size, o.pos, o.mass_kg) for o in info.objects]
    else:
        objs = []
        for name, shape, size, (x, y), _rgba in TABLE_OBJECTS:
            half_h = size[0] if shape == "sphere" else size[-1]
            objs.append((name, shape, size, (x, y, TABLE_TOP_Z_M + half_h + 0.001), OBJECT_MASS_KG))
    for name, shape, size, pos, mass in objs:
        ents.append(
            {
                "kind": "object",
                "id": name,
                "shape": shape,
                "pose": {"pos": list(pos)},
                "size": list(size),
                "mass_kg": mass,
            }
        )
    return ents
