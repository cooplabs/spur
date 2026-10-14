"""The declared world: typed entities, the one world model, immutable snapshots (DESIGN.md §7.1).

Every entity has an id, a pose in a frame, a source (`declared` by a client, `measured` from tags, `perceived` from
the cameras) and the time it was last set. Frames: `base` = v2's base frame, the frame the RT core's arm model and the
executor's surface heights use (rt/include/r1rt/arm_model.hpp "base": origin at the two foot sites' midpoint on the
lower sole, x along the feet's mean yaw, y left, z up; REP-103). Other frames are refused until the frame registry
(measurement, §8.2) exists.

Shapes (all metres; a pose is the shape's own reference point, its yaw / quaternion the shape's axes):
  Surface    a box whose TOP face is the surface: pose = the centre of the top face, size = (x, y, thickness).
  Container  an open box: walls + floor, a hand may be INSIDE above the floor (v2 modelled the basket as a solid slab,
             DESIGN §7.1); pose = the centre of the bottom face (where it stands), size_outer = (x, y, height).
  Obstacle   a box / cylinder / sphere to keep away from: pose = its centre; size = half-sizes as MuJoCo's geom size
             (box (hx, hy, hz), cylinder (r, hz), sphere (r,)).
  Object     the same shapes, something to grasp (it may be touched by the hand; collisions with the arm are not).

The model never names a task: clients declare what is on their table (D32).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Source = Literal["declared", "measured", "perceived"]
Shape = Literal["box", "cylinder", "sphere"]
FRAMES = ("base",)
SOURCES = ("declared", "measured", "perceived")


def _now_ns() -> int:
    return time.monotonic_ns()


@dataclass(frozen=True)
class Pose:
    """A position (m) and an orientation quaternion (w, x, y, z) in `frame`."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    frame: str = "base"

    @staticmethod
    def xyz_yaw(x: float, y: float, z: float, yaw: float = 0.0, frame: str = "base") -> Pose:
        return Pose((float(x), float(y), float(z)), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), frame)

    def rot(self) -> FloatArray:
        w, x, y, z = self.quat
        n = math.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def apply(self, p_local: FloatArray) -> FloatArray:
        return np.asarray(self.rot() @ np.asarray(p_local, dtype=np.float64) + np.asarray(self.pos), dtype=np.float64)


@dataclass(frozen=True)
class _Entity:
    id: str
    pose: Pose
    source: Source = "declared"
    t_ns: int = field(default_factory=_now_ns)

    def check(self) -> None:
        if not self.id or len(self.id) > 64:
            raise ValueError("entity id: 1-64 characters")
        if self.pose.frame not in FRAMES:
            raise ValueError(f"{self.id}: unknown frame {self.pose.frame!r} (known: {', '.join(FRAMES)})")
        if self.source not in SOURCES:
            raise ValueError(f"{self.id}: unknown source {self.source!r}")
        if not all(math.isfinite(v) for v in (*self.pose.pos, *self.pose.quat)):
            raise ValueError(f"{self.id}: non-finite pose")


def _positive(eid: str, name: str, v: tuple[float, ...], n: int) -> None:
    if len(v) != n or not all(math.isfinite(x) and x > 0 for x in v):
        raise ValueError(f"{eid}: {name} must be {n} positive finite numbers")


@dataclass(frozen=True)
class Surface(_Entity):
    size: tuple[float, float, float] = (1.0, 1.0, 0.04)
    kind: Literal["surface"] = "surface"

    def check(self) -> None:
        super().check()
        _positive(self.id, "size", self.size, 3)


@dataclass(frozen=True)
class Container(_Entity):
    size_outer: tuple[float, float, float] = (0.3, 0.2, 0.05)
    wall_m: float = 0.003
    floor_m: float = 0.003
    kind: Literal["container"] = "container"

    def check(self) -> None:
        super().check()
        _positive(self.id, "size_outer", self.size_outer, 3)
        sx, sy, sz = self.size_outer
        if not (0 < self.wall_m < min(sx, sy) / 2 and 0 < self.floor_m < sz):
            raise ValueError(f"{self.id}: wall / floor thickness out of range")


_SHAPE_N = {"box": 3, "cylinder": 2, "sphere": 1}


@dataclass(frozen=True)
class Obstacle(_Entity):
    shape: Shape = "box"
    size: tuple[float, ...] = (0.05, 0.05, 0.05)
    kind: Literal["obstacle"] = "obstacle"

    def check(self) -> None:
        super().check()
        if self.shape not in _SHAPE_N:
            raise ValueError(f"{self.id}: unknown shape {self.shape!r}")
        _positive(self.id, "size", self.size, _SHAPE_N[self.shape])


@dataclass(frozen=True)
class Object(_Entity):
    shape: Shape = "box"
    size: tuple[float, ...] = (0.02, 0.02, 0.02)
    mass_kg: float | None = None
    kind: Literal["object"] = "object"

    def check(self) -> None:
        super().check()
        if self.shape not in _SHAPE_N:
            raise ValueError(f"{self.id}: unknown shape {self.shape!r}")
        _positive(self.id, "size", self.size, _SHAPE_N[self.shape])
        if self.mass_kg is not None and not (math.isfinite(self.mass_kg) and self.mass_kg > 0):
            raise ValueError(f"{self.id}: mass_kg must be positive")

    def top_z(self) -> float:
        """The highest point of the object in its frame (an upright shape: its centre + half height)."""
        h = self.size[0] if self.shape == "sphere" else self.size[-1]
        return float(self.pose.pos[2] + h)


Entity = Surface | Container | Obstacle | Object
KINDS: dict[str, type] = {"surface": Surface, "container": Container, "obstacle": Obstacle, "object": Object}


def entity_from_dict(d: dict[str, object]) -> Entity:
    """A JSON entity (r1d's `world.declare`) -> a checked entity. `pose`: {pos: [3], quat?: [4], yaw?: rad, frame?}."""
    d = dict(d)
    kind = str(d.pop("kind", ""))
    if kind not in KINDS:
        raise ValueError(f"unknown entity kind {kind!r} (known: {', '.join(KINDS)})")
    p = d.pop("pose", None)
    if not isinstance(p, dict) or "pos" not in p:
        raise ValueError("pose: {pos: [x, y, z], yaw | quat, frame}")
    pos = tuple(float(v) for v in p["pos"])
    if len(pos) != 3:
        raise ValueError("pose.pos: 3 numbers")
    frame = str(p.get("frame", "base"))
    if "quat" in p:
        q = tuple(float(v) for v in p["quat"])
        if len(q) != 4:
            raise ValueError("pose.quat: 4 numbers (w, x, y, z)")
        pose = Pose((pos[0], pos[1], pos[2]), (q[0], q[1], q[2], q[3]), frame)
    else:
        pose = Pose.xyz_yaw(pos[0], pos[1], pos[2], float(p.get("yaw", 0.0)), frame)
    for k in ("size", "size_outer"):
        if k in d:
            d[k] = tuple(float(v) for v in d[k])  # type: ignore[attr-defined]
    e = KINDS[kind](pose=pose, **d)
    e.check()
    return e  # type: ignore[no-any-return]


def entity_to_dict(e: Entity) -> dict[str, object]:
    out: dict[str, object] = {"kind": e.kind, "id": e.id, "source": e.source, "t_ns": e.t_ns}
    out["pose"] = {"pos": list(e.pose.pos), "quat": list(e.pose.quat), "frame": e.pose.frame}
    for k in ("size", "size_outer", "wall_m", "floor_m", "shape", "mass_kg"):
        if hasattr(e, k):
            v = getattr(e, k)
            out[k] = list(v) if isinstance(v, tuple) else v
    return out


@dataclass(frozen=True)
class WorldSnapshot:
    """An immutable view of the world at `t_ns` (`version` counts every change): what a plan is computed on."""

    entities: tuple[Entity, ...]
    t_ns: int
    version: int

    def get(self, eid: str) -> Entity | None:
        return next((e for e in self.entities if e.id == eid), None)

    def of_kind(self, kind: str) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.kind == kind)

    def key(self) -> tuple[object, ...]:
        """Equal keys = the same geometry (the planner's model cache)."""
        return tuple((type(e).__name__, e.id, e.pose, getattr(e, "size", None), getattr(e, "size_outer", None),
                      getattr(e, "wall_m", None), getattr(e, "floor_m", None), getattr(e, "shape", None))
                     for e in self.entities)  # fmt: skip


class WorldModel:
    """The one world model (R-e): r1d's `world.*` requests write it, the planner reads snapshots. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ents: dict[str, Entity] = {}
        self._version = 0

    def declare(self, e: Entity) -> Entity:
        e.check()
        with self._lock:
            self._ents[e.id] = e
            self._version += 1
        return e

    def remove(self, eid: str) -> bool:
        with self._lock:
            hit = self._ents.pop(eid, None) is not None
            if hit:
                self._version += 1
            return hit

    def clear(self) -> None:
        with self._lock:
            self._ents.clear()
            self._version += 1

    def snapshot(self) -> WorldSnapshot:
        with self._lock:
            return WorldSnapshot(tuple(self._ents.values()), _now_ns(), self._version)


def moved(e: Entity, pose: Pose, source: Source = "measured") -> Entity:
    """The same entity at a new pose (e.g. re-measured), stamped now."""
    return replace(e, pose=pose, source=source, t_ns=_now_ns())
