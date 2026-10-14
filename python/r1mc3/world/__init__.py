"""r1mc3.world: the one world model (DESIGN.md §7.1) - declared entities, snapshots and the one geometry engine.

w = WorldModel()
w.declare(Surface("table", Pose.xyz_yaw(0.45, 0.0, 0.76), size=(0.6, 1.2, 0.04)))
geo = Geometry(w.snapshot())          # MuJoCo collision / signed distance on the robot's MJCF + the entities
"""

from .entities import (
    FRAMES,
    Container,
    Entity,
    Object,
    Obstacle,
    Pose,
    Surface,
    WorldModel,
    WorldSnapshot,
    entity_from_dict,
    entity_to_dict,
    moved,
)
from .geometry import ARM_JOINTS, POINTS, Geometry, Held, Hit

__all__ = [
    "ARM_JOINTS",
    "FRAMES",
    "POINTS",
    "Container",
    "Entity",
    "Geometry",
    "Held",
    "Hit",
    "Object",
    "Obstacle",
    "Pose",
    "Surface",
    "WorldModel",
    "WorldSnapshot",
    "entity_from_dict",
    "entity_to_dict",
    "moved",
]
