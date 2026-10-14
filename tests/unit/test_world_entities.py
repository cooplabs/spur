"""r1mc3.world entities + WorldModel (DESIGN.md §7.1): declared / measured entities, frames, snapshots."""

from __future__ import annotations

import math

import pytest
from r1mc3.world import Container, Object, Pose, Surface, WorldModel, entity_from_dict, entity_to_dict, moved


def test_declare_snapshot_versions_and_remove() -> None:
    w = WorldModel()
    s0 = w.snapshot()
    w.declare(Surface("table", Pose.xyz_yaw(0.5, 0.0, 0.76), size=(0.6, 1.2, 0.04)))
    w.declare(Object("box", Pose.xyz_yaw(0.4, 0.1, 0.785), size=(0.025, 0.025, 0.025)))
    s1 = w.snapshot()
    assert s1.version == s0.version + 2 and len(s1.entities) == 2
    assert s1.get("box") is not None and s1.get("nope") is None
    assert w.remove("box") and not w.remove("box")
    assert len(s1.entities) == 2  # a snapshot never changes
    assert [e.id for e in w.snapshot().entities] == ["table"]


def test_redeclare_replaces_and_moved_stamps_measured() -> None:
    w = WorldModel()
    t = w.declare(Surface("table", Pose.xyz_yaw(0.5, 0.0, 0.76), size=(0.6, 1.2, 0.04)))
    t2 = moved(t, Pose.xyz_yaw(0.52, 0.0, 0.76))
    w.declare(t2)
    s = w.snapshot()
    assert len(s.entities) == 1 and s.entities[0].pose.pos[0] == 0.52 and s.entities[0].source == "measured"


@pytest.mark.parametrize(
    "bad",
    [
        Surface("t", Pose((0.0, 0.0, 0.7), frame="table"), size=(1.0, 1.0, 0.04)),  # unknown frame
        Surface("t", Pose((math.nan, 0.0, 0.7)), size=(1.0, 1.0, 0.04)),
        Surface("t", Pose(), size=(1.0, -1.0, 0.04)),
        Container("c", Pose(), size_outer=(0.3, 0.2, 0.05), wall_m=0.2),
        Object("o", Pose(), shape="cone", size=(0.1,)),  # type: ignore[arg-type]
        Object("", Pose()),
    ],
)
def test_bad_entities_are_refused(bad: object) -> None:
    with pytest.raises(ValueError):
        WorldModel().declare(bad)  # type: ignore[arg-type]


def test_json_round_trip() -> None:
    d = {"kind": "container", "id": "bin", "pose": {"pos": [0.45, 0.3, 0.76], "yaw": 0.1},
         "size_outer": [0.35, 0.26, 0.045], "wall_m": 0.003, "floor_m": 0.003}  # fmt: skip
    e = entity_from_dict(d)
    assert isinstance(e, Container) and abs(e.pose.quat[3] - math.sin(0.05)) < 1e-12
    back = entity_from_dict({k: v for k, v in entity_to_dict(e).items() if k not in ("source", "t_ns")})
    assert back.pose == e.pose and back.size_outer == e.size_outer  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        entity_from_dict({"kind": "robot", "id": "x", "pose": {"pos": [0, 0, 0]}})
