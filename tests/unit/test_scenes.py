"""The simulator's generic scenes (r1sim.scenes) and the generic default world (session R)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from r1mc3.world.entities import entity_from_dict
from r1sim import scenes, world


def _bodies(m: mujoco.MjModel) -> set[str]:
    return {m.body(i).name for i in range(m.nbody)}


def test_the_default_world_is_the_robot_alone_on_its_floor() -> None:
    w = world.build_world("real", visual=False)
    alone = world.build_world("real", visual=False, scene=None)
    assert w.scene is not None and w.scene.name == "floor" and w.scene.objects == ()
    assert alone.scene is None
    assert _bodies(w.model) == _bodies(alone.model)  # "floor" adds nothing to the robot's own floor
    assert w.model.nbody == alone.model.nbody and w.model.ngeom == alone.model.ngeom
    assert world.build_world("real", visual=False, scene="none").scene is None
    assert world.build_world("real", visual=False, with_scene=False, scene="table").scene is None  # older callers


def test_the_table_scene_holds_a_table_and_three_loose_objects() -> None:
    w = world.build_world("real", visual=False, scene="table")
    info = w.scene
    assert isinstance(info, scenes.SceneInfo) and info.table_top_z == pytest.approx(0.78)
    assert [o.name for o in info.objects] == ["box", "can", "ball"]
    m = w.model
    d = mujoco.MjData(m)
    d.qpos[:] = w.initial_qpos()
    mujoco.mj_forward(m, d)
    top = m.geom("table_top")
    assert d.geom_xpos[top.id][2] + m.geom_size[top.id][2] == pytest.approx(scenes.TABLE_TOP_Z_M)
    near_edge = d.geom_xpos[top.id][0] - m.geom_size[top.id][0]
    assert near_edge == pytest.approx(scenes.TABLE_EDGE_X_M)
    # the objects rest on the table (the table alone on a floor, 0.5 s of physics: no robot to hold up)
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.05])
    alone = scenes.add_table(spec)
    m2 = spec.compile()
    d2 = mujoco.MjData(m2)
    for _ in range(int(0.5 / m2.opt.timestep)):
        mujoco.mj_step(m2, d2)
    for o in alone.objects:
        z = float(d2.xpos[m2.body(o.name).id][2])
        assert z == pytest.approx(scenes.TABLE_TOP_Z_M + o.half_height, abs=0.004), o.name
        assert np.allclose(d2.xpos[m2.body(o.name).id][:2], o.pos[:2], atol=0.005), o.name


def test_the_table_entities_are_r1d_world_entities() -> None:
    w = world.build_world("real", visual=False, scene="table")
    ents = [entity_from_dict(e) for e in scenes.table_entities(w.scene)]
    assert [e.kind for e in ents] == ["surface", "object", "object", "object"]
    table = ents[0]
    assert table.pose.pos[2] == pytest.approx(scenes.TABLE_TOP_Z_M)  # a surface's pose = its top face's centre
    assert scenes.table_entities() == scenes.table_entities(w.scene)  # the constants = what the scene built
    ball = ents[3]
    assert ball.kind == "object" and np.allclose(ball.pose.pos, w.scene.object("ball").pos)


def test_resolve_takes_names_builders_and_module_strings() -> None:
    assert scenes.resolve(None) is None and scenes.resolve("none") is None
    assert scenes.resolve("table") is scenes.add_table
    assert scenes.resolve("r1sim.scenes:add_table") is scenes.add_table

    def mine(spec: mujoco.MjSpec, **_kw: object) -> str:
        b = spec.worldbody.add_body()
        b.name = "my_block"
        g = b.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.05, 0.05, 0.05]
        b.pos = [1.0, 0.0, 0.05]
        return "mine"

    assert scenes.resolve(mine) is mine
    w = world.build_world("real", visual=False, scene=mine)
    assert w.scene == "mine" and "my_block" in _bodies(w.model)
    with pytest.raises(ValueError):
        scenes.resolve("no_such_scene")
    with pytest.raises(ModuleNotFoundError):
        scenes.resolve("no_such_package.module:thing")
