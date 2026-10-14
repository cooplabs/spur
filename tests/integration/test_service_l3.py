"""r1d's L3 + the remaining L2 ops (DESIGN.md §5.1, session 9) against the REAL v3 core in lockstep (the Rig of
test_service_lockstep.py): skill.ready / skill.table_ready (named postures, one arm at a time, through arm.move_joint),
seq.run (steps as child goals; the first failure ends it), head.look_at, body.pose (the policy's lean within its
trained range), and arm.move_joint checked against the world when r1d has a planner.

On the base (6449aee) every op here answered UNKNOWN_COMMAND.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import pytest
import r1mc3.plan as planner
from r1mc3.service.codes import Code
from r1mc3.service.goals import hand_points, port_state
from r1mc3.skills import posture
from r1mc3.world import Geometry
from r1sim import cpp_policy, world
from r1sim.v3_core import GROUP_ARM_LEFT, GROUP_JOINTS, MODE_BALANCE

from tests.integration.test_service_lockstep import COL, Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
LEFT = list(GROUP_JOINTS[GROUP_ARM_LEFT])
QCOLS = [COL["q"] + j - 14 for j in LEFT]
HEAD = [COL["q"] + 24 - 14, COL["q"] + 25 - 14]


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def test_skill_table_ready_then_ready_one_arm(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: res.__setitem__("tr", rig.svc.submit({"op": "skill.table_ready", "arms": "left", "speed": 1.0,
                                                              "wait": True})))  # fmt: skip

    def back() -> None:
        res["tilt"] = float(rig.ctl.core.status()["tilt_rad"])
        res["r"] = rig.svc.submit({"op": "skill.ready", "arms": "left", "speed": 1.0, "wait": True})

    rig.at(5.0, back)
    tr = rig.run(9.0)
    out = res["tr"].result
    assert out is not None and out.ok and out.data["state"] == "succeeded", out
    assert [s["op"] for s in out.data["steps"]] == ["arm.move_joint"] and out.data["steps"][0]["state"] == "succeeded"
    at = tr[(tr[:, COL["t"]] > 4.5) & (tr[:, COL["t"]] < 5.0)]
    assert np.max(np.abs(at[:, QCOLS] - posture("table_ready", "left"))) < 1e-6
    assert np.degrees(res["tilt"]) < 3.0 and int(rig.ctl.core.mode) == MODE_BALANCE  # still standing, balancing
    r = res["r"].result
    assert r is not None and r.ok, r
    end = tr[tr[:, COL["t"]] > 8.5]
    assert np.max(np.abs(end[:, QCOLS] - posture("ready", "left"))) < 1e-6


def test_seq_run_steps_and_the_first_failure(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        q0 = rig.ctl.last_q[LEFT].copy()
        res["q0"] = q0
        good = [
            {"op": "head.look_at", "position": [0.6, 0.1, 0.6]},
            {"op": "arm.move_joint", "arm": "left", "q": (q0 + np.array([0, 0.1, 0, 0, 0])).tolist(), "speed": 1.0},
            {"op": "status"},
        ]
        res["ok"] = rig.svc.submit({"op": "seq.run", "steps": good, "wait": True})

    def bad() -> None:
        q0 = res["q0"]
        steps = [
            {"op": "arm.move_joint", "arm": "left", "q": (q0 + 1.5).tolist(), "duration_s": 0.3},  # too fast: LIMIT
            {"op": "arm.move_joint", "arm": "left", "q": q0.tolist()},
        ]
        res["bad"] = rig.svc.submit({"op": "seq.run", "steps": steps, "wait": True})
        res["nested"] = rig.svc.submit({"op": "seq.run", "steps": [{"op": "seq.run", "steps": [{"op": "status"}]}]})

    rig.at(1.0, go)
    rig.at(4.0, bad)
    tr = rig.run(4.5)
    out = res["ok"].result
    assert out is not None and out.ok and out.data["state"] == "succeeded", out
    assert [s["ok"] for s in out.data["steps"]] == [True, True, True] and out.moved
    b = res["bad"].result
    assert b is not None and b.code == Code.STEP_FAILED and b.data["failed_step"] == 0, b
    assert len(b.data["steps"]) == 1 and b.data["steps"][0]["code"] == "LIMIT"  # the second step never ran
    assert res["nested"].result.code == Code.INVALID_ARGS
    after = tr[tr[:, COL["t"]] > 4.0]
    assert np.ptp(after[:, QCOLS], axis=0).max() < 1e-9  # nothing moved after the failed step


def test_canceling_a_sequence_cancels_its_running_step(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        q0 = rig.ctl.last_q[LEFT].copy()
        steps = [
            {"op": "arm.move_joint", "arm": "left", "q": (q0 + np.array([0, 0.3, 0, -0.3, 0])).tolist(), "speed": 0.2}
        ]
        res["s"] = rig.svc.submit({"op": "seq.run", "steps": steps}).result

    def cancel() -> None:
        res["active"] = rig.svc.status()["goals"]
        res["c"] = rig.svc.submit({"op": "goal.cancel", "goal_id": res["s"].data["goal_id"]}).result

    rig.at(1.0, go)
    rig.at(1.6, cancel)
    tr = rig.run(2.4)
    child = [g for g in res["active"] if g["goal_op"] == "arm.move_joint"]
    assert len(res["active"]) == 2 and child[0]["parent"] == res["s"].data["goal_id"]
    assert res["c"].ok and res["c"].data["state"] == "canceled"
    assert rig.svc.status()["goals"] == []
    held = tr[tr[:, COL["t"]] > 2.0]
    assert np.ptp(held[:, QCOLS], axis=0).max() < 1e-9  # the arm holds where the canceled step left it


def test_head_look_at_points_the_camera(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}
    target = [1.0, 0.25, 0.8]  # ~1 m ahead, left, below the camera
    rig.at(1.0, lambda: res.__setitem__("h", rig.svc.submit({"op": "head.look_at", "position": target, "speed": 1.0,
                                                             "wait": True})))  # fmt: skip
    rig.run(3.5)
    out = res["h"].result
    assert out is not None and out.ok and out.data["clipped"] is False, out
    assert out.data["head_yaw_deg"] > 5.0 and out.data["head_pitch_deg"] > 5.0  # yaw + = look LEFT, pitch + = DOWN
    # where the camera site's forward (x) axis points now, from the measured joints (the robot's own geometry)
    st = port_state(rig.port)
    assert st is not None
    geo = Geometry(None)
    geo.set_stance(np.asarray(st["q"]), np.asarray(st["imu_quat"]))
    site = mujoco.mj_name2id(geo.m, mujoco.mjtObj.mjOBJ_SITE, "head_camera")
    cam = geo.d.site_xpos[site]
    fwd = geo.d.site_xmat[site].reshape(3, 3)[:, 0]
    d = np.asarray(target) - cam
    err = np.degrees(np.arccos(np.dot(fwd, d) / np.linalg.norm(d)))
    assert err < 3.0, err


def test_body_pose_leans_within_the_trained_range(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def lean() -> None:
        res["l3"] = rig.svc.submit({"op": "body.pose", "lean_deg": 3.0}).result
        res["l7"] = rig.svc.submit({"op": "body.pose", "lean_deg": 7.0}).result
        res["h"] = rig.svc.submit({"op": "body.pose", "height": 0.7}).result

    def mid() -> None:
        res["lean_mid"] = float(rig.ctl.core.status()["lean"])
        res["q"] = rig.svc.submit({"op": "body.pose"}).result
        res["l0"] = rig.svc.submit({"op": "body.pose", "lean_deg": 0.0}).result

    rig.at(1.0, lean)
    rig.at(3.0, mid)
    rig.run(5.0)
    assert res["l3"].ok and res["l3"].data["trained_range_deg"] == [0.0, 5.0]
    assert res["l7"].code == Code.LIMIT and not res["l7"].moved
    assert res["h"].code == Code.INVALID_ARGS
    assert abs(res["lean_mid"] - np.radians(3.0)) < 0.005, res["lean_mid"]  # the policy sees the commanded lean
    assert res["q"].data["lean_deg"] == pytest.approx(3.0)
    assert res["l0"].ok
    assert abs(float(rig.ctl.core.status()["lean"])) < 0.005  # back upright after lean 0 (renewal stopped, expired)
    assert int(rig.ctl.core.mode) == MODE_BALANCE


def test_move_joint_is_checked_against_the_world_with_a_planner(w: world.World) -> None:
    rig = Rig(w)
    rig.svc.planner = planner
    res: dict[str, Any] = {}

    def go() -> None:
        q_goal = posture("table_ready", "left")
        st = port_state(rig.port)
        assert st is not None
        q26 = np.asarray(st["q"]).copy()
        q26[LEFT] = q_goal
        pts = hand_points(rig.port, None, {**st, "q": q26})
        assert pts is not None
        box = {"kind": "obstacle", "id": "box", "pose": {"pos": pts[0, 4].tolist()}, "shape": "box",
               "size": [0.12, 0.12, 0.12]}  # fmt: skip
        res["decl"] = rig.svc.submit({"op": "world.declare", "entity": box}).result
        res["blocked"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": q_goal, "wait": True})

    rig.at(1.0, go)
    tr = rig.run(2.0)
    assert res["decl"].ok, res["decl"]
    out = res["blocked"].result
    assert out is not None and out.code == Code.COLLISION and not out.moved, out
    assert np.ptp(tr[tr[:, COL["t"]] > 1.0][:, QCOLS], axis=0).max() < 1e-9


def _declare_the_sims_static_scene(rig: Rig) -> int:
    """Every collidable box welded to the simulator's world (whatever scene it built: table top, legs, walls, mounts)
    declared as an obstacle in the base frame (the feet sites' midpoint on the sole plane, the feet's yaw)."""
    m, d = rig.sim.m, rig.sim.d
    feet = [d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)] for s in ("left_foot", "right_foot")]
    o = np.array([(feet[0][0] + feet[1][0]) / 2, (feet[0][1] + feet[1][1]) / 2, min(feet[0][2], feet[1][2])])
    n = 0
    for g in range(m.ngeom):
        if m.body_weldid[m.geom_bodyid[g]] != 0 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        rot = d.geom_xmat[g].reshape(3, 3)
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        pose = {"pos": (d.geom_xpos[g] - o).tolist(), "yaw": yaw}
        ent = {
            "kind": "obstacle",
            "id": f"geom{g}",
            "shape": "box",
            "size": m.geom_size[g].tolist(),  # half-sizes (r1mc3.world Obstacle); s10: was 2x (every box doubled)
            "pose": pose,
        }
        r = rig.svc.submit({"op": "world.declare", "entity": ent}).result
        assert r is not None and r.ok, r
        n += 1
    return n
