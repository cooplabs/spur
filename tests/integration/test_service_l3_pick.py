"""r1d's session-10 L3 ops (DESIGN.md §5.1): skill.pick / skill.place (generic, over L2: hover -> guarded descent ->
hand.grasp -> lift; carry -> release -> retreat) and par.run (goals together on disjoint resources), with the REAL v3
core in lockstep, the planner inline and the grip tests' geometric fake hand (test_service_l2_hands.HandsRig).

On the base (a041731) every op here answered UNKNOWN_COMMAND (the request types did not exist).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import r1mc3.plan as planner
from r1mc3.hands import grip as G
from r1mc3.service.codes import Code
from r1mc3.service.goals import hand_points
from r1sim import cpp_policy, world
from r1sim.v3_core import GROUP_ARM_LEFT, GROUP_ARM_RIGHT, GROUP_JOINTS

from tests.integration.test_service_l2_hands import HandsRig, _object
from tests.integration.test_service_l2_paths import TABLE

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
LEFT = list(GROUP_JOINTS[GROUP_ARM_LEFT])
RIGHT = list(GROUP_JOINTS[GROUP_ARM_RIGHT])
SUPPORT_TOP = 0.40  # TABLE's top face
OBJ_H = 0.04


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _rig(w: world.World) -> HandsRig:
    rig = HandsRig(w)
    rig.svc.planner = planner
    return rig


def _pick_request(rig: HandsRig) -> dict[str, Any]:
    """An object 3 cm in front of the resting left hand on TABLE; the grasp centre 1.5 cm below where the hover
    leaves it (the hover's own prediction, a dry run: the client's choice of grasp height)."""
    pts = hand_points(rig.port, rig.svc.hands)
    assert pts is not None
    c = pts[0, 4]
    pos = [float(c[0]) + 0.03, float(c[1]), SUPPORT_TOP + OBJ_H / 2]
    hover = {"op": "arm.hover", "arm": "left", "position": [pos[0], pos[1], SUPPORT_TOP], "height_m": OBJ_H + 0.03}
    dry = rig.svc.submit({"op": "arm.plan", "goal": hover}).result
    assert dry is not None and dry.ok, dry
    gz = float(dry.data["predicted"]["grasp_centre_m"][2]) - 0.015
    return {"op": "skill.pick", "arm": "left", "position": pos, "height_m": OBJ_H, "grasp_z_m": gz, "grip": "wrap",
            "width_m": 0.03, "hover_m": 0.03, "lift_m": 0.03}  # fmt: skip


def test_pick_then_place(w: world.World) -> None:
    rig = _rig(w)
    _object(rig, "left", 0.03)  # the fake hand's fingers stop on a 3 cm object
    res: dict[str, Any] = {}

    def pick() -> None:
        rig.svc.submit({"op": "world.declare", "entity": TABLE})
        res["pick"] = rig.svc.submit({**_pick_request(rig), "wait": True})

    def place() -> None:
        res["grips"] = dict(rig.svc.status()["grips"])
        pts = hand_points(rig.port, rig.svc.hands)
        assert pts is not None
        p = [float(x) for x in pts[0, 4] + np.array([0.0, -0.03, -0.01])]
        res["place"] = rig.svc.submit({"op": "skill.place", "arm": "left", "position": p, "lift_m": 0.03,
                                       "retreat_m": 0.02, "wait": True})  # fmt: skip

    def after_pick() -> None:
        if res["pick"].result is None:
            rig.at(rig.t_ns * 1e-9 + 0.2, after_pick)
            return
        place()

    rig.at(1.0, pick)
    rig.at(2.0, after_pick)
    rig.run(22.0)
    pk = res["pick"].result
    assert pk is not None and pk.ok and pk.data["state"] == "succeeded", pk
    ops = [s["op"] for s in pk.data["steps"]]
    assert ops == ["arm.hover", "arm.guarded_move", "hand.grasp", "arm.move_linear"], ops
    assert pk.data["precheck"] and pk.data["precheck"][0]["ok"]  # the hover was planned before anything moved
    assert pk.data["steps"][2]["data"]["holding"] is True
    assert res["grips"]["left"]["state"] == "HOLDING"  # the lift confirmed the grip (the lift check)
    pl = res["place"].result
    assert pl is not None and pl.ok and pl.data["state"] == "succeeded", pl
    assert [s["op"] for s in pl.data["steps"]] == ["arm.carry", "hand.release", "arm.move_linear"]
    assert rig.svc.status()["grips"] == {}  # released
    assert np.max(np.abs(rig.hands.h["left"].q - G.preset("open", "left"))) < 0.1


def test_an_empty_pick_fails_at_the_grasp(w: world.World) -> None:
    rig = _rig(w)  # no object: the fake hand closes on nothing
    res: dict[str, Any] = {}

    def pick() -> None:
        rig.svc.submit({"op": "world.declare", "entity": TABLE})
        res["pick"] = rig.svc.submit({**_pick_request(rig), "wait": True})

    rig.at(1.0, pick)
    rig.run(12.0)
    pk = res["pick"].result
    assert pk is not None and not pk.ok and pk.code == Code.SKILL_FAILED and pk.data["state"] == "aborted", pk
    assert pk.data["failed_step"] == 2 and pk.data["steps"][2]["code"] == "NOT_HOLDING", pk.data["failed_step"]
    assert pk.moved  # the hover and the descent ran
    assert len(pk.data["steps"]) == 3  # no lift after a failed grasp


def test_a_pick_out_of_reach_is_refused_before_anything_moves(w: world.World) -> None:
    rig = _rig(w)
    res: dict[str, Any] = {}

    def pick() -> None:
        rig.svc.submit({"op": "world.declare", "entity": TABLE})
        res["q0"] = rig.ctl.last_q[LEFT].copy()
        res["pick"] = rig.svc.submit({"op": "skill.pick", "arm": "left", "position": [1.2, 0.3, 0.42],
                                      "height_m": 0.04, "wait": True})  # fmt: skip

    rig.at(1.0, pick)
    rig.run(2.5)
    pk = res["pick"].result
    assert pk is not None and pk.code == Code.SKILL_FAILED and not pk.moved, pk
    assert pk.data["stopped_by"]["code"] == "PRECHECK"
    assert np.max(np.abs(rig.ctl.last_q[LEFT] - res["q0"])) < 1e-9


def test_par_run_moves_both_arms_together(w: world.World) -> None:
    rig = _rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        ql = rig.ctl.last_q[LEFT] + np.array([0.0, 0.0, 0.0, -0.2, 0.0])
        qr = rig.ctl.last_q[RIGHT] + np.array([0.0, 0.0, 0.0, -0.2, 0.0])
        steps = [{"op": "arm.move_joint", "arm": "left", "q": ql.tolist()},
                 {"op": "arm.move_joint", "arm": "right", "q": qr.tolist()}]  # fmt: skip
        res["targets"] = (ql, qr)
        res["par"] = rig.svc.submit({"op": "par.run", "steps": steps, "wait": True})
        res["shared"] = rig.svc.submit({"op": "par.run", "steps": [steps[0], steps[0]]}).result
        res["notgoal"] = rig.svc.submit({"op": "par.run", "steps": [{"op": "status"}]}).result

    rig.at(1.0, go)
    rig.run(5.0)
    p = res["par"].result
    assert p is not None and p.ok and p.data["state"] == "succeeded", p
    starts = [rig.svc.l2.table.goals[s["goal_id"]].t_start_ns for s in p.data["steps"]]
    assert abs(starts[0] - starts[1]) <= 20_000_000  # started together (within two r1d ticks)
    assert np.max(np.abs(rig.ctl.last_q[LEFT] - res["targets"][0])) < 1e-3
    assert np.max(np.abs(rig.ctl.last_q[RIGHT] - res["targets"][1])) < 1e-3
    assert res["shared"].code == Code.INVALID_ARGS and res["notgoal"].code == Code.INVALID_ARGS


def test_par_run_first_failure_cancels_the_others(w: world.World) -> None:
    rig = _rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        qr = rig.ctl.last_q[RIGHT] + np.array([0.0, 0.0, 0.0, -0.3, 0.0])
        steps = [{"op": "arm.reach", "arm": "left", "position": [1.5, 0.3, 0.5]},  # refused at its start
                 {"op": "arm.move_joint", "arm": "right", "q": qr.tolist(), "speed": 0.2}]  # fmt: skip
        res["par"] = rig.svc.submit({"op": "par.run", "steps": steps, "wait": True})

    rig.at(1.0, go)
    rig.run(4.0)
    p = res["par"].result
    assert p is not None and not p.ok and p.code == Code.STEP_FAILED and p.data["failed_step"] == 0, p
    states = {s["index"]: s["state"] for s in p.data["steps"]}
    assert states[0] == "rejected" and states[1] == "canceled", states


def test_pick_with_the_level_hand_ranking_on(w: world.World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Session 13 (D48): the level / no-roll ranking at 10 deg (stricter than its default 12) and the pick succeeds -
    the hover is chosen where the grip closes and ends PICK_MIN_DESCENT_M above it. On dcc01a2 (the ranking forced to
    10 deg) the straight-down step was refused: "stop_z_m: 0 < the depth below the grasp centre", depth -0.0208."""
    from r1mc3.plan import orient as O

    monkeypatch.setattr(O, "ROLLED_FREE_DEG", 10.0)
    rig = _rig(w)
    _object(rig, "left", 0.03)
    res: dict[str, Any] = {}

    def pick() -> None:
        rig.svc.submit({"op": "world.declare", "entity": TABLE})
        res["pick"] = rig.svc.submit({**_pick_request(rig), "wait": True})

    rig.at(1.0, pick)
    rig.run(14.0)
    pk = res["pick"].result
    assert pk is not None and pk.ok and pk.data["state"] == "succeeded", pk
    assert [s["op"] for s in pk.data["steps"]] == ["arm.hover", "arm.guarded_move", "hand.grasp", "arm.move_linear"]
