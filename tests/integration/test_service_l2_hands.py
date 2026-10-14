"""r1d's hand goals (DESIGN.md §5.1 / §6.2, session 9): hand.grasp / release / preshape through the grip state machine
(r1mc3.hands.grip.GraspRunner), with the REAL v3 core in lockstep for the arm (the lift check) and the grip tests'
geometric fake hand (tests/unit/grip_fake.py: joints track their command, a group stops where its fingertip meets an
object at a given aperture, pads rise when pushed) stepped at r1d's 100 Hz tick.

On the base (6449aee) every op here answered UNKNOWN_COMMAND.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.hands import grip as G
from r1mc3.service.codes import Code
from r1sim import cpp_policy, world
from r1sim.engine import LowCmd, LowState
from r1sim.v3_core import GROUP_ARM_LEFT, GROUP_JOINTS

from tests.integration.test_service_lockstep import Rig
from tests.unit.grip_fake import FakeHand, aperture_stop

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
LEFT = list(GROUP_JOINTS[GROUP_ARM_LEFT])


class FakeHands:
    """Both hands as r1hands offers them to r1d (command / state), each a FakeHand stepped by the rig."""

    def __init__(self) -> None:
        self.h = {s: FakeHand(s) for s in ("left", "right")}

    def command(self, side: str, q: Any, **kw: Any) -> dict[str, Any]:
        kw.pop("ttl_s", None)
        out: dict[str, Any] = self.h[side].command(side, q, **kw)
        return out

    def state(self, side: str) -> dict[str, Any]:
        out: dict[str, Any] = self.h[side].state(side)
        return out

    def step(self) -> None:
        for h in self.h.values():
            h.step()


class HandsRig(Rig):
    def __init__(self, w: world.World) -> None:
        super().__init__(w)
        self.hands = FakeHands()
        self.svc.hands = self.hands

    def _r1d(self, st: LowState) -> LowCmd | None:
        self.hands.step()
        return super()._r1d(st)


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _object(rig: HandsRig, side: str, aperture_m: float) -> None:
    rig.hands.h[side].stops = {
        g: aperture_stop(side, f, aperture_m) for g, f in (("thumb", "index"), ("index", "index"), ("middle", "middle"))
    }


def test_an_empty_grasp_is_not_holding(w: world.World) -> None:
    rig = HandsRig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: res.__setitem__("g", rig.svc.submit({"op": "hand.grasp", "side": "left", "grip": "wrap",
                                                             "wait": True})))  # fmt: skip
    rig.run(4.0)
    out = res["g"].result
    assert out is not None and not out.ok and out.code == Code.NOT_HOLDING and out.moved, out
    assert out.data["state"] == "aborted" and out.data["holding"] is False
    assert out.data["verdict"]["state"] == "NOT_HOLDING" and out.data["verdict"]["reason"]
    assert out.data["stopped_by"]["code"] == "NOT_HOLDING"


def test_a_not_holding_grasp_keeps_the_grip_until_hand_release(w: world.World) -> None:
    """Session 11, the user's "contact decides": r1d's NOT_HOLDING never relaxes the hand by itself - the grip stays
    ticked (its command re-sent at r1d's tick, the closed fingers where they stopped) until hand.release opens it. (On
    1e55bdd: one back-off command 0.03 rad from where the fingers were, then nothing - the hand drifted open.)"""
    rig = HandsRig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: res.__setitem__("g", rig.svc.submit({"op": "hand.grasp", "side": "left", "grip": "wrap",
                                                             "wait": True})))  # fmt: skip

    def after() -> None:
        h = rig.hands.h["left"]
        res["n0"], res["q0"] = h.commands, h.q.copy()
        res["status"] = dict(rig.svc.status()["grips"])

    def later() -> None:
        h = rig.hands.h["left"]
        res["n1"], res["q1"] = h.commands, h.q.copy()
        res["r"] = rig.svc.submit({"op": "hand.release", "side": "left", "wait": True})

    rig.at(4.0, after)
    rig.at(6.0, later)
    rig.run(7.5)
    out = res["g"].result
    assert out is not None and out.code == Code.NOT_HOLDING, out
    assert res["n1"] - res["n0"] >= 150  # 2 s at r1d's 100 Hz tick: the grip re-sent
    assert np.max(np.abs(res["q1"][1:] - res["q0"][1:])) < 0.01  # the fingers stayed where they closed
    assert res["status"]["left"]["state"] == "NOT_HOLDING" and res["status"]["left"]["keeps_grip"] is True
    r = res["r"].result
    assert r is not None and r.ok and r.data["reached"] is True, r
    assert rig.svc.status()["grips"] == {}
    assert np.max(np.abs(rig.hands.h["left"].q - G.preset("open", "left"))) < 0.1


def test_the_operators_hand_command_takes_the_hand_from_a_kept_grip(w: world.World) -> None:
    """Session 11: a NOT_HOLDING grip keeps its grip (re-sent every tick) - an operator's direct `hand.command` (`r1
    hand right 0,0,...`, the runbooks' open word) must take the hand, not be overridden 10 ms later. (Without
    L2.end_grip in service._hand the kept grip re-closes the hand: the fingers stay where the grip holds them.)"""
    rig = HandsRig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: res.__setitem__("g", rig.svc.submit({"op": "hand.grasp", "side": "left", "grip": "wrap",
                                                             "wait": True})))  # fmt: skip
    q_open = G.preset("open", "left")
    rig.at(4.0, lambda: res.__setitem__("h", rig.svc.submit({"op": "hand.command", "side": "left",
                                                             "q": q_open.tolist(), "ttl": 2.0}).result))  # fmt: skip
    rig.run(5.5)
    assert res["g"].result is not None and res["g"].result.code == Code.NOT_HOLDING
    assert res["h"] is not None and res["h"].ok, res["h"]
    assert rig.svc.status()["grips"] == {}
    assert np.max(np.abs(rig.hands.h["left"].q - q_open)) < 0.1  # opened (the kept grip did not re-close it)


def test_a_skills_hand_step_takes_the_hand_from_a_kept_grip(w: world.World) -> None:
    """Session 11: a skill's finger step (table_ready's HandMoveRunner: fingers closed for the way out, open again)
    after a missed grasp must move the hand - the kept NOT_HOLDING grip ends instead of re-closing it every tick."""
    from r1mc3.service.l2 import HandMoveRunner

    rig = HandsRig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: res.__setitem__("g", rig.svc.submit({"op": "hand.grasp", "side": "left", "grip": "wrap",
                                                             "wait": True})))  # fmt: skip
    q_open = G.preset("open", "left")

    def step() -> None:
        res["runner"] = HandMoveRunner(rig.svc.l2, "left", q_open)
        res["start"] = res["runner"].start(rig.t_ns)

    rig.at(4.0, step)
    rig.run(5.5)
    assert res["g"].result is not None and res["g"].result.code == Code.NOT_HOLDING
    assert res["start"] is None and rig.svc.status()["grips"] == {}
    assert np.max(np.abs(rig.hands.h["left"].q - q_open)) < 0.1  # the step's pose, not the kept grip's


def test_a_grasp_holds_the_lift_check_confirms_then_release(w: world.World) -> None:
    rig = HandsRig(w)
    _object(rig, "left", 0.03)
    res: dict[str, Any] = {}

    def grasp() -> None:
        res["g"] = rig.svc.submit({"op": "hand.grasp", "side": "left", "grip": "wrap", "object_class": "rigid",
                                   "width_m": 0.03, "wait": True})  # fmt: skip

    def lift() -> None:  # elbow flexed more (a SMALLER elbow value = more flexed): the hand rises
        res["busy"] = rig.svc.submit({"op": "hand.preshape", "side": "left", "shape": "open"}).result
        q = rig.ctl.last_q[LEFT].copy() + np.array([0.0, 0.0, 0.0, -0.25, 0.0])
        res["lift"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": q.tolist(), "wait": True})

    def release() -> None:
        res["grips_before"] = dict(rig.svc.status()["grips"])
        res["r"] = rig.svc.submit({"op": "hand.release", "side": "left", "wait": True})

    rig.at(1.0, grasp)
    rig.at(3.0, lift)
    rig.at(5.0, release)
    rig.run(6.5)
    g = res["g"].result
    assert g is not None and g.ok and g.data["state"] == "succeeded" and g.data["holding"] is True, g
    assert g.data["verdict"]["state"] == "TENTATIVE"  # the hand's evidence holds; the lift decides
    assert res["busy"].code == Code.RESOURCE_BUSY  # a preshape while the hand holds a grip
    lift_r = res["lift"].result
    assert lift_r is not None and lift_r.ok, lift_r
    assert lift_r.data["hand_now"][2] - lift_r.data["hand_start"][2] > 0.02  # it really rose (FK of the measured q)
    assert res["grips_before"]["left"]["state"] == "HOLDING"  # the lift check confirmed it
    r = res["r"].result
    assert r is not None and r.ok and r.data["reached"] is True, r
    assert rig.svc.status()["grips"] == {}
    q_open = G.preset("open", "left")
    assert np.max(np.abs(rig.hands.h["left"].q - q_open)) < 0.1


def test_preshape_and_the_refusals(w: world.World) -> None:
    rig = HandsRig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        res["p"] = rig.svc.submit({"op": "hand.preshape", "side": "right", "shape": "pinch", "wait": True})
        res["bad"] = rig.svc.submit({"op": "hand.preshape", "side": "left", "shape": "claw"}).result
        res["busy"] = rig.svc.submit({"op": "hand.preshape", "side": "right", "shape": "open"}).result

    rig.at(1.0, go)
    rig.run(3.0)
    p = res["p"].result
    assert p is not None and p.ok and p.data["reached"] is True, p
    assert np.max(np.abs(rig.hands.h["right"].q - G.preshape("pinch", "right"))) < 0.1
    assert res["bad"].code == Code.INVALID_ARGS and "pinch" in res["bad"].details["shapes"]
    assert res["busy"].code == Code.RESOURCE_BUSY
    rig.svc.hands = None
    nohands = rig.svc.submit({"op": "hand.grasp", "side": "left"}).result
    assert nohands is not None and nohands.code == Code.UNSUPPORTED
