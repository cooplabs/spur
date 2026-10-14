"""r1d's planned arm goals (DESIGN.md §5.1 L2 + §7, session 9): arm.reach / hover / lower / carry / move_relative /
plan and world.declare / list / clear, through the planner (r1mc3.plan, inline here: a pure function of the world
snapshot) and the REAL v3 core in lockstep (the Rig of test_service_lockstep.py). Each leg of a plan is one TRAJECTORY
command; the next leg starts when the executor reports the previous DONE.

On the base (6449aee) every op here answered UNKNOWN_COMMAND.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import r1mc3.plan as planner
from r1mc3.service.codes import Code
from r1mc3.service.goals import hand_points
from r1sim import cpp_policy, world
from r1sim.v3_core import GROUP_ARM_LEFT, GROUP_JOINTS, SOURCE_HOLD

from tests.integration.test_service_lockstep import COL, Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
LEFT = list(GROUP_JOINTS[GROUP_ARM_LEFT])
QCOLS = [COL["q"] + j - 14 for j in LEFT]
# a low table under the hanging-forearm hands (base frame: the soles' midpoint, z up): top z 0.40, x 0.10 .. 0.50
TABLE = {"kind": "surface", "id": "table", "pose": {"pos": [0.30, 0.0, 0.38]}, "size": [0.4, 1.0, 0.04]}
# error_mm = the measured grasp centre (the core arm model's FK of the measured joints) vs the planner's predicted one:
# the same point definition since planner 33b7fb0 (before: ~20 mm apart - the IMU mount rotation, api_TO_PLANNER.md 5)
ERROR_MM_NOW = 10.0


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _rig(w: world.World) -> Rig:
    rig = Rig(w)
    rig.svc.planner = planner
    return rig


def _after(rig: Rig, res: dict[str, Any], key: str, fn: Any, t: float) -> None:
    """Run fn once res[key]'s goal has ended (polled every 0.1 s of simulated time from t)."""

    def poll() -> None:
        if key in res and res[key].result is not None:
            fn()
        else:
            rig.at(rig.t_ns * 1e-9 + 0.1, poll)

    rig.at(t, poll)


def _off(c: np.ndarray, d: list[float]) -> list[float]:
    return [float(x) for x in c + np.asarray(d, dtype=np.float64)]


def _centre(rig: Rig) -> np.ndarray:
    pts = hand_points(rig.port, None)
    assert pts is not None
    return np.asarray(pts[0, 4], dtype=np.float64)


def test_world_ops_and_the_dry_run(w: world.World) -> None:
    rig = _rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        c = _centre(rig)
        res["decl"] = rig.svc.submit({"op": "world.declare", "entity": TABLE}).result
        res["bad"] = rig.svc.submit({"op": "world.declare", "entity": {"kind": "cloud", "id": "x"}}).result
        res["list"] = rig.svc.submit({"op": "world.list"}).result
        res["plan"] = rig.svc.submit(
            {"op": "arm.plan", "goal": {"op": "arm.reach", "arm": "left", "position": _off(c, [0.04, 0, 0.03])}}
        ).result
        res["far"] = rig.svc.submit(
            {"op": "arm.plan", "goal": {"op": "arm.reach", "arm": "left", "position": [1.5, 0.2, 0.5]}}
        ).result
        res["notarm"] = rig.svc.submit({"op": "arm.plan", "goal": {"op": "hand.grasp", "side": "left"}}).result
        res["clear"] = rig.svc.submit({"op": "world.clear", "entity_id": "table"}).result
        res["list2"] = rig.svc.submit({"op": "world.list"}).result

    rig.at(1.0, go)
    tr = rig.run(1.3)
    assert res["decl"].ok and res["decl"].data["entity"]["id"] == "table"
    assert res["bad"].code == Code.INVALID_ARGS
    assert [e["id"] for e in res["list"].data["entities"]] == ["table"]
    p = res["plan"]
    assert p.ok and not p.moved and p.data["legs"][0]["knots"] >= 2 and p.data["plan_id"], p
    assert res["far"].code == Code.UNREACHABLE and not res["far"].moved
    assert res["notarm"].code == Code.INVALID_ARGS
    assert res["clear"].ok and res["list2"].data["entities"] == []
    assert np.ptp(tr[tr[:, COL["t"]] > 1.0][:, QCOLS], axis=0).max() < 1e-9  # a dry run moves nothing


def test_hover_from_the_resting_arm(w: world.World) -> None:
    rig = _rig(w)
    res: dict[str, Any] = {}

    def hover() -> None:
        rig.svc.submit({"op": "world.declare", "entity": TABLE})
        c = _centre(rig)
        # position[2] = the datum (the planner, 33b7fb0): the lowest fingertip 6 cm above the table top (z 0.40)
        p = _off(c, [0.03, 0.0, 0.0])
        p[2] = 0.40
        res["hover"] = rig.svc.submit({"op": "arm.hover", "arm": "left", "position": p, "height_m": 0.06,
                                       "wait": True})  # fmt: skip

    rig.at(1.0, hover)
    rig.run(7.0)
    h = res["hover"].result
    assert h is not None and h.ok and h.data["state"] == "succeeded", h
    # s10 (planner 06f8913, go_line returns the leg's STORED end): from the resting arm the hand is below the hover
    # pose, so the descend is zero-length and dropped (before: a ~3 mrad descend, an artefact of the chaining bug)
    assert h.data["legs"][0] == "lift" and h.data["legs"][-1] in ("transit", "descend")
    assert h.data["legs_done"] == len(h.data["legs"])
    assert abs(h.data["predicted"]["lowest_tip_z_m"] - 0.46) < 0.005  # the lowest fingertip 6 cm above the top
    assert h.data["error_mm"] < ERROR_MM_NOW, h.data
    assert int(rig.ctl.core.status()["upper_rejected"]) == 0


@pytest.mark.slow
def test_reach_lower_carry(w: world.World) -> None:
    rig = _rig(w)
    res: dict[str, Any] = {}

    def reach() -> None:
        rig.svc.submit({"op": "world.declare", "entity": TABLE})
        c = _centre(rig)
        res["reach"] = rig.svc.submit({"op": "arm.reach", "arm": "left", "position": _off(c, [0.06, 0.0, 0.05]),
                                       "wait": True})  # fmt: skip

    def lower() -> None:
        res["c_lower"] = _centre(rig)
        res["lower_contact"] = rig.svc.submit({"op": "arm.lower", "arm": "left", "until": "contact"}).result
        res["lower"] = rig.svc.submit({"op": "arm.lower", "arm": "left", "until": "depth", "max_depth_m": 0.03,
                                       "wait": True})  # fmt: skip

    def carry() -> None:
        c = _centre(rig)
        res["carry"] = rig.svc.submit({"op": "arm.carry", "arm": "left", "position": _off(c, [0.0, -0.03, 0.0]),
                                       "lift_m": 0.03, "wait": True})  # fmt: skip

    rig.at(1.0, reach)
    _after(rig, res, "reach", lower, 1.1)
    _after(rig, res, "lower", carry, 1.2)
    tr = rig.run(16.0)
    r = res["reach"].result
    assert r is not None and r.ok and r.data["state"] == "succeeded", r
    assert r.data["planned_on"] == "inline" and r.data["plan_id"]
    assert r.data["error_mm"] < ERROR_MM_NOW, r.data
    assert res["lower_contact"].code == Code.UNSUPPORTED  # no pad-contact publisher in this r1d: refused, not blind
    lo = res["lower"].result
    assert lo is not None and lo.ok, lo
    dz = lo.data["hand_now"][2] - res["c_lower"][2]
    assert -0.04 < dz < -0.02, dz  # 3 cm down (the measured hand, FK)
    ca = res["carry"].result
    assert ca is not None and ca.ok and ca.data["legs"] == ["rise", "travel", "lower"], ca
    assert ca.data["legs_done"] == 3
    assert int(rig.ctl.core.status()["upper_rejected"]) == 0  # every leg's first knot met the executor's target
    end = tr[tr[:, COL["t"]] > 15.8]
    assert np.all(end[:, COL["src"]] == SOURCE_HOLD)


def test_planned_goal_refusals(w: world.World) -> None:
    rig = _rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        res["far"] = rig.svc.submit({"op": "arm.reach", "arm": "left", "position": [1.5, 0.2, 0.5], "wait": True})
        res["frame"] = rig.svc.submit({"op": "arm.reach", "arm": "left", "position": [0.2, 0.2, 0.5],
                                       "frame": "moon", "wait": True})  # fmt: skip

    rig.at(1.0, go)
    tr = rig.run(1.5)
    far = res["far"].result
    assert far is not None and far.code == Code.UNREACHABLE and not far.moved and far.data["state"] == "rejected"
    fr = res["frame"].result
    assert fr is not None and fr.code == Code.UNKNOWN_FRAME and not fr.moved
    assert np.ptp(tr[tr[:, COL["t"]] > 1.0][:, QCOLS], axis=0).max() < 1e-9
    rig.svc.planner = None
    none = rig.svc.submit({"op": "arm.reach", "arm": "left", "position": [0.2, 0.2, 0.5]}).result
    assert none is not None and none.code == Code.UNSUPPORTED


@pytest.mark.slow
def test_the_planner_worker_path(w: world.World) -> None:
    """r1d's default: the planner in its own process (PlannerWorker): the goal is active while the plan is computed
    (wall clock), r1d's tick keeps running, the plan's legs start when it is back."""
    rig = Rig(w)
    with planner.PlannerWorker() as worker:
        rig.svc.planner = worker
        res: dict[str, Any] = {}

        def go() -> None:
            rig.svc.submit({"op": "world.declare", "entity": TABLE})
            c = _centre(rig)
            res["g"] = rig.svc.submit({"op": "arm.reach", "arm": "left", "position": _off(c, [0.04, 0.0, 0.03]),
                                       "wait": True})  # fmt: skip
            res["dry"] = rig.svc.submit({"op": "arm.plan", "goal": {"op": "arm.reach", "arm": "left",
                                                                    "position": [1.5, 0.2, 0.5]}})  # fmt: skip

        rig.at(1.0, go)
        t_sim = 1.0
        while (res.get("g") is None or res["g"].result is None) and t_sim < 30.0:
            rig.run(0.5)  # the worker plans in wall time while the sim advances
            t_sim += 0.5
    g = res["g"].result
    assert g is not None and g.ok and g.data["planned_on"] == "worker", g
    d = res["dry"].result
    assert d is not None and d.code == Code.UNREACHABLE and not d.moved
