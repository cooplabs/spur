"""r1d's session-10 L2 ops (DESIGN.md §5.1): arm.move_path, arm.guarded_move, arm.slide (service/l2_paths.py),
through the planner (r1mc3.plan, inline: a pure function of the world snapshot) and the REAL v3 core in lockstep (the
Rig of test_service_lockstep.py).

On the base (a041731) every op here answered UNKNOWN_COMMAND (the request types did not exist).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import r1mc3.plan as planner
from r1mc3.service.codes import Code
from r1mc3.service.goals import hand_points
from r1sim import cpp_policy, world
from r1sim.v3_core import GROUP_ARM_LEFT, GROUP_JOINTS

from tests.integration.test_service_lockstep import COL, Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
LEFT = list(GROUP_JOINTS[GROUP_ARM_LEFT])
QCOLS = [COL["q"] + j - 14 for j in LEFT]
TABLE = {"kind": "surface", "id": "table", "pose": {"pos": [0.30, 0.0, 0.38]}, "size": [0.4, 1.0, 0.04]}
ERROR_MM = 10.0  # the measured grasp centre vs the planner's predicted one (test_service_l2_planned's bound)


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _rig(w: world.World) -> Rig:
    rig = Rig(w)
    rig.svc.planner = planner
    return rig


def _centre(rig: Rig) -> np.ndarray:
    pts = hand_points(rig.port, None)
    assert pts is not None
    return np.asarray(pts[0, 4], dtype=np.float64)


def _pinch(rig: Rig) -> np.ndarray:
    """The planner's reach point (`pinch` = the thumb-index midpoint) from the measured joints."""
    pts = hand_points(rig.port, None)
    assert pts is not None
    return np.asarray(0.5 * (pts[0, 1] + pts[0, 3]), dtype=np.float64)


def _off(c: np.ndarray, d: list[float]) -> list[float]:
    return [float(x) for x in c + np.asarray(d, dtype=np.float64)]


def test_the_ops_did_not_exist_and_validate_their_arguments(w: world.World) -> None:
    rig = _rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        sub = rig.svc.submit
        res["two"] = sub({"op": "arm.guarded_move", "arm": "left", "delta": [0, 0, 0.01], "stop_z_m": 0.5}).result
        res["none"] = sub({"op": "arm.guarded_move", "arm": "left"}).result
        res["slide_dz"] = sub({"op": "arm.slide", "arm": "left", "delta": [0.02, 0.0, 0.02]}).result
        res["empty"] = sub({"op": "arm.move_path", "arm": "left", "waypoints": []}).result
        res["contact"] = sub({"op": "arm.guarded_move", "arm": "left", "delta": [0.0, 0.0, -0.02]}).result
        rig.svc.planner = None
        res["noplanner"] = sub({"op": "arm.move_path", "arm": "left", "waypoints": [[0.2, 0.2, 0.5]]}).result
        rig.svc.planner = planner

    rig.at(1.0, go)
    rig.run(1.2)
    for k in ("two", "none", "slide_dz", "empty"):
        assert res[k].code == Code.INVALID_ARGS and not res[k].moved, (k, res[k])
    # until 'contact' (the default) needs r1hands' pad-contact publisher (none in this rig), as arm.lower
    c = res["contact"]
    assert c.code == Code.UNSUPPORTED and c.data["state"] == "rejected" and not c.moved, c
    assert res["noplanner"].code == Code.UNSUPPORTED


def test_move_path_through_three_waypoints(w: world.World) -> None:
    rig = _rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        rig.svc.submit({"op": "world.declare", "entity": TABLE})
        p = _pinch(rig)
        wps = [_off(p, [0.03, 0.0, 0.02]), _off(p, [0.05, 0.0, 0.03]), _off(p, [0.05, -0.02, 0.03])]
        res["wps"] = wps
        res["path"] = rig.svc.submit({"op": "arm.move_path", "arm": "left", "waypoints": wps, "wait": True})

    rig.at(1.0, go)
    rig.run(9.0)
    out = res["path"].result
    assert out is not None and out.ok and out.data["state"] == "succeeded", out
    d = out.data
    assert d["waypoints"] == 3 and d["waypoints_done"] == 3, d
    assert [lg.split(":")[0] for lg in d["legs"]][-1] == "w2" and {lg.split(":")[0] for lg in d["legs"]} == {
        "w0",
        "w1",
        "w2",
    }
    assert d["legs_done"] == len(d["legs"])
    assert d["error_mm"] < ERROR_MM, d  # the final grasp centre vs the planner's prediction for the last waypoint
    assert len(d["predicted"]["waypoints_grasp_centre_m"]) == 3
    assert int(rig.ctl.core.status()["upper_rejected"]) == 0


def test_move_path_is_checked_whole_before_anything_moves(w: world.World) -> None:
    """AP-3: an unreachable LAST waypoint refuses the whole path - nothing moves for the reachable first two."""
    rig = _rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        p = _pinch(rig)
        wps = [_off(p, [0.03, 0.0, 0.02]), _off(p, [0.05, 0.0, 0.03]), [1.5, 0.2, 0.5]]
        res["path"] = rig.svc.submit({"op": "arm.move_path", "arm": "left", "waypoints": wps, "wait": True})

    rig.at(1.0, go)
    tr = rig.run(2.5)
    out = res["path"].result
    assert out is not None and not out.ok and out.code == Code.UNREACHABLE and not out.moved, out
    assert out.details["waypoint"] == 2 and out.details["waypoints"] == 3, out.details
    assert out.data["state"] == "rejected"
    after = tr[tr[:, COL["t"]] > 1.0][:, QCOLS]
    assert np.ptp(after, axis=0).max() < 1e-9  # the commanded arm never moved


def test_guarded_move_in_free_air_reaches_its_end(w: world.World) -> None:
    rig = _rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        res["c0"] = _centre(rig)
        res["g"] = rig.svc.submit({"op": "arm.guarded_move", "arm": "left", "delta": [0.03, 0.0, 0.02],
                                   "until": ["blocked"], "wait": True})  # fmt: skip

    rig.at(1.0, go)
    rig.run(5.0)
    out = res["g"].result
    assert out is not None and out.ok and out.data["state"] == "succeeded", out
    assert not out.data["stopped_early"] and out.data["error_mm"] < ERROR_MM, out.data


def test_slide_along_the_declared_support_under_the_hand(w: world.World) -> None:
    """A surface declared just under the lowest fingertip: the slide runs along it with the surface in the hand's
    allowed touches on every leg; with no support under the hand the slide is refused (nothing moves)."""
    rig = _rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        res["none"] = rig.svc.submit({"op": "arm.slide", "arm": "left", "delta": [0.02, 0.0, 0.0]}).result
        pts = hand_points(rig.port, None)
        assert pts is not None
        tip = pts[0, 1:4][int(np.argmin(pts[0, 1:4, 2]))]
        top = float(tip[2]) - 0.004  # the support's top 4 mm under the lowest fingertip
        res["top"] = top
        sup = {"kind": "surface", "id": "bench", "pose": {"pos": [float(tip[0]), float(tip[1]), top - 0.01]},
               "size": [0.3, 0.3, 0.02]}  # fmt: skip
        rig.svc.submit({"op": "world.declare", "entity": sup})
        res["other"] = rig.svc.submit({"op": "arm.slide", "arm": "left", "delta": [0.02, 0.0, 0.0],
                                       "support": "nope"}).result  # fmt: skip
        res["slide"] = rig.svc.submit({"op": "arm.slide", "arm": "left", "delta": [0.025, 0.0, 0.0], "wait": True})

    rig.at(1.0, go)
    rig.run(5.0)
    assert res["none"].code == Code.INVALID_ARGS and not res["none"].moved, res["none"]
    assert res["other"].code == Code.UNKNOWN_FRAME
    out = res["slide"].result
    assert out is not None and out.ok and out.data["state"] == "succeeded", out
    assert out.data["error_mm"] < ERROR_MM, out.data


class _ThreadPlanner:
    """r1d's planner as a worker that answers later (a Future per goal, like PlannerWorker; a thread, not a process,
    to stay cheap): move_path's chain submits waypoint k+1 only when waypoint k's plan is back."""

    def __init__(self) -> None:
        import concurrent.futures as cf

        self.pool = cf.ThreadPoolExecutor(max_workers=1)
        self.submitted = 0

    def submit(self, snap: Any, goal: Any, q26: Any, **kw: Any) -> Any:
        self.submitted += 1
        return self.pool.submit(planner.plan, snap, goal, q26, **kw)


def test_move_path_with_a_worker_planner_chains_the_futures(w: world.World) -> None:
    rig = Rig(w)
    tp = _ThreadPlanner()
    rig.svc.planner = tp
    res: dict[str, Any] = {}

    def go() -> None:
        p = _pinch(rig)
        wps = [_off(p, [0.03, 0.0, 0.02]), _off(p, [0.05, 0.0, 0.03])]
        res["path"] = rig.svc.submit({"op": "arm.move_path", "arm": "left", "waypoints": wps, "wait": True})

    rig.at(1.0, go)
    rig.run(7.0)
    tp.pool.shutdown()
    out = res["path"].result
    assert out is not None and out.ok and out.data["state"] == "succeeded", out
    assert tp.submitted == 2 and out.data["planned_on"] == "worker" and out.data["waypoints_done"] == 2, out.data
    assert out.data["error_mm"] < ERROR_MM
