"""arm.move_path with `blend_m` (session 11) through r1d + the planner (inline) + the REAL v3 core in lockstep (the Rig
of test_service_lockstep.py): ONE trajectory through the waypoints, the hand moving past the interior ones.

On the base (1e55bdd) `blend_m` was refused (INVALID_ARGS: an unknown field) and every path stopped at each waypoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import r1mc3.plan as planner
from r1mc3.service.codes import Code
from r1mc3.service.goals import hand_points
from r1sim import cpp_policy, world

from tests.integration.test_service_lockstep import Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
TABLE = {"kind": "surface", "id": "table", "pose": {"pos": [0.30, 0.0, 0.38]}, "size": [0.4, 1.0, 0.04]}
ERROR_MM = 10.0  # the measured grasp centre vs the planner's prediction (test_service_l2_paths' bound)


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _pinch(rig: Rig) -> np.ndarray:
    pts = hand_points(rig.port, None)
    assert pts is not None
    return np.asarray(0.5 * (pts[0, 1] + pts[0, 3]), dtype=np.float64)


def _run(
    w: world.World, blend_m: float | None, plan_with: Any = planner
) -> tuple[Any, list[tuple[float, np.ndarray]], list[list[float]]]:
    rig = Rig(w)
    rig.svc.planner = plan_with
    res: dict[str, Any] = {}
    trace: list[tuple[float, np.ndarray]] = []

    def probe(st: Any) -> None:  # the measured pinch point at 50 Hz (the hand's real motion in the sim)
        trace.append((float(st.t), _pinch(rig)))

    def go() -> None:
        rig.svc.submit({"op": "world.declare", "entity": TABLE})
        p = _pinch(rig)
        wps = [[float(x) for x in p + np.asarray(d)] for d in ([0.06, 0, 0.0], [0.06, 0, 0.06], [0.0, 0, 0.06])]
        res["wps"] = wps
        req: dict[str, Any] = {"op": "arm.move_path", "arm": "left", "waypoints": wps, "linear": True, "wait": True}
        if blend_m is not None:
            req["blend_m"] = blend_m
        res["path"] = rig.svc.submit(req)

    rig.sim.add_controller(probe, 0.02, "probe")
    rig.at(1.0, go)
    rig.run(12.0)
    return res["path"].result, trace, res["wps"]


def _stopped_near(trace: list[tuple[float, np.ndarray]], wp: list[float], tol: float) -> tuple[float, float]:
    """(the closest the hand came to wp, the hand's lowest speed within tol of it) from the 50 Hz trace."""
    t = np.array([a for a, _ in trace])
    p = np.array([b for _, b in trace])
    v = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(t)]
    d = np.linalg.norm(p - np.asarray(wp), axis=1)
    near = d <= tol
    return float(d.min()), float(v[near].min()) if near.any() else float("nan")


def test_move_path_blended_does_not_stop_at_the_waypoints(w: world.World) -> None:
    stops, tr0, wps = _run(w, None)
    blended, tr1, wps1 = _run(w, 0.02)
    assert stops is not None and stops.ok and stops.data["state"] == "succeeded", stops
    assert blended is not None and blended.ok and blended.data["state"] == "succeeded", blended
    b = blended.data["blend"]
    assert b["applied"] and blended.data["legs"] == ["w0-w2:blend"], b
    assert blended.data["waypoints_done"] == 3 and blended.data["error_mm"] < ERROR_MM, blended.data
    assert "blend" not in stops.data and len(stops.data["legs"]) >= 3
    # the hand comes to REST at each interior waypoint without the blend, keeps moving past it with it
    for k in (0, 1):
        d0, v0 = _stopped_near(tr0, wps[k], 0.005)
        d1, v1 = _stopped_near(tr1, wps1[k], 0.025)
        assert d0 < 0.005 and v0 < 0.005, (k, d0, v0)  # m, m/s: stopped there
        assert d1 <= 0.02 + 0.004 and v1 > 0.01, (k, d1, v1)  # within the tolerance (+ the arm's tracking), moving
    assert b["duration_s"] < b["unblended_s"], b


def test_move_path_blended_through_the_planner_worker(w: world.World) -> None:
    """r1d's production planner (the worker process): the legs and then the blend run there (PlannerWorker.call)."""
    with planner.PlannerWorker() as wk:
        out, _, _ = _run(w, 0.02, wk)
    assert out is not None and out.ok and out.data["state"] == "succeeded", out
    assert out.data["planned_on"] == "worker" and out.data["blend"]["applied"], out.data
    assert out.data["legs"] == ["w0-w2:blend"] and out.data["error_mm"] < ERROR_MM


def test_a_blended_path_stopped_midway_says_how_far_it_got(w: world.World) -> None:
    """arm.stop inside the one trajectory: the arm HOLDs; waypoints_done = the waypoints its curve had passed."""
    rig = Rig(w)
    rig.svc.planner = planner
    res: dict[str, Any] = {}

    def go() -> None:
        p = _pinch(rig)
        wps = [[float(x) for x in p + np.asarray(d)] for d in ([0.06, 0, 0.0], [0.06, 0, 0.06], [0.0, 0, 0.06])]
        res["path"] = rig.svc.submit({"op": "arm.move_path", "arm": "left", "waypoints": wps, "linear": True,
                                      "blend_m": 0.02, "wait": True})  # fmt: skip

    def stop() -> None:
        res["stop"] = rig.svc.submit({"op": "arm.stop", "arm": "left"}).result

    rig.at(1.0, go)
    rig.at(4.0, stop)  # the blended path takes ~4.4 s from ~1 s: stopped past the first corner, before the end
    rig.run(7.0)
    out = res["path"].result
    assert out is not None and out.data["state"] != "succeeded" and out.data["blend"]["applied"], out
    passed = out.data["blend"]["pass_t_s"]
    assert len(passed) == 2 and 0 < out.data["waypoints_done"] < 3, (out.data["waypoints_done"], passed)


def test_move_path_blend_argument_bounds(w: world.World) -> None:
    rig = Rig(w)
    rig.svc.planner = planner
    res: dict[str, Any] = {}

    def go() -> None:
        wps = [[0.2, 0.2, 0.5], [0.22, 0.2, 0.5]]
        res["big"] = rig.svc.submit({"op": "arm.move_path", "arm": "left", "waypoints": wps, "blend_m": 0.06}).result
        res["neg"] = rig.svc.submit({"op": "arm.move_path", "arm": "left", "waypoints": wps, "blend_m": -0.01}).result

    rig.at(0.5, go)
    rig.run(0.6)
    assert res["big"].code == Code.INVALID_ARGS and not res["big"].moved
    assert res["neg"].code == Code.INVALID_ARGS and not res["neg"].moved
