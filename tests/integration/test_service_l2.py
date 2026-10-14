"""r1d's L2 goals (DESIGN.md §5.1-§5.2, session 9) against the REAL v3 core in lockstep (r1rt_py.Core through
r1sim.v3_core, the real profile, writer 500 Hz, r1d's tick 100 Hz; the Rig of test_service_lockstep.py).

The goal lifecycle (accepted -> active -> succeeded | aborted | canceled | preempted | timed_out), the resource rule
(RESOURCE_BUSY / preempt), the measured §5.2 keys, the stop / cancel / timeout ends - with `arm.move_joint`, the first
L2 op (a minimum-jerk TRAJECTORY with the deadman keepalive renewed at every r1d tick). On the base (6449aee) every
op here answered UNKNOWN_COMMAND.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.service.codes import Code
from r1mc3.service.port import LockstepPort
from r1mc3.service.schemas import GOAL_RESULT_KEYS
from r1sim import cpp_policy, world
from r1sim.v3_core import GROUP_ARM_LEFT, GROUP_JOINTS, SOURCE_HOLD, SOURCE_TRAJECTORY

from tests.integration.test_service_lockstep import COL, Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
LEFT = list(GROUP_JOINTS[GROUP_ARM_LEFT])
QCOLS = [COL["q"] + j - 14 for j in LEFT]
DELTA = np.array([0.0, 0.25, 0.0, -0.30, 0.0])  # shoulder roll out, elbow flexed more (a SMALLER elbow = more flexed)


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _goal(rig: Rig, res: dict[str, Any], key: str, req: dict[str, Any]) -> None:
    res[key] = rig.svc.submit(req)


def _q0_goal(rig: Rig, res: dict[str, Any]) -> list[float]:
    res["q0"] = rig.ctl.last_q[LEFT].copy()
    res["goal"] = res["q0"] + DELTA
    return [float(x) for x in res["goal"]]


def test_move_joint_succeeds_with_the_measured_result_keys(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        q = _q0_goal(rig, res)
        res["acc"] = rig.svc.submit({"op": "arm.move_joint", "id": 7, "arm": "left", "q": q, "speed": 0.5}).result
        gid = res["acc"].data["goal_id"]
        res["wait"] = rig.svc.submit({"op": "goal.wait", "id": 8, "goal_id": gid, "timeout_s": 5.0})
        res["status_mid"] = None

    def mid() -> None:
        res["status_mid"] = rig.svc.submit({"op": "goal.status", "goal_id": res["acc"].data["goal_id"]}).result
        res["busy"] = rig.svc.submit({"op": "joint.command", "group": "arm_left", "q": res["q0"].tolist()}).result

    rig.at(1.0, go)
    rig.at(1.3, mid)
    tr = rig.run(3.0)
    acc = res["acc"]
    assert acc.ok and acc.moved and acc.data["state"] == "accepted" and acc.id == 7, acc
    assert res["status_mid"].data["state"] == "active"
    assert res["busy"].code == Code.RESOURCE_BUSY and not res["busy"].moved  # L0 on a limb a goal holds
    out = res["wait"].result
    assert out is not None and out.ok and out.op == "goal.wait" and out.id == 8, out
    d = out.data
    assert d["state"] == "succeeded" and d["stopped_by"] is None
    assert set(GOAL_RESULT_KEYS) <= set(d), set(GOAL_RESULT_KEYS) - set(d)
    # the planned duration: 1.875 x 0.30 rad / (0.5 x 1 rad/s) = 1.125 s
    assert 1.1 <= d["duration_s"] <= 1.25, d["duration_s"]
    end = tr[(tr[:, COL["t"]] > 2.5)]
    assert np.max(np.abs(end[:, QCOLS] - res["goal"])) < 1e-6  # the core's commanded target IS the goal
    assert np.all(end[:, COL["src"]] == SOURCE_HOLD)
    mid_rows = tr[(tr[:, COL["t"]] > 1.2) & (tr[:, COL["t"]] < 1.9)]
    assert np.all(mid_rows[:, COL["src"]] == SOURCE_TRAJECTORY)
    # measured: the hand moved (FK of the measured joints), the error against the target pose's FK is small
    assert np.linalg.norm(np.subtract(d["hand_now"], d["hand_start"])) > 0.03
    assert d["error_mm"] < 20.0 and d["achieved"] is True, d
    assert len(d["fingertips_m"]) == 3 and len(d["grasp_centre_m"]) == 3
    assert int(rig.ctl.core.status()["upper_rejected"]) == 0


def test_wait_true_replies_with_the_final_result(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: _goal(rig, res, "t", {"op": "arm.move_joint", "arm": "left", "q": _q0_goal(rig, res),
                                              "speed": 0.8, "wait": True}))  # fmt: skip
    rig.run(1.2)
    assert res["t"].result is None  # still moving
    rig.run(1.2)
    out: Any = res["t"].result
    assert out is not None and out.ok and out.op == "arm.move_joint" and out.data["state"] == "succeeded", out


def test_busy_then_preempt(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def first() -> None:
        res["a"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": _q0_goal(rig, res), "speed": 0.3})

    def second() -> None:
        back = res["q0"].tolist()
        res["busy"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": back}).result
        res["b"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": back, "preempt": True, "wait": True})

    rig.at(1.0, first)
    rig.at(1.5, second)
    tr = rig.run(4.5)
    assert res["busy"].code == Code.RESOURCE_BUSY and not res["busy"].moved
    assert res["busy"].details["held_by"][0]["goal_id"] == res["a"].result.data["goal_id"]
    a_id = res["a"].result.data["goal_id"]
    st: Any = rig.svc.submit({"op": "goal.status", "goal_id": a_id}).result
    assert st.data["state"] == "preempted" and st.data["result"]["code"] == "PREEMPTED"
    b = res["b"].result
    assert b is not None and b.ok and b.data["state"] == "succeeded", b
    end = tr[tr[:, COL["t"]] > 4.3]
    assert np.max(np.abs(end[:, QCOLS] - res["q0"])) < 1e-6  # the second goal ended where it was sent
    assert int(rig.ctl.core.status()["upper_rejected"]) == 0  # its first knot met the braked target (deferred start)


def test_a_stop_aborts_the_goal_and_the_arm_holds(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: _goal(rig, res, "g", {"op": "arm.move_joint", "arm": "left", "q": _q0_goal(rig, res),
                                              "speed": 0.3, "wait": True}))  # fmt: skip
    rig.at(1.4, lambda: _goal(rig, res, "stop", {"op": "stop"}))
    tr = rig.run(2.5)
    out = res["g"].result
    assert res["stop"].result.ok
    assert out is not None and not out.ok and out.code == Code.GUARD_STOP and out.moved, out
    assert out.data["state"] == "aborted" and out.data["stopped_by"]["code"] == "OPERATOR_STOP"
    held = tr[tr[:, COL["t"]] > 1.8]
    assert np.all(held[:, COL["src"]] == SOURCE_HOLD)
    assert np.ptp(held[:, QCOLS], axis=0).max() < 1e-9  # holding, not moving
    assert np.max(np.abs(held[-1, QCOLS] - res["goal"])) > 0.05  # it stopped on the way


def test_cancel_and_timeout_end_the_goal_holding(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        q = _q0_goal(rig, res)
        res["g"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": q, "speed": 0.3}).result

    def cancel() -> None:
        res["c"] = rig.svc.submit({"op": "goal.cancel", "goal_id": res["g"].data["goal_id"]}).result
        res["c2"] = rig.svc.submit({"op": "goal.cancel", "goal_id": res["g"].data["goal_id"]}).result
        res["unknown"] = rig.svc.submit({"op": "goal.status", "goal_id": 999}).result

    def timed() -> None:  # on to the goal (~0.25 rad, ~2 s at speed 0.2) with a timeout shorter than the move
        res["t"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": res["goal"].tolist(), "speed": 0.2,
                                   "timeout_s": 0.4, "wait": True})  # fmt: skip

    rig.at(1.0, go)
    rig.at(1.4, cancel)
    rig.at(2.2, timed)
    tr = rig.run(3.6)
    assert res["c"].ok and res["c"].data["state"] == "canceled"
    assert res["c2"].ok and res["c2"].data["already_ended"] is True
    assert res["unknown"].code == Code.UNKNOWN_GOAL
    st: Any = rig.svc.submit({"op": "goal.status", "goal_id": res["g"].data["goal_id"]}).result
    assert st.data["result"]["code"] == "CANCELED" and st.data["result"]["moved"] is True
    seg = tr[(tr[:, COL["t"]] > 1.8) & (tr[:, COL["t"]] < 2.2)]
    assert np.ptp(seg[:, QCOLS], axis=0).max() < 1e-9  # canceled = holding
    t = res["t"].result
    assert t is not None and t.code == Code.TIMEOUT and t.data["state"] == "timed_out", t
    late = tr[tr[:, COL["t"]] > 3.2]
    assert np.all(late[:, COL["src"]] == SOURCE_HOLD) and np.ptp(late[:, QCOLS], axis=0).max() < 1e-9


def test_refusals_say_nothing_moved(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def probe() -> None:
        q = _q0_goal(rig, res)
        res["fast"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": (res["q0"] + 1.5).tolist(),
                                      "duration_s": 0.3}).result  # fmt: skip
        res["bad"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": q[:4]}).result
        res["nan"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": [float("nan")] * 5}).result
        res["stop_idle"] = rig.svc.submit({"op": "arm.stop", "arm": "right"}).result

    rig.at(1.0, probe)
    tr = rig.run(1.3)
    assert res["fast"].code == Code.LIMIT and not res["fast"].moved and res["fast"].details["peak_rad_s"] > 2.0
    assert res["bad"].code == Code.INVALID_ARGS and res["nan"].code == Code.INVALID_ARGS
    assert res["stop_idle"].ok and res["stop_idle"].data["held"] == ["arm_right"]
    assert np.ptp(tr[tr[:, COL["t"]] > 1.0][:, QCOLS], axis=0).max() < 1e-9  # the left arm never moved
    assert rig.svc.status()["goals"] == []


def test_status_reports_the_body_the_hands_and_the_policy(w: world.World) -> None:
    """Session 9 (the compat adapter reads them): body.rpy_deg / tilt_deg / lean_deg, arms.<side>.hand_point +
    fingertips_m (the built-in arm model's FK of the measured state), policy (the policy directory)."""
    rig = Rig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: res.__setitem__("s", rig.svc.submit({"op": "status"}).result))
    rig.at(1.0, lambda: res.__setitem__("low", rig.svc.submit({"op": "arm.lower", "arm": "left", "until": "depth",
                                                               "stop_z_m": -1.0}).result))  # fmt: skip
    rig.run(1.1)
    d = res["s"].data
    assert d["policy"].startswith("r1mc2_manip2_lean")
    assert len(d["body"]["rpy_deg"]) == 3 and abs(d["body"]["rpy_deg"][1]) < 5.0 and d["body"]["tilt_deg"] < 5.0
    assert d["body"]["lean_deg"][0] is None
    left = d["arms"]["left"]
    assert left["hand_point"][1] > 0.05 and d["arms"]["right"]["hand_point"][1] < -0.05  # left hand on the left (+y)
    assert set(left["fingertips_m"]) == {"thumb_tip", "middle_tip", "index_tip"}
    assert res["low"].code == Code.UNSUPPORTED  # no planner in this rig: refused before the stop height is used


class _NoEventsPort(LockstepPort):
    """r1rt's real segment today: no per-group executor events (ShmPort.upper_groups() is None) - r1d must judge a
    goal's end from the commanded target (the mirror's q_target on the robot, the last published command here)."""

    def upper_groups(self) -> dict[str, list[int]] | None:
        return None


def _no_events_rig(w: world.World) -> Rig:
    rig = Rig(w)
    rig.port = _NoEventsPort(rig.ctl.core, rig.ctl.mod, lambda: rig.t_ns, targets_fn=lambda: rig.ctl.last_q.copy())
    rig.svc.port = rig.port
    return rig


def test_without_executor_events_the_end_is_judged_from_the_target(w: world.World) -> None:
    """The robot path: a goal succeeds when the commanded target reached the plan's end; an operator stop (the core
    brakes the arm off the plan) aborts it (STOPPED) - no executor events needed."""
    rig = _no_events_rig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: _goal(rig, res, "a", {"op": "arm.move_joint", "arm": "left", "q": _q0_goal(rig, res),
                                              "speed": 0.8, "wait": True}))  # fmt: skip

    def second() -> None:
        res["b"] = rig.svc.submit({"op": "arm.move_joint", "arm": "left", "q": res["q0"].tolist(), "speed": 0.3,
                                   "wait": True})  # fmt: skip

    rig.at(2.5, second)
    rig.at(3.0, lambda: _goal(rig, res, "stop", {"op": "stop"}))
    rig.run(3.5)
    a = res["a"].result
    assert a is not None and a.ok and a.data["state"] == "succeeded", a
    assert 0.6 <= a.data["duration_s"] <= 1.1, a.data["duration_s"]  # 1.875 x 0.30 / 0.8 = 0.70 s (+ the end check)
    b = res["b"].result
    assert b is not None and not b.ok and b.data["state"] == "aborted", b
    assert b.data["stopped_by"]["code"] == "OPERATOR_STOP"  # r1d's own stop record ends it at once


def test_without_executor_events_a_core_stop_is_seen_as_stopped(w: world.World) -> None:
    """The core stops the path by itself (here: a stop request posted straight to the core, not through r1d - as a
    guard / deadman would): r1d sees the target leave its plan and aborts the goal (STOPPED)."""
    rig = _no_events_rig(w)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: _goal(rig, res, "a", {"op": "arm.move_joint", "arm": "left", "q": _q0_goal(rig, res),
                                              "speed": 0.3, "wait": True}))  # fmt: skip
    rig.at(1.5, lambda: rig.port.request_stop(0, 3, rig.t_ns))
    rig.run(2.2)
    a = res["a"].result
    assert a is not None and not a.ok and a.code == Code.GUARD_STOP and a.data["stopped_by"]["code"] == "STOPPED", a
    assert a.data["duration_s"] < 0.7  # seen within ~0.2 s of the stop
