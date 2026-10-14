"""r1d (python/r1mc3/service) against the REAL v3 core in lockstep (r1rt_py.Core through r1sim.v3_core; real profile +
the empty floor, writer 500 Hz, r1d's tick 100 Hz). DESIGN.md §3.3 (executor sources, TTL, stop epoch), §3.5 (takeover),
§5 (L0 / L1, codes).

1. L0 `joint.command` streamed by a client at 20 Hz: the core's commanded arm target follows the setpoint; when the
   client stops, the record TTL expires and the group decelerates to HOLD (the executor's STREAM -> HOLD).
2. L1 `stream.joint_jog`: r1d integrates the rate at its tick; after the request's TTL nothing more is sent and the
   group HOLDs where the jog ended.
3. A stop through r1d: the group HOLDs, a record stamped before the stop is refused by the core, a new one accepted.
4. Mode checks: legs in BALANCE -> WRONG_MODE; arms in MONITOR -> WRONG_MODE; damp from a non-operator -> NOT_PERMITTED.
5. The takeover sequence driven by r1d (the core's gates -> the emulated built-in's ReleaseMode -> RELEASED) from the
   S1 stand fixture into BALANCE; a ReleaseMode that fails -> ABORT -> MONITOR, the built-in keeps holding.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from r1mc3.service.codes import Code
from r1mc3.service.port import LockstepPort
from r1mc3.service.service import Service
from r1sim import cpp_policy, world
from r1sim import joints as J
from r1sim.engine import LowCmd, LowState, Simulation
from r1sim.model import centred_stand_pose
from r1sim.v3_core import (
    GROUP_ARM_LEFT,
    GROUP_JOINTS,
    MODE_BALANCE,
    MODE_MONITOR,
    MODE_TAKEOVER_BLEND,
    SOURCE_HOLD,
    SOURCE_STREAM,
    EmulatedBuiltin,
    TakeoverPlan,
    V3CoreController,
)

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")

COL = {"t": 0, "mode": 1, "stop": 2, "src": 3, "q": 12}  # V3CoreController.upper_trace
LEFT = list(GROUP_JOINTS[GROUP_ARM_LEFT])


class Rig:
    """The sim + the core + r1d ticking at 100 Hz between the core's writer ticks (r1d's clock = simulated time)."""

    def __init__(self, w: world.World, *, takeover: bool = False, release_ok: bool = True) -> None:
        self.sim = Simulation(w.model, w.index)
        self.t_ns = 0
        self.builtin: EmulatedBuiltin | None = None
        if takeover:
            self.sim.reset(w.initial_qpos(centred_stand_pose(w.model)))
            self.builtin = EmulatedBuiltin(w.model, release_ok=release_ok)
            plan = TakeoverPlan(self.builtin, t_request_s=1e9)  # the controller only feeds the built-in's command
            self.ctl = V3CoreController(w.model, takeover=plan, trace_upper=True, trace_cmd=True)
            self.sim.add_controller(self.builtin, self.ctl.period_s, "builtin")
        else:
            self.sim.reset(w.initial_qpos(J.POSE_STAND.copy()))
            self.ctl = V3CoreController(w.model, trace_upper=True)
        self.port = LockstepPort(
            self.ctl.core, self.ctl.mod, lambda: self.t_ns, targets_fn=lambda: self.ctl.last_q.copy()
        )
        self.svc = Service(self.port, releaser=_SyncReleaser(self.builtin) if self.builtin else None)
        self.script: list[tuple[float, Callable[[], object]]] = []
        self.sim.add_controller(self._r1d, 0.01, "r1d")
        self.sim.add_controller(self.ctl, self.ctl.period_s, "v3")

    def _r1d(self, st: LowState) -> LowCmd | None:
        self.t_ns = round(float(st.t) * 1e9)
        while self.script and self.script[0][0] <= st.t + 1e-9:
            self.script.pop(0)[1]()
        self.svc.tick(self.t_ns)
        if not self.ctl.entered and int(self.ctl.core.mode) == MODE_TAKEOVER_BLEND:  # the policy thread runs from here
            self.ctl.entered = True
            self.ctl._k_entry = self.ctl.calls
        return None

    def at(self, t: float, fn: Callable[[], object]) -> None:
        self.script.append((t, fn))
        self.script.sort(key=lambda e: e[0])

    def run(self, seconds: float) -> np.ndarray:
        self.sim.run_for(seconds)
        return np.array(self.ctl.upper_trace)


class _SyncReleaser:
    synchronous = True

    def __init__(self, builtin: EmulatedBuiltin) -> None:
        self.builtin = builtin

    def release(self, timeout_s: float) -> tuple[bool, dict[str, Any]]:
        return self.builtin.release(), {"emulated": True}


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _rows(tr: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return tr[(tr[:, COL["t"]] >= t0) & (tr[:, COL["t"]] < t1)]


def test_joint_command_stream_follows_then_ttl_holds(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def start() -> None:
        q0 = rig.ctl.last_q[LEFT].copy()
        res["q0"] = q0
        res["goal"] = q0 + np.array([0.0, 0.25, 0.0, 0.25, 0.0])

    rig.at(1.0, start)
    for k in range(int((2.5 - 1.2) / 0.05)):
        t = 1.2 + 0.05 * k
        rig.at(
            t,
            lambda: res.setdefault("results", []).append(
                rig.svc.submit(
                    {"op": "joint.command", "group": "arm_left", "q": res["goal"].tolist(), "ttl": 0.2}
                ).result
            ),
        )
    tr = rig.run(3.5)
    results = res["results"]
    assert all(r is not None and r.ok and r.moved for r in results), results[0]
    qcols = [COL["q"] + j - 14 for j in LEFT]
    src_left = COL["src"] + 0
    held = _rows(tr, 2.2, 2.45)
    assert np.all(held[:, src_left] == SOURCE_STREAM)
    assert np.max(np.abs(held[:, qcols] - res["goal"])) < 1e-9  # the commanded target IS the setpoint
    # the last record was issued at 2.45 s with TTL 0.2: by 2.65 + stop_time (0.15) + the hold ramp the group HOLDs
    after = _rows(tr, 3.0, 3.5)
    assert np.all(after[:, src_left] == SOURCE_HOLD)
    assert np.max(np.abs(after[:, qcols] - res["goal"])) < 0.02  # it stopped near the setpoint (no drift away)
    assert int(rig.ctl.core.status()["upper_rejected"]) == 0


def test_joint_jog_integrates_then_stops(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}
    rate = np.array([0.0, 0.2, 0.0, 0.0, 0.0])  # shoulder roll outwards at 0.2 rad/s
    rig.at(1.0, lambda: res.__setitem__("q0", rig.ctl.last_q[LEFT].copy()))
    for k in range(10):  # the client re-sends the jog every 0.1 s for 1 s (TTL 0.3 s each)
        rig.at(
            1.2 + 0.1 * k,
            lambda: res.setdefault("r", []).append(
                rig.svc.submit({"op": "stream.joint_jog", "group": "arm_left", "dq": rate.tolist(), "ttl": 0.3}).result
            ),
        )
    tr = rig.run(3.0)
    assert all(r is not None and r.ok for r in res["r"])
    qcols = [COL["q"] + j - 14 for j in LEFT]
    end = _rows(tr, 2.6, 3.0)
    moved = end[-1, qcols] - res["q0"]
    # 1.2 s -> 2.2 s (the last request's TTL) at 0.2 rad/s = 0.2 rad (+- one r1d tick, the stop's own travel)
    assert 0.18 < moved[1] < 0.24, moved
    assert np.max(np.abs(np.delete(moved, 1))) < 1e-6
    assert np.all(end[:, COL["src"]] == SOURCE_HOLD)
    assert "arm_left" not in rig.svc.status()["jogs"]


def test_stop_holds_and_refuses_records_issued_before_it(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def go() -> None:
        q0 = rig.ctl.last_q[LEFT].copy()
        res["goal"] = q0 + np.array([0.0, 0.3, 0.0, 0.0, 0.0])
        res["epoch0"] = rig.port.stop_epoch()
        res["cmd"] = rig.svc.submit({"op": "joint.command", "group": "arm_left", "q": res["goal"].tolist(), "ttl": 0.3})

    def stop() -> None:
        res["stop"] = rig.svc.submit({"op": "stop"}).result

    def stale() -> None:  # a slow client's record stamped before the stop
        res["rej0"] = int(rig.ctl.core.status()["upper_rejected"])
        rig.port.push_upper(
            {
                "cmd_id": 999,
                "group": GROUP_ARM_LEFT,
                "source": SOURCE_STREAM,
                "n_joints": 5,
                "t_issue_ns": rig.t_ns,
                "ttl_ns": 200_000_000,
                "q": res["goal"].tolist(),
                "dq": [0.0] * 5,
                "stop_epoch": res["epoch0"],
            }
        )

    def fresh() -> None:
        res["rej1"] = int(rig.ctl.core.status()["upper_rejected"])
        res["new"] = rig.svc.submit({"op": "joint.command", "group": "arm_left", "q": res["goal"].tolist(), "ttl": 0.2})

    rig.at(1.0, go)
    rig.at(1.05, stop)
    rig.at(1.5, stale)
    rig.at(1.6, fresh)
    tr = rig.run(1.9)
    assert res["cmd"].result.ok and res["stop"].ok and res["stop"].data["stop_epoch"] > res["epoch0"]
    assert res["rej1"] == res["rej0"] + 1  # the core refused the stale record
    assert int(rig.ctl.core.status()["upper_last_reject"]) != 0
    assert res["new"].result.ok
    assert int(rig.ctl.core.status()["upper_rejected"]) == res["rej1"]  # the new one was accepted
    stopped = _rows(tr, 1.3, 1.5)
    assert np.all(stopped[:, COL["src"]] == SOURCE_HOLD)
    assert np.all(_rows(tr, 1.62, 1.78)[:, COL["src"]] == SOURCE_STREAM)  # TTL 0.2 s from 1.6 s
    assert int(rig.ctl.core.mode) == MODE_BALANCE  # a stop keeps balancing


def test_mode_checks(w: world.World) -> None:
    rig = Rig(w)
    res: dict[str, Any] = {}

    def probe() -> None:
        res["legs"] = rig.svc.submit({"op": "joint.command", "group": "leg_left", "q": [0.0] * 6}).result
        res["damp"] = rig.svc.submit({"op": "damp"}).result
        res["bad"] = rig.svc.submit({"op": "joint.command", "group": "arm_left", "q": [0.0] * 4}).result
        res["nope"] = rig.svc.submit({"op": "fly"}).result
        res["status"] = rig.svc.submit({"op": "status"}).result

    rig.at(1.0, probe)
    rig.run(1.1)
    assert res["legs"].code == Code.WRONG_MODE and res["legs"].details["mode"] == "balance"
    assert res["damp"].code == Code.NOT_PERMITTED and not res["damp"].moved
    assert res["bad"].code == Code.INVALID_ARGS
    assert res["nope"].code == Code.UNKNOWN_COMMAND
    st = res["status"].data
    assert st["core"]["mode"] == "balance" and st["token"]["owner"] == "r1rt"
    assert int(rig.ctl.core.mode) == MODE_BALANCE  # nothing above changed the mode


@pytest.mark.slow
@pytest.mark.parametrize("release_ok", [True, False], ids=["released", "release_fails"])
def test_takeover_sequence_from_the_builtin(w: world.World, release_ok: bool) -> None:
    rig = Rig(w, takeover=True, release_ok=release_ok)
    res: dict[str, Any] = {}

    def arms_in_monitor() -> None:
        res["arm"] = rig.svc.submit({"op": "joint.command", "group": "arm_left", "q": [0.0] * 5}).result

    rig.at(0.5, arms_in_monitor)
    rig.at(1.0, lambda: res.__setitem__("t", rig.svc.submit({"op": "takeover"})))
    rig.run(3.0)
    assert res["arm"].code == Code.WRONG_MODE and res["arm"].details["mode"] == "monitor"
    r = res["t"].result
    assert r is not None
    assert rig.builtin is not None and rig.builtin.release_calls == 1
    if hasattr(rig.port.view, "request_mode"):  # layout 2: the requests went through the core's mailbox
        assert rig.port._direct == {}
    if release_ok:
        assert r.ok and r.moved, r
        assert [e["stage"] for e in r.data["log"]] == ["gates", "release", "decide"]
        assert int(rig.ctl.core.mode) == MODE_BALANCE  # BLEND (1 s) -> BALANCE by 3 s
    else:
        assert r.code == Code.RELEASE_FAILED and r.details["aborted"]
        assert int(rig.ctl.core.mode) == MODE_MONITOR  # back to MONITOR: nothing of ours published
        assert rig.builtin.released is False
