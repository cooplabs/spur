"""The HW1 rehearsal in lockstep: the robot HANGING on its stand (the core agent's fixture: the pelvis welded 0.10 m
above its standing height, tests/integration/test_v3_core_onstand.py), the emulated built-in damping the legs (Unitree's
damping mode, the normal start on the stand), r1d's `on_stand` (the core's gates through the mailbox ->
ReleaseMode -> RELEASED -> BLEND into the stiff hold -> ON_STAND), then through r1d: a leg L0 setpoint, one STEP and
one CHIRP recording on the left knee (r1mc3.record) - the core's commanded knee target follows the plans exactly, the
measured knee follows it, every record is accepted, the manifests say done (DESIGN §3.5 ON_STAND, §14.1 step 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.record.plans import plan
from r1mc3.record.runner import Recorder
from r1mc3.service.port import LockstepPort
from r1mc3.service.service import Service
from r1sim import cpp_policy
from r1sim.engine import LowCmd, LowState, Simulation
from r1sim.v3_core import MODE_ON_STAND, EmulatedBuiltin, TakeoverPlan, V3CoreController

from tests.integration.test_service_lockstep import _SyncReleaser
from tests.integration.test_v3_core_onstand import _hanging_world

pytestmark = [
    pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp"),
    pytest.mark.slow,
]
KNEE = 3  # left_knee (SDK order); column 3 of leg_left


class HangRig:
    def __init__(self, out: Path) -> None:
        w, qpos = _hanging_world()
        self.sim = Simulation(w.model, w.index)
        self.sim.reset(qpos)
        self.builtin = EmulatedBuiltin(w.model, leg_kp=0.0)  # damping the hanging legs
        plan_ = TakeoverPlan(self.builtin, t_request_s=1e9)  # the controller feeds the built-in's command only
        self.ctl = V3CoreController(w.model, takeover=plan_, trace_cmd=True)
        self.t_ns = 0
        self.port = LockstepPort(self.ctl.core, self.ctl.mod, lambda: self.t_ns, targets_fn=self._targets)
        self.svc = Service(
            self.port, releaser=_SyncReleaser(self.builtin), recorder_factory=lambda s: Recorder(s, out_dir=out)
        )
        self.script: list[tuple[float, Any]] = []
        self.q_meas: list[tuple[float, float]] = []
        self.sim.add_controller(self.builtin, self.ctl.period_s, "builtin")
        self.sim.add_controller(self._r1d, 0.01, "r1d")
        self.sim.add_controller(self.ctl, self.ctl.period_s, "v3")

    def _targets(self) -> np.ndarray | None:
        return self.ctl.last_q.copy() if self.ctl.last_publish else None

    def _r1d(self, st: LowState) -> LowCmd | None:
        self.t_ns = round(float(st.t) * 1e9)
        self.q_meas.append((float(st.t), float(st.q[KNEE])))
        while self.script and self.script[0][0] <= st.t + 1e-9:
            self.script.pop(0)[1]()
        self.svc.tick(self.t_ns)
        return None

    def at(self, t: float, fn: Any) -> None:
        self.script.append((t, fn))
        self.script.sort(key=lambda e: e[0])


def test_on_stand_then_a_step_and_a_chirp_on_the_knee(tmp_path: Path) -> None:
    rig = HangRig(tmp_path)
    res: dict[str, Any] = {}
    step = {"joint": "left_knee", "amplitude_rad": 0.05, "hold_s": 0.5, "repeats": 1}
    chirp = {"joint": "left_knee", "amplitude_rad": 0.03, "f0_hz": 0.5, "f1_hz": 3.0, "duration_s": 4.0}

    def start_step() -> None:
        res["base"] = rig.ctl.last_q[:6].copy()
        res["t_step"] = rig.t_ns * 1e-9
        res["step"] = rig.svc.submit({"op": "record.run", "kind": "step", "params": step, "dry_run": False}).result

    def start_chirp() -> None:
        res["t_chirp"] = rig.t_ns * 1e-9
        res["chirp"] = rig.svc.submit({"op": "record.run", "kind": "chirp", "params": chirp, "dry_run": False}).result

    # the dangling legs settle under the built-in's damping first (the motion gate |dq| <= 0.5), as the core's test
    rig.at(1.5, lambda: res.__setitem__("on", rig.svc.submit({"op": "on_stand"})))

    def leg_l0() -> None:  # one L0 setpoint on the left leg (ON_STAND: legs allowed), at the current target
        res["leg"] = rig.svc.submit(
            {"op": "joint.command", "group": "leg_left", "q": rig.ctl.last_q[:6].tolist(), "ttl": 0.2}
        ).result
        res["arm_in_stand"] = rig.svc.submit({"op": "joint.hold", "group": "arm_left"}).result

    rig.at(2.8, leg_l0)
    rig.at(3.0, start_step)
    rig.at(6.0, start_chirp)
    rig.sim.run_for(10.5)
    on = res["on"].result
    assert on is not None and on.ok, on
    assert int(rig.ctl.core.mode) == MODE_ON_STAND
    assert res["step"].ok and res["chirp"].ok, (res["step"], res["chirp"])
    assert res["leg"].ok and res["leg"].moved and res["arm_in_stand"].ok
    tr = rig.ctl.cmd_trace
    t = np.array([e[0] for e in tr])
    q = np.array([e[4] for e in tr])
    kp = np.array([e[5] for e in tr])
    assert np.min(kp[t > 2.5][:, :12]) > 0.0  # never limp on the stand
    base = res["base"][KNEE]
    sp = plan("step", step, np.zeros(26))
    for t0 in (0.25, 0.75, 1.25, 1.75, 2.25):  # mid-hold of each step segment
        sel = (t >= res["t_step"] + t0) & (t < res["t_step"] + t0 + 0.1)
        want = base + sp.at(t0)[0][KNEE]
        assert np.max(np.abs(q[sel, KNEE] - want)) < 1e-9, t0
    cp = plan("chirp", chirp, np.zeros(26))
    sel = (t >= res["t_chirp"] + 0.6) & (t < res["t_chirp"] + 3.4)
    # the commanded knee = the chirp sampled at r1d's 100 Hz tick, once the executor's per-tick speed cap has let the
    # target reach each new setpoint (the second half of every 10 ms window; a STREAM setpoint is not interpolated)
    x = (t[sel] - res["t_chirp"]) * 100
    late = (x - np.floor(x + 1e-6)) >= 0.5
    k = np.floor(x[late] + 1e-6)
    want = np.array([base + cp.at(kk / 100)[0][KNEE] for kk in k])
    assert np.max(np.abs(q[sel, KNEE][late] - want)) < 1e-9
    meas = np.array(rig.q_meas)
    msel = (meas[:, 0] >= res["t_chirp"] + 0.6) & (meas[:, 0] < res["t_chirp"] + 3.4)
    track = np.max(np.abs(meas[msel, 1] - np.interp(meas[msel, 0], t, q[:, KNEE])))
    assert track < 0.03, track  # the hanging knee follows a 3 Hz, 0.03 rad sweep at the hold gains
    docs = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("*.json"))]
    assert [d["status"] for d in docs] == ["done", "done"] and all(d["refused"] == 0 for d in docs)
    assert int(rig.ctl.core.status()["upper_rejected"]) == 0
