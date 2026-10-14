"""#29 hazard (ii) (session 9): a resume must not blend from a sample whose board tick is provably OLD.

After a whole-machine stop the DDS receive path first drains the lowstate samples queued in the socket during the stop:
each is stamped at its RECEIVE time (fresh) but carries its board tick and pose from the stop's start. r1rt / r1guard
wait for a sample whose tick has caught up with the tick extrapolated from before the stop (r1rt::ResumeClock). Session
8 gave up after R1RT_RESUME_STATE_WAIT_NS (50 ms) and blended from the newest sample "fresh by its stamp" - during a
drain longer than 50 ms that is a pose from the stop's start, i.e. the old target: the snap #29 exists to prevent, with
no stale packet at all. The robot's cyclonedds.xml asks for a 4 MB socket buffer (~1700-3400 LowState samples): a drain
past 50 ms is plausible there (Mac, loaded: ~60-100 us per sample).

Lockstep (test_v3_resume.py's hanging robot + board model + HW1 friction): a 2 s stop, then for DRAIN_S the controller
is fed the BACKLOG (the states recorded from the stop's start, replayed 10x faster than real time, each restamped
"now" - what the drain delivers), then the live state. Base (6449aee): at 50 ms the blend starts from a backlog pose -
a ~kp x sag torque step on the sagged hips. Now: the wait goes on while the backlog drains (counted in
`resume_drain_waits`, up to R1RT_RESUME_DRAIN_MAX_NS = 400 ms), the blend starts from the live pose - no snap.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from r1sim import cpp_policy
from r1sim.engine import LowCmd, LowState, Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.v3_core import EmulatedBuiltin, TakeoverPlan, V3CoreController

from tests.integration.test_v3_core_onstand import RELEASE_S, T_REQUEST
from tests.integration.test_v3_resume import BOARD, BODY, T_SILENCE, hanging  # noqa: F401  (the module fixture)

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py is not built (scripts/ci.sh cpp)")

STOP_S = 2.0
DRAIN_S = 0.08  # the backlog drain after the stop: past the old 50 ms expiry, inside the new 400 ms limit
DRAIN_SPEED = 10.0  # backlog seconds replayed per second of drain (all of it far behind the board's tick)
STEP_MAX_NM = 2.0  # the largest 1 ms torque change after the drain (test_v3_resume's limit; base: ~kp x sag)


class _Rec:
    """Every physics tick's reported state (never stalled): the backlog the drain replays."""

    def __init__(self) -> None:
        self.states: list[LowState] = []

    def __call__(self, st: LowState) -> LowCmd | None:
        self.states.append(dataclasses.replace(st))
        return None

    def at(self, t: float) -> LowState:
        ts = np.fromiter((s.t for s in self.states), dtype=np.float64)
        return self.states[int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))]


class _Drain:
    """Registered as "v3" (stalled by the schedule); after the stop feeds the inner controller the backlog for DRAIN_S
    (the recorded state of the stop's start onward, with its old tick and pose, its time = now), then the live state."""

    def __init__(self, inner: Any, rec: _Rec) -> None:
        self.inner = inner
        self.rec = rec
        self.last_t: float | None = None
        self.t_res: float | None = None
        self.t_stop: float | None = None

    def __call__(self, st: LowState) -> LowCmd | None:
        t = float(st.t)
        if self.last_t is not None and self.t_res is None and t - self.last_t > 0.1:
            self.t_res, self.t_stop = t, self.last_t
        self.last_t = t
        if self.t_res is not None and self.t_stop is not None and t < self.t_res + DRAIN_S:
            old = self.rec.at(self.t_stop + (t - self.t_res) * DRAIN_SPEED)
            st = dataclasses.replace(old, t=st.t)  # stamped now by the controller; tick + pose from the backlog
        out: LowCmd | None = self.inner(st)
        return out


def _run(hanging: tuple[Any, np.ndarray], *, guarded: bool) -> dict[str, Any]:  # noqa: F811
    w, q0 = hanging
    sched = FaultSchedule([Fault("controller_stall", T_SILENCE, STOP_S, {"name": "v3"})], seed=1)
    sim = Simulation(w.model, w.index, faults=sched, driver=BOARD)
    sim.reset(q0)
    builtin = EmulatedBuiltin(w.model)
    plan = TakeoverPlan(builtin, t_request_s=T_REQUEST, release_s=RELEASE_S, target="on_stand", via_mailbox=True)
    ctl: Any
    if guarded:
        from r1sim.v3_guard import V3GuardedController

        ctl = V3GuardedController(w.model, faults=sched, takeover=plan)
        period = ctl.writer_period_s
    else:
        ctl = V3CoreController(w.model, faults=sched, takeover=plan)
        period = ctl.period_s
    rec = _Rec()
    drain = _Drain(ctl, rec)
    sim.add_controller(builtin, period, "builtin")
    sim.add_controller(rec, sim.dt, "rec")
    sim.add_controller(drain, ctl.period_s, "v3")
    rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    t_end = T_SILENCE + STOP_S + DRAIN_S + 1.2
    while sim.time < t_end - 1e-9:
        sim.run_for(0.001)
        if sim.time >= T_SILENCE - 0.05:
            s = sim.read_state()
            rows.append((sim.time, s.q.copy(), s.tau_est.copy()))
    t = np.array([r[0] for r in rows])
    q = np.array([r[1] for r in rows])
    tau = np.array([r[2] for r in rows])
    t_res = T_SILENCE + STOP_S
    win = (t >= t_res - 0.002) & (t <= t_res + DRAIN_S + 0.3)
    dtau = np.abs(np.diff(tau[:, BODY], axis=0))[win[1:]]
    pre = q[(t >= T_SILENCE - 0.05) & (t < T_SILENCE)][-1, BODY]
    k_res = int(np.argmin(np.abs(t - t_res)))
    return {
        "ctl": ctl,
        "step_nm": float(dtau.max()),
        "sag_max": float(np.abs(q[k_res, BODY] - pre).max()),
        "end_err": float(np.abs(q[-1, BODY] - pre).max()),
    }


def test_the_core_waits_out_a_long_backlog_drain_instead_of_blending_from_an_old_pose(hanging: Any) -> None:  # noqa: F811
    r = _run(hanging, guarded=False)
    st = r["ctl"].status()
    waits, unc = st.get("resume_drain_waits"), st["resume_uncaught"]
    print(f"\n[ info ] core, {DRAIN_S * 1e3:.0f} ms drain after a {STOP_S} s stop: step {r['step_nm']:.2f} N m, "
          f"sag {r['sag_max']:.3f} rad, drain_waits {waits}, uncaught {unc}")  # fmt: skip
    assert r["sag_max"] > 0.02, r  # the board released after 0.385 s: there is a sag to snap back
    assert r["step_nm"] < STEP_MAX_NM, r  # base: the blend from a backlog pose at 50 ms = the snap
    assert st["resume_blends"] == 1 and st["resume_uncaught"] == 0 and st["resume_drain_waits"] == 1, st
    assert r["end_err"] < 0.03, r


def test_the_guard_waits_out_a_long_backlog_drain_after_a_whole_machine_stop(hanging: Any) -> None:  # noqa: F811
    r = _run(hanging, guarded=True)
    ctl = r["ctl"]
    g = ctl.guard.status()
    waits, unc = g.get("resume_drain_waits"), g["resume_uncaught"]
    print(f"\n[ info ] guard, {DRAIN_S * 1e3:.0f} ms drain after a {STOP_S} s whole-machine stop: step "
          f"{r['step_nm']:.2f} N m, drain_waits {waits}, uncaught {unc}")  # fmt: skip
    assert ctl.two_writers == 0 and ctl.core_writes_while_guarded == 0
    assert r["sag_max"] > 0.02, r
    assert r["step_nm"] < STEP_MAX_NM, r  # base: the guard's blend from a backlog pose at 50 ms
    assert g["resume_takeovers"] == 1 and g["resume_blends"] == 1 and g["resume_uncaught"] == 0, g
    assert g["resume_drain_waits"] == 1, g
