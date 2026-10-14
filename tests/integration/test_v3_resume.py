"""Fix #29 (session 8, HW1's board-hold test): the resume after a long silence, in lockstep on the hanging robot with
the BOARD MODEL.
 HW1's board-hold test (11 Oct 2026, the 500 ms pause's lowstate recording; tests/assets/hw1/board_hold_README.md):
Unitree's board holds the last LowCmd ~400 ms after the commands stop, then releases into a DAMPED sink (the hips ~0.12
rad/s). A controller that is silent longer than that (r1rt and r1guard stalled together - the 9 Oct class - or r1rt
alone with the guard off, as in the board-hold test) and then resumes its OLD target at full stiffness kicks the sagged
limbs back: at 500 ms the robot's hips came back 0.014 rad at kp 200 (the user stopped the test there). The fix: the
core (and the guard) detect their own silence (`resume_gap_s`, default 0.1 s) and re-enter through a blend from the
MEASURED pose to the old target over takeover_blend_s (ON_STAND / the arms' HOLD), the freeze ramp restarted from the
measured pose (STAND_FREEZE).

The board model here is the engine's command timeout (`DriverConfig(cmd_timeout_s=0.4, on_cmd_timeout="damp",
damp_kd=15)`: the B6 estimate of the session-8 contract, independent of the board agent's fitted value). The torque
"step" is the largest change of the applied torque between two consecutive 1 ms physics steps on the legs and arms in
the 0.3 s after the resume; the base (bad5a5c) snaps the hips back at kp 200 (tens of N m in one step) and fails.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import pytest
from r1sim import cpp_policy
from r1sim.driver import DriverConfig
from r1sim.engine import Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.v3_core import (
    MODE_ON_STAND,
    MODE_STAND_FREEZE,
    REQ_STAND_FREEZE,
    REQUESTER_OPERATOR,
    EmulatedBuiltin,
    TakeoverPlan,
    V3CoreController,
)

from tests.integration.test_v3_core_onstand import RELEASE_S, T_ON_STAND, T_REQUEST, _hanging_world

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py is not built (scripts/ci.sh cpp)")

BOARD = DriverConfig(kind="board", cmd_timeout_s=0.4, on_cmd_timeout="damp", damp_kd=15.0)
T_SILENCE = T_ON_STAND + 0.5
STEP_MAX_NM = 2.0  # the largest 1 ms torque change allowed after the resume (fixed: <= 1.1 N m; base 1 s 5.2, 2 s 11.6)
DQ_MAX = 0.6  # rad/s on any leg / arm joint after the resume (the sag itself is ~0.1-0.3 rad/s)
BODY = slice(0, 24)  # legs, waist, arms (the head carries ~no load)


# The HW1 leg / waist friction (the session-8 contract's provisional table = the wrap-up re-fit of the S4 ramps): the
# model's 2.5 N m on the hips holds a hanging hip against most of its gravity torque and hides the sag (the base then
# sags only ~0.04 rad in 1 s); the real hips sank ~0.12 rad/s (the board-hold test).
HW1_FRICTION = {"hip_pitch": 0.6, "hip_roll": 0.4, "hip_yaw": 0.6, "knee": 0.55, "ankle_pitch": 2.9,
                "ankle_roll": 1.4, "waist_roll": 0.5, "waist_yaw": 0.4}  # fmt: skip


@pytest.fixture(scope="module")
def hanging() -> tuple[Any, np.ndarray]:
    w, q = _hanging_world()
    m = w.model
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for key, fr in HW1_FRICTION.items():
            if name.endswith(f"{key}_joint"):
                m.dof_frictionloss[m.jnt_dofadr[j]] = fr
    return w, q


def _run(
    hanging: tuple[Any, np.ndarray], silence_s: float, *, guarded: str | None = None, freeze: bool = False
) -> dict[str, Any]:
    """ON_STAND through the mailbox, then a silence of `silence_s` at T_SILENCE: the core alone ("v3" stalled), the core
    with the guard OFF (observe only: the board-hold test) or the core + the guard together (the whole-machine stall).
    freeze: the operator's STAND_FREEZE from ON_STAND first (S10), the silence after its ramp (the hold of the stand
    pose)."""
    w, q0 = hanging
    t_sil = T_SILENCE + (0.5 if freeze else 0.0)
    if guarded is None:
        faults = [Fault("controller_stall", t_sil, silence_s, {"name": "v3"})]
    elif guarded == "observe":  # the board-hold test: the guard runs observe-only, r1rt alone stalls
        faults = [Fault("controller_stall", t_sil, silence_s, {"name": "v3_core"})]
    else:  # "both": the whole controller (r1rt AND r1guard) stalls - the guarded controller is registered as "v3"
        faults = [Fault("controller_stall", t_sil, silence_s, {"name": "v3"})]
    sched = FaultSchedule(faults, seed=1)
    sim = Simulation(w.model, w.index, faults=sched, driver=BOARD)
    sim.reset(q0)
    builtin = EmulatedBuiltin(w.model)  # stiff at the stand pose = Unitree's stand lock (HW1: ON_STAND from it)
    plan = TakeoverPlan(builtin, t_request_s=T_REQUEST, release_s=RELEASE_S, target="on_stand", via_mailbox=True)
    ctl: Any
    if guarded is None:
        ctl = V3CoreController(w.model, faults=sched, takeover=plan, trace_cmd=True)
        sim.add_controller(builtin, ctl.period_s, "builtin")
    else:
        from r1sim.v3_guard import V3GuardedController

        params = {"takeover_enabled": 0} if guarded == "observe" else None
        ctl = V3GuardedController(w.model, faults=sched, takeover=plan, trace_cmd=True, guard_params=params)
        sim.add_controller(builtin, ctl.writer_period_s, "builtin")
    sim.add_controller(ctl, ctl.period_s, "v3")
    if freeze:
        ctl.at(T_ON_STAND + 0.1, lambda c, t_ns: c.view.request_mode(REQ_STAND_FREEZE, REQUESTER_OPERATOR, t_ns))
    t_end = t_sil + silence_s + 1.5
    rows: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    while sim.time < t_end - 1e-9:
        sim.run_for(0.001)
        if sim.time >= t_sil - 0.05:
            st = sim.read_state()
            rows.append((sim.time, st.q.copy(), st.dq.copy(), st.tau_est.copy()))
    t = np.array([r[0] for r in rows])
    q = np.array([r[1] for r in rows])
    dq = np.array([r[2] for r in rows])
    tau = np.array([r[3] for r in rows])
    t_res = t_sil + silence_s
    win = (t >= t_res - 0.002) & (t <= t_res + 0.3)
    dtau = np.abs(np.diff(tau[:, BODY], axis=0))[win[1:]]
    pre = (t >= t_sil - 0.05) & (t < t_sil)
    sag = np.abs(q[np.argmin(np.abs(t - t_res)), BODY] - q[pre][-1, BODY])
    after = (t >= t_res) & (t <= t_res + 1.0)
    end = q[-1, BODY] - q[pre][-1, BODY]  # back at the old target after the blend
    return {
        "ctl": ctl,
        "sim": sim,
        "step_nm": float(dtau.max()),
        "dq_max": float(np.abs(dq[after][:, BODY]).max()),
        "sag_max": float(sag.max()),
        "end_err": float(np.abs(end).max()),
        "sag_j": sag,
        "step_j": dtau.max(axis=0),
    }


@pytest.mark.parametrize("silence_s", [0.5, 1.0, 2.0])
def test_the_core_resumes_from_a_long_silence_without_a_snap(hanging: Any, silence_s: float) -> None:
    r = _run(hanging, silence_s)
    st = r["ctl"].status()
    assert st["mode"] == MODE_ON_STAND
    # the board released at 0.4 s: the limbs sagged (0.5 s: only 0.1 s of sink)
    assert r["sag_max"] > (0.004 if silence_s <= 0.5 else 0.02), r
    assert r["step_nm"] < STEP_MAX_NM, r  # base: the snap back to the old target at full stiffness
    assert r["dq_max"] < DQ_MAX, r
    assert r["end_err"] < 0.03, r  # back at the old hold after the blend
    assert st["stops"] == 1 and st["stop_ns_max"] >= int((silence_s - 0.01) * 1e9), st
    assert st["resume_blends"] == 1 and st["resume_waits_expired"] == 0 and st["resume_uncaught"] == 0, st
    mi = r["ctl"].view.mirror()  # what r1d reads for `r1 status` (#31)
    assert mi["stops"] == 1 and mi["stop_ns_max"] == st["stop_ns_max"] and mi["resume_waiting"] == 0, mi
    info = r["ctl"].view.rt_info()  # r1d's `r1 status` reads rt_info() (deploy, #31)
    assert info["stops"] == 1 and abs(info["stop_ms_max"] - st["stop_ns_max"] * 1e-6) < 1e-9, info


def test_a_short_silence_inside_the_boards_hold_changes_nothing(hanging: Any) -> None:
    """0.2 s < the board's 0.4 s hold: no sag, the resume blend from the measured (= held) pose moves nothing."""
    r = _run(hanging, 0.2)
    assert r["sag_max"] < 1e-3, r
    assert r["step_nm"] < 2.0, r
    assert r["end_err"] < 0.01, r


@pytest.mark.parametrize("mode", ["observe", "both"])
def test_the_guarded_controller_resumes_without_a_snap(hanging: Any, mode: str) -> None:
    """The board-hold test (the guard observe-only, r1rt silent 1 s) and the whole-machine stall (r1rt + r1guard silent
    1 s): whoever publishes first after the silence (the core, or the guard taking over a core it finds long silent)
    re-enters through the blend from the measured pose; never two writers."""
    r = _run(hanging, 1.0, guarded=mode)
    ctl = r["ctl"]
    assert ctl.two_writers == 0 and ctl.core_writes_while_guarded == 0
    assert r["sag_max"] > 0.01, r
    assert r["step_nm"] < STEP_MAX_NM, r
    assert r["dq_max"] < DQ_MAX, r
    assert r["end_err"] < 0.03, r


def test_a_stand_freeze_hold_resumes_through_its_ramp_from_the_measured_pose(hanging: Any) -> None:
    """S10's STAND_FREEZE from ON_STAND (the hold of the stand pose after its 0.4 s ramp), then a 2 s silence: the ramp
    restarts from the measured (sagged) pose (stand_ramp_s) instead of snapping to the stand pose."""
    r = _run(hanging, 2.0, freeze=True)
    st = r["ctl"].status()
    assert st["mode"] == MODE_STAND_FREEZE and st["freeze_from_stand"] == 1, st
    assert r["sag_max"] > 0.02, r
    assert r["step_nm"] < STEP_MAX_NM, r  # base: the snap to the stand pose at the stand gains
    assert r["end_err"] < 0.03, r
