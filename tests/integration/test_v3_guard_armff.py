"""Fix #28 (session 8, HW1 T6): r1guard holds the arms / head WITH the core's last feed-forward.

On the robot (11 Oct 2026, T6 x2: `results/hw1_20261010/end/session_part4/records/*t6_3s*`; the wrap-up analysis
the HW1 wrap-up analysis D) the guard took over in 10.7 ms and held the legs within
0.003 rad, but the ARMS moved 0.004-0.020 rad: the core sends the arms' gravity feed-forward from its arm model
(left -1.46 / +1.60 / +0.14 / -0.51 N m before the stop) and the guard published the arms with tau_ff = 0
(guard.cpp, session 6: "no velocity, no feed-forward") - every arm motion's sign matched the lost feed-forward.

Here in lockstep: the hanging robot in ON_STAND, both arms at Unitree's stand-lock preset (the HW1 pose: shoulder
pitch 0.2, roll +-0.2, yaw 0, elbow 0.8, wrist 0), the core stalled 1.5 s (`controller_stall name="v3_core"`: T6), the
guard holds. The shoulder friction is HW1's (~1 N m, the S5 / S9 analysis: the model's 2.5 N m on shoulder pitch / roll
would hide the lost feed-forward); the elbow keeps the model's 0.2 N m. Base (bad5a5c): the arms sag when the guard
takes over; now: < 0.005 rad.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import pytest
from r1sim import cpp_policy
from r1sim.engine import Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.v3_core import GROUP_ARM_LEFT, GROUP_ARM_RIGHT, MODE_ON_STAND, EmulatedBuiltin, TakeoverPlan

from tests.integration.test_v3_core_onstand import RELEASE_S, T_ON_STAND, T_REQUEST, _hanging_world

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py is not built (scripts/ci.sh cpp)")

# Unitree's stand-lock arm preset (HW1 S1.0, fix list #25): shoulder pitch, roll, yaw, elbow, wrist roll
ARM_LEFT = np.array([0.2, 0.2, 0.0, 0.8, 0.0])
ARM_RIGHT = np.array([0.2, -0.2, 0.0, 0.8, 0.0])
ARMS = slice(14, 24)
SHOULDER_FRICTION_HW1 = 1.0  # N m (HW1 S5 / S9: ~1 N m of shoulder friction, the size of the 0.5 kg payload effect)
T_POSE = T_ON_STAND + 0.2
T_STALL = T_POSE + 4.5  # the trajectory ends at T_POSE + 1.5; the arms settle (stick-slip at 1 N m) by then
STALL_S = 1.5
MOTION_MAX = 0.005  # rad (the contract: HW1's 0.004-0.020 must become < 0.005)


@pytest.fixture(scope="module")
def hanging() -> tuple[Any, np.ndarray]:
    w, q = _hanging_world()
    m = w.model
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.endswith(("shoulder_pitch_joint", "shoulder_roll_joint")):
            m.dof_frictionloss[m.jnt_dofadr[j]] = SHOULDER_FRICTION_HW1
    return w, q


def _t6(hanging: tuple[Any, np.ndarray]) -> dict[str, Any]:
    from r1sim.v3_guard import V3GuardedController

    w, q0 = hanging
    sched = FaultSchedule([Fault("controller_stall", T_STALL, STALL_S, {"name": "v3_core"})], seed=1)
    sim = Simulation(w.model, w.index, faults=sched)
    sim.reset(q0)
    builtin = EmulatedBuiltin(w.model)  # Unitree's stand lock (stiff), as at HW1
    plan = TakeoverPlan(builtin, t_request_s=T_REQUEST, release_s=RELEASE_S, target="on_stand", via_mailbox=True)
    ctl = V3GuardedController(w.model, faults=sched, takeover=plan, trace_cmd=True)
    sim.add_controller(builtin, ctl.writer_period_s, "builtin")
    sim.add_controller(ctl, ctl.period_s, "v3")

    def pose(c: Any, t_ns: int) -> None:
        c.send_trajectory(t_ns, GROUP_ARM_LEFT, ARM_LEFT, 1.5)
        c.send_trajectory(t_ns, GROUP_ARM_RIGHT, ARM_RIGHT, 1.5)

    ctl.at(T_POSE, pose)
    rows: list[tuple[float, np.ndarray]] = []
    while sim.time < T_STALL + STALL_S - 0.01:
        sim.run_for(0.002)
        if sim.time >= T_STALL - 0.2:
            rows.append((sim.time, np.asarray(sim.read_state().q, dtype=np.float64).copy()))
    t = np.array([r[0] for r in rows])
    q = np.array([r[1] for r in rows])
    before = q[(t >= T_STALL - 0.1) & (t < T_STALL)].mean(axis=0)
    hold = q[t >= T_STALL + 0.02]
    motion = np.abs(hold - before).max(axis=0)
    return {"ctl": ctl, "motion": motion, "before": before}


def test_t6_the_guard_keeps_the_arms_feed_forward_in_a_loaded_arm_pose(hanging: Any) -> None:
    r = _t6(hanging)
    ctl = r["ctl"]
    gs = ctl.guard_status()
    assert gs["takeovers"] == 1 and gs["mode_from"] == MODE_ON_STAND and gs["act_mode"] == MODE_ON_STAND
    assert ctl.two_writers == 0
    assert np.abs(r["before"][ARMS] - np.r_[ARM_LEFT, ARM_RIGHT]).max() < 0.03  # the arms at the preset (loaded)
    arm = r["motion"][ARMS]
    print(f"\n[ info ] T6 arm motion during the guard's hold (rad): {np.round(arm, 4).tolist()}")
    assert float(arm.max()) < MOTION_MAX, arm  # base: the arms sag at the takeover (the lost feed-forward)
    assert float(r["motion"][:12].max()) < 0.003  # the legs as on the robot (< 0.003 rad)
