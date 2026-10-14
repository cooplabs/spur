"""r1guard during TAKEOVER_BLEND with the REAL core's takeover (session 5; GUARD.md "not solid" 2; lockstep, real
profile on the floor). Unitree's built-in is the S1 fixture (`EmulatedBuiltin`, a stiff stand); r1d's part goes through
the mode-request mailbox. The core stalls (writer + policy thread) 0.4 s into its 1.0 s blend for the 9 Oct stall's
4.018 s: the guard takes over <= 15 ms after the core's last tick and continues the core's OWN blend (the mirror's
blend start and length: its leg gains follow (1 - a) 200 + a kp_policy to the end of the core's blend, then its
policy), no step in the command, every command with stiff legs, one writer; when the core returns it finds the token
gone. Session 4's guard blended 0.3 s x (1 - a) from its takeover instead (a different ramp)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1sim import cpp_policy, world
from r1sim.engine import Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.model import centred_stand_pose
from r1sim.v3_core import MODE_TAKEOVER_BLEND, EmulatedBuiltin, TakeoverPlan


def _has_guard() -> bool:
    if cpp_policy.find_module() is None:
        return False
    try:
        return hasattr(cpp_policy.import_r1rt_py(), "Guard")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_guard(), reason="r1rt_py with Guard not built: run scripts/ci.sh cpp")

T_REQUEST = 1.0
RELEASE_S = 0.3
BLEND_S = 1.0
T_STALL = T_REQUEST + RELEASE_S + 0.4  # 40 % into the blend
STALL_S = 4.018  # the 9 Oct stall
TILT_MAX = 4.0  # deg: the S1 fixture's centred stand leans ~2-3 deg by itself (test_v3_core_takeover.py)


def _tilt_deg(quat: np.ndarray) -> float:
    _, x, y, _ = quat
    return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


@pytest.mark.slow
def test_the_guard_continues_the_cores_takeover_blend_into_balance() -> None:
    from r1sim.v3_guard import OWNER_R1GUARD, V3GuardedController

    w = world.build_world("real", visual=False)
    sched = FaultSchedule([Fault("controller_stall", T_STALL, STALL_S, {"name": "v3_core"})], seed=1)
    sim = Simulation(w.model, w.index, faults=sched)
    sim.reset(w.initial_qpos(centred_stand_pose(w.model)))
    builtin = EmulatedBuiltin(w.model)
    plan = TakeoverPlan(builtin, t_request_s=T_REQUEST, release_s=RELEASE_S, via_mailbox=True)
    ctl: Any = V3GuardedController(w.model, faults=sched, takeover=plan, trace_cmd=True)
    sim.add_controller(builtin, ctl.writer_period_s, "builtin")
    sim.add_controller(ctl, ctl.period_s, "v3")
    pel = w.index.pelvis_body
    tilt_max = 0.0
    t_end = T_STALL + STALL_S + 1.0
    while sim.time < t_end - 1e-9:
        sim.run_for(0.01)
        tilt_max = max(tilt_max, _tilt_deg(sim.d.xquat[pel]))
    assert plan.request_status == 0 and plan.decision_status == 0
    gs = ctl.guard_status()
    fx = ctl.forensics()
    assert gs["takeovers"] == 1 and gs["mode_from"] == MODE_TAKEOVER_BLEND
    assert fx["rt_silence_ns"] <= 15_000_000
    assert ctl.two_writers == 0 and ctl.core_writes_while_guarded == 0
    assert ctl.view.token()[0] == OWNER_R1GUARD
    gt = ctl.trace_array()
    published = gt[(gt[:, 2] + gt[:, 3]) > 0]
    assert np.all(published[published[:, 0] >= T_REQUEST, 5] > 30.0)  # every command after MIRROR: stiff legs
    after = published[published[:, 0] >= T_STALL]
    assert after[0, 0] - T_STALL <= 0.015  # the robot's command gap
    assert tilt_max < TILT_MAX, tilt_max
    # the guard's leg gains continue the core's ramp: at the end of the CORE's blend (T_REQUEST + RELEASE_S + 1.0 s)
    # the guard publishes its policy's gains, not before (session 4: ~0.19 s after the takeover)
    tr = ctl.cmd_trace  # the core's own commands (up to the stall)
    kp_snapshot = float(tr[next(i for i, r in enumerate(tr) if r[1])][5][0])  # MIRROR: the built-in's hip kp
    assert kp_snapshot > 150.0
    st_blend_end = T_REQUEST + RELEASE_S + BLEND_S
    g_after = gt[(gt[:, 0] >= T_STALL) & (gt[:, 3] > 0)]
    kmin_mid = g_after[(g_after[:, 0] > T_STALL + 0.25) & (g_after[:, 0] < st_blend_end - 0.05), 5]
    assert np.all(kmin_mid > 35.0)  # still blending from 200: the legs' min kp above the policy's ankles (31.4)
    kmin_late = g_after[g_after[:, 0] > st_blend_end + 0.01, 5]
    np.testing.assert_allclose(kmin_late, 31.385, atol=0.01)  # the policy's ankle kp once the core's blend ended
