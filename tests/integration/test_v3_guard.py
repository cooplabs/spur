"""r1guard (C++, r1rt_py.Guard) as the v3 core's hot standby on the simulated robot, in lockstep (real profile on the
floor; the core's writer 500 Hz + policy thread 50 Hz, the guard 1 kHz on the same segment; r1sim.v3_guard).

1. Standby is passive: with the core alive the guard publishes nothing, keeps a valid shadow, and the robot's motion is
   bit-identical to the core alone.
2. **A7 v3 part (DESIGN §13.1, the G-HW1 rehearsal's lockstep form): the 9 Oct stall (4.018 s) injected into the core
   ITSELF (writer + policy thread: no heartbeat, no mirror)** -> the guard takes the token <= 15 ms after the core's
   last tick, balances in place (max tilt < 3 deg through the stall and 2 s after), never limp (every command sent has
   stiff legs), forensics captured; when the core comes back it finds the token gone and publishes nothing - one writer
   at every tick. Two stances: the policy's default arms and the logged table_ready arms (9 Oct, telemetry line 1137).
3. A NaN on the guard's own network output while the core is alive: flagged on the guard's heartbeat, nothing changes
   for the robot; the core then stalls -> the guard's first action is invalid -> STAND_FREEZE (rung 3: the ramp from
   the core's last command into the CoM-centred stand), never damp, the robot stays up.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import pytest
from r1sim import cpp_policy, world
from r1sim import joints as J
from r1sim.engine import Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.v3_core import FAULT_BITS, INJECT_POLICY_NAN, MODE_BALANCE, MODE_STAND_FREEZE, V3CoreController
from r1sim.v3_guard import MODE_GUARDED, OWNER_R1GUARD, OWNER_R1RT, V3GuardedController, token_owner


def _has_guard() -> bool:
    if cpp_policy.find_module() is None:
        return False
    try:
        return hasattr(cpp_policy.import_r1rt_py(), "Guard")
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_guard(), reason="r1rt_py with Guard not built: run scripts/ci.sh cpp")

STALL_S = 4.018  # 9 Oct: telemetry t 433.7816 -> 437.8001 (V2_INVENTORY App. D facts row 1)
T_STALL = 2.0
# The arms' commanded pose in the last row before the 9 Oct stall (table_ready; NB r1-motor-controller-v2/results/
# serve-20261010T010206Z/telemetry.jsonl line 1137, q_cmd[14:24]) - as tests/integration/test_v3_core.py.
ARMS_TABLE_READY = (0.0689, 0.685, 0.0192, -0.5269, -1.5534, 0.2079, -0.1853, 0.0042, 0.8141, 0.0039)
STALL_IN_THE_CORE = Fault("controller_stall", T_STALL, STALL_S, {"name": "v3_core"})  # writer + policy thread


def _tilt_deg(quat: np.ndarray) -> float:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(r[8], -1.0, 1.0))))


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _sim(w: world.World, faults: FaultSchedule | None, arms: Any = None) -> Simulation:
    sim = Simulation(w.model, w.index, faults=faults)
    pose = J.POSE_STAND.copy()
    if arms is not None:
        pose[J.ARMS] = arms
    sim.reset(w.initial_qpos(pose))
    return sim


def _run(
    w: world.World,
    faults: list[Fault],
    seconds: float,
    arms: Any = None,
    guard_nan: tuple[float, float] | None = None,
) -> dict[str, Any]:
    sched = FaultSchedule(faults, seed=1)
    sim = _sim(w, sched, arms)
    ctl = V3GuardedController(w.model, faults=sched, trace=True)
    if guard_nan is not None:
        ctl.guard.inject(INJECT_POLICY_NAN, round(guard_nan[0] * 1e9), round(guard_nan[1] * 1e9))
    sim.add_controller(ctl, ctl.period_s, "v3_guarded")
    pel = w.index.pelvis_body
    tilts: list[tuple[float, float]] = []
    statuses: list[tuple[float, dict[str, Any]]] = []
    for _ in range(round(seconds / 0.01)):
        sim.run_for(0.01)
        tilts.append((sim.time, _tilt_deg(sim.d.xquat[pel])))
        if round(sim.time / 0.01) % 50 == 0:
            statuses.append((sim.time, ctl.guard_status()))
    return {
        "ctl": ctl,
        "sched": sched,
        "tilts": np.array(tilts),
        "statuses": statuses,
        "trace": ctl.trace_array(),
        "core_trace": np.array(ctl.trace),
        "gstatus": ctl.guard_status(),
        "fx": ctl.forensics(),
        "qpos": sim.d.qpos.copy(),
    }


def test_standby_is_passive_bit_identical_to_the_core_alone(w: world.World) -> None:
    seconds = 2.0
    sim_a = _sim(w, None)
    alone = V3CoreController(w.model)
    sim_a.add_controller(alone, alone.period_s, "v3")
    sim_a.run_for(seconds)
    sim_b = _sim(w, None)
    ctl = V3GuardedController(w.model)
    sim_b.add_controller(ctl, ctl.period_s, "v3_guarded")
    sim_b.run_for(seconds)
    np.testing.assert_array_equal(sim_b.d.qpos, sim_a.d.qpos)
    g = ctl.guard_status()
    assert g["active"] == 0 and g["publishes"] == 0 and g["takeovers"] == 0
    assert g["shadow_steps"] >= 95 and g["shadow_invalid"] == 0 and g["shadow_valid"] == 1
    # bumpless: every shadow step equals the core's own action of the same instant (same code, state and inputs)
    assert g["shadow_matches"] >= 95
    assert g["shadow_match_err_max_rad"] <= 1e-9, g["shadow_match_err_max_rad"]
    assert ctl.two_writers == 0
    assert g["rt_beat_age_ns"] <= 2_000_000


def _check_takeover(r: dict[str, Any]) -> dict[str, float]:
    ctl: V3GuardedController = r["ctl"]
    tr = r["trace"]  # t, core_tick, core_pub, guard_pub, guard_active, kp_min
    g = r["gstatus"]
    fx = r["fx"]
    t_end = T_STALL + STALL_S
    # the takeover: once, from BALANCE, <= 15 ms after the core's last heartbeat (stale = older than 5 x 2 ms)
    assert g["takeovers"] == 1 and g["active"] == 1 and g["mode"] == MODE_GUARDED
    assert fx["captured"] == 1 and fx["mode_from"] == MODE_BALANCE
    t_take = g["t_takeover_ns"] * 1e-9
    silence_ms = fx["rt_silence_ns"] * 1e-6
    assert T_STALL <= t_take <= T_STALL + 0.015
    assert 10.0 < silence_ms <= 15.0, silence_ms
    assert token_owner(fx["token_before"]) == OWNER_R1RT and token_owner(fx["token_after"]) == OWNER_R1GUARD
    assert fx["mirror_policy_valid"] == 1 and fx["shadow_valid"] == 1
    assert fx["proc_err"] == -1  # /proc capture: the guard process wrapper's (Linux), not lockstep
    # one writer at every call; the core after its stall publishes nothing (it checks the token before each write)
    assert ctl.two_writers == 0 and ctl.core_writes_while_guarded == 0
    after = tr[:, 0] > t_end + 0.01
    assert np.any(after & (tr[:, 1] == 1)) and not np.any(tr[after, 2])
    s = ctl.status()
    assert s["faults"] & FAULT_BITS["not_owner"] and s["mode"] == MODE_BALANCE  # the core thinks it balances
    # the guard publishes from the takeover on, every cycle, with stiff legs; never limp at any call that sent anything
    guarded = tr[:, 0] >= t_take - 1e-9
    assert np.all(tr[guarded, 3] == 1)
    sent = ~np.isnan(tr[:, 5])
    assert np.all(tr[sent, 5] > 1.0), float(np.min(tr[sent, 5]))
    # no gap in what the robot receives: between the core's last command and the guard's first <= 15 ms
    last_core = float(np.max(tr[(tr[:, 2] == 1) & (tr[:, 0] < t_take), 0]))
    gap_ms = (t_take - last_core) * 1e3
    assert gap_ms <= 15.0
    # balance through the stall and 2 s after
    ti = r["tilts"]
    win = (ti[:, 0] >= T_STALL) & (ti[:, 0] <= t_end + 2.0)
    max_tilt = float(np.max(ti[win, 1]))
    assert max_tilt < 3.0, max_tilt
    assert g["act_mode"] == MODE_BALANCE and g["invalid_actions"] == 0
    assert g["policy_steps"] >= 50 * (t_end + 2.0 - t_take) - 2
    return {"t_takeover": t_take, "silence_ms": silence_ms, "gap_ms": gap_ms, "max_tilt": max_tilt}


@pytest.mark.slow
def test_a7_v3_part_the_9_oct_stall_in_the_core_guard_takes_over(w: world.World) -> None:
    r = _run(w, [STALL_IN_THE_CORE], T_STALL + STALL_S + 2.5)
    _check_takeover(r)


@pytest.mark.slow
def test_a7_v3_part_with_the_logged_table_ready_arms(w: world.World) -> None:
    r = _run(w, [STALL_IN_THE_CORE], T_STALL + STALL_S + 2.5, ARMS_TABLE_READY)
    _check_takeover(r)
    # the arms HOLD where the core held them (its executor's targets): still at table_ready 2.5 s after the stall
    # (sag without gravity feed-forward ~0.08 rad, S3)
    arms_q = r["qpos"][w.index.qadr[J.ARMS]]
    assert float(np.max(np.abs(arms_q - np.asarray(ARMS_TABLE_READY)))) < 0.15


@pytest.mark.slow
def test_a_nan_in_the_guards_policy_flagged_on_standby_then_stand_freeze_never_damp(w: world.World) -> None:
    nan = (T_STALL - 0.5, T_STALL + 0.5)  # the guard's own network output only
    r = _run(w, [STALL_IN_THE_CORE], T_STALL + 5.0, guard_nan=nan)
    ctl: V3GuardedController = r["ctl"]
    standby = [s for t, s in r["statuses"] if nan[0] + 0.05 < t < T_STALL]
    assert standby and all(s["shadow_valid"] == 0 and s["faults"] & FAULT_BITS["policy_invalid"] for s in standby)
    core_before = r["core_trace"][r["core_trace"][:, 0] < T_STALL]
    assert np.all(core_before[core_before[:, 0] > 0.1, 1] == MODE_BALANCE)  # the core unaffected
    g = r["gstatus"]
    assert g["takeovers"] == 1 and g["act_mode"] == MODE_STAND_FREEZE and g["invalid_actions"] >= 1
    assert g["faults"] & FAULT_BITS["policy_invalid"]
    tr = r["trace"]
    sent = ~np.isnan(tr[:, 5])
    assert np.all(tr[sent, 5] > 1.0)
    assert ctl.two_writers == 0
    ti = r["tilts"]
    after = ti[:, 0] >= T_STALL
    max_tilt = float(np.max(ti[after, 1]))
    assert max_tilt < 5.0, max_tilt  # the centred stand's own lean is ~3 deg (S3: STAND_FREEZE 2.95 deg)
