"""Fault injection on the engine (DESIGN.md §12.6): each fault acts exactly inside its window, not outside, and a run
with faults is reproducible bit for bit in lockstep."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from r1sim import dex3
from r1sim import model as M
from r1sim.control import StandHold
from r1sim.engine import ControllerThread, HandCmd, LowCmd, LowState, Simulation
from r1sim.faults import Fault, FaultSchedule

from tests.integration.toy import toy_index, toy_model

TOY_A = [3, 9, 14]


def _toy(faults: FaultSchedule) -> Simulation:
    m = toy_model()
    return Simulation(m, toy_index(m), faults=faults)


class _Clock:
    """A 500 Hz controller whose command encodes the time it was made (q = t), so the applied one is visible."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, state: LowState) -> LowCmd:
        self.calls.append(state.t)
        c = LowCmd()
        c.q[TOY_A] = state.t
        c.kp[TOY_A] = 20.0
        return c


def _applied_q(sim: Simulation) -> float:
    a = sim._b_a[0]
    return float(sim.d.ctrl[a] / 20.0)  # board law: ctrl = kp q_des (dq = tau_ff = 0)


# ---------------------------------------------------------------- command path
def test_cmd_gap_discards_every_command_in_the_window() -> None:
    f = FaultSchedule([Fault("cmd_gap", 0.1, 0.1)])
    sim = _toy(f)
    clock = _Clock()
    sim.add_controller(clock, 0.002, "clock")
    sim.run_for(0.15)
    assert _applied_q(sim) == pytest.approx(0.098)  # the last command before the gap is held
    sim.run_for(0.1)
    assert _applied_q(sim) == pytest.approx(0.248)  # commands flow again after it
    assert f.count("cmd_gap", "discard") == 50  # 0.100 .. 0.198 at 500 Hz
    assert all(0.1 <= e.t < 0.2 for e in f.log)


def test_cmd_drop_is_seeded_and_stays_in_its_window() -> None:
    def run() -> tuple[list[float], bytes]:
        f = FaultSchedule([Fault("cmd_drop", 0.1, 0.2, {"p": 0.5})], seed=3)
        sim = _toy(f)
        sim.add_controller(_Clock(), 0.002, "clock")
        sim.run_for(0.4)
        return [e.t for e in f.log], np.asarray(sim.d.qpos).tobytes()

    (t1, q1), (t2, q2) = run(), run()
    assert t1 == t2 and q1 == q2
    assert 25 <= len(t1) <= 75  # ~half of the 100 commands in the window
    assert all(0.1 <= t < 0.3 for t in t1)


def test_cmd_drop_p1_drops_all_and_write_cmd_too() -> None:
    f = FaultSchedule([Fault("cmd_drop", 0.0, 1.0, {"p": 1.0})])
    sim = _toy(f)
    sim.write_cmd(_Clock()(sim.read_state()))
    sim.step(10)
    assert sim.applied_cmds == 0 and f.count("cmd_drop") == 1


# ---------------------------------------------------------------- state side
def test_state_delay_fault_only_inside_its_window() -> None:
    f = FaultSchedule([Fault("state_delay", 0.1, 0.1, {"delay_s": 0.01})])
    sim = _toy(f)
    lags: dict[float, float] = {}
    for _ in range(300):
        sim.step(1)
        lags[round(sim.time, 6)] = sim.time - sim.read_state().t
    assert lags[0.05] == pytest.approx(0.0, abs=1e-9)
    assert lags[0.15] == pytest.approx(0.01)
    assert lags[0.25] == pytest.approx(0.0, abs=1e-9)
    assert f.count("state_delay", "apply") == 1


def test_motor_error_words_inside_the_window() -> None:
    f = FaultSchedule([Fault("motor_error", 0.05, 0.05, {"joints": [3, 14], "word": 7})])
    sim = _toy(f)
    sim.run_for(0.04)
    assert np.all(sim.read_state().motor_err == 0)
    sim.run_for(0.03)
    err = sim.read_state().motor_err
    assert err[3] == 7 and err[14] == 7 and err[9] == 0
    sim.run_for(0.05)
    assert np.all(sim.read_state().motor_err == 0)


def test_hand_state_freeze_holds_the_last_reading() -> None:
    m = M.build_model("real", visual=False)
    hold = StandHold.for_model(m)
    f = FaultSchedule([Fault("hand_state_freeze", 0.1, 0.1, {"side": "left"})])
    sim = Simulation(m, M.RobotIndex(m), faults=f)
    sim.reset(M.stand_qpos(m, hold.pose))
    cmd = hold.command
    for side in dex3.SIDES:  # close both hands so their readings keep changing
        h = HandCmd()
        h.q[:] = dex3.preset("fist", side)
        h.kp[:], h.kd[:] = dex3.KP, dex3.KD
        cmd.hands[side] = h
    sim.write_cmd(cmd)
    sim.run_for(0.098)
    before = sim.read_state()
    sim.run_for(0.05)
    during = sim.read_state()
    np.testing.assert_array_equal(during.hand_q["left"], before.hand_q["left"])
    np.testing.assert_array_equal(during.hand_tau["left"], before.hand_tau["left"])
    assert not np.array_equal(during.hand_q["right"], before.hand_q["right"])  # the other hand moves on
    assert not np.array_equal(during.q, before.q)  # and so does the body
    sim.run_for(0.1)
    after = sim.read_state()
    assert not np.array_equal(after.hand_q["left"], during.hand_q["left"])
    assert f.count("hand_state_freeze", "apply") == 1


# ---------------------------------------------------------------- controller stall (lockstep)
def test_lockstep_controller_stall_skips_calls_and_holds_the_command() -> None:
    f = FaultSchedule([Fault("controller_stall", 0.2, 0.3)])
    sim = _toy(f)
    clock = _Clock()
    sim.add_controller(clock, 0.002, "clock")
    sim.run_for(0.4)
    assert _applied_q(sim) == pytest.approx(0.198)  # held since the stall began
    sim.run_for(0.6)
    assert sim.controller_calls()["clock"] == 500 - 150
    assert f.count("controller_stall", "skip_call") == 150
    assert not any(0.2 <= t < 0.5 for t in clock.calls)


def test_named_controller_stall_affects_only_that_controller() -> None:
    f = FaultSchedule([Fault("controller_stall", 0.0, 0.1, {"name": "b"})])
    sim = _toy(f)
    sim.add_controller(lambda s: None, 0.002, "a")
    sim.add_controller(lambda s: None, 0.002, "b")
    sim.run_for(0.2)
    assert sim.controller_calls() == {"a": 100, "b": 50}


# ---------------------------------------------------------------- reproducibility
def _all_faults(seed: int) -> FaultSchedule:
    return FaultSchedule(
        [
            Fault("controller_stall", 0.30, 0.10),
            Fault("cmd_gap", 0.50, 0.05),
            Fault("cmd_drop", 0.60, 0.20, {"p": 0.3}),
            Fault("state_delay", 0.20, 0.30, {"delay_s": 0.004}),
            Fault("motor_error", 0.10, 0.10, {"joints": [9], "word": 3}),
        ],
        seed=seed,
    )


def test_a_run_with_every_fault_is_bit_identical_on_rerun() -> None:
    def run(seed: int) -> tuple[bytes, list[Any]]:
        f = _all_faults(seed)
        sim = _toy(f)
        sim.add_controller(_Clock(), 0.002, "clock")
        sim.run_for(1.0)
        return np.asarray(sim.d.qpos).tobytes(), [(e.t, e.kind, e.what) for e in f.log]

    a, b = run(1), run(1)
    assert a == b
    assert run(2)[1] != a[1]  # another seed: other drops


# ---------------------------------------------------------------- real time
@pytest.mark.realtime
def test_controller_thread_blocks_during_a_stall_fault() -> None:
    f = FaultSchedule([Fault("controller_stall", 0.3, 0.5)])
    m = toy_model()
    sim = Simulation(m, toy_index(m), faults=f)
    clock = _Clock()
    th = ControllerThread(sim, clock, 500.0, name="clock")
    sim.start_realtime()
    th.start()
    try:
        time.sleep(1.2)
    finally:
        th.stop()
        sim.stop()
    inside = [t for t in clock.calls if 0.305 < t < 0.795]
    assert inside == []
    assert len(th.stalls) == 1 and th.stalls[0].sim_end - th.stalls[0].sim_start >= 0.45
    assert sim.time > 1.0  # physics went on
    assert f.count("controller_stall", "block") == 1
