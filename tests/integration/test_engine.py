"""r1sim.engine: physics independent of the controller (DESIGN.md §12.1, §12.4; R-k; V2_INVENTORY.md §6).

Numbers measured here (real-time factors, stall advance) are printed and also written to out/s1_engine_report.json
(gitignored) for the session report.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from r1sim import model as M
from r1sim.control import StandHold
from r1sim.engine import ControllerThread, LowCmd, LowState, Simulation

from tests.integration.toy import toy_index, toy_model

REPORT = Path(__file__).resolve().parents[2] / "out" / "s1_engine_report.json"


def _report(key: str, value: Any) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    data[key] = value
    REPORT.write_text(json.dumps(data, indent=1, sort_keys=True))
    print(f"[s1 engine] {key}: {value}")


def _tilt_deg(quat: np.ndarray) -> float:
    x, y = float(quat[1]), float(quat[2])
    return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


def _robot_sim(profile: str) -> tuple[Simulation, StandHold]:
    m = M.build_model(profile, visual=False)
    ctrl = StandHold.for_model(m)
    sim = Simulation(m, M.RobotIndex(m))
    sim.reset(M.stand_qpos(m, ctrl.pose))
    return sim, ctrl


# ---------------------------------------------------------------- the driver block (toy model, fast)
def _toy_cmd() -> LowCmd:
    c = LowCmd()
    c.q[[3, 9, 14]] = [0.1, -0.2, 0.5]
    c.dq[[3, 9, 14]] = [0.0, 0.3, -0.1]
    c.kp[[3, 9, 14]] = [80.0, 60.0, 10.0]
    c.kd[[3, 9, 14]] = [2.0, 1.5, 0.5]
    c.tau_ff[[3, 9, 14]] = [0.5, -0.25, 0.1]
    return c


def test_zero_torque_before_any_command() -> None:
    sim = Simulation(toy_model(), toy_index(toy_model()))
    sim.step(5)
    assert np.all(sim.read_state().tau_est == 0.0)
    assert np.all(sim.d.actuator_force == 0.0)


def test_driver_applies_the_board_pd_law_every_step() -> None:
    m = toy_model()
    ix = toy_index(m)
    sim = Simulation(m, ix)
    sim.step(20)
    cmd = _toy_cmd()
    sim.write_cmd(cmd)
    for _ in range(50):  # the law holds at every step, evaluated on the state the step starts from
        before = sim.read_state()
        sim.step(1)
        i = [3, 9, 14]
        expect = cmd.kp[i] * (cmd.q[i] - before.q[i]) + cmd.kd[i] * (cmd.dq[i] - before.dq[i]) + cmd.tau_ff[i]
        expect = np.clip(expect, -ix.tau_max[i], ix.tau_max[i])
        np.testing.assert_allclose(sim.read_state().tau_est[i], expect, rtol=0, atol=1e-9)


def test_driver_clips_to_tau_max() -> None:
    m = toy_model()
    ix = toy_index(m)
    sim = Simulation(m, ix)
    c = LowCmd()
    c.kp[14], c.q[14] = 1e4, 3.0  # asks for far more than the 5 N m motor
    sim.write_cmd(c)
    sim.step(1)
    assert sim.read_state().tau_est[14] == pytest.approx(ix.tau_max[14])


def test_absent_joints_read_zero_and_ignore_commands() -> None:
    m = toy_model()
    sim = Simulation(m, toy_index(m))
    c = LowCmd()
    c.kp[:], c.q[:] = 100.0, 1.0
    sim.write_cmd(c)
    sim.step(10)
    s = sim.read_state()
    absent = np.setdiff1d(np.arange(26), [3, 9, 14])
    assert np.all(s.q[absent] == 0.0) and np.all(s.tau_est[absent] == 0.0)


def test_lockstep_controllers_run_at_their_rates_in_order() -> None:
    m = toy_model()
    sim = Simulation(m, toy_index(m))
    order: list[str] = []

    def ctl_a(state: LowState) -> None:
        order.append("a")

    def ctl_b(state: LowState) -> None:
        order.append("b")

    sim.add_controller(ctl_a, 0.002, "a")
    sim.add_controller(ctl_b, 0.005, "b")
    sim.step(1000)  # 1 s at dt 1 ms
    assert sim.controller_calls() == {"a": 500, "b": 200}
    assert order[:2] == ["a", "b"]  # both due at step 0: registration order
    assert sim.time == pytest.approx(1.0)
    assert sim.read_state().tick == 1000


def test_physics_never_waits_for_a_lockstep_controller_returning_none() -> None:
    m = toy_model()
    sim = Simulation(m, toy_index(m))
    sim.write_cmd(_toy_cmd())

    def silent(state: LowState) -> None:
        return None

    sim.add_controller(silent, 0.002, "silent")
    sim.step(100)
    assert sim.applied_cmds == 1  # the one mailbox command, held (zero-order hold)
    assert np.any(sim.read_state().tau_est != 0.0)


# ---------------------------------------------------------------- determinism (the robot)
@pytest.mark.parametrize("profile", ["real", "train"])
def test_lockstep_reruns_are_bit_identical(profile: str) -> None:
    def run() -> list[bytes]:
        sim, ctrl = _robot_sim(profile)
        sim.add_controller(ctrl, 0.002, "stand")
        sim.run_for(2.0)
        d = sim.d
        return [np.asarray(a).tobytes() for a in (d.qpos, d.qvel, d.act, d.sensordata, d.actuator_force)]

    a, b = run(), run()
    assert a == b


def test_split_stepping_equals_mj_step() -> None:
    """The engine's mj_step1 / mj_step2 split is bit-identical to plain mj_step (MuJoCo 3.5.0, implicitfast)."""
    m = M.build_model("real", visual=False)
    ctrl = StandHold.for_model(m)
    sim = Simulation(m, M.RobotIndex(m))
    q0 = M.stand_qpos(m, ctrl.pose)
    sim.reset(q0)
    sim.write_cmd(ctrl.command)
    sim.step(300)
    ref_m = sim.m  # the same driver-configured model, stepped plainly with the same controls
    ref_d = mujoco.MjData(ref_m)
    ref_d.qpos[:] = q0
    ref_d.ctrl[:] = sim.d.ctrl
    mujoco.mj_step(ref_m, ref_d, 300)
    assert np.asarray(ref_d.qpos).tobytes() == np.asarray(sim.d.qpos).tobytes()


# ---------------------------------------------------------------- real time: physics independent of the controller
@pytest.mark.realtime
@pytest.mark.parametrize("busy", [False, True], ids=["blocked", "gil_hog"])
def test_physics_keeps_running_while_the_controller_is_stalled(busy: bool) -> None:
    """The 9 Oct 2026 stand-in: the controller thread stops for 1 s (blocked in a call; or spinning in Python, holding
    the GIL). Physics must keep advancing at wall-clock pace, holding the last command like the board did."""
    sim, ctrl = _robot_sim("real")
    sim.write_cmd(ctrl.command)  # the command in force before the controller thread starts
    s0 = sim.read_state()
    th = ControllerThread(sim, ctrl, 500.0)
    sim.start_realtime()
    th.start()
    try:
        time.sleep(0.5)
        th.stall(1.0, busy=busy)
        time.sleep(1.6)
    finally:
        th.stop()
        sim.stop()
    assert len(th.stalls) == 1
    st = th.stalls[0]
    wall = st.wall_end - st.wall_start
    advanced = st.sim_end - st.sim_start
    writes_in_stall = [w for w, _ in sim.write_log if st.wall_start < w < st.wall_end]
    s = sim.read_state()
    _report(
        f"stall_{'gil_hog' if busy else 'blocked'}",
        {
            "stall_wall_s": round(wall, 4),
            "sim_advanced_s": round(advanced, 4),
            "writes_during_stall": len(writes_in_stall),
            "max_chunk_steps": sim.rt.max_chunk_steps,
            "achieved_rtf": round(sim.rt.achieved_rtf, 4),
            "tilt_change_deg": round(_tilt_deg(s.base_quat) - _tilt_deg(s0.base_quat), 3),
            "pelvis_dz_m": round(float(s.base_pos[2] - s0.base_pos[2]), 4),
        },
    )
    assert advanced >= 0.9 * wall
    assert writes_in_stall == []
    assert abs(_tilt_deg(s.base_quat) - _tilt_deg(s0.base_quat)) < 5.0
    assert abs(float(s.base_pos[2] - s0.base_pos[2])) < 0.03


@pytest.mark.realtime
def test_the_stall_criterion_fails_on_a_v2_style_coupling() -> None:
    """Negative control for the test above. v2 stepped physics inside its control tick (r1mc2 runtime.py:270-271:
    `Runtime.step_once` calls `backend.step()`). Coupled that way, the same 1 s stall freezes physics, so the criterion
    used above (`advanced >= 0.9 x wall`) fails: the stall test discriminates v3's design from v2's."""
    sim, ctrl = _robot_sim("real")
    stall, stop = threading.Event(), threading.Event()
    marks: dict[str, float] = {}

    def v2_style_loop() -> None:  # one thread: controller tick, then physics, paced at 500 Hz
        period = 0.002
        k = max(1, round(period / sim.dt))
        nxt = time.perf_counter()
        while not stop.is_set():
            if stall.is_set():
                marks["w0"], marks["s0"] = time.perf_counter(), sim.time
                time.sleep(1.0)  # the same stall as above, now in the thread that also steps physics
                marks["w1"], marks["s1"] = time.perf_counter(), sim.time
                stall.clear()
            sim.write_cmd(ctrl(sim.read_state()))
            sim.step(k)
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    th = threading.Thread(target=v2_style_loop, daemon=True)
    th.start()
    try:
        time.sleep(0.3)
        stall.set()
        time.sleep(1.5)
    finally:
        stop.set()
        th.join(5.0)
    wall = marks["w1"] - marks["w0"]
    advanced = marks["s1"] - marks["s0"]
    _report("stall_v2_style_negative_control", {"stall_wall_s": round(wall, 4), "sim_advanced_s": round(advanced, 4)})
    assert wall >= 0.95
    assert advanced < 0.9 * wall  # the v3 criterion fails here
    assert advanced == 0.0  # physics did not move at all while the coupled thread was stalled


@pytest.mark.realtime
def test_realtime_mode_tracks_the_wall_clock() -> None:
    sim, ctrl = _robot_sim("real")
    sim.write_cmd(ctrl.command)
    th = ControllerThread(sim, ctrl, 500.0)
    sim.start_realtime()
    th.start()
    try:
        time.sleep(2.0)
    finally:
        th.stop()
        sim.stop()
    rt = sim.rt
    _report(
        "realtime_real_profile",
        {
            "wall_s": round(rt.wall_s, 3),
            "sim_s": round(rt.sim_s, 3),
            "achieved_rtf": round(rt.achieved_rtf, 4),
            "max_lag_ms": round(rt.max_lag_s * 1e3, 3),
            "max_chunk_steps": rt.max_chunk_steps,
            "overruns": rt.overruns,
            "rephases": rt.rephases,
            "controller_ticks": th.ticks,
            "controller_late": th.late,
        },
    )
    assert rt.wall_s > 1.9
    assert abs(rt.achieved_rtf - 1.0) < 0.05
    assert th.ticks > 0.8 * 500 * 2.0


@pytest.mark.parametrize("profile", ["real", "train"])
def test_lockstep_real_time_factor(profile: str) -> None:
    """As fast as possible, headless, the stand hold in the loop at 500 Hz (train: every 5 ms physics step)."""
    sim, ctrl = _robot_sim(profile)
    sim.add_controller(ctrl, 0.002, "stand")
    sim.run_for(0.2)  # warm-up
    t0 = time.perf_counter()
    sim.run_for(3.0)
    rtf = 3.0 / (time.perf_counter() - t0)
    _report(
        f"lockstep_rtf_{profile}", {"rtf": round(rtf, 2), "dt": sim.dt, "controller_hz": round(1 / max(sim.dt, 0.002))}
    )
    assert rtf > 1.0


# ---------------------------------------------------------------- session 2: the input path and the sensor hook
class _FakeSensors:
    """Records how the engine drives a sensor model (the SensorHook contract) and offsets q[3] by 1000."""

    def __init__(self) -> None:
        self.bound: list[Any] = []
        self.resets: list[int | None] = []
        self.calls: list[tuple[float, float, int]] = []  # (state t, dt, state seq)

    def bind(self, m: Any, joint_map: Any) -> None:
        self.bound.append((m, joint_map))

    def reset(self, seed: int | None = None) -> None:
        self.resets.append(seed)

    def process(self, state: LowState, m: Any, d: Any, dt: float, faults: Any) -> LowState:
        assert float(d.time) == state.t  # called right after the step, on the current data
        self.calls.append((state.t, dt, state.seq))
        q = state.q.copy()
        q[3] += 1000.0
        return dataclasses.replace(state, q=q, truth={"q": state.q})


def test_sensor_hook_contract() -> None:
    m = toy_model()
    fake = _FakeSensors()
    sim = Simulation(m, toy_index(m), seed=5, sensors=fake)
    assert len(fake.bound) == 1 and fake.bound[0][0] is sim.m  # bound once, to the engine's own model copy
    assert fake.resets == [5]  # the reset in __init__
    seen: list[float] = []

    def ctl(state: LowState) -> None:
        seen.append(float(state.q[3]))

    sim.add_controller(ctl, 0.002, "c")
    sim.step(100)
    seqs = [c[2] for c in fake.calls]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))  # once per published state, in order
    assert len(fake.calls) == sim._snap_seq
    ts = [c[0] for c in fake.calls]
    assert fake.calls[0][1] == 0.0 and all(
        dt == pytest.approx(t1 - t0) for (t0, _, _), (t1, dt, _) in zip(fake.calls, fake.calls[1:], strict=False)
    )
    assert ts[-1] == pytest.approx(sim.time)
    assert all(v > 500.0 for v in seen)  # controllers get what process returned
    s = sim.read_state()
    assert s.q[3] > 500.0 and s.truth["q"][3] < 500.0  # and so does read_state
    sim.reset()
    assert fake.resets == [5, 5] and fake.calls[-1][1] == 0.0


def test_input_path_with_nothing_active_is_bit_identical_to_the_s1_path() -> None:
    """The session-2 slow path (queue, ring, per-state processing) with an empty fault schedule gives exactly the S1
    trajectory: same commands, same order, same physics."""
    from r1sim.faults import FaultSchedule

    def run(faults: FaultSchedule | None) -> list[bytes]:
        m = M.build_model("train", visual=False)
        ctrl = StandHold.for_model(m)
        sim = Simulation(m, M.RobotIndex(m), faults=faults)
        assert sim._fast == (faults is None)
        sim.reset(M.stand_qpos(m, ctrl.pose))
        sim.write_cmd(ctrl.command)
        sim.add_controller(ctrl, 0.02, "stand")
        sim.run_for(1.0)
        s = sim.read_state()
        return [np.asarray(a).tobytes() for a in (sim.d.qpos, sim.d.qvel, s.q, s.dq, s.quat, s.gyro, s.tau_est)]

    assert run(None) == run(FaultSchedule())
