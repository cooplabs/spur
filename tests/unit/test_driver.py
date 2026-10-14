"""r1sim.driver + the engine's driver block: the board law vs mjlab's <position> actuators, the command / state delays,
the board's command timeout, non-finite commands (DESIGN.md §12.3 actuators; V2_INVENTORY App. E §4.2 items 1, 4)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from r1mc3.config import load_provenance
from r1sim import model as M
from r1sim.control import StandHold
from r1sim.driver import DriverConfig, driver_for_profile, driver_yaml
from r1sim.engine import LowCmd, LowState, Simulation

from tests.integration.toy import TOY_XML, toy_index, toy_model

ROOT = Path(__file__).resolve().parents[2]
TOY_A = [3, 9, 14]  # the toy's SDK slots


def _cmd(q: float = 0.2, kp: float = 50.0, kd: float = 1.0) -> LowCmd:
    c = LowCmd()
    c.q[TOY_A] = [q, -q, 2 * q]
    c.kp[TOY_A] = kp
    c.kd[TOY_A] = kd
    return c


def _toy(driver: DriverConfig | None = None, xml: str = TOY_XML) -> Simulation:
    m = mujoco.MjModel.from_xml_string(xml)
    return Simulation(m, toy_index(m), driver=driver)


# ---------------------------------------------------------------- config + provenance
def test_driver_config_validates() -> None:
    with pytest.raises(ValueError):
        DriverConfig(kind="pd")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DriverConfig(cmd_delay_s=-0.001)
    with pytest.raises(ValueError):
        DriverConfig(cmd_timeout_s=0.0)
    with pytest.raises(ValueError):
        DriverConfig(on_cmd_timeout="limp")  # type: ignore[arg-type]
    assert DriverConfig().plain and not DriverConfig(state_delay_s=0.002).plain


def test_driver_yaml_validates_and_real_equals_it() -> None:
    p = load_provenance(driver_yaml()).parameters
    for key, e in p.items():
        if e.status == "estimate" and isinstance(e.value, float):
            assert e.range is not None, f"{key}: a numeric estimate needs a range"
    real = DriverConfig.real()
    assert real.kind == "board"
    assert real.cmd_delay_s == p["cmd_delay_s"].value
    assert real.state_delay_s == p["state_delay_s"].value
    assert real.cmd_timeout_s == p["cmd_timeout_s"].value
    assert real.on_cmd_timeout == p["on_cmd_timeout"].value
    assert real.damp_kd == p["damp_kd"].value
    assert p["train_actuator_kind"].value == DriverConfig.train().kind == "position"


def test_driver_for_profile() -> None:
    assert driver_for_profile("train") == DriverConfig.train()
    assert driver_for_profile("real") == DriverConfig.real()
    with pytest.raises(ValueError):
        driver_for_profile("sim")


# ---------------------------------------------------------------- position (mjlab) vs board
TOY_POSITION_XML = TOY_XML.replace(
    '<motor name="j0" joint="j0_joint" ctrlrange="-60 60" ctrllimited="true"/>',
    '<position name="j0" joint="j0_joint" kp="30" kv="1" forcerange="-60 60" forcelimited="true"/>',
)


def test_position_kind_accepts_position_actuators_and_matches_board() -> None:
    """A model with <position> actuators (what the train agent builds) runs under both kinds; same forces."""
    assert "<position" in TOY_POSITION_XML

    def run(kind: str) -> bytes:
        sim = _toy(DriverConfig(kind=kind), TOY_POSITION_XML)  # type: ignore[arg-type]
        sim.write_cmd(_cmd())
        sim.step(300)
        return np.asarray(sim.d.qpos).tobytes()

    assert run("position") == run("board")


def test_position_kind_sets_mjlab_actuator_parameters() -> None:
    sim = _toy(DriverConfig.train())
    sim.step(2)
    assert np.all(sim.d.actuator_force == 0.0)  # zero torque before the first command
    c = _cmd(kp=40.0, kd=1.5)
    sim.write_cmd(c)
    sim.step(1)
    a = sim._b_a
    np.testing.assert_array_equal(sim.m.actuator_gainprm[a, 0], 40.0)
    np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 1], -40.0)
    np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 2], -1.5)
    np.testing.assert_array_equal(sim.d.ctrl[a], c.q[sim._b_idx])  # ctrl = q_des (no kp folded in)
    assert np.all(sim.m.actuator_ctrllimited[a] == 0) and np.all(sim.m.actuator_forcelimited[a] == 1)


def test_position_kind_refuses_velocity_and_feedforward() -> None:
    sim = _toy(DriverConfig.train())
    c = _cmd()
    c.tau_ff[3] = 0.5
    with pytest.raises(ValueError):
        sim.write_cmd(c)
    c = _cmd()
    c.dq[9] = 0.1

    def ctl(state: LowState) -> LowCmd:
        return c

    sim.add_controller(ctl, 0.002)
    with pytest.raises(ValueError):
        sim.step(1)


def test_position_equals_board_bit_identical_on_the_train_robot() -> None:
    """The train-profile robot, the same command sequence (stand hold, then a different pose at 1 s): the board law
    and mjlab's <position> actuators give bit-identical trajectories over 2 s (dq = tau_ff = 0)."""
    m = M.build_model("train", visual=False)
    hold = StandHold.for_model(m)
    other = StandHold(hold.pose + 0.05 * np.sign(np.arange(26) % 3 - 1))

    def run(driver: DriverConfig) -> list[bytes]:
        sim = Simulation(m, M.RobotIndex(m), driver=driver)
        sim.reset(M.stand_qpos(m, hold.pose))

        def ctl(state: LowState) -> LowCmd:
            return hold(state) if state.t < 1.0 else other(state)

        sim.add_controller(ctl, 0.02, "policy_rate")  # 50 Hz, decimation 4 as in training
        out = []
        for _ in range(10):
            sim.run_for(0.2)
            out.append(np.asarray(sim.d.qpos).tobytes())
        return out

    board, position = run(DriverConfig()), run(DriverConfig.train())
    assert board == position


# ---------------------------------------------------------------- command delay
def test_command_delay_applies_at_the_first_boundary_after_the_delay() -> None:
    sim = _toy(DriverConfig(cmd_delay_s=0.005))
    sim.step(10)  # t = 0.010
    c = _cmd()
    sim.write_cmd(c)
    a = sim._b_a
    for _ in range(5):  # boundaries 0.010 .. 0.014: not yet
        sim.step(1)
        assert np.all(sim.d.ctrl[a] == 0.0), sim.time
    sim.step(1)  # the step starting at 0.015 applies it
    assert np.all(sim.d.ctrl[a] != 0.0)


def test_delayed_commands_split_bulk_steps_exactly() -> None:
    """step(n) honours the delayed command's boundary: identical to stepping one at a time (also for commands from a
    lockstep controller, which travel the same delayed path)."""

    def run(bulk: bool) -> bytes:
        sim = _toy(DriverConfig(cmd_delay_s=0.0033))

        def ctl(state: LowState) -> LowCmd:
            return _cmd(q=0.1 + 0.4 * (state.t > 0.05))

        sim.add_controller(ctl, 0.01, "slow")
        if bulk:
            sim.step(200)
        else:
            for _ in range(200):
                sim.step(1)
        return np.asarray(sim.d.qpos).tobytes()

    assert run(True) == run(False)


def test_controller_commands_are_delayed_too() -> None:
    sim = _toy(DriverConfig(cmd_delay_s=0.003))
    seen: list[float] = []

    def ctl(state: LowState) -> LowCmd:
        return _cmd()

    sim.add_controller(ctl, 1.0, "once")  # called at t = 0 only (in 0.01 s)
    applied_at = None
    for _ in range(10):
        before = sim.applied_cmds
        t = sim.time
        sim.step(1)
        if sim.applied_cmds > before and applied_at is None:
            applied_at = t
        seen.append(t)
    assert applied_at == pytest.approx(0.003)


# ---------------------------------------------------------------- state delay
def test_state_delay_returns_the_state_published_delay_ago() -> None:
    ref = _toy()
    sim = _toy(DriverConfig(state_delay_s=0.004))
    for s in (ref, sim):
        s.write_cmd(_cmd())
    hist: dict[int, bytes] = {0: ref.read_state().q.tobytes()}
    for k in range(1, 40):
        ref.step(1)
        sim.step(1)
        hist[k] = ref.read_state().q.tobytes()
        st = sim.read_state()
        if k >= 4:
            assert st.t == pytest.approx(sim.time - 0.004)
            assert st.q.tobytes() == hist[k - 4]
        else:
            assert st.t == 0.0  # not that old yet: the oldest state


def test_lockstep_controllers_see_the_delayed_state() -> None:
    sim = _toy(DriverConfig(state_delay_s=0.006))
    seen: list[tuple[float, float]] = []

    def ctl(state: LowState) -> None:
        seen.append((sim.time, state.t))

    sim.add_controller(ctl, 0.002)
    sim.step(20)
    for now, t in seen[3:]:
        assert t == pytest.approx(now - 0.006)


# ---------------------------------------------------------------- command timeout
@pytest.mark.parametrize("action", ["hold", "damp", "zero_torque"])
def test_command_timeout(action: str) -> None:
    sim = _toy(DriverConfig(cmd_timeout_s=0.05, on_cmd_timeout=action, damp_kd=3.0))  # type: ignore[arg-type]
    c = _cmd()
    sim.write_cmd(c)
    sim.step(50)  # t = 0.050: the last boundary is 0.049 - not yet
    a = sim._b_a
    assert sim.timeouts == 0
    np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 1], -50.0)
    sim.step(1)  # the step starting at 0.050 = 0 + timeout
    assert sim.timeouts == 1
    if action == "hold":
        np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 1], -50.0)
        assert np.all(sim.d.ctrl[a] != 0.0)
    else:
        np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 1], 0.0)
        np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 2], -3.0 if action == "damp" else 0.0)
        np.testing.assert_array_equal(sim.d.ctrl[a], 0.0)
    sim.write_cmd(c)  # a new command ends it
    sim.step(1)
    np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 1], -50.0)
    sim.step(60)
    assert sim.timeouts == 2  # and it can fire again


def test_no_timeout_by_default_holds_forever() -> None:
    sim = _toy()
    sim.write_cmd(_cmd())
    sim.step(5000)
    assert sim.timeouts == 0 and np.all(sim.d.ctrl[sim._b_a] != 0.0)


# ---------------------------------------------------------------- non-finite commands
@pytest.mark.parametrize("driver", [None, DriverConfig(cmd_delay_s=0.002)], ids=["s1_path", "input_path"])
def test_non_finite_command_is_rejected_whole(driver: DriverConfig | None) -> None:
    sim = _toy(driver)
    good = _cmd()
    sim.write_cmd(good)
    sim.step(5)
    ctrl = sim.d.ctrl.copy()
    bad = _cmd(q=0.5)
    bad.q[9] = np.nan
    sim.write_cmd(bad)
    sim.step(5)
    assert sim.rejected_cmds == 1
    np.testing.assert_array_equal(sim.d.ctrl, ctrl)  # the last good command stays in force
    assert np.all(np.isfinite(sim.d.qpos))


def test_default_simulation_keeps_the_s1_path() -> None:
    m = toy_model()
    sim = Simulation(m, toy_index(m))
    assert sim._fast and sim.driver == DriverConfig()
    assert not Simulation(m, toy_index(m), driver=DriverConfig(cmd_delay_s=0.001))._fast
    assert ROOT.exists()


# ---------------------------------------------------------------- session 8 (B6): hand-only writes, per-joint damp kd
@pytest.mark.parametrize("driver", [None, DriverConfig(cmd_timeout_s=0.05, on_cmd_timeout="damp", damp_kd=3.0)],
                         ids=["s1_path", "input_path"])  # fmt: skip
def test_a_hand_only_write_never_reapplies_the_body_command(driver: DriverConfig | None) -> None:
    """The bridge forwards every Dex3 command together with its stored body command: a hand-only write (body=False)
    must leave the body's actuators exactly as the last BODY command set them (the base engine had no such write: the
    body was re-applied on every hand command - after the board's timeout that re-armed the stiff hold)."""
    sim = _toy(driver)
    a = sim._b_a
    sim.write_cmd(_cmd(q=0.2, kp=50.0))
    sim.step(1)
    other = _cmd(q=-0.4, kp=7.0)  # a different body in the hand-only write's arrays: must be ignored
    sim.write_cmd(other, body=False)
    sim.step(1)
    np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 1], -50.0)
    np.testing.assert_allclose(sim.d.ctrl[a], 50.0 * np.array([0.2, -0.2, 0.4]))


def test_hand_only_writes_do_not_refresh_the_body_timeout() -> None:
    """The board-hold test / B6: the body board releases cmd_timeout_s after its last BODY command however busy the
    hands are."""
    sim = _toy(DriverConfig(cmd_timeout_s=0.05, on_cmd_timeout="damp", damp_kd=3.0))
    a = sim._b_a
    sim.write_cmd(_cmd())
    for _ in range(10):  # hand traffic every 10 ms for 100 ms
        sim.step(10)
        sim.write_cmd(_cmd(q=0.3), body=False)
    assert sim.timeouts == 1  # released at 50 ms despite the hand writes
    np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 1], 0.0)
    np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 2], -3.0)
    sim.write_cmd(_cmd())  # a body command ends the release
    sim.step(1)
    np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 1], -50.0)


def test_per_joint_damp_kd() -> None:
    kd = tuple(float(k) for k in range(26))
    sim = _toy(DriverConfig(cmd_timeout_s=0.01, on_cmd_timeout="damp", damp_kd=kd))
    sim.write_cmd(_cmd())
    sim.step(20)
    assert sim.timeouts == 1
    np.testing.assert_array_equal(sim.m.actuator_biasprm[sim._b_a, 2], -np.array(TOY_A, dtype=float))
    with pytest.raises(ValueError):
        DriverConfig(damp_kd=(1.0, 2.0))
    with pytest.raises(ValueError):
        DriverConfig(damp_kd=-1.0)


def test_the_real_profile_board_is_the_hw1_board_hold_measurement() -> None:
    """models/driver.yaml after HW1 (session 8, B6): the release ~0.385 s after the last command, then damping."""
    real = DriverConfig.real()
    assert real.cmd_timeout_s is not None and 0.375 <= real.cmd_timeout_s <= 0.400
    assert real.on_cmd_timeout == "damp"
    assert isinstance(real.damp_kd, float) and 10.0 <= real.damp_kd <= 17.0
    p = load_provenance(driver_yaml()).parameters
    for k in ("cmd_timeout_s", "on_cmd_timeout", "damp_kd"):
        assert p[k].status == "measured" and "hw1_20261011" in p[k].source


def test_a_hand_only_write_is_judged_by_its_hands_only() -> None:
    """A non-finite body in a hand-only write's arrays (the bridge's stored body after a rejected NaN rt/lowcmd) must
    not get the hand command rejected; a non-finite HAND value still is."""
    sim = _toy(DriverConfig(cmd_delay_s=0.0, cmd_timeout_s=1.0))
    bad = _cmd()
    bad.q[TOY_A[0]] = float("nan")
    sim.write_cmd(bad, body=False)
    sim.step(1)
    assert sim.rejected_cmds == 0
    from r1sim.engine import HandCmd

    h = HandCmd()
    h.q[0] = float("inf")
    c = _cmd()
    c.hands = {"left": h}
    sim.write_cmd(c, body=False)
    sim.step(1)
    assert sim.rejected_cmds == 1
