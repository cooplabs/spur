"""The DDS-bridge simulator's STAND (session 6, the hang agent): `DdsBridge(stance="hanging")` - the robot hanging by
its harness's top bar in the stand's padded cradle, the lower harness loop against the pole (a one-sided pitch stop),
feet off the floor (StandSpec: the user's photos IMG_8780 / IMG_8781 of 10 Oct 2026; every number an estimate until
HW1). Unitree's built-in on the stand: zero torque at power-on (publishing kp 0 / kd 0, or silent), or the stiff
stand lock.

Proved here: the hanging robot settles with the feet at the stated clearance and passes ON_STAND's gates (tilt <= 12
deg, every |dq| <= 0.5 rad/s); the pole carries the CoM's moment (without it the robot hangs ~30 deg pitched); leg
motions of the HW1 recordings' sizes (+-0.2 rad) keep the feet off the floor; the cradle's roll / yaw compliance is
finite (measured, reported); with REAL processes (r1rt, r1guard, r1d, the CLI) `r1 on-stand` blends into the stiff
hold and a leg follows an L0 setpoint; a SILENT zero-torque built-in is refused by today's gate (BUILTIN).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1sim import joints as J  # noqa: E402
from r1sim import stack  # noqa: E402
from r1sim.dds_bridge import DdsBridge, StandSpec, foot_geoms, lowest_point_z  # noqa: E402

GATE_TILT_DEG = 12.0  # ON_STAND's gates (RT_CORE_INTERFACES.md session 5; DESIGN D22)
GATE_DQ = 0.5


def _pole_force(br: DdsBridge) -> float:
    """The normal force between the pole and the harness's lower loop now (N)."""
    m, d = br.sim.m, br.sim.d
    pole = m.geom("r1sim_stand_pole").id
    f6 = np.zeros(6)
    tot = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if pole in (c.geom1, c.geom2):
            mujoco.mj_contactForce(m, d, i, f6)
            tot += abs(float(f6[0]))
    return tot


@pytest.mark.parametrize(
    "kw",
    [
        {},  # the default: the cradle + zero torque (publishing)
        {"builtin": "stand_lock"},
        {"builtin": "damp"},
        {"builtin": "damp_legs", "stand": StandSpec(attach="weld")},
    ],
    ids=["cradle-zero_torque", "cradle-stand_lock", "cradle-damp", "weld-damp_legs"],
)
def test_the_hanging_robot_settles_feet_off_the_floor_and_passes_the_gates(kw: dict[str, Any]) -> None:
    br = DdsBridge(domain=40, scene="none", stance="hanging", **kw)  # nothing started: the settle is lockstep
    h = br.hang
    spec = br.stand
    assert spec is not None
    assert abs(h["foot_z_min_m"] - spec.clearance_m) < 0.002, h
    assert h["tilt_deg"] < GATE_TILT_DEG and h["dq_max_at_settle"] < GATE_DQ, h
    br.sim.run_for(2.0)  # it stays where it hangs
    now = br.hang_state()
    assert now["foot_z_min"] > spec.clearance_m - 0.01, now
    assert abs(now["tilt_deg"] - h["tilt_deg"]) < 0.5, (now, h)
    if spec.attach == "cradle":
        assert _pole_force(br) > 50.0  # the CoM's moment about the bar rests on the pole


def test_without_the_pole_the_robot_hangs_pitched() -> None:
    """The pole is what holds the pitch: 0.3 m behind its place the robot swings its CoM under the bar."""
    far = DdsBridge(domain=40, scene="none", stance="hanging", stand=StandSpec(pole_gap_m=0.3))
    near = DdsBridge(domain=40, scene="none", stance="hanging")
    assert far.hang["pitch_deg"] > near.hang["pitch_deg"] + 20.0, (far.hang, near.hang)


def test_leg_motions_of_the_recording_sizes_keep_the_feet_off_the_floor() -> None:
    """From the hanging pose every leg joint +-0.2 rad (the stiction ramp's amplitude; steps 0.05, chirps 0.03),
    clipped to the joint's range: the lowest foot point stays above the floor (kinematics)."""
    br = DdsBridge(domain=40, scene="none", stance="hanging")
    m = br.sim.m
    d = mujoco.MjData(m)
    q0 = br.sim.d.qpos.copy()
    feet = foot_geoms(m)
    worst = np.inf
    for name in J.JOINTS[:12]:
        jid = m.joint(f"{name}_joint").id
        a = int(m.jnt_qposadr[jid])
        lo, hi = m.jnt_range[jid]
        for dq in (-0.2, 0.2):
            d.qpos[:] = q0
            d.qpos[a] = np.clip(q0[a] + dq, lo, hi)
            mujoco.mj_kinematics(m, d)
            worst = min(worst, lowest_point_z(m, d, feet))
    assert worst > 0.02, worst


def test_the_cradle_is_compliant_in_roll_and_yaw() -> None:
    """A 5 N·m torque on the torso about x (roll) and z (yaw): the cradle yields a little and holds (stiffness
    reported; MuJoCo's default constraint softness, unmeasured - until HW1)."""
    out = {}
    for axis, k in (("roll", 0), ("yaw", 2)):
        br = DdsBridge(domain=40, scene="none", stance="hanging", builtin="stand_lock")
        a0 = br.hang_state()
        tid = br.sim.m.body("torso_link").id
        br.sim.d.xfrc_applied[tid, 3 + k] = 5.0
        br.sim.run_for(1.5)
        a1 = br.hang_state()
        dang = abs(a1[f"{axis}_deg"] - a0[f"{axis}_deg"])
        out[axis] = 5.0 / np.radians(max(dang, 1e-6))
        assert 0.005 < dang < 5.0, (axis, dang)
    print("cradle stiffness (N m / rad):", {k: round(v) for k, v in out.items()})


@pytest.mark.realtime
@pytest.mark.slow
def test_on_stand_with_real_processes_on_the_hanging_robot(tmp_path: Path) -> None:
    if not stack.available():
        pytest.skip("needs the DDS build of r1rt / r1guard (scripts/ci.sh dds)")
    with stack.ProcStack(domain=41, workdir=tmp_path, bridge={"scene": "none", "stance": "hanging"}) as st:
        br = st.bridge
        st.wait_mode("monitor", 10)
        time.sleep(0.5)
        t0 = time.monotonic()
        r = st.cli("on-stand")
        assert r["ok"] and r["_exit"] == 0, r
        st.wait_mode("on_stand", 15)
        time.sleep(1.2)
        labels = {h: v["label"] for h, v in br.writers().items()}
        rt = [a for a in list(br.applied) if a.wall > t0 and labels.get(a.writer) == "r1rt"]
        assert rt and min(a.min_leg_kp for a in rt if a.wall > rt[-1].wall - 1.0) >= 199.0  # the stiff hold
        hold = st.cli("hold", "leg_left")
        q = list(hold["data"]["q"])
        q[3] += 0.2  # the left knee +0.2 rad (the ramp's amplitude)
        for _ in range(8):
            j = st.cli("joint", "leg_left", "--ttl", "0.3", "--", ",".join(f"{v:.4f}" for v in q))
            assert j["ok"], j
        time.sleep(0.4)
        assert abs(float(br.sim.read_state().q[3]) - q[3]) < 0.04
        tr = [t for t in list(br.truth) if t.wall > t0]
        assert min(t.foot_z_min for t in tr) > 0.05  # never near the floor
        # session 8: relative to the hang (the default stand HW1_STAND leans 5.1 deg like the real robot's 5.2-5.5;
        # the session-6 stand hung at 1.65 under this absolute 5.0)
        assert max(t.tilt_deg for t in tr) < br.hang["tilt_deg"] + 3.5


@pytest.mark.realtime
@pytest.mark.slow
def test_a_silent_zero_torque_builtin_is_refused_by_todays_gate(tmp_path: Path) -> None:
    """Whether Unitree publishes rt/lowcmd in zero torque is unknown: if it does not, ON_STAND's gate (a FRESH
    built-in command) refuses - R1RT_TAKEOVER_BUILTIN (9). The gate is the core's (not changed here)."""
    if not stack.available():
        pytest.skip("needs the DDS build of r1rt / r1guard (scripts/ci.sh dds)")
    bridge = {"scene": "none", "stance": "hanging", "builtin": "zero_torque_silent"}
    with stack.ProcStack(domain=42, workdir=tmp_path, bridge=bridge) as st:
        st.wait_mode("monitor", 10)
        time.sleep(0.5)
        r = st.cli("on-stand")
        assert not r["ok"] and r["details"]["refusal"] == 9, r
        assert st.mode() == "monitor"
        assert not st.bridge.writers()  # nothing on rt/lowcmd at all


# ---------------------------------------------------------------- session 8 (board agent): the HW1 facts in the bridge
def test_the_hanging_stand_lock_is_unitrees_fixed_preset() -> None:
    """HW1 S1.0 (#25): Unitree's stand lock commands a FIXED preset at the stand gains and the robot moves INTO it -
    the hanging bridge's stand_lock built-in does the same (before session 8 it held the CoM-centred stand: knee 0.455,
    hip roll +-0.025 instead of 0.41 / +-0.04)."""
    from r1sim.dds_bridge import UNITREE_STAND_LOCK_POSE

    br = DdsBridge(domain=40, scene="none", stance="hanging", builtin="stand_lock")
    cmd = br.builtin_cmd
    assert cmd is not None
    np.testing.assert_array_equal(cmd.q, np.asarray(UNITREE_STAND_LOCK_POSE))
    np.testing.assert_array_equal(cmd.kp, J.KP_STIFF)
    np.testing.assert_array_equal(cmd.kd, J.KD_STIFF)
    br.sim.add_controller(lambda _st: cmd, 0.002, "unitree")  # Unitree publishes the lock all the time (lockstep here)
    br.sim.run_for(1.0)
    q = br.sim.read_state().truth.get("q", br.sim.read_state().q)
    legs = np.asarray(q)[J.LEGS]
    assert float(np.max(np.abs(legs - np.asarray(UNITREE_STAND_LOCK_POSE)[J.LEGS]))) < 0.02  # HW1: within 0.017
    assert br.hang["tilt_deg"] < GATE_TILT_DEG


def test_the_hw1_stand_hangs_at_the_measured_attitude() -> None:
    """HW1: the IMU read roll 1.8 / pitch -4.9 deg on the stand (the board-hold test's lowstate); the default StandSpec
    hangs the model at IMU pitch +1.7 deg (6.6 deg off: half the hips' gravity load); dds_bridge.HW1_STAND reproduces
    the pitch."""
    from r1sim.dds_bridge import HW1_STAND, _rpy

    br = DdsBridge(domain=40, scene="none", stance="hanging", builtin="stand_lock", stand=HW1_STAND, sensors=False)
    pitch = float(np.degrees(_rpy(br.sim.read_state().quat))[1])
    assert pitch == pytest.approx(-4.94, abs=1.0)
    br.sim.run_for(2.0)
    assert float(np.degrees(_rpy(br.sim.read_state().quat))[1]) == pytest.approx(pitch, abs=0.3)  # it stays


def test_hand_commands_do_not_keep_the_board_from_releasing() -> None:
    """B6 in the bridge: with the board model on (driver=DriverConfig.real()) the body releases cmd_timeout_s after
    the last rt/lowcmd even while Dex3 commands keep arriving (before session 8 every hand command re-wrote the body
    command into the engine: the body would never have released while the hands were driven)."""
    from r1sim.driver import DriverConfig

    drv = DriverConfig.real()
    assert drv.cmd_timeout_s is not None
    br = DdsBridge(domain=40, scene="none", stance="hanging", builtin="stand_lock", sensors=False, driver=drv)
    sim = br.sim
    br._compose_and_write()  # the last rt/lowcmd (the built-in's stand lock) at t0
    t0 = sim.time
    while sim.time < t0 + drv.cmd_timeout_s + 0.1:  # Dex3 commands every 10 ms (r1hands' 100 Hz)
        sim.run_for(0.01)
        br._compose_and_write(body=False)
    assert sim.timeouts == 1
    a = sim._b_a
    np.testing.assert_array_equal(sim.m.actuator_biasprm[a, 1], 0.0)  # kp 0: released
    np.testing.assert_allclose(sim.m.actuator_biasprm[a, 2], -np.asarray(drv.damp_kd))  # damping
    assert br.log()["board"]["timeouts"] == 1
