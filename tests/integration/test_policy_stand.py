"""The balance policy (r1sim.control.PolicyController + r1mc3.policy.ReferencePolicy) keeps the R1 standing in
lockstep, from the policy's own default pose (Unitree's stand lock, joints.POSE_STAND) - the pose that FALLS under a
plain stiff PD (tests/integration/test_stand.py pins that) - in both profiles; reruns are bit-identical; the policy
fault hooks put NaN / out-of-range actions exactly where scheduled."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import mujoco
import numpy as np
import pytest
from r1mc3.policy import POLICY_DIR, ReferencePolicy
from r1sim import joints as J
from r1sim import world
from r1sim.control import PolicyController
from r1sim.engine import Simulation
from r1sim.faults import Fault, FaultSchedule

CONTROL_HZ = 500.0  # the real profile runs the controller at 500 Hz; the train profile's 5 ms step gives 200 Hz


def _tilt_deg(quat: np.ndarray) -> float:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(r[8], -1.0, 1.0))))


def _run(profile: str, seconds: float) -> tuple[Simulation, PolicyController, dict[str, float]]:
    w = world.build_world(profile, visual=False, with_scene=False)
    sim = Simulation(w.model, w.index)
    sim.reset(w.initial_qpos())  # the policy default pose, soles level on the floor
    ctrl = PolicyController(ReferencePolicy.load(POLICY_DIR))
    sim.add_controller(ctrl, 1.0 / CONTROL_HZ, "policy")
    ix = w.index
    z0 = float(sim.d.xpos[ix.pelvis_body][2])
    feet0 = np.array([sim.d.site_xpos[s][:2] for s in ix.foot_sites])
    worst_tilt = 0.0
    for _ in range(round(seconds / 0.25)):
        sim.run_for(0.25)
        worst_tilt = max(worst_tilt, _tilt_deg(sim.d.xquat[ix.pelvis_body]))
    feet = np.array([sim.d.site_xpos[s][:2] for s in ix.foot_sites])
    return (
        sim,
        ctrl,
        {
            "tilt_max": worst_tilt,
            "tilt_end": _tilt_deg(sim.d.xquat[ix.pelvis_body]),
            "dz_mm": 1000.0 * (float(sim.d.xpos[ix.pelvis_body][2]) - z0),
            "feet_mm": 1000.0 * float(np.linalg.norm(feet - feet0, axis=1).max()),
        },
    )


@pytest.mark.parametrize(("profile", "calls_per_policy_step"), [("train", 4), ("real", 10)])
def test_policy_stands_10s(profile: str, calls_per_policy_step: int) -> None:
    _, ctrl, r = _run(profile, 10.0)
    assert ctrl.policy.policy_steps == 500  # 50 Hz for 10 s
    assert ctrl.calls == 500 * calls_per_policy_step  # 200 Hz (train, 5 ms steps) / 500 Hz (real)
    # measured 10 Oct 2026: train tilt <= 0.42 deg, dz +0.6 mm, feet 3.9 mm; real <= 0.37 deg, +0.6 mm, 2.8 mm
    assert r["tilt_max"] < 3.0, r
    assert abs(r["dz_mm"]) < 20.0, r
    assert r["feet_mm"] < 10.0, r
    assert ctrl.max_abs_action < 3.0


def test_policy_stand_rerun_bit_identical() -> None:
    a, _, _ = _run("train", 2.0)
    b, _, _ = _run("train", 2.0)
    assert np.array_equal(a.d.qpos, b.d.qpos) and np.array_equal(a.d.qvel, b.d.qvel)


def _ctrl_with(faults: FaultSchedule) -> tuple[PolicyController, Callable[[float], np.ndarray]]:
    w = world.build_world("train", visual=False, with_scene=False)
    sim = Simulation(w.model, w.index)
    sim.reset(w.initial_qpos())
    s0 = sim.read_state()
    ctrl = PolicyController(ReferencePolicy.load(POLICY_DIR), faults=faults)

    def at(t: float) -> np.ndarray:
        cmd = ctrl(dataclasses.replace(s0, t=t))
        return cmd.q

    return ctrl, at


def test_policy_out_of_range_fault_on_schedule() -> None:
    f = FaultSchedule([Fault("policy_out_of_range", t_start=0.015, duration=0.02, params={"index": [0], "value": 5.0})])
    ctrl, at = _ctrl_with(f)
    pol = ctrl.policy
    q0 = at(0.0)  # policy step 1: clean
    assert abs(q0[0] - pol.default[0]) < 0.5
    q1 = at(0.02)  # policy step 2: in the window -> action[0] = 5 -> target = default + scale * 5
    assert q1[0] == pytest.approx(pol.default[0] + pol.scale[0] * 5.0)
    assert np.all(np.abs(q1[1:14] - pol.default[1:14]) < 1.0)  # the other actions untouched
    q2 = at(0.04)  # after the window: finite and back near the default pose
    assert np.all(np.isfinite(q2)) and abs(q2[0] - pol.default[0]) < 0.5
    assert f.count("policy_out_of_range", "applied") == 1


def test_policy_nan_fault_gives_nan_targets() -> None:
    """A NaN action reaches the command (the engine / board decides what to do with it). Like v2, the NaN then sits in
    the next observation (`actions` term), so the runner keeps producing NaN after the window until it is reset - a
    fact for the session-3 safety logic (the guard must not rely on the policy recovering by itself)."""
    f = FaultSchedule([Fault("policy_nan", t_start=0.015, duration=0.02)])
    ctrl, at = _ctrl_with(f)
    assert np.all(np.isfinite(at(0.0)))
    q = at(0.02)
    assert np.all(np.isnan(q[:14])) and np.all(np.isfinite(q[14:]))  # legs + waist NaN; arms / head held
    assert np.all(np.isnan(at(0.04)[:14]))  # poisoned through the `actions` observation (v2 semantics)
    ctrl.reset()
    assert np.all(np.isfinite(at(0.06)))
    assert f.count("policy_nan", "applied") == 1


def test_arm_pose_and_gains() -> None:
    w = world.build_world("train", visual=False, with_scene=False)
    sim = Simulation(w.model, w.index)
    sim.reset(w.initial_qpos())
    ctrl = PolicyController(ReferencePolicy.load(POLICY_DIR))
    cmd = ctrl(sim.read_state())
    assert np.allclose(cmd.q[J.ARMS], J.POSE_STAND[J.ARMS], atol=5e-4)  # the policy's default arm angles
    assert np.array_equal(cmd.kp[J.ARMS], [60, 60, 45, 45, 30, 60, 60, 45, 45, 30])
    assert np.array_equal(cmd.q[J.HEAD], [0.0, 0.0]) and np.array_equal(cmd.kp[J.HEAD], J.KP_STIFF[J.HEAD])
    assert set(cmd.hands) == {"left", "right"}
    with pytest.raises(ValueError):
        PolicyController(ReferencePolicy.load(POLICY_DIR), arm_pose=np.zeros(3))
