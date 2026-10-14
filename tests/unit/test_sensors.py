"""r1sim.sensors: the sensor models (encoders, IMU, tau_est, pads, thermal) on top of the exact simulated state.

The engine wires `SensorModel` into `Simulation` (session 2, engine agent); these tests call `process` directly on
states from the S1 engine, as the engine will: once per published state, after mj_step1."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import mujoco
import numpy as np
import pytest
from r1mc3.config import load_provenance
from r1sim import dex3
from r1sim import model as M
from r1sim.control import StandHold
from r1sim.engine import LowState, Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.joints import JOINTS
from r1sim.sensors import (
    CELL_PAD,
    N_CELLS,
    SENSORS_YAML,
    THERMAL_GROUPS,
    SensorConfig,
    SensorModel,
    body_thermal_arrays,
)

OFF = SensorConfig.off()
FIELDS = ("q", "dq", "tau_est", "quat", "gyro", "acc")


@pytest.fixture(scope="module")
def real() -> tuple[Any, Any, M.RobotIndex, LowState]:
    """The real profile standing 0.2 s under the stand hold: (model, data, index, the published state)."""
    m = M.build_model("real", visual=False)
    ctrl = StandHold.for_model(m)
    sim = Simulation(m, M.RobotIndex(m))
    sim.reset(M.stand_qpos(m, ctrl.pose))
    sim.add_controller(ctrl, 0.002, "stand")
    sim.run_for(0.2)
    return sim.m, sim.d, M.RobotIndex(sim.m), sim.read_state()


def _model(cfg: SensorConfig, real: tuple[Any, Any, M.RobotIndex, LowState], seed: int = 0) -> SensorModel:
    sm = SensorModel(cfg, seed=seed)
    sm.bind(real[0], real[2])
    return sm


def _bits(st: LowState) -> bytes:
    parts = [np.asarray(getattr(st, f)).tobytes() for f in (*FIELDS, "motor_temp")]
    for dct in (st.hand_q, st.hand_dq, st.hand_tau, st.hand_temp, st.hand_pressure):
        parts += [np.asarray(dct[s]).tobytes() for s in sorted(dct)]
    return b"".join(parts)


# ---------------------------------------------------------------- configuration
def test_yaml_validates_and_equals_the_defaults() -> None:
    prov = load_provenance(SENSORS_YAML)
    assert SensorConfig.from_yaml() == SensorConfig()
    assert set(prov.parameters) == {f.name for f in dataclasses.fields(SensorConfig)}
    no_range = {"hand_tau_thumb0_raw"}  # a stuck sample value, not a quantity with a spread
    for k, e in prov.parameters.items():
        if e.status in ("estimate", "reported") and k not in no_range and isinstance(e.value, float):
            assert e.range is not None, f"{k}: an estimate needs a range"


# ---------------------------------------------------------------- off / determinism / purity
def test_off_is_an_exact_passthrough(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    m, d, _, s = real
    out = _model(OFF, real).process(s, m, d, 0.0, None)
    for f in FIELDS:
        assert np.asarray(getattr(out, f)).tobytes() == np.asarray(getattr(s, f)).tobytes(), f
    for f in ("hand_q", "hand_dq", "hand_tau"):
        for side in dex3.SIDES:
            assert getattr(out, f)[side].tobytes() == getattr(s, f)[side].tobytes(), (f, side)
    assert set(out.hand_pressure) == set(dex3.SIDES) and set(out.hand_temp) == set(dex3.SIDES)


def test_same_seed_reproduces_and_another_seed_differs(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    m, d, _, s = real

    def run(seed: int) -> list[bytes]:
        sm = _model(SensorConfig(), real, seed)
        return [_bits(sm.process(dataclasses.replace(s, t=s.t + 0.002 * i), m, d, 0.002, None)) for i in range(20)]

    a, b, c = run(3), run(3), run(4)
    assert a == b
    assert all(x != y for x, y in zip(a, c, strict=True))
    sm = _model(SensorConfig(), real, 3)  # reset(seed) restarts the streams and the biases
    first = _bits(sm.process(s, m, d, 0.002, None))
    sm.process(s, m, d, 0.002, None)
    sm.reset(3)
    assert _bits(sm.process(s, m, d, 0.002, None)) == first


def test_process_returns_a_new_state_fills_truth_and_leaves_the_input(
    real: tuple[Any, Any, M.RobotIndex, LowState],
) -> None:
    m, d, _, s = real
    before = copy.deepcopy(s)
    out = _model(SensorConfig(), real, 1).process(s, m, d, 0.002, None)
    assert out is not s
    assert _bits(s) == _bits(before) and s.truth == before.truth
    for k in (*FIELDS, "motor_temp", "hand_q", "hand_dq", "hand_tau", "hand_temp", "hand_pressure", "hand_pad_force"):
        assert k in out.truth, k
    assert out.truth["q"].tobytes() == s.q.tobytes()
    assert not np.array_equal(out.q, s.q)


# ---------------------------------------------------------------- encoders / IMU
def test_noise_statistics(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    m, d, ix, s = real
    cfg = OFF.replace(q_noise_std=0.01, dq_noise_std=0.05, gyro_noise_std=0.02, acc_noise_std=0.1)
    sm = _model(cfg, real, 5)
    outs = [sm.process(s, m, d, 0.002, None) for _ in range(1500)]
    pres = ix.present
    for f, std in (("q", 0.01), ("dq", 0.05), ("gyro", 0.02), ("acc", 0.1)):
        e = np.array([getattr(o, f) - getattr(s, f) for o in outs])
        if f in ("q", "dq"):
            e = e[:, pres]
        assert abs(float(e.std()) / std - 1.0) < 0.1, (f, float(e.std()))
        assert abs(float(e.mean())) < 0.1 * std, f
    head = ~pres
    assert not head.any() or np.all(np.array([o.q[head] for o in outs]) == 0.0)


def test_quantisation_step(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    m, d, ix, s = real
    out = _model(OFF.replace(q_quant=0.001, hand_q_quant=0.002), real).process(s, m, d, 0.0, None)
    k = out.q[ix.present] / 0.001
    assert np.allclose(k, np.round(k), atol=1e-9) and np.max(np.abs(out.q - s.q)) <= 0.0005 + 1e-12
    kh = out.hand_q["left"] / 0.002
    assert np.allclose(kh, np.round(kh), atol=1e-9)


def test_gyro_bias_is_constant_per_seed(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    m, d, _, s = real
    cfg = OFF.replace(gyro_bias_std=0.01, acc_bias_std=0.05)
    sm = _model(cfg, real, 7)
    b1 = sm.process(s, m, d, 0.002, None).gyro - s.gyro
    b2 = sm.process(s, m, d, 0.002, None).gyro - s.gyro
    assert np.array_equal(b1, b2) and np.linalg.norm(b1) > 0
    b3 = _model(cfg, real, 8).process(s, m, d, 0.002, None).gyro - s.gyro
    assert not np.array_equal(b1, b3)


def test_quaternion_stays_unit_and_close(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    m, d, _, s = real
    out = _model(SensorConfig(), real, 2).process(s, m, d, 0.002, None)
    assert abs(float(np.linalg.norm(out.quat)) - 1.0) < 1e-12
    ang = 2.0 * np.arccos(min(1.0, abs(float(np.dot(out.quat, s.quat)))))
    assert 0.0 < ang < 0.05


# ---------------------------------------------------------------- tau_est
def test_tau_scale_and_the_finger_2x(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    m, d, _, s = real
    left = dataclasses.replace(s, hand_tau={k: np.full(7, 0.3) for k in s.hand_tau})
    out = _model(OFF.replace(tau_scale=1.1, hand_tau_gain=2.0), real).process(left, m, d, 0.0, None)
    assert np.allclose(out.tau_est, 1.1 * s.tau_est, rtol=0, atol=1e-12)
    assert np.allclose(out.hand_tau["left"], 0.6, atol=1e-12)
    assert out.truth["hand_tau"]["left"][0] == 0.3


def test_hand_tau_raw_units(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    m, d, _, s = real
    st = dataclasses.replace(s, hand_tau={k: np.full(7, 0.3) for k in s.hand_tau})
    out = _model(OFF.replace(hand_tau_gain=2.0, hand_tau_units="raw"), real).process(st, m, d, 0.0, None)
    r = out.hand_tau["right"]
    assert r[0] == 8498.6  # thumb_0: the stuck sample (not calibrated)
    steps = r[1:] / 418.65
    assert np.allclose(steps, np.round(steps), atol=1e-9)
    assert np.allclose(r[1:] * 1e-6, 0.6, atol=418.65e-6)


def test_junk_values_at_the_configured_rate(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    m, d, ix, s = real
    sm = _model(OFF.replace(tau_junk_p=0.05, hand_tau_junk_p=0.05), real, 9)
    outs = [sm.process(s, m, d, 0.002, None) for _ in range(400)]
    tau = np.array([o.tau_est[ix.present] for o in outs])
    junk = np.abs(tau) == 1674.594
    assert abs(float(junk.mean()) - 0.05) < 0.01
    assert np.array_equal(tau[~junk], np.broadcast_to(s.tau_est[ix.present], tau.shape)[~junk])
    hj = np.array([np.abs(o.hand_tau["left"]) == 1674.594 for o in outs])
    assert abs(float(hj.mean()) - 0.05) < 0.02


# ---------------------------------------------------------------- pads
def _probe_world() -> tuple[Any, Any, M.RobotIndex]:
    """The real robot standing (no physics run) + a small free box `probe` that the tests place against the hand."""
    spec = M.robot_spec("real", visual=False)
    b = spec.worldbody.add_body(name="probe", pos=[1.0, 1.0, 0.1])
    b.add_freejoint()
    b.add_geom(name="probe_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.003, 0.0015, 0.0015], mass=1.0)
    m = spec.compile()
    d = mujoco.MjData(m)
    ix = M.RobotIndex(m)
    d.qpos[:] = M.stand_qpos(m)
    mujoco.mj_kinematics(m, d)
    return m, d, ix


def _place(m: Any, d: Any, centre: np.ndarray, rot: np.ndarray) -> LowState:
    """Put the probe at `centre` with orientation `rot` (its x = the pressing axis), solve the contact forces as the
    engine's state would be after mj_step1, and return a LowState shell for process()."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "probe")
    adr = int(m.jnt_qposadr[m.body_jntadr[b]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    d.qpos[adr : adr + 3] = centre
    d.qpos[adr + 3 : adr + 7] = quat
    d.qvel[:] = 0.0
    mujoco.mj_step1(m, d)
    z7 = {sd: np.zeros(7) for sd in dex3.SIDES}
    z = np.zeros(26)
    return LowState(0.0, 0, z, z, z, np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3), z7, z7, z7,
                    np.zeros(3), np.array([1.0, 0, 0, 0]), 0)  # fmt: skip


def _press_pad(m: Any, d: Any, side: str, k: int, s_frac: float, t_frac: float, depth: float = 0.0006) -> LowState:
    """The probe pressed `depth` into pad k at (s_frac towards the tip, t_frac across) x its half sizes."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_pad_{k}")
    half = m.geom_size[g]
    thin = int(np.argmin(half))
    inplane = [a for a in range(3) if a != thin]
    long = max(inplane, key=lambda a: half[a])
    short = next(a for a in inplane if a != long)
    rot = d.geom_xmat[g].reshape(3, 3)
    sm = SensorModel(OFF)
    sm.bind(m, M.RobotIndex(m))
    distal = next(p.distal for p in sm._hands[side].pads if p.k == k)
    out_sign = 1.0 if m.geom_pos[g][thin] >= 0 else -1.0
    loc = np.zeros(3)
    loc[thin] = out_sign * (half[thin] + 0.003 - depth)
    loc[long] = distal * s_frac * half[long]
    loc[short] = t_frac * half[short]
    centre = d.geom_xpos[g] + rot @ loc
    axes = np.zeros((3, 3))  # probe x along the pad normal, y / z in plane
    axes[:, 0] = rot[:, thin]
    axes[:, 1] = rot[:, long]
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    return _place(m, d, centre, axes)


@pytest.fixture(scope="module")
def probe() -> tuple[Any, Any, M.RobotIndex]:
    return _probe_world()


def _sense(m: Any, d: Any, ix: M.RobotIndex, st: LowState) -> LowState:
    sm = SensorModel(OFF)
    sm.bind(m, ix)
    return sm.process(st, m, d, 0.0, None)


def test_pressing_a_tip_pad_lights_its_cells_proximal_to_distal(probe: tuple[Any, Any, M.RobotIndex]) -> None:
    m, d, ix = probe
    k = dex3.FINGER_PADS["index"][1]  # the index tip pad (3x1: slots 3, 6, 8)
    for s_frac, strongest in ((0.8, 8), (-0.8, 3), (0.0, 6)):
        out = _sense(m, d, ix, _press_pad(m, d, "left", k, s_frac, 0.0))
        p = out.hand_pressure["left"] - 101325.0
        force = out.truth["hand_pad_force"]["left"]
        assert force[k] > 0.2 and np.count_nonzero(force) == 1, force
        cells = {sl: p[k, sl] for sl in dex3.PAD_SLOTS[k]}
        assert max(cells, key=lambda sl: cells[sl]) == strongest, (s_frac, cells)
        others = np.delete(np.arange(9), k)
        assert np.all(out.hand_pressure["left"][others][:, [0, 2, 9, 11]] <= 101325.0)  # nothing elsewhere


def test_pressing_a_2x2_pad_lights_the_nearest_corner(probe: tuple[Any, Any, M.RobotIndex]) -> None:
    m, d, ix = probe
    k = dex3.FINGER_PADS["middle"][0]  # middle proximal pad (2x2: slots 0, 2 proximal; 9, 11 distal)
    out = _sense(m, d, ix, _press_pad(m, d, "right", k, 0.9, 0.9))
    p = out.hand_pressure["right"][k]
    assert max((0, 2, 9, 11), key=lambda sl: p[sl]) == 11
    assert np.all(p[[1, 3, 4, 5, 6, 7, 8, 10]] == 30000.0)  # inactive slots: no data


def test_a_light_fingertip_contact_gives_no_pad_signal(probe: tuple[Any, Any, M.RobotIndex]) -> None:
    """The rounded tip past the tip pad (Dex3-1: the cells sit behind the tips) feels nothing (A10)."""
    m, d, ix = probe
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_hand_index_1_link")
    rot = d.xmat[b].reshape(3, 3)
    # the distal capsule ends at x = 0.040 + 0.0085 in its link frame; the probe (half 3 mm along x) touches the cap
    centre = d.xpos[b] + rot @ np.array([0.0485 + 0.003 - 0.0006, 0.0, 0.0])
    st = _place(m, d, centre, rot)
    g_link = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "left_hand_index_1_geom")
    g_probe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_geom")
    pairs = {(int(c.geom1), int(c.geom2)) for c in d.contact[: d.ncon]}
    assert (g_link, g_probe) in pairs or (g_probe, g_link) in pairs  # it touches
    out = _sense(m, d, ix, st)
    assert np.all(out.truth["hand_pad_force"]["left"] == 0.0)
    active = out.hand_pressure["left"][CELL_PAD, [s for sl in dex3.PAD_SLOTS.values() for s in sl]]
    assert active.shape == (N_CELLS,) and np.all(active == 101325.0)


def test_force_solve_leaves_the_trajectory_bit_identical(probe: tuple[Any, Any, M.RobotIndex]) -> None:
    """process() solves the constraint forces after mj_step1 when a hand geom touches: mj_step2 must not notice."""
    m, _, ix = probe

    def run(sense: bool) -> bytes:
        d = mujoco.MjData(m)
        d.qpos[:] = M.stand_qpos(m)
        mujoco.mj_kinematics(m, d)
        st = _press_pad(m, d, "left", dex3.FINGER_PADS["index"][1], 0.0, 0.0, depth=0.0002)
        sm = SensorModel(SensorConfig(), seed=1)
        sm.bind(m, ix)
        for _ in range(40):
            if sense:
                sm.process(st, m, d, 0.001, None)
            mujoco.mj_step2(m, d)
            mujoco.mj_step1(m, d)
        if sense:
            assert sm.force_solves > 0
        return bytes(d.qpos.tobytes() + d.qvel.tobytes())

    assert run(True) == run(False)


def test_pad_glitches_and_drift(probe: tuple[Any, Any, M.RobotIndex]) -> None:
    m, d, ix = probe
    st = _place(m, d, np.array([1.0, 1.0, 0.1]), np.eye(3))  # probe far away: no contact
    sm = SensorModel(OFF.replace(pad_glitch_p=0.02), seed=4)
    sm.bind(m, ix)
    outs = [sm.process(st, m, d, 0.002, None).hand_pressure["left"] for _ in range(500)]
    up = np.array([[o[k, dex3.PAD_SLOTS[k][0]] > 200000.0 for k in range(9)] for o in outs])
    assert abs(float(up.mean()) - 0.02) < 0.01
    closed = dataclasses.replace(st, hand_q={sd: dex3.preset("fist", sd) for sd in dex3.SIDES})
    out = _sense_cfg(m, d, ix, closed, OFF.replace(pad_shape_drift_pa=600.0))
    k = dex3.FINGER_PADS["index"][0]
    assert out.hand_pressure["left"][k, 0] > 101325.0 + 300.0
    assert out.hand_pressure["left"][7, 0] == 101325.0  # the palm does not drift with finger shape


def _sense_cfg(m: Any, d: Any, ix: M.RobotIndex, st: LowState, cfg: SensorConfig) -> LowState:
    sm = SensorModel(cfg)
    sm.bind(m, ix)
    return sm.process(st, m, d, 0.0, None)


# ---------------------------------------------------------------- thermal
def test_thermal_two_pole_response(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    """The A4 model (session 3): per motor T = ambient + idle rise + D_fast + D_slow, each pole exact for constant
    torque; the body starts at its powered idle; reported in whole degrees. The values are A4's fit
    (models/sensors.yaml; the fit itself: tests/validation/test_a4_thermal.py)."""
    m, d, _, s = real
    c = SensorConfig()
    th = body_thermal_arrays(c)
    tau = np.zeros(26)
    tau[1] = 25.0  # the 5 Oct hip-roll load (v2 runtime.py:86-87)
    st = dataclasses.replace(s, tau_est=tau, hand_tau={k: np.full(7, 0.45) for k in s.hand_tau})
    sm = _model(OFF.replace(temp_quant_c=1.0), real)
    idle = c.body_ambient_c + th["idle_rise_c"]
    u_h = 0.45**c.hand_heat_exponent
    for i in range(1, 121):
        out = sm.process(st, m, d, 1.0, None)
        exact = out.truth["motor_temp"]
        want = idle[1] + sum(
            th[f"gain_{k}"][1] * 625.0 * (1.0 - np.exp(-i / th[f"tau_{k}_s"][1])) for k in ("fast", "slow")
        )
        assert exact[1] == pytest.approx(want, rel=1e-12)
        assert exact[0] == pytest.approx(idle[0], rel=1e-15)  # no load: the powered idle, no heat
        assert out.motor_temp[1] == np.round(exact[1])  # reported in whole degrees
        h = out.truth["hand_temp"]["left"][5]
        want_h = c.hand_ambient_c + sum(
            getattr(c, f"hand_heat_gain_{k}") * u_h * (1.0 - np.exp(-i / getattr(c, f"hand_heat_tau_{k}_s")))
            for k in ("fast", "slow")
        )
        assert h == pytest.approx(want_h, rel=1e-12)
    # 12 min at 25 N m: past v2's 60 C warn level (the left hip roll's protection tripped after 12 min at this load)
    out = sm.process(st, m, d, 600.0, None)
    assert float(out.truth["motor_temp"][1]) > 60.0
    out = sm.process(st, m, d, 1e6, None)
    ss = idle[1] + (th["gain_fast"][1] + th["gain_slow"][1]) * 625.0
    assert out.truth["motor_temp"][1] == pytest.approx(ss, rel=1e-9)


def test_thermal_s2_placeholder_fields_are_gone() -> None:
    """The one-pole placeholders (body_heat_k / body_heat_tau_s / hand_heat_k / hand_heat_tau_s) are replaced by the
    fitted two-pole fields; a stale config naming them is refused (SensorConfig has no such field)."""
    names = {f.name for f in dataclasses.fields(SensorConfig)}
    assert not {"body_heat_k", "body_heat_tau_s", "hand_heat_k", "hand_heat_tau_s"} & names
    assert set(SensorConfig().body_heat_gain_fast) == set(THERMAL_GROUPS)
    assert sorted(SensorConfig().body_idle_rise_c) == sorted(JOINTS)


# ---------------------------------------------------------------- faults
def test_sensor_spike_only_inside_its_window(real: tuple[Any, Any, M.RobotIndex, LowState]) -> None:
    m, d, _, s = real
    fs = FaultSchedule(
        [
            Fault("sensor_spike", 0.5, 0.1, {"field": "q", "index": 3, "magnitude": 0.2}),
            Fault(
                "sensor_spike", 0.5, 0.1, {"field": "hand_pressure", "index": [7, 0], "side": "left", "magnitude": 5e4}
            ),
        ]
    )
    sm = _model(OFF, real)
    for t in (0.45, 0.5, 0.55, 0.5999, 0.6, 0.65):
        out = sm.process(dataclasses.replace(s, t=t), m, d, 0.0, fs)
        inside = 0.5 <= t < 0.6
        assert out.q[3] == pytest.approx(s.q[3] + (0.2 if inside else 0.0), abs=1e-15)
        assert out.hand_pressure["left"][7, 0] - 101325.0 == pytest.approx(5e4 if inside else 0.0)
        assert out.hand_pressure["right"][7, 0] == 101325.0
        assert out.truth["q"][3] == s.q[3]
    assert fs.count("sensor_spike", "applied") == 2


# ---------------------------------------------------------------- train profile (no hands)
def test_train_profile_has_no_hand_sensors() -> None:
    m = M.build_model("train", visual=False)
    sim = Simulation(m, M.RobotIndex(m))
    sim.reset(M.stand_qpos(m))
    s = sim.read_state()
    sm = SensorModel(seed=1)
    sm.bind(sim.m, M.RobotIndex(sim.m))
    out = sm.process(s, sim.m, sim.d, 0.0, None)
    assert out.hand_pressure == {} and out.hand_temp == {}
    head = ~M.RobotIndex(sim.m).present
    assert head.any() and np.all(out.q[head] == 0.0)
    assert not np.array_equal(out.q, s.q)


# ---------------------------------------------------------------- through the engine (Simulation(sensors=...))
def test_through_the_engine_reruns_bit_identical_and_physics_untouched() -> None:
    """The engine calls process once per publish; the stand hold ignores the readings, so the physics must equal a run
    without sensors bit for bit (truth = the exact state), and two runs with the same seed must be identical."""
    m = M.build_model("real", visual=False)
    pose = StandHold.for_model(m).pose

    def run(sensors: SensorModel | None, seed: int = 0) -> tuple[list[LowState], Simulation]:
        sim = Simulation(m, M.RobotIndex(m), seed=seed, sensors=sensors)
        sim.reset(M.stand_qpos(m, pose))
        seen: list[LowState] = []
        hold = StandHold(pose)

        def ctrl(st: LowState) -> Any:
            seen.append(st)
            return hold(st)

        sim.add_controller(ctrl, 0.002, "stand")
        sim.run_for(0.3)
        return seen, sim

    plain, sim0 = run(None)
    a, sim_a = run(SensorModel(), seed=11)
    b, _ = run(SensorModel(), seed=11)
    assert [_bits(x) for x in a] == [_bits(x) for x in b]
    assert sim_a.d.qpos.tobytes() == sim0.d.qpos.tobytes()
    for x, y in zip(a, plain, strict=True):
        assert x.truth["q"].tobytes() == y.q.tobytes()
    last = a[-1]
    assert not np.array_equal(last.q, plain[-1].q)  # the controller saw noisy readings
    assert set(last.hand_pressure) == set(dex3.SIDES) and last.hand_pressure["left"].shape == (9, 12)
    knee = 3
    c0 = SensorConfig()
    idle_knee = c0.body_ambient_c + body_thermal_arrays(c0)["idle_rise_c"][knee]
    assert last.truth["motor_temp"][knee] > idle_knee  # the loaded knee warms above its idle (exact value)
    c, _ = run(SensorModel(), seed=12)
    assert _bits(c[-1]) != _bits(last)
