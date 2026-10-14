"""Sensor models of r1sim (DESIGN.md §12.3 "Sensors"): what the robot's sensors report, not what the physics knows.

The engine (r1sim.engine.Simulation, `sensors=`) calls `SensorModel.process` exactly once per published state, in
order, in the physics thread; what it returns is what `read_state()` and the lockstep controllers get. Every value it
changes keeps its exact value in `LowState.truth` (tests and the validation suite read that; controllers never do).

Models (every parameter in `SensorConfig`, nominal values with sources and ranges in models/sensors.yaml):

* **Encoders** (body 26, hands 7 per side): a per-seed constant offset (the joint zero calibration; nominal 0),
  Gaussian noise on q and dq, quantisation of q. Absent joints (the head in the train profile) stay 0.
* **IMU** at the model's `imu` site (real profile: the URDF IMU position, rotated by the IMU's mounting offset -
  model.IMU_MOUNT_OFFSET_DEG, session 4: geometry, so the exact readings carry it too): gyro noise + a per-seed
  constant bias (+ an optional random walk), accelerometer noise + bias, the orientation as the exact one times a
  small per-seed constant rotation (the session-to-session variation around the mounting offset) and a small random
  rotation per sample.
* **tau_est**: body = scale x the applied torque + noise; hands = `hand_tau_gain` x the applied torque + noise (gain
  1.0 since session 3: the 7 Oct calibration fitted the reading to kp x error per joint; v2's "~2x" (HG-16) compared a
  finger's two-joint sum with one joint's command - models/sensors.yaml), in N·m (the calibrated reading,
  default) or in the hand service's raw units (`hand_tau_units="raw"`: N·m / 1e-6, quantised in 418.65 steps; thumb_0,
  which the 7 Oct calibration could not fit, reads its stuck sample value). Junk readings (a garbage magnitude with a
  random sign) at a configurable rate per sample and motor.
* **Pads** (real profile; hands with pad boxes only): the 9 x 12 Dex3 PressSensorState layout. The 33 active cells
  (r1sim.dex3.PAD_SLOTS) are placed on the pad boxes of the hand model: a 2x2 pad has its cells at the box quadrants
  (slots 0, 2 proximal, 9, 11 distal), a 3x1 tip pad has three cells along the finger (slots 3, 6, 8, proximal to
  distal) - the slot-to-place map is an ESTIMATE (manual p.19 grid; confirm by pressing on HW1). Each contact's normal
  force on a pad box (or on the finger's own link shape within the pad's face + 2 mm, the v2 "real" pad model) is split
  bilinearly between the nearest cells; a cell reads absolute pressure baseline + gain x F / area (+ offset, shape
  and heat drift, noise, single-frame glitches), above a detection floor (10 g per taxel) and up to saturation
  (2.5 kg per taxel); inactive slots read 30000 ("no data"). The rounded fingertip past the tip pad feels nothing:
  light tip contacts give no signal (A10).
* **Thermal** (A4, session 3: fitted to v2's logged temperatures, r1sim.tier_a.a4_thermal): a parallel TWO-POLE
  model per motor, T = T_ambient + idle_rise + D_fast + D_slow with tau_k dD_k/dt = -D_k + G_k u, integrated exactly for
  piecewise-constant torque over the time since the previous publish; reported like the robot (whole degrees). Body:
  u = tau^2, gains and time constants per joint group (THERMAL_GROUPS), an idle rise per motor (powered, no load:
  the head yaw runs ~26 C over the room); the body starts at its powered idle (ambient + idle rise). Hands: u =
  |tau|^1.5 (fitted exponent), ambient = the powered idle hand (~40 C), no idle rise. One time constant could not fit
  the logs: the reported temperature (the hotter of a motor's two sensors) follows load in seconds (36 -> 59 C in
  ~30 s at a ~24 N m hip-roll load, 5 Oct) on top of a housing that drifts over tens of minutes.

`SensorConfig.off()`: no noise, bias, scale, quantisation, junk, glitch or drift - q / dq / tau_est / quat / gyro /
acc and the hand q / dq / tau pass through bit-identical; pads and temperatures are still computed (noise-free).

Contact forces: the engine publishes after `mj_step1` (positions / velocities fresh, the constraint forces NOT yet
solved for the new contacts). When a contact touches a hand geom, `process` solves the acceleration stage for the
current state (`mj_fwdActuation`, `mj_fwdAcceleration`, `mj_fwdConstraint`) before reading forces. These are pure
functions of the state and `mj_step2` recomputes them, so the trajectory is unchanged (test_force_solve_leaves_the_
trajectory_bit_identical); with no hand contact nothing extra runs.
"""

from __future__ import annotations

import dataclasses
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import numpy.typing as npt

from . import dex3
from .engine import JointMap, LowState
from .faults import FaultSchedule
from .joints import JOINTS

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

SENSORS_YAML = Path(__file__).resolve().parents[2] / "models" / "sensors.yaml"

# ------------------------------------------------------------------ thermal model constants (A4 fit, session 3)
#: body joint groups that share the thermal gains / time constants (SDK joint indices; motor sizes per v2 joints.py
#: TAU_MAX 60 / 50 / 33 N m)
THERMAL_GROUPS: dict[str, tuple[int, ...]] = {
    "hip_pitch": (0, 6),
    "hip_roll": (1, 7),
    "hip_yaw": (2, 8),
    "knee": (3, 9),
    "ankle": (4, 5, 10, 11),
    "waist": (12, 13),
    "shoulder": (14, 15, 16, 19, 20, 21),
    "elbow": (17, 22),
    "wrist": (18, 23),
    "head": (24, 25),
}
# fitted on v2's 5-8 Oct sessions (5 s windows of the commanded torque^2; r1sim.tier_a.a4_thermal; sources and
# residuals in models/sensors.yaml)
BODY_HEAT_GAIN_FAST = {"hip_pitch": 0.0514, "hip_roll": 0.0161, "hip_yaw": 0.0309, "knee": 0.0213, "ankle": 0.0316,
                       "waist": 0.0209, "shoulder": 0.000826, "elbow": 0.0, "wrist": 0.0, "head": 0.0561}  # fmt: skip
BODY_HEAT_TAU_FAST_S = {"hip_pitch": 45.0, "hip_roll": 5.0, "hip_yaw": 45.0, "knee": 20.0, "ankle": 45.0,
                        "waist": 20.0, "shoulder": 45.0, "elbow": 3.0, "wrist": 3.0, "head": 45.0}  # fmt: skip
BODY_HEAT_GAIN_SLOW = {"hip_pitch": 0.0, "hip_roll": 0.0672, "hip_yaw": 0.481, "knee": 0.181, "ankle": 0.103,
                       "waist": 0.0, "shoulder": 0.141, "elbow": 0.844, "wrist": 0.499, "head": 0.00147}  # fmt: skip
BODY_HEAT_TAU_SLOW_S = {"hip_pitch": 1500.0, "hip_roll": 900.0, "hip_yaw": 3600.0, "knee": 2400.0,
                        "ankle": 5400.0, "waist": 1500.0, "shoulder": 1500.0, "elbow": 1500.0, "wrist": 2400.0,
                        "head": 2400.0}  # fmt: skip
BODY_IDLE_RISE_C = {
    "left_hip_pitch": 6.0, "left_hip_roll": 7.0, "left_hip_yaw": 8.2, "left_knee": 5.2, "left_ankle_pitch": 14.5,
    "left_ankle_roll": 17.2, "right_hip_pitch": 6.1, "right_hip_roll": 7.6, "right_hip_yaw": 10.5, "right_knee": 10.2,
    "right_ankle_pitch": 10.8, "right_ankle_roll": 20.9, "waist_roll": 8.3, "waist_yaw": 17.3,
    "left_shoulder_pitch": 13.6, "left_shoulder_roll": 7.3, "left_shoulder_yaw": 12.2, "left_elbow": 14.6,
    "left_wrist_roll": 11.1, "right_shoulder_pitch": 12.0, "right_shoulder_roll": 8.1, "right_shoulder_yaw": 17.1,
    "right_elbow": 20.1, "right_wrist_roll": 9.1, "head_pitch": 11.3, "head_yaw": 26.0,
}  # fmt: skip


def body_thermal_arrays(c: SensorConfig) -> dict[str, FloatArray]:
    """Per-joint (26,) arrays of the body thermal parameters of `c` (group values spread over THERMAL_GROUPS)."""

    def per_joint(vals: Mapping[str, float]) -> FloatArray:
        out = np.zeros(N_BODY)
        for g, js in THERMAL_GROUPS.items():
            out[list(js)] = float(vals[g])
        return out

    return {
        "idle_rise_c": np.array([float(c.body_idle_rise_c[n]) for n in JOINTS], dtype=np.float64),
        "gain_fast": per_joint(c.body_heat_gain_fast),
        "tau_fast_s": per_joint(c.body_heat_tau_fast_s),
        "gain_slow": per_joint(c.body_heat_gain_slow),
        "tau_slow_s": per_joint(c.body_heat_tau_slow_s),
    }


N_BODY = 26
N_PAD, N_SLOT = 9, 12
SPIKE_FIELDS = ("q", "dq", "tau_est", "gyro", "acc", "hand_pressure")


@dataclass(frozen=True)
class SensorConfig:
    """Every sensor parameter (SI: rad, rad/s, m/s^2, N·m, N, Pa, m^2, deg C, s). Nominal = models/sensors.yaml."""

    # encoders (body)
    q_noise_std: float = 2.0e-4
    q_quant: float = 3.0e-5
    q_bias_std: float = 0.0
    dq_noise_std: float = 0.02
    # encoders (hands)
    hand_q_noise_std: float = 0.002
    hand_q_quant: float = 0.0
    hand_dq_noise_std: float = 0.02
    # IMU
    gyro_noise_std: float = 0.01
    gyro_bias_std: float = 0.003
    gyro_bias_rw: float = 0.0  # rad/s per sqrt(s)
    acc_noise_std: float = 0.05
    acc_bias_std: float = 0.05
    quat_noise_std: float = 0.002
    quat_bias_std: float = 0.0035
    # tau_est (body)
    tau_scale: float = 1.0
    tau_noise_std: float = 0.05
    tau_junk_p: float = 0.0
    tau_junk_abs: float = 1674.594
    # tau_est (hands)
    hand_tau_gain: float = 1.0
    hand_tau_noise_std: float = 0.001
    hand_tau_units: Literal["nm", "raw"] = "nm"
    hand_tau_nm_per_raw: float = 1.0e-6
    hand_tau_raw_step: float = 418.65
    hand_tau_thumb0_raw: float = 8498.6
    hand_tau_junk_p: float = 0.0
    hand_tau_junk_abs: float = 1674.594
    # pads
    pads: bool = True
    pad_baseline_pa: float = 101325.0
    pad_no_data: float = 30000.0
    pad_cell_area_m2: float = 1.2e-5
    tip_cell_area_m2: float = 0.9e-5
    pad_gain: float = 1.0
    pad_min_force_n: float = 0.098
    pad_max_force_n: float = 24.5
    pad_noise_std_pa: float = 100.0
    pad_offset_std_pa: float = 500.0
    pad_shape_drift_pa: float = 600.0
    pad_heat_drift_pa_per_c: float = 190.0
    pad_heat_ref_c: float = 40.0
    pad_glitch_p: float = 2.0e-5
    pad_glitch_min_pa: float = 254000.0
    pad_glitch_max_pa: float = 421000.0
    pad_face_margin_m: float = 0.002
    # thermal (two poles per motor; body gains per joint group, see THERMAL_GROUPS; values = models/sensors.yaml)
    thermal: bool = True
    body_ambient_c: float = 23.3
    body_idle_rise_c: Mapping[str, float] = field(default_factory=lambda: dict(BODY_IDLE_RISE_C))
    body_heat_gain_fast: Mapping[str, float] = field(default_factory=lambda: dict(BODY_HEAT_GAIN_FAST))  # C/(N m)^2
    body_heat_tau_fast_s: Mapping[str, float] = field(default_factory=lambda: dict(BODY_HEAT_TAU_FAST_S))
    body_heat_gain_slow: Mapping[str, float] = field(default_factory=lambda: dict(BODY_HEAT_GAIN_SLOW))  # C/(N m)^2
    body_heat_tau_slow_s: Mapping[str, float] = field(default_factory=lambda: dict(BODY_HEAT_TAU_SLOW_S))
    hand_ambient_c: float = 40.0
    hand_heat_exponent: float = 1.5
    hand_heat_gain_fast: float = 20.5  # C per (N m)^1.5
    hand_heat_tau_fast_s: float = 3.0
    hand_heat_gain_slow: float = 124.0  # C per (N m)^1.5
    hand_heat_tau_slow_s: float = 300.0
    temp_quant_c: float = 1.0

    @classmethod
    def off(cls) -> SensorConfig:
        """Exact passthrough of the joint / IMU / torque readings; pads and temperatures noise-free."""
        zero = (
            "q_noise_std q_quant q_bias_std dq_noise_std hand_q_noise_std hand_q_quant hand_dq_noise_std "
            "gyro_noise_std gyro_bias_std gyro_bias_rw acc_noise_std acc_bias_std quat_noise_std quat_bias_std "
            "tau_noise_std tau_junk_p hand_tau_noise_std hand_tau_junk_p pad_noise_std_pa pad_offset_std_pa "
            "pad_shape_drift_pa pad_heat_drift_pa_per_c pad_glitch_p temp_quant_c"
        ).split()
        kw: dict[str, Any] = {k: 0.0 for k in zero}
        return cls(**kw, tau_scale=1.0, hand_tau_gain=1.0, hand_tau_units="nm")

    @classmethod
    def from_yaml(cls, path: str | Path = SENSORS_YAML) -> SensorConfig:
        """The nominal values of a provenance file (schema r1mc3.provenance/1; parameter id = field name)."""
        from r1mc3.config import load_provenance

        prov = load_provenance(path)
        names = {f.name for f in fields(cls)}
        unknown = set(prov.parameters) - names
        if unknown:
            raise ValueError(f"{path}: parameters that SensorConfig does not have: {sorted(unknown)}")
        kw: dict[str, Any] = {k: e.value for k, e in prov.parameters.items()}
        return cls(**kw)

    def replace(self, **kw: Any) -> SensorConfig:
        return dataclasses.replace(self, **kw)


@dataclass
class _Pad:
    """One pad box of one hand, in its geom frame (fixed at bind time)."""

    k: int
    geom: int
    half: FloatArray  # (3,) box half sizes
    thin: int  # axis index of the thickness
    long: int  # in-plane axis along the finger (cells proximal -> distal)
    short: int  # the other in-plane axis
    distal: float  # +1 / -1: the sign of `long` that points towards the fingertip
    slots: tuple[int, ...]
    area: float


@dataclass
class _Hand:
    side: str
    pads: list[_Pad]
    link_pads: dict[int, tuple[int, ...]] = field(default_factory=dict)  # link geom -> pad ids whose face it carries


def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(name.encode())])


def _quat_small(rotvec: FloatArray) -> FloatArray:
    """Unit quaternion (w, x, y, z) of a rotation vector."""
    out = np.zeros(4)
    ang = float(np.linalg.norm(rotvec))
    if ang == 0.0:
        out[0] = 1.0
        return out
    mujoco.mju_axisAngle2Quat(out, rotvec / ang, ang)
    return out


def _qmul(a: FloatArray, b: FloatArray) -> FloatArray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return out


def _quantise(x: FloatArray, step: float) -> FloatArray:
    return np.asarray(np.round(x / step) * step, dtype=np.float64)


# Which pad each finger link shape carries (r1sim.dex3.hand_xml): thumb_1 -> thumb pad, thumb_2 -> thumb tip, the
# proximal / distal finger links -> that finger's pad / tip (dex3.FINGER_PADS), the palm box -> the three palm pads.
def _link_pad_map() -> dict[str, tuple[int, ...]]:
    fp = dex3.FINGER_PADS
    return {
        "thumb_1": (fp["thumb"][0],),
        "thumb_2": (fp["thumb"][1],),
        "middle_0": (fp["middle"][0],),
        "middle_1": (fp["middle"][1],),
        "index_0": (fp["index"][0],),
        "index_1": (fp["index"][1],),
        "palm": (6, 7, 8),
    }


# The 33 active cells as flat arrays (pad, slot, tip cell?, finger) so a hand's pressures are one vectorised step.
# Finger: 0 thumb, 1 middle, 2 index (its closure and hottest motor drive the pad's drift), -1 palm (no drift).
_FINGER_OF_PAD: dict[int, int] = {}
for _fi, _f in enumerate(("thumb", "middle", "index")):
    for _k in dex3.FINGER_PADS[_f]:
        _FINGER_OF_PAD[_k] = _fi
CELL_PAD: IntArray = np.array([k for k, sl in dex3.PAD_SLOTS.items() for _ in sl], dtype=np.int64)
CELL_SLOT: IntArray = np.array([s for sl in dex3.PAD_SLOTS.values() for s in sl], dtype=np.int64)
CELL_TIP: npt.NDArray[np.bool_] = np.array([len(dex3.PAD_SLOTS[k]) == 3 for k in CELL_PAD], dtype=bool)
CELL_FINGER: IntArray = np.array([_FINGER_OF_PAD.get(int(k), -1) for k in CELL_PAD], dtype=np.int64)
# motors per finger for the shape drift (thumb_0 has no closing direction: thumb_1, thumb_2, as v2 sim.py:498) and the
# heat drift (all of the finger's motors), as (3, 7) averaging / selection matrices
_FINGER_CLOSE_AVG = np.zeros((3, dex3.N))
_FINGER_HEAT_SEL = np.zeros((3, dex3.N), dtype=bool)
for _fi, (_ic, _ih) in enumerate(((dex3.THUMB[1:], dex3.THUMB), (dex3.MIDDLE, dex3.MIDDLE), (dex3.INDEX, dex3.INDEX))):
    _FINGER_CLOSE_AVG[_fi, _ic] = 1.0 / len(_ic)
    _FINGER_HEAT_SEL[_fi, _ih] = True
N_CELLS = len(CELL_PAD)  # 33
_CLOSE: dict[str, tuple[FloatArray, FloatArray]] = {}
for _s in dex3.SIDES:
    _d, _lo, _hi = dex3.CLOSE_DIR[_s], dex3.Q_MIN[_s], dex3.Q_MAX[_s]
    _CLOSE[_s] = (_d, np.maximum(np.where(_d > 0, _hi, np.where(_d < 0, -_lo, 1.0)), 1e-6))


class SensorModel:
    """The robot's sensors on top of the exact simulated state (see the module docstring)."""

    def __init__(self, cfg: SensorConfig | None = None, *, seed: int = 0) -> None:
        self.cfg = cfg if cfg is not None else SensorConfig()
        self.seed = int(seed)
        self._bound = False
        self._fresh = True
        self._present = np.ones(N_BODY, dtype=bool)
        self._hands: dict[str, _Hand] = {}
        self._sides: tuple[str, ...] = ()
        self._hand_geom = np.zeros(0, dtype=bool)
        self._spikes_logged: set[int] = set()
        self.force_solves = 0  # how often the acceleration stage was solved for pad forces (cost counter)

    # ------------------------------------------------------------------ set-up
    def bind(self, m: Any, joint_map: JointMap) -> None:
        """Index the model once: present body joints, the hands' pad boxes and the link shapes that carry them."""
        self._present = np.asarray(joint_map.present, dtype=bool).copy()
        self._sides = tuple(s for s in dex3.SIDES if s in joint_map.hand_qadr)
        self._hands = {}
        self._hand_geom = np.zeros(int(m.ngeom), dtype=bool)
        d0 = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d0)
        for side in joint_map.hand_qadr:
            pads: list[_Pad] = []
            for k in range(N_PAD):
                g = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_pad_{k}"))
                if g < 0 or int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                    pads = []
                    break
                pads.append(self._pad(m, d0, side, k, g))
            if not pads:
                continue
            hand = _Hand(side, pads)
            for link, ks in _link_pad_map().items():
                g = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_hand_{link}_geom"))
                if g >= 0:
                    hand.link_pads[g] = ks
                    self._hand_geom[g] = True
            for p in pads:
                self._hand_geom[p.geom] = True
            self._hands[side] = hand
        self._bound = True
        self._fresh = True

    @staticmethod
    def _pad(m: Any, d0: Any, side: str, k: int, g: int) -> _Pad:
        half = np.asarray(m.geom_size[g], dtype=np.float64).copy()
        thin = int(np.argmin(half))
        inplane = [a for a in range(3) if a != thin]
        long = max(inplane, key=lambda a: half[a])
        short = next(a for a in inplane if a != long)
        # distal = the in-plane direction towards the fingertip (thumb / finger tip sites; palm pads: the middle tip)
        tip = {0: "thumb", 1: "thumb", 6: "middle", 7: "middle", 8: "middle"}.get(k)
        if tip is None:
            tip = "middle" if k in dex3.FINGER_PADS["middle"] else "index"
        s = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand_{tip}_tip"))
        rot = d0.geom_xmat[g].reshape(3, 3)
        to_tip = rot.T @ (d0.site_xpos[s] - d0.geom_xpos[g]) if s >= 0 else np.eye(3)[long]
        distal = 1.0 if to_tip[long] >= 0 else -1.0
        slots = dex3.PAD_SLOTS[k]
        return _Pad(k, g, half, thin, long, short, distal, slots, 0.0)

    def reset(self, seed: int | None = None) -> None:
        """Back to the start: new random streams for `seed` (None = keep), biases redrawn, temperatures ambient."""
        if seed is not None:
            self.seed = int(seed)
        self._fresh = True

    def _start(self) -> None:
        c, seed = self.cfg, self.seed
        self._r = _rng(seed, "sample")  # per-sample draws: one normal and one uniform block per call (fixed sizes)
        rb = _rng(seed, "bias")  # per-seed constants, drawn in a fixed order
        self._q_bias = rb.normal(0.0, 1.0, N_BODY) * c.q_bias_std * self._present
        self._gyro_bias = rb.normal(0.0, 1.0, 3) * c.gyro_bias_std
        self._acc_bias = rb.normal(0.0, 1.0, 3) * c.acc_bias_std
        self._quat_bias = _quat_small(rb.normal(0.0, 1.0, 3) * c.quat_bias_std)
        self._pad_offset = {s: rb.normal(0.0, 1.0, (N_PAD, N_SLOT)) * c.pad_offset_std_pa for s in dex3.SIDES}
        th = body_thermal_arrays(c)
        self._body_th = th
        self._body_poles = np.zeros((2, N_BODY))  # D_fast, D_slow: the body starts at its powered idle
        self._t_body = c.body_ambient_c + th["idle_rise_c"] * self._present
        self._hand_poles = {s: np.zeros((2, dex3.N)) for s in dex3.SIDES}
        self._t_hand = {s: np.full(dex3.N, c.hand_ambient_c) for s in dex3.SIDES}
        self._spikes_logged = set()
        self._fresh = False

    # ------------------------------------------------------------------ the per-publish step
    def process(self, state: LowState, m: Any, d: Any, dt: float, faults: FaultSchedule | None = None) -> LowState:
        if not self._bound:
            raise RuntimeError("SensorModel.process before bind()")
        if self._fresh:
            self._start()
        c = self.cfg
        dt = max(0.0, float(dt))
        truth: dict[str, Any] = dict(state.truth)
        pres = self._present

        # draws: one fixed-size block per call (the stream position depends only on the number of calls)
        nh = len(self._sides)
        nrm = self._r.normal(size=3 * N_BODY + 12 + nh * (3 * dex3.N + N_CELLS))
        uni = self._r.random(2 * N_BODY + nh * (2 * dex3.N + 2 * N_PAD))
        n_q, n_dq, n_tau = nrm[:N_BODY], nrm[N_BODY : 2 * N_BODY], nrm[2 * N_BODY : 3 * N_BODY]
        n_g, n_a, n_r, n_rw = (nrm[3 * N_BODY + 3 * i : 3 * N_BODY + 3 * i + 3] for i in range(4))
        u_tau, s_tau = uni[:N_BODY], uni[N_BODY : 2 * N_BODY]
        hand_draws: dict[str, tuple[FloatArray, FloatArray]] = {}
        on, ou = 3 * N_BODY + 12, 2 * N_BODY
        for side in self._sides:
            hand_draws[side] = (nrm[on : on + 3 * dex3.N + N_CELLS], uni[ou : ou + 2 * dex3.N + 2 * N_PAD])
            on += 3 * dex3.N + N_CELLS
            ou += 2 * dex3.N + 2 * N_PAD

        # encoders
        q = state.q
        if c.q_bias_std > 0 or c.q_noise_std > 0:
            q = q + self._q_bias + n_q * c.q_noise_std * pres
        if c.q_quant > 0:
            q = np.where(pres, _quantise(q, c.q_quant), 0.0)
        dq = state.dq + n_dq * c.dq_noise_std * pres if c.dq_noise_std > 0 else state.dq

        # IMU
        if c.gyro_bias_rw > 0 and dt > 0:
            self._gyro_bias = self._gyro_bias + n_rw * c.gyro_bias_rw * np.sqrt(dt)
        gyro = state.gyro
        if c.gyro_bias_std > 0 or c.gyro_noise_std > 0 or c.gyro_bias_rw > 0:
            gyro = gyro + self._gyro_bias + n_g * c.gyro_noise_std
        acc = (
            state.acc + self._acc_bias + n_a * c.acc_noise_std
            if (c.acc_bias_std > 0 or c.acc_noise_std > 0)
            else state.acc
        )
        quat = state.quat
        if c.quat_bias_std > 0 or c.quat_noise_std > 0:
            quat = _qmul(_qmul(quat, self._quat_bias), _quat_small(n_r * c.quat_noise_std))

        # tau_est (body)
        tau = state.tau_est
        if c.tau_scale != 1.0 or c.tau_noise_std > 0:
            tau = c.tau_scale * tau + n_tau * c.tau_noise_std * pres
        if c.tau_junk_p > 0:
            junk = (u_tau < c.tau_junk_p) & pres
            tau = np.where(junk, np.where(s_tau < 0.5, -1.0, 1.0) * c.tau_junk_abs, tau)

        # thermal (true applied torques)
        if c.thermal:
            th = self._body_th
            u = np.square(state.tau_est * pres)
            self._body_poles = self._heat(self._body_poles, u, dt, th["gain_fast"], th["tau_fast_s"],
                                          th["gain_slow"], th["tau_slow_s"])  # fmt: skip
            self._t_body = c.body_ambient_c + (th["idle_rise_c"] + self._body_poles.sum(axis=0)) * pres
        temp = self._t_body.copy()
        if c.temp_quant_c > 0:
            temp = _quantise(temp, c.temp_quant_c)

        # hands
        hq, hdq, htau, htemp, hpress = dict(state.hand_q), dict(state.hand_dq), dict(state.hand_tau), {}, {}
        hpress_true: dict[str, FloatArray] = {}
        pad_force: dict[str, FloatArray] = {}
        cells = self._cell_forces(m, d) if self._hands and c.pads else {}
        for side in state.hand_q:
            hn, hu = hand_draws.get(side, (np.zeros(3 * dex3.N + N_CELLS), np.ones(2 * dex3.N + 2 * N_PAD)))
            n_hq, n_hdq, n_ht = hn[: dex3.N], hn[dex3.N : 2 * dex3.N], hn[2 * dex3.N : 3 * dex3.N]
            u_ht, s_ht = hu[: dex3.N], hu[dex3.N : 2 * dex3.N]
            x = state.hand_q[side]
            if c.hand_q_noise_std > 0:
                x = x + n_hq * c.hand_q_noise_std
            if c.hand_q_quant > 0:
                x = _quantise(x, c.hand_q_quant)
            hq[side] = x
            if c.hand_dq_noise_std > 0:
                hdq[side] = state.hand_dq[side] + n_hdq * c.hand_dq_noise_std
            t_true = state.hand_tau[side]
            if c.thermal:
                u_h = np.abs(t_true) ** c.hand_heat_exponent
                self._hand_poles[side] = self._heat(self._hand_poles[side], u_h, dt, c.hand_heat_gain_fast,
                                                    c.hand_heat_tau_fast_s, c.hand_heat_gain_slow,
                                                    c.hand_heat_tau_slow_s)  # fmt: skip
                self._t_hand[side] = c.hand_ambient_c + self._hand_poles[side].sum(axis=0)
            ht = self._t_hand[side].copy()
            htemp[side] = _quantise(ht, c.temp_quant_c) if c.temp_quant_c > 0 else ht
            r = t_true
            if c.hand_tau_gain != 1.0 or c.hand_tau_noise_std > 0:
                r = c.hand_tau_gain * t_true + n_ht * c.hand_tau_noise_std
            if c.hand_tau_junk_p > 0:
                r = np.where(u_ht < c.hand_tau_junk_p, np.where(s_ht < 0.5, -1.0, 1.0) * c.hand_tau_junk_abs, r)
            if c.hand_tau_units == "raw":
                r = _quantise(r / c.hand_tau_nm_per_raw, c.hand_tau_raw_step)
                if c.hand_tau_thumb0_raw > 0:
                    r[0] = c.hand_tau_thumb0_raw
            htau[side] = r
            if side in self._hands and c.pads:
                f_cells = cells.get(side, np.zeros((N_PAD, N_SLOT)))
                n_p = hn[3 * dex3.N :]
                u_g, m_g = hu[2 * dex3.N : 2 * dex3.N + N_PAD], hu[2 * dex3.N + N_PAD :]
                hpress[side], hpress_true[side] = self._pressure(side, f_cells, n_p, u_g, m_g, state.hand_q[side])
                pad_force[side] = f_cells.sum(axis=1)

        out = dataclasses.replace(
            state,
            q=q,
            dq=dq,
            tau_est=tau,
            quat=quat,
            gyro=gyro,
            acc=acc,
            motor_temp=temp,
            hand_q=hq,
            hand_dq=hdq,
            hand_tau=htau,
            hand_temp=htemp if htemp else dict(state.hand_temp),
            hand_pressure=hpress if hpress else dict(state.hand_pressure),
            truth=truth,
        )
        if faults is not None:
            out = self._spike(out, faults)
        truth.update(
            q=state.q, dq=state.dq, tau_est=state.tau_est, quat=state.quat, gyro=state.gyro, acc=state.acc,
            motor_temp=self._t_body.copy(), hand_q=dict(state.hand_q), hand_dq=dict(state.hand_dq),
            hand_tau=dict(state.hand_tau),
        )  # fmt: skip
        if htemp:
            truth["hand_temp"] = {s: v.copy() for s, v in self._t_hand.items() if s in htemp}
        if hpress_true:
            truth["hand_pressure"] = hpress_true
            truth["hand_pad_force"] = pad_force
        return out

    @staticmethod
    def _heat(
        poles: FloatArray, u: FloatArray, dt: float, g_fast: Any, tau_fast: Any, g_slow: Any, tau_slow: Any
    ) -> FloatArray:
        """Exact step over dt of the two poles (2, n), tau_k dD_k/dt = -D_k + G_k u, for constant u."""
        if dt <= 0:
            return poles
        a_f, a_s = (
            np.exp(-dt / np.asarray(tau_fast, dtype=np.float64)),
            np.exp(-dt / np.asarray(tau_slow, dtype=np.float64)),
        )
        fast = poles[0] * a_f + g_fast * u * (1.0 - a_f)
        slow = poles[1] * a_s + g_slow * u * (1.0 - a_s)
        return np.stack([fast, slow]).astype(np.float64)

    # ------------------------------------------------------------------ pads
    def _cell_forces(self, m: Any, d: Any) -> dict[str, FloatArray]:
        """side -> (9, 12) normal force [N] per cell from the contacts on the hands' pad boxes / pad faces."""
        out = {s: np.zeros((N_PAD, N_SLOT)) for s in self._hands}
        ncon = int(d.ncon)
        if ncon == 0:
            return out
        g1 = np.asarray(d.contact.geom1[:ncon], dtype=np.int64)
        g2 = np.asarray(d.contact.geom2[:ncon], dtype=np.int64)
        hit = np.flatnonzero(self._hand_geom[g1] | self._hand_geom[g2])
        if hit.size == 0:
            return out
        # Constraint forces for the CURRENT contacts (see the module docstring: the engine publishes after mj_step1).
        mujoco.mj_fwdActuation(m, d)
        mujoco.mj_fwdAcceleration(m, d)
        mujoco.mj_fwdConstraint(m, d)
        self.force_solves += 1
        f6 = np.zeros(6)
        for i in hit:
            mujoco.mj_contactForce(m, d, int(i), f6)
            fn = abs(float(f6[0]))
            if fn == 0.0:
                continue
            pos = np.asarray(d.contact.pos[i], dtype=np.float64)
            for g in (int(g1[i]), int(g2[i])):
                if not self._hand_geom[g]:
                    continue
                for side, hand in self._hands.items():
                    hit_pad = self._which_pad(d, hand, g, pos)
                    if hit_pad is not None:
                        pad, loc = hit_pad
                        out[side][pad.k] += fn * self._weights(pad, loc)
        return out

    def _which_pad(self, d: Any, hand: _Hand, g: int, pos: FloatArray) -> tuple[_Pad, FloatArray] | None:
        """The pad a contact at world `pos` on geom g presses, and its position in that pad's frame (or None)."""
        for p in hand.pads:
            if p.geom == g:
                return p, self._local(d, p, pos)
        ks = hand.link_pads.get(g)
        if ks is None:
            return None
        # Ported from r1mc2 (v2) backends/sim.py _on_pad at git r21_try 2a4a165 (pad_model "real"); changes: returns the
        # pad and the local position; the margin is a parameter. A contact on a link's own shape counts for its pad only
        # within the pad box's in-plane extent + margin, on the pad's side of the finger (within 6 mm of its plane).
        mg = self.cfg.pad_face_margin_m
        for k in ks:
            p = hand.pads[k]
            loc = self._local(d, p, pos)
            if (
                abs(loc[p.long]) <= p.half[p.long] + mg
                and abs(loc[p.short]) <= p.half[p.short] + mg
                and abs(loc[p.thin]) <= 0.006
            ):
                return p, loc
        return None

    @staticmethod
    def _local(d: Any, p: _Pad, pos: FloatArray) -> FloatArray:
        rot = d.geom_xmat[p.geom].reshape(3, 3)
        return np.asarray(rot.T @ (pos - d.geom_xpos[p.geom]), dtype=np.float64)

    @staticmethod
    def _weights(p: _Pad, loc: FloatArray) -> FloatArray:
        """(12,) share of a contact's force per slot: bilinear between the nearest cell centres (sums to 1)."""
        w = np.zeros(N_SLOT)
        s = float(np.clip(p.distal * loc[p.long] / p.half[p.long], -1.0, 1.0))  # -1 proximal end .. +1 distal end
        if len(p.slots) == 4:  # 2x2: cells at s, t = +-0.5; slots (prox -, prox +, dist -, dist +)
            t = float(np.clip(loc[p.short] / p.half[p.short], -1.0, 1.0))
            wd, wp_ = float(np.clip(s + 0.5, 0.0, 1.0)), float(np.clip(t + 0.5, 0.0, 1.0))
            for slot, a, b in zip(p.slots, (1 - wd, 1 - wd, wd, wd), (1 - wp_, wp_, 1 - wp_, wp_), strict=True):
                w[slot] = a * b
        else:  # 3x1 along the finger: cells at s = -2/3, 0, +2/3 (proximal -> distal)
            centres = np.array([-2.0 / 3.0, 0.0, 2.0 / 3.0])
            for j, slot in enumerate(p.slots):
                w[slot] = float(np.interp(s, centres, np.eye(3)[j]))
        return w

    @staticmethod
    def _closure(q: FloatArray, side: str) -> FloatArray:
        """r1sim.dex3.closure without the per-call set-up (0 open .. 1 closed per joint, thumb_0 -> 0)."""
        d, span = _CLOSE[side]
        return np.asarray(np.clip(d * q / span, 0.0, 1.0), dtype=np.float64)

    def _pressure(
        self, side: str, f_cells: FloatArray, n_p: FloatArray, u_g: FloatArray, m_g: FloatArray, q_true: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """(reported, ideal) 9 x 12 absolute pressure [Pa]; inactive slots NO_DATA. n_p: (33,) normal draws; u_g, m_g:
        (9,) uniform draws for the glitches."""
        c = self.cfg
        f = f_cells[CELL_PAD, CELL_SLOT]
        area = np.where(CELL_TIP, c.tip_cell_area_m2, c.pad_cell_area_m2)
        rep = np.full((N_PAD, N_SLOT), c.pad_no_data)
        ideal = rep.copy()
        ideal[CELL_PAD, CELL_SLOT] = c.pad_baseline_pa + f / area
        felt = np.where(f >= c.pad_min_force_n, np.minimum(f, c.pad_max_force_n), 0.0)
        v = c.pad_baseline_pa + c.pad_gain * felt / area
        if c.pad_offset_std_pa > 0:
            v = v + self._pad_offset[side][CELL_PAD, CELL_SLOT]
        if c.pad_shape_drift_pa > 0 or c.pad_heat_drift_pa_per_c > 0:
            temps = self._t_hand[side]
            hot = np.where(_FINGER_HEAT_SEL, temps, -np.inf).max(axis=1)
            per_finger = np.zeros(4)  # index 3 (= -1) is the palm: no drift
            per_finger[:3] = c.pad_shape_drift_pa * (_FINGER_CLOSE_AVG @ self._closure(q_true, side))
            per_finger[:3] += c.pad_heat_drift_pa_per_c * np.maximum(0.0, hot - c.pad_heat_ref_c)
            v = v + per_finger[CELL_FINGER]
        if c.pad_noise_std_pa > 0:
            v = v + n_p * c.pad_noise_std_pa
        if c.pad_glitch_p > 0:
            g = np.where(
                u_g < c.pad_glitch_p, c.pad_glitch_min_pa + m_g * (c.pad_glitch_max_pa - c.pad_glitch_min_pa), 0.0
            )
            v = v + g[CELL_PAD]
        rep[CELL_PAD, CELL_SLOT] = v
        return rep, ideal

    # ------------------------------------------------------------------ faults
    def _spike(self, out: LowState, faults: FaultSchedule) -> LowState:
        """Apply the active `sensor_spike` faults (params: field, index, magnitude; hand_pressure: index = (pad, slot),
        optional side)."""
        active = faults.active("sensor_spike", out.t)
        if not active:
            return out
        repl: dict[str, Any] = {}
        for f in active:
            fld = str(f.params.get("field", "q"))
            if fld not in SPIKE_FIELDS:
                raise ValueError(f"sensor_spike: field must be one of {SPIKE_FIELDS}, got {fld!r}")
            mag = float(f.params.get("magnitude", 0.0))
            idx = f.params.get("index", 0)
            if fld == "hand_pressure":
                hp = {s: v.copy() for s, v in (repl.get(fld) or out.hand_pressure).items()}
                sides = [f.params["side"]] if "side" in f.params else list(hp)
                pad, slot = (int(idx[0]), int(idx[1])) if isinstance(idx, (list, tuple)) else (int(idx), 0)
                for s in sides:
                    if s in hp:
                        hp[s][pad, slot] += mag
                repl[fld] = hp
            else:
                arr = np.array(repl.get(fld, getattr(out, fld)), dtype=np.float64, copy=True)
                arr[int(idx)] += mag
                repl[fld] = arr
            key = id(f)
            if key not in self._spikes_logged:
                self._spikes_logged.add(key)
                faults.record(out.t, "sensor_spike", "applied", field=fld, index=idx, magnitude=mag)
        return dataclasses.replace(out, **repl)
