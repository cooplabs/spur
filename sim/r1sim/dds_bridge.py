"""The DDS-bridge simulator (DESIGN.md §12.4 "DDS bridge (real time)"; session 5, the bridge agent): r1sim in REAL TIME
behind the robot's own DDS interface, so the real v3 binaries (`r1rt`, `r1guard`, r1d's clients) run unchanged
against it - timing-faithful integration, the stall tests with real processes, operator rehearsal.

What it does, as the R1's board + Unitree's built-in controller + the Dex3 service do on the robot:

* **Physics in its own thread** (`Simulation.start_realtime`): paced by the wall clock, never waits for anybody.
* **rt/lowstate** (unitree_hg LowState_, Unitree's CRC, mode_machine 1, mode_pr 0, the board tick at 1 kHz, the IMU
  as r1sim's sensor model reports it - noise, the mounting offset in the model geometry) from a paced publisher thread
  at `rate_hz` (default 500 Hz; the robot's board publishes ~1 kHz - what this Mac sustains is measured in
  docs/sim/DDS_BRIDGE.md). A state is published only when physics produced a new one (`stats["repeats_skipped"]`).
* **rt/lowcmd** is applied like the board: every sample whose CRC matches (a bad CRC is dropped and counted), the
  latest sample wins (last writer wins, whoever wrote it), held until the next one (zero-order hold, the 9 Oct
  behaviour). A slot with `mode` 0 is driven with zero gains (the board's disabled motor). Every applied command is
  recorded with its writer's publication handle, the time and the smallest leg kp (command gaps, two writers,
  "never limp").
* **rt/dex3/{left,right}/state** (HandState_: 7 motors + the 9 x 12 pads from the sensor model) at `hand_rate_hz`,
  **rt/dex3/*/cmd** applied per side (the body command and each hand's are kept apart, as V2Backend does). Until a
  hand command arrives, the bridge holds the hands itself (open; `stance="9oct"`: v2's closure 0.9 of that session) -
  the stand-in for r1hands; the Dex3 timeout bit is not modelled (r1sim has no hand timeout).
* **The emulated built-in** (`builtin="stand"`): Unitree's controller as v3's takeover sees it - the S1 stand fixture
  (`r1sim.control.StandHold`, a stiff joint PD at the CoM-centred stand; Unitree's own balance is NOT modelled)
  publishing rt/lowcmd from its OWN DomainParticipant (a foreign writer for r1rt, as on the robot) at 500 Hz until
  a ReleaseMode arrives over the motion-switcher RPC (`r1mc3.dds.rpc`, served here: CheckMode answers name "ai"
  while it runs, "" after). SelectMode is not emulated (answered with an error).

The world: `scene` (r1sim.scenes) - "floor" (default: the robot alone on its floor; "none" the same), "table" (a plain
table with a box, a can and a ball), a builder, or a "package.module:attr" string naming one; `w` = a caller's world.

Configurations: `stance` "centred" (the fixture's CoM-centred stand) or "9oct" (the 9 Oct fall session: the logged
table_ready arms, the session's own IMU offset 0.63 / 0.53 deg, the hands at closure 0.9; the fixture holds the
CoM-centred stand WITH those arms); `safety` = the user's U4 stall / fall configuration (no leg stiction:
`model.SAFETY_LEGS_FRICTIONLOSS`); `sensors` (default on) = the S2 sensor model seeded by `seed`.

CycloneDDS: the loopback configuration (`r1mc3.dds.rpc.LOOPBACK_URI`) unless `loopback=False`; QoS as rpc.py
(reliable writers, best-effort readers: matches whatever QoS the peer chose).

Session 6 (the hang agent): `stance="hanging"` - the robot on its STAND for the HW1 rehearsal (`StandSpec`: the
harness's top bar in the stand's padded cradle = a compliant hinge about the lateral axis, the lower harness loop
against the pole = a one-sided pitch stop preloaded by gravity, feet off the floor; variant `attach="weld"` = S5's
rigid pelvis weld), the built-in's stand variants (`builtin="zero_torque"` - Unitree's power-on state, publishing kp 0 /
kd 0 - the stand's default; `"zero_torque_silent"`; `"stand_lock"`; `"damp"`; `"damp_legs"`), and scheduled fault
injections (`inject(kind, at_s=, duration_s=, every_s=, count=, **params)`: the lowstate drops / delays / CRC errors,
the link down, a second writer, motor error words, over-temperature, the hands' freeze / death / timeout, the bridge's
own publisher / command-reader stalls; every window recorded in `faults()`). What is ASSUMED about the stand is in
`StandSpec`'s docstring and docs/sim/DDS_BRIDGE.md.

Not modelled (said plainly): the board's own behaviour without commands (it holds the last one: the 9 Oct evidence;
the board-hold test measures it), rt/arm_sdk, Unitree's balance, the Dex3 timeout (only as the opt-in `hand_timeout_s` /
inject("dex3_timeout"): v2's knowledge, not measured), the wireless remote (zeros), BMS / power.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import json
import math
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from r1mc3.dds import hg, rpc

from . import dex3, model, world
from . import joints as J
from .control import StandHold
from .driver import DriverConfig
from .engine import N_BODY, N_HAND, HandCmd, LowCmd, LowState, Simulation
from .sensors import SensorConfig, SensorModel

FloatArray = npt.NDArray[np.float64]
# CycloneDDS's IDL dataclasses type their fields as its annotation classes (types.array / sequence), which mypy cannot
# index or assign lists to: the conversions below go through this untyped alias of the generated module.
_hg: Any = hg

BUILTIN_HZ = 500.0  # the emulated built-in's command rate (the real one ~650 Hz, v2 robot.py docstring)
BUILTIN_MODE_NAME = "ai"  # CheckMode's name while Unitree's built-in runs (v2 runtime.py restore_builtin default)
STANCES = ("centred", "9oct", "hanging")
# The built-in controller as v3 meets it (session 6: the stand). "stand" = the S1 fixture (a stiff PD at the
# CoM-centred stand; = Unitree's "stand lock" as far as v3 can tell, alias "stand_lock"); "zero_torque" = Unitree's
# power-on state (the user, 10 Oct 2026: "at power-on Unitree's controller is in ZERO TORQUE mode") publishing kp 0 /
# kd 0 commands; "zero_torque_silent" = the same state publishing NOTHING (whether Unitree publishes rt/lowcmd in zero
# torque is unknown: both variants); "damp" = kp 0 + v3's damp kd (Unitree's damping mode as a stand-in); "damp_legs"
# = S5's lockstep built-in on the stand (legs kp 0, the rest stiff); "none" = no built-in at all.
BUILTINS = ("stand", "stand_lock", "zero_torque", "zero_torque_silent", "damp", "damp_legs", "none")
# Session 8 (HW1 S1.0, fix #25): Unitree's STAND LOCK commands a FIXED preset pose (constant over 133 + 66 messages on
# 11 Oct 2026; the robot moves INTO it when stand lock is selected), at exactly v3's stand gains (joints.KP_STIFF /
# KD_STIFF: legs 200 / 3, waist 300 / 5, arms 100 / 2, wrists 50 / 2, head 50 / 2 and 10 / 0.1). SDK order; source
# results/hw1_20261010/part2/s1_0_lowcmd_lock*.txt, NOTES.md S1.0. Used by the HANGING stance's "stand" / "stand_lock"
# built-in; the standing stances keep the S1 fixture (the CoM-centred stand: the floor case was never recorded).
UNITREE_STAND_LOCK_POSE = (
    -0.305, 0.04, -0.14, 0.41, -0.137, -0.06,  # left leg: hip pitch / roll / yaw, knee, ankle pitch / roll
    -0.305, -0.04, 0.14, 0.41, -0.137, 0.06,  # right leg
    0.0, 0.0,  # waist roll / yaw
    0.2, 0.2, 0.0, 0.8, 0.0,  # left arm: shoulder pitch / roll / yaw, elbow, wrist roll
    0.2, -0.2, 0.0, 0.8, 0.0,  # right arm
    0.0, 0.0,  # head pitch / yaw
)  # fmt: skip
ATTACHMENTS = ("cradle", "weld")
STAND_BODY = "r1sim_stand"  # the stand (a mocap body: the pole, the cradle; raised / lowered without a recompile)
STAND_COLLIDE_BIT = 8  # contype / conaffinity bit shared ONLY by the pole and the harness's lower loop

# The arms' commanded pose in the last row before the 9 Oct stall (table_ready; NB r1-motor-controller-v2/results/
# serve-20261010T010206Z/telemetry.jsonl line 1137, q_cmd[14:24]; = a7_fall.LEFT_ARM_LOGGED + RIGHT_ARM_LOGGED).
ARMS_TABLE_READY = (0.0689, 0.685, 0.0192, -0.5269, -1.5534, 0.2079, -0.1853, 0.0042, 0.8141, 0.0039)
IMU_OFFSET_9OCT_DEG = (0.63, 0.53)  # (roll, pitch): the fall session's own IMU - legs offset (a7_fall)
# v2's closure 0.9 (offboard.table_ready, r15): the logged left hand q_cmd of that session (a7_fall docstring step 3)
HAND_CLOSURE_09_LEFT = (0.0, 0.81, 1.355, -1.31, -1.445, -1.31, -1.445)
TORSO_BODY = "torso_link"
PUSH_S = 0.2  # the A7 nudge's duration (a7_fall.PUSH_S)


def tilt_deg(quat_wxyz: npt.ArrayLike) -> float:
    """The angle between the body's z axis and the world's (deg), as v2 / r1sim measure tilt."""
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat_wxyz, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(r[8], -1.0, 1.0))))


# ---------------------------------------------------------------------------------------------- Unitree's CRC, fast
class HgCrc:
    """Unitree's hg CRC (r1mc3.dds.hg.crc_of_image) for images of ONE length, vectorised with numpy.

    The CRC (CRC-32/MPEG-2 over the big-endian bytes of each 32-bit word, init 0xFFFFFFFF, no xorout) is affine over
    GF(2) for a fixed length: crc(m) = Z^n(init) xor (xor over byte positions p of TP[p, m_p]), where Z feeds one zero
    byte and TP[p, b] = Z^(n-1-p)(T[b]) is the contribution of byte b at position p. ~20 us per LowState_ image vs ~200
    us for the pure-Python loop (equal on every golden message and random images: tests/integration/test_dds_bridge.py).
    """

    def __init__(self, image_len: int) -> None:
        self.image_len = int(image_len)
        n = self.image_len - 4  # the crc word is not covered
        if n <= 0 or n % 4:
            raise ValueError("image length must be 4 x words, > 4")
        t = np.array(hg._TABLE, dtype=np.uint32)
        tp = np.empty((n, 256), dtype=np.uint32)
        row = t.copy()
        tp[n - 1] = row
        for p in range(n - 2, -1, -1):
            row = ((row << np.uint32(8)) & np.uint32(0xFFFFFFFF)) ^ t[row >> np.uint32(24)]
            tp[p] = row
        self._tp = tp
        self._idx = np.arange(n)
        c = 0xFFFFFFFF
        for _ in range(n):
            c = ((c << 8) & 0xFFFFFFFF) ^ hg._TABLE[(c >> 24) & 0xFF]
        self._c0 = np.uint32(c)

    def __call__(self, image: bytes) -> int:
        if len(image) != self.image_len:
            raise ValueError(f"image of {len(image)} bytes, this table is for {self.image_len}")
        seq = np.frombuffer(image, dtype=np.uint8, count=self.image_len - 4).reshape(-1, 4)[:, ::-1].ravel()
        return int(self._c0 ^ np.bitwise_xor.reduce(self._tp[self._idx, seq]))


_CRC_LOWSTATE: HgCrc | None = None
_CRC_LOWCMD: HgCrc | None = None


def crc_lowstate(msg: hg.LowState_) -> int:
    global _CRC_LOWSTATE
    if _CRC_LOWSTATE is None:
        _CRC_LOWSTATE = HgCrc(2092)
    return _CRC_LOWSTATE(hg.lowstate_image(msg))


def crc_lowcmd(msg: hg.LowCmd_) -> int:
    global _CRC_LOWCMD
    if _CRC_LOWCMD is None:
        _CRC_LOWCMD = HgCrc(1004)
    return _CRC_LOWCMD(hg.lowcmd_image(msg))


# ---------------------------------------------------------------------------------------------- message conversions
def _rpy(quat_wxyz: FloatArray) -> list[float]:
    w, x, y, z = (float(v) for v in quat_wxyz)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [float(roll), float(pitch), float(yaw)]


def lowstate_msg(st: LowState, present: npt.NDArray[np.bool_]) -> hg.LowState_:
    """The board's LowState_ from an r1sim state (26 joints -> their hg slots, J.SLOT), CRC filled."""
    motors: list[Any] = [hg.motor_state_zero() for _ in range(hg.N_SLOTS)]
    temp = np.asarray(st.motor_temp, dtype=np.float64)
    err = np.asarray(st.motor_err)
    for j in range(N_BODY):
        m = motors[int(J.SLOT[j])]
        m.mode = 1 if present[j] else 0
        m.q = float(st.q[j])
        m.dq = float(st.dq[j])
        m.tau_est = float(st.tau_est[j])
        t_c = round(float(temp[j]))
        m.temperature = [t_c, t_c]
        m.motorstate = int(err[j])
    imu = _hg.IMUState_(
        quaternion=[float(v) for v in st.quat],
        gyroscope=[float(v) for v in st.gyro],
        accelerometer=[float(v) for v in st.acc],
        rpy=_rpy(st.quat),
        temperature=30,
    )
    msg = _hg.LowState_(
        version=[0, 0],
        mode_pr=J.MODE_PR,
        mode_machine=J.MODE_MACHINE,
        tick=int(st.tick) & 0xFFFFFFFF,
        imu_state=imu,
        motor_state=motors,
        wireless_remote=[0] * 40,
        reserve=[0, 0, 0, 0],
        crc=0,
    )
    msg.crc = crc_lowstate(msg)
    out: hg.LowState_ = msg
    return out


def lowcmd_msg(cmd: LowCmd, *, mode_machine: int = J.MODE_MACHINE) -> hg.LowCmd_:
    """A LowCmd_ (CRC filled) from an r1sim body command: the emulated built-in's writer and the tests' peers."""
    msg: Any = hg.lowcmd_zero()
    msg.mode_pr = J.MODE_PR
    msg.mode_machine = mode_machine
    for j in range(N_BODY):
        m = msg.motor_cmd[int(J.SLOT[j])]
        m.mode = 1
        m.q, m.dq, m.tau = float(cmd.q[j]), float(cmd.dq[j]), float(cmd.tau_ff[j])
        m.kp, m.kd = float(cmd.kp[j]), float(cmd.kd[j])
    msg.crc = crc_lowcmd(msg)
    out: hg.LowCmd_ = msg
    return out


def body_cmd_of(msg: Any) -> tuple[LowCmd, int]:
    """The board's reading of a LowCmd_: (the 26-joint command, how many slots of ours were disabled). A disabled
    slot (`mode` 0) is driven with zero gains and zero feed-forward (the motor does nothing)."""
    c = LowCmd()
    disabled = 0
    for j in range(N_BODY):
        m = msg.motor_cmd[int(J.SLOT[j])]
        if int(m.mode) & 0x01 == 0:
            disabled += 1
            continue
        c.q[j], c.dq[j], c.tau_ff[j], c.kp[j], c.kd[j] = m.q, m.dq, m.tau, m.kp, m.kd
    return c, disabled


def hand_state_msg(st: LowState, side: str) -> hg.HandState_:
    msg: Any = hg.hand_state_zero()
    q, dq, tau = st.hand_q[side], st.hand_dq[side], st.hand_tau[side]
    temp = st.hand_temp.get(side)
    for i in range(N_HAND):
        m = msg.motor_state[i]
        m.mode = dex3.mode_byte(i)
        m.q, m.dq, m.tau_est = float(q[i]), float(dq[i]), float(tau[i])
        t_c = 30 if temp is None else round(float(temp[i]))
        m.temperature = [t_c, t_c]
    press = st.hand_pressure.get(side)
    if press is None:  # no sensor model: the 33 active cells read atmospheric pressure, the rest "no data"
        press = np.full((hg.N_DEX3_PRESS, 12), 30000.0)
        for k, slots in dex3.PAD_SLOTS.items():
            press[k, list(slots)] = 101325.0
    for k in range(hg.N_DEX3_PRESS):
        msg.press_sensor_state[k].pressure = [float(v) for v in press[k]]
        msg.press_sensor_state[k].temperature = [30.0] * 12
    msg.power_v = msg.system_v = msg.device_v = 24.0
    msg.power_a = 0.3
    out: hg.HandState_ = msg
    return out


def hand_cmd_of(msg: Any) -> HandCmd:
    h = HandCmd()
    for i, m in enumerate(list(msg.motor_cmd)[:N_HAND]):
        h.q[i], h.dq[i], h.tau_ff[i], h.kp[i], h.kd[i] = m.q, m.dq, m.tau, m.kp, m.kd
    return h


def hand_hold(q: npt.ArrayLike) -> HandCmd:
    h = HandCmd()
    h.q[:] = np.asarray(q, dtype=np.float64)
    h.kp[:] = dex3.KP
    h.kd[:] = dex3.KD
    return h


def snapshot(dq: deque[Any]) -> list[Any]:
    """A copy of a deque another thread appends to (list(dq) raises if an append lands mid-copy: retry)."""
    while True:
        try:
            return list(dq)
        except RuntimeError:
            continue


def instance_handle(entity: Any) -> int:
    """A DDS entity's instance handle (= the publication handle a local reader sees for this writer's samples).
    cyclonedds-python 0.10.2's Entity.get_instance_handle() fails on int() of its ctypes value; read `.value`."""
    import ctypes as ct

    from cyclonedds.internal import dds_c_t

    h = dds_c_t.instance_handle()
    if entity._get_instance_handle(entity._ref, ct.byref(h)) != 0:
        raise RuntimeError("dds_get_instance_handle failed")
    return int(h.value) & 0xFFFFFFFFFFFFFFFF  # unsigned, as sample_info.publication_handle


# ---------------------------------------------------------------------------------------------- records
@dataclass
class Applied:
    """One rt/lowcmd sample the board applied (or dropped)."""

    wall: float  # time.monotonic() at application
    sim_t: float  # simulated time then
    writer: int  # the sample's publication handle (one per DDS writer)
    min_leg_kp: float  # the smallest effective leg kp (a disabled slot counts 0): "never limp" = > 0 always
    disabled_legs: int
    mode_machine: int
    crc_ok: bool


@dataclass
class Truth:
    wall: float
    sim_t: float
    tilt_deg: float
    base_z: float
    # stance "hanging" only (nan otherwise): the lowest point of any foot collision geom above the floor (m; > 0 = the
    # feet are off the floor) and the pelvis's attitude (deg, from the ground-truth quaternion; yaw from the start)
    foot_z_min: float = float("nan")
    pitch_deg: float = float("nan")
    roll_deg: float = float("nan")
    yaw_deg: float = float("nan")


# ---------------------------------------------------------------------------------------------- faults (session 6)
FAULT_KINDS = (
    "lowstate_gap",  # rt/lowstate AND the hands' state withheld (S5 semantics)
    "lowstate_drop",  # fraction: each rt/lowstate sample dropped with this probability
    "lowstate_delay",  # delay_ms: rt/lowstate published delay_ms late (a FIFO)
    "lowstate_crc",  # fraction: rt/lowstate samples with a corrupted CRC
    "lowcmd_drop",  # fraction: rt/lowcmd samples lost before the board
    "lowcmd_crc",  # fraction: rt/lowcmd samples corrupted on the way (the board drops them: bad CRC)
    "link_down",  # the motor link: no rt/lowstate out, rt/lowcmd ignored (the hands' USB path is unaffected)
    "second_writer",  # payload ("stand" | "damp" | "zero"), rate_hz: a foreign participant on rt/lowcmd
    "motor_error",  # joint (SDK index or name), word: LowState_ motorstate on that joint's slot
    "over_temp",  # joint, to_c, rate_c_s: the joint's reported temperature ramps up to to_c
    "hand_freeze",  # side ("left" | "right" | "both"): the hand's state repeats one sample
    "dex3_death",  # side: the hand's state stops and its commands are ignored (dex3_server gone)
    "dex3_timeout",  # timeout_s: a hand goes passive timeout_s after its last HandCmd (v2's knowledge, ~1 s)
    "publisher_stall",  # the bridge's rt/lowstate publisher thread blocked for each window (physics continues)
    "cmd_reader_stall",  # the bridge's rt/lowcmd reader thread blocked for each window (the board holds the last)
)


@dataclass
class FaultWindow:
    """One scheduled fault: windows of `duration_s` from `t0` (wall, time.monotonic()), repeated `every_s` up to `count`
    times (None = once / forever); `first_*` / `last_*` / `hits` = when the bridge actually applied it."""

    kind: str
    params: dict[str, Any]
    t0: float
    duration_s: float | None
    every_s: float | None = None
    count: int | None = None
    first_wall: float | None = None
    first_sim: float | None = None
    last_wall: float | None = None
    last_sim: float | None = None
    hits: int = 0
    cleared: bool = False
    stalls: list[tuple[float, float]] | None = None  # the stall kinds: each actual (start, end) wall time

    def active(self, now: float) -> bool:
        if self.cleared or now < self.t0:
            return False
        off = now - self.t0
        if self.every_s:
            k = int(off // self.every_s)
            if self.count is not None and k >= self.count:
                return False
            off -= k * self.every_s
        return self.duration_s is None or off < self.duration_s

    def window_end(self, now: float) -> float:
        """The end of the window `now` is in (wall)."""
        start = self.t0
        if self.every_s:
            start += int((now - self.t0) // self.every_s) * self.every_s
        return start + (self.duration_s if self.duration_s is not None else 0.0)

    def end(self) -> float | None:
        """When the last window ends (None = never by itself)."""
        if self.duration_s is None:
            return None
        if self.every_s:
            return self.t0 + ((self.count or 1) - 1) * self.every_s + self.duration_s if self.count else None
        return self.t0 + self.duration_s


def _joint_index(joint: int | str) -> int:
    """An SDK joint index (0-25) from an index or a name ("left_knee" / "left_knee_joint")."""
    if isinstance(joint, str):
        name = joint.removesuffix("_joint")
        if name not in J.IDX:
            raise ValueError(f"unknown joint {joint!r}")
        return int(J.IDX[name])
    j = int(joint)
    if not 0 <= j < N_BODY:
        raise ValueError(f"joint index {j} outside 0..{N_BODY - 1}")
    return j


# ---------------------------------------------------------------------------------------------- the stand (session 6)
@dataclass(frozen=True)
class StandSpec:
    """The robot's stand for HW1 (DESIGN §14.1, D12a) as modelled - every number an ESTIMATE until HW1.

    What the stand is (the user's photos IMG_8780 / IMG_8781 and words, 10 Oct 2026; no robot measurement yet): one
    vertical pole on a floor base; at its top a padded cradle (a horizontal rubber-ribbed plate with a vertical tab on
    each side). The robot wears a black tubular HARNESS FRAME bolted to its upper back around the backpack; the frame's
    TOP BAR rests in the cradle between the tabs ("by the top metal bar on the harness on the back at the upper
    torso"); the frame's lower loop / the backpack sits against the pole below. So the robot hangs from a bar behind
    its upper back - a line contact, about a lateral axis; its CoM (in front of the bar) swings the lower body
    backwards until the lower loop rests against the pole: in pitch it is held between the cradle and the pole; roll
    and yaw are resisted by the bar's width on the pad and the tabs (rubber: compliant); the feet hang off the floor.

    How it is modelled (attach="cradle", the default):
    * the bar in the cradle = two CONNECT equality constraints between `torso_link` and the stand body at the bar's
      two contact ends (y = +-bar_half_width_m): together a hinge about the lateral axis through the bar; their
      softness (`cradle_solref`, MuJoCo's (time constant, damping ratio)) is the rubber pad - it gives the roll / yaw
      compliance (the stiffness this produces is MEASURED in the tests and reported; nothing was fitted). Unlike the
      real cradle the constraint is bilateral: the bar cannot lift off (gravity keeps it seated; nothing in HW1 pushes
      the robot up with more than its weight);
    * the pole = a vertical capsule on the stand body that collides ONLY with the harness's lower loop (a lateral
      capsule fixed to `torso_link`; bit STAND_COLLIDE_BIT): a one-sided contact, preloaded by the CoM's moment about
      the bar. `pole_gap_m` > 0 lets the loop swing back that far before it touches (the robot then hangs tilted
      forwards by ~atan(gap / (bar_z - loop_z))); 0 = the loop touches the pole with the robot upright;
    * the harness frame's geoms are MASSLESS (the frame's mass, if it stays on the robot, belongs to the robot's mass
      / CoM: the reaction-board CoM and the weighing decide, U6); legs / arms do not collide with the pole (the pole
      is behind the backpack; the recordings' leg motions are +-0.2 rad);
    * placement: positions are given in the STANDING reference frame (the CoM-centred stand, feet on the floor;
      torso_link at x -0.004, z 0.778; the shoulder pitch joints at z 0.975); the robot is raised, settles under the
      built-in's command for `settle_s` (lockstep, before real time starts), then robot + stand are raised together
      so the lowest foot point is `clearance_m` above the floor.

    attach="weld": S5's lockstep fixture (tests/integration/test_v3_core_onstand.py: the pelvis welded rigidly, here
    to the stand body; no pole) - the stiffest possible stand, for comparison.
    """

    attach: str = "cradle"
    clearance_m: float = 0.08  # the feet's height above the floor once hanging (estimate: "feet off the floor")
    bar_x_m: float = -0.17  # the harness's top bar behind the torso (the back ~-0.10 + backpack / frame ~0.07)
    bar_z_m: float = 0.93  # just below the shoulder line (0.975): the cradle sits at the upper back in the photos
    bar_half_width_m: float = 0.06  # half the bar's length resting on the pad between the tabs
    loop_z_m: float = 0.66  # the frame's lower loop (about the hip / waist bottom level in IMG_8781)
    loop_half_width_m: float = 0.06
    loop_radius_m: float = 0.01  # the frame's tube (~2 cm)
    pole_radius_m: float = 0.02  # the pole (~4 cm in the photos)
    pole_gap_m: float = 0.0
    cradle_solref: tuple[float, float] = (0.02, 1.0)  # MuJoCo's default softness: rubber pad, unmeasured
    settle_s: float = 4.0
    raise0_m: float = 0.25  # the first raise before settling (feet clear while the legs hang out)

    def __post_init__(self) -> None:
        if self.attach not in ATTACHMENTS:
            raise ValueError(f"attach must be one of {ATTACHMENTS}")
        if self.clearance_m <= 0.0:
            raise ValueError("clearance_m must be > 0 (the feet off the floor)")

    @property
    def pole_x_m(self) -> float:
        """The pole's axis (reference frame): the loop's back surface + the pole's radius + the gap behind it."""
        return self.bar_x_m - self.loop_radius_m - self.pole_radius_m - self.pole_gap_m


# Session 8 (B6): the stand as HW1 found it. The real robot hung with the IMU at roll 1.8 / pitch -4.9 deg (the
# board-hold test's pre-stop lowstate, the 500 ms pause's recording; tilt 5.2-5.5 deg all day),
# the default StandSpec hangs it at pitch +1.7 (the IMU): 6.6 deg further forward, which halves the hips' gravity load
# at the board-hold pose (the model at the measured attitude gives the measured hold torques: hip pitch -2.83 vs -3.53 /
# -2.70, shoulder pitch -1.44 vs -1.48 / -1.86 N m). A pole 3.2 cm further forward (pole_gap_m -0.032: the harness loop
# rests on it with the robot leaning back) reproduces the attitude (IMU pitch -4.8 / -4.9 deg). It is the hanging
# stance's DEFAULT (the integrator's decision, 11 Oct: the sim matches the real robot by default); StandSpec() stays
# the session-6 photo estimate (`--stand s6`). The 1.8 deg roll is not modelled (the cradle is symmetric).
HW1_STAND = StandSpec(pole_gap_m=-0.032)


def _stand_of_world(w: world.World) -> StandSpec:
    """The StandSpec a caller's world was built with (stand_extra compiles the pole in at pole_x_m): HW1_STAND or the
    session-6 StandSpec(); any other stand must be passed explicitly (stand=...)."""
    bid = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, STAND_BODY)
    if bid < 0:
        raise ValueError("stance='hanging' with a caller's world: build it with extra=stand_extra(StandSpec(...))")
    x = float(w.model.body_pos[bid, 0])
    for spec in (HW1_STAND, StandSpec()):
        if abs(x - spec.pole_x_m) < 1e-9:
            return spec
    raise ValueError(f"the caller's world has a stand at pole x {x:.4f} m: pass the StandSpec it was built with")


@functools.lru_cache(maxsize=2)
def torso_reference(profile: str = "real") -> tuple[tuple[float, ...], tuple[float, ...]]:
    """torso_link's world position and rotation matrix (row-major) in the STANDING reference pose (the CoM-centred
    stand, feet on the floor) - the frame StandSpec's numbers are given in. Geoms fixed to the torso must be placed
    BEFORE the compile: MuJoCo's per-body mid-phase boxes are built then (a geom moved afterwards is not collided)."""
    from .model import centred_stand_pose

    w0 = world.build_world(profile, visual=False, with_scene=False)
    d = mujoco.MjData(w0.model)
    d.qpos[:] = w0.initial_qpos(centred_stand_pose(w0.model))
    mujoco.mj_kinematics(w0.model, d)
    tid = w0.model.body(TORSO_BODY).id
    return tuple(float(v) for v in d.xpos[tid]), tuple(float(v) for v in d.xmat[tid])


def stand_extra(stand: StandSpec) -> Any:
    """`world.build_world(extra=...)`: the stand body (mocap), its pole, the harness's loop and the attachment's
    equality constraint(s) with placeholder data - `place_stand` sets the real numbers after the compile (the
    anchors depend on the robot's pose, known only then)."""

    def extra(spec: mujoco.MjSpec, _info: Any) -> None:
        st = spec.worldbody.add_body()
        st.name = STAND_BODY
        st.mocap = True
        st.pos = [stand.pole_x_m, 0.0, stand.raise0_m]
        top = stand.bar_z_m - 0.04
        if stand.attach == "cradle":
            pole = st.add_geom()
            pole.name = "r1sim_stand_pole"
            pole.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            pole.fromto = [0.0, 0.0, 0.30, 0.0, 0.0, top]
            pole.size = [stand.pole_radius_m, 0.0, 0.0]
            pole.contype, pole.conaffinity = 0, STAND_COLLIDE_BIT
            pole.rgba = [0.08, 0.08, 0.08, 1.0]
            torso = spec.body(TORSO_BODY)
            loop = torso.add_geom()
            loop.name = "r1sim_harness_loop"
            loop.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            loop.size = [stand.loop_radius_m, stand.loop_half_width_m, 0.0]
            loop.contype, loop.conaffinity = STAND_COLLIDE_BIT, 0
            loop.density = 0.0  # massless: the robot's inertia is unchanged
            xt_, rt_ = torso_reference()
            xt, rt = np.asarray(xt_), np.asarray(rt_).reshape(3, 3)
            loop.pos = rt.T @ (np.array([stand.bar_x_m, 0.0, stand.loop_z_m]) - xt)
            qz = np.zeros(4)
            mujoco.mju_quatZ2Vec(qz, rt.T @ np.array([0.0, 1.0, 0.0]))  # the capsule's axis along the world's y
            loop.quat = qz
            loop.rgba = [0.05, 0.05, 0.05, 1.0]
            for k, side in enumerate((1.0, -1.0)):
                eq = spec.add_equality()
                eq.type = mujoco.mjtEq.mjEQ_CONNECT
                eq.objtype = mujoco.mjtObj.mjOBJ_BODY
                eq.name1, eq.name2 = TORSO_BODY, STAND_BODY
                eq.name = f"r1sim_cradle_{'left' if side > 0 else 'right'}"
                eq.solref = list(stand.cradle_solref)
                eq.data = np.zeros(11)
                del k
        else:
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY
            eq.name1, eq.name2 = "pelvis", STAND_BODY
            eq.name = "r1sim_stand_weld"
            data = np.zeros(11)
            data[6] = 1.0  # a relative pose given (set by place_stand); torquescale 1
            data[10] = 1.0
            eq.data = data

    return extra


def _quat_conj(q: FloatArray) -> FloatArray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def place_stand(m: Any, stand: StandSpec, qpos_hang: FloatArray) -> None:
    """Set the attachment's anchors / relative pose for the robot at `qpos_hang` (the reference stand raised by
    stand.raise0_m), the stand body at (pole_x, 0, raise0). (The loop geom was placed before the compile.)"""
    d = mujoco.MjData(m)
    d.qpos[:] = qpos_hang
    mujoco.mj_kinematics(m, d)
    sid = m.body(STAND_BODY).id
    stand_pos = np.asarray(m.body_pos[sid], dtype=np.float64)
    if stand.attach == "cradle":
        tid = m.body(TORSO_BODY).id
        xt, rt = d.xpos[tid].copy(), d.xmat[tid].reshape(3, 3).copy()
        dz = stand.raise0_m
        for name, side in (("r1sim_cradle_left", 1.0), ("r1sim_cradle_right", -1.0)):
            p = np.array([stand.bar_x_m, side * stand.bar_half_width_m, stand.bar_z_m + dz])
            e = m.equality(name).id
            m.eq_data[e, 0:3] = rt.T @ (p - xt)  # in torso_link's frame
            m.eq_data[e, 3:6] = p - stand_pos  # in the stand's frame (world-aligned)
    else:
        pid = m.body("pelvis").id
        xp, qp = d.xpos[pid].copy(), d.xquat[pid].copy()
        rp = d.xmat[pid].reshape(3, 3)
        e = m.equality("r1sim_stand_weld").id
        m.eq_data[e, 0:3] = 0.0
        m.eq_data[e, 3:6] = rp.T @ (stand_pos - xp)  # the stand's origin in the pelvis frame
        m.eq_data[e, 6:10] = _quat_conj(qp)  # the stand (world-aligned) in the pelvis frame


def foot_geoms(m: Any) -> list[int]:
    """The feet's collision geoms (bodies *ankle_roll*, any collision bit)."""
    out = []
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if "ankle_roll" in b and (int(m.geom_contype[g]) | int(m.geom_conaffinity[g])):
            out.append(g)
    return out


def lowest_point_z(m: Any, d: Any, geoms: list[int]) -> float:
    """The lowest point of these geoms (capsules / spheres exactly, others by their bounding sphere), world z."""
    z = float("inf")
    for g in geoms:
        c = d.geom_xpos[g]
        t = int(m.geom_type[g])
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ax = d.geom_xmat[g].reshape(3, 3)[:, 2]
            z = min(z, float(c[2] - abs(ax[2]) * m.geom_size[g, 1] - m.geom_size[g, 0]))
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = min(z, float(c[2] - m.geom_size[g, 0]))
        else:
            z = min(z, float(c[2] - m.geom_rbound[g]))
    return z


def builtin_command(kind: str, fixture: StandHold) -> LowCmd | None:
    """The emulated built-in's rt/lowcmd for `kind` (BUILTINS); None = it publishes nothing."""
    from .v3_core import DAMP_KD

    if kind in ("stand", "stand_lock"):
        return fixture.command
    if kind in ("none", "zero_torque_silent"):
        return None
    c = fixture.command
    c.hands = {}
    if kind == "zero_torque":
        c.kp[:] = 0.0
        c.kd[:] = 0.0
    elif kind == "damp":
        c.kp[:] = 0.0
        c.kd[:] = DAMP_KD
    elif kind == "damp_legs":
        c.kp[J.LEGS] = 0.0  # S5's EmulatedBuiltin(leg_kp=0.0): kd stays the stiff set
    else:
        raise ValueError(f"builtin must be one of {BUILTINS}")
    return c


class DdsBridge:
    """The DDS-bridge simulator (module docstring). A context manager: `with DdsBridge(domain=41) as br: ...`."""

    def __init__(
        self,
        *,
        domain: int,
        scene: Any = "floor",
        stance: str = "centred",
        safety: bool = False,
        seed: int = 0,
        rate_hz: float = 500.0,
        hand_rate_hz: float = 100.0,
        builtin: str | None = None,
        sensors: bool = True,
        imu_offset_deg: tuple[float, float] | None = None,
        truth_hz: float = 200.0,
        loopback: bool = True,
        w: world.World | None = None,
        stand: StandSpec | None = None,
        hand_timeout_s: float | None = None,
        driver: DriverConfig | None = None,
        board_tick: str = "sim",
    ) -> None:
        # session 9 (the stale agent, U19 / D25): the LowState tick's clock. "sim" (default, unchanged) = simulated
        # time x 1 kHz; "wall" = the bridge's monotonic clock x 1 kHz, as the real board's tick runs in real time - a
        # bridge slower than real time (RTF 0.4-0.6 with 11 agents on this Mac) then does not read as OLD state at
        # r1rt's board-tick content age (docs/sim/STATE_AGE.md); a delayed sample (lowstate_delay) keeps its tick.
        if board_tick not in ("sim", "wall"):
            raise ValueError("board_tick must be 'sim' or 'wall'")
        self.board_tick = board_tick
        if stance not in STANCES:
            raise ValueError(f"stance must be one of {STANCES}")
        hanging = stance == "hanging"
        if builtin is None:  # the stand's default: Unitree's power-on zero-torque state (the user, 10 Oct 2026)
            builtin = "zero_torque" if hanging else "stand"
        if builtin not in BUILTINS:
            raise ValueError(f"builtin must be one of {BUILTINS}")
        if stand is not None and not hanging:
            raise ValueError("a StandSpec needs stance='hanging'")
        if hanging and stand is None and w is not None:
            stand = _stand_of_world(w)  # a caller's world carries its stand (the pole is compiled in)
        # session 8 (the integrator's decision: the sim matches the real robot by default): the hanging stance's
        # default stand is HW1_STAND (HW1's measured hang attitude); StandSpec() = the session-6 photo estimate
        self.stand: StandSpec | None = (stand or HW1_STAND) if hanging else None
        if loopback:
            rpc.use_loopback()
        self.domain = int(domain)
        self.stance = stance
        self.safety = bool(safety)
        self.seed = int(seed)
        self.rate_hz = float(rate_hz)
        self.hand_rate_hz = float(hand_rate_hz)
        self.truth_hz = float(truth_hz)
        if w is None:
            w = world.build_world(
                "real",
                scene=scene,  # r1sim.scenes: "floor" (default) / "none" / "table" / a builder / "package.module:attr"
                extra=stand_extra(self.stand) if self.stand is not None else None,
            )
        elif self.stand is not None and mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, STAND_BODY) < 0:
            raise ValueError("stance='hanging' with a caller's world: build it with extra=stand_extra(StandSpec(...))")
        self.world = w
        m = copy.deepcopy(w.model)  # the configuration below never changes the caller's world
        if self.safety:  # U4 (DESIGN D16): the least stable plausible robot - no leg / waist stiction
            from .model import SAFETY_LEGS_FRICTIONLOSS

            for name in J.JOINTS[:14]:
                m.dof_frictionloss[m.joint(f"{name}_joint").dofadr[0]] = (
                    SAFETY_LEGS_FRICTIONLOSS[1] if "ankle" in name else SAFETY_LEGS_FRICTIONLOSS[0]
                )
        off = imu_offset_deg if imu_offset_deg is not None else (IMU_OFFSET_9OCT_DEG if stance == "9oct" else None)
        if off is not None:
            model.set_imu_offset(m, float(off[0]), float(off[1]))
        self.imu_offset_deg = off
        pose = J.POSE_STAND.copy()
        if stance == "9oct":
            pose[J.ARMS] = ARMS_TABLE_READY
        if hanging and builtin in ("stand", "stand_lock"):  # session 8 (#25): Unitree's fixed stand-lock preset
            self.fixture = StandHold(np.asarray(UNITREE_STAND_LOCK_POSE), hands=())
        else:
            self.fixture = StandHold(model.centred_stand_pose(m, pose), hands=())
        sm = SensorModel(SensorConfig.from_yaml(), seed=self.seed) if sensors else None
        # session 8 (B6): the motor board's input path. Default (None) = DriverConfig.real() = models/driver.yaml: the
        # HW1 board-hold release ~0.385 s after the last rt/lowcmd (then damping), the B2 delays - the sim matches the
        # real robot by default (the integrator, 11 Oct). Opt-out: driver=DriverConfig() = the S1 board law, no delay,
        # the last rt/lowcmd held forever (every bridge run before session 8); `--board hold` on the CLI
        driver = DriverConfig.real() if driver is None else driver
        self.driver = driver
        self.sim = Simulation(m, w.index, seed=self.seed, sensors=sm, driver=driver)
        self.sim.reset(w.initial_qpos(self.fixture.pose))
        self._present = np.asarray(w.index.present, dtype=bool)
        self._torso = m.body(TORSO_BODY).id
        self.builtin_cmd: LowCmd | None = builtin_command(builtin, self.fixture)
        self._feet = foot_geoms(m)
        hand_q = (
            {s: dex3.clamp(np.asarray(HAND_CLOSURE_09_LEFT) * (1.0 if s == "left" else dex3.MIRROR), s)
             for s in dex3.SIDES}
            if stance == "9oct"
            else {s: dex3.preset("open", s) for s in dex3.SIDES}
        )  # fmt: skip
        self._lock = threading.Lock()
        self._body: LowCmd = self.builtin_cmd.copy() if self.builtin_cmd is not None else LowCmd()
        self._hands: dict[str, HandCmd] = {s: hand_hold(q) for s, q in hand_q.items()} if w.index.hand_qadr else {}
        self._hand_from_dds: set[str] = set()
        self._compose_and_write()  # the board has the built-in's command before physics runs
        self.builtin_mode = builtin
        self.builtin_active = builtin != "none"  # Unitree's controller runs (silent or not) until ReleaseMode
        self.hang: dict[str, Any] = {}
        self._yaw0 = 0.0
        if self.stand is not None:
            self.hang = self._hang(w)
        self._kin = mujoco.MjData(m) if self.stand is not None else None
        self._kin_qadr = np.asarray(w.index.qadr, dtype=np.int64)
        free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        self._base_qadr = int(m.jnt_qposadr[free[0]]) if free else -1
        self.builtin_handle: int | None = None
        self.release_wall: float | None = None
        self.applied: deque[Applied] = deque(maxlen=2_000_000)
        self.truth: deque[Truth] = deque(maxlen=2_000_000)
        self.events: list[dict[str, Any]] = []
        self.stats: dict[str, int] = dict(
            lowstate_published=0,
            repeats_skipped=0,
            lowcmd_rx=0,
            lowcmd_bad_crc=0,
            lowcmd_applied=0,
            lowcmd_disabled_slots=0,
            lowcmd_wrong_mode_machine=0,
            hand_cmd_rx=0,
            hand_state_published=0,
            builtin_writes=0,
        )
        self._pub_gaps: list[float] = []
        self._pause_until = 0.0  # time.monotonic() until which rt/lowstate is withheld (inject "lowstate_gap")
        # session 6: scheduled faults (inject), their random draws on their own stream (reproducible per seed)
        self._windows: dict[str, list[FaultWindow]] = {}
        self._frng = np.random.default_rng(self.seed + 7919)
        self._delayed: deque[tuple[float, Any]] = deque()
        self._frozen_hand: dict[str, Any] = {}
        self._over_temp_t0: dict[int, float] = {}
        self.hand_timeout_s = hand_timeout_s
        self._hand_last_rx: dict[str, float] = {}
        self._hand_passive: set[str] = set()
        for k in ("lowstate_dropped", "lowstate_crc_corrupted", "lowstate_link_down", "lowcmd_dropped",
                  "lowcmd_corrupted", "lowcmd_link_down", "hand_cmd_ignored", "hand_state_withheld",
                  "publisher_stalls", "cmd_reader_stalls", "second_writer_writes"):  # fmt: skip
            self.stats[k] = 0
        # rt/lowcmd writer handle -> its DDS user_data ("r1mc3:r1rt" / "r1mc3:r1guard": proc's writers say who they are)
        self.writer_tags: dict[int, str] = {}
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._dds_up = False
        self.wall0: float | None = None

    # ------------------------------------------------------------------ the stand
    def _hang(self, w: world.World) -> dict[str, Any]:
        """Hang the robot on its stand (StandSpec): raise it, settle it under the built-in's command (lockstep, before
        real time), raise robot + stand together to the foot clearance, restart from the settled pose at rest."""
        stand = self.stand
        assert stand is not None
        m = self.sim.m
        free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        b = int(m.jnt_qposadr[free[0]])
        q = w.initial_qpos(self.fixture.pose)
        # session 8: hang by the TORSO - StandSpec's numbers are in the reference frame in which torso_link sits where
        # the CoM-centred stand puts it (torso_reference); a pose with other legs (Unitree's stand-lock preset) stood
        # with its feet centred would put the torso ~3.5 cm elsewhere, the harness loop inside the pole (no change for
        # the centred pose: the shift is 0 there)
        d0 = mujoco.MjData(m)
        d0.qpos[:] = q
        mujoco.mj_kinematics(m, d0)
        q[b : b + 3] += np.asarray(torso_reference("real")[0]) - d0.xpos[m.body(TORSO_BODY).id]
        q[b + 2] += stand.raise0_m
        if stand.attach == "cradle" and stand.pole_gap_m < 0.0:
            # a pole FORWARD of the upright loop (HW1_STAND: the robot leans back on it): start leaned back about the
            # bar by the angle at which the loop just touches the pole - started upright the loop would sit inside the
            # pole and the contact would throw a limp robot over it (73 deg)
            phi = -math.asin(min(1.0, -stand.pole_gap_m / (stand.bar_z_m - stand.loop_z_m)))
            pivot = np.array([stand.bar_x_m, 0.0, stand.bar_z_m + stand.raise0_m])
            rq = np.array([math.cos(phi / 2.0), 0.0, math.sin(phi / 2.0), 0.0])
            rm = np.zeros(9)
            mujoco.mju_quat2Mat(rm, rq)
            q[b : b + 3] = pivot + rm.reshape(3, 3) @ (q[b : b + 3] - pivot)
            qn = np.zeros(4)
            mujoco.mju_mulQuat(qn, rq, q[b + 3 : b + 7].copy())
            q[b + 3 : b + 7] = qn
        place_stand(m, stand, q)
        self.sim.reset(q)
        self._compose_and_write()
        # the built-in publishes all the time on the robot; in this lockstep settle nothing re-sends its command, so a
        # board command timeout (driver.cmd_timeout_s, session 8) is suspended for the settle only
        drv = self.sim.driver
        if drv.cmd_timeout_s is not None:
            self.sim.driver = dataclasses.replace(drv, cmd_timeout_s=None)
        try:
            self.sim.run_for(stand.settle_s)
        finally:
            self.sim.driver = drv
        d = self.sim.d
        qs = d.qpos.copy()
        body_dofs = [int(m.jnt_dofadr[m.joint(f"{n}_joint").id]) for n in J.JOINTS if self._present[J.IDX[n]]]
        dq_max = float(np.max(np.abs(d.qvel[body_dofs])))
        base_w = float(np.linalg.norm(d.qvel[int(m.jnt_dofadr[free[0]]) + 3 : int(m.jnt_dofadr[free[0]]) + 6]))
        kd = mujoco.MjData(m)
        kd.qpos[:] = qs
        mujoco.mj_kinematics(m, kd)
        z_settled = lowest_point_z(m, kd, self._feet)
        shift = stand.clearance_m - z_settled
        qs[b + 2] += shift
        m.body_pos[m.body(STAND_BODY).id, 2] += shift
        self.sim.reset(qs)
        self._compose_and_write()
        kd.qpos[:] = qs
        mujoco.mj_kinematics(m, kd)
        quat = qs[b + 3 : b + 7]
        rpy = np.degrees(_rpy(quat))
        self._yaw0 = float(rpy[2])
        return dict(
            attach=stand.attach,
            builtin=self.builtin_mode if hasattr(self, "builtin_mode") else None,
            settle_s=stand.settle_s,
            raise_m=round(stand.raise0_m + shift, 4),
            foot_z_min_m=round(lowest_point_z(m, kd, self._feet), 4),
            tilt_deg=round(tilt_deg(quat), 3),
            pitch_deg=round(float(rpy[1]), 3),
            roll_deg=round(float(rpy[0]), 3),
            dq_max_at_settle=round(dq_max, 4),
            base_w_at_settle=round(base_w, 4),
            pelvis_z_m=round(float(qs[b + 2]), 4),
            spec=asdict(stand),
        )

    def hang_state(self) -> dict[str, Any]:
        """Now (ground truth): the lowest foot point (m), tilt / pitch / roll / yaw (deg) - stance hanging only."""
        st = self.sim.read_state()
        return self._hang_truth(st)

    def _hang_truth(self, st: LowState) -> dict[str, Any]:
        kd = self._kin
        if kd is None:
            return {}
        b = self._base_qadr
        kd.qpos[b : b + 3] = st.base_pos
        kd.qpos[b + 3 : b + 7] = st.base_quat
        q = np.asarray(st.truth.get("q", st.q), dtype=np.float64)
        pres = self._present
        kd.qpos[self._kin_qadr[pres]] = q[pres]
        mujoco.mj_kinematics(self.sim.m, kd)
        rpy = np.degrees(_rpy(np.asarray(st.base_quat, dtype=np.float64)))
        yaw = (float(rpy[2]) - self._yaw0 + 180.0) % 360.0 - 180.0
        return dict(
            foot_z_min=lowest_point_z(self.sim.m, kd, self._feet),
            tilt_deg=tilt_deg(st.base_quat),
            pitch_deg=float(rpy[1]),
            roll_deg=float(rpy[0]),
            yaw_deg=yaw,
        )

    # ------------------------------------------------------------------ DDS set-up
    def _setup_dds(self) -> None:
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        dp = DomainParticipant(self.domain)
        self.dp = dp
        self._w_state = DataWriter(dp, Topic(dp, hg.TOPIC_LOWSTATE, hg.LowState_), qos=rpc.WRITER_QOS)
        self._r_cmd = DataReader(dp, Topic(dp, hg.TOPIC_LOWCMD, hg.LowCmd_), qos=rpc.READER_QOS)
        self._w_hand = {
            "left": DataWriter(dp, Topic(dp, hg.TOPIC_DEX3_LEFT_STATE, hg.HandState_), qos=rpc.WRITER_QOS),
            "right": DataWriter(dp, Topic(dp, hg.TOPIC_DEX3_RIGHT_STATE, hg.HandState_), qos=rpc.WRITER_QOS),
        }
        self._r_hand = {
            "left": DataReader(dp, Topic(dp, hg.TOPIC_DEX3_LEFT_CMD, hg.HandCmd_), qos=rpc.READER_QOS),
            "right": DataReader(dp, Topic(dp, hg.TOPIC_DEX3_RIGHT_CMD, hg.HandCmd_), qos=rpc.READER_QOS),
        }
        # Unitree's built-in: its own participant (a foreign writer on rt/lowcmd for r1rt)
        bdp = DomainParticipant(self.domain)
        self._bdp = bdp
        self._w_builtin = DataWriter(bdp, Topic(bdp, hg.TOPIC_LOWCMD, hg.LowCmd_), qos=rpc.WRITER_QOS)
        self.builtin_handle = instance_handle(self._w_builtin)
        from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication

        self._r_pubs = BuiltinDataReader(dp, BuiltinTopicDcpsPublication)
        self.switcher = rpc.MotionSwitcherServer(
            mode=lambda: BUILTIN_MODE_NAME if self.builtin_active else "",
            release=self._release_builtin,
            select=lambda name: False,  # not emulated (Unitree brings it back in damping: not wanted in a rehearsal)
            participant=dp,
        )
        self._dds_up = True

    # ------------------------------------------------------------------ the board
    def _compose_and_write(self, *, body: bool = True) -> None:
        """Under the lock: the latest body command + the latest command per hand -> the engine. body=False (a hand
        command arrived, session 8): the engine takes the hands only - a Dex3 command is not an rt/lowcmd, so it must
        not refresh the body board's command timeout (B6: the release ~0.385 s after the last rt/lowcmd)."""
        c = self._body.copy()
        c.hands = {s: h.copy() for s, h in self._hands.items()}
        self.sim.write_cmd(c, body=body)

    def _apply_lowcmd(self, msg: Any, writer: int, *, corrupt: bool = False) -> None:
        self.stats["lowcmd_rx"] += 1
        crc_ok = not corrupt and crc_lowcmd(msg) == int(msg.crc)
        now = time.monotonic()
        if not crc_ok:
            self.stats["lowcmd_bad_crc"] += 1
            self.applied.append(Applied(now, self.sim.time, writer, float("nan"), 0, int(msg.mode_machine), False))
            return
        if int(msg.mode_machine) != J.MODE_MACHINE:
            self.stats["lowcmd_wrong_mode_machine"] += 1  # applied anyway (the real board's reaction is unknown)
        body, disabled = body_cmd_of(msg)
        self.stats["lowcmd_disabled_slots"] += disabled
        with self._lock:
            self._body = body
            self._compose_and_write()
        self.stats["lowcmd_applied"] += 1
        legs_disabled = sum(1 for j in J.LEGS if int(msg.motor_cmd[int(J.SLOT[j])].mode) & 0x01 == 0)
        self.applied.append(
            Applied(now, self.sim.time, writer, float(np.min(body.kp[J.LEGS])), legs_disabled, int(msg.mode_machine),
                    True)
        )  # fmt: skip

    def _scan_publications(self) -> None:
        """Learn the user_data of rt/lowcmd writers from the builtin DCPSPublication topic (its instance handle is the
        publication handle our reader's samples carry)."""
        found: list[Any] = self._r_pubs.take(N=64)
        for e in found:
            if getattr(e, "topic_name", None) != hg.TOPIC_LOWCMD:
                continue
            tag = ""
            for pol in e.qos:
                if type(pol).__name__ == "Userdata":
                    tag = bytes(pol.data).decode("utf-8", "replace")
            self.writer_tags[int(e.sample_info.instance_handle)] = tag

    def _rx_loop(self) -> None:
        k = 0
        while not self._stop.is_set():
            k += 1
            now = time.monotonic()
            stall = self._on("cmd_reader_stall", now)
            if stall is not None:  # this thread blocked (starved): samples queue in the reader, the board holds
                self._stall(stall, now, "cmd_reader_stalls")
                continue
            if k % 50 == 1:
                self._scan_publications()
            got = 0
            link = self._on("link_down", now)
            drop = self._on("lowcmd_drop", now)
            bad = self._on("lowcmd_crc", now)
            samples: list[Any] = self._r_cmd.take(N=64)
            for s in samples:
                if isinstance(s, hg.LowCmd_):
                    got += 1
                    if link is not None:  # the link is down: the board never sees it
                        self._hit(link, now)
                        self.stats["lowcmd_link_down"] += 1
                        continue
                    if drop is not None and self._frng.random() < float(drop.params["fraction"]):
                        self._hit(drop, now)
                        self.stats["lowcmd_dropped"] += 1
                        continue
                    corrupt = bad is not None and self._frng.random() < float(bad.params["fraction"])
                    if corrupt:
                        assert bad is not None
                        self._hit(bad, now)
                        self.stats["lowcmd_corrupted"] += 1
                    info: Any = getattr(s, "sample_info", None)  # set on every taken sample by CycloneDDS
                    self._apply_lowcmd(s, int(info.publication_handle) if info is not None else 0, corrupt=corrupt)
            for side, r in self._r_hand.items():
                hs: list[Any] = r.take(N=16)
                dead = self._hand_fault("dex3_death", side, now)
                for h in hs:
                    if isinstance(h, hg.HandCmd_):
                        self.stats["hand_cmd_rx"] += 1
                        if dead is not None:  # dex3_server gone: nobody passes the command to the hand
                            self._hit(dead, now)
                            self.stats["hand_cmd_ignored"] += 1
                            continue
                        with self._lock:
                            self._hands[side] = hand_cmd_of(h)
                            self._hand_from_dds.add(side)
                            self._hand_last_rx[side] = now
                            if side in self._hand_passive:
                                self._hand_passive.discard(side)
                                self.events.append(
                                    dict(wall=now, sim_t=self.sim.time, what=f"hand {side} driven again")
                                )
                            self._compose_and_write(body=False)
            self._hand_timeouts(now)
            if not got:
                time.sleep(0.0003)

    def _hand_timeouts(self, now: float) -> None:
        """The Dex3 timeout bit (opt-in; v2's knowledge, not measured): a hand that had DDS commands goes passive
        (kp 0, kd 0) `timeout` s after its last one, until the next arrives."""
        tw = self._on("dex3_timeout", now)
        timeout = float(tw.params.get("timeout_s", 1.0)) if tw is not None else self.hand_timeout_s
        if timeout is None:
            return
        for side in list(self._hand_from_dds):
            if side in self._hand_passive or now - self._hand_last_rx.get(side, now) <= timeout:
                continue
            with self._lock:
                self._hands[side] = HandCmd()
                self._hand_passive.add(side)
                self._compose_and_write(body=False)
            if tw is not None:
                self._hit(tw, now)
            self.events.append(
                dict(wall=now, sim_t=self.sim.time, what=f"hand {side} passive (Dex3 timeout {timeout} s)")
            )

    # ------------------------------------------------------------------ publishers
    def _pub_loop(self) -> None:
        period = 1.0 / self.rate_hz
        hand_every = max(1, round(self.rate_hz / self.hand_rate_hz))
        nxt = time.monotonic()
        k = 0
        last_seq = -1
        last_pub: float | None = None
        while not self._stop.is_set():
            now = time.monotonic()
            stall = self._on("publisher_stall", now)
            if stall is not None:  # this thread blocked (starved): nothing published, physics runs on
                self._stall(stall, now, "publisher_stalls")
                nxt = time.monotonic()
                continue
            st = self.sim.read_state()
            gap = self._on("lowstate_gap", now)
            link = self._on("link_down", now)
            if now < self._pause_until or gap is not None:
                if gap is not None:
                    self._hit(gap, now)
                # a link gap: nothing published (inject "lowstate_gap")
            elif st.seq != last_seq:
                last_seq = st.seq
                if link is not None:
                    self._hit(link, now)
                    self.stats["lowstate_link_down"] += 1
                else:
                    raw = lowstate_msg(st, self._present)
                    if self.board_tick == "wall":  # session 9: the real board's real-time tick (see __init__)
                        raw.tick = int(time.monotonic() * 1000.0) & 0xFFFFFFFF
                        raw.crc = crc_lowstate(raw)
                    msg = self._faulted_lowstate(raw, now)
                    if msg is not None:
                        self._w_state.write(msg)
                        t_w = time.monotonic()
                        if last_pub is not None:
                            self._pub_gaps.append(t_w - last_pub)
                        last_pub = t_w
                        self.stats["lowstate_published"] += 1
                if k % hand_every == 0 and st.hand_q:
                    for side, wr in self._w_hand.items():
                        if side in st.hand_q:
                            hm = self._faulted_hand(side, st, now)
                            if hm is not None:
                                wr.write(hm)
                                self.stats["hand_state_published"] += 1
            else:
                self.stats["repeats_skipped"] += 1
            k += 1
            nxt += period
            slp = nxt - time.monotonic()
            if slp > 0:
                time.sleep(slp)
            elif slp < -0.1:
                nxt = time.monotonic()

    def _faulted_lowstate(self, msg: Any, now: float) -> Any | None:
        """rt/lowstate through the active faults: motor error words / over-temperature (CRC recomputed), drops, a
        corrupted CRC, a delay (FIFO). None = nothing to publish this tick."""
        changed = False
        me = self._on("motor_error", now)
        if me is not None:
            self._hit(me, now)
            j = _joint_index(me.params["joint"])
            msg.motor_state[int(J.SLOT[j])].motorstate = int(me.params.get("word", 1))
            changed = True
        ot = self._on("over_temp", now)
        if ot is not None:
            self._hit(ot, now)
            j = _joint_index(ot.params["joint"])
            m = msg.motor_state[int(J.SLOT[j])]
            t_start = self._over_temp_t0.setdefault(j, float(m.temperature[0]))
            assert ot.first_wall is not None
            t_c = min(
                float(ot.params.get("to_c", 80.0)),
                t_start + float(ot.params.get("rate_c_s", 2.0)) * (now - ot.first_wall),
            )
            m.temperature = [round(t_c), round(t_c)]
            changed = True
        elif self._over_temp_t0:
            self._over_temp_t0.clear()
        if changed:
            msg.crc = crc_lowstate(msg)
        drop = self._on("lowstate_drop", now)
        if drop is not None and self._frng.random() < float(drop.params["fraction"]):
            self._hit(drop, now)
            self.stats["lowstate_dropped"] += 1
            return None
        bad = self._on("lowstate_crc", now)
        if bad is not None and self._frng.random() < float(bad.params["fraction"]):
            self._hit(bad, now)
            msg.crc = int(msg.crc) ^ 0x1
            self.stats["lowstate_crc_corrupted"] += 1
        dl = self._on("lowstate_delay", now)
        if dl is None:
            if self._delayed:
                self._delayed.clear()  # the delay ended: the backlog is stale, publish fresh states again
            return msg
        self._hit(dl, now)
        self._delayed.append((now, msg))
        due = now - float(dl.params["delay_ms"]) * 1e-3
        out = None
        while self._delayed and self._delayed[0][0] <= due:
            out = self._delayed.popleft()[1]  # the newest state that is old enough (older ones are superseded)
        return out

    def _faulted_hand(self, side: str, st: LowState, now: float) -> Any | None:
        dead = self._hand_fault("dex3_death", side, now)
        if dead is not None:
            self._hit(dead, now)
            self.stats["hand_state_withheld"] += 1
            return None
        fr = self._hand_fault("hand_freeze", side, now)
        if fr is not None:
            self._hit(fr, now)
            if side not in self._frozen_hand:
                self._frozen_hand[side] = hand_state_msg(st, side)
            return self._frozen_hand[side]
        self._frozen_hand.pop(side, None)
        return hand_state_msg(st, side)

    def _builtin_loop(self) -> None:
        if self.builtin_cmd is None:
            return  # "zero_torque_silent" / "none": Unitree's controller publishes nothing
        period = 1.0 / BUILTIN_HZ
        msg = lowcmd_msg(self.builtin_cmd)
        nxt = time.monotonic()
        while not self._stop.is_set() and self.builtin_active:
            self._w_builtin.write(msg)
            self.stats["builtin_writes"] += 1
            nxt += period
            slp = nxt - time.monotonic()
            if slp > 0:
                time.sleep(slp)
            elif slp < -0.1:
                nxt = time.monotonic()

    def _release_builtin(self) -> bool:
        if self.builtin_active:
            self.builtin_active = False
            self.release_wall = time.monotonic()
            self.events.append(dict(wall=self.release_wall, sim_t=self.sim.time, what="ReleaseMode: built-in stops"))
        return True

    def _truth_loop(self) -> None:
        period = 1.0 / self.truth_hz
        nxt = time.monotonic()
        while not self._stop.is_set():
            st = self.sim.read_state()
            if self._kin is not None:  # the stand: the feet's clearance and the attitude too
                h = self._hang_truth(st)
                self.truth.append(
                    Truth(time.monotonic(), float(st.t), h["tilt_deg"], float(st.base_pos[2]), h["foot_z_min"],
                          h["pitch_deg"], h["roll_deg"], h["yaw_deg"])
                )  # fmt: skip
            else:
                self.truth.append(Truth(time.monotonic(), float(st.t), tilt_deg(st.base_quat), float(st.base_pos[2])))
            nxt += period
            slp = nxt - time.monotonic()
            if slp > 0:
                time.sleep(slp)
            elif slp < -0.1:
                nxt = time.monotonic()

    # ------------------------------------------------------------------ life cycle
    def start(self) -> DdsBridge:
        if not self._dds_up:
            self._setup_dds()
        self._stop.clear()
        self.wall0 = time.monotonic()
        self.sim.start_realtime()
        self.switcher.start()
        targets = [("rx", self._rx_loop), ("pub", self._pub_loop), ("truth", self._truth_loop)]
        if self.builtin_active:
            targets.append(("builtin", self._builtin_loop))
        for name, fn in targets:
            th = threading.Thread(target=fn, name=f"dds-bridge-{name}", daemon=True)
            th.start()
            self._threads.append(th)
        return self

    def stop(self) -> None:
        self._stop.set()
        self.builtin_active = False
        for th in self._threads:
            th.join(2.0)
        self._threads.clear()
        if self._dds_up:
            self.switcher.stop()
        self.sim.stop()

    def __enter__(self) -> DdsBridge:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ queries / faults
    def tilt_deg(self) -> float:
        """The pelvis's tilt now (ground truth, deg)."""
        return tilt_deg(self.sim.read_state().base_quat)

    def push(self, impulse_ns: float, *, direction: tuple[float, float, float] = (-1.0, 0.0, 0.0),
             duration_s: float = PUSH_S) -> None:  # fmt: skip
        """An impulse on the torso (N·s; default backwards, as A7's deterministic nudge) spread over `duration_s`
        from now (a force set in d.xfrc_applied and cleared by a timer; recorded in `events`)."""
        f = np.asarray(direction, dtype=np.float64) * (float(impulse_ns) / duration_s)
        d = self.sim.d
        t0w, t0s = time.monotonic(), self.sim.time
        d.xfrc_applied[self._torso, :3] = f

        def clear() -> None:
            d.xfrc_applied[self._torso, :3] = 0.0
            self.events.append(dict(wall=time.monotonic(), sim_t=self.sim.time, what="push end"))

        threading.Timer(duration_s, clear).start()
        self.events.append(dict(wall=t0w, sim_t=t0s, what=f"push {impulse_ns} N s along {list(direction)}"))

    def inject(
        self,
        kind: str,
        *,
        at_s: float = 0.0,
        duration_s: float | None = None,
        every_s: float | None = None,
        count: int | None = None,
        **params: Any,
    ) -> FaultWindow | None:
        """A fault on the robot's side, scheduled: from `at_s` s from now (wall), for `duration_s` (None = until
        `clear(kind)`), repeated every `every_s` s up to `count` times. Kinds: "push" (impulse_ns, direction,
        duration_s: see `push`; at once or after at_s) and FAULT_KINDS (their parameters in the tuple's comments).
        Every window is recorded (`faults()`: the schedule + when the bridge actually applied it; `events`)."""
        now = time.monotonic()
        if kind == "push":
            kw = {k: v for k, v in params.items() if k != "impulse_ns"}
            if duration_s is not None:
                kw["duration_s"] = duration_s
            if at_s > 0.0:
                threading.Timer(at_s, self.push, args=(float(params["impulse_ns"]),), kwargs=kw).start()
            else:
                self.push(float(params["impulse_ns"]), **kw)
            return None
        if kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault {kind!r} (known: push, {', '.join(FAULT_KINDS)})")
        need = {
            "lowstate_drop": ("fraction",), "lowstate_crc": ("fraction",), "lowcmd_drop": ("fraction",),
            "lowcmd_crc": ("fraction",), "lowstate_delay": ("delay_ms",), "motor_error": ("joint",),
            "over_temp": ("joint",),
        }.get(kind, ())  # fmt: skip
        missing = [n for n in need if n not in params]
        if missing:
            raise ValueError(f"fault {kind!r} needs {missing}")
        if kind in ("lowstate_gap", "publisher_stall", "cmd_reader_stall") and duration_s is None:
            raise ValueError(f"fault {kind!r} needs duration_s")
        if kind in ("motor_error", "over_temp"):
            _joint_index(params["joint"])  # validated now, not in the publisher thread
        if kind in ("hand_freeze", "dex3_death") and params.setdefault("side", "both") not in ("left", "right", "both"):
            raise ValueError("side must be 'left', 'right' or 'both'")
        if kind == "second_writer" and params.setdefault("payload", "stand") not in ("stand", "damp", "zero"):
            raise ValueError("second_writer payload must be 'stand', 'damp' or 'zero'")
        f = FaultWindow(kind, dict(params), now + float(at_s), duration_s, every_s, count)
        if kind in ("publisher_stall", "cmd_reader_stall"):
            f.stalls = []
        self._windows.setdefault(kind, []).append(f)
        what = "lowstate gap" if kind == "lowstate_gap" else f"inject {kind}"  # S5's event text kept for the gap
        self.events.append(
            dict(wall=now, sim_t=self.sim.time, what=f"{what} {params} at +{at_s} s for {duration_s} s"
                 + (f" every {every_s} s x {count}" if every_s else ""))
        )  # fmt: skip
        if kind == "second_writer":
            th = threading.Thread(target=self._second_writer, args=(f,), name="dds-bridge-second-writer", daemon=True)
            th.start()
            self._threads.append(th)
        return f

    def clear(self, kind: str) -> None:
        """End every window of this fault kind now."""
        for f in self._windows.get(kind, []):
            f.cleared = True
        self.events.append(dict(wall=time.monotonic(), sim_t=self.sim.time, what=f"clear {kind}"))

    def faults(self) -> list[dict[str, Any]]:
        """Every scheduled fault window with its actual application times (relative to `wall0` where given)."""
        out = []
        w0 = self.wall0 or 0.0
        for ws in self._windows.values():
            for f in ws:
                e = asdict(f)
                for key in ("t0", "first_wall", "last_wall"):
                    if e[key] is not None:
                        e[key + "_rel"] = round(e[key] - w0, 4)
                if f.stalls is not None:
                    e["stalls"] = [(round(a - w0, 4), round(b - a, 5)) for a, b in f.stalls[-2000:]]
                    e["n_stalls"] = len(f.stalls)
                out.append(e)
        return out

    def _on(self, kind: str, now: float) -> FaultWindow | None:
        ws = self._windows.get(kind)
        if not ws:
            return None
        for f in ws:
            if f.active(now):
                return f
        return None

    def _hand_fault(self, kind: str, side: str, now: float) -> FaultWindow | None:
        f = self._on(kind, now)
        if f is None or f.params.get("side", "both") not in ("both", side):
            return None
        return f

    def _hit(self, f: FaultWindow, now: float) -> None:
        f.hits += 1
        if f.first_wall is None:
            f.first_wall, f.first_sim = now, self.sim.time
        f.last_wall, f.last_sim = now, self.sim.time

    def _stall(self, f: FaultWindow, now: float, stat: str) -> None:
        """Block the calling bridge thread until the window ends (a starved thread); the actual times recorded."""
        self._hit(f, now)
        end = f.window_end(now)
        while not self._stop.is_set():
            left = end - time.monotonic()
            if left <= 0:
                break
            time.sleep(min(left, 0.05))
        t1 = time.monotonic()
        if f.stalls is not None and len(f.stalls) < 100_000:
            f.stalls.append((now, t1))
        f.last_wall, f.last_sim = t1, self.sim.time
        self.stats[stat] += 1

    def _second_writer(self, f: FaultWindow) -> None:
        """A foreign participant on rt/lowcmd (inject "second_writer"): its own DomainParticipant, user_data
        "r1sim:second_writer" (NOT "r1mc3:": r1rt takes it for a foreign writer, as it would take any other program
        on the robot), publishing its payload at rate_hz during the windows."""
        from cyclonedds.core import Policy, Qos
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic
        from cyclonedds.util import duration

        from .v3_core import DAMP_KD

        payload = str(f.params.get("payload", "stand"))
        c = self.fixture.command
        c.hands = {}
        if payload != "stand":
            c.kp[:] = 0.0
            c.kd[:] = DAMP_KD if payload == "damp" else 0.0
        msg = lowcmd_msg(c)
        dp = DomainParticipant(self.domain)
        qos = Qos(
            Policy.Reliability.Reliable(duration(milliseconds=100)),
            Policy.History.KeepLast(16),
            Policy.Userdata(data=b"r1sim:second_writer"),
        )
        wr = DataWriter(dp, Topic(dp, hg.TOPIC_LOWCMD, hg.LowCmd_), qos=qos)
        f.params["writer_handle"] = instance_handle(wr)
        period = 1.0 / float(f.params.get("rate_hz", 500.0))
        end = f.end()
        nxt = time.monotonic()
        while not self._stop.is_set() and not f.cleared:
            now = time.monotonic()
            if end is not None and now > end:
                break
            if f.active(now):
                wr.write(msg)
                self._hit(f, now)
                self.stats["second_writer_writes"] += 1
            nxt += period
            slp = nxt - time.monotonic()
            if slp > 0:
                time.sleep(slp)
            elif slp < -0.1:
                nxt = time.monotonic()

    def writers(self) -> dict[int, dict[str, Any]]:
        """Per writer handle of rt/lowcmd: the label ("builtin" = the emulated built-in; a writer whose DDS user_data
        is "r1mc3:<role>" = <role>, e.g. "r1rt" / "r1guard"; else "w<k>" in order of first appearance), how many
        samples were applied, first / last wall + sim time, its user_data tag."""
        out: dict[int, dict[str, Any]] = {}
        for a in snapshot(self.applied):
            e = out.get(a.writer)
            if e is None:
                tag = self.writer_tags.get(a.writer, "")
                if a.writer == self.builtin_handle:
                    lab = "builtin"
                elif tag.startswith(("r1mc3:", "r1sim:")):
                    lab = tag.split(":", 1)[1]  # proc's writers: "r1rt" / "r1guard"; inject's "second_writer"
                else:
                    lab = f"w{sum(1 for x in out.values() if x['label'] != 'builtin')}"
                e = out[a.writer] = dict(label=lab, tag=tag, n=0, first_wall=a.wall, first_sim=a.sim_t)
            e["n"] += 1
            e["last_wall"], e["last_sim"] = a.wall, a.sim_t
        return out

    def log(self) -> dict[str, Any]:
        gaps = np.asarray(self._pub_gaps[-20000:], dtype=np.float64)
        rt = self.sim.rt
        return dict(
            domain=self.domain,
            stance=self.stance,
            safety=self.safety,
            seed=self.seed,
            imu_offset_deg=self.imu_offset_deg,
            rate_hz=self.rate_hz,
            stats=dict(self.stats),
            lowstate_rate_hz=round(float(1.0 / gaps.mean()), 1) if gaps.size else None,
            lowstate_gap_p99_ms=round(float(np.percentile(gaps, 99) * 1e3), 2) if gaps.size else None,
            lowstate_gap_max_ms=round(float(gaps.max() * 1e3), 2) if gaps.size else None,
            physics=dict(
                rtf=round(rt.achieved_rtf, 3),
                max_lag_s=round(rt.max_lag_s, 4),
                rephases=rt.rephases,
                overruns=rt.overruns,
            ),
            builtin_active=self.builtin_active,
            release_wall=self.release_wall,
            writers={str(k): v for k, v in self.writers().items()},
            events=list(self.events),
            switcher_requests=list(self.switcher.log) if self._dds_up else [],
            builtin=self.builtin_mode,
            stand=self.hang or None,
            faults=self.faults(),
            board=dict(  # session 8 (B6): the board's command timeout (None = holds forever) and how often it fired
                cmd_timeout_s=self.sim.driver.cmd_timeout_s,
                on_cmd_timeout=self.sim.driver.on_cmd_timeout,
                cmd_delay_s=self.sim.driver.cmd_delay_s,
                timeouts=self.sim.timeouts,
            ),
        )

    def records(self) -> dict[str, Any]:
        """Everything recorded (for --record / the A7 harness): the summary + every applied command + the truth."""
        out = self.log()
        out["applied"] = [asdict(a) for a in snapshot(self.applied)]
        out["truth"] = [asdict(t) for t in snapshot(self.truth)]
        return out


# ---------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1sim.dds_bridge", description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", type=int, required=True)
    ap.add_argument(
        "--scene",
        default="floor",
        help="the world around the robot (r1sim.scenes): floor (default), none, table, or package.module:attr",
    )
    ap.add_argument("--stance", choices=STANCES, default="centred")
    ap.add_argument("--safety", action="store_true", help="U4: no leg stiction (every stall / fall safety test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate-hz", type=float, default=500.0)
    ap.add_argument(
        "--builtin",
        choices=BUILTINS,
        default=None,
        help="default: stand (standing stances), zero_torque (hanging: Unitree's power-on state)",
    )
    ap.add_argument("--attach", choices=ATTACHMENTS, default="cradle", help="hanging: the stand's attachment")
    ap.add_argument(
        "--stand",
        choices=("hw1", "s6"),
        default="hw1",
        help="hanging: hw1 (default) = HW1_STAND (HW1's measured attitude); s6 = the session-6 photo estimate",
    )
    ap.add_argument("--clearance", type=float, default=StandSpec.clearance_m, help="hanging: the feet's height (m)")
    ap.add_argument("--hand-timeout-s", type=float, default=None, help="opt-in Dex3 timeout (v2's ~1 s)")
    ap.add_argument("--no-sensors", action="store_true")
    ap.add_argument(
        "--board",
        choices=("hold", "real"),
        default="real",
        help="the motor board's input path: real (default) = models/driver.yaml (the HW1 board-hold release after "
        "~0.385 s, the fitted delays); hold = the last rt/lowcmd forever, no delay (every bridge run before session 8)",
    )
    ap.add_argument("--record", type=Path, default=None, help="write the records (JSON) here at exit")
    ap.add_argument("--seconds", type=float, default=None, help="run this long, then exit (default: until SIGINT)")
    a = ap.parse_args(argv)
    base = HW1_STAND if a.stand == "hw1" else StandSpec()
    stand = dataclasses.replace(base, attach=a.attach, clearance_m=a.clearance) if a.stance == "hanging" else None
    br = DdsBridge(domain=a.domain, scene=a.scene, stance=a.stance, safety=a.safety, seed=a.seed, rate_hz=a.rate_hz,
                   builtin=a.builtin, sensors=not a.no_sensors, stand=stand,
                   hand_timeout_s=a.hand_timeout_s,
                   driver=DriverConfig.real() if a.board == "real" else DriverConfig())  # fmt: skip
    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    br.start()
    print(json.dumps(dict(ready=True, domain=a.domain, stance=a.stance, safety=a.safety, seed=a.seed,
                          rate_hz=a.rate_hz, builtin=br.builtin_mode, stand=br.hang or None)), flush=True)  # fmt: skip
    done.wait(a.seconds)
    br.stop()
    summary = br.log()
    print(json.dumps(dict(stopped=True, stats=summary["stats"], lowstate_rate_hz=summary["lowstate_rate_hz"],
                          physics=summary["physics"])), flush=True)  # fmt: skip
    if a.record is not None:
        a.record.write_text(json.dumps(br.records()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
