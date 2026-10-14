"""Unitree's hg messages (LowCmd_ / LowState_ / HandCmd_ / HandState_ ...) for CycloneDDS's Python binding.

Generated from OUR IDL (`rt/idl/src/unitree/idl/hg/*.idl`, proven byte-identical to Unitree's generated C++ headers,
rt/idl/SOURCE.md) by CycloneDDS 0.10.2's idlc Python backend (`idlc -x final -l <cyclonedds>/_idlpy.so`, 10 Oct 2026,
v3 session 5) and merged into one module: the type names (`unitree_hg::msg::dds_::X_`), field order, field types and
`@final` extensibility are the generator's; only the forward references became direct class references. The C++ side
uses the same IDL, so a Python peer and r1rt / r1guard exchange the same bytes (the bridge's cross-language test).

The CRC (`lowcmd_crc`, `lowstate_crc`) is Unitree's word CRC over the C struct image (unitree_sdk2 crc.h crc32_core =
CRC-32/MPEG-2 over the big-endian bytes of each little-endian 32-bit word, all but the last word): the struct formats
are tools/crc_golden.py's (`FMT_HG_LOWCMD` / `FMT_HG_LOWSTATE`, read from unitree_sdk2py utils/crc.py), checked
against tests/assets/crc_golden/ (tests/unit/test_dds_hg.py). Table-driven: ~0.2 ms per message in CPython.
"""

# No `from __future__ import annotations`: CycloneDDS's IDL reads the field annotations at run time.
import struct
from dataclasses import dataclass
from typing import Any

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

# Topic names (unitree_sdk2: the hg robots' low-level topics; the hands' topics as dex3_server serves them).
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_ARM_SDK = "rt/arm_sdk"
TOPIC_DEX3_LEFT_CMD = "rt/dex3/left/cmd"
TOPIC_DEX3_RIGHT_CMD = "rt/dex3/right/cmd"
TOPIC_DEX3_LEFT_STATE = "rt/dex3/left/state"
TOPIC_DEX3_RIGHT_STATE = "rt/dex3/right/state"

N_SLOTS = 35  # hg motor slots (the R1 uses 26 of them: r1sim.joints / R1RT_HG_SLOT_MAP_INIT)
N_DEX3_MOTORS = 7
N_DEX3_PRESS = 9  # pressure sensor modules per Dex3-1 hand (12 taxels each)


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.MotorCmd_"):
    mode: types.uint8
    q: types.float32
    dq: types.float32
    tau: types.float32
    kp: types.float32
    kd: types.float32
    reserve: types.uint32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.MotorState_"):
    mode: types.uint8
    q: types.float32
    dq: types.float32
    ddq: types.float32
    tau_est: types.float32
    temperature: types.array[types.int16, 2]
    vol: types.float32
    sensor: types.array[types.uint32, 2]
    motorstate: types.uint32
    reserve: types.array[types.uint32, 4]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class IMUState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.IMUState_"):
    quaternion: types.array[types.float32, 4]
    gyroscope: types.array[types.float32, 3]
    accelerometer: types.array[types.float32, 3]
    rpy: types.array[types.float32, 3]
    temperature: types.int16


@dataclass
@annotate.final
@annotate.autoid("sequential")
class PressSensorState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.PressSensorState_"):
    pressure: types.array[types.float32, 12]
    temperature: types.array[types.float32, 12]
    lost: types.uint32
    reserve: types.uint32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowCmd_"):
    mode_pr: types.uint8
    mode_machine: types.uint8
    motor_cmd: types.array[MotorCmd_, 35]
    reserve: types.array[types.uint32, 4]
    crc: types.uint32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LowState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowState_"):
    version: types.array[types.uint32, 2]
    mode_pr: types.uint8
    mode_machine: types.uint8
    tick: types.uint32
    imu_state: IMUState_
    motor_state: types.array[MotorState_, 35]
    wireless_remote: types.array[types.uint8, 40]
    reserve: types.array[types.uint32, 4]
    crc: types.uint32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class HandCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.HandCmd_"):
    motor_cmd: types.sequence[MotorCmd_]
    reserve: types.array[types.uint32, 4]


@dataclass
@annotate.final
@annotate.autoid("sequential")
class HandState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.HandState_"):
    motor_state: types.sequence[MotorState_]
    press_sensor_state: types.sequence[PressSensorState_]
    imu_state: IMUState_
    power_v: types.float32
    power_a: types.float32
    system_v: types.float32
    device_v: types.float32
    error: types.array[types.uint32, 2]
    reserve: types.array[types.uint32, 2]


# --- constructors with every field zeroed (the IDL dataclasses have no defaults) ---------------------------------


def motor_cmd_zero() -> MotorCmd_:
    return MotorCmd_(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=0)


def motor_state_zero() -> MotorState_:
    return MotorState_(
        mode=0,
        q=0.0,
        dq=0.0,
        ddq=0.0,
        tau_est=0.0,
        temperature=[0, 0],
        vol=0.0,
        sensor=[0, 0],
        motorstate=0,
        reserve=[0, 0, 0, 0],
    )


def imu_zero() -> IMUState_:
    return IMUState_(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[0.0, 0.0, 0.0],
        accelerometer=[0.0, 0.0, 0.0],
        rpy=[0.0, 0.0, 0.0],
        temperature=0,
    )


def press_zero() -> PressSensorState_:
    return PressSensorState_(pressure=[0.0] * 12, temperature=[0.0] * 12, lost=0, reserve=0)


def lowcmd_zero() -> LowCmd_:
    return LowCmd_(
        mode_pr=0,
        mode_machine=0,
        motor_cmd=[motor_cmd_zero() for _ in range(N_SLOTS)],
        reserve=[0, 0, 0, 0],
        crc=0,
    )


def lowstate_zero() -> LowState_:
    return LowState_(
        version=[0, 0],
        mode_pr=0,
        mode_machine=0,
        tick=0,
        imu_state=imu_zero(),
        motor_state=[motor_state_zero() for _ in range(N_SLOTS)],
        wireless_remote=[0] * 40,
        reserve=[0, 0, 0, 0],
        crc=0,
    )


def hand_cmd_zero(n_motors: int = N_DEX3_MOTORS) -> HandCmd_:
    return HandCmd_(motor_cmd=[motor_cmd_zero() for _ in range(n_motors)], reserve=[0, 0, 0, 0])


def hand_state_zero(n_motors: int = N_DEX3_MOTORS, n_press: int = N_DEX3_PRESS) -> HandState_:
    return HandState_(
        motor_state=[motor_state_zero() for _ in range(n_motors)],
        press_sensor_state=[press_zero() for _ in range(n_press)],
        imu_state=imu_zero(),
        power_v=0.0,
        power_a=0.0,
        system_v=0.0,
        device_v=0.0,
        error=[0, 0],
        reserve=[0, 0],
    )


# --- Unitree's CRC over the C struct image -------------------------------------------------------------------------

# tools/crc_golden.py (from unitree_sdk2py utils/crc.py: "#size 1004" / "#size 2092")
FMT_HG_LOWCMD = "<2B2x" + "B3x5fI" * N_SLOTS + "5I"
FMT_HG_LOWSTATE = "<2I2B2xI" + "13fh2x" + "B3x4f2hf7I" * N_SLOTS + "40B5I"
_POLY = 0x04C11DB7


def _make_table() -> tuple[int, ...]:
    table = []
    for b in range(256):
        c = b << 24
        for _ in range(8):
            c = ((c << 1) & 0xFFFFFFFF) ^ _POLY if c & 0x80000000 else (c << 1) & 0xFFFFFFFF
        table.append(c)
    return tuple(table)


_TABLE = _make_table()


def crc_of_image(image: bytes) -> int:
    """Unitree's CRC of a packed message image (crc field included as the last word, which is excluded)."""
    n = (len(image) >> 2) - 1
    crc = 0xFFFFFFFF
    tab = _TABLE
    for i in range(n):
        w = image[4 * i : 4 * i + 4]
        # big-endian bytes of the little-endian word: w[3], w[2], w[1], w[0]
        for byte in (w[3], w[2], w[1], w[0]):
            crc = ((crc << 8) & 0xFFFFFFFF) ^ tab[((crc >> 24) ^ byte) & 0xFF]
    return crc


def lowcmd_image(msg: LowCmd_) -> bytes:
    vals: list[Any] = [msg.mode_pr, msg.mode_machine]
    for m in msg.motor_cmd:
        vals += [m.mode, m.q, m.dq, m.tau, m.kp, m.kd, m.reserve]
    vals += [*msg.reserve, msg.crc]
    return struct.pack(FMT_HG_LOWCMD, *vals)


def lowstate_image(msg: LowState_) -> bytes:
    imu = msg.imu_state
    vals: list[Any] = [*msg.version, msg.mode_pr, msg.mode_machine, msg.tick]
    vals += [*imu.quaternion, *imu.gyroscope, *imu.accelerometer, *imu.rpy, imu.temperature]
    for m in msg.motor_state:
        vals += [m.mode, m.q, m.dq, m.ddq, m.tau_est, *m.temperature, m.vol, *m.sensor, m.motorstate, *m.reserve]
    vals += [*msg.wireless_remote, *msg.reserve, msg.crc]
    return struct.pack(FMT_HG_LOWSTATE, *vals)


def lowcmd_crc(msg: LowCmd_) -> int:
    return crc_of_image(lowcmd_image(msg))


def lowstate_crc(msg: LowState_) -> int:
    return crc_of_image(lowstate_image(msg))
