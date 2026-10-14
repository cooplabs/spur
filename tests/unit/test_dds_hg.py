# mypy: disable-error-code="index,assignment,arg-type"
# (CycloneDDS IDL fields are typed as its annotation classes - types.array / types.sequence - not as list / int)
"""The Python hg types + CRC (python/r1mc3/dds/hg.py): the CRC reproduces the committed golden set (the same file the
C++ CRC test reads), and a LowCmd_ / LowState_ / HandState_ survives a CycloneDDS loopback inside one process."""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Any, cast

import pytest
from r1mc3 import dds

pytestmark = pytest.mark.needs_dds
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1mc3.dds import hg  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "assets" / "crc_golden" / "crc_golden.txt"


def _golden() -> list[tuple[str, str, int, bytes]]:
    rows = []
    for line in GOLDEN.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, kind, crc, data = line.split()
        rows.append((name, kind, int(crc, 16), bytes.fromhex(data)))
    return rows


def _lowcmd_from_image(image: bytes) -> hg.LowCmd_:
    v = struct.unpack(hg.FMT_HG_LOWCMD, image)
    msg = hg.lowcmd_zero()
    msg.mode_pr, msg.mode_machine = v[0], v[1]
    for i in range(hg.N_SLOTS):
        m = v[2 + 7 * i : 9 + 7 * i]
        msg.motor_cmd[i] = hg.MotorCmd_(mode=m[0], q=m[1], dq=m[2], tau=m[3], kp=m[4], kd=m[5], reserve=m[6])
    msg.reserve = list(v[2 + 7 * hg.N_SLOTS : 6 + 7 * hg.N_SLOTS])
    msg.crc = v[-1]
    return msg


def _lowstate_from_image(image: bytes) -> hg.LowState_:
    v = list(struct.unpack(hg.FMT_HG_LOWSTATE, image))
    msg = hg.lowstate_zero()
    msg.version, msg.mode_pr, msg.mode_machine, msg.tick = v[0:2], v[2], v[3], v[4]
    i = 5
    imu = msg.imu_state
    imu.quaternion, imu.gyroscope, imu.accelerometer, imu.rpy = (
        v[i : i + 4],
        v[i + 4 : i + 7],
        v[i + 7 : i + 10],
        v[i + 10 : i + 13],
    )
    imu.temperature = v[i + 13]
    i += 14
    for k in range(hg.N_SLOTS):
        m = v[i : i + 15]
        msg.motor_state[k] = hg.MotorState_(
            mode=m[0], q=m[1], dq=m[2], ddq=m[3], tau_est=m[4], temperature=m[5:7], vol=m[7], sensor=m[8:10],
            motorstate=m[10], reserve=m[11:15],
        )  # fmt: skip
        i += 15
    msg.wireless_remote = v[i : i + 40]
    msg.reserve = v[i + 40 : i + 44]
    msg.crc = v[i + 44]
    return msg


def test_the_crc_reproduces_the_committed_golden_set() -> None:
    rows = _golden()
    assert {k for _, k, _, _ in rows} == {"lowcmd", "lowstate"}
    for name, kind, crc, image in rows:
        if kind == "lowcmd":
            msg_c = _lowcmd_from_image(image)
            assert hg.lowcmd_image(msg_c) == image, name
            assert hg.lowcmd_crc(msg_c) == crc, name
        else:
            msg_s = _lowstate_from_image(image)
            assert hg.lowstate_image(msg_s) == image, name
            assert hg.lowstate_crc(msg_s) == crc, name


def test_image_sizes_are_unitrees() -> None:
    assert len(hg.lowcmd_image(hg.lowcmd_zero())) == 1004
    assert len(hg.lowstate_image(hg.lowstate_zero())) == 2092


def test_loopback_in_one_process() -> None:
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic

    dp = DomainParticipant(0)
    cmd = hg.lowcmd_zero()
    cmd.mode_machine = 1
    cmd.motor_cmd[3].q = 0.25
    cmd.motor_cmd[3].kp = 84.302
    cmd.crc = hg.lowcmd_crc(cmd)
    st = hg.lowstate_zero()
    st.tick = 12345
    st.motor_state[7].q = -0.5
    hs = hg.hand_state_zero()
    hs.press_sensor_state[2].pressure[5] = 30000.0
    pairs = []
    for name, typ, sample in (
        ("r1mc3_test/lowcmd", hg.LowCmd_, cmd),
        ("r1mc3_test/lowstate", hg.LowState_, st),
        ("r1mc3_test/hand", hg.HandState_, hs),
    ):
        topic = Topic(dp, name, typ)
        pairs.append((DataReader(dp, topic), DataWriter(dp, topic), sample))
    time.sleep(0.2)
    for _, w, sample in pairs:
        w.write(sample)
    deadline = time.monotonic() + 3.0
    got: dict[int, object] = {}
    while len(got) < len(pairs) and time.monotonic() < deadline:
        for i, (r, _, _) in enumerate(pairs):
            for s in r.take(N=4):
                got[i] = s
        time.sleep(0.01)
    assert len(got) == len(pairs)
    rx_cmd = got[0]
    assert isinstance(rx_cmd, hg.LowCmd_)
    assert rx_cmd.crc == hg.lowcmd_crc(rx_cmd) == cmd.crc
    assert rx_cmd.motor_cmd[3].kp == pytest.approx(84.302, abs=1e-5)
    rx_st = got[1]
    assert isinstance(rx_st, hg.LowState_)
    assert rx_st.tick == 12345 and rx_st.motor_state[7].q == pytest.approx(-0.5)
    rx_hs = got[2]
    assert isinstance(rx_hs, hg.HandState_)
    press = cast(list[Any], rx_hs.press_sensor_state)
    assert len(press) == hg.N_DEX3_PRESS
    assert press[2].pressure[5] == 30000.0
