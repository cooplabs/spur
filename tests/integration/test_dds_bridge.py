# mypy: disable-error-code="index,assignment,arg-type,attr-defined,var-annotated"
# (CycloneDDS IDL fields are typed as its annotation classes - types.array / types.sequence - not as list / int)
"""The DDS-bridge simulator (r1sim.dds_bridge, DESIGN §12.4): r1sim in real time behind the robot's DDS interface.

A peer in this process (its own DomainParticipant, like r1rt / r1guard) sees rt/lowstate with Unitree's CRC at the
publish rate, the emulated built-in on rt/lowcmd until a ReleaseMode over the motion-switcher RPC, and the board
applies the peer's commands (bad CRC dropped, a disabled slot = zero gains, every applied sample recorded with its
writer). Plus the vectorised CRC against the reference and the 9 Oct / safety configurations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3 import dds

pytestmark = pytest.mark.needs_dds
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from cyclonedds.domain import DomainParticipant  # noqa: E402
from cyclonedds.pub import DataWriter  # noqa: E402
from cyclonedds.sub import DataReader  # noqa: E402
from cyclonedds.topic import Topic  # noqa: E402
from r1mc3.dds import hg, rpc  # noqa: E402
from r1sim import joints as J  # noqa: E402
from r1sim import world  # noqa: E402
from r1sim.dds_bridge import (  # noqa: E402
    ARMS_TABLE_READY,
    DdsBridge,
    HgCrc,
    crc_lowcmd,
    instance_handle,
    lowcmd_msg,
)
from r1sim.driver import DriverConfig  # noqa: E402
from r1sim.model import REAL_JOINT_FRICTIONLOSS_HW1, SAFETY_LEGS_FRICTIONLOSS  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "assets" / "crc_golden" / "crc_golden.txt"


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real")


def _take_all(r: Any, seconds: float) -> list[Any]:
    out: list[Any] = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        out += r.take(N=256)
        time.sleep(0.002)
    return out


def test_the_vectorised_crc_is_unitrees() -> None:
    crcs = {2092: HgCrc(2092), 1004: HgCrc(1004)}
    for line in GOLDEN.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _, _, crc, data = line.split()
        image = bytes.fromhex(data)
        assert crcs[len(image)](image) == int(crc, 16)
    rng = np.random.default_rng(5)
    for n, c in crcs.items():
        for _ in range(20):
            im = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
            assert c(im) == hg.crc_of_image(im)


@pytest.mark.realtime
def test_lowstate_builtin_release_and_a_peers_command(w: world.World) -> None:
    domain = 47
    # session 8: the board model is the bridge's default; this test's peer pauses 0.4 s between its commands (a wire /
    # CRC / writer test, not a board-timeout test): the S1 board that holds the last command, explicitly
    with DdsBridge(domain=domain, w=w, stance="centred", driver=DriverConfig()) as br:
        dp = DomainParticipant(domain)
        r_state = DataReader(dp, Topic(dp, hg.TOPIC_LOWSTATE, hg.LowState_), qos=rpc.READER_QOS)
        r_cmd = DataReader(dp, Topic(dp, hg.TOPIC_LOWCMD, hg.LowCmd_), qos=rpc.READER_QOS)
        w_cmd = DataWriter(dp, Topic(dp, hg.TOPIC_LOWCMD, hg.LowCmd_), qos=rpc.WRITER_QOS)
        states = _take_all(r_state, 1.5)
        builtin_seen = _take_all(r_cmd, 0.3)
        # rt/lowstate: Unitree's CRC, the R1's machine mode, the board tick advancing at ~1 kHz, our joints in slots
        assert len(states) > 300, len(states)  # >= ~200 Hz received under load (published ~500 Hz)
        assert all(hg.lowstate_crc(s) == s.crc for s in states)
        assert all(s.mode_machine == J.MODE_MACHINE for s in states)
        ticks = np.array([s.tick for s in states], dtype=np.int64)
        assert np.all(np.diff(ticks) > 0)
        sim_q = br.sim.read_state().q
        last = states[-1]
        assert abs(last.motor_state[int(J.SLOT[3])].q - sim_q[3]) < 0.02  # left knee, as the sensors report it
        assert last.motor_state[14].mode == 0  # an unused hg slot
        # the emulated built-in publishes the stand fixture from its own participant (a foreign writer)
        assert builtin_seen and all(crc_lowcmd(c) == c.crc for c in builtin_seen)
        assert br.builtin_handle in {int(c.sample_info.publication_handle) for c in builtin_seen}
        # ReleaseMode over the motion-switcher RPC: the built-in stops
        cli = rpc.MotionSwitcherClient(domain, participant=dp)
        assert cli.check_mode() == (0, {"form": "0", "name": "ai"})
        assert cli.release_mode() == 0
        assert cli.check_mode()[1]["name"] == ""
        time.sleep(0.1)
        n_builtin = br.writers()[br.builtin_handle]["n"]
        time.sleep(0.3)
        assert br.writers()[br.builtin_handle]["n"] == n_builtin  # nothing more from the built-in
        # a peer's command: the head pitched 0.3 rad; a bad CRC dropped; then a command with a disabled leg slot
        ours = instance_handle(w_cmd)
        cmd = br.fixture.command
        cmd.q[24] = 0.3
        good = lowcmd_msg(cmd)
        bad = lowcmd_msg(cmd)
        bad.crc ^= 1
        for _ in range(100):
            w_cmd.write(good)
            time.sleep(0.004)
        w_cmd.write(bad)
        time.sleep(0.4)
        q_head = br.sim.read_state().q[24]
        assert abs(q_head - 0.3) < 0.05, q_head
        assert br.stats["lowcmd_bad_crc"] >= 1
        mine = [a for a in br.applied if a.writer == ours]
        assert len([a for a in mine if a.crc_ok]) >= 60
        assert all(a.min_leg_kp > 10.0 for a in mine if a.crc_ok)  # never limp so far
        limp = lowcmd_msg(cmd)
        limp.motor_cmd[int(J.SLOT[3])].mode = 0  # the left knee's motor disabled
        limp.crc = crc_lowcmd(limp)
        w_cmd.write(limp)
        time.sleep(0.2)
        last_mine = [a for a in br.applied if a.writer == ours][-1]
        assert last_mine.disabled_legs == 1 and last_mine.min_leg_kp == 0.0  # recorded as limp (kp 0 on a leg)
        w_cmd.write(good)  # back to stiff before the bridge stops
        time.sleep(0.05)
        log = br.log()
    assert log["writers"][str(br.builtin_handle)]["label"] == "builtin"
    assert log["writers"][str(ours)]["label"] == "w0"
    assert log["physics"]["rephases"] == 0


@pytest.mark.realtime
def test_hands_over_dds(w: world.World) -> None:
    domain = 48
    with DdsBridge(domain=domain, w=w, stance="centred", builtin="stand") as br:
        dp = DomainParticipant(domain)
        r_hand = DataReader(dp, Topic(dp, hg.TOPIC_DEX3_LEFT_STATE, hg.HandState_), qos=rpc.READER_QOS)
        w_hand = DataWriter(dp, Topic(dp, hg.TOPIC_DEX3_LEFT_CMD, hg.HandCmd_), qos=rpc.WRITER_QOS)
        hs = _take_all(r_hand, 0.6)
        assert len(hs) >= 20  # ~100 Hz published
        h = hs[-1]
        assert len(h.motor_state) == hg.N_DEX3_MOTORS and len(h.press_sensor_state) == hg.N_DEX3_PRESS
        cmd = hg.hand_cmd_zero()
        for i, m in enumerate(cmd.motor_cmd):
            m.kp, m.kd = 1.5, 0.2
            m.q = 0.6 if i == 2 else 0.0  # thumb_2 (left: + closes)
        for _ in range(60):
            w_hand.write(cmd)
            time.sleep(0.01)
        q2 = br.sim.read_state().hand_q["left"][2]
        assert br.stats["hand_cmd_rx"] >= 30
        assert abs(q2 - 0.6) < 0.15, q2


def test_the_9oct_and_safety_configurations(w: world.World) -> None:
    br = DdsBridge(domain=49, stance="9oct", safety=True, sensors=False)
    m = br.sim.m
    for name in J.JOINTS[:14]:
        f = m.dof_frictionloss[m.joint(f"{name}_joint").dofadr[0]]
        assert f == (SAFETY_LEGS_FRICTIONLOSS[1] if "ankle" in name else SAFETY_LEGS_FRICTIONLOSS[0])
    np.testing.assert_allclose(br.fixture.pose[J.ARMS], ARMS_TABLE_READY)
    np.testing.assert_allclose(br.sim.read_state().q[J.ARMS], ARMS_TABLE_READY, atol=1e-9)
    assert br.imu_offset_deg == (0.63, 0.53)
    nominal = DdsBridge(domain=49, stance="centred", sensors=False)
    knee = nominal.sim.m.joint("left_knee_joint").dofadr[0]
    # the nominal model keeps the legs' MEASURED stiction (HW1, session 8: was issue #51's 2.5); safety = 0
    assert nominal.sim.m.dof_frictionloss[knee] == pytest.approx(REAL_JOINT_FRICTIONLOSS_HW1["left_knee"])


@pytest.mark.realtime
def test_a_lowstate_gap_is_a_gap_on_the_wire(w: world.World) -> None:
    domain = 46
    with DdsBridge(domain=domain, w=w, stance="centred") as br:
        dp = DomainParticipant(domain)
        r_state = DataReader(dp, Topic(dp, hg.TOPIC_LOWSTATE, hg.LowState_), qos=rpc.READER_QOS)
        _take_all(r_state, 0.3)
        br.inject("lowstate_gap", duration_s=0.3)
        rx: list[float] = []
        t_end = time.monotonic() + 0.8
        while time.monotonic() < t_end:
            if r_state.take(N=256):
                rx.append(time.monotonic())
            time.sleep(0.002)
        gaps = np.diff(np.asarray(rx))
        assert gaps.max() > 0.25, gaps.max()
        assert any("lowstate gap" in e["what"] for e in br.events)


@pytest.mark.realtime
def test_writers_are_labelled_by_their_user_data(w: world.World) -> None:
    from cyclonedds.core import Policy, Qos

    domain = 45
    with DdsBridge(domain=domain, w=w, stance="centred") as br:
        dp = DomainParticipant(domain)
        tagged = DataWriter(
            dp,
            Topic(dp, hg.TOPIC_LOWCMD, hg.LowCmd_),
            qos=Qos(Policy.Userdata(b"r1mc3:r1guard"), Policy.History.KeepLast(1)),
        )
        msg = lowcmd_msg(br.fixture.command)
        for _ in range(40):
            tagged.write(msg)
            time.sleep(0.005)
        time.sleep(0.3)
        labels = {v["label"] for v in br.writers().values()}
    assert labels == {"builtin", "r1guard"}
