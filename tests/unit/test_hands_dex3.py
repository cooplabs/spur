# mypy: disable-error-code="index"
# (CycloneDDS IDL sequences are typed as its annotation class, not as list)
"""r1hands minimal (python/r1mc3/hands/dex3.py) against a fake Dex3 over CycloneDDS (loopback, domain 61): the
commands on the wire (mode byte, timeout bit cleared for the first ticks, limits), the state read back (joints,
the 9 x 12 taxels raw), the TTL -> hold-where-it-is path, a stale state reported invalid, bad commands refused."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pytest
from r1mc3 import dds

pytestmark = pytest.mark.needs_dds
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1mc3.dds import hg, rpc  # noqa: E402
from r1mc3.hands import dex3  # noqa: E402

DOMAIN = 61
LOOPBACK = rpc.LOOPBACK_URI  # lo0 on macOS, lo on Linux
os.environ.setdefault("CYCLONEDDS_URI", LOOPBACK)


class FakeDex3:
    """dex3_server's side: publishes HandState_ for the left hand, reads HandCmd_."""

    def __init__(self) -> None:
        from cyclonedds.core import Policy, Qos
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        self.dp = DomainParticipant(DOMAIN)
        self.w = DataWriter(self.dp, Topic(self.dp, dex3.TOPIC_STATE["left"], hg.HandState_))
        keep_all = Qos(Policy.History.KeepLast(1000))  # the test reads every command (the default keeps the last one)
        self.r = DataReader(self.dp, Topic(self.dp, dex3.TOPIC_CMD["left"], hg.HandCmd_), qos=keep_all)
        self.q = np.array([0.1, 0.2, 0.3, -0.4, -0.5, -0.6, -0.7])
        self.rng = np.random.default_rng(0)

    def publish(self) -> None:
        m = hg.hand_state_zero()
        for i in range(dex3.N):
            m.motor_state[i].q = float(self.q[i])
            # a real sample is never bit-identical to the previous one (session 6: r1hands calls a repeating stream
            # FROZEN after dex3.FROZEN_S): the current-based tau_est carries the noise here, q stays exact
            m.motor_state[i].tau_est = float(self.rng.normal(0.0, 1e-3))
            m.motor_state[i].temperature = [40 + i, 41 + i]
        m.press_sensor_state[4].pressure[3] = 123.0
        self.w.write(m)

    def cmds(self) -> list[Any]:
        return list(self.r.take(N=1000))


def _until(cond: Any, timeout: float = 3.0) -> bool:
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _collect(fake: FakeDex3, got: list[Any], n: int) -> bool:
    got.extend(fake.cmds())
    return len(got) >= n


@pytest.fixture(scope="module")
def rig() -> Any:
    fake = FakeDex3()
    hands = dex3.Dex3Hands(DOMAIN, sides=("left",))
    yield fake, hands
    hands.close()


def test_mode_byte_is_unitrees_layout() -> None:
    assert dex3.mode_byte(0, 1, 1) == 0x90
    assert dex3.mode_byte(6, 1, 0) == 0x16
    assert dex3.mode_byte(15, 7, 1) == 0xFF


def test_state_read_and_commands_on_the_wire(rig: Any) -> None:
    fake, hands = rig
    time.sleep(0.3)  # discovery
    assert _until(lambda: (fake.publish(), hands.step(), hands.state("left")["valid"])[-1])
    st = hands.state("left")
    np.testing.assert_allclose(st["q"], fake.q, atol=1e-6)
    assert st["pressure"].shape == (9, 12) and st["pressure"][4, 3] == 123.0
    assert st["temperature"][2].tolist() == [42.0, 43.0]
    out = hands.command("left", [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ttl_s=2.0)  # thumb_0 beyond its limit
    assert out["clamped_q"] and abs(out["q"][0] - (1.0472 - dex3.MARGIN)) < 1e-9
    fake.cmds()
    for _ in range(6):
        fake.publish()
        hands.step()
        time.sleep(0.01)
    got: list[Any] = []
    assert _until(lambda: _collect(fake, got, 6))
    modes = [[c.motor_cmd[i].mode for i in range(dex3.N)] for c in got]
    assert modes[0] == [dex3.mode_byte(i, 1, 0) for i in range(dex3.N)]  # the first ticks clear the timeout flag
    assert modes[-1] == [dex3.mode_byte(i, 1, 1) for i in range(dex3.N)]  # then the hardware deadman is armed
    assert got[-1].motor_cmd[0].q == pytest.approx(1.0472 - dex3.MARGIN, abs=1e-6)
    assert got[-1].motor_cmd[3].kp == pytest.approx(1.5)


def test_ttl_expiry_holds_the_measured_position(rig: Any) -> None:
    fake, hands = rig
    hands.command("left", [0.0, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5], kp=[3.0] * 7, ttl_s=0.05)
    fake.publish()
    time.sleep(0.1)
    fake.cmds()
    for _ in range(3):
        fake.publish()
        hands.step()
        time.sleep(0.01)
    assert hands.state("left")["held"]
    got: list[Any] = []
    assert _until(lambda: _collect(fake, got, 2))
    q = [got[-1].motor_cmd[i].q for i in range(dex3.N)]
    np.testing.assert_allclose(q, dex3.clamp_q(fake.q, "left"), atol=1e-6)  # where it is, not the old target
    assert got[-1].motor_cmd[1].kp == pytest.approx(1.5) and got[-1].motor_cmd[1].kd == pytest.approx(0.2)


def test_stale_state_is_invalid(rig: Any) -> None:
    _, hands = rig
    time.sleep(dex3.STATE_STALE_S + 0.05)
    hands.step()
    st = hands.state("left")
    assert st["valid"] is False and st["age_ms"] > dex3.STATE_STALE_S * 1e3


def test_bad_commands_are_refused(rig: Any) -> None:
    _, hands = rig
    with pytest.raises(ValueError):
        hands.command("left", [0.0] * 6)
    with pytest.raises(ValueError):
        hands.command("left", [0.0] * 7, kp=[9.0] * 7)
    with pytest.raises(ValueError):
        hands.command("left", [0.0] * 7, ttl_s=10.0)
    with pytest.raises(ValueError):
        hands.command("right", [0.0] * 7)  # not a side this instance drives
