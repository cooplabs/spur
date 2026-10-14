"""The R1's battery in r1d's status (session 13; r1mc3.battery, read-only): day 2 read the percentage with a helper in
v2's environment over SSH (results/hw2_20261013/tools/r1battery.py) and planned the swaps from it. Now r1d's status
carries the reading, the drain (~0.7 %/min hanging, ~1 %/min balancing on day 2) and the minutes to the swap level."""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from r1mc3 import dds
from r1mc3.battery import TOPIC_MAIN, TOPIC_SECONDARY, BatteryReader, decode, drain_pct_per_min
from r1mc3.service.service import Service

from tests.unit.test_service_server import FakePort, _NoRelease


class Msg:
    """A BmsState_ as unitree_sdk2py hands it over (attribute access only)."""

    def __init__(self, soc: int, mv: int = 36530, ma: int = -2346, cell_mv: int = 4063, temp: int = 27) -> None:
        self.soc, self.soh, self.cycle, self.current = soc, 99, 8, ma
        self.bmsvoltage = [mv, mv - 50, 0]
        self.cell_vol = [cell_mv] * 9 + [0] * 31
        self.temperature = [23, 19, 20, temp] + [0] * 8


def test_decode_is_v2s_reading_plus_the_power() -> None:
    """v2's fake SDK carries a real R1 reading (30 Sep): soc 95, 36530 mV, -2346 mA, cells 4063 mV, 27 C max."""
    s = decode(Msg(95))
    assert (s.soc_pct, s.voltage_v, s.current_a, s.cell_min_v, s.temp_max_c, s.soh_pct, s.cycles) == (
        95, 36.53, -2.35, 4.063, 27, 99, 8)  # fmt: skip
    assert s.power_w == pytest.approx(85.8, abs=0.1)  # discharging = positive watts


def test_the_drain_is_a_line_through_the_percentages() -> None:
    hist = [(60.0 * k, 64.0 - k) for k in range(6)]  # 1 %/min for 5 min
    assert drain_pct_per_min(hist) == pytest.approx(1.0)
    assert drain_pct_per_min(hist[:2]) is None  # 1 min: still measuring
    assert drain_pct_per_min([(0.0, 64.0), (300.0, 64.0)]) == 0.0


class Feed:
    """A fake wire + clock: queued messages arrive at the next poll."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.q: list[tuple[str, Any]] = []

    def clock(self) -> float:
        return self.t

    def source(self) -> list[tuple[str, Any]]:
        out, self.q = self.q, []
        return out


def test_the_reader_reports_the_drain_and_the_minutes_to_the_swap() -> None:
    f = Feed()
    rd = BatteryReader(source=f.source, clock=f.clock)
    assert rd.status() == {"available": False, "reason": f"no message on {TOPIC_MAIN} yet", "words": []}
    for k in range(6):  # day 2 balancing: ~1 %/min, ~115 W
        f.q.append((TOPIC_MAIN, Msg(64 - k, mv=34590, ma=-3330)))
        rd.poll()
        f.t += 60.0
    f.t -= 60.0
    st = rd.status()
    assert st["available"] and st["soc_pct"] == 59 and st["drain_pct_per_min"] == pytest.approx(1.0)
    assert st["minutes_to_swap"] == 29 and st["power_w"] == pytest.approx(115.2, abs=0.1)
    assert st["words"] == ["battery 59 % (34.6 V, 115 W, cells >= 4.06 V) - ~1.0 %/min: ~29 min to 30 %"]
    f.q.append((TOPIC_SECONDARY, Msg(80)))
    rd.poll()
    assert rd.status()["secondary_soc_pct"] == 80


def test_low_and_old_readings_say_so() -> None:
    f = Feed()
    rd = BatteryReader(source=f.source, clock=f.clock)
    f.q.append((TOPIC_MAIN, Msg(28)))
    rd.poll()
    w = rd.status()["words"]
    assert w[0].startswith("battery 28 %") and "measuring" in w[0]
    assert "at or below the swap level (30 %): plan the battery swap now" in w[1]
    f.t += 30.0
    assert "the last message is 30 s old" in rd.status()["words"][-1]


def test_a_broken_source_is_reported_never_raised() -> None:
    def boom() -> list[tuple[str, Any]]:
        raise OSError("link down")

    rd = BatteryReader(source=boom)
    rd.poll()
    assert rd.status()["available"] is False and "link down" in rd.status()["reason"]


def test_r1d_status_carries_the_battery() -> None:
    """Base: no `battery` in r1d's status (the reading lived in a v2 helper)."""
    f = Feed()
    svc = Service(FakePort(), releaser=_NoRelease())
    svc.battery = BatteryReader(source=f.source, clock=f.clock)  # type: ignore[attr-defined]
    f.q.append((TOPIC_MAIN, Msg(64, mv=34590, ma=-3060)))
    svc.battery.poll()  # type: ignore[attr-defined]
    out = svc.handle({"op": "status"})
    b = out["data"]["battery"]
    assert b["soc_pct"] == 64 and b["voltage_v"] == 34.59 and b["words"][0].startswith("battery 64 %")


@pytest.mark.needs_dds
@pytest.mark.skipif(not dds.available(), reason="CycloneDDS's Python binding is not installed")
def test_the_reader_over_dds_loopback() -> None:
    """A BMS publisher (the robot's role) on loopback domain 100 -> the reader's status (read-only subscriber)."""
    from r1mc3.dds import rpc

    os.environ.setdefault("CYCLONEDDS_URI", rpc.LOOPBACK_URI)
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.pub import DataWriter
    from cyclonedds.topic import Topic
    from r1mc3.dds.bms import BmsState_, bms_zero

    dp = DomainParticipant(100)
    w = DataWriter(dp, Topic(dp, TOPIC_MAIN, BmsState_))
    rd = BatteryReader(100).start()
    try:
        msg = bms_zero(soc=64, bmsvoltage=[34590, 34540, 0], current=-3060, cell_vol=[3847] * 9 + [0] * 31,
                       temperature=[32, 30] + [0] * 10)  # fmt: skip
        t0 = time.monotonic()
        while time.monotonic() - t0 < 8.0 and not rd.status()["available"]:
            w.write(msg)
            time.sleep(0.2)
            rd.poll()
        st = rd.status()
        assert st["available"], st
        assert (st["soc_pct"], st["voltage_v"], st["current_a"], st["cell_min_v"], st["temp_max_c"]) == (
            64, 34.59, -3.06, 3.847, 32)  # fmt: skip
    finally:
        rd.close()
