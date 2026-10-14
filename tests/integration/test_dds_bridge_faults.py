# mypy: disable-error-code="index,assignment,attr-defined"
# (CycloneDDS IDL fields are typed as its annotation classes; samples carry a run-time sample_info)
"""The DDS-bridge simulator's scheduled fault injections (session 6, the hang agent; DESIGN §12.6): every kind of
`DdsBridge.inject(kind, at_s=, duration_s=, every_s=, count=, ...)` observed ON THE WIRE by a peer in this process (as
r1rt / r1guard / r1hands see it), each window's actual times in `faults()`. Wall-clock on a loaded Mac: windows of
0.3-0.5 s, effects asserted with wide margins (a fraction 0.5 -> 0.25-0.75)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from cyclonedds.domain import DomainParticipant  # noqa: E402
from cyclonedds.pub import DataWriter  # noqa: E402
from cyclonedds.sub import DataReader  # noqa: E402
from cyclonedds.topic import Topic  # noqa: E402
from r1mc3.dds import hg, rpc  # noqa: E402
from r1sim import joints as J  # noqa: E402
from r1sim.dds_bridge import DdsBridge, HandCmd, crc_lowstate, lowcmd_msg  # noqa: E402

KNEE = J.IDX["left_knee"]


def _window(br: DdsBridge, kind: str) -> tuple[float, float]:
    f = next(e for e in br.faults() if e["kind"] == kind)
    assert f["first_wall"] is not None, f
    return float(f["first_wall"]), float(f["last_wall"])


def test_lowstate_side_faults_on_the_wire() -> None:
    domain = 43
    with DdsBridge(domain=domain, scene="none", stance="centred") as br:
        dp = DomainParticipant(domain)
        rd = DataReader(dp, Topic(dp, hg.TOPIC_LOWSTATE, hg.LowState_), qos=rpc.READER_QOS)
        w_cmd = DataWriter(dp, Topic(dp, hg.TOPIC_LOWCMD, hg.LowCmd_), qos=rpc.WRITER_QOS)
        peer = lowcmd_msg(br.fixture.command)
        time.sleep(0.5)  # discovery
        rd.take(N=4096)
        br.inject("lowstate_drop", at_s=0.3, duration_s=0.5, fraction=0.5)
        br.inject("lowstate_crc", at_s=1.0, duration_s=0.4, fraction=0.5)
        br.inject("motor_error", at_s=1.6, duration_s=0.3, joint="left_knee", word=0x20)
        br.inject("over_temp", at_s=2.1, duration_s=0.3, joint=KNEE, to_c=75.0, rate_c_s=400.0)
        br.inject("lowstate_delay", at_s=2.6, duration_s=0.5, delay_ms=50.0)
        br.inject("link_down", at_s=3.3, duration_s=0.3)
        br.inject("publisher_stall", at_s=3.9, duration_s=0.03, every_s=0.1, count=3)
        rx: list[dict[str, Any]] = []
        t_end = time.monotonic() + 4.6
        while time.monotonic() < t_end:
            w_cmd.write(peer)  # a command source for the link-down check
            got: list[Any] = rd.take(N=256)
            now = time.monotonic()
            tick_now = int(br.sim.read_state().tick)
            for s in got:
                if isinstance(s, hg.LowState_):
                    m = s.motor_state[int(J.SLOT[KNEE])]
                    rx.append(dict(t=now, lag=tick_now - int(s.tick), crc_ok=crc_lowstate(s) == int(s.crc),
                                   err=int(m.motorstate), temp=int(m.temperature[0])))  # fmt: skip
            time.sleep(0.002)
        t = np.array([r["t"] for r in rx])

        def inside(kind: str) -> list[dict[str, Any]]:
            a, b = _window(br, kind)
            return [r for r in rx if a <= r["t"] <= b + 0.003]

        base = [r for r in rx if r["t"] < _window(br, "lowstate_drop")[0]]
        rate0 = len(base) / (base[-1]["t"] - base[0]["t"])
        a, b = _window(br, "lowstate_drop")
        assert 0.25 < len(inside("lowstate_drop")) / ((b - a) * rate0) < 0.75
        bad = [not r["crc_ok"] for r in inside("lowstate_crc")]
        assert 0.25 < np.mean(bad) < 0.75
        assert any(r["err"] == 0x20 for r in inside("motor_error"))
        assert not any(r["err"] for r in base)
        assert max(r["temp"] for r in inside("over_temp")) == 75
        lag_in = np.median([r["lag"] for r in inside("lowstate_delay")[5:]])
        assert 35 <= lag_in <= 80 and np.median([r["lag"] for r in base]) < 15, lag_in  # 1 kHz board ticks
        a, b = _window(br, "link_down")
        assert not [r for r in rx if a + 0.01 < r["t"] < b - 0.01]  # nothing on rt/lowstate while the link is down
        assert br.stats["lowcmd_link_down"] > 0
        stalls = next(e for e in br.faults() if e["kind"] == "publisher_stall")
        assert stalls["n_stalls"] == 3 and all(0.025 < d < 0.08 for _, d in stalls["stalls"]), stalls
        gaps = np.diff(t[t > _window(br, "publisher_stall")[0] - 0.01])
        assert (gaps > 0.02).sum() >= 3


def test_command_side_faults_at_the_board() -> None:
    domain = 44
    with DdsBridge(domain=domain, scene="none", stance="centred", builtin="none") as br:
        dp = DomainParticipant(domain)
        w_cmd = DataWriter(dp, Topic(dp, hg.TOPIC_LOWCMD, hg.LowCmd_), qos=rpc.WRITER_QOS)
        peer = lowcmd_msg(br.fixture.command)
        time.sleep(0.5)
        br.inject("lowcmd_drop", at_s=0.3, duration_s=0.4, fraction=0.5)
        br.inject("lowcmd_crc", at_s=0.9, duration_s=0.4, fraction=0.5)
        br.inject("cmd_reader_stall", at_s=1.5, duration_s=0.05, every_s=0.2, count=3)
        sw = br.inject("second_writer", at_s=2.3, duration_s=0.4, payload="damp")
        t_end = time.monotonic() + 3.1
        nxt = time.monotonic()
        while time.monotonic() < t_end:
            w_cmd.write(peer)
            nxt += 0.002
            time.sleep(max(0.0, nxt - time.monotonic()))
        assert 0.25 < br.stats["lowcmd_dropped"] / (0.4 * 500) < 0.75
        assert 0.25 < br.stats["lowcmd_corrupted"] / (0.4 * 500) < 0.75
        assert br.stats["lowcmd_bad_crc"] >= br.stats["lowcmd_corrupted"]  # the board drops them (bad CRC)
        a, _ = _window(br, "cmd_reader_stall")
        applied = [x.wall for x in list(br.applied) if x.crc_ok and a - 0.01 < x.wall < a + 0.6]
        assert (np.diff(applied) > 0.04).sum() >= 3  # the board held the last command through each stall
        labels = {h: v for h, v in br.writers().items()}
        sec = [v for v in labels.values() if v["label"] == "second_writer"]
        assert sec and sec[0]["n"] > 20, labels
        assert sw is not None and sw.hits > 50
        damp_applied = [x for x in list(br.applied) if x.writer == sw.params["writer_handle"]]
        assert damp_applied and max(x.min_leg_kp for x in damp_applied) == 0.0  # its damp reached the motors


def test_hand_faults() -> None:
    domain = 47
    with DdsBridge(domain=domain, scene="none", stance="centred") as br:
        dp = DomainParticipant(domain)
        r_l = DataReader(dp, Topic(dp, hg.TOPIC_DEX3_LEFT_STATE, hg.HandState_), qos=rpc.READER_QOS)
        r_r = DataReader(dp, Topic(dp, hg.TOPIC_DEX3_RIGHT_STATE, hg.HandState_), qos=rpc.READER_QOS)
        w_l = DataWriter(dp, Topic(dp, hg.TOPIC_DEX3_LEFT_CMD, hg.HandCmd_), qos=rpc.WRITER_QOS)
        w_r = DataWriter(dp, Topic(dp, hg.TOPIC_DEX3_RIGHT_CMD, hg.HandCmd_), qos=rpc.WRITER_QOS)
        cmd: Any = hg.hand_cmd_zero()
        for m in cmd.motor_cmd:
            m.kp, m.kd = 1.5, 0.1
        time.sleep(0.5)
        br.inject("hand_freeze", at_s=0.2, duration_s=0.4, side="left")
        br.inject("dex3_death", at_s=0.2, duration_s=0.4, side="right")
        br.inject("dex3_timeout", at_s=0.0, duration_s=None, timeout_s=0.3)
        left: list[tuple[float, tuple[float, ...]]] = []
        right: list[float] = []
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            w_r.write(cmd)  # ignored while the right dex3_server is dead
            if time.monotonic() < t_end - 0.6:
                w_l.write(cmd)  # the left hand's commands stop 0.6 s before the end -> passive 0.3 s later
            now = time.monotonic()
            for s in r_l.take(N=64):
                if isinstance(s, hg.HandState_):
                    left.append((now, tuple(round(float(m.q), 9) for m in s.motor_state)))
            right += [now for s in r_r.take(N=64) if isinstance(s, hg.HandState_)]
            time.sleep(0.005)
        a, b = _window(br, "hand_freeze")
        frozen = {q for t, q in left if a + 0.02 < t < b - 0.02}
        assert len(frozen) == 1  # one repeated sample
        a, b = _window(br, "dex3_death")
        assert not [t for t in right if a + 0.02 < t < b - 0.02]
        assert br.stats["hand_cmd_ignored"] > 0
        assert "left" in br._hand_passive and float(np.max(br._hands["left"].kp)) == 0.0
        assert isinstance(br._hands["right"], HandCmd) and "right" not in br._hand_passive  # commanded again
        assert any("passive" in e["what"] for e in br.events)


def test_inject_refuses_bad_parameters() -> None:
    br = DdsBridge(domain=48, scene="none", stance="centred")  # not started
    with pytest.raises(ValueError):
        br.inject("lowstate_drop", duration_s=1.0)  # no fraction
    with pytest.raises(ValueError):
        br.inject("motor_error", joint="left_elbow_x")
    with pytest.raises(ValueError):
        br.inject("publisher_stall")  # a stall needs its length
    with pytest.raises(ValueError):
        br.inject("nonsense")
