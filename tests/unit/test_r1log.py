"""r1log on a real POSIX shared-memory segment (r1rt_py.SegmentView.create) with records pushed by the test (the
producer side; on the robot r1rt is the only producer): order, drops, rotation, the flight recorder's triggers,
window and hold-off, and the CLI as a separate process attached by name.
"""

from __future__ import annotations

import itertools
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.log import Logger, LoggerConfig, info, read
from r1mc3.log import records as R
from r1mc3.log._native import has_segment_reader, native

pytestmark = pytest.mark.skipif(not has_segment_reader(), reason="r1rt_py with SegmentView not built")

_n = itertools.count()
MS = 1_000_000


@pytest.fixture
def seg() -> Iterator[Any]:
    name = f"/r1lt{os.getpid() % 100000}_{next(_n)}"
    v = native().SegmentView.create(name, True)
    assert v.name == name
    yield v
    v.close()


def tel(
    seq0: int, n: int, *, dt_ns: int = 2 * MS, faults: int = 0, mode: int = 3, owner: int = 1
) -> np.ndarray[Any, Any]:
    r = np.zeros(n, R.TELEMETRY)
    r["seq"] = np.arange(seq0, seq0 + n)
    r["t_wake_ns"] = r["seq"].astype(np.int64) * dt_ns
    r["t_send_ns"] = r["t_wake_ns"] + 50_000
    r["imu_quat"][:, 0] = 1.0
    r["faults"] = faults
    r["mode"] = mode
    r["token_owner"] = owner
    r["q"] = np.random.default_rng(seq0).normal(size=(n, R.NJ)).astype(np.float32)
    return r


def pol(seq0: int, n: int) -> np.ndarray[Any, Any]:
    r = np.zeros(n, R.POLICY)
    r["seq"] = np.arange(seq0, seq0 + n)
    r["t_obs_ns"] = r["seq"].astype(np.int64) * 20 * MS
    r["t_action_ns"] = r["t_obs_ns"] + 40_000
    r["n_obs"] = 74
    r["action"] = 0.1
    return r


def test_every_record_in_order(seg: Any, tmp_path: Path) -> None:
    lg = Logger(seg, LoggerConfig(out_dir=tmp_path, stamp="T", flight=False))
    sent = []
    for k in range(6):  # 6 drains of 500 records (1 s each)
        r = tel(1 + 500 * k, 500)
        assert seg.push_telemetry(r.tobytes()) == 500
        assert seg.push_policy(pol(1 + 50 * k, 50).tobytes()) == 50
        sent.append(r)
        assert lg.poll() == 550
    out = lg.close()
    d = read(out["files"][0])
    got = d.structs["/r1rt/telemetry"]
    assert got.tobytes() == np.concatenate(sent).tobytes()  # byte-identical, in order
    assert len(d.structs["/r1rt/policy"]) == 300
    assert out["telemetry"] == {"records": 3000, "seq_gaps": 0, "seq_backwards": 0, "ring_drops": 0}
    assert len(d.json["/r1log/stats"]) >= 5  # 1 Hz in record time
    np.testing.assert_array_equal(d.log_time["/r1rt/telemetry"], got["t_wake_ns"])


def test_drops_are_counted_twice(seg: Any, tmp_path: Path) -> None:
    """The ring keeps the oldest 4096 and refuses the rest (the producer never waits); r1log reports the ring's drop
    counter and, independently, the hole it sees in the writer's tick numbers."""
    lg = Logger(seg, LoggerConfig(out_dir=tmp_path, stamp="T", flight=False))
    assert seg.push_telemetry(tel(1, 5000).tobytes()) == 4096
    lg.poll()
    seg.push_telemetry(tel(5001, 10).tobytes())
    out = lg.close()
    assert out["telemetry"]["records"] == 4106
    assert out["telemetry"]["ring_drops"] == 904 == out["telemetry"]["seq_gaps"]
    assert seg.ring_stats()["telemetry"]["drops"] == 904


def test_rotation_by_record_time_and_size(seg: Any, tmp_path: Path) -> None:
    lg = Logger(seg, LoggerConfig(out_dir=tmp_path / "span", stamp="T", flight=False, max_span_s=1.0))
    for k in range(7):  # 3.5 s
        seg.push_telemetry(tel(1 + 250 * k, 250).tobytes())
        lg.poll()
    out = lg.close()
    assert len(out["files"]) == 3, out["files"]  # files close once they span >= 1 s (checked per drain)
    parts = [read(f).structs["/r1rt/telemetry"] for f in out["files"]]
    assert np.concatenate(parts)["seq"].tolist() == list(range(1, 1751))
    for f in out["files"]:
        assert read(f).metadata["r1log_close"]["closed_by"] in ("rotation", "close")

    lg2 = Logger(seg, LoggerConfig(out_dir=tmp_path / "size", stamp="T", flight=False, max_bytes=200_000,
                                   chunk_size=64_000))  # fmt: skip
    for k in range(10):
        seg.push_telemetry(tel(10_000 + 300 * k, 300).tobytes())
        lg2.poll()
    out2 = lg2.close()
    assert len(out2["files"]) >= 2
    sizes = [Path(f).stat().st_size for f in out2["files"]]
    assert max(sizes[:-1]) < 200_000 + 2 * 64_000 + 50_000  # a file closes within about one chunk of the limit
    assert sum(len(read(f).structs["/r1rt/telemetry"]) for f in out2["files"]) == 3000


def test_flight_dump_window_and_triggers(seg: Any, tmp_path: Path) -> None:
    cfg = LoggerConfig(out_dir=tmp_path, stamp="T", flight_window_s=2.0, flight_post_s=0.5, flight_holdoff_s=10.0)
    lg = Logger(seg, cfg)
    for k in range(12):  # 6 s at 500 Hz; policy_late rises at seq 2000 (t = 4.0 s) and stays on
        r = tel(1 + 250 * k, 250)
        r["faults"][r["seq"] >= 2000] = R.FAULT_BITS["policy_late"] | R.FAULT_BITS["lowstate_late"]
        seg.push_telemetry(r.tobytes())
        lg.poll()
    out = lg.close()
    assert len(out["flight_files"]) == 1
    d = read(out["flight_files"][0])
    t = d.structs["/r1rt/telemetry"]["t_wake_ns"]
    assert t.min() == 2000 * 2 * MS - 2_000_000_000 and t.max() == 2000 * 2 * MS + 500 * MS
    trig = d.json["/r1log/trigger"]
    assert trig[0]["reason"] == "fault:policy_late" and trig[0]["t_ns"] == 4_000_000_000  # not the 10 ms late bit
    assert d.metadata["r1log"]["reasons"] == "fault:policy_late"
    ev = [e for e in read(out["files"][0]).json["/r1log/event"] if e["kind"] == "fault"]
    assert ev[0]["rising"] == ["lowstate_late", "policy_late"]


def test_holdoff_mode_takeover_tilt_operator(seg: Any, tmp_path: Path) -> None:
    cfg = LoggerConfig(out_dir=tmp_path, stamp="T", flight_window_s=0.5, flight_post_s=0.1, flight_holdoff_s=2.0)
    lg = Logger(seg, cfg)
    seq = 1

    def push(n: int, **kw: Any) -> np.ndarray[Any, Any]:
        nonlocal seq
        r = tel(seq, n, **kw)
        seq += n
        seg.push_telemetry(r.tobytes())
        lg.poll()
        return r

    push(250)
    push(10, faults=R.FAULT_BITS["loop_gap"])  # t 0.5 s: dump 1
    push(240)
    push(10, faults=R.FAULT_BITS["loop_gap"])  # t 1.0 s: same category within 2 s -> suppressed
    push(240)
    push(100, mode=5)  # STAND_FREEZE: a new category -> dump 2
    push(150, mode=5, owner=2)  # the guard takes the token: takeover -> dump 3 (after dump 2's post window)
    r = tel(seq, 200)
    a = np.radians(np.linspace(0.0, 30.0, 200))  # the robot tips past 10 deg
    r["imu_quat"][:, 0], r["imu_quat"][:, 2] = np.cos(a / 2), np.sin(a / 2)
    seq += 200
    seg.push_telemetry(r.tobytes())
    lg.poll()
    push(100)
    seg.request_stop(0, 7, int(seq * 2 * MS))  # an operator STOP on the stop channel
    push(100)
    push(100)
    assert lg.request_dump("bench") in (True, False)
    out = lg.close()
    reasons = [p.split("_", 3)[3].removesuffix(".mcap") for p in (Path(f).name for f in out["flight_files"])]
    assert reasons[:2] == ["fault-loop_gap", "mode-STAND_FREEZE"], reasons
    assert "takeover" in reasons and "tilt" in reasons and "operator-stop" in reasons, reasons
    assert out["flight_suppressed"] >= 1
    ev = read(out["files"][0]).json["/r1log/event"]
    kinds = {e["kind"] for e in ev}
    assert {"fault", "fault_clear", "mode", "token_owner", "stop_request", "dump_request"} <= kinds
    tilt_dump = read(next(f for f in out["flight_files"] if "tilt" in f))
    tr = tilt_dump.json["/r1log/trigger"][0]
    assert tr["reason"] == "tilt" and 10.0 < tr["tilt_deg"] < 10.3


def test_seqlock_samples(seg: Any, tmp_path: Path) -> None:
    """Nothing writes the hands / guard seqlocks yet (r1hands, r1guard): they are sampled only when written."""
    lg = Logger(seg, LoggerConfig(out_dir=tmp_path, stamp="T", flight=False))
    seg.push_telemetry(tel(1, 10).tobytes())
    lg.poll()
    out = lg.close()
    assert out["sampled"] == {"hands_contact": 0, "guard_heartbeat": 0}
    assert seg.hands_contact() is None and seg.guard_heartbeat() is None


def test_cli_attaches_by_name_and_dumps_on_sigusr1(seg: Any, tmp_path: Path) -> None:
    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, "-m", "r1mc3.log", "run", "--out", str(tmp_path), "--segment", seg.name, "--duration-s", "4",
         "--period-ms", "10", "--flight-s", "1", "--post-s", "0", "--nice", "0", "--wait-s", "5"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
    )  # fmt: skip
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert "attached" in line, line
        seg.push_telemetry(tel(1, 600).tobytes())
        time.sleep(0.5)
        proc.send_signal(signal.SIGUSR1)
        time.sleep(0.3)
        seg.push_telemetry(tel(601, 100).tobytes())
        rest, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, rest
    logs = sorted(tmp_path.glob("r1log_*.mcap"))
    dumps = sorted(tmp_path.glob("flight_*request-operator.mcap"))
    assert len(logs) == 1 and len(dumps) == 1, (logs, dumps, rest)
    s = info(logs[0])
    assert s["topics"]["/r1rt/telemetry"]["records"] == 700 and s["topics"]["/r1rt/telemetry"]["seq_gaps"] == 0
    assert read(dumps[0]).json["/r1log/trigger"][0]["reason"] == "request:operator"


def test_a_cut_file_is_read_up_to_the_cut(tmp_path: Path) -> None:
    """r1log killed / power lost: the file has no summary or footer and may end inside a chunk. The indexed reader
    cannot open it; read() walks it sequentially and returns every complete record, flagged truncated."""
    from r1mc3.log import McapFile

    path = tmp_path / "cut.mcap"
    f = McapFile(path, metadata={"t": "cut"}, chunk_size=200_000)
    rec = tel(1, 3000)
    for r in rec:
        f.write("telemetry", int(r["t_wake_ns"]), r.tobytes(), int(r["seq"]))
    f._f.flush()  # the closed chunks are on disk; the open one and the summary are not (no close())
    cut = tmp_path / "copy.mcap"
    cut.write_bytes(path.read_bytes()[:-1000])  # and the last write was torn
    d = read(cut)
    assert d.truncated and d.metadata["r1log"] == {"t": "cut"}
    got = d.structs["/r1rt/telemetry"]
    assert 1000 < len(got) < 3000
    assert got.tobytes() == rec[: len(got)].tobytes()  # a prefix, byte-identical
    assert info(cut)["truncated"] is True
    f.close()
    assert not read(path).truncated and len(read(path).structs["/r1rt/telemetry"]) == 3000


def test_no_flight_dump_for_the_start_before_the_first_lowstate(seg: Any, tmp_path: Path) -> None:
    """Session 8 (HW1 fix #33): r1rt raises lowstate_stale + lowstate_lost from its first tick until the first lowstate
    arrives; both HW1 power-ups wrote a flight file at every start for it. Before the core's first lowstate
    (t_state_rx_ns 0) the two bits are events only; a later real loss still dumps. On bad5a5c: 2 flight files."""
    cfg = LoggerConfig(out_dir=tmp_path, stamp="T", flight_window_s=1.0, flight_post_s=0.2, flight_holdoff_s=0.5)
    lg = Logger(seg, cfg)
    lost = R.FAULT_BITS["lowstate_stale"] | R.FAULT_BITS["lowstate_lost"]
    for k in range(12):  # 6 s at 500 Hz: no state for 0.5 s, then state, then a real loss at 4.0 s
        r = tel(1 + 250 * k, 250, mode=0, owner=0)
        r["t_state_rx_ns"] = np.where(r["seq"] >= 250, r["t_wake_ns"] - 500_000, 0)
        r["faults"][r["seq"] < 250] = lost
        late = r["seq"] >= 2000
        r["t_state_rx_ns"][late] = 2000 * 2 * MS - 500_000  # the last state before the loss, frozen
        r["faults"][late & (r["seq"] >= 2010)] = lost
        seg.push_telemetry(r.tobytes())
        lg.poll()
    out = lg.close()
    reasons = [read(f).metadata["r1log"]["reasons"] for f in out["flight_files"]]
    assert reasons == ["fault:lowstate_stale+lowstate_lost"], reasons
    trig = read(out["flight_files"][0]).json["/r1log/trigger"]
    assert trig[0]["t_ns"] == 2010 * 2 * MS  # the real loss, not the start
    ev = [e for e in read(out["files"][0]).json["/r1log/event"] if e["kind"] == "fault"]
    assert ev[0]["rising"] == ["lowstate_stale", "lowstate_lost"] and ev[0]["seq"] == 1  # the start: still an event
