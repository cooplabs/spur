"""r1d's session-6 words against a fake core port (no C++ needed): `r1 status` in plain words, `r1 reclaim`, `r1
board-hold`, `r1 guard-takeover`, `r1 shutdown`, the record aliases / --label, and the pause tool on a REAL child
process (SIGSTOP / SIGCONT timing, the detached SIGCONT backstop when r1d's own SIGCONT never comes). Every op here is
new in session 6: on the base (26e8bce) each request answers UNKNOWN_COMMAND (the schema has no such op), so every test
below fails there."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.cli.main import build_request
from r1mc3.hw1 import pause
from r1mc3.hw1.ops import Hw1Ops
from r1mc3.hw1.session import find_r1log, save_session
from r1mc3.record.plans import plan
from r1mc3.service.codes import CoreTables
from r1mc3.service.port import Heartbeat
from r1mc3.service.service import Service

MODES = CoreTables(None).modes
GUARD_ABSENT = 1 << 14
NOT_OWNER = 1 << 13


class FakePort:
    """The segment as r1d sees it, with the guard's heartbeat, the token, the armed flag and r1rt's pid settable."""

    def __init__(self, mode: str = "on_stand") -> None:
        self.tables = CoreTables(None)
        self.tables.requests = {**self.tables.requests, "on_stand": 9, "reclaim": 10, "stand_hold": 11}
        self.tables.refusals = {"not_operator": 20, "mode": 21, "state": 24, "far": 25, "stand_session": 26}
        self.tables.refusal_names = {v: k for k, v in self.tables.refusals.items()}
        self.tables.status_codes = {"ok": 0, "refused": -6, "not_owner": -5}
        self.tables.status_names = {v: k for k, v in self.tables.status_codes.items()}
        self.mode = MODES[mode]
        self.t = 10_000_000_000
        self.core_age_ns = 1_000_000
        self.guard_age_ns: int | None = 1_000_000
        self.owner = 1
        self.gen = 3
        self.faults = 0
        self.armed: bool | None = True
        self.pid: int | None = None
        self.requests: list[tuple[int, int]] = []
        self.replies: dict[int, tuple[int, int, int]] = {}
        self.shm_name = "/r1mc3.fake"
        self.rt: dict[str, Any] | None = None  # SegmentView.rt_info() (None: a binding without it)

    def rt_info(self) -> dict[str, Any] | None:
        return self.rt

    def missing(self) -> list[str]:
        return []

    def now_ns(self) -> int:
        self.t += 1_000
        return self.t

    def heartbeat(self) -> Heartbeat:
        return Heartbeat(1, self.t - self.core_age_ns, 0, 500, self.faults, 1.0, self.mode, False)

    def guard_heartbeat(self) -> Heartbeat | None:
        if self.guard_age_ns is None:
            return None
        return Heartbeat(1, self.t - self.guard_age_ns, 0, 1000, 0, 0.0, MODES["monitor"], False)

    def guard_armed(self) -> bool | None:
        return self.armed

    def rt_pid(self) -> int | None:
        return self.pid

    def token(self) -> tuple[int, int]:
        return (self.owner, self.gen)

    def request_stop(self, level: int, requester: int, t_ns: int) -> None:
        pass

    def stop_epoch(self) -> int:
        return 0

    def push_upper(self, rec: Any) -> int:
        return 0

    def upper_keepalive(self, group: int, cmd_id: int, t_ns: int) -> int:
        return 0

    def request_mode(self, req: int, requester: int, t_ns: int) -> int:
        self.requests.append((req, requester))
        return len(self.requests)

    def mode_reply(self, seq: int) -> tuple[int, int, int] | None:
        return self.replies.get(seq, (0, 0, self.mode))

    def targets(self) -> np.ndarray:
        return np.zeros(26)

    def measured(self) -> np.ndarray:
        return np.zeros(26)


class FakeRecorder:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def arrays(self) -> dict[str, np.ndarray]:
        t = np.arange(pause.monotonic_ns() - 4_000_000_000, pause.monotonic_ns(), 2_000_000, dtype=np.int64)
        return {"t_rx_ns": t, "tick": np.arange(t.size, dtype=np.uint32) * 2, "q": np.zeros((t.size, 35), np.float32)}


def _svc(port: FakePort, tmp: Path, **hw1: Any) -> Service:
    return Service(
        port,
        journal_dir=tmp,
        hw1_factory=lambda s: Hw1Ops(s, record_dir=tmp / "records", log_dir=tmp / "logs", **hw1),
    )


def _run(svc: Service, req: dict[str, Any], ticks: int = 3) -> dict[str, Any]:
    t = svc.submit(req)
    for _ in range(ticks):
        if t.done.is_set():
            break
        svc.tick()
    assert t.done.wait(30.0), req
    assert t.result is not None
    return t.result.model_dump(mode="json")


def _op(op: str, **kw: Any) -> dict[str, Any]:
    return {"op": op, "requester": "operator", **kw}


# ---------------------------------------------------------------------------------------------- status in words
def test_status_alarms_include_the_over_temperature_warning(tmp_path: Path) -> None:
    """core's R1RT_FAULT_OVER_TEMP (bit 18, session 6: a warning only, never an action) is an operator alarm in
    `r1 status` like MOTOR_ERROR (integrator; fails before OVER_TEMP joined ALARM_FAULTS)."""
    port = FakePort("on_stand")
    port.faults = 1 << 18
    st = _svc(port, tmp_path).status()
    assert "OVER_TEMP" in st["alarms"], st["alarms"]


def test_status_words_guard_states(tmp_path: Path) -> None:
    port = FakePort("on_stand")
    svc = _svc(port, tmp_path)
    st = svc.status()
    assert st["guard"]["alive"] and st["guard"]["armed"] is True and not st["guard"]["active"]
    assert any("ARMED" in w for w in st["summary"]) and any("owns the motors" in w for w in st["summary"])
    port.armed = False
    assert any("OBSERVE-ONLY" in w for w in svc.status()["summary"])
    port.guard_age_ns = 2_000_000_000  # stopped
    assert any("STOPPED" in w and "r1guard" in w for w in svc.status()["summary"])
    port.guard_age_ns = None
    port.faults = GUARD_ABSENT
    st = svc.status()
    assert st["guard"] is None and "GUARD_ABSENT" in st["alarms"] and any("ABSENT" in w for w in st["summary"])
    port.guard_age_ns, port.owner, port.faults, port.armed = 1_000_000, 2, NOT_OWNER, True
    st = svc.status()
    assert st["guard"]["active"] and st["guard"]["taken_over_from"] == "on_stand"
    assert any("SILENT" in w for w in st["summary"]) and any("ACTIVE" in w for w in st["summary"])
    port.armed = None  # a build that cannot tell
    port.owner = 1
    assert any("not reported" in w for w in svc.status()["summary"])


# ---------------------------------------------------------------------------------------------- reclaim
def test_reclaim_operator_only_and_preconditions(tmp_path: Path) -> None:
    port = FakePort("on_stand")
    svc = _svc(port, tmp_path)
    r = _run(svc, {"op": "reclaim", "requester": "r1d"})
    assert r["code"] == "NOT_PERMITTED" and not r["moved"]
    r = _run(svc, _op("reclaim"))  # r1rt owns the token: nothing to reclaim
    assert r["code"] == "WRONG_MODE" and "nothing to reclaim" in r["details"]["reason"]
    port.owner = 2
    port.core_age_ns = 3_000_000_000  # r1rt still stopped
    r = _run(svc, _op("reclaim"))
    assert r["code"] == "FAULT" and "not running" in r["details"]["reason"]
    assert port.requests == []  # nothing reached the core's mailbox so far
    port.core_age_ns = 1_000_000
    port.replies[1] = (-6, 21, MODES["on_stand"])  # the core refuses: REFUSE_MODE (e.g. the guard took over in BALANCE)
    r = _run(svc, _op("reclaim"))
    assert r["code"] == "WRONG_MODE" and r["details"]["refusal_name"] == "mode"
    assert port.requests == [(10, 4)]  # the core's `reclaim`, requester OPERATOR
    r = _run(svc, _op("reclaim"))
    assert r["ok"] and r["data"]["mode"] == "on_stand" and "token" in r["data"]


def test_reclaim_unsupported_without_the_request(tmp_path: Path) -> None:
    port = FakePort("on_stand")
    port.tables.requests = {k: v for k, v in port.tables.requests.items() if k != "reclaim"}
    port.owner = 2
    r = _run(_svc(port, tmp_path), _op("reclaim"))
    assert r["code"] == "UNSUPPORTED"


# ---------------------------------------------------------------------------------------------- board-hold /
# guard-takeover checks
def test_board_hold_refusals_and_dry_run(tmp_path: Path) -> None:
    port = FakePort("on_stand")
    port.pid = os.getpid()
    svc = _svc(port, tmp_path)
    r = _run(svc, {"op": "board_hold.pause", "pause_ms": 100, "dry_run": False})  # not the operator
    assert r["code"] == "NOT_PERMITTED"
    r = _run(svc, _op("board_hold.pause", pause_ms=100, dry_run=False))  # the guard is ARMED
    assert r["code"] == "WRONG_MODE" and r["details"]["failed"] == ["guard OFF"]
    port.armed = None  # cannot tell -> refused as well
    assert _run(svc, _op("board_hold.pause", pause_ms=100))["details"]["failed"] == ["guard OFF"]
    r = _run(svc, _op("board_hold.pause", pause_ms=150))  # not one of the five durations (20 / 100 / 300 / 400 / 500)
    assert r["code"] == "INVALID_ARGS"
    port.armed = False  # observe-only
    port.mode = MODES["balance"]
    r = _run(svc, _op("board_hold.pause", pause_ms=100))
    assert r["code"] == "WRONG_MODE" and r["details"]["failed"] == ["ON_STAND"]
    port.mode = MODES["on_stand"]
    port.guard_age_ns = None  # stopped: fine
    r = _run(svc, _op("board_hold.pause", pause_ms=500))
    assert r["ok"] and r["data"]["dry_run"] and all(c["ok"] for c in r["data"]["checks"]) and not r["moved"]
    port.pid = None
    assert _run(svc, _op("board_hold.pause", pause_ms=500))["details"]["failed"] == ["r1rt pid"]


def test_guard_takeover_refusals_and_dry_run(tmp_path: Path) -> None:
    port = FakePort("on_stand")
    port.pid = os.getpid()
    svc = _svc(port, tmp_path)
    port.armed = False
    assert _run(svc, _op("guard_takeover.stall", stop_s=2.0))["details"]["failed"] == ["guard armed"]
    port.armed, port.guard_age_ns = True, None
    assert _run(svc, _op("guard_takeover.stall", stop_s=2.0))["details"]["failed"] == ["guard alive"]
    port.guard_age_ns = 1_000_000
    r = _run(svc, _op("guard_takeover.stall", stop_s=2.0))
    assert r["ok"] and r["data"]["dry_run"] and r["data"]["plan"]["next"] == "r1 reclaim"
    assert _run(svc, _op("guard_takeover.stall", stop_s=20.0))["code"] == "INVALID_ARGS"  # <= 8 s


def test_board_hold_run_with_fake_pause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    port = FakePort("on_stand")
    port.pid, port.armed = os.getpid(), False
    calls: list[tuple[int, float]] = []

    def fake_pause(pid: int, dur_s: float, **kw: Any) -> pause.Paused:
        calls.append((pid, dur_s))
        p = pause.Paused(pid=pid, requested_s=dur_s)
        p.t_stop_ns = pause.monotonic_ns()
        time.sleep(dur_s)
        p.t_cont_ns = pause.monotonic_ns()
        return p

    svc = _svc(port, tmp_path, recorder_factory=FakeRecorder, pauser=fake_pause)
    import r1mc3.hw1.ops as ops

    monkeypatch.setattr(ops, "RECORD_PRE_S", 0.01)
    monkeypatch.setattr(ops, "RECORD_POST_S", 0.01)
    r = _run(svc, _op("board_hold.pause", pause_ms=20, dry_run=False))
    assert r["ok"], r
    assert calls == [(os.getpid(), 0.02)]
    d = r["data"]
    assert d["lowstate"]["recorded"] and Path(d["files"]["lowstate"]).exists() and Path(d["files"]["manifest"]).exists()
    assert [e["event"] for e in d["timeline"]][1] == "SIGSTOP r1rt"
    man = json.loads(Path(d["files"]["manifest"]).read_text())
    assert man["schema"] == "r1mc3.hw1_board_hold/1" and man["pause"]["requested_ms"] == 20.0


def test_takeover_net_continues_when_the_guard_does_not_take_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard never takes the token -> r1d continues r1rt after 0.2 s and reports FAULT (not S seconds silent)."""
    port = FakePort("on_stand")
    port.pid = os.getpid()
    seen: dict[str, Any] = {}

    def fake_pause(pid: int, dur_s: float, *, watch: Any = None, **kw: Any) -> pause.Paused:
        p = pause.Paused(pid=pid, requested_s=dur_s)
        p.t_stop_ns = pause.monotonic_ns()
        t0 = time.monotonic()
        while time.monotonic() - t0 < dur_s:
            why = watch(time.monotonic() - t0)
            if why:
                p.early = why
                break
            time.sleep(0.005)
        seen["elapsed"] = time.monotonic() - t0
        p.t_cont_ns = pause.monotonic_ns()
        return p

    import r1mc3.hw1.ops as ops

    monkeypatch.setattr(ops, "TAKEOVER_PRE_S", 0.01)
    monkeypatch.setattr(ops, "TAKEOVER_POST_S", 0.01)
    svc = _svc(port, tmp_path, recorder_factory=FakeRecorder, pauser=fake_pause)
    r = _run(svc, _op("guard_takeover.stall", stop_s=3.0, dry_run=False))
    assert r["code"] == "FAULT" and "did not take the token" in r["details"]["reason"]
    assert seen["elapsed"] < 0.5


# ---------------------------------------------------------------------------------------------- the pause tool, real
def _ticker() -> subprocess.Popen[str]:
    code = (
        "import time\nwhile True:\n print(time.clock_gettime_ns(time.CLOCK_MONOTONIC), flush=True)\n"
        " time.sleep(0.001)\n"
    )
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)


def _max_gap_ms(p: subprocess.Popen[str], first_ns: int, until_ns: int) -> float:
    """The ticker's largest gap from its line `first_ns` (read before the pause) to its first line after `until_ns`."""
    assert p.stdout is not None
    ts: list[int] = [first_ns]
    for _ in range(100_000):
        ts.append(int(p.stdout.readline()))
        if ts[-1] > until_ns:
            break
    return float(np.diff(ts).max()) * 1e-6


@pytest.mark.realtime  # wall-clock: a 100 ms pause measured to < 140 ms (a shared CI VM: ~158)
@pytest.mark.parametrize("in_process", [False, True])
def test_sigstop_for_real_child(in_process: bool) -> None:
    child = _ticker()
    try:
        assert child.stdout is not None
        first = int(child.stdout.readline())
        out = pause.sigstop_for(child.pid, 0.1, in_process=in_process)
        assert 99.0 <= out.actual_ms < 140.0 and not out.errors, out.as_dict()
        assert _max_gap_ms(child, first, out.t_cont_ns) >= 95.0  # the child really stopped ~0.1 s
        assert child.poll() is None  # and runs again
    finally:
        child.kill()
        child.wait()


def test_backstop_continues_when_r1d_never_sends_sigcont(monkeypatch: pytest.MonkeyPatch) -> None:
    """r1d dies between SIGSTOP and SIGCONT (emulated: its own SIGCONT dropped) -> the detached backstop continues."""
    import signal as sig

    real_kill = os.kill
    child = _ticker()
    try:
        assert child.stdout is not None
        child.stdout.readline()
        monkeypatch.setattr(os, "kill", lambda pid, s: None if s == sig.SIGCONT else real_kill(pid, s))
        out = pause.sigstop_for(child.pid, 0.05, in_process=True)  # no pause child either: r1d "died"
        monkeypatch.setattr(os, "kill", real_kill)
        assert out.backstop_pid is not None
        for _ in range(5000):  # the lines printed before the stop are still buffered in the pipe
            if (ts := int(child.stdout.readline())) > out.t_stop_ns:
                break
        resumed_s = (ts - out.t_stop_ns) * 1e-9  # the first line after the stop: the backstop's SIGCONT (0.55 s)
        assert 0.45 < resumed_s < 2.0, resumed_s
    finally:
        child.kill()
        child.wait()


# ---------------------------------------------------------------------------------------------- shutdown
def test_shutdown_refused_while_we_own_the_motors(tmp_path: Path) -> None:
    for mode, owner in (("on_stand", 1), ("balance", 1), ("on_stand", 2)):
        port = FakePort(mode)
        port.owner = owner
        r = _run(_svc(port, tmp_path), _op("shutdown"))
        assert r["code"] == "WRONG_MODE" and not r["moved"], (mode, owner)
    port = FakePort("damp")
    assert _run(_svc(port, tmp_path), {"op": "shutdown", "requester": "r1d"})["code"] == "NOT_PERMITTED"


def test_shutdown_saves_and_verifies(tmp_path: Path) -> None:
    logs, recs = tmp_path / "logs", tmp_path / "records"
    logs.mkdir()
    recs.mkdir()
    (logs / "r1log_a.mcap").write_bytes(os.urandom(4096))
    (recs / "20261010T000000Z_ramp_left_knee.json").write_text("{}")
    old = logs / "old_session.mcap"
    old.write_bytes(b"x")
    # session 8 (HW1 #20): older than the 24 h look-back = an earlier day's file (it was "1 h old = not this session's"
    # on bad5a5c; a 1 h old unsaved file of the same day IS saved now - test_deploy_s8 covers it)
    os.utime(old, (time.time() - 25 * 3600, time.time() - 25 * 3600))
    port = FakePort("damp")
    svc = _svc(port, tmp_path)
    svc.handle(_op("damp"))  # a journaled word
    r = _run(svc, _op("shutdown"))
    assert r["ok"], r
    d = r["data"]
    assert d["message"] == "safe to power off now" and d["saved"]["verified"]
    folder = Path(d["saved"]["folder"])
    man = json.loads((folder / "manifest.json").read_text())
    files = {e["file"] for e in man["files"]}
    assert "r1log/r1log_a.mcap" in files and "records/20261010T000000Z_ramp_left_knee.json" in files
    assert "r1log/old_session.mcap" not in files and "r1d_journal.jsonl" in files
    assert all(e["verified"] for e in man["files"])
    journal = (folder / "r1d_journal.jsonl").read_text().splitlines()
    assert json.loads(journal[0])["request"]["op"] == "damp"


def test_save_session_detects_a_bad_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logs, recs = tmp_path / "logs", tmp_path / "records"
    logs.mkdir()
    recs.mkdir()
    (logs / "a.mcap").write_bytes(b"abc")
    import r1mc3.hw1.session as sess

    real = sess.sha256_of
    monkeypatch.setattr(sess, "sha256_of", lambda p: "bad" if "sessions" in str(p) else real(p))
    out = save_session(
        session_dir=recs / "sessions" / "x", log_dir=logs, record_dir=recs, since_wall=time.time() - 10, journal=[]
    )
    assert not out["verified"] and out["failed"] == ["r1log/a.mcap"]


def test_find_and_stop_r1log_by_segment(tmp_path: Path) -> None:
    code = (
        "import signal,time,sys\nsignal.signal(signal.SIGINT, lambda *a: sys.exit(0))\nwhile True: time.sleep(0.05)\n"
    )
    script = tmp_path / "fake_r1log.py"
    script.write_text(code)
    # a process whose command line looks like r1log's for OUR segment (argv: ... r1mc3.log run --segment NAME)
    p = subprocess.Popen([sys.executable, str(script), "r1mc3.log", "run", "--segment", "/r1mc3.svctest"])
    other = subprocess.Popen([sys.executable, str(script), "r1mc3.log", "run", "--segment", "/r1mc3.other"])
    try:
        time.sleep(0.3)
        assert find_r1log("/r1mc3.svctest") == [p.pid]
        from r1mc3.hw1.session import stop_r1log

        assert stop_r1log([p.pid], timeout_s=5.0) == {str(p.pid): "exited"}
    finally:
        for q in (p, other):
            q.kill()
            q.wait()


# ---------------------------------------------------------------------------------------------- words + records
def test_cli_words() -> None:
    assert build_request(["reclaim"])[0] == {"requester": "operator", "op": "reclaim"}
    assert build_request(["shutdown"])[0]["op"] == "shutdown"
    r, _ = build_request(["board-hold", "--pause-ms", "500"])
    assert r == {"requester": "operator", "op": "board_hold.pause", "pause_ms": 500, "dry_run": True}
    assert build_request(["board-hold", "--pause-ms", "500", "--yes"])[0]["dry_run"] is False
    r, _ = build_request(["guard-takeover", "--stop-s", "2", "--yes"])
    assert r["op"] == "guard_takeover.stall" and r["stop_s"] == 2.0 and r["dry_run"] is False
    with pytest.raises(SystemExit):  # session 8 (#34): 300 is a board-hold step now, 1000 / 2000 are refused
        build_request(["board-hold", "--pause-ms", "1000"])
    r, _ = build_request(["record", "run", "torque_ramp", "--joint", "left_knee", "--yes"])
    assert r["kind"] == "torque_ramp" and r["params"] == {"joint": "left_knee"} and r["dry_run"] is False
    r, _ = build_request(["record", "run", "statics", "--label", "com_board", "duration_s=30", "--yes", "--dry-run"])
    assert r["params"] == {"duration_s": 30, "label": "com_board"} and r["dry_run"] is True


def test_cli_negative_first_value() -> None:
    """hang's report: `r1 joint leg_left -0.25,...` failed (argparse took the list for an option)."""
    r, _ = build_request(["joint", "leg_left", "-0.25,0.01,0,0.5,-0.2,0", "--ttl", "0.3"])
    assert r["q"] == [-0.25, 0.01, 0.0, 0.5, -0.2, 0.0] and r["ttl"] == 0.3
    r, _ = build_request(["jog", "arm_left", "-.1,0,0,0,0"])
    assert r["dq"][0] == -0.1
    r, _ = build_request(["hand", "left", "0,0,0,0,0,0,0", "--kp", "-1e-3,1,1,1,1,1,1"])
    assert r["kp"][0] == -1e-3


def test_record_aliases_and_label() -> None:
    p = plan("statics", {"label": "com_board", "duration_s": 30}, None)
    assert p.kind == "imu_static" and p.params["asked_as"] == "statics" and p.params["label"] == "com_board"
    assert not p.joints
    r = plan("torque_ramp", {"joint": "left_knee"}, np.zeros(26))
    assert r.kind == "ramp" and r.joints == (3,)
    from r1mc3.record.plans import PlanError

    with pytest.raises(PlanError):
        plan("statics", {"label": "../x"}, None)


# ---------------------------------------------------------------------------------------------- hands contact
class FakeHands:
    def __init__(self) -> None:
        self.p = {s: np.full((9, 12), 100000.0) for s in ("left", "right")}  # streaming pads (HW2: ~1e5 raw)
        self.valid = True

    def state(self, side: str) -> dict[str, Any]:
        return {"pressure": self.p[side].copy(), "valid": self.valid}

    def command(self, side: str, q: Any, **kw: Any) -> dict[str, Any]:
        return {"ok": True}


def test_contact_bits_into_the_segment_writer(tmp_path: Path) -> None:
    from r1mc3.hands.contact import ContactPublisher

    hands = FakeHands()
    calls: list[tuple[int, int, int, bool]] = []
    pub = ContactPublisher(hands, lambda pl, pr, t_ns, valid: calls.append((pl, pr, t_ns, valid)), tare_samples=5)
    svc = Service(FakePort("on_stand"), hands=hands, contact=pub, journal_dir=tmp_path)
    for _ in range(5):
        svc.tick()
    pl, pr, _, valid = calls[-1]
    assert (pl, pr, valid) == (0, 0, True)  # taring: valid, no contact
    hands.p["left"][2, 7] = 100060.0  # pad 2 of the left hand pressed (+60 over its tare; session 12: 40 raw)
    hands.p["right"][8, 0] = 100030.0  # +30: below the threshold
    svc.tick()
    assert calls[-1][:2] == (1 << 2, 0)
    hands.p["right"][8, 0] = 100045.0
    svc.tick()
    assert calls[-1][:2] == (1 << 2, 1 << 8)
    hands.valid = False
    svc.tick()
    pl, pr, _, valid = calls[-1]
    assert (pl, pr, valid) == (0, 0, False)
    st = svc.status()
    assert st["hands_contact"]["writes"] == 8 and st["hands_contact"]["valid"] is False


def test_takeover_watch_continues_early_with_the_pause_child() -> None:
    """The real pause child + the guard-takeover test's watch: a watch that fires at 50 ms continues the process at once
    (not after 2 s)."""
    child = _ticker()
    try:
        assert child.stdout is not None
        first = int(child.stdout.readline())
        t0 = time.monotonic()
        out = pause.sigstop_for(child.pid, 2.0, watch=lambda el: "no takeover" if el > 0.05 else None)
        assert out.early == "no takeover" and out.actual_ms < 200.0 and time.monotonic() - t0 < 1.0, out.as_dict()
        assert _max_gap_ms(child, first, out.t_cont_ns) >= 45.0
    finally:
        child.kill()
        child.wait()


# ---------------------------------------------------------------------------------------------- stand freeze (S10)
def _cfg(tmp: Path) -> Path:
    stand = [-0.3, 0.025, -0.13, 0.455, -0.21, -0.05, -0.3, -0.025, 0.13, 0.455, -0.21, 0.05] + [0.0] * 14
    p = tmp / "r1rt.cfg"
    p.write_text("format = r1rt-config/1\nstand_ramp_s = 0.4\nstand_q = " + " ".join(map(str, stand)) + "\n")
    return p


def test_stand_freeze_checks_and_mapping(tmp_path: Path) -> None:
    port = FakePort("on_stand")
    svc = Service(port, journal_dir=tmp_path, rt_config=_cfg(tmp_path))
    assert _run(svc, {"op": "stand_freeze", "requester": "r1d"})["code"] == "NOT_PERMITTED"
    port.mode = MODES["balance"]
    assert _run(svc, _op("stand_freeze"))["code"] == "WRONG_MODE"  # only on the stand
    port.mode = MODES["on_stand"]
    dry = _run(svc, _op("stand_freeze"))  # targets = zeros: the knees 0.455 rad from the stand pose
    assert dry["ok"] and dry["data"]["plan"]["ramp_s"] == 0.4 and dry["data"]["far_check"]["ok"] is False
    assert dry["data"]["far_check"]["worst_joint"] in ("left_knee", "right_knee")
    far = _run(svc, _op("stand_freeze", dry_run=False))
    assert far["code"] == "LIMIT" and port.requests == []  # r1d's own far check: nothing reached the core
    svc.rt_config_path = None  # without r1rt's config r1d cannot check: the core decides
    port.replies[1] = (-6, 25, MODES["on_stand"])  # the core's own far refusal
    r = _run(svc, _op("stand_freeze", dry_run=False))
    assert r["code"] == "LIMIT" and r["details"]["refusal_name"] == "far" and "0.15 rad" in r["details"]["reason"]
    port.replies[2] = (0, 0, MODES["stand_freeze"])
    r = _run(svc, _op("stand_freeze", dry_run=False))
    assert r["ok"] and r["moved"] and "stand-hold" in r["data"]["next"] and port.requests[-1] == (5, 4)
    port.mode = MODES["stand_freeze"]
    st = svc.status()
    assert st["freeze_from_stand"] and any("STAND_FREEZE (from the stand)" in w for w in st["summary"])
    hold = _run(svc, _op("stand_hold"))
    assert hold["ok"] and port.requests[-1] == (11, 4) and "resume" in hold["data"]["next"]
    port.mode = MODES["on_stand"]
    assert _run(svc, _op("stand_hold"))["code"] == "WRONG_MODE"
    assert not any("STAND_FREEZE" in w for w in svc.status()["summary"])


def test_stand_freeze_words() -> None:
    assert build_request(["stand-freeze"])[0] == {"requester": "operator", "op": "stand_freeze", "dry_run": True}
    assert build_request(["stand-freeze", "--yes"])[0]["dry_run"] is False
    assert build_request(["stand-hold"])[0]["op"] == "stand_hold"


def test_shutdown_manifest_survives_later_requests(tmp_path: Path) -> None:
    """Rehearsal R4: r1d's own live journal (in the records dir, appended after every request) was HARD-LINKED into
    the session folder, so a request after `r1 shutdown` changed the saved copy and broke the manifest. Now it is a
    copied snapshot: every manifest entry still verifies after more requests."""
    recs = tmp_path / "records"
    port = FakePort("damp")
    svc = Service(
        port,
        journal_dir=recs,  # as on the robot: r1d's live journal goes into R1_RECORD_DIR
        hw1_factory=lambda s: Hw1Ops(s, record_dir=recs, log_dir=tmp_path / "logs"),
    )
    svc.handle(_op("damp"))
    out = svc.handle(_op("shutdown"))
    assert out["ok"], out
    folder = Path(out["data"]["saved"]["folder"])
    for _ in range(3):
        svc.handle(_op("stop"))  # r1d keeps running and journaling after the save
    from r1mc3.hw1.session import sha256_of

    man = json.loads((folder / "manifest.json").read_text())
    bad = [e["file"] for e in man["files"] if sha256_of(folder / e["file"]) != e["sha256"]]
    assert bad == [], bad  # on 7d9c62f: ['records/r1d_journal_<UTC>_<pid>.jsonl'] (linked, then appended)
    live = [e for e in man["files"] if e["file"].startswith("records/r1d_journal_")]
    assert live and live[0]["how"] == "snapshot"


# ---------------------------------------------------------------------------------------------- stand session
class CountingReleaser:
    synchronous = True

    def __init__(self) -> None:
        self.calls = 0

    def release(self, timeout_s: float) -> tuple[bool, dict[str, Any]]:
        self.calls += 1
        return True, {}


def test_takeover_refused_before_anything_in_a_stand_session(tmp_path: Path) -> None:
    """r1rt --stand-session: `r1 takeover` is refused by r1d itself - nothing reaches the core's mailbox and NO
    ReleaseMode is sent (Unitree's controller keeps the robot); `r1 on-stand` still runs."""
    port = FakePort("monitor")
    port.rt = {"stand_session": True}
    rel = CountingReleaser()
    svc = Service(port, releaser=rel, journal_dir=tmp_path)
    r = _run(svc, _op("takeover"))
    assert r["code"] == "WRONG_MODE" and not r["moved"] and "--stand-session" in r["details"]["reason"], r
    assert port.requests == [] and rel.calls == 0
    st = svc.status()
    assert st["core"]["stand_session"] is True
    assert "stand session: balance REFUSED (r1rt --stand-session)" in st["summary"]
    on = _run(svc, _op("on_stand"), ticks=10)
    assert on["ok"] and port.requests[0][0] == port.tables.requests["on_stand"] and rel.calls == 1


def test_stand_session_refusal_from_the_core_without_rt_info(tmp_path: Path) -> None:
    """A binding without rt_info: r1d cannot tell, the core refuses at the gates (refusal `stand_session`) -> the
    same WRONG_MODE + reason, and still no ReleaseMode (the gates come first)."""
    port = FakePort("monitor")
    port.rt = None
    port.replies[1] = (-6, 26, MODES["monitor"])
    rel = CountingReleaser()
    svc = Service(port, releaser=rel, journal_dir=tmp_path)
    assert svc.status()["core"]["stand_session"] is None
    assert not any("stand session" in w for w in svc.status()["summary"])
    r = _run(svc, _op("takeover"), ticks=5)
    assert r["code"] == "WRONG_MODE" and r["details"]["refusal_name"] == "stand_session", r
    assert "--stand-session" in r["details"]["reason"] and rel.calls == 0
    port.rt = {"stand_session": False}
    assert svc.status()["core"]["stand_session"] is False
