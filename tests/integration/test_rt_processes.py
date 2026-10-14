# mypy: disable-error-code="index,assignment,attr-defined,unreachable"
# (CycloneDDS IDL fields are typed as its annotation classes - types.array - and samples carry a run-time sample_info)
"""r1rt and r1guard as REAL processes over DDS (session 5, M4d item 1; DESIGN §2.1, §4.3; PROCESSES.md).

A FAKE robot in this test process speaks the robot's side of the wire (r1mc3.dds.hg = Unitree's hg types for
CycloneDDS's Python binding): a static double-support stand on rt/lowstate at 500 Hz (Unitree's CRC), Unitree's
built-in controller as a separate participant publishing a stiff stand on rt/lowcmd until our takeover, and a reader
of rt/lowcmd that records every command of OUR processes (the built-in's own writer is in this process: ignored with
IgnoreLocal) with its source timestamp, writer handle and the legs' lowest kp. No physics: the state does not move,
so these tests prove the processes, the wire and the timing; the physical stall test is the A7 contrast on the
DDS-bridge simulator (r1sim.tier_a.a7_processes).

What is proved here:
  * both processes start and print their ready line (rt_granted false on macOS, said), the config file written by
    r1mc3.rtconfig parses in C++ (every key, nothing unknown);
  * MONITOR publishes nothing; r1guard on standby publishes nothing;
  * THE CROSS-LANGUAGE CHECK: our C++ hg types and the Python peer exchange the same bytes - r1rt reads Python's
    LowState_ with a valid CRC (and flags a corrupted one), Python reads r1rt's LowCmd_ and its CRC verifies;
  * the takeover through the segment's mode mailbox, as r1d sends it (TAKEOVER -> MIRROR of the built-in -> the
    built-in lets go -> TAKEOVER_RELEASED -> BLEND -> BALANCE) with the guard gate; DAMP only from the operator;
  * the 9 Oct stall (4.018 s) injected into r1rt (SIGSTOP of the whole process, and the writer-sleep variant):
    r1guard takes over - the command gap seen by the robot, one writer at a time, legs never limp; after SIGCONT r1rt
    stays silent (NOT_OWNER; at most one packet of the residual race);
  * kill -9 of r1rt -> r1guard takes over; r1guard absent at start -> the takeover is refused (refusal GUARD).
The timing numbers are wall-clock on a loaded Mac without real-time scheduling: asserted with margins, reported as
measured (the integrator re-measures them alone).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1mc3.dds import hg, rpc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LOOPBACK_URI = rpc.LOOPBACK_URI  # lo0 on macOS, lo on Linux (r1mc3.dds.rpc)
os.environ["CYCLONEDDS_URI"] = LOOPBACK_URI  # this process's participants (set before the first one is created)

STALL_S = 4.018  # the 9 Oct stall (serve-20261010T010206Z telemetry:1137-1138)
LEGS = range(12)  # SDK joints 0-11 = hg slots 0-11


def _apps_dir() -> Path | None:
    roots = [Path(os.environ["R1MC3_BUILD_DIR"])] if os.environ.get("R1MC3_BUILD_DIR") else []
    roots.append(REPO / "build")
    for r in roots:
        d = r / "rt_dds" / "apps"
        if (d / "r1rt").exists() and (d / "r1guard").exists():
            return d
    return None


_APPS = _apps_dir()
if _APPS is None:
    pytest.skip("r1rt / r1guard not built (scripts/ci.sh dds: -DR1RT_WITH_CYCLONEDDS=ON)", allow_module_level=True)
APPS: Path = _APPS  # not None past the skip (assignment code disabled above)


# --------------------------------------------------------------------------------------------------------------------
# the processes


class Proc:
    """A child process whose stdout JSON lines are collected by a reader thread."""

    def __init__(self, args: list[str]) -> None:
        self.args = args
        env = dict(os.environ, CYCLONEDDS_URI=LOOPBACK_URI)
        self.p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.lines: list[dict[str, Any]] = []
        self.raw: list[str] = []
        self._cv = threading.Condition()
        self._t = threading.Thread(target=self._read, daemon=True)
        self._t.start()

    def _read(self) -> None:
        assert self.p.stdout is not None
        for line in self.p.stdout:
            with self._cv:
                self.raw.append(line)
                with contextlib.suppress(json.JSONDecodeError):
                    self.lines.append(json.loads(line))
                self._cv.notify_all()

    def wait_for(self, pred: Callable[[dict[str, Any]], bool], timeout: float) -> dict[str, Any]:
        end = time.monotonic() + timeout
        with self._cv:
            while True:
                for d in self.lines:
                    if pred(d):
                        return d
                left = end - time.monotonic()
                if left <= 0 or self.p.poll() is not None:
                    err = self.p.stderr.read() if self.p.poll() is not None and self.p.stderr else ""
                    raise AssertionError(f"{Path(self.args[0]).name}: no matching line in {timeout} s; got "
                                         f"{self.raw[-5:]} stderr={err[-500:]}")  # fmt: skip
                self._cv.wait(left)

    def events(self, name: str) -> list[dict[str, Any]]:
        with self._cv:
            return [d for d in self.lines if d.get("event") == name]

    def last_status(self) -> dict[str, Any] | None:
        with self._cv:
            st = [d for d in self.lines if "status" in d]
        return st[-1] if st else None

    def stop(self) -> None:
        if self.p.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                self.p.send_signal(signal.SIGCONT)
                self.p.kill()  # tests only: the processes refuse SIGTERM while they publish (never damp)
            self.p.wait(timeout=10)


def _unlink(name: str) -> None:
    with contextlib.suppress(Exception):
        from _posixshmem import shm_unlink  # type: ignore[import-not-found]

        shm_unlink(name)


@pytest.fixture(scope="module")
def config_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from r1mc3 import rtconfig

    cfg = rtconfig.core_dict()
    return rtconfig.write(tmp_path_factory.mktemp("rtcfg") / "r1rt.cfg", cfg)


@pytest.fixture(scope="module")
def stand_q(config_file: Path) -> list[float]:
    for line in config_file.read_text().splitlines():
        if line.startswith("stand_q ="):
            return [float(x) for x in line.split("=", 1)[1].split()]
    raise AssertionError("no stand_q in the config")


# --------------------------------------------------------------------------------------------------------------------
# the fake robot


@dataclass
class CmdRecord:
    src_ns: int  # the writer's source timestamp (CLOCK_REALTIME of this machine)
    rx_ns: int  # time.time_ns() when this process took it
    handle: int
    min_leg_kp: float
    crc_ok: bool
    leg_q0: float
    max_leg_kp: float = 0.0  # session 6: the operator's damp through the guard (every leg kp 0)


@dataclass
class FakeRobot:
    """The robot's side of the wire (see the module docstring). `bad_crc_every`: corrupt every n-th lowstate."""

    domain: int
    stand_q: list[float]
    rate_hz: float = 500.0
    bad_crc_every: int = 0
    builtin_on: bool = True
    release_on_first_command: bool = True  # Unitree's controller lets go when ours mirrors (the release)
    records: list[CmdRecord] = field(default_factory=list)
    states_sent: int = 0
    bad_sent: int = 0

    def __post_init__(self) -> None:
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.qos import Policy, Qos
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        self._dp = DomainParticipant(self.domain)
        self._w_state = DataWriter(self._dp, Topic(self._dp, hg.TOPIC_LOWSTATE, hg.LowState_))
        t_cmd = Topic(self._dp, hg.TOPIC_LOWCMD, hg.LowCmd_)
        self._r_cmd = DataReader(self._dp, t_cmd, qos=Qos(Policy.IgnoreLocal.Process))
        self._dp_b = DomainParticipant(self.domain)  # Unitree's built-in controller: its own participant
        self._w_builtin = DataWriter(self._dp_b, Topic(self._dp_b, hg.TOPIC_LOWCMD, hg.LowCmd_))
        st = hg.lowstate_zero()
        st.mode_machine = 1
        slots = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30]
        for j, s in enumerate(slots):
            st.motor_state[s].q = float(self.stand_q[j])
            st.motor_state[s].temperature = [30, 30]
        st.imu_state.accelerometer = [0.0, 0.0, 9.81]
        self._state = st
        cmd = hg.lowcmd_zero()
        cmd.mode_machine = 1
        for j, s in enumerate(slots):
            m = cmd.motor_cmd[s]
            m.mode = 1
            m.q = float(self.stand_q[j])
            m.kp = 200.0 if j < 12 else 60.0  # the stand lock's stiff legs (v2 notes: kp 200)
            m.kd = 5.0
        cmd.crc = hg.lowcmd_crc(cmd)
        self._builtin_cmd = cmd
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self) -> FakeRobot:
        self._t.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._t.join(timeout=5)

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        nxt = time.monotonic()
        tick = 0
        while not self._stop.is_set():
            nxt += period
            tick += 1
            st = self._state
            st.tick = tick
            st.crc = hg.lowstate_crc(st)
            if self.bad_crc_every and tick % self.bad_crc_every == 0:
                st.crc ^= 0x1  # corrupted in transit
                self.bad_sent += 1
            self._w_state.write(st)
            self.states_sent += 1
            if self.builtin_on:
                self._w_builtin.write(self._builtin_cmd)
            for s in self._r_cmd.take(N=64):
                if not isinstance(s, hg.LowCmd_):
                    continue
                info = s.sample_info
                rec = CmdRecord(
                    src_ns=int(info.source_timestamp),
                    rx_ns=time.time_ns(),
                    handle=int(info.publication_handle),
                    min_leg_kp=min(float(s.motor_cmd[i].kp) for i in LEGS),
                    crc_ok=hg.lowcmd_crc(s) == s.crc,
                    leg_q0=float(s.motor_cmd[0].q),
                    max_leg_kp=max(float(s.motor_cmd[i].kp) for i in LEGS),
                )
                with self._lock:
                    self.records.append(rec)
                if self.release_on_first_command:
                    self.builtin_on = False
            dt = nxt - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                nxt = time.monotonic()

    def snapshot(self) -> list[CmdRecord]:
        with self._lock:
            return list(self.records)


@contextlib.contextmanager
def rig(
    config: Path, stand_q: list[float], domain: int, *, guard: bool = True, r1rt_args: tuple[str, ...] = (),
    robot_kw: dict[str, Any] | None = None,
) -> Iterator[tuple[FakeRobot, Proc, Proc | None, str]]:  # fmt: skip
    shm = f"/r1mc3.proc.{os.getpid()}.{domain}"
    _unlink(shm)
    robot = FakeRobot(domain, stand_q, **(robot_kw or {})).start()
    rt = Proc([str(APPS / "r1rt"), "--config", str(config), "--shm-name", shm, "--domain", str(domain), *r1rt_args])
    gd = None
    try:
        rt.wait_for(lambda d: d.get("ready") is True, 15)
        if guard:
            gd = Proc([str(APPS / "r1guard"), "--config", str(config), "--shm-name", shm, "--domain", str(domain)])
            gd.wait_for(lambda d: d.get("ready") is True, 15)
        yield robot, rt, gd, shm
    finally:
        for p in (gd, rt):
            if p is not None:
                p.stop()
        robot.close()
        _unlink(shm)


def _handles(recs: list[CmdRecord]) -> list[int]:
    seen: list[int] = []
    for r in recs:
        if r.handle not in seen:
            seen.append(r.handle)
    return seen


def _wait_mode(rt: Proc, mode: str, timeout: float) -> None:
    rt.wait_for(lambda d: d.get("status") == "r1rt" and d.get("mode") == mode, timeout)


# --------------------------------------------------------------------------------------------------------------------
# tests


def test_config_written_by_rtconfig_parses_in_cpp(config_file: Path) -> None:
    out = subprocess.run(
        [str(APPS / "r1rt"), "--config", str(config_file), "--check-config"], capture_output=True, text=True, check=True
    )
    d = json.loads(out.stdout.strip().splitlines()[-1])
    assert d["config_ok"] is True and d["writer_hz"] == 500 and d["takeover_require_guard"] == 1
    bad = config_file.with_name("bad.cfg")
    bad.write_text(config_file.read_text() + "unknown_key = 1\n")
    r = subprocess.run([str(APPS / "r1rt"), "--config", str(bad), "--check-config"], capture_output=True, text=True)
    assert r.returncode != 0 and "unknown key 'unknown_key'" in r.stdout


def test_monitor_and_standby_publish_nothing_and_the_wire_agrees(config_file: Path, stand_q: list[float]) -> None:
    with rig(config_file, stand_q, 50, robot_kw={"bad_crc_every": 100}) as (robot, rt, gd, _):
        assert gd is not None
        ready_rt = rt.wait_for(lambda d: d.get("ready") is True, 1)
        ready_gd = gd.wait_for(lambda d: d.get("ready") is True, 1)
        assert ready_rt["rate_hz"] == 500 and ready_rt["policy_loaded"] is True
        if ready_rt["os"] == "darwin":
            assert ready_rt["rt_granted"] is False  # stated, not hidden
        assert ready_gd["takeover_enabled"] is True and ready_gd["rt_pid"] == ready_rt["pid"]
        time.sleep(2.5)
        st = rt.wait_for(lambda d: d.get("status") == "r1rt" and d.get("lowstate_rx", 0) > 300, 5)
        assert st["mode"] == "MONITOR" and st["publishes"] == 0 and st["dds_writes"] == 0
        # the cross-language check, Python -> C++: our C++ reader verifies Unitree's CRC over Python's LowState_
        # bytes (every good one passes, the corrupted ones are flagged)
        assert robot.bad_sent > 0
        assert st["lowstate_crc_bad"] > 0
        assert st["lowstate_crc_bad"] <= robot.bad_sent
        assert st["lowstate_crc_bad"] < st["lowstate_rx"] / 20
        assert st["builtin_rx"] > 100 and st["foreign_writers"] >= 1  # the built-in is seen as foreign
        gs = gd.wait_for(lambda d: d.get("status") == "r1guard" and d.get("lowstate_rx", 0) > 300, 5)
        assert gs["active"] is False and gs["writes"] == 0
        assert robot.snapshot() == []  # nothing of ours on rt/lowcmd


def mailbox(shm: str, req: str, *, requester: str = "r1d", timeout: float = 5.0) -> tuple[int, int, int]:
    """One request through the segment's mode mailbox (core, layout 2), as r1d / the CLI send it: (status, refusal,
    mode after). Uses the lockstep binding's SegmentView on the process's POSIX segment (the same headers)."""
    from r1sim.cpp_policy import import_r1rt_py

    mod = import_r1rt_py()
    v = mod.SegmentView.attach(shm)
    try:
        t_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)  # the core's clock (CLOCK_MONOTONIC)
        seq = v.request_mode(mod.requests()[req], mod.requesters()[requester], t_ns, timeout_s=1.0)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rep = v.mode_reply(seq)
            if rep is not None:
                return int(rep[0]), int(rep[1]), int(rep[2])
            time.sleep(0.005)
        raise AssertionError(f"no reply to {req} (seq {seq}) in {timeout} s")
    finally:
        v.close()


def _takeover_into_balance(rt: Proc, robot: FakeRobot, shm: str) -> None:
    """r1d's takeover (DESIGN §3.5): TAKEOVER (gates; MIRROR of the built-in) -> ReleaseMode (here: the fake built-in
    lets go when it sees our first command) -> TAKEOVER_RELEASED (BLEND 1 s) -> BALANCE."""
    status, refusal, mode = mailbox(shm, "takeover")
    assert (status, refusal, mode) == (0, 0, 1), (status, refusal, mode)  # R1RT_OK, -, TAKEOVER_MIRROR
    end = time.monotonic() + 3.0
    while robot.builtin_on and time.monotonic() < end:
        time.sleep(0.01)
    assert not robot.builtin_on, "the built-in never saw our MIRROR"
    time.sleep(0.1)
    status, _, mode = mailbox(shm, "takeover_released")
    assert (status, mode) == (0, 2), (status, mode)  # TAKEOVER_BLEND
    _wait_mode(rt, "BALANCE", 15)


def test_takeover_and_the_stall_sigstop_guard_takes_over(config_file: Path, stand_q: list[float]) -> None:
    with rig(config_file, stand_q, 51) as (robot, rt, gd, shm):
        assert gd is not None
        _takeover_into_balance(rt, robot, shm)
        time.sleep(1.0)
        recs = robot.snapshot()
        assert recs and all(r.crc_ok for r in recs)  # the cross-language check, C++ -> Python: r1rt's CRC verifies
        assert len(_handles(recs)) == 1
        # the 9 Oct stall: the whole r1rt process stopped for 4.018 s
        t_stop = time.time_ns()
        os.kill(rt.p.pid, signal.SIGSTOP)
        time.sleep(STALL_S)
        os.kill(rt.p.pid, signal.SIGCONT)
        t_cont = time.time_ns()
        time.sleep(1.5)
        ev = gd.wait_for(lambda d: d.get("event") == "takeover", 5)
        recs = robot.snapshot()
    hs = _handles(recs)
    assert len(hs) == 2, f"expected r1rt then r1guard on rt/lowcmd, writers {hs}"
    core_h, guard_h = hs
    core = [r for r in recs if r.handle == core_h]
    guard = [r for r in recs if r.handle == guard_h]
    last_core_before = max(r.src_ns for r in core if r.src_ns <= t_stop + 5_000_000)
    first_guard = min(r.src_ns for r in guard)
    gap_ms = (first_guard - last_core_before) / 1e6
    late_core = [r for r in core if r.src_ns > first_guard]
    print(
        f"\n[ info ] SIGSTOP stall: command gap {gap_ms:.2f} ms (r1rt last -> r1guard first, source timestamps); "
        f"guard: silence {ev['rt_silence_ns'] / 1e6:.2f} ms at the takeover, mode_from {ev['mode_from']}, proc_err "
        f"{ev['proc_err']}; r1rt packets after the guard's first: {len(late_core)}; guard commands {len(guard)}"
    )
    assert ev["mode_from"] == "BALANCE"
    assert gap_ms <= 50.0, f"guard takeover gap {gap_ms:.1f} ms (target <= 15 ms on the robot; this Mac is loaded)"
    assert len(late_core) <= 1, "after the takeover r1rt must stay silent (one packet = the residual race)"
    assert all(r.src_ns < t_cont + 2_000_000_000 for r in late_core)
    assert min(r.min_leg_kp for r in recs) >= 20.0, "legs never limp (the lowest is the ankles' 31.4)"
    assert all(r.crc_ok for r in recs)
    assert guard[-1].src_ns > t_cont + 1_000_000_000  # the guard kept publishing after r1rt came back


def test_writer_sleep_stall_guard_takes_over(config_file: Path, stand_q: list[float]) -> None:
    args = ("--fault-writer-stall", f"1.0:{STALL_S}")
    with rig(config_file, stand_q, 52, r1rt_args=args) as (robot, rt, gd, shm):
        assert gd is not None
        _takeover_into_balance(rt, robot, shm)
        rt.wait_for(lambda d: d.get("event") == "fault_writer_stall_end", 10)
        time.sleep(1.0)
        ev = gd.wait_for(lambda d: d.get("event") == "takeover", 5)
        st = rt.wait_for(lambda d: d.get("status") == "r1rt" and d.get("token_owner") == 2, 5)
        recs = robot.snapshot()
    hs = _handles(recs)
    assert len(hs) == 2
    core = [r for r in recs if r.handle == hs[0]]
    guard = [r for r in recs if r.handle == hs[1]]
    first_guard = min(r.src_ns for r in guard)
    last_core_before = max(r.src_ns for r in core if r.src_ns < first_guard)
    gap_ms = (first_guard - last_core_before) / 1e6
    late = [r for r in core if r.src_ns > first_guard]
    print(f"\n[ info ] writer-sleep stall: gap {gap_ms:.2f} ms; guard silence {ev['rt_silence_ns'] / 1e6:.2f} ms; "
          f"r1rt after: {len(late)}; r1rt not_owner_skips {st['not_owner_skips']}")  # fmt: skip
    assert gap_ms <= 50.0
    assert len(late) == 0  # the writer slept BEFORE its step: the core sees the token gone, the wrapper never writes
    assert min(r.min_leg_kp for r in recs) >= 20.0


def test_kill9_of_r1rt_guard_takes_over(config_file: Path, stand_q: list[float]) -> None:
    with rig(config_file, stand_q, 53) as (robot, rt, gd, shm):
        assert gd is not None
        _takeover_into_balance(rt, robot, shm)
        time.sleep(0.5)
        rt.p.kill()
        rt.p.wait(timeout=5)
        time.sleep(1.0)
        ev = gd.wait_for(lambda d: d.get("event") == "takeover", 5)
        recs = robot.snapshot()
    hs = _handles(recs)
    assert len(hs) == 2
    core = [r for r in recs if r.handle == hs[0]]
    guard = [r for r in recs if r.handle == hs[1]]
    gap_ms = (min(r.src_ns for r in guard) - max(r.src_ns for r in core)) / 1e6
    print(f"\n[ info ] kill -9: gap {gap_ms:.2f} ms; guard silence {ev['rt_silence_ns'] / 1e6:.2f} ms")
    assert ev["mode_from"] == "BALANCE"
    assert gap_ms <= 50.0
    assert min(r.min_leg_kp for r in recs) >= 20.0


def test_without_r1guard_the_takeover_is_refused(config_file: Path, stand_q: list[float]) -> None:
    with rig(config_file, stand_q, 54, guard=False, r1rt_args=("--test-takeover", "1.5:2.0")) as (robot, rt, _, shm):
        time.sleep(0.5)
        status, refusal, mode = mailbox(shm, "takeover")  # through the mailbox (r1d's path)
        ev = rt.wait_for(lambda d: d.get("event") == "test_takeover", 10)  # and through the test hook
        time.sleep(1.0)
        st = rt.last_status()
        recs = robot.snapshot()
    assert (status, refusal, mode) == (-6, 8, 0)  # R1RT_E_REFUSED, R1RT_TAKEOVER_GUARD (step_api.h), MONITOR
    assert ev["status"] == -6 and ev["refusal"] == 8
    assert st is not None and st["mode"] == "MONITOR" and st["publishes"] == 0
    assert recs == []


def test_damp_through_the_mailbox_only_from_the_operator(config_file: Path, stand_q: list[float]) -> None:
    with rig(config_file, stand_q, 58) as (robot, rt, gd, shm):
        assert gd is not None
        _takeover_into_balance(rt, robot, shm)
        st_r1d = mailbox(shm, "damp", requester="r1d")
        _wait_mode(rt, "BALANCE", 3)
        st_op = mailbox(shm, "damp", requester="operator")
        _wait_mode(rt, "DAMP", 3)
        time.sleep(0.3)
        recs = robot.snapshot()
        rt.p.send_signal(signal.SIGTERM)  # DAMP is the operator's state: a stop request is accepted there
        rt.wait_for(lambda d: d.get("event") == "exit", 5)
    assert st_r1d[0] == -6 and st_r1d[2] == 3  # refused (not the operator), still BALANCE
    assert st_op[0] == 0 and st_op[2] == 8  # DAMP
    assert recs[-1].min_leg_kp == 0.0  # the only limp state, entered by the operator only


def test_sigterm_while_publishing_is_refused_and_monitor_exits(config_file: Path, stand_q: list[float]) -> None:
    with rig(config_file, stand_q, 55) as (robot, rt, gd, shm):
        assert gd is not None
        _takeover_into_balance(rt, robot, shm)
        rt.p.send_signal(signal.SIGTERM)
        ev = rt.wait_for(lambda d: d.get("event") == "signal_refused", 5)
        time.sleep(0.5)
        assert rt.p.poll() is None and ev["mode"] == "BALANCE"
        n = len(robot.snapshot())
        time.sleep(0.3)
        assert len(robot.snapshot()) > n + 50  # still publishing at ~500 Hz
        gd.p.send_signal(signal.SIGTERM)  # the guard on standby: a clean exit
        gd.wait_for(lambda d: d.get("event") == "exit", 5)
        assert gd.p.wait(timeout=5) == 0
    with rig(config_file, stand_q, 56, guard=False) as (_robot, rt2, _g, _s):
        rt2.p.send_signal(signal.SIGTERM)  # MONITOR: exits cleanly
        rt2.wait_for(lambda d: d.get("event") == "exit", 5)
        assert rt2.p.wait(timeout=5) == 0


def test_on_stand_then_the_core_stalls_and_the_guard_holds(config_file: Path, stand_q: list[float]) -> None:
    """HW1's T6 rehearsed with the processes (DESIGN §14.1 step 7): ON_STAND through the mailbox (MIRROR -> the
    built-in lets go -> RELEASED -> the blend into a stiff hold), then SIGSTOP r1rt: r1guard takes over HOLDING the
    core's last command (not balancing), r1rt silent after SIGCONT."""
    with rig(config_file, stand_q, 56) as (robot, rt, gd, shm):
        assert gd is not None
        status, refusal, mode = mailbox(shm, "on_stand")
        assert (status, refusal, mode) == (0, 0, 1), (status, refusal, mode)  # MIRROR
        end = time.monotonic() + 3.0
        while robot.builtin_on and time.monotonic() < end:
            time.sleep(0.01)
        time.sleep(0.1)
        status, _, mode = mailbox(shm, "takeover_released")
        assert (status, mode) == (0, 2), (status, mode)  # the blend into the hold
        _wait_mode(rt, "ON_STAND", 10)
        time.sleep(0.5)
        t_stop = time.time_ns()
        os.kill(rt.p.pid, signal.SIGSTOP)
        time.sleep(2.0)
        t_cont = time.time_ns()
        os.kill(rt.p.pid, signal.SIGCONT)
        time.sleep(1.0)
        ev = gd.wait_for(lambda d: d.get("event") == "takeover", 5)
        gs = gd.wait_for(lambda d: d.get("status") == "r1guard" and d.get("active") is True, 5)
        recs = robot.snapshot()
    hs = _handles(recs)
    assert len(hs) == 2
    core = [r for r in recs if r.handle == hs[0]]
    guard = [r for r in recs if r.handle == hs[1]]
    last_core = max((r for r in core if r.src_ns <= t_stop + 5_000_000), key=lambda r: r.src_ns)
    first_guard = min(guard, key=lambda r: r.src_ns)
    gap_ms = (first_guard.src_ns - last_core.src_ns) / 1e6
    print(f"\n[ info ] ON_STAND stall: gap {gap_ms:.2f} ms; guard {gs['act_mode']}; legs kp {first_guard.min_leg_kp}")
    assert ev["mode_from"] == "ON_STAND" and gs["act_mode"] == "ON_STAND"
    assert gap_ms <= 50.0
    assert first_guard.leg_q0 == pytest.approx(last_core.leg_q0, abs=1e-6)  # the core's last command held
    assert first_guard.min_leg_kp == pytest.approx(last_core.min_leg_kp)
    assert min(r.min_leg_kp for r in recs) >= 20.0  # stiff hold throughout, never limp
    # r1rt after the guard's first packet: DESIGN §4.3 bounds the residual race to ONE stale packet (r1rt checks the
    # token immediately before each write; a SIGSTOP landing between that check and the write lets that one packet out
    # at SIGCONT, stamped then). Session 12: the test asserted none - stricter than the design - and failed once in the
    # session-11 CI under load (one such packet; 8 / 8 quiet re-runs passed). What it pins now is the design's bound and
    # its meaning: at most one, carrying the core's last command (nothing new reaches the board), never a second.
    _at_most_the_residual_stale_packet([r for r in core if r.src_ns > first_guard.src_ns], last_core, t_cont)


def _at_most_the_residual_stale_packet(late: list[CmdRecord], last_core: CmdRecord, t_cont_ns: int) -> None:
    """DESIGN §4.3's bound (session 12): r1rt after the guard owns the board - at most ONE packet, the write that was in
    progress when SIGSTOP landed (stamped as it completes, at SIGCONT: within 0.2 s of it), carrying r1rt's last
    command (nothing new reaches the board). Tests at lines ~418 / ~734 of this file pin the same bound."""
    assert len(late) <= 1, [(r.src_ns - t_cont_ns, r.leg_q0, r.min_leg_kp) for r in late]
    for r in late:
        assert -1_000_000 <= r.src_ns - t_cont_ns <= 200_000_000, (r.src_ns - t_cont_ns, r)  # written at SIGCONT
        assert r.leg_q0 == pytest.approx(last_core.leg_q0, abs=1e-6), (r, last_core)  # the core's last command
        assert r.min_leg_kp == pytest.approx(last_core.min_leg_kp) and r.max_leg_kp == pytest.approx(
            last_core.max_leg_kp
        ), (r, last_core)


def test_rt_grants_and_loop_timing_reach_the_core(tmp_path: Path, stand_q: list[float]) -> None:
    """Session 5 (integrator, proc's requests to core): the processes hand their real-time grants and the writer's
    loop timing to the core. (1) A config written with `--require-rt` refuses TAKEOVER on a host without real-time
    grants (macOS never grants them: refusal R1RT_TAKEOVER_RT = 11), and both heartbeats carry rt_granted = 0;
    (2) the telemetry records carry the writer's wake-up lateness and the previous iteration's duration (before this
    wiring both were always 0: core.cpp wrote 0, the process kept them in its own status line only)."""
    import numpy as np
    from r1mc3 import rtconfig
    from r1mc3.log import records
    from r1sim.cpp_policy import import_r1rt_py

    cfg = rtconfig.write(tmp_path / "r1rt_rt.cfg", rtconfig.core_dict(require_rt=True))
    assert "takeover_require_rt = 1" in cfg.read_text()
    mod = import_r1rt_py()
    with rig(cfg, stand_q, 59) as (robot, rt, gd, shm):
        assert gd is not None
        if rt.wait_for(lambda d: d.get("ready") is True, 1).get("rt_granted"):
            pytest.skip("this host grants real-time scheduling: the refusal case does not apply")
        time.sleep(0.6)
        status, refusal, mode = mailbox(shm, "takeover")
        v = mod.SegmentView.attach(shm)
        try:
            raw = v.pop_telemetry(4096)
            hb_rt = v.rt_heartbeat()
            hb_g = v.guard_heartbeat()
        finally:
            v.close()
        recs = robot.snapshot()
    assert (status, refusal, mode) == (-6, 11, 0), (status, refusal, mode)  # R1RT_E_REFUSED, R1RT_TAKEOVER_RT, MONITOR
    assert recs == []  # nothing published
    assert hb_rt is not None and hb_g is not None
    assert int(np.frombuffer(hb_rt[1], dtype=records.HEARTBEAT)[0]["rt_granted"]) == 0
    assert int(np.frombuffer(hb_g[1], dtype=records.HEARTBEAT)[0]["rt_granted"]) == 0
    data = b"".join(raw) if isinstance(raw, list) else bytes(raw)
    tel = np.frombuffer(data, dtype=records.TELEMETRY)
    assert len(tel) > 50
    assert np.count_nonzero(tel["wake_lateness_ns"]) > 0.5 * len(tel)
    assert np.count_nonzero(tel["iteration_ns"][1:]) == len(tel) - 1
    assert 0 < np.median(tel["iteration_ns"][1:]) < 2_000_000  # inside one 2 ms writer period


# --------------------------------------------------------------------------------------------------------------------
# session 6: the operator's damp through the guard; the operator's hand-back (RECLAIM) and the guard re-arming (T6)


def _into_on_stand(rt: Proc, robot: FakeRobot, shm: str) -> None:
    """r1d's on_stand (DESIGN §3.5): ON_STAND (MIRROR of the built-in) -> the built-in lets go -> RELEASED ->
    ON_STAND."""
    status, refusal, mode = mailbox(shm, "on_stand")
    assert (status, refusal, mode) == (0, 0, 1), (status, refusal, mode)  # MIRROR
    end = time.monotonic() + 3.0
    while robot.builtin_on and time.monotonic() < end:
        time.sleep(0.01)
    time.sleep(0.1)
    status, _, mode = mailbox(shm, "takeover_released")
    assert (status, mode) == (0, 2), (status, mode)
    _wait_mode(rt, "ON_STAND", 10)


def _stop_request(shm: str, requester: str) -> None:
    """r1d's `damp` (service.py _stop): the stop channel's damp level (2) with the requester's id."""
    from r1sim.cpp_policy import import_r1rt_py

    mod = import_r1rt_py()
    v = mod.SegmentView.attach(shm)
    try:
        v.request_stop(2, mod.requesters()[requester], time.clock_gettime_ns(time.CLOCK_MONOTONIC))
    finally:
        v.close()


def _guard_info(shm: str) -> tuple[int, dict[str, Any] | None]:
    """(the segment's creator pid, r1guard's decoded heartbeat) - what r1d's status and `r1 t3` / `r1 t6` read."""
    from r1sim.cpp_policy import import_r1rt_py

    v = import_r1rt_py().SegmentView.attach(shm)
    try:
        return int(v.creator_pid()), v.guard_info()
    finally:
        v.close()


def test_the_operators_damp_reaches_the_motors_through_the_guard(config_file: Path, stand_q: list[float]) -> None:
    """r1rt frozen in ON_STAND, r1guard holding: a damp request from r1d's id changes nothing; the operator's
    (`r1 damp`) makes r1guard publish damp (legs kp 0); r1guard, active in the operator's damp, then exits on SIGTERM
    (nothing of ours holds anything), and r1rt after SIGCONT is in DAMP (not the owner: silent) and exits too.
    Session 5's guard ignored the stop channel (the legs stayed stiff: fails on 26e8bce)."""
    with rig(config_file, stand_q, 57) as (robot, rt, gd, shm):
        assert gd is not None
        _into_on_stand(rt, robot, shm)
        time.sleep(0.3)
        t_stop = time.time_ns()
        os.kill(rt.p.pid, signal.SIGSTOP)
        try:
            gd.wait_for(lambda d: d.get("event") == "takeover", 5)
            time.sleep(0.3)
            t_r1d = time.time_ns()
            _stop_request(shm, "r1d")
            gd.wait_for(lambda d: d.get("event") == "damp_ignored", 5)
            time.sleep(0.3)
            t_op = time.time_ns()
            _stop_request(shm, "operator")
            gd.wait_for(lambda d: d.get("event") == "operator_damp", 5)
            time.sleep(0.3)
            recs = robot.snapshot()
        finally:
            t_cont = time.time_ns()
            os.kill(rt.p.pid, signal.SIGCONT)
        hs = _handles(recs)
        assert len(hs) == 2
        guard = [r for r in recs if r.handle == hs[1]]
        last_core = max(
            (r for r in recs if r.handle == hs[0] and r.src_ns <= t_stop + 5_000_000), key=lambda r: r.src_ns
        )
        between = [r for r in guard if t_r1d + 20_000_000 < r.rx_ns < t_op]
        after = [r for r in guard if r.rx_ns > t_op + 20_000_000]
        assert between and min(r.min_leg_kp for r in between) >= 20.0  # r1d's damp: still the stiff hold
        assert after and max(r.max_leg_kp for r in after) == 0.0  # the operator's damp: every leg kp 0
        time.sleep(0.5)
        _wait_mode(rt, "DAMP", 5)  # r1rt saw the operator's damp too (not the owner: it publishes nothing)
        # r1rt after SIGCONT publishes nothing but, at most, the residual race's one stale packet (DESIGN §4.3; session
        # 12: it showed up once under load - the ON_STAND hold r1rt was writing when SIGSTOP landed)
        late = [r for r in robot.snapshot() if r.handle == hs[0] and r.rx_ns > t_op + 50_000_000]
        _at_most_the_residual_stale_packet(late, last_core, t_cont)
        gd.p.send_signal(signal.SIGTERM)
        assert gd.p.wait(timeout=5) == 0  # active, but in the operator's damp: it exits (session 6)
        rt.p.send_signal(signal.SIGTERM)
        assert rt.p.wait(timeout=5) == 0  # DAMP: exits cleanly


def test_t6_the_operators_reclaim_is_bumpless_and_the_guard_rearms(config_file: Path, stand_q: list[float]) -> None:
    """HW1's T6 end to end with the processes (DESIGN §14.1 step 7): ON_STAND -> SIGSTOP r1rt -> r1guard holds ->
    SIGCONT: r1rt silent -> RECLAIM from r1d's id refused -> the operator's RECLAIM: r1rt publishes again from exactly
    the held command (bumpless), r1guard stops (at most one packet after r1rt's first) and re-arms -> a second SIGSTOP
    is taken over again. Session 5 had no hand-back (the request unknown: fails on 26e8bce)."""
    with rig(config_file, stand_q, 58) as (robot, rt, gd, shm):
        assert gd is not None
        pid, gi = _guard_info(shm)
        assert pid == rt.p.pid  # SegmentView.creator_pid(): r1rt (the SIGSTOP target of `r1 t6`)
        assert gi is not None and gi["armed"] and not gi["active"]  # the heartbeat's flags (board-hold step 4: "armed")
        _into_on_stand(rt, robot, shm)
        time.sleep(0.3)
        os.kill(rt.p.pid, signal.SIGSTOP)
        time.sleep(1.0)
        _, gi = _guard_info(shm)
        os.kill(rt.p.pid, signal.SIGCONT)
        assert gi is not None and gi["active"] and gi["armed"] and gi["mode_from"] == 6  # took over from ON_STAND
        gd.wait_for(lambda d: d.get("event") == "takeover", 5)
        time.sleep(0.5)
        status, refusal, _ = mailbox(shm, "reclaim", requester="r1d")
        assert (status, refusal) == (-6, 20), (status, refusal)  # R1RT_E_REFUSED, R1RT_REFUSE_NOT_OPERATOR
        recs0 = robot.snapshot()
        hs = _handles(recs0)
        assert len(hs) == 2
        t_reclaim = time.time_ns()
        status, refusal, mode = mailbox(shm, "reclaim", requester="operator")
        assert (status, refusal, mode) == (0, 0, 2), (status, refusal, mode)  # OK, TAKEOVER_BLEND into ON_STAND
        gd.wait_for(lambda d: d.get("event") == "rearmed", 5)
        _wait_mode(rt, "ON_STAND", 5)
        time.sleep(0.3)
        recs1 = robot.snapshot()
        core_new = [r for r in recs1 if r.handle == hs[0] and r.rx_ns > t_reclaim]
        guard_all = [r for r in recs1 if r.handle == hs[1]]
        assert core_new, "r1rt did not publish after the reclaim"
        first_core = min(core_new, key=lambda r: r.src_ns)
        last_guard = max(guard_all, key=lambda r: r.src_ns)
        late_guard = [r for r in guard_all if r.src_ns > first_core.src_ns]
        gap_ms = (first_core.src_ns - last_guard.src_ns) / 1e6
        n_late = len(late_guard)
        print(f"\n[ info ] reclaim: guard -> r1rt gap {gap_ms:.2f} ms; guard packets after r1rt's first {n_late}")
        assert len(late_guard) <= 1  # the residual race: at most one stale packet (the same held command)
        assert first_core.leg_q0 == pytest.approx(last_guard.leg_q0, abs=1e-9)  # bumpless: the held targets
        assert first_core.min_leg_kp == pytest.approx(last_guard.min_leg_kp, abs=1e-6)  # the same gains
        assert min(r.min_leg_kp for r in recs1) >= 20.0  # stiff throughout, never limp
        # the second T6: the re-armed guard catches the next stall
        t_stop2 = time.time_ns()
        os.kill(rt.p.pid, signal.SIGSTOP)
        time.sleep(1.0)
        os.kill(rt.p.pid, signal.SIGCONT)
        end = time.monotonic() + 5.0
        while len(gd.events("takeover")) < 2 and time.monotonic() < end:
            time.sleep(0.05)
        assert len(gd.events("takeover")) == 2
        recs2 = robot.snapshot()
        guard2 = [r for r in recs2 if r.handle == hs[1] and r.rx_ns > t_stop2]
        assert guard2 and min(r.min_leg_kp for r in guard2) >= 20.0
        assert not [r for r in recs2 if r.handle == hs[0] and r.src_ns > min(g.src_ns for g in guard2) + 5_000_000]


def test_fault_policy_nan_freezes_into_the_stand_with_the_processes(tmp_path: Path, stand_q: list[float]) -> None:
    """r1rt's --fault-policy-nan hook (session 6, for the fault table): the policy's output NaN 2.5 s after BALANCE
    entry -> rung 3: STAND_FREEZE (the ramp to the stand at stand gains), legs stiff throughout, never damp; the window
    is reported as an event line."""
    from r1mc3 import rtconfig

    cfg = rtconfig.write(tmp_path / "r1rt.cfg", rtconfig.core_dict())
    # 2.5 s after BALANCE entry: the 1 Hz status line shows BALANCE first (the takeover helper waits for it)
    with rig(cfg, stand_q, 59, r1rt_args=("--fault-policy-nan", "2.5:0.3")) as (robot, rt, gd, shm):
        assert gd is not None
        ready = rt.wait_for(lambda d: d.get("ready") is True, 5)
        assert ready["fault_policy_nan"] == [2.5, 0.3]
        _takeover_into_balance(rt, robot, shm)
        ev = rt.wait_for(lambda d: d.get("event") == "fault_policy_nan", 5)
        assert ev["t1_ns"] - ev["t0_ns"] == 300_000_000
        _wait_mode(rt, "STAND_FREEZE", 5)
        time.sleep(0.5)
        recs = robot.snapshot()
        st = rt.last_status()
    assert st is not None and st["mode"] == "STAND_FREEZE"
    assert min(r.min_leg_kp for r in recs) >= 20.0  # never limp
    assert len(_handles(recs)) == 1  # r1rt alone (the guard stayed on standby)


def test_r1guard_waits_while_r1rt_is_still_sizing_the_segment(config_file: Path, stand_q: list[float]) -> None:
    """Session 13 (HW2 day 2, 08:35 CST: r1guard started 8 ms after r1rt and exited "attach /r1mc3.rt: segment smaller
    than expected" - once in four starts; fixed on the day by `sudo systemctl start r1guard`): r1rt's create() is
    shm_open THEN ftruncate THEN init, and the guard treated the 0-byte object in between as fatal. Here the name holds
    a 0-byte object for ~0.4 s (r1rt between its shm_open and its ftruncate), then r1rt makes the real segment:
    r1guard waits and becomes ready (base dcc01a2: it exits at once, "segment smaller than expected")."""
    from _posixshmem import shm_open

    domain = 97
    shm = f"/r1mc3.proc.{os.getpid()}.{domain}"
    _unlink(shm)
    os.close(shm_open(shm, os.O_CREAT | os.O_EXCL | os.O_RDWR, mode=0o600))  # 0 bytes, like r1rt after shm_open
    robot = FakeRobot(domain, stand_q).start()
    gd = Proc([str(APPS / "r1guard"), "--config", str(config_file), "--shm-name", shm, "--domain", str(domain)])
    rt: Proc | None = None
    try:
        time.sleep(0.4)
        assert gd.p.poll() is None, f"r1guard exited on the 0-byte segment: {gd.p.stderr.read() if gd.p.stderr else ''}"
        _unlink(shm)  # r1rt "finishes" the creation: the real segment under the same name
        rt = Proc([str(APPS / "r1rt"), "--config", str(config_file), "--shm-name", shm, "--domain", str(domain)])
        rt.wait_for(lambda d: d.get("ready") is True, 15)
        ready = gd.wait_for(lambda d: d.get("ready") is True, 15)
    finally:
        for p in (gd, rt):
            if p is not None:
                p.stop()
        robot.close()
        _unlink(shm)
    assert ready.get("ready") is True
