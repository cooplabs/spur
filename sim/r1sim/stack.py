"""The v3 process stack against the DDS-bridge simulator, in one object (session 6 foundation).

Session 5 proved each process alone and in pairs (PROCESSES.md, DDS_BRIDGE.md, `tests/integration/
test_service_processes.py`); session 6 rehearses the stand session (HW1, DESIGN §14.1) and runs the fault tests with
ALL of them at once - the bridge (the robot: rt/lowstate, rt/lowcmd, the hands, the motion-switcher RPC, the emulated
built-in), `r1rt`, `r1guard`, `r1d` (`python -m r1mc3.service`), `r1log` (`python -m r1mc3.log run`) and the `r1` CLI
(`python -m r1mc3.cli`, one process per word, as the operator types it). This module starts and stops them the same
way for every harness (the rehearsal, the fault table, the starvation test), so their numbers are comparable:

    with ProcStack(domain=31, workdir=tmp, bridge={"stance": "9oct", "safety": True}) as st:
        st.cli("status")                      # -> r1d's typed result (dict), `_exit` = the CLI's exit code
        st.cli("takeover"); st.wait_mode("balance", 10)
        st.stop_guard(); st.start_guard()     # the board-hold test: the guard off, back on, checked
        st.bridge.writers(); st.bridge.tilt_deg()

The processes run on this Mac WITHOUT real-time scheduling (macOS, PROCESSES.md): every timing is "macOS, under load
/ quiet". Teardown never damps anything - it kills the processes (SIGKILL, tests only: r1rt refuses SIGTERM while it
publishes, DESIGN §2.1) and unlinks the segment.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def loopback_uri() -> str:
    """CycloneDDS on the loopback (lo0 / lo), unicast to 127.0.0.1, no multicast (`r1mc3.dds.rpc.LOOPBACK_URI`)."""
    from r1mc3.dds.rpc import LOOPBACK_URI

    return LOOPBACK_URI


def apps_dir() -> Path | None:
    """proc's binaries: R1RT_APPS_DIR, else $R1MC3_BUILD_DIR/rt_dds/apps, else <repo>/build/rt_dds/apps."""
    cands: list[Path] = []
    if os.environ.get("R1RT_APPS_DIR"):
        cands.append(Path(os.environ["R1RT_APPS_DIR"]))
    if os.environ.get("R1MC3_BUILD_DIR"):
        cands.append(Path(os.environ["R1MC3_BUILD_DIR"]) / "rt_dds" / "apps")
    cands.append(ROOT / "build" / "rt_dds" / "apps")
    for d in cands:
        if (d / "r1rt").exists() and (d / "r1guard").exists():
            return d
    return None


def available() -> bool:
    """True when the stack can run here: CycloneDDS's Python binding and the DDS build of r1rt / r1guard."""
    from r1mc3 import dds

    return dds.available() and apps_dir() is not None


def unlink_shm(name: str) -> None:
    """Remove a POSIX shared-memory segment (no error when it does not exist)."""
    with contextlib.suppress(Exception):
        from _posixshmem import shm_unlink  # type: ignore[import-not-found]

        shm_unlink(name)


class Proc:
    """A child process whose stdout JSON lines are collected by a reader thread (stderr kept for the report)."""

    def __init__(self, name: str, argv: Sequence[str], env: Mapping[str, str], *, log_dir: Path | None = None) -> None:
        self.name = name
        self.argv = list(argv)
        self.t_start = time.monotonic()
        err = open(log_dir / f"{name}.stderr", "w") if log_dir is not None else subprocess.PIPE
        self.p = subprocess.Popen(self.argv, stdout=subprocess.PIPE, stderr=err, text=True, env=dict(env))
        self.lines: list[dict[str, Any]] = []
        self.raw: list[str] = []
        self._cv = threading.Condition()
        self._t = threading.Thread(target=self._read, name=f"{name}-stdout", daemon=True)
        self._t.start()

    def _read(self) -> None:
        assert self.p.stdout is not None
        for line in self.p.stdout:
            with self._cv:
                self.raw.append(line)
                with contextlib.suppress(json.JSONDecodeError):
                    d = json.loads(line)
                    if isinstance(d, dict):
                        d.setdefault("_t", time.monotonic())
                        self.lines.append(d)
                self._cv.notify_all()

    @property
    def pid(self) -> int:
        return int(self.p.pid)

    def alive(self) -> bool:
        return self.p.poll() is None

    def wait_for(self, pred: Callable[[dict[str, Any]], bool], timeout: float) -> dict[str, Any]:
        """The first JSON line matching `pred` (already seen or arriving within `timeout` s); AssertionError if not."""
        end = time.monotonic() + timeout
        with self._cv:
            while True:
                for d in self.lines:
                    if pred(d):
                        return d
                left = end - time.monotonic()
                if left <= 0 or self.p.poll() is not None:
                    raise AssertionError(f"{self.name}: no matching line in {timeout} s; last {self.raw[-5:]}")
                self._cv.wait(min(left, 0.5))

    def json_lines(self) -> list[dict[str, Any]]:
        with self._cv:
            return list(self.lines)

    def events(self, name: str) -> list[dict[str, Any]]:
        with self._cv:
            return [d for d in self.lines if d.get("event") == name]

    def last_status(self) -> dict[str, Any] | None:
        with self._cv:
            st = [d for d in self.lines if "status" in d]
        return st[-1] if st else None

    def signal(self, sig: int) -> None:
        if self.p.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.p.pid, sig)

    def terminate_and_wait(self, timeout: float = 10.0) -> int | None:
        """SIGTERM, then wait: the exit code, or None if it is still running (r1rt refuses SIGTERM while publishing)."""
        self.signal(signal.SIGTERM)
        try:
            return self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> None:
        """Tests / teardown only: SIGCONT (a SIGSTOPped process), then SIGKILL - and its descendants too (session 11:
        r1d's multiprocessing children - the planner / measurement workers and their resource tracker - outlived a
        SIGKILLed r1d, ~29 orphans after one agent's rehearsal runs; on the robot systemd stops the unit's cgroup)."""
        kids = descendants(self.p.pid) if self.p.poll() is None else []
        if self.p.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                self.p.send_signal(signal.SIGCONT)
                self.p.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.p.wait(timeout=10)
        for pid in kids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)


def descendants(pid: int) -> list[int]:
    """Every live descendant of `pid` (children first, then theirs) via `pgrep -P` (macOS and Linux; no psutil)."""
    out: list[int] = []
    todo = [int(pid)]
    while todo:
        parent = todo.pop()
        try:
            r = subprocess.run(["pgrep", "-P", str(parent)], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        for tok in r.stdout.split():
            if tok.isdigit() and int(tok) not in out:
                out.append(int(tok))
                todo.append(int(tok))
    return out


class ProcStack:
    """The DDS-bridge simulator + r1rt + r1guard + r1d + r1log (+ the CLI per word), started in that order.

    `bridge`: keyword arguments for `r1sim.dds_bridge.DdsBridge` (domain is added; e.g. stance / safety / seed / scene
    / builtin), or None for no bridge (the caller runs the robot side). `cfg`: the config dict for the processes
    (default `r1mc3.rtconfig.core_dict()`); `rt_args` / `guard_args` / `r1d_args` / `r1log_args`: extra command-line
    words. `guard` / `r1d` / `r1log`: start that process. `workdir`: the config file, the processes' stderr, r1log's
    files (`workdir/logs`), the records (`R1_RECORD_DIR` = `workdir/records`). `camera` (session 11): True or the
    keyword arguments of `r1sim.bridge_camera.BridgeCamera` (+ `extra`: a world-building extra, e.g. tags) - the
    bridge's world gets the calibrated head camera and r1d a real r1cam server rendering it (`R1CAM_PORT`) and the
    calibration file it renders through (`R1MC3_HEAD_CALIB`: by default the robot's own v3-frame file).
    """

    def __init__(
        self,
        *,
        domain: int,
        workdir: Path,
        bridge: Mapping[str, Any] | None = None,
        cfg: Mapping[str, Any] | None = None,
        shm_name: str | None = None,
        guard: bool = True,
        r1d: bool = True,
        r1log: bool = False,
        rt_args: Sequence[str] = (),
        guard_args: Sequence[str] = (),
        r1d_args: Sequence[str] = (),
        r1log_args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        camera: bool | Mapping[str, Any] = False,
    ) -> None:
        apps = apps_dir()
        if apps is None:
            raise RuntimeError("r1rt / r1guard not built (scripts/ci.sh dds: -DR1RT_WITH_CYCLONEDDS=ON)")
        self.apps = apps
        self.domain = int(domain)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.bridge_kw = dict(bridge) if bridge is not None else None
        self.cfg_dict = dict(cfg) if cfg is not None else None
        self.shm = shm_name or f"/r1mc3.s6.{os.getpid()}.{self.domain}"
        if len(self.shm) > 31:
            raise ValueError(f"shm name too long for macOS (31): {self.shm}")
        self.want_guard, self.want_r1d, self.want_r1log = guard, r1d, r1log
        self.rt_args, self.guard_args = list(rt_args), list(guard_args)
        self.r1d_args, self.r1log_args = list(r1d_args), list(r1log_args)
        self.env = dict(os.environ)
        self.env.update(env or {})
        self.env["CYCLONEDDS_URI"] = loopback_uri()
        self.env.setdefault("R1_RECORD_DIR", str(self.workdir / "records"))
        self.env.setdefault("R1LOG_DIR", str(self.workdir / "logs"))
        # r1d reads r1rt's config for the stand pose / the STAND_FREEZE ramp (`r1 stand-freeze`'s dry run, svc S6)
        self.env.setdefault("R1RT_CONFIG", str(self.workdir / "r1rt.cfg"))
        self.bridge: Any = None
        self.rt: Proc | None = None
        self.guard: Proc | None = None
        self.r1d: Proc | None = None
        self.r1log: Proc | None = None
        self.port: int = 0
        self.cfg_path: Path = self.workdir / "r1rt.cfg"
        self.old_guards: list[Proc] = []
        # session 11: the head camera on the bridge stack (r1sim.bridge_camera)
        self.camera_kw: dict[str, Any] | None = None if not camera else ({} if camera is True else dict(camera))
        self.camera: Any = None

    # ------------------------------------------------------------------ start / stop
    def write_config(self) -> Path:
        from r1mc3 import rtconfig

        cfg = self.cfg_dict if self.cfg_dict is not None else rtconfig.core_dict()
        return rtconfig.write(self.cfg_path, cfg)

    def start(self, *, ready_timeout_s: float = 30.0) -> ProcStack:
        """Start everything wanted; returns self. On any failure the started parts are stopped and it raises."""
        os.environ["CYCLONEDDS_URI"] = loopback_uri()  # this process's participants (the in-process bridge)
        try:
            self.write_config()
            if self.bridge_kw is not None:
                from r1sim.dds_bridge import DdsBridge

                kw = dict(self.bridge_kw)
                if self.camera_kw is not None and "w" not in kw:  # session 11: the same world + the head camera
                    from r1sim.bridge_camera import camera_world

                    kw["w"] = camera_world(
                        scene=kw.get("scene", "floor"), stance=str(kw.get("stance", "centred")),
                        stand=kw.get("stand"), extra=self.camera_kw.get("extra"),
                    )  # fmt: skip
                self.bridge = DdsBridge(domain=self.domain, **kw)
                self.bridge.start()
                if self.camera_kw is not None:
                    from r1sim.bridge_camera import BridgeCamera, attach_env

                    ckw = {k: v for k, v in self.camera_kw.items() if k != "extra"}
                    ckw.setdefault("workdir", self.workdir)
                    self.camera = BridgeCamera(self.bridge, Path(self.env["R1_RECORD_DIR"]), **ckw)
                    attach_env(self.camera, self.env, self.workdir)
            self.rt = Proc(
                "r1rt",
                [str(self.apps / "r1rt"), "--config", str(self.cfg_path), "--shm-name", self.shm,
                 "--domain", str(self.domain), *self.rt_args],
                self.env,
                log_dir=self.workdir,
            )  # fmt: skip
            self.rt.wait_for(lambda d: d.get("ready") is True, ready_timeout_s)
            if self.want_guard:
                self.start_guard(*self.guard_args, timeout_s=ready_timeout_s)
            if self.want_r1log:
                self.start_r1log(timeout_s=ready_timeout_s)
            if self.want_r1d:
                self.start_r1d(timeout_s=ready_timeout_s)
        except BaseException:
            self.stop()
            raise
        return self

    def start_guard(self, *extra: str, timeout_s: float = 30.0) -> Proc:
        """Start r1guard (again): e.g. after `stop_guard()` for the board-hold test, or `--observe-only`."""
        if self.guard is not None and self.guard.alive():
            raise RuntimeError("r1guard is already running")
        if self.guard is not None:
            self.old_guards.append(self.guard)
        self.guard = Proc(
            "r1guard",
            [str(self.apps / "r1guard"), "--config", str(self.cfg_path), "--shm-name", self.shm,
             "--domain", str(self.domain), *extra],
            self.env,
            log_dir=self.workdir,
        )  # fmt: skip
        self.guard.wait_for(lambda d: d.get("ready") is True, timeout_s)
        return self.guard

    def stop_guard(self, timeout_s: float = 10.0) -> int | None:
        """SIGTERM r1guard (it exits only on standby - PROCESSES.md); its exit code, None if it refused."""
        if self.guard is None or not self.guard.alive():
            return None
        return self.guard.terminate_and_wait(timeout_s)

    def start_r1d(self, *, timeout_s: float = 30.0) -> Proc:
        self.r1d = Proc(
            "r1d",
            [sys.executable, "-m", "r1mc3.service", "--shm-name", self.shm, "--domain", str(self.domain),
             "--port", "0", *self.r1d_args],
            self.env,
            log_dir=self.workdir,
        )  # fmt: skip
        ready = self.r1d.wait_for(lambda d: d.get("ready") is True, timeout_s)
        self.port = int(ready["port"])
        return self.r1d

    def start_r1log(self, *, timeout_s: float = 30.0) -> Proc:
        out = Path(self.env["R1LOG_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        self.r1log = Proc(
            "r1log",
            [sys.executable, "-m", "r1mc3.log", "run", "--out", str(out), "--segment", self.shm, *self.r1log_args],
            self.env,
            log_dir=self.workdir,
        )
        return self.r1log

    def stop(self) -> None:
        """Teardown (tests / harnesses): r1log gets SIGTERM (its final drain), the rest SIGKILL; the segment unlinked.
        Nothing here publishes anything - the bridge's physics simply stops."""
        if self.r1log is not None and self.r1log.alive():
            if self.r1log.terminate_and_wait(10.0) is None:
                self.r1log.kill()
        for p in (self.r1d, self.guard, *self.old_guards, self.rt):
            if p is not None:
                p.kill()
        if self.camera is not None:  # session 11
            with contextlib.suppress(Exception):
                self.camera.close()
            self.camera = None
        if self.bridge is not None:
            with contextlib.suppress(Exception):
                self.bridge.stop()
        unlink_shm(self.shm)

    def __enter__(self) -> ProcStack:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ the operator's side
    def cli(self, *words: str, timeout_s: float = 60.0) -> dict[str, Any]:
        """One `r1` word as the operator types it (`python -m r1mc3.cli --port P ...`): r1d's result + `_exit`."""
        t0 = time.monotonic()
        out = subprocess.run(
            [sys.executable, "-m", "r1mc3.cli", "--port", str(self.port), *words],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=timeout_s,
        )
        lines = out.stdout.strip().splitlines()
        res: dict[str, Any] = json.loads(lines[-1]) if lines else {"ok": False, "code": "NO_OUTPUT"}
        res["_exit"] = out.returncode
        res["_wall_s"] = time.monotonic() - t0
        if not lines:
            res["_stderr"] = out.stderr[-800:]
        return res

    def status(self) -> dict[str, Any]:
        return self.cli("status")

    def mode(self) -> str | None:
        st = self.status()
        return ((st.get("data") or {}).get("core") or {}).get("mode")

    def wait_mode(self, mode: str, timeout_s: float, *, poll_s: float = 0.3) -> dict[str, Any]:
        """Poll `r1 status` until the core's mode is `mode`; AssertionError with the last status if not."""
        end = time.monotonic() + timeout_s
        st: dict[str, Any] = {}
        while time.monotonic() < end:
            st = self.status()
            if ((st.get("data") or {}).get("core") or {}).get("mode") == mode:
                return st
            time.sleep(poll_s)
        raise AssertionError(f"mode {mode} not reached in {timeout_s} s: {st}")

    def pids(self) -> dict[str, int]:
        return {p.name: p.pid for p in (self.rt, self.guard, self.r1d, self.r1log) if p is not None and p.alive()}
