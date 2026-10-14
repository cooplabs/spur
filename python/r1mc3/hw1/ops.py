"""r1d's stand-test ops: `board_hold.pause`, `guard_takeover.stall`, `shutdown` (the `r1 board-hold`,
`r1 guard-takeover`, `r1 shutdown` words; R1_CLI.md).

Each is OPERATOR ONLY, checks the robot's state through the segment first (a refusal moves nothing and says which
check failed), and runs in r1d's HW1 worker thread (one at a time: RESOURCE_BUSY otherwise), so the service's tick -
the streams' TTLs, the hands - keeps running while r1rt is paused. Without `--yes` (dry_run) the board-hold and the
guard-takeover tests return the plan + the checks only.

Records: `<UTC>_board_hold_<N>ms.*` / `<UTC>_guard_takeover_<S>s.*` (records written before session R carry the
older numbered stems; nothing parses the stem, the JSON says what it is).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from r1mc3.service.codes import OWNERS, Code
from r1mc3.service.schemas import BoardHoldReq, GuardTakeoverReq, Result, ShutdownReq, fail, ok

from .pause import (
    RECORD_POST_S,
    RECORD_PRE_S,
    TAKEOVER_DEADLINE_S,
    ChildLowstateRecorder,
    monotonic_ns,
    sigstop_for,
    window_stats,
)
from .session import find_r1log, lookback_s, new_session_folder, save_session, stop_r1log

if TYPE_CHECKING:
    from r1mc3.service.service import Service, Ticket

ALIVE_MS = 100.0  # a heartbeat older than this = that process is stopped / not running (the core: 50 ms GUARD_ABSENT)
TAKEOVER_PRE_S = 0.5
TAKEOVER_POST_S = 1.0
OWNER_RT, OWNER_GUARD = 1, 2


class Hw1Ops:
    def __init__(
        self,
        svc: Service,
        *,
        domain: int | None = None,
        recorder_factory: Callable[[], Any] | None = None,
        record_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        session_dir: str | Path | None = None,
        pauser: Callable[..., Any] = sigstop_for,
    ) -> None:
        self.svc = svc
        self.domain = domain
        self._recorder_factory = recorder_factory
        self.record_dir = Path(record_dir or os.environ.get("R1_RECORD_DIR", "out/records"))
        ld = log_dir if log_dir is not None else os.environ.get("R1LOG_DIR")
        self.log_dir = Path(ld) if ld else None
        sd = session_dir if session_dir is not None else os.environ.get("R1_SESSION_DIR")
        self.session_dir = Path(sd) if sd else None
        self.pauser = pauser
        self.started_wall = time.time()
        self._pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="r1d-hw1")
        self._busy = threading.Lock()
        self.last: dict[str, Any] | None = None

    # ------------------------------------------------------------------ the robot's state, by checks
    def snapshot(self) -> dict[str, Any]:
        port = self.svc.port
        now = port.now_ns()
        hb, gb = port.heartbeat(), port.guard_heartbeat()
        owner, gen = port.token()
        armed_fn = getattr(port, "guard_armed", None)
        pid_fn = getattr(port, "rt_pid", None)
        return {
            "mode": None if hb is None else port.tables.mode_name(hb.mode),
            "core_age_ms": None if hb is None else (now - hb.t_ns) * 1e-6,
            "core_faults": 0 if hb is None else hb.faults,
            "guard_age_ms": None if gb is None else (now - gb.t_ns) * 1e-6,
            "guard_mode": None if gb is None else port.tables.mode_name(gb.mode),
            "guard_armed": armed_fn() if callable(armed_fn) else None,
            "owner": OWNERS.get(owner, str(owner)),
            "owner_id": owner,
            "generation": gen,
            "rt_pid": pid_fn() if callable(pid_fn) else None,
        }

    @staticmethod
    def _alive(age_ms: float | None) -> bool:
        return age_ms is not None and age_ms < ALIVE_MS

    def _pid_check(self, s: dict[str, Any]) -> tuple[bool, str]:
        pid = s["rt_pid"]
        if pid is None:
            return False, "r1rt's pid is not readable (no r1rt process: lockstep, or the segment header unreadable)"
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False, f"r1rt pid {pid} does not exist"
        except PermissionError:
            return False, f"r1rt pid {pid}: no permission to signal it (r1d must run as r1rt's user)"
        return True, f"r1rt pid {pid}"

    def checks_board_hold(self, s: dict[str, Any]) -> list[dict[str, Any]]:
        out = [
            _chk("r1rt alive", self._alive(s["core_age_ms"]), f"heartbeat {_ms(s['core_age_ms'])}"),
            _chk(
                "ON_STAND",
                s["mode"] == "on_stand",
                f"mode {s['mode']} (the board-hold test runs only on the stand, in ON_STAND)",
            ),
            _chk("r1rt owns the motors", s["owner"] == "r1rt", f"token {s['owner']}"),
        ]
        g_alive = self._alive(s["guard_age_ms"])
        if not g_alive:
            out.append(_chk("guard OFF", True, f"r1guard stopped (heartbeat {_ms(s['guard_age_ms'])})"))
        elif s["guard_armed"] is False:
            out.append(_chk("guard OFF", True, "r1guard runs observe-only (will not take over)"))
        elif s["guard_armed"] is True:
            out.append(
                _chk("guard OFF", False, "r1guard is ARMED: it would take over at ~10 ms - stop it for this test")
            )
        else:
            out.append(
                _chk(
                    "guard OFF",
                    False,
                    "r1guard runs and this build cannot tell whether it is armed: stop it for this test",
                )
            )
        ok_pid, why = self._pid_check(s)
        out.append(_chk("r1rt pid", ok_pid, why))
        return out

    def checks_guard_takeover(self, s: dict[str, Any]) -> list[dict[str, Any]]:
        out = [
            _chk("r1rt alive", self._alive(s["core_age_ms"]), f"heartbeat {_ms(s['core_age_ms'])}"),
            _chk(
                "ON_STAND",
                s["mode"] == "on_stand",
                f"mode {s['mode']} (the guard-takeover test runs only on the stand, in ON_STAND)",
            ),
            _chk("r1rt owns the motors", s["owner"] == "r1rt", f"token {s['owner']}"),
            _chk("guard alive", self._alive(s["guard_age_ms"]), f"r1guard heartbeat {_ms(s['guard_age_ms'])}"),
            _chk(
                "guard armed",
                s["guard_armed"] is not False,
                "observe-only: it would NOT take over"
                if s["guard_armed"] is False
                else ("armed" if s["guard_armed"] else "armed: unknown on this build (r1d's 0.2 s net applies)"),
            ),
        ]
        ok_pid, why = self._pid_check(s)
        out.append(_chk("r1rt pid", ok_pid, why))
        return out

    # ------------------------------------------------------------------ the board-hold test
    def board_hold(self, r: BoardHoldReq, t: Ticket) -> Result | None:
        if r.requester != "operator":
            return fail(
                r.op,
                r.id,
                Code.NOT_PERMITTED,
                reason="the board-hold test is an operator step (an explicit OK on the day)",
            )
        s = self.snapshot()
        checks = self.checks_board_hold(s)
        plan = {
            "pause_ms": r.pause_ms,
            "record_s": [-RECORD_PRE_S, r.pause_ms * 1e-3 + RECORD_POST_S],
            "what": "r1rt is stopped (SIGSTOP) for the pause: no rt/lowcmd from us; the board's behaviour is recorded",
            "moves": "nothing is commanded; the board holds or lets go (that is what this test measures)",
            "how_to_stop": "it ends by itself (SIGCONT; a detached backstop SIGCONT too); robot shell: kill -CONT "
            "$(pidof r1rt); you: the remote / E-stop",
        }
        failed = [c for c in checks if not c["ok"]]
        if failed:
            return fail(r.op, r.id, Code.WRONG_MODE, checks=checks, failed=[c["check"] for c in failed], plan=plan)
        if r.dry_run:
            return ok(r.op, r.id, dry_run=True, plan=plan, checks=checks)
        return self._submit(r, t, lambda: self._run_board_hold(r, s, checks))

    def _run_board_hold(self, r: BoardHoldReq, s0: dict[str, Any], checks: list[dict[str, Any]]) -> Result:
        rec = self._recorder()
        t_rec0 = monotonic_ns()
        time.sleep(RECORD_PRE_S)
        p = self.pauser(int(s0["rt_pid"]), r.pause_ms * 1e-3)
        time.sleep(RECORD_POST_S)
        t_rec1 = monotonic_ns()
        s1 = self.snapshot()
        data = self._finish_record(rec, "board_hold", r.pause_ms, t_rec0, t_rec1, p, s0, s1, checks)
        if s1["owner"] != "r1rt":
            data["warning"] = (
                f"the token moved to {s1['owner']} during the test: the guard was not off - not a valid board-hold test"
            )
        return ok(r.op, r.id, **data)

    # ------------------------------------------------------------------ the guard-takeover test
    def guard_takeover(self, r: GuardTakeoverReq, t: Ticket) -> Result | None:
        if r.requester != "operator":
            return fail(
                r.op,
                r.id,
                Code.NOT_PERMITTED,
                reason="the guard-takeover test is an operator step (an explicit OK on the day)",
            )
        s = self.snapshot()
        checks = self.checks_guard_takeover(s)
        plan = {
            "stop_s": r.stop_s,
            "what": "r1rt is stopped (SIGSTOP) in ON_STAND: the guard must take over within ~10 ms and hold the last "
            "command; after SIGCONT r1rt stays silent until `r1 reclaim`",
            "moves": "nothing should move (the guard holds the same pose, stiff)",
            "net": f"no guard takeover within {TAKEOVER_DEADLINE_S} s -> r1d continues r1rt at once",
            "next": "r1 reclaim",
            "how_to_stop": "it ends by itself (SIGCONT; a detached backstop SIGCONT too); `r1 damp` is always yours",
        }
        failed = [c for c in checks if not c["ok"]]
        if failed:
            return fail(r.op, r.id, Code.WRONG_MODE, checks=checks, failed=[c["check"] for c in failed], plan=plan)
        if r.dry_run:
            return ok(r.op, r.id, dry_run=True, plan=plan, checks=checks)
        return self._submit(r, t, lambda: self._run_guard_takeover(r, s, checks))

    def _run_guard_takeover(self, r: GuardTakeoverReq, s0: dict[str, Any], checks: list[dict[str, Any]]) -> Result:
        rec = self._recorder()
        t_rec0 = monotonic_ns()
        time.sleep(TAKEOVER_PRE_S)
        taken: dict[str, int] = {}

        def watch(elapsed_s: float) -> str | None:
            o, g = self.svc.port.token()
            if o == OWNER_GUARD and "t_ns" not in taken:
                taken["t_ns"] = monotonic_ns()
                taken["generation"] = g
            if "t_ns" not in taken and elapsed_s > TAKEOVER_DEADLINE_S:
                return f"the guard did not take the token within {TAKEOVER_DEADLINE_S} s"
            return None

        p = self.pauser(int(s0["rt_pid"]), r.stop_s, watch=watch)
        time.sleep(TAKEOVER_POST_S)
        t_rec1 = monotonic_ns()
        s1 = self.snapshot()
        data = self._finish_record(rec, "guard_takeover", r.stop_s, t_rec0, t_rec1, p, s0, s1, checks)
        from r1mc3.log.records import fault_names

        core_faults = [str(f).upper() for f in fault_names(int(s1["core_faults"]))]
        data["takeover"] = {
            "token_seen_r1guard_ms_after_stop": None
            if "t_ns" not in taken
            else round((taken["t_ns"] - p.t_stop_ns) * 1e-6, 3),
            "seen_by": "r1d polling the token every 1 ms (coarse; the board-side gap is the bridge's / r1log's)",
        }
        data["after_sigcont"] = {
            "r1rt_alive": self._alive(s1["core_age_ms"]),
            "r1rt_mode": s1["mode"],
            "r1rt_faults": core_faults,
            "token": s1["owner"],
            "r1rt_silent": s1["owner"] == "r1guard" and "NOT_OWNER" in core_faults,
        }
        if p.early is not None or "t_ns" not in taken:
            return fail(r.op, r.id, Code.FAULT, reason=p.early or "the guard never took the token", **data)
        data["next"] = "r1 reclaim (operator): r1rt takes the motors back from the guard"
        return ok(r.op, r.id, **data)

    # ------------------------------------------------------------------ shared
    def _recorder(self) -> Any:
        """r1d's lowstate recorder for one test: its own process (no GIL shared with the pause), ready = DDS matched."""
        if self._recorder_factory is not None:
            return self._recorder_factory()
        if self.domain is None:
            return None
        self.record_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.record_dir / f".lowstate_{os.getpid()}_{monotonic_ns()}.npz"
        rec = ChildLowstateRecorder(self.domain, tmp)
        if not rec.ready:
            rec.close()
            raise RuntimeError("r1d's lowstate recorder saw no rt/lowstate within 5 s (DDS / the robot's link?)")
        return rec

    def _submit(self, r: Any, t: Ticket, fn: Callable[[], Result]) -> Result | None:
        if not self._busy.acquire(blocking=False):
            return fail(r.op, r.id, Code.RESOURCE_BUSY, reason="another HW1 test / shutdown is running")

        def run() -> None:
            try:
                res = fn()
            except Exception as e:  # never leave the ticket open; the pauser's finally already sent SIGCONT
                res = fail(r.op, r.id, Code.FAULT, reason=repr(e))
            finally:
                self._busy.release()
            self.last = res.model_dump(mode="json")
            t.finish(res)

        self._pool.submit(run)
        return None

    def _finish_record(
        self,
        rec: Any,
        test: str,
        size: float,
        t_rec0: int,
        t_rec1: int,
        p: Any,
        s0: dict[str, Any],
        s1: dict[str, Any],
        checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        stem = f"{stamp}_{test}_{size:g}{'ms' if test == 'board_hold' else 's'}"
        self.record_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        lowstate: dict[str, Any] = {"recorded": False}
        if rec is not None:
            rec.stop()
            a = rec.arrays()
            npz = self.record_dir / f"{stem}.lowstate.npz"
            np.savez(npz, **a)
            files["lowstate"] = str(npz)
            lowstate = {
                "recorded": True,
                "samples": int(a["t_rx_ns"].size),
                "before": window_stats(a, t_rec0, p.t_stop_ns),
                "during": window_stats(a, p.t_stop_ns, p.t_cont_ns),
                "after": window_stats(a, p.t_cont_ns, t_rec1),
            }
            close = getattr(rec, "close", None)
            if callable(close):
                close()
        t0 = p.t_stop_ns
        timeline = [
            {"t_s": round((t_rec0 - t0) * 1e-9, 4), "event": "lowstate recorder on"},
            {"t_s": 0.0, "event": "SIGSTOP r1rt"},
            {
                "t_s": round((p.t_cont_ns - t0) * 1e-9, 4),
                "event": "SIGCONT r1rt" + (f" ({p.early})" if p.early else ""),
            },
            {"t_s": round((t_rec1 - t0) * 1e-9, 4), "event": "recorder off"},
        ]
        data = {
            "test": test,
            "pause": p.as_dict(),
            "timeline": sorted(timeline, key=lambda e: e["t_s"]),
            "lowstate": lowstate,
            "before": {k: s0[k] for k in ("mode", "owner", "generation", "guard_armed")},
            "after": {k: s1[k] for k in ("mode", "owner", "generation", "guard_armed")},
            "checks": checks,
            "files": files,
        }
        man = self.record_dir / f"{stem}.json"
        man.write_text(json.dumps({"schema": f"r1mc3.hw1_{test}/1", "utc": stamp, **data}, indent=1, default=str))
        files["manifest"] = str(man)
        return data

    # ------------------------------------------------------------------ shutdown
    def shutdown(self, r: ShutdownReq, t: Ticket) -> Result | None:
        if r.requester != "operator":
            return fail(r.op, r.id, Code.NOT_PERMITTED, reason="shutdown is the operator's word")
        s = self.snapshot()
        core_alive = self._alive(s["core_age_ms"])
        mode = s["mode"] if core_alive else None
        allowed = mode in ("monitor", "damp") if core_alive else s["owner"] == "none"
        if s["owner"] == "r1guard" and mode != "damp":
            allowed = False  # the guard publishes (holds / balances): we still own the motors
        if not allowed:
            return fail(
                r.op,
                r.id,
                Code.WRONG_MODE,
                mode=mode if core_alive else "r1rt not running",
                token=s["owner"],
                needs="monitor or damp (the session end: hold the robot -> r1 damp -> stand -> r1 shutdown)",
            )
        return self._submit(r, t, lambda: self._run_shutdown(r, s))

    def _run_shutdown(self, r: ShutdownReq, s: dict[str, Any]) -> Result:
        name = getattr(self.svc.port, "shm_name", "") or ""
        pids = find_r1log(name) if name else []
        stopped = stop_r1log(pids) if pids else {}
        still = [p for p, st in stopped.items() if st != "exited"]
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        # session 8 (HW1 #13 / #20): R1_SESSION_DIR is the PARENT; each shutdown gets a new folder, and the files are
        # every file of the look-back window not yet saved (not only those since r1d's last start)
        parent = self.session_dir or (self.record_dir / "sessions")
        folder = new_session_folder(parent, r.label, stamp)
        saved = save_session(
            session_dir=folder,
            log_dir=self.log_dir,
            record_dir=self.record_dir,
            since_wall=min(self.started_wall, time.time() - lookback_s()),
            sessions_root=parent,
            journal=list(self.svc.journal),
            extra={"mode_at_shutdown": s["mode"], "token": s["owner"], "r1log": stopped},
            live=[p for p in (getattr(self.svc, "_journal_path", None),) if p is not None],  # r1d keeps appending
            log_closed=bool(pids) and not still,  # r1log not found or not stopped: its files are snapshots
        )
        safe = bool(saved["verified"]) and not still
        msg = (
            "safe to power off now"
            if safe
            else "NOT safe to power off yet: " + ("r1log did not stop; " if still else "") + "a file did not verify"
        )
        data: dict[str, Any] = {
            "message": msg,
            "safe_to_power_off": safe,
            "r1log": {"found": pids, "stop": stopped, "log_dir": None if self.log_dir is None else str(self.log_dir)},
            "saved": saved,
            "processes": "r1d stops r1log only (its final drain) and stays up; r1rt / r1guard are stopped by "
            "`r1mc3-svc stop` (systemd; refused while we own the motors) or by the power-off",
        }
        if not safe:
            return fail(r.op, r.id, Code.FAULT, **data)
        return ok(r.op, r.id, **data)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=False)


def _chk(name: str, good: bool, value: str) -> dict[str, Any]:
    return {"check": name, "ok": bool(good), "value": value}


def _ms(age: float | None) -> str:
    return "none" if age is None else f"{age:.0f} ms old"
