"""`Service`: r1d's requests and its tick (DESIGN.md §5, §3.3, §3.5).

    svc = Service(port, releaser=MotionSwitcherReleaser(domain), hands=Dex3Hands(domain))
    t = svc.submit({"op": "joint.command", "group": "arm_left", "q": [...], "ttl": 0.2})   # -> Ticket
    svc.tick()                                     # 100 Hz: sequences, jogs, recordings (server: its own thread)
    t.result                                       # Result (pydantic) once done

Every op completes inside `submit` except the two mode sequences, which need the core and the motion switcher:

* `takeover`: the core's gates through the mailbox (`R1RT_REQ_TAKEOVER`: MIRROR republishes Unitree's last command)
  -> the motion switcher's ReleaseMode (an RPC in r1d, never in an RT thread; v2 waited <= 4 s) -> RELEASED (BLEND
  into BALANCE) or ABORT (back to MONITOR, the built-in still drives the motors). Nothing in r1d ever damps.
* `on_stand`: the same with `R1RT_REQ_ON_STAND` (the core blends into a stiff hold instead of the policy).
* `reclaim` (operator only, session 6): r1rt takes the motors back from r1guard after the guard-takeover test
  (the core's `reclaim`).
* `board_hold.pause` / `guard_takeover.stall` / `shutdown` (operator only, session 6): the stand tests and the
  session end with the log save built in (`r1mc3.hw1.ops`), run in r1d's HW1 worker thread.

They advance in `tick()`, so the same code runs in real time (the server's tick thread; the release in a worker
thread so a slow RPC never blocks the streams) and in lockstep (the simulator calls `tick()` between writer ticks).

L0 / L1 go into the core's upper_cmd ring stamped with the stop epoch (a stop refuses older records):
`joint.command` = one STREAM record (TTL <= 300 ms: the executor decelerates to HOLD when it expires),
`joint.hold` = a HOLD record, `stream.joint_jog` / `stream.head` rates = STREAM setpoints r1d integrates at its tick
until the request's TTL (then nothing more is sent; the record TTL stops the group). Legs / waist only in ON_STAND
(WRONG_MODE otherwise); arms / head in BALANCE or ON_STAND; nothing is queued while the core does not execute
records (a record left in the ring would run at the next entry).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from pydantic import ValidationError

from .codes import GROUP_JOINTS, LOWER_GROUPS, OWNERS, SOURCES, STOP_LEVELS, Code, group_of_joint
from .port import FloatArray, Port
from .schemas import (
    GOAL_OPS,
    OPS,
    REQUEST,
    BoardHoldReq,
    CameraDepthImageReq,
    CameraExtentReq,
    CameraObserveReq,
    CameraPointReq,
    CameraProjectReq,
    CameraReq,
    FreezeReq,
    GuardTakeoverReq,
    HandCommandReq,
    HandStateReq,
    HeadStreamReq,
    JointCommandReq,
    JointHoldReq,
    JointJogReq,
    ReclaimReq,
    RecordReq,
    Result,
    ResumeReq,
    ShutdownReq,
    StandFreezeReq,
    StandHoldReq,
    StatusReq,
    StopReq,
    TakeoverReq,
    fail,
    ok,
)

R1RT_OK = 0
EXEC_MODES = ("balance", "on_stand")  # modes whose writer runs the executor on fresh records (EXECUTOR.md)
JOG_RECORD_TTL_S = 0.1  # each integrated jog setpoint lives this long (several r1d ticks: one late tick is harmless)
LIMIT_MARGIN = 0.05  # rad inside the profile's position limits (the executor's own margin, EXECUTOR.md)
ALIVE_MS = 100.0  # a heartbeat older than this: that process is stopped / not running (the core's GUARD_ABSENT: 50 ms)
# r1d waits this long for these results
LONG_OPS = {"board_hold.pause": 30.0, "guard_takeover.stall": 30.0, "shutdown": 60.0}
ALARM_FAULTS = {"GUARD_ABSENT", "NOT_OWNER", "MOTOR_ERROR", "TILT_FALL", "LOWSTATE_STALE", "LOWSTATE_LOST",
                "POLICY_INVALID", "BUILTIN_ACTIVE", "TAKEOVER_ALARM", "RT_NOT_GRANTED", "OVER_TEMP"}  # fmt: skip
STAND_SESSION_REASON = (
    "stand session: r1rt was started with --stand-session - balance is refused until r1rt is restarted without it "
    "(runbook: lift the stand lock)"
)
STAND_FREEZE_MAX_DEV_RAD = 0.15  # rt/include/r1rt/step_api.h R1RT_ON_STAND_FREEZE_MAX_DEV_RAD (the core decides)
LOWER_JOINTS = tuple(range(14))  # legs + waist (SDK 0-13): what the stand's freeze ramps
READ_ONLY_OPS = ("status", "hand.state", "record.list", "record.plan")  # not journaled (polled often, move nothing)


class Releaser(Protocol):
    """The motion switcher's ReleaseMode (Unitree's built-in controller lets go of the motors)."""

    def release(self, timeout_s: float) -> tuple[bool, dict[str, Any]]: ...


class HandsLike(Protocol):
    def command(
        self,
        side: str,
        q: Any,
        *,
        dq: Any = None,
        kp: Any = None,
        kd: Any = None,
        tau: Any = None,
        ttl_s: float = 1.0,
    ) -> dict[str, Any]: ...
    def state(self, side: str) -> dict[str, Any]: ...


@dataclass
class Ticket:
    request: dict[str, Any]
    result: Result | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def finish(self, r: Result) -> None:
        self.result = r
        self.done.set()


@dataclass
class _Sequence:
    """A takeover / on_stand in progress (advanced by tick)."""

    ticket: Ticket
    req: TakeoverReq
    first: str  # "takeover" | "on_stand"
    stage: str = "gates"  # gates -> release -> decide -> done
    seq: int = 0
    t0_ns: int = 0
    future: cf.Future[tuple[bool, dict[str, Any]]] | None = None
    release: tuple[bool, dict[str, Any]] | None = None
    log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Jog:
    group: str
    cmd_id: int
    q_set: FloatArray
    dq: FloatArray
    until_ns: int
    last_ns: int


class Service:
    def __init__(
        self,
        port: Port,
        *,
        releaser: Releaser | None = None,
        hands: HandsLike | None = None,
        limits: Mapping[str, tuple[float, float]] | None = None,
        tick_hz: float = 100.0,
        recorder_factory: Callable[[Service], Any] | None = None,
        hw1_factory: Callable[[Service], Any] | None = None,
        journal_dir: str | Path | None = None,
        contact: Any | None = None,
        rt_config: str | Path | None = None,
        planner: Any | None = None,
        world: Any | None = None,
        grasp_factory: Callable[..., Any] | None = None,
        camera: Any | None = None,
    ) -> None:
        self.port = port
        self.releaser = releaser
        self.hands = hands
        if hands is not None:  # HW2: the grasp start's model-only references once, now - computed at every grasp's
            # start (~100 ms on the Orin) they starved r1hands' reader thread past its 100 ms freshness limit and
            # hand.grasp / hand.calibrate refused "no valid hand state" (r1mc3.hands.grip.empty_aperture)
            from r1mc3.hands.grip import warm_grip_caches

            warm_grip_caches()
        self.tick_hz = tick_hz
        self.limits = limits  # joint name -> (q_min, q_max); None = the core's own limits only
        self._lock = threading.RLock()
        self._next_id = 1
        self._seqs: list[_Sequence] = []
        self._jogs: dict[str, _Jog] = {}
        self._last_set: dict[str, FloatArray] = {}  # group -> the last setpoint r1d sent
        self._pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="r1d-release")
        self.pushed: list[tuple[int, str, int, int]] = []  # (t_ns, group, source, rc) - for tests / status
        self.recorder = recorder_factory(self) if recorder_factory is not None else None
        # r1d's request journal (every request that can change something + its result): kept for `r1 shutdown`'s log
        # save, and appended live to a file in the records dir so a crashed r1d's journal survives too
        self.journal: list[dict[str, Any]] = []
        jd = journal_dir if journal_dir is not None else os.environ.get("R1_RECORD_DIR")
        self._journal_path = (
            Path(jd) / f"r1d_journal_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}Z_{os.getpid()}.jsonl"
            if jd
            else None
        )
        self.hw1 = hw1_factory(self) if hw1_factory is not None else None
        self.contact = contact  # r1hands' pad-contact bits -> the segment's hands_contact, at the tick (100 Hz)
        rc = rt_config if rt_config is not None else os.environ.get("R1RT_CONFIG")
        self.rt_config_path = Path(rc) if rc else None  # r1rt's own config file (the stand pose, stand_ramp_s)
        self.freeze_from_stand = False  # r1d's own record: the current STAND_FREEZE came from `stand_freeze` (S10)
        self.contact_errors = 0
        # session 9: L2 / L3 (l2.py) and the pieces they use (None = those ops answer UNSUPPORTED)
        self.planner = planner  # r1mc3.plan (a module or a PlannerWorker with plan(snapshot, goal, q_now))
        self.world = world  # r1mc3.world.WorldModel
        self.grasp_factory = grasp_factory  # (hands, side, grip, object_class, ...) -> r1mc3.hands.grip.GraspRunner
        self.camera = camera  # r1cam's client (request(dict) -> dict); None = r1mc3.record.runner.CameraClient()
        from .l2 import L2

        self.l2 = L2(self)

    # ------------------------------------------------------------------ requests
    def submit(self, req: Mapping[str, Any]) -> Ticket:
        t = Ticket(dict(req))
        rid = req.get("id") if isinstance(req, Mapping) else None
        op = str(req.get("op", "?")) if isinstance(req, Mapping) else "?"
        try:
            parsed = REQUEST.validate_python(dict(req))
        except ValidationError as e:
            code = Code.UNKNOWN_COMMAND if "op" not in req or "union_tag_invalid" in str(e) else Code.INVALID_ARGS
            t.finish(fail(op, rid, code, errors=[{"loc": list(x["loc"]), "msg": x["msg"]} for x in e.errors()]))
            return t
        with self._lock:
            try:
                r = self._dispatch(parsed, t)
            except NotImplementedError as e:
                r = fail(op, rid, Code.UNSUPPORTED, reason=str(e))
            except BlockingIOError as e:  # the core's mailbox holds another client's pending request
                r = fail(op, rid, Code.RESOURCE_BUSY, reason=str(e))
        if r is not None:
            t.finish(r)
        return t

    def handle(self, req: Mapping[str, Any], timeout_s: float = 15.0) -> dict[str, Any]:
        """Submit and wait (the server's path; needs the tick thread for the mode sequences and the goals)."""
        op = str(req.get("op", "?"))
        timeout_s = max(timeout_s, LONG_OPS.get(op, 0.0))
        if op == "goal.wait" or (op in GOAL_OPS and req.get("wait")):  # the goal's own timeout + its end
            timeout_s = max(timeout_s, float(req.get("timeout_s", 30.0)) + 2.0)
        t0 = time.time()
        t = self.submit(req)
        if not t.done.wait(timeout_s):
            out = fail(op, req.get("id"), Code.TIMEOUT, waited_s=timeout_s).model_dump(mode="json")
        else:
            assert t.result is not None
            out = t.result.model_dump(mode="json")
        if op not in READ_ONLY_OPS:
            self._journal(t0, req, out)
        return out

    def _journal(self, t0: float, req: Mapping[str, Any], out: Mapping[str, Any]) -> None:
        e = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0)) + f".{int((t0 % 1) * 1000):03d}Z",
            "request": dict(req),
            "ok": out.get("ok"),
            "code": out.get("code"),
            "moved": out.get("moved"),
        }
        with self._lock:
            self.journal.append(e)
        if self._journal_path is not None:
            try:
                self._journal_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._journal_path, "a") as f:
                    f.write(json.dumps(e, default=str) + "\n")
            except OSError:
                pass  # the in-memory journal still goes into the session folder

    def _dispatch(self, r: Any, t: Ticket) -> Result | None:
        from . import leases  # session 10 (api): off until a client takes a lease

        refused = leases.check(self, r)
        if refused is not None:
            return refused
        if isinstance(r, StatusReq):
            return ok(r.op, r.id, **self.status())
        if isinstance(r, StopReq):
            return self._stop(r)
        if isinstance(r, ResumeReq):
            return self._mode_request_now(r.op, r.id, "resume", r.requester, t)
        if isinstance(r, FreezeReq):
            return self._mode_request_now(r.op, r.id, "stand_freeze", r.requester, t)
        if isinstance(r, TakeoverReq):
            return self._start_sequence(r, t)
        if isinstance(r, ReclaimReq):
            return self._reclaim(r, t)
        if isinstance(r, StandFreezeReq):
            return self._stand_freeze(r, t)
        if isinstance(r, StandHoldReq):
            if r.requester != "operator":
                return fail(r.op, r.id, Code.NOT_PERMITTED, reason="stand_hold is the operator's word")
            if self.mode_name() != "stand_freeze":
                return fail(r.op, r.id, Code.WRONG_MODE, mode=self.mode_name(), needs="stand_freeze (from the stand)")
            return self._mode_request_now(r.op, r.id, "stand_hold", r.requester, t)
        if isinstance(r, BoardHoldReq | GuardTakeoverReq | ShutdownReq):
            if self.hw1 is None:
                return fail(r.op, r.id, Code.UNSUPPORTED, reason="no HW1 tools in this r1d")
            if isinstance(r, BoardHoldReq):
                res3: Result | None = self.hw1.board_hold(r, t)
                return res3
            if isinstance(r, GuardTakeoverReq):
                res6: Result | None = self.hw1.guard_takeover(r, t)
                return res6
            ress: Result | None = self.hw1.shutdown(r, t)
            return ress
        if isinstance(r, JointCommandReq):
            return self._joint_command(r)
        if isinstance(r, JointHoldReq):
            return self._joint_hold(r)
        if isinstance(r, JointJogReq):
            return self._jog(r.op, r.id, r.group, np.asarray(r.dq, dtype=np.float64), r.ttl)
        if isinstance(r, HeadStreamReq):
            if r.rate is not None:
                return self._jog(r.op, r.id, "head", np.asarray(r.rate, dtype=np.float64), r.ttl)
            assert r.q is not None
            return self._joint_command(JointCommandReq(op="joint.command", id=r.id, group="head", q=r.q, ttl=r.ttl))
        if isinstance(r, HandCommandReq | HandStateReq):
            return self._hand(r)
        if isinstance(r, RecordReq):
            if self.recorder is None:
                return fail(r.op, r.id, Code.UNSUPPORTED, reason="no recorder in this r1d")
            res: Result | None = self.recorder.request(r, t)
            return res
        if isinstance(r, CameraReq):
            return self._camera(r)
        # session 10: each agent dispatches its new request types between its markers
        # >>> s10:cam
        if isinstance(r, CameraObserveReq | CameraPointReq | CameraExtentReq | CameraProjectReq | CameraDepthImageReq):
            from .perception import perception_of

            return perception_of(self).handle(r, t)
        # <<< s10:cam
        # >>> s10:measure
        from .schemas import MeasurePhoneReq, MeasureSetupReq

        if isinstance(r, MeasureSetupReq | MeasurePhoneReq):  # off the lock in a worker (measure_ops.py)
            from .measure_ops import measure_ops

            return measure_ops(self).dispatch(r, t)
        # <<< s10:measure
        # >>> s10:grasp
        # <<< s10:grasp
        # >>> s10:api
        from . import l2_paths  # arm.move_path / guarded_move / slide, skill.pick / place, par.run (session 10)

        if l2_paths.handles(r):
            return l2_paths.dispatch(self.l2, r, t)
        if leases.handles(r):
            return leases.handle(self, r)
        from . import stream_l1

        if stream_l1.handles(r):
            return stream_l1.arm_twist(self, r)
        # <<< s10:api
        # >>> s11:core
        from . import balance  # `r1 balance`: ENTER_BALANCE from a freeze, operator only (session 11)

        if balance.handles(r):
            return balance.request(self, r, t)
        # <<< s11:core
        # >>> s11:grip
        # <<< s11:grip
        # >>> s11:cam
        # <<< s11:cam
        # >>> s11:api
        # <<< s11:api
        # >>> s11:episode
        # <<< s11:episode
        # >>> s12:approach
        # <<< s12:approach
        # >>> s12:grip
        # <<< s12:grip
        # >>> s12:cam
        # <<< s12:cam
        # >>> s12:fixes
        # <<< s12:fixes
        # >>> s13:tools
        # <<< s13:tools
        # >>> s13:click
        # <<< s13:click
        # >>> s13:setup
        # <<< s13:setup
        # >>> s13:pose
        # <<< s13:pose
        # >>> s13:grip
        # <<< s13:grip
        # >>> s13:core
        # <<< s13:core
        # >>> s13:ops
        # <<< s13:ops
        if self.l2.handles(getattr(r, "op", "")):
            return self.l2.dispatch(r, t)
        if getattr(r, "op", None) in OPS:  # session 9: an L2 / L3 op this r1d does not run yet
            return fail(r.op, r.id, Code.UNSUPPORTED, reason=f"{r.op}: not implemented in this r1d yet")
        return fail(getattr(r, "op", "?"), getattr(r, "id", None), Code.UNKNOWN_COMMAND)

    # ------------------------------------------------------------------ status
    def mode_name(self) -> str:
        hb = self.port.heartbeat()
        return "absent" if hb is None else self.port.tables.mode_name(hb.mode)

    def status(self) -> dict[str, Any]:
        """The core, the guard, the token, in numbers AND in plain words (`summary`; R1_CLI.md `r1 status`)."""
        now = self.port.now_ns()
        hb = self.port.heartbeat()
        gb = self.port.guard_heartbeat()
        owner, gen = self.port.token()
        owner_name = OWNERS.get(owner, str(owner))
        armed_fn = getattr(self.port, "guard_armed", None)
        armed: bool | None = armed_fn() if callable(armed_fn) else None
        info_fn = getattr(self.port, "guard_info", None)
        ginfo: dict[str, Any] | None = info_fn() if callable(info_fn) else None
        out: dict[str, Any] = {
            "core": None,
            "guard": None,
            "token": {"owner": owner_name, "generation": gen},
            "jogs": sorted(self._jogs),
            "goals": self.l2.status(),
            "grips": self.l2.grip_status(),
            "sequences": [s.first for s in self._seqs],
            "binding_missing": self.port.missing(),
        }
        core_mode = None
        if hb is not None:
            core_mode = self.port.tables.mode_name(hb.mode)
            out["core"] = {
                "mode": core_mode,
                "faults": _fault_names(hb.faults),
                "age_ms": (now - hb.t_ns) * 1e-6,
                "alive": (now - hb.t_ns) * 1e-6 < ALIVE_MS,
                "rate_hz": hb.rate_hz,
                "rt_granted": hb.rt_granted,
                "blend_progress": hb.blend_progress,
                "seq": hb.seq,
                "stand_session": self.stand_session(),  # None: this binding / r1rt does not say
                **self.stop_counts(),  # session 8 (HW1 #31): deliberate stops / long silences, apart from overruns
            }
        if gb is not None:
            g_age = (now - gb.t_ns) * 1e-6
            out["guard"] = {
                "mode": self.port.tables.mode_name(gb.mode),
                "faults": _fault_names(gb.faults),
                "age_ms": g_age,
                "seq": gb.seq,
                "alive": g_age < ALIVE_MS,
                "armed": armed,  # None: this build cannot tell (the core's heartbeat field, session 6)
                "active": owner_name == "r1guard",
                # the guard's own record of what it took over from (ABI 6); else the core's last mode
                "taken_over_from": (
                    (
                        self.port.tables.mode_name(int(ginfo["mode_from"]))
                        if ginfo and "mode_from" in ginfo
                        else core_mode
                    )
                    if owner_name == "r1guard"
                    else None
                ),
                "damp": None if ginfo is None else bool(ginfo.get("damp", False)),
            }
        out["alarms"] = sorted(
            {str(f).upper() for f in (out["core"] or {}).get("faults", [])} & ALARM_FAULTS
        )  # the faults an operator must see in words (the full lists stay in core / guard "faults")
        if core_mode != "stand_freeze":
            self.freeze_from_stand = False
        out["freeze_from_stand"] = self.freeze_from_stand if core_mode == "stand_freeze" else None
        out.update(self._body_arms())  # session 9: what the compat adapter / clients read (IMU, hands, the policy)
        # session 10: each agent adds its status keys between its markers
        # >>> s10:cam
        if getattr(self, "_perception", None) is not None:
            out["camera"] = self._perception.status()  # type: ignore[attr-defined]
        # <<< s10:cam
        # >>> s10:measure
        if (mops := getattr(self, "_measure_ops", None)) is not None:
            out["measure"] = mops.status()
        # <<< s10:measure
        # >>> s10:grasp
        if self.hands is not None:  # session 10: the hands' calibration references (hand.calibrate)
            out["hand_calibration"] = self.l2.hand_calibration()
        # <<< s10:grasp
        # >>> s10:core
        # <<< s10:core
        # >>> s10:stale
        from .state_age import state_age_status  # the STATE_OLD alarm + episodes + a stale freeze (D34)

        out["state_age"] = state_age_status(self.port, None if out["core"] is None else out["core"]["faults"])
        # <<< s10:stale
        # >>> s10:api
        if getattr(self, "_leases", None):  # session 10: the live leases (leases.py; absent until one was taken)
            from . import leases

            out["leases"] = [le.view(now) for le in leases.live(self, now)]
        # <<< s10:api
        # >>> s11:core
        from . import balance  # session 11: a freeze of a balancing robot names its way back (`r1 balance`)

        balance.status_hint(out, core_mode)
        # <<< s11:core
        # >>> s11:grip
        # <<< s11:grip
        # >>> s11:cam
        if "camera" not in out:  # session 11: the calibration r1d WILL use, before the first camera word
            from .perception import calibration_status

            out["camera"] = calibration_status()
        # <<< s11:cam
        # >>> s11:api
        # <<< s11:api
        # >>> s11:episode
        # <<< s11:episode
        # >>> s12:approach
        # <<< s12:approach
        # >>> s12:grip
        # <<< s12:grip
        # >>> s12:cam
        # <<< s12:cam
        # >>> s12:fixes
        # <<< s12:fixes
        # >>> s13:tools
        # <<< s13:tools
        # >>> s13:click
        # <<< s13:click
        # >>> s13:setup
        # <<< s13:setup
        # >>> s13:pose
        # <<< s13:pose
        # >>> s13:grip
        if self.hands is not None:  # session 13: each hand's heat before the grip's 72 C limit (r1mc3.hands.heat)
            from . import hand_heat

            out["hands_heat"] = hand_heat.status(self.l2, now)
        # <<< s13:grip
        # >>> s13:core
        from . import stop_latch  # session 13 (HW2 day 2's F12): the core's latched stop holds the lean at 0

        stop_latch.status(out, self.port, self.l2._lean_ended)
        # <<< s13:core
        # >>> s13:ops
        if (bat := getattr(self, "battery", None)) is not None:  # session 13: the R1's battery (read-only)
            out["battery"] = bat.status()  # r1mc3.battery.BatteryReader: reading, drain, minutes to the swap, words
        # <<< s13:ops
        out["summary"] = _status_words(out)
        out["summary"].extend((out.get("battery") or {}).get("words", []))  # session 13 (ops): the battery, in words
        from .hand_heat import words as hand_heat_words  # session 13 (grip): a warm hand in words, before the limit

        out["summary"].extend(hand_heat_words(out.get("hands_heat") or {}))
        if self.hands is not None:
            out["hands"] = {s: _hand_summary(self.hands.state(s)) for s in ("left", "right")}
            for side, h in out["hands"].items():
                if h.get("invalid_reason") == "frozen":
                    out["summary"].append(
                        f"hands: {side} state FROZEN (unchanged {h.get('unchanged_ms') or 0:.0f} ms): not valid, "
                        "recordings abort (dex3_server / the hand's link?)"
                    )
                elif h.get("valid") is False:
                    out["summary"].append(f"hands: {side} state not valid ({h.get('invalid_reason')})")
        if self.contact is not None:
            out["hands_contact"] = {**self.contact.last, "writes": self.contact.writes, "errors": self.contact_errors}
            # session 12 (grip): why the pads' contact is not valid yet (no tare until both hands' pads stream with the
            # hand open: contact-guarded moves stop at once until then) - in the operator's summary, not only the dict
            out["summary"].extend(f"pads: {w}" for w in self.contact.last.get("words", []))
        return out

    def _body_arms(self) -> dict[str, Any]:
        """`body` (the IMU roll / pitch / yaw and tilt in degrees, the lean commanded by body.pose and the one the core
        uses), `arms` (each hand's grasp centre + fingertips in the base frame, by the built-in arm model's FK of the
        measured state) and `policy` (the policy directory's name)."""
        from r1mc3.policy import POLICY_DIR

        from .goals import POINT_GRASP_CENTRE, hand_points, port_state, tips_dict

        st = port_state(self.port)
        out: dict[str, Any] = {"policy": POLICY_DIR.name, "body": None, "arms": None}
        if st is None:
            return out
        rpy = st.get("imu_rpy")
        lean_now = None
        mirror = getattr(getattr(self.port, "view", None), "mirror", None)
        if callable(mirror):
            try:
                m = mirror()
                lean_now = None if m is None else float(np.degrees(float(m["base_cmd"][3])))
            except Exception:
                lean_now = None
        out["body"] = {
            "rpy_deg": None if rpy is None else [round(float(np.degrees(x)), 3) for x in rpy],
            "tilt_deg": None if st.get("tilt_rad") is None else round(float(np.degrees(st["tilt_rad"])), 3),
            "lean_deg": [
                None if self.l2._lean is None else round(float(np.degrees(self.l2._lean)), 3),
                None if lean_now is None else round(lean_now, 3),
            ],
        }
        pts = hand_points(self.port, self.hands, st)
        if pts is not None:
            out["arms"] = {
                side: {
                    "hand_point": [round(float(x), 5) for x in pts[k, POINT_GRASP_CENTRE]],
                    "fingertips_m": tips_dict(pts[k]),
                    "approach": None,
                }
                for k, side in enumerate(("left", "right"))
            }
        return out

    # ------------------------------------------------------------------ stops and simple mode requests
    def _requester_id(self, who: str) -> int:
        rid = self.port.tables.requester(who)
        if rid is not None:
            return rid
        # an ABI-4 binding has no requester table: the stop channel only records who asked (the core's own id is
        # R1RT_OWNER_R1RT = 1); 3 / 4 are r1d's / the operator's until the binding says otherwise
        return 4 if who == "operator" else 3

    def _stop(self, r: StopReq) -> Result:
        if r.op == "damp" and r.requester != "operator":
            return fail(r.op, r.id, Code.NOT_PERMITTED, reason="damp is accepted from an operator client only")
        level = STOP_LEVELS[r.op]
        now = self.port.now_ns()
        self.port.request_stop(level, self._requester_id(r.requester), now)
        self._jogs.clear()  # a stop ends r1d's own streams too (their records would be refused anyway)
        self.l2.on_stop(r.op, now)  # session 9: and every goal (aborted, stopped_by OPERATOR_STOP)
        if self.recorder is not None:
            self.recorder.abort("stop")
        return ok(r.op, r.id, level=r.op, t_ns=now, stop_epoch=self._epoch_or_none())

    def _epoch_or_none(self) -> int | None:
        try:
            return self.port.stop_epoch()
        except NotImplementedError:
            return None

    def _mode_request_now(self, op: str, rid: Any, name: str, who: str, t: Ticket) -> Result | None:
        """A one-shot mode request (resume, freeze) through the mailbox; answered in tick() when the core replies
        (the server waits for it in `handle`; lockstep: by the next ticks)."""
        code = self.port.tables.requests.get(name)
        if code is None:
            return fail(op, rid, Code.UNSUPPORTED, reason=f"the core has no request '{name}'")
        now = self.port.now_ns()
        seq = self.port.request_mode(code, self._requester_id(who), now)
        self._seqs.append(
            _Sequence(
                ticket=t,
                req=TakeoverReq.model_validate({"op": "takeover", "id": rid}),
                first=name,
                stage="one",
                seq=seq,
                t0_ns=now,
            )
        )
        return None

    # ------------------------------------------------------------------ reclaim (T6)
    def _reclaim(self, r: ReclaimReq, t: Ticket) -> Result | None:
        """OPERATOR ONLY: the core's `reclaim` through the mailbox - r1rt takes the token back from r1guard (the core
        decides: only after a guard takeover from ON_STAND, r1rt alive and in ON_STAND). r1d checks the obvious first
        so the refusal says it in plain words; the core's own refusal is mapped by name."""
        if r.requester != "operator":
            return fail(r.op, r.id, Code.NOT_PERMITTED, reason="reclaim is the operator's word (DESIGN §4.4)")
        code = self.port.tables.requests.get("reclaim")
        if code is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="the core has no 'reclaim' request (core, session 6)")
        owner, _ = self.port.token()
        if OWNERS.get(owner) != "r1guard":
            return fail(
                r.op,
                r.id,
                Code.WRONG_MODE,
                token=OWNERS.get(owner, str(owner)),
                reason="the guard does not hold the motors: nothing to reclaim",
            )
        hb = self.port.heartbeat()
        age = None if hb is None else (self.port.now_ns() - hb.t_ns) * 1e-6
        if age is None or age > ALIVE_MS:
            return fail(
                r.op,
                r.id,
                Code.FAULT,
                core_age_ms=age,
                reason="r1rt is not running (its heartbeat is stale): it cannot take the motors back - the guard "
                "keeps holding (robot shell: is r1rt stopped? kill -CONT $(pidof r1rt))",
            )
        return self._mode_request_now(r.op, r.id, "reclaim", r.requester, t)

    # ------------------------------------------------------------------ the stand's freeze (S10)
    def _rt_config(self, *keys: str) -> dict[str, list[float]]:
        """Values from r1rt's config file (`--rt-config` / R1RT_CONFIG: the file r1rt itself reads), {} if not given."""
        if self.rt_config_path is None or not self.rt_config_path.exists():
            return {}
        out: dict[str, list[float]] = {}
        for line in self.rt_config_path.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() in keys:
                try:
                    out[k.strip()] = [float(x) for x in v.split()]
                except ValueError:
                    continue
        return out

    def _stand_freeze(self, r: StandFreezeReq, t: Ticket) -> Result | None:
        """OPERATOR ONLY: the core's `stand_freeze` from ON_STAND (S10). The dry run shows the ramp time (r1rt's config)
        and the far check (every leg / waist target within 0.15 rad of the stand pose) against the core's targets."""
        if r.requester != "operator":
            return fail(r.op, r.id, Code.NOT_PERMITTED, reason="the stand's freeze is the operator's word (S10, U3)")
        mode = self.mode_name()
        if mode != "on_stand":
            return fail(r.op, r.id, Code.WRONG_MODE, mode=mode, needs="on_stand (the robot hanging on its stand)")
        cfg = self._rt_config("stand_ramp_s", "stand_q")
        ramp = cfg["stand_ramp_s"][0] if cfg.get("stand_ramp_s") else None
        tgt = self.port.targets()
        check: dict[str, Any] = {"max_dev_rad": STAND_FREEZE_MAX_DEV_RAD}
        if cfg.get("stand_q") and tgt is not None:
            from .codes import JOINT_NAMES

            stand = np.asarray(cfg["stand_q"], dtype=np.float64)
            dev = np.abs(np.asarray(tgt, dtype=np.float64)[list(LOWER_JOINTS)] - stand[list(LOWER_JOINTS)])
            k = int(np.argmax(dev))
            check.update(
                worst_joint=JOINT_NAMES[LOWER_JOINTS[k]],
                worst_dev_rad=round(float(dev[k]), 4),
                ok=bool(dev.max() <= STAND_FREEZE_MAX_DEV_RAD),
            )
        else:
            check.update(ok=None, why="r1rt's config not given to r1d (--rt-config / R1RT_CONFIG): the core checks")
        plan = {
            "ramp_s": ramp if ramp is not None else "unknown (r1rt's config not given to r1d)",
            "what": "legs + waist ramp linearly to the stand pose at the stand gains, then hold; arms / head HOLD",
            "moves": "both legs and the waist, by up to 0.15 rad, in the ramp time",
            "how_to_stop": "it ends by itself (hold); `r1 stand-hold` back to the ON_STAND hold; `r1 damp` is yours",
            "after": "`r1 stand-hold`, then `r1 resume` (the freeze posts a STOP)",
        }
        if r.dry_run:
            return ok(r.op, r.id, dry_run=True, plan=plan, far_check=check)
        if check.get("ok") is False:
            return fail(r.op, r.id, Code.LIMIT, reason=REFUSAL_WORDS["far"], far_check=check, plan=plan)
        res = self._mode_request_now(r.op, r.id, "stand_freeze", r.requester, t)
        if res is None:
            self._seqs[-1].first = "stand_freeze_onstand"
        return res

    # ------------------------------------------------------------------ takeover / on_stand
    def _start_sequence(self, r: TakeoverReq, t: Ticket) -> Result | None:
        if any(s.stage not in ("done",) and s.first in ("takeover", "on_stand") for s in self._seqs):
            return fail(r.op, r.id, Code.RESOURCE_BUSY, reason="a takeover / on_stand is already running")
        name = r.op
        code = self.port.tables.requests.get(name)
        if code is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason=f"the core has no request '{name}'")
        if self.releaser is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="no motion-switcher client (releaser) in this r1d")
        mode = self.mode_name()
        if mode != "monitor":
            return fail(r.op, r.id, Code.WRONG_MODE, mode=mode, needs="monitor")
        if name == "takeover" and self.stand_session() is True:
            # refused HERE, before the mailbox and before any ReleaseMode: Unitree's controller keeps the robot
            return fail(r.op, r.id, Code.WRONG_MODE, reason=STAND_SESSION_REASON, refusal_name="stand_session")
        now = self.port.now_ns()
        seq = self.port.request_mode(code, self._requester_id(r.requester), now)
        self._seqs.append(_Sequence(ticket=t, req=r, first=name, seq=seq, t0_ns=now))
        return None  # answered by tick()

    def stand_session(self) -> bool | None:
        """r1rt's `--stand-session` (the core refuses every entry into BALANCE for its lifetime); None = unknown."""
        fn = getattr(self.port, "rt_info", None)
        info = fn() if callable(fn) else None
        if not info or "stand_session" not in info:
            return None
        return bool(info["stand_session"])

    def stop_counts(self) -> dict[str, Any]:
        """HW1 fix #31: r1rt's skipped_deadlines / lateness included the deliberate SIGSTOP windows (a board-hold test +
        2 guard-takeover tests = 3306 ticks, max 3.0 s), so a real overrun would have been lost in them. The core now
        counts its long silences (resume events) apart: `stops` and the longest `stop_ms_max` from the heartbeat
        (rt_info); None = this binding / r1rt does not report them."""
        fn = getattr(self.port, "rt_info", None)
        info = fn() if callable(fn) else None
        out: dict[str, Any] = {"stops": None, "stop_ms_max": None}
        if info:
            for key in out:
                if key in info and info[key] is not None:
                    out[key] = info[key]
        return out

    def _advance(self, s: _Sequence, now: int) -> None:
        tb = self.port.tables
        if s.stage == "one":  # resume / freeze / reclaim: just the reply
            rep = self.port.mode_reply(s.seq)
            if rep is not None:
                st, refusal, mode_after = rep
                op1, rid1 = s.ticket.request["op"], s.ticket.request.get("id")
                if st == R1RT_OK:
                    extra: dict[str, Any] = {}
                    if s.first == "reclaim":
                        o, g = self.port.token()
                        extra = {"token": {"owner": OWNERS.get(o, str(o)), "generation": g}}
                    if s.first == "stand_freeze_onstand":
                        self.freeze_from_stand = True
                        extra = {
                            "moved": True,
                            "next": "the legs + waist hold the stand pose; `r1 stand-hold` back to the ON_STAND hold, "
                            "then `r1 resume` (the freeze posted a STOP)",
                        }
                    if s.first == "stand_hold":
                        self.freeze_from_stand = False
                        extra = {"next": "`r1 resume` clears the freeze's STOP (nothing moves by itself)"}
                    moved = bool(extra.pop("moved", False))
                    res = ok(op1, rid1, moved=moved, mode=tb.mode_name(mode_after), **extra)
                else:
                    rname = tb.refusal_name(refusal)
                    res = fail(
                        op1,
                        rid1,
                        _refusal_code(tb.status_name(st), rname),
                        status=st,
                        status_name=tb.status_name(st),
                        refusal=refusal,
                        refusal_name=rname,
                        reason=REFUSAL_WORDS.get(rname or "", "the core refused"),
                        mode=tb.mode_name(mode_after),
                    )
                s.stage = "done"
                s.ticket.finish(res)
            elif now - s.t0_ns > 1_000_000_000:
                s.stage = "done"
                s.ticket.finish(fail(s.ticket.request["op"], s.ticket.request.get("id"), Code.TIMEOUT))
            return
        op, rid = s.req.op, s.req.id
        if s.stage == "gates":
            rep = self.port.mode_reply(s.seq)
            if rep is None:
                if now - s.t0_ns > s.req.reply_timeout_s * 1e9:
                    s.stage = "done"
                    s.ticket.finish(fail(op, rid, Code.TIMEOUT, stage="gates", waited_s=s.req.reply_timeout_s))
                return
            st, refusal, mode_after = rep
            s.log.append({"stage": "gates", "status": st, "refusal": refusal, "mode": tb.mode_name(mode_after)})
            if st != R1RT_OK:
                s.stage = "done"
                rname = tb.refusal_name(refusal)
                if rname == "stand_session":  # the core's own refusal (a binding without rt_info: r1d could not tell)
                    s.ticket.finish(
                        fail(op, rid, Code.WRONG_MODE, stage="gates", refusal=refusal, refusal_name=rname,
                             reason=STAND_SESSION_REASON, status=st, log=s.log)
                    )  # fmt: skip
                    return
                s.ticket.finish(
                    fail(op, rid, Code.CONTROLLER_REJECTED, stage="gates", refusal=refusal, refusal_name=rname,
                         status=st, log=s.log)
                )  # fmt: skip
                return
            s.stage = "release"
            assert self.releaser is not None
            if getattr(self.releaser, "synchronous", False):
                s.release = self.releaser.release(s.req.release_timeout_s)
            else:
                s.future = self._pool.submit(self.releaser.release, s.req.release_timeout_s)
            s.t0_ns = now
        if s.stage == "release":
            if s.release is None and s.future is not None and s.future.done():
                try:
                    s.release = s.future.result()
                except Exception as e:  # the RPC failed: the built-in may still drive the motors -> ABORT
                    s.release = (False, {"error": repr(e)})
            if s.release is None:
                if now - s.t0_ns > (s.req.release_timeout_s + 1.0) * 1e9:
                    s.release = (False, {"error": "release timed out"})
                else:
                    return
            released, info = s.release
            s.log.append({"stage": "release", "released": released, **info})
            name = "takeover_released" if released else "takeover_abort"
            s.seq = self.port.request_mode(tb.requests[name], self._requester_id(s.req.requester), now)
            s.stage = "decide"
            s.t0_ns = now
            return
        if s.stage == "decide":
            rep = self.port.mode_reply(s.seq)
            if rep is None:
                if now - s.t0_ns > s.req.reply_timeout_s * 1e9:
                    s.stage = "done"
                    s.ticket.finish(fail(op, rid, Code.TIMEOUT, stage="decide", log=s.log))
                return
            st, refusal, mode_after = rep
            s.log.append({"stage": "decide", "status": st, "refusal": refusal, "mode": tb.mode_name(mode_after)})
            s.stage = "done"
            assert s.release is not None
            if not s.release[0]:
                s.ticket.finish(fail(op, rid, Code.RELEASE_FAILED, aborted=st == R1RT_OK, log=s.log))
            elif st != R1RT_OK:
                s.ticket.finish(fail(op, rid, Code.CONTROLLER_REJECTED, stage="decide", status=st, log=s.log))
            else:
                s.ticket.finish(ok(op, rid, moved=True, mode=tb.mode_name(mode_after), log=s.log))

    # ------------------------------------------------------------------ L0 / L1
    def _busy(self, op: str, rid: Any, group: str) -> Result | None:
        """Session 9: an L0 / L1 command on a limb an L2 goal holds is refused (RESOURCE_BUSY: cancel the goal first,
        or `arm.stop`); `stop` is always accepted."""
        g = self.l2.holder(group)
        if g is None:
            return None
        return fail(op, rid, Code.RESOURCE_BUSY, held_by={"goal_id": g.goal_id, "op": g.op})

    def _group_check(self, op: str, rid: Any, group: str) -> tuple[int, str] | Result:
        mode = self.mode_name()
        if group in LOWER_GROUPS:
            if mode != "on_stand":
                return fail(op, rid, Code.WRONG_MODE, mode=mode, needs="on_stand", group=group)
            gid = self.port.tables.groups.get(group)
            if gid is None:
                return fail(op, rid, Code.UNSUPPORTED, reason=f"the core has no '{group}' group (ON_STAND L0)")
            return gid, mode
        if mode not in EXEC_MODES:
            return fail(op, rid, Code.WRONG_MODE, mode=mode, needs=list(EXEC_MODES), group=group)
        return self.port.tables.groups[group], mode

    def _check_limits(self, group: str, q: FloatArray) -> dict[str, Any] | None:
        if self.limits is None:
            return None
        from .codes import JOINT_NAMES

        for j, v in zip(GROUP_JOINTS[group], q, strict=True):
            lo, hi = self.limits[JOINT_NAMES[j]]
            if not lo + LIMIT_MARGIN <= v <= hi - LIMIT_MARGIN:
                return {"joint": JOINT_NAMES[j], "q": float(v), "min": lo + LIMIT_MARGIN, "max": hi - LIMIT_MARGIN}
        return None

    def new_cmd_id(self) -> int:
        cid = self._next_id
        self._next_id += 1
        return cid

    def _push(self, group: str, gid: int, source: str, rec: dict[str, Any]) -> int:
        rec = {"group": gid, "source": SOURCES[source], "n_joints": len(GROUP_JOINTS[group]), **rec}
        if "stop_epoch" not in rec:
            rec["stop_epoch"] = self.port.stop_epoch()
        rc = self.port.push_upper(rec)
        self.pushed.append((int(rec.get("t_issue_ns", 0)), group, SOURCES[source], rc))
        return rc

    def _vec(self, v: list[float] | None, n: int, name: str) -> FloatArray | None:
        if v is None:
            return None
        a = np.asarray(v, dtype=np.float64)
        if a.shape != (n,):
            raise ValueError(f"{name}: {n} values expected, got {a.size}")
        return a

    def _joint_command(self, r: JointCommandReq) -> Result:
        busy = self._busy(r.op, r.id, r.group)
        if busy is not None:
            return busy
        chk = self._group_check(r.op, r.id, r.group)
        if isinstance(chk, Result):
            return chk
        gid, _ = chk
        n = len(GROUP_JOINTS[r.group])
        try:
            q = self._vec(r.q, n, "q")
            vecs = {k: self._vec(getattr(r, k), n, k) for k in ("dq", "kp", "kd", "tau_ff")}
        except ValueError as e:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason=str(e))
        assert q is not None
        lim = self._check_limits(r.group, q)
        if lim is not None:
            return fail(r.op, r.id, Code.LIMIT, **lim)
        now = self.port.now_ns()
        self._jogs.pop(r.group, None)
        cid = self.new_cmd_id()
        rec: dict[str, Any] = {"cmd_id": cid, "t_issue_ns": now, "ttl_ns": round(r.ttl * 1e9), "q": q.tolist()}
        for k, v in vecs.items():
            if v is not None:
                rec[k] = v.tolist()
        if "dq" not in rec:
            rec["dq"] = [0.0] * n
        rc = self._push(r.group, gid, "stream", rec)
        if rc != R1RT_OK:
            return fail(r.op, r.id, Code.RESOURCE_BUSY, reason="upper_cmd ring full", rc=rc)
        self._last_set[r.group] = q.copy()
        return ok(r.op, r.id, moved=True, cmd_id=cid, t_issue_ns=now, ttl_s=r.ttl)

    def _joint_hold(self, r: JointHoldReq) -> Result:
        busy = self._busy(r.op, r.id, r.group)
        if busy is not None:
            return busy
        chk = self._group_check(r.op, r.id, r.group)
        if isinstance(chk, Result):
            return chk
        gid, _ = chk
        n = len(GROUP_JOINTS[r.group])
        if r.q is not None:
            try:
                q = self._vec(r.q, n, "q")
            except ValueError as e:
                return fail(r.op, r.id, Code.INVALID_ARGS, reason=str(e))
        else:
            q = self._current(r.group)
            if q is None:
                return fail(r.op, r.id, Code.UNSUPPORTED, reason="the core's current targets are not readable")
        assert q is not None
        lim = self._check_limits(r.group, q)
        if lim is not None:
            return fail(r.op, r.id, Code.LIMIT, **lim)
        self._jogs.pop(r.group, None)
        cid = self.new_cmd_id()
        rc = self._push(r.group, gid, "hold", {"cmd_id": cid, "t_issue_ns": self.port.now_ns(), "q": q.tolist()})
        if rc != R1RT_OK:
            return fail(r.op, r.id, Code.RESOURCE_BUSY, reason="upper_cmd ring full", rc=rc)
        self._last_set[r.group] = q.copy()
        return ok(r.op, r.id, moved=True, cmd_id=cid, q=q.tolist())

    def _current(self, group: str) -> FloatArray | None:
        """Where a group's target is now: the core's last commanded q (else the measured q), else r1d's last."""
        for src in (self.port.targets(), self.port.measured()):
            if src is not None:
                return np.asarray(src, dtype=np.float64)[list(GROUP_JOINTS[group])].copy()
        last = self._last_set.get(group)
        return None if last is None else last.copy()

    def _jog(self, op: str, rid: Any, group: str, dq: FloatArray, ttl: float) -> Result:
        busy = self._busy(op, rid, group)
        if busy is not None:
            return busy
        chk = self._group_check(op, rid, group)
        if isinstance(chk, Result):
            return chk
        n = len(GROUP_JOINTS[group])
        if dq.shape != (n,) or not np.all(np.isfinite(dq)):
            return fail(op, rid, Code.INVALID_ARGS, reason=f"dq: {n} finite values expected")
        now = self.port.now_ns()
        jog = self._jogs.get(group)
        if jog is None:
            q0 = self._current(group)
            if q0 is None:
                return fail(op, rid, Code.UNSUPPORTED, reason="the core's current targets are not readable")
            jog = _Jog(group=group, cmd_id=self.new_cmd_id(), q_set=q0, dq=dq, until_ns=0, last_ns=now)
            self._jogs[group] = jog
        jog.dq = dq.copy()
        jog.until_ns = now + round(ttl * 1e9)
        return ok(op, rid, moved=True, cmd_id=jog.cmd_id, until_ns=jog.until_ns)

    def _tick_jog(self, jog: _Jog, now: int) -> None:
        gid = self.port.tables.groups.get(jog.group)
        if gid is None:
            self._jogs.pop(jog.group, None)
            return
        if now >= jog.until_ns:
            self._jogs.pop(jog.group, None)  # nothing more: the last record's TTL decelerates the group to HOLD
            return
        dt = max(0.0, (now - jog.last_ns) * 1e-9)
        jog.last_ns = now
        q = jog.q_set + jog.dq * dt
        if self.limits is not None:
            from .codes import JOINT_NAMES

            for k, j in enumerate(GROUP_JOINTS[jog.group]):
                lo, hi = self.limits[JOINT_NAMES[j]]
                q[k] = min(max(q[k], lo + LIMIT_MARGIN), hi - LIMIT_MARGIN)
        jog.q_set = q
        ttl = min(JOG_RECORD_TTL_S, (jog.until_ns - now) * 1e-9 + 2.0 / self.tick_hz)
        self._push(
            jog.group,
            gid,
            "stream",
            {
                "cmd_id": jog.cmd_id,
                "t_issue_ns": now,
                "ttl_ns": round(ttl * 1e9),
                "q": q.tolist(),
                "dq": jog.dq.tolist(),
            },
        )
        self._last_set[jog.group] = q.copy()

    def stream_setpoint(
        self,
        group: str,
        q: FloatArray,
        dq: FloatArray | None,
        ttl_s: float,
        cmd_id: int,
        now: int,
        gains: tuple[FloatArray, FloatArray] | None = None,
    ) -> int:
        """One STREAM setpoint for a running script (the recording tools); the group's mode checked by the caller.
        `gains` = (kp, kd) per joint; a joint with 0 / 0 runs at the group's hold gains (the executor's rule)."""
        gid = self.port.tables.groups[group]
        rec: dict[str, Any] = {"cmd_id": cmd_id, "t_issue_ns": now, "ttl_ns": round(ttl_s * 1e9), "q": q.tolist()}
        rec["dq"] = (dq if dq is not None else np.zeros_like(q)).tolist()
        if gains is not None:
            rec["kp"], rec["kd"] = gains[0].tolist(), gains[1].tolist()
        self._last_set[group] = q.copy()
        return self._push(group, gid, "stream", rec)

    # ------------------------------------------------------------------ hands
    def _camera(self, r: CameraReq) -> Result:
        """r1cam's status / the camera-services switch (r1cam refuses the switch unless operator + confirm)."""
        from r1mc3.record.runner import CameraClient

        req: dict[str, Any] = {"op": "status"}
        if r.op == "camera.services":
            req = {"op": "services", "action": r.action, "requester": r.requester, "confirm": r.confirm}
        try:
            out = (self.camera or CameraClient()).request(req)
        except OSError as e:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason=f"r1cam does not answer ({e})")
        if out.get("ok"):
            return ok(r.op, r.id, **dict(out.get("data", {})))
        try:
            code = Code(out.get("code", "FAULT"))
        except ValueError:
            code = Code.FAULT
        return fail(r.op, r.id, code, **dict(out.get("details", {})))

    def _hand(self, r: HandCommandReq | HandStateReq) -> Result:
        if self.hands is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="no hands in this r1d")
        if isinstance(r, HandStateReq):
            sides = [r.side] if r.side is not None else ["left", "right"]
            data: dict[str, Any] = {s: _jsonable(self.hands.state(s)) for s in sides}
            return ok(r.op, r.id, **data)
        self.l2.end_grip(r.side)  # session 11 (grip): the operator's hand command takes the hand from any grip
        try:
            out = self.hands.command(r.side, r.q, dq=r.dq, kp=r.kp, kd=r.kd, tau=r.tau, ttl_s=r.ttl)
        except ValueError as e:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason=str(e))
        if not out.get("ok", True):
            return fail(r.op, r.id, Code.HAND_FAULT, **{k: v for k, v in out.items() if k != "ok"})
        return ok(r.op, r.id, moved=True, **{k: v for k, v in out.items() if k != "ok"})

    # ------------------------------------------------------------------ tick
    def tick(self, now_ns: int | None = None) -> None:
        now = self.port.now_ns() if now_ns is None else now_ns
        with self._lock:
            for s in list(self._seqs):
                self._advance(s, now)
                if s.stage == "done":
                    self._seqs.remove(s)
            for jog in list(self._jogs.values()):
                self._tick_jog(jog, now)
            self.l2.tick(now)
            per = getattr(self, "_perception", None)  # cam's line (the pose-at-capture history), session 10
            if per is not None:
                per.tick(now)
            if self.recorder is not None:
                self.recorder.tick(now)
        if self.contact is not None:
            try:
                self.contact.tick(now)
            except Exception:  # a hands hiccup never stops the tick (the core sees `valid` age out instead)
                self.contact_errors += 1

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self.hw1 is not None:
            self.hw1.close()


REFUSAL_WORDS = {  # the core's refusals (r1rt_py.refusals() names) in plain words for the operator
    "not_operator": "only the operator may do this",
    "mode": "the core is not in the mode this needs (reclaim: ON_STAND only - a balancing hand-back is not built)",
    "reclaim_token": "the guard does not hold the motors: nothing to reclaim",
    "reclaim_state": "no fresh lowstate or an executor not ready in r1rt: the guard keeps holding; try again",
    "guard": "no live r1guard",
    "far": "a leg / waist target is more than 0.15 rad from the stand pose: move closer first",
    "state": "no fresh lowstate in r1rt",
}


def _refusal_code(status_name: str, refusal_name: str | None) -> Code:
    """The core's refusal -> a DESIGN §5 code (by the binding's names, never by number)."""
    if refusal_name == "not_operator":
        return Code.NOT_PERMITTED
    if refusal_name in ("mode", "reclaim_token"):
        return Code.WRONG_MODE
    if status_name == "not_owner" or refusal_name == "token":
        return Code.NOT_OWNER
    if refusal_name == "far":
        return Code.LIMIT
    return Code.CONTROLLER_REJECTED


def _status_words(st: Mapping[str, Any]) -> list[str]:
    """`r1 status` in plain words (one line per process; the numbers stay in the other keys)."""
    owner = st["token"]["owner"]
    core, guard = st.get("core"), st.get("guard")
    lines: list[str] = []
    if core is not None and core["mode"] == "stand_freeze" and st.get("freeze_from_stand"):
        lines.append(
            "r1rt: STAND_FREEZE (from the stand): legs + waist held at the stand pose; `r1 stand-hold` returns to the "
            "ON_STAND hold (balance is never entered from here)"
        )
    if core is None:
        lines.append("r1rt: no heartbeat (not running?)")
    else:
        mode = str(core["mode"]).upper()
        if not core["alive"]:
            lines.append(f"r1rt: {mode}, heartbeat {core['age_ms']:.0f} ms old - STOPPED or stalled")
        elif owner == "r1rt":
            lines.append(f"r1rt: {mode}, owns the motors (publishes rt/lowcmd)")
        elif owner == "r1guard":
            # session 11 (floor's F13): `r1 reclaim` is ON_STAND only - after a takeover from BALANCE the words must not
            # send the operator to a refused word at the moment they need the right one
            back = (
                "r1 reclaim gives them back"
                if str(core["mode"]).lower() == "on_stand"
                else "no hand-back from balancing: you hold the robot, then r1 damp (the standard end)"
            )
            lines.append(f"r1rt: {mode}, running but SILENT - the guard holds the motors ({back})")
        else:
            lines.append(f"r1rt: {mode}, publishes nothing (Unitree's controller in charge)")
    if guard is None:
        lines.append("r1guard: ABSENT (no heartbeat): takeover / on-stand refused; GUARD_ABSENT while we own motors")
    elif not guard["alive"]:
        lines.append(f"r1guard: heartbeat {guard['age_ms']:.0f} ms old - STOPPED (it will not take over)")
    elif guard["active"]:
        lines.append(f"r1guard: ACTIVE - holds the motors (took over from {str(guard['taken_over_from']).upper()})")
    elif guard["armed"] is True:
        lines.append("r1guard: alive, ARMED (takes over ~10 ms after r1rt stops)")
    elif guard["armed"] is False:
        lines.append("r1guard: alive, OBSERVE-ONLY (will NOT take over)")
    else:
        lines.append("r1guard: alive (armed or observe-only: not reported by this build)")
    if core is not None and core.get("stand_session"):
        lines.append("stand session: balance REFUSED (r1rt --stand-session)")
    if core is not None and core.get("stops"):
        mx = core.get("stop_ms_max")
        lines.append(
            f"r1rt: {int(core['stops'])} long silence(s) since its start (stopped / stalled, e.g. the board-hold / "
            "guard-takeover tests; longest "
            + (f"{float(mx):.0f} ms" if mx is not None else "?")
            + ") - counted apart from skipped deadlines"
        )
    from .state_age import state_age_words  # session 10 (stale): the state-age alarm in words

    lines.extend(state_age_words(st.get("state_age"), None if core is None else str(core["mode"])))
    if st.get("alarms"):
        lines.append("alarms: " + ", ".join(st["alarms"]))
    # session 13 (core, HW2 day 2's F12): a stop stays latched in r1rt until `resume` - the lean is held at 0 meanwhile
    if (st.get("stop") or {}).get("latched"):
        lines.append("r1rt: " + str(st["stop"]["words"]))
    return lines


def _fault_names(word: int) -> list[str]:
    from r1mc3.log.records import fault_names

    return list(fault_names(int(word)))


def _jsonable(d: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        elif isinstance(v, Mapping):
            out[k] = _jsonable(v)
        else:
            out[k] = v
    return out


def _hand_summary(st: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("valid", "invalid_reason", "frozen", "unchanged_ms", "age_ms", "commanding", "writes")
    return {k: st.get(k) for k in keys if k in st}


__all__ = ["Releaser", "Service", "Ticket", "group_of_joint"]
