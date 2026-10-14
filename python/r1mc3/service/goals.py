"""r1d's L2 / L3 goals (DESIGN.md §5.1-§5.2): the goal table, the lifecycle, the runners, the measured results.

A goal op (`arm.move_joint`, `hand.grasp`, `skill.table_ready`, ...) is accepted or refused at once; an accepted goal
holds its RESOURCES (`arm_left`, `arm_right`, `head`, `hand_left`, `hand_right`, `base`, `body`) until it ends, and
a second goal on a held resource is refused (RESOURCE_BUSY) unless it asks to `preempt`. r1d's tick advances every
goal: accepted -> active -> succeeded | aborted | canceled | preempted | timed_out (v2's lifecycle, INV part B §1).
`stop` / `halt` / `damp` are always accepted and end every motion goal (`aborted`, stopped_by OPERATOR_STOP; the
core's executor decelerates the arms to HOLD by itself, EXECUTOR.md).

Arm motions are TRAJECTORY records (quintic Hermite knots, <= 8 per record, consecutive segments) with the deadman
keepalive renewed at every r1d tick (EXECUTOR.md: 50 ms); the core decides and reports how they end:

* with per-group status from the binding (`Port.upper_groups()`: the lockstep core's status today) a goal ends on the
  executor's own event for ITS cmd_id: DONE -> succeeded; a guard / the deadman / a starved path / a resume after a
  stop -> aborted (GUARD_STOP + the event's name); the operator's stop -> aborted (or canceled when r1d stopped it);
* without it (r1rt's segment does not publish the executor's per-group events yet - session 10, core) r1d compares the
  core's commanded target (the mirror's q_target) with its own plan: the plan's end reached and held -> succeeded; the
  target leaving the plan by more than PATH_DEVIATION_RAD -> aborted (stopped_by {"code": "STOPPED", "detail": ...}).

The measured result keys (§5.2) come from the core's measured state through the built-in arm model's forward
kinematics (`r1rt_py.fingertips_base`, ARM_MODEL.md: v2's base frame - the soles' midpoint, x along the feet, z up)
- the same numbers the executor's surface guard uses, never a planner's prediction.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .codes import GROUP_JOINTS, OWNERS, SOURCES, Code
from .schemas import GOAL_TERMINAL, Result

FloatArray = npt.NDArray[np.float64]
MAX_KNOTS = 8  # R1RT_MAX_KNOTS (types.h): knots per upper_cmd record
KEEPALIVE_TTL_NS = 50_000_000  # the TRAJECTORY deadman (EXECUTOR.md: 50 ms); r1d renews it at its 100 Hz tick
PATH_DEVIATION_RAD = 0.02  # without per-group events: the commanded target this far from r1d's plan = stopped
DONE_TOL_RAD = 1e-3  # ... and the plan's end reached when the commanded target is this close to the last knot
END_MARGIN_S = 0.3  # how long after the plan's end r1d waits for the core's DONE before it decides from the target
ACHIEVED_MM = 20.0  # `achieved` = the measured hand within this of the target (v2 reported error_mm, not a flag)
START_TOL_RAD = 0.05  # the executor's traj_start_tol_rad: the first knot within this of the current target
PEAK_SPEED_RAD_S = 1.0  # speed 1.0 = a 1 rad/s peak joint speed (v2 planned at <= 1 rad/s, EXECUTOR.md)
MIN_DURATION_S = 0.3
# executor events (executor.hpp ExecEvent) by name; r1sim.v3_core.EVENT_NAMES has the same table for the simulator
EVENT_NAMES = {
    1: "guard_lag",
    2: "guard_torque",
    3: "guard_contact",
    4: "guard_surface",
    5: "guard_tilt",
    6: "guard_contact_stale",
    16: "deadman",
    17: "stream_ttl",
    18: "operator_stop",
    19: "starved",
    20: "freedrive_end",
    21: "bad_state",
    22: "motor_error",
    23: "resume_after_silence",
    32: "done",
}
# executor refusals (executor.hpp ExecReject) by name
REJECT_NAMES = {
    1: "not_ready",
    2: "group",
    3: "source",
    4: "n_joints",
    5: "non_finite",
    6: "gains",
    7: "stop_epoch",
    8: "ttl",
    9: "knots",
    10: "segment",
    11: "too_long",
    12: "start",
    13: "torque_guard",
    14: "surface_guard",
    15: "contact_guard",
    16: "no_gravity_ff",
    17: "group_held",
}
EVENT_DONE = 32
EVENT_OPERATOR_STOP = 18
CONTACT_EVENT = 3
ARM_GROUPS = {"left": "arm_left", "right": "arm_right"}
HAND_RESOURCES = {"left": "hand_left", "right": "hand_right"}
SIDE_INDEX = {"left": 0, "right": 1}
POINT_GRASP_CENTRE = 4  # fingertips_base rows: palm, thumb tip, middle tip, index tip, grasp centre
POINT_PALM = 0


# ---------------------------------------------------------------------------------------------- trajectories
Knot = tuple[int, FloatArray, FloatArray, FloatArray]  # (t_ns from the start, q, dq, ddq)


def min_jerk_knots(q0: npt.ArrayLike, q1: npt.ArrayLike, duration_s: float, n_knots: int = MAX_KNOTS) -> list[Knot]:
    """Exact samples (q, dq, ddq) of the minimum-jerk move q0 -> q1 (q0 + d (10 s^3 - 15 s^4 + 6 s^5)): the
    executor's quintic Hermite between them reproduces the polynomial (r1sim.v3_core.min_jerk_knots, the same)."""
    a = np.asarray(q0, dtype=np.float64)
    d = np.asarray(q1, dtype=np.float64) - a
    out: list[Knot] = []
    for i in range(n_knots):
        s = i / (n_knots - 1)
        p = 10 * s**3 - 15 * s**4 + 6 * s**5
        v = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration_s
        acc = (60 * s - 180 * s**2 + 120 * s**3) / duration_s**2
        out.append((round(s * duration_s * 1e9), a + d * p, d * v, d * acc))
    return out


def min_jerk_duration(q0: npt.ArrayLike, q1: npt.ArrayLike, speed: float) -> float:
    """The duration whose minimum-jerk peak speed (1.875 |d| / T) is speed x PEAK_SPEED_RAD_S on the farthest joint."""
    d = float(np.max(np.abs(np.asarray(q1, dtype=np.float64) - np.asarray(q0, dtype=np.float64))))
    return max(MIN_DURATION_S, 1.875 * d / (speed * PEAK_SPEED_RAD_S))


def records_for(
    cmd_id: int, gid: int, n_joints: int, knots: Sequence[Knot], t_issue_ns: int, stop_epoch: int | None,
    guards: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """A TRAJECTORY as upper_cmd records of <= MAX_KNOTS knots (segment_index 0.., last_segment on the last)."""
    chunks = [knots[i : i + MAX_KNOTS] for i in range(0, len(knots), MAX_KNOTS)]
    out = []
    for k, ch in enumerate(chunks):
        rec: dict[str, Any] = {
            "cmd_id": cmd_id,
            "group": gid,
            "source": SOURCES["trajectory"],
            "n_joints": n_joints,
            "t_issue_ns": t_issue_ns,
            "ttl_ns": KEEPALIVE_TTL_NS,
            "segment_index": k,
            "last_segment": int(k == len(chunks) - 1),
            "knots": [
                (int(t), [float(x) for x in q], [float(x) for x in dq], [float(x) for x in a]) for t, q, dq, a in ch
            ],
        }
        if stop_epoch is not None:
            rec["stop_epoch"] = stop_epoch
        if guards:
            rec["guards"] = dict(guards)
        out.append(rec)
    return out


def sample_knots(knots: Sequence[Knot], t_ns: int) -> FloatArray:
    """The quintic Hermite through the knots at t (clamped to the ends) - the executor's interpolation (EXECUTOR.md)."""
    if t_ns <= knots[0][0]:
        return np.asarray(knots[0][1], dtype=np.float64).copy()
    if t_ns >= knots[-1][0]:
        return np.asarray(knots[-1][1], dtype=np.float64).copy()
    for (t0, q0, v0, a0), (t1, q1, v1, a1) in itertools.pairwise(knots):
        if t0 <= t_ns <= t1:
            h = (t1 - t0) * 1e-9
            s = (t_ns - t0) * 1e-9 / h
            h00 = 1 - 10 * s**3 + 15 * s**4 - 6 * s**5
            h10 = s - 6 * s**3 + 8 * s**4 - 3 * s**5
            h20 = 0.5 * s**2 - 1.5 * s**3 + 1.5 * s**4 - 0.5 * s**5
            h01 = 10 * s**3 - 15 * s**4 + 6 * s**5
            h11 = -4 * s**3 + 7 * s**4 - 3 * s**5
            h21 = 0.5 * s**3 - s**4 + 0.5 * s**5
            out: FloatArray = (
                h00 * np.asarray(q0)
                + h10 * h * np.asarray(v0)
                + h20 * h * h * np.asarray(a0)
                + h01 * np.asarray(q1)
                + h11 * h * np.asarray(v1)
                + h21 * h * h * np.asarray(a1)
            )
            return out
    return np.asarray(knots[-1][1], dtype=np.float64).copy()


# ---------------------------------------------------------------------------------------------- measurement
def hand_points(svc_port: Any, hands: Any, state: Mapping[str, Any] | None = None) -> FloatArray | None:
    """Both hands' points (2, 5, 3) [palm, thumb tip, middle tip, index tip, grasp centre] in the base frame from
    the measured state (the built-in arm model; the hands' measured q when r1hands is present, else the open hand).
    None when the binding has no arm model or no state is in."""
    mod = getattr(svc_port, "mod", None)
    fn = getattr(mod, "fingertips_base", None) if mod is not None else None
    st = state if state is not None else port_state(svc_port)
    if fn is None or st is None or "q" not in st or "imu_quat" not in st:
        return None
    q = np.asarray(st["q"], dtype=np.float64).reshape(1, 26)
    quat = np.asarray(st["imu_quat"], dtype=np.float64).reshape(1, 4)
    hq: list[FloatArray | None] = [None, None]
    if hands is not None:
        for k, side in enumerate(("left", "right")):
            try:
                hs = hands.state(side)
            except Exception:  # a hands hiccup: the open hand
                continue
            v = hs.get("q") if hs.get("valid", True) else None
            if v is not None and np.asarray(v).size == 7 and np.all(np.isfinite(v)):
                hq[k] = np.asarray(v, dtype=np.float64).reshape(1, 7)
    try:
        pts, _low, _feet = fn(q, quat, hq[0], hq[1])
    except ValueError:
        return None
    out: FloatArray = np.asarray(pts, dtype=np.float64)[0]
    return out


def port_state(port: Any) -> dict[str, Any] | None:
    fn = getattr(port, "state", None)
    if callable(fn):
        try:
            st = fn()
        except Exception:
            return None
        return None if st is None else dict(st)
    return None


def tips_dict(hand: FloatArray) -> dict[str, list[float]]:
    """One hand's fingertips by name (base frame), from a (5, 3) row of hand_points (the planner's names)."""
    return {
        n: [round(float(x), 5) for x in hand[i]] for n, i in (("thumb_tip", 1), ("middle_tip", 2), ("index_tip", 3))
    }


def _r(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [round(float(x), 5) for x in v.ravel()] if v.ndim <= 1 else [_r(r) for r in v]
    return v


# ---------------------------------------------------------------------------------------------- the table
class Runner(Protocol):
    """Drives one goal: `ready` until its limb can start (after a preempted goal's brake), `start` once (a Result =
    refused at the start, nothing moved), `tick` until it returns a terminal state, `stop` when the goal is canceled /
    preempted / timed out (it must leave the limb holding), `end` when something outside ended it (an operator stop:
    the core already holds the limbs), `outcome` for the final result."""

    def ready(self, now_ns: int) -> bool: ...
    def start(self, now_ns: int) -> Result | None: ...
    def tick(self, now_ns: int) -> str | None: ...
    def stop(self, now_ns: int, why: str) -> None: ...
    def end(self, state: str, code: Code, stopped_by: dict[str, Any] | None) -> None: ...
    def outcome(self) -> tuple[Code, dict[str, Any], dict[str, Any]]: ...


@dataclass
class Goal:
    goal_id: int
    op: str
    rid: Any
    resources: frozenset[str]
    runner: Runner
    timeout_ns: int
    t_accept_ns: int
    state: str = "accepted"
    t_start_ns: int = 0
    t_end_ns: int = 0
    result: Result | None = None
    waiters: list[tuple[Any, int, str, Any]] = field(default_factory=list)  # (Ticket, deadline_ns, op, request id)
    parent: int | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_op": self.op,
            "state": self.state,
            "resources": sorted(self.resources),
            "t_accept_ns": self.t_accept_ns,
            "t_start_ns": self.t_start_ns,
            "t_end_ns": self.t_end_ns,
            "parent": self.parent,
        }


class GoalTable:
    """Every goal r1d accepted (the terminal ones kept for `goal.status` / `goal.wait`, the last KEPT_MAX of them)."""

    KEPT_MAX = 256

    def __init__(self) -> None:
        self.goals: dict[int, Goal] = {}
        self._ids = itertools.count(1)
        self.starting: dict[int, int | None] = {}  # goal id -> parent, while its runner's start() runs (L2.accept)

    def new_id(self) -> int:
        return next(self._ids)

    def active(self) -> list[Goal]:
        return [g for g in self.goals.values() if g.state not in GOAL_TERMINAL]

    def holding(self, resources: frozenset[str]) -> list[Goal]:
        return [g for g in self.active() if g.resources & resources]

    def ancestors(self, parent: int | None) -> set[int]:
        """`parent` and its parents up the chain (the goals that hold a child's resources for it)."""
        out: set[int] = set()
        while parent is not None and parent not in out:
            out.add(parent)
            p = self.goals.get(parent)
            parent = self.starting.get(parent) if p is None else p.parent  # (a goal still in its start: not added yet)
        return out

    def add(self, g: Goal) -> None:
        self.goals[g.goal_id] = g
        done = [k for k, v in self.goals.items() if v.state in GOAL_TERMINAL]
        for k in done[: max(0, len(done) - self.KEPT_MAX)]:
            del self.goals[k]


# ---------------------------------------------------------------------------------------------- arm trajectories
StopFn = Callable[[], None]


PlanFn = Callable[[FloatArray], "list[Knot] | Result"]


class TrajectoryRunner:
    """One TRAJECTORY of one upper group (arm or head): planned at its start from the group's current target
    (`plan(q_current) -> knots | Result`), pushed, kept alive at every tick, ended by the core.

    `target_point` is the requested hand point (base frame) for error_mm, else the forward kinematics of the last
    knot; `point_index` = which hand point (the grasp centre by default)."""

    def __init__(
        self,
        svc: Any,
        group: str,
        plan: PlanFn,
        *,
        side: str | None = None,
        target_point: FloatArray | None = None,
        point_index: int = POINT_GRASP_CENTRE,
        guards: Mapping[str, Any] | None = None,
        plan_id: str | None = None,
        planned_on: str = "r1d",
        extra: dict[str, Any] | None = None,
        contact_ends_ok: bool = False,
    ) -> None:
        self.svc = svc
        self.group = group
        self.plan = plan
        self.knots: list[Knot] = []
        self.side = side
        self.target_point = target_point
        self.point_index = point_index
        self.guards = guards
        self.plan_id = plan_id
        self.planned_on = planned_on
        self.extra = extra if extra is not None else {}  # (a plan function may fill it at the start)
        self.contact_ends_ok = contact_ends_ok  # arm.lower until contact: the contact guard IS the success
        self.cmd_id = 0
        self.gid = -1
        self.t0_ns = 0
        self.end_state: str | None = None
        self.stopped_by: dict[str, Any] | None = None
        self.code = Code.OK
        self.p_start: FloatArray | None = None
        self._why = ""
        self._prev: FloatArray | None = None
        self._rej0: tuple[int, int] | None = None
        self.refused = False

    @property
    def duration_ns(self) -> int:
        return int(self.knots[-1][0])

    def ready(self, now_ns: int) -> bool:
        """The group's commanded target is still (a preempted / stopped goal's brake has ended): two r1d ticks apart
        it moved less than 1e-4 rad (the executor's stop takes ~0.15 s)."""
        cur = self.svc._current(self.group)
        if cur is None:
            return True  # start() refuses with the reason
        prev, self._prev = self._prev, cur
        return prev is not None and float(np.max(np.abs(cur - prev))) < 1e-4

    def start(self, now_ns: int) -> Result | None:
        from .schemas import Result, fail

        port = self.svc.port
        self.gid = int(port.tables.groups[self.group])
        cur = self.svc._current(self.group)
        if cur is None:
            return fail("trajectory", None, Code.UNSUPPORTED, reason="the core's current targets are not readable")
        planned = self.plan(cur)
        if isinstance(planned, Result):
            return planned
        self.knots = list(planned)
        if float(np.max(np.abs(np.asarray(self.knots[0][1]) - cur))) > START_TOL_RAD:
            return fail(
                "trajectory",
                None,
                Code.CONTROLLER_REJECTED,
                reason="the path's first point is more than 0.05 rad from the group's current target",
                current=_r(cur),
                first=_r(np.asarray(self.knots[0][1])),
            )
        self.p_start = hand_points(port, self.svc.hands)
        self.cmd_id = self.svc.new_cmd_id()
        n = len(GROUP_JOINTS[self.group])
        epoch = self.svc._epoch_or_none()
        for rec in records_for(self.cmd_id, self.gid, n, self.knots, now_ns, epoch, self.guards):
            rc = port.push_upper(rec)
            if rc != 0:
                return fail("trajectory", None, Code.RESOURCE_BUSY, reason="upper_cmd ring full", rc=rc)
        port.upper_keepalive(self.gid, self.cmd_id, now_ns)
        self.t0_ns = now_ns
        self._rej0 = self._rejected()
        self.svc._last_set[self.group] = np.asarray(self.knots[-1][1], dtype=np.float64).copy()
        return None

    def _rejected(self) -> tuple[int, int] | None:
        fn = getattr(self.svc.port, "upper_groups", None)
        info = fn() if callable(fn) else None
        if info is None or "rejected" not in info:
            return None
        return int(info["rejected"][0]), int(info["last_reject"][0])

    def _group_info(self) -> tuple[int, int, int, int | None] | None:
        """(source, event, cmd_id, the core's publish time or None) of this goal's group; None = no per-group status
        (an old binding, or r1rt's status too old to describe now)."""
        fn = getattr(self.svc.port, "upper_groups", None)
        if not callable(fn):
            return None
        info = fn()
        if info is None or self.gid >= len(info["source"]):
            return None
        t_core = info.get("t_ns")
        return (
            int(info["source"][self.gid]),
            int(info["event"][self.gid]),
            int(info["cmd_id"][self.gid]),
            None if t_core is None else int(t_core[0]),
        )

    def _guard_holds(self) -> bool:
        """The ownership token is r1guard's (DESIGN §4.3: it took over from a silent r1rt). r1rt's executor no longer
        runs the path and its per-group status is frozen at the moment it stopped - the goal ends here, named."""
        try:
            owner, _ = self.svc.port.token()
        except Exception:
            return False
        return OWNERS.get(int(owner)) == "r1guard"

    def tick(self, now_ns: int) -> str | None:
        if self.end_state is not None:
            return self.end_state
        port = self.svc.port
        port.upper_keepalive(self.gid, self.cmd_id, now_ns)
        elapsed = now_ns - self.t0_ns
        if self._guard_holds():  # session 10: r1rt went silent, r1guard took the motors and holds them
            self._end(
                "aborted",
                Code.GUARD_STOP,
                {"code": "GUARD_TAKEOVER", "detail": "r1guard holds the motors (r1rt went silent); the arm holds"},
            )
            return self.end_state
        info = self._group_info()
        if info is not None:
            src, ev, cid, t_core = info
            # how long the CORE has had the record (r1rt's segment: its own publish time; lockstep: r1d's clock) - a
            # status read from before the push (r1rt a tick behind, or a load hiccup on a test machine) is not a
            # "replaced" or "left the trajectory" verdict
            seen = elapsed if t_core is None else t_core - self.t0_ns
            if cid != self.cmd_id:
                rej = self._rejected()
                if rej is not None and self._rej0 is not None and rej[0] > self._rej0[0]:  # the core refused it
                    self.refused = True
                    name = REJECT_NAMES.get(rej[1], f"reject_{rej[1]}")
                    self._end("aborted", Code.CONTROLLER_REJECTED, {"code": "REFUSED_BY_CORE", "detail": name})
                elif seen > 50_000_000:  # the core runs another command on this group now (an L0 stream, a stop)
                    self._end("aborted", Code.GUARD_STOP, {"code": "REPLACED", "detail": f"cmd_id {cid} on the group"})
                return self.end_state
            if src == SOURCES["trajectory"]:
                return None
            if ev == EVENT_DONE:
                self._end("succeeded", Code.OK, None)
            elif ev == CONTACT_EVENT and self.contact_ends_ok:
                self._end("succeeded", Code.OK, {"code": "GUARD_CONTACT", "detail": "contact (the goal's end)"})
            elif ev == EVENT_OPERATOR_STOP:
                self._end("aborted", Code.GUARD_STOP, {"code": "OPERATOR_STOP", "detail": "a stop request"})
            elif ev != 0:
                name = EVENT_NAMES.get(ev, f"event_{ev}")
                self._end("aborted", Code.GUARD_STOP, {"code": name.upper(), "detail": f"executor event {ev}"})
            elif seen > 50_000_000:
                self._end("aborted", Code.GUARD_STOP, {"code": "STOPPED", "detail": "the group left the trajectory"})
            return self.end_state
        # no per-group events (r1rt's segment): judge from the commanded target vs the plan
        tgt = self.svc._current(self.group)
        if tgt is None:
            return None
        plan_now = sample_knots(self.knots, elapsed - 4_000_000)  # the core runs a tick or two behind r1d's clock
        if elapsed >= self.duration_ns:
            if float(np.max(np.abs(tgt - np.asarray(self.knots[-1][1])))) <= DONE_TOL_RAD:
                self._end("succeeded", Code.OK, None)
            elif elapsed > self.duration_ns + END_MARGIN_S * 1e9:
                self._end("aborted", Code.GUARD_STOP, {"code": "STOPPED", "detail": "the path's end was not reached"})
        elif elapsed > 20_000_000 and float(np.max(np.abs(tgt - plan_now))) > PATH_DEVIATION_RAD:
            self._end(
                "aborted",
                Code.GUARD_STOP,
                {"code": "STOPPED", "detail": "the core stopped the path (this binding reports no per-group event)"},
            )
        return self.end_state

    def _end(self, state: str, code: Code, stopped_by: dict[str, Any] | None) -> None:
        self.end_state = state
        self.code = code
        self.stopped_by = stopped_by

    def end(self, state: str, code: Code, stopped_by: dict[str, Any] | None) -> None:
        if self.end_state is None:
            self._end(state, code, stopped_by)

    def stop(self, now_ns: int, why: str) -> None:
        """Cancel / preempt / timeout: a HOLD at the group's current target (the executor brakes to it, no jump)."""
        self._why = why
        if self.end_state is not None:
            return
        q = self.svc._current(self.group)
        if q is not None and self.gid >= 0:
            self.svc._push(self.group, self.gid, "hold", {"cmd_id": self.svc.new_cmd_id(), "t_issue_ns": now_ns,
                                                           "q": q.tolist()})  # fmt: skip
        self.end_state = why
        self.stopped_by = {"code": why.upper(), "detail": "r1d stopped the goal"}
        self.code = {"canceled": Code.CANCELED, "preempted": Code.PREEMPTED, "timed_out": Code.TIMEOUT}.get(
            why, Code.CANCELED
        )

    def outcome(self) -> tuple[Code, dict[str, Any], dict[str, Any]]:
        if not self.knots:  # never started (deferred, then stopped / refused): nothing moved
            return self.code, {"moved": False, "stopped_by": self.stopped_by}, {"stopped_by": self.stopped_by}
        pts = hand_points(self.svc.port, self.svc.hands)
        data: dict[str, Any] = {
            "moved": not self.refused,
            "stopped_by": self.stopped_by,
            "plan_id": self.plan_id,
            "planned_on": self.planned_on,
            "cmd_id": self.cmd_id,
            "target_q": _r(np.asarray(self.knots[-1][1])),
            **self.extra,
        }
        meas = port_state(self.svc.port)
        if meas is not None and "q" in meas:
            data["achieved_q"] = _r(np.asarray(meas["q"], dtype=np.float64)[list(GROUP_JOINTS[self.group])])
        if self.side is not None and pts is not None:
            k = SIDE_INDEX[self.side]
            target = self.target_point
            if target is None:
                st = dict(meas or {})
                if "q" in st:
                    qt = np.asarray(st["q"], dtype=np.float64).copy()
                    qt[list(GROUP_JOINTS[self.group])] = np.asarray(self.knots[-1][1])
                    tp = hand_points(self.svc.port, self.svc.hands, {**st, "q": qt})
                    target = None if tp is None else tp[k, self.point_index]
            now_pt = pts[k, self.point_index]
            data["hand_start"] = None if self.p_start is None else _r(self.p_start[k, self.point_index])
            data["hand_now"] = _r(now_pt)
            data["fingertips_m"] = tips_dict(pts[k])
            data["grasp_centre_m"] = _r(pts[k, POINT_GRASP_CENTRE])
            if target is not None:
                err = float(np.linalg.norm(now_pt - target)) * 1e3
                data["target"] = _r(np.asarray(target))
                data["error_mm"] = round(err, 2)
                data["achieved"] = bool(err <= ACHIEVED_MM and self.code == Code.OK)
        else:
            data.update({"hand_start": None, "hand_now": None, "fingertips_m": None, "grasp_centre_m": None})
            data.setdefault("target", None)
            data.setdefault("error_mm", None)
            data["achieved"] = self.code == Code.OK
        details = {} if self.code == Code.OK else {"stopped_by": self.stopped_by}
        return self.code, data, details
