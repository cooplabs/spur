"""The L2 / L3 ops of r1d (DESIGN.md §5.1): the goal machinery around goals.py's runners, op by op.

`Service._dispatch` hands every L2 / L3 / goal.* request here (`L2.dispatch`); `Service.tick` calls `L2.tick` at
100 Hz; a stop / halt / damp calls `L2.on_stop` (every goal ends `aborted`, stopped_by OPERATOR_STOP: the core holds
the limbs by itself). The planner (`r1mc3.plan`), the world model (`r1mc3.world`) and the grip state machine
(`r1mc3.hands.grip`) are injected (`Service(planner=..., world=..., grasp_factory=...)`); an op whose piece is absent
answers UNSUPPORTED with the reason, never a guess.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ..hands.trace import trace_dir
from . import stop_latch
from .codes import GROUP_JOINTS, Code
from .goals import (
    ARM_GROUPS,
    HAND_RESOURCES,
    POINT_GRASP_CENTRE,
    SIDE_INDEX,
    Goal,
    GoalTable,
    Knot,
    Runner,
    TrajectoryRunner,
    hand_points,
    min_jerk_duration,
    min_jerk_knots,
    tips_dict,
)
from .schemas import (
    GOAL_OPS,
    GOAL_TERMINAL,
    REQUEST,
    ArmLowerReq,
    ArmMoveJointReq,
    ArmPlanReq,
    ArmStopReq,
    BodyPoseReq,
    GoalCancelReq,
    GoalListReq,
    GoalStatusReq,
    GoalWaitReq,
    HandGraspReq,
    HandPreshapeReq,
    HandReleaseReq,
    HandTareReq,
    HeadLookAtReq,
    Result,
    SeqRunReq,
    SkillReq,
    WorldClearReq,
    WorldDeclareReq,
    WorldListReq,
    fail,
    ok,
)

if TYPE_CHECKING:
    from .service import Service, Ticket

DQ_MAX_RAD_S = 2.0  # the executor's dq_max on the arms (EXECUTOR.md, R1 defaults): a path above it is refused
RECENT_RELEASE_NS = 300_000_000  # a resource freed this recently: the next goal waits for the limb to be still
DEFER_MAX_NS = 1_500_000_000  # a deferred goal starts after this even if its limb never looked still (it may refuse)
HAND_MOVE_TOL_RAD = 0.1  # hand.release / preshape: done when every finger is this close to the target
HAND_MOVE_MAX_S = 1.5  # ... or after this (the result says whether it got there: a blocked finger is not an error)
HAND_HOLD_TTL_S = 5.0  # the last hand command's TTL (r1hands then holds where the fingers are, dex3.py)
GRIP_HOLDING = ("TENTATIVE", "HOLDING")
LAG_RAD = 0.035  # the lag guard's margin (v2 controller.py:325 guard_lag; the executor adds friction / kp + latency)
ALL_PADS = 0x1FF  # the contact guard counts any of the hand's 9 pads that rose after the path began (EXECUTOR.md)
PLAN_WAIT_S = 10.0  # a plan not back from the planner worker within this (wall clock): the goal aborts (TIMEOUT)
HEAD_PITCH_SOFT = (-0.578, 0.578)  # rad, head pitch's soft limits (joints.py SOFT_MIN / MAX; HW1 S8: no hard stop)
HEAD_YAW_MAX = 1.4  # rad (v2 skills.py look_at; the executor's head yaw limit is 1.5)
LEAN_TTL_NS = 500_000_000  # the base command's TTL (DESIGN §5.1: <= 0.5 s, expiry = zero)
PLANNED_KINDS = {
    "arm.reach": "reach",
    "arm.move_linear": "move_linear",
    "arm.move_relative": "move_relative",
    "arm.hover": "hover",
    "arm.lower": "lower",
    "arm.carry": "carry",
    "arm.move_joint": "move_joint",
}


class L2:
    def __init__(self, svc: Service) -> None:
        self.svc = svc
        self.table = GoalTable()
        self._released_ns: dict[str, int] = {}
        self.handlers: dict[str, Callable[[Any, Ticket], Result | None]] = {
            "goal.status": self._goal_status,
            "goal.cancel": self._goal_cancel,
            "goal.wait": self._goal_wait,
            "goal.list": self._goal_list,
            "arm.move_joint": self._arm_move_joint,
            "arm.stop": self._arm_stop,
            "arm.reach": self._arm_planned,
            "arm.move_linear": self._arm_planned,
            "arm.move_relative": self._arm_planned,
            "arm.hover": self._arm_planned,
            "arm.lower": self._arm_planned,
            "arm.carry": self._arm_planned,
            "arm.plan": self._arm_plan,
            "world.declare": self._world_declare,
            "world.clear": self._world_clear,
            "world.list": self._world_list,
            "head.look_at": self._head_look_at,
            "body.pose": self._body_pose,
            "seq.run": self._seq_run,
            "skill.ready": self._skill,
            "skill.table_ready": self._skill,
            "hand.grasp": self._hand_grasp,
            "hand.release": self._hand_release,
            "hand.preshape": self._hand_preshape,
            "hand.tare": self._hand_tare,
            # session 10: each agent registers its L2 / L3 handlers between its markers
            # >>> s10:cam
            # <<< s10:cam
            # >>> s10:measure
            # <<< s10:measure
            # >>> s10:grasp
            "hand.calibrate": self._hand_calibrate,
            # <<< s10:grasp
            # >>> s10:planner
            # <<< s10:planner
            # >>> s10:api
            # <<< s10:api
            # >>> s11:core
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
        }
        self._pending_plans: list[tuple[Any, Ticket, str, Any, int]] = []  # arm.plan dry runs on the worker
        self._parent_ctx: int | None = None  # set while a sequence submits its step (the child goal's parent)
        self._head_geo: Any = None  # a Geometry of the robot alone (head.look_at's camera position)
        self._lean: float | None = None  # body.pose: the policy's lean command r1d renews (rad), None = none
        self._lean_ended: str | None = None  # session 13: why the last lean ended (a stop), for `r1 status`
        self._base_seq = 0
        self.grips: dict[str, Any] = {}  # side -> the GraspRunner that holds (ticked until hand.release)
        self._grip_traces: dict[str, tuple[Any, Any]] = {}  # side -> (runner, its TracingHands): r1mc3.hands.trace
        self._grip_z0: dict[str, float] = {}  # side -> the grasp centre's height at the verdict (the lift check)

    # ------------------------------------------------------------------ dispatch
    def handles(self, op: str) -> bool:
        return op in self.handlers

    def dispatch(self, r: Any, t: Ticket) -> Result | None:
        out = self.handlers[r.op](r, t)
        if out is not None and not out.ok and r.op in GOAL_OPS and "state" not in out.data:
            out = out.model_copy(update={"data": {**out.data, "state": "rejected"}})  # refused before it was a goal
        return out

    def status(self) -> list[dict[str, Any]]:
        return [g.summary() for g in self.table.active()]

    # ------------------------------------------------------------------ the goal machinery
    def accept(
        self, r: Any, t: Ticket, resources: frozenset[str], runner: Runner, *, parent: int | None = None
    ) -> Result | None:
        """Start a goal now (or defer it until its limb is still), or refuse it. Returns the immediate reply (None =
        the ticket finishes later: `wait`)."""
        now = self.svc.port.now_ns()
        if parent is None:
            parent = self._parent_ctx
        # a goal's own ancestors (the sequence / skill / par that started it, and theirs) hold its resources for it
        # (session 10: a nested composite's grandchild was RESOURCE_BUSY by its grandparent)
        anc = self.table.ancestors(parent)
        busy = [g for g in self.table.holding(resources) if g.goal_id not in anc]
        if busy and not getattr(r, "preempt", False):
            return fail(
                r.op,
                r.id,
                Code.RESOURCE_BUSY,
                held_by=[
                    {"goal_id": g.goal_id, "op": g.op, "resources": sorted(g.resources & resources)} for g in busy
                ],
            )
        for g in busy:
            g.runner.stop(now, "preempted")
            self._finish(g, now)
        g = Goal(
            goal_id=self.table.new_id(),
            op=r.op,
            rid=r.id,
            resources=resources,
            runner=runner,
            timeout_ns=round(float(getattr(r, "timeout_s", 30.0)) * 1e9),
            t_accept_ns=now,
            parent=parent,
        )
        if hasattr(runner, "goal_id"):
            runner.goal_id = g.goal_id
        recent = any(now - self._released_ns.get(x, -(10**18)) < RECENT_RELEASE_NS for x in resources)
        if not recent:
            self.table.starting[g.goal_id] = parent  # its children (started inside start()) find their ancestors
            try:
                refused = runner.start(now)
            finally:
                self.table.starting.pop(g.goal_id, None)
            if refused is not None:
                return Result(ok=False, code=refused.code, op=r.op, id=r.id, moved=False, details=refused.details)
            g.state, g.t_start_ns = "active", now
        self.table.add(g)
        if getattr(r, "wait", False):
            g.waiters.append((t, now + g.timeout_ns + 1_000_000_000, r.op, r.id))
            return None
        return ok(r.op, r.id, moved=g.state == "active", goal_id=g.goal_id, state="accepted")

    def tick(self, now: int) -> None:
        self._tick_grips(now)
        if self._lean is not None:
            self._send_lean(now)
        for p in list(self._pending_plans):
            fut, tk, op, rid, t0 = p
            if fut.done() or now - t0 > PLAN_WAIT_S * 1e9:
                self._pending_plans.remove(p)
                tk.finish(_plan_reply(op, rid, fut.result() if fut.done() else None))
        for g in list(self.table.active()):
            if g.state == "accepted":
                anc = self.table.ancestors(g.parent)  # (session 10: a deferred child waited for its own parent)
                others = [o for o in self.table.holding(g.resources)
                          if o.goal_id != g.goal_id and o.goal_id not in anc and o.state == "active"]  # fmt: skip
                if others:
                    continue
                if not g.runner.ready(now) and now - g.t_accept_ns < DEFER_MAX_NS:
                    continue
                refused = g.runner.start(now)
                if refused is not None:
                    g.runner.end("aborted", refused.code, {"code": "REFUSED", "detail": refused.details})
                    self._finish(g, now)
                    continue
                g.state, g.t_start_ns = "active", now
                continue
            st = g.runner.tick(now)
            if st is None and now - g.t_start_ns > g.timeout_ns:
                g.runner.stop(now, "timed_out")
                st = "timed_out"
            if st is not None:
                self._finish(g, now)
        for g in self.table.active():
            for w in list(g.waiters):
                if now > w[1]:
                    g.waiters.remove(w)
                    w[0].finish(fail(w[2], w[3], Code.TIMEOUT, goal_id=g.goal_id, state=g.state))

    def _finish(self, g: Goal, now: int) -> None:
        code, data, details = g.runner.outcome()
        state = getattr(g.runner, "end_state", None) or ("succeeded" if code == Code.OK else "aborted")
        if state not in GOAL_TERMINAL:
            state = "succeeded" if code == Code.OK else "aborted"
        g.state = state
        g.t_end_ns = now
        data = {
            **data,
            "goal_id": g.goal_id,
            "state": state,
            "duration_s": round((now - (g.t_start_ns or now)) * 1e-9, 4),
        }
        g.result = Result(
            ok=code == Code.OK, code=code, op=g.op, id=g.rid, moved=bool(data.get("moved", False)), data=data,
            details=details,
        )  # fmt: skip
        for x in g.resources:
            self._released_ns[x] = now
        for w in g.waiters:
            w[0].finish(g.result.model_copy(update={"op": w[2], "id": w[3]}))
        g.waiters.clear()

    def on_stop(self, level: str, now: int) -> None:
        """An operator / client stop (always accepted): every goal ends - the core's executor holds the limbs."""
        if self._lean is not None:  # session 13: the core zeroes the lean until a resume - r1d's renewal ends too
            self._lean = None
            self._lean_ended = f"the lean ended with a '{level}' (body.pose again after `r1 resume`)"
        for g in list(self.table.active()):
            g.runner.end("aborted", Code.GUARD_STOP, {"code": "OPERATOR_STOP", "detail": f"a '{level}' request"})
            self._finish(g, now)

    def _tick_grips(self, now: int) -> None:
        """Every grip keeps squeezing at r1d's tick (the GraspRunner re-sends its command, TTL 0.5 s); a TENTATIVE grip
        gets the lift check from the arm's measured motion: the grasp centre's rise since the verdict (FK)."""
        pts: Any = None
        c = self.svc.contact  # session 12: the pads' contact record carries the grips (stops / state) and the
        if c is not None and hasattr(c, "set_grip"):  # pads' drift-following tare pauses on a side that holds a grip
            for s in ("left", "right"):
                c.set_grip(s, self.grips.get(s))
        for side, (tgr, tr) in list(self._grip_traces.items()):  # session 12: a grip that ended -> its trace to disk
            if self.grips.get(side) is not tgr:
                del self._grip_traces[side]
                d = trace_dir(self.svc)
                if d is not None:
                    tr.save(d, tgr)
        for side, gr in list(self.grips.items()):
            tr = self._grip_traces.get(side, (None, None))[1]
            if tr is not None:
                tr.now_ns = now
            try:
                st = gr.tick(now)
            except Exception:  # a hands hiccup: the grip's own state says it (stale -> NOT_HOLDING)
                continue
            if st == "TENTATIVE":
                if pts is None:
                    pts = hand_points(self.svc.port, self.svc.hands)
                if pts is not None:
                    z = float(pts[SIDE_INDEX[side], POINT_GRASP_CENTRE, 2])
                    z0 = self._grip_z0.setdefault(side, z)
                    gr.notify_lift(max(0.0, z - z0))
                    if tr is not None:
                        tr.note_lift(max(0.0, z - z0))
            if tr is not None and st != "OPEN":
                lv = tr.last_verdict
                if lv is None or lv.get("state") != st or lv.get("reason") != getattr(gr, "reason", None):
                    tr.last_verdict = gr.verdict().as_dict()
        from . import hand_heat  # session 13: the hands' heat in words before the grip's 72 C limit (HW2 day 2)

        hand_heat.tick(self, now)

    def surface_below(self, side: str) -> float | None:
        """The declared support surface's top under the hand (base frame z), from the world model: the highest
        surface / container floor whose footprint holds the hand's grasp centre and whose top is below it. None
        without a world model or a measured hand (the grip then does not judge the contact height - its evidence
        says so)."""
        world = self.svc.world
        if world is None:
            return None
        pts = hand_points(self.svc.port, self.svc.hands)
        if pts is None:
            return None
        c = pts[SIDE_INDEX[side], POINT_GRASP_CENTRE]
        try:
            snap = world.snapshot()
        except Exception:
            return None
        return support_top_below(snap, c)

    def contact_height(self, side: str, surface_z: float) -> Callable[[Any], dict[str, float]] | None:
        """For the grip's contact-height term: the fingertips' heights above the surface for a given hand pose (the
        arm's measured joints + the hand's q, through the built-in arm model's FK)."""
        port = self.svc.port
        fn = getattr(getattr(port, "mod", None), "fingertips_base", None)
        if fn is None:
            return None
        k = SIDE_INDEX[side]

        def f(q_hand: Any) -> dict[str, float]:
            from .goals import port_state

            st = port_state(port)
            if st is None or "q" not in st:
                return {}
            q = np.asarray(st["q"], dtype=np.float64).reshape(1, 26)
            quat = np.asarray(st["imu_quat"], dtype=np.float64).reshape(1, 4)
            qh = np.asarray(q_hand, dtype=np.float64).reshape(1, 7)
            pts, _low, _feet = fn(q, quat, qh if k == 0 else None, qh if k == 1 else None)
            p = np.asarray(pts, dtype=np.float64)[0, k]
            return {"thumb": float(p[1, 2] - surface_z), "middle": float(p[2, 2] - surface_z),
                    "index": float(p[3, 2] - surface_z)}  # fmt: skip

        return f

    def end_grip(self, side: str) -> bool:
        """The operator's direct hand command (`hand.command`, `r1 hand`) takes the hand: the side's grip ends here - it
        is no longer ticked, so it no longer re-sends its command over the operator's (session 11: a NOT_HOLDING grip
        keeps its grip until told to open - contact decides - and must never fight the operator's word). True if a
        grip ended."""
        gr = self.grips.pop(side, None)
        self._grip_z0.pop(side, None)
        return gr is not None

    def grip_status(self) -> dict[str, Any]:
        # keeps_grip (session 11): NOT_HOLDING with the hand still gripping - it opens only on hand.release
        return {
            side: {**gr.verdict().as_dict(), "keeps_grip": bool(getattr(gr, "keeps_grip", False))}
            for side, gr in self.grips.items()
        }

    def holder(self, resource: str) -> Goal | None:
        held = self.table.holding(frozenset([resource]))
        return held[0] if held else None

    # ------------------------------------------------------------------ goal.*
    def _get(self, op: str, rid: Any, goal_id: int) -> Goal | Result:
        g = self.table.goals.get(goal_id)
        if g is None:
            return fail(op, rid, Code.UNKNOWN_GOAL, goal_id=goal_id)
        return g

    def _goal_status(self, r: GoalStatusReq, t: Ticket) -> Result | None:
        g = self._get(r.op, r.id, r.goal_id)
        if isinstance(g, Result):
            return g
        out: dict[str, Any] = dict(g.summary())
        if g.result is not None:
            out["result"] = g.result.model_dump(mode="json")
        return ok(r.op, r.id, **out)

    def _goal_wait(self, r: GoalWaitReq, t: Ticket) -> Result | None:
        g = self._get(r.op, r.id, r.goal_id)
        if isinstance(g, Result):
            return g
        if g.result is not None:
            return g.result.model_copy(update={"op": r.op, "id": r.id})
        g.waiters.append((t, self.svc.port.now_ns() + round(r.timeout_s * 1e9), r.op, r.id))
        return None

    def _goal_cancel(self, r: GoalCancelReq, t: Ticket) -> Result | None:
        g = self._get(r.op, r.id, r.goal_id)
        if isinstance(g, Result):
            return g
        if g.state in GOAL_TERMINAL:
            return ok(r.op, r.id, goal_id=g.goal_id, state=g.state, already_ended=True)
        now = self.svc.port.now_ns()
        for child in [c for c in self.table.active() if c.parent == g.goal_id]:
            child.runner.stop(now, "canceled")
            self._finish(child, now)
        g.runner.stop(now, "canceled")
        self._finish(g, now)
        return ok(r.op, r.id, moved=True, goal_id=g.goal_id, state=g.state)

    def _goal_list(self, r: GoalListReq, t: Ticket) -> Result | None:
        done = [g for g in self.table.goals.values() if g.state in GOAL_TERMINAL][-20:]
        return ok(r.op, r.id, active=self.status(), recent=[g.summary() for g in done])

    # ------------------------------------------------------------------ arm.move_joint / arm.stop
    def _arm_move_joint(self, r: ArmMoveJointReq, t: Ticket) -> Result | None:
        svc = self.svc
        group = ARM_GROUPS[r.arm]
        chk = svc._group_check(r.op, r.id, group)
        if isinstance(chk, Result):
            return chk
        q = np.asarray(r.q, dtype=np.float64)
        lim = svc._check_limits(group, q)
        if lim is not None:
            return fail(r.op, r.id, Code.LIMIT, **lim)

        def plan(cur: np.ndarray) -> list[Knot] | Result:
            dur = r.duration_s if r.duration_s is not None else min_jerk_duration(cur, q, r.speed)
            peak = 1.875 * float(np.max(np.abs(q - cur))) / dur
            if peak > DQ_MAX_RAD_S:
                return fail(r.op, r.id, Code.LIMIT, reason="the move is too fast for the joints' dq limit",
                            peak_rad_s=round(peak, 3), dq_max_rad_s=DQ_MAX_RAD_S)  # fmt: skip
            return min_jerk_knots(cur, q, dur)

        if svc.planner is not None and r.duration_s is None:  # world-checked (DESIGN §5.1: planned against the world)
            goal = {"kind": "move_joint", "arm": r.arm, "q": [float(x) for x in q], "speed": r.speed}
            return self.accept(r, t, frozenset([group]), PlannedRunner(self, r.op, r.arm, goal))
        if self.world_has_entities():  # an unchecked path in a declared world: refused (the compat finding, session 9)
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="the world has declared entities and this path would not "
                        "be checked against them: " + ("omit duration_s" if svc.planner is not None else
                                                       "r1d has no planner"))  # fmt: skip
        runner = TrajectoryRunner(svc, group, plan, side=r.arm, planned_on="r1d")
        return self.accept(r, t, frozenset([group]), runner)

    def _arm_stop(self, r: ArmStopReq, t: Ticket) -> Result | None:
        """Stop one arm (or both): its goal ends `canceled`; with no goal running, the arm HOLDs where its target is
        (an L0 stream stops too). Accepted in every mode the arms execute in."""
        svc = self.svc
        arms = [r.arm] if r.arm is not None else ["left", "right"]
        now = svc.port.now_ns()
        stopped: list[int] = []
        held: list[str] = []
        for arm in arms:
            group = ARM_GROUPS[arm]
            g = self.holder(group)
            if g is not None:
                g.runner.stop(now, "canceled")
                self._finish(g, now)
                stopped.append(g.goal_id)
                continue
            chk = svc._group_check(r.op, r.id, group)
            if isinstance(chk, Result):
                continue  # nothing of ours runs on an arm outside BALANCE / ON_STAND
            gid, _ = chk
            svc._jogs.pop(group, None)
            cur = svc._current(group)
            if cur is not None:
                svc._push(group, gid, "hold", {"cmd_id": svc.new_cmd_id(), "t_issue_ns": now, "q": cur.tolist()})
                held.append(group)
        return ok(r.op, r.id, moved=bool(stopped or held), canceled_goals=stopped, held=held)

    # ------------------------------------------------------------------ planned arm goals (r1mc3.plan)
    def world_has_entities(self) -> bool:
        w = self.svc.world
        if w is None:
            return False
        try:
            return len(w.snapshot().entities) > 0
        except Exception:
            return True  # unknown: assume there is something to check

    def world_model(self) -> Any:
        """The world model (created empty on first use: `world.declare` fills it)."""
        if self.svc.world is None:
            from r1mc3.world import WorldModel

            self.svc.world = WorldModel()
        return self.svc.world

    def held_object(self, side: str) -> dict[str, Any] | None:
        """The object a grip on `side` holds, for the planner (session 10): a sphere of half the grasp's width at the
        grasp centre (the grip knows only the width it was asked to close on; None without a width / a holding grip).
        It keeps a carried object off a container's rim and the table like the hand itself."""
        gr = self.grips.get(side)
        w = None if gr is None else getattr(gr, "width_m", None)
        if gr is None or w is None or getattr(gr, "state", "") not in ("HOLDING", "TENTATIVE"):
            return None
        return {"shape": "sphere", "size": [round(float(w) / 2.0, 4)]}

    def plan_goal_dict(self, r: Any) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": PLANNED_KINDS[r.op], "arm": r.arm, "speed": float(getattr(r, "speed", 0.5))}
        for k in (
            "q",
            "position",
            "delta",
            "frame",
            "approach",
            "height_m",
            "lift_m",
            "max_depth_m",
            "until",
            "prefer",
            "grasp_yaw_deg",  # session 10 (planner): the hover's closing axis
            "grip",
            "width_m",
        ):
            v = getattr(r, k, None)
            if v is not None:
                d[k] = v
        tid = getattr(r, "target_id", None)
        if tid:  # session 10 round B: the goal's own target entity - only the hand may touch it (the planner's rule)
            d["allow_hand"] = [str(tid)]
        # session 12: the requested approach (orient.STYLES) + the grasp height of a hover; a lower ends at the last
        # hover's grasp pose when its line ends there (the planner checks it: core.END_Q_TOL_M)
        for k in ("approach_style", "approach_tol_deg", "grasp_z_m", "min_descent_m"):
            v = getattr(r, k, None)
            if v is not None:
                d[k] = v
        if r.op == "arm.lower":
            gq = getattr(self, "_hover_grasp_q", {}).get(r.arm)
            if gq is not None:
                d["end_q"], d["end_point"] = list(gq[0]), gq[1]
        held = None if self is None else self.held_object(r.arm)
        if held is not None:
            d["held"] = held
        return d

    def note_hover_plan(self, arm: str, details: dict[str, Any]) -> None:
        """Session 12: a planned hover's grasp pose (plan_details.orientation.grasp_q) for the next arm.lower."""
        store: dict[str, Any] = self.__dict__.setdefault("_hover_grasp_q", {})
        o = details.get("orientation") or {}
        gq = o.get("grasp_q")
        if gq is None:
            store.pop(arm, None)
        else:
            store[arm] = ([float(v) for v in gq], o.get("grasp_point"))

    def _arm_planned(self, r: Any, t: Ticket) -> Result | None:
        svc = self.svc
        if svc.planner is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="no planner in this r1d (r1mc3.plan.PlannerWorker)")
        group = ARM_GROUPS[r.arm]
        chk = svc._group_check(r.op, r.id, group)
        if isinstance(chk, Result):
            return chk
        if isinstance(r, ArmLowerReq) and r.stop_z_m is not None:  # a stop height -> the depth from the hand now
            pts = hand_points(svc.port, svc.hands)
            if pts is None:
                return fail(r.op, r.id, Code.UNSUPPORTED, reason="no measured hand for stop_z_m")
            depth = float(pts[SIDE_INDEX[r.arm], POINT_GRASP_CENTRE, 2]) - r.stop_z_m
            if not 0.0 < depth <= 0.4:
                return fail(r.op, r.id, Code.LIMIT, reason="stop_z_m: 0 < the depth from the grasp centre <= 0.4 m",
                            depth_m=round(depth, 4))  # fmt: skip
            r = r.model_copy(update={"max_depth_m": depth, "until": "depth" if r.until == "depth" else r.until})
        if isinstance(r, ArmLowerReq) and r.until in ("contact", "surface") and svc.contact is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="a contact-guarded descent needs the pads' contact "
                        "publisher (r1hands) - none in this r1d; until='depth' runs without it")  # fmt: skip
        runner = PlannedRunner(self, r.op, r.arm, self.plan_goal_dict(r), contact_ends_ok=isinstance(r, ArmLowerReq))
        return self.accept(r, t, frozenset([group]), runner)

    def _arm_plan(self, r: ArmPlanReq, t: Ticket) -> Result | None:
        """The dry run: the whole goal planned on the current world snapshot from the arm's current target; nothing
        moves. With a planner worker the reply comes when the plan is back."""
        svc = self.svc
        if svc.planner is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="no planner in this r1d")
        try:
            g: Any = REQUEST.validate_python({**r.goal, "id": None})
        except Exception as e:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason=f"goal: {e}")
        if getattr(g, "op", None) not in PLANNED_KINDS:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason="goal: an arm goal", kinds=sorted(PLANNED_KINDS))
        out = submit_plan(self, self.plan_goal_dict(g), g.arm)
        if isinstance(out, Result):
            return out.model_copy(update={"op": r.op, "id": r.id})
        if hasattr(out, "done"):  # a Future from the worker
            self._pending_plans.append((out, t, r.op, r.id, svc.port.now_ns()))
            return None
        return _plan_reply(r.op, r.id, out)

    # ------------------------------------------------------------------ head.look_at
    def look_angles(self, position: Any) -> tuple[float, float, dict[str, Any]] | Result:
        """Head (pitch, yaw) that point the head camera at `position` (base frame): v2 skills.py:1259 look_at (yaw =
        atan2(dy, dx), pitch = atan2(-dz, horizontal), pitch + = down, yaw + = left), here relative to the torso's own
        yaw / pitch (the waist and the lean) and re-solved 3 times for the camera's offset from the joints; clipped to
        yaw +-1.4 (v2) and the head pitch's soft limits."""
        import mujoco

        from .goals import port_state

        st = port_state(self.svc.port)
        if st is None or "q" not in st:
            return fail("head.look_at", None, Code.UNSUPPORTED, reason="no measured state from the core")
        if self._head_geo is None:
            from r1mc3.world import Geometry

            self._head_geo = Geometry(None)
        geo = self._head_geo
        q26 = np.asarray(st["q"], dtype=np.float64).copy()
        quat = st.get("imu_quat")
        site = int(mujoco.mj_name2id(geo.m, mujoco.mjtObj.mjOBJ_SITE, "head_camera"))
        torso = int(mujoco.mj_name2id(geo.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"))
        if site < 0 or torso < 0:
            return fail("head.look_at", None, Code.UNSUPPORTED, reason="the model has no head_camera site")
        p = np.asarray(position, dtype=np.float64)
        pitch = yaw = 0.0
        lo, hi = HEAD_PITCH_SOFT
        for _ in range(3):
            geo.set_stance(q26, None if quat is None else np.asarray(quat, dtype=np.float64))
            cam = np.asarray(geo.d.site_xpos[site], dtype=np.float64)
            rt = np.asarray(geo.d.xmat[torso], dtype=np.float64).reshape(3, 3)
            d = rt.T @ (p - cam)  # the target in the torso's axes
            yaw_raw = float(np.arctan2(d[1], d[0]))
            pitch_raw = float(np.arctan2(-d[2], np.hypot(d[0], d[1])))
            yaw = float(np.clip(yaw_raw, -HEAD_YAW_MAX, HEAD_YAW_MAX))
            pitch = float(np.clip(pitch_raw, lo, hi))
            q26[24], q26[25] = pitch, yaw
        clipped = abs(yaw - yaw_raw) > 1e-3 or abs(pitch - pitch_raw) > 1e-3
        info = {
            "camera_m": [round(float(x), 4) for x in cam],
            "clipped": bool(clipped),
            "head_pitch_deg": round(float(np.degrees(pitch)), 2),
            "head_yaw_deg": round(float(np.degrees(yaw)), 2),
        }
        return pitch, yaw, info

    def _head_look_at(self, r: HeadLookAtReq, t: Ticket) -> Result | None:
        svc = self.svc
        if r.frame != "base":
            return fail(r.op, r.id, Code.UNKNOWN_FRAME, frame=r.frame, known=["base"])
        chk = svc._group_check(r.op, r.id, "head")
        if isinstance(chk, Result):
            return chk
        info: dict[str, Any] = {}

        def plan(cur: np.ndarray) -> list[Knot] | Result:
            ang = self.look_angles(r.position)
            if isinstance(ang, Result):
                return ang
            pitch, yaw, more = ang
            info.update(more)
            q = np.array([pitch, yaw])
            return min_jerk_knots(cur, q, min_jerk_duration(cur, q, r.speed))

        runner = TrajectoryRunner(svc, "head", plan, planned_on="r1d", extra=info)
        return self.accept(r, t, frozenset(["head"]), runner)

    # ------------------------------------------------------------------ body.pose
    def policy_envelope(self, arm: str) -> Any:
        """The balance policy's trained arm range for `arm` (its ONNX metadata, the right arm mirrored as v2) - None
        when the policy has none."""
        if not hasattr(self, "_arm_ranges"):
            from r1mc3.policy import POLICY_DIR, ReferencePolicy

            try:
                self._arm_ranges: Any = ReferencePolicy.load(POLICY_DIR).arm_ranges_left
            except Exception:
                self._arm_ranges = None
        if self._arm_ranges is None:
            return None
        from r1mc3.plan import policy_envelope

        return policy_envelope(self._arm_ranges, arm)

    def policy_lean_range(self) -> tuple[float, float] | None:
        """The live policy's trained lean range (deg) from its metadata (policies/<dir>/r1mc_policy.json)."""
        import json

        from r1mc3.policy import POLICY_DIR

        try:
            meta = json.loads((POLICY_DIR / "r1mc_policy.json").read_text())
            lo, hi = meta["lean_range_deg"]
            return float(lo), float(hi)
        except (OSError, KeyError, ValueError, TypeError):
            return None

    def _send_lean(self, now: int) -> None:
        # session 13 (HW2 day 2's F12): a stop latched in the core zeroes the base command until a resume - the lean
        # ENDS with it (as the stop ends every goal): r1d stops renewing it, and after `r1 resume` the robot stays
        # upright until a new body.pose (never a lean coming back by itself)
        if stop_latch.latched(self.svc.port):
            self._lean = None
            self._lean_ended = "the lean ended with a stop (body.pose again after `r1 resume`)"
            return
        self._base_seq += 1
        fn = getattr(self.svc.port, "set_base_cmd", None)
        if fn is not None:
            fn(self._base_seq, now, LEAN_TTL_NS, 0.0, 0.0, 0.0, float(self._lean or 0.0))

    def _body_pose(self, r: BodyPoseReq, t: Ticket) -> Result | None:
        """The policy's own lean command (DESIGN §5.1 body.pose): within its trained range, renewed by r1d at its
        tick (TTL 0.5 s: if r1d stops, the core's command expires to zero = upright); lean 0 ends it. BALANCE only."""
        svc = self.svc
        if r.height is not None:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason="this policy has no height command")
        if r.lean_deg is None:
            return ok(r.op, r.id, lean_deg=None if self._lean is None else round(float(np.degrees(self._lean)), 3))
        mode = svc.mode_name()
        if mode != "balance":
            return fail(r.op, r.id, Code.WRONG_MODE, mode=mode, needs="balance")
        rng = self.policy_lean_range()
        if rng is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="the policy's metadata has no lean range")
        if not rng[0] <= r.lean_deg <= rng[1]:
            return fail(r.op, r.id, Code.LIMIT, lean_deg=r.lean_deg, trained_range_deg=list(rng))
        if getattr(svc.port, "set_base_cmd", None) is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="this port cannot write the policy's base command")
        # session 13 (HW2 day 2's F12: `r1 lean 3` was ACCEPTED and nothing happened - a client's stop six minutes
        # earlier had latched in the core, which zeroes the base command until a resume): refused in words instead
        if r.lean_deg != 0.0 and stop_latch.latched(svc.port):
            return fail(r.op, r.id, Code.STOPPED, reason=stop_latch.WORDS, lean_deg=r.lean_deg, needs="r1 resume")
        self._lean_ended = None
        lean = float(np.radians(r.lean_deg))
        self._lean = lean if r.lean_deg != 0.0 else None
        try:
            if self._lean is not None:
                self._send_lean(svc.port.now_ns())
        except NotImplementedError as e:
            self._lean = None
            return fail(r.op, r.id, Code.UNSUPPORTED, reason=str(e))
        return ok(r.op, r.id, moved=True, lean_deg=r.lean_deg, trained_range_deg=list(rng))

    # ------------------------------------------------------------------ seq.run / skills (L3)
    def _seq_run(self, r: SeqRunReq, t: Ticket) -> Result | None:
        bad = [i for i, s in enumerate(r.steps) if s.get("op") in ("seq.run", "damp", "shutdown")]
        if bad:
            return fail(
                r.op, r.id, Code.INVALID_ARGS, reason="seq.run does not run seq.run / damp / shutdown", steps=bad
            )
        runner = SeqRunner(self, r.op, list(r.steps), r.stop_on_failure, Code.STEP_FAILED)
        return self.accept(r, t, steps_resources(r.steps), runner)

    def _skill(self, r: SkillReq, t: Ticket) -> Result | None:
        """skill.ready / skill.table_ready: the named posture (r1mc3.skills.postures), one arm at a time through
        arm.move_joint (world-checked with a planner); a failed arm ends the skill (SKILL_FAILED, the step's result)."""
        from r1mc3.skills import posture

        name = r.op.split(".", 1)[1]
        arms: list[str] = ["left", "right"] if r.arms == "both" else [r.arms]
        if name == "table_ready" and self.svc.planner is not None:
            # session 10: from ANY start, incl. the hands under a declared table (v2's out-and-around, the fingers
            # closed while a hand comes out): r1mc3.skills.table_ready
            from r1mc3.skills.table_ready import TableReadyRunner

            res = frozenset(ARM_GROUPS[a] for a in arms)
            if self.svc.hands is not None:
                res = res | frozenset(HAND_RESOURCES[a] for a in arms)
            # ~10-13 s per arm at the default speed (fingers, out, over, on, fingers): the goal's default 30 s would
            # time out a two-arm reset - at least 90 s (a stop / cancel ends it at any time, as always)
            r = r.model_copy(update={"timeout_s": max(float(r.timeout_s), 90.0)})
            return self.accept(r, t, res, TableReadyRunner(self, r.op, name, arms, r.speed))
        steps = [{"op": "arm.move_joint", "arm": a, "q": posture(name, a), "speed": r.speed} for a in arms]
        # v2's rule (skills.py table_ready): every arm is planned before anything moves (each again just before it
        # moves - its own step): a refused arm ends the skill with nothing moved
        pre = [{"op": "arm.plan", "goal": st} for st in steps] if self.svc.planner is not None else []
        runner = SeqRunner(self, r.op, steps, True, Code.SKILL_FAILED, precheck=pre)
        return self.accept(r, t, frozenset(ARM_GROUPS[a] for a in arms), runner)

    # ------------------------------------------------------------------ world.*
    def _world_declare(self, r: WorldDeclareReq, t: Ticket) -> Result | None:
        from r1mc3.world import entity_from_dict, entity_to_dict

        try:
            e = self.world_model().declare(entity_from_dict(dict(r.entity)))
        except (ValueError, TypeError, KeyError) as ex:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason=str(ex))
        return ok(r.op, r.id, entity=entity_to_dict(e), version=self.world_model().snapshot().version)

    def _world_clear(self, r: WorldClearReq, t: Ticket) -> Result | None:
        w = self.world_model()
        if r.entity_id is None:
            w.clear()
            return ok(r.op, r.id, cleared="all")
        if not w.remove(r.entity_id):
            return fail(r.op, r.id, Code.INVALID_ARGS, reason="no such entity", entity_id=r.entity_id)
        return ok(r.op, r.id, cleared=r.entity_id)

    def _world_list(self, r: WorldListReq, t: Ticket) -> Result | None:
        from r1mc3.world import entity_to_dict

        snap = self.world_model().snapshot()
        return ok(r.op, r.id, entities=[entity_to_dict(e) for e in snap.entities], version=snap.version)

    # ------------------------------------------------------------------ hands
    def _hands_or_fail(self, op: str, rid: Any) -> Result | None:
        if self.svc.hands is None:
            return fail(op, rid, Code.UNSUPPORTED, reason="no hands in this r1d")
        return None

    def _hand_grasp(self, r: HandGraspReq, t: Ticket) -> Result | None:
        bad = self._hands_or_fail(r.op, r.id)
        if bad is not None:
            return bad
        return self.accept(r, t, frozenset([HAND_RESOURCES[r.side]]), GraspGoalRunner(self, r))

    def _hand_release(self, r: HandReleaseReq, t: Ticket) -> Result | None:
        bad = self._hands_or_fail(r.op, r.id)
        if bad is not None:
            return bad
        from r1mc3.hands import grip as G

        runner = HandMoveRunner(self, r.side, G.preset("open", r.side), release=True)
        return self.accept(r.model_copy(update={"preempt": True}), t, frozenset([HAND_RESOURCES[r.side]]), runner)

    def _hand_preshape(self, r: HandPreshapeReq, t: Ticket) -> Result | None:
        bad = self._hands_or_fail(r.op, r.id)
        if bad is not None:
            return bad
        from r1mc3.hands import grip as G

        if r.shape in G.GRIPS:
            q = G.preshape(r.shape, r.side)
        elif r.shape in G.PRESETS_LEFT:
            q = G.preset(r.shape, r.side)
        else:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason="unknown shape",
                        shapes=sorted([*G.GRIPS, *G.PRESETS_LEFT]))  # fmt: skip
        if r.side in self.grips:
            return fail(r.op, r.id, Code.RESOURCE_BUSY, reason="the hand holds a grip: hand.release first")
        return self.accept(r, t, frozenset([HAND_RESOURCES[r.side]]), HandMoveRunner(self, r.side, q))

    def _hand_calibrate(self, r: Any, t: Ticket) -> Result | None:
        """hand.calibrate (session 10): the hand's air-pinch reference (r1mc3.service.hand_cal)."""
        from .hand_cal import hand_calibrate

        return hand_calibrate(self, r, t)

    def hand_calibration(self) -> dict[str, Any]:
        from .hand_cal import store

        out: dict[str, Any] = {"air_pinch": store(self).summary()}
        return out

    def layer_zero(self, side: str) -> float | None:
        """The hand's air-pinch reference (m) from hand.calibrate (or R1_HAND_CAL_FILE), None = not calibrated."""
        from .hand_cal import store

        z: float | None = store(self).layer_zero(side)
        return z

    def _hand_tare(self, r: HandTareReq, t: Ticket) -> Result | None:
        """Re-tare one hand's pads (the contact publisher re-collects its tare over the next 0.5 s; the hand must be
        open and touching nothing - v2's rule). Nothing moves."""
        c = self.svc.contact
        if c is None or not hasattr(c, "tare"):
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="no pad-contact publisher in this r1d")
        if r.side in self.grips:
            return fail(r.op, r.id, Code.RESOURCE_BUSY, reason="the hand holds a grip: hand.release first")
        if hasattr(c, "retare"):
            c.retare(r.side)
        else:
            c.tare[r.side] = None
            c._tare_buf[r.side] = []
        return ok(r.op, r.id, side=r.side, tare_samples=int(getattr(c, "tare_samples", 0)))


class GraspGoalRunner:
    """hand.grasp as a goal: the grip state machine (r1mc3.hands.grip.GraspRunner) until its verdict - TENTATIVE /
    HOLDING -> succeeded (the grip keeps holding, ticked by L2 until hand.release; the lift check runs when an arm
    lifts the hand), NOT_HOLDING -> aborted with NOT_HOLDING and the reason. Session 11 (the user's "contact decides"):
    a NOT_HOLDING verdict never relaxes the fingers by itself - the grip stays in `L2.grips`, ticked (its command
    re-sent) until hand.release / another hand goal; only the heat abort backs off (powered, never limp)."""

    def __init__(self, l2: L2, r: HandGraspReq) -> None:
        self.l2, self.r = l2, r
        self.gr: Any = None
        self.end_state: str | None = None
        self.code = Code.OK
        self.stopped_by: dict[str, Any] | None = None

    def ready(self, now_ns: int) -> bool:
        return True

    def start(self, now_ns: int) -> Result | None:
        svc, r = self.l2.svc, self.r
        factory = svc.grasp_factory
        if factory is None:
            from r1mc3.hands.grip import GraspRunner

            factory = GraspRunner
        surface = self.l2.surface_below(r.side)
        height = self.l2.contact_height(r.side, surface) if surface is not None else None
        # a thin layer is judged against the hand's own air-pinch reference (hand.calibrate); none -> the grip refuses
        z = self.l2.layer_zero(r.side) if r.object_class == "thin_flexible" else None
        kw: dict[str, Any] = {} if z is None else {"layer_zero_m": z}
        # session 12: the grasp's hand data as the runner sees it, for offline replay (r1mc3.hands.trace; HW2 day 1's
        # tape pinch read NOT_HOLDING while held and nothing on the robot kept what the verdict saw)
        tracer: Any = None
        hands: Any = svc.hands
        if trace_dir(svc) is not None and hands is not None:
            from r1mc3.hands.trace import TracingHands

            params = {"grip": r.grip, "object_class": r.object_class, "width_m": r.width_m, "squeeze": r.squeeze,
                      "surface_z_m": surface, "contact_height": height is not None, **kw}  # fmt: skip
            tracer = TracingHands(hands, r.side, params)
            tracer.now_ns = now_ns
            hands = tracer
            height = tracer.wrap_contact_height(height)
        try:
            self.gr = factory(hands, r.side, r.grip, r.object_class, width_m=r.width_m, squeeze=r.squeeze,
                              surface_z_m=surface, contact_height=height, **kw)  # fmt: skip
        except ValueError as e:
            reason = str(e)
            if r.object_class == "thin_flexible" and z is None:
                reason += (f" - r1d: run hand.calibrate {{side: {r.side}, kind: air_pinch}} first"
                           f" (`r1 calibrate-pinch {r.side}`)")  # fmt: skip
            return fail(r.op, r.id, Code.INVALID_ARGS, reason=reason)
        old = self.l2.grips.pop(r.side, None)
        self.l2._grip_z0.pop(r.side, None)
        del old
        old_tr = self.l2._grip_traces.pop(r.side, None)
        if old_tr is not None and trace_dir(svc) is not None:
            old_tr[1].save(trace_dir(svc), old_tr[0])
        self.gr.start(now_ns)
        self.l2.grips[r.side] = self.gr
        if tracer is not None:
            self.l2._grip_traces[r.side] = (self.gr, tracer)
        return None

    def tick(self, now_ns: int) -> str | None:
        if self.end_state is not None:
            return self.end_state
        st = self.gr.state  # ticked by L2._tick_grips this tick
        if st in GRIP_HOLDING:
            self.end_state = "succeeded"
        elif st == "NOT_HOLDING":
            self.end_state, self.code = "aborted", Code.NOT_HOLDING
            self.stopped_by = {"code": "NOT_HOLDING", "detail": self.gr.verdict().reason}
        return self.end_state

    def stop(self, now_ns: int, why: str) -> None:
        if self.end_state is not None:
            return
        if self.gr is not None and self.gr.state not in GRIP_HOLDING:  # nothing held yet: open again
            self.gr.release(now_ns)
            self.l2.grips.pop(self.r.side, None)
        self.end_state = why
        self.code = {"preempted": Code.PREEMPTED, "timed_out": Code.TIMEOUT}.get(why, Code.CANCELED)
        self.stopped_by = {"code": why.upper(), "detail": "r1d stopped the grasp"}

    def end(self, state: str, code: Code, stopped_by: dict[str, Any] | None) -> None:
        if self.end_state is None:
            self.end_state, self.code, self.stopped_by = state, code, stopped_by

    def outcome(self) -> tuple[Code, dict[str, Any], dict[str, Any]]:
        v = self.gr.verdict().as_dict() if self.gr is not None else None
        data: dict[str, Any] = {
            "moved": self.gr is not None,
            "side": self.r.side,
            "grip": self.r.grip,
            "object_class": self.r.object_class,
            "holding": None if v is None else v["state"] in GRIP_HOLDING,
            "verdict": v,
            "stopped_by": self.stopped_by,
        }
        pts = hand_points(self.l2.svc.port, self.l2.svc.hands)
        if pts is not None:
            k = SIDE_INDEX[self.r.side]
            data["fingertips_m"] = tips_dict(pts[k])
            data["grasp_centre_m"] = [round(float(x), 5) for x in pts[k, POINT_GRASP_CENTRE]]
        # session 13 (HW2 day 2: the 4th back-to-back box wrap ended "hand hot (72 C >= 72)" with nothing said
        # before): the hand's heat with every grasp reply - level ok / warm / hot and the words (r1mc3.hands.heat)
        from . import hand_heat

        data["hand_heat"] = hand_heat.status(self.l2, self.l2.svc.port.now_ns()).get(self.r.side)
        return self.code, data, ({} if self.code == Code.OK else {"verdict": v})


class HandMoveRunner:
    """hand.release / hand.preshape: the hand to a pose (re-sent every tick, the last command with HAND_HOLD_TTL_S);
    done when every finger is within HAND_MOVE_TOL_RAD or after HAND_MOVE_MAX_S (`reached` says which)."""

    def __init__(self, l2: L2, side: str, q: Any, *, release: bool = False) -> None:
        self.l2, self.side, self.q, self.release = l2, side, np.asarray(q, dtype=np.float64), release
        self.end_state: str | None = None
        self.code = Code.OK
        self.stopped_by: dict[str, Any] | None = None
        self.t0 = 0
        self.reached = False
        self.q_end: list[float] | None = None

    def ready(self, now_ns: int) -> bool:
        return True

    def _send(self, ttl: float) -> None:
        from r1mc3.hands import grip as G

        assert self.l2.svc.hands is not None
        self.l2.svc.hands.command(self.side, self.q, kp=np.full(7, G.KP_GRIP), kd=np.full(7, G.KD_GRIP), ttl_s=ttl)

    def start(self, now_ns: int) -> Result | None:
        if self.release:
            gr = self.l2.grips.pop(self.side, None)
            self.l2._grip_z0.pop(self.side, None)
            if gr is not None:
                gr.release(now_ns)
        else:
            # session 11: a hand moved to a pose (a skill's finger step - table_ready closes / opens the fingers) takes
            # it from any grip: a NOT_HOLDING grip keeps its grip (contact decides) and would re-send its command over
            # this one every tick (hand.preshape itself is refused while a grip exists: RESOURCE_BUSY)
            self.l2.end_grip(self.side)
        try:
            self._send(1.0)
        except ValueError as e:
            return fail("hand", None, Code.INVALID_ARGS, reason=str(e))
        self.t0 = now_ns
        return None

    def tick(self, now_ns: int) -> str | None:
        if self.end_state is not None:
            return self.end_state
        assert self.l2.svc.hands is not None
        st = self.l2.svc.hands.state(self.side)
        q = st.get("q")
        if q is not None and st.get("valid", True):
            self.q_end = [round(float(x), 4) for x in q]
            self.reached = float(np.max(np.abs(np.asarray(q, dtype=np.float64) - self.q))) <= HAND_MOVE_TOL_RAD
        if self.reached or now_ns - self.t0 >= HAND_MOVE_MAX_S * 1e9:
            self._send(HAND_HOLD_TTL_S)
            self.end_state = "succeeded"
        else:
            self._send(1.0)
        return self.end_state

    def stop(self, now_ns: int, why: str) -> None:
        if self.end_state is None:
            self.end_state = why
            self.code = {"preempted": Code.PREEMPTED, "timed_out": Code.TIMEOUT}.get(why, Code.CANCELED)
            self.stopped_by = {"code": why.upper(), "detail": "r1d stopped the hand"}

    def end(self, state: str, code: Code, stopped_by: dict[str, Any] | None) -> None:
        if self.end_state is None:
            self.end_state, self.code, self.stopped_by = state, code, stopped_by

    def outcome(self) -> tuple[Code, dict[str, Any], dict[str, Any]]:
        data = {"moved": True, "side": self.side, "reached": self.reached, "q_hand": self.q_end,
                "target_q_hand": [round(float(x), 4) for x in self.q], "stopped_by": self.stopped_by}  # fmt: skip
        return self.code, data, {}


def plan_inputs(l2: L2, arm: str) -> tuple[Any, Any, dict[str, Any] | None, Any] | Result:
    """For the planner: the measured joints (26: where the arms really are), the IMU quaternion, the hands' joints and
    the goal arm's commanded target (the executor continues from its TARGET: the plan's first knot = it; the measured
    pose sags below it - the planner's arm_target)."""
    from .goals import port_state

    svc = l2.svc
    st = port_state(svc.port)
    if st is None or "q" not in st:
        return fail("plan", None, Code.UNSUPPORTED, reason="no measured state from the core")
    q26 = np.asarray(st["q"], dtype=np.float64).copy()
    target = svc._current(ARM_GROUPS[arm])
    hands: dict[str, Any] | None = None
    if svc.hands is not None:
        hands = {}
        for side in ("left", "right"):
            try:
                hs = svc.hands.state(side)
            except Exception:
                continue
            if hs.get("q") is not None and hs.get("valid", True):
                hands[side] = np.asarray(hs["q"], dtype=np.float64)
    quat = st.get("imu_quat")
    return q26, None if quat is None else np.asarray(quat, dtype=np.float64), hands or None, target


def submit_plan(l2: L2, goal: dict[str, Any], arm: str) -> Any:
    """Plan through the injected planner: a worker (`submit` -> Future) or an inline planner (`plan(...)` / a
    callable -> PlanResult). Returns the Future, the PlanResult, or a refusal Result."""
    from r1mc3.plan import PlanGoal

    inputs = plan_inputs(l2, arm)
    if isinstance(inputs, Result):
        return inputs
    q26, quat, hands, target = inputs
    pg = PlanGoal.from_dict(goal)
    snap = l2.world_model().snapshot()
    kw: dict[str, Any] = {"imu_quat": quat, "hands": hands, "arm_target": target}
    if l2.svc.mode_name() == "balance":  # under the balance policy the arms stay inside its trained range (v2)
        kw["envelope"] = l2.policy_envelope(arm)
    planner: Any = l2.svc.planner
    assert planner is not None
    if hasattr(planner, "submit"):
        return planner.submit(snap, pg, q26, **kw)
    fn = planner.plan if hasattr(planner, "plan") else planner
    return fn(snap, pg, q26, **kw)


def _fixed(knots: list[Knot]) -> Callable[[Any], list[Knot] | Result]:
    def plan(cur: Any) -> list[Knot] | Result:
        return knots

    return plan


def _plan_reply(op: str, rid: Any, res: Any) -> Result:
    if res is None:
        return fail(op, rid, Code.TIMEOUT, reason="the planner worker did not answer")
    d = res.to_dict()
    d.pop("segments", None)
    d["plan_code"] = d.pop("code", None)
    d.pop("ok", None)
    d["legs"] = [{"leg": s.leg, "duration_s": round(s.duration_s, 3), "knots": len(s.t), "guards": s.guards}
                 for s in res.segments]  # fmt: skip
    if not res.ok:
        return fail(op, rid, _code(res.code), **d)
    return ok(op, rid, **d)


def _code(name: str) -> Code:
    try:
        return Code(name)
    except ValueError:
        return Code.FAULT


class PlannedRunner:
    """A planned arm goal: the planner (worker or inline) -> the plan's legs, each one TRAJECTORY command run in
    order (the next starts when the executor reports the previous DONE: legs chain at rest); the descent of
    `arm.lower` may end on its contact guard (the goal's success)."""

    def __init__(self, l2: L2, op: str, arm: str, goal: dict[str, Any], *, contact_ends_ok: bool = False) -> None:
        self.l2, self.op, self.arm, self.goal = l2, op, arm, goal
        self.group = ARM_GROUPS[arm]
        self.contact_ends_ok = contact_ends_ok
        self.future: Any = None
        self.plan: Any = None
        self.leg = -1
        self.cur: TrajectoryRunner | None = None
        self.end_state: str | None = None
        self.code = Code.OK
        self.stopped_by: dict[str, Any] | None = None
        self.t0 = 0
        self.wall0 = 0.0
        self.planned_on = "worker" if hasattr(l2.svc.planner, "submit") else "inline"
        self.dropped: list[str] = []
        self._prev: Any = None
        self.p_start: Any = None

    def ready(self, now_ns: int) -> bool:
        cur = self.l2.svc._current(self.group)
        if cur is None:
            return True
        prev, self._prev = self._prev, cur
        return prev is not None and float(np.max(np.abs(cur - prev))) < 1e-4

    def start(self, now_ns: int) -> Result | None:
        import time

        out = submit_plan(self.l2, self.goal, self.arm)
        if isinstance(out, Result):
            return out
        self.p_start = hand_points(self.l2.svc.port, self.l2.svc.hands)
        self.t0, self.wall0 = now_ns, time.monotonic()
        if hasattr(out, "done"):
            self.future = out
            return None
        return self._take(out, now_ns)

    def _take(self, res: Any, now_ns: int) -> Result | None:
        self.plan = res
        if res.ok and self.op == "arm.hover":  # session 12: its grasp pose for the next arm.lower
            self.l2.note_hover_plan(self.arm, dict(res.details))
        if not res.ok:
            return fail(self.op, None, _code(res.code), reason=res.reason, plan_id=res.plan_id,
                        plan_details=dict(res.details))  # fmt: skip
        return self._next_leg(now_ns)

    def _guards(self, bits: int) -> dict[str, Any]:
        g: dict[str, Any] = {"enable": int(bits), "lag_rad": LAG_RAD}
        if bits & 4:
            if self.l2.svc.contact is None:
                g["enable"] = int(bits) & ~4
                self.dropped.append("contact (no pad-contact publisher in this r1d)")
            else:
                g.update({"contact_mask": ALL_PADS, "contact_all": 0, "hand": 0 if self.arm == "left" else 1})
        return g

    def _next_leg(self, now_ns: int) -> Result | None:
        self.leg += 1
        seg = self.plan.segments[self.leg]
        knots: list[Knot] = [
            (round(float(t) * 1e9), np.asarray(q), np.asarray(dq), np.asarray(a))
            for t, q, dq, a in zip(seg.t, seg.q, seg.dq, seg.ddq, strict=True)
        ]
        last = self.leg == len(self.plan.segments) - 1
        # the planner's predicted grasp centre = the core arm model's (0.5 thumb + 0.25 (index + middle)), the point r1d
        # measures the result with (planner 33b7fb0; `point_m` is the goal's own point, e.g. the pinch midpoint)
        pred = self.plan.predicted
        target = (pred.get("grasp_centre_m") or pred.get("point_m")) if last else None
        self.cur = TrajectoryRunner(
            self.l2.svc,
            self.group,
            _fixed(knots),
            side=self.arm,
            target_point=None if target is None else np.asarray(target, dtype=np.float64),
            guards=self._guards(seg.guards),
            plan_id=self.plan.plan_id,
            planned_on=self.planned_on,
            contact_ends_ok=self.contact_ends_ok and bool(seg.guards & 4),
        )
        refused = self.cur.start(now_ns)
        if refused is not None:
            return refused
        return None

    def tick(self, now_ns: int) -> str | None:
        import time

        if self.end_state is not None:
            return self.end_state
        if self.plan is None:
            assert self.future is not None
            if not self.future.done():
                if time.monotonic() - self.wall0 > PLAN_WAIT_S:
                    self.end("aborted", Code.TIMEOUT, {"code": "PLAN_TIMEOUT", "detail": "the planner did not answer"})
                return self.end_state
            refused = self._take(self.future.result(), now_ns)
            if refused is not None:
                self.end("aborted", refused.code, {"code": "PLAN_REFUSED", "detail": refused.details})
            return self.end_state
        assert self.cur is not None
        st = self.cur.tick(now_ns)
        if st is None:
            return None
        if st == "succeeded" and self.cur.stopped_by is None and self.leg < len(self.plan.segments) - 1:
            refused = self._next_leg(now_ns)
            if refused is not None:
                self.end("aborted", refused.code, {"code": "LEG_REFUSED", "detail": refused.details})
            return self.end_state
        self.end(st, self.cur.code, self.cur.stopped_by)
        return self.end_state

    def stop(self, now_ns: int, why: str) -> None:
        if self.end_state is not None:
            return
        if self.cur is not None:
            self.cur.stop(now_ns, why)
        self.end_state = why
        self.code = {"preempted": Code.PREEMPTED, "timed_out": Code.TIMEOUT}.get(why, Code.CANCELED)
        self.stopped_by = {"code": why.upper(), "detail": "r1d stopped the goal"}

    def end(self, state: str, code: Code, stopped_by: dict[str, Any] | None) -> None:
        if self.end_state is None:
            self.end_state, self.code, self.stopped_by = state, code, stopped_by
            if self.cur is not None:
                self.cur.end(state, code, stopped_by)

    def outcome(self) -> tuple[Code, dict[str, Any], dict[str, Any]]:
        if self.cur is None or not self.cur.knots:
            d: dict[str, Any] = {"moved": False, "stopped_by": self.stopped_by, "planned_on": self.planned_on,
                                 "plan_id": None if self.plan is None else self.plan.plan_id}  # fmt: skip
            if self.plan is not None and not self.plan.ok:
                d.update({"reason": self.plan.reason, "plan_details": self.plan.details})
            return self.code, d, {"stopped_by": self.stopped_by, **({"reason": self.plan.reason}
                                                                     if self.plan is not None else {})}  # fmt: skip
        self.cur.code, self.cur.stopped_by = self.code, self.stopped_by
        code, data, details = self.cur.outcome()
        if self.p_start is not None:
            data["hand_start"] = [round(float(x), 5) for x in self.p_start[SIDE_INDEX[self.arm], POINT_GRASP_CENTRE]]
        data.update({
            "legs": [s.leg for s in self.plan.segments],
            "legs_done": self.leg + (1 if self.end_state == "succeeded" else 0),
            "predicted": self.plan.predicted,
            "clearance_m": self.plan.clearance_m,
            "planning_s": round(float(self.plan.planning_s), 4),
            "guards_dropped": self.dropped,
            "plan_details": dict(self.plan.details),  # session 10: e.g. the out-and-around's keyframes, a left start
        })  # fmt: skip
        return code, data, details


def steps_resources(steps: list[dict[str, Any]]) -> frozenset[str]:
    """The resources a sequence's steps will use (a sequence holds them all, so nobody slips in between steps)."""
    out: set[str] = set()
    for s in steps:
        op = str(s.get("op", ""))
        if op.startswith("arm.") and s.get("arm") in ARM_GROUPS:
            out.add(ARM_GROUPS[str(s["arm"])])
        elif op.startswith("hand.") and s.get("side") in HAND_RESOURCES:
            out.add(HAND_RESOURCES[str(s["side"])])
        elif op == "head.look_at":
            out.add("head")
        elif op.startswith("skill."):
            arms = s.get("arms", "both")
            out |= {ARM_GROUPS[a] for a in (("left", "right") if arms == "both" else (arms,)) if a in ARM_GROUPS}
    return frozenset(out)


class SeqRunner:
    """seq.run / the L3 skills: the steps one after another through r1d's own request path (each goal step is a child
    goal of this one; a non-goal step's result is taken at once). The first failed step ends the sequence (`fail_code`
    STEP_FAILED / SKILL_FAILED with the step's index and result) unless stop_on_failure is False."""

    def __init__(
        self,
        l2: L2,
        op: str,
        steps: list[dict[str, Any]],
        stop_on_failure: bool,
        fail_code: Code,
        *,
        precheck: list[dict[str, Any]] | None = None,
    ) -> None:
        self.l2, self.op, self.steps = l2, op, steps
        self.stop_on_failure, self.fail_code = stop_on_failure, fail_code
        self.precheck = list(precheck or [])  # dry runs that must all pass before the first step (skills)
        self.pre_tickets: list[Any] = []
        self.pre_results: list[dict[str, Any]] = []
        self.goal_id = 0
        self.i = -1
        self.child: int | None = None
        self.pending: Any = None  # a non-goal step's Ticket still running (e.g. a mode sequence)
        self.results: list[dict[str, Any]] = []
        self.end_state: str | None = None
        self.code = Code.OK
        self.stopped_by: dict[str, Any] | None = None
        self.failed: int | None = None

    def ready(self, now_ns: int) -> bool:
        return True

    def start(self, now_ns: int) -> Result | None:
        if self.precheck:
            self.pre_tickets = [self.l2.svc.submit(req) for req in self.precheck]
            self._precheck_done(now_ns)  # inline planners answer at once; a worker's answers come in tick()
            return None
        self._next(now_ns)
        return None

    def _precheck_done(self, now_ns: int) -> bool | None:
        """None = dry runs still out; True = all passed (the steps start); False = one refused (the skill ends)."""
        if not all(tk.done.is_set() for tk in self.pre_tickets):
            return None
        results = [tk.result for tk in self.pre_tickets]
        self.pre_tickets = []
        self.precheck = []
        for k, res in enumerate(results):
            self.pre_results.append({"index": k, "ok": res.ok, "code": str(res.code), "details": res.details})
            if not res.ok:
                self.failed = k
                self.end_state, self.code = "aborted", self.fail_code
                self.stopped_by = {"code": "PRECHECK", "detail": f"step {k} refused before anything moved: {res.code}"}
                self.results.append({"index": k, "op": self.steps[k].get("op"), "ok": False, "code": str(res.code),
                                     "state": "rejected", "goal_id": None, "moved": False, "data": res.data,
                                     "details": res.details})  # fmt: skip
                return False
        self._next(now_ns)
        return True

    def _record(self, res: Result) -> None:
        self.results.append({"index": self.i, "op": res.op, "ok": res.ok, "code": str(res.code),
                             "state": res.data.get("state"), "goal_id": res.data.get("goal_id"),
                             "moved": res.moved, "data": res.data, "details": res.details})  # fmt: skip
        if not res.ok and self.failed is None:
            self.failed = self.i

    def _next(self, now_ns: int) -> None:
        while True:
            if self.failed is not None and self.stop_on_failure:
                self.end_state, self.code = "aborted", self.fail_code
                self.stopped_by = {"code": self.fail_code.value, "detail": f"step {self.failed} failed"}
                return
            self.i += 1
            if self.i >= len(self.steps):
                self.end_state = "succeeded" if self.failed is None else "aborted"
                self.code = Code.OK if self.failed is None else self.fail_code
                return
            step = {k: v for k, v in self.steps[self.i].items() if k not in ("wait", "id")}
            self.l2._parent_ctx = self.goal_id
            try:
                tk = self.l2.svc.submit(step)
            finally:
                self.l2._parent_ctx = None
            res = tk.result
            if res is None:
                self.pending = tk
                return
            if res.ok and res.data.get("state") == "accepted" and "goal_id" in res.data:
                self.child = int(res.data["goal_id"])
                return
            self._record(res)

    def tick(self, now_ns: int) -> str | None:
        if self.end_state is not None:
            return self.end_state
        if self.pre_tickets:
            self._precheck_done(now_ns)
            return self.end_state
        if self.pending is not None:
            if not self.pending.done.is_set():
                return None
            res = self.pending.result
            self.pending = None
            self._record(res)
            self._next(now_ns)
        elif self.child is not None:
            g = self.l2.table.goals.get(self.child)
            if g is None or g.result is None:
                return None
            self.child = None
            self._record(g.result)
            self._next(now_ns)
        return self.end_state

    def stop(self, now_ns: int, why: str) -> None:
        if self.end_state is not None:
            return
        if self.child is not None:
            g = self.l2.table.goals.get(self.child)
            if g is not None and g.state not in GOAL_TERMINAL:
                g.runner.stop(now_ns, why)
                self.l2._finish(g, now_ns)
        self.end_state = why
        self.code = {"preempted": Code.PREEMPTED, "timed_out": Code.TIMEOUT}.get(why, Code.CANCELED)
        self.stopped_by = {"code": why.upper(), "detail": f"r1d stopped the sequence at step {self.i}"}

    def end(self, state: str, code: Code, stopped_by: dict[str, Any] | None) -> None:
        if self.end_state is None:
            self.end_state, self.code, self.stopped_by = state, code, stopped_by

    def outcome(self) -> tuple[Code, dict[str, Any], dict[str, Any]]:
        moved = any(r["moved"] for r in self.results)
        data = {"moved": moved, "steps": self.results, "n_steps": len(self.steps), "failed_step": self.failed,
                "stopped_by": self.stopped_by, "precheck": self.pre_results}  # fmt: skip
        details: dict[str, Any] = {}
        if self.failed is not None:
            details = {"failed_step": self.failed, "step_result": self.results[-1] if self.results else None}
        return self.code, data, details


def support_top_below(snapshot: Any, point: Any) -> float | None:
    """The highest support top (a surface's top face, a container's floor) below `point` (base frame) whose
    footprint holds it. Structurally typed over the world snapshot's entities (r1mc3.world: `entities` with `kind`,
    `pose` (position [3] + optional quaternion), `size`; a container's `floor_m`)."""
    ents = getattr(snapshot, "entities", None)
    if ents is None:
        return None
    if isinstance(ents, dict):
        ents = list(ents.values())
    p = np.asarray(point, dtype=np.float64)
    best: float | None = None
    for e in ents:
        kind = str(getattr(e, "kind", type(e).__name__)).lower()
        if kind not in ("surface", "container"):
            continue
        pose = getattr(e, "pose", None)
        size = getattr(e, "size", None) or getattr(e, "size_outer", None)
        if pose is None or size is None:
            continue
        pos = np.asarray(getattr(pose, "pos", None) if hasattr(pose, "pos") else pose[:3], dtype=np.float64)
        sz = np.asarray(size, dtype=np.float64)
        if abs(p[0] - pos[0]) > sz[0] / 2 or abs(p[1] - pos[1]) > sz[1] / 2:
            continue  # (axis-aligned footprint in the base frame: the declared yaw is ignored here - small objects)
        if kind == "surface":
            top = float(pos[2] + sz[2] / 2)
        else:
            top = float(pos[2] - sz[2] / 2 + float(getattr(e, "floor_m", 0.0)))
        if top < p[2] and (best is None or top > best):
            best = top
    return best


def group_joints(group: str) -> list[int]:
    return list(GROUP_JOINTS[group])
