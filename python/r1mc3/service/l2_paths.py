"""Session 10 L2 / L3 ops of r1d (DESIGN.md §5.1): arm.move_path, arm.guarded_move, arm.slide, skill.pick,
skill.place, par.run - built ONLY on the session-9 machinery (l2.py's goal table, PlannedRunner, SeqRunner, the
planner `r1mc3.plan`), nothing new in the core.

`Service._dispatch` hands these request types here (`handles` / `dispatch`, the api agent's marker block); the same
refusal shape as l2.py (`data.state = "rejected"`, moved False). Generic (D32): no task objects or layouts - the
client declares its world and names the points.

What each op is (and is not):
* arm.move_path - the waypoints as consecutive planner goals (`reach`, or `move_linear` with `linear`), leg k+1
  planned from leg k's planned END (the arm's joints replaced by its last knot) so the WHOLE path is checked before
  anything moves (AP-3, as hover / carry); the legs then run in order, each one TRAJECTORY ending at rest - or, with
  `blend_m` > 0 (session 11), the checked legs become ONE trajectory that does not stop at the interior waypoints,
  each passed within blend_m (r1mc3.plan.blend: corners cut inside that ball, the executed curve re-checked with the
  planner's clearance before it is used; a blend that does not pass falls back to a smaller corner, then to the
  stops - `data.blend` says which).
* arm.guarded_move - a straight line (planner `move_linear`) whose end by a named guard is the goal's SUCCESS
  (`stopped_early`): `contact` = the executor's pad-contact guard (armed on the line; refused without r1hands'
  pad-contact publisher, as arm.lower), `blocked` = the lag / torque guard. A guard always stops the arm; only the
  result's meaning changes. `touch` passes the planner's allow_hand (the hand may touch those declared entities).
* arm.slide - a straight line along the declared support under the hand with that support in allow_hand on every
  leg (the planner otherwise lets a hand touch what it starts on during the FIRST leg only); refused when no declared
  support lies within `max_gap_m` under the lowest fingertip. No press.
* skill.pick / skill.place - seq.run's machinery (SeqRunner, SKILL_FAILED) over the L2 goals above, with the first
  planned step dry-run before anything moves (v2's skills rule).
* par.run - goal requests started together as child goals of one parent that holds all their resources (disjoint,
  checked); ends when all have ended.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .codes import Code
from .goals import ARM_GROUPS, POINT_GRASP_CENTRE, SIDE_INDEX, hand_points
from .l2 import PlannedRunner, SeqRunner, plan_inputs, steps_resources, support_top_below
from .schemas import (
    GOAL_OPS,
    GOAL_TERMINAL,
    ArmGuardedMoveReq,
    ArmMovePathReq,
    ArmSlideReq,
    ParRunReq,
    Result,
    SkillPickReq,
    SkillPlaceReq,
    fail,
)

if TYPE_CHECKING:
    from .l2 import L2
    from .service import Ticket

GUARD_CONTACT_BIT = 4  # r1rt_traj_guards (EXECUTOR.md; r1mc3.plan.core GUARD_CONTACT)
BLOCKED_CODES = ("GUARD_LAG", "GUARD_TORQUE")  # the executor events that mean "held back" (goals.EVENT_NAMES upper)
TYPES = (ArmMovePathReq, ArmGuardedMoveReq, ArmSlideReq, SkillPickReq, SkillPlaceReq, ParRunReq)


def handles(r: Any) -> bool:
    return isinstance(r, TYPES)


def dispatch(l2: L2, r: Any, t: Ticket) -> Result | None:
    table: dict[str, Callable[..., Result | None]] = {
        "arm.move_path": arm_move_path,
        "arm.guarded_move": arm_guarded_move,
        "arm.slide": arm_slide,
        "skill.pick": skill_pick,
        "skill.place": skill_place,
        "par.run": par_run,
    }
    out = table[r.op](l2, r, t)
    if out is not None and not out.ok and "state" not in out.data:
        out = out.model_copy(update={"data": {**out.data, "state": "rejected"}})  # refused before it was a goal
    return out


# ---------------------------------------------------------------------------------------------- helpers
def _arm_ready(l2: L2, r: Any) -> Result | None:
    svc = l2.svc
    if svc.planner is None:
        return fail(r.op, r.id, Code.UNSUPPORTED, reason="no planner in this r1d (r1mc3.plan.PlannerWorker)")
    chk = svc._group_check(r.op, r.id, ARM_GROUPS[r.arm])
    return chk if isinstance(chk, Result) else None


def support_under(snapshot: Any, point: Any, max_gap_m: float) -> tuple[str, float] | None:
    """The declared support (a surface's top face, a container's floor) whose footprint holds `point` (base frame)
    and whose top lies at most `max_gap_m` below it (or up to 5 mm above it: a fingertip resting on it), the highest
    first -> (entity id, top z). Structurally typed like l2.support_top_below (axis-aligned footprints)."""
    ents = getattr(snapshot, "entities", None) or ()
    if isinstance(ents, dict):
        ents = list(ents.values())
    p = np.asarray(point, dtype=np.float64)
    best: tuple[str, float] | None = None
    for e in ents:
        kind = str(getattr(e, "kind", "")).lower()
        if kind not in ("surface", "container"):
            continue
        pose = getattr(e, "pose", None)
        size = getattr(e, "size", None) or getattr(e, "size_outer", None)
        if pose is None or size is None:
            continue
        pos = np.asarray(pose.pos if hasattr(pose, "pos") else pose[:3], dtype=np.float64)
        sz = np.asarray(size, dtype=np.float64)
        if abs(p[0] - pos[0]) > sz[0] / 2 or abs(p[1] - pos[1]) > sz[1] / 2:
            continue
        top = float(pos[2] + sz[2] / 2) if kind == "surface" else float(pos[2] - sz[2] / 2 + float(e.floor_m))
        if -0.005 <= p[2] - top <= max_gap_m and (best is None or top > best[1]):
            best = (str(e.id), top)
    return best


def _lowest_tip(l2: L2, arm: str) -> np.ndarray | None:
    """The hand's lowest fingertip (base frame) from the measured joints (the core's arm model)."""
    pts = hand_points(l2.svc.port, l2.svc.hands)
    if pts is None:
        return None
    tips = np.asarray(pts[SIDE_INDEX[arm], 1:4], dtype=np.float64)  # thumb / middle / index tips (goals.tips_dict)
    low: np.ndarray = tips[int(np.argmin(tips[:, 2]))]
    return low


# ---------------------------------------------------------------------------------------------- arm.move_path
class _PlanChain:
    """A Future-like chain of planner goals: goal k+1 planned from goal k's planned end (the arm's joints = its last
    knot, the hand where it will be) - the whole path validated before anything moves. `done()` advances the chain
    (it submits the next goal when the previous plan is back), `result()` = one PlanResult with every leg."""

    def __init__(self, runner: PathRunner, goals: list[dict[str, Any]]) -> None:
        from r1mc3.world.geometry import ARM_JOINTS

        self.runner, self.goals = runner, goals
        self.joints = list(ARM_JOINTS[runner.arm])
        self.results: list[Any] = []
        self.fut: Any = None
        self.final: Any = None
        inputs = plan_inputs(runner.l2, runner.arm)
        if isinstance(inputs, Result):
            raise _Refused(inputs)
        self.q26, self.quat, self.hands, self.target = inputs
        self.snap = runner.l2.world_model().snapshot()
        self.kw: dict[str, Any] = {"imu_quat": self.quat, "hands": self.hands}
        if runner.l2.svc.mode_name() == "balance":  # the policy's trained arm range (as l2.submit_plan)
            self.kw["envelope"] = runner.l2.policy_envelope(runner.arm)
        self.blending = False
        self._submit()

    def _submit(self) -> None:
        from r1mc3.plan import PlanGoal

        k = len(self.results)
        q26 = np.asarray(self.q26, dtype=np.float64).copy()
        target = self.target
        if k > 0:
            target = np.asarray(self.results[-1].segments[-1].q[-1], dtype=np.float64)
            q26[self.joints] = target
        pg = PlanGoal.from_dict(self.goals[k])
        planner: Any = self.runner.l2.svc.planner
        kw = {**self.kw, "arm_target": target}
        self.runner.wall0 = time.monotonic()  # PLAN_WAIT_S counts per leg
        if hasattr(planner, "submit"):
            self.fut = planner.submit(self.snap, pg, q26, **kw)
            return
        fn = planner.plan if hasattr(planner, "plan") else planner
        self._take(fn(self.snap, pg, q26, **kw))

    def _take(self, res: Any) -> None:
        self.fut = None
        if self.blending:  # the blended plan (or the joined one with the reason it was not blended)
            self.final = res
            return
        self.results.append(res)
        if not res.ok or len(self.results) == len(self.goals):
            joined = _joined(self.results, len(self.goals))
            if joined.ok and self.runner.blend_m > 0.0 and len(self.goals) > 1:
                self._blend(joined)
                return
            self.final = joined
            return
        self._submit()

    def _blend(self, joined: Any) -> None:
        """Session 11: the checked legs -> one trajectory through the waypoints (r1mc3.plan.blend), computed where the
        planner runs (its worker: the compiled geometry lives there; inline otherwise)."""
        from r1mc3.plan.blend import blend_path

        self.blending = True
        args = (self.snap, joined, np.asarray(self.q26, dtype=np.float64))
        kw = {"arm": self.runner.arm, "radius_m": self.runner.blend_m, "speed": self.runner.speed,
              "imu_quat": self.quat, "hands": self.hands, "envelope": self.kw.get("envelope"),
              "arm_target": None if self.target is None else np.asarray(self.target, dtype=np.float64)}  # fmt: skip
        planner: Any = self.runner.l2.svc.planner
        self.runner.wall0 = time.monotonic()
        if hasattr(planner, "call"):
            self.fut = planner.call(blend_path, *args, **kw)
            return
        self._take(blend_path(*args, **kw))

    def done(self) -> bool:
        while self.final is None and self.fut is not None and self.fut.done():
            self._take(self.fut.result())
        return self.final is not None

    def result(self) -> Any:
        return self.final


class _Refused(Exception):
    def __init__(self, res: Result) -> None:
        super().__init__(str(res.details))
        self.res = res


def _joined(results: list[Any], n_goals: int) -> Any:
    """One PlanResult for the whole path (the planner's dataclass): every leg's segments in order (named
    `w<k>:<leg>`), the last waypoint's prediction, the smallest clearance; a refused waypoint refuses the path (its
    code, `waypoint` = its index, nothing planned after it)."""
    from r1mc3.plan import PlanResult

    last = results[-1]
    plan_id = "path:" + "+".join(str(r.plan_id) for r in results)
    planning_s = float(sum(float(r.planning_s) for r in results))
    if not last.ok:
        k = len(results) - 1
        return PlanResult(False, last.code, plan_id, reason=f"waypoint {k}: {last.reason}",
                          details={**dict(last.details), "waypoint": k, "waypoints": n_goals},
                          planning_s=planning_s)  # fmt: skip
    segs = tuple(replace(s, leg=f"w{k}:{s.leg}") for k, r in enumerate(results) for s in r.segments)
    clr = [r.clearance_m for r in results if r.clearance_m is not None]
    pred = {**last.predicted, "waypoints_grasp_centre_m": [r.predicted.get("grasp_centre_m") for r in results]}
    return PlanResult(True, "OK", plan_id, segs, pred, min(clr) if clr else None, "",
                      {"waypoints": n_goals, "legs": [s.leg for s in segs]}, planning_s)  # fmt: skip


class PathRunner(PlannedRunner):
    """arm.move_path: PlannedRunner with the plan = the chained waypoints (the legs run as PlannedRunner runs any
    multi-leg plan: each leg one TRAJECTORY, the next on the executor's DONE)."""

    def __init__(
        self, l2: L2, op: str, arm: str, goals: list[dict[str, Any]], *, blend_m: float = 0.0, speed: float = 1.0
    ) -> None:
        super().__init__(l2, op, arm, goals[-1])
        self.goals = goals
        self.blend_m, self.speed = float(blend_m), float(speed)

    def start(self, now_ns: int) -> Result | None:
        try:
            chain = _PlanChain(self, self.goals)
        except _Refused as e:
            return e.res
        self.p_start = hand_points(self.l2.svc.port, self.l2.svc.hands)
        self.t0 = now_ns
        if chain.done():
            return self._take(chain.result(), now_ns)
        self.future = chain
        return None

    def _take(self, res: Any, now_ns: int) -> Result | None:
        out = super()._take(res, now_ns)
        if out is not None and not res.ok and "waypoint" in res.details:  # which waypoint refused, at the top
            out = out.model_copy(update={"details": {**out.details, "waypoint": res.details["waypoint"],
                                                     "waypoints": len(self.goals)}})  # fmt: skip
        return out

    def outcome(self) -> tuple[Code, dict[str, Any], dict[str, Any]]:
        code, data, details = super().outcome()
        data["waypoints"] = len(self.goals)
        if self.plan is not None and self.plan.ok:
            done = data.get("legs_done", 0)
            legs = [s.leg for s in self.plan.segments]
            data["waypoints_done"] = (
                len({lg.split(":", 1)[0] for lg in legs[:done]}) if done < len(legs) else len(self.goals)
            )
        if self.blend_m > 0.0 and self.plan is not None:  # what the blend did (or why the hand stops at each waypoint)
            blend = dict(self.plan.details).get("blend") or {}
            data["blend"] = blend or None
            if blend.get("applied") and data.get("legs_done", 0) < 1 and data.get("achieved_q") is not None:
                # stopped inside the one trajectory: the waypoints its curve had passed where the arm stopped
                seg = self.plan.segments[0]
                k = int(np.argmin(np.max(np.abs(np.asarray(seg.q) - np.asarray(data["achieved_q"])), axis=1)))
                data["waypoints_done"] = sum(1 for t in blend.get("pass_t_s", []) if t <= float(seg.t[k]))
        return code, data, details


def arm_move_path(l2: L2, r: ArmMovePathReq, t: Ticket) -> Result | None:
    bad = _arm_ready(l2, r)
    if bad is not None:
        return bad
    kind = "move_linear" if r.linear else "reach"
    goals = [{"kind": kind, "arm": r.arm, "position": list(w), "frame": r.frame, "speed": r.speed}
             for w in r.waypoints]  # fmt: skip
    runner = PathRunner(l2, r.op, r.arm, goals, blend_m=r.blend_m, speed=r.speed)
    return l2.accept(r, t, frozenset([ARM_GROUPS[r.arm]]), runner)


# ---------------------------------------------------------------------------------------------- arm.guarded_move
class GuardedRunner(PlannedRunner):
    """A planned line whose end by one of the named guards is a success (`stopped_early`): the contact guard armed on
    every leg when `contact` is named; a lag / torque stop counted as success when `blocked` is named."""

    def __init__(self, l2: L2, op: str, arm: str, goal: dict[str, Any], until: list[str]) -> None:
        super().__init__(l2, op, arm, goal, contact_ends_ok="contact" in until)
        self.until = list(until)
        self.stopped_early = False

    def _take(self, res: Any, now_ns: int) -> Result | None:
        if res.ok and "contact" in self.until:
            res = replace(res, segments=tuple(replace(s, guards=s.guards | GUARD_CONTACT_BIT) for s in res.segments))
        return super()._take(res, now_ns)

    def end(self, state: str, code: Code, stopped_by: dict[str, Any] | None) -> None:
        sb = (stopped_by or {}).get("code")
        if state == "aborted" and code == Code.GUARD_STOP and "blocked" in self.until and sb in BLOCKED_CODES:
            state, code = "succeeded", Code.OK
            self.stopped_early = True
        elif state == "succeeded" and sb == "GUARD_CONTACT":
            self.stopped_early = True
        super().end(state, code, stopped_by)

    def outcome(self) -> tuple[Code, dict[str, Any], dict[str, Any]]:
        code, data, details = super().outcome()
        sb = (self.stopped_by or {}).get("code")
        data.update({"stopped_early": self.stopped_early, "until": self.until,
                     "touched": self.stopped_early and sb == "GUARD_CONTACT",
                     "blocked": self.stopped_early and sb in BLOCKED_CODES})  # fmt: skip
        return code, data, details


def arm_guarded_move(l2: L2, r: ArmGuardedMoveReq, t: Ticket) -> Result | None:
    bad = _arm_ready(l2, r)
    if bad is not None:
        return bad
    if "contact" in r.until and l2.svc.contact is None:
        return fail(r.op, r.id, Code.UNSUPPORTED, reason="until 'contact' needs the pads' contact publisher (r1hands) "
                    "- none in this r1d; until ['blocked'] runs without it")  # fmt: skip
    goal: dict[str, Any] = {"kind": "move_linear", "arm": r.arm, "frame": r.frame, "speed": r.speed,
                            "allow_hand": list(r.touch)}  # fmt: skip
    if r.stop_z_m is not None:
        pts = hand_points(l2.svc.port, l2.svc.hands)
        if pts is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="no measured hand for stop_z_m")
        depth = float(pts[SIDE_INDEX[r.arm], POINT_GRASP_CENTRE, 2]) - r.stop_z_m
        if not 0.0 < depth <= 0.4:
            return fail(r.op, r.id, Code.LIMIT, reason="stop_z_m: 0 < the depth below the grasp centre <= 0.4 m",
                        depth_m=round(depth, 4))  # fmt: skip
        goal["delta"] = [0.0, 0.0, -depth]
    elif r.delta is not None:
        goal["delta"] = list(r.delta)
    else:
        goal["position"] = list(r.position or ())
    return l2.accept(r, t, frozenset([ARM_GROUPS[r.arm]]), GuardedRunner(l2, r.op, r.arm, goal, list(r.until)))


# ---------------------------------------------------------------------------------------------- arm.slide
def arm_slide(l2: L2, r: ArmSlideReq, t: Ticket) -> Result | None:
    bad = _arm_ready(l2, r)
    if bad is not None:
        return bad
    tip = _lowest_tip(l2, r.arm)
    if tip is None:
        return fail(r.op, r.id, Code.UNSUPPORTED, reason="no measured hand (the core's arm model)")
    snap = l2.world_model().snapshot()
    sup = support_under(snap, tip, r.max_gap_m)
    if r.support is not None:
        ids = {str(e.id) for e in snap.entities}
        if r.support not in ids:
            return fail(r.op, r.id, Code.UNKNOWN_FRAME, reason=f"support {r.support!r} is not a declared entity")
        if sup is None or sup[0] != r.support:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason=f"the hand is not on / within {r.max_gap_m} m above "
                        f"{r.support!r}", lowest_tip=[round(float(x), 4) for x in tip])  # fmt: skip
    if sup is None:
        return fail(r.op, r.id, Code.INVALID_ARGS, reason=f"no declared support within {r.max_gap_m} m under the "
                    "hand's lowest fingertip (lower the hand onto it first)",
                    lowest_tip=[round(float(x), 4) for x in tip])  # fmt: skip
    goal = {"kind": "move_linear", "arm": r.arm, "frame": r.frame, "speed": r.speed, "delta": list(r.delta),
            "allow_hand": [sup[0]]}  # fmt: skip
    runner = GuardedRunner(l2, r.op, r.arm, goal, [])
    out = l2.accept(r, t, frozenset([ARM_GROUPS[r.arm]]), runner)
    if out is not None and out.ok:
        out = out.model_copy(update={"data": {**out.data, "support": sup[0], "support_top_z_m": round(sup[1], 4)}})
    return out


# ---------------------------------------------------------------------------------------------- skill.pick / place
def _skill(l2: L2, r: Any, t: Ticket, steps: list[dict[str, Any]], resources: frozenset[str]) -> Result | None:
    """A generic L3 skill = SeqRunner over L2 goal steps (SKILL_FAILED on the first failure), the first planned step
    dry-run before anything moves (v2's rule: a refused first step ends the skill with nothing moved)."""
    bad = _arm_ready(l2, r)
    if bad is not None:
        return bad
    pre = [{"op": "arm.plan", "goal": steps[0]}] if steps[0]["op"] in ("arm.hover", "arm.carry") else []
    runner = SeqRunner(l2, r.op, steps, True, Code.SKILL_FAILED, precheck=pre)
    return l2.accept(r, t, resources, runner)


def skill_pick(l2: L2, r: SkillPickReq, t: Ticket) -> Result | None:
    from r1mc3.skills.pick import pick_steps

    snap = l2.world_model().snapshot()
    centre = np.asarray(r.position, dtype=np.float64)
    top = support_top_below(snap, centre)
    if top is None or top <= 0.0:
        top = float(centre[2] - r.height_m / 2)  # no declared support: the object's own bottom (the hover datum)
    steps = pick_steps(r.model_dump(), support_top_z=top)
    res = frozenset([ARM_GROUPS[r.arm], f"hand_{r.arm}"])
    return _skill(l2, r, t, steps, res)


def skill_place(l2: L2, r: SkillPlaceReq, t: Ticket) -> Result | None:
    from r1mc3.skills.pick import place_steps

    steps = place_steps(r.model_dump())
    res = frozenset([ARM_GROUPS[r.arm], f"hand_{r.arm}"])
    return _skill(l2, r, t, steps, res)


# ---------------------------------------------------------------------------------------------- par.run
class ParRunner:
    """par.run: every step submitted at the start as a child goal of this one (this goal holds all their
    resources, so the children pass the busy check); ends when every child has ended."""

    def __init__(self, l2: L2, op: str, steps: list[dict[str, Any]], stop_on_failure: bool) -> None:
        self.l2, self.op, self.steps, self.stop_on_failure = l2, op, steps, stop_on_failure
        self.goal_id = 0
        self.children: list[int | None] = []
        self.results: list[dict[str, Any] | None] = []
        self.end_state: str | None = None
        self.code = Code.OK
        self.stopped_by: dict[str, Any] | None = None
        self.failed: int | None = None

    def ready(self, now_ns: int) -> bool:
        return True

    def start(self, now_ns: int) -> Result | None:
        for k, st in enumerate(self.steps):
            step = {kk: v for kk, v in st.items() if kk not in ("wait", "id")}
            self.l2._parent_ctx = self.goal_id
            try:
                res = self.l2.svc.submit(step).result
            finally:
                self.l2._parent_ctx = None
            if res is not None and res.ok and res.data.get("state") == "accepted" and "goal_id" in res.data:
                self.children.append(int(res.data["goal_id"]))
                self.results.append(None)
                continue
            self.children.append(None)
            self.results.append(_step_record(k, res))
            if self.failed is None:
                self.failed = k
        if self.failed is not None and self.stop_on_failure:
            self._cancel_children(now_ns, "canceled")
        return None

    def _cancel_children(self, now_ns: int, why: str) -> None:
        for k, gid in enumerate(self.children):
            g = None if gid is None else self.l2.table.goals.get(gid)
            if g is not None and g.state not in GOAL_TERMINAL:
                g.runner.stop(now_ns, why)
                self.l2._finish(g, now_ns)
            if g is not None and g.result is not None and self.results[k] is None:
                self.results[k] = _step_record(k, g.result)

    def tick(self, now_ns: int) -> str | None:
        if self.end_state is not None:
            return self.end_state
        for k, gid in enumerate(self.children):
            if gid is None or self.results[k] is not None:
                continue
            g = self.l2.table.goals.get(gid)
            if g is None or g.result is None:
                continue
            self.results[k] = _step_record(k, g.result)
            if not g.result.ok and self.failed is None:
                self.failed = k
                if self.stop_on_failure:
                    self._cancel_children(now_ns, "canceled")
        if all(x is not None for x in self.results):
            self.end_state = "succeeded" if self.failed is None else "aborted"
            self.code = Code.OK if self.failed is None else Code.STEP_FAILED
            if self.failed is not None:
                self.stopped_by = {"code": "STEP_FAILED", "detail": f"step {self.failed} failed"}
        return self.end_state

    def stop(self, now_ns: int, why: str) -> None:
        if self.end_state is not None:
            return
        self._cancel_children(now_ns, why)
        self.end_state = why
        self.code = {"preempted": Code.PREEMPTED, "timed_out": Code.TIMEOUT}.get(why, Code.CANCELED)
        self.stopped_by = {"code": why.upper(), "detail": "r1d stopped the parallel goals"}

    def end(self, state: str, code: Code, stopped_by: dict[str, Any] | None) -> None:
        if self.end_state is None:
            self.end_state, self.code, self.stopped_by = state, code, stopped_by

    def outcome(self) -> tuple[Code, dict[str, Any], dict[str, Any]]:
        rows = [x for x in self.results if x is not None]
        data = {"moved": any(x["moved"] for x in rows), "steps": rows, "n_steps": len(self.steps),
                "failed_step": self.failed, "stopped_by": self.stopped_by}  # fmt: skip
        details = {} if self.failed is None else {"failed_step": self.failed, "step_result": self.results[self.failed]}
        return self.code, data, details


def _step_record(k: int, res: Result | None) -> dict[str, Any]:
    if res is None:
        return {"index": k, "ok": False, "code": "TIMEOUT", "state": None, "goal_id": None, "moved": False,
                "data": {}, "details": {}}  # fmt: skip
    return {"index": k, "op": res.op, "ok": res.ok, "code": str(res.code), "state": res.data.get("state"),
            "goal_id": res.data.get("goal_id"), "moved": res.moved, "data": res.data,
            "details": res.details}  # fmt: skip


def par_run(l2: L2, r: ParRunReq, t: Ticket) -> Result | None:
    per = []
    for k, s in enumerate(r.steps):
        if str(s.get("op", "")) not in GOAL_OPS:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason=f"step {k}: par.run runs goal ops only", step=s.get("op"))
        res = steps_resources([s])
        if not res:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason=f"step {k}: no resource (arm / hand / head) named")
        per.append(res)
    for i in range(len(per)):
        for j in range(i + 1, len(per)):
            if per[i] & per[j]:
                return fail(r.op, r.id, Code.INVALID_ARGS, reason=f"steps {i} and {j} share {sorted(per[i] & per[j])}"
                            " (par.run needs disjoint resources; use seq.run)")  # fmt: skip
    resources = frozenset().union(*per)
    return l2.accept(r, t, resources, ParRunner(l2, r.op, list(r.steps), r.stop_on_failure))
