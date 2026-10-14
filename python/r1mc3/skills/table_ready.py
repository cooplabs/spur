"""`skill.table_ready` from ANY start, incl. the hands hanging under a declared table (session 10; the user, 11 Oct
~23:00: a client's "reset the robot to ready" and every episode start need it).

v2's skill (r1mc2/skills.py table_ready, git r21_try 2a4a165 skills.py:587-760) under a manipulation policy: both arms
planned before anything moves, each again just before it moves (the first arm up shifts the body), one arm at a time;
a hand under the table's near edge goes out to the side and around (r1mc2/outaround.py: the way a person moved the arms
on the real robot), its fingers CLOSED while it comes out (v2: closure 0.6, the operator's console 0.9 - "a more closed,
smaller hand" cleared the edge where 0.6 did not, offboard.py:16), set back as they were once the hand is up. v3: the
planner routes a move_joint whose hand starts under a declared surface out and around (r1mc3.plan.core._out_around,
every part 2 cm from the world); this runner adds the order and the fingers:

  precheck  every arm planned on the current world (the hand as it is; when that is refused or goes out and around
            and r1d has hands: planned again with the hand closed - `fist`, the smallest Dex3 shape); a refusal ends the
            skill with NOTHING moved (SKILL_FAILED, stopped_by PRECHECK)
  per arm   [fingers closed] -> the arm (planned again now, with the hand as closed) -> [fingers back as they were]

Generic: no task names; the surface is whatever the client declared (D32). Without hands (r1d has no r1hands) the arm
plans with the hand as measured / open and the fingers are not touched.

Session 13 (HW2 day 2: a client's reset after a teleop run was REFUSED in planning - "COLLISION: at the left arm's
target its index would touch the right hand's index": the right hand sat over the board's LEFT side; done by hand,
right then left): with both arms, an arm the precheck refuses is planned once more AS IF the other arm were already at
its posture; when that clears, the other arm goes FIRST (`order` / `reordered` in the result). Each arm is still planned
again just before it moves (the first arm up shifts the body), and nothing moves unless both plans cleared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .postures import posture

if TYPE_CHECKING:
    from r1mc3.service.l2 import L2

CLOSED_SHAPE = "fist"  # r1sim.dex3 PRESETS_LEFT["fist"]: the fingers curled (v2's closure 0.9 while coming out)
DEFER_MAX_NS = 3_000_000_000  # an inner runner starts at the latest this long after it was due (L2's own deferral)


class TableReadyRunner:
    """The goal runner of skill.table_ready / skill.ready (the Runner protocol of service/goals.py): composes the
    hand moves and the planned arm moves itself (the skill holds both arms and both hands)."""

    def __init__(self, l2: L2, op: str, name: str, arms: list[str], speed: float) -> None:
        self.l2, self.op, self.name, self.arms, self.speed = l2, op, name, arms, float(speed)
        self.goal_id = 0
        self.pre: dict[str, Any] = {}  # arm -> the precheck's pending plan (Future) / PlanResult
        self.pre_closed: dict[str, bool] = {}  # arm -> planned with the hand closed
        self.pre_results: list[dict[str, Any]] = []
        self.items: list[tuple[str, str, Any]] = []  # (kind "hand" | "arm", side, q / goal dict)
        self.i = -1
        self.cur: Any = None
        self.cur_started = False
        self.cur_due = 0
        self.results: list[dict[str, Any]] = []
        self.end_state: str | None = None
        self.code: Any = None
        self.stopped_by: dict[str, Any] | None = None
        self.failed: int | None = None
        self.hands0: dict[str, Any] = {}
        # session 13: the arm the precheck refused, planned again with the other arm at its posture (Future / result)
        self.alt: dict[str, Any] = {}
        self.alt_tried: set[str] = set()
        self.alt_closed: dict[str, bool] = {}
        self.reordered: str | None = None

    # ------------------------------------------------------------------ helpers
    def _goal(self, arm: str, closed: bool) -> dict[str, Any]:
        g: dict[str, Any] = {"kind": "move_joint", "arm": arm, "q": posture(self.name, arm), "speed": self.speed}
        if closed:
            g["hand"] = CLOSED_SHAPE
        return g

    def _plan(self, arm: str, closed: bool) -> Any:
        from r1mc3.service.l2 import submit_plan

        self.pre_closed[arm] = closed
        return submit_plan(self.l2, self._goal(arm, closed), arm)

    def _plan_other_moved(self, arm: str, closed: bool) -> Any:
        """Session 13: `arm` planned as if the OTHER arm were already at its posture (l2.submit_plan with that arm's
        measured joints replaced) -> the Future / PlanResult / a refusal Result."""
        from r1mc3.plan import PlanGoal
        from r1mc3.service.l2 import plan_inputs
        from r1mc3.service.schemas import Result
        from r1mc3.world.geometry import ARM_JOINTS

        inputs = plan_inputs(self.l2, arm)
        if isinstance(inputs, Result):
            return inputs
        q26, quat, hands, target = inputs
        other = "right" if arm == "left" else "left"
        q26 = np.asarray(q26, dtype=np.float64).copy()
        q26[ARM_JOINTS[other]] = posture(self.name, other)
        kw: dict[str, Any] = {"imu_quat": quat, "hands": hands, "arm_target": target}
        if self.l2.svc.mode_name() == "balance":  # as submit_plan: inside the balance policy's trained range
            kw["envelope"] = self.l2.policy_envelope(arm)
        snap = self.l2.world_model().snapshot()
        pg = PlanGoal.from_dict(self._goal(arm, closed))
        planner: Any = self.l2.svc.planner
        if hasattr(planner, "submit"):
            return planner.submit(snap, pg, q26, **kw)
        fn = planner.plan if hasattr(planner, "plan") else planner
        return fn(snap, pg, q26, **kw)

    def _try_other_first(self) -> bool:
        """Session 13: with both arms, exactly one refused by the precheck -> its plan with the other arm at its posture
        (once; again with the fingers closed when r1d has hands). Clears -> the other arm goes first. Returns True
        while such a plan is pending."""
        if len(self.arms) != 2:
            return False
        bad = [a for a in self.arms if not self._ok(self._value(self.pre[a]))]
        if len(bad) != 1:
            return False
        arm = bad[0]
        if arm not in self.alt:
            if arm in self.alt_tried:
                return False
            self.alt_tried.add(arm)
            if not self._submit_alt(arm, self.pre_closed[arm]):
                return False
        while True:
            p = self.alt[arm]
            if not self._done(p):
                return True
            res = self._value(p)
            if self._ok(res):
                other = "right" if arm == "left" else "left"
                first = self._value(self.pre[arm])
                why = "" if first is None else f"{first.code} {first.reason}"[:200]
                self.reordered = (f"the {other} arm first: the {arm} arm's way was refused while the {other} arm is "
                                  f"where it is ({why}); with the {other} arm at its posture it clears")  # fmt: skip
                self.pre[arm] = res
                self.pre_closed[arm] = self.alt_closed[arm]
                self.arms = [other, arm]
                return False
            if self.alt_closed[arm] or self.l2.svc.hands is None or not self._submit_alt(arm, True):
                return False

    def _submit_alt(self, arm: str, closed: bool) -> bool:
        out = self._plan_other_moved(arm, closed)
        if hasattr(out, "code") and not hasattr(out, "segments") and not hasattr(out, "done"):
            return False  # a refusal Result (no measured state): nothing more to try
        self.alt[arm], self.alt_closed[arm] = out, closed
        return True

    @staticmethod
    def _ok(res: Any) -> bool:
        return res is not None and bool(getattr(res, "ok", False))

    @staticmethod
    def _done(p: Any) -> bool:
        return p is None or not hasattr(p, "done") or bool(p.done())

    @staticmethod
    def _value(p: Any) -> Any:
        if p is None:
            return None
        return p.result() if hasattr(p, "result") and callable(p.result) else p

    def _hand_q(self, side: str) -> Any:
        h = self.l2.svc.hands
        if h is None:
            return None
        try:
            st = h.state(side)
        except Exception:
            return None
        q = st.get("q") if isinstance(st, dict) else None
        return None if q is None else np.asarray(q, dtype=np.float64).copy()

    # ------------------------------------------------------------------ Runner protocol
    def ready(self, now_ns: int) -> bool:
        return True

    def start(self, now_ns: int) -> Any:
        from r1mc3.service.codes import Code
        from r1mc3.service.schemas import Result

        if self.l2.svc.planner is None:
            return Result(ok=False, code=Code.UNSUPPORTED, op=self.op, details={"reason": "no planner in this r1d"})
        for arm in self.arms:
            out = self._plan(arm, closed=False)
            if isinstance(out, Result):
                return out
            self.pre[arm] = out
        self._precheck(now_ns)
        return None

    def _precheck(self, now_ns: int) -> None:
        from r1mc3.service.codes import Code
        from r1mc3.service.l2 import _code

        if not all(self._done(p) for p in self.pre.values()):
            return
        for arm in list(self.pre):
            res = self._value(self.pre[arm])
            if res is None:
                continue
            det = dict(res.details)
            under = (det.get("out_around") or {}).get("under") or det.get("out_around_under")
            if under is not None and not self.pre_closed[arm] and self.l2.svc.hands is not None:
                p = self._plan(arm, closed=True)  # the hand comes out closed (v2): planned with that shape
                self.pre[arm] = None if hasattr(p, "code") and not hasattr(p, "segments") else p  # a refusal Result
                if not self._done(p):
                    return
                res = self._value(p)
        if self._try_other_first():  # session 13: the arm in the other's way goes first
            return
        k = 0
        for arm in self.arms:
            res = self._value(self.pre[arm])
            entry = {"index": k, "arm": arm, "ok": bool(res is not None and res.ok),
                     "code": "TIMEOUT" if res is None else res.code, "hand_closed": self.pre_closed[arm],
                     "legs": [] if res is None else [s.leg for s in res.segments],
                     "reason": "" if res is None else res.reason}  # fmt: skip
            self.pre_results.append(entry)
            if res is None or not res.ok:
                self.failed = k
                self.end_state, self.code = "aborted", Code.SKILL_FAILED
                self.stopped_by = {"code": "PRECHECK", "detail": f"the {arm} arm refused before anything moved: "
                                   f"{entry['code']} {entry['reason']}"}  # fmt: skip
                self.results.append({"index": k, "op": "arm.move_joint", "ok": False,
                                     "code": str(_code(str(entry["code"]))), "state": "rejected", "moved": False,
                                     "data": {"plan_details": {} if res is None else dict(res.details)},
                                     "details": {"reason": entry["reason"]}})  # fmt: skip
                return
            k += 1
        self.pre = {}
        for arm in self.arms:
            closed = self.pre_closed[arm]
            if closed:
                self.hands0[arm] = self._hand_q(arm)
                self.items.append(("hand", arm, CLOSED_SHAPE))
            self.items.append(("arm", arm, self._goal(arm, closed)))
            if closed:
                self.items.append(("hand", arm, "restore"))
        self._next(now_ns)

    def _make(self, kind: str, side: str, what: Any) -> Any:
        from r1mc3.hands import grip as G
        from r1mc3.service.l2 import HandMoveRunner, PlannedRunner

        if kind == "arm":
            return PlannedRunner(self.l2, "arm.move_joint", side, what)
        if what == "restore":
            q = self.hands0.get(side)
            return HandMoveRunner(self.l2, side, q if q is not None else G.preset("open", side))
        return HandMoveRunner(self.l2, side, G.preset(what, side))

    def _next(self, now_ns: int) -> None:
        self.i += 1
        if self.i >= len(self.items):
            from r1mc3.service.codes import Code

            self.end_state, self.code = "succeeded", Code.OK
            self.cur = None
            return
        kind, side, what = self.items[self.i]
        self.cur, self.cur_started, self.cur_due = self._make(kind, side, what), False, now_ns

    def tick(self, now_ns: int) -> str | None:
        from r1mc3.service.codes import Code

        if self.end_state is not None:
            return self.end_state
        if self.pre:
            self._precheck(now_ns)
            return self.end_state
        if self.cur is None:
            return self.end_state
        kind, side, _ = self.items[self.i]
        if not self.cur_started:
            if not self.cur.ready(now_ns) and now_ns - self.cur_due < DEFER_MAX_NS:
                return None
            refused = self.cur.start(now_ns)
            if refused is not None:
                self._record(kind, side, "rejected", refused.code, {"reason": refused.details}, moved=False)
                self.failed = self.i
                self.end_state, self.code = "aborted", Code.SKILL_FAILED
                self.stopped_by = {"code": "SKILL_FAILED", "detail": f"step {self.i} ({kind} {side}) refused"}
                return self.end_state
            self.cur_started = True
            return None
        st = self.cur.tick(now_ns)
        if st is None:
            return None
        code, data, details = self.cur.outcome()
        self._record(kind, side, st, code, data, moved=bool(data.get("moved", True)), details=details)
        if st != "succeeded" or code != Code.OK:
            self.failed = self.i
            self.end_state, self.code = "aborted", Code.SKILL_FAILED
            self.stopped_by = {"code": "SKILL_FAILED", "detail": f"step {self.i} ({kind} {side}) {st}"}
            return self.end_state
        self._next(now_ns)
        return self.end_state

    def _record(self, kind: str, side: str, state: str, code: Any, data: dict[str, Any], *, moved: bool,
                details: dict[str, Any] | None = None) -> None:  # fmt: skip
        self.results.append({"index": self.i, "op": "arm.move_joint" if kind == "arm" else "hand.preshape",
                             "side": side, "ok": state == "succeeded", "code": str(code), "state": state,
                             "moved": moved, "data": data, "details": details or {}})  # fmt: skip

    def stop(self, now_ns: int, why: str) -> None:
        from r1mc3.service.codes import Code

        if self.end_state is not None:
            return
        if self.cur is not None and self.cur_started:
            self.cur.stop(now_ns, why)
        self.end_state = why
        self.code = {"preempted": Code.PREEMPTED, "timed_out": Code.TIMEOUT}.get(why, Code.CANCELED)
        self.stopped_by = {"code": why.upper(), "detail": f"r1d stopped the skill at step {self.i}"}

    def end(self, state: str, code: Any, stopped_by: dict[str, Any] | None) -> None:
        if self.cur is not None and self.cur_started:
            self.cur.end(state, code, stopped_by)
        if self.end_state is None:
            self.end_state, self.code, self.stopped_by = state, code, stopped_by

    def outcome(self) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        from r1mc3.service.codes import Code

        moved = any(r["moved"] for r in self.results)
        data = {"moved": moved, "steps": self.results, "n_steps": len(self.items), "failed_step": self.failed,
                "stopped_by": self.stopped_by, "precheck": self.pre_results,
                "hands_closed": [a for a, c in self.pre_closed.items() if c],
                "order": list(self.arms), "reordered": self.reordered}  # fmt: skip
        details: dict[str, Any] = {}
        if self.failed is not None:
            details = {"failed_step": self.failed, "step_result": self.results[-1] if self.results else None}
        return self.code if self.code is not None else Code.FAULT, data, details
