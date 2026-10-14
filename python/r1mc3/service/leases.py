"""Resource leases (DESIGN.md §5: "one lease model"; session 10). OFF until a client takes one.

A client takes a lease on resource groups (`lease.acquire {resources, client, ttl_s}` -> `lease_id`), renews it
(`lease.renew`) and gives it back (`lease.release`); it expires by itself after its TTL (r1d's clock: the core's).
While a lease is live, a request that would MOVE one of its resources must carry that `lease` id, else NOT_OWNER
(details: the holder, the time left). With no live lease nothing changes (every existing client works as before).

Never gated: `status`, `stop` / `halt` / `damp` (always accepted: DESIGN §5), `arm.stop` (anyone may stop an arm),
the read-only ops, the dry run
`arm.plan`, the lease ops themselves, and the steps a composite goal (seq.run, a skill, par.run) submits for itself
(the composite was checked). `goal.cancel` of a goal on a leased resource needs the lease too (session 9: "any client
may cancel any goal").
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .codes import Code
from .schemas import LeaseReq, Result, fail, ok

if TYPE_CHECKING:
    from .service import Service

RESOURCES = ("base", "body", "head", "arm_left", "arm_right", "hand_left", "hand_right", "leg_left", "leg_right",
             "waist")  # fmt: skip
MAX_TTL_S = 300.0
NEVER_GATED = {"status", "stop", "halt", "damp", "arm.stop", "arm.plan", "goal.status", "goal.wait", "goal.list",
               "hand.state", "record.list", "record.plan", "camera.status", "world.list"}  # fmt: skip
_ids = itertools.count(1)


@dataclass
class Lease:
    lease_id: str
    client: str
    resources: frozenset[str]
    until_ns: int

    def view(self, now_ns: int) -> dict[str, Any]:
        return {"lease_id": self.lease_id, "client": self.client, "resources": sorted(self.resources),
                "expires_in_s": round(max(0, self.until_ns - now_ns) * 1e-9, 3)}  # fmt: skip


def _table(svc: Service) -> dict[str, Lease]:
    tb: dict[str, Lease] | None = getattr(svc, "_leases", None)
    if tb is None:
        tb = {}
        svc._leases = tb  # type: ignore[attr-defined]
    return tb


def live(svc: Service, now_ns: int) -> list[Lease]:
    tb = _table(svc)
    for k in [k for k, v in tb.items() if v.until_ns <= now_ns]:
        del tb[k]
    return list(tb.values())


def resources_of(svc: Service, r: Any) -> frozenset[str]:
    """The resource groups a request would move (from its op and fields; the steps' union for seq.run / par.run;
    the goal's for goal.cancel; nothing for the never-gated ops)."""
    from .l2 import steps_resources

    op = str(getattr(r, "op", ""))
    if op in NEVER_GATED or op.startswith("lease."):
        return frozenset()
    if op == "stream.arm_twist":
        return frozenset([f"arm_{r.arm}"])
    if op in ("joint.command", "joint.hold", "stream.joint_jog"):
        return frozenset([str(r.group)])
    if op == "stream.head":
        return frozenset(["head"])
    if op in ("hand.command",):
        return frozenset([f"hand_{r.side}"])
    if op == "goal.cancel":
        g = svc.l2.table.goals.get(int(r.goal_id))
        return frozenset() if g is None else frozenset(g.resources)
    if op == "body.pose":
        return frozenset(["body"])
    if op in ("seq.run", "par.run"):
        return steps_resources(list(r.steps))
    if op in ("takeover", "on_stand", "resume", "freeze", "reclaim", "stand_freeze", "stand_hold"):
        return frozenset(RESOURCES)  # a mode change moves everything
    d = r.model_dump() if hasattr(r, "model_dump") else {}
    return steps_resources([d]) | (
        frozenset([f"hand_{d['arm']}"]) if op in ("skill.pick", "skill.place") else frozenset()
    )


def check(svc: Service, r: Any) -> Result | None:
    """None = the request may run; else NOT_OWNER (a live lease on a resource it would move, not presented)."""
    if not getattr(svc, "_leases", None):
        return None  # no lease was ever taken: off
    if svc.l2._parent_ctx is not None:
        return None  # a composite goal's own step (the composite was checked)
    now = svc.port.now_ns()
    held = live(svc, now)
    if not held:
        return None
    need = resources_of(svc, r)
    if not need:
        return None
    given = getattr(r, "lease", None)
    for le in held:
        if le.resources & need and le.lease_id != given:
            return fail(str(r.op), getattr(r, "id", None), Code.NOT_OWNER, held_by=le.view(now),
                        resources=sorted(le.resources & need),
                        reason="a lease holds these resources: send its `lease` id (or wait for its end)")  # fmt: skip
    return None


def handles(r: Any) -> bool:
    return isinstance(r, LeaseReq)


def handle(svc: Service, r: LeaseReq) -> Result:
    now = svc.port.now_ns()
    held = live(svc, now)
    tb = _table(svc)
    if r.op == "lease.list":
        return ok(r.op, r.id, leases=[le.view(now) for le in held])
    if r.op == "lease.acquire":
        res = frozenset(r.resources)
        bad = sorted(res - set(RESOURCES))
        if bad or not res:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason="resources: known groups", unknown=bad,
                        known=list(RESOURCES))  # fmt: skip
        clash = [le for le in held if le.resources & res and le.client != r.client]
        if clash:
            return fail(r.op, r.id, Code.RESOURCE_BUSY, held_by=[le.view(now) for le in clash])
        le = Lease(f"L{next(_ids)}", r.client, res, now + round(min(r.ttl_s, MAX_TTL_S) * 1e9))
        tb[le.lease_id] = le
        return ok(r.op, r.id, **le.view(now))
    le2 = tb.get(r.lease_id or "")
    if le2 is None:
        return fail(r.op, r.id, Code.NOT_OWNER, reason="no live lease with this id", lease_id=r.lease_id)
    if r.op == "lease.renew":
        le2.until_ns = now + round(min(r.ttl_s, MAX_TTL_S) * 1e9)
        return ok(r.op, r.id, **le2.view(now))
    del tb[le2.lease_id]  # lease.release
    return ok(r.op, r.id, released=le2.lease_id)
