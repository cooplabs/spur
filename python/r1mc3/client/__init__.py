"""`r1mc3.client`: the typed Python SDK for r1d (DESIGN.md §5.3) - one method per L0-L3 op, sync.

    from r1mc3.client import R1
    with R1() as r:                                  # 127.0.0.1:8780 (through the ssh tunnel from the Mac)
        r.status().data["summary"]
        r.world_declare({"kind": "surface", "id": "table", "pose": {"pos": [0.45, 0, 0.76]}, "size": [0.6, 1.2, 0.04]})
        res = r.reach("left", [0.35, 0.15, 0.85])    # waits for the goal's end (wait=False: the goal id at once)
        res.ok, res.code, res.data["error_mm"]
        r.grasp("left", grip="pinch", object_class="rigid", width_m=0.03).raise_for_error()

Every method returns a `Reply` (the result: ok / code / op / id / moved / data / details); `raise_for_error()` turns
a refusal into `R1Refused` (with the code and the details). The requests are r1d's JSON schemas
(`r1mc3.service.schemas.REQUEST`): what this module sends is validated against them by its tests. `damp` is not
here on purpose: the one limp state is the operator's word (`r1 damp`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from r1mc3.service.client import Client, R1dError
from r1mc3.service.server import DEFAULT_PORT

Side = Literal["left", "right"]
GOAL_WAIT_S = 60.0


class R1Refused(RuntimeError):
    def __init__(self, reply: Reply) -> None:
        super().__init__(f"{reply.op}: {reply.code} {reply.details}")
        self.reply = reply


@dataclass(frozen=True)
class Reply:
    ok: bool
    code: str
    op: str
    id: int | str | None = None
    moved: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def of(d: dict[str, Any]) -> Reply:
        return Reply(
            bool(d.get("ok")),
            str(d.get("code")),
            str(d.get("op")),
            d.get("id"),
            bool(d.get("moved")),
            dict(d.get("data") or {}),
            dict(d.get("details") or {}),
        )

    def raise_for_error(self) -> Reply:
        if not self.ok:
            raise R1Refused(self)
        return self


class R1:
    """A connection to r1d with one typed method per op (the transport: r1mc3.service.client.Client)."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, *, timeout_s: float = 20.0, transport: Any = None
    ) -> None:
        self.c = transport if transport is not None else Client(host, port, timeout_s=timeout_s)
        self.lease: str | None = None  # session 10: set by lease_acquire (sent with every request while set)

    def close(self) -> None:
        self.c.close()

    def __enter__(self) -> R1:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def call(self, op: str, **kw: Any) -> Reply:
        if self.lease is not None and "lease" not in kw:
            kw["lease"] = self.lease
        return Reply.of(self.c.call(op, **{k: v for k, v in kw.items() if v is not None}))

    def _goal(self, op: str, wait: bool, timeout_s: float, **kw: Any) -> Reply:
        if wait and hasattr(self.c, "timeout_s"):
            self.c.timeout_s = max(float(self.c.timeout_s), timeout_s + 5.0)
        return self.call(op, wait=wait, timeout_s=timeout_s, **kw)

    # ---- system / L0 / L1
    def status(self) -> Reply:
        return self.call("status")

    def stop(self, level: Literal["stop", "halt"] = "stop") -> Reply:
        return self.call(level)

    def resume(self) -> Reply:
        return self.call("resume")

    def joint_command(self, group: str, q: Sequence[float], ttl: float = 0.2, **gains: Sequence[float]) -> Reply:
        return self.call("joint.command", group=group, q=list(q), ttl=ttl, **{k: list(v) for k, v in gains.items()})

    def joint_hold(self, group: str, q: Sequence[float] | None = None) -> Reply:
        return self.call("joint.hold", group=group, q=None if q is None else list(q))

    def jog(self, group: str, dq: Sequence[float], ttl: float = 0.3) -> Reply:
        return self.call("stream.joint_jog", group=group, dq=list(dq), ttl=ttl)

    # ---- L2 arm goals
    def move_joint(
        self, arm: Side, q: Sequence[float], *, speed: float = 0.5, wait: bool = True, timeout_s: float = GOAL_WAIT_S
    ) -> Reply:
        return self._goal("arm.move_joint", wait, timeout_s, arm=arm, q=list(q), speed=speed)

    def reach(
        self,
        arm: Side,
        position: Sequence[float],
        *,
        frame: str = "base",
        approach: Sequence[float] | None = None,
        prefer: str | None = None,
        speed: float = 0.5,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal(
            "arm.reach",
            wait,
            timeout_s,
            arm=arm,
            position=list(position),
            frame=frame,
            approach=None if approach is None else list(approach),
            prefer=prefer,
            speed=speed,
        )

    def move_linear(
        self,
        arm: Side,
        *,
        position: Sequence[float] | None = None,
        delta: Sequence[float] | None = None,
        frame: str = "base",
        speed: float = 0.3,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal(
            "arm.move_linear",
            wait,
            timeout_s,
            arm=arm,
            frame=frame,
            speed=speed,
            position=None if position is None else list(position),
            delta=None if delta is None else list(delta),
        )

    def move_relative(
        self,
        arm: Side,
        delta: Sequence[float],
        *,
        frame: str = "base",
        speed: float = 0.5,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal("arm.move_relative", wait, timeout_s, arm=arm, delta=list(delta), frame=frame, speed=speed)

    def hover(
        self,
        arm: Side,
        position: Sequence[float],
        height_m: float = 0.10,
        *,
        prefer: str | None = None,
        frame: str = "base",
        speed: float = 0.5,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal(
            "arm.hover",
            wait,
            timeout_s,
            arm=arm,
            position=list(position),
            height_m=height_m,
            prefer=prefer,
            frame=frame,
            speed=speed,
        )

    def lower(
        self,
        arm: Side,
        *,
        until: Literal["contact", "surface", "depth"] = "contact",
        max_depth_m: float = 0.10,
        stop_z_m: float | None = None,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal(
            "arm.lower", wait, timeout_s, arm=arm, until=until, max_depth_m=max_depth_m, stop_z_m=stop_z_m
        )

    def carry(
        self,
        arm: Side,
        position: Sequence[float],
        lift_m: float = 0.10,
        *,
        frame: str = "base",
        speed: float = 0.4,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal(
            "arm.carry", wait, timeout_s, arm=arm, position=list(position), lift_m=lift_m, frame=frame, speed=speed
        )

    def plan(self, goal: dict[str, Any]) -> Reply:
        return self.call("arm.plan", goal=goal)

    def arm_stop(self, arm: Side | None = None) -> Reply:
        return self.call("arm.stop", arm=arm)

    # ---- L2 hands / head / body
    def grasp(
        self,
        side: Side,
        *,
        grip: Literal["wrap", "pinch", "tip"] = "wrap",
        object_class: Literal["rigid", "thin_flexible", "soft"] = "rigid",
        width_m: float | None = None,
        squeeze: float | None = None,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal(
            "hand.grasp",
            wait,
            timeout_s,
            side=side,
            grip=grip,
            object_class=object_class,
            width_m=width_m,
            squeeze=squeeze,
        )

    def release(self, side: Side, *, wait: bool = True, timeout_s: float = GOAL_WAIT_S) -> Reply:
        return self._goal("hand.release", wait, timeout_s, side=side)

    def preshape(self, side: Side, shape: str = "open", *, wait: bool = True, timeout_s: float = GOAL_WAIT_S) -> Reply:
        return self._goal("hand.preshape", wait, timeout_s, side=side, shape=shape)

    def tare(self, side: Side) -> Reply:
        return self.call("hand.tare", side=side)

    def look_at(self, position: Sequence[float], *, wait: bool = True, timeout_s: float = GOAL_WAIT_S) -> Reply:
        return self._goal("head.look_at", wait, timeout_s, position=list(position))

    def lean(self, deg: float) -> Reply:
        return self.call("body.pose", lean_deg=deg)

    # ---- L3
    def skill(
        self,
        name: Literal["ready", "table_ready"],
        *,
        arms: Literal["both", "left", "right"] = "both",
        speed: float = 0.4,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal(f"skill.{name}", wait, timeout_s, arms=arms, speed=speed)

    def seq(
        self,
        steps: Sequence[dict[str, Any]],
        *,
        stop_on_failure: bool = True,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal("seq.run", wait, timeout_s, steps=list(steps), stop_on_failure=stop_on_failure)

    # ---- goals / world
    def goal_status(self, goal_id: int) -> Reply:
        return self.call("goal.status", goal_id=goal_id)

    def goal_wait(self, goal_id: int, timeout_s: float = GOAL_WAIT_S) -> Reply:
        if hasattr(self.c, "timeout_s"):
            self.c.timeout_s = max(float(self.c.timeout_s), timeout_s + 5.0)
        return self.call("goal.wait", goal_id=goal_id, timeout_s=timeout_s)

    def goal_cancel(self, goal_id: int) -> Reply:
        return self.call("goal.cancel", goal_id=goal_id)

    def goals(self) -> Reply:
        return self.call("goal.list")

    def world_declare(self, entity: dict[str, Any]) -> Reply:
        return self.call("world.declare", entity=entity)

    def world_list(self) -> Reply:
        return self.call("world.list")

    def world_clear(self, entity_id: str | None = None) -> Reply:
        return self.call("world.clear", entity_id=entity_id)

    # ---- session 10 (api): paths, guarded moves, slides, pick / place, par.run, leases, twists
    def move_path(
        self,
        arm: Side,
        waypoints: Sequence[Sequence[float]],
        *,
        linear: bool = False,
        frame: str = "base",
        speed: float = 0.4,
        blend_m: float = 0.0,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        """blend_m > 0 (session 11): pass each interior waypoint within blend_m without stopping (data.blend)."""
        extra: dict[str, Any] = {"blend_m": blend_m} if blend_m else {}  # an older r1d refuses an unknown field
        return self._goal("arm.move_path", wait, timeout_s, arm=arm, waypoints=[list(w) for w in waypoints],
                          linear=linear, frame=frame, speed=speed, **extra)  # fmt: skip

    def guarded_move(
        self,
        arm: Side,
        *,
        position: Sequence[float] | None = None,
        delta: Sequence[float] | None = None,
        stop_z_m: float | None = None,
        until: Sequence[Literal["contact", "blocked"]] = ("contact",),
        touch: Sequence[str] = (),
        frame: str = "base",
        speed: float = 0.2,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal("arm.guarded_move", wait, timeout_s, arm=arm,
                          position=None if position is None else list(position),
                          delta=None if delta is None else list(delta), stop_z_m=stop_z_m, until=list(until),
                          touch=list(touch), frame=frame, speed=speed)  # fmt: skip

    def slide(
        self,
        arm: Side,
        delta: Sequence[float],
        *,
        support: str | None = None,
        frame: str = "base",
        speed: float = 0.2,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal("arm.slide", wait, timeout_s, arm=arm, delta=list(delta), support=support, frame=frame,
                          speed=speed)  # fmt: skip

    def pick(
        self,
        arm: Side,
        position: Sequence[float],
        *,
        height_m: float = 0.05,
        grasp_z_m: float | None = None,
        grip: str = "wrap",
        object_class: str = "rigid",
        width_m: float | None = None,
        squeeze: float | None = None,
        hover_m: float = 0.03,
        lift_m: float = 0.05,
        object_id: str | None = None,
        speed: float = 0.4,
        wait: bool = True,
        timeout_s: float = 60.0,
    ) -> Reply:
        return self._goal("skill.pick", wait, timeout_s, arm=arm, position=list(position), height_m=height_m,
                          grasp_z_m=grasp_z_m, grip=grip, object_class=object_class, width_m=width_m,
                          squeeze=squeeze, hover_m=hover_m, lift_m=lift_m, object_id=object_id,
                          speed=speed)  # fmt: skip

    def place(
        self,
        arm: Side,
        position: Sequence[float],
        *,
        lift_m: float = 0.08,
        retreat_m: float = 0.05,
        speed: float = 0.4,
        wait: bool = True,
        timeout_s: float = 60.0,
    ) -> Reply:
        return self._goal("skill.place", wait, timeout_s, arm=arm, position=list(position), lift_m=lift_m,
                          retreat_m=retreat_m, speed=speed)  # fmt: skip

    def par(
        self,
        steps: Sequence[dict[str, Any]],
        *,
        stop_on_failure: bool = True,
        wait: bool = True,
        timeout_s: float = GOAL_WAIT_S,
    ) -> Reply:
        return self._goal("par.run", wait, timeout_s, steps=list(steps), stop_on_failure=stop_on_failure)

    def lease_acquire(self, resources: Sequence[str], *, client: str = "sdk", ttl_s: float = 30.0) -> Reply:
        """Take a lease; on success this client sends its id with every request until lease_release."""
        rep = self.call("lease.acquire", resources=list(resources), client=client, ttl_s=ttl_s)
        if rep.ok and rep.data.get("lease_id"):
            self.lease = str(rep.data["lease_id"])
        return rep

    def lease_renew(self, ttl_s: float = 30.0) -> Reply:
        return self.call("lease.renew", lease_id=self.lease, ttl_s=ttl_s)

    def lease_release(self) -> Reply:
        rep = self.call("lease.release", lease_id=self.lease)
        self.lease = None
        return rep

    def leases(self) -> Reply:
        return self.call("lease.list")

    def twist(self, arm: Side, v: Sequence[float], ttl: float = 0.3) -> Reply:
        return self.call("stream.arm_twist", arm=arm, v=list(v), ttl=ttl)

    # ---- session 10: the other agents' ops (the request shapes of their schemas)
    def calibrate_pinch(self, side: Side, *, wait: bool = True, timeout_s: float = GOAL_WAIT_S) -> Reply:
        """hand.calibrate air_pinch (grasp): the hand's air-pinch reference - thin_flexible grasps need it."""
        return self._goal("hand.calibrate", wait, timeout_s, side=side, kind="air_pinch")

    def observe(self, *, depth: bool = True, right: bool = False) -> Reply:
        """camera.observe (cam): the newest head frame -> obs_id + the camera pose at capture (+ the JPEG)."""
        return self.call("camera.observe", depth=depth, right=right)

    def point(self, obs_id: str, px: float, py: float, *, use_depth: bool = True) -> Reply:
        """camera.point (cam): a pixel of frame obs_id -> a base point on the declared surface its ray meets."""
        return self.call("camera.point", obs_id=obs_id, px=float(px), py=float(py), use_depth=use_depth)

    def object_extent(self, obs_id: str, px: float, py: float) -> Reply:
        return self.call("camera.object_extent", obs_id=obs_id, px=float(px), py=float(py))

    def project(self, obs_id: str, points: Sequence[Sequence[float]]) -> Reply:
        return self.call("camera.project", obs_id=obs_id, points=[list(p) for p in points])

    def measure_setup(self, entities: Sequence[dict[str, Any]], *, frames: int = 3) -> Reply:
        """measure.setup (measure): the declared entities' poses from their tags in head frames (moves nothing)."""
        if hasattr(self.c, "timeout_s"):
            self.c.timeout_s = max(float(self.c.timeout_s), 10.0 * frames + 5.0)
        return self.call("measure.setup", entities=list(entities), frames=frames)

    def measure_phone(
        self,
        image_b64: str,
        intrinsics: dict[str, Any],
        table_tags: Sequence[dict[str, Any]],
        *,
        entities: Sequence[dict[str, Any]] = (),
        points: Sequence[dict[str, Any]] = (),
    ) -> Reply:
        """measure.phone (measure): a still's camera pose from the table tags + entities / pixels on the table."""
        return self.call("measure.phone", image_b64=image_b64, intrinsics=intrinsics, table_tags=list(table_tags),
                         entities=list(entities), points=list(points))  # fmt: skip


__all__ = ["R1", "R1Refused", "R1dError", "Reply"]
