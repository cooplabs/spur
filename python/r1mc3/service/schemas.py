"""Typed requests and the one result shape of r1d (DESIGN.md §5: typed schemas, codes, measured results).

A request is one JSON object with an `op` (the model is chosen by it) and a client-chosen `id` echoed in the result.
Every result has the same keys: `ok`, `code` (codes.Code), `op`, `id`, `moved` (whether anything was commanded to
move: refusals say False by construction, DESIGN §5.2), `data` (op-specific, measured values) and `details`
(structured reasons, never prose to match on). The JSON Schemas are `request_schema()` / `Result.model_json_schema()`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .codes import Code

Group = Literal["arm_left", "arm_right", "head", "leg_left", "leg_right", "waist"]
UpperGroup = Literal["arm_left", "arm_right", "head"]
Side = Literal["left", "right"]
Requester = Literal["r1d", "operator"]


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: int | str | None = None
    requester: Requester = "r1d"  # "operator" = a human at the CLI (damp needs it)
    lease: str | None = None  # session 10: the client's lease id (leases.py; needed only while a lease is live)


class StatusReq(_Req):
    op: Literal["status"]


class StopReq(_Req):
    """stop: all groups HOLD + base zero, keep balancing; halt: decelerate to rest (resumable); damp: limp (OPERATOR
    ONLY: the one limp state, DESIGN §3.5)."""

    op: Literal["stop", "halt", "damp"]


class ResumeReq(_Req):
    op: Literal["resume"]


class FreezeReq(_Req):
    op: Literal["freeze"]


class TakeoverReq(_Req):
    """MONITOR -> MIRROR (the core's gates) -> ReleaseMode (motion switcher) -> BLEND -> BALANCE, or ABORT."""

    op: Literal["takeover", "on_stand"]
    release_timeout_s: float = Field(4.0, gt=0.0, le=10.0)  # v2: ReleaseMode within 4 s (DESIGN §3.5)
    reply_timeout_s: float = Field(1.0, gt=0.0, le=5.0)  # the core answers a mailbox request within a few ticks


class ReclaimReq(_Req):
    """OPERATOR ONLY: r1rt takes the motors back from r1guard (the guard-takeover test; the core's `reclaim`: only after
    the guard took over from ON_STAND and r1rt runs again - a balancing hand-back is refused, DESIGN §4.4)."""

    op: Literal["reclaim"]


class BoardHoldReq(_Req):
    """OPERATOR ONLY (the board-hold test, DESIGN §14.1 item 6): LowCmd paused `pause_ms` by SIGSTOP of r1rt, the guard
    OFF, ON_STAND; r1d records rt/lowstate from 1 s before to 2 s after. dry_run (default): the plan + its checks
    only."""

    op: Literal["board_hold.pause"]
    pause_ms: Literal[20, 100, 300, 400, 500]  # HW1 #34: the board releases at ~400 ms; 1000 / 2000 refused for now
    dry_run: bool = True


class GuardTakeoverReq(_Req):
    """OPERATOR ONLY (the guard-takeover test, DESIGN §14.1 item 7): r1rt stopped `stop_s` in ON_STAND with the guard
    ARMED; the guard must hold; after SIGCONT r1rt stays silent until `reclaim`. dry_run (default): the plan + its
    checks only."""

    op: Literal["guard_takeover.stall"]
    stop_s: float = Field(3.0, ge=0.5, le=8.0)
    dry_run: bool = True


class StandFreezeReq(_Req):
    """OPERATOR ONLY (HW1 S10, the user's U3): in ON_STAND, the legs + waist ramp to the stand pose over the config's
    stand_ramp_s at the stand gains and hold (the same STAND_FREEZE as a balancing robot's rung 3; arms HOLD; the core
    posts a STOP -> `resume` after). dry_run (default): the ramp time + the far check against the current targets."""

    op: Literal["stand_freeze"]
    dry_run: bool = True


class StandHoldReq(_Req):
    """OPERATOR (r1d's word): from the stand's STAND_FREEZE back into the ON_STAND hold (a blend, no release)."""

    op: Literal["stand_hold"]


class ShutdownReq(_Req):
    """OPERATOR ONLY: the session end's last r1d word (only in MONITOR / DAMP): r1log finishes, the session's logs
    are saved + verified into one folder, then "safe to power off now". Never damps, never moves anything.
    Session 8 (HW1 fixes #13 / #20): every shutdown writes a NEW folder `<sessions>/<label>_<UTC>` (never one that
    exists) with every file of the last 24 h not already saved in an earlier session folder."""

    op: Literal["shutdown"]
    label: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")


class JointCommandReq(_Req):
    """L0: per-joint setpoints for ONE group (STREAM with a TTL; DESIGN §5.1). Legs / waist only in ON_STAND."""

    op: Literal["joint.command"]
    group: Group
    q: list[float]
    dq: list[float] | None = None
    kp: list[float] | None = None
    kd: list[float] | None = None
    tau_ff: list[float] | None = None
    ttl: float = Field(0.2, gt=0.0, le=0.3)  # s; the executor caps a STREAM TTL at 300 ms (arms)

    @model_validator(mode="after")
    def _finite(self) -> JointCommandReq:
        for k in ("q", "dq", "kp", "kd", "tau_ff"):
            v = getattr(self, k)
            if v is not None and not all(x == x and abs(x) != float("inf") for x in v):
                raise ValueError(f"{k}: non-finite value")
        return self


class JointHoldReq(_Req):
    """L0: HOLD a group at q (or where it is now when q is omitted)."""

    op: Literal["joint.hold"]
    group: Group
    q: list[float] | None = None


class JointJogReq(_Req):
    """L1: joint velocities for one group; r1d integrates them at its tick rate into STREAM setpoints until `ttl`."""

    op: Literal["stream.joint_jog"]
    group: Group
    dq: list[float]
    ttl: float = Field(0.3, gt=0.0, le=0.5)


class HeadStreamReq(_Req):
    """L1: the head - a setpoint (pitch, yaw) or a rate (dpitch, dyaw), TTL-limited."""

    op: Literal["stream.head"]
    q: list[float] | None = None
    rate: list[float] | None = None
    ttl: float = Field(0.3, gt=0.0, le=0.5)

    @model_validator(mode="after")
    def _one(self) -> HeadStreamReq:
        if (self.q is None) == (self.rate is None):
            raise ValueError("give exactly one of q (pitch, yaw) or rate (dpitch, dyaw)")
        return self


class HandCommandReq(_Req):
    """Hands low level: per-motor q / dq / kp / kd / tau for one hand (7 motors, Dex3 DDS order), held for `ttl`."""

    op: Literal["hand.command"]
    side: Side
    q: list[float]
    dq: list[float] | None = None
    kp: list[float] | None = None
    kd: list[float] | None = None
    tau: list[float] | None = None
    ttl: float = Field(1.0, gt=0.0, le=5.0)


class HandStateReq(_Req):
    op: Literal["hand.state"]
    side: Side | None = None


class RecordReq(_Req):
    """HW1 Tier B recordings (r1mc3.record): a plan by name + parameters; dry_run (default) only plans and checks."""

    op: Literal["record.plan", "record.run", "record.list"]
    kind: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = True


class CameraReq(_Req):
    """r1cam through r1d (session 9; R1CAM_PLAN.md): `camera.status` (read-only); `camera.services` = Unitree's
    camera-services switch (the stereo service vs video_hub - moves nothing): OPERATOR + confirm only, never automatic
    (the user's Q2, 11 Oct; its first robot use only with the user's confirmation, HW2)."""

    op: Literal["camera.status", "camera.services"]
    action: Literal["status", "stereo", "video_hub"] = "status"
    confirm: bool = False


# ---------------------------------------------------------------------------------------------- L2 / L3 (session 9)
# DESIGN §5.1 / §5.2. A goal op answers at once: `data.goal_id`, `data.state` ("accepted"; a refusal is a failed
# Result with moved False) - unless `wait` is True, then the reply is the goal's final result. `goal.status` /
# `goal.cancel` / `goal.wait` follow a goal through accepted -> active -> succeeded | aborted | canceled | preempted |
# timed_out. A finished motion goal's data carries the measured §5.2 keys (GOAL_RESULT_KEYS).

Frame = str  # "base" (DESIGN §5: the pelvis-yaw frame on the soles, REP-103) or a declared world entity's id
GripName = Literal["wrap", "pinch", "tip"]
ObjectClass = Literal["rigid", "thin_flexible", "soft"]  # generic classes (DESIGN §6.2; D32: no task names)
GOAL_STATES = ("accepted", "active", "succeeded", "aborted", "canceled", "preempted", "timed_out")
GOAL_TERMINAL = ("succeeded", "aborted", "canceled", "preempted", "timed_out")
GOAL_RESULT_KEYS = (
    "moved",
    "hand_start",
    "hand_now",
    "target",
    "achieved",
    "error_mm",
    "fingertips_m",
    "grasp_centre_m",
    "stopped_by",
    "duration_s",
    "planned_on",
    "plan_id",
)
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Arm5 = Annotated[list[float], Field(min_length=5, max_length=5)]


def _finite_all(obj: BaseModel, keys: tuple[str, ...]) -> None:
    for k in keys:
        v = getattr(obj, k, None)
        if v is None:
            continue
        vals = v if isinstance(v, list) else [v]
        if not all(isinstance(x, int | float) and x == x and abs(x) != float("inf") for x in vals):
            raise ValueError(f"{k}: non-finite value")


class _Goal(_Req):
    """Common to every goal: `wait` (reply with the final result), `timeout_s` (the goal times out after it),
    `preempt` (replace a running goal on the same resources instead of RESOURCE_BUSY)."""

    wait: bool = False
    timeout_s: float = Field(30.0, gt=0.0, le=600.0)
    preempt: bool = False


class GoalStatusReq(_Req):
    op: Literal["goal.status"]
    goal_id: int


class GoalCancelReq(_Req):
    op: Literal["goal.cancel"]
    goal_id: int


class GoalWaitReq(_Req):
    op: Literal["goal.wait"]
    goal_id: int
    timeout_s: float = Field(30.0, gt=0.0, le=600.0)


class GoalListReq(_Req):
    op: Literal["goal.list"]


class ArmMoveJointReq(_Goal):
    """L2: a planned joint-space move of one arm to q (5 joints, SDK order) - TRAJECTORY records with the deadman
    keepalive; `speed` scales the joints' planning speed (1.0 = 1 rad/s peak, v2's planning speed); `duration_s`
    overrides it (refused when the path would exceed the joints' dq limit)."""

    op: Literal["arm.move_joint"]
    arm: Side
    q: Arm5
    speed: float = Field(0.5, gt=0.0, le=1.0)
    duration_s: float | None = Field(None, gt=0.0, le=30.0)

    @model_validator(mode="after")
    def _ok(self) -> ArmMoveJointReq:
        _finite_all(self, ("q",))
        return self


class ArmReachReq(_Goal):
    """L2: the hand (its grasp centre for `grip`, else the palm) to a position; `approach` = the direction the hand
    travels in the last centimetres (a unit vector in `frame`), `orientation` = a wxyz quaternion for the palm."""

    op: Literal["arm.reach"]
    arm: Side
    position: Vec3
    frame: Frame = "base"
    orientation: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None
    approach: Vec3 | None = None
    grip: GripName | None = None
    prefer: Literal["front", "inward", "outward", "across"] | None = None  # the hand-orientation family (orient.py)
    speed: float = Field(0.5, gt=0.0, le=1.0)
    # session 10 round B: the declared world entity this goal is FOR (e.g. the object to grasp): the HAND may touch it
    # (the planner's allow_hand); the arm keeps its clearance and every other entity blocks as before
    target_id: str | None = Field(None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _ok(self) -> ArmReachReq:
        _finite_all(self, ("position", "orientation", "approach"))
        return self


class ArmMoveLinearReq(_Goal):
    """L2: a straight Cartesian line of the hand to `position` or by `delta` (exactly one), in `frame`."""

    op: Literal["arm.move_linear"]
    arm: Side
    position: Vec3 | None = None
    delta: Vec3 | None = None
    frame: Frame = "base"
    speed: float = Field(0.3, gt=0.0, le=1.0)
    # session 10 round B: the declared world entity this goal is FOR (e.g. the object to grasp): the HAND may touch it
    # (the planner's allow_hand); the arm keeps its clearance and every other entity blocks as before
    target_id: str | None = Field(None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _one(self) -> ArmMoveLinearReq:
        if (self.position is None) == (self.delta is None):
            raise ValueError("give exactly one of position or delta")
        _finite_all(self, ("position", "delta"))
        return self


class ArmMoveRelativeReq(_Goal):
    """L2: the hand moved by `delta` (m) in `frame` (planned path, not necessarily a line)."""

    op: Literal["arm.move_relative"]
    arm: Side
    delta: Vec3
    frame: Frame = "base"
    speed: float = Field(0.5, gt=0.0, le=1.0)
    # session 10 round B: the declared world entity this goal is FOR (e.g. the object to grasp): the HAND may touch it
    # (the planner's allow_hand); the arm keeps its clearance and every other entity blocks as before
    target_id: str | None = Field(None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _ok(self) -> ArmMoveRelativeReq:
        _finite_all(self, ("delta",))
        return self


class ArmHoverReq(_Goal):
    """L2: the hand to `height_m` above `position` (lift + transit + descend, planned end to end before it moves)."""

    op: Literal["arm.hover"]
    arm: Side
    position: Vec3
    height_m: float = Field(0.10, ge=0.0, le=0.5)
    frame: Frame = "base"
    grip: GripName | None = None
    prefer: Literal["front", "inward", "outward", "across"] | None = None  # the hand-orientation family (orient.py)
    speed: float = Field(0.5, gt=0.0, le=1.0)
    # session 10 (planner): the axis the fingers close along, seen from above (base frame, deg, a line: mod 180; e.g.
    # across a box's narrow side), and a preferred finger direction - both rank the orientation search's clear poses
    grasp_yaw_deg: float | None = Field(None, ge=-180.0, le=180.0)
    approach: Vec3 | None = None
    # session 10 round B: the declared world entity this goal is FOR (e.g. the object to grasp): the HAND may touch it
    # (the planner's allow_hand); the arm keeps its clearance and every other entity blocks as before
    target_id: str | None = Field(None, min_length=1, max_length=64)
    # session 12 (HW2 day 1, the user: "the approach is a MODEL decision, the controller must follow it very well"):
    # front / inward = the fingers' heading family delivered; around = for a TALL object, the palm turned to face it,
    # the closing line level, the pose chosen where the grip closes with the open hand clear of the target (needs
    # target_id declared as an object, else grasp_z_m); low = for a FLAT object, the grip's point down at the grasp
    # height with the hand clear of the support. approach_tol_deg: None = the nearest deliverable pose, its error
    # reported (plan_details.orientation.approach_err_deg); a number = refused beyond it, saying why
    approach_style: Literal["front", "inward", "around", "low"] | None = None
    approach_tol_deg: float | None = Field(None, ge=0.0, le=90.0)
    grasp_z_m: float | None = None  # where the grip's point will close (base z): the pose is chosen there
    # session 13: with grasp_z_m, the hover ends at least this far above that grasp pose (a straight-down step follows)
    min_descent_m: float | None = Field(None, ge=0.0, le=0.2)

    @model_validator(mode="after")
    def _ok(self) -> ArmHoverReq:
        _finite_all(self, ("position",))
        return self


class ArmLowerReq(_Goal):
    """L2: a guarded descent from where the hand is: until a pad contact (`contact`), the declared surface below
    (`surface`), or `max_depth_m` (`depth`); never deeper than max_depth_m."""

    op: Literal["arm.lower"]
    arm: Side
    until: Literal["contact", "surface", "depth"] = "contact"
    max_depth_m: float = Field(0.10, gt=0.0, le=0.4)
    stop_z_m: float | None = None  # the grasp centre stops at this height (base frame): a depth from where it is now
    frame: Frame = "base"
    speed: float = Field(1.0, gt=0.0, le=1.0)  # the planner runs a guarded descent at a quarter of its speed
    # session 10 round B: the declared world entity this goal is FOR (e.g. the object to grasp): the HAND may touch it
    # (the planner's allow_hand); the arm keeps its clearance and every other entity blocks as before
    target_id: str | None = Field(None, min_length=1, max_length=64)


class ArmCarryReq(_Goal):
    """L2: rise by `lift_m`, travel, lower to `position` (planned end to end first)."""

    op: Literal["arm.carry"]
    arm: Side
    position: Vec3
    lift_m: float = Field(0.10, ge=0.0, le=0.5)
    frame: Frame = "base"
    speed: float = Field(0.4, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ok(self) -> ArmCarryReq:
        _finite_all(self, ("position",))
        return self


class ArmPlanReq(_Req):
    """L2 dry run: plan any arm goal (`goal` = its request object without id) and report; nothing moves."""

    op: Literal["arm.plan"]
    goal: dict[str, Any]


class ArmStopReq(_Req):
    """L2: stop one arm's (or both arms') goal: decelerate to HOLD (the goal ends `canceled`)."""

    op: Literal["arm.stop"]
    arm: Side | None = None


class HandGraspReq(_Goal):
    """L2: preshape -> close -> per-finger contact stop -> squeeze -> verdict (the grip state machine, DESIGN §6.2)."""

    op: Literal["hand.grasp"]
    side: Side
    grip: GripName = "wrap"
    object_class: ObjectClass = "rigid"
    width_m: float | None = Field(None, gt=0.0, le=0.2)
    squeeze: float | None = Field(None, ge=0.0, le=0.8)


class HandReleaseReq(_Goal):
    op: Literal["hand.release"]
    side: Side


class HandPreshapeReq(_Goal):
    op: Literal["hand.preshape"]
    side: Side
    shape: str = "open"


class HandTareReq(_Req):
    op: Literal["hand.tare"]
    side: Side


class HeadLookAtReq(_Goal):
    """L2: point the head camera's optical axis at a position (pitch / yaw within their limits)."""

    op: Literal["head.look_at"]
    position: Vec3
    frame: Frame = "base"
    speed: float = Field(0.5, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ok(self) -> HeadLookAtReq:
        _finite_all(self, ("position",))
        return self


class BodyPoseReq(_Req):
    """L2: the balance policy's own command inputs (lean; height only if the policy has one), within the policy's
    trained ranges (its metadata); TTL-renewed by r1d until changed."""

    op: Literal["body.pose"]
    lean_deg: float | None = Field(None, ge=-15.0, le=15.0)
    height: float | None = None


class SkillReq(_Goal):
    """L3: named arm postures built on L2 (`ready` = arms down at the sides, `table_ready` = forearms forward, hands
    open above a work surface); `arms` = which arms."""

    op: Literal["skill.ready", "skill.table_ready"]
    arms: Literal["both", "left", "right"] = "both"
    speed: float = Field(0.4, gt=0.0, le=1.0)


class SeqRunReq(_Goal):
    """L3: run steps (request objects) one after another; the first failure ends the sequence (STEP_FAILED with the
    step's index and result) unless `stop_on_failure` is False."""

    op: Literal["seq.run"]
    steps: list[dict[str, Any]] = Field(min_length=1, max_length=64)
    stop_on_failure: bool = True


class WorldDeclareReq(_Req):
    """World model: declare / replace an entity (`entity` = {kind: surface | container | obstacle | object, id, pose,
    size, ...}, the planner's r1mc3.world types)."""

    op: Literal["world.declare"]
    entity: dict[str, Any]


class WorldClearReq(_Req):
    op: Literal["world.clear"]
    entity_id: str | None = None  # None = every declared entity


class WorldListReq(_Req):
    op: Literal["world.list"]


# session 10: each agent defines its new request models between its own markers (merge-friendly)
# >>> s10:cam (request models)
class CameraObserveReq(_Req):
    """camera.observe (session 10, DESIGN §8.1): r1cam's newest head frame -> an obs_id with the camera's pose in
    `base` at capture (v2 headcam.py's contract); `depth` (default on) for pointing heights / object_extent."""

    op: Literal["camera.observe"]
    depth: bool = True
    right: bool = False
    jpeg: bool = True  # False: the reply without the image (the `r1 observe` word: obs_id + pose for pointing)
    max_age_s: float = Field(1.0, gt=0.0, le=5.0)


class CameraPointReq(_Req):
    """camera.point: a pixel of frame obs_id -> a `base` point on the declared surface its line of sight meets (depth
    only for the height of a raised thing; v2's rules)."""

    op: Literal["camera.point"]
    obs_id: str
    px: float
    py: float
    patch: int = Field(7, ge=1, le=31)
    use_depth: bool = True


class CameraExtentReq(_Req):
    """camera.object_extent: what stands at the pixel (footprint, heights, shape) from the depth; the robot's own hands
    are dropped unless exclude_hands is false."""

    op: Literal["camera.object_extent"]
    obs_id: str
    px: float
    py: float
    exclude_hands: bool = True


class CameraProjectReq(_Req):
    op: Literal["camera.project"]
    obs_id: str
    points: list[Vec3] = Field(min_length=1, max_length=256)


class CameraDepthImageReq(_Req):
    op: Literal["camera.depth_image"]
    obs_id: str


# <<< s10:cam
# >>> s10:measure (request models)
# measure.* (session 10; DESIGN §8.2, docs/architecture/MEASUREMENT.md): GENERIC - the client declares its entities and
# where their tags sit (D32: no task layout in r1d); r1d detects the tags in head frames (r1cam) or in a phone still,
# fits each entity's pose and returns the registrations; the checks / tolerances / words are the client's.
class MeasureTagSpec(BaseModel):
    """A tag on an entity: AprilTag 36h11 `id`, the black square's edge `size_m`, and its pose on the entity - either
    the full 4 x 4 `T_entity_tag` (the tag frame of r1mc3.measure.tags) or, for a tag lying flat on the entity's top
    face, its centre `xyz` (entity frame, m) + `up_yaw_deg` (the printed top row's direction, 0 = +x)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: int = Field(ge=0, le=586)  # tag36h11 has 587 codes
    size_m: float = Field(gt=0.005, le=0.6)
    T_entity_tag: Annotated[list[Annotated[list[float], Field(min_length=4, max_length=4)]],
                            Field(min_length=4, max_length=4)] | None = None  # fmt: skip
    xyz: Annotated[list[float], Field(min_length=3, max_length=3)] | None = None
    up_yaw_deg: float = 0.0

    @model_validator(mode="after")
    def _one_pose(self) -> MeasureTagSpec:
        if (self.T_entity_tag is None) == (self.xyz is None):
            raise ValueError("give exactly one of T_entity_tag / xyz")
        return self


class MeasureEntitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1, max_length=64)
    tags: list[MeasureTagSpec] = Field(min_length=1, max_length=64)
    # x, y, z, yaw with the face horizontal (a table / board / container on a level floor); False = 6 DoF
    level: bool = True


class MeasureSetupReq(_Req):
    """measure.setup: `frames` new head frames from r1cam (each taken while the head is still, its camera pose from
    the joints at the frame) -> every entity's pose in `base` from all its tags in all frames (one fit per entity).
    Moves nothing; valid for a robot standing on both feet."""

    op: Literal["measure.setup"]
    entities: list[MeasureEntitySpec] = Field(min_length=1, max_length=16)
    frames: int = Field(3, ge=1, le=10)
    wait_s: float = Field(5.0, gt=0.0, le=10.0)  # for each new frame
    max_head_motion_rad: float = Field(0.003, gt=0.0, le=0.01)  # head joints between the reads around a frame


class PhoneIntrinsics(BaseModel):
    """A still camera's model (OpenCV): K (3 x 3) or a centred pinhole `f_px`; `dist` = k1 k2 p1 p2 [k3]."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    K: Annotated[list[Annotated[list[float], Field(min_length=3, max_length=3)]],
                 Field(min_length=3, max_length=3)] | None = None  # fmt: skip
    f_px: float | None = Field(None, gt=0.0)
    dist: list[float] = Field(default_factory=lambda: [0.0] * 5, min_length=4, max_length=8)
    width: int = Field(gt=0, le=20000)
    height: int = Field(gt=0, le=20000)

    @model_validator(mode="after")
    def _one_k(self) -> PhoneIntrinsics:
        if (self.K is None) == (self.f_px is None):
            raise ValueError("give exactly one of K / f_px")
        return self


class MeasurePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=64)
    px: Annotated[list[float], Field(min_length=2, max_length=2)]
    z_m: float = Field(0.0, ge=-0.1, le=1.0)  # the plane above the table the pixel lies on (an object's top face)


class MeasurePhoneReq(_Req):
    """measure.phone: a still from a hand-placed camera -> its pose from the table's tags (the placement correction),
    the declared entities' poses ON THE TABLE (table frame) and the given pixels on the table plane (at z_m)."""

    op: Literal["measure.phone"]
    image_b64: str = Field(min_length=16, max_length=40_000_000)  # a JPEG / PNG
    intrinsics: PhoneIntrinsics
    table_tags: list[MeasureTagSpec] = Field(min_length=3, max_length=64)
    entities: list[MeasureEntitySpec] = Field(default_factory=list, max_length=16)
    points: list[MeasurePoint] = Field(default_factory=list, max_length=64)
    # the phone's own check: its table tags' on-plane fit (mm; a tag left out may be predicted 2x it off) - the default
    # = r1mc3.measure.phone.MAX_FIT_MM; a client whose camera sits closer / more oblique may ask for more (<= 5 mm)
    max_fit_mm: float = Field(2.0, gt=0.0, le=5.0)


# <<< s10:measure
# >>> s10:grasp (request models)
class HandCalibrateReq(_Goal):
    """L2 (session 10): measure one hand's calibration reference - `air_pinch`: a tip pinch closed on nothing (the hand
    empty, in free air), its tip-shell clearance = the reference every `thin_flexible` grasp of that hand needs."""

    op: Literal["hand.calibrate"]
    side: Side
    kind: Literal["air_pinch"] = "air_pinch"


# <<< s10:grasp
# >>> s10:planner (request models)
# <<< s10:planner
# >>> s10:api (request models)
class ArmMovePathReq(_Goal):
    """L2 (session 10): the hand through `waypoints` (positions in `frame`, the pinch point as arm.reach) one after
    another. EVERY leg is planned end to end before anything moves (AP-3: a refused waypoint refuses the whole path,
    nothing moved); each leg a straight line (`linear`) or any planned path. `blend_m` 0: the hand comes to rest at
    each waypoint (the legs chain on the executor's DONE); > 0 (session 11): ONE trajectory that passes each interior
    waypoint within blend_m without stopping and ends exactly at the last one (r1mc3.plan.blend: re-checked with the
    planner's clearance; else a smaller corner, else the stops - `data.blend`)."""

    op: Literal["arm.move_path"]
    arm: Side
    waypoints: list[Vec3] = Field(min_length=1, max_length=16)
    frame: Frame = "base"
    linear: bool = False
    speed: float = Field(0.4, gt=0.0, le=1.0)
    blend_m: float = Field(0.0, ge=0.0, le=0.05)  # r1mc3.plan.blend.BLEND_MAX_M

    @model_validator(mode="after")
    def _ok(self) -> ArmMovePathReq:
        if not all(x == x and abs(x) != float("inf") for w in self.waypoints for x in w):
            raise ValueError("waypoints: non-finite value")
        return self


GuardEnd = Literal["contact", "blocked"]


class ArmGuardedMoveReq(_Goal):
    """L2 (session 10): a straight hand move to `position`, by `delta`, or straight down until the grasp centre is at
    `stop_z_m` (exactly one), that ENDS AS A SUCCESS when one of `until` fires: `contact` = a pad contact of the hand
    (the executor's contact guard; needs r1hands' pad-contact publisher), `blocked` = the lag / torque guard (the arm
    held back by something the world model does not know). Every guard stays armed (a guard always stops the arm; only
    the result's meaning changes: `stopped_early`). `touch` = declared entity ids the HAND may touch on the way (the
    planner's allow_hand: e.g. the surface the move is meant to reach)."""

    op: Literal["arm.guarded_move"]
    arm: Side
    position: Vec3 | None = None
    delta: Vec3 | None = None
    stop_z_m: float | None = None
    frame: Frame = "base"
    until: list[GuardEnd] = Field(default_factory=lambda: list[GuardEnd](["contact"]), max_length=2)
    touch: list[str] = Field(default_factory=list, max_length=8)
    speed: float = Field(0.2, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _one(self) -> ArmGuardedMoveReq:
        if sum(x is not None for x in (self.position, self.delta, self.stop_z_m)) != 1:
            raise ValueError("give exactly one of position, delta or stop_z_m")
        _finite_all(self, ("position", "delta", "stop_z_m"))
        return self


class ArmSlideReq(_Goal):
    """L2 (session 10): slide the hand along the declared support under it (a surface's top / a container's floor
    within `max_gap_m` below the lowest fingertip) by `delta` (x, y in `frame`; z within +-5 mm): a straight line the
    hand may keep touching the support on (the planner's allow_hand for that support), the lag guard armed. No press
    into the support (a pressing slide = the wipe question, not here)."""

    op: Literal["arm.slide"]
    arm: Side
    delta: Vec3
    frame: Frame = "base"
    support: str | None = None  # the support's entity id (default: the declared support under the hand)
    max_gap_m: float = Field(0.03, gt=0.0, le=0.10)
    speed: float = Field(0.2, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ok(self) -> ArmSlideReq:
        _finite_all(self, ("delta",))
        if abs(self.delta[2]) > 0.005:
            raise ValueError("delta: a slide keeps the hand's height (|dz| <= 5 mm)")
        return self


class SkillPickReq(_Goal):
    """L3 (session 10, generic): pick the object whose centre is at `position` with `arm`'s hand - hover (the lowest
    fingertip `hover_m` above the object's top, datum = the support top under it) -> straight down until the grasp
    centre is at `grasp_z_m` (default: the object's centre) -> hand.grasp (the verdict) -> lift `lift_m`. Each step an
    L2 goal (a child of this one); the hover is planned before anything moves; any failed step = SKILL_FAILED (a
    NOT_HOLDING grasp included). `object_id` = the object's declared entity id (the hand may touch it on the way)."""

    op: Literal["skill.pick"]
    arm: Side
    position: Vec3
    height_m: float = Field(0.05, gt=0.0, le=0.3)  # the object's height
    grasp_z_m: float | None = None
    grip: GripName = "wrap"
    object_class: ObjectClass = "rigid"
    width_m: float | None = Field(None, gt=0.0, le=0.2)
    squeeze: float | None = Field(None, ge=0.0, le=0.8)
    hover_m: float = Field(0.03, ge=0.01, le=0.3)
    lift_m: float = Field(0.05, ge=0.0, le=0.2)
    object_id: str | None = None
    frame: Frame = "base"
    speed: float = Field(0.4, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ok(self) -> SkillPickReq:
        _finite_all(self, ("position",))
        return self


class SkillPlaceReq(_Goal):
    """L3 (session 10, generic): place what `arm`'s hand holds - arm.carry (rise `lift_m`, travel, lower) until the
    grasp centre is at `position` -> hand.release -> retreat straight up `retreat_m`. Each step an L2 goal; the carry
    is planned before anything moves; a failed step = SKILL_FAILED."""

    op: Literal["skill.place"]
    arm: Side
    position: Vec3
    lift_m: float = Field(0.08, ge=0.0, le=0.3)
    retreat_m: float = Field(0.05, ge=0.0, le=0.2)
    frame: Frame = "base"
    speed: float = Field(0.4, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ok(self) -> SkillPlaceReq:
        _finite_all(self, ("position",))
        return self


class ParRunReq(_Goal):
    """L3 (session 10): run goal requests TOGETHER on disjoint resources (e.g. the left arm and the head); the goal
    ends when all have ended - succeeded if all did; with `stop_on_failure` the first failure cancels the others
    (STEP_FAILED with the step's index and result)."""

    op: Literal["par.run"]
    steps: list[dict[str, Any]] = Field(min_length=1, max_length=8)
    stop_on_failure: bool = True


class LeaseReq(_Req):
    """Leases (session 10, DESIGN §5): `lease.acquire {resources, client, ttl_s}` -> lease_id; `lease.renew {lease_id,
    ttl_s}`; `lease.release {lease_id}`; `lease.list`. While a lease is live, requests that move its resources must
    carry its id in `lease` (else NOT_OWNER); stop / halt / damp / arm.stop / status are never gated."""

    op: Literal["lease.acquire", "lease.renew", "lease.release", "lease.list"]
    resources: list[str] = Field(default_factory=list, max_length=10)
    client: str = Field("client", min_length=1, max_length=64)
    ttl_s: float = Field(30.0, gt=0.0, le=300.0)
    lease_id: str | None = None


class ArmTwistReq(_Req):
    """L1 (session 10): the hand's grasp centre at velocity `v` (m/s, base frame, |v| <= 0.10) for `ttl` -
    resolved rate from the arm model's Jacobian at every request, integrated at r1d's tick like stream.joint_jog;
    refused while the world has declared entities (no collision look-ahead yet)."""

    op: Literal["stream.arm_twist"]
    arm: Side
    v: Vec3
    ttl: float = Field(0.3, gt=0.0, le=0.5)

    @model_validator(mode="after")
    def _ok(self) -> ArmTwistReq:
        _finite_all(self, ("v",))
        if sum(x * x for x in self.v) > 0.10**2 + 1e-12:
            raise ValueError("v: |v| <= 0.10 m/s")
        return self


# <<< s10:api
# >>> s11:core (request models)
class BalanceReq(_Req):
    """OPERATOR ONLY (session 11): from a freeze of a balancing robot (`r1 freeze`, an invalid policy action, the
    state-age rung) back into BALANCE - the core's ENTER_BALANCE (service/balance.py: r1d's checks, then the core's:
    fresh state, never older than the state-age limit, a valid first action). Never from the stand's freeze."""

    op: Literal["balance"]


# <<< s11:core
# >>> s11:grip (request models)
# <<< s11:grip
# >>> s11:cam (request models)
# <<< s11:cam
# >>> s11:api (request models)
# <<< s11:api
# >>> s11:episode (request models)
# <<< s11:episode
# >>> s12:approach (request models)
# <<< s12:approach
# >>> s12:grip (request models)
# <<< s12:grip
# >>> s12:cam (request models)
# <<< s12:cam
# >>> s12:fixes (request models)
# <<< s12:fixes
# >>> s13:tools (request models)
# <<< s13:tools
# >>> s13:click (request models)
# <<< s13:click
# >>> s13:setup (request models)
# <<< s13:setup
# >>> s13:pose (request models)
# <<< s13:pose
# >>> s13:grip (request models)
# <<< s13:grip
# >>> s13:core (request models)
# <<< s13:core
# >>> s13:ops (request models)
# <<< s13:ops


Request = Annotated[
    StatusReq
    | StopReq
    | ResumeReq
    | FreezeReq
    | TakeoverReq
    | ReclaimReq
    | BoardHoldReq
    | GuardTakeoverReq
    | ShutdownReq
    | StandFreezeReq
    | StandHoldReq
    | JointCommandReq
    | JointHoldReq
    | JointJogReq
    | HeadStreamReq
    | HandCommandReq
    | HandStateReq
    | RecordReq
    | CameraReq
    | GoalStatusReq
    | GoalCancelReq
    | GoalWaitReq
    | GoalListReq
    | ArmMoveJointReq
    | ArmReachReq
    | ArmMoveLinearReq
    | ArmMoveRelativeReq
    | ArmHoverReq
    | ArmLowerReq
    | ArmCarryReq
    | ArmPlanReq
    | ArmStopReq
    | HandGraspReq
    | HandReleaseReq
    | HandPreshapeReq
    | HandTareReq
    | HeadLookAtReq
    | BodyPoseReq
    | SkillReq
    | SeqRunReq
    | WorldDeclareReq
    | WorldClearReq
    # >>> s10:cam (union members: '    | YourReq')
    | CameraObserveReq
    | CameraPointReq
    | CameraExtentReq
    | CameraProjectReq
    | CameraDepthImageReq
    # <<< s10:cam
    # >>> s10:measure (union members: '    | YourReq')
    | MeasureSetupReq
    | MeasurePhoneReq
    # <<< s10:measure
    # >>> s10:grasp (union members: '    | YourReq')
    | HandCalibrateReq
    # <<< s10:grasp
    # >>> s10:planner (union members: '    | YourReq')
    # <<< s10:planner
    # >>> s10:api (union members: '    | YourReq')
    | ArmMovePathReq
    | ArmGuardedMoveReq
    | ArmSlideReq
    | SkillPickReq
    | SkillPlaceReq
    | ParRunReq
    | LeaseReq
    | ArmTwistReq
    # <<< s10:api
    # >>> s11:core (union members: '    | YourReq')
    | BalanceReq
    # <<< s11:core
    # >>> s11:grip (union members: '    | YourReq')
    # <<< s11:grip
    # >>> s11:cam (union members: '    | YourReq')
    # <<< s11:cam
    # >>> s11:api (union members: '    | YourReq')
    # <<< s11:api
    # >>> s11:episode (union members: '    | YourReq')
    # <<< s11:episode
    # >>> s12:approach (union members: '    | YourReq')
    # <<< s12:approach
    # >>> s12:grip (union members: '    | YourReq')
    # <<< s12:grip
    # >>> s12:cam (union members: '    | YourReq')
    # <<< s12:cam
    # >>> s12:fixes (union members: '    | YourReq')
    # <<< s12:fixes
    # >>> s13:tools (union members: '    | YourReq')
    # <<< s13:tools
    # >>> s13:click (union members: '    | YourReq')
    # <<< s13:click
    # >>> s13:setup (union members: '    | YourReq')
    # <<< s13:setup
    # >>> s13:pose (union members: '    | YourReq')
    # <<< s13:pose
    # >>> s13:grip (union members: '    | YourReq')
    # <<< s13:grip
    # >>> s13:core (union members: '    | YourReq')
    # <<< s13:core
    # >>> s13:ops (union members: '    | YourReq')
    # <<< s13:ops
    | WorldListReq,
    Field(discriminator="op"),
]
REQUEST = TypeAdapter(Request)
OPS: tuple[str, ...] = (
    "status",
    "stop",
    "halt",
    "damp",
    "resume",
    "freeze",
    "takeover",
    "on_stand",
    "reclaim",
    "board_hold.pause",
    "guard_takeover.stall",
    "shutdown",
    "stand_freeze",
    "stand_hold",
    "joint.command",
    "joint.hold",
    "stream.joint_jog",
    "stream.head",
    "hand.command",
    "hand.state",
    "record.plan",
    "record.run",
    "record.list",
    "camera.status",
    "camera.services",
    # session 9: L2 / L3
    "goal.status",
    "goal.cancel",
    "goal.wait",
    "goal.list",
    "arm.move_joint",
    "arm.reach",
    "arm.move_linear",
    "arm.move_relative",
    "arm.hover",
    "arm.lower",
    "arm.carry",
    "arm.plan",
    "arm.stop",
    "hand.grasp",
    "hand.release",
    "hand.preshape",
    "hand.tare",
    "head.look_at",
    "body.pose",
    "skill.ready",
    "skill.table_ready",
    "seq.run",
    "world.declare",
    "world.clear",
    # >>> s10:cam (op names)
    "camera.observe",
    "camera.point",
    "camera.object_extent",
    "camera.project",
    "camera.depth_image",
    # <<< s10:cam
    # >>> s10:measure (op names)
    "measure.setup",
    "measure.phone",
    # <<< s10:measure
    # >>> s10:grasp (op names)
    "hand.calibrate",
    # <<< s10:grasp
    # >>> s10:planner (op names)
    # <<< s10:planner
    # >>> s10:api (op names)
    "arm.move_path",
    "arm.guarded_move",
    "arm.slide",
    "skill.pick",
    "skill.place",
    "par.run",
    "lease.acquire",
    "lease.renew",
    "lease.release",
    "lease.list",
    "stream.arm_twist",
    # <<< s10:api
    # >>> s11:core (op names)
    "balance",
    # <<< s11:core
    # >>> s11:grip (op names)
    # <<< s11:grip
    # >>> s11:cam (op names)
    # <<< s11:cam
    # >>> s11:api (op names)
    # <<< s11:api
    # >>> s11:episode (op names)
    # <<< s11:episode
    # >>> s12:approach (op names)
    # <<< s12:approach
    # >>> s12:grip (op names)
    # <<< s12:grip
    # >>> s12:cam (op names)
    # <<< s12:cam
    # >>> s12:fixes (op names)
    # <<< s12:fixes
    # >>> s13:tools (op names)
    # <<< s13:tools
    # >>> s13:click (op names)
    # <<< s13:click
    # >>> s13:setup (op names)
    # <<< s13:setup
    # >>> s13:pose (op names)
    # <<< s13:pose
    # >>> s13:grip (op names)
    # <<< s13:grip
    # >>> s13:core (op names)
    # <<< s13:core
    # >>> s13:ops (op names)
    # <<< s13:ops
    "world.list",
)
GOAL_OPS: tuple[str, ...] = (
    "arm.move_joint",
    "arm.reach",
    "arm.move_linear",
    "arm.move_relative",
    "arm.hover",
    "arm.lower",
    "arm.carry",
    "hand.grasp",
    "hand.release",
    "hand.preshape",
    "head.look_at",
    "skill.ready",
    "skill.table_ready",
    # >>> s10:cam (goal op names)
    # <<< s10:cam
    # >>> s10:measure (goal op names)
    # <<< s10:measure
    # >>> s10:grasp (goal op names)
    "hand.calibrate",
    # <<< s10:grasp
    # >>> s10:planner (goal op names)
    # <<< s10:planner
    # >>> s10:api (goal op names)
    "arm.move_path",
    "arm.guarded_move",
    "arm.slide",
    "skill.pick",
    "skill.place",
    "par.run",
    # <<< s10:api
    # >>> s11:core (goal op names)
    # <<< s11:core
    # >>> s11:grip (goal op names)
    # <<< s11:grip
    # >>> s11:cam (goal op names)
    # <<< s11:cam
    # >>> s11:api (goal op names)
    # <<< s11:api
    # >>> s11:episode (goal op names)
    # <<< s11:episode
    # >>> s12:approach (goal op names)
    # <<< s12:approach
    # >>> s12:grip (goal op names)
    # <<< s12:grip
    # >>> s12:cam (goal op names)
    # <<< s12:cam
    # >>> s12:fixes (goal op names)
    # <<< s12:fixes
    # >>> s13:tools (goal op names)
    # <<< s13:tools
    # >>> s13:click (goal op names)
    # <<< s13:click
    # >>> s13:setup (goal op names)
    # <<< s13:setup
    # >>> s13:pose (goal op names)
    # <<< s13:pose
    # >>> s13:grip (goal op names)
    # <<< s13:grip
    # >>> s13:core (goal op names)
    # <<< s13:core
    # >>> s13:ops (goal op names)
    # <<< s13:ops
    "seq.run",
)


def request_schema() -> dict[str, Any]:
    return REQUEST.json_schema()


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    code: Code
    op: str
    id: int | str | None = None
    moved: bool = False
    data: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


def ok(op: str, rid: int | str | None, *, moved: bool = False, **data: Any) -> Result:
    return Result(ok=True, code=Code.OK, op=op, id=rid, moved=moved, data=data)


def fail(op: str, rid: int | str | None, code: Code, **details: Any) -> Result:
    return Result(ok=False, code=code, op=op, id=rid, moved=False, details=details)
