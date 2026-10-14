"""Error codes (DESIGN.md §5) and the core's enumerations as r1d uses them.

The core's numbers (modes, requests, requesters, groups) come from the C++ binding when it exports them
(`r1rt_py.modes()`, `requests()`, `requesters()`, `groups()`: the session-5 mailbox contract), so a renumbering in
the headers cannot silently desynchronise r1d. The fallback tables below are the ABI-4 headers' values
(rt/include/r1rt/types.h, step_api.h) and are used only when the binding predates those functions (it then also
lacks the mailbox: the mode ops answer UNSUPPORTED).
"""

from __future__ import annotations

from enum import StrEnum
from types import ModuleType
from typing import Any


class Code(StrEnum):
    """Result codes (DESIGN §5: codes with structured details, never prose matched by keywords, R-h)."""

    OK = "OK"
    INVALID_ARGS = "INVALID_ARGS"
    UNKNOWN_COMMAND = "UNKNOWN_COMMAND"
    WRONG_MODE = "WRONG_MODE"
    NOT_OWNER = "NOT_OWNER"
    RESOURCE_BUSY = "RESOURCE_BUSY"
    LIMIT = "LIMIT"
    CONTROLLER_REJECTED = "CONTROLLER_REJECTED"
    FAULT = "FAULT"
    TIMEOUT = "TIMEOUT"
    HAND_FAULT = "HAND_FAULT"
    NOT_PERMITTED = "NOT_PERMITTED"  # e.g. damp from a non-operator client
    UNSUPPORTED = "UNSUPPORTED"  # the connected core / binding does not implement it (yet)
    RELEASE_FAILED = "RELEASE_FAILED"  # the motion switcher did not release Unitree's controller (takeover aborted)
    # session 9 (L2 / L3, DESIGN §5)
    UNKNOWN_FRAME = "UNKNOWN_FRAME"  # a frame that is neither "base" nor a declared world entity
    STALE_FRAME = "STALE_FRAME"  # a measured frame older than its validity
    UNKNOWN_GOAL = "UNKNOWN_GOAL"  # goal.status / cancel / wait with an id r1d does not know
    UNREACHABLE = "UNREACHABLE"  # no joint solution / path inside the limits
    COLLISION = "COLLISION"  # every candidate path collides (self, the other arm, the world)
    GUARD_STOP = "GUARD_STOP"  # the core's executor stopped the motion (a guard, the deadman, a starved path)
    SKILL_FAILED = "SKILL_FAILED"  # an L3 skill's step failed (details: the step and its result)
    STEP_FAILED = "STEP_FAILED"  # seq.run: a step failed (details: index + the step's result)
    NOT_HOLDING = "NOT_HOLDING"  # the grip verdict says the hand holds nothing
    CANCELED = "CANCELED"  # the goal was canceled (goal.cancel / arm.stop / an operator stop)
    PREEMPTED = "PREEMPTED"  # a newer goal on the same resources replaced it (`preempt`)
    # session 10 (each agent adds its codes between its markers)
    # >>> s10:cam
    NOT_CALIBRATED = "NOT_CALIBRATED"  # pointing without a head-camera calibration (v2's rule: refuse)
    NO_DEPTH = "NO_DEPTH"  # no depth where it is needed (a frame observed without depth, no reading there)
    OFF_SURFACE = "OFF_SURFACE"  # the pixel's line of sight meets no declared surface / the point lies off it
    # <<< s10:cam
    # >>> s10:measure
    NOT_STILL = "NOT_STILL"  # a measurement needs the camera still: the head moved during a frame
    # <<< s10:measure
    # >>> s10:grasp
    CALIBRATION_FAILED = "CALIBRATION_FAILED"  # hand.calibrate: no valid reference (why in the details; the old kept)
    # <<< s10:grasp
    # >>> s10:planner
    # <<< s10:planner
    # >>> s10:api
    # <<< s10:api
    # >>> s10:core
    # <<< s10:core
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
    STOPPED = "STOPPED"  # the core's stop / halt is latched: the base command (the policy's lean) is 0 until a resume
    # <<< s13:core
    # >>> s13:ops
    # <<< s13:ops


# ---- the core's enumerations (ABI 4 fallbacks; the binding's tables win) ----
MODES_ABI4: dict[str, int] = {
    "monitor": 0,
    "takeover_mirror": 1,
    "takeover_blend": 2,
    "balance": 3,
    "guarded": 4,
    "stand_freeze": 5,
    "on_stand": 6,
    "overlay_arm_sdk": 7,
    "damp": 8,
}
REQUESTS_ABI4: dict[str, int] = {
    "enter_balance": 1,
    "stop": 2,
    "resume": 3,
    "damp": 4,
    "stand_freeze": 5,
    "takeover": 6,
    "takeover_released": 7,
    "takeover_abort": 8,
}
OWNERS: dict[int, str] = {0: "none", 1: "r1rt", 2: "r1guard"}
STOP_LEVELS: dict[str, int] = {"stop": 0, "halt": 1, "damp": 2}  # types.h R1RT_STOP_*
SOURCES: dict[str, int] = {"hold": 0, "trajectory": 1, "stream": 2, "freedrive": 3}  # types.h R1RT_SOURCE_*
# upper-body groups (types.h R1RT_GROUP_*) and their joints (SDK order, r1sim.joints / executor.hpp)
UPPER_GROUPS: dict[str, int] = {"arm_left": 0, "arm_right": 1, "head": 2}
GROUP_JOINTS: dict[str, tuple[int, ...]] = {
    "arm_left": (14, 15, 16, 17, 18),
    "arm_right": (19, 20, 21, 22, 23),
    "head": (24, 25),
    "leg_left": (0, 1, 2, 3, 4, 5),
    "leg_right": (6, 7, 8, 9, 10, 11),
    "waist": (12, 13),
}
LOWER_GROUPS = ("leg_left", "leg_right", "waist")  # L0 only in ON_STAND (DESIGN §5.1)
# layout 2 (session 5, types.h R1RT_GROUP_*): the leg / waist groups of ON_STAND's per-joint L0 (6-joint records)
GROUPS_LAYOUT2: dict[str, int] = {**UPPER_GROUPS, "leg_left": 3, "leg_right": 4, "waist": 5}
# The 26 joints in SDK order (r1sim.joints.JOINTS; v2 r1mc2/joints.py) - names for the CLI / the recording plans.
JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_roll",
    "waist_yaw",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "head_pitch",
    "head_yaw",
)


def group_of_joint(j: int) -> str:
    """The smallest named group holding joint j (arm_left / arm_right / head / leg_left / leg_right / waist)."""
    for g in ("arm_left", "arm_right", "head", "leg_left", "leg_right", "waist"):
        if j in GROUP_JOINTS[g]:
            return g
    raise ValueError(f"joint index {j} out of range")


class CoreTables:
    """The core's numbers from the binding (or the ABI-4 fallbacks), resolved once per connection."""

    def __init__(self, mod: ModuleType | Any | None) -> None:
        def table(name: str, fallback: dict[str, int]) -> tuple[dict[str, int], bool]:
            fn = getattr(mod, name, None) if mod is not None else None
            if callable(fn):
                return {str(k): int(v) for k, v in dict(fn()).items()}, True
            return dict(fallback), False

        self.modes, self.from_binding = table("modes", MODES_ABI4)
        self.requests, _ = table("requests", REQUESTS_ABI4)
        self.requesters, self.has_requesters = table("requesters", {})
        layout = int(getattr(mod, "LAYOUT_VERSION", 1)) if mod is not None else 1
        self.groups, self.has_groups = table("groups", GROUPS_LAYOUT2 if layout >= 2 else UPPER_GROUPS)
        self.mode_names = {v: k for k, v in self.modes.items()}
        self.refusals, _ = table("refusals", {})  # the core's refusal codes by name (gates, mode, operator, ...)
        self.status_codes, _ = table("status_codes", {})
        self.refusal_names = {v: k for k, v in self.refusals.items()}
        self.status_names = {v: k for k, v in self.status_codes.items()}

    def refusal_name(self, refusal: int) -> str | None:
        return None if int(refusal) == 0 else self.refusal_names.get(int(refusal), f"refusal_{int(refusal)}")

    def status_name(self, status: int) -> str:
        return self.status_names.get(int(status), f"status_{int(status)}")

    def mode_name(self, mode: int) -> str:
        return self.mode_names.get(int(mode), f"mode_{int(mode)}")

    def requester(self, name: str) -> int | None:
        """The id the core knows a requester by (None when the binding has no requester table)."""
        return self.requesters.get(name)
