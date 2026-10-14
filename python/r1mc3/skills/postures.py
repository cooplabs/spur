"""Named arm postures for the L3 skills (`skill.ready`, `skill.table_ready`), 5 joints per arm in SDK order
(shoulder pitch, shoulder roll, shoulder yaw, elbow, wrist roll). Directions (docs/hardware/JOINT_DIRECTIONS.md,
verified on the robot 11 Oct): pitch + = the limb BACK, roll + = toward the robot's LEFT on both sides, a SMALLER elbow
value = MORE flexed (forearm up). Each posture names its source; the skills move one arm at a time (v2's lesson: the
first arm up shifts the body) through `arm.move_joint` (world-checked when r1d has a planner).

Postures (left arm; the right = its mirror unless named):
* ready - the balance policy's default arm pose (forearms forward-down, hands at the hips): the live policy's default
  pose R1MC-Manip2 HOME (r1sim.joints.POSE_STAND = the ONNX default_joint_pos; the arms the policy was trained
  around); the right arm = POSE_STAND's own right arm.
* table_ready - the forearm up and forward, the palm turned in (elbow -0.527, wrist -1.553): the left arm as v2's
  `skill table_ready` left it on the real robot, 9 Oct 2026 (sim/r1sim/dds_bridge.py ARMS_TABLE_READY, the pose B7
  balanced 100 / 100 seeds). **The right arm's mirror is NOT hardware-proven and B7 did not test both arms up** - say
  so before it is used on the robot.
"""

from __future__ import annotations

# mirror: roll, yaw and wrist roll change sign between the arms (the SDK's left / right conventions, joints.py)
_MIRROR = (1.0, -1.0, -1.0, 1.0, -1.0)


def _mirror(q: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(m * v for m, v in zip(_MIRROR, q, strict=True))


_READY_LEFT = (0.22, 0.193, 0.039, 0.76, 0.0)
_READY_RIGHT = (0.22, -0.194, -0.033, 0.76, 0.0)  # POSE_STAND's own right arm (not an exact mirror)
_TABLE_READY_LEFT = (0.0689, 0.685, 0.0192, -0.5269, -1.5534)

POSTURES: dict[str, dict[str, tuple[float, ...]]] = {
    "ready": {"left": _READY_LEFT, "right": _READY_RIGHT},
    "table_ready": {"left": _TABLE_READY_LEFT, "right": _mirror(_TABLE_READY_LEFT)},
}


def posture(name: str, arm: str) -> list[float]:
    """The posture's 5 joint targets for `arm` (KeyError for an unknown name / arm)."""
    return list(POSTURES[name][arm])
