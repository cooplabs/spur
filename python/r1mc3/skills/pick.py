"""The generic pick / place skills as L2 steps (DESIGN.md §5.1 L3: built only on L2; session 10).

Pure functions: a skill request (its fields as a dict) -> the list of L2 requests r1d runs one after another
(`r1mc3.service.l2_paths`, SeqRunner: the first failure ends the skill, SKILL_FAILED). Generic (D32): the client names
the object's centre, size and grip; nothing here knows a task, an object or a layout.

pick: hover (the lowest fingertip `hover_m` above the object's top, the datum = the support top under it; the pose
      chosen where the grip closes and the hover at least PICK_MIN_DESCENT_M above it: session 13) ->
      straight down until the grasp centre is at `grasp_z_m` (default: the object's centre; the hand may touch the
      declared object `object_id`) -> hand.grasp (the grip verdict: NOT_HOLDING fails the skill) -> lift `lift_m`.
place: carry (rise `lift_m`, travel, lower) until the grasp centre is at `position` -> hand.release -> retreat up.
"""

from __future__ import annotations

from typing import Any

# session 13 (D48, S12's blocker): the hover is chosen WHERE THE GRIP CLOSES (arm.hover grasp_z_m: the pose is ranked at
# the grasp height, the hover straight above it) and ends at least this far above that pose. Before, the hover was
# ranked at its own height and the grasp centre's height above the lowest fingertip depended on the pose the planner
# picked there: a level hand (the "avoid a rolled hand" ranking) left it 1.65 cm BELOW the client's grasp height and the
# straight-down step was refused ("stop_z_m: 0 < the depth below the grasp centre", tests/integration/
# test_service_l3_pick.py). 3 cm covers the grip point vs the grasp centre (<= ~2 cm apart) + the hover's tracking.
PICK_MIN_DESCENT_M = 0.03


def pick_steps(r: dict[str, Any], *, support_top_z: float) -> list[dict[str, Any]]:
    """skill.pick's steps. `support_top_z` = the top of what the object stands on (base z: the hover's datum)."""
    x, y, z = (float(v) for v in r["position"])
    arm, frame, speed = r["arm"], r.get("frame", "base"), float(r.get("speed", 0.4))
    height = float(r["height_m"])
    obj_top = max(support_top_z + height, z + height / 2)
    grasp_z = float(r["grasp_z_m"]) if r.get("grasp_z_m") is not None else z
    grasp: dict[str, Any] = {"op": "hand.grasp", "side": arm, "grip": r.get("grip", "wrap"),
                             "object_class": r.get("object_class", "rigid")}  # fmt: skip
    for k in ("width_m", "squeeze"):
        if r.get(k) is not None:
            grasp[k] = r[k]
    steps: list[dict[str, Any]] = [
        {
            "op": "arm.hover",
            "arm": arm,
            "position": [x, y, support_top_z],
            "frame": frame,
            "speed": speed,
            "height_m": round(obj_top - support_top_z + float(r.get("hover_m", 0.03)), 4),
            **({"grip": r["grip"]} if r.get("grip") else {}),
            "grasp_z_m": round(grasp_z, 4),
            "min_descent_m": PICK_MIN_DESCENT_M,
            **({"target_id": r["object_id"]} if r.get("object_id") else {}),
        },
        {
            "op": "arm.guarded_move",
            "arm": arm,
            "stop_z_m": grasp_z,
            "frame": frame,
            "until": [],
            "touch": [r["object_id"]] if r.get("object_id") else [],
            "speed": min(speed, 0.3),
        },
        grasp,
    ]
    if float(r.get("lift_m", 0.0)) > 0.0:
        steps.append({"op": "arm.move_linear", "arm": arm, "delta": [0.0, 0.0, float(r["lift_m"])], "frame": frame,
                      "speed": min(speed, 0.3)})  # fmt: skip
    return steps


def place_steps(r: dict[str, Any]) -> list[dict[str, Any]]:
    """skill.place's steps: carry to `position` (the grasp centre there), release, retreat straight up."""
    arm, frame, speed = r["arm"], r.get("frame", "base"), float(r.get("speed", 0.4))
    steps: list[dict[str, Any]] = [
        {
            "op": "arm.carry",
            "arm": arm,
            "position": [float(v) for v in r["position"]],
            "frame": frame,
            "lift_m": float(r.get("lift_m", 0.08)),
            "speed": speed,
        },
        {"op": "hand.release", "side": arm},
    ]
    if float(r.get("retreat_m", 0.0)) > 0.0:
        steps.append({"op": "arm.move_linear", "arm": arm, "delta": [0.0, 0.0, float(r["retreat_m"])],
                      "frame": frame, "speed": min(speed, 0.3)})  # fmt: skip
    return steps
