"""The joint-direction check (HW1 fix list #38): what v3's model says each joint does for a small POSITIVE step, in
the words an operator watching the robot uses, and the step plan that shows it on the robot.

Why it exists: the user's requirement at HW1 part 2 (11 Oct 2026, "i dont want a reverse movement which can mess
everything up"). A joint whose sign is mirrored between the model / policy and the robot would make every controller
push the wrong way. HW1 checked all 26 joints live: each moved +0.10 rad (ankles / wrists / head +0.15, head pitch
re-run at +0.20) and back while the user watched, and every answer matched this model's prediction (26 / 26,
docs/hardware/JOINT_DIRECTIONS.md; NOTES.md 15:08-15:16 PDT). Re-run it after ANY firmware, model or joint-table
change (the user's rule: never guess a direction, check it against the model).

What it computes (the session's `dir_expect.py`, now in the repo): the real-profile model at a pose, the torso held
(the torso is what the stand holds, so the torso frame is the frame the operator sees: x forward, y = the robot's
LEFT, z up); for each joint a point on the moving limb (the sole, the toe, the outer edge of the foot, the palm, the
front of the hand, the face) - its displacement and the limb's rotation for +delta on that joint alone. Each joint's
PRIMARY motion is the one the operator judges by: a translation for the joints that swing a limb (hip pitch / roll,
knee, waist roll, shoulder pitch / roll, elbow), a rotation for the joints that turn or tilt the end (yaws, ankle pitch
/ roll, wrist roll, head pitch). `HW1_VERIFIED` holds what the user saw on 11 Oct; tests/unit/test_hw1_dircheck.py
checks the model still predicts exactly that.

    python -m r1mc3.hw1.dircheck                       # the table at the HW1 check pose
    python -m r1mc3.hw1.dircheck --pose stand_lock     # at Unitree's stand-lock preset (docs/hardware/STAND_LOCK.md)
    python -m r1mc3.hw1.dircheck --plan --json out.json   # + the `r1 record run static_pose` steps, JSON to a file

Nothing here moves the robot: `--plan` only PRINTS the words (run them one at a time, the robot on its stand in
ON_STAND, the user watching; how to stop: `r1 stop`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# The 26 body joints in SDK order (r1sim.joints.JOINTS; repeated so the plan / words work without the simulator).
JOINTS: tuple[str, ...] = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_roll", "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
    "head_pitch", "head_yaw",
)  # fmt: skip

# r1d's groups (python/r1mc3/service/codes.py GROUP_JOINTS): the step plan moves one group's setpoint at a time.
GROUPS: dict[str, tuple[int, ...]] = {
    "leg_left": (0, 1, 2, 3, 4, 5),
    "leg_right": (6, 7, 8, 9, 10, 11),
    "waist": (12, 13),
    "arm_left": (14, 15, 16, 17, 18),
    "arm_right": (19, 20, 21, 22, 23),
    "head": (24, 25),
}

# The pose of the HW1 check (11 Oct 2026 15:08 PDT): v3's ON_STAND legs (the stand pose the S2-S10 steps returned
# to) and the arms as the user corrected them before the board-hold test (sh_p 0, sh_r +-0.19, elbow 1.40 = forearm
# LOWERED).
HW1_POSE: tuple[float, ...] = (
    -0.3, 0.025, -0.13, 0.455, -0.2136, -0.05,
    -0.3, -0.025, 0.13, 0.455, -0.2136, 0.05,
    0.0, 0.0,
    0.0, 0.19, 0.0, 1.40, 0.0,
    0.0, -0.19, 0.0, 1.40, 0.0,
    0.0, 0.0,
)  # fmt: skip

# Unitree's STAND LOCK = a FIXED preset pose (HW1 part 2 S1.0, 133 constant rt/lowcmd messages; fix list #25;
# docs/hardware/STAND_LOCK.md). The robot moves INTO it when the stand lock is selected.
STAND_LOCK_PRESET: tuple[float, ...] = (
    -0.305, 0.04, -0.14, 0.41, -0.137, -0.06,
    -0.305, -0.04, 0.14, 0.41, -0.137, 0.06,
    0.0, 0.0,
    0.2, 0.2, 0.0, 0.8, 0.0,
    0.2, -0.2, 0.0, 0.8, 0.0,
    0.0, 0.0,
)  # fmt: skip

POSES: dict[str, tuple[float, ...]] = {"hw1": HW1_POSE, "stand_lock": STAND_LOCK_PRESET}

# The step HW1 used per joint (rad): 0.10; the ankles, wrists and head 0.15 (smaller links: easier to see); the head
# pitch was re-run at 0.20 (the user missed the first, 0.15, move). Out in 1.5 s, hold 2 s, back in 1.5 s, hold 1 s
# (a return with hold_s 0.5 was refused at HW1: the static_pose plan needs hold_s >= 1).
STEP_RAD: dict[str, float] = {
    **{j: 0.10 for j in JOINTS},
    **{j: 0.15 for j in JOINTS if "ankle" in j or "wrist" in j or j.startswith("head")},
    "head_pitch": 0.20,
}
MOVE_S = 1.5
HOLD_S = 2.0
BACK_HOLD_S = 1.0

# Which point the operator watches, and whether that joint is judged by where the point goes ("move") or by how the
# limb turns ("turn").
_POINT_KIND: dict[str, tuple[str, str]] = {
    "hip_pitch": ("sole", "move"),
    "hip_roll": ("sole", "move"),
    "hip_yaw": ("toe", "turn"),
    "knee": ("sole", "move"),
    "ankle_pitch": ("toe", "turn"),
    "ankle_roll": ("outer_edge", "turn"),
    "shoulder_pitch": ("hand", "move"),
    "shoulder_roll": ("hand", "move"),
    "shoulder_yaw": ("hand_front", "turn"),
    "elbow": ("hand", "move"),
    "wrist_roll": ("hand_front", "turn"),
}

# Primary-motion keys (torso frame): translations of the watched point, rotations of its limb.
MOVE_KEYS = ("forward", "back", "left", "right", "up", "down")
TURN_KEYS = ("roll_left_up", "roll_left_down", "pitch_down", "pitch_up", "turn_left", "turn_right")


@dataclass(frozen=True)
class Verified:
    """What the user saw on the robot at HW1 (11 Oct 2026) for +STEP_RAD on one joint, and the primary key it is."""

    key: str
    seen: str


# The 26 answers of the live check (NOTES.md "Direction check", 15:08-15:16 PDT; HW1.md table). Every one matched the
# model; the two first answers that did not (left ankle pitch "toes inward", head pitch "head goes up") were re-watched
# and corrected by the user (toes DOWN; the head went down - the first answer was the return).
HW1_VERIFIED: dict[str, Verified] = {
    "left_hip_pitch": Verified("back", "the leg swings BACKWARD"),
    "left_hip_roll": Verified("left", "the leg swings OUTWARD"),
    "left_hip_yaw": Verified("turn_left", "the toes turn OUT"),
    "left_knee": Verified("back", "the knee BENDS MORE"),
    "left_ankle_pitch": Verified("pitch_down", "the toes go DOWN"),
    "left_ankle_roll": Verified("roll_left_up", "the OUTER edge of the foot goes UP"),
    "right_hip_pitch": Verified("back", "the leg swings BACKWARD"),
    "right_hip_roll": Verified("left", "the leg swings INWARD"),
    "right_hip_yaw": Verified("turn_left", "the toes turn INWARD"),
    "right_knee": Verified("back", "the knee BENDS MORE"),
    "right_ankle_pitch": Verified("pitch_down", "the toes go DOWN"),
    "right_ankle_roll": Verified("roll_left_up", "the INNER edge of the foot goes UP"),
    "waist_roll": Verified("right", "the legs swing to the robot's RIGHT"),
    "waist_yaw": Verified("turn_right", "the pelvis rotates to the RIGHT (the toes turn right)"),
    "left_shoulder_pitch": Verified("back", "the arm swings BACKWARD"),
    "left_shoulder_roll": Verified("left", "the arm swings OUTWARD"),
    "left_shoulder_yaw": Verified("turn_left", "the hand turns toward the robot's LEFT"),
    "left_elbow": Verified("back", "the elbow STRAIGHTENS a little"),
    "left_wrist_roll": Verified("turn_right", "the front of the hand rotates INWARD"),
    "right_shoulder_pitch": Verified("back", "the arm swings BACKWARD"),
    "right_shoulder_roll": Verified("left", "the arm swings INWARD"),
    "right_shoulder_yaw": Verified("turn_left", "the hand turns inward, to the robot's LEFT"),
    "right_elbow": Verified("back", "the elbow STRAIGHTENS"),
    "right_wrist_roll": Verified("turn_right", "the forearm / hand turns OUTWARD, to the robot's RIGHT"),
    "head_pitch": Verified("pitch_down", "the head nods DOWN"),
    "head_yaw": Verified("turn_left", "the head turns LEFT"),
}


def point_and_kind(joint: str) -> tuple[str, str]:
    """The watched point (e.g. "left_sole", "face") and "move" / "turn" for one joint."""
    if joint == "waist_roll":
        return "left_sole", "move"  # the torso is held: the waist moves the pelvis + both legs
    if joint == "waist_yaw":
        return "left_toe", "turn"
    if joint.startswith("head"):
        return "face", "turn"
    side, part = joint.split("_", 1)
    pt, kind = _POINT_KIND[part]
    return f"{side}_{pt}", kind


def operator_words(joint: str, key: str) -> str:
    """The plain words an operator watching the robot uses for `key` on `joint` (inward / outward depend on the side:
    for the right limbs the robot's LEFT is inward)."""
    side = "left" if joint.startswith("left_") else "right" if joint.startswith("right_") else ""
    part = joint.split("_", 1)[1] if side else joint
    lateral = {"left": "OUTWARD" if side == "left" else "INWARD", "right": "INWARD" if side == "left" else "OUTWARD"}
    if part in ("hip_pitch", "shoulder_pitch"):
        limb = "leg" if part.startswith("hip") else "arm"
        return {"back": f"the {limb} swings BACKWARD", "forward": f"the {limb} swings FORWARD"}.get(key, key)
    if part in ("hip_roll", "shoulder_roll"):
        limb = "leg" if part.startswith("hip") else "arm"
        return f"the {limb} swings {lateral[key]}" if key in lateral else key
    if part == "knee":
        return {"back": "the knee BENDS MORE", "forward": "the knee STRAIGHTENS"}.get(key, key)
    if part == "elbow":
        # a LARGER elbow value = straighter (HW1 correction): +step moves the hand back toward hanging
        return {"back": "the elbow STRAIGHTENS", "forward": "the elbow FLEXES (forearm up)"}.get(key, key)
    if part == "hip_yaw":
        toes_out = (key == "turn_left") == (side == "left")
        return "the toes turn OUT" if toes_out else "the toes turn INWARD"
    if part == "ankle_pitch":
        return {"pitch_down": "the toes go DOWN", "pitch_up": "the toes go UP"}.get(key, key)
    if part == "ankle_roll":
        outer_up = (key == "roll_left_up") == (side == "left")
        return "the OUTER edge of the foot goes UP" if outer_up else "the INNER edge of the foot goes UP"
    if part == "shoulder_yaw":
        side_word = {"turn_left": "LEFT", "turn_right": "RIGHT"}.get(key)
        return f"the hand turns toward the robot's {side_word}" if side_word else key
    if part == "wrist_roll":
        inward = (key == "turn_right") == (side == "left")
        return "the front of the hand rotates INWARD" if inward else "the front of the hand rotates OUTWARD"
    if joint == "waist_roll":
        side_word = {"right": "RIGHT", "left": "LEFT"}.get(key)
        return f"the legs swing to the robot's {side_word}" if side_word else key
    if joint == "waist_yaw":
        side_word = {"turn_right": "RIGHT", "turn_left": "LEFT"}.get(key)
        return f"the pelvis rotates to the {side_word} (the toes turn {side_word.lower()})" if side_word else key
    if joint == "head_pitch":
        return {"pitch_down": "the head nods DOWN", "pitch_up": "the head tilts UP"}.get(key, key)
    if joint == "head_yaw":
        return {"turn_left": "the head turns LEFT", "turn_right": "the head turns RIGHT"}.get(key, key)
    return key


@dataclass(frozen=True)
class Expectation:
    """The model's prediction for +delta on one joint at a pose (torso frame: x forward, y the robot's LEFT, z up)."""

    joint: str
    delta_rad: float
    point: str
    kind: str  # "move" (judged by the point's displacement) or "turn" (by the limb's rotation)
    d_cm: tuple[float, float, float]  # the point's displacement, cm
    rot: tuple[float, float, float]  # the limb's rotation vector (small angle), rad
    key: str  # the primary motion (MOVE_KEYS / TURN_KEYS)
    words: str  # the displacement in words (the session's dir_expect.json "words")
    rot_words: str  # the rotation in words
    operator: str  # what the operator should see


def _move_key(d: FloatArray) -> str:
    i = int(np.argmax(np.abs(d)))
    return (("forward", "back"), ("left", "right"), ("up", "down"))[i][0 if d[i] > 0 else 1]


def _turn_key(w: FloatArray) -> str:
    i = int(np.argmax(np.abs(w)))
    return (("roll_left_up", "roll_left_down"), ("pitch_down", "pitch_up"), ("turn_left", "turn_right"))[i][
        0 if w[i] > 0 else 1
    ]


def _move_words(d: FloatArray) -> str:
    words = []
    for v, pos, neg in (
        (d[0], "forward", "back"),
        (d[1], "to the robot's LEFT", "to the robot's RIGHT"),
        (d[2], "up", "down"),
    ):
        if abs(v) >= 0.35 * float(np.abs(d).max()):
            words.append(f"{pos if v > 0 else neg} {abs(v):.1f} cm")
    return ", ".join(words)


def _rot_words(w: FloatArray) -> str:
    out = []
    for v, pos, neg in (
        (w[0], "rolls: left side UP (toward the robot's right)", "rolls: left side DOWN (toward the robot's left)"),
        (w[1], "pitches: front DOWN", "pitches: front UP"),
        (w[2], "turns: front toward the robot's LEFT", "turns: front toward the robot's RIGHT"),
    ):
        if abs(v) >= 0.5 * float(np.abs(w).max()):
            out.append(pos if v > 0 else neg)
    return "; ".join(out)


class _Kinematics:
    """The real-profile model with the torso as the reference (the base at the origin, unrotated: the torso's own
    pose is then fixed by the waist only, and every point is expressed in the torso frame)."""

    def __init__(self, profile: str = "real") -> None:
        import mujoco
        from r1sim import model as M

        self._mj = mujoco
        self.m = M.build_model(profile, visual=False)
        self.d = mujoco.MjData(self.m)
        self.ix = M.RobotIndex(self.m)
        self.points: dict[str, tuple[int, FloatArray]] = {}
        self._foot_sites = M.FOOT_SITES
        self._head_site = M.HEAD_CAMERA_SITE

    def set_q(self, q26: Sequence[float]) -> None:
        d, ix = self.d, self.ix
        d.qpos[:] = 0.0
        d.qpos[ix.base_qadr + 3] = 1.0
        for k in range(len(JOINTS)):
            if ix.present[k]:
                d.qpos[ix.qadr[k]] = q26[k]
        self._mj.mj_kinematics(self.m, self.d)

    def _site(self, name: str) -> int:
        return int(self._mj.mj_name2id(self.m, self._mj.mjtObj.mjOBJ_SITE, name))

    def _body(self, name: str) -> int:
        return int(self._mj.mj_name2id(self.m, self._mj.mjtObj.mjOBJ_BODY, name))

    def _local(self, body: int, p_world: FloatArray) -> tuple[int, FloatArray]:
        return body, self.d.xmat[body].reshape(3, 3).T @ (p_world - self.d.xpos[body])

    def anchor_points(self, q26: Sequence[float]) -> None:
        """Fix the watched points in their bodies at the pose `q26` (the torso's axes give "forward" / "left")."""
        self.set_q(q26)
        tb = self.ix.torso_body
        r0 = self.d.xmat[tb].reshape(3, 3)
        ex, ey = r0[:, 0], r0[:, 1]
        pts: dict[str, tuple[int, FloatArray]] = {}
        for side, sg, site in (("left", 1.0, self._foot_sites[0]), ("right", -1.0, self._foot_sites[1])):
            sid = self._site(site)
            b, p = int(self.m.site_bodyid[sid]), self.d.site_xpos[sid].copy()
            pts[f"{side}_sole"] = self._local(b, p)
            pts[f"{side}_toe"] = self._local(b, p + 0.12 * ex)
            pts[f"{side}_outer_edge"] = self._local(b, p + sg * 0.05 * ey)
            palm = self._body(f"{side}_hand_palm_link")
            pts[f"{side}_hand"] = self._local(palm, self.d.xpos[palm].copy())
            pts[f"{side}_hand_front"] = self._local(palm, self.d.xpos[palm] + 0.06 * ex)
        cam = self._site(self._head_site)
        pts["face"] = self._local(int(self.m.site_bodyid[cam]), self.d.site_xpos[cam].copy())
        self.points = pts

    def where(self, point: str) -> tuple[FloatArray, FloatArray]:
        """The point (torso frame, m) and its body's rotation relative to the torso."""
        b, pl = self.points[point]
        tb = self.ix.torso_body
        rt = self.d.xmat[tb].reshape(3, 3)
        rb = self.d.xmat[b].reshape(3, 3)
        p = rt.T @ (self.d.xpos[b] + rb @ pl - self.d.xpos[tb])
        return p, rt.T @ rb


def expectations(
    pose: Sequence[float] = HW1_POSE, *, delta: float | Mapping[str, float] = 0.10, profile: str = "real"
) -> dict[str, Expectation]:
    """The model's prediction for +delta on each joint alone, from `pose` (26 values, SDK order)."""
    if len(pose) != len(JOINTS):
        raise ValueError(f"pose: {len(JOINTS)} values, got {len(pose)}")
    kin = _Kinematics(profile)
    kin.anchor_points(pose)
    out: dict[str, Expectation] = {}
    for k, jn in enumerate(JOINTS):
        if not kin.ix.present[k]:
            continue
        dl = float(delta[jn]) if isinstance(delta, Mapping) else float(delta)
        point, kind = point_and_kind(jn)
        kin.set_q(pose)
        p0, r0 = kin.where(point)
        q = list(pose)
        q[k] += dl
        kin.set_q(q)
        p1, r1 = kin.where(point)
        dv = (p1 - p0) * 100.0
        rd = r1 @ r0.T
        w = np.array([rd[2, 1] - rd[1, 2], rd[0, 2] - rd[2, 0], rd[1, 0] - rd[0, 1]]) / 2.0
        key = _move_key(dv) if kind == "move" else _turn_key(w)
        out[jn] = Expectation(
            joint=jn,
            delta_rad=dl,
            point=point,
            kind=kind,
            d_cm=(round(float(dv[0]), 2), round(float(dv[1]), 2), round(float(dv[2]), 2)),
            rot=(round(float(w[0]), 3), round(float(w[1]), 3), round(float(w[2]), 3)),
            key=key,
            words=_move_words(dv),
            rot_words=_rot_words(w),
            operator=operator_words(jn, key),
        )
    return out


def step_plan(pose: Sequence[float] = HW1_POSE) -> list[dict[str, Any]]:
    """The HW1 step per joint as `r1 record run static_pose` words (one group's setpoint: only that joint changes)."""
    steps: list[dict[str, Any]] = []
    for group, idx in GROUPS.items():
        base = [round(float(pose[i]), 4) for i in idx]
        for n, j in enumerate(idx):
            jn = JOINTS[j]
            up = list(base)
            up[n] = round(up[n] + STEP_RAD[jn], 4)
            fmt = "[" + ",".join(f"{v:g}" for v in up) + "]"
            back = "[" + ",".join(f"{v:g}" for v in base) + "]"
            steps.append(
                dict(
                    joint=jn,
                    group=group,
                    step_rad=STEP_RAD[jn],
                    out=f"r1 record run static_pose group={group} 'q={fmt}' move_s={MOVE_S:g} hold_s={HOLD_S:g} "
                    f"--label dir_{group}_{n} --yes",
                    back=f"r1 record run static_pose group={group} 'q={back}' move_s={MOVE_S:g} "
                    f"hold_s={BACK_HOLD_S:g} --label dir_{group}_{n}_back --yes",
                )
            )
    return steps


def check_against_hw1(exp: Mapping[str, Expectation]) -> dict[str, bool]:
    """Per joint: does the model's primary motion equal what the user saw at HW1?"""
    return {jn: jn in exp and exp[jn].key == v.key for jn, v in HW1_VERIFIED.items()}


def table(exp: Mapping[str, Expectation]) -> str:
    ok = check_against_hw1(exp)
    lines = [f"{'joint':22s} {'step':>5s}  {'watch':17s} {'model (torso frame)':44s} operator sees  [HW1 11 Oct]"]
    for jn, e in exp.items():
        model = e.words if e.kind == "move" else e.rot_words
        seen = "match" if ok.get(jn) else f"DIFFERS from HW1: {HW1_VERIFIED[jn].seen}" if jn in HW1_VERIFIED else "-"
        lines.append(f"{jn:22s} {e.delta_rad:5.2f}  {e.point:17s} {model[:44]:44s} {e.operator}  [{seen}]")
    n_ok = sum(ok.values())
    lines.append(f"{n_ok} / {len(HW1_VERIFIED)} joints: the model's primary motion = what the user saw at HW1")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1mc3.hw1.dircheck", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--pose", default="hw1", help="hw1 (the HW1 check pose), stand_lock, or 26 comma-separated values")
    ap.add_argument("--delta", type=float, default=None, help="one step for every joint (default: HW1's per joint)")
    ap.add_argument("--plan", action="store_true", help="also print the step words (nothing is sent)")
    ap.add_argument("--json", default=None, help="write the expectations (+ plan) as JSON to this file")
    a = ap.parse_args(argv)
    pose = POSES[a.pose] if a.pose in POSES else tuple(float(x) for x in a.pose.strip("[]").split(","))
    exp = expectations(pose, delta=a.delta if a.delta is not None else STEP_RAD)
    print(table(exp))
    plan = step_plan(pose) if a.plan else []
    for s in plan:
        print(f"\n# {s['joint']} +{s['step_rad']:g} rad - expect: {exp[s['joint']].operator}\n{s['out']}\n{s['back']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"pose": list(pose), "expect": {k: asdict(v) for k, v in exp.items()}, "plan": plan}, f, indent=1)
    return 0 if all(check_against_hw1(exp).values()) else 1


if __name__ == "__main__":
    sys.exit(main())
