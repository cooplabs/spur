"""Generate the RT core's built-in arm model (DESIGN.md §3.4) from the simulator's compiled MJCF.

The real-time core needs two small models that must not drift from the simulator: the arms' (and head's) gravity
torque for the feed-forward (FREEDRIVE, HOLD without sag) and the fingertip forward kinematics for the surface guard.
Both are pure functions of the kinematic tree and the link inertials, so this tool exports exactly those numbers from
`r1sim.model.build_model("real", visual=False)` - the same model every simulator test runs (hands as built: the
articulated Dex3-1 palm + 7 finger links, ~0.697 kg per hand) - into a C++ data table, and the hand-written evaluator
(`rt/src/arm_model.cpp`) walks it. No general dynamics library enters the RT core.

    uv run python tools/gen_arm_model.py            # rewrite rt/src/arm_model_gen.cpp
    uv run python tools/gen_arm_model.py --check    # exit 1 if the committed file differs (CI: the test does this)

Run it after any change to the robot model's kinematics or inertials (r1sim.model / dex3 / the MJCF); the test
`tests/unit/test_arm_model.py::test_generated_file_is_current` fails until you do.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import mujoco
import numpy as np
from r1sim import dex3, joints
from r1sim import model as M

OUT = Path(__file__).resolve().parents[1] / "rt" / "src" / "arm_model_gen.cpp"
PROFILE = "real"

# The subtrees the RT core needs, by root body: the waist (to place the torso), both arms + hands, the head, both
# legs (to place the feet: v2's base frame hangs on them). The fixed parts of the trunk (the backpack 0.454 kg and the
# rear mass 1.6 kg of session 5's CoM fix, both welded to the torso) never move relative to it and load no arm / head
# joint (the gravity torques walk only the arm / head subtrees), but the whole-body CoM needs them: session 11 exports
# them too (the user kept the arm-aware STAND_FREEZE pose for HW2 - r1rt/stand_pose.hpp's ankle shift read ~10 % large
# without them). MuJoCo appends them after the head, so every other node index stays what it was.
ROOTS = ("waist_roll_link", "left_hip_pitch_link", "right_hip_pitch_link")
SITES = (
    "left_foot",
    "right_foot",
    "left_hand_palm",
    "left_hand_thumb_tip",
    "left_hand_middle_tip",
    "left_hand_index_tip",
    "right_hand_palm",
    "right_hand_thumb_tip",
    "right_hand_middle_tip",
    "right_hand_index_tip",
    "imu",
)
INPUTS = [f"{n}_joint" for n in joints.JOINTS] + [n for s in dex3.SIDES for n in dex3.joint_names(s)]


def _name(m: mujoco.MjModel, kind: mujoco.mjtObj, i: int) -> str:
    return str(mujoco.mj_id2name(m, kind, i))


def _f(x: float) -> str:
    return repr(float(x))  # shortest round-trip decimal: the C++ double is bit-identical to the model's


def _v(xs: np.ndarray) -> str:
    return "{" + ", ".join(_f(x) for x in np.asarray(xs, dtype=np.float64).ravel()) + "}"


def export(m: mujoco.MjModel) -> dict[str, object]:
    """The exported tree: nodes (bodies) in MuJoCo order (parents first), the sites, the input map."""
    pelvis = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    keep: list[int] = [pelvis]
    roots = {int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, r)) for r in ROOTS}
    torso = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"))
    for b in range(1, m.nbody):
        p = int(m.body_parentid[b])
        fixed_to_trunk = int(m.body_jntnum[b]) == 0 and p in (pelvis, torso)  # the backpack, the rear mass (above)
        if b in roots or (p in keep and p != pelvis) or fixed_to_trunk:
            keep.append(b)
    node_of = {b: i for i, b in enumerate(keep)}
    nodes = []
    for b in keep:
        nj = int(m.body_jntnum[b])
        if b == pelvis or nj == 0:  # the root / a link welded to its parent (the palms)
            inp, axis, jpos, qref = -1, np.zeros(3), np.zeros(3), 0.0
        else:
            if nj != 1:
                raise SystemExit(f"{_name(m, mujoco.mjtObj.mjOBJ_BODY, b)}: {nj} joints (the evaluator handles 0 / 1)")
            j = int(m.body_jntadr[b])
            if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise SystemExit(f"{_name(m, mujoco.mjtObj.mjOBJ_JOINT, j)}: not a hinge")
            inp = INPUTS.index(_name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
            axis, jpos = m.jnt_axis[j].copy(), m.jnt_pos[j].copy()
            qref = float(m.qpos0[m.jnt_qposadr[j]])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.body_quat[b])
        nodes.append(
            dict(
                name=_name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                parent=-1 if b == pelvis else node_of[int(m.body_parentid[b])],
                pos=np.zeros(3) if b == pelvis else m.body_pos[b].copy(),
                rot=np.eye(3).ravel() if b == pelvis else R,
                input=inp,
                axis=axis,
                jpos=jpos,
                qref=qref,
                mass=float(m.body_mass[b]),
                ipos=m.body_ipos[b].copy(),
            )
        )
    for i in range(len(nodes)):  # MuJoCo orders bodies depth first: a subtree is a contiguous run of nodes
        end = i + 1
        while end < len(nodes) and _descends(nodes, end, i):
            end += 1
        nodes[i]["end"] = end
    sites = []
    for s in SITES:
        sid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.site_quat[sid])
        sites.append(dict(name=s, node=node_of[int(m.site_bodyid[sid])], pos=m.site_pos[sid].copy(), rot=R))
    return dict(nodes=nodes, sites=sites, gravity=m.opt.gravity.copy())


def _descends(nodes: list[dict[str, object]], k: int, i: int) -> bool:
    while k != -1:
        if k == i:
            return True
        k = int(nodes[k]["parent"])  # type: ignore[call-overload]
    return False


def _digest(tree: dict[str, object]) -> str:
    h = hashlib.sha256()
    for n in tree["nodes"]:  # type: ignore[attr-defined]
        for k in ("pos", "rot", "axis", "jpos", "ipos"):
            h.update(np.asarray(n[k], dtype="<f8").tobytes())
        h.update(np.asarray([n["parent"], n["end"], n["input"]], dtype="<i8").tobytes())
        h.update(np.asarray([n["mass"], n["qref"]], dtype="<f8").tobytes())
    for s in tree["sites"]:  # type: ignore[attr-defined]
        h.update(np.asarray(s["pos"], dtype="<f8").tobytes())
        h.update(np.asarray(s["rot"], dtype="<f8").tobytes())
    return h.hexdigest()


def _iv(xs: list[int]) -> str:
    return "{" + ", ".join(str(int(x)) for x in xs) + "}"


def render(tree: dict[str, object]) -> str:
    nodes = tree["nodes"]
    sites = tree["sites"]
    assert isinstance(nodes, list) and isinstance(sites, list)
    c = constants(tree)
    lines = [
        '// GENERATED by tools/gen_arm_model.py from r1sim.model.build_model("real", visual=False) - DO NOT EDIT.',
        "// Regenerate after any change to the robot model's kinematics / inertials:",
        "//   uv run python tools/gen_arm_model.py",
        "// (tests/unit/test_arm_model.py::test_generated_file_is_current fails until you do).",
        f"// Model digest (sha256 of the exported numbers): {_digest(tree)}",
        '#include "r1rt/arm_model.hpp"',
        "",
        "namespace r1rt::arm {",
        "",
        f'const char kModelDigest[] = "{_digest(tree)}";',
        f"const double kGravityWorld[3] = {_v(np.asarray(tree['gravity']))};",
        "",
        f"const int kNumNodes = {len(nodes)};",
        f'static_assert({len(nodes)} <= kMaxNodes, "raise kMaxNodes");',
        f'static_assert({len(sites)} == kNumSiteDefs, "the site order is fixed by the header");',
        f"const int kTorsoNode = {c['kTorsoNode']};",
        f"const int kArmJointNode[2][5] = {{{_iv(c['arm_left'])}, {_iv(c['arm_right'])}}};",
        f"const int kHeadJointNode[2] = {_iv(c['head'])};",
        '// the open hand (dex3.preset("open", side), Dex3 DDS order): the hand pose used when no hand q is given',
        f"const double kHandOpen[2][7] = {{{_v(dex3.preset('open', 'left'))}, {_v(dex3.preset('open', 'right'))}}};",
        "",
        "// parent, end (one past the subtree), pos (in parent), rot (row-major, in parent), input (q index or -1),",
        "// axis, jpos, qref, mass, ipos",
        "const Node kNodes[kMaxNodes] = {",
    ]
    for i, n in enumerate(nodes):
        lines.append(
            f"    /* {i:2d} {n['name']} */ {{{n['parent']}, {n['end']}, {_v(n['pos'])}, {_v(n['rot'])}, "
            f"{n['input']}, {_v(n['axis'])}, {_v(n['jpos'])}, {_f(n['qref'])}, {_f(n['mass'])}, {_v(n['ipos'])}}},"
        )
    lines += [
        "};",
        "",
        "// node, pos (in the node's body frame), rot (row-major)",
        "const SiteDef kSites[kNumSiteDefs] = {",
    ]
    for i, s in enumerate(sites):
        lines.append(f"    /* {i:2d} {s['name']} */ {{{s['node']}, {_v(s['pos'])}, {_v(s['rot'])}}},")
    lines += ["};", "", "}  // namespace r1rt::arm", ""]
    return "\n".join(lines)


def constants(tree: dict[str, object]) -> dict[str, object]:
    """The node / site indices the hand-written header hard-codes (checked by the test against this export)."""
    nodes = tree["nodes"]
    sites = tree["sites"]
    assert isinstance(nodes, list) and isinstance(sites, list)
    by_input = {int(n["input"]): i for i, n in enumerate(nodes) if int(n["input"]) >= 0}
    out: dict[str, object] = {"kNumNodes": len(nodes), "kNumSites": len(sites)}
    out["kTorsoNode"] = next(i for i, n in enumerate(nodes) if n["name"] == "torso_link")
    out["arm_left"] = [by_input[k] for k in range(14, 19)]
    out["arm_right"] = [by_input[k] for k in range(19, 24)]
    out["head"] = [by_input[24], by_input[25]]
    return out


def generate() -> str:
    return render(export(M.build_model(PROFILE, visual=False)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed file differs")
    a = ap.parse_args(argv)
    text = generate()
    if a.check:
        cur = OUT.read_text() if OUT.exists() else ""
        if cur != text:
            print(f"{OUT} is stale: run `uv run python tools/gen_arm_model.py`", file=sys.stderr)
            return 1
        print(f"{OUT}: current")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
