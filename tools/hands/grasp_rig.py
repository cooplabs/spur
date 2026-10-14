"""A one-hand grasp rig: where the Dex3-1 closes on an object, with the S1 collision primitives vs the mesh hulls.

Session 3 (DESIGN §13.1 A9: "decompose the cause in sim (aim geometry vs arm sag vs calibration)"; the hand
geometry part). The hand is fixed in space (palm frame = world frame), gravity off, the object free and light (its
weighed mass), centred on v2's OPEN-hand grasp centre (0.5 thumb tip + 0.25 (index tip + middle tip) of the open
preset, r1mc2 api.py:1724-1728 - the point v2 aims at the object). The hand closes from the open preset to a closed
preset under Dex3 position control (kp 1.5, kd 0.2: xr_teleoperate's gains, dex3.KP / KD; torque within the URDF
efforts) for `seconds`. Reported, in the palm frame (x along the fingers, y the palm normal on the grasp side, z
across the fingers; mm):

* every hand-object contact at the end (geom, position, normal force) and their force-weighted centroid;
* where the object ended up relative to where it was aimed (the free object is pushed / pulled to where the hand
  actually closes: + x = forward along the fingers);
* the first-contact closure (how far the fingers had closed when they first touched the object);
* the joint angles at the end.

Usage: uv run python tools/hands/grasp_rig.py [--side right] [--object cylinder60|bar_flat|box_tall] [--preset fist]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
from r1sim import dex3  # noqa: E402

FloatArray = npt.NDArray[np.float64]
MM = 1000.0

# Objects (sizes and weighed masses of three real test objects, 10 Oct): MJCF geom strings, centred.
OBJECTS: dict[str, tuple[str, float]] = {
    # a 6 cm round object (6.0 cm footprint, 6.2 cm tall), 29 g; axis across the fingers
    "cylinder60": ('<geom type="cylinder" size="0.030 0.031" friction="1.3 0.005 0.0008" condim="6"/>', 0.029),
    # a tall box 7.8 x 6.7 x 11.2 cm, 69 g: 6.7 cm between thumb and fingers, the 11.2 cm side across the fingers
    "box_tall": ('<geom type="box" size="0.039 0.0335 0.056" friction="1.0 0.004 0.0004"/>', 0.069),
    # a flat bar 4.8 x 12.4 x 3.1 cm, 10 g: 3.1 cm between thumb and fingers, the long side across the fingers
    "bar_flat": ('<geom type="box" size="0.024 0.0155 0.062" friction="0.6 0.003 0.0003"/>', 0.010),
}


def open_grasp_centre(side: str) -> FloatArray:
    """v2's aim point in the palm frame: 0.5 thumb tip + 0.25 (index tip + middle tip), open preset, tip sites."""
    g = -1.0 if side == "left" else 1.0
    poses = dex3.link_poses(side, dex3.preset("open", side))
    tips = {}
    for link, loc in (
        ("thumb_2", (0.0, g * dex3.TIP, 0.0)),
        ("middle_1", (dex3.TIP, 0, 0)),
        ("index_1", (dex3.TIP, 0, 0)),
    ):
        pos, quat = poses[link]
        v = np.zeros(3)
        mujoco.mju_rotVecQuat(v, np.asarray(loc, dtype=np.float64), quat)
        tips[link] = pos + v
    return np.asarray(0.5 * tips["thumb_2"] + 0.25 * (tips["index_1"] + tips["middle_1"]))


def rig_xml(side: str, collision: str, obj: str, obj_pos: FloatArray, fixed: bool = False) -> str:
    hand = dex3.hand_xml(side, collision=collision)
    assets = ""
    if collision == "mesh":
        assets = "".join(f'<mesh name="{a["name"]}" file="{a["file"]}"/>' for a in dex3.collision_mesh_assets(side))
    acts = "".join(
        f'<position name="{n}" joint="{j}" kp="{dex3.KP[k]}" kv="{dex3.KD[k]}" '
        f'forcerange="{-dex3.TAU_MAX[k]} {dex3.TAU_MAX[k]}"/>'
        for k, (n, j) in enumerate(zip(dex3.actuator_names(side), dex3.joint_names(side), strict=True))
    )
    geom, mass = OBJECTS[obj]
    excl = "".join(
        f'<exclude body1="{a}" body2="{b}"/>'
        for i, a in enumerate(dex3.link_names(side))
        for b in dex3.link_names(side)[i + 1 :]
    )
    return f"""<mujoco model="dex3_rig">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 0" cone="elliptic" impratio="10"/>
  <asset>{assets}</asset>
  <worldbody>
    <body name="mount">
      {hand.replace(f'pos="{dex3._fmt(dex3.MOUNT_POS)}"', 'pos="0 0 0"', 1)}
    </body>
    <body name="object" pos="{dex3._fmt(obj_pos.tolist())}">
      {"" if fixed else "<freejoint/>"}
      <inertial pos="0 0 0" mass="{mass}" diaginertia="1e-5 1e-5 1e-5"/>
      {geom.replace("<geom ", '<geom name="object" ')}
    </body>
  </worldbody>
  <contact>{excl}</contact>
  <actuator>{acts}</actuator>
</mujoco>"""


@dataclass
class RigResult:
    side: str
    collision: str
    obj: str
    preset: str
    fixed: bool
    aim_mm: list[float]
    object_end_mm: list[float]
    object_shift_mm: list[float]
    contacts: list[dict[str, Any]] = field(default_factory=list)
    centroid_mm: list[float] | None = None
    first_contact_t_s: float | None = None
    first_contact_closure: float | None = None
    q_end: list[float] = field(default_factory=list)
    touching: list[str] = field(default_factory=list)


def run(side: str = "right", collision: str = "primitives", obj: str = "cylinder60", preset: str = "fist",
        seconds: float = 1.5, ramp_s: float = 0.5, fixed: bool = False) -> RigResult:  # fmt: skip
    """fixed=True: the object is welded at the aim point (geometry only: where the fingers touch it and how far they
    close); fixed=False: it is free and light (no gravity, no table: the closing fingers rake it toward the palm)."""
    aim = open_grasp_centre(side)
    m = mujoco.MjModel.from_xml_string(rig_xml(side, collision, obj, aim, fixed))
    d = mujoco.MjData(m)
    q0, q1 = dex3.preset("open", side), dex3.preset(preset, side)
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in dex3.joint_names(side)]
    d.qpos[qadr] = q0
    mujoco.mj_forward(m, d)
    obj_b = m.body("object").id
    obj_g = m.geom("object").id
    first_t: float | None = None
    first_c: float | None = None
    n = int(seconds / m.opt.timestep)
    for i in range(n):
        a = min(1.0, (i * m.opt.timestep) / ramp_s)
        d.ctrl[:] = q0 + a * (q1 - q0)
        mujoco.mj_step(m, d)
        if first_t is None and any(obj_g in (c.geom1, c.geom2) for c in d.contact[: d.ncon]):
            first_t = d.time
            first_c = float(np.mean(dex3.closure(d.qpos[qadr], side)[1:]))
    mujoco.mj_forward(m, d)
    contacts = []
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if obj_g not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == obj_g else c.geom1
        mujoco.mj_contactForce(m, d, i, f6)
        contacts.append(
            {
                "geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other),
                "pos_mm": (np.asarray(c.pos) * MM).round(1).tolist(),
                "fn_n": round(float(f6[0]), 3),
            }
        )
    cen = None
    if contacts:
        w = np.array([max(cc["fn_n"], 1e-9) for cc in contacts])
        P = np.array([cc["pos_mm"] for cc in contacts])
        cen = (w @ P / w.sum()).round(1).tolist()
    end = d.xpos[obj_b] * MM
    return RigResult(
        side=side, collision=collision, obj=obj, preset=preset, fixed=fixed, aim_mm=(aim * MM).round(1).tolist(),
        object_end_mm=end.round(1).tolist(), object_shift_mm=(end - aim * MM).round(1).tolist(), contacts=contacts,
        centroid_mm=cen, first_contact_t_s=None if first_t is None else round(first_t, 3),
        first_contact_closure=None if first_c is None else round(first_c, 3),
        q_end=np.asarray(d.qpos[qadr]).round(3).tolist(),
        touching=sorted({str(cc["geom"]) for cc in contacts}),
    )  # fmt: skip


def closure_str(r: RigResult) -> str:
    c = dex3.closure(np.asarray(r.q_end), r.side)
    return f"{c[1]:.2f}, {c[2]:.2f}, {np.mean(c[3:]):.2f}"


def table(rows: list[RigResult]) -> str:
    L = ["| object | preset | object | collision | first touch (s, closure) | object shift x / y / z (mm) | contact "
         "centroid (mm, palm) | final closure (thumb_1, thumb_2, fingers) | touching |",
         "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in rows:
        L.append(
            f"| {r.obj} | {r.preset} | {'fixed' if r.fixed else 'free'} | {r.collision} | {r.first_contact_t_s}, "
            f"{r.first_contact_closure} | {r.object_shift_mm[0]} / {r.object_shift_mm[1]} / {r.object_shift_mm[2]} | "
            f"{r.centroid_mm} | {closure_str(r)} | "
            f"{', '.join(t.replace(r.side + '_', '') for t in r.touching)} |"
        )
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--side", default="right", choices=dex3.SIDES)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = []
    for fixed in (True, False):
        for obj, preset in (("cylinder60", "fist"), ("box_tall", "fist"), ("bar_flat", "pinch")):
            for coll in dex3.COLLISIONS:
                rows.append(run(a.side, coll, obj, preset, fixed=fixed))
    print(f"aim (v2's open grasp centre, palm frame, mm): {rows[0].aim_mm}")
    print(table(rows))
    if a.json:
        Path(a.json).write_text(json.dumps([r.__dict__ for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
