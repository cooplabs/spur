"""The one geometry engine (DESIGN.md §7.1): MuJoCo collision and signed distance on the robot's MJCF + the world.

v2 had ten geometry implementations (INV part B §5); v3 answers every query - "does this arm pose touch anything?",
"how far is the hand from the table?", "where is the pinch point?" - from ONE compiled model: the same robot MJCF the
simulator runs (r1sim.model, the "real" profile, collision geoms only) with the snapshot's entities added as static
geoms. The robot is posed so that its base frame (arm_model.hpp "base": the feet sites' midpoint on the lower sole,
the feet's mean yaw, z up) IS the model's world frame: entity poses in `base` go in unchanged.

Contacts that matter for an arm: any geom of that arm's subtree (shoulder to fingertips) against (a) the torso, legs,
head (self), (b) the other arm (other_arm), (c) a world entity (world) - a world entity counts when nearer than the
clearance (its geoms carry the clearance as MuJoCo's contact margin, so `contact.dist` is the signed distance).
Pairs inside one arm (the fingers vs the forearm) are not checked: the hand's own shapes are the hand driver's job.
Cost on the Mac: mj_kinematics ~1.4 us + mj_collision ~3.2 us for the 65 collision geoms (session 9 probe).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt
from r1sim import dex3
from r1sim import joints as J
from r1sim import model as M

from .entities import Container, Entity, Object, Obstacle, Surface, WorldSnapshot

FloatArray = npt.NDArray[np.float64]

ARM_JOINTS: dict[str, npt.NDArray[np.int64]] = {"left": J.LEFT_ARM, "right": J.RIGHT_ARM}
SIDES = ("left", "right")
POINTS = (
    "pinch",
    "palm",
    "fingertips",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "wrist",
)  # + "grasp:<grip>[@<width m>]"
TIPS = ("thumb_tip", "index_tip", "middle_tip")
DEFAULT_CLEARANCE_M = 0.01  # world entities: an arm closer than this counts as touching (v2's planner used -4 mm
#                             at the table, a TOLERATED penetration; v3 keeps a positive clearance and lets a goal
#                             allow the touches it means, e.g. the fingertips on the surface of a guarded descent)
BOX = int(mujoco.mjtGeom.mjGEOM_BOX)  # distances(): MuJoCo 3.5's box-box distance quirks
HAND_SUPPORT_CLEARANCE_M = 0.002  # the HAND near a support (a surface's top, a container's floor): it works there -
#                                   v2 planned with -4 mm at the table (a tolerated penetration) and made 1.5 cm moves
#                                   with the middle tip 8 mm above the board; 1 cm stopped those (compat, 11 Oct).
#                                   The arm, the robot and every obstacle / object / wall keep DEFAULT_CLEARANCE_M.
SELF_MARGIN_M = 0.0  # robot vs robot: a contact = penetration (the collision shapes are already padded)
GEOM_PREFIX = "w:"


@dataclass(frozen=True)
class Held:
    """An object held by `side`'s hand, in its palm frame (session 10: the planner checks it like the hand - the
    world clearance, the hand's support rule, the body / other arm - so a carry keeps it off a container's rim)."""

    side: str
    shape: str  # box | cylinder | sphere (MuJoCo's geom sizes: half-sizes, radius / half-height)
    size: tuple[float, ...]
    pos: tuple[float, float, float]  # its centre in the palm body's frame (m)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Hit:
    """One contact that blocks: the arm's body, what it touches (`entity:<id>`, `robot:<body>`), the signed distance
    (m, < 0 = penetration) and the kind (self / other_arm / world)."""

    arm_body: str
    other: str
    dist: float
    kind: str


def _add_entity(spec: mujoco.MjSpec, e: Entity, clearance: float) -> None:
    R = e.pose.rot()
    pos = np.asarray(e.pose.pos, dtype=np.float64)
    quat = list(e.pose.quat)

    def box(name: str, centre_local: tuple[float, float, float], half: tuple[float, float, float]) -> None:
        g = spec.worldbody.add_geom()
        g.name = f"{GEOM_PREFIX}{e.id}:{name}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = list(half)
        g.pos = list(pos + R @ np.asarray(centre_local))
        g.quat = quat
        g.margin = clearance
        g.contype = 1
        g.conaffinity = 1
        g.group = 4

    if isinstance(e, Surface):
        sx, sy, t = e.size
        box("top", (0.0, 0.0, -t / 2), (sx / 2, sy / 2, t / 2))
    elif isinstance(e, Container):
        sx, sy, sz = e.size_outer
        w, f = e.wall_m, e.floor_m
        box("floor", (0.0, 0.0, f / 2), (sx / 2, sy / 2, f / 2))
        box("wall_xp", (sx / 2 - w / 2, 0.0, sz / 2), (w / 2, sy / 2, sz / 2))
        box("wall_xm", (-sx / 2 + w / 2, 0.0, sz / 2), (w / 2, sy / 2, sz / 2))
        box("wall_yp", (0.0, sy / 2 - w / 2, sz / 2), (sx / 2 - w, w / 2, sz / 2))
        box("wall_ym", (0.0, -sy / 2 + w / 2, sz / 2), (sx / 2 - w, w / 2, sz / 2))
    elif isinstance(e, Obstacle | Object):
        g = spec.worldbody.add_geom()
        g.name = f"{GEOM_PREFIX}{e.id}:shape"
        g.type = {
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        }[e.shape]
        size = list(e.size) + [0.0] * (3 - len(e.size))
        g.size = size
        g.pos = list(pos)
        g.quat = quat
        g.margin = clearance
        g.contype = 1
        g.conaffinity = 1
        g.group = 4


def _subtree(m: mujoco.MjModel, root: int) -> set[int]:
    out = {root}
    for b in range(root + 1, m.nbody):
        if int(m.body_parentid[b]) in out:
            out.add(b)
    return out


def _rp_quat(quat: FloatArray) -> FloatArray:
    """The IMU's roll / pitch only (yaw removed, v2 controller.rp_matrix): the pelvis orientation for the base frame."""
    w, x, y, z = (float(v) for v in quat)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(-yaw / 2), np.sin(-yaw / 2)  # q_yaw(-yaw) * q
    q = np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])
    return np.asarray(q / np.linalg.norm(q), dtype=np.float64)


def _mat_to_quat(R: FloatArray) -> FloatArray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return q


class Geometry:
    """The robot posed at a stance + the snapshot's entities in one MuJoCo model; queries for one arm at a time.

    `set_stance(q26, imu_quat, hands)` once per plan (legs, waist, head, the other arm, the hands as measured), then
    `set_arm(side, q5)` + `hits(side)` / `point(side, name)` per candidate. Not thread-safe (one per worker)."""

    def __init__(
        self,
        snapshot: WorldSnapshot | None = None,
        *,
        profile: str = "real",
        clearance_m: float = DEFAULT_CLEARANCE_M,
        held: tuple[Held, ...] = (),
    ) -> None:
        spec = M.robot_spec(profile, visual=False)
        self.clearance_m = float(clearance_m)
        self.entities: tuple[Entity, ...] = snapshot.entities if snapshot is not None else ()
        for e in self.entities:
            _add_entity(spec, e, self.clearance_m)
        self.held = tuple(held)
        for h in self.held:  # a held object travels with the hand: a collision geom on the palm (session 10)
            body = spec.body(f"{h.side}_hand_palm_link")
            g = body.add_geom()
            g.name = f"held_{h.side}"
            g.type = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                      "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}[h.shape]  # fmt: skip
            g.size = [*h.size, *([0.0] * (3 - len(h.size)))]
            g.pos = list(h.pos)
            g.quat = list(h.quat)
            g.contype, g.conaffinity, g.group = 1, 1, 3
            g.mass = 0.0
            g.density = 0.0
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.ix = M.RobotIndex(m)
        self.geom_entity: dict[int, str] = {}
        for g in range(m.ngeom):
            n = m.geom(g).name
            if n.startswith(GEOM_PREFIX):
                self.geom_entity[g] = n[len(GEOM_PREFIX) :].split(":")[0]
        self.support_geoms: set[int] = {
            g for g in self.geom_entity if m.geom(g).name.endswith((":top", ":floor"))
        }  # Surface tops + Container floors
        self.entity_geoms: dict[str, list[int]] = {}
        for g, eid in self.geom_entity.items():
            self.entity_geoms.setdefault(eid, []).append(g)
        self.arm_bodies: dict[str, set[int]] = {}
        self.hand_bodies: dict[str, set[int]] = {}
        for s in SIDES:
            root = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_shoulder_pitch_link"))
            self.arm_bodies[s] = _subtree(m, root)
            palm = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hand_palm_link"))
            self.hand_bodies[s] = _subtree(m, palm) if palm >= 0 else set()
        self.site: dict[str, dict[str, int]] = {
            s: {k: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_hand_{k}")) for k in ("palm", *TIPS)}
            for s in SIDES
        }
        self.palm_body = {s: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hand_palm_link")) for s in SIDES}
        self.wrist_body = {
            s: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_roll_link")) for s in SIDES
        }
        self._body_name = [m.body(b).name for b in range(m.nbody)]
        body_of = np.asarray(m.geom_bodyid, dtype=np.int64)
        self._geom_arm = {s: np.isin(body_of, sorted(self.arm_bodies[s])) for s in SIDES}
        self._arm_geoms = {
            s: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.arm_bodies[s] and m.geom_contype[g]]
            for s in SIDES
        }
        self.arm_qadr = {s: self.ix.qadr[ARM_JOINTS[s]] for s in SIDES}
        self.arm_vadr = {s: self.ix.vadr[ARM_JOINTS[s]] for s in SIDES}
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))
        self.q26 = J.POSE_STAND.copy()
        self.hands = {s: dex3.preset("open", s) for s in SIDES}
        self._grasp_cache: dict[tuple[str, str, float | None], FloatArray] = {}
        self._closing_q_cache: dict[tuple[str, str, float | None], FloatArray] = {}  # session 13
        self.set_stance(J.POSE_STAND)

    # ------------------------------------------------------------------ pose
    def set_stance(
        self,
        q26: FloatArray,
        imu_quat: FloatArray | None = None,
        hands: Mapping[str, FloatArray] | None = None,
        *,
        imu_mount: bool = True,
    ) -> None:
        """Pose the whole robot (the measured joints, the pelvis's roll / pitch from the IMU, the hands) and move it
        so its base frame is the model's world frame.

        imu_quat = what the IMU reports (w, x, y, z; its yaw is ignored). The IMU sits in the pelvis with a mounting
        rotation (r1sim.model.IMU_MOUNT_OFFSET_DEG, nominal 0.50 roll / 0.96 deg pitch; the model's imu site): the
        pelvis = the IMU's orientation x that rotation's inverse (arm_model.hpp undoes it the same way). v2 used the
        IMU's roll / pitch AS the pelvis's (ArmKin.sync) - A9: its belief of where the open hand was came out 1-2.5 cm
        off along the approach (S15); `imu_mount=False` reproduces v2 for that comparison."""
        m, d, ix = self.m, self.d, self.ix
        self.q26 = np.asarray(q26, dtype=np.float64).copy()
        # session 12 (fixes): a side without a given hand pose is posed OPEN (dex3's "open" preset = a fresh
        # Geometry's), never left as an EARLIER call set it - the planner caches geometries (core.geometry_for), so on
        # r1d's long-lived planner worker a plan made without that hand's state (r1d passes none when the hand state is
        # not valid: the hand driver starting, a stale hand) used the previous plan's fingers (an 11 cm different reach)
        for s in SIDES:
            h = None if hands is None else hands.get(s)
            self.hands[s] = dex3.preset("open", s) if h is None else np.asarray(h, dtype=np.float64).copy()
        d.qpos[:] = m.qpos0
        b = ix.base_qadr
        d.qpos[b : b + 3] = (0.0, 0.0, 0.8)
        qi = np.array([1.0, 0, 0, 0]) if imu_quat is None else np.asarray(imu_quat, dtype=np.float64)
        if imu_mount and imu_quat is not None:
            qs = np.asarray(m.site_quat[ix.imu_site], dtype=np.float64)  # the site in the pelvis
            q_pel = np.zeros(4)
            mujoco.mju_mulQuat(q_pel, qi, np.array([qs[0], -qs[1], -qs[2], -qs[3]]))
            qi = q_pel
        d.qpos[b + 3 : b + 7] = _rp_quat(qi)
        for i in range(J.N):
            if ix.present[i]:
                d.qpos[ix.qadr[i]] = self.q26[i]
        for s, adr in ix.hand_qadr.items():
            d.qpos[adr] = self.hands[s]
        mujoco.mj_kinematics(m, d)
        pl, pr = d.site_xpos[ix.foot_sites[0]].copy(), d.site_xpos[ix.foot_sites[1]].copy()

        def yaw_of(R: FloatArray) -> float:
            return float(np.arctan2(R[1, 0], R[0, 0]))

        yaw = 0.5 * (
            yaw_of(d.site_xmat[ix.foot_sites[0]].reshape(3, 3)) + yaw_of(d.site_xmat[ix.foot_sites[1]].reshape(3, 3))
        )
        o = np.array([(pl[0] + pr[0]) / 2, (pl[1] + pr[1]) / 2, min(pl[2], pr[2])])
        c, s_ = np.cos(yaw), np.sin(yaw)
        Rb = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        p_pel = d.qpos[b : b + 3].copy()
        R_pel = np.zeros(9)
        mujoco.mju_quat2Mat(R_pel, d.qpos[b + 3 : b + 7])
        d.qpos[b : b + 3] = Rb.T @ (p_pel - o)
        d.qpos[b + 3 : b + 7] = _mat_to_quat(Rb.T @ R_pel.reshape(3, 3))
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    def set_arm(self, side: str, q5: FloatArray) -> None:
        self.d.qpos[self.arm_qadr[side]] = q5
        mujoco.mj_kinematics(self.m, self.d)

    def set_hand(self, side: str, hand_q: FloatArray) -> None:
        self.hands[side] = np.asarray(hand_q, dtype=np.float64).copy()
        self.d.qpos[self.ix.hand_qadr[side]] = self.hands[side]
        mujoco.mj_kinematics(self.m, self.d)

    def arm_q(self, side: str) -> FloatArray:
        return np.asarray(self.d.qpos[self.arm_qadr[side]], dtype=np.float64).copy()

    # ------------------------------------------------------------------ points
    def point(self, side: str, name: str = "pinch") -> FloatArray:
        """A hand point in the base frame (pinch = between the thumb and index tips, v2 arms.py:340)."""
        d, S = self.d, self.site[side]
        if name.startswith("grasp:"):
            grip, _, w = name[6:].partition("@")
            off = self.grasp_offset(side, grip, float(w) if w else None)
            pb = self.palm_body[side]
            return np.asarray(d.xmat[pb].reshape(3, 3) @ off + d.xpos[pb], dtype=np.float64)
        if name == "pinch":
            return np.asarray(0.5 * (d.site_xpos[S["thumb_tip"]] + d.site_xpos[S["index_tip"]]), dtype=np.float64)
        if name == "fingertips":
            return np.asarray(sum(d.site_xpos[S[k]] for k in TIPS) / 3.0, dtype=np.float64)
        if name == "grasp_centre":  # v2 api._grasp_centre = the RT core's arm model (arm_model.hpp grasp_centre)
            c = 0.5 * d.site_xpos[S["thumb_tip"]] + 0.25 * (d.site_xpos[S["index_tip"]] + d.site_xpos[S["middle_tip"]])
            return np.asarray(c, dtype=np.float64)
        if name == "wrist":
            return np.asarray(d.xpos[self.wrist_body[side]], dtype=np.float64).copy()
        if name not in S:
            raise KeyError(f"unknown hand point {name!r} (known: {', '.join(POINTS)})")
        return np.asarray(d.site_xpos[S[name]], dtype=np.float64).copy()

    def grasp_offset(self, side: str, grip: str, width_m: float | None = None) -> FloatArray:
        """Where `grip`'s fingers close, in the PALM frame (HG-2, DESIGN §7.2 "grasp aim": the aim point is where the
        chosen grip's fingers really close, not the open hand's midpoint - INV part B §5.2). From the hand model:
        the hand is swept from the open preset to the grip's closed preset (r1sim.dex3 GRASPS: wrap -> fist, pinch /
        tip -> pinch_index) and the closing point is the midpoint of the thumb tip and the opposing finger tip(s)
        (wrap: index + middle) where their gap equals the object's width (no width: fully closed). Cached."""
        key = (side, grip, None if width_m is None else round(float(width_m), 4))
        if key in self._grasp_cache:
            return self._grasp_cache[key]
        closed = {"wrap": "fist", "power": "fist", "pinch": "pinch_index", "tip": "pinch_index"}.get(grip)
        if closed is None:
            raise KeyError(f"unknown grip {grip!r} (wrap | pinch | tip)")
        fingers = ("index_tip", "middle_tip") if closed == "fist" else ("index_tip",)
        h_open, h_closed = dex3.preset("open", side), dex3.preset(closed, side)
        h_save = self.hands[side].copy()
        d = self.d
        pb = self.palm_body[side]
        best: FloatArray | None = None
        prev: tuple[float, FloatArray] | None = None
        # session 12 (fixes; the S11 CI leak): computed at ONE canonical stance (the model's qpos0), not the caller's.
        # The palm-frame result is a constant of the hand model, but computed at the current stance its floats varied
        # ~1e-16 m with that stance - and cached here, inside a geometry the planner caches per world snapshot
        # (core.geometry_for), plans depended on which earlier goal first asked (the closed-loop simulation grew that
        # into another episode: the whole-episode exact replay's 130 / 135 after other tests declared the same world)
        q_save = d.qpos.copy()
        d.qpos[:] = self.m.qpos0
        try:
            for s in np.linspace(0.0, 1.0, 101):
                self.set_hand(side, h_open + s * (h_closed - h_open))
                th = d.site_xpos[self.site[side]["thumb_tip"]]
                fi = sum(d.site_xpos[self.site[side][k]] for k in fingers) / len(fingers)
                mid = 0.5 * (th + fi)
                R = d.xmat[pb].reshape(3, 3)
                local = R.T @ (mid - d.xpos[pb])
                gap = float(np.linalg.norm(th - fi))
                if width_m is not None and gap <= width_m:
                    if prev is None:
                        best = local
                    else:  # interpolate between the last two samples to the exact width
                        g0, l0 = prev
                        f = (g0 - width_m) / max(g0 - gap, 1e-9)
                        best = l0 + f * (local - l0)
                    break
                prev = (gap, local)
                best = local
        finally:
            d.qpos[:] = q_save
            self.set_hand(side, h_save)  # (restores the hand + runs the kinematics of the restored stance)
        assert best is not None
        self._grasp_cache[key] = np.asarray(best, dtype=np.float64)
        return self._grasp_cache[key]

    def closing_hand_q(self, side: str, grip: str, width_m: float | None) -> FloatArray:
        """Session 13: the hand's joint angles where `grip`'s sweep (open preset -> the grip's closed preset, as
        grasp_offset) has the thumb tip and the opposing tip(s) `width_m` apart = the moment the fingers meet an object
        that wide (no width: fully closed). Palm-frame geometry only: computed at the model's qpos0. Cached."""
        key = (side, grip, None if width_m is None else round(float(width_m), 4))
        if key in self._closing_q_cache:
            return self._closing_q_cache[key]
        closed = {"wrap": "fist", "power": "fist", "pinch": "pinch_index", "tip": "pinch_index"}.get(grip)
        if closed is None:
            raise KeyError(f"unknown grip {grip!r} (wrap | pinch | tip)")
        fingers = ("index_tip", "middle_tip") if closed == "fist" else ("index_tip",)
        h_open, h_closed = dex3.preset("open", side), dex3.preset(closed, side)
        h_save = self.hands[side].copy()
        d = self.d
        q_save = d.qpos.copy()
        d.qpos[:] = self.m.qpos0
        best = h_closed
        try:
            for s in np.linspace(0.0, 1.0, 101):
                hq = h_open + s * (h_closed - h_open)
                self.set_hand(side, hq)
                th = d.site_xpos[self.site[side]["thumb_tip"]]
                fi = sum(d.site_xpos[self.site[side][k]] for k in fingers) / len(fingers)
                if width_m is not None and float(np.linalg.norm(th - fi)) <= width_m:
                    best = hq
                    break
        finally:
            d.qpos[:] = q_save
            self.set_hand(side, h_save)
        self._closing_q_cache[key] = np.asarray(best, dtype=np.float64)
        return self._closing_q_cache[key]

    def point_jac(self, side: str, name: str = "pinch") -> FloatArray:
        """d point / d q5 (3 x 5); needs mj_comPos (set_arm keeps the stance's - exact for these hinge chains? no:
        recomputed here)."""
        m, d, S = self.m, self.d, self.site[side]
        mujoco.mj_comPos(m, d)
        if name.startswith("grasp:"):  # a point fixed in the palm body
            p = self.point(side, name)
            mujoco.mj_jac(m, d, self._jp, self._jr, p, self.palm_body[side])
            return np.asarray(self._jp[:, self.arm_vadr[side]], dtype=np.float64).copy()
        if name == "grasp_centre":
            J3 = np.zeros((3, m.nv))
            for n, w in (("thumb_tip", 0.5), ("index_tip", 0.25), ("middle_tip", 0.25)):
                mujoco.mj_jacSite(m, d, self._jp, self._jr, S[n])
                J3 += w * self._jp
            return np.asarray(J3[:, self.arm_vadr[side]], dtype=np.float64)
        names = {"pinch": ["thumb_tip", "index_tip"], "fingertips": list(TIPS)}.get(name, [name])
        J3 = np.zeros((3, m.nv))
        for n in names:
            if n == "wrist":
                mujoco.mj_jacBody(m, d, self._jp, self._jr, self.wrist_body[side])
            else:
                mujoco.mj_jacSite(m, d, self._jp, self._jr, S[n])
            J3 += self._jp
        return np.asarray(J3[:, self.arm_vadr[side]] / len(names), dtype=np.float64)

    def hand_axes(self, side: str) -> tuple[FloatArray, FloatArray]:
        """(approach = the palm's x axis: along the fingers; palm normal: out of the palm) in the base frame."""
        R = self.d.xmat[self.palm_body[side]].reshape(3, 3)
        g = -1.0 if side == "left" else 1.0
        return np.asarray(R[:, 0], dtype=np.float64).copy(), np.asarray(g * R[:, 1], dtype=np.float64).copy()

    def palm_rot_jac(self, side: str) -> FloatArray:
        """d(rotation vector of the palm) / d q5 (3 x 5)."""
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacBody(self.m, self.d, self._jp, self._jr, self.palm_body[side])
        return np.asarray(self._jr[:, self.arm_vadr[side]], dtype=np.float64).copy()

    def lowest(self, side: str, bodies: Iterable[int] | None = None) -> float:
        """The lowest point (z, base frame) of the hand's collision geoms (sphere / capsule / box bounding)."""
        m, d = self.m, self.d
        bs = set(self.hand_bodies[side] if bodies is None else bodies)
        lo = np.inf
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in bs or m.geom_contype[g] == 0:
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            sz = m.geom_size[g]
            t = int(m.geom_type[g])
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                ext = sz[0]
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ext = abs(R[2, 2]) * sz[1] + sz[0]
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                ext = float(np.abs(R[2, :]) @ sz)
            else:
                ext = float(m.geom_rbound[g])
            lo = min(lo, float(d.geom_xpos[g][2] - ext))
        return float(lo)

    # ------------------------------------------------------------------ collisions
    def hits(
        self,
        side: str,
        *,
        allow: Iterable[str] = (),
        allow_hand: Iterable[str] = (),
        stop_at_first: bool = False,
        hand_support_m: float = HAND_SUPPORT_CLEARANCE_M,
        hand_clear: dict[str, float] | None = None,
    ) -> list[Hit]:
        """Blocking contacts of `side`'s arm at the current pose. allow: entity ids the WHOLE arm may touch;
        allow_hand: entity ids only the hand may touch (a target object, the surface of a guarded descent).
        hand_support_m: how close the HAND may work above a support (a surface top, a container floor) - 2 mm by
        default; a path that must keep its full margin from a table (the out-and-around way, session 10) passes the
        geometry's own clearance (or more) so the hand gets no exception.
        hand_clear (session 12): entity id -> the distance the HAND must keep from it (below the geometry's own
        clearance), checked before allow / allow_hand - the OPEN hand around a grasp target at the grasp pose must not
        be inside it (the target is otherwise the hand's to touch)."""
        m, d = self.m, self.d
        mujoco.mj_collision(m, d)
        if d.ncon == 0:
            return []
        con = d.contact
        g1, g2, dist = con.geom1, con.geom2, con.dist
        in1, in2 = self._geom_arm[side][g1], self._geom_arm[side][g2]
        sel = np.nonzero(in1 != in2)[0]  # one geom on this arm, the other not (inside one arm / elsewhere: skip)
        if sel.size == 0:
            return []
        allow_s, allow_h = set(allow), set(allow_hand)
        hand = self.hand_bodies[side]
        other = self.arm_bodies["right" if side == "left" else "left"]
        out: list[Hit] = []
        for i in sel:
            ga, gb = (int(g1[i]), int(g2[i])) if in1[i] else (int(g2[i]), int(g1[i]))
            ba, bb = int(m.geom_bodyid[ga]), int(m.geom_bodyid[gb])
            eid = self.geom_entity.get(gb)
            dd = float(dist[i])
            if eid is not None:
                if hand_clear and eid in hand_clear and ba in hand:
                    if dd >= hand_clear[eid]:
                        continue
                    out.append(Hit(self._body_name[ba], f"entity:{eid}", dd, "world"))
                    if stop_at_first:
                        break
                    continue
                if eid in allow_s or (eid in allow_h and ba in hand):
                    continue
                if gb in self.support_geoms and ba in hand and dd >= hand_support_m:
                    continue  # the hand working just above a table / a container's floor
                kind, what = "world", f"entity:{eid}"
            else:
                if dd > -SELF_MARGIN_M and dd > 0.0:
                    continue
                if bb == 0:  # the floor plane (world body, not an entity)
                    kind, what = "world", "floor"
                else:
                    kind = "other_arm" if bb in other else "self"
                    what = f"robot:{self._body_name[bb]}"
            out.append(Hit(self._body_name[ba], what, dd, kind))
            if stop_at_first:
                break
        return out

    def clear(self, side: str, q5: FloatArray, **kw: object) -> bool:
        self.set_arm(side, q5)
        return not self.hits(side, stop_at_first=True, **kw)  # type: ignore[arg-type]

    def distance(self, side: str, eid: str, distmax: float = 0.5) -> float:
        """Signed distance (m) from `side`'s arm to entity `eid` at the current pose (distmax if farther)."""
        return self.distances(side, distmax, only=eid).get(eid, float(distmax))

    def distances(self, side: str, distmax: float = 0.5, only: str | None = None) -> dict[str, float]:
        """{entity id: signed distance from `side`'s arm} for the entities nearer than distmax. mj_geomDistance per
        geom pair; MuJoCo gives no distance for a box-box pair (0 and an empty fromto) - those (the hand's pads) use
        a conservative bound instead: the pad's bounding sphere to the entity box (at most ~1 cm pessimistic).
        Session 11: MuJoCo 3.5's box-box pair also answers an exact 0 WITH a fromto (random left-arm poses at a table:
        35 of 6000 box-box evaluations, the true distance 2-47 cm; e.g. the palm box 0.3 m from the table box) - so an
        exact 0 of a box-box pair takes the same bound (only the reported clearance read it: the contact checks are
        mj_collision's, not this)."""
        m, d = self.m, self.d
        ft = np.zeros(6)
        out: dict[str, float] = {}
        for eid, gbs in self.entity_geoms.items():
            if only is not None and eid != only:
                continue
            best = float(distmax)
            for gb in gbs:
                for ga in self._arm_geoms[side]:
                    ft[:] = 0.0
                    v = float(mujoco.mj_geomDistance(m, d, ga, gb, best, ft))
                    if v == 0.0 and (not np.any(ft) or (m.geom_type[ga] == BOX and m.geom_type[gb] == BOX)):
                        v = min(self._sphere_to_box(ga, gb), best)
                    best = min(best, v)
            if best < distmax:
                out[eid] = best
        return out

    def _sphere_to_box(self, ga: int, gb: int) -> float:
        m, d = self.m, self.d
        R = d.geom_xmat[gb].reshape(3, 3)
        local = R.T @ (d.geom_xpos[ga] - d.geom_xpos[gb])
        out = np.maximum(np.abs(local) - m.geom_size[gb], 0.0)
        return float(np.linalg.norm(out) - m.geom_rbound[ga])

    def surfaces_under(self, xy: FloatArray) -> list[tuple[str, float]]:
        """The surfaces / container floors / object tops under a point (id, top z in base), highest first."""
        out: list[tuple[str, float]] = []
        for e in self.entities:
            R = e.pose.rot()
            local = R.T @ (np.array([xy[0], xy[1], e.pose.pos[2]]) - np.asarray(e.pose.pos))
            if isinstance(e, Surface):
                if abs(local[0]) <= e.size[0] / 2 and abs(local[1]) <= e.size[1] / 2:
                    out.append((e.id, float(e.pose.pos[2])))
            elif isinstance(e, Container):
                if abs(local[0]) <= e.size_outer[0] / 2 and abs(local[1]) <= e.size_outer[1] / 2:
                    out.append((e.id, float(e.pose.pos[2] + e.floor_m)))
            elif isinstance(e, Object | Obstacle):
                if e.shape == "box":
                    inside = abs(local[0]) <= e.size[0] and abs(local[1]) <= e.size[1]
                else:
                    inside = float(np.hypot(local[0], local[1])) <= e.size[0]
                if inside:
                    top = e.pose.pos[2] + (e.size[0] if e.shape == "sphere" else e.size[-1])
                    out.append((e.id, float(top)))
        return sorted(out, key=lambda t: -t[1])
