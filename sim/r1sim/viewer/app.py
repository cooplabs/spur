"""The viewer's state, shared by the interactive window and the headless renderer (`r1sim.viewer`).

`App(ViewerConfig(...))` builds the world (a generic scene), the simulation (the real-equivalent driver profile, a
fault schedule) and the controller, and answers what both front ends need: `advance(t)` (lockstep to a simulated
time), `key(code)` (the keyboard: faults, camera presets, overlay toggles, jog, object drop-in, pause), `overlay()`
(the text lines), `camera(preset)`, `scene_option()` (the contact / frame flags) and `decorate(scn)` (the declared world
boxes as translucent geoms). Nothing here opens a window: the window is MuJoCo's passive viewer (`__main__`), the
headless path is `mujoco.Renderer`.

The controllers are the controller's own C++ code, bound to Python (`r1rt_py`, built by `cmake -S rt`): `guarded`
(the default) = the real-time core AND r1guard on one shared segment, exactly the code of the `r1rt` and `r1guard`
binaries, stepped in lockstep with the physics (`r1sim.v3_guard`); `core` = the core alone (no hot standby: a stall is
the board's to survive); `policy` = the Python reference policy; `hold` = a stiff stand fixture; `none`. The process
version - the real binaries over DDS against the real-time simulator - is `r1sim.viewer.stack` (`--stack`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from .. import dex3, scenes, world
from .. import joints as J
from ..faults import Fault, FaultSchedule

FloatArray = npt.NDArray[np.float64]

# Generic scenes: r1sim.scenes' floor / table + the viewer's stand. Another package's scene (your own task scene) is
# named by its "package.module:attr" string (r1sim.scenes.resolve) - this package never names one.
SCENES = ("floor", "table", "stand")
CONTROLLERS = ("guarded", "core", "policy", "hold", "none")
PRESETS = ("free", "follow", "front", "side", "overhead", "head")
CONTROL_HZ = 500.0
PUSH_N = 60.0  # the push key: a horizontal force on the pelvis (N) for PUSH_S (an 0.15 s / 9 N s impulse)
PUSH_S = 0.15
PUSH_DIRS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))  # forward, back, left, right (base frame)
PUSH_NAMES = ("forward", "back", "left", "right")
# r1sim.v3_core MODE_*
MODE_NAMES = {0: "MONITOR", 1: "TAKEOVER_MIRROR", 2: "TAKEOVER_BLEND", 3: "BALANCE", 4: "GUARDED",
              5: "STAND_FREEZE", 6: "ON_STAND", 8: "DAMP"}  # fmt: skip
OWNER_NAMES = {0: "none", 1: "r1rt (the core)", 2: "r1guard"}  # the ownership token's owner byte (types.h)
# key -> (fault kind, duration s, params): the engine's / the core's fault injector (r1sim.faults)
FAULT_KEYS: dict[str, tuple[str, float, dict[str, Any]]] = {
    "N": ("policy_nan", 0.2, {}),  # the policy's action NaN -> the core's STAND_FREEZE (rung 3)
    "G": ("cmd_gap", 0.3, {}),  # the board sees no commands (the board model holds ~0.4 s, then damps)
    "D": ("state_delay", 2.0, {"delay_s": 0.04}),  # the robot's state 40 ms old for 2 s
    "M": ("motor_error", 1.0, {"joints": [3], "word": 1}),  # a motor error word on the left knee for 1 s
    "S": ("controller_stall", 0.5, {}),  # the controller stalls (guarded: the core alone, 2 s - see STALL_CORE_S)
}
STALL_CORE_S = 2.0  # guarded: `S` stalls the core (writer + policy: the whole r1rt) this long - r1guard takes over
# Jog (the keys J, [ ], - =; arrows in the window): the selected group / joint moved through the core's upper-body
# executor - the records r1d sends (a minimum-jerk TRAJECTORY with its keepalive) - or a hand's closure.
JOG_TARGETS = ("arm_left", "arm_right", "head", "hand_left", "hand_right")
JOG_STEP_RAD = {"arm_left": 0.10, "arm_right": 0.10, "head": 0.05}
JOG_S = 0.4  # one jog step's duration
JOG_RANGE_MARGIN_RAD = 0.05  # a jog stops this far inside the model's joint range
HAND_STEP = 0.25  # a hand's closure step (0 = the open preset, 1 = the fist preset)
# The window's special keys (GLFW key codes) -> the viewer's key names
SPECIAL_KEYS = {262: "]", 263: "[", 264: "-", 265: "="}  # right / left = next / previous joint, down / up = - / +
# Object drop-in: a pool of generic free bodies, parked out of the way (no contacts, gravity compensated) until `O`
# drops the next one in front of the robot (onto the table in the table scene).
POOL = (("box", mujoco.mjtGeom.mjGEOM_BOX, (0.025, 0.025, 0.025), (0.85, 0.30, 0.20, 1.0)),
        ("can", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.03, 0.05, 0.0), (0.25, 0.50, 0.85, 1.0)),
        ("ball", mujoco.mjtGeom.mjGEOM_SPHERE, (0.03, 0.0, 0.0), (0.95, 0.75, 0.20, 1.0)),
        ("block", mujoco.mjtGeom.mjGEOM_BOX, (0.04, 0.02, 0.015), (0.30, 0.70, 0.35, 1.0)),
        ("cup", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.035, 0.045, 0.0), (0.90, 0.90, 0.92, 1.0)),
        ("puck", mujoco.mjtGeom.mjGEOM_CYLINDER, (0.04, 0.012, 0.0), (0.20, 0.20, 0.22, 1.0)))  # fmt: skip
POOL_PARK = (-3.0, 3.0, 0.5)  # where the pool waits (world m; spaced along -x)
POOL_LIVE = (1, 1)  # a dropped object's contype / conaffinity (MuJoCo's defaults: it meets the floor and the robot)
DROP_FLOOR_XY = ((0.55, 0.0), (0.50, 0.20), (0.50, -0.20), (0.65, 0.10), (0.65, -0.10), (0.45, 0.0))
DROP_TABLE_XY = ((0.40, 0.10), (0.42, -0.10), (0.50, 0.25), (0.50, -0.25), (0.55, 0.0), (0.38, 0.0))
DROP_HEIGHT_M = 0.10  # dropped from this far above the support
# The stand scene: the robot hanging on a stand, feet off the floor - the pelvis held by a weld to the world (the
# stiffest stand; the DDS bridge's `stance="hanging"` models the real cradle + pole), a visual-only pole behind it.
HANG_DZ = 0.10  # m above the standing height
STAND_REQUEST_S = 1.5  # the hanging legs settle under the built-in's damping before ON_STAND is asked
STAND_RELEASE_S = 0.3
TABLE_TOP_Z = scenes.TABLE_TOP_Z_M  # the generic table's top (m)
HELP = (
    "1-6 camera (free follow front side overhead head)   P push   N policy NaN   G command gap   D state delay   "
    "M motor error   S stall the core   U reclaim   J jog target   [ ] joint   - = move (arrows)   O drop an object   "
    "C contacts   F frames   W world boxes   H text   R record   . screenshot   K pause"
)


@dataclass
class ViewerConfig:
    scene: str = "floor"
    controller: str = "guarded"
    profile: str = "real"
    model_path: str | None = None  # any MJCF file instead of a scene (passive: no controller)
    world_json: str | None = None  # declared entities (r1d's world.declare format) drawn as boxes
    seed: int | None = None
    camera: str = "free"
    width: int = 960
    height: int = 720


@dataclass
class Toggles:
    contacts: bool = False
    frames: bool = False
    world: bool = True
    text: bool = True
    paused: bool = False


@dataclass
class Event:
    t: float
    what: str


@dataclass
class _Push:
    t_end: float
    force: FloatArray


@dataclass
class Jog:
    """The jog selection and the hands' closures (0 open .. 1 fist)."""

    target: int = 0
    joint: dict[str, int] = field(default_factory=dict)
    closure: dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in dex3.SIDES})

    @property
    def name(self) -> str:
        return JOG_TARGETS[self.target % len(JOG_TARGETS)]


def _half_height(typ: Any, size: Any) -> float:
    if typ == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(size[0])
    if typ == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(size[1])
    return float(size[2])


def _pool(spec: mujoco.MjSpec, _info: Any) -> None:
    """The drop-in pool: free bodies parked out of the way (`App._set_parked` switches their contacts / gravity off)."""
    for k, (name, typ, size, rgba) in enumerate(POOL):
        b = spec.worldbody.add_body()
        b.name = f"viewer_drop_{k}_{name}"
        b.pos = [POOL_PARK[0] - 0.3 * k, POOL_PARK[1], POOL_PARK[2]]
        b.add_freejoint()
        g = b.add_geom()
        g.name = f"viewer_drop_{k}"
        g.type = typ
        g.size = list(size)
        g.mass = 0.08
        g.rgba = [*rgba[:3], 0.0]  # parked: invisible, no contacts, gravity compensated (set_parked)
        g.contype = g.conaffinity = 0
        b.gravcomp = 1.0


def set_parked(m: mujoco.MjModel, body: int, geom: int, parked: bool) -> None:
    """A pool object parked (no contacts, gravity compensated) or live (POOL_LIVE's bits, gravity on). The broad phase
    filters bodies by `body_contype` / `body_conaffinity` (each the OR of its geoms' bits, fixed at compile time), so
    both levels are switched."""
    ct, ca = (0, 0) if parked else POOL_LIVE
    m.geom_contype[geom], m.geom_conaffinity[geom] = ct, ca
    m.body_contype[body], m.body_conaffinity[body] = ct, ca
    m.body_gravcomp[body] = 1.0 if parked else 0.0
    m.geom_rgba[geom, 3] = 0.0 if parked else 1.0  # parked = invisible


def _stand_visual(spec: mujoco.MjSpec, _info: Any) -> None:
    """The stand's look only (no collisions, no mass): a floor plate, a pole behind the robot, an arm to its back."""
    st = spec.worldbody.add_body()
    st.name = "viewer_stand"
    st.pos = [-0.32, 0.0, 0.0]
    parts = ((mujoco.mjtGeom.mjGEOM_BOX, [0.22, 0.22, 0.01], [0.0, 0.0, 0.01]),
             (mujoco.mjtGeom.mjGEOM_CYLINDER, [0.022, 0.62, 0.0], [0.0, 0.0, 0.64]),
             (mujoco.mjtGeom.mjGEOM_BOX, [0.10, 0.07, 0.015], [0.08, 0.0, 1.20]))  # fmt: skip
    for k, (typ, size, pos) in enumerate(parts):
        g = st.add_geom()
        g.name = f"viewer_stand_{k}"
        g.type = typ
        g.size = size
        g.pos = pos
        g.contype = g.conaffinity = 0
        g.density = 0.0
        g.rgba = [0.08, 0.08, 0.09, 1.0]


def _hang_weld(z_hang: float) -> Callable[[mujoco.MjSpec, Any], None]:
    """A weld of the pelvis to the world, upright at height z_hang (the fixture of the ON_STAND lockstep test,
    tests/integration/test_v3_core_onstand.py): the robot hangs, feet off the floor."""

    def hang(spec: mujoco.MjSpec, _info: Any) -> None:
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = "pelvis"  # body2 = the world
        eq.name = "viewer_stand_hook"
        data = np.zeros(11)
        data[5] = -z_hang  # relpose: the world's origin in the pelvis frame (pelvis upright at z_hang)
        data[6] = 1.0
        data[10] = 1.0  # torquescale
        eq.data = data

    return hang


def valid_scene(scene: str) -> bool:
    """A built-in viewer scene or a "package.module:attr" string naming another package's scene builder."""
    return scene in SCENES or ":" in scene


def build_viewer_world(
    scene: str,
    *,
    profile: str = "real",
    seed: int | None = None,
    head_camera: bool = True,
    extra: Callable[[mujoco.MjSpec, Any], None] | None = None,
) -> tuple[world.World, float]:
    """The viewer's world for a scene (+ the drop-in pool, + `extra`) and how far the robot starts above its standing
    height. floor / table = r1sim.scenes; stand = no scene + the hanging weld + the stand's look; a "module:attr"
    string = that builder (r1sim.scenes.resolve)."""
    if not valid_scene(scene):
        raise ValueError(f"scene: one of {SCENES} or 'package.module:attr'")
    extras: list[Callable[[mujoco.MjSpec, Any], None]] = [_pool]
    dz = 0.0
    built: str | None = scene
    if scene == "stand":
        from ..model import centred_stand_pose

        w0 = world.build_world(profile, scene=None, visual=False)
        z_hang = float(w0.initial_qpos(centred_stand_pose(w0.model))[w0.index.base_qadr + 2]) + HANG_DZ
        extras += [_hang_weld(z_hang), _stand_visual]
        built, dz = None, HANG_DZ
    if extra is not None:
        extras.append(extra)

    def add(spec: mujoco.MjSpec, info: Any) -> None:
        for fn in extras:
            fn(spec, info)

    w = world.build_world(profile, scene=built, seed=seed, head_camera=head_camera, extra=add)
    return w, dz


def load_entities(path: str | Path) -> list[dict[str, Any]]:
    """Declared world entities: a JSON list (or {"entities": [...]}) of r1d `world.declare` entities - kind surface /
    container / object / obstacle, pose {pos, yaw?}, size / size_outer (full sizes, m)."""
    d = json.loads(Path(path).read_text())
    ents = d.get("entities", d) if isinstance(d, dict) else d
    if not isinstance(ents, list):
        raise ValueError(f"{path}: a list of entities or {{'entities': [...]}}")
    return [dict(e) for e in ents]


ENTITY_RGBA = {"surface": (0.2, 0.8, 0.3, 0.25), "container": (0.2, 0.5, 0.9, 0.3), "object": (0.95, 0.6, 0.1, 0.35),
               "obstacle": (0.9, 0.15, 0.15, 0.3)}  # fmt: skip


def tilt_of(quat_wxyz: npt.ArrayLike) -> float:
    """The tilt of a body's z axis from vertical (deg) for its world quaternion (w, x, y, z)."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)))))


def decorate_entities(scn: mujoco.MjvScene, entities: list[dict[str, Any]]) -> int:
    """Declared world entities as translucent boxes / spheres / cylinders appended to a scene; returns how many."""
    n = 0
    for e in entities:
        if scn.ngeom >= scn.maxgeom:
            break
        kind = str(e.get("kind", "obstacle"))
        pose = e.get("pose", {})
        pos = np.asarray(pose.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        yaw = float(pose.get("yaw", 0.0))
        size = np.asarray(e.get("size") or e.get("size_outer") or [0.05, 0.05, 0.05], dtype=np.float64)
        shape = str(e.get("shape", "box"))
        c, s = math.cos(yaw), math.sin(yaw)
        mat = np.array([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0])
        rgba = np.asarray(ENTITY_RGBA.get(kind, ENTITY_RGBA["obstacle"]), dtype=np.float32)
        g = scn.geoms[scn.ngeom]
        if shape == "sphere":
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size[0], 0, 0]), pos, mat, rgba)
        elif shape == "cylinder":
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([size[0], size[-1], 0]), pos, mat, rgba)
        else:  # r1d entities give FULL box sizes for surfaces / containers, half sizes for objects / obstacles
            half = size / 2.0 if kind in ("surface", "container") else size
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, half[:3], pos, mat, rgba)
        scn.ngeom += 1
        n += 1
    return n


def camera_for(m: mujoco.MjModel, preset: str, pelvis_body: int | None) -> mujoco.MjvCamera:
    """The camera for a preset: free (an orbit around the robot), follow (tracks the pelvis), front / side /
    overhead (fixed free cameras), head (the robot's head camera, when the model has it)."""
    p = preset
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    if p == "head" and pelvis_body is not None:
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
        if cid >= 0:
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cid
            return cam
        p = "front"
    look = np.array([0.15, 0.0, 0.75])
    if pelvis_body is not None and p == "follow":
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = pelvis_body
        cam.distance, cam.azimuth, cam.elevation = 2.2, 150.0, -18.0
        return cam
    views = {"free": (2.6, 145.0, -20.0), "front": (2.4, 180.0, -12.0), "side": (2.4, 90.0, -10.0),
             "overhead": (2.2, 180.0, -85.0)}  # fmt: skip
    cam.distance, cam.azimuth, cam.elevation = views.get(p, views["free"])
    cam.lookat[:] = look if pelvis_body is not None else m.stat.center
    if pelvis_body is None:
        cam.distance = 2.0 * float(m.stat.extent)
    return cam


def scene_option_for(t: Toggles) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = t.contacts
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = t.contacts
    opt.frame = mujoco.mjtFrame.mjFRAME_BODY if t.frames else mujoco.mjtFrame.mjFRAME_NONE
    return opt


def contact_rows(m: mujoco.MjModel, d: mujoco.MjData) -> tuple[str, str]:
    f = np.zeros(6)
    fmax = 0.0
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f)
        fmax = max(fmax, abs(float(f[0])))
    return ("contacts", f"{d.ncon} (max normal {fmax:.0f} N)")


class App:
    def __init__(self, cfg: ViewerConfig) -> None:
        if not valid_scene(cfg.scene) and cfg.model_path is None:
            raise ValueError(f"scene: one of {SCENES}, 'package.module:attr' (or --model FILE)")
        if cfg.controller not in CONTROLLERS:
            raise ValueError(f"controller: one of {CONTROLLERS}")
        if cfg.camera not in PRESETS:
            raise ValueError(f"camera: one of {PRESETS}")
        self.cfg = cfg
        self.toggles = Toggles()
        self.events: list[Event] = []
        self.preset = cfg.camera
        self.entities = load_entities(cfg.world_json) if cfg.world_json else []
        self.faults = FaultSchedule(seed=cfg.seed or 0)
        self._push: _Push | None = None
        self._n_push = 0
        self.w: world.World | None = None
        self.ctl: Any = None
        self.sim: Any = None
        self.controller = "none"
        self.jog = Jog()
        self._pool_bodies: list[int] = []
        self._pool_geoms: list[tuple[int, int, int]] = []  # (geom, contype, conaffinity) once dropped
        self._n_drop = 0
        self._reclaim_seq: int | None = None
        if cfg.model_path is not None:
            self.m = mujoco.MjModel.from_xml_path(str(cfg.model_path))
            self.d = mujoco.MjData(self.m)
            mujoco.mj_forward(self.m, self.d)
        else:
            self._build_robot()
        self.time0 = float(self.d.time)

    # ------------------------------------------------------------------------------------------------- building
    def _build_robot(self) -> None:
        from ..driver import driver_for_profile
        from ..engine import Simulation

        cfg = self.cfg
        self.w, dz = build_viewer_world(cfg.scene, profile=cfg.profile, seed=cfg.seed)
        w = self.w
        self.sim = Simulation(w.model, w.index, driver=driver_for_profile(w.profile), faults=self.faults)
        self.m, self.d = self.sim.m, self.sim.d
        self._init_pool()
        ctl, pose, name, builtin = self._controller(w)
        q = w.initial_qpos(pose)
        q[w.index.base_qadr + 2] += dz
        self.sim.reset(q)
        self._park_all()
        if builtin is not None:  # Unitree's built-in, damping the hanging legs until ON_STAND's ReleaseMode
            self.sim.add_controller(builtin, getattr(ctl, "writer_period_s", ctl.period_s), "builtin")
        if ctl is not None:
            self.sim.add_controller(self._with_hands(ctl), getattr(ctl, "period_s", 1.0 / CONTROL_HZ), name)
        self.ctl, self.controller = ctl, name

    def _controller(self, w: world.World) -> tuple[Any, FloatArray, str, Any]:
        """The controller (the C++ core + r1guard in lockstep, the core alone, the reference policy, the stand-hold
        fixture or none), the robot's start pose and, on the stand, Unitree's emulated built-in. "guarded" / "core"
        fall back to "policy" when the binding (r1rt_py) is not built (on the stand: to "hold")."""
        from ..control import PolicyController, StandHold
        from ..model import centred_stand_pose

        want = self.cfg.controller
        stand = self.cfg.scene == "stand"
        if want in ("guarded", "core"):
            from .. import cpp_policy

            if cpp_policy.find_module() is None:
                fallback = "hold" if stand else "policy"
                self.events.append(Event(0.0, f"r1rt_py not built: {fallback} instead of the C++ {want}"))
                want = fallback
            else:
                from ..v3_core import EmulatedBuiltin, TakeoverPlan, V3CoreController
                from ..v3_guard import V3GuardedController

                cls = V3GuardedController if want == "guarded" else V3CoreController
                if stand:  # r1d's part of ON_STAND through the mode mailbox, as on the robot (DESIGN §3.5)
                    builtin = EmulatedBuiltin(w.model, leg_kp=0.0)
                    plan = TakeoverPlan(builtin, t_request_s=STAND_REQUEST_S, release_s=STAND_RELEASE_S,
                                        target="on_stand", via_mailbox=True)  # fmt: skip
                    ctl = cls(w.model, faults=self.faults, takeover=plan)
                    return ctl, centred_stand_pose(w.model), want, builtin
                return cls(w.model, faults=self.faults), J.POSE_STAND.copy(), want, None
        if want == "policy" and stand:
            self.events.append(Event(0.0, "on the stand: the stand fixture instead of the balance policy"))
            want = "hold"
        if want == "policy":
            from r1mc3.policy import POLICY_DIR, ReferencePolicy

            return PolicyController(ReferencePolicy.load(POLICY_DIR)), J.POSE_STAND.copy(), "policy", None
        if want == "hold":
            hold = StandHold.for_model(w.model)
            return hold, hold.pose, "hold", None
        return None, centred_stand_pose(w.model), "none", None

    def _with_hands(self, ctl: Any) -> Callable[[Any], Any]:
        """The controller's command with the hands at the jog's closures (the core holds the hands open: the hands are
        r1hands' business - the viewer stands in for it, as the DDS bridge's hand hold does)."""
        open_ = {s: dex3.preset("open", s) for s in dex3.SIDES}
        fist = {s: dex3.preset("fist", s) for s in dex3.SIDES}

        def call(st: Any) -> Any:
            cmd = ctl(st)
            if cmd is not None:
                for side, h in cmd.hands.items():
                    c = self.jog.closure.get(side, 0.0)
                    if c > 0.0:
                        h.q[:] = (1.0 - c) * open_[side] + c * fist[side]
            return cmd

        return call

    # ------------------------------------------------------------------------------------------------- drop-in pool
    def _init_pool(self) -> None:
        self._pool_bodies, self._pool_geoms = [], []
        for k in range(len(POOL)):
            bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"viewer_drop_{k}_{POOL[k][0]}")
            gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"viewer_drop_{k}")
            if bid < 0 or gid < 0:
                return
            self._pool_bodies.append(bid)
            self._pool_geoms.append((gid, *POOL_LIVE))

    def _park_all(self) -> None:
        for k in range(len(self._pool_bodies)):
            self._set_parked(k, True)

    def _set_parked(self, k: int, parked: bool) -> None:
        set_parked(self.m, self._pool_bodies[k], self._pool_geoms[k][0], parked)

    def _free_adr(self, body: int) -> tuple[int, int]:
        j = int(self.m.body_jntadr[body])
        return int(self.m.jnt_qposadr[j]), int(self.m.jnt_dofadr[j])

    def drop(self) -> str:
        """`O`: the next pool object dropped in front of the robot (onto the table's top in the table scene; the
        oldest one comes back once all have been dropped)."""
        if self.w is None or not self._pool_bodies:
            return "drop: no robot scene"
        k = self._n_drop % len(self._pool_bodies)
        name, typ, size, _ = POOL[k]
        table = self.cfg.scene == "table"
        x, y = (DROP_TABLE_XY if table else DROP_FLOOR_XY)[(self._n_drop // len(POOL) + k) % 6]
        z = (TABLE_TOP_Z if table else 0.0) + _half_height(typ, size) + DROP_HEIGHT_M
        qa, va = self._free_adr(self._pool_bodies[k])
        self._set_parked(k, False)
        self.d.qpos[qa : qa + 7] = (x, y, z, 1.0, 0.0, 0.0, 0.0)
        self.d.qvel[va : va + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self._n_drop += 1
        return f"dropped a {name} at ({x:.2f}, {y:.2f}) m"

    # ------------------------------------------------------------------------------------------------- jog
    def _jog_group(self) -> tuple[int, tuple[int, ...]] | None:
        from ..v3_core import GROUP_ARM_LEFT, GROUP_ARM_RIGHT, GROUP_HEAD, GROUP_JOINTS

        g = {"arm_left": GROUP_ARM_LEFT, "arm_right": GROUP_ARM_RIGHT, "head": GROUP_HEAD}.get(self.jog.name)
        return None if g is None else (g, GROUP_JOINTS[g])

    def jog_label(self) -> str:
        name = self.jog.name
        if name.startswith("hand_"):
            return f"{name}: closure {self.jog.closure[name[5:]]:.2f}"
        grp = self._jog_group()
        assert grp is not None
        joints = grp[1]
        j = joints[self.jog.joint.get(name, 0) % len(joints)]
        if self.controller in ("guarded", "core"):
            return f"{name}: {J.JOINTS[j]} {float(self.ctl.last_q[j]):+.2f} rad"
        return f"{name}: {J.JOINTS[j]}"

    def jog_step(self, sign: float) -> str:
        """`-` / `=`: the selected joint one step (a 0.4 s minimum-jerk TRAJECTORY through the core's executor, with
        its keepalive - what r1d sends), or the selected hand's closure one step."""
        name = self.jog.name
        if name.startswith("hand_"):
            side = name[5:]
            c = min(1.0, max(0.0, self.jog.closure[side] + sign * HAND_STEP))
            self.jog.closure[side] = c
            return f"{side} hand closure {c:.2f}"
        if self.controller not in ("guarded", "core"):
            return "jog: only with the C++ core (--controller guarded or core)"
        if self.token_owner() == 2:
            return "jog: r1guard holds the motors (the core is silent) - no arm motions until the operator's hand-back"
        mode = MODE_NAMES.get(int(self.ctl.core.mode), str(int(self.ctl.core.mode)))
        if mode not in ("BALANCE", "ON_STAND"):
            return f"jog: the core takes arm / head motions in BALANCE or ON_STAND (now {mode})"
        grp = self._jog_group()
        assert grp is not None
        group, joints = grp
        k = self.jog.joint.get(name, 0) % len(joints)
        q_goal = np.asarray(self.ctl.last_q, dtype=np.float64)[list(joints)].copy()
        j = joints[k]
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{J.JOINTS[j]}_joint")
        lo, hi = (float(self.m.jnt_range[jid][0]), float(self.m.jnt_range[jid][1])) if jid >= 0 else (-9.0, 9.0)
        target = q_goal[k] + sign * JOG_STEP_RAD[name]
        q_goal[k] = min(hi - JOG_RANGE_MARGIN_RAD, max(lo + JOG_RANGE_MARGIN_RAD, target))
        t = float(self.d.time)
        self.ctl.send_trajectory(round(t * 1e9), group, q_goal, JOG_S, keepalive_until_s=t + JOG_S + 0.2)
        return f"jog {J.JOINTS[j]} -> {q_goal[k]:+.2f} rad"

    def jog_select(self, d_joint: int = 0, d_target: int = 0) -> str:
        if d_target:
            self.jog.target = (self.jog.target + d_target) % len(JOG_TARGETS)
        elif d_joint:
            name = self.jog.name
            grp = self._jog_group()
            if grp is not None:
                self.jog.joint[name] = (self.jog.joint.get(name, 0) + d_joint) % len(grp[1])
        return f"jog target {self.jog_label()}"

    def reclaim(self) -> str:
        """`U`: the operator's hand-back from r1guard to the core (the core's mode mailbox, requester OPERATOR). The
        core refuses it while balancing (a balancing hand-back is not offered: the guard keeps the robot); on the
        stand it is allowed."""
        if self.controller != "guarded":
            return "reclaim: only with --controller guarded"
        from ..v3_core import REQ_RECLAIM, REQUESTER_OPERATOR

        t_ns = round(float(self.d.time) * 1e9)
        self._reclaim_seq = int(self.ctl.view.request_mode(REQ_RECLAIM, REQUESTER_OPERATOR, t_ns))
        return f"reclaim asked (request {self._reclaim_seq})"

    def _reclaim_reply(self) -> None:
        if self._reclaim_seq is None or self.controller != "guarded":
            return
        rep = self.ctl.view.mode_reply(self._reclaim_seq)
        if rep is not None:
            status, refusal, mode = (int(x) for x in rep)
            self._reclaim_seq = None
            reasons = {int(v): k for k, v in dict(self.ctl.mod.refusals()).items()}
            word = "accepted" if status == 0 else f"refused ({reasons.get(refusal, refusal)})"
            self.events.append(Event(self.t, f"reclaim {word}; mode {MODE_NAMES.get(mode, mode)}"))

    # ------------------------------------------------------------------------------------------------- stepping
    @property
    def t(self) -> float:
        return float(self.d.time) - self.time0

    def advance(self, t_target: float, max_steps: int = 2000) -> int:
        """Lockstep to simulated time t_target (relative to the start), at most max_steps physics steps; returns
        how many ran. A push in progress is applied as an external force on the pelvis."""
        if self.toggles.paused:
            return 0
        dt = float(self.m.opt.timestep)
        n = 0
        while self.t + 0.5 * dt < t_target and n < max_steps:
            self._apply_push()
            if self.sim is not None:
                self.sim.step(1)
            else:
                mujoco.mj_step(self.m, self.d)
            n += 1
        self._apply_push()
        self._reclaim_reply()
        return n

    def _apply_push(self) -> None:
        if self.w is None:
            return
        b = self.w.index.pelvis_body
        if self._push is not None and self.t < self._push.t_end:
            self.d.xfrc_applied[b, :3] = self._push.force
        else:
            self.d.xfrc_applied[b, :3] = 0.0
            self._push = None

    def push(self, force_xy: tuple[float, float] | None = None, duration_s: float = PUSH_S) -> str:
        """A horizontal push on the pelvis (N, base frame): the given force, else the next of PUSH_DIRS at PUSH_N."""
        if self.w is None:
            return "push: no robot in this model"
        if force_xy is None:
            k = self._n_push % len(PUSH_DIRS)
            self._n_push += 1
            dx, dy = PUSH_DIRS[k]
            fx, fy, label = dx * PUSH_N, dy * PUSH_N, PUSH_NAMES[k]
        else:
            fx, fy = float(force_xy[0]), float(force_xy[1])
            label = f"{math.degrees(math.atan2(fy, fx)):.0f} deg"
        self._push = _Push(self.t + duration_s, np.array([fx, fy, 0.0]))
        return f"push {math.hypot(fx, fy):.0f} N for {duration_s} s ({label})"

    # ------------------------------------------------------------------------------------------------- keys
    def key(self, ch: str) -> str:
        """One key (a character: GLFW's keycode as chr, letters upper case) -> what it did (also kept in events)."""
        c = ch.upper()
        what = ""
        if c in "123456" and c:
            self.preset = PRESETS[int(c) - 1]
            what = f"camera: {self.preset}"
        elif c == "P":
            what = self.push()
        elif c in FAULT_KEYS:
            what = self._fault_key(c)
        elif c == "U":
            what = self.reclaim()
        elif c == "J":
            what = self.jog_select(d_target=1) if self.w is not None else "jog: no robot in this model"
        elif c in "[]" and c and self.w is not None:
            what = self.jog_select(d_joint=1 if c == "]" else -1)
        elif c in "-=+" and c and self.w is not None:
            what = self.jog_step(-1.0 if c == "-" else 1.0)
        elif c == "O":
            what = self.drop()
        elif c == "C":
            self.toggles.contacts = not self.toggles.contacts
            what = f"contacts {'on' if self.toggles.contacts else 'off'}"
        elif c == "F":
            self.toggles.frames = not self.toggles.frames
            what = f"frames {'on' if self.toggles.frames else 'off'}"
        elif c == "W":
            self.toggles.world = not self.toggles.world
            what = f"world boxes {'on' if self.toggles.world else 'off'}"
        elif c == "H":
            self.toggles.text = not self.toggles.text
            what = f"text {'on' if self.toggles.text else 'off'}"
        elif c == "K":
            self.toggles.paused = not self.toggles.paused
            what = "paused" if self.toggles.paused else "running"
        if what:
            self.events.append(Event(self.t, what))
        return what

    def _fault_key(self, c: str) -> str:
        kind, dur, params = FAULT_KEYS[c]
        if self.sim is None:
            return f"{kind}: no controller in this model"
        now = float(self.d.time)
        if kind.startswith("policy_"):
            if self.controller not in ("guarded", "core"):
                return f"{kind}: only with the C++ core (--controller guarded or core)"
            from ..v3_core import INJECT_POLICY_NAN  # the core takes network-output faults itself

            t0 = round(now * 1e9)
            self.ctl.core.inject(INJECT_POLICY_NAN, t0, t0 + round(dur * 1e9), 0.0)
            self.faults.add(Fault(kind, now, dur, params))  # type: ignore[arg-type]
            return f"fault {kind} for {dur} s"
        if kind == "controller_stall" and self.controller == "guarded":
            # the whole r1rt (writer + policy) stalls; r1guard - its own process on the robot - keeps running
            self.faults.add(Fault(kind, now, STALL_CORE_S, {"name": "v3_core"}))  # type: ignore[arg-type]
            return f"fault {kind} of the core for {STALL_CORE_S} s (r1guard keeps running)"
        self.faults.add(Fault(kind, now, dur, params))  # type: ignore[arg-type]
        return f"fault {kind} for {dur} s" + (f" {params}" if params else "")

    # ------------------------------------------------------------------------------------------------- views
    def camera(self, preset: str | None = None) -> mujoco.MjvCamera:
        return camera_for(self.m, preset or self.preset, None if self.w is None else self.w.index.pelvis_body)

    def scene_option(self) -> mujoco.MjvOption:
        return scene_option_for(self.toggles)

    def decorate(self, scn: mujoco.MjvScene) -> int:
        """The declared world entities as translucent boxes / spheres / cylinders in the scene (appended after
        mjv_updateScene, as MuJoCo's own decorations); returns how many were added."""
        return decorate_entities(scn, self.entities if self.toggles.world else [])

    # ------------------------------------------------------------------------------------------------- text
    def mode_name(self) -> str:
        if self.controller in ("guarded", "core") and self.ctl is not None:
            return MODE_NAMES.get(int(self.ctl.core.mode), str(int(self.ctl.core.mode)))
        return ""

    def token_owner(self) -> int | None:
        """The ownership token's owner (OWNER_NAMES) with the guarded controller, else None. The guard's own word
        first: the segment's state record is the core's, and a stalled core does not refresh it."""
        if self.controller != "guarded" or self.ctl is None:
            return None
        if bool(self.ctl.guard.active):
            return 2
        st = self.ctl.view.state()
        return None if st is None else int(st["token_owner"])

    def guard_takeovers(self) -> int:
        return int(self.ctl.guard_status()["takeovers"]) if self.controller == "guarded" else 0

    def tilt_deg(self) -> float:
        if self.w is None:
            return math.nan
        return tilt_of(self.d.xquat[self.w.index.pelvis_body])

    def pelvis_z(self) -> float:
        return math.nan if self.w is None else float(self.d.xpos[self.w.index.pelvis_body][2])

    def overlay(self) -> list[tuple[str, str]]:
        """(label, value) lines: time, scene / controller and the core's mode, the token's owner, base tilt, pelvis
        height, the largest joint torque, contacts, the hands' largest pad reading, the jog, active faults, the last
        event."""
        rows: list[tuple[str, str]] = [("t", f"{self.t:7.2f} s{'  (paused)' if self.toggles.paused else ''}")]
        rows.append(("scene", self.cfg.model_path or self.cfg.scene))
        mode = self.mode_name()
        owner = self.token_owner()
        if owner == 2:  # r1guard holds the motors: the core's last mode, silent (it never writes without the token)
            mode = f"core {mode}, silent"
        rows.append(("controller", f"{self.controller}{'  ' + mode if mode else ''}"))
        if owner is not None:
            who = OWNER_NAMES.get(owner, str(owner))
            rows.append(("motors owned by", f"{who}  (guard takeovers {self.guard_takeovers()})"))
        if self.w is not None:
            rows.append(("tilt / pelvis z", f"{self.tilt_deg():5.2f} deg / {self.pelvis_z():.3f} m"))
            st = self.sim.read_state() if self.sim is not None else None
            if st is not None:
                k = int(np.argmax(np.abs(st.tau_est)))
                rows.append(("max |torque|", f"{abs(float(st.tau_est[k])):5.1f} N m ({J.JOINTS[k]})"))
                pads = {s: float(np.max(p)) for s, p in st.hand_pressure.items() if np.size(p)}
                if pads:
                    rows.append(("pads max", "  ".join(f"{s} {v:.0f}" for s, v in sorted(pads.items()))))
            rows.append(("jog", self.jog_label()))
        rows.append(contact_rows(self.m, self.d))
        now = float(self.d.time)
        active = [f"{x.kind} ({x.t_end - now:.1f} s)" for x in self.faults.faults if x.active(now)]
        if self._push is not None:
            active.append("push")
        rows.append(("faults", ", ".join(active) or "-"))
        if self.events:
            rows.append(("last", f"{self.events[-1].what} @ {self.events[-1].t:.2f} s"))
        return rows

    def overlay_text(self) -> tuple[str, str]:
        rows = self.overlay()
        return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)

    def summary(self) -> dict[str, Any]:
        return {"t": round(self.t, 4), "scene": self.cfg.model_path or self.cfg.scene, "controller": self.controller,
                "overlay": dict(self.overlay()), "events": [{"t": round(e.t, 3), "what": e.what} for e in self.events],
                "world_entities": len(self.entities)}  # fmt: skip


__all__ = [
    "CONTROLLERS", "FAULT_KEYS", "HELP", "JOG_TARGETS", "OWNER_NAMES", "PRESETS", "SCENES", "SPECIAL_KEYS", "App",
    "ViewerConfig", "build_viewer_world", "camera_for", "decorate_entities", "load_entities", "tilt_of", "valid_scene",
]  # fmt: skip
