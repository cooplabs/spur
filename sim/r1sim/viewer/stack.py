"""The viewer on the REAL processes (`python -m r1sim.viewer --stack`): the window shows the DDS-bridge simulator
(`r1sim.dds_bridge`, physics in real time behind the robot's own DDS topics) while the controller's real binaries run
against it - `r1rt` (the real-time core), `r1guard` (the hot standby) and `r1d` (the API service), started by
`r1sim.stack.ProcStack` exactly as the integration tests start them.

What it needs beyond the in-process viewer: CycloneDDS's Python binding (`uv sync --extra dds`; on macOS built against
our CycloneDDS: `rt/idl/build_cyclonedds.sh`) and the DDS build of `rt/` (`-DR1RT_WITH_CYCLONEDDS=ON`: the `r1rt` /
`r1guard` binaries, found through `R1RT_APPS_DIR` or `$R1MC3_BUILD_DIR/rt_dds/apps`). `available()` says whether both
are here.

How it runs: the stack starts (the bridge's emulated built-in holds the robot, r1rt in MONITOR), then the viewer asks
r1d for the takeover (`takeover`; on the stand `on_stand`) in the background - the robot is then balanced (or held) by
the real r1rt process. The window shows a COPY of the bridge's state (its qpos / qvel copied every frame, kinematics
recomputed): the physics thread is never touched for drawing. The keys go through the real interfaces: `S` stops the
r1rt PROCESS for 2 s (SIGSTOP / SIGCONT) - r1guard takes the motors (watch "motors owned by"); `U` asks r1d for the
operator's hand-back (`reclaim`); the jog keys send `stream.joint_jog` (r1d integrates the rate into STREAM setpoints,
TTL 0.3 s) and the hands `hand.preshape`; `P` pushes the torso, `G` / `D` / `M` are the bridge's own faults (lost
commands, late state, a motor error word). Object drop-in writes the object's pose into the running physics between
two steps of its thread (a visual demo feature; the robot's own state is never written).

The processes run WITHOUT real-time scheduling on a desktop: timings are "desktop, not real time"; a loaded machine
shows it (the bridge re-anchors its clock when it falls behind).
"""

from __future__ import annotations

import contextlib
import os
import signal
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco

from .. import joints as J
from .app import (
    DROP_FLOOR_XY,
    DROP_HEIGHT_M,
    DROP_TABLE_XY,
    HAND_STEP,
    JOG_TARGETS,
    OWNER_NAMES,
    POOL,
    PRESETS,
    PUSH_DIRS,
    PUSH_NAMES,
    SCENES,
    TABLE_TOP_Z,
    Event,
    Jog,
    Toggles,
    _half_height,
    _pool,
    camera_for,
    contact_rows,
    decorate_entities,
    load_entities,
    scene_option_for,
    set_parked,
    tilt_of,
    valid_scene,
)

STALL_S = 2.0  # `S`: r1rt SIGSTOPped this long (the guard takes over within ~10 ms)
PUSH_IMPULSE_NS = 9.0  # `P`: as the in-process viewer's 60 N x 0.15 s
JOG_RATE = {"arm_left": 0.3, "arm_right": 0.3, "head": 0.15}  # rad/s for the jog's TTL (0.3 s)
JOG_TTL_S = 0.3
STATUS_EVERY_S = 0.25  # r1d's status is polled this often (a background thread)
BRIDGE_FAULTS: dict[str, tuple[str, dict[str, Any], float]] = {
    "G": ("lowcmd_drop", {"fraction": 1.0}, 0.3),  # every rt/lowcmd lost for 0.3 s (the board holds, then damps)
    "D": ("lowstate_delay", {"delay_ms": 40.0}, 2.0),  # rt/lowstate 40 ms late for 2 s
    "M": ("motor_error", {"joint": 3, "word": 1}, 1.0),  # a motor error word on the left knee for 1 s
}


def available() -> bool:
    """True when the stack mode can run here: CycloneDDS's Python binding and the DDS build of r1rt / r1guard."""
    from r1mc3 import dds

    if not dds.available():
        return False
    from ..stack import apps_dir

    return apps_dir() is not None


def why_not() -> str:
    """Why the stack mode cannot run here (empty when it can)."""
    from r1mc3 import dds

    if not dds.available():
        return ("CycloneDDS's Python binding is not installed (uv sync --extra dds; on macOS first "
                "rt/idl/build_cyclonedds.sh and CYCLONEDDS_HOME)")  # fmt: skip
    from ..stack import apps_dir

    if apps_dir() is None:
        return "r1rt / r1guard are not built with DDS (cmake -S rt -B build/rt_dds -DR1RT_WITH_CYCLONEDDS=ON)"
    return ""


@dataclass
class StackConfig:
    scene: str = "floor"  # floor | table | stand (the bridge's hanging stance on its modelled stand)
    seed: int = 0
    domain: int | None = None  # DDS domain id (default: from the pid, in 70-79 + 10 x k)
    world_json: str | None = None
    camera: str = "free"
    width: int = 960
    height: int = 720
    takeover: bool = True  # ask r1d for the takeover / ON_STAND at start
    workdir: str | None = None


@dataclass
class _Status:
    data: dict[str, Any] = field(default_factory=dict)
    t_wall: float = 0.0


class StackApp:
    """The viewer's state on the process stack - the same interface as `App` (advance / key / overlay / camera /
    scene_option / decorate / summary / m / d / toggles / preset / t)."""

    def __init__(self, cfg: StackConfig) -> None:
        if not valid_scene(cfg.scene):
            raise ValueError(f"scene: one of {SCENES} or 'package.module:attr'")
        if cfg.camera not in PRESETS:
            raise ValueError(f"camera: one of {PRESETS}")
        reason = why_not()
        if reason:
            raise RuntimeError(f"--stack cannot run here: {reason}")
        from .. import world
        from ..dds_bridge import HW1_STAND, stand_extra
        from ..stack import ProcStack

        self.cfg = cfg
        self.toggles = Toggles()
        self.events: list[Event] = []
        self.preset = cfg.camera
        self.entities = load_entities(cfg.world_json) if cfg.world_json else []
        self.jog = Jog()
        self.controller = "r1rt + r1guard + r1d (processes)"
        hanging = cfg.scene == "stand"
        extras: list[Any] = [_pool]
        if hanging:  # the bridge's modelled stand (the cradle + the pole, HW1's measured hang), not the viewer's weld
            extras.append(stand_extra(HW1_STAND))

        def extra(spec: mujoco.MjSpec, info: Any) -> None:
            for fn in extras:
                fn(spec, info)

        w = world.build_world("real", scene=None if hanging else cfg.scene, head_camera=True, extra=extra)
        self.w = w
        self._tmp = tempfile.TemporaryDirectory(prefix="r1viewer_stack_") if cfg.workdir is None else None
        workdir = Path(cfg.workdir) if cfg.workdir is not None else Path(self._tmp.name)  # type: ignore[union-attr]
        domain = cfg.domain if cfg.domain is not None else 70 + os.getpid() % 10
        self.stack = ProcStack(
            domain=domain, workdir=workdir, shm_name=f"/r1mc3.viewer.{os.getpid()}",
            bridge={"w": w, "stance": "hanging" if hanging else "centred", "seed": int(cfg.seed)},
        )  # fmt: skip
        self.stack.start()
        br = self.stack.bridge
        self.br = br
        self.m: mujoco.MjModel = br.sim.m
        self.d = mujoco.MjData(self.m)
        self._pool_bodies = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"viewer_drop_{k}_{n[0]}")
                             for k, n in enumerate(POOL)]  # fmt: skip
        self._pool_geoms = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"viewer_drop_{k}")
                            for k in range(len(POOL))]  # fmt: skip
        self._n_drop = 0
        self._n_push = 0
        self.sim_t0 = float(br.sim.time)
        self._copy()
        from r1mc3.client import R1

        self.r1 = R1(port=self.stack.port, timeout_s=30.0)
        self._status = _Status()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._poller = threading.Thread(target=self._poll, name="viewer-status", daemon=True)
        self._poller.start()
        if cfg.takeover:
            threading.Thread(target=self._takeover, name="viewer-takeover", daemon=True).start()

    # ------------------------------------------------------------------ life cycle
    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(Exception):
            self.r1.close()
        self.stack.stop()
        if self._tmp is not None:
            self._tmp.cleanup()

    def __enter__(self) -> StackApp:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _event(self, what: str) -> str:
        self.events.append(Event(float(self.br.sim.time) - self.sim_t0, what))  # the robot's time, not the picture's
        return what

    def _call(self, op: str, **kw: Any) -> dict[str, Any]:
        """One r1d request on its own connection (the poller and the key handlers run in different threads)."""
        from r1mc3.client import R1

        with R1(port=self.stack.port, timeout_s=30.0) as c:
            rep = c.call(op, **kw)
        return {"ok": rep.ok, "code": rep.code, "data": rep.data, "details": rep.details}

    def _takeover(self) -> None:
        word = "on_stand" if self.cfg.scene == "stand" else "takeover"
        try:
            rep = self._call(word, requester="operator")
            self._event(f"{word}: {'accepted' if rep.get('ok') else rep.get('code', rep)}")
        except Exception as e:  # pragma: no cover - a dead r1d is reported, never raised into the window
            self._event(f"{word}: {e}")

    def _poll(self) -> None:
        from r1mc3.client import R1

        with R1(port=self.stack.port, timeout_s=5.0) as c:
            while not self._stop.is_set():
                with contextlib.suppress(Exception):
                    rep = c.status()
                    data = dict(rep.data)
                    with self._lock:
                        self._status = _Status(data, time.monotonic())
                self._stop.wait(STATUS_EVERY_S)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status.data)

    # ------------------------------------------------------------------ stepping (real time: physics is the bridge's)
    @property
    def t(self) -> float:
        return float(self.d.time) - self.sim_t0

    def _copy(self) -> None:
        src = self.br.sim.d
        self.d.qpos[:] = src.qpos
        self.d.qvel[:] = src.qvel
        self.d.time = float(src.time)
        mujoco.mj_forward(self.m, self.d)

    realtime = True  # the window draws `refresh()` every frame instead of advancing a lockstep clock

    def refresh(self) -> None:
        """The bridge's latest state, for drawing (paused: the picture holds, the robot keeps running)."""
        if not self.toggles.paused:
            self._copy()

    def advance(self, t_target: float, max_steps: int = 0, *, timeout_s: float = 120.0) -> int:
        """Wait (real time) until the bridge's simulated time reaches t_target (relative to the start), then copy its
        state for drawing. Paused: the picture stops, the robot does not (it runs in the bridge's thread)."""
        del max_steps
        if self.toggles.paused:
            return 0
        end = time.monotonic() + timeout_s
        while float(self.br.sim.time) - self.sim_t0 < t_target and time.monotonic() < end:
            time.sleep(0.005)
        self._copy()
        return 1

    # ------------------------------------------------------------------ keys
    def key(self, ch: str) -> str:
        c = ch.upper()
        what = ""
        if c in "123456" and c:
            self.preset = PRESETS[int(c) - 1]
            what = f"camera: {self.preset}"
        elif c == "P":
            k = self._n_push % len(PUSH_DIRS)
            self._n_push += 1
            dx, dy = PUSH_DIRS[k]
            self.br.push(PUSH_IMPULSE_NS, direction=(dx, dy, 0.0), duration_s=0.15)
            what = f"push {PUSH_IMPULSE_NS:.0f} N s on the torso ({PUSH_NAMES[k]})"
        elif c in BRIDGE_FAULTS:
            kind, params, dur = BRIDGE_FAULTS[c]
            self.br.inject(kind, duration_s=dur, **params)
            what = f"the bridge's fault {kind} for {dur} s"
        elif c == "N":
            what = "policy NaN: only in-process (the processes' policy cannot be reached from here)"
        elif c == "S":
            what = self.stall_core()
        elif c == "U":
            rep = self._call("reclaim", requester="operator")
            what = f"reclaim: {'accepted' if rep.get('ok') else rep.get('code', rep)}"
        elif c == "J":
            self.jog.target = (self.jog.target + 1) % len(JOG_TARGETS)
            what = f"jog target {self.jog_label()}"
        elif c in "[]" and c:
            name = self.jog.name
            n = len(_group_joints(name))
            if n:
                self.jog.joint[name] = (self.jog.joint.get(name, 0) + (1 if c == "]" else -1)) % n
            what = f"jog target {self.jog_label()}"
        elif c in "-=+" and c:
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
            what = "picture paused (the robot keeps running)" if self.toggles.paused else "picture running"
        return self._event(what) if what else ""

    def stall_core(self, seconds: float = STALL_S) -> str:
        """SIGSTOP the r1rt process for `seconds`, then SIGCONT: r1guard takes the motors over (~10 ms) and keeps
        them - the core stays silent until the operator's hand-back."""
        rt = self.stack.rt
        if rt is None or not rt.alive():
            return "stall: r1rt is not running"
        rt.signal(signal.SIGSTOP)
        threading.Timer(seconds, rt.signal, args=(signal.SIGCONT,)).start()
        return f"r1rt stopped (SIGSTOP) for {seconds} s - r1guard takes over"

    def jog_label(self) -> str:
        name = self.jog.name
        if name.startswith("hand_"):
            return f"{name}: closure {self.jog.closure[name[5:]]:.2f}"
        joints = _group_joints(name)
        j = joints[self.jog.joint.get(name, 0) % len(joints)]
        return f"{name}: {J.JOINTS[j]} {float(self.d.qpos[self.w.index.qadr[j]]):+.2f} rad"

    def jog_step(self, sign: float) -> str:
        """`-` / `=`: through r1d - `stream.joint_jog` (a rate on the selected joint for 0.3 s) or `hand.preshape`."""
        name = self.jog.name
        if name.startswith("hand_"):
            side = name[5:]
            c = min(1.0, max(0.0, self.jog.closure[side] + sign * HAND_STEP))
            self.jog.closure[side] = c
            shape = "open" if c < 0.25 else ("relax" if c < 0.75 else "fist")
            rep = self._call("hand.preshape", side=side, shape=shape, wait=False, requester="operator")
            return f"{side} hand preshape {shape}: {'ok' if rep.get('ok') else rep.get('code', rep)}"
        joints = _group_joints(name)
        k = self.jog.joint.get(name, 0) % len(joints)
        dq = [0.0] * len(joints)
        dq[k] = sign * JOG_RATE[name]
        rep = self._call("stream.joint_jog", group=name, dq=dq, ttl=JOG_TTL_S, requester="operator")
        word = "ok" if rep.get("ok") else rep.get("code", rep)
        return f"jog {J.JOINTS[joints[k]]} {sign * JOG_RATE[name]:+.2f} rad/s: {word}"

    def drop(self) -> str:
        """`O`: the next pool object placed in the running physics (between two of its steps)."""
        k = self._n_drop % len(POOL)
        name, typ, size, _ = POOL[k]
        table = self.cfg.scene == "table"
        x, y = (DROP_TABLE_XY if table else DROP_FLOOR_XY)[(self._n_drop // len(POOL) + k) % 6]
        z = (TABLE_TOP_Z if table else 0.0) + _half_height(typ, size) + DROP_HEIGHT_M
        m, d = self.br.sim.m, self.br.sim.d
        b, g = self._pool_bodies[k], self._pool_geoms[k]
        j = int(m.body_jntadr[b])
        qa, va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        set_parked(m, b, g, False)
        d.qvel[va : va + 6] = 0.0
        d.qpos[qa : qa + 7] = (x, y, z, 1.0, 0.0, 0.0, 0.0)
        self._n_drop += 1
        return f"dropped a {name} at ({x:.2f}, {y:.2f}) m"

    # ------------------------------------------------------------------ views
    def camera(self, preset: str | None = None) -> mujoco.MjvCamera:
        return camera_for(self.m, preset or self.preset, self.w.index.pelvis_body)

    def scene_option(self) -> mujoco.MjvOption:
        return scene_option_for(self.toggles)

    def decorate(self, scn: mujoco.MjvScene) -> int:
        return decorate_entities(scn, self.entities if self.toggles.world else [])

    def tilt_deg(self) -> float:
        return tilt_of(self.d.xquat[self.w.index.pelvis_body])

    def overlay(self) -> list[tuple[str, str]]:
        st = self.status()
        core = st.get("core") or {}
        guard = st.get("guard") or {}
        owner = (st.get("token") or {}).get("owner")
        rows: list[tuple[str, str]] = [("t", f"{self.t:7.2f} s{'  (picture paused)' if self.toggles.paused else ''}")]
        rows.append(("scene", f"{self.cfg.scene} (the DDS-bridge simulator, real time)"))
        rows.append(("r1rt", f"{core.get('mode', '?')}{'' if core.get('alive', True) else '  (silent)'}"))
        rows.append(("r1guard", f"{guard.get('mode', '?')}{'  ACTIVE' if guard.get('active') else ''}"))
        rows.append(("motors owned by", str(owner) if owner is not None else "?"))
        pz = float(self.d.xpos[self.w.index.pelvis_body][2])
        rows.append(("tilt / pelvis z", f"{self.tilt_deg():5.2f} deg / {pz:.3f} m"))
        rows.append(("jog", self.jog_label()))
        rows.append(contact_rows(self.m, self.d))
        rt = getattr(self.br.sim, "rt", None)
        if rt is not None and getattr(rt, "wall_s", 0.0) > 0:
            rows.append(("sim / wall", f"{rt.sim_s / rt.wall_s:.2f}"))
        if self.events:
            rows.append(("last", f"{self.events[-1].what} @ {self.events[-1].t:.2f} s"))
        return rows

    def overlay_text(self) -> tuple[str, str]:
        rows = self.overlay()
        return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)

    def summary(self) -> dict[str, Any]:
        return {"t": round(self.t, 4), "scene": self.cfg.scene, "controller": self.controller,
                "overlay": dict(self.overlay()), "events": [{"t": round(e.t, 3), "what": e.what} for e in self.events],
                "world_entities": len(self.entities), "status": self.status(),
                "owner_names": dict(OWNER_NAMES)}  # fmt: skip


def _group_joints(name: str) -> tuple[int, ...]:
    from ..v3_core import GROUP_ARM_LEFT, GROUP_ARM_RIGHT, GROUP_HEAD, GROUP_JOINTS

    g = {"arm_left": GROUP_ARM_LEFT, "arm_right": GROUP_ARM_RIGHT, "head": GROUP_HEAD}.get(name)
    return () if g is None else GROUP_JOINTS[g]


__all__ = ["StackApp", "StackConfig", "available", "why_not"]
