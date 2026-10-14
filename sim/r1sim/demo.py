"""Demo: the R1 with both Dex3-1 hands standing UNDER THE BALANCE POLICY in a scene (the plain table by default).

    uv run python -m r1sim.demo              # headless: PNG + numbers (scripts/run.sh python -m r1sim.demo: the same
                                             #   through the local site settings)
    uv run python -m r1sim.demo --fixture    # the session-1 stand-hold fixture instead, for comparison
    uv run mjpython -m r1sim.demo --viewer   # macOS: also opens MuJoCo's window (scripts/viewer.sh shows how)

The one-command demo of the whole controller - its C++ core and r1guard balancing the simulated R1 in a window - is
`scripts/demo.sh` (r1sim.viewer); this module is the numbers check of the balance policy alone.

What it does, in order:
1. builds the world (robot profile, default "real", + the scene: `--scene`, r1sim.scenes - "table" by default: a plain
   table with a box, a can and a ball; "floor"; or a "package.module:attr" scene);
2. LOCKSTEP: the live balance policy (R1MC-Manip2-Lean via r1mc3.policy.ReferencePolicy, bit-identical to v2's
   runner) in `r1sim.control.PolicyController` - 50 Hz network inside a 500 Hz controller (the train profile's 5 ms
   steps give 200 Hz) - from the policy's own default pose (Unitree's stand lock, joints.POSE_STAND) on the floor, for
   `--seconds` of simulated time, as fast as the Mac can step it -> the lockstep real-time factor;
   with `--fixture`: the session-1 stand hold (stiff joint PD at the CoM-centred pose; balances nothing);
3. checks the robot stands (base tilt, pelvis height, feet slip) and the scene stays put (object displacement);
4. writes a headless PNG (the default views - behind-right, right side - and the hands);
5. REAL TIME: physics in its own thread paced by the wall clock, the same controller in another thread at 500 Hz ->
   the achieved real-time factor and the pacing counters;
6. prints a summary; exit code 1 if the robot did not stand.
The arms are held at the policy's default arm pose (the policy is arms_external) and the hands open; no hand, arm
or task controller runs. The physics is the session-1 model; its parameters are estimates (models/provenance.yaml).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import joints, render, world
from .control import PolicyController, StandHold
from .driver import driver_for_profile
from .engine import Controller, ControllerThread, Simulation

CONTROL_HZ = 500.0  # the v3 writer rate (DESIGN §0.1 D7)
TILT_LIMIT_DEG = 10.0  # "stands": the base stays within this of its start tilt


@dataclass
class StandCheck:
    tilt_deg: float  # base tilt change from the start (deg)
    tilt_now_deg: float  # base tilt from upright at the end (deg)
    pelvis_dz_m: float  # pelvis height change from the start (m)
    feet_slip_mm: float  # largest horizontal foot-site movement (mm)
    objects_moved_mm: float  # largest horizontal displacement of any scene object (mm)

    @property
    def stands(self) -> bool:
        return self.tilt_deg < TILT_LIMIT_DEG and abs(self.pelvis_dz_m) < 0.05


def _tilt_deg(quat_wxyz: np.ndarray) -> float:
    """Angle between the body z axis and the world z axis."""
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat_wxyz, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(r[8], -1.0, 1.0))))


class _Probe:
    """Ground-truth readings of the simulated world for the stand check."""

    def __init__(self, sim: Simulation, w: world.World) -> None:
        self.sim, self.w = sim, w
        m = sim.m
        self.obj_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o.body) for o in (w.scene.objects if w.scene else ())
        ]
        self.base = self.read()

    def read(self) -> dict[str, np.ndarray | float]:
        d, ix = self.sim.d, self.w.index
        return {
            "tilt": _tilt_deg(d.xquat[ix.pelvis_body]),
            "z": float(d.xpos[ix.pelvis_body][2]),
            "feet": np.array([d.site_xpos[s][:2] for s in ix.foot_sites]),
            "objs": np.array([d.xpos[b][:2] for b in self.obj_ids]).reshape(-1, 2),
        }

    def check(self) -> StandCheck:
        now, b = self.read(), self.base
        feet = np.asarray(now["feet"]) - np.asarray(b["feet"])
        objs = np.asarray(now["objs"]) - np.asarray(b["objs"])
        return StandCheck(
            tilt_deg=abs(float(now["tilt"]) - float(b["tilt"])),
            tilt_now_deg=float(now["tilt"]),
            pelvis_dz_m=float(now["z"]) - float(b["z"]),
            feet_slip_mm=1000.0 * float(np.linalg.norm(feet, axis=1).max()),
            objects_moved_mm=1000.0 * float(np.linalg.norm(objs, axis=1).max()) if len(objs) else 0.0,
        )


def _new_controller(w: world.World, fixture: bool) -> tuple[Controller, np.ndarray]:
    """The controller and the robot's start pose: the balance policy from its default pose, or the S1 fixture from
    the CoM-centred pose (the only pose a plain PD can hold)."""
    if fixture:
        hold = StandHold.for_model(w.model)
        return hold, hold.pose
    from r1mc3.policy import POLICY_DIR, ReferencePolicy

    return PolicyController(ReferencePolicy.load(POLICY_DIR)), joints.POSE_STAND.copy()


def _new_sim(w: world.World, fixture: bool = False) -> tuple[Simulation, Controller]:
    ctrl, pose = _new_controller(w, fixture)
    sim = Simulation(w.model, w.index, driver=driver_for_profile(w.profile))
    sim.reset(w.initial_qpos(pose))
    return sim, ctrl


def lockstep(
    w: world.World, seconds: float, *, fixture: bool = False
) -> tuple[Simulation, StandCheck, float, Controller]:
    """The controller in lockstep for `seconds` of sim time. Returns the sim, the stand check, the lockstep RTF and
    the controller (for its counters)."""
    sim, ctrl = _new_sim(w, fixture)
    probe = _Probe(sim, w)
    sim.add_controller(ctrl, 1.0 / CONTROL_HZ, "stand_hold" if fixture else "policy")
    t0 = time.perf_counter()
    sim.run_for(seconds)
    wall = time.perf_counter() - t0
    return sim, probe.check(), sim.time / wall, ctrl


@dataclass
class RealtimeResult:
    seconds: float
    achieved_rtf: float
    overruns: int
    max_lag_ms: float
    rephases: int
    controller_ticks: int
    controller_late: int
    check: StandCheck


def realtime(w: world.World, seconds: float, *, fixture: bool = False) -> RealtimeResult:
    """Physics in its own wall-clock-paced thread, the controller in a controller thread at 500 Hz."""
    sim, ctrl = _new_sim(w, fixture)
    probe = _Probe(sim, w)
    first = ctrl(sim.read_state())  # the first command before physics starts (as the board would have one)
    assert first is not None
    sim.write_cmd(first)
    sim.start_realtime(1.0)
    ct = ControllerThread(sim, ctrl, CONTROL_HZ, name="stand_hold" if fixture else "policy")
    ct.start()
    try:
        time.sleep(seconds)
    finally:
        ct.stop()
        sim.stop()
    st = sim.rt
    return RealtimeResult(
        seconds=st.wall_s,
        achieved_rtf=st.achieved_rtf,
        overruns=st.overruns,
        max_lag_ms=1000.0 * st.max_lag_s,
        rephases=st.rephases,
        controller_ticks=ct.ticks,
        controller_late=ct.late,
        check=probe.check(),
    )


def viewer(w: world.World, *, fixture: bool = False) -> None:
    """Interactive viewer (macOS: run under mjpython), lockstep stepping paced to real time in this thread."""
    import mujoco.viewer

    sim, ctrl = _new_sim(w, fixture)
    sim.add_controller(ctrl, 1.0 / CONTROL_HZ, "stand_hold" if fixture else "policy")
    with mujoco.viewer.launch_passive(sim.m, sim.d) as v:
        t0, s0 = time.perf_counter(), sim.time
        while v.is_running():
            target = s0 + (time.perf_counter() - t0)
            due = int((target - sim.time) / sim.dt)
            if due > 0:
                sim.step(min(due, 100))
            v.sync()
            time.sleep(1.0 / 120.0)


def _controller_rows(ctrl: Controller, fixture: bool) -> list[tuple[str, str]]:
    if fixture or not isinstance(ctrl, PolicyController):
        return [("controller", "session-1 stand-hold FIXTURE (stiff PD at the CoM-centred pose; balances nothing)")]
    pol = ctrl.policy
    return [
        ("controller", f"balance policy {pol.source} (sha256 {pol.sha256[:12]}...), 50 Hz in the control loop"),
        ("  policy steps / max |action|", f"{pol.policy_steps} / {ctrl.max_abs_action:.3f}"),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1sim.demo", description=__doc__.splitlines()[0])
    ap.add_argument("--profile", choices=("real", "train"), default="real", help="robot profile (default real)")
    ap.add_argument("--scene", default="table", help="r1sim.scenes: table (default), floor, or package.module:attr")
    ap.add_argument("--seconds", type=float, default=5.0, help="lockstep stand duration, simulated seconds")
    ap.add_argument("--realtime-seconds", type=float, default=3.0, help="real-time mode duration (0 = skip)")
    ap.add_argument("--fixture", action="store_true", help="the session-1 stand-hold fixture instead of the policy")
    ap.add_argument("--out", default="out/s2_demo.png", help="PNG path")
    ap.add_argument("--viewer", action="store_true", help="open the interactive viewer afterwards (mjpython)")
    a = ap.parse_args(argv)

    w = world.build_world(a.profile, scene=a.scene)
    what = "the session-1 stand-hold fixture" if a.fixture else "the balance policy"
    print(f"r1sim demo - {what}; profile {a.profile!r}, scene {a.scene!r}, mujoco {mujoco.__version__}")
    top, edge = getattr(w.scene, "table_top_z", None), getattr(w.scene, "table_edge_x", None)
    table = f"; table top {top:.3f} m, near edge {100 * edge:.1f} cm in front of the feet" if top and edge else ""
    print(
        f"  robot mass {w.index.total_mass:.3f} kg; physics dt {1000 * w.model.opt.timestep:g} ms; "
        f"driver {driver_for_profile(a.profile).kind!r}{table}"
    )

    sim, chk, rtf, ctrl = lockstep(w, a.seconds, fixture=a.fixture)
    views = (
        *render.DEFAULT_VIEWS,
        render.View("hands", lookat=(0.12, 0.0, 0.62), distance=1.1, azimuth=60.0, elevation=-8.0),
    )
    png = render.render_png(sim.m, sim.d, Path(a.out), views, width=480, height=360)
    # the other profile's lockstep speed with the same controller, for the record (short run, no PNG)
    other = "train" if a.profile == "real" else "real"
    _, chk_other, rtf_other, _ = lockstep(world.build_world(other, scene=a.scene, visual=False), 2.0, fixture=a.fixture)
    rt = realtime(w, a.realtime_seconds, fixture=a.fixture) if a.realtime_seconds > 0 else None

    rows = [
        *_controller_rows(ctrl, a.fixture),
        ("stands (lockstep)", "yes" if chk.stands else "NO"),
        (
            "  base tilt change",
            f"{chk.tilt_deg:.2f} deg after {a.seconds:g} s (now {chk.tilt_now_deg:.2f} deg from upright)",
        ),
        ("  pelvis height change", f"{1000 * chk.pelvis_dz_m:+.1f} mm"),
        ("  feet slip", f"{chk.feet_slip_mm:.2f} mm"),
        ("  scene objects moved", f"{chk.objects_moved_mm:.2f} mm (max)"),
        (f"real-time factor, lockstep ({a.profile})", f"{rtf:.1f}x (headless physics + controller, scene, 500 Hz)"),
        (
            f"real-time factor, lockstep ({other})",
            f"{rtf_other:.1f}x (2 s; stands: {'yes' if chk_other.stands else 'NO'}, "
            f"tilt change {chk_other.tilt_deg:.2f} deg)",
        ),
    ]
    if rt is not None:
        rows += [
            ("real-time mode, achieved", f"{rt.achieved_rtf:.3f}x over {rt.seconds:.1f} s wall"),
            ("  overruns / rephases / max lag", f"{rt.overruns} / {rt.rephases} / {rt.max_lag_ms:.1f} ms"),
            ("  controller ticks (late)", f"{rt.controller_ticks} ({rt.controller_late})"),
            ("  stands (real time)", f"{'yes' if rt.check.stands else 'NO'} (tilt change {rt.check.tilt_deg:.2f} deg)"),
        ]
    rows.append(("PNG", str(png.resolve())))
    width = max(len(k) for k, _ in rows)
    print()
    for k, v in rows:
        print(f"  {k:<{width}}  {v}")
    ok = chk.stands and (rt is None or rt.check.stands)
    print("\nRESULT:", f"the R1 stands in the scene under {what}" if ok else "FAILED: the robot did not stand")

    if a.viewer:
        viewer(w, fixture=a.fixture)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
