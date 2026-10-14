"""What the examples share: the simulated world (a plain table with a box, a can and a ball on it), the two ways to run
the controller against it, and a progress printer.

The two ways to run (the same requests, the same r1d code, the same answers):

* `processes` (the default) - what runs on the robot: the DDS-bridge simulator stands in for the robot (rt/lowstate,
  rt/lowcmd, the Dex3-1 hands, the motion switcher), and the real `r1rt` (the real-time core), `r1guard` (the hot
  standby) and `r1d` (the API service) run as separate processes against it (`r1sim.stack.ProcStack`). Real time, so
  the numbers wobble when the machine is busy. Needs CycloneDDS (README.md, "What you need").
* `lockstep` - one process, one clock: the simulator, the same C++ core (`r1rt_py`, the code `r1rt` runs), r1d's
  service and the simulated hands stepped together (`r1sim.hands_rig.ServiceRig`). Deterministic for a seed, no DDS.

Both hand the example an `r1mc3.client.R1` (the typed SDK: one method per API op) - the example code cannot tell them
apart.
"""

from __future__ import annotations

import contextlib
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from r1mc3.client import R1
from r1sim import scenes, world

FloatArray = npt.NDArray[np.float64]

# ------------------------------------------------------------------------------------------------- the table scene
# The simulator's built-in "table" scene (r1sim.scenes): a plain table 0.6 x 1.2 m, its top 0.78 m above the floor, its
# near edge 0.25 m in front of the robot's feet, with a 5 cm box, a can and a ball loose on it. Positions are in the
# robot's base frame (x forward, y left, z up, the origin on the floor between the feet) - the frame r1d's API uses.
TABLE_TOP_Z = scenes.TABLE_TOP_Z_M


def table_world() -> world.World:
    """The robot (the real-equivalent profile) standing in front of the table."""
    return world.build_world("real", scene="table", visual=False)


def table_object(name: str) -> scenes.SceneObject:
    """One of the table's objects as the scene built it (shape, MuJoCo half sizes, start position)."""
    return scenes.add_table(mujoco.MjSpec()).object(name)


def table_entity() -> dict[str, Any]:
    """The table as r1d's world model takes it: a surface whose pose is the centre of its TOP face."""
    return next(e for e in scenes.table_entities() if e["id"] == "table")


def object_entity(name: str, xyz: FloatArray) -> dict[str, Any]:
    """An object on the table as r1d's world model takes it (its centre, its MuJoCo half sizes) - at `xyz`, where it
    is now (the scene's start position is only where it began)."""
    e = next(e for e in scenes.table_entities() if e["id"] == name)
    return {**e, "pose": {"pos": [round(float(v), 4) for v in xyz]}}


# ------------------------------------------------------------------------------------------------- running it
@dataclass
class Sim:
    """What an example gets: the SDK client, a way to let time pass, the simulator's truth (an object's position, a
    joint's angle, the body's tilt), the mode's name."""

    r1: R1
    wait: Callable[[float], None]
    body_pos: Callable[[str], FloatArray]
    joint_q: Callable[[str], float]
    tilt_deg: Callable[[], float]
    mode: str


class _LockstepTransport:
    """r1mc3.client.R1's transport over the lockstep rig: a request -> r1d's reply, the simulator stepped meanwhile."""

    def __init__(self, rig: Any) -> None:
        self.rig = rig
        self.timeout_s = 20.0

    def call(self, op: str, **kw: Any) -> dict[str, Any]:
        out: dict[str, Any] = self.rig.call({"op": op, **kw}, max_sim_s=float(kw.get("timeout_s", 30.0)) + 5.0)
        return out

    def close(self) -> None:
        pass


def _quat_tilt_deg(q: FloatArray) -> float:
    """The angle between the body's z axis and the vertical (deg) from a (w, x, y, z) quaternion."""
    _, x, y, _ = (float(v) for v in q)
    zz = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


def _world(table: bool) -> world.World:
    return table_world() if table else world.build_world("real", scene="floor", visual=False)


def _truth(w: world.World, data: Callable[[], Any]) -> tuple[Callable[[str], FloatArray], Callable[[str], float],
                                                              Callable[[], float]]:  # fmt: skip
    """The simulator's ground truth, read from `data()` (MuJoCo's MjData): a body's position, a joint's angle (rad,
    by the SDK's joint name, e.g. "left_elbow"), the pelvis's tilt (deg)."""
    from r1sim import joints as J

    def pos(name: str) -> FloatArray:
        return np.asarray(data().xpos[mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, name)]).copy()

    def joint(name: str) -> float:
        return float(data().qpos[int(w.index.qadr[J.IDX[name]])])

    def tilt() -> float:
        return _quat_tilt_deg(np.asarray(data().xquat[w.index.pelvis_body], dtype=np.float64))

    return pos, joint, tilt


@contextlib.contextmanager
def lockstep(*, seed: int = 0, table: bool = True) -> Iterator[Sim]:
    """The lockstep rig: the robot balancing from the start (the core enters BALANCE at t = 0), arms at the policy's
    stand pose; `table`: the table scene (else the bare floor)."""
    from r1sim.hands_rig import ServiceRig

    w = _world(table)
    rig = ServiceRig(w, arms=None, seed=seed)
    rig.run(1.0)  # the objects settle, the policy finds its stance
    pos, joint, tilt = _truth(w, lambda: rig.sim.d)
    yield Sim(R1(transport=_LockstepTransport(rig)), rig.run, pos, joint, tilt, "lockstep")


@contextlib.contextmanager
def processes(*, domain: int = 0, seed: int = 0, table: bool = True, workdir: Path | None = None) -> Iterator[Sim]:
    """The real processes on the DDS-bridge simulator: started, then the operator's takeover (the simulated robot's
    built-in controller lets go, r1rt balances); everything is stopped on the way out (the simulator simply stops -
    nothing is damped). `table`: the table scene (else the bare floor)."""
    from r1sim import stack

    if not stack.available():
        raise SystemExit(
            "the process stack needs CycloneDDS's Python binding and the DDS build of r1rt / r1guard "
            "(examples/README.md, 'What you need'); or run with --lockstep"
        )
    w = _world(table)
    tmp = tempfile.TemporaryDirectory(prefix="r1mc3_example_") if workdir is None else None
    wd = Path(tmp.name) if tmp is not None else Path(str(workdir))
    try:
        with stack.ProcStack(domain=domain, workdir=wd, bridge={"w": w, "seed": seed}) as st:
            r = R1(port=st.port, timeout_s=60.0)
            st.wait_mode("monitor", 30)
            r.call("takeover").raise_for_error()
            st.wait_mode("balance", 30)
            time.sleep(1.0)
            pos, joint, tilt = _truth(w, lambda: st.bridge.sim.d)
            try:
                yield Sim(r, time.sleep, pos, joint, tilt, "processes")
            finally:
                r.close()
    finally:
        if tmp is not None:
            tmp.cleanup()


def say(t0: float) -> Callable[..., None]:
    """A progress printer: seconds since `t0` (wall clock) in front of every line."""

    def _say(*words: object) -> None:
        print(f"[{time.monotonic() - t0:6.1f} s]", *words, flush=True)

    return _say
