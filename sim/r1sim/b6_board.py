"""B6 (DESIGN §13.2): the motor board without LowCmd - HW1's board-hold test and the simulator's reproduction of it.

What the real board does (HW1 part 2, 11 Oct 2026, the board-hold test: r1rt SIGSTOPped in ON_STAND with the guard
observe-only, the robot hanging on its stand; the 20 / 100 / 500 ms pauses' lowstate recordings by r1d's lowstate
recorder; the board's own 1 kHz `tick` gives the timing, the receive clock only its offset):

* 20 / 100 ms pauses: nothing moves (<= 3e-5 rad) - the board HOLDS the last command;
* 500 ms pause: at ONE board tick, 383.8 ms after the SIGSTOP (all 19 loaded joints in the same tick), the holding
  torques collapse (L hip pitch tau_est -3.55 -> -2.54 -> -1.92 N m in 2 ms) and the joints start to move; from then
  on the loaded joints sink at a nearly constant speed (hips 0.11-0.17 rad/s) with tau_est = -kd_eff x dq (the motor
  DAMPS: a free fall would accelerate) - a damped release, not zero torque, until the next LowCmd (r1rt's SIGCONT).

The hold time from the last LowCmd: r1rt publishes at 500 Hz, so its last command left 0-2 ms before the SIGSTOP and
reached the board ~0.3-0.5 ms later; the tick clock is aligned to the receive clock by the fastest delivery (another
~0.3-0.5 ms): the board released 383-387 ms after its last LowCmd (one event; the 1 ms tick quantises it).

The model (models/driver.yaml, `DriverConfig.real()`): `cmd_timeout_s` after the last body command the board applies
kp 0, dq 0, tau_ff 0 and kd `damp_kd` (one value: the groups do not separate) until the next command (engine
`Simulation._service_inputs`). `simulate_board_hold` replays the board-hold test on the simulated stand (the DDS
bridge's hanging robot at HW1's attitude, lockstep, no DDS): the robot held at the board-hold pose at the core's
ON_STAND gains, the command stopped, the board's timeout and release, and the numbers `board_hold_reference()` measures
on the real recording. tests/validation/test_b6_board.py holds the B6 targets; `python -m r1sim.b6_board` prints the
table (docs/sim/B6_BOARD.md).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

from . import joints as J
from . import world
from .driver import DriverConfig
from .engine import LowCmd

FloatArray = npt.NDArray[np.float64]

# The real board-hold recording cut for the tests (tests/assets/hw1/board_hold_README.md says how).
BOARD_HOLD_EXCERPT = Path(__file__).resolve().parents[2] / "tests" / "assets" / "hw1" / "board_hold_500ms_excerpt.npz"
# The core's ON_STAND command during the board-hold test (HW1 part 2 S1b: legs 200 / 3, waist 300 / 5, arms upper_kp
# 60 60 45 45 30, head 50): the kd are the stiff set (joints.KD_STIFF) - they only shape the settle before the stop.
BOARD_HOLD_KP = np.array([200.0] * 12 + [300.0] * 2 + [60.0, 60.0, 45.0, 45.0, 30.0] * 2 + [50.0, 10.0])
# The joints whose board-hold motion is compared (loaded enough to move after the release: |tau0| >= 1 N m or knees).
COMPARED = ("left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee", "waist_roll", "left_shoulder_pitch",
            "left_shoulder_roll", "right_shoulder_pitch", "right_shoulder_roll", "left_elbow")  # fmt: skip
# The window after the release over which the sink speed is averaged (ms after the stop).
SINK_WINDOW_MS = (395.0, 495.0)

# The joints' friction the real profile does not identify yet: the shoulders keep issue #51's 2.5 N m in the model
# (session 8: the legs / waist carry the HW1 S4 refit, model.REAL_JOINT_FRICTIONLOSS_HW1), far above what the arms
# show - B3's friction band of the S5 holds is ~1 N m (the board-hold sink implies ~0.7 with kd 12.5; the replay's steps
# <= 0.05 at speed: Stribeck). N m; applied to both sides.
HW1_FRICTIONLOSS: dict[str, float] = {"shoulder_pitch": 1.0, "shoulder_roll": 1.0}


@dataclass(frozen=True)
class BoardHoldNumbers:
    """What B6 compares: the release time, each compared joint's motion at the end of the pause and its sink speed."""

    release_ms: float  # the first tick after the stop at which the loaded joints' torque collapses / they move
    motion_end_rad: dict[str, float]  # |q - q_stop| at the end of the 500 ms pause
    sink_rad_s: dict[str, float]  # mean dq over SINK_WINDOW_MS (signed)
    tau_ratio_kd: dict[str, float]  # median -tau_est / dq over the sink window (the board's damping, motor side)
    motion_100ms_rad: float  # the largest motion of any joint in the first 100 ms (the hold)

    def as_dict(self) -> dict[str, Any]:
        return dict(
            release_ms=round(self.release_ms, 2),
            motion_end_rad={k: round(v, 5) for k, v in self.motion_end_rad.items()},
            sink_rad_s={k: round(v, 4) for k, v in self.sink_rad_s.items()},
            tau_ratio_kd={k: round(v, 2) for k, v in self.tau_ratio_kd.items()},
            motion_100ms_rad=float(f"{self.motion_100ms_rad:.3g}"),
        )


def _numbers(rel_ms: FloatArray, q: FloatArray, dq: FloatArray, tau: FloatArray, end_ms: float) -> BoardHoldNumbers:
    pre = rel_ms < 0.0
    q0 = q[pre][-50:].mean(axis=0)
    tau0 = tau[pre][-50:].mean(axis=0)
    mot = np.abs(q - q0)
    loaded = [J.IDX[n] for n in COMPARED]
    # the release: the first sample after 300 ms at which a loaded joint's torque left its hold by > 0.25 N m
    late = rel_ms > 300.0
    rel_t = [rel_ms[late][np.argmax(np.abs(tau[late, j] - tau0[j]) > 0.25)] for j in loaded
             if np.any(np.abs(tau[late, j] - tau0[j]) > 0.25)]  # fmt: skip
    release = float(min(rel_t)) if rel_t else float("nan")
    k_end = int(np.searchsorted(rel_ms, end_ms))
    k_end = min(k_end, len(rel_ms) - 1)
    w = (rel_ms >= SINK_WINDOW_MS[0]) & (rel_ms <= SINK_WINDOW_MS[1])
    sink = {n: float(dq[w, J.IDX[n]].mean()) for n in COMPARED}
    ratio: dict[str, float] = {}
    for n in COMPARED:
        j = J.IDX[n]
        moving = w & (np.abs(dq[:, j]) > 0.02)
        ratio[n] = float(np.median(-tau[moving, j] / dq[moving, j])) if moving.sum() >= 10 else float("nan")
    hold = (rel_ms >= 0.0) & (rel_ms <= 100.0)
    return BoardHoldNumbers(
        release_ms=release,
        motion_end_rad={n: float(mot[k_end, J.IDX[n]]) for n in COMPARED},
        sink_rad_s=sink,
        tau_ratio_kd=ratio,
        motion_100ms_rad=float(mot[hold].max()),
    )


def load_board_hold_excerpt(path: str | Path = BOARD_HOLD_EXCERPT) -> dict[str, FloatArray]:
    """The real board-hold 500 ms recording (tests/assets/hw1): rel_ms (board tick time after the SIGSTOP), q / dq /
    tau_est (26 body joints, SDK order), rpy."""
    z = np.load(path)
    return {k: np.asarray(z[k], dtype=np.float64) for k in z.files}


def board_hold_reference(path: str | Path = BOARD_HOLD_EXCERPT) -> BoardHoldNumbers:
    """B6's numbers on the REAL recording."""
    z = load_board_hold_excerpt(path)
    return _numbers(z["rel_ms"], z["q"], z["dq"], z["tau_est"], float(z["actual_ms"]))


def board_hold_pose(path: str | Path = BOARD_HOLD_EXCERPT) -> FloatArray:
    """The measured pose at the stop (mean of the last 50 ticks before it)."""
    z = load_board_hold_excerpt(path)
    pre = z["rel_ms"] < 0.0
    return np.asarray(z["q"][pre][-50:].mean(axis=0), dtype=np.float64)


def _set_friction(m: Any, frictionloss: Mapping[str, float]) -> None:
    for side in ("left_", "right_", ""):
        for key, v in frictionloss.items():
            name = f"{side}{key}_joint"
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                m.dof_frictionloss[m.jnt_dofadr[jid]] = float(v)


def _no_hand_contacts(m: Any) -> None:
    """The board-hold arm pose (shoulder roll +-0.19, elbow 1.40) puts the simulated open hands' thumbs on the hip roll
    links (penetration ~0.5 mm; the real hands hung free in the board-hold test): a contact the model's hand / hip
    geometry makes, not the board - so the hands do not collide in the B6 replay (said in docs/sim/B6_BOARD.md)."""
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
        if "hand" in name:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0


def simulate_board_hold(
    pause_ms: float = 500.0,
    *,
    driver: DriverConfig | None = None,
    frictionloss: Mapping[str, float] | None = None,
    pose: Sequence[float] | None = None,
    hold_s: float = 1.5,
    after_ms: float = 150.0,
    stop: bool = True,
) -> dict[str, FloatArray]:
    """The board-hold test on the simulated stand (lockstep): the hanging robot (r1sim.dds_bridge's cradle stand, no DDS
    started) held at `pose` (default: the measured board-hold pose) at BOARD_HOLD_KP for `hold_s` (the targets corrected
    so the robot settles AT the pose), then NO command for `pause_ms` (the board's timeout acts: `driver`, default
    DriverConfig.real()), then the hold command again for `after_ms` (r1rt's SIGCONT; its blend is not modelled here).
    `frictionloss`: per joint key (both sides) on top of the real profile's own values, default HW1_FRICTIONLOSS (the
    shoulders). stop=False: the same run with the command never stopped (the baseline: the simulated arms still creep
    ~1e-4 rad/s against their friction after the settle, so "nothing moved" is judged against this run, not against the
    pose before the stop). Returns rel_ms (after the stop), q, dq, tau (the applied actuator torque)."""
    from .dds_bridge import HW1_STAND, DdsBridge, stand_extra

    drv = driver if driver is not None else DriverConfig.real()
    stand = HW1_STAND
    w = world.build_world("real", visual=False, with_scene=False, extra=stand_extra(stand))
    _set_friction(w.model, HW1_FRICTIONLOSS if frictionloss is None else frictionloss)
    _no_hand_contacts(w.model)
    br = DdsBridge(domain=0, scene="none", stance="hanging", builtin="stand_lock", sensors=False, w=w, stand=stand,
                   loopback=False, driver=drv)  # fmt: skip
    sim = br.sim
    target = np.asarray(pose if pose is not None else board_hold_pose(), dtype=np.float64)
    cmd = LowCmd()
    cmd.kp[:] = BOARD_HOLD_KP
    cmd.kd[:] = J.KD_STIFF
    q_cmd = target.copy()
    state: dict[str, Any] = {"cmd": True}

    def ctrl(_st: Any) -> LowCmd | None:
        if not state["cmd"]:
            return None
        c = cmd.copy()
        c.q[:] = q_cmd
        return c

    sim.add_controller(ctrl, 0.002, "board_hold")
    for _ in range(3):  # settle AT the pose: shift the targets by the steady error (gravity at the soft arm gains)
        sim.run_for(hold_s / 3.0)
        q_cmd += target - sim.read_state().q
    sim.run_for(0.3)
    dt = sim.dt
    rows: list[tuple[float, FloatArray, FloatArray, FloatArray]] = []
    t_stop = sim.time
    n_pre = round(0.1 / dt)

    def record(n: int) -> None:
        for _ in range(n):
            sim.step(1)
            st = sim.read_state()
            rows.append(((sim.time - t_stop) * 1e3, st.q.copy(), st.dq.copy(), st.tau_est.copy()))

    # 100 ms of the hold before the stop for the reference values, then the stop
    t_stop = sim.time + n_pre * dt
    record(n_pre)
    state["cmd"] = not stop
    record(round(pause_ms * 1e-3 / dt))
    state["cmd"] = True
    record(round(after_ms * 1e-3 / dt))
    rel = np.array([r[0] for r in rows])
    return dict(
        rel_ms=rel,
        q=np.array([r[1] for r in rows]),
        dq=np.array([r[2] for r in rows]),
        tau=np.array([r[3] for r in rows]),
        timeouts=np.array([sim.timeouts]),
    )


def hold_motion(pause_ms: float, **kw: Any) -> float:
    """The largest joint motion the stop causes during a `pause_ms` pause: max |q(stopped) - q(never stopped)| over
    the pause (both runs lockstep and deterministic, so the difference is the board's doing alone)."""
    a = simulate_board_hold(pause_ms, after_ms=0.0, **kw)
    b = simulate_board_hold(pause_ms, after_ms=0.0, stop=False, **kw)
    w = (a["rel_ms"] >= 0.0) & (a["rel_ms"] <= pause_ms)
    return float(np.abs(a["q"][w] - b["q"][w]).max())


def sim_numbers(run: Mapping[str, FloatArray], pause_ms: float = 500.0) -> BoardHoldNumbers:
    """B6's numbers on a simulate_board_hold run (tau = the simulated motor torque, the counterpart of tau_est)."""
    return _numbers(run["rel_ms"], run["q"], run["dq"], run["tau"], pause_ms)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1sim.b6_board", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="write the table (JSON) here")
    a = ap.parse_args(argv)
    real = board_hold_reference()
    out: dict[str, Any] = {"real": real.as_dict()}
    for ms in (20.0, 100.0):
        out[f"sim_{int(ms)}ms"] = {"max_motion_vs_unstopped_rad": float(f"{hold_motion(ms):.3g}")}
    run = simulate_board_hold(500.0)
    out["sim_500ms"] = {**sim_numbers(run, 500.0).as_dict(), "board_timeouts": int(run["timeouts"][0])}
    old = simulate_board_hold(
        500.0, driver=DriverConfig()
    )  # the board before session 8: holds the last command forever
    out["sim_500ms_hold_forever"] = sim_numbers(old, 500.0).as_dict()
    print(json.dumps(out, indent=1))
    if a.out is not None:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
