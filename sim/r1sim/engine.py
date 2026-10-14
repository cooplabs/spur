"""Physics engine of r1sim: physics independent of any controller (DESIGN.md §12.1, §12.4; requirement R-k).

v2's simulator stepped physics inside its control thread (r1mc2 runtime.py:270-271), so a stalled controller also froze
the world and the 9 Oct 2026 fall could not happen in simulation (V2_INVENTORY.md §6). Here physics owns its clock:

* **Lockstep** (`Simulation.step(n)`): a simulated clock; controllers registered with `add_controller` are called at
  their rates, in registration order, at step boundaries. Fully deterministic (bit-identical reruns).
* **Real time** (`Simulation.start_realtime()`): physics runs in its OWN thread, paced by the wall clock, and never
  waits for a controller. Controllers talk to it only through `write_cmd` (a latest-value mailbox) and `read_state` (a
  snapshot copy), e.g. from a `ControllerThread`. A controller that stalls stops nothing but itself: the board-like
  driver keeps applying the last command (zero-order hold), as the real board did on 9 Oct.

**The driver block** (board semantics, DESIGN.md §3.1 step 9): every physics step applies, per joint,
``tau = clip(kp * (q_des - q) + kd * (dq_des - dq) + tau_ff, -tau_max, tau_max)``. It is evaluated INSIDE MuJoCo: each
joint's `motor` actuator is turned into an affine actuator (gain 1, bias ``[0, -kp, -kd]``, control
``kp * q_des + kd * dq_des + tau_ff``, force limited to +-tau_max). Two reasons:
1. Physics can advance many steps per call (`mj_step(nstep=k)`) without Python in the loop. `mj_step` releases the
   GIL, so a Python thread that stepped one physics step per call would be starved by any CPU-busy Python thread
   (measured on the Mac, 9 Oct 2026: 20,000 steps of a small model took 0.16 s alone and 125 s next to one busy Python
   thread - the GIL convoy effect).
2. With the `implicitfast` integrator the kd term is integrated implicitly (stable for the 0.0005 armature fingers).
The same affine form is MuJoCo's `<position kp kv>` actuator, which the train-equivalent profile needs in session 2.

Before the first command every joint gets zero torque (kp = kd = tau_ff = 0).

**Session 2** (DESIGN §12.3 actuators/sensors, §12.6 faults) adds, all optional:
* the driver profile (`r1sim.driver.DriverConfig`): the law ("board", above, or "position" = mjlab's training
  <position> actuators: gain kp, ctrl q_des - the same forces for dq = tau_ff = 0), a command delay (a command takes
  effect at the first step boundary >= its write time + delay; lockstep controllers' commands too: both model the DDS
  path), a state delay (`read_state` returns the newest state published at or before t - delay; then every step is a
  boundary so every step's state is kept), and the board's command timeout (hold / damp / zero torque);
* a sensor model (`SensorHook`): called once per published state in the physics thread; its result is what
  `read_state` and the lockstep controllers see (noise, quantisation, pads, temperatures; the exact values in
  `LowState.truth`);
* a fault schedule (`r1sim.faults.FaultSchedule`): the engine applies controller_stall (lockstep: due controllers are
  not called; real time: `ControllerThread` blocks), cmd_gap, cmd_drop (at write time), state_delay, motor_error and
  hand_state_freeze (on the reported state, after the sensor model), and logs each application.
A command with a non-finite value is rejected whole (the last command stays; `rejected_cmds`): our choice, the real
board's reaction is unknown. With none of the above configured the engine runs exactly the S1 code path.

State timing: at every step boundary the simulator has run `mj_step1` for the current state (positions, velocities and
the position / velocity sensors such as the gyro are fresh); `mj_step2` completes the step after the command is
applied. `mj_step1` + `mj_step2` is bit-identical to `mj_step` (checked on this Mac, MuJoCo 3.5.0, implicitfast). The
accelerometer is an acceleration-stage sensor, so it lags one physics step.
"""

from __future__ import annotations

import collections
import copy
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import mujoco
import numpy as np
import numpy.typing as npt

from .driver import DriverConfig
from .faults import Fault, FaultSchedule

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

N_BODY = 26  # the R1 SDK's 26 body joints (r1sim.joints.JOINTS order)
N_HAND = 7  # Dex3-1 motors per hand (DDS order)
BOARD_TICK_HZ = 1000.0  # the R1 board's tick counter runs at ~1 kHz (V2_INVENTORY App. E §0 item 1)
AMBIENT_C = 30.0  # temperature reported when no thermal model runs (as v2's simulator: sim.py:425-430)


class JointMap(Protocol):
    """What the engine needs from a model index (r1sim.model.RobotIndex satisfies it).

    Body arrays are in SDK order (26,), with -1 where a joint is absent in the model (e.g. the head in the
    train-equivalent profile). Hand mappings are side -> (7,) in Dex3 DDS order; empty when the hands are rigid."""

    @property
    def present(self) -> npt.NDArray[Any]: ...
    @property
    def qadr(self) -> npt.NDArray[Any]: ...
    @property
    def vadr(self) -> npt.NDArray[Any]: ...
    @property
    def act(self) -> npt.NDArray[Any]: ...
    @property
    def tau_max(self) -> npt.NDArray[Any]: ...
    @property
    def hand_qadr(self) -> Mapping[str, npt.NDArray[Any]]: ...
    @property
    def hand_vadr(self) -> Mapping[str, npt.NDArray[Any]]: ...
    @property
    def hand_act(self) -> Mapping[str, npt.NDArray[Any]]: ...
    @property
    def hand_tau_max(self) -> npt.NDArray[Any]: ...
    @property
    def imu_site(self) -> int: ...


def _zeros(n: int) -> FloatArray:
    return np.zeros(n, dtype=np.float64)


@dataclass
class HandCmd:
    """One hand's low-level command (Dex3 DDS order), the same PD law as the body."""

    q: FloatArray = field(default_factory=lambda: _zeros(N_HAND))
    dq: FloatArray = field(default_factory=lambda: _zeros(N_HAND))
    kp: FloatArray = field(default_factory=lambda: _zeros(N_HAND))
    kd: FloatArray = field(default_factory=lambda: _zeros(N_HAND))
    tau_ff: FloatArray = field(default_factory=lambda: _zeros(N_HAND))

    def copy(self) -> HandCmd:
        return HandCmd(self.q.copy(), self.dq.copy(), self.kp.copy(), self.kd.copy(), self.tau_ff.copy())


@dataclass
class LowCmd:
    """The body's low-level command, like the board's LowCmd: per joint q, dq, kp, kd, tau_ff (26, SDK order).

    `hands` holds an optional command per side ("left" / "right"); a side that is absent keeps its last command."""

    q: FloatArray = field(default_factory=lambda: _zeros(N_BODY))
    dq: FloatArray = field(default_factory=lambda: _zeros(N_BODY))
    kp: FloatArray = field(default_factory=lambda: _zeros(N_BODY))
    kd: FloatArray = field(default_factory=lambda: _zeros(N_BODY))
    tau_ff: FloatArray = field(default_factory=lambda: _zeros(N_BODY))
    hands: dict[str, HandCmd] = field(default_factory=dict)

    def copy(self) -> LowCmd:
        return LowCmd(
            self.q.copy(),
            self.dq.copy(),
            self.kp.copy(),
            self.kd.copy(),
            self.tau_ff.copy(),
            {s: h.copy() for s, h in self.hands.items()},
        )


@dataclass(frozen=True)
class LowState:
    """A snapshot of the simulated robot, like the board's LowState (+ ground truth for tests).

    q / dq / tau_est are (26,) in SDK order (absent joints read 0); tau_est is the torque the driver applied during the
    last step. quat (w, x, y, z) is the IMU site's orientation in the world, gyro its angular velocity (site frame),
    acc its accelerometer (site frame, includes gravity; one physics step old)."""

    t: float
    tick: int
    q: FloatArray
    dq: FloatArray
    tau_est: FloatArray
    quat: FloatArray
    gyro: FloatArray
    acc: FloatArray
    hand_q: dict[str, FloatArray]
    hand_dq: dict[str, FloatArray]
    hand_tau: dict[str, FloatArray]
    base_pos: FloatArray  # ground truth: the free base's position (world)
    base_quat: FloatArray  # ground truth: the free base's orientation (w, x, y, z)
    seq: int  # snapshot sequence number (increases with every published state)
    # Session 2 (sensors, faults). The defaults mean "not modelled": ambient temperature, no error words, no pads.
    motor_temp: FloatArray = field(default_factory=lambda: np.full(N_BODY, AMBIENT_C))  # (26,) deg C
    motor_err: IntArray = field(default_factory=lambda: np.zeros(N_BODY, dtype=np.int_))  # (26,) board error words
    hand_temp: dict[str, FloatArray] = field(default_factory=dict)  # side -> (7,) deg C
    hand_pressure: dict[str, FloatArray] = field(default_factory=dict)  # side -> (9, 12) Dex3 pads x slots, raw units
    # Ground truth of every value a sensor model changed (keys "q", "dq", "tau_est", "quat", "gyro", "acc",
    # "hand_q", ...); empty when the sensors are off. Tests and the validation suite read this, controllers never do.
    truth: dict[str, Any] = field(default_factory=dict)


Controller = Callable[[LowState], LowCmd | None]


@dataclass
class _Scheduled:
    name: str
    fn: Controller
    period_steps: int
    calls: int = 0


@dataclass
class RealtimeStats:
    """Real-time mode counters (all wall-clock based)."""

    rate: float = 1.0
    wall_s: float = 0.0  # wall time since start_realtime
    sim_s: float = 0.0  # simulated time advanced since start_realtime
    steps: int = 0
    chunks: int = 0
    max_chunk_steps: int = 0
    overruns: int = 0  # chunks that hit the per-chunk step cap (physics could not keep up)
    max_lag_s: float = 0.0  # largest (target sim time - sim time) seen after a chunk
    rephases: int = 0  # times the clock was re-anchored because the lag exceeded rephase_s

    @property
    def achieved_rtf(self) -> float:
        """Simulated seconds per wall second since start (1.0 x rate when keeping up)."""
        return self.sim_s / self.wall_s if self.wall_s > 0 else 0.0


class SensorHook(Protocol):
    """The sensor-model interface the engine drives (r1sim.sensors.SensorModel implements it).

    `bind` once in `Simulation.__init__` (after the model copy and the driver set-up), `reset(seed)` in every
    `Simulation.reset`, `process` exactly once per published state, in order, in the physics thread, right after the
    step that produced it (`d` is the current data; `dt` = simulated time since the previous publish, 0.0 after a
    reset). What `process` returns is what `read_state()` and the lockstep controllers get."""

    def bind(self, m: Any, joint_map: JointMap) -> None: ...
    def reset(self, seed: int | None = None) -> None: ...
    def process(self, state: LowState, m: Any, d: Any, dt: float, faults: FaultSchedule | None) -> LowState: ...


def _finite(cmd: LowCmd, body: bool = True) -> bool:
    """Every value of the command is finite (one concatenation: ~2 us, vs ~14 us array by array); body=False: the
    hands only (a hand-only write's body arrays are not applied)."""
    arrays = [cmd.q, cmd.dq, cmd.kp, cmd.kd, cmd.tau_ff] if body else []
    for h in cmd.hands.values():
        arrays += [h.q, h.dq, h.kp, h.kd, h.tau_ff]
    return not arrays or bool(np.isfinite(np.concatenate(arrays)).all())


class Simulation:
    """One MuJoCo world whose physics does not depend on any controller (see the module docstring).

    The model is deep-copied: the driver block re-configures the joints' actuators (PD inside MuJoCo).

    Keyword arguments (session 2): `driver` (r1sim.driver.DriverConfig; None = the S1 board law, no delays), `sensors`
    (a SensorHook, e.g. r1sim.sensors.SensorModel; None = exact readings), `faults` (r1sim.faults.FaultSchedule; the
    engine applies controller_stall, cmd_gap, cmd_drop, state_delay, hand_state_freeze and motor_error). With all three
    at their defaults the engine runs the S1 code path unchanged (latest-value mailbox, lazy state snapshots)."""

    def __init__(
        self,
        model: Any,
        joint_map: JointMap,
        *,
        seed: int = 0,
        driver: DriverConfig | None = None,
        sensors: SensorHook | None = None,
        faults: FaultSchedule | None = None,
    ) -> None:
        self.m = copy.deepcopy(model)
        self.d = mujoco.MjData(self.m)
        self.seed = seed  # passed to the sensor model's reset (noise streams)
        self.driver = driver if driver is not None else DriverConfig()
        self.sensors = sensors
        self.faults = faults
        # The S1 fast path: no input path to model (delay / timeout), no per-state processing (sensors / faults).
        self._fast = self.driver.plain and sensors is None and faults is None
        self._position = self.driver.kind == "position"
        self.dt = float(self.m.opt.timestep)
        jm = joint_map

        present = np.asarray(jm.present, dtype=bool)
        self._present = present
        self._b_idx = np.flatnonzero(present)  # SDK indices present in the model
        self._b_q = np.asarray(jm.qadr, dtype=np.int64)[self._b_idx]
        self._b_v = np.asarray(jm.vadr, dtype=np.int64)[self._b_idx]
        self._b_a = np.asarray(jm.act, dtype=np.int64)[self._b_idx]
        self._b_tau = np.asarray(jm.tau_max, dtype=np.float64)[self._b_idx]
        self._hands: dict[str, tuple[IntArray, IntArray, IntArray]] = {}
        for side, qa in jm.hand_qadr.items():
            self._hands[side] = (
                np.asarray(qa, dtype=np.int64),
                np.asarray(jm.hand_vadr[side], dtype=np.int64),
                np.asarray(jm.hand_act[side], dtype=np.int64),
            )
        self._h_tau = np.asarray(jm.hand_tau_max, dtype=np.float64) if self._hands else _zeros(N_HAND)
        self._configure_driver()

        # IMU: the site and its sensors (r1.xml: imu_ang_vel = gyro, imu_lin_acc = accelerometer, at site "imu")
        self._imu_site = int(jm.imu_site)
        self._gyro_adr = self._sensor_adr("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO)
        self._acc_adr = self._sensor_adr("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER)
        free = [j for j in range(self.m.njnt) if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self._base_qadr = int(self.m.jnt_qposadr[free[0]]) if free else -1

        self._controllers: list[_Scheduled] = []
        self._steps = 0
        self._lock = threading.Lock()  # guards the mailbox / queue and the published state (tiny critical sections)
        self._pending: LowCmd | None = None
        self._pending_body = True  # False: the pending write carries hand commands only (session 8, B6)
        self._snap_seq = 0
        self._snap: tuple[Any, ...] = ()
        self.cmd_writes = 0
        self.write_log: collections.deque[tuple[float, float]] = collections.deque(maxlen=200_000)  # (wall, sim t)
        self.applied_cmds = 0
        self.rejected_cmds = 0  # commands with a non-finite value (rejected whole; the last command stays in force)
        self.timeouts = 0  # times the board's command timeout fired (driver.cmd_timeout_s)
        # Input path (slow path only): commands in arrival order with the sim time they take effect, and whether the
        # write carried a body command (True) or hand commands only (False: the Dex3 boards' path, session 8).
        self._queue: collections.deque[tuple[float, LowCmd, bool]] = collections.deque()
        self._last_rx_t: float | None = None  # sim time the board last received (applied) a command
        self._timed_out = False
        # Published states (slow path): (t, reported state); read_state() picks by the state delay.
        self._states: collections.deque[LowState] = collections.deque()
        self._last_pub_t: float | None = None
        self._frozen: dict[tuple[int, str], tuple[Any, ...]] = {}  # (fault index, side) -> the frozen hand reading
        self._fault_started: set[int] = set()  # state-side faults already recorded (index in faults.faults)
        if sensors is not None:
            sensors.bind(self.m, jm)

        self._rt_thread: threading.Thread | None = None
        self._rt_stop = threading.Event()
        self.rt = RealtimeStats()
        self.rephase_s = 0.25
        self.max_chunk_s = 0.05  # at most this much simulated time per chunk (bounds command latency)
        self.reset()

    # ------------------------------------------------------------------ set-up
    def _sensor_adr(self, name: str, kind: Any) -> int:
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0 or self.m.sensor_type[sid] != kind:
            return -1
        return int(self.m.sensor_adr[sid])

    def _configure_driver(self) -> None:
        """Turn each commanded joint's actuator (a <motor> or a <position>) into the driver's PD actuator: affine bias
        [0, -kp, -kd]; gain 1 with ctrl = kp q_des + kd dq_des + tau_ff ("board"), or gain kp with ctrl = q_des
        ("position", mjlab's <position>). No ctrl clamp (mjlab: ctrllimited false), force limited to +-tau_max."""
        acts = self._driver_acts()
        taus = np.concatenate([self._b_tau, *[self._h_tau for _ in self._hands]])
        for a, t in zip(acts, taus, strict=True):
            gear = self.m.actuator_gear[a]
            if gear[0] != 1.0 or np.any(gear[1:] != 0.0):
                raise ValueError(f"actuator {a}: the driver block needs gear (1, 0, ...), got {gear}")
            if self.m.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE:
                raise ValueError(f"actuator {a}: the driver block needs a direct (no dynamics) actuator")
            self.m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            self.m.actuator_gainprm[a, :] = 0.0
            self.m.actuator_gainprm[a, 0] = 0.0 if self._position else 1.0
            self.m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
            self.m.actuator_biasprm[a, :] = 0.0
            self.m.actuator_ctrllimited[a] = 0
            self.m.actuator_forcelimited[a] = 1
            self.m.actuator_forcerange[a, :] = (-float(t), float(t))

    def add_controller(self, fn: Controller, period_s: float, name: str = "") -> None:
        """Lockstep only: call `fn(state)` every round(period_s / dt) physics steps (at least 1), in registration
        order, at step boundaries from step 0. A returned LowCmd is applied before the step (after the driver's
        command delay, if any); None keeps the last."""
        k = max(1, round(period_s / self.dt))
        self._controllers.append(_Scheduled(name or f"controller{len(self._controllers)}", fn, k))

    @property
    def time(self) -> float:
        return float(self.d.time)

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def realtime_running(self) -> bool:
        return self._rt_thread is not None and self._rt_thread.is_alive()

    def controller_calls(self) -> dict[str, int]:
        return {c.name: c.calls for c in self._controllers}

    # ------------------------------------------------------------------ state and commands
    def reset(self, qpos: npt.ArrayLike | None = None) -> None:
        """Back to t = 0 with zero velocities, zero torque (no command) and an optional qpos."""
        self._check_not_rt("reset")
        mujoco.mj_resetData(self.m, self.d)
        if qpos is not None:
            self.d.qpos[:] = np.asarray(qpos, dtype=np.float64)
        acts = self._driver_acts()
        self.m.actuator_biasprm[acts, :] = 0.0
        if self._position:
            self.m.actuator_gainprm[acts, 0] = 0.0
        self.d.ctrl[:] = 0.0
        self._steps = 0
        for c in self._controllers:
            c.calls = 0
        with self._lock:
            self._pending = None
            self._pending_body = True
            self._queue.clear()
            self._states.clear()
        self._last_rx_t = None
        self._timed_out = False
        self._last_pub_t = None
        self._frozen.clear()
        self._fault_started.clear()
        if self.sensors is not None:
            self.sensors.reset(self.seed)
        mujoco.mj_step1(self.m, self.d)
        self._publish()

    def _driver_acts(self) -> IntArray:
        return np.concatenate([self._b_a, *[h[2] for h in self._hands.values()]]).astype(np.int64)

    def _check_position(self, cmd: LowCmd) -> None:
        """A <position> actuator has no velocity target or feed-forward: refuse a command that needs them."""
        i = self._b_idx
        if np.any(cmd.dq[i] != 0.0) or np.any(cmd.tau_ff[i] != 0.0):
            raise ValueError("driver kind 'position' (mjlab <position> actuators): dq and tau_ff must be 0")
        for side, h in cmd.hands.items():
            if side in self._hands and (np.any(h.dq != 0.0) or np.any(h.tau_ff != 0.0)):
                raise ValueError(f"driver kind 'position': hand {side} dq and tau_ff must be 0")

    def write_cmd(self, cmd: LowCmd, *, body: bool = True) -> None:
        """Thread-safe: the latest command wins; physics applies it at its next step boundary (after the driver's
        command delay, unless a cmd_gap / cmd_drop fault discards it).

        body=False (session 8, B6): the write carries the HAND commands only (`cmd.hands`; its body arrays are
        ignored). The body's board and the Dex3 boards are separate input paths on the robot (rt/lowcmd vs the hands'
        own topics): a hand command must neither re-apply the body command nor count as a body command for the board's
        timeout (driver.cmd_timeout_s - the board-hold test: the body board releases ~0.385 s after its last rt/lowcmd
        however busy the hands are)."""
        c = cmd.copy()
        if self._position:
            self._check_position(c)
        with self._lock:
            self.cmd_writes += 1
            t = float(self.d.time)
            self.write_log.append((time.perf_counter(), t))
            if self._fast:
                if not body and self._pending is not None and self._pending_body:
                    self._pending.hands.update(c.hands)  # a body command still pending keeps its body part
                else:
                    self._pending = c
                    self._pending_body = body
            else:
                self._enqueue(c, t, body=body)

    def _enqueue(self, cmd: LowCmd, t: float, *, body: bool = True) -> None:
        """Slow path, under the lock: a command enters the board's input path at sim time t."""
        f = self.faults
        if f is not None:
            if f.active("cmd_gap", t):
                f.record(t, "cmd_gap", "discard")
                return
            for fd in f.active("cmd_drop", t):
                if f.rng("cmd_drop").random() < float(fd.params.get("p", 0.5)):
                    f.record(t, "cmd_drop", "discard")
                    return
        self._queue.append((t + self.driver.cmd_delay_s, cmd, body))

    def read_state(self) -> LowState:
        """Thread-safe snapshot of the last published state (published at every step boundary); with a state delay
        (driver.state_delay_s, state_delay faults) the newest one published at or before t - delay."""
        if self._fast:
            with self._lock:
                snap = self._snap
            return self._state_from(*snap)
        with self._lock:
            return self._delayed_state()

    def _delayed_state(self) -> LowState:
        """Slow path, under the lock."""
        states = self._states
        newest = states[-1]
        delay = self._state_delay(newest.t)
        if delay <= 0.0:
            return newest
        cutoff = newest.t - delay + 1e-6 * self.dt
        for s in reversed(states):
            if s.t <= cutoff:
                return s
        return states[0]  # not that old yet (just after a reset): the oldest state there is

    def _state_delay(self, t: float) -> float:
        delay = self.driver.state_delay_s
        f = self.faults
        if f is not None:
            for fd in f.active("state_delay", t):
                delay += float(fd.params["delay_s"])
                self._note_start(fd, t, delay_s=float(fd.params["delay_s"]))
        return delay

    def _max_state_delay(self) -> float:
        extra = 0.0
        if self.faults is not None:
            extra = max([float(f.params["delay_s"]) for f in self.faults.faults if f.kind == "state_delay"] + [0.0])
        return self.driver.state_delay_s + extra

    def _note_start(self, fault: Fault, t: float, **detail: Any) -> None:
        """Record the first application of a state-side fault (once per scheduled fault)."""
        f = self.faults
        if f is None:
            return
        k = next(i for i, x in enumerate(f.faults) if x is fault)
        if k not in self._fault_started:
            self._fault_started.add(k)
            f.record(t, fault.kind, "apply", **detail)

    def _apply(self, cmd: LowCmd, *, body: bool = True) -> None:
        if not _finite(cmd, body):
            self.rejected_cmds += 1
            if self.faults is not None:
                self.faults.record(float(self.d.time), "rejected_cmd", "non_finite")
            return
        i = self._b_idx
        if body:
            kp, kd = cmd.kp[i], cmd.kd[i]
            self.m.actuator_biasprm[self._b_a, 1] = -kp
            self.m.actuator_biasprm[self._b_a, 2] = -kd
            if self._position:
                self.m.actuator_gainprm[self._b_a, 0] = kp
                self.d.ctrl[self._b_a] = cmd.q[i]
            else:
                self.d.ctrl[self._b_a] = kp * cmd.q[i] + kd * cmd.dq[i] + cmd.tau_ff[i]
        for side, h in cmd.hands.items():
            if side not in self._hands:
                continue
            a = self._hands[side][2]
            self.m.actuator_biasprm[a, 1] = -h.kp
            self.m.actuator_biasprm[a, 2] = -h.kd
            if self._position:
                self.m.actuator_gainprm[a, 0] = h.kp
                self.d.ctrl[a] = h.q
            else:
                self.d.ctrl[a] = h.kp * h.q + h.kd * h.dq + h.tau_ff
        self.applied_cmds += 1

    def _take_pending(self) -> tuple[LowCmd, bool] | None:
        with self._lock:
            c, self._pending = self._pending, None
            body, self._pending_body = self._pending_body, True
        return None if c is None else (c, body)

    def _service_inputs(self) -> None:
        """Slow path, at a step boundary: apply every queued command now due (in arrival order; a hand side a
        command leaves out keeps its last command), then the board's command timeout."""
        t = float(self.d.time)
        eps = 1e-6 * self.dt
        due: list[tuple[LowCmd, bool]] = []
        with self._lock:
            while self._queue and self._queue[0][0] <= t + eps:
                _, c0, b0 = self._queue.popleft()
                due.append((c0, b0))
        for c, body in due:
            before = self.rejected_cmds
            self._apply(c, body=body)
            if body and self.rejected_cmds == before:  # only a body command is a receive for the body's board
                self._last_rx_t = t
                self._timed_out = False
        to = self.driver.cmd_timeout_s
        if to is not None and self._last_rx_t is not None and not self._timed_out and t >= self._last_rx_t + to - eps:
            self._timed_out = True
            self.timeouts += 1
            action = self.driver.on_cmd_timeout
            if action != "hold":
                c = LowCmd()  # body only: the Dex3 hands have their own boards (and timeout bit)
                if action == "damp":
                    c.kd[:] = self.driver.damp_kd
                self._apply(c)

    def _steps_until(self, t_event: float) -> int:
        """Physics steps from now to the first step boundary at or after t_event (>= 1)."""
        n = math.ceil((t_event - float(self.d.time)) / self.dt - 1e-6)
        return max(1, n)

    def _publish(self) -> None:
        d = self.d
        t, qpos, qvel, sens, force = (
            float(d.time),
            d.qpos.copy(),
            d.qvel.copy(),
            d.sensordata.copy(),
            d.actuator_force.copy(),
        )
        xmat = d.site_xmat[self._imu_site].copy() if self._imu_site >= 0 else np.eye(3).ravel()
        if self._fast:
            with self._lock:
                self._snap_seq += 1
                self._snap = (t, qpos, qvel, sens, force, xmat, self._snap_seq)
            return
        with self._lock:
            self._snap_seq += 1
            seq = self._snap_seq
        state = self._state_from(t, qpos, qvel, sens, force, xmat, seq)
        dt = 0.0 if self._last_pub_t is None else t - self._last_pub_t
        self._last_pub_t = t
        if self.sensors is not None:
            state = self.sensors.process(state, self.m, d, dt, self.faults)
        if self.faults is not None:
            state = self._state_faults(state, t)
        with self._lock:
            self._snap = (t, qpos, qvel, sens, force, xmat, seq)
            self._states.append(state)
            keep = self._max_state_delay() + 2.0 * self.dt
            while len(self._states) > 2 and self._states[1].t <= t - keep:
                self._states.popleft()

    def _state_faults(self, state: LowState, t: float) -> LowState:
        """motor_error and hand_state_freeze on a reported state (after the sensor model)."""
        f = self.faults
        assert f is not None
        changes: dict[str, Any] = {}
        for fd in f.active("motor_error", t):
            err = (changes["motor_err"] if "motor_err" in changes else state.motor_err).copy()
            joints = np.asarray(fd.params.get("joints", np.arange(N_BODY)), dtype=np.int64)
            err[joints] = int(fd.params.get("word", 1))
            changes["motor_err"] = err
            self._note_start(fd, t, joints=joints.tolist(), word=int(fd.params.get("word", 1)))
        freezes = f.active("hand_state_freeze", t)
        if freezes:
            prev = self._states[-1] if self._states else state
            fields = ("hand_q", "hand_dq", "hand_tau", "hand_pressure", "hand_temp")
            new = {k: dict(getattr(state, k)) for k in fields}
            for fd in freezes:
                k = next(i for i, x in enumerate(f.faults) if x is fd)
                sides = [fd.params["side"]] if "side" in fd.params else list(state.hand_q)
                for side in sides:
                    key = (k, side)
                    if key not in self._frozen:  # the last reading before the window
                        self._frozen[key] = tuple(
                            None if getattr(prev, n).get(side) is None else getattr(prev, n)[side].copy()
                            for n in fields
                        )
                    for n, v in zip(fields, self._frozen[key], strict=True):
                        if v is not None:
                            new[n][side] = v.copy()
                self._note_start(fd, t, sides=sides)
            changes.update(new)
        return replace(state, **changes) if changes else state

    def _state_from(
        self,
        t: float,
        qpos: FloatArray,
        qvel: FloatArray,
        sens: FloatArray,
        force: FloatArray,
        imu_xmat: FloatArray,
        seq: int,
    ) -> LowState:
        q, dq, tau = _zeros(N_BODY), _zeros(N_BODY), _zeros(N_BODY)
        q[self._b_idx] = qpos[self._b_q]
        dq[self._b_idx] = qvel[self._b_v]
        tau[self._b_idx] = force[self._b_a]
        hq = {s: qpos[h[0]].copy() for s, h in self._hands.items()}
        hdq = {s: qvel[h[1]].copy() for s, h in self._hands.items()}
        htau = {s: force[h[2]].copy() for s, h in self._hands.items()}
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(imu_xmat, dtype=np.float64))
        rot = np.asarray(imu_xmat, dtype=np.float64).reshape(3, 3)
        gyro = sens[self._gyro_adr : self._gyro_adr + 3].copy() if self._gyro_adr >= 0 else _zeros(3)
        if self._acc_adr >= 0:
            acc = sens[self._acc_adr : self._acc_adr + 3].copy()
        else:  # no accelerometer in the model: the static reading (gravity only), in the site frame
            acc = rot.T @ np.array([0.0, 0.0, 9.81])
        if self._base_qadr >= 0:
            bp = qpos[self._base_qadr : self._base_qadr + 3].copy()
            bq = qpos[self._base_qadr + 3 : self._base_qadr + 7].copy()
        else:
            bp, bq = _zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        return LowState(
            t=t,
            tick=round(t * BOARD_TICK_HZ),
            q=q,
            dq=dq,
            tau_est=tau,
            quat=quat,
            gyro=gyro,
            acc=acc,
            hand_q=hq,
            hand_dq=hdq,
            hand_tau=htau,
            base_pos=bp,
            base_quat=bq,
            seq=seq,
        )

    # ------------------------------------------------------------------ lockstep
    def _check_not_rt(self, what: str) -> None:
        if self.realtime_running:
            raise RuntimeError(f"{what}: not allowed while the real-time thread runs (call stop() first)")

    def _steps_to_next_due(self) -> int:
        if not self._controllers:
            return 1 << 62
        return min(c.period_steps - (self._steps % c.period_steps) for c in self._controllers)

    def _steps_to_next_event(self) -> int:
        """Slow path: the next boundary something happens at - a controller, a queued command taking effect, the
        command timeout; every step while a state delay can apply (the ring then holds every step's state)."""
        k = self._steps_to_next_due()
        if self._max_state_delay() > 0.0:
            return 1
        with self._lock:
            nxt = self._queue[0][0] if self._queue else None
        if nxt is not None:
            k = min(k, self._steps_until(nxt))
        to = self.driver.cmd_timeout_s
        if to is not None and self._last_rx_t is not None and not self._timed_out:
            k = min(k, self._steps_until(self._last_rx_t + to))
        return k

    def _advance(self, k: int) -> None:
        """Complete the current step (mj_step2, the command is already applied), k - 1 more in C, then mj_step1 for
        the new state and publish it. Keeps the invariant: at a boundary, mj_step1 has run for the current state."""
        mujoco.mj_step2(self.m, self.d)
        if k > 1:
            mujoco.mj_step(self.m, self.d, k - 1)
        mujoco.mj_step1(self.m, self.d)
        self._steps += k
        self._publish()

    def _stalled(self, name: str, t: float) -> bool:
        """A controller_stall fault covers this lockstep controller now (`name` param: one controller; default all)."""
        f = self.faults
        if f is None:
            return False
        for fd in f.active("controller_stall", t):
            if fd.params.get("name") in (None, name):
                f.record(t, "controller_stall", "skip_call", name=name)
                return True
        return False

    def step(self, n: int = 1) -> None:
        """Lockstep: advance n physics steps, calling the registered controllers when due."""
        self._check_not_rt("step")
        if not self._fast:
            self._step_slow(int(n))
            return
        remaining = int(n)
        while remaining > 0:
            pending = self._take_pending()
            if pending is not None:
                self._apply(pending[0], body=pending[1])
            due = [c for c in self._controllers if self._steps % c.period_steps == 0]
            if due:
                with self._lock:
                    snap = self._snap
                state = self._state_from(*snap)
                for c in due:
                    cmd = c.fn(state)
                    c.calls += 1
                    if cmd is not None:
                        if self._position:
                            self._check_position(cmd)
                        self._apply(cmd)
            k = min(remaining, self._steps_to_next_due())
            self._advance(k)
            remaining -= k

    def _step_slow(self, n: int) -> None:
        """Lockstep with the input path (delay / timeout / faults) and processed states (sensors / faults)."""
        remaining = n
        while remaining > 0:
            self._service_inputs()
            t = float(self.d.time)
            due = [c for c in self._controllers if self._steps % c.period_steps == 0]
            if due:
                with self._lock:
                    state = self._delayed_state()
                submitted = False
                for c in due:
                    if self._stalled(c.name, t):
                        continue
                    cmd = c.fn(state)
                    c.calls += 1
                    if cmd is not None:
                        c2 = cmd.copy()
                        if self._position:
                            self._check_position(c2)
                        with self._lock:
                            self._enqueue(c2, t)
                        submitted = True
                if submitted:
                    self._service_inputs()  # zero delay: the command applies before this step, as in S1
            k = min(remaining, self._steps_to_next_event())
            self._advance(k)
            remaining -= k

    def run_for(self, seconds: float) -> None:
        """Lockstep: advance by round(seconds / dt) steps."""
        self.step(max(0, round(seconds / self.dt)))

    # ------------------------------------------------------------------ real time
    def start_realtime(self, rate: float = 1.0) -> None:
        """Run physics in its own thread, paced by the wall clock (sim time = rate x wall time). Lockstep
        controllers are NOT called in this mode; use ControllerThread (or any thread) with read_state / write_cmd."""
        self._check_not_rt("start_realtime")
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self._rt_stop.clear()
        self.rt = RealtimeStats(rate=rate)
        self._rt_thread = threading.Thread(target=self._rt_loop, args=(rate,), name="r1sim-physics", daemon=True)
        self._rt_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the real-time thread (no-op in lockstep)."""
        th = self._rt_thread
        if th is None:
            return
        self._rt_stop.set()
        th.join(timeout)
        if th.is_alive():
            raise RuntimeError("the physics thread did not stop")
        self._rt_thread = None

    def _rt_loop(self, rate: float) -> None:
        dt = self.dt
        cap = max(1, round(self.max_chunk_s / dt))
        st = self.rt
        wall0 = time.perf_counter()
        sim0 = float(self.d.time)
        start_wall, start_sim = wall0, sim0
        while not self._rt_stop.is_set():
            now = time.perf_counter()
            target = sim0 + rate * (now - wall0)
            due = int((target - float(self.d.time)) / dt + 1e-9)
            if due > 0:
                k = min(due, cap)
                if self._fast:
                    pending = self._take_pending()
                    if pending is not None:
                        self._apply(pending[0], body=pending[1])
                else:
                    self._service_inputs()
                    k = min(k, self._steps_to_next_event())
                self._advance(k)
                st.steps += k
                st.chunks += 1
                st.max_chunk_steps = max(st.max_chunk_steps, k)
                if due > cap:
                    st.overruns += 1
                lag = target - float(self.d.time)
                st.max_lag_s = max(st.max_lag_s, lag)
                if lag > self.rephase_s:  # cannot keep up: re-anchor (sim time falls behind wall time; counted)
                    wall0, sim0 = time.perf_counter(), float(self.d.time)
                    st.rephases += 1
            st.wall_s = time.perf_counter() - start_wall
            st.sim_s = float(self.d.time) - start_sim
            nxt = wall0 + (float(self.d.time) + dt - sim0) / rate
            slp = nxt - time.perf_counter()
            if slp > 0:
                time.sleep(min(slp, 0.002))


@dataclass
class StallRecord:
    wall_start: float
    wall_end: float
    sim_start: float
    sim_end: float
    busy: bool


class ControllerThread:
    """Runs a controller against a Simulation in its own thread, at a fixed rate (absolute deadlines).

    `stall(seconds)` makes the next tick block for that long before it reads or writes anything - the 9 Oct 2026
    stand-in (a thread stuck in a blocking call). With busy=True the tick spins in Python instead (a GIL hog)."""

    def __init__(self, sim: Simulation, fn: Controller, rate_hz: float, name: str = "controller") -> None:
        self.sim = sim
        self.fn = fn
        self.period = 1.0 / float(rate_hz)
        self.name = name
        self.ticks = 0
        self.writes = 0
        self.late = 0
        self.stalls: list[StallRecord] = []
        self._stall_req: tuple[float, bool] | None = None
        self._stall_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"r1sim-{self.name}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def stall(self, seconds: float, busy: bool = False) -> None:
        with self._stall_lock:
            self._stall_req = (float(seconds), busy)

    @property
    def stalled(self) -> bool:
        with self._stall_lock:
            return self._stall_req is not None

    def _do_stall(self, seconds: float, busy: bool) -> None:
        w0, s0 = time.perf_counter(), self.sim.time
        if busy:
            x = 0
            while time.perf_counter() - w0 < seconds:
                x += 1
        else:
            time.sleep(seconds)
        self.stalls.append(StallRecord(w0, time.perf_counter(), s0, self.sim.time, busy))

    def _fault_stall(self) -> bool:
        """Block while a controller_stall fault (all controllers, or `name` = this one) is active at sim time."""
        f = self.sim.faults
        if f is None:
            return False
        t = self.sim.time
        act = [x for x in f.active("controller_stall", t) if x.params.get("name") in (None, self.name)]
        if not act:
            return False
        end = max(x.t_end for x in act)
        f.record(t, "controller_stall", "block", name=self.name)
        w0, s0 = time.perf_counter(), t
        while self.sim.time < end and not self._stop.is_set():
            time.sleep(0.0005)
        self.stalls.append(StallRecord(w0, time.perf_counter(), s0, self.sim.time, False))
        return True

    def _loop(self) -> None:
        nxt = time.perf_counter()
        while not self._stop.is_set():
            with self._stall_lock:
                req = self._stall_req
            if req is not None:
                self._do_stall(*req)
                with self._stall_lock:
                    self._stall_req = None
                nxt = time.perf_counter()
            if self._fault_stall():
                nxt = time.perf_counter()
            cmd = self.fn(self.sim.read_state())
            self.ticks += 1
            if cmd is not None:
                self.sim.write_cmd(cmd)
                self.writes += 1
            nxt += self.period
            slp = nxt - time.perf_counter()
            if slp > 0:
                time.sleep(slp)
            elif slp < -self.period:
                self.late += 1
                nxt = time.perf_counter()
