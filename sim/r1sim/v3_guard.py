"""r1guard (C++, `r1rt_py.Guard`) next to the v3 core on ONE shared segment, in lockstep (DESIGN.md §2.1, §4.3
rung 2; docs/architecture/GUARD.md).

    from r1sim.v3_guard import V3GuardedController
    ctl = V3GuardedController(w.model, faults=sched)       # the core (writer 500 Hz + policy 50 Hz) + the guard (1 kHz)
    sim.add_controller(ctl, ctl.period_s, "v3_guarded")    # called every 1 ms - NOT under the name "v3_core" (below)

Per call (= one guard cycle, 1 kHz): every second call is the core's writer tick (+ its policy thread on its schedule)
exactly as `V3CoreController` does it, unless a fault `controller_stall name="v3_core"` covers the time: then the
whole r1rt process is stalled - writer AND policy thread, no heartbeat, no mirror (the 9 Oct 4.018 s stall moved into
r1rt itself, DESIGN §13.1 A7 v3 part). Then the guard's cycle, with its own copy of the state (on the robot its own DDS
reader; `controller_stall name="v3_guard_lowstate"` withholds it). The guard reads the core's heartbeat and mirror
from the segment, keeps its shadow policy warm, and takes the ownership token by CAS when the heartbeat is older than
5 writer periods. At most ONE of them may publish in a call - `two_writers` counts violations (the core checks the
token before every write, so after the takeover it publishes nothing). The published command goes to the robot; None
(the board holds the last command) when neither publishes.

Note: the engine applies `controller_stall` faults to a whole registered controller by its registration name, so this
controller must not be registered as "v3_core" (that would stall the guard too). Faults on the guard's own network
output: `ctl.guard.inject(INJECT_POLICY_NAN, t0_ns, t1_ns)` (the parent's `policy_nan` faults go into the core).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import dex3
from .engine import HandCmd, LowCmd, LowState
from .faults import FaultSchedule
from .v3_core import V3CoreController

FloatArray = npt.NDArray[np.float64]

GUARD_HZ = 1000  # DESIGN §2.1: r1guard checks the heartbeat at 1 kHz
OWNER_NONE, OWNER_R1RT, OWNER_R1GUARD = 0, 1, 2  # types.h r1rt_owner
MODE_GUARDED = 4
LEG_JOINTS = slice(0, 12)


def token_owner(word: int) -> int:
    """The owner byte of an ownership-token word (shm_layout.h OwnershipToken::owner_of)."""
    return int(word) & 0xFF


def _state_dict(st: LowState, t_ns: int) -> dict[str, Any]:
    return {
        "t_rx_ns": t_ns,
        "board_tick": int(st.tick),
        "q": st.q,
        "dq": st.dq,
        "tau_est": st.tau_est,
        "quat": st.quat,
        "gyro": st.gyro,
        "acc": st.acc,
        "motor_state": np.asarray(st.motor_err, dtype=np.int64),
    }


class V3GuardedController(V3CoreController):
    """The core and r1guard as one lockstep simulator controller `(LowState) -> LowCmd | None` (module docstring).

    `period_s` is the guard's period (register the controller with it). `guard_params`: r1rt_py.Guard params
    (stale_periods, takeover_blend_s, takeover_enabled). Records per call in `gtrace` when `trace_guard=True`:
    (t, core_tick, core_published, guard_published, guard_active, min leg kp of the command sent (NaN: none sent))."""

    def __init__(
        self,
        model: Any,
        policy_dir: str | Path | None = None,
        *,
        writer_hz: int = 500,
        guard_hz: int = GUARD_HZ,
        guard_params: Mapping[str, Any] | None = None,
        faults: FaultSchedule | None = None,
        trace_guard: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, policy_dir, writer_hz=writer_hz, faults=faults, **kwargs)
        if guard_hz % writer_hz != 0:
            raise ValueError(f"guard_hz {guard_hz} must be a multiple of writer_hz {writer_hz} (lockstep order)")
        params = {"cycle_hz": int(guard_hz), **dict(guard_params or {})}
        self.guard: Any = self.mod.Guard(self.config, self.core, params)
        self.writer_period_s = 1.0 / writer_hz
        self.period_s = 1.0 / guard_hz
        self._core_every = guard_hz // writer_hz
        self._gcalls = 0
        self.two_writers = 0
        self.core_writes_while_guarded = 0  # core commands published while the guard held the token (must be 0)
        self.last_core_tick_t: float | None = None
        self.trace_guard = trace_guard
        self.gtrace: list[tuple[float, int, int, int, int, float]] = []

    def __call__(self, st: LowState) -> LowCmd | None:
        t = float(st.t)
        t_ns = round(t * 1e9)
        core_tick = self._gcalls % self._core_every == 0
        self._gcalls += 1
        core_cmd: LowCmd | None = None
        core_pub = 0
        if core_tick and not self._stalled("v3_core", t):
            core_cmd = super().__call__(st)
            core_pub = int(core_cmd is not None)
            self.last_core_tick_t = t
        gstate = None if self._stalled("v3_guard_lowstate", t) else _state_dict(st, t_ns)
        gpub, q, dq, kp, kd, tau = self.guard.cycle(t_ns, gstate)
        gactive = int(self.guard.active)
        if core_pub and gpub:
            self.two_writers += 1
        if core_pub and gactive:
            self.core_writes_while_guarded += 1
        out: LowCmd | None = core_cmd
        if gpub:
            out = self._lowcmd(q, dq, kp, kd, tau)
        if self.trace_guard:
            kmin = float(np.min(out.kp[LEG_JOINTS])) if out is not None else math.nan
            self.gtrace.append((t, int(core_tick), core_pub, int(gpub), gactive, kmin))
        return out

    def _lowcmd(self, q: Any, dq: Any, kp: Any, kd: Any, tau: Any) -> LowCmd:
        cmd = LowCmd()
        cmd.q[:] = q
        cmd.dq[:] = dq
        cmd.kp[:] = kp
        cmd.kd[:] = kd
        cmd.tau_ff[:] = tau
        for side in self._hands:  # the hands as the core's controller holds them (r1hands' business)
            h = HandCmd()
            h.q[:] = dex3.preset("open", side)
            h.kp[:] = dex3.KP
            h.kd[:] = dex3.KD
            cmd.hands[side] = h
        return cmd

    def guard_status(self) -> dict[str, Any]:
        s: dict[str, Any] = self.guard.status()
        return s

    def forensics(self) -> dict[str, Any]:
        f: dict[str, Any] = self.guard.forensics()
        return f

    def trace_array(self) -> FloatArray:
        """gtrace as an (n, 6) float array (columns as in the class docstring)."""
        return np.asarray(self.gtrace, dtype=np.float64).reshape(-1, 6)
