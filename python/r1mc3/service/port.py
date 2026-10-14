"""How r1d reaches the real-time core: the shared segment (DESIGN.md §2 principle 3), two ways.

* `ShmPort`: r1rt's POSIX segment through `r1rt_py.SegmentView` (the robot; the DDS-bridge rehearsal). Every write
  goes through the binding's C++ ring / seqlock / mailbox code (acquire / release order; a Python mmap view would
  have no barriers - r1mc3.log._native explains why).
* `LockstepPort`: an `r1rt_py.Core` stepped by the simulator (r1sim.v3_core) - tests and rehearsals run the same
  service code against the real core in simulated time.

Both expose the same small surface (`Port`): the clock, the heartbeats, the token, the stop channel and its epoch, the
upper_cmd ring + keepalives, the mode-request mailbox (session 5) and the core's last commanded / measured joints.
What a binding lacks is reported by `missing()` (the service answers UNSUPPORTED for the ops that need it), never
emulated: r1d must not write the segment by any path other than the core's own C++ code.

Time: r1d stamps records with CLOCK_MONOTONIC (`time.clock_gettime_ns(time.CLOCK_MONOTONIC)`), the clock the core
compares TTLs and keepalives against on the robot (DESIGN §3.1); in lockstep the simulator's clock.
"""

from __future__ import annotations

import contextlib
import os
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .codes import UPPER_GROUPS, CoreTables

FloatArray = npt.NDArray[np.float64]
R1RT_OK = 0

# r1rt_heartbeat (types.h; r1mc3.log.records.HEARTBEAT is the checked dtype)
_HB = np.dtype(
    [
        ("seq", "<u8"),
        ("t_ns", "<i8"),
        ("lowstate_tick", "<u8"),
        ("rate_hz", "<u4"),
        ("faults", "<u4"),
        ("blend_progress", "<f4"),
        ("mode", "<u2"),
        ("rt_granted", "u1"),
        ("_pad0", "u1"),
    ]
)
assert _HB.itemsize == 40


@dataclass(frozen=True)
class Heartbeat:
    seq: int
    t_ns: int
    lowstate_tick: int
    rate_hz: int
    faults: int
    blend_progress: float
    mode: int
    rt_granted: bool

    @staticmethod
    def parse(raw: Any) -> Heartbeat | None:
        """From SegmentView.rt_heartbeat()'s (writes, bytes) or None."""
        if raw is None:
            return None
        writes, data = raw
        if int(writes) == 0 or len(data) < _HB.itemsize:
            return None
        r = np.frombuffer(bytes(data)[: _HB.itemsize], dtype=_HB)[0]
        return Heartbeat(
            seq=int(r["seq"]),
            t_ns=int(r["t_ns"]),
            lowstate_tick=int(r["lowstate_tick"]),
            rate_hz=int(r["rate_hz"]),
            faults=int(r["faults"]),
            blend_progress=float(r["blend_progress"]),
            mode=int(r["mode"]),
            rt_granted=bool(r["rt_granted"]),
        )


class Port(Protocol):
    tables: CoreTables

    def now_ns(self) -> int: ...
    def heartbeat(self) -> Heartbeat | None: ...
    def guard_heartbeat(self) -> Heartbeat | None: ...
    def token(self) -> tuple[int, int]: ...
    def request_stop(self, level: int, requester: int, t_ns: int) -> None: ...
    def stop_epoch(self) -> int: ...
    def push_upper(self, rec: Mapping[str, Any]) -> int: ...
    def upper_keepalive(self, group: int, cmd_id: int, t_ns: int) -> int: ...
    def request_mode(self, req: int, requester: int, t_ns: int) -> int: ...
    def mode_reply(self, seq: int) -> tuple[int, int, int] | None: ...
    def targets(self) -> FloatArray | None: ...
    def measured(self) -> FloatArray | None: ...
    def missing(self) -> list[str]: ...

    # session 9 (L2 goals), optional - a port without them answers None:
    #   state() -> the core's latest lowstate as a dict (q, dq, tau_est, imu_quat, ...) for the measured results
    #   upper_groups() -> {"source": [3], "event": [3], "cmd_id": [3], "rejected": [1], "last_reject": [1]} per upper
    #     group (the executor's own status; ShmPort adds "t_ns": [r1rt's publish time], session 10 layout 4)


# r1rt's executor status older than this = r1rt is not ticking (stopped / stalled; the guard's own threshold is 10 ms,
# r1d's "alive" 100 ms): its last per-group events would describe the moment it stopped, not now
EXEC_STATUS_STALE_NS = 100_000_000


def monotonic_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


# SegmentHeader (shm_layout.h): magic u64 @0, layout u32 @8, abi u32 @12, size u64 @16, created_ns i64 @24,
# creator_pid i32 @32 - written ONCE by r1rt before `ready` is released, read-only afterwards.
_HDR_PID_OFFSET = 32
_MAGIC = b"R1MC3SHM"


def read_creator_pid(name: str) -> int | None:
    """r1rt's pid from its segment header (a read-only mapping of the first bytes; the field is written once before
    the segment is marked ready, so a plain read is enough - no seqlock needed). None when it cannot be read."""
    try:
        import mmap
        from _posixshmem import shm_open  # type: ignore[import-not-found]

        fd = shm_open(name, os.O_RDONLY, 0)
    except (ImportError, OSError):
        return None
    try:
        with mmap.mmap(fd, 64, mmap.MAP_SHARED, mmap.PROT_READ) as mm:
            magic = bytes(mm[0:8])
            if magic not in (_MAGIC, _MAGIC[::-1]):
                return None
            pid = int(struct.unpack_from("<i", mm, _HDR_PID_OFFSET)[0])
    except (OSError, ValueError):
        return None
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
    return pid if pid > 0 else None


class ShmPort:
    """r1rt's segment through `r1rt_py.SegmentView` (attach: header checked by the C++ side)."""

    NEEDS = ("push_upper", "upper_keepalive", "stop_epoch", "request_mode", "mode_reply")

    def __init__(self, view: Any, mod: ModuleType | Any) -> None:
        self.view = view
        self.mod = mod
        self.tables = CoreTables(mod)

    @classmethod
    def attach(cls, name: str, *, wait_s: float = 10.0, mod: ModuleType | Any | None = None) -> ShmPort:
        """Attach to r1rt's segment, waiting (bounded) for r1rt to create it."""
        if mod is None:
            from r1mc3.log._native import native

            mod = native()
        t_end = time.monotonic() + wait_s
        while True:
            try:
                return cls(mod.SegmentView.attach(name), mod)
            except RuntimeError:
                if time.monotonic() >= t_end:
                    raise
                time.sleep(0.2)

    def missing(self) -> list[str]:
        return [n for n in self.NEEDS if not hasattr(self.view, n)]

    def _need(self, name: str) -> Callable[..., Any]:
        fn = getattr(self.view, name, None)
        if fn is None:
            raise NotImplementedError(f"r1rt_py.SegmentView.{name} (the session-5 binding) is not in this build")
        return fn  # type: ignore[no-any-return]

    def now_ns(self) -> int:
        return monotonic_ns()

    def heartbeat(self) -> Heartbeat | None:
        return Heartbeat.parse(self.view.rt_heartbeat())

    def guard_heartbeat(self) -> Heartbeat | None:
        return Heartbeat.parse(self.view.guard_heartbeat())

    def token(self) -> tuple[int, int]:
        o, g = self.view.token()
        return int(o), int(g)

    def request_stop(self, level: int, requester: int, t_ns: int) -> None:
        self.view.request_stop(level, requester, t_ns)

    def stop_epoch(self) -> int:
        return int(self._need("stop_epoch")())

    def push_upper(self, rec: Mapping[str, Any]) -> int:
        return int(self._need("push_upper")(dict(rec)))

    def upper_keepalive(self, group: int, cmd_id: int, t_ns: int) -> int:
        return int(self._need("upper_keepalive")(group, cmd_id, t_ns))

    def request_mode(self, req: int, requester: int, t_ns: int) -> int:
        return int(self._need("request_mode")(req, requester, t_ns))

    def mode_reply(self, seq: int) -> tuple[int, int, int] | None:
        r = self._need("mode_reply")(seq)
        return None if r is None else (int(r[0]), int(r[1]), int(r[2]))

    def _vec(self, name: str, field: str) -> FloatArray | None:
        fn = getattr(self.view, name, None)
        if fn is None:
            return None
        d = fn()
        if d is None:
            return None
        if isinstance(d, Mapping):
            v = d.get(field)
            return None if v is None else np.asarray(v, dtype=np.float64)
        return None

    def targets(self) -> FloatArray | None:
        """The core's last commanded q (the guard mirror's q_target), when the binding exposes it."""
        return self._vec("mirror", "q_target")

    def measured(self) -> FloatArray | None:
        return self._vec("state", "q")

    def state(self) -> dict[str, Any] | None:
        """The core's latest lowstate (SegmentView.state(): q, dq, tau_est, imu_quat, tilt_rad, ...)."""
        fn = getattr(self.view, "state", None)
        if fn is None:
            return None
        d = fn()
        return None if d is None else dict(d)

    def exec_status(self) -> dict[str, Any] | None:
        """r1rt's executor status (layout 4, session 10: `SegmentView.exec_status()`), None when this binding predates
        it or r1rt never published it."""
        fn = getattr(self.view, "exec_status", None)
        if not callable(fn):
            return None
        d = fn()
        return None if d is None else dict(d)

    def upper_groups(self) -> dict[str, list[int]] | None:
        """The executor's per-group source / last event / cmd_id (+ the refusal count / reason) from r1rt's segment
        (layout 4, session 10) - the same shape as LockstepPort's, so goals.py ends a goal on the executor's DONE and
        names the guard that stopped it. None when the binding predates layout 4 or r1rt's status is older than
        EXEC_STATUS_STALE_NS (r1rt stopped / stalled: its last status would hide that) - r1d then judges the goal from
        the mirror's commanded target, as before."""
        d = self.exec_status()
        if d is None or monotonic_ns() - int(d["t_ns"]) > EXEC_STATUS_STALE_NS:
            return None
        n = len(UPPER_GROUPS)
        return {
            "source": [int(x) for x in d["source"][:n]],
            "event": [int(x) for x in d["event"][:n]],
            "cmd_id": [int(x) for x in d["cmd_id"][:n]],
            "rejected": [int(d["upper_rejected"])],
            "last_reject": [int(d["last_reject"])],
            "t_ns": [int(d["t_ns"])],  # when r1rt published it (goals.py: how long the core has had a record)
        }

    def set_base_cmd(self, seq: int, t_ns: int, ttl_ns: int, vx: float, vy: float, wz: float, lean: float) -> int:
        """The policy's base command (vx, vy, wz, lean; TTL <= 0.5 s, expiry = zero) into r1rt's segment
        (`SegmentView.set_base_cmd`, session 10; r1d is its single writer). The core zeroes an expired / non-finite
        command and keeps a lean outside the policy's trained range out (PolicyRunner.set_lean_target)."""
        fn = getattr(self.view, "set_base_cmd", None)
        if fn is None:
            raise NotImplementedError("r1rt_py.SegmentView.set_base_cmd is not in this build (session 10)")
        return int(fn(seq, t_ns, ttl_ns, vx, vy, wz, lean))

    @property
    def shm_name(self) -> str:
        return str(getattr(self.view, "name", "") or "")

    def rt_pid(self) -> int | None:
        """r1rt's pid (the segment's creator): `SegmentView.creator_pid()` (ABI 6), else the header read here."""
        fn = getattr(self.view, "creator_pid", None)
        if callable(fn):
            with contextlib.suppress(Exception):
                pid = int(fn())
                if pid > 0:
                    return pid
        return read_creator_pid(self.shm_name) if self.shm_name else None

    def guard_info(self) -> dict[str, Any] | None:
        return guard_info_of(self.view)

    def rt_info(self) -> dict[str, Any] | None:
        return rt_info_of(self.view)

    def guard_armed(self) -> bool | None:
        """Whether r1guard would take over (True) or only observes (False); None when this build cannot tell."""
        info = guard_info_of(self.view)
        return None if info is None or "armed" not in info else bool(info["armed"])


def rt_info_of(view: Any) -> dict[str, Any] | None:
    """r1rt's start-time facts through the binding (session 6, `SegmentView.rt_info()`: e.g. `stand_session`), None
    when the binding predates it or r1rt has not published them."""
    fn = getattr(view, "rt_info", None)
    if not callable(fn):
        return None
    with contextlib.suppress(Exception):
        d = fn()
        return None if d is None else dict(d)
    return None


def guard_info_of(view: Any) -> dict[str, Any] | None:
    """The guard heartbeat decoded by the binding (ABI 6, `SegmentView.guard_info()`: armed / active / damp /
    not_owner / mode_from / flags), None when the binding predates it or the guard never beat."""
    fn = getattr(view, "guard_info", None)
    if not callable(fn):
        return None
    with contextlib.suppress(Exception):
        d = fn()
        return None if d is None else dict(d)
    return None


class LockstepPort:
    """An `r1rt_py.Core` in the simulator's lockstep (r1sim.v3_core.V3CoreController's `core`).

    `clock_ns` returns the simulated time of the writer tick being prepared; `targets_fn` / `measured_fn` give the
    last published command / the last state (the lockstep driver has them). The mailbox: through the core's segment
    (`SegmentView.of_core`) when the binding has it; otherwise - an ABI-4 core - the request is applied directly with
    `Core.request` (legal in lockstep: one thread calls everything) and its reply recorded at once."""

    def __init__(
        self,
        core: Any,
        mod: ModuleType | Any,
        clock_ns: Callable[[], int],
        *,
        targets_fn: Callable[[], FloatArray | None] | None = None,
        measured_fn: Callable[[], FloatArray | None] | None = None,
    ) -> None:
        self.core = core
        self.mod = mod
        self.tables = CoreTables(mod)
        self._clock = clock_ns
        self._targets = targets_fn
        self._measured = measured_fn
        self.view = mod.SegmentView.of_core(core)
        self._direct: dict[int, tuple[int, int, int]] = {}
        self._seq = 0

    def missing(self) -> list[str]:
        return []

    def now_ns(self) -> int:
        return int(self._clock())

    def heartbeat(self) -> Heartbeat | None:
        return Heartbeat.parse(self.view.rt_heartbeat())

    def guard_heartbeat(self) -> Heartbeat | None:
        return Heartbeat.parse(self.view.guard_heartbeat())

    def token(self) -> tuple[int, int]:
        o, g = self.view.token()
        return int(o), int(g)

    def request_stop(self, level: int, requester: int, t_ns: int) -> None:
        self.core.stop_channel(level, requester, t_ns)

    def stop_epoch(self) -> int:
        return int(self.core.stop_epoch())

    def push_upper(self, rec: Mapping[str, Any]) -> int:
        return int(self.core.push_upper(dict(rec)))

    def upper_keepalive(self, group: int, cmd_id: int, t_ns: int) -> int:
        return int(self.core.upper_keepalive(group, cmd_id, t_ns))

    def request_mode(self, req: int, requester: int, t_ns: int) -> int:
        fn = getattr(self.view, "request_mode", None)
        if fn is not None:
            return int(fn(req, requester, t_ns))
        self._seq += 1
        rc = int(self.core.request(req, t_ns))
        st = self.core.status()
        self._direct[self._seq] = (rc, int(st.get("takeover_refusal", 0)), int(self.core.mode))
        return self._seq

    def mode_reply(self, seq: int) -> tuple[int, int, int] | None:
        fn = getattr(self.view, "mode_reply", None)
        if fn is not None:
            r = fn(seq)
            return None if r is None else (int(r[0]), int(r[1]), int(r[2]))
        return self._direct.get(seq)

    def targets(self) -> FloatArray | None:
        return None if self._targets is None else self._targets()

    def measured(self) -> FloatArray | None:
        return None if self._measured is None else self._measured()

    def state(self) -> dict[str, Any] | None:
        """The core's latest lowstate through its segment (the same SegmentView.state() r1d reads on the robot)."""
        fn = getattr(self.view, "state", None)
        if fn is None:
            return None
        d = fn()
        return None if d is None else dict(d)

    def upper_groups(self) -> dict[str, list[int]] | None:
        """The executor's per-group source / last event / cmd_id from the lockstep core's status."""
        st = self.core.status()
        if "upper_source" not in st:
            return None
        return {
            "source": [int(x) for x in st["upper_source"]],
            "event": [int(x) for x in st["upper_event"]],
            "cmd_id": [int(x) for x in st["upper_cmd_id"]],
            "rejected": [int(st.get("upper_rejected", 0))],
            "last_reject": [int(st.get("upper_last_reject", 0))],
        }

    def set_base_cmd(self, seq: int, t_ns: int, ttl_ns: int, vx: float, vy: float, wz: float, lean: float) -> int:
        """The policy's base command into the lockstep core (Core.set_base_cmd)."""
        return int(self.core.set_base_cmd(seq, t_ns, ttl_ns, vx, vy, wz, lean))

    shm_name = ""

    def rt_pid(self) -> int | None:
        """No r1rt process in lockstep (the simulator steps the core): the board-hold / guard-takeover tests answer
        UNSUPPORTED."""
        return None

    def guard_info(self) -> dict[str, Any] | None:
        return guard_info_of(self.view)

    def rt_info(self) -> dict[str, Any] | None:
        return rt_info_of(self.view)

    def guard_armed(self) -> bool | None:
        info = guard_info_of(self.view)
        return None if info is None or "armed" not in info else bool(info["armed"])
