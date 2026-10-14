"""`PlannerWorker`: the planner in its own process (DESIGN.md §7.2: "all in worker processes; pure functions of a
world snapshot"; K10: a Python stall must not stop the arms' keepalives in r1d).

    w = PlannerWorker()                       # spawns one process (the model compiles there once per world)
    fut = w.submit(snapshot, goal, q_now)     # -> concurrent.futures.Future[PlanResult]
    res = fut.result(timeout=5.0)
    w.close()

The process start method is `spawn` (no fork of r1d's threads, no inherited DDS participants / shared memory). A
worker that dies is restarted on the next submit; its pending futures fail with BrokenProcessPool (r1d answers the
goal with a code, nothing moves).
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from ..world.entities import WorldSnapshot
from .core import PlanGoal, PlanResult, plan

FloatArray = npt.NDArray[np.float64]


def _run(snapshot: WorldSnapshot, goal: PlanGoal, q_now: FloatArray, kw: dict[str, Any]) -> PlanResult:
    return plan(snapshot, goal, q_now, **kw)


PARENT_POLL_S = 0.5  # how often the worker checks that r1d is still its parent


def _exit_with_parent(parent_pid: int) -> None:
    """The worker's initializer: a daemon thread ends the worker when r1d (its parent) is gone (session 11: a
    ProcessPoolExecutor worker blocks on its call queue forever after its parent dies - the queue's pipe has a write end
    in the worker itself, so no EOF; ~550 orphaned workers + resource trackers had piled up on the Mac from SIGKILLed
    r1d's, and on the robot r1d.service's KillMode=process would keep them in the unit's cgroup across r1d crashes)."""

    def watch() -> None:
        while True:
            if os.getppid() != parent_pid:  # re-parented: r1d died (SIGKILL, OOM, a crash)
                os._exit(0)
            time.sleep(PARENT_POLL_S)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def _warm() -> bool:
    from ..world.entities import WorldModel
    from .core import geometry_for

    geometry_for(WorldModel().snapshot())
    return True


class PlannerWorker:
    def __init__(self, *, warm: bool = True) -> None:
        self._ctx = mp.get_context("spawn")
        self._pool: cf.ProcessPoolExecutor | None = None
        self._warm = warm
        self._start()

    def _start(self) -> None:
        self._pool = cf.ProcessPoolExecutor(
            max_workers=1, mp_context=self._ctx, initializer=_exit_with_parent, initargs=(os.getpid(),)
        )
        if self._warm:
            self._pool.submit(_warm)

    def submit(
        self,
        snapshot: WorldSnapshot,
        goal: PlanGoal,
        q_now: FloatArray,
        **kw: Any,
    ) -> cf.Future[PlanResult]:
        assert self._pool is not None
        if getattr(self._pool, "_broken", False):  # the process died (killed, crashed): start a new one
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._start()
            assert self._pool is not None
        try:
            return self._pool.submit(_run, snapshot, goal, np.asarray(q_now, dtype=np.float64), kw)
        except cf.process.BrokenProcessPool:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._start()
            assert self._pool is not None
            return self._pool.submit(_run, snapshot, goal, np.asarray(q_now, dtype=np.float64), kw)

    def call(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> cf.Future[Any]:
        """Another pure planner-side function in the worker's process (session 11: arm.move_path's blend check needs
        the compiled geometry, which lives here). `fn` must be importable at module level (spawn pickles it by
        name); a dead process is restarted as in `submit`."""
        assert self._pool is not None
        if getattr(self._pool, "_broken", False):
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._start()
            assert self._pool is not None
        try:
            return self._pool.submit(fn, *args, **kw)
        except cf.process.BrokenProcessPool:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._start()
            assert self._pool is not None
            return self._pool.submit(fn, *args, **kw)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def __enter__(self) -> PlannerWorker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
