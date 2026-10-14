"""PlannerWorker (DESIGN.md §7.2: planning in worker processes; K10): the same result as in-process, the caller's
thread is never blocked, a dead worker is replaced."""

from __future__ import annotations

import os
import signal
import time

import numpy as np
from r1mc3.plan import PlanGoal, PlannerWorker, plan
from r1mc3.world import Pose, Surface, WorldModel
from r1sim import joints as J


def test_worker_matches_in_process_and_survives_a_kill() -> None:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.55, 0.0, 0.76), size=(0.6, 1.2, 0.04)))
    snap = w.snapshot()
    q = J.POSE_STAND.copy()
    goal = PlanGoal("reach", "left", position=(0.35, 0.10, 0.90))
    local = plan(snap, goal, q)
    with PlannerWorker() as pw:
        t0 = time.monotonic()
        fut = pw.submit(snap, goal, q)
        assert time.monotonic() - t0 < 0.5  # submit returns at once
        res = fut.result(timeout=60.0)
        assert res.ok and res.code == local.code
        assert np.allclose(res.segments[-1].q[-1], local.segments[-1].q[-1], atol=1e-9)
        # kill the worker process: the next submit starts a new one
        pool = pw._pool
        assert pool is not None
        pid = next(iter(pool._processes))
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        res2 = None
        for _ in range(2):
            try:
                res2 = pw.submit(snap, goal, q).result(timeout=60.0)
                break
            except Exception:  # the broken pool's future
                continue
        assert res2 is not None and res2.ok
