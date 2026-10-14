"""Fix #29 (session 8) with REAL processes: r1rt + r1guard + r1d + the CLI on the DDS-bridge sim's HANGING robot with
the board model (the bridge's engine: `DriverConfig(cmd_timeout_s=0.4, on_cmd_timeout="damp", damp_kd=15)` - HW1's
board-hold test: the board holds the last LowCmd ~400 ms, then a damped release). `r1 on-stand` from Unitree's stand
lock, then:

* the board-hold test's case: r1guard OBSERVE-ONLY, r1rt SIGSTOPped 2.0 s, SIGCONT - r1rt must re-enter through the
blend from the
  measured pose (no snap back to its old target);
* the whole-machine case (the 9 Oct class): r1rt AND r1guard SIGSTOPped together 2.0 s, SIGCONT both - whoever
  publishes first re-enters through the blend; never two writers interleaved, never limp.

The snap is measured on the robot side: the applied joint torque (the engine's ground truth) sampled every ~1 ms by a
thread here; "step" = the largest change between consecutive samples on legs / waist / arms in the 0.3 s after the
first command after the stop. macOS without real-time scheduling: the numbers carry the Mac's jitter (quoted as such).
"""

from __future__ import annotations

import itertools
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

import mujoco  # noqa: E402
from r1sim import stack, world  # noqa: E402
from r1sim.dds_bridge import HW1_STAND, DdsBridge, stand_extra  # noqa: E402
from r1sim.driver import DriverConfig  # noqa: E402

from tests.integration.test_v3_resume import HW1_FRICTION  # noqa: E402

BODY = slice(0, 24)
STOP_S = 2.0
STEP_MAX_NM = 4.0  # lockstep: <= 1.1 N m fixed, 11.6 N m base (2 s); the processes add the Mac's jitter


class _Sampler:
    """The bridge's ground truth every ~1 ms: (sim t, q, tau applied) - a thread of the test process."""

    def __init__(self, br: DdsBridge) -> None:
        self.br = br
        self.rows: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        last = -1.0
        while not self._stop.is_set():
            st = self.br.sim.read_state()
            if st.t != last:
                last = float(st.t)
                q = np.asarray(st.truth.get("q", st.q), dtype=np.float64).copy()
                tau = np.asarray(st.truth.get("tau_est", st.tau_est), dtype=np.float64).copy()
                self.rows.append((last, q, tau))
            time.sleep(0.0005)

    def __enter__(self) -> _Sampler:
        self._th.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._th.join(2.0)


def _hw1_world() -> world.World:
    """The hanging robot (the bridge's stand) with the HW1 leg / waist friction (the model's 2.5 N m holds a hanging
    hip against most of its gravity and hides the sag: test_v3_resume.HW1_FRICTION)."""
    w = world.build_world("real", with_scene=False, extra=stand_extra(HW1_STAND))  # session 8: the default stand
    m = w.model
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for key, fr in HW1_FRICTION.items():
            if name.endswith(f"{key}_joint"):
                m.dof_frictionloss[m.jnt_dofadr[j]] = fr
    return w


def _run(tmp_path: Path, domain: int, *, whole: bool) -> dict[str, Any]:
    br = DdsBridge(domain=domain, scene="none", stance="hanging", builtin="stand_lock", w=_hw1_world())
    # the board model = the bridge's default since session 8 (DriverConfig.real(): HW1's board-hold 0.385 s / kd 12.5;
    # this test's own BOARD 0.4 s / kd 15 was the provisional stand-in); slow path: the bridge's sim has sensors
    assert br.sim.driver == DriverConfig.real()
    br.start()
    try:
        with stack.ProcStack(domain=domain, workdir=tmp_path, bridge=None) as st:
            st.wait_mode("monitor", 15)
            time.sleep(0.5)
            r = st.cli("on-stand")
            assert r["ok"], r
            st.wait_mode("on_stand", 15)
            time.sleep(1.5)
            if not whole:  # the board-hold test: the guard observe-only (the user's sudo at HW1: r1guard-observe)
                assert st.stop_guard() == 0
                st.start_guard("--observe-only")
                time.sleep(0.5)
            assert st.rt is not None and st.guard is not None
            procs = [st.rt.p.pid] + ([st.guard.p.pid] if whole else [])
            with _Sampler(br) as smp:
                time.sleep(0.3)
                t_sim_stop = float(br.sim.read_state().t)
                wall_stop = time.monotonic()
                for pid in procs:
                    os.kill(pid, signal.SIGSTOP)
                time.sleep(STOP_S)
                for pid in reversed(procs):
                    os.kill(pid, signal.SIGCONT)
                wall_cont = time.monotonic()
                time.sleep(2.0)
            status = st.status()
            guard_line = st.guard.last_status()
            rt_line = st.rt.last_status()
            applied = [a for a in list(br.applied) if a.wall > wall_stop - 0.5]
            labels = {h: v["label"] for h, v in br.writers().items()}
    finally:
        br.stop()
    t = np.array([x[0] for x in smp.rows])
    q = np.array([x[1] for x in smp.rows])
    tau = np.array([x[2] for x in smp.rows])
    after = [a for a in applied if a.wall >= wall_cont]
    assert after, "nothing published after the stop"
    t_first = after[0].sim_t  # the first command after the stop, on the board's clock
    win = (t >= t_first - 0.003) & (t <= t_first + 0.3)
    dtau = np.abs(np.diff(tau[:, BODY], axis=0))[win[1:]]
    pre = t < t_sim_stop
    k0 = int(np.flatnonzero(t <= t_first)[-1])
    sag = np.abs(q[k0, BODY] - q[pre][-1, BODY])
    writers_after = [labels.get(a.writer, "?") for a in after]
    switches = sum(1 for x, y in itertools.pairwise(writers_after) if x != y)
    return {
        "step_nm": float(dtau.max()) if dtau.size else float("nan"),
        "sag_max": float(sag.max()),
        "writers_after": sorted(set(writers_after)),
        "switches": switches,
        "min_leg_kp": min(a.min_leg_kp for a in applied),
        "end_err": float(np.abs(q[-1, BODY] - q[pre][-1, BODY]).max()),
        "status": status,
        "rt": {
            k: (rt_line or {}).get(k)
            for k in (
                "stops",
                "stop_skipped",
                "overruns",
                "stop_ms_max",
                "resume_blends",
                "resume_waits_expired",
                "skipped_deadlines",
            )
        },
        "guard": {
            k: (guard_line or {}).get(k)
            for k in ("takeovers", "stops", "resume_takeovers", "resume_blends", "resume_waits_expired", "act_mode")
        },
        "gap_s": t_first - t_sim_stop,
        "step_j": dtau.max(axis=0) if dtau.size else None,
        "step_t": (t[1:][win[1:]][np.argmax(dtau.max(axis=1))] - t_first) if dtau.size else None,
        "dt_samples_ms": float(np.median(np.diff(t[win])) * 1e3) if win.sum() > 2 else None,
        "dt_max_ms": float(np.max(np.diff(t[win])) * 1e3) if win.sum() > 2 else None,
    }


@pytest.mark.realtime
@pytest.mark.slow
def test_board_hold_r1rt_resumes_after_a_2s_stop_without_a_snap(tmp_path: Path) -> None:
    if not stack.available():
        pytest.skip("needs the DDS build of r1rt / r1guard (scripts/ci.sh dds)")
    r = _run(tmp_path, 51, whole=False)
    print(f"\n[ info ] board-hold 2 s (macOS, under load): {r}")
    assert r["sag_max"] > 0.01, r  # the board let go after 0.4 s: the hanging limbs sagged
    assert r["writers_after"] == ["r1rt"] and r["switches"] == 0, r
    assert r["min_leg_kp"] > 0.0, r  # never limp (every applied command)
    assert r["step_nm"] < STEP_MAX_NM, r  # base: the snap back to the old target at full stiffness
    assert r["end_err"] < 0.03, r


@pytest.mark.realtime
@pytest.mark.slow
def test_a_whole_machine_stop_of_r1rt_and_r1guard_resumes_without_a_snap(tmp_path: Path) -> None:
    if not stack.available():
        pytest.skip("needs the DDS build of r1rt / r1guard (scripts/ci.sh dds)")
    r = _run(tmp_path, 52, whole=True)
    print(f"\n[ info ] whole-machine 2 s (macOS, under load): {r}")
    assert r["sag_max"] > 0.01, r
    assert r["switches"] <= 1, r  # one hand-over at most (r1rt first, then a late guard takeover), never interleaved
    assert r["min_leg_kp"] > 0.0, r
    assert r["step_nm"] < STEP_MAX_NM, r
