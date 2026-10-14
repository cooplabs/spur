"""The v3 real-time core (C++, r1rt_py.Core) balancing the simulated robot in lockstep (real profile on the floor,
writer 500 Hz, its policy thread 50 Hz; r1sim.v3_core.V3CoreController): the session-3 checks of DESIGN §4.2-§4.3.

1. It stands 10 s on its own C++ policy runner (no inline runs, no fault bits).
2. A7 moved into v3: the 9 Oct stall (4.018 s) injected into the v3 POLICY THREAD -> the writer runs the policy
   inline (rung 1) within one writer tick of the 40 ms threshold, every policy period, never limp, the robot stays
   upright (max tilt < 3 deg); the thread comes back and the writer hands back. Two stances: the policy's default
   arm pose and the logged table_ready arm pose of the 9 Oct stall (telemetry line 1137). NOTE: in this sim a plain
   policy controller that simply STOPS for 4.018 s (the board holding its last command) survives in both stances too
   (s3 probe: max tilt 0.39 deg) - the real robot fell; the v2-falls-v3-does-not contrast needs the `fall` agent's
   A7 v2 reproduction (what else differed on 9 Oct), so this test proves the rung, not the contrast.
3. A NaN policy action -> STAND_FREEZE (rung 3) on the next writer tick after the policy step: a 0.4 s ramp into the
   CoM-centred stand at stand gains; PINNED: the robot stays up for the 20 s simulated after it (S2's v2 semantics
   held the last good command and tipped at ~19.5 s).
4. The lowstate stops reaching the writer (the DDS receive thread stalled) for 15 ms / 150 ms / 1.5 s -> late /
   stale (the last balanced command resent) / lost (a fault, still publishing) - never limp, the robot stays up.
5. Lockstep reruns are bit-identical.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import pytest
from r1sim import cpp_policy, world
from r1sim import joints as J
from r1sim.engine import Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.v3_core import FAULT_BITS, MODE_BALANCE, MODE_DAMP, MODE_STAND_FREEZE, V3CoreController, faults_of

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")

STALL_S = 4.018  # 9 Oct: telemetry t 433.7816 -> 437.8001 (V2_INVENTORY App. D facts row 1)


def _tilt_deg(quat: np.ndarray) -> float:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(r[8], -1.0, 1.0))))


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


# The arms' commanded pose in the last row before the 9 Oct stall (table_ready; NB r1-motor-controller-v2/results/
# serve-20261010T010206Z/telemetry.jsonl line 1137, t 433.7816, q_cmd[14:24]). The core HOLDs the arms where they are
# at BALANCE entry, so the robot starts there. (v2 also sent arm gravity feed-forward, tau_ff; the core's HOLD has none
# until the executor's built-in arm model, DESIGN §3.4.)
ARMS_TABLE_READY = (0.0689, 0.685, 0.0192, -0.5269, -1.5534, 0.2079, -0.1853, 0.0042, 0.8141, 0.0039)


def _run(w: world.World, faults: list[Fault], seconds: float, arms: Any = None) -> dict[str, Any]:
    sched = FaultSchedule(faults, seed=1)
    sim = Simulation(w.model, w.index, faults=sched)
    pose = J.POSE_STAND.copy()
    if arms is not None:
        pose[J.ARMS] = arms
    sim.reset(w.initial_qpos(pose))
    ctl = V3CoreController(w.model, faults=sched, trace=True)
    sim.add_controller(ctl, ctl.period_s, "v3")
    pel = w.index.pelvis_body
    tilts: list[tuple[float, float]] = []
    z_min = np.inf
    for _ in range(round(seconds / 0.05)):
        sim.run_for(0.05)
        tilts.append((sim.time, _tilt_deg(sim.d.xquat[pel])))
        z_min = min(z_min, float(sim.d.xpos[pel][2]))
    return {
        "ctl": ctl,
        "sched": sched,
        "status": ctl.status(),
        "max_tilt": max(a for t, a in tilts if t >= 1.0),
        "end_tilt": tilts[-1][1],
        "z_min": z_min,
        "qpos": sim.d.qpos.copy(),
        "trace": np.array(ctl.trace),
    }


def test_stands_10_s_under_the_v3_core(w: world.World) -> None:
    r = _run(w, [], 10.0)
    s = r["status"]
    print(f"v3 core 10 s: max tilt {r['max_tilt']:.3f} deg, pelvis min {r['z_min']:.4f} m, {s}")
    assert r["ctl"].enter_status == 0
    assert s["mode"] == MODE_BALANCE and s["inline_runs"] == 0 and s["faults_seen"] == 0
    assert s["policy_thread_runs"] == 499  # 10 s at 50 Hz after the entry step
    assert r["max_tilt"] < 1.0 and r["z_min"] > 0.70
    tr = r["trace"]
    assert np.all(tr[:, 1] == MODE_BALANCE)


@pytest.mark.parametrize("stance", ["default", "table_ready_arms"])
def test_a7_stall_in_the_v3_policy_thread_runs_the_policy_inline(w: world.World, stance: str) -> None:
    t0 = 3.0
    arms = ARMS_TABLE_READY if stance == "table_ready_arms" else None
    r = _run(w, [Fault("controller_stall", t0, STALL_S, {"name": "v3_policy"})], 10.0, arms)
    s, tr = r["status"], r["trace"]
    late = (tr[:, 2].astype(np.int64) & FAULT_BITS["policy_late"]) != 0
    first_late = float(tr[np.argmax(late), 0])
    stalled = [e.t for e in r["sched"].log if e.kind == "controller_stall"]
    last_thread = stalled[0] - 0.02  # the thread's last iteration before its first skipped one
    print(
        f"A7 v3 ({stance}): stall {STALL_S} s at {t0} s in the policy thread: first inline tick at "
        f"{first_late:.3f} s, inline runs {s['inline_runs']}, entries {s['inline_entries']}, max tilt "
        f"{r['max_tilt']:.3f} deg, pelvis min {r['z_min']:.4f} m, end mode {s['mode']}"
    )
    assert s["inline_entries"] == 1 and abs(s["inline_runs"] - STALL_S / 0.02) <= 3
    assert 0.040 < first_late - last_thread <= 0.040 + 0.002 + 1e-9  # within one writer tick of the 40 ms age
    assert np.all(tr[:, 1] == MODE_BALANCE)  # never limp, never frozen: balance throughout
    assert abs(r["sched"].count("controller_stall", "skip_call") - STALL_S / 0.02) <= 1
    assert s["inline_active"] == 0 and s["policy_thread_runs"] > 400 - 200  # handed back to the thread
    assert r["max_tilt"] < 3.0 and r["z_min"] > 0.70


def test_nan_action_freezes_into_the_stand_and_stays_up(w: world.World) -> None:
    t0 = 3.0
    r = _run(w, [Fault("policy_nan", t0, 0.1)], 23.0)
    s, tr = r["status"], r["trace"]
    frozen = tr[:, 1] == MODE_STAND_FREEZE
    t_freeze = float(tr[np.argmax(frozen), 0])
    print(
        f"NaN at {t0} s: STAND_FREEZE at {t_freeze:.3f} s; 20 s later: tilt {r['end_tilt']:.2f} deg, max "
        f"{r['max_tilt']:.2f} deg, pelvis min {r['z_min']:.4f} m; faults {faults_of(s['faults_seen'])}"
    )
    assert t0 <= t_freeze <= t0 + 0.022  # the next policy step produced it, the next writer tick froze
    assert np.all(frozen[tr[:, 0] >= t_freeze]) and not np.any(tr[:, 1] == MODE_DAMP)
    assert s["freeze_entries"] == 1 and s["faults"] & FAULT_BITS["policy_invalid"]
    # PINNED (session 3): 20 s in the freeze, upright within 5 deg, the pelvis above 0.70 m
    assert r["max_tilt"] < 5.0 and r["z_min"] > 0.70


@pytest.mark.parametrize(
    ("gap_s", "want", "not_want"),
    [
        (0.015, "lowstate_late", "lowstate_stale"),
        (0.150, "lowstate_stale", "lowstate_lost"),
        (1.5, "lowstate_lost", None),
    ],
)
def test_lowstate_withheld(w: world.World, gap_s: float, want: str, not_want: str | None) -> None:
    t0 = 3.0
    r = _run(w, [Fault("controller_stall", t0, gap_s, {"name": "v3_lowstate"})], 6.0)
    s, tr = r["status"], r["trace"]
    seen = int(np.bitwise_or.reduce(tr[:, 2].astype(np.int64)))
    print(
        f"lowstate withheld {gap_s * 1e3:.0f} ms: {faults_of(seen)}, stale resends {s['stale_resends']}, "
        f"max tilt {r['max_tilt']:.3f} deg"
    )
    assert seen & FAULT_BITS[want]
    if not_want is not None:
        assert not seen & FAULT_BITS[not_want]
    assert np.all(tr[:, 1] == MODE_BALANCE)  # still publishing the balanced command, never limp
    assert r["ctl"].last_publish == 1
    if gap_s >= 0.15:
        assert s["stale_resends"] > 0
    after = tr[tr[:, 0] > t0 + gap_s + 0.05]
    assert not np.any(after[:, 2].astype(np.int64) & (FAULT_BITS["lowstate_stale"] | FAULT_BITS["lowstate_lost"]))
    assert r["max_tilt"] < 3.0 and r["z_min"] > 0.70


def test_lockstep_rerun_is_bit_identical(w: world.World) -> None:
    a = _run(w, [Fault("controller_stall", 1.0, 0.3, {"name": "v3_policy"})], 2.0)
    b = _run(w, [Fault("controller_stall", 1.0, 0.3, {"name": "v3_policy"})], 2.0)
    assert np.array_equal(a["qpos"], b["qpos"])
