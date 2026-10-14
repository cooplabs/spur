"""The takeover sequence of the v3 core on the simulated robot (real profile on the floor, lockstep; DESIGN §3.5, ABI
4).

Unitree's built-in controller is emulated by the S1 stand fixture (`r1sim.v3_core.EmulatedBuiltin`: a stiff PD at the
CoM-centred stand, publishing on rt/lowcmd until ReleaseMode; Unitree's own balancer is not modelled). The controller
plays r1d's part (`TakeoverPlan`): TAKEOVER at 1.0 s (the core's gates), ReleaseMode 0.3 s later, then RELEASED (BLEND
1.0 s into BALANCE) or ABORT. v3's failure branches: a fault in MIRROR keeps mirroring (alarm), a failed release goes
back to MONITOR (nothing published, the built-in keeps holding), a fault in BLEND finishes the blend (alarm), an
invalid action in BLEND freezes into the stand from the blended command; nothing ever damps. The fixture's CoM-centred
stand itself leans ~2-3 deg (S2), so tilt bounds before BALANCE are 4 deg.
Why the blend is 1.0 s (v2's takeover default; 0.3 s was its into=walk): rung 3 (STAND_FREEZE, a static stand) does not
catch a robot the policy has just started moving - an invalid action in the policy's first ~0.26 s fell with an
instant switch, not with the 1.0 s blend (pinned below; the sweep is in the S4 notes).
The same logic on a synthetic state stream, exactly: rt/tests/test_core_takeover.cpp.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1sim import cpp_policy, world
from r1sim.engine import Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.model import centred_stand_pose
from r1sim.v3_core import (
    FAULT_BITS,
    MODE_BALANCE,
    MODE_DAMP,
    MODE_MONITOR,
    MODE_STAND_FREEZE,
    MODE_TAKEOVER_BLEND,
    MODE_TAKEOVER_MIRROR,
    EmulatedBuiltin,
    TakeoverPlan,
    V3CoreController,
)

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py is not built (scripts/ci.sh cpp)")

T_REQUEST = 1.0
RELEASE_S = 0.3
BLEND_S = 1.0
TILT_MAX = 4.0  # deg: the fixture's centred stand leans ~2-3 deg by itself
LEGS = slice(0, 12)


def _tilt_deg(quat: np.ndarray) -> float:
    _, x, y, _ = quat
    return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _run(
    w: world.World,
    seconds: float,
    *,
    release_ok: bool = True,
    faults: list[Fault] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sched = FaultSchedule(faults or [], seed=1)
    sim = Simulation(w.model, w.index, faults=sched)
    sim.reset(w.initial_qpos(centred_stand_pose(w.model)))
    builtin = EmulatedBuiltin(w.model, release_ok=release_ok)
    plan = TakeoverPlan(builtin, t_request_s=T_REQUEST, release_s=RELEASE_S)
    ctl = V3CoreController(w.model, faults=sched, takeover=plan, config=config, trace_cmd=True)
    sim.add_controller(builtin, ctl.period_s, "builtin")  # first: the core's write of a tick comes after it
    sim.add_controller(ctl, ctl.period_s, "v3")
    pel = w.index.pelvis_body
    tilts: list[tuple[float, float]] = []
    z_min = np.inf
    for _ in range(round(seconds / 0.02)):
        sim.run_for(0.02)
        tilts.append((sim.time, _tilt_deg(sim.d.xquat[pel])))
        z_min = min(z_min, float(sim.d.xpos[pel][2]))
    tr = ctl.cmd_trace
    t = np.array([e[0] for e in tr])
    pub = np.array([e[1] for e in tr])
    mode = np.array([e[2] for e in tr])
    flt = np.array([e[3] for e in tr], dtype=np.int64)
    q = np.array([e[4] for e in tr])
    return {
        "ctl": ctl,
        "plan": plan,
        "builtin": builtin,
        "status": ctl.status(),
        "t": t,
        "pub": pub,
        "mode": mode,
        "faults": flt,
        "q": q,
        "max_tilt": max(a for _, a in tilts),
        "end_tilt": tilts[-1][1],
        "z_min": z_min,
    }


def _step_at(r: dict[str, Any], mode: int) -> float:
    """The largest change of a leg command on the first tick of `mode` (vs the tick before)."""
    k = int(np.argmax(r["mode"] == mode))
    return float(np.max(np.abs(r["q"][k, LEGS] - r["q"][k - 1, LEGS])))


def _policy_steps(r: dict[str, Any], t0: float, t1: float) -> float:
    """The largest per-tick change of a published leg command in [t0, t1) (the policy's own 50 Hz action steps)."""
    sel = (r["t"] >= t0) & (r["t"] < t1) & (r["pub"] == 1)
    q = r["q"][sel][:, LEGS]
    return float(np.max(np.abs(np.diff(q, axis=0)))) if len(q) > 1 else 0.0


def test_takeover_from_the_builtin_stand_into_balance(w: world.World) -> None:
    r = _run(w, 6.0)
    s, plan, mode, t = r["status"], r["plan"], r["mode"], r["t"]
    t_mirror = float(t[np.argmax(mode == MODE_TAKEOVER_MIRROR)])
    t_blend = float(t[np.argmax(mode == MODE_TAKEOVER_BLEND)])
    t_bal = float(t[np.argmax(mode == MODE_BALANCE)])
    snap = r["q"][mode == MODE_TAKEOVER_MIRROR][0]
    first_bal = r["q"][mode == MODE_BALANCE][0]
    jump = float(np.max(np.abs(first_bal[LEGS] - snap[LEGS])))  # an instant switch would step the legs this far
    into_blend = _step_at(r, MODE_TAKEOVER_BLEND)
    into_balance = _step_at(r, MODE_BALANCE)
    policy_steps = _policy_steps(r, t_bal, t_bal + 1.0)
    end_tilt = r["end_tilt"]
    print(
        f"takeover: refusal {plan.refusal}, used mirror {s['takeover_used_mirror']}, snapshot vs measured "
        f"{s['mirror_dev_rad']:.4f} rad; MIRROR {t_mirror:.3f} s, BLEND {t_blend:.3f} s, BALANCE {t_bal:.3f} s; legs: "
        f"snapshot -> first BALANCE command {jump:.4f} rad, step into BLEND {into_blend:.5f}, into BALANCE "
        f"{into_balance:.5f}, the policy's own steps in its first BALANCE second <= {policy_steps:.4f} rad; max "
        f"tilt {r['max_tilt']:.3f} deg, end {end_tilt:.3f}, pelvis min {r['z_min']:.4f} m, alarms "
        f"{s['takeover_alarm_ticks']}"
    )
    assert plan.request_status == 0 and plan.refusal == "ok" and plan.release_ok and plan.decision_status == 0
    assert s["takeover_used_mirror"] == 1 and s["mirror_dev_rad"] < 0.1  # the stand lock sits on the measured pose
    assert np.all(r["pub"][t < T_REQUEST - 1e-9] == 0) and np.all(mode[t < T_REQUEST - 1e-9] == MODE_MONITOR)
    assert t_mirror == pytest.approx(T_REQUEST, abs=1e-9)
    assert t_blend == pytest.approx(T_REQUEST + RELEASE_S, abs=1e-9)
    assert t_bal == pytest.approx(T_REQUEST + RELEASE_S + BLEND_S, abs=1e-9)
    # MIRROR republishes the built-in's command exactly (its stand pose at its stiff gains)
    fix = r["builtin"].fixture.command
    mirrored = r["q"][mode == MODE_TAKEOVER_MIRROR]
    np.testing.assert_array_equal(mirrored, np.tile(fix.q, (len(mirrored), 1)))
    # bumpless: no step where the mode changes (into BLEND: 1/500 of the distance; into BALANCE: no more than the
    # policy's own 50 Hz steps), while an instant switch would step the legs by `jump`
    assert into_blend < 0.005 and into_blend < 0.05 * jump
    # the step into BALANCE as bumpless as the step into BLEND (the blend ends at a = 1 on the policy's own command;
    # what remains is that tick's policy change): measured 0.0003 rad vs the 0.156 rad an instant switch would step
    assert into_balance < 0.005 and into_balance < 0.05 * jump
    assert s["mode"] == MODE_BALANCE and s["faults_seen"] & FAULT_BITS["takeover_alarm"] == 0
    assert s["faults_seen"] & FAULT_BITS["builtin_active"] == 0  # the built-in stopped at the release
    assert not np.any(mode == MODE_DAMP)
    assert r["max_tilt"] < TILT_MAX and end_tilt < 1.0 and r["z_min"] > 0.70
    assert s["policy_thread_runs"] > 100 and s["inline_runs"] == 0


def test_a_release_that_fails_aborts_back_to_monitor(w: world.World) -> None:
    r = _run(w, 3.0, release_ok=False)
    s, plan, mode, t = r["status"], r["plan"], r["mode"], r["t"]
    t_abort = T_REQUEST + RELEASE_S
    print(f"abort: decision {plan.decision_status}, aborts {s['takeover_aborts']}, max tilt {r['max_tilt']:.3f} deg")
    assert plan.request_status == 0 and plan.release_ok is False and plan.decision_status == 0
    assert np.all(mode[t >= t_abort - 1e-9] == MODE_MONITOR)
    assert np.all(r["pub"][t >= t_abort - 1e-9] == 0)  # nothing of ours competes with the built-in that still drives
    assert s["takeover_aborts"] == 1 and s["faults_seen"] & FAULT_BITS["takeover_alarm"]
    assert r["builtin"].released is False and r["max_tilt"] < TILT_MAX  # the built-in kept holding the robot


def test_a_gate_that_fails_changes_nothing(w: world.World) -> None:
    # the profile demands a stiffer built-in than the stand lock's kp 200: refused, the core stays silent
    r = _run(w, 2.0, config={"builtin_min_leg_kp": 250.0})
    s, plan = r["status"], r["plan"]
    assert plan.request_status != 0 and plan.refusal == "builtin"
    assert np.all(r["mode"] == MODE_MONITOR) and np.all(r["pub"] == 0)
    assert s["takeover_requests"] == 1 and r["max_tilt"] < TILT_MAX


def test_a_lowstate_gap_during_mirror_keeps_mirroring_with_an_alarm(w: world.World) -> None:
    gap = Fault("controller_stall", T_REQUEST + 0.05, 0.15, {"name": "v3_lowstate"})  # 150 ms: STALE in MIRROR
    r = _run(w, 4.5, faults=[gap])
    s, mode, t, flt = r["status"], r["mode"], r["t"], r["faults"]
    in_gap = (t >= T_REQUEST + 0.05) & (t < T_REQUEST + 0.2)
    assert np.all(mode[in_gap] == MODE_TAKEOVER_MIRROR) and np.all(r["pub"][in_gap] == 1)
    stale = in_gap & ((flt & FAULT_BITS["lowstate_stale"]) != 0)
    assert np.any(stale) and np.all(flt[stale] & FAULT_BITS["takeover_alarm"])
    fix = r["builtin"].fixture.command
    np.testing.assert_array_equal(r["q"][in_gap], np.tile(fix.q, (int(np.sum(in_gap)), 1)))
    assert s["mode"] == MODE_BALANCE and r["max_tilt"] < TILT_MAX


def test_a_policy_stall_during_the_blend_finishes_into_balance(w: world.World) -> None:
    stall = Fault("controller_stall", T_REQUEST + RELEASE_S + 0.05, 0.4, {"name": "v3_policy"})
    r = _run(w, 4.5, faults=[stall])
    s, mode, t, flt = r["status"], r["mode"], r["t"], r["faults"]
    blend = mode == MODE_TAKEOVER_BLEND
    assert np.any(blend & ((flt & FAULT_BITS["policy_late"]) != 0))  # rung 1 inside the blend
    assert np.any(blend & ((flt & FAULT_BITS["takeover_alarm"]) != 0))
    t_bal = float(t[np.argmax(mode == MODE_BALANCE)])
    print(f"policy stall in BLEND: inline runs {s['inline_runs']}, BALANCE at {t_bal:.3f} s, tilt {r['max_tilt']:.3f}")
    assert t_bal == pytest.approx(T_REQUEST + RELEASE_S + BLEND_S, abs=1e-9)  # the blend finished on time
    assert s["mode"] == MODE_BALANCE and s["inline_runs"] > 0 and r["max_tilt"] < TILT_MAX
    assert not np.any(mode == MODE_DAMP)


@pytest.mark.slow
@pytest.mark.parametrize(("blend_s", "ramp_s"), [(BLEND_S, None), (0.002, None), (0.002, 0.05)])
def test_an_invalid_action_in_the_policys_first_moments_freezes_never_damps(
    w: world.World, blend_s: float, ramp_s: float | None
) -> None:
    """A NaN action 0.1 s after the release (the policy's start-up transient): STAND_FREEZE from the last command,
    publishing, never damp - in every configuration. Whether the static stand catches the robot depends on the freeze
    RAMP (S4 sweep, 12 NaN times in the policy's first 1.2 s: the default 0.4 s ramp falls 3/12 with the 1.0 s blend,
    5/12 with 0.3 s, 7/12 with an instant switch; a 0.05 s ramp 0/12 even with an instant switch, and 0/8 in steady
    BALANCE like the 0.4 s ramp). The default ramp (DESIGN §3.5: 0.3-0.5 s) is a decision for the user; this test
    asserts the outcome only for the 0.05 s ramp (the proposal) and prints the others."""
    nan = Fault("policy_nan", T_REQUEST + RELEASE_S + 0.1, 10.0)
    cfg: dict[str, Any] = {"takeover_blend_s": blend_s}
    if ramp_s is not None:
        cfg["stand_ramp_s"] = ramp_s
    r = _run(w, 5.0, faults=[nan], config=cfg)
    s, mode = r["status"], r["mode"]
    k = int(np.argmax(mode == MODE_STAND_FREEZE))
    step_in = float(np.max(np.abs(r["q"][k, LEGS] - r["q"][k - 1, LEGS])))  # the freeze starts at the last command
    print(
        f"NaN 0.1 s after the release, blend {blend_s} s, ramp {ramp_s or 'default'}: freeze at {r['t'][k]:.3f} s "
        f"(from mode {mode[k - 1]}), first step {step_in:.5f} rad, max tilt {r['max_tilt']:.1f} deg, end "
        f"{r['end_tilt']:.1f} deg"
    )
    assert step_in < 0.01
    assert s["mode"] == MODE_STAND_FREEZE and s["faults_seen"] & FAULT_BITS["policy_invalid"]
    assert not np.any(mode == MODE_DAMP) and np.all(r["pub"][r["t"] >= T_REQUEST - 1e-9] == 1)
    if blend_s == BLEND_S:
        assert mode[k - 1] == MODE_TAKEOVER_BLEND
    if ramp_s == 0.05:
        assert r["end_tilt"] < 5.0 and r["z_min"] > 0.65
