"""ON_STAND (session 5, DESIGN §3.5) on the simulated robot HANGING on its stand (lockstep, real profile, no scene).

The stand: the pelvis welded to the world 0.10 m above its standing height (feet off the floor; the user's stand holds
the robot hanging, DESIGN D12a) - a test fixture, not part of the scene. Unitree's built-in is the S1 fixture
(`EmulatedBuiltin`) DAMPING the legs (kp 0: Unitree's damping mode, the normal start on the stand; a takeover would
refuse it). r1d's part goes through the mode-request mailbox: ON_STAND (the gates), ReleaseMode, RELEASED -> the 1.0 s
blend into a stiff hold of the pose measured at RELEASED -> ON_STAND. Then per-joint L0 on a leg (a STREAM on the left
leg group with TTL, renewed, then left to expire -> HOLD) and the stop; with r1guard: the core stalls in ON_STAND and
the guard holds the core's last command (holding, not balancing). C++ on a synthetic stream:
rt/tests/test_core_onstand.cpp, test_guard.cpp.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import pytest
from r1sim import cpp_policy, world
from r1sim.engine import Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.model import centred_stand_pose
from r1sim.v3_core import (
    GROUP_LEG_LEFT,
    MODE_DAMP,
    MODE_ON_STAND,
    MODE_TAKEOVER_BLEND,
    MODE_TAKEOVER_MIRROR,
    REQ_DAMP,
    REQ_STOP,
    REQUESTER_OPERATOR,
    REQUESTER_R1D,
    EmulatedBuiltin,
    TakeoverPlan,
    V3CoreController,
)

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py is not built (scripts/ci.sh cpp)")

HANG_DZ = 0.10  # m above the standing height: feet off the floor
T_REQUEST = 1.5  # the dangling legs settle under the built-in's damping first (the motion gate: |dq| <= 0.5)
RELEASE_S = 0.3
BLEND_S = 1.0
T_ON_STAND = T_REQUEST + RELEASE_S + BLEND_S
KNEE_L = 3


def _hanging_world() -> tuple[world.World, np.ndarray]:
    w0 = world.build_world("real", visual=False, with_scene=False)
    q0 = w0.initial_qpos(centred_stand_pose(w0.model))
    z_hang = float(q0[2]) + HANG_DZ

    def hang(spec: mujoco.MjSpec, _info: Any) -> None:
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = "pelvis"  # body2 = the world
        eq.name = "test_stand_hook"
        data = np.zeros(11)
        data[5] = -z_hang  # relpose: the world's origin in the pelvis frame (pelvis upright at z_hang)
        data[6] = 1.0
        data[10] = 1.0  # torquescale
        eq.data = data

    w = world.build_world("real", visual=False, with_scene=False, extra=hang)
    q = w.initial_qpos(centred_stand_pose(w.model))
    q[2] += HANG_DZ
    return w, q


@pytest.fixture(scope="module")
def hanging() -> tuple[world.World, np.ndarray]:
    return _hanging_world()


def _foot_floor_contacts(m: Any, d: Any) -> int:
    n = 0
    for i in range(d.ncon):
        names = {
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom1) or "",
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, d.contact[i].geom2) or "",
        }
        if "floor" in names:
            n += 1
    return n


def _run(
    hanging: tuple[world.World, np.ndarray], seconds: float, *, guarded: bool = False, faults: list[Fault] | None = None
) -> dict[str, Any]:
    w, q = hanging
    sched = FaultSchedule(faults or [], seed=1)
    sim = Simulation(w.model, w.index, faults=sched)
    sim.reset(q)
    builtin = EmulatedBuiltin(w.model, leg_kp=0.0)  # damping the hanging legs
    plan = TakeoverPlan(builtin, t_request_s=T_REQUEST, release_s=RELEASE_S, target="on_stand", via_mailbox=True)
    ctl: Any
    if guarded:
        from r1sim.v3_guard import V3GuardedController

        ctl = V3GuardedController(w.model, faults=sched, takeover=plan, trace_cmd=True)
        sim.add_controller(builtin, ctl.writer_period_s, "builtin")
    else:
        ctl = V3CoreController(w.model, faults=sched, takeover=plan, trace_cmd=True)
        sim.add_controller(builtin, ctl.period_s, "builtin")
    sim.add_controller(ctl, ctl.period_s, "v3")
    contacts = 0
    for _ in range(round(seconds / 0.01)):
        sim.run_for(0.01)
        contacts = max(contacts, _foot_floor_contacts(w.model, sim.d))
    return {"sim": sim, "ctl": ctl, "plan": plan, "builtin": builtin, "contacts": contacts, "w": w}


def _state_q(r: dict[str, Any]) -> np.ndarray:
    return np.asarray(r["sim"].read_state().q, dtype=np.float64)


def test_the_hanging_robot_enters_on_stand_through_the_mailbox_and_holds(hanging: Any) -> None:
    r = _run(hanging, T_ON_STAND + 0.5)
    ctl, plan = r["ctl"], r["plan"]
    assert plan.request_status == 0 and plan.refusal == "ok" and plan.decision_status == 0, plan.refusal
    assert plan.seqs == [1, 2]
    st = ctl.status()
    assert st["mode"] == MODE_ON_STAND and st["takeover_target"] == MODE_ON_STAND
    assert st["policy_commits"] == 0  # nothing balances on the stand
    tr = ctl.cmd_trace
    modes = [m for _, _, m, _, _, _ in tr]
    assert MODE_TAKEOVER_MIRROR in modes and MODE_TAKEOVER_BLEND in modes
    t_mirror = next(t for t, _, m, _, _, _ in tr if m == MODE_TAKEOVER_MIRROR)
    t_stand = next(t for t, _, m, _, _, _ in tr if m == MODE_ON_STAND)
    assert abs(t_mirror - T_REQUEST) < 0.003
    assert abs(t_stand - T_ON_STAND) < 0.01
    # the legs stiffen over the blend from the built-in's damping (kp 0) to the ON_STAND hold, never a step
    kps = np.array([kp[:12] for t, p, m, _, _, kp in tr if p and t >= T_REQUEST + RELEASE_S])
    assert float(np.min(kps[-1])) > 30.0  # stiff at the end (the stand gains)
    assert float(np.max(np.abs(np.diff(kps, axis=0)))) < 1.0  # gains ramp: < 1 N m/rad per 2 ms tick
    qs = np.array([q for t, p, m, _, q, _ in tr if p])
    assert float(np.max(np.abs(np.diff(qs[:, :14], axis=0)))) < 0.01  # no position step
    # holding: the measured legs stay at the command (hanging, nothing else touches them), the feet never touch
    q_meas = _state_q(r)
    q_cmd = tr[-1][4]
    assert float(np.max(np.abs(q_meas[:12] - q_cmd[:12]))) < 0.05
    assert r["contacts"] == 0


def test_a_leg_follows_l0_streams_holds_when_they_expire_and_stops(hanging: Any) -> None:
    w, q = hanging
    sched = FaultSchedule([], seed=1)
    sim = Simulation(w.model, w.index, faults=sched)
    sim.reset(q)
    builtin = EmulatedBuiltin(w.model, leg_kp=0.0)
    plan = TakeoverPlan(builtin, t_request_s=T_REQUEST, release_s=RELEASE_S, target="on_stand", via_mailbox=True)
    ctl = V3CoreController(w.model, faults=sched, takeover=plan, trace_cmd=True)
    sim.add_controller(builtin, ctl.period_s, "builtin")
    sim.add_controller(ctl, ctl.period_s, "v3")
    sim.run_for(T_ON_STAND + 0.2)
    assert ctl.status()["mode"] == MODE_ON_STAND
    leg0 = ctl.last_q[0:6].copy()
    goal = leg0.copy()
    goal[KNEE_L] += 0.25  # bend the left knee 0.25 rad (a hanging leg)
    cid = {"id": 0}

    def stream(c: V3CoreController, t_ns: int) -> None:
        cid["id"] = c.send_stream(t_ns, GROUP_LEG_LEFT, goal, 0.3, cmd_id=cid["id"] or None)

    t0 = sim.time
    for k in range(6):  # renewed every 0.1 s for 0.6 s, then left to expire (TTL 0.3 s)
        ctl.at(t0 + 0.1 * k, stream)
    sim.run_for(0.55)
    st = ctl.status()
    assert st["lower_accepted"] >= 6 and st["lower_rejected"] == 0
    assert st["lower_source"][0] == 2  # STREAM
    assert abs(ctl.last_q[KNEE_L] - goal[KNEE_L]) < 1e-9  # the target reached (at <= 2 rad/s)
    q_meas = _state_q({"sim": sim})
    assert abs(q_meas[KNEE_L] - goal[KNEE_L]) < 0.05  # the hanging leg followed
    sim.run_for(0.6)  # the stream expired: HOLD at its last target
    st = ctl.status()
    assert st["lower_source"][0] == 0 and st["lower_event"][0] == 17  # HOLD after a STREAM TTL expiry
    assert abs(ctl.last_q[KNEE_L] - goal[KNEE_L]) < 1e-6
    # the operator's stop, then damp (the session end); damp through the mailbox only from the operator
    view = ctl.view
    seq = view.request_mode(REQ_STOP, REQUESTER_R1D, round(sim.time * 1e9))
    sim.run_for(0.01)
    assert view.mode_reply(seq)[0] == 0 and ctl.status()["mode"] == MODE_ON_STAND
    seq = view.request_mode(REQ_DAMP, REQUESTER_R1D, round(sim.time * 1e9))
    sim.run_for(0.01)
    assert view.mode_reply(seq)[:2] == (-6, 20) and ctl.status()["mode"] == MODE_ON_STAND
    seq = view.request_mode(REQ_DAMP, REQUESTER_OPERATOR, round(sim.time * 1e9))
    sim.run_for(0.01)
    assert view.mode_reply(seq)[0] == 0 and ctl.status()["mode"] == MODE_DAMP


def test_the_guard_holds_the_hanging_robot_when_the_core_stalls_in_on_stand(hanging: Any) -> None:
    stall = Fault("controller_stall", T_ON_STAND + 0.3, 1.0, {"name": "v3_core"})
    r = _run(hanging, T_ON_STAND + 1.6, guarded=True, faults=[stall])
    ctl = r["ctl"]
    gs = ctl.guard_status()
    assert gs["takeovers"] == 1 and gs["mode_from"] == MODE_ON_STAND and gs["act_mode"] == MODE_ON_STAND
    assert gs["policy_steps"] == 0  # holding, not balancing
    assert ctl.two_writers == 0 and ctl.core_writes_while_guarded == 0
    gt = ctl.trace_array()
    after = gt[gt[:, 0] >= T_ON_STAND + 0.3]
    assert np.all(np.nan_to_num(after[:, 5], nan=0.0)[after[:, 3] + after[:, 2] > 0] > 30.0)  # stiff legs throughout
    gap = after[(after[:, 2] + after[:, 3]) > 0][0, 0] - (T_ON_STAND + 0.3)
    assert gap <= 0.015  # the guard's first command within 15 ms of the core's silence
    q_meas = _state_q(r)
    q_cmd = ctl.cmd_trace[-1][4]  # the core's last command (the guard holds it)
    assert float(np.max(np.abs(q_meas[:12] - q_cmd[:12]))) < 0.05
    assert r["contacts"] == 0


def test_t6_the_operators_reclaim_on_the_hanging_robot_is_bumpless_and_the_guard_rearms(hanging: Any) -> None:
    """HW1's T6 in lockstep with physics (session 6): the core stalls in ON_STAND (the guard holds), resumes silent (not
    the owner), the operator's RECLAIM through the mailbox gives the token back: the core's first command is the one
    the guard held, the hanging legs do not move at the hand-back, the guard re-arms and catches a second stall.
    Session 5 had no hand-back (the request unknown: fails on 26e8bce)."""
    r1rt_py = cpp_policy.import_r1rt_py()
    t_stall1, t_reclaim, t_stall2 = T_ON_STAND + 0.3, T_ON_STAND + 1.8, T_ON_STAND + 3.3
    faults = [
        Fault("controller_stall", t_stall1, 1.0, {"name": "v3_core"}),
        Fault("controller_stall", t_stall2, 0.5, {"name": "v3_core"}),
    ]
    w, q0 = hanging
    sched = FaultSchedule(faults, seed=1)
    sim = Simulation(w.model, w.index, faults=sched)
    sim.reset(q0)
    builtin = EmulatedBuiltin(w.model, leg_kp=0.0)
    plan = TakeoverPlan(builtin, t_request_s=T_REQUEST, release_s=RELEASE_S, target="on_stand", via_mailbox=True)
    from r1sim.v3_guard import V3GuardedController

    ctl = V3GuardedController(w.model, faults=sched, takeover=plan, trace_cmd=True)
    sim.add_controller(builtin, ctl.writer_period_s, "builtin")
    sim.add_controller(ctl, ctl.period_s, "v3")
    reqs = r1rt_py.requests()
    who = r1rt_py.requesters()
    seqs: dict[str, int] = {}

    def reclaim(c: Any, t_ns: int) -> None:
        seqs["r1d"] = c.view.request_mode(reqs["reclaim"], who["r1d"], t_ns)

    def reclaim_op(c: Any, t_ns: int) -> None:
        seqs["operator"] = c.view.request_mode(reqs["reclaim"], who["operator"], t_ns)

    ctl.at(t_reclaim - 0.1, reclaim)
    ctl.at(t_reclaim, reclaim_op)
    qs: list[tuple[float, np.ndarray]] = []
    contacts = 0
    for _ in range(round((t_stall2 + 1.2) / 0.01)):
        sim.run_for(0.01)
        qs.append((sim.time, np.asarray(sim.read_state().q, dtype=np.float64)[:14].copy()))
        contacts = max(contacts, _foot_floor_contacts(w.model, sim.d))
    assert ctl.view.mode_reply(seqs["r1d"])[:2] == (-6, 20)  # refused: not the operator
    assert ctl.view.mode_reply(seqs["operator"])[:2] == (0, 0)
    gs = ctl.guard_status()
    assert gs["takeovers"] == 2 and gs["rearms"] == 1 and gs["act_mode"] == MODE_ON_STAND
    assert ctl.two_writers == 0 and ctl.core_writes_while_guarded == 0
    assert ctl.status()["reclaims"] == 1
    tr = ctl.cmd_trace
    held = [qq for t, p, _, _, qq, _ in tr if p and t < t_stall1][-1]  # the core's last command before the stall
    first = next((t, qq, kp) for t, p, _, _, qq, kp in tr if p and t >= t_reclaim)
    assert first[0] - t_reclaim < 0.005  # the core publishes at its next tick
    assert float(np.max(np.abs(np.asarray(first[1]) - np.asarray(held)))) < 1e-9  # bumpless: the held command
    assert float(np.min(np.asarray(first[2])[:12])) > 30.0
    win = np.array([qq for t, qq in qs if t_reclaim - 0.2 <= t <= t_reclaim + 1.2])
    assert float(np.max(np.ptp(win[:, :12], axis=0))) < 0.01  # the hanging legs do not move at the hand-back
    gt = ctl.trace_array()
    sent = gt[(gt[:, 2] + gt[:, 3]) > 0]
    assert np.all(np.nan_to_num(sent[:, 5], nan=0.0)[sent[:, 0] >= T_ON_STAND] > 30.0)  # stiff, never limp
    second = gt[(gt[:, 0] >= t_stall2) & (gt[:, 3] > 0)]
    assert len(second) and second[0, 0] - t_stall2 <= 0.015  # the re-armed guard: within 15 ms again
    assert contacts == 0


def _stand_freeze_step(hanging: Any, ramp_s: float) -> dict[str, Any]:
    """HW1 S10 (U3) in lockstep: ON_STAND on the hanging robot, both legs streamed to the stand pose + 0.10 rad on hip
    pitch and knee (ADDITIONS S5 step 1), held, then the operator's STAND_FREEZE with a `ramp_s` ramp: the peak change
    of the applied leg torque against the hold before it (the sim's estimate of what S10 measures on the robot)."""
    from r1sim.v3_core import GROUP_LEG_LEFT, GROUP_LEG_RIGHT, REQ_STAND_FREEZE

    w, q0 = hanging
    sim = Simulation(w.model, w.index, faults=FaultSchedule([], seed=1))
    sim.reset(q0)
    builtin = EmulatedBuiltin(w.model, leg_kp=0.0)
    plan = TakeoverPlan(builtin, t_request_s=T_REQUEST, release_s=RELEASE_S, target="on_stand", via_mailbox=True)
    ctl = V3CoreController(w.model, faults=None, takeover=plan, config={"stand_ramp_s": ramp_s})
    sim.add_controller(builtin, ctl.period_s, "builtin")
    sim.add_controller(ctl, ctl.period_s, "v3")
    sim.run_for(T_ON_STAND + 0.2)
    assert ctl.status()["mode"] == MODE_ON_STAND
    goal = np.asarray(ctl.config["stand_q"], dtype=np.float64)[:12].copy()
    for j in (0, 3, 6, 9):  # hip pitch + knee, both legs (SDK order)
        goal[j] += 0.10
    ids: dict[int, int] = {}

    def stream(c: V3CoreController, t_ns: int) -> None:
        for g, sl in ((GROUP_LEG_LEFT, slice(0, 6)), (GROUP_LEG_RIGHT, slice(6, 12))):
            ids[g] = c.send_stream(t_ns, g, goal[sl], 0.3, cmd_id=ids.get(g))

    t0 = sim.time
    for k in range(20):  # 2 s of renewed streams (<= 2 rad/s), then the hold at the goal
        ctl.at(t0 + 0.1 * k, stream)
    sim.run_for(3.0)
    pre: list[np.ndarray] = []
    for _ in range(50):
        sim.run_for(0.002)
        pre.append(np.asarray(sim.read_state().tau_est, dtype=np.float64)[:12].copy())
    tau_hold = np.mean(pre, axis=0)
    seq = ctl.view.request_mode(REQ_STAND_FREEZE, REQUESTER_OPERATOR, round(sim.time * 1e9))
    peak = np.zeros(12)
    contacts = 0
    for _ in range(500):  # 1 s at 2 ms
        sim.run_for(0.002)
        tau = np.asarray(sim.read_state().tau_est, dtype=np.float64)[:12]
        peak = np.maximum(peak, np.abs(tau - tau_hold))
        contacts = max(contacts, _foot_floor_contacts(w.model, sim.d))
    rep = ctl.view.mode_reply(seq)
    q_end = np.asarray(sim.read_state().q, dtype=np.float64)[:12]
    return {
        "reply": rep,
        "peak": peak,
        "contacts": contacts,
        "q_err": q_end - goal + 0.10 * np.isin(np.arange(12), (0, 3, 6, 9)),
        "mode": ctl.status()["mode"],
    }


def test_s10_the_freeze_ramp_torque_step_on_the_hanging_robot_04_vs_005_s(hanging: Any) -> None:
    """The sim's preview of HW1 S10 (the user's U3: see the real torque step before choosing the takeover-window ramp):
    the same freeze from 0.10 rad off the stand pose with the 0.4 s ramp (the default) and a 0.05 s ramp. Printed for
    S6.md; asserted: both reach the stand pose and stay hanging, the fast ramp's step is the larger."""
    slow = _stand_freeze_step(hanging, 0.4)
    fast = _stand_freeze_step(hanging, 0.05)
    names = ["L hip pitch", "L knee", "R hip pitch", "R knee"]
    idx = [0, 3, 6, 9]
    print("\n[ info ] S10 sim preview: peak |tau - tau_hold| (N m) over the freeze, 0.4 s vs 0.05 s ramp")
    for n, j in zip(names, idx, strict=True):
        print(f"[ info ]   {n:12s} {slow['peak'][j]:7.2f}  {fast['peak'][j]:7.2f}")
    for r in (slow, fast):
        assert r["reply"][0] == 0 and r["mode"] == 5  # STAND_FREEZE
        assert r["contacts"] == 0
        assert float(np.max(np.abs(r["q_err"][idx]))) < 0.03  # reached the stand pose (the legs hang: small sag)
    assert float(np.max(fast["peak"][idx])) > float(np.max(slow["peak"][idx]))


def test_a_core_stall_in_mirror_of_an_on_stand_entry_from_soft_legs_ends_stiff(hanging: Any) -> None:
    """Session 6: the core stalls 0.1 s after the ON_STAND request (MIRROR of the built-in's DAMPED legs, kp 0, while
    ReleaseMode runs) for 3 s. The guard continues the core's blend to the ON_STAND hold gains (DESIGN §4.3 "continues
    the same ramp") instead of holding the soft snapshot for good (80cac26: legs kp 0 until the operator acted - limp by
    our software; fails there): stiff again within the blend length, no policy, feet never touch."""
    stall = Fault("controller_stall", T_REQUEST + 0.1, 3.0, {"name": "v3_core"})
    r = _run(hanging, T_REQUEST + 3.0, guarded=True, faults=[stall])
    ctl = r["ctl"]
    gs = ctl.guard_status()
    assert gs["takeovers"] == 1 and gs["mode_from"] == MODE_TAKEOVER_MIRROR and gs["act_mode"] == MODE_ON_STAND
    assert gs["policy_steps"] == 0
    gt = ctl.trace_array()
    g = gt[(gt[:, 0] >= T_REQUEST + 0.1) & (gt[:, 3] > 0)]  # the guard's commands
    assert len(g) and float(g[0, 5]) < 1.0  # it starts from the soft snapshot (kp 0: what the motors had)
    t_stiff = g[np.argmax(g[:, 5] >= 0.99 * float(g[-1, 5])), 0] - g[0, 0]
    assert 0.9 <= t_stiff <= 1.1  # the whole 1 s blend (a stall in MIRROR: a = 0)
    assert float(g[-1, 5]) > 30.0  # stiff at the end: the ON_STAND hold
    assert ctl.two_writers == 0
    assert r["contacts"] == 0
