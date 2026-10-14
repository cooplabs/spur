"""The upper-body executor inside the v3 core (r1rt::Core + r1rt::UpperBodyExecutor, C++, through r1rt_py.Core) while
the core balances the simulated robot in lockstep (real profile on the floor, writer 500 Hz, policy thread 50 Hz;
r1sim.v3_core.V3CoreController plays r1d: records into the segment's upper_cmd ring, keepalives, the stop channel).
DESIGN.md §3.3 (executor), §4.2-§4.3 (stops, STAND_FREEZE: arms HOLD, never limp).

1. An arm TRAJECTORY out and back: each arm from its pose at BALANCE entry to "out to the side, elbow bent, wrist
   turned" (shoulder roll +-0.5, elbow +0.5, wrist roll -+1.0 rad - the wrist turn of table_ready) as ONE trajectory of
   two minimum-jerk legs (15 knots in two records: multi-segment). The logged table_ready arm pose of the 9 Oct stall
   is NOT used as the goal: from this stance every straight joint-space path to it (or through the via poses tried)
   puts the hand or elbow on the table edge (s3 `wire` probes: the table is close in front; the executor follows
   paths, r1d's planner plans them). The commanded target equals the planned path (the knots are exact samples of it)
   to 1e-9 rad; the MEASURED arm follows within a stated error, with the built-in arm model's gravity feed-forward
   (DESIGN §3.4, session 4: r1rt::arm::gravity_ff26 -> the executor) - without it (session 3) the arm sagged 0.081 rad
   at the goal and lagged 0.107 rad on the way; the robot stays upright (tilt < 1 deg); the move ends "done" and the
   group HOLDs.
2. The keepalive withheld mid-move: the deadman (50 ms) stops the arm within stop_time (0.15 s) - velocity smoothly to
   zero, never reversing (no overshoot), then HOLD; balance continues.
3. A STREAM whose setpoints stop coming: TTL expiry -> decelerate -> HOLD.
4. An operator stop (the stop channel) during two arm moves: every group stops and HOLDs, balance continues; a record
   issued before the stop (older stop epoch) is refused, a new one accepted.
5. STAND_FREEZE (a NaN action) during an arm move: the arm stops and HOLDs at its hold gains, the robot stays up.
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
from r1sim.v3_core import (
    EVENTS,
    GROUP_ARM_LEFT,
    GROUP_ARM_RIGHT,
    GROUP_JOINTS,
    MODE_BALANCE,
    MODE_DAMP,
    MODE_STAND_FREEZE,
    SOURCE_HOLD,
    SOURCE_STREAM,
    SOURCE_TRAJECTORY,
    STOP_STOP,
    V3CoreController,
    min_jerk_path_sample,
)

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")

EV = {v: k for k, v in EVENTS.items()}
MIRROR = np.array([1.0, -1.0, -1.0, 1.0, -1.0])  # left -> right arm: roll, yaw, wrist roll change sign
# out to the side, elbow bent, wrist turned (left arm; table-clear in dynamics: s3 `wire` probe, tilt 0.46 deg)
SIDE_LEFT = np.array([0.0, 0.5, 0.0, 0.5, -1.0])
SIDE = {GROUP_ARM_LEFT: SIDE_LEFT, GROUP_ARM_RIGHT: SIDE_LEFT * MIRROR}
LEG_S = 1.5  # per leg (two legs of half the move: peak 1.875 x 0.5 / 1.5 = 0.63 rad/s; dq_max 2 rad/s)
MOVE_S = 3.0  # single-leg moves of the stop tests
TRACK_TOL_RAD = 0.07  # measured-tracking bound with the gravity ff (kp 60/45/30; 0.051 measured, 0.107 without)
STEADY_TOL_RAD = 0.03  # steady offset at the goal with the gravity ff (0.012 measured; 0.081 without it)
STOP_TIME_S = 0.15
TICK_S = 0.002
COL = {"t": 0, "mode": 1, "stop": 2, "src": 3, "ev": 6, "stopping": 9, "q": 12}  # V3CoreController.upper_trace


def _cols(group: int) -> list[int]:
    return [COL["q"] + j - 14 for j in GROUP_JOINTS[group]]


def _tilt_deg(quat: np.ndarray) -> float:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(r[8], -1.0, 1.0))))


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _run(w: world.World, ctl_setup: Any, seconds: float, faults: list[Fault] | None = None) -> dict[str, Any]:
    sched = FaultSchedule(faults or [], seed=1)
    sim = Simulation(w.model, w.index, faults=sched)
    sim.reset(w.initial_qpos(J.POSE_STAND.copy()))
    ctl = V3CoreController(w.model, faults=sched, trace_upper=True)
    ctl_setup(ctl)
    sim.add_controller(ctl, ctl.period_s, "v3")
    pel = w.index.pelvis_body
    qadr = {j: int(w.model.jnt_qposadr[w.model.joint(J.JOINTS[j] + "_joint").id]) for j in range(14, 26)}
    rows: list[tuple[float, float, np.ndarray]] = []
    for _ in range(round(seconds / 0.01)):
        sim.run_for(0.01)
        rows.append((sim.time, _tilt_deg(sim.d.xquat[pel]), np.array([sim.d.qpos[qadr[j]] for j in range(14, 26)])))
    tr = np.array(ctl.upper_trace)
    return {
        "ctl": ctl,
        "tr": tr,
        "rows": rows,
        "status": ctl.status(),
        "max_tilt": max(a for t, a, _ in rows if t >= 1.0),
    }


@pytest.mark.parametrize("group", [GROUP_ARM_LEFT, GROUP_ARM_RIGHT], ids=["left", "right"])
def test_arm_trajectory_out_to_the_side_and_back(w: world.World, group: int) -> None:
    info: dict[str, Any] = {}
    cols = _cols(group)
    durs = [LEG_S, LEG_S]
    dur = sum(durs)
    points: list[np.ndarray] = []

    def setup(ctl: V3CoreController) -> None:
        def out(c: V3CoreController, t_ns: int) -> None:
            info["q0"] = c.last_q[list(GROUP_JOINTS[group])].copy()
            info["t0"] = t_ns * 1e-9
            points[:] = [info["q0"] + 0.5 * SIDE[group], info["q0"] + SIDE[group]]
            c.send_path(t_ns, group, points, durs)

        def back(c: V3CoreController, t_ns: int) -> None:
            info["t1"] = t_ns * 1e-9
            c.send_path(t_ns, group, [points[0], info["q0"]], durs)

        ctl.at(1.0, out)
        ctl.at(1.0 + dur + 1.0, back)

    r = _run(w, setup, 1.0 + 2 * dur + 2.0)
    tr, s = r["tr"], r["status"]
    n_rec = len(r["ctl"].push_results)
    assert all(rc == 0 for rc in r["ctl"].push_results) and s["upper_accepted"] == n_rec and s["upper_rejected"] == 0
    goal = points[-1]
    legs = [(info["q0"], points, info["t0"]), (goal, [points[0], info["q0"]], info["t1"])]
    # the commanded target = the planned path (tick by tick), out and back
    err_cmd = 0.0
    for k, (a, pts, t0) in enumerate(legs):
        sel = (tr[:, 0] >= t0) & (tr[:, 0] <= t0 + dur)
        assert np.all(tr[sel, COL["src"] + group] == SOURCE_TRAJECTORY), k
        for row in tr[sel]:
            want = min_jerk_path_sample(a, pts, durs, row[0] - t0)
            err_cmd = max(err_cmd, float(np.abs(row[cols] - want).max()))
    # the measured arm (physics) against the planned path, and the steady offset at the goal (gravity ff on)
    jj = [j - 14 for j in GROUP_JOINTS[group]]
    err_meas = 0.0
    for t, _, qm in r["rows"]:
        for a, pts, t0 in legs:
            if t0 <= t <= t0 + dur:
                err_meas = max(err_meas, float(np.abs(qm[jj] - min_jerk_path_sample(a, pts, durs, t - t0)).max()))
    at_goal = next(qm[jj] for t, _, qm in r["rows"] if t >= info["t0"] + dur + 0.9)
    print(
        f"group {group}: {n_rec} records, |goal - start| max {np.abs(goal - info['q0']).max():.3f} rad; commanded "
        f"vs planned {err_cmd:.2e} rad; measured vs planned max {err_meas:.4f} rad; steady offset at the goal "
        f"{np.round(at_goal - goal, 4)} rad; max tilt {r['max_tilt']:.3f} deg"
    )
    assert err_cmd < 1e-9
    assert err_meas < TRACK_TOL_RAD
    assert float(np.abs(at_goal - goal).max()) < STEADY_TOL_RAD  # fails without the gravity ff (0.081 rad)
    assert np.allclose(tr[-1, cols], info["q0"], atol=1e-9)  # back where it started, exactly (commanded)
    assert tr[-1, COL["src"] + group] == SOURCE_HOLD and tr[-1, COL["ev"] + group] == EV["done"]
    assert np.all(tr[:, COL["mode"]] == MODE_BALANCE) and s["faults_seen"] == 0
    assert r["max_tilt"] < 1.0


def test_keepalive_withheld_stops_the_arm_smoothly_then_holds(w: world.World) -> None:
    group, cols = GROUP_ARM_LEFT, _cols(GROUP_ARM_LEFT)
    t_cut = 2.2  # the last renewal is at or before this (mid-move: the min-jerk speed is near its peak)

    def setup(ctl: V3CoreController) -> None:
        ctl.at(
            1.0,
            lambda c, t_ns: c.send_trajectory(
                t_ns, group, c.last_q[14:19] + SIDE_LEFT, MOVE_S, keepalive_until_s=t_cut
            ),
        )

    r = _run(w, setup, 4.0)
    tr = r["tr"]
    ev = tr[:, COL["ev"] + group]
    k_dm = int(np.argmax(ev == EV["deadman"]))
    assert ev[k_dm] == EV["deadman"]
    t_dm = tr[k_dm, 0]
    hold = (tr[:, 0] > t_dm) & (tr[:, COL["src"] + group] == SOURCE_HOLD) & (tr[:, COL["stopping"] + group] == 0)
    t_hold = float(tr[np.argmax(hold), 0])
    q = tr[:, cols]
    v = np.diff(q, axis=0) / TICK_S
    v0 = v[k_dm - 2]  # the commanded velocity just before the deadman fired
    during = slice(k_dm - 1, int(np.argmax(hold)) + 50)
    reversed_ = np.any(np.sign(v[during]) * np.sign(v0) < 0) and np.any(np.abs(v[during]) > 1e-9)
    travel = np.abs(q[np.argmax(hold)] - q[k_dm - 1])
    print(
        f"deadman at {t_dm:.3f} s (last renewal <= {t_cut} s, + 50 ms); HOLD at {t_hold:.3f} s "
        f"({(t_hold - t_dm) * 1e3:.0f} ms); |v0| max {np.abs(v0).max():.3f} rad/s; stop travel {np.round(travel, 4)}"
    )
    assert t_cut + 0.05 - 0.021 <= t_dm <= t_cut + 0.05 + 0.003
    assert t_hold - t_dm <= STOP_TIME_S + 2 * TICK_S
    assert not reversed_
    assert np.all(travel <= np.abs(v0) * STOP_TIME_S / 2 + 1e-6)  # v0 T / 2: the profile's travel, no overshoot
    assert np.all(q[-1] == q[np.argmax(hold)])  # then it holds still
    assert np.all(tr[:, COL["mode"]] == MODE_BALANCE) and r["max_tilt"] < 1.0


def test_stream_ttl_expiry_decelerates_to_hold(w: world.World) -> None:
    group, cols = GROUP_ARM_RIGHT, _cols(GROUP_ARM_RIGHT)
    state: dict[str, Any] = {}

    def setup(ctl: V3CoreController) -> None:
        def stream(c: V3CoreController, t_ns: int) -> None:
            if "q0" not in state:
                state["q0"] = c.last_q[list(GROUP_JOINTS[group])].copy()
            goal = state["q0"] + np.array([-0.3, 0.0, 0.0, 0.3, 0.0])
            state["id"] = c.send_stream(t_ns, group, goal, 0.1, cmd_id=state.get("id"))
            state["last"] = t_ns * 1e-9

        for k in range(26):  # setpoints every 20 ms from 1.0 to 1.5 s, then nothing
            ctl.at(1.0 + 0.02 * k, stream)

    r = _run(w, setup, 2.5)
    tr = r["tr"]
    src = tr[:, COL["src"] + group]
    ev = tr[:, COL["ev"] + group]
    t_exp = float(tr[np.argmax(ev == EV["stream_ttl"]), 0])
    hold = (tr[:, 0] > t_exp) & (src == SOURCE_HOLD) & (tr[:, COL["stopping"] + group] == 0)
    t_hold = float(tr[np.argmax(hold), 0])
    print(f"last setpoint {state['last']:.3f} s, TTL expiry at {t_exp:.3f} s, HOLD at {t_hold:.3f} s")
    assert np.all(src[(tr[:, 0] > 1.01) & (tr[:, 0] < state["last"])] == SOURCE_STREAM)
    assert state["last"] + 0.1 - 1e-9 <= t_exp <= state["last"] + 0.1 + TICK_S + 1e-9
    assert t_hold - t_exp <= STOP_TIME_S + 2 * TICK_S
    q = tr[:, cols]
    assert np.all(q[-1] == q[np.argmax(hold)])
    assert np.all(tr[:, COL["mode"]] == MODE_BALANCE) and r["max_tilt"] < 1.0


def test_operator_stop_holds_every_group_and_refuses_older_records(w: world.World) -> None:
    info: dict[str, Any] = {}

    def setup(ctl: V3CoreController) -> None:
        def moves(c: V3CoreController, t_ns: int) -> None:
            info["epoch0"] = int(c.core.stop_epoch())
            c.send_trajectory(t_ns, GROUP_ARM_LEFT, c.last_q[14:19] + SIDE[GROUP_ARM_LEFT], MOVE_S)
            c.send_trajectory(t_ns, GROUP_ARM_RIGHT, c.last_q[19:24] + SIDE[GROUP_ARM_RIGHT], MOVE_S)

        def stop(c: V3CoreController, t_ns: int) -> None:
            c.core.stop_channel(STOP_STOP, 9, t_ns)  # the operator's stop (any client)
            info["t_stop"] = t_ns * 1e-9

        def stale(c: V3CoreController, t_ns: int) -> None:  # r1d had not seen the stop yet: the old epoch
            q = c.last_q[list(GROUP_JOINTS[GROUP_ARM_LEFT])]
            info["rc_stale"] = c._push(
                {"cmd_id": 99, "group": GROUP_ARM_LEFT, "source": SOURCE_HOLD, "n_joints": 5, "q": list(q),
                 "stop_epoch": info["epoch0"]}
            )  # fmt: skip

        def fresh(c: V3CoreController, t_ns: int) -> None:
            info["fresh_id"] = c.send_hold(t_ns, GROUP_ARM_LEFT, c.last_q[list(GROUP_JOINTS[GROUP_ARM_LEFT])])

        ctl.at(1.0, moves)
        ctl.at(2.0, stop)
        ctl.at(2.5, stale)
        ctl.at(2.6, fresh)

    r = _run(w, setup, 3.5)
    tr, s = r["tr"], r["status"]
    after = tr[:, 0] >= info["t_stop"] + STOP_TIME_S + 2 * TICK_S
    for g in (GROUP_ARM_LEFT, GROUP_ARM_RIGHT, 2):
        assert np.all(tr[after, COL["src"] + g] == SOURCE_HOLD), g
        assert np.all(tr[after, COL["stopping"] + g] == 0), g
    for g in (GROUP_ARM_LEFT, GROUP_ARM_RIGHT):
        ev = tr[(tr[:, 0] >= info["t_stop"]) & (tr[:, 0] < 2.5), COL["ev"] + g]
        assert np.all(ev == EV["operator_stop"]), g
    print(
        f"stop at {info['t_stop']:.3f} s: every group HOLD by +{STOP_TIME_S + 2 * TICK_S:.3f} s; stale record "
        f"rejected (reason {s['upper_last_reject']}), fresh accepted; accepted {s['upper_accepted']}, rejected "
        f"{s['upper_rejected']}; stop epoch {s['upper_stop_epoch']}; max tilt {r['max_tilt']:.3f} deg"
    )
    assert info["rc_stale"] == 0  # the ring took it; the executor refuses it
    assert s["upper_rejected"] == 1 and s["upper_last_reject"] == 7  # kRejectStopEpoch
    assert s["upper_accepted"] == 3 and s["upper_cmd_id"][GROUP_ARM_LEFT] == info["fresh_id"]
    assert s["stop_active"] == 1
    assert np.all(tr[:, COL["mode"]] == MODE_BALANCE) and r["max_tilt"] < 1.0


def test_stand_freeze_stops_the_arm_and_holds_it(w: world.World) -> None:
    group, cols = GROUP_ARM_LEFT, _cols(GROUP_ARM_LEFT)
    info: dict[str, Any] = {}

    def setup(ctl: V3CoreController) -> None:
        def move(c: V3CoreController, t_ns: int) -> None:
            info["goal"] = c.last_q[14:19] + SIDE_LEFT
            c.send_trajectory(t_ns, group, info["goal"], MOVE_S)

        ctl.at(1.0, move)

    r = _run(w, setup, 7.0, [Fault("policy_nan", 2.0, 0.1)])
    tr, s = r["tr"], r["status"]
    frozen = tr[:, COL["mode"]] == MODE_STAND_FREEZE
    t_freeze = float(tr[np.argmax(frozen), 0])
    hold = (tr[:, 0] > t_freeze) & (tr[:, COL["src"] + group] == SOURCE_HOLD) & (tr[:, COL["stopping"] + group] == 0)
    t_hold = float(tr[np.argmax(hold), 0])
    q = tr[:, cols]
    print(
        f"NaN at 2.0 s: STAND_FREEZE at {t_freeze:.3f} s, arm HOLD at {t_hold:.3f} s; arm stopped "
        f"{np.round(np.abs(q[-1] - info['goal']), 3)} rad short of the goal; max tilt {r['max_tilt']:.2f} deg"
    )
    assert 2.0 <= t_freeze <= 2.022
    assert np.all(frozen[tr[:, 0] >= t_freeze]) and not np.any(tr[:, COL["mode"]] == MODE_DAMP)
    assert tr[np.argmax(frozen), COL["ev"] + group] == EV["operator_stop"]
    assert t_hold - t_freeze <= STOP_TIME_S + 2 * TICK_S
    assert np.all(q[-1] == q[np.argmax(hold)])
    assert s["stop_active"] == 1 and s["freeze_entries"] == 1
    assert r["max_tilt"] < 5.0  # the S3 rtcore pin for STAND_FREEZE
