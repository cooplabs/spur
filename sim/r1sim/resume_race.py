"""#29's open race, the decision (d) in numbers: what a stale pre-stop packet does, and what publishing during the
resume wait would change (session 9; tests/integration/test_v3_resume_race.py is the process version).

After a whole-machine stop, one r1rt packet computed BEFORE the stop can reach the board (the stop landed inside the DDS
write: no user-space check closes that window). The board then holds that stale target - the pre-stop pose at full
stiffness, on limbs that sagged in the board's damping - until the next command replaces it. The core and the guard
publish their next command only once a LIVE lowstate is in (the resume rule, #29): after the stop the DDS receive path
first drains the samples queued during it (macOS, this session: 16-33 samples seen over an 18-34 ms wait after a
0.5-2 s stop; the robot's 4 MB socket buffer holds ~4 s of its 1052/s stream). So the stale target acts for the DRAIN.

This module replays that in lockstep on the bridge's hanging robot (HW1_STAND, the identified board: 0.385 s hold, then
damping kd 12.5; ON_STAND through the core's mailbox), stops the controller `stop_s`, then sends the stale packet (the
core's last pre-stop command) and, for `drain_s`, one of:

* ``nothing``  - today (r1rt / r1guard publish nothing until a live sample): the board keeps the stale target;
* ``damp``     - the board's OWN damping (kp 0, kd = the board's damp_kd, no feed-forward) - what the board was already
                 doing since 0.385 s into the stop - from the next tick until the live sample (NOT LIMP-FREE: needs the
                 user's explicit OK before it could ever be built);
* ``drained``  - a hold at the newest DRAINED sample (the pose the board reported `drain_s` x ... ago, sweeping from
                 the stop's start to now as the queue empties) + the pre-stop hold offset, at the hold gains;

then the core's own resume (the blend from the live measured pose + the offset, over 1 s). Numbers per run: the largest
joint speed on legs / waist / arms in the 0.1 s after the stale packet, the largest motion back toward the old target
in its first 50 ms, the largest 1 ms torque step, and whether any leg was commanded at kp 0 (limp) at any time.

    uv run python -m r1sim.resume_race --stop-s 2.0 --drain-ms 2 5 10 20 35 50
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .engine import LowCmd, LowState
from .faults import Fault, FaultSchedule

BODY = slice(0, 24)  # legs, waist, arms (the head carries ~no load)
LEGS = slice(0, 12)
STALE_TICK_S = 0.002  # the stale packet is the board's command until the next writer tick at least (500 Hz)
SCENARIOS = ("nothing", "damp", "drained")


@dataclass
class _Rec:
    """Every physics tick's truth (t, q, tau) - registered BEFORE the stalled controller, never stalled itself."""

    rows: list[tuple[float, np.ndarray, np.ndarray]] = field(default_factory=list)

    def __call__(self, st: LowState) -> LowCmd | None:
        self.rows.append(
            (
                float(st.t),
                np.asarray(st.truth.get("q", st.q), dtype=np.float64).copy(),
                np.asarray(st.truth.get("tau_est", st.tau_est), dtype=np.float64).copy(),
            )
        )
        return None


class _Race:
    """Wraps the core's lockstep controller (registered as "v3", so the fault schedule stalls it): passes its commands
    through, remembers the last one before the stop, and after the stop plays the stale packet + the scenario."""

    def __init__(self, inner: Any, rec: _Rec, scenario: str, drain_s: float, damp_kd: float) -> None:
        self.inner = inner
        self.rec = rec
        self.scenario = scenario
        self.drain_s = drain_s
        self.damp_kd = damp_kd
        self.last_cmd: LowCmd | None = None
        self.last_t: float | None = None
        self.t_res: float | None = None
        self.t_stop: float | None = None
        self.min_leg_kp = np.inf
        self.sent: list[tuple[float, str]] = []

    def __call__(self, st: LowState) -> LowCmd | None:
        t = float(st.t)
        if self.last_t is not None and t - self.last_t > 0.1 and self.t_res is None:
            self.t_res = t
            self.t_stop = self.last_t
        self.last_t = t
        in_window = self.t_res is not None and t < self.t_res + self.drain_s
        # inside the window the core has no live sample yet (the drain: the "v3_lowstate" stall below): it waits
        out: LowCmd | None = self.inner(st)
        if self.t_res is None:
            if out is not None:
                self.last_cmd = out.copy()
                self.min_leg_kp = min(self.min_leg_kp, float(np.min(out.kp[LEGS])))
            return out
        if not in_window:
            if out is not None:
                self.min_leg_kp = min(self.min_leg_kp, float(np.min(out.kp[LEGS])))
            return out
        assert self.last_cmd is not None
        if t < self.t_res + STALE_TICK_S or self.scenario == "nothing":
            if t < self.t_res + STALE_TICK_S and not any(k == "stale" for _, k in self.sent):
                self.sent.append((t, "stale"))
                return self.last_cmd.copy()  # the stale packet: the pre-stop command
            return None  # nothing new: the board keeps the stale target
        cmd = self.last_cmd.copy()
        if self.scenario == "damp":
            cmd.kp[:] = 0.0
            cmd.kd[:] = self.damp_kd
            cmd.dq[:] = 0.0
            cmd.tau_ff[:] = 0.0
            cmd.q[:] = np.asarray(st.truth.get("q", st.q), dtype=np.float64)  # irrelevant at kp 0
        elif self.scenario == "drained":
            assert self.t_stop is not None
            frac = (t - self.t_res) / self.drain_s
            t_seen = self.t_stop + frac * (self.t_res - self.t_stop)  # the newest drained sample's time
            q_seen = _q_at(self.rec, t_seen)
            q_pre = _q_at(self.rec, self.t_stop)
            off = np.clip(
                self.last_cmd.q - q_pre, -0.15, 0.15
            )  # the pre-stop hold offset (core.cpp kResumeOffsetMaxRad)
            cmd.q[:] = q_seen + off
            cmd.dq[:] = 0.0
        self.min_leg_kp = min(self.min_leg_kp, float(np.min(cmd.kp[LEGS])))
        return cmd


def _q_at(rec: _Rec, t: float) -> np.ndarray:
    ts = np.fromiter((r[0] for r in rec.rows), dtype=np.float64)
    k = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
    return rec.rows[k][1]


def run(stop_s: float = 2.0, drain_s: float = 0.02, scenario: str = "nothing") -> dict[str, Any]:
    """One lockstep run (the setup of tests/integration/test_v3_resume_hw1stand.py) -> the numbers."""
    from .dds_bridge import DdsBridge  # needs the dds extra (the bridge's own hanging robot; DDS never started)
    from .v3_core import EmulatedBuiltin, TakeoverPlan, V3CoreController

    if scenario not in SCENARIOS:
        raise ValueError(f"scenario {scenario!r} not in {SCENARIOS}")
    t_request, release_s, t_on_stand = 1.5, 0.3, 2.8  # test_v3_core_onstand's T_REQUEST / RELEASE_S / T_ON_STAND
    br = DdsBridge(domain=99, scene="none", stance="hanging")
    sim = br.sim
    t0 = sim.time
    t_sil = t0 + t_on_stand + 1.0
    # the stop, then the drain: the core gets no new (live) sample for drain_s after it (its resume wait runs on)
    faults = [Fault("controller_stall", t_sil, stop_s, {"name": "v3"})]
    if drain_s > 0.0:
        faults.append(Fault("controller_stall", t_sil + stop_s, drain_s, {"name": "v3_lowstate"}))
    sim.faults = FaultSchedule(faults, seed=1)
    builtin = EmulatedBuiltin(br.world.model)
    plan = TakeoverPlan(builtin, t_request_s=t0 + t_request, release_s=release_s, target="on_stand", via_mailbox=True)
    ctl = V3CoreController(br.world.model, faults=sim.faults, takeover=plan)
    rec = _Rec()
    race = _Race(ctl, rec, scenario, drain_s, float(np.asarray(sim.driver.damp_kd).max()))
    sim.add_controller(builtin, ctl.period_s, "builtin")
    sim.add_controller(rec, sim.dt, "rec")
    sim.add_controller(race, ctl.period_s, "v3")
    while sim.time < t_sil + stop_s + 1.2:
        sim.run_for(0.001)
    t = np.array([r[0] for r in rec.rows])
    q = np.array([r[1] for r in rec.rows])
    tau = np.array([r[2] for r in rec.rows])
    assert race.t_res is not None and race.t_stop is not None and race.last_cmd is not None
    t_res = race.t_res
    dq = np.vstack([np.zeros((1, q.shape[1])), np.diff(q, axis=0) / np.maximum(np.diff(t), 1e-9)[:, None]])
    k0 = int(np.flatnonzero(t <= t_res)[-1])
    q0 = q[k0, BODY]
    toward = np.sign(race.last_cmd.q[BODY] - q0)
    w50 = (t >= t_res) & (t <= t_res + 0.05)
    w100 = (t >= t_res) & (t <= t_res + 0.1)
    win = (t >= t_res - 0.002) & (t <= t_res + 0.3)
    dtau = np.abs(np.diff(tau[:, BODY], axis=0))[win[1:]]
    pre = q[t <= race.t_stop][-1, BODY]
    return {
        "scenario": scenario,
        "stop_s": stop_s,
        "drain_ms": round(drain_s * 1e3, 1),
        "sag_max": round(float(np.max(np.abs(q0 - pre))), 4),
        "dq_peak": round(float(np.max(np.abs(dq[w100][:, BODY]))), 3),
        "kick_rad": round(float(np.max((q[w50][:, BODY] - q0) * toward)), 4),
        "step_nm": round(float(dtau.max()), 2),
        "min_leg_kp": round(float(race.min_leg_kp), 1),
        "end_err": round(float(np.abs(q[-1, BODY] - pre).max()), 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stop-s", type=float, nargs="+", default=[2.0])
    ap.add_argument("--drain-ms", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0, 35.0, 50.0])
    ap.add_argument("--scenario", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    a = ap.parse_args(argv)
    for s in a.stop_s:
        for d in a.drain_ms:
            for sc in a.scenario:
                print(json.dumps(run(s, d * 1e-3, sc)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
