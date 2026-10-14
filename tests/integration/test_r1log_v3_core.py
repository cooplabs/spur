"""r1log on the v3 core in lockstep (real profile on the floor, writer 500 Hz, policy thread 50 Hz): the logger drains
the core's own segment (`r1rt_py.SegmentView.of_core`) and the MCAP files hold exactly what the core published.

1. Drained every writer tick: every telemetry record, in order (seq consecutive, one per writer tick), no ring drops,
   every policy record; the logged joint angles are the simulated encoders (float32).
2. Not drained for longer than the ring (4096 records = 8.19 s at 500 Hz): the ring's drop counter, the core's own
   telemetry_drops, and the gap r1log sees in the writer's tick numbers all agree.
3. A NaN policy action -> STAND_FREEZE: the flight recorder dumps its own file around it (the trigger records, the
   window before, the post window after).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.log import Logger, LoggerConfig, read
from r1mc3.log import records as R
from r1mc3.log._native import has_segment_reader, native
from r1sim import world
from r1sim.engine import LowState, Simulation
from r1sim.faults import Fault, FaultSchedule
from r1sim.v3_core import MODE_BALANCE, MODE_STAND_FREEZE, V3CoreController

pytestmark = pytest.mark.skipif(not has_segment_reader(), reason="r1rt_py with SegmentView not built")


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _sim(w: world.World, faults: list[Fault] | None = None) -> tuple[Simulation, V3CoreController]:
    sched = FaultSchedule(faults or [], seed=1)
    sim = Simulation(w.model, w.index, faults=sched)
    sim.reset(w.initial_qpos())
    ctl = V3CoreController(w.model, faults=sched)
    sim.add_controller(ctl, ctl.period_s, "v3")
    return sim, ctl


class _Drain:
    """A lockstep 'controller' registered after the core: drains the rings once per writer tick, sends nothing."""

    def __init__(self, lg: Logger) -> None:
        self.lg = lg
        self.encoders: list[tuple[float, np.ndarray[Any, Any]]] = []

    def __call__(self, st: LowState) -> None:
        self.lg.poll()
        self.encoders.append((float(st.t), np.asarray(st.q, dtype=np.float64).copy()))


def test_every_writer_tick_logged_in_order(w: world.World, tmp_path: Path) -> None:
    sim, ctl = _sim(w)
    view = native().SegmentView.of_core(ctl.core)
    lg = Logger(view, LoggerConfig(out_dir=tmp_path, stamp="T"), source="lockstep test")
    drain = _Drain(lg)
    sim.add_controller(drain, ctl.period_s, "r1log")
    sim.run_for(3.0)
    out = lg.close()
    assert out["telemetry"]["ring_drops"] == 0 and out["telemetry"]["seq_gaps"] == 0
    assert out["policy"]["ring_drops"] == 0 and out["policy"]["seq_gaps"] == 0
    d = read(out["files"][0])
    tel = d.structs["/r1rt/telemetry"]
    calls = sim.controller_calls()["v3"]
    # one record per writer step, none lost: the controller's 1500 ticks + the entry step at t = 0 (v3_core.py: the
    # first call steps again in the same instant after ENTER_BALANCE)
    assert calls == 1500 and len(tel) == calls + 1
    assert np.all(np.diff(tel["seq"].astype(np.int64)) == 1)
    assert tel["t_wake_ns"][0] == tel["t_wake_ns"][1] == 0
    assert np.all(np.diff(tel["t_wake_ns"][1:]) == 2_000_000)  # 500 Hz, in order
    assert np.all(tel["mode"][-100:] == MODE_BALANCE) and np.all(tel["token_owner"][-100:] == 1)
    pol = d.structs["/r1rt/policy"]
    assert 145 <= len(pol) <= 151 and np.all(np.diff(pol["seq"].astype(np.int64)) == 1)
    assert np.all(pol["n_obs"] == 74) and np.all(np.isfinite(pol["action"]))
    # the logged joint angles are the encoder values the core read on that tick (float32 of the state)
    enc = {round(t * 1e9): q for t, q in drain.encoders}
    k = 0
    for r in tel[::97]:
        q = enc.get(int(r["t_wake_ns"]))
        if q is not None:
            np.testing.assert_allclose(r["q"], q.astype(np.float32), atol=1e-6)
            k += 1
    assert k >= 10
    assert d.metadata["r1log"]["abi_version"] == str(native().shm_layout()["abi_version"])
    assert d.metadata["r1log_close"]["telemetry_records"] == "1501"


def test_drop_counter_when_not_drained(w: world.World, tmp_path: Path) -> None:
    sim, ctl = _sim(w)
    view = native().SegmentView.of_core(ctl.core)
    lg = Logger(view, LoggerConfig(out_dir=tmp_path, stamp="T", flight=False))
    sim.run_for(9.0)  # 4500 writer ticks into a 4096-record ring, nobody draining
    lg.poll()
    sim.run_for(0.2)  # 100 more, drained: r1log now sees the hole in the tick numbers
    out = lg.close()
    cap = int(native().shm_layout()["telemetry_capacity"])
    assert cap == 4096
    pushed_total = sim.controller_calls()["v3"] + 1  # + the entry step at t = 0
    assert out["telemetry"]["records"] == cap + 100
    assert out["telemetry"]["ring_drops"] == pushed_total - cap - 100 == 405
    assert out["telemetry"]["seq_gaps"] == out["telemetry"]["ring_drops"]
    assert int(ctl.status()["telemetry_drops"]) == out["telemetry"]["ring_drops"]


def test_flight_recorder_dumps_around_stand_freeze(w: world.World, tmp_path: Path) -> None:
    sim, ctl = _sim(w, [Fault("policy_nan", 2.0, 0.1)])
    view = native().SegmentView.of_core(ctl.core)
    cfg = LoggerConfig(out_dir=tmp_path, stamp="T", flight_window_s=1.0, flight_post_s=0.5)
    lg = Logger(view, cfg)
    sim.add_controller(_Drain(lg), ctl.period_s, "r1log")
    sim.run_for(3.0)
    out = lg.close()
    assert len(out["flight_files"]) == 1, out
    d = read(out["flight_files"][0])
    trig = d.json["/r1log/trigger"]
    reasons = [t["reason"] for t in trig]
    assert any(r.startswith("fault:") and "policy_invalid" in r for r in reasons), reasons
    assert "mode:STAND_FREEZE" in reasons
    t_trig = int(d.metadata["r1log"]["t_trigger_ns"])
    tel = d.structs["/r1rt/telemetry"]
    t = tel["t_wake_ns"]
    assert t.min() >= t_trig - 1_000_000_000 and t.min() <= t_trig - 990_000_000  # the 1 s window before
    assert t.max() >= t_trig + 490_000_000  # the post window after
    assert np.all(np.diff(tel["seq"].astype(np.int64)) == 1)
    assert np.any(tel["mode"] == MODE_STAND_FREEZE) and np.any(tel["mode"] == MODE_BALANCE)
    assert any(e["kind"] == "mode" and e["to"] == "STAND_FREEZE" for e in d.json["/r1log/event"])
    assert R.FAULT_BITS["policy_invalid"] & int(tel["faults"][-1])


def test_the_guards_cycle_records_are_logged(w: world.World, tmp_path: Path) -> None:
    """Session 5 (layout 2): r1guard's own ring -> /r1guard/telemetry, every cycle in order; the takeover after a
    stall of the core is visible in it (event, token owner, the command with stiff legs)."""
    from r1sim.v3_guard import V3GuardedController

    sched = FaultSchedule([Fault("controller_stall", 1.0, 0.5, {"name": "v3_core"})], seed=1)
    sim = Simulation(w.model, w.index, faults=sched)
    sim.reset(w.initial_qpos())
    ctl = V3GuardedController(w.model, faults=sched)
    sim.add_controller(ctl, ctl.period_s, "v3")
    view = native().SegmentView.of_core(ctl.core)
    lg = Logger(view, LoggerConfig(out_dir=tmp_path, stamp="G", flight=False), source="lockstep guard test")
    sim.add_controller(_Drain(lg), ctl.writer_period_s, "r1log")
    sim.run_for(2.0)
    out = lg.close()
    assert out["guard"]["ring_drops"] == 0 and out["guard"]["seq_gaps"] == 0
    d = read(out["files"][0])
    g = d.structs["/r1guard/telemetry"]
    assert len(g) == sim.controller_calls()["v3"]  # one per 1 kHz guard cycle
    assert np.all(np.diff(g["seq"].astype(np.int64)) == 1)
    take = np.flatnonzero(g["event"] == 1)  # kGuardEventTakeover
    assert len(take) == 1
    k = int(take[0])
    t_take = g["t_ns"][k] * 1e-9
    assert 1.0 < t_take <= 1.015
    assert g["active"][k - 1] == 0 and np.all(g["active"][k:] == 1)
    assert np.all(g["token_owner"][k:] == 2) and np.all(g["published"][k:] == 1)
    assert float(np.min(g["kp"][k:, :12])) > 30.0  # never limp
