"""Layout 4 (session 10, core): the executor status region and the base-command writer of r1rt's segment, in lockstep
(the C++ core in the simulator, read through `r1rt_py.SegmentView` exactly as r1d reads r1rt's POSIX segment).

* `SegmentView.exec_status()` says what the core's own status says (per group: the record followed, the source, the
  last event; the refusals; the state-age counters) - so `ShmPort.upper_groups()` gives r1d on real processes what
  `LockstepPort.upper_groups()` gives it in lockstep, and nothing once r1rt's status is older than 100 ms (r1rt
  stopped: its last events would describe the moment it stopped);
* `SegmentView.set_base_cmd()` writes the base command the core reads (the policy's lean follows it; a TTL above 0.5 s
  is refused) - `ShmPort.set_base_cmd()` = body.pose on real processes.

On the base (a041731) SegmentView had neither (AttributeError) and ShmPort.upper_groups() returned None always.
The process-level proof (r1rt / r1guard / r1d on the DDS bridge): test_service_robot_events.py."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1sim import cpp_policy, world
from r1sim.engine import Simulation
from r1sim.model import centred_stand_pose
from r1sim.v3_core import GROUP_ARM_LEFT, SOURCE_TRAJECTORY, V3CoreController

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py is not built (scripts/ci.sh cpp)")


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _mod() -> Any:
    return cpp_policy.import_r1rt_py()


def _balancing(w: world.World) -> tuple[Simulation, V3CoreController]:
    sim = Simulation(w.model, w.index)
    sim.reset(w.initial_qpos(centred_stand_pose(w.model)))
    ctl = V3CoreController(w.model)
    sim.add_controller(ctl, ctl.period_s, "v3")
    sim.run_for(0.3)
    return sim, ctl


def _arm_move(ctl: V3CoreController, t_ns: int, cmd_id: int, delta: float, epoch: int) -> dict[str, Any]:
    """r1d's own TRAJECTORY record (goals.records_for): the left shoulder pitch + delta in 0.4 s, minimum jerk."""
    from r1mc3.service.goals import min_jerk_knots, records_for

    q0 = ctl.last_q[14:19].copy()
    q1 = q0.copy()
    q1[0] += delta
    (rec,) = records_for(cmd_id, GROUP_ARM_LEFT, 5, min_jerk_knots(q0, q1, 0.4), t_ns, epoch)
    return rec


def test_the_exec_status_says_what_the_core_status_says(w: world.World, monkeypatch: pytest.MonkeyPatch) -> None:
    from r1mc3.service import port as port_mod
    from r1mc3.service.port import ShmPort

    sim, ctl = _balancing(w)
    mod = _mod()
    view = mod.SegmentView.of_core(ctl.core)
    shm_port = ShmPort(view, mod)
    monkeypatch.setattr(port_mod, "monotonic_ns", lambda: round(sim.time * 1e9))  # the lockstep core's clock

    def check(when: str) -> None:
        es = view.exec_status()
        st = ctl.status()
        assert es is not None, when
        assert es["upper_accepted"] == st["upper_accepted"] and es["upper_rejected"] == st["upper_rejected"], when
        assert es["last_reject"] == st["upper_last_reject"], when
        assert es["cmd_id"][:3] == list(st["upper_cmd_id"]), when
        assert es["source"][:3] == list(st["upper_source"]), when
        assert es["event"][:3] == list(st["upper_event"]), when
        assert es["state_old_ticks"] == st["state_old_ticks"], when
        assert es["state_content_age_ns"] == st["state_content_age_ns"], when
        ug = shm_port.upper_groups()
        assert ug == {
            "source": list(st["upper_source"]),
            "event": list(st["upper_event"]),
            "cmd_id": list(st["upper_cmd_id"]),
            "rejected": [st["upper_rejected"]],
            "last_reject": [st["upper_last_reject"]],
            "t_ns": [es["t_ns"]],
        }, when

    check("idle")
    t_ns = round(sim.time * 1e9)
    assert view.push_upper(_arm_move(ctl, t_ns, 41, 0.10, view.stop_epoch())) == 0
    for _ in range(10):  # r1d's keepalive every 50 ms, in the simulated clock
        view.upper_keepalive(GROUP_ARM_LEFT, 41, round(sim.time * 1e9))
        sim.run_for(0.02)
    check("running")
    assert view.exec_status()["cmd_id"][GROUP_ARM_LEFT] == 41
    assert view.exec_status()["source"][GROUP_ARM_LEFT] == SOURCE_TRAJECTORY
    for _ in range(25):
        view.upper_keepalive(GROUP_ARM_LEFT, 41, round(sim.time * 1e9))
        sim.run_for(0.02)
    check("done")
    assert view.exec_status()["event"][GROUP_ARM_LEFT] == 32  # kEventDone: the path's end, not a stop
    # a record stamped before a stop is refused: the count + the reason reach the segment
    old = view.stop_epoch()
    view.request_stop(0, 2, round(sim.time * 1e9))
    sim.run_for(0.02)
    assert view.push_upper(_arm_move(ctl, round(sim.time * 1e9), 42, 0.05, old)) == 0
    sim.run_for(0.02)
    check("refused")
    assert view.exec_status()["upper_rejected"] == 1
    # the state-age counters under the names r1d's `r1 status` words read from rt_info() (the stale agent's ask)
    info = view.rt_info()
    st = ctl.status()
    assert info["stale_rung"] == 0  # the robot profile's D34 value (OFF until after HW2's log check)
    assert info["state_old_ticks"] == st["state_old_ticks"] and info["state_old_episodes"] == st["state_old_episodes"]
    assert info["content_age_valid"] is bool(st["content_age_valid"]) and info["stale_freeze"] is False
    assert info["state_content_age_max_ms"] == pytest.approx(st["state_content_age_max_ns"] * 1e-6)
    # r1rt silent (its status older than 100 ms): no per-group answer (r1d falls back to the mirror / the token)
    monkeypatch.setattr(port_mod, "monotonic_ns", lambda: round(sim.time * 1e9) + 150_000_000)
    assert shm_port.upper_groups() is None


def test_the_base_command_through_the_segment_view(w: world.World) -> None:
    from r1mc3.service.port import ShmPort

    sim, ctl = _balancing(w)
    mod = _mod()
    view = mod.SegmentView.of_core(ctl.core)
    shm_port = ShmPort(view, mod)
    assert view.base_cmd() is None
    lean = float(np.radians(3.0))
    seq = 0
    for _ in range(100):  # r1d renews it at its tick (TTL 0.5 s); 2 s
        seq += 1
        assert shm_port.set_base_cmd(seq, round(sim.time * 1e9), 500_000_000, 0.0, 0.0, 0.0, lean) == 0
        sim.run_for(0.02)
    assert abs(ctl.status()["lean"] - lean) < 0.005  # the policy's ramped lean reached the command (3 deg +- 0.3)
    assert view.mirror()["base_cmd"][3] == pytest.approx(ctl.status()["lean"])  # what r1d's status reads
    writes, raw = view.base_cmd()
    base_cmd = [("seq", "<u8"), ("t", "<i8"), ("ttl", "<i8"), ("v", "<f8", 3), ("lean", "<f8"), ("waist", "<f8", 2),
                ("req", "<u4"), ("_p", "<u4")]  # fmt: skip
    rec = np.frombuffer(raw, dtype=np.dtype(base_cmd))  # types.h r1rt_base_cmd (80 B)
    assert int(rec["seq"][0]) == seq and float(rec["lean"][0]) == pytest.approx(lean) and writes == seq
    with pytest.raises(ValueError):
        view.set_base_cmd(seq + 1, round(sim.time * 1e9), 600_000_000, 0.0, 0.0, 0.0, lean)  # TTL > 0.5 s
    # r1d silent: the command expires (TTL) and the lean ramps back to upright
    sim.run_for(2.5)
    assert abs(ctl.status()["lean"]) < 0.005
