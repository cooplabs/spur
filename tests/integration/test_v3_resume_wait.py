"""#29's open race (session 9): the resume wait ends at the FIRST live sample (no 10 ms floor with a known tick rate).

The race (tests/integration/test_v3_resume_race.py): a whole-machine stop inside r1rt's write path lets ONE pre-stop
packet out after the stop; the board keeps that stale target until our next command, and after a stop r1rt and r1guard
publish nothing until a live lowstate is in. Session 8 waited at least R1RT_RESUME_SETTLE_NS (10 ms) even when the
sample in hand was provably current (its board tick caught up with the tick extrapolated from before the stop:
r1rt::ResumeClock); now that floor applies only while no tick rate has been learned.

Lockstep (the same hanging robot + the board model as test_v3_resume.py): the state is current at the first tick after
the stall, so the first command after a 2 s stop - the core's own resume blend, or the guard's after a whole-machine
stall - comes at that tick. Base (6449aee): 10 ms of silence first (5 writer ticks).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1sim import cpp_policy

from tests.integration.test_v3_resume import T_SILENCE, _run, hanging  # noqa: F401  (the module fixture)

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py is not built (scripts/ci.sh cpp)")

STOP_S = 2.0
FIRST_CMD_MAX_S = 0.0045  # <= 2 writer ticks after the stall's end (base: >= 0.010)


def test_the_core_publishes_its_resume_blend_at_the_first_live_sample(hanging: Any) -> None:  # noqa: F811
    r = _run(hanging, STOP_S)
    ctl = r["ctl"]
    t_res = T_SILENCE + STOP_S
    rows = [(t, pub) for (t, pub, *_rest) in ctl.cmd_trace if t >= t_res - 1e-9]
    t_first = next(t for t, pub in rows if pub)
    st = ctl.status()
    print(f"\n[ info ] the core's first command {1e3 * (t_first - t_res):.1f} ms after a {STOP_S} s stop")
    assert st["resume_blends"] == 1 and st["resume_uncaught"] == 0, st
    assert t_first - t_res <= FIRST_CMD_MAX_S, (t_first - t_res, st)  # base: 10 ms of nothing
    assert r["step_nm"] < 2.0, r  # still the blend from the measured pose (no snap)


def test_the_guard_takes_over_a_whole_machine_stop_with_its_first_live_sample(hanging: Any) -> None:  # noqa: F811
    r = _run(hanging, STOP_S, guarded="both")
    ctl = r["ctl"]
    t_res = T_SILENCE + STOP_S
    g = ctl.trace_array()  # (t, core_tick, core_pub, guard_pub, guard_active, kmin)
    after = g[g[:, 0] >= t_res - 1e-9]
    pubs = after[(after[:, 2] > 0) | (after[:, 3] > 0)]
    t_first = float(pubs[0, 0])
    print(f"\n[ info ] the first command after a {STOP_S} s whole-machine stop {1e3 * (t_first - t_res):.1f} ms")
    assert ctl.two_writers == 0 and ctl.core_writes_while_guarded == 0
    assert t_first - t_res <= FIRST_CMD_MAX_S, t_first - t_res  # base: 10 ms of nothing
    assert np.all(after[:, 5] > 0.0), "a limp command"  # never limp
    assert r["step_nm"] < 2.0, r


def test_the_decision_d_tool_shows_what_the_drain_costs() -> None:
    """r1sim.resume_race (the numbers behind the user's decision (d), core_RACE_NUMBERS.md): the same stale packet after
    a 2 s stop, then 20 ms of drain - publishing nothing lets it kick the sagged hips back; the board's own damping
    during the drain would not (but commands kp 0 = limp: the user's call, NOT built into r1rt / r1guard)."""
    from r1mc3 import dds

    if not dds.available():
        pytest.skip("the bridge's robot needs the dds extra")
    from r1sim import resume_race

    nothing = resume_race.run(2.0, 0.020, "nothing")
    damp = resume_race.run(2.0, 0.020, "damp")
    ideal = resume_race.run(2.0, 0.002, "nothing")  # the drain as short as one writer tick
    print(f"\n[ info ] {nothing}\n[ info ] {damp}\n[ info ] {ideal}")
    assert nothing["sag_max"] > 0.1 and nothing["min_leg_kp"] > 0.0
    # numbers on the legs' identified armature 0.008 (session 9, merged after core measured on 0.05 - in brackets):
    # nothing 3.64 rad/s, 0.066 rad (2.75, 0.054); damp 1.00, 0.0035 (0.44, 0.003); ideal 1.00, 0.0083 (0.44, 0.005)
    assert nothing["dq_peak"] > 2.0 and nothing["kick_rad"] > 0.04
    assert damp["dq_peak"] < 1.5 and damp["kick_rad"] < 0.006 and damp["min_leg_kp"] == 0.0  # LIMP: not built
    assert ideal["dq_peak"] < 1.5 and ideal["kick_rad"] < 0.012  # one tick of the stale packet
    assert damp["dq_peak"] < 0.5 * nothing["dq_peak"] and ideal["kick_rad"] < 0.25 * nothing["kick_rad"]
