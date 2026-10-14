"""r1sim.faults: the schedule (DESIGN.md §12.6) - windows, validation, per-kind random streams, the log."""

from __future__ import annotations

import numpy as np
import pytest
from r1sim.faults import FAULT_KINDS, Fault, FaultSchedule


def test_window_is_half_open() -> None:
    f = Fault("cmd_gap", 1.0, 0.5)
    assert not f.active(0.999) and f.active(1.0) and f.active(1.499) and not f.active(1.5)
    assert f.t_end == 1.5


@pytest.mark.parametrize(
    ("kind", "t0", "dur"),
    [("bogus", 0.0, 1.0), ("cmd_gap", 0.0, 0.0), ("cmd_gap", float("nan"), 1.0), ("cmd_gap", 0.0, -1.0)],
)
def test_invalid_faults(kind: str, t0: float, dur: float) -> None:
    with pytest.raises(ValueError):
        Fault(kind, t0, dur)  # type: ignore[arg-type]


def test_active_filters_by_kind_and_time() -> None:
    s = FaultSchedule([Fault("cmd_gap", 0.0, 1.0), Fault("cmd_drop", 0.5, 1.0), Fault("cmd_gap", 2.0, 1.0)])
    assert [f.t_start for f in s.active("cmd_gap", 0.7)] == [0.0]
    assert [f.t_start for f in s.active("cmd_drop", 0.7)] == [0.5]
    assert s.active("cmd_gap", 1.5) == []
    s.add(Fault("motor_error", 1.4, 0.2, {"joints": [3], "word": 2}))
    assert len(s.active("motor_error", 1.5)) == 1


def test_rng_streams_are_per_kind_and_reproducible() -> None:
    a, b = FaultSchedule(seed=7), FaultSchedule(seed=7)
    x = a.rng("cmd_drop").random(5)
    b.rng("sensor_spike").random(3)  # another kind's draws do not shift this kind's stream
    np.testing.assert_array_equal(x, b.rng("cmd_drop").random(5))
    assert not np.array_equal(x, FaultSchedule(seed=8).rng("cmd_drop").random(5))


def test_log() -> None:
    s = FaultSchedule()
    s.record(0.1, "cmd_gap", "discard")
    s.record(0.2, "cmd_gap", "discard")
    s.record(0.3, "motor_error", "apply", joints=[3])
    assert s.count("cmd_gap") == 2 and s.count("motor_error", "apply") == 1 and s.count("cmd_drop") == 0
    assert s.log[2].detail == {"joints": [3]}
    assert set(FAULT_KINDS) >= {"controller_stall", "cmd_gap", "cmd_drop", "state_delay", "hand_state_freeze"}
