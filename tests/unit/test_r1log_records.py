"""r1log's record layouts against the C++ build (types.h / shm_layout.h through r1rt_py.shm_layout()).

r1log reads raw C records; a numpy dtype that drifted from the C struct would write misread logs silently. These
tests pin the dtypes to this build's offsetof / sizeof and show check_layout refuses a drifted layout.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from r1mc3.log import records as R
from r1mc3.log._native import has_segment_reader, native

needs_native = pytest.mark.skipif(not has_segment_reader(), reason="r1rt_py with SegmentView not built")


@needs_native
def test_dtypes_match_this_build() -> None:
    layout = native().shm_layout()
    R.check_layout(layout)  # raises on any difference
    for name, dt in R.DTYPES.items():
        assert dt.itemsize == layout["sizes"][name] == R.SIZES[name]
        assert dt.fields is not None
        for f, off in layout[name].items():
            assert dt.fields[f][1] == off, (name, f)


@needs_native
@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda L: L["r1rt_telemetry_record"].__setitem__("q", L["r1rt_telemetry_record"]["q"] + 4), "offset"),
        (lambda L: L["sizes"].__setitem__("r1rt_policy_record", 608), "size"),
        (lambda L: L["r1rt_policy_record"].__setitem__("new_field", 600), "does not know"),
        (lambda L: L.__setitem__("layout_version", 5), "layout version"),  # session 10: the real one is 4
    ],
)
def test_a_drifted_layout_is_refused(mutate: Any, match: str) -> None:
    layout = copy.deepcopy(dict(native().shm_layout()))
    layout = {k: (dict(v) if isinstance(v, dict) else v) for k, v in layout.items()}
    mutate(layout)
    with pytest.raises(RuntimeError, match=match):
        R.check_layout(layout)


def test_schema_round_trip() -> None:
    for name, dt in R.DTYPES.items():
        back = R.dtype_from_schema(R.schema_json(name, abi_version=3))
        assert back.itemsize == dt.itemsize
        assert back.fields is not None and dt.fields is not None
        for f, spec in dt.fields.items():
            assert back.fields[f][1] == spec[1] and back.fields[f][0] == spec[0]
        rec = np.zeros(3, dt)
        rec["seq"] = [1, 2, 3]
        assert np.frombuffer(rec.tobytes(), back)["seq"].tolist() == [1, 2, 3]


def test_enumerations_match_the_simulator_mirror() -> None:
    """types.h has one source; r1sim.v3_core mirrors the fault bits and modes too: both mirrors must agree."""
    from r1sim import v3_core

    for k, bit in v3_core.FAULT_BITS.items():  # the sim's mirror names the bits it uses
        assert R.FAULT_BITS[k] == bit, k
    # + session 4 takeover bits, + session 6 over_temp (bit 18), + session 9 state_old (bit 19, types.h)
    assert len(set(R.FAULT_BITS.values())) == 20 and max(R.FAULT_BITS.values()) == 1 << 19
    assert R.MODES[v3_core.MODE_MONITOR] == "MONITOR" and R.MODES[v3_core.MODE_BALANCE] == "BALANCE"
    assert R.MODES[v3_core.MODE_STAND_FREEZE] == "STAND_FREEZE" and R.MODES[v3_core.MODE_DAMP] == "DAMP"


def test_tilt_from_quaternion() -> None:
    a = np.radians([0.0, 10.0, 85.0])
    q_pitch = np.stack([np.cos(a / 2), np.zeros(3), np.sin(a / 2), np.zeros(3)], axis=1)
    np.testing.assert_allclose(R.tilt_deg(q_pitch), [0.0, 10.0, 85.0], atol=1e-9)
    q_roll = np.stack([np.cos(a / 2), np.sin(a / 2), np.zeros(3), np.zeros(3)], axis=1)
    np.testing.assert_allclose(R.tilt_deg(q_roll), [0.0, 10.0, 85.0], atol=1e-9)
    q_yaw = np.array([[np.cos(0.5), 0.0, 0.0, np.sin(0.5)]])
    np.testing.assert_allclose(R.tilt_deg(q_yaw), [0.0], atol=1e-9)
