"""The C records r1log reads from r1rt's segment, as numpy structured dtypes, and their MCAP schemas.

The dtypes mirror rt/include/r1rt/types.h (r1rt_telemetry_record, r1rt_policy_record, r1rt_hands_contact,
r1rt_heartbeat, and since layout 2 (session 5) r1rt_guard_record - r1guard's own cycle record: little-endian, natural
C alignment). They are checked against the C++ build at start-up
(`check_layout(r1rt_py.shm_layout())`: every field's offsetof and every size) - r1log refuses to run on a drifted
layout instead of writing misread logs (tests/unit/test_r1log_records.py fails on any drift).

In the MCAP files a record travels as its raw C bytes; the schema (encoding "r1rt-struct", a custom encoding -
MCAP allows them) is a JSON document with every field's name, numpy type, C type, offset and shape plus the layout /
ABI versions. A reader rebuilds the dtype from the FILE's schema (`dtype_from_schema`), so an old log stays readable
after the layout changes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

NJ = 26  # R1RT_NUM_JOINTS
MAX_OBS = 128  # R1RT_MAX_POLICY_OBS
NACT = 14  # R1RT_NUM_POLICY_ACTIONS
NHANDS = 2  # R1RT_NUM_HANDS
LAYOUT_VERSION = 4  # R1RT_LAYOUT_VERSION this module was written for (shm_layout.h; 2 = session 5; 3 = session 8: the
# guard mirror's tau_ff; 4 = session 10: the executor status region appended - none of the records below changed)
STRUCT_ENCODING = "r1rt-struct"

_C_TYPES = {
    "u1": "uint8_t",
    "u2": "uint16_t",
    "u4": "uint32_t",
    "u8": "uint64_t",
    "i2": "int16_t",
    "i4": "int32_t",
    "i8": "int64_t",
    "f4": "float",
    "f8": "double",
}

# (field, numpy base type, shape) in declaration order (types.h).
_FIELDS: dict[str, list[tuple[str, str, tuple[int, ...]]]] = {
    "r1rt_telemetry_record": [
        ("seq", "u8", ()),
        ("t_wake_ns", "i8", ()),
        ("t_send_ns", "i8", ()),
        ("t_state_rx_ns", "i8", ()),
        ("board_tick", "u4", ()),
        ("wake_lateness_ns", "i4", ()),
        ("iteration_ns", "i4", ()),
        ("faults", "u4", ()),
        ("clamp_count", "u4", ()),
        ("mode", "u2", ()),
        ("token_owner", "u1", ()),
        ("guard_event", "u1", ()),
        ("q", "f4", (NJ,)),
        ("dq", "f4", (NJ,)),
        ("tau_est", "f4", (NJ,)),
        ("q_cmd", "f4", (NJ,)),
        ("dq_cmd", "f4", (NJ,)),
        ("kp", "f4", (NJ,)),
        ("kd", "f4", (NJ,)),
        ("tau_ff", "f4", (NJ,)),
        ("imu_quat", "f4", (4,)),
        ("imu_gyro", "f4", (3,)),
        ("imu_acc", "f4", (3,)),
        ("temperature", "i2", (NJ, 2)),
        ("motor_state", "u4", (NJ,)),
    ],
    "r1rt_policy_record": [
        ("seq", "u8", ()),
        ("t_obs_ns", "i8", ()),
        ("t_action_ns", "i8", ()),
        ("n_obs", "u4", ()),
        ("inline_run", "u1", ()),
        ("_pad0", "u1", (3,)),
        ("obs", "f4", (MAX_OBS,)),
        ("action", "f4", (NACT,)),
    ],
    "r1rt_hands_contact": [
        ("seq", "u8", ()),
        ("t_ns", "i8", ()),
        ("pad_contact", "u4", (NHANDS,)),
        ("finger_stalled", "u1", (NHANDS,)),
        ("finger_blocked", "u1", (NHANDS,)),
        ("grip_state", "u1", (NHANDS,)),
        ("valid", "u1", ()),
        ("_pad0", "u1", ()),
    ],
    "r1rt_heartbeat": [
        ("seq", "u8", ()),
        ("t_ns", "i8", ()),
        ("lowstate_tick", "u8", ()),
        ("rate_hz", "u4", ()),
        ("faults", "u4", ()),
        ("blend_progress", "f4", ()),
        ("mode", "u2", ()),
        ("rt_granted", "u1", ()),
        ("guard_flags", "u1", ()),  # ABI 6 (session 6): the former padding byte - r1guard's R1RT_GUARD_FLAG_*
    ],
    "r1rt_guard_record": [  # layout 2 (session 5): r1guard's cycle (1 kHz, standby and active)
        ("seq", "u8", ()),
        ("t_ns", "i8", ()),
        ("t_state_rx_ns", "i8", ()),
        ("rt_beat_age_ns", "i8", ()),
        ("rt_beat_seq", "u8", ()),
        ("board_tick", "u4", ()),
        ("faults", "u4", ()),
        ("mode", "u2", ()),
        ("mode_from", "u2", ()),
        ("act_mode", "u2", ()),
        ("token_owner", "u1", ()),
        ("event", "u1", ()),
        ("active", "u1", ()),
        ("published", "u1", ()),
        ("shadow_valid", "u1", ()),
        ("_pad0", "u1", (5,)),
        ("shadow_dev_rad", "f4", ()),
        ("tilt_rad", "f4", ()),
        ("q_cmd", "f4", (NJ,)),
        ("kp", "f4", (NJ,)),
        ("kd", "f4", (NJ,)),
    ],
}

# The sizes types.h pins with static asserts.
SIZES = {
    "r1rt_telemetry_record": 1136,
    "r1rt_policy_record": 600,
    "r1rt_hands_contact": 32,
    "r1rt_heartbeat": 40,
    "r1rt_guard_record": 384,
}


def _dtype(name: str) -> np.dtype[Any]:
    fields = _FIELDS[name]
    dt = np.dtype([(f, "<" + t, shape) for f, t, shape in fields], align=True)
    if dt.itemsize != SIZES[name]:  # natural alignment must reproduce the C size exactly
        raise AssertionError(f"{name}: numpy itemsize {dt.itemsize} != C size {SIZES[name]}")
    return dt


TELEMETRY = _dtype("r1rt_telemetry_record")
POLICY = _dtype("r1rt_policy_record")
HANDS_CONTACT = _dtype("r1rt_hands_contact")
HEARTBEAT = _dtype("r1rt_heartbeat")
GUARD = _dtype("r1rt_guard_record")
DTYPES: dict[str, np.dtype[Any]] = {
    "r1rt_telemetry_record": TELEMETRY,
    "r1rt_policy_record": POLICY,
    "r1rt_hands_contact": HANDS_CONTACT,
    "r1rt_heartbeat": HEARTBEAT,
    "r1rt_guard_record": GUARD,
}

# types.h enumerations (names for logs and events).
MODES = {
    0: "MONITOR",
    1: "TAKEOVER_MIRROR",
    2: "TAKEOVER_BLEND",
    3: "BALANCE",
    4: "GUARDED",
    5: "STAND_FREEZE",
    6: "ON_STAND",
    7: "OVERLAY_ARM_SDK",
    8: "DAMP",
}
OWNERS = {0: "none", 1: "r1rt", 2: "r1guard"}
# guard.hpp GuardEvent (the guard record's `event`)
GUARD_EVENTS = {
    0: "none",
    1: "takeover",
    2: "cas_failed",
    3: "token_lost",
    4: "shadow_invalid",
    5: "freeze",
    6: "mode_not_handled",
}
FAULT_BITS: dict[str, int] = {
    "lowstate_late": 1 << 0,
    "lowstate_stale": 1 << 1,
    "lowstate_lost": 1 << 2,
    "crc": 1 << 3,
    "tilt_arm_stop": 1 << 4,
    "tilt_fall": 1 << 5,
    "joint_clamp": 1 << 6,
    "velocity_clamp": 1 << 7,
    "torque_clamp": 1 << 8,
    "motor_error": 1 << 9,
    "policy_late": 1 << 10,
    "policy_invalid": 1 << 11,
    "loop_gap": 1 << 12,
    "not_owner": 1 << 13,
    "guard_absent": 1 << 14,
    "rt_not_granted": 1 << 15,
    "takeover_alarm": 1 << 16,  # session 4 (rtcore): an alarm only - a takeover failure branch was taken
    "builtin_active": 1 << 17,  # session 4 (rtcore): the built-in controller still writes after the release
    "over_temp": 1 << 18,  # session 6: a joint at or above the takeover's temperature gate (60 C): a warning only
    "state_old": 1 << 19,  # session 9 (U19 / D25): the state older than state_age_limit_ns - an alarm (STATE_AGE.md)
}
STOP_LEVELS = ("stop", "halt", "damp")


def fault_names(word: int) -> list[str]:
    return [k for k, b in FAULT_BITS.items() if word & b]


def check_layout(layout: Mapping[str, Any]) -> None:
    """Refuse a C++ build whose records differ from these dtypes (sizes, every field's offsetof, the layout
    version). `layout` = r1rt_py.shm_layout()."""
    if int(layout["layout_version"]) != LAYOUT_VERSION:
        raise RuntimeError(f"segment layout version {layout['layout_version']} != {LAYOUT_VERSION} (r1log is stale)")
    for name, dt in DTYPES.items():
        size = int(layout["sizes"][name])
        if size != dt.itemsize:
            raise RuntimeError(f"{name}: C size {size} != r1log's {dt.itemsize}")
        assert dt.fields is not None
        offs = layout[name]
        for f, spec in dt.fields.items():
            off = spec[1]
            if f.startswith("_pad"):
                continue
            if f not in offs or int(offs[f]) != off:
                raise RuntimeError(f"{name}.{f}: C offset {offs.get(f)} != r1log's {off}")
        missing = set(offs) - set(dt.fields)
        if missing:
            raise RuntimeError(f"{name}: C fields r1log does not know: {sorted(missing)}")


def schema_json(name: str, *, abi_version: int | None = None) -> bytes:
    """The MCAP schema data for a struct channel (encoding "r1rt-struct"): self-describing field list."""
    dt = DTYPES[name]
    assert dt.fields is not None
    fields = []
    for f, t, shape in _FIELDS[name]:
        fields.append(
            {"name": f, "type": "<" + t, "c_type": _C_TYPES[t], "offset": int(dt.fields[f][1]), "shape": list(shape)}
        )
    doc = {
        "struct": name,
        "header": "rt/include/r1rt/types.h",
        "size": dt.itemsize,
        "endian": "little",
        "layout_version": LAYOUT_VERSION,
        "abi_version": abi_version,
        "fields": fields,
    }
    return json.dumps(doc, separators=(",", ":")).encode()


def dtype_from_schema(data: bytes) -> np.dtype[Any]:
    """Rebuild a record dtype from an "r1rt-struct" schema (the reader trusts the file, not this module)."""
    doc = json.loads(data)
    names, formats, offsets = [], [], []
    for f in doc["fields"]:
        names.append(f["name"])
        formats.append((f["type"], tuple(f["shape"])) if f["shape"] else f["type"])
        offsets.append(int(f["offset"]))
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(doc["size"])})


def tilt_deg(quat_wxyz: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Tilt of the body z axis from the world vertical for (n, 4) quaternions (w, x, y, z): cos t = 1 - 2(x^2 + y^2).
    Absolute (from the IMU), not the core's learned-stance tilt - the flight recorder only needs a large-tilt cue."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    c = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    out: np.ndarray[Any, Any] = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    return out
