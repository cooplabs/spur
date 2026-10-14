"""Unitree's battery message `unitree_hg.msg.dds_.BmsState_` (rt/lf/bmsstate) for CycloneDDS's Python binding
(session 13; read-only use: r1mc3.battery).

Unitree ships no `.idl` for it (rt/idl/SOURCE.md: only generated code, and not for this type). The layout = the
unitree_sdk2py class v2 read on the R1 (r1mc2/backends/robot.py:138-146 subscribed with it; the field order, types and
array sizes of v2's fake SDK, tests/fake_sdk/unitree_sdk2py/idl/unitree_hg/msg/dds_/__init__.py:56-69, which carries
a real reading: version_low 44, fn 120, cell_vol 9 x 4063 mV, bmsvoltage [36530, 36480, 0], current -2346 mA, soc
95, soh 99, temperature [23, 19, 20, 27], cycle 8, bmsstate [528, 0, 0, 0, 9]); written like r1mc3.dds.hg (the
generator's style: `@final`, sequential ids, uint8 for Unitree's octets). `python -m r1mc3.battery --check` on the
robot prints the remote's type name from discovery and a decoded reading - the check after a firmware change.
"""

# No `from __future__ import annotations`: CycloneDDS's IDL reads the field annotations at run time.
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

TOPIC_BMS = "rt/lf/bmsstate"
TOPIC_BMS_SECONDARY = "rt/lf/secondary_bmsstate"


@dataclass
@annotate.final
@annotate.autoid("sequential")
class BmsState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.BmsState_"):
    version_high: types.uint8
    version_low: types.uint8
    fn: types.uint8
    cell_vol: types.array[types.uint16, 40]  # type: ignore[type-arg, valid-type]
    bmsvoltage: types.array[types.uint32, 3]  # type: ignore[type-arg, valid-type]
    current: types.int32
    soc: types.uint8
    soh: types.uint8
    temperature: types.array[types.int16, 12]  # type: ignore[type-arg, valid-type]
    cycle: types.uint16
    manufacturer_date: types.uint8
    bmsstate: types.array[types.uint32, 5]  # type: ignore[type-arg, valid-type]
    reserve: types.array[types.uint32, 3]  # type: ignore[type-arg, valid-type]


def bms_zero(**over: object) -> BmsState_:
    """A BmsState_ with every field zero (+ the given ones): the tests' / the sim's publisher."""
    base: dict[str, object] = {
        "version_high": 0, "version_low": 0, "fn": 0, "cell_vol": [0] * 40, "bmsvoltage": [0] * 3, "current": 0,
        "soc": 0, "soh": 0, "temperature": [0] * 12, "cycle": 0, "manufacturer_date": 0, "bmsstate": [0] * 5,
        "reserve": [0] * 3,
    }  # fmt: skip
    base.update(over)
    return BmsState_(**base)  # type: ignore[arg-type]
