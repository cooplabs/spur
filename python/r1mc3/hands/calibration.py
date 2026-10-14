"""The hands' calibration references r1d keeps per side (session 10): the AIR-PINCH reference of each hand.

A `thin_flexible` grasp (DESIGN §6.2 / D27: a layer of cloth between the fingertips) is judged by the tip-shell
clearance where the tips stop: with nothing between them the real tips (and the simulated hand's capsules) stop a
little short of the model's shells (the simulated hand reads 2.19 mm), so the layer term needs each hand's own zero -
`layer_zero_m`, read from a tip pinch closed on NOTHING (`r1mc3.hands.grip.air_pinch_layer`). Without it the grip
refuses `thin_flexible` (an uncalibrated layer term passes an air pinch).

r1d's `hand.calibrate {side, kind: "air_pinch"}` measures it (r1mc3.service.hand_cal) and stores it here; every later
thin pinch of that hand uses it. `accept_air_pinch(verdict)` decides whether the close really ended tip on tip.

Session 12 (HW2 day 1, H3, 12 Oct 2026): on the robot both hands' air pinches RAN - thumb and index both "stall", the
user saw the tips touch - but were refused: the tip pads read 0.00-0.02 kPa (the real tip pads answer only above ~10 N
and by where the tip lands, H4) against the 1.5 kPa rule, and the capsule model put the touching tips 6.42 (right) /
9.30 mm (left) INSIDE each other - beyond the -2 mm floor. So the tips' meeting is accepted from the STALL too: the
thumb and the index both stopped on a stall / torque with the tips' points within AIR_PINCH_APERTURE_MAX_M of the
model's tips-met distance (H3: 1.55 / 5.48 mm), no pad needed; and the plausible range reaches -12 mm (the real
shells vs the model's capsules: H3's -9.3 mm + a 2.7 mm margin).

The store is in memory, and in a JSON file when `R1_HAND_CAL_FILE` names one (loaded when r1d starts, written at each
calibration) - so a restarted r1d keeps the session's references.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SIDES = ("left", "right")
# a plausible air-pinch reference (m): the tips stop between 12 mm INSIDE the model's shells (session 12: the real
# tips touching read -6.42 / -9.30 mm in the capsule model, HW2 H3; was -2 mm from S9's ~4 mm-per-side estimate) and
# 6 mm short of them; outside -> the close did not end tip on tip (something between the tips, the thumb on the
# finger's side) and the calibration is refused
AIR_PINCH_MIN_M = -0.012
AIR_PINCH_MAX_M = 0.006
AIR_PINCH_APERTURE_MAX_M = 0.010  # the tips' POINTS at most this past the model's tips-met distance (H3: 1.6 / 5.5 mm)
STALL_STOPS = ("stall", "torque")  # a stop on the other tip without a pad (the real tip pads: a ~10 N switch, H4)
ENV_FILE = "R1_HAND_CAL_FILE"


def accept_air_pinch(verdict: Mapping[str, Any]) -> tuple[float | None, str, str | None]:
    """(the reference in m or None, how the tips' meeting was shown: "tip pressure" | "stall" | "", why refused) from
    the verdict (as_dict) of a `tip` pinch closed on nothing with layer_zero_m 0 (r1mc3.service.hand_cal)."""
    ev = verdict.get("evidence") or {}
    terms = ev.get("terms") or {}
    layer = (terms.get("layer") or {}).get("layer_mm")
    if layer is None:
        return None, "", "not a thin_flexible verdict (no layer term): calibrate with a tip pinch on nothing"
    tp = terms.get("tip_pressure") or {}
    stops = ev.get("stops") or {}
    ap = ev.get("aperture_mm") or {}
    fingers = [f for f in ("index", "middle") if f in stops]
    stalled = stops.get("thumb") in STALL_STOPS and any(
        stops.get(f) in STALL_STOPS and f in ap and float(ap[f]) <= 1000 * AIR_PINCH_APERTURE_MAX_M for f in fingers
    )
    if tp.get("ok"):
        basis = "tip pressure"
    elif stalled:
        basis = "stall"
    else:
        return (
            None,
            "",
            (
                f"the tips did not press each other (tip pressure: {tp}; stops {stops}, tip gap past the meeting "
                f"point {ap} mm; a stall of thumb AND index within {1000 * AIR_PINCH_APERTURE_MAX_M:.0f} mm counts)"
            ),
        )
    z = float(layer) / 1000.0
    if not AIR_PINCH_MIN_M <= z <= AIR_PINCH_MAX_M:
        return None, basis, (f"air-pinch reference {1000 * z:.2f} mm outside {1000 * AIR_PINCH_MIN_M:.0f} .. "
                             f"{1000 * AIR_PINCH_MAX_M:.0f} mm")  # fmt: skip
    return z, basis, None


@dataclass(frozen=True)
class AirPinch:
    side: str
    layer_zero_m: float
    t_unix: float
    source: str  # "measured" (hand.calibrate) | "file"


class HandCalibration:
    """Per side: the air-pinch reference (None until measured)."""

    def __init__(self, path: str | Path | None = None) -> None:
        p = path if path is not None else os.environ.get(ENV_FILE)
        self.path = Path(p) if p else None
        self.air_pinch: dict[str, AirPinch] = {}
        if self.path is not None and self.path.exists():
            self.load()

    def layer_zero(self, side: str) -> float | None:
        a = self.air_pinch.get(side)
        return None if a is None else a.layer_zero_m

    def set_air_pinch(self, side: str, layer_zero_m: float) -> AirPinch:
        if side not in SIDES:
            raise ValueError(f"side: one of {SIDES}")
        z = float(layer_zero_m)
        if not AIR_PINCH_MIN_M <= z <= AIR_PINCH_MAX_M:
            raise ValueError(f"air-pinch reference {1000 * z:.2f} mm outside {1000 * AIR_PINCH_MIN_M:.0f} .. "
                             f"{1000 * AIR_PINCH_MAX_M:.0f} mm")  # fmt: skip
        a = AirPinch(side, z, time.time(), "measured")
        self.air_pinch[side] = a
        self.save()
        return a

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for s in SIDES:
            a = self.air_pinch.get(s)
            out[s] = None if a is None else {"layer_zero_mm": round(1000 * a.layer_zero_m, 3), "source": a.source}
        return out

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "schema": "r1mc3.hand_calibration/1",
                        "air_pinch": {s: asdict(a) for s, a in self.air_pinch.items()},
                    },
                    indent=1,
                )
            )
            tmp.replace(self.path)
        except OSError:
            pass  # the in-memory reference still holds for this r1d

    def load(self) -> None:
        assert self.path is not None
        try:
            d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        for s, a in (d.get("air_pinch") or {}).items():
            z = float(a.get("layer_zero_m", float("nan")))
            if s in SIDES and AIR_PINCH_MIN_M <= z <= AIR_PINCH_MAX_M:
                self.air_pinch[s] = AirPinch(s, z, float(a.get("t_unix", 0.0)), "file")
