"""A Dex3-1 hand's heat, in words BEFORE the grip's limit (session 13; HW2 day 2).

    w = HeatWatch()
    w.update(now_s, temperature_c, gripping=True)     # every r1d tick a grip runs (L2._tick_grips)
    w.status(now_s, temperature_c)                    # {"temperature_c", "level", "peak_c", "words", ...}

Why (HW2 day 2, 13 Oct 2026): the right hand reached the grip's limit (r1mc3.hands.grip.TEMP_ABORT_C, 72 C - the
finger motors' protection: the grasp ends relaxed, powered) on the 4th back-to-back wrap of a 6.7 cm box - "hand hot
(72 C >= 72)" - with nothing said before it. The day's 33 grasp traces (results/hw2_20261013/floor{,_a}/records,
the max over the hand's 14 sensors at every read):
- while a grip squeezes the hand heats 1-2.7 C/s (pinch ~2.2, wrap ~0.7-2.4); a pinch levels off at 60 C (TEMP_EASE_C:
  the squeeze halves there) - a firm WRAP on a 6.7 cm box kept heating: its four peaks 63 / 60 / 67 / 72 C;
- after the release it cools fast: -11..-13 C within 10-14 s (60 -> 47-49);
- back to back (~12-20 s apart) each grasp starts ~1 C warmer (44 -> 49 over four pinches, 46 -> 50 over four wraps); a
  2-3 min pause brings the start back to 44-46 C (a 30 min pause: 43).
So a peak at or above HEAT_WARN_C predicts the next firm grip will reach the limit: the words say so and how long to
wait. The limit itself is never moved here (never a loosening); nothing here opens a hand or refuses a goal.
Generic (D32): one hand's temperatures, no task names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .grip import TEMP_ABORT_C, TEMP_EASE_C

HEAT_WARN_C = 65.0  # a grip's temperature at / above this: the next firm grip may reach TEMP_ABORT_C (day 2: 67 -> 72)
HEAT_RATE_C_S = 2.0  # while squeezing, the day's median heating (1-2.7 C/s) - for "about N s of grip left"
COOL_WAIT_S = 90.0  # a pause that brought the start temperature back to ~45 C on day 2 (2-3 min pauses: 44-46 C)
PEAK_FORGET_S = 180.0  # a grip's peak stops warning this long after the release (day 2: 2-3 min -> 44-46 C)


@dataclass
class HeatWatch:
    """One hand: the current grip's peak and the last grip's peak + when it ended."""

    peak_c: float | None = None  # the running grip's peak
    last_peak_c: float | None = None  # the last ended grip's peak
    last_end_s: float | None = None
    _gripping: bool = False

    def update(self, now_s: float, temperature_c: float | None, *, gripping: bool) -> None:
        if gripping:
            if temperature_c is not None:
                self.peak_c = temperature_c if self.peak_c is None else max(self.peak_c, temperature_c)
            self._gripping = True
            return
        if self._gripping:  # the grip ended: its peak is what predicts the next one
            self.last_peak_c, self.last_end_s = self.peak_c, now_s
            self.peak_c = None
            self._gripping = False

    def status(self, now_s: float, temperature_c: float | None) -> dict[str, Any]:
        t = temperature_c
        recent = (
            self.last_peak_c
            if self.last_peak_c is not None and self.last_end_s is not None and now_s - self.last_end_s < PEAK_FORGET_S
            else None
        )
        peak = max((p for p in (self.peak_c, recent) if p is not None), default=None)
        level = "ok"
        words: list[str] = []
        if t is not None and t >= TEMP_ABORT_C:
            level = "hot"
            words.append(f"{t:.0f} C: at the grip's limit ({TEMP_ABORT_C:.0f} C) - a grip ends relaxed; let it cool")
        elif self._gripping and t is not None and t >= HEAT_WARN_C:
            level = "warm"
            left = max(0.0, (TEMP_ABORT_C - t) / HEAT_RATE_C_S)
            words.append(
                f"{t:.0f} C while gripping: about {left:.0f} s of squeeze left before the grip's limit "
                f"({TEMP_ABORT_C:.0f} C; it heats ~1-2.5 C/s while it squeezes) - put the object down soon"
            )
        elif peak is not None and peak >= HEAT_WARN_C:
            level = "warm"
            since = None if self.last_end_s is None else now_s - self.last_end_s
            wait = COOL_WAIT_S if since is None else max(0.0, COOL_WAIT_S - since)
            words.append(
                f"the last grip peaked at {peak:.0f} C (the grip's limit {TEMP_ABORT_C:.0f} C): the next firm grip may "
                f"reach it - wait ~{wait:.0f} s before it (back to back each grasp starts ~1 C warmer; a 2-3 min "
                f"pause brings the hand back to ~45 C)"
            )
        return {
            "temperature_c": None if t is None else round(float(t), 1),
            "level": level,
            "peak_c": None if peak is None else round(float(peak), 1),
            "warn_c": HEAT_WARN_C,
            "ease_c": TEMP_EASE_C,
            "limit_c": TEMP_ABORT_C,
            "words": words,
        }


def hand_temperature_c(state: Any) -> float | None:
    """The hottest of a hand state's temperature sensors (Dex3-1: 7 motors x 2), None without one."""
    if not isinstance(state, Mapping):
        return None
    temp = state.get("temperature", state.get("temp"))
    if temp is None:
        return None
    try:
        a = np.asarray(temp, dtype=np.float64)
        return float(np.nanmax(a)) if a.size and np.isfinite(a).any() else None
    except (TypeError, ValueError):
        return None
