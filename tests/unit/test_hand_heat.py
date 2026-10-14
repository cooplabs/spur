"""The hands' heat in words BEFORE the grip's 72 C limit (session 13; r1mc3.hands.heat).

HW2 day 2 (13 Oct 2026): four back-to-back wraps of a box peaked at 63 / 60 / 67 / 72 C (the grasp traces'
temperatures, results/hw2_20261013/floor/records/grasp_20261014T0222*-0224*) - the 4th ended "hand hot (72 C >= 72)"
with nothing said before it. On the base (dcc01a2) there is no heat module, no `hand_heat` in a grasp reply and no
`hands_heat` in r1d's status: every test here fails."""

from __future__ import annotations

from r1mc3.hands import heat as H
from r1mc3.hands.grip import TEMP_ABORT_C


def _grip(w: H.HeatWatch, t0: float, temps: list[float]) -> float:
    """One grip: the temperatures at 1 s steps, then the release."""
    t = t0
    for c in temps:
        w.update(t, c, gripping=True)
        t += 1.0
    w.update(t, None, gripping=False)
    return t


def test_day2s_tea_box_sequence_warns_before_the_4th_wrap() -> None:
    w = H.HeatWatch()
    t = _grip(w, 0.0, [46, 50, 55, 60, 63])  # grasp 1 peaked 63: below the warning
    assert w.status(t + 12.0, 47.0)["level"] == "ok"
    t = _grip(w, t + 15.0, [47, 52, 58, 60])  # grasp 2: 60
    assert w.status(t + 12.0, 48.0)["level"] == "ok"
    t = _grip(w, t + 15.0, [48, 55, 61, 65, 67])  # grasp 3: 67 -> the 4th would reach 72 (it did on the robot)
    s = w.status(t + 12.0, 50.0)
    assert s["level"] == "warm" and s["peak_c"] == 67.0 and s["temperature_c"] == 50.0
    assert "the next firm grip may reach it" in s["words"][0] and "wait ~78 s" in s["words"][0]
    assert s["limit_c"] == TEMP_ABORT_C == 72.0  # the limit itself is not moved


def test_a_grip_at_66_c_says_how_long_it_can_still_squeeze() -> None:
    w = H.HeatWatch()
    w.update(0.0, 60.0, gripping=True)
    w.update(1.0, 66.0, gripping=True)
    s = w.status(1.0, 66.0)
    assert s["level"] == "warm" and "about 3 s of squeeze left" in s["words"][0]


def test_at_the_limit_it_is_hot_and_a_long_pause_forgets_the_peak() -> None:
    w = H.HeatWatch()
    t = _grip(w, 0.0, [50, 60, 72])
    assert w.status(t, 72.0)["level"] == "hot"
    assert w.status(t + 30.0, 49.0)["level"] == "warm"  # the 72 C peak still predicts the next grip
    assert w.status(t + H.PEAK_FORGET_S + 1.0, 45.0)["level"] == "ok"  # day 2: 2-3 min -> back to 44-46 C


def test_a_pinch_levelling_at_60_c_is_normal_operation() -> None:
    """Day 2's pinches all levelled off at 60 C (the squeeze halves there, TEMP_EASE_C) - no warning for that."""
    w = H.HeatWatch()
    t = _grip(w, 0.0, [44, 50, 55, 60, 60, 60, 60])
    assert w.status(t + 12.0, 47.0)["level"] == "ok"


def test_the_hand_temperature_is_the_hottest_sensor() -> None:
    assert H.hand_temperature_c({"temperature": [[40, 41], [66, 39]]}) == 66.0
    assert H.hand_temperature_c({"temperature": None}) is None and H.hand_temperature_c(None) is None
