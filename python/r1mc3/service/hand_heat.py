"""r1d's view of the hands' heat (session 13; r1mc3.hands.heat): each hand's HeatWatch, fed from the running grips at
L2's tick and read by `status` (`hands_heat`) and every hand.grasp reply (`hand_heat`). Words before the grip's limit
(72 C), nothing automatic: no hand is opened, no goal refused, the limit never moved."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from r1mc3.hands.heat import HeatWatch, hand_temperature_c

if TYPE_CHECKING:
    from .l2 import L2

SIDES = ("left", "right")


def watches(l2: L2) -> dict[str, HeatWatch]:
    w: dict[str, HeatWatch] | None = getattr(l2, "_hand_heat", None)
    if w is None:
        w = {s: HeatWatch() for s in SIDES}
        l2._hand_heat = w  # type: ignore[attr-defined]
    return w


def tick(l2: L2, now_ns: int) -> None:
    """From L2._tick_grips: a side with a grip (squeezing - a NOT_HOLDING grip that keeps its grip too) feeds its
    runner's last temperature; a side whose grip ended closes that grip's peak."""
    for side, w in watches(l2).items():
        gr = l2.grips.get(side)
        t = getattr(gr, "temperature_c", None) if gr is not None else None
        w.update(now_ns * 1e-9, t, gripping=gr is not None and getattr(gr, "state", "OPEN") != "OPEN")


def status(l2: L2, now_ns: int) -> dict[str, Any]:
    """{side: HeatWatch.status} with the hand's current temperature (r1hands' state; None without hands)."""
    hands = getattr(l2.svc, "hands", None)
    out: dict[str, Any] = {}
    for side, w in watches(l2).items():
        t: float | None = None
        if hands is not None:
            try:
                t = hand_temperature_c(hands.state(side))
            except Exception:  # a hands hiccup: the words wait for the next read
                t = None
        gr = l2.grips.get(side)
        if t is None and gr is not None:
            t = getattr(gr, "temperature_c", None)
        out[side] = w.status(now_ns * 1e-9, t)
    return out


def words(heat: dict[str, Any]) -> list[str]:
    """`r1 status` lines: "hands: right hand warm - ..." for every side that is not ok."""
    out: list[str] = []
    for side, h in heat.items():
        if h.get("level") != "ok":
            out.extend(f"hands: {side} hand {h['level']} - {w}" for w in h["words"])
    return out
