"""The core's latched stop, as r1d sees it (session 13; HW2 day 2's F12).

A `stop` / `halt` (any client's: r1d's own word, a task client's end of a run, the core's own stop on a fault, a
freeze's STOP) LATCHES in r1rt until a `resume` (core.cpp: `stop_active_`, cleared only by R1RT_REQ_RESUME): the arms
take new goals at once (the executor's stop epoch), but the policy's base command - its lean - stays ZERO (core.cpp
`current_cmd`). On day 2 a task client stopped the robot at 02:47:08 UTC (the end of a teleop run), nobody resumed,
and at 02:53:00 `r1 lean 3` was ACCEPTED while the core used 0 (status `lean_deg [3.0, 0.0]`; the robot never leaned;
results/hw2_20261013/floor/r1d_journal.jsonl). r1d could not see the latch; since session 13 r1rt's heartbeat flags
carry it (R1RT_RT_FLAG_STOP_ACTIVE -> `SegmentView.rt_info()["stop_active"]`).
"""

from __future__ import annotations

from typing import Any

WORDS = (
    "the robot is STOPPED (a stop / halt latched in the core - it keeps balancing, the arms take new goals): the "
    "policy's lean stays 0 until `r1 resume`"
)


def latched(port: Any) -> bool | None:
    """True while the core's stop is latched, False when it is not, None when this r1rt / binding cannot say."""
    fn = getattr(port, "rt_info", None)
    if not callable(fn):
        return None
    try:
        info = fn()
    except Exception:
        return None
    if not isinstance(info, dict) or "stop_active" not in info:
        return None
    return bool(info["stop_active"])


def status(out: dict[str, Any], port: Any, lean_ended: str | None) -> None:
    """`r1 status`: `stop` {latched, words} and, under `body`, why a lean is not in use (`lean_note`). The words line
    in `summary` comes from service._status_words (`stop.words` when latched)."""
    lat = latched(port)
    out["stop"] = {"latched": lat, "words": WORDS if lat else None}
    body = out.get("body")
    if isinstance(body, dict):
        if lat:
            body["lean_note"] = WORDS
        elif lean_ended:
            body["lean_note"] = lean_ended


__all__ = ["WORDS", "latched", "status"]
