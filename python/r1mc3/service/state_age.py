"""The state-age alarm in r1d's `status` and in `r1 status` words (session 10; docs/sim/STATE_AGE.md, DESIGN D34).

Why: the core raises fault bit 19 `STATE_OLD` whenever the robot state it balances on is older than
`state_age_limit_ns` (10 ms) - by the receive age or the board tick's CONTENT age - and counts an EPISODE when that
lasts `state_age_sustain_ns`; with `stale_rung` >= 1 such an episode freezes a balancing robot into its stand
(STAND_FREEZE). D34 (the user, 11 Oct 2026): HW2 runs with the rung OFF and the alarm LOGGING, and the rung is armed
only after HW2's log check shows 0 false episodes. So the operator must be able to read, in words, (a) that the state
is old right now, (b) how many episodes there were since r1rt started and how old the state got, (c) that a
STAND_FREEZE came from the rung (not from the stand / an invalid action) and how BALANCE comes back.

Sources (each optional: a missing one leaves its keys None, never an error):
* the heartbeat's fault word (both ports): `state_old` in `core.faults` = old right now;
* `SegmentView.exec_status()` (shm layout 4, session 10 core: r1rt writes the state-age counters every tick - the
  robot path) and `SegmentView.rt_info()` keys if r1rt ever publishes them there; an older r1rt / binding has neither,
  then only the fault bit is known;
* in lockstep the core's own status (`LockstepPort.core.status()`: the same counters, in ns).
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

# the keys r1d reports under status["state_age"] (ms / counts / flags; None = not reported by this r1rt)
KEYS = (
    "old_now",
    "content_age_ms",
    "content_age_max_ms",
    "content_age_valid",
    "episodes",
    "old_ticks",
    "old_ms_max",
    "rung_entries",
    "rung_returns",
    "stale_freeze",
    "relearns",
    "stale_rung",
)
# the core's status names (Core.status() / rt_info) -> ours, with the ns -> ms scale where it applies
_NS_KEYS = {
    "state_content_age_ns": "content_age_ms",
    "state_content_age_max_ns": "content_age_max_ms",
    "state_old_ns_max": "old_ms_max",
}
_MS_KEYS = {
    "state_content_age_ms": "content_age_ms",
    "state_content_age_max_ms": "content_age_max_ms",
    "state_old_ms_max": "old_ms_max",
}
_COUNT_KEYS = {
    "state_old_episodes": "episodes",
    "state_old_ticks": "old_ticks",
    "stale_rung_entries": "rung_entries",
    "stale_rung_returns": "rung_returns",
    "content_age_relearns": "relearns",
    "stale_rung": "stale_rung",
}
_FLAG_KEYS = {"content_age_valid": "content_age_valid", "stale_freeze": "stale_freeze"}


def _from_core_dict(d: Mapping[str, Any], out: dict[str, Any]) -> None:
    for src, dst in _NS_KEYS.items():
        if d.get(src) is not None:
            out[dst] = round(1e-6 * float(d[src]), 2)
    for src, dst in _MS_KEYS.items():
        if d.get(src) is not None:
            out[dst] = round(float(d[src]), 2)
    for src, dst in _COUNT_KEYS.items():
        if d.get(src) is not None:
            out[dst] = int(d[src])
    for src, dst in _FLAG_KEYS.items():
        if d.get(src) is not None:
            out[dst] = bool(d[src])


def state_age_status(port: Any, core_faults: list[str] | None) -> dict[str, Any] | None:
    """status["state_age"]: None when r1rt has no heartbeat; else every KEY (None = not reported)."""
    if core_faults is None:
        return None
    out: dict[str, Any] = dict.fromkeys(KEYS)
    out["old_now"] = "state_old" in core_faults
    info_fn = getattr(port, "rt_info", None)
    if callable(info_fn):
        with contextlib.suppress(Exception):
            info = info_fn()
            if info:
                _from_core_dict(info, out)
    exec_fn = getattr(getattr(port, "view", None), "exec_status", None)  # layout 4: the robot path
    if callable(exec_fn):
        with contextlib.suppress(Exception):
            es = exec_fn()
            if es:
                _from_core_dict(es, out)
    core = getattr(port, "core", None)  # lockstep: the core itself (one thread calls everything)
    status_fn = getattr(core, "status", None)
    if callable(status_fn):
        with contextlib.suppress(Exception):
            _from_core_dict(dict(status_fn()), out)
    return out


def state_age_words(sa: Mapping[str, Any] | None, core_mode: str | None) -> list[str]:
    """The `r1 status` lines for the state-age alarm (none while nothing is or was old)."""
    if not sa:
        return []
    lines: list[str] = []
    eps = sa.get("episodes")
    if sa.get("stale_freeze") and core_mode == "stand_freeze":
        lines.append(
            "r1rt: STAND_FREEZE by the state-age rung (the robot state stayed older than the limit for the sustain "
            "time): legs ramped to the stand pose at stand gains, arms / head held - never limp; BALANCE comes back "
            "only by the operator once the state is live again (`r1 balance`, then `r1 resume`)"
        )
    elif sa.get("freeze_way_back") and core_mode == "stand_freeze":  # session 11 (core): the operator's / rung 3's
        lines.append(
            "r1rt: STAND_FREEZE from BALANCE: legs ramped to the stand pose at stand gains (the ankles shifted for the "
            "arms it holds), arms / head held - never limp; back: " + str(sa["freeze_way_back"])
        )
    if sa.get("old_now"):
        age = sa.get("content_age_ms")
        lines.append(
            "r1rt: STATE_OLD - the robot state it balances on is older than its state-age limit (10 ms in the robot "
            "profile)"
            + (f" (content age {float(age):.0f} ms)" if age is not None else "")
            + " (the receive path is late or stopped: the network / DDS / r1rt's receive thread)"
        )
    if eps:
        worst = sa.get("old_ms_max")
        lines.append(
            f"r1rt: {int(eps)} STATE_OLD episode(s) since its start"
            + (f" (the longest {float(worst):.0f} ms)" if worst is not None else "")
            + " - D34: the hold-the-stand rung is armed only after a session with 0 false episodes"
        )
    if sa.get("relearns"):
        lines.append(
            f"r1rt: the board-tick clock re-based {int(sa['relearns'])} time(s) (a board tick reset, or a steady "
            "backlog longer than 2 s that now reads as fresh)"
        )
    return lines


__all__ = ["KEYS", "state_age_status", "state_age_words"]
