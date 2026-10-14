"""r1d's state-age block + `r1 status` words (session 10, stale; docs/sim/STATE_AGE.md, D34) on fake ports.

Every test fails on the base a041731: `r1mc3.service.state_age` does not exist and r1d's status has no `state_age`.
"""

from __future__ import annotations

from typing import Any

from r1mc3.service.state_age import KEYS, state_age_status, state_age_words


class _Port:
    def __init__(self, info: dict[str, Any] | None = None) -> None:
        self._info = info

    def rt_info(self) -> dict[str, Any] | None:
        return self._info


class _Core:
    def __init__(self, st: dict[str, Any]) -> None:
        self._st = st

    def status(self) -> dict[str, Any]:
        return self._st


class _LockstepLike(_Port):
    def __init__(self, st: dict[str, Any]) -> None:
        super().__init__(None)
        self.core = _Core(st)


def test_no_heartbeat_no_block() -> None:
    assert state_age_status(_Port(), None) is None
    assert state_age_words(None, None) == []


def test_fault_bit_only_robot_without_counters() -> None:
    """An r1rt that publishes no counters (rt_info without them): only the fault bit, the rest None."""
    sa = state_age_status(_Port({"stops": 0}), ["state_old"])
    assert sa is not None and set(sa) == set(KEYS)
    assert sa["old_now"] is True and sa["episodes"] is None and sa["content_age_ms"] is None
    words = state_age_words(sa, "balance")
    assert len(words) == 1 and words[0].startswith("r1rt: STATE_OLD")
    assert "content age" not in words[0]


def test_quiet_core_says_nothing() -> None:
    st = {"state_content_age_ns": 600_000, "state_old_episodes": 0, "content_age_valid": 1, "stale_freeze": 0}
    sa = state_age_status(_LockstepLike(st), [])
    assert sa is not None and sa["old_now"] is False and sa["content_age_ms"] == 0.6 and sa["episodes"] == 0
    assert state_age_words(sa, "balance") == []


def test_robot_counters_from_rt_info_in_ms() -> None:
    info = {"state_content_age_ms": 31.0, "state_old_episodes": 2, "state_old_ms_max": 412.0, "stale_rung": 0}
    sa = state_age_status(_Port(info), ["state_old", "lowstate_late"])
    assert sa is not None and sa["content_age_ms"] == 31.0 and sa["episodes"] == 2 and sa["stale_rung"] == 0
    words = state_age_words(sa, "balance")
    assert words[0] == (
        "r1rt: STATE_OLD - the robot state it balances on is older than its state-age limit (10 ms in the robot "
        "profile) (content age 31 ms) (the receive path is late or stopped: the network / DDS / r1rt's receive thread)"
    )
    assert words[1].startswith("r1rt: 2 STATE_OLD episode(s) since its start (the longest 412 ms) - D34")


def test_a_stale_freeze_is_named_and_how_balance_comes_back() -> None:
    st = {
        "state_content_age_ns": 0,
        "state_old_episodes": 1,
        "state_old_ns_max": 1_570_000_000,
        "stale_rung_entries": 1,
        "stale_freeze": 1,
        "content_age_relearns": 0,
    }
    sa = state_age_status(_LockstepLike(st), [])
    words = state_age_words(sa, "stand_freeze")
    assert words[0].startswith("r1rt: STAND_FREEZE by the state-age rung")
    assert "never limp" in words[0] and "only by the operator" in words[0]
    assert words[1].startswith("r1rt: 1 STATE_OLD episode(s) since its start (the longest 1570 ms)")
    # the freeze flag outside STAND_FREEZE (e.g. after the operator's ENTER_BALANCE) is not shown as a freeze
    assert not any("STAND_FREEZE" in w for w in state_age_words(sa, "balance"))


def test_relearns_are_worded() -> None:
    sa = state_age_status(_LockstepLike({"content_age_relearns": 3}), [])
    words = state_age_words(sa, "balance")
    assert words == [
        "r1rt: the board-tick clock re-based 3 time(s) (a board tick reset, or a steady backlog longer than 2 s that "
        "now reads as fresh)"
    ]


def test_robot_counters_from_the_segments_exec_status() -> None:
    """Layout 4 (session 10 core): r1rt writes the counters into the segment's executor status every tick."""

    class _View:
        def exec_status(self) -> dict[str, Any]:
            return {"seq": 9, "state_content_age_ns": 12_400_000, "state_old_episodes": 1, "stale_freeze": False}

    class _ShmLike(_Port):
        view = _View()

    sa = state_age_status(_ShmLike(None), ["state_old"])
    assert sa is not None and sa["content_age_ms"] == 12.4 and sa["episodes"] == 1 and sa["stale_freeze"] is False


def test_a_failing_source_is_ignored() -> None:
    class _Bad:
        def rt_info(self) -> dict[str, Any]:
            raise RuntimeError("torn")

    sa = state_age_status(_Bad(), ["state_old"])
    assert sa is not None and sa["old_now"] is True and sa["episodes"] is None
