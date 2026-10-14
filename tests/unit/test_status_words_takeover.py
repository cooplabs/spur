"""`r1 status`' words after a guard takeover (session 11, floor's F13 finding): `r1 reclaim` is ON_STAND only, so
after a takeover from BALANCE the words must not offer it. Fails on 1e55bdd: "(r1 reclaim gives them back)" in
BALANCE."""

from __future__ import annotations

from typing import Any

from r1mc3.service.service import _status_words


def _st(mode: str) -> dict[str, Any]:
    return {"token": {"owner": "r1guard"}, "core": {"mode": mode, "alive": True, "age_ms": 1.0}, "guard": None}


def _r1rt_line(mode: str) -> str:
    return next(w for w in _status_words(_st(mode)) if w.startswith("r1rt:"))


def test_after_a_takeover_from_balance_no_reclaim_is_offered() -> None:
    line = _r1rt_line("balance")
    assert "reclaim" not in line
    assert "r1 damp" in line and "SILENT" in line


def test_after_a_takeover_from_the_stand_reclaim_is_offered() -> None:
    assert "r1 reclaim gives them back" in _r1rt_line("on_stand")
