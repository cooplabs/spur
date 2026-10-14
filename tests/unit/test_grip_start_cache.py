"""HW2 (12 Oct 2026): a grasp's start must not do the hand model's slow reference search.

On the robot every `hand.grasp` / `hand.calibrate` refused "no valid hand state" ~2 of 3 times before any finger closed:
GraspRunner computed `empty_aperture` (a 401-step capsule search, ~50 ms per call on the Orin) for index + middle at
its construction; r1hands' reader thread could not run meanwhile (measured gaps 107-124 ms) and the hand state aged
past its 100 ms freshness limit by the time `start()` read it. The fix: the reference is cached (a constant of the hand
model) and r1d computes it at start-up (`warm_grip_caches`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.hands import grip as G

SIDES = ("left", "right")


class _CountTipClearance:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.n = 0
        real = G.tip_clearance

        def counted(*a: Any, **kw: Any) -> float:
            self.n += 1
            return real(*a, **kw)

        monkeypatch.setattr(G, "tip_clearance", counted)


def _clear_cache() -> None:
    clear = getattr(getattr(G, "_empty_aperture", None), "cache_clear", None)
    if clear is not None:
        clear()


def _uncached(side: str, finger: str, grip: str) -> float:
    """The reference computed from scratch (the search as it was before the cache)."""
    spec = G.GRIPS[grip]
    q0, q1 = G.preshape(grip, side), G.preset(spec.closed, side)
    for s in np.linspace(0.0, 1.0, 401):
        q = q0 + s * (q1 - q0)
        if G.tip_clearance(side, q, finger) <= 0.0:
            tp = G.tips(side, q)
            return float(np.linalg.norm(tp["thumb"] - tp[finger]))
    tp = G.tips(side, q1)
    return float(np.linalg.norm(tp["thumb"] - tp[finger]))


def test_a_second_empty_aperture_call_does_not_search_again(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cache()
    count = _CountTipClearance(monkeypatch)
    first = G.empty_aperture("right", "index", "pinch")
    assert count.n > 0  # the first call searches
    n1 = count.n
    assert G.empty_aperture("right", "index", "pinch") == first
    assert G.empty_aperture("right", "index") == first  # the default grip is the same key
    assert count.n == n1


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("finger", ["index", "middle"])
@pytest.mark.parametrize("grip", ["pinch", "tip", "wrap"])
def test_the_cached_reference_equals_the_search_exactly(side: str, finger: str, grip: str) -> None:
    _clear_cache()
    assert G.empty_aperture(side, finger, grip) == _uncached(side, finger, grip)
    assert G.empty_aperture(side, finger, grip) == _uncached(side, finger, grip)  # the cached value too


def test_after_the_warm_up_a_grasp_runner_starts_without_the_search(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cache()
    G.warm_grip_caches()
    count = _CountTipClearance(monkeypatch)
    for side in SIDES:
        G.GraspRunner(object(), side, "wrap", "rigid", width_m=0.06)
        G.GraspRunner(object(), side, "pinch", "rigid", width_m=0.03)
        G.GraspRunner(object(), side, "tip", "thin_flexible", layer_zero_m=0.0)  # hand.calibrate's air pinch
    assert count.n == 0


def test_r1d_warms_the_grip_caches_at_start_when_it_has_hands(monkeypatch: pytest.MonkeyPatch) -> None:
    from r1mc3.service.service import Service

    calls: list[int] = []
    monkeypatch.setattr(G, "warm_grip_caches", lambda: calls.append(1), raising=False)
    Service(object(), hands=object())  # type: ignore[arg-type]
    assert calls == [1]
    Service(object())  # type: ignore[arg-type]
    assert calls == [1]  # no hands: nothing to warm
