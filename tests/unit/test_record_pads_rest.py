"""`python -m r1mc3.record.pads_rest` (HW2 runbook H2): the at-rest pad numbers vs r1hands' contact threshold."""

from __future__ import annotations

from typing import Any

import numpy as np
from r1mc3.hands.contact import CONTACT_THRESHOLD, N_PADS
from r1mc3.record import pads_rest as P


def _series(sd: float, n: int = 250, hz: float = 25.0, seed: int = 0) -> tuple[list[float], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    base = np.full(N_PADS, 101325.0)
    t = [k / hz for k in range(n)]
    return t, [base + rng.normal(0.0, sd, N_PADS) for _ in range(n)]


def test_quiet_pads_read_no_contact() -> None:
    t, pk = _series(sd=0.5)
    s = P.summarize(t, pk)
    assert s["pads_over_threshold_most_of_the_time"] == 0
    assert s["noise_sd_median"] < 1.0
    assert s["tare_samples"] == 13  # 0.5 s at 25 Hz (t < 0.5: 0.00 .. 0.48)


def test_sim_like_noise_reads_contact_at_rest() -> None:
    """The simulator's ~100-unit pad noise against the 40 raw threshold (session 12; 5.0 before): a large share of the
    samples of every pad still exceeds it (a zero-mean noise around a 0.5 s median tare), so in sim the guards see
    'contact' on and off at rest - the real pads' sd is 5-8 (HW2 H2)."""
    t, pk = _series(sd=100.0)
    s = P.summarize(t, pk)
    assert s["pads_ever_over_threshold"] == N_PADS
    assert s["noise_vs_threshold"] > 2
    assert all(0.1 < p["share_over_threshold"] < 0.6 for p in s["pads"])  # the tare is a 0.5 s median of noise
    assert "never changes it" in P.words("left", s)


def test_hw2s_real_noise_reads_no_contact_at_40() -> None:
    """HW2's H2 (the real pads: sd 5-8, max +16 in 10 s) at the user's 40 raw: no pad over the threshold, ever."""
    t, pk = _series(sd=6.0)
    s = P.summarize(t, pk)
    assert s["threshold"] == 40.0 and s["pads_ever_over_threshold"] == 0


def test_a_pressed_pad_is_seen() -> None:
    t, pk = _series(sd=0.5)
    for k in range(100, 250):
        pk[k] = pk[k].copy()
        pk[k][1] += 100.0  # the thumb tip pressed after 4 s (HW1: a ~0.9 kg firm press +100..330 raw)
    s = P.summarize(t, pk)
    assert s["pads"][1]["max"] > 95 and s["pads"][1]["share_over_threshold"] > 0.6
    assert s["pads_over_threshold_most_of_the_time"] == 1


def test_pad_peaks_and_too_few_samples() -> None:
    p = np.arange(108, dtype=float).reshape(9, 12)
    peaks = P.pad_peaks(p.tolist())
    assert peaks is not None and np.allclose(peaks, p.max(axis=1))
    assert P.pad_peaks(None) is None and P.pad_peaks([1.0, 2.0]) is None
    assert "error" in P.summarize([0.0, 0.1], [np.zeros(9), np.zeros(9)])


def test_read_hand_polls_and_skips_invalid() -> None:
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(d: float) -> None:
        now[0] += d

    calls: list[dict[str, Any]] = []

    def call(op: str, **kw: Any) -> dict[str, Any]:
        calls.append({"op": op, **kw})
        valid = len(calls) % 5 != 0
        st = {"valid": valid, "pressure": np.full((9, 12), 101325.0).tolist()}
        return {"ok": True, "data": {kw["side"]: st}}

    _t, pk, bad = P.read_hand(call, "right", 2.0, 10.0, clock=clock, sleep=sleep)
    assert {c["op"] for c in calls} == {"hand.state"} and all(c["side"] == "right" for c in calls)
    assert len(calls) == 20 and bad == 4 and len(pk) == 16
    assert CONTACT_THRESHOLD == 40.0  # the guard's value is read, never changed, by the tool (session 12: 5 -> 40)
