"""r1hands' pad contact on HW2's real numbers (session 12; r1mc3.hands.contact; results/hw2_20261012/HW2_PART1.md H2 /
H4, the user's decision 12 Oct 2026 ~17:55: CONTACT_THRESHOLD 5 -> 40 raw).

The fake hand reproduces what HW2 day 1 measured on the Dex3 pads (tests/assets/hands_hw2/README.md for the sources):
every taxel reads exactly 0 until the hand's first command (the state itself is valid), then each pad's largest taxel
sits at its own level near 100,000 raw (H2's after-tare levels, left hand) with a noise sd ~6, quantised in steps of 8
and at most +16 over its tare in 10 s; a touch the pads register is >= ~1,800 raw (H4); over tens of minutes the tip
and palm pads creep UP (stand_b's r1log: >= ~20 raw after ~30 min, ~+100 by ~3 h later) while the proximal pads fall.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from r1mc3.hands import contact as C

# H2's tares (tests/assets/hands_hw2/h2_pads_rest_after_tare.json, left hand: pad 0..8)
_H2 = json.loads(
    (Path(__file__).resolve().parents[1] / "assets" / "hands_hw2" / "h2_pads_rest_after_tare.json").read_text()
)
H2_LEVELS = np.array([p["tare"] for p in _H2["hands"]["left"]["pads"]], dtype=np.float64)
H2_MAX_ABOVE = max(p["max"] for s in ("left", "right") for p in _H2["hands"][s]["pads"])  # +16 raw in 10 s
TIPS_PALMS = [1, 3, 5, 6, 7, 8]  # the pads that drift up (stand_b); 0 / 2 / 4 (proximal) drift down
DT_NS = 10_000_000  # r1d's 100 Hz tick


class Pads:
    """Both hands: `streaming` False = every taxel 0 (before the first hand command); noise sd 6 quantised in 8s."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.streaming = {"left": False, "right": False}
        self.offset = {s: np.zeros(9) for s in ("left", "right")}  # drift / presses on top of the levels
        self.noise = True
        self.q: dict[str, Any] = {"left": None, "right": None}  # the hand's joints (None: no q in the state)

    def state(self, side: str) -> dict[str, Any]:
        q = {} if self.q[side] is None else {"q": np.asarray(self.q[side], dtype=np.float64)}
        if not self.streaming[side]:
            return {"valid": True, "pressure": np.zeros((9, 12)), **q}
        p = np.full((9, 12), 30000.0)  # NO_DATA slots
        n = self.rng.normal(0.0, 6.0, 9) if self.noise else np.zeros(9)
        peak = H2_LEVELS + self.offset[side] + 8.0 * np.round(np.clip(n, -24.0, 16.0) / 8.0)
        p[:, 3] = peak
        p[:, 5] = peak - 40.0
        return {"valid": True, "pressure": p, **q}


def _pub(pads: Pads, **kw: Any) -> tuple[C.ContactPublisher, list[tuple[int, int, bool]]]:
    calls: list[tuple[int, int, bool]] = []
    pub = C.ContactPublisher(pads, lambda pl, pr, t_ns, valid, **_k: calls.append((pl, pr, valid)), **kw)
    return pub, calls


def _run(pub: C.ContactPublisher, t: list[int], n: int, dt_ns: int = DT_NS) -> None:
    for _ in range(n):
        t[0] += dt_ns
        pub.tick(t[0])


def test_the_threshold_is_the_users_40_raw() -> None:
    """The user's decision (12 Oct ~17:55): 40 raw = 5 steps of 8, between H2's at-rest noise (<= 16) and the weakest
    touch the pads register (>= ~1,800, H4). Fails on the base: 5.0."""
    assert C.CONTACT_THRESHOLD == 40.0
    assert H2_MAX_ABOVE == 16.0 and H2_MAX_ABOVE < C.CONTACT_THRESHOLD < 1800.0


def test_the_tare_waits_for_the_pads_to_stream_hw2_day1() -> None:
    """HW2 day 1, H2: r1d's tare was taken over the first 50 VALID states - before the pads streamed (all zeros) - so
    all 9 pads of both hands read "contact" from then on (511 / 511) and the executor's contact guard (it counts only
    pads that ROSE after a path began) could never fire. Now: not valid until the pads stream (a contact-guarded move
    stops at once instead of running blind), the tare from streaming states only, then 0 at rest and a touch seen.
    Fails on the base: valid True with 0 bits during the zeros, then 511 / 511 at rest."""
    pads = Pads()
    pub, calls = _pub(pads)
    t = [0]
    _run(pub, t, 100)  # 1 s of valid states, the pads not streaming yet
    assert calls[-1] == (0, 0, False)
    assert pub.last["tare"] == {"left": "waiting", "right": "waiting"}
    assert any("not streaming" in w for w in pub.last["words"])
    pads.streaming = {"left": True, "right": True}  # the operator opens the hands: the pads stream
    _run(pub, t, C.TARE_SAMPLES)
    assert pub.last["tare"] == {"left": "ok", "right": "ok"}
    _run(pub, t, 1000)  # 10 s at rest (H2's duration)
    assert calls[-1][2] is True
    assert max(pl | pr for pl, pr, _ in calls[-1000:]) == 0  # at rest: no pad "in contact", ever
    pads.offset["right"][5] = 1800.0  # finger B's tip registers a touch (H4's weakest registered touch)
    _run(pub, t, 1)
    assert calls[-1] == (0, 1 << 5, True)


def test_at_rest_noise_never_reads_contact_but_a_firm_press_does() -> None:
    """H2's noise (sd 5-8, max +16 in 10 s, both hands) at 40 raw: 0 of 9 pads in 60 s; at the base's 5 raw the same
    stream reads contact on most pads at once. A ~0.9 kg firm press (HW1: +100..330 raw) is a contact."""
    pads = Pads(seed=3)
    pads.streaming = {"left": True, "right": True}
    pub, calls = _pub(pads)
    t = [0]
    _run(pub, t, C.TARE_SAMPLES + 6000)
    assert max(pl | pr for pl, pr, _ in calls[-6000:]) == 0
    old_pads = Pads(seed=3)
    old_pads.streaming = {"left": True, "right": True}
    old, old_calls = _pub(old_pads, threshold=5.0)
    _run(old, [0], C.TARE_SAMPLES + 200)
    assert bin(max(pl for pl, _, _ in old_calls[-200:])).count("1") >= 5
    pads.offset["left"][1] = 100.0
    _run(pub, t, 1)
    assert calls[-1][0] == 1 << 1


def test_a_stream_that_stops_is_tared_again_when_it_comes_back() -> None:
    """The hand driver restarts (its pads read 0 again until the next command): never a tare over zeros, not valid
    meanwhile; a fresh tare at the new levels once the pads stream again."""
    pads = Pads()
    pads.streaming = {"left": True, "right": True}
    pub, calls = _pub(pads)
    t = [0]
    _run(pub, t, C.TARE_SAMPLES + 10)
    pads.streaming["left"] = False
    _run(pub, t, 20)
    assert calls[-1][2] is False and pub.last["tare"]["left"] == "waiting"
    assert pub.tare_state("left") == "waiting"
    pads.offset["left"][:] = 500.0  # the restarted pads come back at other levels
    pads.streaming["left"] = True
    _run(pub, t, C.TARE_SAMPLES + 100)
    assert calls[-1] == (0, 0, True)
    assert pub.tare["left"] is not None and abs(float(pub.tare["left"][1]) - (H2_LEVELS[1] + 500.0)) < 16.0


def test_the_tare_follows_the_pads_slow_drift_over_a_session() -> None:
    """HW2's stand_b: the tip / palm pads crept up >= ~20 raw in ~30 min and ~+100 raw over ~3 h after one tare. With a
    fixed tare they read "contact" at rest once the drift passes 40 (here after ~75 min) - a blind guard again; the
    drift-following tare keeps them at 0, and a touch (H4: >= 1,800) during or after is still a contact - also one held
    for 10 min (it is absorbed at most DRIFT_UP_RAW_S x 600 s = 60 raw). Fails on the base (no drift following)."""
    pads = Pads(seed=5)
    pads.streaming = {"left": True, "right": True}
    pub, calls = _pub(pads)
    t = [0]
    _run(pub, t, C.TARE_SAMPLES)
    rate = 100.0 / (3 * 3600.0)  # raw / s: ~+100 in 3 h
    for _ in range(2 * 3600):  # 2 h at 1 Hz ticks (the drift is slow; the tare's rate limit uses the real dt)
        for s in ("left", "right"):
            pads.offset[s][TIPS_PALMS] += rate
            pads.offset[s][[0, 2, 4]] -= 0.15  # the proximal pads fall (~-1,000 over ~2 h)
        _run(pub, t, 1, dt_ns=1_000_000_000)
    assert pads.offset["left"][1] > 60.0  # the drift is past the threshold by now
    assert max(pl | pr for pl, pr, _ in calls[-600:]) == 0  # the last 10 min at rest: no contact
    assert pub.last["drift_raw"]["left"] is not None and pub.last["drift_raw"]["left"][1] > 50.0
    base = C.ContactPublisher(pads, lambda *a, **k: None, drift_up_raw_s=0.0, drift_down_raw_s=0.0)
    base.tare = {s: H2_LEVELS.copy() for s in ("left", "right")}  # the fixed tare of the base, collected at the start
    base.tare0 = {s: H2_LEVELS.copy() for s in ("left", "right")}
    base.tick(t[0] + DT_NS)
    assert base.last["pad_left"] & (1 << 1)  # the fixed tare reads the drifted tip "in contact" at rest
    pads.offset["right"][3] += 1800.0  # a touch on finger A's tip, held 10 min
    hits = []
    for _ in range(600):
        _run(pub, t, 1, dt_ns=1_000_000_000)
        hits.append(calls[-1][1] & (1 << 3))
    assert all(hits)


def test_the_tare_stands_still_while_a_grip_holds() -> None:
    """A held object presses the pads for minutes: while r1d holds a grip on that side (set_grip, synced at r1d's tick)
    the tare does not follow - the pads keep reading their contact; after the release it follows again."""

    class Grip:
        state = "HOLDING"

        def stops(self) -> dict[str, str]:
            return {"thumb": "pad", "index": "pad"}

    pads = Pads(seed=7)
    pads.streaming = {"left": True, "right": True}
    pub, calls = _pub(pads)
    t = [0]
    _run(pub, t, C.TARE_SAMPLES)
    tare0 = np.asarray(pub.tare["right"]).copy()
    pub.set_grip("right", Grip())
    pads.offset["right"][[1, 3]] += 300.0
    _run(pub, t, 600, dt_ns=1_000_000_000)  # 10 min holding
    assert np.allclose(np.asarray(pub.tare["right"]), tare0)
    assert calls[-1][1] & (1 << 1) and calls[-1][1] & (1 << 3)
    pub.set_grip("right", None)
    pads.offset["right"][[1, 3]] -= 300.0
    _run(pub, t, 30, dt_ns=1_000_000_000)
    assert calls[-1][1] == 0


def test_hand_tare_re_collects_through_retare() -> None:
    """`hand.tare` (`r1 tare <side>`): the operator's re-tare forgets the tare and its drift; collected again."""
    pads = Pads()
    pads.streaming = {"left": True, "right": True}
    pub, _ = _pub(pads)
    t = [0]
    _run(pub, t, C.TARE_SAMPLES + 5)
    pub.retare("right")
    assert pub.tare["right"] is None and pub.tare0["right"] is None and pub.tare_state("right") == "waiting"
    _run(pub, t, 1)
    assert pub.tare_state("right") == "taring"
    _run(pub, t, C.TARE_SAMPLES)
    assert pub.tare_state("right") == "ok" and pub.drift_raw("right") == [0.0] * 9


def test_the_tare_waits_for_the_hand_to_open() -> None:
    """v2's rule (the hand open, touching nothing): r1d's first hand command may CLOSE the hand (a skill's fist) - its
    own curl presses the pads, and a tare taken then would leave every later reading below the zero (a less sensitive
    guard). The tare waits until the hand is open; the drift follower stands still while it is closed. Fails on the
    base (the tare over the zeros / over the closed hand)."""
    from r1mc3.hands import grip as G

    pads = Pads()
    pads.streaming = {"left": True, "right": True}
    fist = {s: G.preset("fist", s) for s in ("left", "right")}
    pads.q = dict(fist)
    pads.offset["right"][[0, 2, 4]] = 600.0  # the curled fingers press the proximal pads
    pub, calls = _pub(pads)
    t = [0]
    _run(pub, t, 200)
    assert calls[-1] == (0, 0, False) and pub.tare_state("right") == "waiting"
    assert any("not open" in w for w in pub.last["words"])
    pads.q = {s: G.preset("open", s) for s in ("left", "right")}
    pads.offset["right"][[0, 2, 4]] = 0.0
    _run(pub, t, C.TARE_SAMPLES + 50)
    assert calls[-1] == (0, 0, True) and pub.tare_state("right") == "ok"
    tare = np.asarray(pub.tare["right"]).copy()
    pads.q = dict(fist)  # closed again (after the tare): its pads read their curl, the zero stays where it was
    pads.offset["right"][[0, 2, 4]] = 600.0
    _run(pub, t, 300, dt_ns=1_000_000_000)
    assert np.allclose(np.asarray(pub.tare["right"]), tare)
    assert calls[-1][1] & 0b10101 == 0b10101  # the curl reads as contact on pads 0 / 2 / 4 (what the pads feel)
