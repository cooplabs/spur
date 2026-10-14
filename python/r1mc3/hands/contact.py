"""r1hands' contact summary into the segment (`hands_contact`, types.h `r1rt_hands_contact`; DESIGN §3.3 / §6.3).

At the hand rate (r1d's 100 Hz tick) the Dex3 pads' state becomes one bit per pad (bit k = pad k of 9 in contact) per
hand, written through the binding's `SegmentView.write_hands_contact` (r1hands is that region's single writer); the
core's trajectory guards read it. `valid` = 0 when a hand's state is stale (> STATE_STALE_S) or the hands are absent.

A pad is "in contact" when its largest taxel rises more than `threshold` above that pad's TARE (the median of its
largest taxel over the first `TARE_SAMPLES` states in which the pads STREAM, the hand open at start).

Session 12 (HW2 day 1, 12 Oct 2026, results/hw2_20261012/HW2_PART1.md H2 / H4):
  - the threshold is **40 raw** (5 steps of the pads' 8-unit quantisation; the user's decision 12 Oct ~17:55): at rest
    the real pads read ~100,000 raw with a noise sd of 5-8 and at most +16 above their tare in 10 s (H2, both hands),
    while every touch the pads register at all is >= ~1,800 raw (H4, the force gauge: the TIP pads answer only above
    ~10 N and by WHERE the tip lands - a firm-contact switch, no usable raw-per-newton gain). 5 raw (the estimate
    before HW2) put 0.4-45 % of the at-rest samples "in contact" per pad;
  - **the tare waits for the pads to stream**: the Dex3 hand's state is valid before its pads stream - every taxel reads
    exactly 0 until the first hand command - and the tare taken over those zeros made all 9 pads of both hands "in
    contact" from then on (511 / 511), so the executor's contact guard (it counts only pads that ROSE after a path
    began) could never fire. Now a state counts toward the tare only when every pad's largest taxel is above
    `STREAM_MIN_RAW` and the hand is OPEN (v2's rule: a first command that closes it would put the curl's own pressure
    into the zero); while a hand's pads do not stream (or stop streaming: back to zeros, e.g. the hand driver
    restarted) the record is NOT valid - a contact-guarded move stops at once (the executor's CONTACT_STALE), never
    runs with a blind guard - and `last["tare"]` / `last["words"]` say why (r1 status). The tare is re-collected by
    itself when the stream comes back; `hand.tare` (`r1 tare <side>`) stays the operator's word for a re-tare with the
    hand open and touching nothing;
  - **the tare follows the pads' slow drift** while the side holds no grip: HW2's r1log (stand_b, the 30 B4 grasps,
    one tare at 01:30 UTC) shows the TIP and PALM pads of BOTH hands - the idle left one too - creeping UP: their share
    of samples > 5 raw over the tare went 0.3 -> 0.9 in 10-15 min and 1.0 from ~30 min on (a mean >= ~20 raw), and
    ~3 h later (H4) they sat ~+100 raw above H2's tares; the proximal pads drift DOWN (-41 within 10 s of opening,
    -1,000..-1,500 by H4). At 40 raw a single tare would read "contact" at rest within an hour or so - a blind guard
    again. So each pad's tare moves toward its reading by at most `DRIFT_UP_RAW_S` / `DRIFT_DOWN_RAW_S` per second
    (frozen while r1d holds a grip on that side - `set_grip` - and while the hand is not open): ~10 x the observed
    drift (~0.01 raw / s), while a touch the pads register at all (>= ~1,800 raw) would take hours to be absorbed and
    a contact-guarded move decides in tens of milliseconds. Equal rates both ways keep the tare at the pad's median (a
    faster fall would park it at the low edge of the noise: the sim's 100-unit pad noise would then read "contact" all
    the time).

Session 9: while a grasp runs (r1mc3.hands.grip.GraspRunner, registered with `set_grip`), its finger stops and state go
into the same record: `finger_stalled` / `finger_blocked` bit per finger (thumb 0, middle 1, index
2: types.h r1rt_hands_contact) and `grip_state` (GRIP_STATE_CODE); with no grasp registered the record carries pads only
(as before).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

N_PADS = 9
TARE_SAMPLES = 50  # 0.5 s at 100 Hz
CONTACT_THRESHOLD = 40.0  # raw pad units above the tare (HW2 H2 / H4 + the user's decision 12 Oct 2026; was 5.0)
DRIFT_UP_RAW_S = 0.1  # raw / s the tare may rise (HW2 stand_b: tip / palm pads ~+20 raw in ~30 min, ~+100 in ~3 h)
DRIFT_DOWN_RAW_S = 0.1  # raw / s it may fall (the proximal pads: ~-1,000 over ~3 h)
OPEN_TOL_RAD = 0.25  # the hand counts as open for the tare: every closing joint within this of the open preset (HW2
#                     H1: open within 0.07 rad, thumb_1 0.04-0.06 short); the thumb's rotation (joint 0) is not judged
STREAM_MIN_RAW = 1000.0  # a pad whose largest taxel is at / below this is not streaming (HW2: exactly 0 before the
#                          first hand command; ~100,000 raw at rest once streaming; the sim's pads ~101 kPa absolute)
SIDES = ("left", "right")
FINGER_BIT = {"thumb": 0, "middle": 1, "index": 2}  # types.h r1rt_hands_contact.finger_stalled / finger_blocked
GRIP_STATE_CODE = {"OPEN": 0, "CLOSING": 1, "CONTACT": 2, "TENTATIVE": 3, "HOLDING": 4, "NOT_HOLDING": 5}
STALL_STOPS = ("stall", "torque", "self")  # a finger stopped without a pad (on something - or the hand itself)


def grip_bits(runner: Any) -> tuple[int, int, int]:
    """(stalled bits, blocked bits, grip state code) of a GraspRunner (anything with `stops()` and `state`)."""
    st = runner.stops()
    stalled = sum(1 << FINGER_BIT[g] for g, why in st.items() if why in STALL_STOPS)
    blocked = sum(1 << FINGER_BIT[g] for g, why in st.items() if why == "blocked")
    return stalled, blocked, GRIP_STATE_CODE.get(str(runner.state), 0)


class ContactPublisher:
    """`tick(now_ns)` at the hand rate: the hands' pad state -> `writer(pad_left, pad_right, t_ns=..., valid=...)`."""

    def __init__(
        self,
        hands: Any,
        writer: Callable[..., Any],
        *,
        threshold: float = CONTACT_THRESHOLD,
        tare_samples: int = TARE_SAMPLES,
        drift_up_raw_s: float = DRIFT_UP_RAW_S,
        drift_down_raw_s: float = DRIFT_DOWN_RAW_S,
    ) -> None:
        self.hands = hands
        self.writer = writer
        self.threshold = float(threshold)
        self.tare_samples = int(tare_samples)
        self._tare_buf: dict[str, list[np.ndarray]] = {s: [] for s in SIDES}
        self.tare: dict[str, np.ndarray | None] = {s: None for s in SIDES}
        self.last: dict[str, Any] = {}
        self.writes = 0
        self.grips: dict[str, Any] = {}  # side -> the GraspRunner of the grasp in progress (set_grip)
        self.drift_up = float(drift_up_raw_s)
        self.drift_down = float(drift_down_raw_s)
        self._t_last: dict[str, int | None] = {s: None for s in SIDES}
        self.tare0: dict[str, np.ndarray | None] = {s: None for s in SIDES}  # as collected (drift = tare - tare0)
        self._tare_status: dict[str, str] = {s: "waiting" for s in SIDES}
        self._why: dict[str, str] = {s: "" for s in SIDES}

    def set_grip(self, side: str, runner: Any | None) -> None:
        """r1d registers the grasp of `side` (None: none) so its finger stops / state go into the contact record."""
        if side not in SIDES:
            raise ValueError(f"side: one of {SIDES}")
        if runner is None:
            self.grips.pop(side, None)
        else:
            self.grips[side] = runner

    def retare(self, side: str) -> None:
        """Forget `side`'s tare: it is collected again over the next `tare_samples` streaming states (hand.tare)."""
        if side not in SIDES:
            raise ValueError(f"side: one of {SIDES}")
        self.tare[side] = None
        self.tare0[side] = None
        self._tare_buf[side] = []
        self._tare_status[side] = "waiting"

    def drift_raw(self, side: str) -> list[float] | None:
        """How far each pad's tare has followed its drift since it was collected (raw; None = no tare)."""
        t, t0 = self.tare[side], self.tare0[side]
        return None if t is None or t0 is None else [round(float(x), 1) for x in t - t0]

    @staticmethod
    def _open(side: str, st: Mapping[str, Any]) -> bool:
        """The hand open (every closing joint within OPEN_TOL_RAD of the open preset); a state without joints counts
        as open (nothing to judge)."""
        q = st.get("q")
        if q is None:
            return True
        from .grip import preset

        qa = np.asarray(q, dtype=np.float64)
        if qa.size != 7:
            return True
        return bool(np.max(np.abs(qa[1:] - preset("open", side)[1:])) <= OPEN_TOL_RAD)

    def tare_state(self, side: str) -> str:
        """The tare: "ok", "taring" (collecting over streaming states) or "waiting" (the pads do not stream yet)."""
        return "ok" if self.tare[side] is not None else self._tare_status.get(side, "waiting")

    def _bits(self, side: str, now_ns: int = 0) -> tuple[int, bool]:
        st = self.hands.state(side)
        p = st.get("pressure")
        if not st.get("valid") or p is None:
            self._why[side] = f"{side} hand state not valid ({st.get('invalid_reason') or 'no pressure'})"
            return 0, False
        peak = np.max(np.asarray(p, dtype=np.float64).reshape(N_PADS, -1), axis=1)
        dead = [k for k in range(N_PADS) if not peak[k] > STREAM_MIN_RAW]
        if dead:  # not streaming (HW2: every taxel 0 until the first hand command) - never a tare, never "valid"
            if self.tare[side] is not None or self._tare_buf[side]:
                self.retare(side)  # the stream stopped (e.g. the hand driver restarted): a fresh tare when it is back
            self._tare_status[side] = "waiting"
            self._why[side] = (
                f"{side} pads not streaming yet ({len(dead)} of {N_PADS} pads read <= {STREAM_MIN_RAW:.0f} raw): no "
                f"tare - send any hand command (e.g. open the hand) first; contact-guarded moves stop until then"
            )
            return 0, False
        if self.tare[side] is None and not self._open(side, st):
            # v2's rule: tare with the hand open (touching nothing) - a first hand command that closes it (a skill's
            # fist) would put the curl's own pad pressure into the zero; the tare waits for the hand to open
            self._tare_status[side] = "waiting"
            self._why[side] = (
                f"{side} pads stream but the hand is not open: the tare waits for it to open (e.g. "
                f"`r1 hand {side} ...` with H1's open words); contact-guarded moves stop until then"
            )
            return 0, False
        if self.tare[side] is None:
            self._tare_buf[side].append(peak)
            self._tare_status[side] = "taring"
            self._why[side] = f"{side} pads taring ({len(self._tare_buf[side])} of {self.tare_samples} states)"
            if len(self._tare_buf[side]) >= self.tare_samples:
                tare_new = np.median(np.stack(self._tare_buf[side]), axis=0)
                self.tare[side], self.tare0[side] = tare_new, tare_new.copy()
                self._t_last[side] = int(now_ns)
                self._why[side] = ""
            return 0, True  # taring: no contact claimed yet (valid state, the pads stream)
        tare = self.tare[side]
        assert tare is not None
        on = peak - tare > self.threshold
        t_last = self._t_last[side]
        dt = 0.0 if t_last is None else max(0.0, min(1.0, (int(now_ns) - t_last) * 1e-9))
        self._t_last[side] = int(now_ns)
        if side not in self.grips and dt > 0.0 and self._open(side, st):  # follow the slow drift (module
            # docstring) only while the hand is open: frozen during a grip and while closed (its own curl presses pads)
            self.tare[side] = tare + np.clip(peak - tare, -self.drift_down * dt, self.drift_up * dt)
        return int(sum(1 << k for k in range(N_PADS) if on[k])), True

    def tick(self, now_ns: int) -> None:
        lb, lv = self._bits("left", now_ns)
        rb, rv = self._bits("right", now_ns)
        info = {
            "tare": {s: self.tare_state(s) for s in SIDES},
            "drift_raw": {s: self.drift_raw(s) for s in SIDES},
            "threshold_raw": self.threshold,
            "words": [w for w in (self._why[s] for s in SIDES) if w],
        }
        if not self.grips:
            self.writer(lb, rb, t_ns=int(now_ns), valid=bool(lv and rv))
            self.last = {"pad_left": lb, "pad_right": rb, "valid": bool(lv and rv), "t_ns": int(now_ns), **info}
        else:
            g = {s: grip_bits(self.grips[s]) if s in self.grips else (0, 0, 0) for s in SIDES}
            kw = {
                "stalled_left": g["left"][0],
                "stalled_right": g["right"][0],
                "blocked_left": g["left"][1],
                "blocked_right": g["right"][1],
                "grip_left": g["left"][2],
                "grip_right": g["right"][2],
            }
            self.writer(lb, rb, t_ns=int(now_ns), valid=bool(lv and rv), **kw)
            self.last = {"pad_left": lb, "pad_right": rb, "valid": bool(lv and rv), "t_ns": int(now_ns), **kw, **info}
        self.writes += 1
