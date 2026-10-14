"""The grasp's finger stops and state in the hands-contact record (r1mc3.hands.contact.grip_bits / set_grip; the
types.h r1rt_hands_contact fields that stayed 0 until session 9)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.hands import contact as C
from r1mc3.hands import grip as G

from tests.unit.grip_fake import FakeHand, closure_stop, run


class _Hands:
    """Both hands for the publisher: the right one is a FakeHand, the left one idle."""

    def __init__(self, right: FakeHand) -> None:
        self.right = right
        self.left = FakeHand("left")

    def state(self, side: str) -> dict[str, Any]:
        return (self.right if side == "right" else self.left).state(side)


def test_without_a_grasp_the_record_is_pads_only_as_before() -> None:
    calls: list[tuple[Any, ...]] = []
    pub = C.ContactPublisher(_Hands(FakeHand()), lambda pl, pr, t_ns, valid: calls.append((pl, pr, t_ns, valid)))
    pub.tick(1)
    assert calls == [(0, 0, 1, True)]


def test_a_grasp_puts_its_stops_and_state_into_the_record() -> None:
    h = FakeHand()
    # the index stalls on a surface without a pad signal (pads silent), the thumb closes on nothing
    h.pad_gain_kpa = 0.0
    h.stops = {"index": closure_stop("right", (5, 6), 0.4)}
    r = G.GraspRunner(h, "right", "wrap", "rigid")
    r.start(0)
    run(r, h, 9.0)
    assert r.stops()["index"] == "stall" and r.state == "NOT_HOLDING"
    stalled, blocked, code = C.grip_bits(r)
    assert stalled == 1 << C.FINGER_BIT["index"] and blocked == 0 and code == C.GRIP_STATE_CODE["NOT_HOLDING"]
    got: dict[str, Any] = {}
    pub = C.ContactPublisher(_Hands(h), lambda pl, pr, **kw: got.update(kw, pl=pl, pr=pr))
    pub.set_grip("right", r)
    pub.tick(5)
    assert got["stalled_right"] == stalled and got["grip_right"] == code and got["stalled_left"] == 0
    pub.set_grip("right", None)
    assert pub.grips == {}
    with pytest.raises(ValueError):
        pub.set_grip("middle", r)


def test_the_state_codes_cover_every_state() -> None:
    assert set(C.GRIP_STATE_CODE) == set(G.STATES)
    assert len(set(C.GRIP_STATE_CODE.values())) == len(G.STATES)
    assert np.all(np.asarray(sorted(C.FINGER_BIT.values())) == np.arange(3))
