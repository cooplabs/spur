"""Session 13: r1d says a hand is warm BEFORE the grip's 72 C limit - in every hand.grasp reply (`hand_heat`) and in
its status (`hands_heat`); the REAL v3 core in lockstep + the grip tests' geometric fake hands (test_service_l2_hands's
rig). HW2 day 2's 4th back-to-back wrap of a box ended "hand hot (72 C >= 72)" with nothing said before. On the base
(dcc01a2) neither key exists."""

from __future__ import annotations

from typing import Any

import pytest
from r1sim import cpp_policy, world

from tests.integration.test_service_l2_hands import HandsRig, _object

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def test_a_warm_hand_is_said_in_the_grasp_reply_and_the_status(w: world.World) -> None:
    rig = HandsRig(w)
    _object(rig, "left", 0.040)
    rig.hands.h["left"].temperature = 66.0  # a firm wrap's heat (HW2 day 2's wraps of a box: 63 / 60 / 67 / 72 C)
    res: dict[str, Any] = {}
    rig.at(1.0, lambda: res.__setitem__("g", rig.svc.submit({"op": "hand.grasp", "side": "left", "grip": "wrap",
                                                             "width_m": 0.04, "wait": True})))  # fmt: skip
    rig.run(4.0)
    g = res["g"].result
    assert g is not None and g.ok, g
    heat = g.data["hand_heat"]
    assert heat["level"] == "warm" and heat["temperature_c"] == 66.0 and heat["limit_c"] == 72.0, heat
    assert "s of squeeze left" in heat["words"][0]
    st = rig.svc.status()
    assert st["hands_heat"]["left"]["level"] == "warm" and st["hands_heat"]["right"]["level"] == "ok"
    rig.svc.submit({"op": "hand.release", "side": "left", "wait": True})
    rig.run(1.0)
    rig.hands.h["left"].temperature = 50.0  # cooled after the release: the 66 C peak still predicts the next grip
    st = rig.svc.status()
    assert st["hands_heat"]["left"]["level"] == "warm" and st["hands_heat"]["left"]["peak_c"] == 66.0
    assert "the next firm grip may reach it" in st["hands_heat"]["left"]["words"][0]
