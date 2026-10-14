"""r1d's `hand.calibrate {side, kind: "air_pinch"}` (session 10, DESIGN §6.2 / D27) with the PHYSICAL simulated Dex3
hands (r1sim.hands_rig: the sim + the v3 core + r1d in lockstep, the hands through r1hands' interface).

Session 9 left `thin_flexible` grasps refused in r1d (INVALID_ARGS): the grip needs each hand's air-pinch reference
(the simulated hand's tips stop 2.19 mm short of the model's shells with nothing between them) and r1d had no word to
measure it. Now: the hand closes a tip pinch on nothing, r1d keeps the clearance as `layer_zero_m`, and the thin pinch
runs - an air pinch with the reference is NOT_HOLDING (without it, it would read a 2.2 mm "layer" and pass).

On the base (a041731): hand.calibrate -> UNKNOWN_COMMAND; a thin_flexible grasp -> INVALID_ARGS before and after.
"""

from __future__ import annotations

from typing import Any

import pytest
from r1sim import cpp_policy, world
from r1sim.hands_rig import ServiceRig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _thin(rig: ServiceRig, side: str = "right") -> dict[str, Any]:
    return rig.call({"op": "hand.grasp", "side": side, "grip": "tip", "object_class": "thin_flexible", "wait": True},
                    max_sim_s=12.0)  # fmt: skip


def test_the_air_pinch_word_makes_a_thin_pinch_judgeable(w: world.World, tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("R1_HAND_CAL_FILE", str(tmp_path / "hand_cal.json"))
    rig = ServiceRig(w, planner=False)
    rig.run(1.0)
    before = _thin(rig)
    assert not before["ok"] and before["code"] == "INVALID_ARGS" and not before["moved"], before
    assert "hand.calibrate" in before["details"]["reason"]  # the refusal names r1d's word
    assert rig.svc.status()["hand_calibration"]["air_pinch"]["right"] is None
    cal = rig.call({"op": "hand.calibrate", "side": "right", "kind": "air_pinch", "wait": True}, max_sim_s=12.0)
    assert cal["ok"] and cal["data"]["state"] == "succeeded", cal
    z = cal["data"]["layer_zero_m"]
    # the simulated hand's air pinch: the tips stop ~2.2 mm short of the model's capsules (S9.md, grip)
    assert 0.0010 < z < 0.0040, cal["data"]
    assert cal["data"]["verdict"]["evidence"]["terms"]["tip_pressure"]["ok"]
    ap = rig.svc.status()["hand_calibration"]["air_pinch"]["right"]
    assert ap["layer_zero_mm"] == pytest.approx(1000 * z, abs=1e-3)
    assert (tmp_path / "hand_cal.json").exists()  # kept for a restarted r1d
    # the hand opened again after the calibration
    q = rig.hands.state("right")["q"]
    assert max(abs(float(v)) for v in q[1:]) < 0.3
    after = _thin(rig)  # an air pinch, now judged against the hand's own reference
    assert not after["ok"] and after["code"] == "NOT_HOLDING", after
    v = after["data"]["verdict"]
    assert v["state"] == "NOT_HOLDING" and v["reason"].startswith("layer"), v
    assert abs(v["evidence"]["terms"]["layer"]["layer_mm"]) < 0.5
    # the other hand is still uncalibrated: refused
    left = _thin(rig, "left")
    assert left["code"] == "INVALID_ARGS"
    # a restarted r1d loads the file
    rig2 = ServiceRig(w, planner=False)
    assert rig2.svc.status()["hand_calibration"]["air_pinch"]["right"]["source"] == "file"
