"""Hand self-contact (session 3, the A8 finding; models/provenance.yaml "hand_self_contact").

Session 1 excluded EVERY pair of a Dex3 hand's links (after v2's sim: "the real hand is built not to collide with
itself"), so the simulated thumb passed through the curled fingers and the 9 Oct false hold could not happen (A8).
The real hand does stop on itself: the logged thumb_1 stayed at -0.56..-0.63 of its -0.9 command while thumb_2 ran on,
palm pads 0 (two logged steps), and v2's verdict has a rule for pinch tips that meet (hands.py TIP_MEET_M). Since
session 3 r1sim.model excludes only the palm-thumb pairs (model.HAND_SELF_EXCLUDES), whose collision shapes overlap
by construction. Pinned here, for the default primitives and the opt-in mesh-hull variant (dex3.COLLISION):

* the compiled model excludes exactly those pairs (+ the elbow-palm pair) - FAILS on the S1/S2 model (28 pairs/hand);
* the thumb can touch the fingers: the pinch presets put thumb_2 into index_1 / middle_1 - FAILS on the S1/S2 model;
* why the palm-thumb pairs stay excluded: with them on, palm-thumb_1 overlaps from thumb_1 ~ -0.12..-0.15 rad
  (primitives; the real thumb closes much further freely) and in EVERY pose with the mesh hull - shape artefacts;
* (guards, pass on S1 too) no hand self-contact in the open / relax / spread / open_wide presets, and none over 1 s
  of the robot standing under the policy with the hands open or relaxed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator

import mujoco
import numpy as np
import pytest
from r1mc3.policy import POLICY_DIR, ReferencePolicy
from r1sim import control, dex3, model, world
from r1sim.control import PolicyController
from r1sim.engine import LowCmd, LowState, Simulation

SIDE = "right"
NAMES = ("palm", *dex3.NAMES)


@pytest.fixture(params=dex3.COLLISIONS)
def collision(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setattr(dex3, "COLLISION", request.param)
    yield str(request.param)


def _hand_bodies(m: mujoco.MjModel, side: str) -> dict[int, str]:
    return {m.body(f"{side}_hand_{n}_link").id: n for n in NAMES}


def _self_pairs(m: mujoco.MjModel, d: mujoco.MjData, side: str) -> dict[tuple[str, str], float]:
    """Hand-link pairs in contact (deepest penetration, m) after mj_forward."""
    hb = _hand_bodies(m, side)
    out: dict[tuple[str, str], float] = {}
    for c in d.contact[: d.ncon]:
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 in hb and b2 in hb:
            a, b = sorted((hb[b1], hb[b2]))
            out[(a, b)] = min(out.get((a, b), 1.0), float(c.dist))
    return out


def _posed(m: mujoco.MjModel, q: np.ndarray) -> mujoco.MjData:
    d = mujoco.MjData(m)
    idx = model.RobotIndex(m)
    for s in dex3.SIDES:
        d.qpos[idx.hand_qadr[s]] = q if s == SIDE else dex3.preset("open", s)
    mujoco.mj_forward(m, d)
    return d


def _model_without_hand_excludes() -> mujoco.MjModel:
    root = ET.fromstring(model.robot_xml("real", visual=False))
    contact = root.find("contact")
    assert contact is not None
    for ex in list(contact):
        if "_hand_" in ex.get("body1", "") and "_hand_" in ex.get("body2", ""):
            contact.remove(ex)
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode")).compile()


def test_only_the_palm_thumb_pairs_are_excluded(collision: str) -> None:
    m = model.build_model("real", visual=False)
    got = set()
    for i in range(int(m.nexclude)):
        v = int(m.exclude_signature[i])
        got.add(tuple(sorted((m.body(v >> 16).name, m.body(v & 0xFFFF).name))))
    want = set()
    for side in dex3.SIDES:
        want.add(tuple(sorted((f"{side}_elbow_link", f"{side}_hand_palm_link"))))
        for a, b in model.HAND_SELF_EXCLUDES:
            want.add(tuple(sorted((f"{side}_hand_{a}_link", f"{side}_hand_{b}_link"))))
    hand_got = {p for p in got if "_hand_" in p[0] or "_hand_" in p[1]}
    assert hand_got == want, sorted(hand_got ^ want)
    assert model.HAND_SELF_EXCLUDES == (("palm", "thumb_1"), ("palm", "thumb_2"))


@pytest.mark.parametrize(("preset", "finger"), [("pinch_index", "index_1"), ("pinch_middle", "middle_1")])
def test_the_thumb_touches_the_finger_it_pinches(collision: str, preset: str, finger: str) -> None:
    """The pinch presets were solved for tip-to-tip contact: the thumb's distal link now meets the finger (S1: it
    passed through)."""
    m = model.build_model("real", visual=False)
    pairs = _self_pairs(m, _posed(m, dex3.preset(preset, SIDE)), SIDE)
    assert (finger, "thumb_2") in pairs, pairs
    assert not any("palm" in p for p in pairs)  # the excluded pairs stay quiet


@pytest.mark.parametrize("preset", ["open", "relax", "spread", "open_wide"])
def test_no_self_contact_in_the_resting_presets(collision: str, preset: str) -> None:
    m = model.build_model("real", visual=False)
    assert _self_pairs(m, _posed(m, dex3.preset(preset, SIDE)), SIDE) == {}


def test_why_the_palm_thumb_pairs_stay_excluded(collision: str) -> None:
    """With every hand exclude dropped, palm-thumb_1 overlaps early (primitives) or always (mesh hull): the shapes'
    artefacts, not contacts of the real hand. If a shape fix removes this overlap, revisit HAND_SELF_EXCLUDES."""
    m = _model_without_hand_excludes()
    first = None
    for t1 in np.linspace(0.0, -1.0, 101):  # right thumb_1 closes negative
        q = dex3.preset("open", SIDE).copy()
        q[1] = t1
        if ("palm", "thumb_1") in _self_pairs(m, _posed(m, q), SIDE):
            first = float(t1)
            break
    assert first is not None
    if collision == "mesh":
        assert first == 0.0  # the hull fills the thumb slot: overlapping in every pose
    else:
        assert -0.2 <= first <= -0.1, first  # long before the real thumb's free closing (-0.56 and beyond)
    fist = _self_pairs(m, _posed(m, dex3.preset("fist", SIDE)), SIDE)
    assert set(fist) == {("palm", "thumb_1"), ("palm", "thumb_2")}, fist  # the fist preset: only palm-thumb


class _HandsAt:
    """The balance policy with both hands held at a preset."""

    def __init__(self, preset: str) -> None:
        self.ctl = PolicyController(ReferencePolicy.load(POLICY_DIR))
        self.preset = preset

    def __call__(self, st: LowState) -> LowCmd:
        cmd = self.ctl(st)
        control._open_hands(cmd, dex3.SIDES, preset=self.preset)
        return cmd


@pytest.mark.parametrize("preset", ["open", "relax"])
def test_no_self_contact_while_standing(preset: str) -> None:
    w = world.build_world("real", visual=False)
    sim = Simulation(w.model, w.index)
    sim.reset(w.initial_qpos())
    sim.add_controller(_HandsAt(preset), 1.0 / 500.0, "policy")
    worst: dict[tuple[str, str], float] = {}
    for _ in range(250):  # 1 s, checked every 4 ms
        sim.run_for(0.004)
        for side in dex3.SIDES:
            for k, v in _self_pairs(w.model, sim.d, side).items():
                worst[k] = min(worst.get(k, 1.0), v)
    assert worst == {}, worst
