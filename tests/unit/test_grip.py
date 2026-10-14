"""r1mc3.hands.grip (DESIGN.md §6.2, D27): the grasp sequence and the evidence verdict, per object class, on a fake
hand whose stops are geometric (tests/unit/grip_fake.py: the model's kinematics decide where a finger is blocked).

The cases are the ones the verdict exists for: an empty close, a real wrap, the 9 Oct false hold's mechanism (the
rotated thumb landing on the curled index - the tips still ~15 mm apart, so a gap rule alone passes it), fingertips
pressing the surface, a low grasp on an object's sides, a slip at the lift, a thin layer between the tips vs tips that
met (an air pinch), a hot hand, a stale hand. v2's verdict (force alone) would say "holding" in several of them: the
force term is reported in each, so the test shows the force passing where the verdict correctly refuses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.hands import grip as G

from tests.unit.grip_fake import FakeHand, aperture_stop, closure_stop, layer_stop, run

SIDE = "right"
DONE = ("HOLDING", "NOT_HOLDING")


def _grasp(hand: FakeHand, grip: str = "wrap", cls: str = "rigid", **kw: Any) -> tuple[G.GraspRunner, float]:
    r = G.GraspRunner(hand, SIDE, grip, cls, **kw)
    r.start(0)
    t = run(r, hand, 9.0)
    return r, t


def _kept(r: G.GraspRunner) -> bool:
    """keeps_grip read through a call (mypy narrows a property across the runner's own state changes)."""
    return bool(r.keeps_grip)


def _lift(r: G.GraspRunner, hand: FakeHand, t: float, dz: float = 0.015) -> float:
    r.notify_lift(dz)
    return run(r, hand, 1.0, t, DONE)


# ------------------------------------------------------------------------------------------------ the hand model
def test_presets_and_kinematics_equal_the_simulators() -> None:
    import mujoco
    from r1sim import dex3

    for side in ("left", "right"):
        for name in G.PRESETS_LEFT:
            assert np.allclose(G.preset(name, side), dex3.preset(name, side))
        assert np.allclose(G.close_dir(side), dex3.CLOSE_DIR[side])
        for grip, kind in (("wrap", "power"), ("pinch", "pinch_index")):
            assert np.allclose(G.preshape(grip, side), dex3.preshape(kind, side))
        rng = np.random.default_rng(3)
        for _ in range(5):
            q = dex3.clamp(rng.uniform(dex3.Q_MIN[side], dex3.Q_MAX[side]), side)
            sim = dex3.link_poses(side, q)
            ours = G.link_frames(side, q)
            for link, (p, quat) in sim.items():
                r = np.zeros(9)
                mujoco.mju_quat2Mat(r, quat)
                assert np.allclose(ours[link][0], p, atol=1e-9)
                assert np.allclose(ours[link][1], r.reshape(3, 3), atol=1e-9)
            # the tip sites (r1sim.dex3.hand_xml: fingers TIP along x, the thumb along the grasp side g*y)
            g = -1.0 if side == "left" else 1.0
            tp = G.tips(side, q)
            for name, link, off in (("thumb", "thumb_2", (0, dex3.TIP * g, 0)), ("index", "index_1", (dex3.TIP, 0, 0))):
                o = np.zeros(3)
                mujoco.mju_rotVecQuat(o, np.asarray(off, dtype=np.float64), sim[link][1])
                assert np.allclose(tp[name], sim[link][0] + o, atol=1e-9)


def test_the_rotated_thumb_wrap_runs_the_thumb_into_the_index_with_the_tips_apart() -> None:
    """The 9 Oct mechanism in the model: a wrap from the pinch pre-shape (thumb rotated) - the thumb's capsules meet
    the index's while the tip points are still ~15 mm apart (a gap rule alone would not see it)."""
    q0 = G.preshape("pinch", SIDE)
    qc = G.preset("fist", SIDE)
    qc[0] = q0[0]
    hit = None
    for s in np.linspace(0.0, 1.0, 201):
        q = q0 + s * (qc - q0)
        if G.thumb_finger_clearance(SIDE, q)["index"] <= 0.0:
            hit = q
            break
    assert hit is not None
    tp = G.tips(SIDE, hit)
    assert 0.012 < float(np.linalg.norm(tp["thumb"] - tp["index"])) < 0.025
    # an open-thumb wrap never does (the capsules stay >= 10 mm apart)
    q0 = G.preshape("wrap", SIDE)
    assert (
        min(G.thumb_finger_clearance(SIDE, q0 + s * (G.preset("fist", SIDE) - q0))["index"] for s in (0.5, 1.0)) > 0.01
    )


def test_empty_aperture_is_where_the_tip_shells_meet() -> None:
    z = G.empty_aperture(SIDE, "index")
    assert 0.003 < z < 0.012  # the presets put the tip POINTS together (0.2 mm): the shells meet ~7 mm earlier
    assert G.empty_aperture("left", "index") == pytest.approx(z, abs=1e-6)


def test_bad_arguments_are_refused() -> None:
    h = FakeHand()
    for kw in (
        {"side": "middle"},
        {"grip": "fist"},
        {"object_class": "rubber_toy"},
        {"width_m": 0.0},
        {"squeeze": 1.0},
    ):
        args: dict[str, Any] = {"side": SIDE, "grip": "wrap", "object_class": "rigid", **kw}
        with pytest.raises(ValueError):
            G.GraspRunner(h, args.pop("side"), args.pop("grip"), args.pop("object_class"), **args)


# ------------------------------------------------------------------------------------------------ rigid
def test_an_empty_wrap_is_not_holding_and_keeps_its_pose_without_pushing() -> None:
    """Session 11 (contact decides): an empty close is NOT_HOLDING and the hand stays where it closed - no push (every
    group closed on nothing: its hold is where it stopped), no back-off (a verdict never relaxes the hand by itself;
    before session 11 every closing joint was commanded 0.03 rad back), powered (kp 1.5), until the caller opens it."""
    h = FakeHand()
    r, t = _grasp(h)
    v = r.verdict()
    assert v.state == "NOT_HOLDING" and v.reason is not None and v.reason.startswith("opposition")
    assert v.evidence["stops"] == {"thumb": "closed", "index": "closed", "middle": "closed"}
    q_verdict = h.q.copy()
    run(r, h, 1.0, t, ())
    d = G.close_dir(SIDE)
    assert np.all(np.abs(h.q[1:] - q_verdict[1:]) < 0.01), h.q - q_verdict  # stayed (base: backed off 0.03)
    assert np.all(np.abs(d[1:] * (h.cmd[1:] - h.q[1:])) < 0.01), h.cmd - h.q
    assert np.all(h.kp == pytest.approx(1.5)) and r.keeps_grip


def test_a_rigid_wrap_on_an_object_is_tentative_then_holding_after_the_lift() -> None:
    h = FakeHand()
    h.stops = {
        "thumb": aperture_stop(SIDE, "index", 0.05),
        "index": aperture_stop(SIDE, "index", 0.05),
        "middle": aperture_stop(SIDE, "middle", 0.05),
    }
    r, t = _grasp(h, width_m=0.05)
    v = r.verdict()
    assert v.state == "TENTATIVE", v.reason
    terms = v.evidence["terms"]
    assert terms["opposition"]["ok"] and terms["gap"]["ok"] and terms["force"]["ok"]
    assert terms["gap"]["aperture_mm"] == pytest.approx(50.0, abs=2.0)
    assert v.confidence == pytest.approx(0.6)
    _lift(r, h, t)
    v = r.verdict()
    assert v.state == "HOLDING" and v.evidence["terms"]["lift"]["ok"] and v.confidence == pytest.approx(1.0)


def test_a_slip_at_the_lift_is_not_holding() -> None:
    """The object leaves the hand as the arm lifts: the fingers run on past the verdict's pose."""
    h = FakeHand()
    st = aperture_stop(SIDE, "index", 0.05)
    h.stops = {"thumb": st, "index": st, "middle": aperture_stop(SIDE, "middle", 0.05)}
    r, t = _grasp(h, width_m=0.05)
    assert r.state == "TENTATIVE"
    h.stops = {}
    h.blocked = {}
    _lift(r, h, t)
    v = r.verdict()
    assert v.state == "NOT_HOLDING" and v.reason is not None and v.reason.startswith("lift")
    assert v.evidence["terms"]["lift"]["ok"] is False


def _creep_stop(idx: list[int], limit: float) -> Any:
    """Block a group once it closes past `limit` (max over its joints of q along the closing direction, rad)."""
    d = G.close_dir(SIDE)
    return lambda q: bool(np.max(d[idx] * q[idx]) >= limit)


def _held(squeeze: float | None = None) -> tuple[G.GraspRunner, FakeHand, float]:
    """A rigid wrap on a 5 cm object, lifted: HOLDING."""
    h = FakeHand()
    h.stops = {
        "thumb": aperture_stop(SIDE, "index", 0.05),
        "index": aperture_stop(SIDE, "index", 0.05),
        "middle": aperture_stop(SIDE, "middle", 0.05),
    }
    r, t = _grasp(h, width_m=0.05, squeeze=squeeze)
    t = _lift(r, h, t)
    assert r.state == "HOLDING", r.verdict()
    return r, h, t


def test_after_holding_one_groups_evidence_dropping_out_is_not_a_slip() -> None:
    """Session 10 (a simulated small round toy, r1sim.hands_rig): held 9 cm up, but judged NOT_HOLDING 2.3 cm into the
    carry because the middle finger's pad stopped reading as the load shifted onto thumb + index (the finger crept 0.04
    rad, no run-on) - and the NOT_HOLDING relaxed the fingers, which dropped the toy. After the lift check confirmed the
    hold, the object has left only when a finger runs on or no group keeps any evidence. (On a041731: NOT_HOLDING,
    'lift'.)"""
    r, h, t = _held(squeeze=0.15)
    d = G.close_dir(SIDE)
    idx = list(G.GROUPS["middle"])
    c_now = float(np.max(d[idx] * h.q[idx]))
    h.stops["middle"] = _creep_stop(idx, c_now + 0.09)  # creeps 0.09 rad (< RUN_ON_RAD) and rests
    h.blocked["middle"] = False
    h.pad_off = {"middle"}  # ... with its pad no longer reading, not held back by its squeeze (0.15 x 0.5)
    t = run(r, h, 1.5, t, ("NOT_HOLDING",))
    v = r.verdict()
    assert v.state == "HOLDING", v.reason
    assert [e["event"] for e in r.events].count("holding") == 1  # one transition, not one event per tick
    # the object really leaving after the hold: every finger runs on -> NOT_HOLDING (unchanged)
    h.stops, h.blocked = {}, {}
    run(r, h, 1.5, t, ("NOT_HOLDING",))
    v = r.verdict()
    assert v.state == "NOT_HOLDING" and v.reason is not None and v.reason.startswith("lift")
    assert v.evidence["terms"]["lift"]["run_on_rad"] > G.RUN_ON_RAD


def test_after_holding_fingers_settling_further_with_the_contact_persisting_is_not_a_slip() -> None:
    """Session 11, the user's "contact decides" (round B, the simulated grasp grid: a tall cylinder HOLDING, 8.0 N,
    every group touching, ~7 cm up, when its fingers settled 0.101 rad past the verdict's pose > RUN_ON_RAD 0.10 ->
    NOT_HOLDING -> the fingers backed off -> it fell). After the hold is confirmed a finger closing further while the
    contact persists is NOT a slip; only closing on nothing is. (On 1e55bdd: NOT_HOLDING, 'lift: ... ran on'.)"""
    r, h, t = _held()
    d = G.close_dir(SIDE)
    for g in ("thumb", "index", "middle"):  # every group settles 0.15 rad further (> RUN_ON_RAD) and rests, pads on
        idx = [i for i in G.GROUPS[g] if i != 0]
        h.stops[g] = _creep_stop(idx, float(np.max(d[idx] * h.q[idx])) + 0.15)
        h.blocked[g] = False
    t = run(r, h, 1.5, t, ("NOT_HOLDING",))
    v = r.verdict()
    assert v.state == "HOLDING", v.reason
    # ... and the object leaving for real (nothing resists any more: no pad, not held back) is still a slip
    h.stops, h.blocked = {}, {}
    run(r, h, 1.5, t, ("NOT_HOLDING",))
    v = r.verdict()
    assert v.state == "NOT_HOLDING" and v.evidence["terms"]["lift"]["contact_persists"] is False


def test_a_not_holding_verdict_never_relaxes_the_fingers_by_itself() -> None:
    """Session 11, the user's rule: NOT_HOLDING (here: the object slips out at the lift) keeps the hand's grip - the
    grip's command is re-sent every tick (r1hands would hold the MEASURED pose when a TTL runs out: the squeeze would
    fade) and nothing backs off - until `release` opens the hand. (On 1e55bdd: the joints commanded 0.03 rad back from
    where they were, one command, then silence.)"""
    h = FakeHand()
    st = aperture_stop(SIDE, "index", 0.05)
    h.stops = {"thumb": st, "index": st, "middle": aperture_stop(SIDE, "middle", 0.05)}
    r, t = _grasp(h, width_m=0.05)
    assert r.state == "TENTATIVE"
    h.stops, h.blocked = {}, {}
    t = _lift(r, h, t)
    assert r.state == "NOT_HOLDING"
    d = G.close_dir(SIDE)
    n0 = h.commands
    t = run(r, h, 1.0, t, ())
    ix = [1, 2, 3, 4, 5, 6]
    assert np.all(d[ix] * (h.cmd[ix] - h.q[ix]) > -0.005), h.cmd - h.q  # never backed off
    assert h.commands - n0 >= 90  # re-sent at the 100 Hz tick
    assert np.all(h.kp == pytest.approx(1.5)) and r.keeps_grip
    r.release(round((t + 0.01) * 1e9))
    assert r.verdict().state == "OPEN" and not _kept(r)
    assert np.allclose(h.cmd[1:], G.preset("open", SIDE)[1:])
    n1 = h.commands
    run(r, h, 0.2, t, ())
    assert h.commands == n1  # opened: nothing re-sent


def test_a_kept_grip_still_relaxes_when_the_hand_gets_hot() -> None:
    """The one exception (the motors' protection, HW1 B3.3: 0.9 N m heats a finger ~6 C/s): a NOT_HOLDING hand that
    keeps its grip backs off at TEMP_ABORT_C (powered: kp 1.5, never limp)."""
    h = FakeHand()
    st = aperture_stop(SIDE, "index", 0.05)
    h.stops = {"thumb": st, "index": st, "middle": aperture_stop(SIDE, "middle", 0.05)}
    r, t = _grasp(h, width_m=0.05)
    h.stops, h.blocked = {}, {}
    t = _lift(r, h, t)
    assert r.state == "NOT_HOLDING" and r.keeps_grip
    h.temperature = 73.0
    run(r, h, 0.05, t, ())
    d = G.close_dir(SIDE)
    assert not _kept(r) and np.all(d[1:] * (h.cmd[1:] - h.q[1:]) < 0.0)
    assert np.all(h.kp == pytest.approx(1.5))


def _three_group_grasp_in_the_lift_window() -> tuple[FakeHand, G.GraspRunner, float]:
    h = FakeHand()
    h.stops = {
        "thumb": aperture_stop(SIDE, "index", 0.05),
        "index": aperture_stop(SIDE, "index", 0.05),
        "middle": aperture_stop(SIDE, "middle", 0.05),
    }
    r, t = _grasp(h, width_m=0.05, squeeze=0.15)
    assert r.state == "TENTATIVE"
    return h, r, t


def _lose_the_evidence(h: FakeHand, group: str) -> None:
    d = G.close_dir(SIDE)
    idx = list(G.GROUPS[group])
    h.stops[group] = _creep_stop(idx, float(np.max(d[idx] * h.q[idx])) + 0.09)
    h.blocked[group] = False
    h.pad_off = {*h.pad_off, group}


def test_in_the_lift_check_one_group_losing_its_evidence_is_not_a_slip() -> None:
    """Contact decides in the lift check too (the user's answer at the end of session 11; was: the window stays
    strict - this test pinned NOT_HOLDING): the middle finger creeps 0.09 rad and loses its pad while the thumb and the
    index keep their evidence -> HOLDING (grip's grid: 12 of 71 picks that really carried the object were called
    NOT_HOLDING by the strict window as the fingers settled under the weight)."""
    h, r, t = _three_group_grasp_in_the_lift_window()
    _lose_the_evidence(h, "middle")
    _lift(r, h, t)
    v = r.verdict()
    assert v.state == "HOLDING", v.evidence
    lift = v.evidence["terms"]["lift"]
    assert lift["contact_persists"] is True and lift["run_on_judged"] is False


def test_in_the_lift_check_every_group_closing_on_nothing_is_a_slip() -> None:
    """The slip contact decides still catches in the lift window: EVERY touching group loses its evidence (the object
    left - the fingers close on nothing) -> NOT_HOLDING."""
    h, r, t = _three_group_grasp_in_the_lift_window()
    for g in ("thumb", "index", "middle"):
        _lose_the_evidence(h, g)
    _lift(r, h, t)
    v = r.verdict()
    assert v.state == "NOT_HOLDING" and v.evidence["terms"]["lift"]["contact_persists"] is False


def test_the_thumb_on_its_own_curled_finger_is_not_a_hold() -> None:
    """The 9 Oct false hold's mechanism: the rotated thumb stalls on the curled index, the index presses the board -
    both stop with force, the tips ~15 mm apart (the gap rule passes, the force passes): NOT_HOLDING, because the
    thumb's stop is the hand touching itself."""
    h = FakeHand(q0=G.preshape("pinch", SIDE))
    h.stops = {
        "thumb": lambda q: G.thumb_finger_clearance(SIDE, q)["index"] <= 0.0005,
        "index": closure_stop(SIDE, (5, 6), 0.5),
        "middle": closure_stop(SIDE, (3, 4), 0.5),
    }
    r, _ = _grasp(h)
    v = r.verdict()
    assert v.state == "NOT_HOLDING", v.evidence
    assert v.evidence["stops"]["thumb"] == "self"
    assert v.evidence["terms"]["opposition"]["thumb_on_own_finger"] is True
    assert v.evidence["terms"]["gap"]["ok"]  # the tips are apart: a gap rule alone would have passed it


def test_fingertips_pressing_the_surface_are_not_a_hold() -> None:
    """Thumb and fingers stopped by the surface under them (pads + torque, opposition by evidence, the tips apart), the
    object's width known: the tips are at the surface and the aperture does not match the width -> pressing."""
    h = FakeHand()
    h.stops = {"thumb": closure_stop(SIDE, (1, 2), 0.3), "index": closure_stop(SIDE, (5, 6), 0.3)}
    h.stops["middle"] = closure_stop(SIDE, (3, 4), 0.3)
    r, _ = _grasp(h, width_m=0.03, contact_height=lambda q: {"thumb": 0.009, "index": 0.008, "middle": 0.0085})
    v = r.verdict()
    assert v.state == "NOT_HOLDING", v.evidence
    assert v.reason is not None and v.reason.startswith("contact_height")
    t = v.evidence["terms"]
    assert t["opposition"]["ok"] and t["force"]["ok"] and t["contact_height"]["at_surface"]


def test_a_low_grasp_on_the_objects_sides_is_tentative() -> None:
    """Tips at the table (v2's lower-then-close) closed on a 48 mm object's sides: the aperture matches the width."""
    h = FakeHand()
    st = aperture_stop(SIDE, "index", 0.046)
    h.stops = {"thumb": st, "index": st, "middle": aperture_stop(SIDE, "middle", 0.046)}
    r, _ = _grasp(h, width_m=0.048, contact_height=lambda q: {"thumb": 0.009, "index": 0.007, "middle": 0.008})
    v = r.verdict()
    assert v.state == "TENTATIVE", v.evidence
    assert v.evidence["terms"]["contact_height"]["at_surface"] and v.evidence["terms"]["contact_height"]["ok"]
    # the same with no width: not decidable from the hand - the lift decides (term not judged)
    h2 = FakeHand()
    h2.stops = dict(h.stops)
    r2, _ = _grasp(h2, contact_height=lambda q: {"thumb": 0.009, "index": 0.007, "middle": 0.008})
    assert r2.state == "TENTATIVE" and r2.verdict().evidence["terms"]["contact_height"]["ok"] is None


def test_a_rigid_pinch_on_a_card_reads_the_tips_met_but_a_12_mm_object_holds() -> None:
    """Session 13: a RIGID pinch stopped 3 mm apart cannot be told from tips that met (the real air pinch reads +1.55 /
    +5.48 mm of aperture, HW2 day 1's H3) - the pinch now also needs the distal shells clear of each other
    (PINCH_MET_MARGIN_M); a thin layer is the `thin_flexible` class's job (its layer term + the hand's air-pinch
    reference). Session 12 called this card TENTATIVE (0.5 x 3 mm = 1.5 mm of aperture)."""
    h = FakeHand()
    st = aperture_stop(SIDE, "index", 0.003)
    h.stops = {"thumb": st, "index": st}
    r, _ = _grasp(h, "pinch", width_m=0.003)
    v = r.verdict()
    assert v.state == "NOT_HOLDING", v.evidence
    assert v.reason is not None and v.reason.startswith("gap") and v.evidence["terms"]["gap"]["tips_met"]
    h2 = FakeHand()
    st2 = aperture_stop(SIDE, "index", 0.012)
    h2.stops = {"thumb": st2, "index": st2}
    r2, _ = _grasp(h2, "pinch", width_m=0.012)
    assert r2.state == "TENTATIVE", r2.verdict().evidence
    assert not r2.verdict().evidence["terms"]["gap"]["tips_met"]


# ------------------------------------------------------------------------------------------------ thin_flexible
@pytest.mark.parametrize("layer_m", [0.0015, 0.004])
def test_a_thin_layer_between_the_tips_holds_after_the_lift(layer_m: float) -> None:
    h = FakeHand()
    st = layer_stop(SIDE, "index", layer_m)
    h.stops = {"thumb": st, "index": st}
    r, t = _grasp(h, "tip", "thin_flexible", layer_zero_m=0.0)
    v = r.verdict()
    assert v.state == "TENTATIVE", v.evidence
    assert v.evidence["terms"]["tip_pressure"]["ok"]
    assert v.evidence["terms"]["layer"]["layer_mm"] == pytest.approx(1000 * layer_m, abs=0.6)
    _lift(r, h, t, 0.02)
    assert r.verdict().state == "HOLDING"


def test_an_air_pinch_tips_that_met_is_not_holding() -> None:
    """Tips pressing each other with nothing between: tip pressure on both (as a real pinch) - the layer says 0."""
    h = FakeHand()
    st = layer_stop(SIDE, "index", 0.0)
    h.stops = {"thumb": st, "index": st}
    r, _ = _grasp(h, "tip", "thin_flexible", layer_zero_m=0.0)
    v = r.verdict()
    assert v.state == "NOT_HOLDING" and v.reason is not None and v.reason.startswith("layer")
    assert v.evidence["terms"]["tip_pressure"]["ok"]  # the pressure alone would have passed it
    assert abs(v.evidence["terms"]["layer"]["layer_mm"]) < 0.5


def test_a_thin_layer_slipping_out_at_the_lift_is_not_holding() -> None:
    h = FakeHand()
    st = layer_stop(SIDE, "index", 0.002)
    h.stops = {"thumb": st, "index": st}
    r, t = _grasp(h, "tip", "thin_flexible", layer_zero_m=0.0)
    assert r.state == "TENTATIVE"
    met = layer_stop(SIDE, "index", 0.0)
    h.stops = {"thumb": met, "index": met}
    h.blocked = {}
    _lift(r, h, t, 0.02)
    v = r.verdict()
    assert v.state == "NOT_HOLDING" and v.evidence["terms"]["lift"]["ok"] is False


def test_the_rigid_gap_rule_would_reject_a_good_thin_pinch() -> None:
    """Why the class exists (D27): the same 1.5 mm layer judged as `rigid` without a width fails the gap rule."""
    h = FakeHand()
    st = layer_stop(SIDE, "index", 0.0015)
    h.stops = {"thumb": st, "index": st}
    r, _ = _grasp(h, "tip", "rigid")
    v = r.verdict()
    assert v.state == "NOT_HOLDING" and v.evidence["terms"]["gap"]["ok"] is False


# ------------------------------------------------------------------------------------------------ the hand's state
def test_a_hot_hand_ends_the_grasp_relaxed() -> None:
    h = FakeHand()
    st = aperture_stop(SIDE, "index", 0.05)
    h.stops = {"thumb": st, "index": st, "middle": aperture_stop(SIDE, "middle", 0.05)}
    r, t = _grasp(h, width_m=0.05)
    assert r.state == "TENTATIVE"
    h.temperature = 73.0
    run(r, h, 0.2, t, DONE)
    v = r.verdict()
    assert v.state == "NOT_HOLDING" and v.reason is not None and "hot" in v.reason
    assert np.all(h.kp == pytest.approx(1.5))  # relaxed, not limp


def test_a_stale_hand_state_during_the_close_is_not_holding() -> None:
    h = FakeHand()
    r = G.GraspRunner(h, SIDE, "wrap", "rigid")
    r.start(0)
    run(r, h, 0.5)
    h.valid = False
    run(r, h, 0.1)
    assert r.state == "NOT_HOLDING" and r.reason is not None and "stale" in r.reason


def test_the_squeeze_is_kp_capped_and_synced() -> None:
    """A side that touches first rests on the object (CONTACT_HOLD) until the other side touches; then the squeeze
    ramps to the grip's value; kp x error never above MAX_ERR_TAU."""
    h = FakeHand()
    st = aperture_stop(SIDE, "index", 0.05)
    h.stops = {"thumb": st, "index": st, "middle": aperture_stop(SIDE, "middle", 0.05)}
    r, _ = _grasp(h, width_m=0.05)
    d = G.close_dir(SIDE)
    push = d[[1, 2, 5, 6]] * (h.cmd[[1, 2, 5, 6]] - h.q[[1, 2, 5, 6]])
    assert np.max(push) == pytest.approx(G.GRIPS["wrap"].squeeze, abs=0.02)
    assert np.all(1.5 * np.abs(h.cmd - h.q) <= G.MAX_ERR_TAU + 1e-9)
    assert r.events[0]["event"] == "start" and any(e["event"] == "tentative" for e in r.events)


def test_a_thin_pinch_needs_the_hands_air_pinch_reference() -> None:
    """Uncalibrated, the layer term passes an air pinch whenever the real tips stop short of the model's capsules (the
    sim's hand reads 2.2 mm with nothing between its tips): thin_flexible refuses to run without the reference, and
    `air_pinch_layer` reads it from an air pinch."""
    with pytest.raises(ValueError, match="air-pinch"):
        G.GraspRunner(FakeHand(), SIDE, "tip", "thin_flexible")
    h = FakeHand()
    st = layer_stop(SIDE, "index", 0.0022)  # a hand whose tips stop 2.2 mm short of the model's capsules (no layer)
    h.stops = {"thumb": st, "index": st}
    r, _ = _grasp(h, "tip", "thin_flexible", layer_zero_m=0.0)
    assert r.state == "TENTATIVE"  # uncalibrated: the air pinch would pass
    z = G.air_pinch_layer(r.verdict())
    assert z == pytest.approx(0.0022, abs=0.0005)
    h2 = FakeHand()
    h2.stops = {"thumb": st, "index": st}
    r2, _ = _grasp(h2, "tip", "thin_flexible", layer_zero_m=z)
    assert r2.state == "NOT_HOLDING" and r2.reason is not None and r2.reason.startswith("layer")
