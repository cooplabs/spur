"""Session 13 (HW2 day 2's FG, 13 Oct 2026): the hand's pose (PLANNER.md "Session 13"; orient.py / core.py).

FG on the robot (the robot balancing at the real task table, results/hw2_20261013/b4/): the DEFAULT hand closed at 14-16
deg on the tape / low box ("the index finger keeps hitting the table first - a recurring problem"); the small round
object's trace replays put the index tip 5-13 mm over the board, the thumb 42-60 mm and the MIDDLE tip 70-80 mm - above
the 62 mm small round object: a two-tip grip "with the fingertips" (default 2 / 4, `around` 2 / 4); `around` on the tall
box grazed / pushed a corner (4 mm of open-hand clearance vs ~1 cm of tracking); `low` pressed the fingertips below the
thin block's bottom. Generic shapes here (D32): a short cylinder 6 x 6.2 cm, a tall box 7.8 x 6.7 x 11.2, a flat box 4.8
x 12.4 x 3.1. Every test fails on the base dcc01a2 (named per test).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.hands.grip import preshape
from r1mc3.plan import PlanGoal, plan, policy_envelope
from r1mc3.plan import orient as O
from r1mc3.plan.core import geometry_for
from r1mc3.policy import POLICY_DIR, ReferencePolicy
from r1mc3.service.schemas import REQUEST, ArmHoverReq
from r1mc3.skills.pick import PICK_MIN_DESCENT_M, pick_steps

from tests.unit.test_plan_approach import BOARD_H, FLAT_BOX, LAYOUTS, SHORT_CYL, TALL_BOX, _q, _world

LAYOUTS.setdefault("fg", (0.75843, 0.08943, -0.00936))  # FG part 2's measured table (b4/layout_measured_sw.json)


@pytest.fixture(scope="module")
def env() -> tuple[np.ndarray, np.ndarray]:
    ranges = ReferencePolicy.load(POLICY_DIR).arm_ranges_left
    assert ranges is not None
    return policy_envelope(ranges, "right")


def _hover(layout: str, obj: tuple[Any, ...], env: Any, **kw: Any) -> tuple[Any, Any]:
    _, sz, at, grip = obj
    top, edge, yo = LAYOUTS[layout]
    snap = _world(layout, obj)
    g = PlanGoal("hover", "right", position=(edge + at[0] / 100, yo + at[1] / 100, top + BOARD_H),
                 height_m=sz[2] / 100 + 0.02, grip=grip, grasp_yaw_deg=90.0, allow_hand=("obj",), **kw)  # fmt: skip
    return snap, plan(snap, g, _q("table_ready"), hands={"right": preshape(grip, "right")}, envelope=env)


def _closing_tips(snap: Any, r: Any, grip: str, width: float) -> dict[str, float]:
    """The grip's tips at the grasp pose with the hand where its fingers meet an object `width` wide (base z, m)."""
    o = r.details["orientation"]
    geo = geometry_for(snap)
    geo.set_stance(_q("table_ready"), None, {"right": preshape(grip, "right")})
    geo.set_arm("right", np.asarray(o["grasp_q"]))
    geo.set_hand("right", geo.closing_hand_q("right", grip, width))
    return {k: float(geo.point("right", f"{k}_tip")[2]) for k in ("thumb", "index", "middle")}


def test_skill_pick_hovers_where_the_grip_closes_and_leaves_a_descent() -> None:
    """S12's blocker for D48: the pick's hover was ranked at its own height, and a level hand left the grasp centre
    1.65 cm BELOW the client's grasp height -> the straight-down step refused. The hover now carries the grasp height
    + a minimum descent. On dcc01a2: no grasp_z_m / min_descent_m in the hover step (KeyError)."""
    st = pick_steps({"arm": "left", "position": [0.3, 0.2, 0.42], "height_m": 0.04, "grasp_z_m": 0.415},
                    support_top_z=0.40)  # fmt: skip
    hv = st[0]
    assert hv["op"] == "arm.hover" and hv["grasp_z_m"] == 0.415 and hv["min_descent_m"] == PICK_MIN_DESCENT_M
    req = REQUEST.validate_python({**hv, "id": None})
    assert isinstance(req, ArmHoverReq) and req.min_descent_m == PICK_MIN_DESCENT_M  # r1d takes the field


def test_min_descent_keeps_the_hover_above_its_grasp_pose(env: Any) -> None:
    """A grasp height whose pose would put the hand AT the hover height: the hover still ends min_descent_m above it
    (the descent the next straight-down step needs). On dcc01a2: PlanGoal has no min_descent_m (TypeError)."""
    top, edge, yo = LAYOUTS["lab"]
    snap = _world("lab", None)
    gz = top + BOARD_H + 0.07  # above where the hover's own height leaves the hand
    g = PlanGoal("hover", "right", position=(edge + 0.20, yo - 0.10, top + BOARD_H), height_m=0.04, grip="wrap",
                 grasp_yaw_deg=90.0, grasp_z_m=gz, min_descent_m=0.03)  # fmt: skip
    r = plan(snap, g, _q("table_ready"), hands={"right": preshape("wrap", "right")}, envelope=env)
    assert r.ok, r.reason
    assert r.details["orientation"]["descent_m"] >= 0.03 - 0.002


def test_the_default_grasp_hand_is_level(env: Any) -> None:
    """D48 ON: a grasp hover without a style keeps the closing line within ROLLED_FREE_DEG (12) of level (the robot's
    tape 14.4-14.8, low box 16.3 deg) at FG's spot and a standard layout's tape spot. (Near the robot's centre line -
    a standard layout's small round object spot - a level line closing across the object does not exist for the right
    hand: the best is 20.6 deg, the first level one closes 29 deg off the asked line; the closing line keeps the
    priority there.) On dcc01a2 ROLLED_FREE_DEG is None (the ranking off)."""
    assert O.ROLLED_FREE_DEG == 12.0
    for at in ((20.21, -9.56), (20.5, -9.5)):  # FG's spot + a standard layout's tape spot
        top, edge, yo = LAYOUTS["fg"]
        snap = _world("fg", None)
        for grip, h in (("pinch", 0.066), ("pinch", 0.039), ("wrap", 0.062)):
            g = PlanGoal("hover", "right", position=(edge + at[0] / 100, yo + at[1] / 100, top + BOARD_H),
                         height_m=h, grip=grip, grasp_yaw_deg=90.0)  # fmt: skip
            r = plan(snap, g, _q("table_ready"), hands={"right": preshape(grip, "right")}, envelope=env)
            assert r.ok, (at, grip, r.reason)
            assert abs(r.details["orientation"]["tilt_deg"]) <= O.ROLLED_FREE_DEG + 1e-6, (at, grip)


def test_a_short_wrap_closes_deep_with_all_three_tips_on_the_object(env: Any) -> None:
    """The small round object: a wrap onto a declared SHORT object (no style - the default B4 / compat path that
    declares it) is planned where the grip closes, its closing point WRAP_DEEP_FRAC of the width in (the hand further
    over it), and the three tips meet it on its side (between FIT_LO_M over its bottom and FIT_HI_M under its top) - not
    the robot's index on the board + middle above the top. On dcc01a2: a no-style hover had no grasp pose (no descent_m,
    no grasp_q: KeyError)."""
    snap, r = _hover("fg", SHORT_CYL, env)
    assert r.ok, r.reason
    o = r.details["orientation"]
    assert o["width_m"] == pytest.approx(0.06 * 0.6, abs=1e-4) and o["descent_m"] > 0.0
    assert o["tips_out_mm"] <= 2.0
    top, _, _ = LAYOUTS["fg"]
    bottom = top + BOARD_H
    tips = _closing_tips(snap, r, "wrap", o["width_m"])
    for k, z in tips.items():  # on its side: over the board, under its top (the robot: index 5-13 mm, middle 70-80)
        assert bottom + O.FIT_LO_M - 0.002 <= z <= bottom + 0.062 - O.FIT_HI_M + 0.002, (k, z - bottom)


def test_a_tall_wrap_keeps_its_grasp_point(env: Any) -> None:
    """The robot's 8 / 8 `around` tall items keep their closing point (the deep wrap is for SHORT objects only). On
    dcc01a2 the reported clearance was 4 mm (target_clear_m 0.004)."""
    _snap, r = _hover("fg", TALL_BOX, env, approach_style="around")
    assert r.ok, r.reason
    o = r.details["orientation"]
    assert o["width_m"] == pytest.approx(0.067, abs=1e-4) and o["approach_err_deg"] == 0.0
    assert o["target_clear_m"] == O.AROUND_CLEAR_M == 0.010


def test_around_keeps_the_open_hand_10_mm_from_a_tall_box(env: Any) -> None:
    """FG's tall box: "very close to / grazing the box sometimes, pushed down on a corner once" (4 mm of open-hand
    clearance, a real hover ends 5-10 mm off). On dcc01a2 the grasp pose came within 4-10 mm (a hand part inside 9
    mm)."""
    snap, r = _hover("fg", TALL_BOX, env, approach_style="around")
    assert r.ok, r.reason
    geo = geometry_for(snap)
    geo.set_stance(_q("table_ready"), None, {"right": preshape("wrap", "right")})
    geo.set_arm("right", np.asarray(r.details["orientation"]["grasp_q"]))
    assert not geo.hits("right", hand_clear={"obj": 0.009}, allow_hand=("obj",))


def test_low_keeps_the_closing_tips_over_the_flat_objects_bottom(env: Any) -> None:
    """FG's thin block `low`: the fingertips went below its bottom and grazed the board (the NOMINAL table); `low` keeps
    the closing tips LOW_TIP_MARGIN_M over the object's bottom. On dcc01a2 no tips band was reported (KeyError)."""
    snap, r = _hover("fg", FLAT_BOX, env, approach_style="low")
    assert r.ok, r.reason
    o = r.details["orientation"]
    assert o["tips_out_mm"] == 0.0 and o["approach_err_deg"] == 0.0
    top, _, _ = LAYOUTS["fg"]
    tips = _closing_tips(snap, r, "pinch", o["width_m"])
    assert min(tips["thumb"], tips["index"]) >= top + BOARD_H + 0.006 - 1e-4


def _day2_measured_setup() -> Any:
    """HW2 day 2's measured setup as declared to r1d by hand before FG part 2 / SW9 (results/hw2_20261013/STEPS.md "THE
    MEASURED SETUP DECLARED TO r1d BY HAND"): table 0.698 x 1.200 top centre (0.43835, -0.00179, 0.75843) yaw 0.02168,
    board 0.355 x 0.355 x 0.012 at (0.27889, -0.00301, 0.77053) yaw 0.018226, bin (container 0.35 x 0.26 x 0.045) at
    (0.30725, -0.45595, 0.75843) yaw 0.028472."""
    from r1mc3.world import Container, Pose, Surface, WorldModel

    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.43835, -0.00179, 0.75843 - 0.01, 0.02168), size=(0.698, 1.2, 0.02)))
    w.declare(Surface("board", Pose.xyz_yaw(0.27889, -0.00301, 0.77053 - 0.006, 0.018226), size=(0.355, 0.355, 0.012)))
    w.declare(Container("bin", Pose.xyz_yaw(0.30725, -0.45595, 0.75843 + 0.0225, 0.028472),
                        size_outer=(0.35, 0.26, 0.045)))  # fmt: skip
    return w.snapshot()


def test_the_way_out_from_under_the_table_keeps_its_margin_past_the_tray_on_the_last_leg() -> None:
    """MUST 3 (HW2 day 2, SW9: `skill table_ready` with the measured setup + the bin declared planned the right arm
    1.19 cm from the bin and the hand touched it): the out-and-around's LAST leg ("on": over -> the table_ready
    posture) is checked at the out / over legs' margin (2-3 cm), not the planner's default 1 cm (the setup agent's
    patch, setup_TO_pose.md). On dcc01a2 the right arm's "on" leg came within ~1.1 cm of the bin."""
    from r1mc3.plan import path as PP
    from r1mc3.skills import posture
    from r1mc3.world import Geometry
    from r1sim import dex3
    from r1sim import joints as J

    snap = _day2_measured_setup()
    q = J.POSE_STAND.copy()
    r = plan(snap, PlanGoal("move_joint", "right", q=tuple(posture("table_ready", "right")), speed=1.0, hand="fist"),
             q)  # fmt: skip
    assert r.ok, r.reason
    assert r.details["legs"] == ["out", "over", "on"], r.details["legs"]
    m = float(r.details["out_around"]["clearance_m"])
    assert m >= 0.02
    # every leg judged at 5 mrad steps against 0.9 x the margin (the planner checks every 0.02 rad), the fist included
    geo = Geometry(snap, clearance_m=0.9 * m)
    geo.set_stance(q, None, {"right": dex3.preset("fist", "right")})
    for s in r.segments:
        for qq in PP.densify(s.q, 0.005):
            geo.set_arm("right", qq)
            near_tray = [h for h in geo.hits("right", hand_support_m=0.9 * m) if "bin" in str(h.other)]
            assert not near_tray, (s.leg, near_tray[0])
