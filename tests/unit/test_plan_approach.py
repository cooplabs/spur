"""Session 12 (HW2 day 1, 12 Oct 2026): the requested approach (orient.py's "session 12" section; PLANNER.md).

On the robot (hanging at a table, B4's scripted grasps) the tall cylinder grazed on the way down and slipped, the tall
box was pushed, the flat box got ~40 of 56-72 mm of its descent; the user: "the approach angle is the problem, nothing
else", "the forearm just needs to rotate more and the shoulder can come more down", "from around, not on top", "the
approach is technically a MODEL decision but the controller has to be very good about following that model decision".
The planner reproduced it on the robot's own layout (the table's near edge 12.9 cm ahead of the base origin, the board
on it - b4_layout_twin.json's numbers; generic shapes here, D32): the default hover chose the pose at the HOVER height
and the descent found what it found - the palm on the tall cylinder's top ~half way down, the tall box's knuckle at
once, the flat box 8 of 52 mm allowed - and front / inward / outward all came out as ONE -57 deg pose.

Every test here fails on the base 1cf3afd (no approach_style: PlanGoal TypeError / ArmHoverReq INVALID_ARGS; the
retrace and the hand clearance do not exist); the "base behaviour" asserts document what the robot saw.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.hands.grip import preshape
from r1mc3.plan import PlanGoal, plan, policy_envelope
from r1mc3.plan import orient as O
from r1mc3.plan.core import geometry_for
from r1mc3.policy import POLICY_DIR, ReferencePolicy
from r1mc3.service.l2 import L2
from r1mc3.service.schemas import REQUEST
from r1mc3.skills import posture
from r1mc3.world import Geometry, Object, Pose, Surface, WorldModel, WorldSnapshot
from r1sim import joints as J

ASSETS = Path(__file__).resolve().parents[1] / "assets"
# the robot's day-1 table relative to the standing base (b4_layout_twin.json) and the standing lab table
LAYOUTS = {"hang": (0.754, 0.129, 0.02), "lab": (0.751, 0.081, 0.0)}
BOARD_H = 0.012
# generic shapes (cm, as world entities take them: centre + half sizes)
TALL_CYL = ("cylinder", (8.0, 8.0, 13.2), (20.0, -10.0, 0.0), "wrap")
TALL_BOX = ("box", (7.8, 6.7, 11.2), (20.0, -10.0, 0.0), "wrap")
SHORT_CYL = ("cylinder", (6.0, 6.0, 6.2), (20.0, -10.0, 0.0), "wrap")
FLAT_BOX = ("box", (4.8, 12.4, 3.1), (30.0, 0.0, 90.0), "pinch")
LOW_BOX = ("box", (9.8, 5.4, 3.9), (20.0, -10.0, 0.0), "pinch")
READY_RIGHT = np.array([-0.4, -0.3, 0.0, -0.4, 0.0])  # HW2 day 1's ready pose (r1 move-joint right -0.4,-0.3,0,-0.4,0)


def _world(layout: str, obj: tuple[Any, ...] | None) -> WorldSnapshot:
    top, edge, yo = LAYOUTS[layout]
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(edge + 0.711 / 2, yo, top), size=(0.711, 1.22, 0.02)))
    x0, x1 = edge + 0.01, edge + 0.366
    w.declare(Surface("board", Pose.xyz_yaw((x0 + x1) / 2, yo, top + BOARD_H), size=(x1 - x0, 0.356, BOARD_H)))
    if obj is not None:
        shape, sz, at, _ = obj
        cx, cy, cz = edge + at[0] / 100, yo + at[1] / 100, top + BOARD_H + sz[2] / 200
        if shape == "cylinder":
            w.declare(Object("obj", Pose.xyz_yaw(cx, cy, cz), shape="cylinder", size=(sz[0] / 200, sz[2] / 200)))
        else:
            w.declare(Object("obj", Pose.xyz_yaw(cx, cy, cz, np.radians(at[2])),
                             size=(sz[0] / 200, sz[1] / 200, sz[2] / 200)))  # fmt: skip
    return w.snapshot()


@pytest.fixture(scope="module")
def env() -> tuple[np.ndarray, np.ndarray]:
    ranges = ReferencePolicy.load(POLICY_DIR).arm_ranges_left
    assert ranges is not None
    return policy_envelope(ranges, "right")


def _q(start: str = "ready") -> np.ndarray:
    q = J.POSE_STAND.copy()
    q[J.LEFT_ARM] = posture("table_ready", "left")
    q[J.RIGHT_ARM] = READY_RIGHT if start == "ready" else posture("table_ready", "right")
    return q


def _hover(layout: str, obj: tuple[Any, ...], env: Any, *, declare: bool, start: str = "ready", **kw: Any) -> Any:
    _, sz, at, grip = obj
    top, edge, yo = LAYOUTS[layout]
    snap = _world(layout, obj if declare else None)
    g = PlanGoal("hover", "right", position=(edge + at[0] / 100, yo + at[1] / 100, top + BOARD_H), height_m=sz[2] / 100,
                 grip=grip, grasp_yaw_deg=90.0, allow_hand=("obj",) if declare else (), **kw)  # fmt: skip
    return snap, plan(snap, g, _q(start), hands={"right": preshape(grip, "right")}, envelope=env)


def _after(q26: np.ndarray, r: Any) -> np.ndarray:
    q = q26.copy()
    q[J.RIGHT_ARM] = r.segments[-1].q[-1]
    return q


def _lower(snap: WorldSnapshot, q26: np.ndarray, grip: str, env: Any, depth: float, r_hover: Any,
           declare: bool) -> Any:  # fmt: skip
    o = r_hover.details.get("orientation", {})
    gq, gp = o.get("grasp_q"), o.get("grasp_point")
    g = PlanGoal("lower", "right", until="depth", max_depth_m=depth, allow_hand=("obj",) if declare else (),
                 end_q=tuple(gq) if gq else None, end_point=gp)  # fmt: skip
    return plan(snap, g, q26, hands={"right": preshape(grip, "right")}, envelope=env)


def _first_touch(layout: str, obj: tuple[Any, ...], q26: np.ndarray, r_lower: Any) -> float | None:
    """The descent (m of the grasp centre's drop) at which a hand part first comes within 1 mm of the object."""
    geo = geometry_for(_world(layout, obj), 0.001)
    geo.set_stance(q26, None, {"right": preshape(obj[3], "right")})
    Q = r_lower.segments[-1].q
    z0 = None
    for k in range(len(Q) - 1):
        for s in np.linspace(0.0, 1.0, 11):
            geo.set_arm("right", Q[k] + (Q[k + 1] - Q[k]) * s)
            z = float(geo.point("right", "grasp_centre")[2])
            z0 = z if z0 is None else z0
            if any(h.other == "entity:obj" for h in geo.hits("right")):
                return z0 - z
    return None


def test_base_behaviour_the_default_hover_puts_a_hand_part_on_the_tall_object_on_the_way_down(env: Any) -> None:
    """What the robot saw, reproduced by the planner: no style, B4's recipe (the grasp centre to half the height)."""
    snap, r = _hover("hang", TALL_CYL, env, declare=False)
    assert r.ok
    q2 = _after(_q(), r)
    geo = geometry_for(snap)
    geo.set_stance(q2, None, {"right": preshape("wrap", "right")})
    want = float(r.predicted["grasp_centre_m"][2]) - (LAYOUTS["hang"][0] + BOARD_H + 0.066)
    lo = _lower(snap, q2, "wrap", env, want, r, False)
    assert lo.ok
    touch = _first_touch("hang", TALL_CYL, q2, lo)
    assert touch is not None and touch < 0.8 * want, (touch, want)
    assert r.details["orientation"]["palm_down_deg"] > 0.0 and "approach_style" in r.details["orientation"]


@pytest.mark.parametrize("obj", [TALL_BOX, SHORT_CYL])
def test_around_on_the_other_wrapped_shapes(env: Any, obj: tuple[Any, ...]) -> None:
    snap, r = _hover("hang", obj, env, declare=True, approach_style="around")
    assert r.ok, r.reason
    o = r.details["orientation"]
    assert o["approach_err_deg"] == 0.0 and o["palm_down_deg"] <= O.AROUND_PALM_DOWN_MAX_DEG
    q2 = _after(_q(), r)
    lo = _lower(snap, q2, "wrap", env, o["descent_m"], r, True)
    assert lo.ok and lo.details.get("ended_at_grasp_pose") is True
    assert _first_touch("hang", obj, q2, lo) is None


def test_low_lets_the_flat_object_be_reached_at_its_grasp_height(env: Any) -> None:
    """The flat box 30 cm out on the robot's day-1 layout: the default allowed 8 of 52 mm (the robot: ~40 of 56-72);
    'low' plans where the pinch closes (its middle) and the whole descent is accepted."""
    snap0, r0 = _hover("hang", FLAT_BOX, env, declare=False)
    assert r0.ok
    q0 = _after(_q(), r0)
    want0 = float(r0.predicted["point_m"][2]) - (LAYOUTS["hang"][0] + BOARD_H + 0.0155)
    assert not _lower(snap0, q0, "pinch", env, want0, r0, False).ok  # base behaviour: the full descent refused
    snap, r = _hover("hang", FLAT_BOX, env, declare=True, approach_style="low")
    assert r.ok, r.reason
    o = r.details["orientation"]
    assert o["approach_err_deg"] == 0.0 and abs(o["tilt_deg"]) <= O.TILT_FREE_DEG + 1e-6
    q2 = _after(_q(), r)
    lo = _lower(snap, q2, "pinch", env, o["descent_m"], r, True)
    assert lo.ok, lo.reason
    geo = geometry_for(snap)
    geo.set_stance(q2, None, {"right": preshape("pinch", "right")})
    geo.set_arm("right", lo.segments[-1].q[-1])
    assert float(geo.point("right", o["grasp_point"])[2]) == pytest.approx(o["grasp_z_m"], abs=0.003)
    assert _first_touch("hang", FLAT_BOX, q2, lo) is None


def test_front_and_inward_are_delivered_as_different_poses(env: Any) -> None:
    """The base: prefer front / inward / outward = one -57 deg pose (the closing-line term outweighed the family)."""
    heads = {}
    for st in ("front", "inward"):
        _, r = _hover("lab", SHORT_CYL, env, declare=True, approach_style=st)
        assert r.ok, r.reason
        o = r.details["orientation"]
        lo, hi = O.FAMILIES[st]
        assert lo - 1.0 <= o["heading_deg"] <= hi + 1.0 and o["approach_err_deg"] <= 1.0
        heads[st] = o["heading_deg"]
    assert abs(heads["front"] - heads["inward"]) >= 30.0
    _, rf = _hover("lab", SHORT_CYL, env, declare=False, prefer="front")
    _, ri = _hover("lab", SHORT_CYL, env, declare=False, prefer="inward")
    assert abs(rf.details["orientation"]["heading_deg"] - ri.details["orientation"]["heading_deg"]) < 5.0  # base


def test_a_strict_request_beyond_its_tolerance_is_refused_in_words(env: Any) -> None:
    _, rn = _hover("hang", FLAT_BOX, env, declare=True, approach_style="front")  # NEAREST: ok, the error reported
    assert rn.ok, rn.reason
    err = rn.details["orientation"]["approach_err_deg"]
    assert err > 5.0 and rn.details["orientation"]["approach_delivered"] is False
    _, rs = _hover("hang", FLAT_BOX, env, declare=True, approach_style="front", approach_tol_deg=5.0)
    assert not rs.ok and rs.code == "UNREACHABLE"
    assert "'front' approach" in rs.reason and "within 5 deg" in rs.reason and "nearest clear pose" in rs.reason
    assert "approach_err_deg" in rs.details and "nearest" in rs.details


@pytest.mark.parametrize("style,obj", [("low", LOW_BOX), ("around", SHORT_CYL)])
def test_the_style_choice_does_not_depend_on_where_the_arm_starts(env: Any, style: str, obj: tuple[Any, ...]) -> None:
    """The user on the low box: "instead of choosing the most optimal one the planner just chooses the fastest one" -
    with a style the poses that deliver it are ranked by their quality (palm, closing line, heading), travel last."""
    got = []
    for start in ("ready", "table_ready"):
        _, r = _hover("lab", obj, env, declare=True, approach_style=style, start=start)
        assert r.ok, r.reason
        got.append(np.asarray(r.details["orientation"]["grasp_q"]))
    assert float(np.max(np.abs(got[0] - got[1]))) < 0.02, got


def test_an_unknown_style_or_tolerance_is_refused() -> None:
    snap = _world("lab", None)
    for kw in ({"approach_style": "top"}, {"approach_style": "front", "approach_tol_deg": 120.0}):
        r = plan(snap, PlanGoal("hover", "right", position=(0.3, -0.1, 0.0), height_m=0.1, **kw), _q())
        assert not r.ok and r.code == "INVALID_ARGS", r.reason


def test_r1d_passes_the_style_and_the_next_lower_retraces_to_the_grasp_pose() -> None:
    hv = {"op": "arm.hover", "arm": "right", "position": [0.3, -0.1, 0.76], "height_m": 0.13, "grip": "wrap",
          "grasp_yaw_deg": 90.0, "target_id": "obj", "approach_style": "around", "approach_tol_deg": 10.0,
          "grasp_z_m": 0.83}  # fmt: skip
    req: Any = REQUEST.validate_python(hv)
    d = L2.plan_goal_dict(None, req)  # type: ignore[arg-type]  # (uses nothing of the instance)
    assert d["approach_style"] == "around" and d["approach_tol_deg"] == 10.0 and d["grasp_z_m"] == 0.83
    g = PlanGoal.from_dict(d)
    assert g.approach_style == "around" and g.allow_hand == ("obj",)
    ns: Any = types.SimpleNamespace(held_object=lambda arm: None)  # (the two methods use nothing else of L2)
    L2.note_hover_plan(ns, "right", {"orientation": {"grasp_q": [0.1, -1.0, 0.9, -0.5, -1.8],
                                                     "grasp_point": "grasp:wrap@0.0800"}})  # fmt: skip
    low: Any = REQUEST.validate_python({"op": "arm.lower", "arm": "right", "until": "depth", "max_depth_m": 0.08})
    dl = L2.plan_goal_dict(ns, low)
    assert dl["end_q"] == [0.1, -1.0, 0.9, -0.5, -1.8] and dl["end_point"] == "grasp:wrap@0.0800"
    assert PlanGoal.from_dict(dl).end_q == (0.1, -1.0, 0.9, -0.5, -1.8)
    L2.note_hover_plan(ns, "right", {"orientation": {}})  # a hover without a grasp pose forgets it
    assert "end_q" not in L2.plan_goal_dict(ns, low)
    with pytest.raises(ValueError):
        REQUEST.validate_python({"op": "arm.hover", "arm": "right", "position": [0.3, -0.1, 0.76],
                                 "approach_style": "top"})  # fmt: skip


def test_the_hand_must_keep_its_distance_from_a_hand_clear_entity() -> None:
    """Geometry.hits(hand_clear=...): a target the hand may touch (allow_hand) still counts within the distance."""
    snap = _world("lab", SHORT_CYL)
    geo = Geometry(snap)
    q = _q()
    geo.set_stance(q, None, {"right": preshape("wrap", "right")})
    from r1mc3.plan import ik

    # the palm put into the object's middle: the hand touches it
    tgt = np.array([LAYOUTS["lab"][1] + 0.2, -0.1, LAYOUTS["lab"][0] + BOARD_H + 0.031])
    s = ik.solve(geo, "right", tgt, point="palm")
    geo.set_arm("right", s.q)
    assert not [h for h in geo.hits("right", allow_hand=("obj",)) if h.other == "entity:obj"]
    assert [h for h in geo.hits("right", allow_hand=("obj",), hand_clear={"obj": 0.004}) if h.other == "entity:obj"]


def test_the_operators_hover_word_takes_the_approach() -> None:
    from r1mc3.cli.main import build_request

    req, _ = build_request(["hover", "right", "0.3,-0.1,0.76", "--height", "0.13", "--approach", "around", "--target",
                            "obj", "--grip", "wrap", "--grasp-yaw", "90", "--approach-tol", "10"])  # fmt: skip
    assert req["op"] == "arm.hover" and req["approach_style"] == "around" and req["approach_tol_deg"] == 10.0
    assert req["target_id"] == "obj" and req["grip"] == "wrap" and req["grasp_yaw_deg"] == 90.0
    REQUEST.validate_python({k: v for k, v in req.items() if k not in ("requester", "wait", "timeout_s")})
    req2, _ = build_request(["lower", "right", "--until", "depth", "--depth", "0.08", "--target", "obj"])
    assert req2["op"] == "arm.lower" and req2["target_id"] == "obj"


def _touches_in_legs(layout: str, obj: tuple[Any, ...], q26: np.ndarray, r: Any) -> dict[str, int]:
    """Per hover leg: the sampled poses whose hand comes within 1 mm of the object."""
    geo = geometry_for(_world(layout, obj), 0.001)
    geo.set_stance(q26, None, {"right": preshape(obj[3], "right")})
    out: dict[str, int] = {}
    for seg in r.segments:
        n = 0
        for k in range(len(seg.q) - 1):
            for s in np.linspace(0.0, 1.0, 6):
                geo.set_arm("right", seg.q[k] + (seg.q[k + 1] - seg.q[k]) * s)
                n += any(h.other == "entity:obj" for h in geo.hits("right"))
        out[seg.leg] = out.get(seg.leg, 0) + n
    return out


@pytest.mark.parametrize("style", [None, "around"])
def test_the_transit_keeps_the_hand_off_a_declared_tall_target(env: Any, style: str | None) -> None:
    """b4's trace of the base (HW2's cases in the sim twin): from the table_ready arms (day 2's start after F8) the
    hover's transit swept the hand across the tall target's top - the target was the hand's to touch for the WHOLE goal.
    Now only the final descend may touch it (a style: not even that)."""
    for layout in ("lab", "hang"):
        kw: dict[str, Any] = {} if style is None else {"approach_style": style}
        _, r = _hover(layout, TALL_CYL, env, declare=True, start="table_ready", **kw)
        assert r.ok, r.reason
        t = _touches_in_legs(layout, TALL_CYL, _q("table_ready"), r)
        assert all(n == 0 for leg, n in t.items() if leg != "descend" or style is not None), (layout, t)


def test_the_lower_still_ends_at_the_grasp_pose_when_the_hover_ended_a_little_off(env: Any) -> None:
    """Session 12 (a small round object): an EXECUTED hover ends 5-12 mm from its plan (the task hand-over re-picks in
    lockstep: 10.3 / 12.3 mm; r1d's hover error median 5.9 mm). The lower then fell back to the plain straight descent
    (it ended 0.25 rad elsewhere or was refused at the board: the tall box never picked). Now it still ends EXACTLY at
    the hover's checked grasp pose (the open hand clear of the object on the way). On the base (ddd74c2)
    ended_at_grasp_pose is missing."""
    snap, r = _hover("lab", TALL_CYL, env, declare=True, approach_style="around")
    assert r.ok, r.reason
    o = r.details["orientation"]
    q2 = _after(_q(), r)
    geo = geometry_for(snap, 0.001)
    geo.set_stance(q2, None, {"right": preshape("wrap", "right")})
    z_plan = float(geo.point("right", "pinch")[2])
    q_off = q2.copy()
    q_off[J.RIGHT_ARM[0]] -= 0.03  # the shoulder pitch 0.03 rad forward: the hand ~8 mm above its planned point
    geo.set_arm("right", q_off[J.RIGHT_ARM])
    off_mm = 1000.0 * abs(float(geo.point("right", "pinch")[2]) - z_plan)
    assert 4.0 < off_mm < 15.0, off_mm
    lo = _lower(snap, q_off, "wrap", env, o["descent_m"], r, True)
    assert lo.ok, lo.reason
    assert lo.details.get("ended_at_grasp_pose") is True
    assert np.allclose(lo.segments[-1].q[-1], o["grasp_q"], atol=1e-3)
    assert _first_touch("lab", TALL_CYL, q_off, lo) is None


def test_a_grasp_hover_without_a_style_does_not_roll_the_hand(env: Any) -> None:
    """Session 12 (a small round object): without a style the ranking never saw the closing line's tilt - a short round
    object near the robot's centre line (a standard layout's small round object spot, 8 cm from the near edge) got a
    hand rolled -23.6 deg (the thumb at the board, the fingers ~7 cm higher over its top): in the episodes the close
    tipped the small round object 25-35 deg, it rode the carry on its side and landed on the bin's rim (3 / 9 in the
    bin). Now a rolled GRASP hand (> ROLLED_FREE_DEG) costs: -9.3 deg, held upright (5 / 9). Poses within 20 deg (the
    robot's held small round object / tape / low box) are unchanged. On the base (ddd74c2) the tilt is -23.6. Session 12
    left it OFF (skill.pick's stop height first); session 13 (D48) turns it ON at 12 deg (HW2 day 2's tape / low box
    closed at 14-16 deg, "the index finger keeps hitting the table first") - skill.pick plans its hover where the grip
    closes now."""
    assert O.ROLLED_FREE_DEG == 12.0
    top, edge, yo = LAYOUTS["lab"]
    snap = _world("lab", None)
    g = PlanGoal("hover", "right", position=(edge + 0.081, yo - 0.012, top + BOARD_H), height_m=0.11, grip="wrap",
                 grasp_yaw_deg=90.0)  # fmt: skip
    r = plan(snap, g, _q(), hands={"right": preshape("wrap", "right")}, envelope=env)
    assert r.ok, r.reason
    o = r.details["orientation"]
    assert abs(o["tilt_deg"]) <= O.ROLLED_FREE_DEG + 1e-6
    assert O.axis_err_deg(o["close_yaw_deg"], 90.0) <= O.CLOSE_TOL_DEG
