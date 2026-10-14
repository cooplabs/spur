"""Session 9 follow-ups from api / compat driving the planner through r1d and the real core (api_TO_PLANNER.md,
compat_TO_API.md 21:25): each test fails on 411f571."""

from __future__ import annotations

import numpy as np
import pytest
from r1mc3.plan import PlanGoal, ik, plan
from r1mc3.world import Geometry, Obstacle, Pose, Surface, WorldModel, WorldSnapshot
from r1sim import joints as J

TZ = 0.76


@pytest.fixture(scope="module")
def table() -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.55, 0.0, TZ), size=(0.6, 1.2, 0.04)))
    return w.snapshot()


def test_knots_stay_inside_the_soft_limits(table: WorldSnapshot) -> None:
    """api 1: the executor refuses a knot past its q_min / q_max (= the soft limits); a knot AT the limit came out a
    hair outside (kRejectKnots). Now every knot is >= 1 mrad inside - a target at the limit and a start at it too.
    411f571: the last knot = SOFT_MIN exactly."""
    q = J.POSE_STAND.copy()
    tgt = q[J.LEFT_ARM].copy()
    tgt[4] = J.SOFT_MAX[J.LEFT_ARM][4]  # the left wrist roll at its soft limit (api's case: shoulder roll -0.177)
    r = plan(table, PlanGoal("move_joint", "left", q=tuple(float(v) for v in tgt)), q)
    assert r.ok, r.reason
    lo, hi = J.SOFT_MIN[J.LEFT_ARM], J.SOFT_MAX[J.LEFT_ARM]
    for s in r.segments:
        assert np.all(s.q >= lo + 1e-3) and np.all(s.q <= hi - 1e-3)
    start = tgt.copy()
    r2 = plan(table, PlanGoal("move_joint", "left", q=tuple(float(v) for v in q[J.LEFT_ARM])), q, arm_target=start)
    assert r2.ok and np.all(r2.segments[0].q >= lo + 1e-3) and np.max(np.abs(r2.segments[0].q[0] - start)) <= 0.05
    assert ik.KNOT_MARGIN_RAD >= 1e-3


@pytest.mark.parametrize(
    "goal",
    [
        PlanGoal("lower", "left", until="depth", max_depth_m=0.03),
        PlanGoal("lower", "left", until="depth", max_depth_m=0.04),  # 4.5 cm down is beyond the hanging arm's reach
        PlanGoal("move_linear", "left", delta=(0.0, 0.0, -0.03)),
    ],
)
def test_straight_descents_from_the_resting_arm(table: WorldSnapshot, goal: PlanGoal) -> None:
    """api 3: from the hanging arm a 1 cm step down needs ~0.11 rad at the elbow per cm (near the workspace's edge) -
    the line's 0.25 rad 'joint jump' test refused it; the step is now halved down to 2 mm before a jump is called.
    411f571: 'the line needs a joint jump at 100%'."""
    r = plan(table, goal, J.POSE_STAND.copy())
    assert r.ok, r.reason
    geo = Geometry(table)
    geo.set_stance(J.POSE_STAND)
    p0 = geo.point("left", "pinch")
    geo.set_arm("left", r.segments[-1].q[-1])
    p1 = geo.point("left", "pinch")
    dz = goal.max_depth_m if goal.kind == "lower" else 0.03
    assert np.linalg.norm(p1 - (p0 - np.array([0.0, 0.0, dz]))) < 0.003


def test_hover_datum_is_position_z_and_the_lowest_fingertip() -> None:
    """api 4 / compat 2b: height_m = the LOWEST FINGERTIP (v2's datum, = the arm model's lowest_tip_z) above
    position[2] when > 0 (compat sends the table top as v2 did), else above the highest declared top under (x, y);
    predicted.lowest_tip_z_m reports it, landing_z_m = the lowest collision point (can be lower: api's 3.6 vs 6 cm).
    411f571: no lowest_tip_z_m, and the datum was always the highest declared top (here the board, +1.2 cm)."""
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.30, 0.0, 0.40), size=(0.4, 1.0, 0.04)))
    w.declare(Surface("board", Pose.xyz_yaw(0.25, 0.15, 0.412), size=(0.2, 0.2, 0.012)))
    snap = w.snapshot()
    q = J.POSE_STAND.copy()
    for h in (0.10, 0.12, 0.15):
        r = plan(snap, PlanGoal("hover", "left", position=(0.25, 0.15, 0.40), height_m=h), q)
        assert r.ok, r.reason
        assert r.predicted["lowest_tip_z_m"] == pytest.approx(0.40 + h, abs=0.0015), h
        assert r.predicted["landing_z_m"] <= r.predicted["lowest_tip_z_m"] + 1e-9
    r0 = plan(snap, PlanGoal("hover", "left", position=(0.25, 0.15, 0.0), height_m=0.12), q)
    assert r0.ok and r0.predicted["lowest_tip_z_m"] == pytest.approx(0.532, abs=0.0015)  # the board's top + 12 cm


def test_the_hand_works_just_above_a_support(table: WorldSnapshot) -> None:
    """compat 6: after a lower near the support the client's next small move was refused COLLISION (the 1 cm clearance
    against the table for the HAND), though v2 made it on the robot with the middle tip 8 mm above the board. The hand
    may now come within 2 mm of a support (a surface's top, a container's floor); the arm / robot / obstacles keep 1 cm.
    411f571: 'blocked at 50%: right_hand_thumb_2_link touches entity:table'."""
    q = J.POSE_STAND.copy()
    h = plan(table, PlanGoal("hover", "right", position=(0.38, -0.12, 0.0), height_m=0.04), q)
    assert h.ok, h.reason
    q2 = q.copy()
    q2[J.RIGHT_ARM] = h.segments[-1].q[-1]
    lo = plan(
        table, PlanGoal("lower", "right", until="depth", max_depth_m=h.predicted["landing_z_m"] - (TZ + 0.0135)), q2
    )
    assert lo.ok, lo.reason
    assert lo.predicted["landing_z_m"] > TZ + 0.010  # the start is clear of the old 1 cm clearance
    q3 = q.copy()
    q3[J.RIGHT_ARM] = lo.segments[-1].q[-1]
    mv = plan(table, PlanGoal("move_linear", "right", delta=(0.015, 0.0, -0.006)), q3)
    assert mv.ok, mv.reason
    assert TZ + 0.002 <= mv.predicted["landing_z_m"] < TZ + 0.010


def test_the_arm_and_obstacles_keep_the_full_clearance() -> None:
    """The 2 mm is for the HAND over a SUPPORT only: an obstacle 5 mm under the hand's lowest shape, or a surface 5 mm
    under the forearm, still blocks (DEFAULT_CLEARANCE_M 1 cm)."""
    q26 = J.POSE_STAND.copy()
    probe = Geometry()
    probe.set_stance(q26)
    m, d = probe.m, probe.d
    low_hand = probe.lowest("right")
    hand_geoms = [
        g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in probe.hand_bodies["right"] and m.geom_contype[g]
    ]
    p = d.geom_xpos[min(hand_geoms, key=lambda g: float(d.geom_xpos[g][2]))].copy()  # under the hand's lowest shape
    elbow = int(m.body("right_elbow_link").id)
    caps = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == elbow and m.geom_contype[g]]
    ends = []  # the forearm's shapes' lowest points (a capsule: its lower end)
    for g in caps:
        ax = d.geom_xmat[g].reshape(3, 3)[:, 2] * (m.geom_size[g][1] if m.geom_type[g] == 3 else 0.0)
        c = d.geom_xpos[g]
        ends.append(min((c + ax, c - ax), key=lambda v: float(v[2])) - np.array([0.0, 0.0, m.geom_size[g][0]]))
    ep = min(ends, key=lambda v: float(v[2]))
    low_elbow = float(ep[2])
    hand = {m.body(b).name for b in probe.hand_bodies["right"]}

    def hits_with(e: Surface | Obstacle) -> list[str]:
        w = WorldModel()
        w.declare(e)
        g = Geometry(w.snapshot())
        g.set_stance(q26)
        return [f"{h.arm_body}>{h.other}" for h in g.hits("right")]

    assert hits_with(Surface("s", Pose.xyz_yaw(p[0], p[1], low_hand - 0.005), size=(0.06, 0.06, 0.01))) == []
    box = Obstacle("o", Pose.xyz_yaw(p[0], p[1], low_hand - 0.015), shape="box", size=(0.03, 0.03, 0.01))
    assert any(h.endswith("entity:o") for h in hits_with(box))
    hs = hits_with(Surface("s", Pose.xyz_yaw(ep[0], ep[1], low_elbow - 0.005), size=(0.02, 0.02, 0.01)))
    assert any(h.endswith("entity:s") and h.split(">")[0] not in hand for h in hs), hs


def test_the_planner_fk_is_the_rt_cores_arm_model() -> None:
    """api 2 / 5: the planner's points = the RT core's built-in arm model (r1rt_py.fingertips_base, the executor's
    surface guard and r1d's measured results) for the same joints + IMU reading + hands: the IMU's mounting rotation
    undone in both (ed81598). 411f571 already agrees; before ed81598 the gap was 1-2 cm (api's 20:40 reading)."""
    from r1mc3.log import _native

    try:
        R = _native.native()
    except ImportError:
        pytest.skip("r1rt_py not built (scripts/ci.sh cpp)")
    if not hasattr(R, "fingertips_base"):
        pytest.skip("r1rt_py without the arm model")
    from r1sim import dex3

    g = Geometry()
    rng = np.random.default_rng(0)
    for _ in range(5):
        q26 = J.POSE_STAND.copy()
        q26[J.ARMS] = rng.uniform(J.SOFT_MIN[J.ARMS], J.SOFT_MAX[J.ARMS])
        quat = np.array([1.0, *rng.uniform(-0.03, 0.03, 3)])
        quat /= np.linalg.norm(quat)
        hl, hr = dex3.preset("open", "left"), dex3.preset("open", "right")
        pts, low, _ = R.fingertips_base(q26[None], quat[None], hl[None], hr[None])
        g.set_stance(q26, quat, {"left": hl, "right": hr})
        for i, s in enumerate(("left", "right")):
            mine = [g.point(s, k) for k in ("palm", "thumb_tip", "middle_tip", "index_tip", "grasp_centre")]
            assert np.max(np.abs(np.array(mine) - pts[0, i])) < 1e-6
            assert min(float(g.point(s, k)[2]) for k in ("thumb_tip", "middle_tip", "index_tip")) == pytest.approx(
                float(low[0, i]), abs=1e-6
            )
