"""arm.move_path's blend (session 11, r1mc3.plan.blend): the planned, checked legs -> ONE trajectory through the
waypoints that does not stop at the interior ones, re-checked with the planner's clearance before it is used.

On the base (1e55bdd) the module did not exist (S10: "move_path blending - slipped"; every path stopped at each
waypoint). Planner-level: the legs are planned inline as r1d's path chain plans them (leg k+1 from leg k's planned end).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3.plan import PlanGoal, PlanResult, geometry_for, plan
from r1mc3.plan import path as P
from r1mc3.plan.blend import BLEND_MAX_M, SAMPLE_DT, blend_path
from r1mc3.plan.ik import arm_limits
from r1mc3.plan.timing import V_MAX, Timed, sample
from r1mc3.service.l2_paths import _joined
from r1mc3.world import Obstacle, Pose, Surface, WorldModel, WorldSnapshot
from r1mc3.world.geometry import ARM_JOINTS
from r1sim import joints as J

ARM = "left"
IDX = list(ARM_JOINTS[ARM])
SPEED = 0.4


@pytest.fixture(scope="module")
def q() -> np.ndarray:
    return J.POSE_STAND.copy()


@pytest.fixture(scope="module")
def pinch(q: np.ndarray) -> np.ndarray:
    geo = geometry_for(WorldModel().snapshot())
    geo.set_stance(q, None, None)
    geo.set_arm(ARM, q[IDX])
    return np.asarray(geo.point(ARM, "pinch"), dtype=np.float64)


def _table() -> WorldSnapshot:
    w = WorldModel()
    w.declare(Surface("table", Pose.xyz_yaw(0.55, 0.0, 0.76), size=(0.6, 1.2, 0.04)))
    return w.snapshot()


def _chain(snap: WorldSnapshot, q: np.ndarray, wps: list[np.ndarray], kind: str = "move_linear") -> PlanResult:
    """r1d's path chain (service/l2_paths._PlanChain) inline: each waypoint planned from the previous planned end."""
    res: list[Any] = []
    target, q26 = q[IDX].copy(), q.copy()
    for wp in wps:
        r = plan(snap, PlanGoal.from_dict({"kind": kind, "arm": ARM, "position": list(wp), "speed": SPEED}), q26,
                 arm_target=target)  # fmt: skip
        res.append(r)
        assert r.ok, r.reason
        target = r.segments[-1].q[-1]
        q26 = q26.copy()
        q26[IDX] = target
    out: PlanResult = _joined(res, len(wps))
    return out


def _timed(seg: Any) -> Timed:
    return Timed(seg.t, seg.q, seg.dq, seg.ddq, 0.0, 0.0, 0.0)


def _hand_path(snap: WorldSnapshot, q: np.ndarray, seg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geo = geometry_for(snap)
    geo.set_stance(q, None, None)
    ts, qs, _ = sample(_timed(seg), SAMPLE_DT)
    pts = []
    for qq in qs:
        geo.set_arm(ARM, qq)
        pts.append(geo.point(ARM, "pinch"))
    return ts, qs, np.asarray(pts)


def _all_valid(snap: WorldSnapshot, q: np.ndarray, qs: np.ndarray, from_k: int = 0) -> list[int]:
    """An INDEPENDENT re-check: the planner's checker on every sample (the default 1 cm clearance) -> the bad ones."""
    geo = geometry_for(snap)
    geo.set_stance(q, None, None)
    chk = P.Checker(geo, ARM, arm_limits(ARM))
    return [k for k in range(from_k, len(qs)) if not chk.valid(qs[k])]


def test_a_blended_path_does_not_stop_at_the_waypoints_and_is_faster(q: np.ndarray, pinch: np.ndarray) -> None:
    snap = _table()
    wps = [pinch + np.array(o) for o in ((0.08, 0, 0.0), (0.08, 0, 0.08), (0.0, 0, 0.08))]
    joined = _chain(snap, q, wps)
    assert len(joined.segments) == 3  # unblended: one trajectory per leg, each from rest to rest
    b = blend_path(snap, joined, q, arm=ARM, radius_m=0.02, speed=SPEED, arm_target=q[IDX])
    info = b.details["blend"]
    assert b.ok and info["applied"] and len(b.segments) == 1, info
    seg = b.segments[0]
    assert np.allclose(seg.q[0], joined.segments[0].q[0]) and np.allclose(seg.q[-1], joined.segments[-1].q[-1])
    assert np.allclose(seg.dq[0], 0) and np.allclose(seg.dq[-1], 0)  # from rest, to rest - only at the ends
    assert b.predicted["q_end"] == joined.predicted["q_end"]  # the end (and its prediction) unchanged
    assert b.duration_s < joined.duration_s, (b.duration_s, joined.duration_s)
    _, qs, pts = _hand_path(snap, q, seg)
    _, _, dqs = sample(_timed(seg), SAMPLE_DT)
    assert np.max(np.abs(dqs)) <= V_MAX * SPEED * 1.001
    # each interior waypoint passed within the tolerance, the hand MOVING there (unblended: at rest, 0 m/s)
    speed = np.linalg.norm(np.diff(pts, axis=0), axis=1) / SAMPLE_DT
    for k, w in enumerate(wps[:-1]):
        d = np.linalg.norm(pts - w, axis=1)
        i = int(np.argmin(d))
        assert d[i] <= 0.02 + 0.001, (k, d[i])
        assert speed[min(i, len(speed) - 1)] > 0.01, (k, speed[i])  # m/s
    assert np.linalg.norm(pts[-1] - wps[-1]) < 0.003  # the last waypoint reached as planned (at rest)
    assert _all_valid(snap, q, qs) == []
    # the unblended plan for comparison: its legs end at rest at the waypoints
    assert all(np.allclose(s.dq[-1], 0) for s in joined.segments)


def test_a_blend_never_trades_clearance_for_smoothness(q: np.ndarray, pinch: np.ndarray) -> None:
    """Up, then forward over a box the hanging fingers start next to (inside the 1 cm clearance: the planner's leave
    rule takes the first leg away from it): the 5 cm corner would put the middle finger on the box -> a smaller corner,
    or (as here: the half radius touches it too, the pass-through puts the palm on the hip) the planned stops - and
    whatever comes back re-checks clear once clear of the start (an independent checker on every sample)."""
    w = WorldModel()
    w.declare(Obstacle("box", Pose.xyz_yaw(pinch[0] + 0.07, pinch[1], pinch[2] - 0.09), size=(0.05, 0.06, 0.05)))
    snap = w.snapshot()
    wps = [pinch + np.array(o) for o in ((0.0, 0, 0.08), (0.10, 0, 0.08))]
    joined = _chain(snap, q, wps)
    b = blend_path(snap, joined, q, arm=ARM, radius_m=BLEND_MAX_M, speed=SPEED, arm_target=q[IDX])
    info = b.details["blend"]
    tries = info.get("tries", [])
    assert tries and "box" in tries[0]["why"], info
    if info["applied"]:
        assert info["radius_m"] < BLEND_MAX_M, info
    else:
        assert b.segments == joined.segments, info  # the stops, exactly as planned
    for seg in b.segments:
        _, qs, _ = _hand_path(snap, q, seg)
        bad = _all_valid(snap, q, qs)
        first_clear = next(k for k in range(len(qs)) if k not in set(bad))
        assert _all_valid(snap, q, qs, first_clear) == [], seg.leg


def test_a_smaller_corner_when_the_big_one_leaves_the_tolerance(q: np.ndarray, pinch: np.ndarray) -> None:
    """The tolerance is the client's blend_m everywhere: the hand within it of the planned path and of each waypoint."""
    snap = _table()
    wps = [pinch + np.array(o) for o in ((0.08, 0, 0.0), (0.08, 0, 0.08), (0.0, 0, 0.08))]
    joined = _chain(snap, q, wps)
    for r in (0.01, 0.03):
        b = blend_path(snap, joined, q, arm=ARM, radius_m=r, speed=SPEED, arm_target=q[IDX])
        info = b.details["blend"]
        assert info["applied"], info
        assert info["deviation_mm"] <= 1000 * r + 1.0 and max(info["closest_mm"]) <= 1000 * r + 1.0, info


def test_a_blend_that_cannot_pass_returns_the_stops(
    q: np.ndarray, pinch: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If no corner (not even the pass-through) re-checks clear, the result is the UNBLENDED plan with the reason."""
    snap = _table()
    wps = [pinch + np.array(o) for o in ((0.05, 0, 0.0), (0.05, 0, 0.05))]
    joined = _chain(snap, q, wps)
    calls = {"n": 0}

    def never(self: P.Checker, qq: np.ndarray) -> bool:
        calls["n"] += 1
        self.last_hit = "a test obstacle"
        return False

    monkeypatch.setattr(P.Checker, "valid", never)
    b = blend_path(snap, joined, q, arm=ARM, radius_m=0.03, speed=SPEED, arm_target=q[IDX])
    info = b.details["blend"]
    assert not info["applied"] and calls["n"] >= 3, info
    assert [t["radius_m"] for t in info["tries"]] == [0.03, 0.015, 0.0]
    assert b.segments == joined.segments and b.plan_id == joined.plan_id  # the stops, exactly as planned


def test_what_is_never_blended(q: np.ndarray, pinch: np.ndarray) -> None:
    snap = _table()
    one = _chain(snap, q, [pinch + np.array([0.04, 0, 0.0])])
    assert blend_path(snap, one, q, arm=ARM, radius_m=0.02).details["blend"]["applied"] is False
    two = _chain(snap, q, [pinch + np.array([0.04, 0, 0.0]), pinch + np.array([0.04, 0, 0.04])])
    for r in (0.0, BLEND_MAX_M + 0.01):
        assert blend_path(snap, two, q, arm=ARM, radius_m=r).details["blend"]["applied"] is False
    guarded = PlanResult(True, "OK", "x", (two.segments[0], type(two.segments[1])(**{
        **{k: getattr(two.segments[1], k) for k in ("arm", "t", "q", "dq", "ddq", "leg")}, "guards": 1 | 4})),
        two.predicted)  # fmt: skip
    why = blend_path(snap, guarded, q, arm=ARM, radius_m=0.02).details["blend"]["why"]
    assert "guard" in why
    refused = PlanResult(False, "UNREACHABLE", "y", reason="no")
    assert blend_path(snap, refused, q, arm=ARM, radius_m=0.02) is refused


def test_the_blend_runs_in_the_planner_worker(q: np.ndarray, pinch: np.ndarray) -> None:
    """r1d computes the blend where the planner runs (PlannerWorker.call: the compiled geometry lives there)."""
    from r1mc3.plan import PlannerWorker

    snap = _table()
    joined = _chain(snap, q, [pinch + np.array([0.05, 0, 0.0]), pinch + np.array([0.05, 0, 0.05])])
    inline = blend_path(snap, joined, q, arm=ARM, radius_m=0.02, speed=SPEED, arm_target=q[IDX])
    with PlannerWorker(warm=False) as wk:
        res = wk.call(blend_path, snap, joined, q, arm=ARM, radius_m=0.02, speed=SPEED, arm_target=q[IDX]).result(60)
    assert res.ok and res.details["blend"]["applied"] == inline.details["blend"]["applied"] is True
    assert np.allclose(res.segments[0].q, inline.segments[0].q) and np.allclose(res.segments[0].t, inline.segments[0].t)
