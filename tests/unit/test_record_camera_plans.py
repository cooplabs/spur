"""The camera record kinds (session 9, r1cam): nothing moves; the S11 / S12 schedules (HW1_RUNBOOK_ADDITIONS S6 / S7:
3 light levels + turned; 3 distances x 9 places x 3 tilts, 2 s still each) and the r1cam streams per kind."""

from __future__ import annotations

import pytest
from r1mc3.record.plans import CAMERA_KINDS, PlanError, plan


@pytest.mark.parametrize("kind", CAMERA_KINDS)
def test_camera_kinds_move_nothing(kind: str) -> None:
    p = plan(kind, {"label": "x"}, None)
    assert p.group is None and p.joints == () and p.hand is None and p.setpoint is None and p.needs_mode is None
    assert p.camera is not None and p.camera["schedule"] and p.describe()["moves"] == []


def test_the_charuco_sweep_is_s12s_81_poses_with_depth_at_2_fps() -> None:
    p = plan("charuco", {}, None)
    assert p.camera is not None
    sched = p.camera["schedule"]
    assert len(sched) == 3 * 9 * 3 and p.camera["streams"] == ["left", "right", "depth"]
    assert p.camera["depth_every"] == 5  # ~2 of the 10 depth frames per second (R1CAM_PLAN.md §2)
    assert all(e["t_end"] - e["t_start"] == 2.0 for e in sched)
    assert sched[0]["say"] == "0.30 m, top-left, flat: hold still"
    assert 320 <= p.duration_s <= 330  # 81 x (2 s move + 2 s still) + 1 s: ~5.4 min
    assert p.params["board"] == "poster_6x4_90"
    with pytest.raises(PlanError):
        plan("charuco", {"distances_m": [0.05]}, None)


def test_flat_field_and_depth_targets() -> None:
    ff = plan("flat_field", {}, None)
    assert ff.camera is not None and len(ff.camera["schedule"]) == 6 and "depth" not in ff.camera["streams"]
    dt = plan("depth_targets", {"label": "board_flat_0p45", "distance_m": 0.45}, None)
    assert dt.camera is not None and dt.camera["streams"] == ["left", "depth"] and dt.duration_s == 6.0
    with pytest.raises(PlanError):
        plan("depth_targets", {}, None)  # what the target is must be said
