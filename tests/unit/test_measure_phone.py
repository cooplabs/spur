"""r1mc3.measure.phone: the hand-placed phone's pose from table tags (DESIGN §8.2: "the top phone might not be
positioned perfectly and there needs to be a correction algorithm"). Synthetic stills: a calibrated phone (1920 x 1080,
f 1450 px, barrel distortion) placed differently each time, the tag corners projected with 0.5 px noise."""

from __future__ import annotations

import math

import numpy as np
import pytest
from r1mc3.measure import OpenCVLens, TagObs, flat_tag
from r1mc3.measure.phone import correct_phone
from r1mc3.measure.register import rodrigues

LENS = OpenCVLens.pinhole(1450.0, 1920, 1080, dist=[-0.06, 0.02, 0.0, 0.0, 0.0])
TABLE_TAGS = [  # table frame: x from the near edge, y to the robot's left, z = the table top; 6 tags in the view
    flat_tag(20, 0.06, 0.08, 0.45),
    flat_tag(21, 0.06, 0.08, -0.45),
    flat_tag(22, 0.08, 0.55, 0.50),
    flat_tag(23, 0.08, 0.55, -0.50),
    flat_tag(24, 0.06, 0.30, 0.55),
    flat_tag(25, 0.06, 0.30, -0.55),
]


def phone_pose(x: float, y: float, dist: float, tilt_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """A camera `dist` m from the table point (x, y) along its optical axis, looking down at that point tilted /
    rolled / turned (table frame, OpenCV camera; image up = +x, the far table)."""
    down = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    R = rodrigues([0, 0, math.radians(yaw_deg)]) @ rodrigues([math.radians(roll_deg), math.radians(tilt_deg), 0]) @ down
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([x, y, 0.0]) - dist * R[:, 2]
    return T


def still(T_table_cam: np.ndarray, tags=TABLE_TAGS, noise: float = 0.5, seed: int = 0) -> dict[int, TagObs]:
    rng = np.random.default_rng(seed)
    T = np.linalg.inv(T_table_cam)
    out = {}
    for t in tags:
        P = (T @ np.c_[t.corners_entity(), np.ones(4)].T).T[:, :3]
        px = LENS.project(P) + rng.normal(0, noise, (4, 2))
        if np.all((px >= 0) & (px < [1920, 1080])):
            out[t.tag_id] = TagObs(t.tag_id, px)
    return out


@pytest.mark.parametrize(
    "pose",
    [(0.30, 0.02, 0.95, 0, 0, 0), (0.33, -0.04, 0.98, 10, -5, 5), (0.28, 0.05, 1.00, -12, 6, -8)],
)
def test_the_correction_recovers_a_moved_tilted_phone_to_a_millimetre(pose: tuple[float, ...]) -> None:
    T_true = phone_pose(*pose)
    pc = correct_phone(still(T_true), LENS, TABLE_TAGS)
    assert pc.ok, pc.reason
    assert pc.T_table_cam is not None
    assert 1000 * np.linalg.norm(pc.T_table_cam[:3, 3] - T_true[:3, 3]) < 10.0
    # what matters: table points from pixels - a grid over the table, through the TRUE camera, back through the fit
    g = np.array([[x, y, 0.0] for x in np.linspace(0.05, 0.60, 7) for y in np.linspace(-0.5, 0.5, 9)])
    px = LENS.project((np.linalg.inv(T_true)[:3, :3] @ g.T).T + np.linalg.inv(T_true)[:3, 3])
    inside = np.all((px >= 0) & (px < [1920, 1080]), axis=1)
    err = 1000 * np.linalg.norm(pc.to_plane(px[inside])[:, :2] - g[inside, :2], axis=1)
    assert err.max() < 2.0, err.max()  # room under a 5 mm object-placement rule
    assert pc.fit_rms_mm < 1.0 and pc.loo_max_mm < 2.0
    assert abs(pc.tilt_deg - math.hypot(pose[3], pose[4])) < 1.0
    assert 0.4 < pc.mm_per_px < 0.8


def test_an_object_top_maps_without_parallax_at_its_height() -> None:
    T_true = phone_pose(0.30, 0.10, 0.90, 10, 0, 0)
    pc = correct_phone(still(T_true), LENS, TABLE_TAGS)
    top = np.array([[0.20, -0.10, 0.132]])  # a 13.2 cm tall object's top centre
    px = LENS.project((np.linalg.inv(T_true)[:3, :3] @ top.T).T + np.linalg.inv(T_true)[:3, 3])
    on_table = pc.to_plane(px)[0]
    at_height = pc.to_plane(px, z=0.132)[0]
    assert 1000 * np.linalg.norm(at_height[:2] - top[0, :2]) < 1.5
    assert 1000 * np.linalg.norm(on_table[:2] - top[0, :2]) > 10.0  # the error a plain homography would make


def test_too_few_tags_too_oblique_and_a_moved_tag_are_refused_with_words() -> None:
    T_true = phone_pose(0.30, 0.0, 0.95, 0, 0, 0)
    two = {k: v for k, v in still(T_true).items() if k in (20, 21)}
    r = correct_phone(two, LENS, TABLE_TAGS)
    assert not r.ok and "only 2 table tags in the phone still (need 3)" in r.reason and "22, 23, 24, 25" in r.reason
    steep_still = still(phone_pose(0.30, 0.0, 1.2, -55, 0, 0))
    assert len(steep_still) >= 3
    steep = correct_phone(steep_still, LENS, TABLE_TAGS)
    assert not steep.ok and "too oblique" in steep.reason
    moved = [*TABLE_TAGS[:5], flat_tag(25, 0.06, 0.33, -0.55)]
    r2 = correct_phone(still(T_true, tags=moved), LENS, TABLE_TAGS)  # tag 25 lies 3 cm off its plan
    assert not r2.ok and ("fit error" in r2.reason or "left out" in r2.reason)
