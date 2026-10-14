"""r1mc3.measure.register: an entity's pose from its tags (DESIGN §8.2). Synthetic views: the tag corners projected
through the head camera's calibrated lens (pinhole + k1, f 127 px) from a head-like pose, with pixel noise."""

from __future__ import annotations

import math

import numpy as np
import pytest
from r1mc3.measure import HeadLens, TagObs, View, flat_tag, locate_entity
from r1mc3.measure.register import rodrigues, rotvec
from r1mc3.measure.tags import rot_z


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """An OpenCV-frame camera (z forward, y down) at `eye` looking at `target` (world z up)."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = eye
    return T


def pose(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_z(math.radians(yaw_deg))
    T[:3, 3] = [x, y, z]
    return T


BOARD = [  # a 35.6 cm square board's corner tags (4 cm near, 8 cm far), the board frame at its near-left corner
    flat_tag(0, 0.04, 0.035, -0.035),
    flat_tag(1, 0.04, 0.035, -0.321),
    flat_tag(2, 0.08, 0.300, -0.056),
    flat_tag(3, 0.08, 0.300, -0.300),
]


def synth_view(lens: HeadLens, T_ref_cam: np.ndarray, T_ref_ent: np.ndarray, tags, noise_px: float, rng) -> View:
    T_cam_ref = np.linalg.inv(T_ref_cam)
    out = {}
    for t in tags:
        P = (T_cam_ref @ T_ref_ent @ np.c_[t.corners_entity(), np.ones(4)].T).T[:, :3]
        px = lens.project(P) + rng.normal(0.0, noise_px, (4, 2))
        if np.all(np.isfinite(px)) and np.all((px >= 0) & (px < [lens.width, lens.height])):
            out[t.tag_id] = TagObs(t.tag_id, px)
    return View(out, lens, T_ref_cam)


@pytest.fixture(scope="module")
def lens() -> HeadLens:
    return HeadLens.load()


def test_rotation_vector_round_trip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        rv = rng.normal(0, 1.0, 3)
        assert np.allclose(rotvec(rodrigues(rv)), rv, atol=1e-9) or np.linalg.norm(rv) > math.pi


def test_a_board_seen_by_the_head_camera_is_located_within_the_5_mm_target(lens: HeadLens) -> None:
    rng = np.random.default_rng(1)
    T_cam = look_at(np.array([0.05, 0.0, 1.22]), np.array([0.35, 0.0, 0.75]))
    truth = pose(0.09, 0.178, 0.763, 1.5)
    errs, yaws = [], []
    for _ in range(10):
        v = synth_view(lens, T_cam, truth, BOARD, 0.3, rng)
        r = locate_entity("board", BOARD, [v])
        assert r.ok, r.reason
        errs.append(1000 * np.linalg.norm(r.xyz - truth[:3, 3]))
        yaws.append(abs(r.yaw_deg - 1.5))
    # DESIGN §8.2 target: 5 mm and 0.5 deg (0.3 px corner noise ~ the JPEG + subpixel corners)
    assert max(errs) < 5.0 and max(yaws) < 0.5, (errs, yaws)
    assert r.used == (0, 1, 2, 3) and r.rms_px < 0.6
    assert r.sigma["x_mm"] < 5 and r.sigma["yaw_deg"] < 0.5


def test_one_tag_is_refused_and_says_which_are_missing(lens: HeadLens) -> None:
    rng = np.random.default_rng(2)
    T_cam = look_at(np.array([0.05, 0.0, 1.22]), np.array([0.35, 0.0, 0.75]))
    v = synth_view(lens, T_cam, pose(0.09, 0.178, 0.763, 0.0), BOARD[:1], 0.3, rng)
    r = locate_entity("board", BOARD, [v])
    assert not r.ok and r.missing == (1, 2, 3)
    assert "only 1 of 4 tags seen on the board" in r.reason and "1, 2, 3" in r.reason


def test_two_views_combine_and_a_shifted_tag_shows_in_its_error(lens: HeadLens) -> None:
    rng = np.random.default_rng(3)
    truth = pose(0.081, 0.0, 0.751, -0.8)
    tags = [flat_tag(10, 0.12, 0.65, 0.55), flat_tag(11, 0.12, 0.65, -0.48), flat_tag(12, 0.05, 0.05, -0.30)]
    v1 = synth_view(lens, look_at(np.array([0.05, 0.0, 1.22]), np.array([0.5, 0.35, 0.75])), truth, tags, 0.3, rng)
    v2 = synth_view(lens, look_at(np.array([0.05, 0.0, 1.22]), np.array([0.5, -0.35, 0.75])), truth, tags, 0.3, rng)
    r = locate_entity("table", tags, [v1, v2])
    assert r.ok and r.n_views == 2 and r.used == (10, 11, 12)
    assert 1000 * np.linalg.norm(r.xyz - truth[:3, 3]) < 5.0 and abs(r.yaw_deg + 0.8) < 0.5
    # a tag placed 2 cm off its plan: the rigid fit no longer fits it (its own error grows, the rms rises)
    bad = [flat_tag(10, 0.12, 0.65, 0.55), flat_tag(11, 0.12, 0.65, -0.48), flat_tag(12, 0.05, 0.07, -0.30)]
    v1b = synth_view(lens, look_at(np.array([0.05, 0.0, 1.22]), np.array([0.5, 0.35, 0.75])), truth, bad, 0.3, rng)
    v2b = synth_view(lens, look_at(np.array([0.05, 0.0, 1.22]), np.array([0.5, -0.35, 0.75])), truth, bad, 0.3, rng)
    rb = locate_entity("table", tags, [v1b, v2b])
    assert rb.rms_px > 3 * r.rms_px
    assert max(rb.tag_err_mm.values()) > 5.0


def test_the_free_fit_recovers_a_tilted_entity(lens: HeadLens) -> None:
    rng = np.random.default_rng(4)
    truth = pose(0.09, 0.178, 0.763, 3.0)
    truth[:3, :3] = truth[:3, :3] @ rodrigues([0.02, -0.03, 0.0])
    T_cam = look_at(np.array([0.05, 0.0, 1.22]), np.array([0.35, 0.0, 0.75]))
    v = synth_view(lens, T_cam, truth, BOARD, 0.2, rng)
    r = locate_entity("board", BOARD, [v], level=False)
    assert r.ok and not r.level
    assert 1000 * np.linalg.norm(r.xyz - truth[:3, 3]) < 5.0
    assert r.T_ref_entity is not None
    dR = rotvec(r.T_ref_entity[:3, :3].T @ truth[:3, :3])
    # the yaw is pinned (0.15 deg here); roll / pitch of a 36 cm board from one view are not (0.7 deg at 0.2 px): the
    # reason `level=True` is the default for horizontal faces
    assert abs(math.degrees(dR[2])) < 0.5 and math.degrees(np.linalg.norm(dR[:2])) < 1.5
