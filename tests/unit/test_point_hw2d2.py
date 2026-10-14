"""Session 13 (HW2 day 2's teleop runs: the click -> object layer 2-6 cm off): r1d's generic pointing on synthetic
depth with the fitted native-depth model's effects (r1sim.depth, A6) - a click whose patch straddles an object's edge
reads the surface the clicked pixel shows (the object, not what lies behind it), a click with no depth is refused in
words (the foot click / the height named), and the object extent no longer inflates on a tilted depth bias and a
low-passed outline. Each test fails on the base (dcc01a2): the median read the surface behind (height 0), no depth was
"taken as on the surface", the extents grew 2-4 cm along the line of sight."""

from __future__ import annotations

import math

import numpy as np
import pytest
from r1mc3.cam import pointing as PT
from r1sim import depth as D

from tests.unit.test_pointing import MODEL, TABLE_Z, _camera, _render_depth, _supports

# a box seen from the head at ~33 deg (day 2's head pitch 0.5729 rad); the click near its far top edge
Box = tuple[float, float, float, float, float]
CAR: Box = (0.24, 0.338, -0.027, 0.027, TABLE_Z + 0.039)  # the low box's 9.8 x 5.4 x 3.9 cm, centre (0.289, 0)
TEA: Box = (0.25, 0.328, 0.06, 0.127, TABLE_Z + 0.112)  # 7.8 x 6.7 x 11.2 cm, centre (0.289, 0.0935)


def _px(T: np.ndarray, p: tuple[float, float, float]) -> tuple[float, float]:
    pc = np.linalg.inv(T) @ np.r_[np.asarray(p, float), 1.0]
    uv = PT.project(MODEL, pc[None, :3])[0]
    return float(uv[0]), float(uv[1])


def test_a_click_straddling_an_objects_edge_reads_the_clicked_surface() -> None:
    T = _camera(math.degrees(0.5729))
    box = (0.30, 0.38, -0.04, 0.04, TABLE_Z + 0.10)
    depth = _render_depth(T, [box])
    # the pixel 2 px inside the box top's FAR edge: most of the 7 x 7 patch looks past it at the table behind
    u, v = _px(T, (0.38, 0.0, TABLE_Z + 0.10))
    r = PT.point(depth, MODEL, T, u, v + 2.0, _supports())
    assert r["ok"] and r["depth_pick"] == "nearer of two", r
    assert r["height_above_surface_m"] == pytest.approx(0.10, abs=0.01)  # base: 0.0 (the median = the table behind)
    assert r["point_base"][0] < 0.385  # base: the table hit ~6 cm beyond the box
    # a uniform patch keeps v2's median; a click on the TABLE just beyond the box (the patch reaching onto the box's
    # top edge) stays on the table
    u2, v2 = _px(T, (0.34, 0.0, TABLE_Z + 0.10))
    assert PT.point(depth, MODEL, T, u2, v2, _supports())["depth_pick"] == "median"
    t = PT.point(depth, MODEL, T, u, v - 2.0, _supports())
    assert t["ok"] and t["height_above_surface_m"] == 0.0, t
    assert t["point_base"][0] > 0.40  # the table behind the box (the true hit of that line of sight)


def test_no_depth_at_the_click_is_refused_in_words_and_the_foot_click_needs_none() -> None:
    T = _camera(math.degrees(0.5729))
    depth = _render_depth(T, [CAR])
    u, v = _px(T, (0.289, 0.0, TABLE_Z + 0.039))
    holed = depth.copy()
    holed[int(v) - 6 : int(v) + 7, int(u) - 6 : int(u) + 7] = np.nan  # a clear / shiny top: no stereo match
    r = PT.point(holed, MODEL, T, u, v, _supports())
    assert r["ok"] is False and r["code_hint"] == "NO_DEPTH", r  # base: ok, "taken as on the surface", 2.9 cm beyond
    assert "foot" in r["error"] and "height" in r["error"]
    # the foot click (use_depth False): where the line of sight meets the surface, never refused for depth
    uf, vf = _px(T, (0.24, 0.0, TABLE_Z))
    f = PT.point(holed, MODEL, T, uf, vf, _supports(), use_depth=False)
    assert f["ok"] and f["via"] == "plane" and f["point_base"][0] == pytest.approx(0.24, abs=0.002)


def _a6(depth: np.ndarray, T: np.ndarray, *, seed: int) -> np.ndarray:
    """The fitted native-depth model (models/depth.yaml: the outline low-pass - objects 2.34 px, the plain table among
    itself -, the disparity field's tilt across the image, the correlated noise), seeded."""
    cfg = D.DepthConfig.from_yaml()
    z = np.nan_to_num(depth, nan=0.0)
    o = T[:3, 3]
    v, u = np.mgrid[0:448, 0:544].astype(float)
    rays = PT.unproject(MODEL, u.ravel(), v.ravel(), np.ones(u.size)) @ T[:3, :3].T
    plane_t = ((TABLE_Z - o[2]) / rays[:, 2]).reshape(448, 544)
    plain = np.abs(z - plane_t) < 1e-6
    out = D.apply(z, cfg, seed=seed, plain=plain).astype(float) / 1000.0
    out[out <= 0] = np.nan
    return out


@pytest.mark.parametrize("name,box,size", [("low box", CAR, (0.098, 0.054)), ("tall box", TEA, (0.078, 0.067))])
def test_the_object_extent_does_not_inflate_on_a_tilted_low_passed_depth(
    name: str, box: Box, size: tuple[float, float]
) -> None:
    T = _camera(math.degrees(0.5729))
    truth = _render_depth(T, [box])
    cx, cy = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
    u, v = _px(T, (cx, cy, box[4]))
    s = _supports()
    L, W = sorted(size, reverse=True)
    got = []
    for seed in range(3):
        e = PT.object_extent(_a6(truth, T, seed=seed), MODEL, T, u, v, lambda X, Y, g: PT.top_at(s, X, Y, g))
        assert e["ok"], e
        got.append((e["length_m"], e["width_m"], math.hypot(e["centre_base"][0] - cx, e["centre_base"][1] - cy)))
    lm, wm, cm = (float(np.mean([g[k] for g in got])) for k in range(3))
    # base (dcc01a2) on the simulator's day-2 scene: the low box 11.9 x 6.6 cm, centre 2.0 cm off; the tall box
    # 10.8 x 8.3, 1.3 cm (the robot on day 2: the low box 14.7 x 7.1)
    assert lm == pytest.approx(L, abs=0.015) and wm == pytest.approx(W, abs=0.012), got
    assert cm < 0.012, got


def test_a_click_into_a_container_without_depth_is_its_floor_never_refused() -> None:
    """The integrator / tools (session 13): the shiny bin gives no depth (shiny) - a click INTO it (the carry's "over
    the basket", the empty hand's park) is the container's floor by geometry, as v2 answered; refusing it (s13_click
    before this fix: NO_DEPTH) would make every item undeliverable. The board / table beside it still refuse."""
    from r1mc3.world import Container, Pose

    T = _camera(math.degrees(0.5729))
    bin = Container("bin", Pose((0.35, -0.30, TABLE_Z)), size_outer=(0.35, 0.26, 0.045), wall_m=0.003, floor_m=0.003)
    depth = _render_depth(T, [])
    holed = np.full_like(depth, np.nan)  # no depth anywhere (a shiny bin, and for the check below the table too)
    u, v = _px(T, (0.35, -0.30, TABLE_Z + 0.003))
    r = PT.point(holed, MODEL, T, u, v, _supports(bin))
    assert r["ok"] and r["surface_id"] == "bin" and r["height_from"] == PT.CONTAINER_NO_DEPTH_LABEL, r
    assert r["point_base"][:2] == pytest.approx([0.35, -0.30], abs=0.003)
    ut, vt = _px(T, (0.30, 0.15, TABLE_Z))  # the table beside it: still refused
    assert PT.point(holed, MODEL, T, ut, vt, _supports(bin))["code_hint"] == "NO_DEPTH"
