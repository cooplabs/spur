"""The simulated head camera (r1sim.lens): the calibrated camera model, the wide-pinhole render and the remap.

Data: tests/assets/render/s3_cal/ (the 4 Oct 2026 calibration capture: calibration.json, marks_used.txt, the pose keys
of head.json; README there).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from r1sim import lens

ASSET = Path(__file__).resolve().parents[1] / "assets" / "render" / "s3_cal"
# the 4 Oct file as calib_head.py wrote it (its mount pairs with the capture's v2-logged T_base_site); session 11: the
# simulated camera's own mount is the v3-frame re-fit (lens.CALIBRATION_FILE) - the lens model is the same in both
CAL = lens.load_calibration(lens.CALIBRATION_FILE_V2_FRAME)
M = CAL.model


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    a, b = u - u0, v - v0
    return np.asarray(
        img[v0, u0] * (1 - a) * (1 - b)
        + img[v0, u0 + 1] * a * (1 - b)
        + img[v0 + 1, u0] * (1 - a) * b
        + img[v0 + 1, u0 + 1] * a * b
    )


def _points_in_view(n: int, seed: int = 0) -> np.ndarray:
    """Camera-frame points (OpenCV) whose pixels fall inside the image, 0.4-1.5 m away."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(5, M.width - 6, n)
    v = rng.uniform(5, M.height - 6, n)
    z = rng.uniform(0.4, 1.5, n)
    return lens.unproject(M, u, v, z)


def test_the_repo_copy_is_the_calibration() -> None:
    assert lens.CALIBRATION_FILE_V2_FRAME.read_bytes() == (ASSET / "calibration.json").read_bytes()
    assert (M.lens, M.width, M.height) == ("pinhole", 544, 448)
    assert M.f == pytest.approx(126.938, abs=1e-3) and M.k1 == pytest.approx(0.02487, abs=1e-5)


def test_the_simulated_camera_sits_where_the_robots_calibration_file_says() -> None:
    """Session 11 (the user's decision, 12 Oct ~03:05): the simulator's head camera = the head_camera site @ the mount
    of models/head_camera/calibration_s3_cal_20261004_v3frame.json - the file the robot installs as
    /etc/r1mc3/head_camera.json - so r1d in the sim and on the robot read the same file. Base (1e55bdd): the sim
    rendered through the 4 Oct mount, 1.08 deg / ~20 mm (at the camera) from the robot's file through v3's pose."""
    from r1sim import joints, model, world

    assert lens.CALIBRATION_FILE.name == "calibration_s3_cal_20261004_v3frame.json"
    robot = json.loads(lens.CALIBRATION_FILE.read_text())
    assert "v3" in robot["frame"]
    v3 = lens.load_calibration()
    assert v3.model == M  # the lens model is unchanged; only the mount differs
    w = world.build_world("real", visual=False, with_scene=False, head_camera=True)
    d = mujoco.MjData(w.model)
    pose = np.asarray(joints.POSE_STAND, dtype=np.float64).copy()
    pose[joints.IDX["head_pitch"]] = 0.6  # the head pitched toward the table
    d.qpos[:] = model.stand_qpos(w.model, pose)
    mujoco.mj_forward(w.model, d)
    sid = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_SITE, lens.HEAD_CAMERA_SITE)
    T_ws = np.eye(4)
    T_ws[:3, :3], T_ws[:3, 3] = d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid]
    T_cam = lens.T_world_cam(w.model, d)
    want = T_ws @ np.asarray(robot["T_site_cam"], dtype=np.float64)
    assert np.allclose(T_cam[:3, 3], want[:3, 3], atol=1e-6)  # the MuJoCo camera's float32 frame: ~1e-7
    assert np.allclose(T_cam[:3, :3], want[:3, :3], atol=1e-5)
    old = T_ws @ CAL.T_site_cam
    dR = old[:3, :3].T @ T_cam[:3, :3]
    assert np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) > 0.9  # the 4 Oct mount is ~1.1 deg away


def test_project_inverts_unproject() -> None:
    P = _points_in_view(500)
    uv = lens.project(M, P)
    back = lens.unproject(M, uv[:, 0], uv[:, 1], P[:, 2])
    assert np.abs(back - P).max() < 1e-9


def test_render_frame_covers_every_output_ray() -> None:
    fr = lens.render_frame(M)
    assert fr.f == pytest.approx(2.0 * M.f)
    mx, my = lens.remap_maps(M, fr)
    assert mx.shape == (M.height, M.width)
    assert mx.min() >= 1.0 and my.min() >= 1.0 and mx.max() <= fr.width - 2.0 and my.max() <= fr.height - 2.0
    # the size DESIGN §12.5 planned ("~970 x 825"): within 10 %
    assert 0.9 * 970 <= fr.width <= 1.1 * 1040 and 0.9 * 825 <= fr.height <= 1.15 * 825


def test_remap_maps_agree_with_the_camera_model() -> None:
    """For points in view: the map, sampled at the point's output pixel (project), gives the point's render pixel."""
    fr = lens.render_frame(M)
    mx, my = lens.remap_maps(M, fr)
    P = _points_in_view(2000, seed=1)
    uv = lens.project(M, P)
    want = fr.pixel(P[:, :2] / P[:, 2:3])
    got = np.c_[
        _bilinear(mx.astype(np.float64), uv[:, 0], uv[:, 1]), _bilinear(my.astype(np.float64), uv[:, 0], uv[:, 1])
    ]
    assert np.abs(got - want).max() < 0.05  # render px (float32 maps + bilinear over a smooth field)


def _blob_centroids(img: np.ndarray, *, dark: bool) -> np.ndarray:
    import cv2

    g = img.mean(axis=2) if img.ndim == 3 else img
    w = (255.0 - g) if dark else g.astype(np.float64)
    mask = (w > 40.0).astype(np.uint8)  # dots on a uniform background (black / white): an absolute threshold
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for k in range(1, n):
        if not 3 <= stats[k, cv2.CC_STAT_AREA] <= 400:  # a dot, not the background beyond the table
            continue
        ys, xs = np.nonzero(lab == k)
        wt = w[ys, xs]
        out.append((float((xs * wt).sum() / wt.sum()), float((ys * wt).sum() / wt.sum())))
    return np.asarray(out)


def _match(found: np.ndarray, want: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(found[None, :, :] - want[:, None, :], axis=2)
    return np.asarray(d.min(axis=1))


def test_rendered_point_grid_lands_where_the_camera_model_says() -> None:
    """Small bright spheres on a grid in front of the camera, rendered (wide pinhole) and remapped: each blob's centroid
    in the 544 x 448 frame lies within 0.5 px of calib_head's / v2's projection of the sphere's centre."""
    fr = lens.render_frame(M)
    spec = mujoco.MjSpec()
    spec.visual.headlight.ambient = [1.0, 1.0, 1.0]
    T = np.eye(4)  # camera at the origin, OpenCV frame = world frame
    lens.add_camera_at(spec, T, fr, "cam")
    uu, vv = np.meshgrid(np.linspace(30, M.width - 31, 9), np.linspace(30, M.height - 31, 7))
    P = lens.unproject(M, uu.ravel(), vv.ravel(), np.full(uu.size, 1.0))
    for k, p in enumerate(P):
        spec.worldbody.add_geom(
            name=f"dot{k}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.015, 0, 0],
            pos=list(p),
            rgba=[1, 1, 1, 1],
            contype=0,
            conaffinity=0,
        )
    m = spec.compile()
    m.vis.headlight.diffuse[:] = 0.0
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    with mujoco.Renderer(m, fr.height, fr.width) as r:
        r.update_scene(d, camera="cam")
        wide = np.asarray(r.render(), dtype=np.uint8).copy()
    out = lens.remap(wide, lens.remap_maps(M, fr), prefilter_sigma=1.0)
    found = _blob_centroids(out, dark=False)
    want = lens.project(M, P)
    err = _match(found, want)
    print(f"grid of {len(P)}: median {np.median(err):.3f} px, max {err.max():.3f} px")
    assert len(found) == len(P)
    assert np.median(err) < 0.25 and err.max() < 0.5


def _marks() -> list[tuple[tuple[float, float, float], tuple[float, float]]]:
    out = []
    for part in (ASSET / "marks_used.txt").read_text().strip().split(";"):
        a, b = part.strip().split(":")
        xyz = [float(v) for v in a.split(",")] + [0.0]
        out.append(((xyz[0], xyz[1], xyz[2]), (float(b.split(",")[0]), float(b.split(",")[1]))))
    return out


def test_table_marks_render_where_the_calibration_puts_them() -> None:
    """The 4 Oct calibration capture rebuilt: the camera at T_base_site @ T_site_cam (the logged site pose and the
    fitted mount), dark dots at the 16 taped marks on a white table (table cm -> base: near edge 0.08 m, top 0.751 m,
    calib_head's lab defaults). The rendered dots land within 0.5 px of the camera model's projection; their distance
    to the operator's clicked pixels is the calibration's own residual (reported, ~1-3 px)."""
    hp = json.loads((ASSET / "head_pose.json").read_text())
    T_base_cam = np.asarray(hp["T_base_site"]) @ CAL.T_site_cam
    edge_x, table_h = 0.08, 0.751
    marks = _marks()
    assert len(marks) == 16
    fr = lens.render_frame(M)
    spec = mujoco.MjSpec()
    spec.visual.headlight.ambient = [1.0, 1.0, 1.0]
    lens.add_camera_at(spec, T_base_cam, fr, "cam")
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.36, 0.61, 0.01],
        pos=[edge_x + 0.36, 0, table_h - 0.01],
        rgba=[1, 1, 1, 1],
    )
    world_pts = []
    for k, ((x, y, z), _) in enumerate(marks):
        p = [edge_x + x / 100.0, y / 100.0, table_h + z / 100.0]
        world_pts.append(p)
        if z > 0:  # a raised mark: a small post up to the mark (the real one sat on a box)
            spec.worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.012, 0.012, z / 200.0],
                pos=[p[0], p[1], table_h + z / 200.0 - 0.0002],
                rgba=[1, 1, 1, 1],
            )
        spec.worldbody.add_geom(
            name=f"mark{k}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.008, 0.0002, 0],
            pos=[p[0], p[1], p[2] + 0.0002],
            rgba=[0, 0, 0, 1],
        )
    m = spec.compile()
    m.vis.headlight.diffuse[:] = 0.0
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    with mujoco.Renderer(m, fr.height, fr.width) as r:
        r.update_scene(d, camera="cam")
        wide = np.asarray(r.render(), dtype=np.uint8).copy()
    out = lens.remap(wide, lens.remap_maps(M, fr), prefilter_sigma=1.0)
    Pc = (np.linalg.inv(T_base_cam) @ np.c_[np.asarray(world_pts), np.ones(len(world_pts))].T).T[:, :3]
    want = lens.project(M, Pc)
    found = _blob_centroids(out, dark=True)
    err = _match(found, want)
    clicked = np.asarray([c for _, c in marks])
    resid = np.linalg.norm(want - clicked, axis=1)
    print(
        f"render vs model: median {np.median(err):.3f} max {err.max():.3f} px; model vs clicked (the calibration's "
        f"residual): median {np.median(resid):.2f} max {resid.max():.2f} px"
    )
    assert np.median(err) < 0.3 and err.max() < 0.5
    assert np.median(resid) < 3.0  # the fit put the marks where they were clicked (16 marks, median 3.3 mm)


def test_the_head_chain_darkens_toward_the_edge_like_the_real_lens() -> None:
    """Session 4 (A13): the real head frames darken toward the image edge (measured real / sim over the 14 paired A13
    frames: 0.94 at 0.675 r_max, 0.76 at 0.875, 0.56 at 0.975); the pinhole render + remap had no vignetting, so the
    sim pictures' bottom rows matched the grey pad of a square-padded frame. The gain is 1 at the principal point and is
    applied by HeadCamera.rgb (vignetting=False keeps the pure geometry)."""
    from r1sim import lens

    cm = lens.load_calibration().model
    g = lens.vignetting_gain(cm)
    assert g.shape == (cm.height, cm.width)
    assert g[round(cm.cy), round(cm.cx)] == pytest.approx(1.0, abs=1e-6)
    assert 0.45 < float(g[-1, 0]) < 0.8 and 0.45 < float(g[-1, -1]) < 0.8  # the bottom corners (0.63 / 0.50)
    assert float(g[round(cm.cy), round(cm.cx) + 100]) > 0.99  # flat in the middle
