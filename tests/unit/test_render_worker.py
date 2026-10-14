"""The Filament render worker (DESIGN D13): the export, the pose identity of the two compiles, the head chain on it.

The worker runs MuJoCo 3.15.0 / Python 3.12 in its own venv (render/, $R1_RENDER_VENV); the physics stays on 3.5.0.
Tests that need the worker skip when the render venv is absent (`scripts/ci.sh render` creates it)."""

from __future__ import annotations

import time

import mujoco
import numpy as np
import pytest
from r1sim import joints, lens, model, render_client, world

needs_render = pytest.mark.skipif(
    not render_client.available(), reason="no render venv (R1_RENDER_VENV; `scripts/ci.sh render` creates it)"
)


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", head_camera=True)


def random_qpos(m: mujoco.MjModel, n: int, seed: int = 0) -> np.ndarray:
    """n qpos rows: hinge / slide joints uniform in their range (+-pi if unlimited), free joints anywhere in a 2 m box
    with a random orientation, ball joints random - every body moves, nothing needs to be physical."""
    rng = np.random.default_rng(seed)
    out = np.tile(m.qpos0, (n, 1))
    for j in range(m.njnt):
        a, t = int(m.jnt_qposadr[j]), int(m.jnt_type[j])
        if t == mujoco.mjtJoint.mjJNT_FREE:
            out[:, a : a + 3] = rng.uniform(-1.0, 1.0, (n, 3)) + np.array([0.0, 0.0, 1.0])
            q = rng.normal(size=(n, 4))
            out[:, a + 3 : a + 7] = q / np.linalg.norm(q, axis=1, keepdims=True)
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            q = rng.normal(size=(n, 4))
            out[:, a : a + 4] = q / np.linalg.norm(q, axis=1, keepdims=True)
        else:
            lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
            out[:, a] = rng.uniform(lo, hi, n)
    return out


def test_world_keeps_its_spec_and_the_export_recompiles_the_same_model(w: world.World, tmp_path) -> None:
    """The exported MJCF compiles on THIS MuJoCo (3.5.0) to the same sizes; its placements are the physics model's to
    to_xml's 6 significant digits (the phones' fitted poses come back ~4e-7 m off), and after the worker's "sync" copy
    (render_client.apply_sync) the kinematics are identical."""
    p = render_client.export_world(w, tmp_path)
    m2 = mujoco.MjModel.from_xml_path(str(p))
    for f in ("nq", "nv", "nbody", "njnt", "ngeom", "ncam", "nlight", "nmesh", "ntex", "nmat"):
        assert getattr(m2, f) == getattr(w.model, f), f
    q = random_qpos(w.model, 5, seed=3)
    d1, d2 = mujoco.MjData(w.model), mujoco.MjData(m2)

    def worst() -> float:
        out = 0.0
        for qi in q:
            d1.qpos[:], d2.qpos[:] = qi, qi
            mujoco.mj_kinematics(w.model, d1)
            mujoco.mj_kinematics(m2, d2)
            out = max(out, float(np.abs(d1.xpos - d2.xpos).max()))
        return out

    before = worst()
    render_client.apply_sync(m2, w.model)
    after = worst()
    print(f"export -> 3.5.0 recompile: max |dxpos| {before:.1e} m (6-digit XML); after the sync {after:.1e}")
    assert before < 2e-6 and after == 0.0
    with pytest.raises(ValueError):
        render_client.export_world(object())


def _quat_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-body distance between unit quaternions (sign-insensitive)."""
    return np.asarray(np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1)))


@needs_render
def test_worker_puts_every_body_where_the_physics_does(w: world.World) -> None:
    """D13's condition: for the same qpos, MuJoCo 3.15.0 in the worker and 3.5.0 here give the same body poses (60
    random configurations of every joint + the stand) - so the pictures show what the physics did."""
    q = np.vstack([w.initial_qpos(), model.stand_qpos(w.model, joints.POSE_STAND), random_qpos(w.model, 60, seed=1)])
    with render_client.RenderWorker(w) as wk:
        assert wk.version == render_client.RENDER_MUJOCO_VERSION
        assert (wk.nq, wk.nbody) == (w.model.nq, w.model.nbody)
        assert wk.bodies == [w.model.body(i).name for i in range(w.model.nbody)]
        xpos, xquat = wk.body_poses(q)
    d = mujoco.MjData(w.model)
    dp, dq = [], []
    for k, qi in enumerate(q):
        d.qpos[:] = qi
        mujoco.mj_kinematics(w.model, d)
        dp.append(np.abs(xpos[k] - d.xpos).max())
        dq.append(_quat_dist(xquat[k], d.xquat).max())
    print(f"{len(q)} poses x {w.model.nbody} bodies: max |dxpos| {max(dp):.2e} m, max |dquat| {max(dq):.2e}")
    assert max(dp) <= 1e-9 and max(dq) <= 1e-9


def _dot_grid() -> tuple[mujoco.MjSpec, np.ndarray, lens.RenderFrame]:
    """Bright spheres on a grid in front of an OpenCV-frame camera at the origin (as test_lens), emissive so both
    renderers draw them white on black whatever their lighting."""
    M = lens.load_calibration().model
    fr = lens.render_frame(M)
    spec = mujoco.MjSpec()
    spec.visual.headlight.ambient = [1.0, 1.0, 1.0]
    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    lens.add_camera_at(spec, np.eye(4), fr, "cam")
    uu, vv = np.meshgrid(np.linspace(30, M.width - 31, 9), np.linspace(30, M.height - 31, 7))
    P = lens.unproject(M, uu.ravel(), vv.ravel(), np.full(uu.size, 1.0))
    spec.add_material(name="dot", rgba=[1, 1, 1, 1], emission=1.0)
    for k, p in enumerate(P):
        spec.worldbody.add_geom(
            name=f"dot{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0], pos=list(p), material="dot",
            contype=0, conaffinity=0,
        )  # fmt: skip
    return spec, P, fr


def _centroids(img: np.ndarray) -> np.ndarray:
    import cv2

    g = img.mean(axis=2).astype(np.float64)
    mask = (g > 40.0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for k in range(1, n):
        if not 3 <= stats[k, cv2.CC_STAT_AREA] <= 400:
            continue
        ys, xs = np.nonzero(lab == k)
        wt = g[ys, xs]
        out.append((float((xs * wt).sum() / wt.sum()), float((ys * wt).sum() / wt.sum())))
    return np.asarray(out)


def _nearest(found: np.ndarray, want: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.norm(found[None, :, :] - want[:, None, :], axis=2).min(axis=1))


@needs_render
def test_filament_head_chain_puts_points_where_the_camera_model_and_the_classic_renderer_do(tmp_path) -> None:
    """The worker's pinhole (frustum from the camera's fovy, square pixels) + our remap: each dot of a 63-point grid
    lands within 0.5 px of calib_head's projection and of the classic renderer's dot."""
    spec, P, fr = _dot_grid()
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    M = lens.load_calibration().model
    maps = lens.remap_maps(M, fr)
    with mujoco.Renderer(m, fr.height, fr.width) as r:
        r.update_scene(d, camera="cam")
        classic = lens.remap(np.asarray(r.render(), dtype=np.uint8).copy(), maps, prefilter_sigma=1.0)
    with render_client.RenderWorker(render_client.export_world(spec, tmp_path)) as wk:
        wide = wk.render(d.qpos, "cam", fr.width, fr.height)
    fil = lens.remap(wide, maps, prefilter_sigma=1.0)
    want = lens.project(M, P)
    f_fil, f_cls = _centroids(fil), _centroids(classic)
    e_model, e_classic = _nearest(f_fil, want), _nearest(f_fil, f_cls)
    print(f"filament vs model: median {np.median(e_model):.3f} max {e_model.max():.3f} px; "
          f"vs classic: median {np.median(e_classic):.3f} max {e_classic.max():.3f} px")  # fmt: skip
    assert len(f_fil) == len(P) == len(f_cls)
    assert e_model.max() < 0.5 and e_classic.max() < 0.5


@needs_render
def test_head_camera_filament_backend_gives_the_robot_frame(w: world.World) -> None:
    """HeadCamera(backend="filament"): 544 x 448 frames from the worker (the floor in front of the robot fills the lower
    image), the depth still the classic ground truth; the classic backend refuses nothing it did before."""
    d = mujoco.MjData(w.model)
    d.qpos[:] = w.initial_qpos()
    mujoco.mj_forward(w.model, d)
    with pytest.raises(ValueError):
        lens.HeadCamera(w.model, backend="filament")
    with render_client.RenderWorker(w) as wk, lens.HeadCamera(w.model, backend="filament", worker=wk) as hc:
        rgb = hc.rgb(d)
        t0 = time.perf_counter()
        for _ in range(5):
            hc.rgb(d)
        ms = (time.perf_counter() - t0) / 5 * 1000
        z = hc.depth_z(d)
    with lens.HeadCamera(w.model) as hc2:
        rgb2 = hc2.rgb(d)
    print(f"filament head frame {ms:.1f} ms (render + transfer + remap)")
    assert rgb.shape == rgb2.shape == (448, 544, 3) and rgb.dtype == np.uint8
    # the floor in front of the robot fills the lower middle: lit in both (the looks differ by design: PBR +
    # tone mapping vs the classic renderer; their brightness is compared with real frames in docs/sessions/s4)
    assert rgb[300:, 150:400].mean() > 90 and rgb2[300:, 150:400].mean() > 90
    assert rgb[300:, 150:400].std() < rgb.std() + 40
    assert np.count_nonzero(z) > 0.9 * z.size


@needs_render
def test_worker_reports_errors_and_keeps_serving(w: world.World) -> None:
    with render_client.RenderWorker(w) as wk:
        with pytest.raises(ValueError):
            wk.render(np.zeros(3), "head_cam", 64, 48)
        with pytest.raises(render_client.WorkerError, match="no camera"):
            wk.render(w.initial_qpos(), "no_such_camera", 64, 48)
        img = wk.render(w.initial_qpos(), "head_cam", 64, 48)
        assert img.shape == (48, 64, 3)
