"""r1mc3.measure.phone_calib (session 11): a still camera's lens model from photos of the printed ChArUco board -> the
`--intrinsics` file of the pre-flight's phone stills (measure.phone). Synthetic check: photos RENDERED through a known
pinhole + radial / tangential distortion (every output pixel undistorted -> its ray meets the board plane -> the
board image sampled), at a phone's 4:3 shape; the fit must give the camera back. Base (1e55bdd): no such module
(ModuleNotFoundError) - the runbook's D35 phone steps had no way to make the file."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from r1mc3.measure import phone_calib as PC  # noqa: E402
from r1mc3.service.schemas import PhoneIntrinsics  # noqa: E402

W, H = 1600, 1200  # a phone's 4:3 (the real stills are 4032 x 3024; the same maths at a test-friendly size)
K_TRUE = np.array([[1250.0, 0.0, 803.0], [0.0, 1250.0, 596.0], [0.0, 0.0, 1.0]])
D_TRUE = np.array([0.08, -0.15, 0.0005, -0.0003, 0.0])
PPM = 4000.0  # the board image's pixels per metre
MARGIN = 100


def _board_image() -> np.ndarray:
    board, _ = PC.board_and_detector()
    sq = PC.BOARD["square_mm"] / 1e3
    w, h = int(PC.BOARD["cols"] * sq * PPM) + 2 * MARGIN, int(PC.BOARD["rows"] * sq * PPM) + 2 * MARGIN
    return np.asarray(board.generateImage((w, h), marginSize=MARGIN), dtype=np.uint8)


def _render(img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """The board (z = 0 plane, OpenCV board frame: x right, y down) seen by K_TRUE / D_TRUE from (rvec, tvec)."""
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([uu, vv], -1).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-10)
    n = cv2.undistortPointsIter(pts, K_TRUE, D_TRUE, None, None, crit).reshape(-1, 2)
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec.reshape(3)
    d = (R.T @ np.c_[n, np.ones(len(n))].T).T
    s = -C[2] / d[:, 2]
    X = C + s[:, None] * d
    mx = (X[:, 0] * PPM + MARGIN - 0.5).reshape(H, W).astype(np.float32)
    my = (X[:, 1] * PPM + MARGIN - 0.5).reshape(H, W).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=200)
    return np.asarray(out, dtype=np.uint8)


def _views(n: int, seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """n photos: the board at 0.6-1.0 m, at different places in the frame (edges included), tilted up to ~35 deg."""
    img = _board_image()
    rng = np.random.default_rng(seed)
    centre = np.array([0.27, 0.18, 0.0])  # the board's middle in its own frame
    out = []
    for k in range(n):
        tilt = np.radians(rng.uniform(-35, 35, 2))
        R, _ = cv2.Rodrigues(np.array([tilt[0], tilt[1], rng.uniform(-0.3, 0.3)]))
        z = rng.uniform(0.6, 1.0)
        # aim the board's middle at a spread of image places (a 3 x 3 pattern + random)
        u = W * (0.2 + 0.3 * (k % 3)) + rng.uniform(-60, 60)
        v = H * (0.2 + 0.3 * ((k // 3) % 3)) + rng.uniform(-60, 60)
        ray = np.linalg.inv(K_TRUE) @ np.array([u, v, 1.0])
        t = ray / ray[2] * z - R @ centre
        rvec, _ = cv2.Rodrigues(R)
        out.append((f"photo_{k:02d}", _render(img, rvec, t)))
    return out


@pytest.fixture(scope="module")
def views() -> list[tuple[str, np.ndarray]]:
    return _views(12)


def test_the_fit_gives_the_rendering_camera_back(views: list[tuple[str, np.ndarray]]) -> None:
    res = PC.run(views)
    assert res["ok"], res["reason"]
    assert res["usable"] >= 10 and res["rms_px"] < 0.5
    K = np.asarray(res["intrinsics"]["K"])
    assert K[0, 0] == pytest.approx(1250.0, rel=0.005) and K[1, 1] == pytest.approx(K[0, 0], rel=1e-6)
    assert abs(K[0, 2] - 803.0) < 4.0 and abs(K[1, 2] - 596.0) < 4.0
    d = res["intrinsics"]["dist"]
    assert d[0] == pytest.approx(0.08, abs=0.02) and d[1] == pytest.approx(-0.15, abs=0.06) and d[4] == 0.0
    # what matters for the pre-flight: the pixels' rays relative to each other (each still's pose is fitted from the
    # table tags, which absorbs a common small rotation - e.g. the principal point traded against the poses: ~3 px
    # here): after the mean ray offset the fitted model is sub-pixel over the frame (median), ~2 px only at the extreme
    # corners 20 px from the edge, where the photos' board corners did not reach (extrapolation)
    grid = np.stack(np.meshgrid(np.linspace(20, W - 20, 9), np.linspace(20, H - 20, 7)), -1).reshape(-1, 1, 2)
    n_true = cv2.undistortPoints(grid, K_TRUE, D_TRUE).reshape(-1, 2)
    n_fit = cv2.undistortPoints(grid, K, np.asarray(d)).reshape(-1, 2)
    dn = n_true - n_fit
    e_px = np.linalg.norm(dn - dn.mean(axis=0), axis=1) * 1250.0  # px at the image's scale
    assert float(np.median(e_px)) < 0.6 and float(e_px.max()) < 2.5, e_px
    assert res["coverage"] >= 0.75 and "calibrated" in res["reason"]
    assert math.isclose(res["hfov_deg"], math.degrees(2 * math.atan(W / 2500.0)), abs_tol=0.5)
    # exactly measure.phone's intrinsics (PhoneIntrinsics refuses extra keys)
    PhoneIntrinsics(**res["intrinsics"])


def test_refusals_say_why(views: list[tuple[str, np.ndarray]]) -> None:
    few = PC.run(views[:3])
    assert not few["ok"] and "only 3 photos" in few["reason"]
    small = np.asarray(cv2.resize(views[0][1], (800, 600)), dtype=np.uint8)
    mixed = PC.run([*views[:6], ("small", small)])
    assert not mixed["ok"] and "different sizes" in mixed["reason"]
    blank = PC.run([("blank", np.full((H, W), 200, np.uint8))])
    assert not blank["ok"]


def test_the_command_writes_the_intrinsics_file(views: list[tuple[str, np.ndarray]], tmp_path: Path) -> None:
    from PIL import Image

    paths = []
    for name, g in views[:8]:
        p = tmp_path / f"{name}.png"
        Image.fromarray(g).save(p)
        paths.append(str(p))
    out = tmp_path / "phone.json"
    assert PC.main([*paths, "--out", str(out), "--report", str(tmp_path / "r.json")]) == 0
    spec = json.loads(out.read_text())
    assert set(spec) == {"K", "dist", "width", "height"} and (spec["width"], spec["height"]) == (W, H)
    assert json.loads((tmp_path / "r.json").read_text())["usable"] == 8
    assert PC.main([paths[0], "--out", str(tmp_path / "no.json")]) == 1 and not (tmp_path / "no.json").exists()
