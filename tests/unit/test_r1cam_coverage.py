"""The live S12 count (r1mc3.cam.coverage, session 10): a ChArUco frame is binned by distance x place x tilt, the
status says "hold still" until the cell has enough frames, then "next" with the first missing cell; a charuco
recording carries it in r1cam's status and its manifest. New on the base (a041731: no live count)."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from PIL import Image  # noqa: E402
from r1mc3.cam.coverage import CharucoCoverage  # noqa: E402

SPEC = {"cols": 6, "rows": 4, "square_mm": 90.0, "marker_mm": 68.0, "dict": "DICT_4X4_50"}


def _frame(u: float, v: float, z: float, tilt: str = "flat", f: float = 127.0) -> bytes:
    """The board drawn into a 544 x 448 frame by a pinhole homography: its centre at pixel fraction (u, v), at depth
    z, turned / tipped 30 deg (no lens distortion: enough for the binning)."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    b = cv2.aruco.CharucoBoard((6, 4), 0.09, 0.068, d)
    px_per_m = 2000.0
    img = b.generateImage((int(0.54 * px_per_m), int(0.36 * px_per_m)), marginSize=0)
    hw, hh = 0.27, 0.18
    a = math.radians(30.0)
    R = np.eye(3)
    if tilt == "turned":
        R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    elif tilt == "tipped":
        R = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    c = np.array([(u * 543 - 271.5) / f * z, (v * 447 - 223.5) / f * z, z])
    src, dst = [], []
    for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        P = c + R @ np.array([x, y, 0.0])
        dst.append([P[0] / P[2] * f + 271.5, P[1] / P[2] * f + 223.5])
        src.append([(x + hw) * px_per_m, (y + hh) * px_per_m])
    H = cv2.getPerspectiveTransform(np.asarray(src, np.float32), np.asarray(dst, np.float32))
    out = cv2.warpPerspective(img, H, (544, 448), borderValue=200)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_cells_hold_still_then_next() -> None:
    cov = CharucoCoverage(SPEC, need=3)
    for _ in range(2):
        cov.feed(_frame(0.5, 0.5, 0.45))
    s = cov.status()
    assert s["detected"] == 2 and s["say"].startswith("hold still (0.45 m, centre, flat): 2/3"), s
    cov.feed(_frame(0.5, 0.5, 0.45))
    s = cov.status()
    assert s["say"].startswith("next: 0.30 m, top-left, flat") and s["cells_done"] == 1 and s["cells"] == 81, s
    cov.feed(_frame(0.2, 0.2, 0.45, "flat"))
    cov.feed(_frame(0.5, 0.5, 0.3, "turned"))
    cov.feed(_frame(0.5, 0.5, 0.6, "tipped"))
    keys = set(cov.status()["counts"])
    assert {"0.45 m, centre, flat", "0.45 m, top-left, flat", "0.30 m, centre, turned ~30 deg left-right",
            "0.60 m, centre, tipped ~30 deg"} <= keys, keys  # fmt: skip
    blank = CharucoCoverage(SPEC)
    buf = io.BytesIO()
    Image.fromarray(np.full((448, 544), 128, np.uint8)).save(buf, format="JPEG")
    blank.feed(buf.getvalue())
    assert blank.status()["say"].startswith("no board in view")


def test_a_charuco_recording_reports_the_count(tmp_path: Path) -> None:
    import time

    from r1mc3.cam.recorder import CamRecorder
    from r1mc3.cam.sim import OnDemandSource

    frames = [_frame(0.5, 0.5, 0.45)] * 3
    src = OnDemandSource(lambda s: frames[0], streams=("left",))
    rec = CamRecorder(src, tmp_path, kind="charuco", params={"board": "poster_6x4_90"}, streams=("left",)).start()
    for _ in range(3):
        src.demand(["left"])
    t_end = time.monotonic() + 10.0
    while rec.coverage.frames < 3 and time.monotonic() < t_end:
        time.sleep(0.05)
    doc = rec.stop()
    assert doc["coverage"]["detected"] == 3 and "centre" in doc["coverage"]["say"]


def test_the_count_works_without_pillow_as_on_the_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session 12 (HW2 day 1: `frames 0` for the whole C3 recording): Pillow is a DEV dependency only - the robot's
    `--no-dev` install has none, and feed() imported it for every frame (the recorder swallowed the ImportError).
    Base 1cf3afd: ImportError here."""
    import sys

    jpeg = _frame(0.5, 0.5, 0.45)
    monkeypatch.setitem(sys.modules, "PIL", None)  # `from PIL import ...` raises ImportError, as on the robot
    cov = CharucoCoverage(SPEC)
    cov.feed(jpeg)
    st = cov.status()
    assert st["frames"] == 1 and st["detected"] == 1 and st["undecodable"] == 0
    cov.feed(b"\xff\xd8 not a jpeg \xff\xd9")
    assert cov.status()["undecodable"] == 1 and cov.status()["frames"] == 2
