"""A still camera's lens model from photos of the printed ChArUco board (session 11): the `--intrinsics` file the
measurement words take for a phone still (`measure.phone`; DESIGN §8.2: the top phone "calibrated once with the
ChArUco board", docs/architecture/MEASUREMENT.md).

    python -m r1mc3.measure.phone_calib still_01.jpg still_02.jpg ... --out phone.json [--report report.json]

Input: 10-20 photos of the board taken with the SAME camera at the SAME resolution as the measurement stills (a phone:
full resolution, 4:3, the same lens / zoom, no crop), the board filling a good part of the frame at different
places - the corners and edges of the image too - and tilted up to ~40 deg. Output: `phone.json` = exactly what
`measure.phone`'s `intrinsics` accepts - `{"K": 3 x 3, "dist": [k1, k2, p1, p2, k3], "width", "height"}` - and a
report (stdout / `--report`): the reprojection RMS (px), per photo, how much of the image the corners covered, and a
verdict in words. Refused (exit 1, nothing written) with fewer than MIN_VIEWS usable photos, photos of different
sizes, or an RMS above MAX_RMS_PX.

The board (docs/hardware/CHARUCO.md, r1mc3.cam.coverage.BOARDS): 6 x 4 squares of 90.0 mm, 68.0 mm markers,
OpenCV DICT_4X4_50 by default; another printed board via --cols / --rows / --square-mm / --marker-mm / --dict. The
model: OpenCV's calibrateCamera with square pixels (a phone) and k3 fixed at 0 (10-20 photos do not pin a 6th-order
term; `--k3` frees it). OpenCV is imported lazily (the `perception` extra), as the rest of r1mc3.measure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

BOARD: dict[str, Any] = {
    "cols": 6,
    "rows": 4,
    "square_mm": 90.0,
    "marker_mm": 68.0,
    "dict": "DICT_4X4_50",
}  # CHARUCO.md, printed
MIN_CORNERS = 6  # per photo: fewer gives no useful pose (a 6x4 board has 15 inner corners)
MIN_VIEWS = 6  # usable photos (OpenCV needs >= 3; 6+ for the distortion terms to be meaningful)
MAX_RMS_PX = 1.5  # above this the photos (blur, a bent board, a mixed resolution) are not a calibration
COVER_GRID = 4  # the image split 4 x 4 for the coverage number


@dataclass(frozen=True)
class Detection:
    name: str
    obj: FloatArray  # N x 3 board points (m)
    px: FloatArray  # N x 2 pixels


def board_and_detector(spec: Mapping[str, Any] = BOARD) -> tuple[Any, Any]:
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, str(spec["dict"])))
    sq, mk = float(spec["square_mm"]) / 1e3, float(spec["marker_mm"]) / 1e3
    board = cv2.aruco.CharucoBoard((int(spec["cols"]), int(spec["rows"])), sq, mk, d)
    return board, cv2.aruco.CharucoDetector(board)


def detect(gray: npt.NDArray[np.uint8], board: Any, detector: Any, name: str = "") -> Detection | None:
    """The board's inner corners in one grey image (None below MIN_CORNERS)."""
    corners, ids, _mc, _mi = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    obj = np.asarray(board.getChessboardCorners(), dtype=np.float64)[ids]
    return Detection(name, obj, np.asarray(corners, dtype=np.float64).reshape(-1, 2))


def coverage(dets: Sequence[Detection], width: int, height: int, grid: int = COVER_GRID) -> float:
    """The fraction of the image's grid x grid cells that hold at least one detected corner (1.0 = the whole frame)."""
    cells: set[tuple[int, int]] = set()
    for d in dets:
        for u, v in d.px:
            cells.add((min(grid - 1, int(grid * u / width)), min(grid - 1, int(grid * v / height))))
    return len(cells) / float(grid * grid)


def calibrate(dets: Sequence[Detection], width: int, height: int, *, k3: bool = False) -> dict[str, Any]:
    """OpenCV's calibrateCamera (square pixels; k3 fixed unless asked) -> K, dist, RMS, per-view RMS."""
    import cv2

    obj = [d.obj.astype(np.float32).reshape(-1, 1, 3) for d in dets]
    img = [d.px.astype(np.float32).reshape(-1, 1, 2) for d in dets]
    f0 = 0.8 * max(width, height)  # a phone's main camera: ~26 mm equivalent (f ~ 0.75-0.85 x the long side)
    K0 = np.array([[f0, 0.0, (width - 1) / 2.0], [0.0, f0, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | (0 if k3 else cv2.CALIB_FIX_K3)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, (width, height), K0, np.zeros(5), flags=flags)
    per_view = []
    for d, o, rv, tv in zip(dets, obj, rvecs, tvecs, strict=True):
        p, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per_view.append(
            {"photo": d.name, "corners": len(d.px), "rms_px": round(float(np.sqrt(np.mean(np.sum(
                (p.reshape(-1, 2) - d.px) ** 2, axis=1)))), 3),
             "distance_m": round(float(np.linalg.norm(tv)), 3)}
        )  # fmt: skip
    return {"rms_px": float(rms), "K": np.asarray(K, dtype=np.float64), "dist": np.asarray(dist, np.float64).ravel(),
            "per_view": per_view}  # fmt: skip


def intrinsics_file(K: FloatArray, dist: FloatArray, width: int, height: int) -> dict[str, Any]:
    """Exactly `measure.phone`'s `intrinsics` (r1mc3.service.schemas.PhoneIntrinsics: extra keys are refused)."""
    d = [float(v) for v in np.asarray(dist, dtype=np.float64).ravel()[:5]]
    d += [0.0] * (5 - len(d))
    return {"K": [[float(v) for v in row] for row in np.asarray(K, dtype=np.float64)], "dist": d,
            "width": int(width), "height": int(height)}  # fmt: skip


def run(
    images: Sequence[tuple[str, npt.NDArray[np.uint8]]], *, spec: Mapping[str, Any] = BOARD, k3: bool = False
) -> dict[str, Any]:
    """Grey images (name, H x W) -> {"ok", "reason", "intrinsics" (when ok), the report}."""
    if not images:
        return {"ok": False, "reason": "no photos"}
    sizes = {(int(g.shape[1]), int(g.shape[0])) for _n, g in images}
    if len(sizes) != 1:
        return {"ok": False, "reason": f"the photos have different sizes {sorted(sizes)}: use one camera, one "
                "resolution (the measurement stills' - a phone: full resolution 4:3)"}  # fmt: skip
    ((width, height),) = sizes
    board, det = board_and_detector(spec)
    dets, skipped = [], []
    for name, g in images:
        d = detect(g, board, det, name)
        if d is None:
            skipped.append(name)
        else:
            dets.append(d)
    out: dict[str, Any] = {"width": width, "height": height, "board": dict(spec), "photos": len(images),
                           "usable": len(dets), "skipped": skipped}  # fmt: skip
    if len(dets) < MIN_VIEWS:
        return {**out, "ok": False, "reason": f"only {len(dets)} photos show the board well enough (need "
                f"{MIN_VIEWS}; >= {MIN_CORNERS} corners each) - take more, the board sharp and filling more of the "
                "frame"}  # fmt: skip
    cal = calibrate(dets, width, height, k3=k3)
    cov = coverage(dets, width, height)
    K = cal["K"]
    out.update(
        rms_px=round(cal["rms_px"], 3), per_view=cal["per_view"], coverage=round(cov, 2),
        f_px=round(float(K[0, 0]), 2), cx=round(float(K[0, 2]), 2), cy=round(float(K[1, 2]), 2),
        dist=[round(float(v), 5) for v in cal["dist"][:5]],
        hfov_deg=round(math.degrees(2 * math.atan(width / (2 * float(K[0, 0])))), 1),
    )  # fmt: skip
    if cal["rms_px"] > MAX_RMS_PX:
        return {**out, "ok": False, "reason": f"reprojection RMS {cal['rms_px']:.2f} px > {MAX_RMS_PX} px: blurred "
                "photos, a board that is not flat, or photos from different cameras / zooms"}  # fmt: skip
    words = f"calibrated: RMS {cal['rms_px']:.2f} px over {len(dets)} photos, f {K[0, 0]:.1f} px"
    if cov < 0.75:
        words += f"; the corners covered only {100 * cov:.0f} % of the image - add photos with the board near the edges"
    return {**out, "ok": True, "reason": words, "intrinsics": intrinsics_file(K, cal["dist"], width, height)}


def _gray(path: Path) -> npt.NDArray[np.uint8]:
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(path))  # a phone's EXIF rotation: the pixels as the camera saw them
    return np.asarray(im.convert("L"), dtype=np.uint8)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1mc3.measure.phone_calib", description=__doc__.splitlines()[0])
    ap.add_argument("photos", nargs="+", help="the board photos (JPEG / PNG), one camera, one resolution")
    ap.add_argument("--out", required=True, help="the intrinsics file to write (measure.phone's --intrinsics)")
    ap.add_argument("--report", help="also write the full report (JSON)")
    ap.add_argument("--cols", type=int, default=BOARD["cols"])
    ap.add_argument("--rows", type=int, default=BOARD["rows"])
    ap.add_argument("--square-mm", type=float, default=BOARD["square_mm"])
    ap.add_argument("--marker-mm", type=float, default=BOARD["marker_mm"])
    ap.add_argument("--dict", default=BOARD["dict"])
    ap.add_argument("--k3", action="store_true", help="also fit k3 (wide lenses, many photos)")
    a = ap.parse_args(argv)
    spec = {"cols": a.cols, "rows": a.rows, "square_mm": a.square_mm, "marker_mm": a.marker_mm, "dict": a.dict}
    try:
        res = run([(str(p), _gray(Path(p))) for p in a.photos], spec=spec, k3=a.k3)
    except ImportError as e:
        print(json.dumps({"ok": False, "reason": f"needs OpenCV (the perception extra): {e}"}))
        return 1
    report = {k: v for k, v in res.items() if k != "intrinsics"}
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if not res["ok"]:
        print(f"REFUSED: {res['reason']}", file=sys.stderr)
        return 1
    Path(a.out).write_text(json.dumps(res["intrinsics"], indent=1))
    print(f"{res['reason']} -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
