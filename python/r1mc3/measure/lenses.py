"""Camera models for measurement: pixels <-> normalised image coordinates (x / z, y / z; OpenCV camera frame: x right,
y down, z forward; the pixel CENTRE at the integer index).

`HeadLens` = the R1 head camera's calibrated model (pinhole + k1 or equidistant; the v2 / calib_head.py format,
r1sim.lens - the same wheel; the camera model moves next to r1cam / pointing in session 10). `OpenCVLens` = a phone or
any camera calibrated with OpenCV (K + the k1 k2 p1 p2 k3 distortion), e.g. from a ChArUco calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class Lens(Protocol):
    """What the measurement needs from a camera model."""

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def f_px(self) -> float:
        """The central focal length (px): turns normalised residuals into pixels."""
        ...

    def normalise(self, px: npt.ArrayLike) -> FloatArray:
        """N x 2 pixels -> N x 2 normalised coordinates (the undistorted ray's x / z, y / z)."""
        ...

    def project(self, P_cam: npt.ArrayLike) -> FloatArray:
        """N x 3 camera-frame points -> N x 2 pixels (NaN behind the camera)."""
        ...


@dataclass(frozen=True)
class HeadLens:
    """The head camera's calibrated model (r1sim.lens.CameraModel: pinhole + k1, or equidistant)."""

    model: Any  # r1sim.lens.CameraModel

    @property
    def width(self) -> int:
        return int(self.model.width)

    @property
    def height(self) -> int:
        return int(self.model.height)

    @property
    def f_px(self) -> float:
        return float(self.model.f)

    @classmethod
    def load(cls, path: str | Path | None = None) -> HeadLens:
        from r1sim import lens

        cal = lens.load_calibration() if path is None else lens.load_calibration(path)
        return cls(cal.model)

    def normalise(self, px: npt.ArrayLike) -> FloatArray:
        from r1sim import lens

        p = np.atleast_2d(np.asarray(px, dtype=np.float64))
        return np.ascontiguousarray(lens.normalised_rays(self.model, p[:, 0], p[:, 1])[:, :2])

    def project(self, P_cam: npt.ArrayLike) -> FloatArray:
        from r1sim import lens

        return lens.project(self.model, P_cam)


@dataclass(frozen=True)
class OpenCVLens:
    """K (3 x 3) + OpenCV distortion (k1, k2, p1, p2[, k3]) at an image size."""

    K: FloatArray
    dist: FloatArray = field(default_factory=lambda: np.zeros(5))
    width: int = 0
    height: int = 0

    @property
    def f_px(self) -> float:
        return float(0.5 * (self.K[0, 0] + self.K[1, 1]))

    @classmethod
    def pinhole(cls, f_px: float, width: int, height: int, dist: npt.ArrayLike | None = None) -> OpenCVLens:
        """A centred pinhole (square pixels) - e.g. a phone before its calibration, or a simulated camera."""
        K = np.array([[f_px, 0.0, (width - 1) / 2.0], [0.0, f_px, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
        d = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
        return cls(K, d, width, height)

    def normalise(self, px: npt.ArrayLike) -> FloatArray:
        import cv2

        p = np.atleast_2d(np.asarray(px, dtype=np.float64)).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
        eye = np.eye(3)
        out = cv2.undistortPointsIter(p, self.K, self.dist, eye, eye, crit)  # R = P = I: normalised output
        return np.asarray(out, dtype=np.float64).reshape(-1, 2)

    def project(self, P_cam: npt.ArrayLike) -> FloatArray:
        import cv2

        P = np.atleast_2d(np.asarray(P_cam, dtype=np.float64))
        out = np.full((len(P), 2), np.nan)
        ok = P[:, 2] > 1e-9
        if ok.any():
            px, _ = cv2.projectPoints(P[ok].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
            out[ok] = np.asarray(px, dtype=np.float64).reshape(-1, 2)
        return out


@dataclass(frozen=True)
class ModelLens:
    """The head camera's calibration as r1d holds it (calib_head.py's model dict: lens, f, cx, cy, k1, depth -
    `r1mc3.cam.observe.HeadCalibration.model`, session 10) through the SAME projection pointing uses
    (`r1mc3.cam.pointing`), so a measurement and a pointed pixel use one lens model."""

    model: Any  # Mapping[str, Any]
    width: int = 544
    height: int = 448

    @property
    def f_px(self) -> float:
        return float(self.model["f"])

    def normalise(self, px: npt.ArrayLike) -> FloatArray:
        from r1mc3.cam.pointing import unproject

        p = np.atleast_2d(np.asarray(px, dtype=np.float64))
        r = unproject(self.model, p[:, 0], p[:, 1], np.ones(len(p)))
        return np.ascontiguousarray(r[:, :2] / r[:, 2:3])

    def project(self, P_cam: npt.ArrayLike) -> FloatArray:
        from r1mc3.cam.pointing import project

        return project(self.model, P_cam)
