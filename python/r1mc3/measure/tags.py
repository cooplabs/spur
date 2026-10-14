"""Fiducial tags: detection (AprilTag 36h11 through OpenCV's aruco module) and where a tag sits on an entity.

AprilTag 36h11 (8 x 8 cells incl. the black border; minimum Hamming distance 11: no false IDs in practice - the
measurement standard; the session-3 tag render test compared it with ArUco 4x4_50, docs/sim/RENDERING.md). A tag's
`size_m` is the edge of its outer BLACK square (OpenCV's marker length); the printed sheet adds a white quiet zone of
at least one cell around it.

The tag frame = OpenCV's marker frame: x along the marker's top row (left -> right as printed), y toward the top row,
z out of the printed face. Corners come back in OpenCV's order: top-left, top-right, bottom-right, bottom-left, i.e. at
(-h, h), (h, h), (h, -h), (-h, -h) in the tag frame (h = size / 2) - the order solvePnP's IPPE_SQUARE expects.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
FAMILY = "DICT_APRILTAG_36h11"


@dataclass(frozen=True)
class TagObs:
    """One detected tag in one image: its id and its 4 corner pixels (OpenCV order)."""

    tag_id: int
    corners_px: FloatArray  # 4 x 2

    @property
    def centre_px(self) -> FloatArray:
        """The diagonals' intersection (the projective centre, not the corners' mean)."""
        p0, p1, p2, p3 = self.corners_px
        d1, d2 = p2 - p0, p3 - p1
        st = np.linalg.solve(np.array([d1, -d2]).T, p1 - p0)
        return np.asarray(p0 + st[0] * d1)

    @property
    def edge_px(self) -> float:
        """The tag's mean edge length in the image (px): the resolution it was seen at."""
        c = self.corners_px
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


@dataclass(frozen=True)
class TagPlacement:
    """A tag on an entity: id, black-square edge (m) and the tag frame's pose in the entity frame (4 x 4)."""

    tag_id: int
    size_m: float
    T_entity_tag: FloatArray

    def corners_entity(self) -> FloatArray:
        """The 4 corners in the entity frame (OpenCV corner order)."""
        h = self.size_m / 2.0
        c = np.array([[-h, h, 0, 1], [h, h, 0, 1], [h, -h, 0, 1], [-h, -h, 0, 1.0]])
        return np.asarray((self.T_entity_tag @ c.T).T[:, :3])

    @property
    def centre_entity(self) -> FloatArray:
        return np.asarray(self.T_entity_tag[:3, 3])


def rot_z(yaw: float) -> FloatArray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def flat_tag(tag_id: int, size_m: float, x: float, y: float, z: float = 0.0, up_yaw_deg: float = 0.0) -> TagPlacement:
    """A tag lying flat on a horizontal face (the face's frame: x forward, y left, z up), printed face up, its top row
    pointing along the face's yaw `up_yaw_deg` (0 = the marker's up along +x: read upright from a viewer at -x, e.g. the
    robot looking across a table)."""
    R = rot_z(math.radians(up_yaw_deg)) @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return TagPlacement(int(tag_id), float(size_m), T)


APRILTAG_CORNER_SHIFT_PX = 0.5  # OpenCV's APRILTAG corner refinement reports pixel-EDGE coordinates: corners come back
# +0.5 px from the pixel-centre convention of everything else (measured, session 9: exactly 0.5 on a pixel grid,
# 0.45-0.49 on warped tags) - subtracted here


def _detect(gray: npt.NDArray[np.uint8], family: str, refine: str) -> dict[int, list[FloatArray]]:
    import cv2

    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = {
        "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
        "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
        "none": cv2.aruco.CORNER_REFINE_NONE,
    }[refine]
    if refine == "subpix":
        p.cornerRefinementWinSize = 3  # small: a far 36h11 tag is only ~15-25 px across (render test)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, family)), p)
    corners, ids_, _ = det.detectMarkers(np.ascontiguousarray(gray))
    found: object = ids_  # None when nothing is found (the stubs say ndarray)
    out: dict[int, list[FloatArray]] = {}
    if found is None:
        return out
    shift = APRILTAG_CORNER_SHIFT_PX if refine == "apriltag" else 0.0
    for i, c in zip(np.asarray(found).ravel(), corners, strict=True):
        out.setdefault(int(i), []).append(np.asarray(c, dtype=np.float64).reshape(4, 2) - shift)
    return out


def detect_tags(
    gray: npt.NDArray[np.uint8], *, family: str = FAMILY, refine: str = "auto", ids: Iterable[int] | None = None
) -> dict[int, TagObs]:
    """Every tag of `family` in a grey image -> {id: TagObs}. refine: "auto" (default: the AprilTag edge refinement -
    max corner error 0.14-0.32 px on 18-80 px tags vs 0.65-1.0 px for cv2.cornerSubPix, which suits saddle corners,
    not a black square's outer corner - and, for a tag it rejects (it drops some tags under ~18 px), the sub-pixel
    method), "apriltag", "subpix" or "none". `ids`: keep only these. A tag id seen twice (impossible on one print; a
    reflection, a second sheet) is dropped - it cannot be told apart."""
    if refine == "auto":
        found = _detect(gray, family, "apriltag")
        for k, v in _detect(gray, family, "subpix").items():
            found.setdefault(k, v)
    else:
        found = _detect(gray, family, refine)
    keep = None if ids is None else {int(i) for i in ids}
    return {k: TagObs(k, v[0]) for k, v in found.items() if len(v) == 1 and (keep is None or k in keep)}
