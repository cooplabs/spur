"""Offscreen rendering to PNG (no extra dependencies): free or named cameras, tiled side by side.

Works headless on macOS from the main thread with MuJoCo's default GL backend (no MUJOCO_GL setting needed; checked
on the session-1 Mac, MuJoCo 3.5.0). On Linux CI use MUJOCO_GL=egl or osmesa.

World frame = the base frame (r1sim.model): origin on the floor under the middle of the feet, x forward (toward the
table), y left, z up. The default views frame the robot standing at a table in front of it.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]


def png_bytes(rgb: npt.ArrayLike) -> bytes:
    """Encode an (H, W, 3) or (H, W, 4) uint8 image as PNG (8-bit RGB / RGBA, no filtering)."""
    img = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"expected (H, W, 3|4) uint8, got {img.shape}")
    h, w, c = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))  # filter type 0 per scanline

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    color_type = 2 if c == 3 else 6
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def write_png(path: str | Path, rgb: npt.ArrayLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(png_bytes(rgb))
    return p


@dataclass(frozen=True)
class View:
    """A free camera (lookat, distance, azimuth, elevation in MuJoCo's convention: azimuth about +z in degrees, 0 =
    looking along +x; negative elevation looks down) or, if `camera` is set, a named model camera."""

    name: str
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.8)
    distance: float = 2.5
    azimuth: float = 0.0
    elevation: float = -20.0
    camera: str | None = None
    fovy: float | None = None  # free camera's vertical field of view (deg); None = the model's visual/global fovy

    def mjv_camera(self, m: Any) -> Any:
        cam = mujoco.MjvCamera()
        if self.camera is not None:
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
            if cid < 0:
                raise KeyError(f"no camera named {self.camera!r}")
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cid
            return cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.lookat
        cam.distance = self.distance
        cam.azimuth = self.azimuth
        cam.elevation = self.elevation
        return cam


# The robot stands at the origin facing +x; the views were set for a table close in front of it (its near edge ~8 cm in
# front of the feet, the top ~0.75 m high; a panel ~27 cm out, a container at y = -0.45, the robot's right).
VIEW_BEHIND_LEFT = View("behind_left", lookat=(0.40, -0.05, 0.76), distance=2.1, azimuth=-50.0, elevation=-38.0)
VIEW_FRONT = View("front", lookat=(0.25, 0.0, 0.85), distance=2.4, azimuth=180.0, elevation=-18.0)
VIEW_SIDE = View("side_right", lookat=(0.25, 0.0, 0.8), distance=2.4, azimuth=90.0, elevation=-12.0)
VIEW_TOP = View("top", lookat=(0.40, -0.10, 0.75), distance=2.2, azimuth=0.0, elevation=-85.0)  # robot at the bottom
# A scene with a room (walls behind the table and on its left) closes two sides: the default views look in from the
# open sides (behind-right and right). VIEW_FRONT / VIEW_BEHIND_LEFT only work without such a room.
VIEW_BEHIND_RIGHT = View("behind_right", lookat=(0.40, 0.0, 0.80), distance=2.3, azimuth=37.0, elevation=-26.0)
DEFAULT_VIEWS: tuple[View, ...] = (VIEW_BEHIND_RIGHT, VIEW_SIDE)


def render_views(
    m: Any,
    d: Any,
    views: Sequence[View] = DEFAULT_VIEWS,
    *,
    width: int = 800,
    height: int = 600,
    gap: int = 6,
) -> Image:
    """Render each view at width x height and tile them left to right (white gaps). Calls mj_forward on d."""
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
    mujoco.mj_forward(m, d)
    tiles: list[Image] = []
    with mujoco.Renderer(m, height, width) as r:
        fovy0 = float(m.vis.global_.fovy)
        for v in views:
            m.vis.global_.fovy = fovy0 if v.fovy is None else v.fovy
            r.update_scene(d, camera=v.mjv_camera(m))
            tiles.append(np.asarray(r.render(), dtype=np.uint8).copy())
        m.vis.global_.fovy = fovy0
    if len(tiles) == 1:
        return tiles[0]
    sep = np.full((height, gap, 3), 255, dtype=np.uint8)
    parts: list[Image] = []
    for i, t in enumerate(tiles):
        if i:
            parts.append(sep)
        parts.append(t)
    return np.concatenate(parts, axis=1)


def render_png(
    m: Any,
    d: Any,
    path: str | Path,
    views: Sequence[View] = DEFAULT_VIEWS,
    *,
    width: int = 800,
    height: int = 600,
) -> Path:
    """Render `views` of (m, d) and write one tiled PNG. Returns the path."""
    return write_png(path, render_views(m, d, views, width=width, height=height))


def image_stats(img: npt.ArrayLike) -> dict[str, float]:
    """Simple checks for tests: mean brightness and the fraction of non-background pixels."""
    a = np.asarray(img, dtype=np.float64)
    gray = a[..., :3].mean(axis=-1)
    return {"mean": float(gray.mean()), "std": float(gray.std())}
