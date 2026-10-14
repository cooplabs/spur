"""The live S12 count (R1CAM_PLAN.md §2 "record words"): while `r1 record run charuco` records, every left frame's
ChArUco detection is binned by WHERE the board is in the image (3 x 3 places) and HOW it is tilted (flat / turned left-
right / tipped), and the recorder's status says "hold still" until the current cell has enough frames, then "next" with
the first cell still missing - so the operator covers the whole field (B5: full-field RMS <= 1 px needs the corners).

The tilt class comes from the foreshortening of the detected corner grid: the mean spacing of horizontally adjacent
corners over that of vertically adjacent ones, divided by 1 for square cells (turned ~30 deg left-right -> ~0.87, tipped
~30 deg -> ~1.15; the threshold sits halfway at 0.93 / 1.07); the image's own lens compression near the edges is the
same for both directions to first order. Generic: a board is named by its spec (`BOARDS`: the printed board of
docs/hardware/CHARUCO.md); needs OpenCV (the camera process's `perception` extra) - without it the status says so.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

# the printed board (docs/hardware/CHARUCO.md; r1sim.charuco_render.CHOSEN - the same grid)
BOARDS: dict[str, dict[str, Any]] = {
    "poster_6x4_90": {"cols": 6, "rows": 4, "square_mm": 90.0, "marker_mm": 68.0, "dict": "DICT_4X4_50"},
}
PLACES = ("top-left", "top", "top-right", "left", "centre", "right", "bottom-left", "bottom", "bottom-right")
TILTS = ("flat", "turned ~30 deg left-right", "tipped ~30 deg")  # record.plans.CHARUCO_TILTS
MIN_CORNERS = 6
TURNED_MAX, TIPPED_MIN = 0.93, 1.07


class CharucoCoverage:
    """Feed it JPEG frames (the recorder's thread); read `status()` from anywhere."""

    def __init__(
        self,
        board: str | dict[str, Any] = "poster_6x4_90",
        *,
        need: int = 8,
        width: int = 544,
        height: int = 448,
        distances_m: tuple[float, ...] = (0.3, 0.45, 0.6),
        f_px: float = 127.0,
    ) -> None:
        self.spec = dict(BOARDS[board]) if isinstance(board, str) else dict(board)
        self.need = int(need)
        self.distances = tuple(float(z) for z in distances_m)
        self.f_px = float(f_px)  # the distance class from the corners' spacing: z ~ f x square / spacing
        self.w, self.h = int(width), int(height)
        self.counts: dict[tuple[int, int, int], int] = {}
        self.frames = 0
        self.detected = 0
        self.undecodable = 0
        self.last: dict[str, Any] | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._det: Any = None
        try:
            import cv2

            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, str(self.spec["dict"])))
            sq, mk = float(self.spec["square_mm"]) / 1e3, float(self.spec["marker_mm"]) / 1e3
            b = cv2.aruco.CharucoBoard((int(self.spec["cols"]), int(self.spec["rows"])), sq, mk, d)
            self._det = cv2.aruco.CharucoDetector(b)
        except Exception as e:  # no OpenCV here: the count is off, the recording is not
            self.error = f"no live count (OpenCV: {type(e).__name__}: {e})"

    def classify(self, ids: np.ndarray, px: np.ndarray) -> tuple[int, int, int] | None:
        """(distance, place, tilt) of a detection (ids = ChArUco corner ids, px = their pixels) or None (too few
        corners). The distance: the nearest planned one to f x square / the larger mean spacing (the untilted side)."""
        if len(ids) < MIN_CORNERS:
            return None
        c = px.mean(axis=0)
        place = 3 * min(2, int(3 * c[1] / self.h)) + min(2, int(3 * c[0] / self.w))
        nx = int(self.spec["cols"]) - 1
        pos = {int(i): p for i, p in zip(ids, px, strict=True)}
        dh = [np.linalg.norm(pos[i + 1] - p) for i, p in pos.items() if (i % nx) < nx - 1 and i + 1 in pos]
        dv = [np.linalg.norm(pos[i + nx] - p) for i, p in pos.items() if i + nx in pos]
        if not dh or not dv:
            return None
        ratio = float(np.mean(dh) / np.mean(dv))
        tilt = 1 if ratio < TURNED_MAX else (2 if ratio > TIPPED_MIN else 0)
        z = self.f_px * float(self.spec["square_mm"]) / 1e3 / max(float(np.mean(dh)), float(np.mean(dv)), 1e-6)
        dist = min(range(len(self.distances)), key=lambda k: abs(self.distances[k] - z))
        return dist, place, tilt

    def feed(self, jpeg: bytes) -> None:
        """One left JPEG. Decoded with OpenCV (session 12: it was Pillow - a DEV dependency only, absent from the
        robot's `--no-dev` install: on HW2 day 1 every frame raised ImportError inside the recorder's guard and the
        count stayed at 0 frames for the whole 325 s C3 recording). An undecodable frame is counted and said."""
        if self._det is None:
            return
        import cv2

        gray = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            with self._lock:
                self.frames += 1
                self.undecodable += 1
            return
        cc, ci, _, _ = self._det.detectBoard(gray)
        ids = np.zeros(0, dtype=np.int64) if ci is None else np.asarray(ci, dtype=np.int64).ravel()
        px = np.zeros((0, 2)) if cc is None else np.asarray(cc, dtype=np.float64).reshape(-1, 2)
        cell = self.classify(ids, px)
        with self._lock:
            self.frames += 1
            if cell is not None:
                self.detected += 1
                self.counts[cell] = self.counts.get(cell, 0) + 1
            self.last = {"corners": len(ids), "cell": cell}

    def missing(self) -> list[tuple[int, int, int]]:
        """The cells still short of `need` frames, in the plan's order (distance, place, tilt)."""
        return [(d, p, t) for d in range(len(self.distances)) for p in range(9) for t in range(3)
                if self.counts.get((d, p, t), 0) < self.need]  # fmt: skip

    def _cell(self, c: tuple[int, int, int]) -> str:
        return f"{self.distances[c[0]]:.2f} m, {PLACES[c[1]]}, {TILTS[c[2]]}"

    def status(self) -> dict[str, Any]:
        with self._lock:
            last = dict(self.last) if self.last else None
            miss = self.missing()
            n = 27 * len(self.distances)
            if self.error:
                say = self.error
            elif self.frames and self.undecodable == self.frames:
                say = f"no live count: none of the {self.frames} frames could be decoded (JPEG)"
            elif last is None or last["cell"] is None:
                say = "no board in view: show the board" + (f" ({self._cell(miss[0])})" if miss else "")
            elif self.counts.get(last["cell"], 0) < self.need:
                say = f"hold still ({self._cell(last['cell'])}): {self.counts[last['cell']]}/{self.need}"
            elif miss:
                say = f"next: {self._cell(miss[0])}"
            else:
                say = "every distance, place and tilt covered: stop the recording"
            return {
                "say": say,
                "cells_done": n - len(miss),
                "cells": n,
                "frames": self.frames,
                "detected": self.detected,
                "undecodable": self.undecodable,
                "counts": {self._cell(c): k for c, k in sorted(self.counts.items())},
            }
