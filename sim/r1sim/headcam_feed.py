"""The simulated head camera as r1cam's frame source (session 10): `HeadFeed(world, data_fn).render(stream)` gives
r1cam's SimSource / OnDemandSource exactly what the robot's stereo module gives RobotSource - the left image as a JPEG
(544 x 448, the lens chain of r1sim.lens: render -> remap -> vignetting -> JPEG q90) and the depth as uint16 mm (0 =
none), at the current state of the simulation's MjData.

depth: "truth" (default; the ground-truth depth along the camera axis) or "r1" (r1sim.depth.apply: the fitted R1
noise model - holes, smooth bias, noise; seeded per frame) or (session 13) "r1_scene": the same with the scene's own
masks - the plain surfaces low-passed among themselves, an object's outline sharp, a clear plastic object partly seen
through (r1sim.depth's A6 schema 3 as fitted: `plain`, `z_behind`) - day 2's clear-tape click read the board behind
it. The world must be built with head_camera=True (and visual=True for the photo look). The right image is not
simulated here (v2's connector does not ask for it; a right calibration does not exist yet) - asking for it raises, and
r1cam reports the stream's error.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from r1sim import lens


class HeadFeed:
    def __init__(
        self,
        w: Any,
        data_fn: Callable[[], Any],
        *,
        depth: str = "truth",
        seed: int = 0,
        backend: str = "classic",
        worker: Any = None,
        vignetting: bool = True,
        calibration: lens.HeadCalibration | None = None,
    ) -> None:
        if depth not in ("truth", "r1", "r1_scene"):
            raise ValueError("depth: 'truth', 'r1' or 'r1_scene'")
        self.cam = lens.HeadCamera(w.model, calibration, backend=backend, worker=worker, vignetting=vignetting)
        self.data_fn = data_fn
        self.depth = depth
        self.seed = int(seed)
        self.frames = 0

    @property
    def calibration(self) -> lens.HeadCalibration:
        return self.cam.cal

    def calibration_dict(self) -> dict[str, Any]:
        """The camera the simulator renders through, in calib_head.py's format (for r1d's HeadCalibration)."""
        return {"T_site_cam": np.asarray(self.cam.cal.T_site_cam).tolist(), "model": self.cam.model.as_dict(),
                "note": f"the simulator's head camera ({self.cam.cal.source})"}  # fmt: skip

    def render(self, stream: str) -> Any:
        d = self.data_fn()
        self.frames += 1
        if stream == "left":
            return lens.jpeg(self.cam.rgb(d))
        if stream == "depth":
            z = self.cam.depth_z(d)
            if self.depth == "r1":
                from r1sim import depth as D

                return np.asarray(D.apply(z, seed=self.seed + self.frames), dtype="<u2")
            if self.depth == "r1_scene":
                from r1sim import depth as D

                plain = lens.plain_surface_mask(self.cam, d)
                behind = lens.depth_without(self.cam, d, lens.clear_plastic_geoms(self.cam._m))
                out = D.apply(z, seed=self.seed + self.frames, plain=plain, z_behind=behind)
                return np.asarray(out, dtype="<u2")
            mm = np.where(z > 0, np.round(z * 1000.0), 0.0)
            return np.clip(mm, 0, 65534).astype("<u2")
        raise KeyError(f"stream {stream!r}: the simulated head camera gives left + depth")

    def close(self) -> None:
        self.cam.close()
