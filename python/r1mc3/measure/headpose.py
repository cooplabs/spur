"""The head camera's pose in the robot's base frame from the joints (the `T_ref_cam` of a head `View`).

Base frame = the floor under the middle of the feet, x along the feet's mean heading, z up, the soles level - the
simulator's standing frame (`r1sim.model.stand_qpos`: the same model, the robot standing at those joints with both feet
flat) and v2's T_base_cam convention at capture. Camera = the head_camera site @ the calibration's mount `T_site_cam`
(OpenCV frame). Valid for a robot STANDING on both feet (the pre-flight's stance); a lean, a foot off the floor or a
non-level floor is not in it (the IMU's roll / pitch would be the next term). A5 (session 4): the site's FK agrees with
v2's logged T_base_site to 12 / 5 / 1 mm and 0.7 deg - the two stance conventions differ, not the head chain.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@lru_cache(maxsize=2)
def _model(profile: str) -> Any:
    from r1sim import model

    return model.build_model(profile, visual=False)


def head_T_base_cam(q26: npt.ArrayLike, *, profile: str = "real", T_site_cam: FloatArray | None = None) -> FloatArray:
    """4 x 4: the head camera (OpenCV frame) in the base frame for the 26 body joints (SDK order)."""
    import mujoco
    from r1sim import lens, model

    m = _model(profile)
    d = mujoco.MjData(m)
    d.qpos[:] = model.stand_qpos(m, np.asarray(q26, dtype=np.float64))
    mujoco.mj_kinematics(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, lens.HEAD_CAMERA_SITE)
    T_bs = np.eye(4)
    T_bs[:3, :3] = d.site_xmat[sid].reshape(3, 3)
    T_bs[:3, 3] = d.site_xpos[sid]
    Tsc = lens.load_calibration().T_site_cam if T_site_cam is None else np.asarray(T_site_cam, dtype=np.float64)
    return np.asarray(T_bs @ Tsc)


_GEO_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _geometry(profile: str) -> Any:
    from r1mc3.world import Geometry

    return Geometry(None, profile=profile)


def head_T_base_cam_stance(
    q26: npt.ArrayLike,
    imu_quat: npt.ArrayLike | None = None,
    *,
    profile: str = "real",
    T_site_cam: FloatArray | None = None,
) -> FloatArray:
    """4 x 4: the head camera (OpenCV frame) in the PLANNER's base frame (session 10) - `r1mc3.world.Geometry.
    set_stance`: the pelvis's roll / pitch from the IMU with its mounting rotation undone, the origin at the foot
    sites' midpoint (z = the lower site), x along the feet's mean heading. The world model, the planner and pointing
    use this frame, so an entity measured here and declared into the world lands where the planner expects it.

    Why not `head_T_base_cam` (level soles from the joints alone): in the simulator, the robot balancing at the task
    view, that convention put the camera 1.8 / -4.0 / 2.0 mm and 0.24 deg off the truth (the pelvis's real roll /
    pitch under the policy) -> the table 2.3 mm, a container 6.7-9 mm off; the IMU term removes the rotation part.
    `imu_quat` None = the pelvis level (as set_stance)."""
    import mujoco
    from r1sim import lens

    with _GEO_LOCK:
        geo = _geometry(profile)
        geo.set_stance(
            np.asarray(q26, dtype=np.float64), None if imu_quat is None else np.asarray(imu_quat, dtype=np.float64)
        )
        sid = mujoco.mj_name2id(geo.m, mujoco.mjtObj.mjOBJ_SITE, lens.HEAD_CAMERA_SITE)
        T_bs = np.eye(4)
        T_bs[:3, :3] = geo.d.site_xmat[sid].reshape(3, 3)
        T_bs[:3, 3] = geo.d.site_xpos[sid]
    Tsc = lens.load_calibration().T_site_cam if T_site_cam is None else np.asarray(T_site_cam, dtype=np.float64)
    return np.asarray(T_bs @ Tsc)
