"""The simulated world: our R1 (one of the two profiles) standing in a scene.

Composes `r1sim.model.robot_spec` (robot + floor + light) with a scene (`r1sim.scenes`: the empty floor by default, a
plain table with loose objects, or any builder / "package.module:attr" a caller names) in one MjSpec, compiles it, and
gives the initial state: the robot at the nominal stand pose with its feet centred on the world origin (= the
controller's base frame), the scene's objects where the scene puts them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from . import joints, lens, model, robot_look, scenes


@dataclass(frozen=True)
class World:
    profile: str
    model: mujoco.MjModel
    index: model.RobotIndex
    scene: Any  # what the scene's builder returned (r1sim.scenes.SceneInfo for the built-ins), None without a scene
    seed: int | None = None  # None = the nominal robot and scene; else the variant (r1sim.variants) of this seed
    # the MjSpec the model was compiled from (session 4: r1sim.render_client exports it with spec.to_xml() for the
    # Filament render worker, DESIGN D13); not part of the world's identity
    spec: Any = field(default=None, repr=False, compare=False)

    def initial_qpos(self, pose: np.ndarray | None = None) -> np.ndarray:
        """The robot at `pose` (default: joints.POSE_STAND) on the floor, feet centred on the origin; the scene's free
        bodies at their placements (their qpos0)."""
        return model.stand_qpos(self.model, joints.POSE_STAND if pose is None else pose)


def _dress(spec: mujoco.MjSpec) -> None:
    """Pictures only (no physics): a gradient sky (the floor's look comes with the scene)."""
    spec.add_texture(
        name="r1sim_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.55, 0.62, 0.70],
        rgb2=[0.12, 0.13, 0.15],
        width=256,
        height=1536,
    )


def build_world(
    profile: str = "real",
    *,
    scene: str | scenes.SceneBuilder | None = "floor",
    visual: bool = True,
    seed: int | None = None,
    look: str = "photo",
    head_camera: bool = False,
    extra: Callable[[mujoco.MjSpec, Any], None] | None = None,
    with_scene: bool = True,
) -> World:
    """Robot (`profile`: "real" | "train") + a scene (`scene`: "floor" (default) / "table" / None, a builder
    `builder(spec, *, seed, visual, look) -> info`, or a "package.module:attr" string naming one; r1sim.scenes).
    seed None = the nominal robot (and scene); a seed = one robot variant within the parameter ranges (r1sim.variants;
    the scene's builder gets the same seed for its own variants). look "photo" (default, with visual) = the robot's
    white / navy / dark plastic (visual only, the physics is identical); "plain" = the model's stock colours.
    head_camera: add the calibrated head camera "head_cam" (r1sim.lens: models/head_camera/, rendered through
    lens.HeadCamera). extra(spec, scene_info): adds anything else before the compile. with_scene=False = no scene
    whatever `scene` says (kept for older callers; the same as scene=None)."""
    if look not in ("photo", "plain"):
        raise ValueError(f"look must be 'photo' or 'plain', not {look!r}")
    builder = scenes.resolve(scene) if with_scene else None
    spec = model.robot_spec(profile, visual=visual, seed=seed)
    if visual:
        _dress(spec)
        if look == "photo":
            robot_look.dress_robot(spec)  # the R1's white / navy / dark plastic (session 4; visual only)
    info: Any = None
    if builder is not None:
        info = builder(spec, seed=seed, visual=visual, look=look)
    if extra is not None:
        extra(spec, info)
    if head_camera:
        cal = lens.load_calibration()
        lens.add_head_camera(spec, cal, lens.render_frame(cal.model))
    m = spec.compile()
    return World(profile=profile, model=m, index=model.RobotIndex(m), scene=info, seed=seed, spec=spec)
