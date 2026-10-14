"""The R1 head camera chain in simulation (DESIGN §12.5): a wide pinhole render -> our lens remap -> the 544 x 448
frame the robot sends -> JPEG.

The robot's head image (left stereo camera, 544 x 448) follows the calibrated camera model of v2 (our previous
controller): pinhole with one radial term (x_d = x (1 + k1 r^2), normalised coordinates) or an equidistant fisheye;
pixel coordinates with the pixel CENTRE at the integer index (OpenCV convention), camera frame x right, y down, z
forward. The pose is the head_camera site (forward kinematics) @ T_site_cam (the mount, fitted by calib_head.py). Our
calibration: runs_calib/s3_cal (4 Oct 2026, 16 marks, left-out median 3.3 mm): pinhole + k1, f 126.9 px - a ~130 deg
wide image. Copied to models/head_camera/ (README there). Session 11: the simulated camera's MOUNT is the v3-frame
re-fit of the same capture (CALIBRATION_FILE; the 4 Oct file stays as CALIBRATION_FILE_V2_FRAME for v2-logged poses).

MuJoCo renders a centred pinhole only, so the simulator renders a larger pinhole frame that covers every ray of the
output image (for this calibration: ~980 x 840 at twice the output's central focal length, the "970 x 825" of DESIGN
§12.5) and resamples it through the camera model (`remap_maps`, cv2.remap). JPEG quality 90: the quantisation tables of
the real head frames (runs_calib/s3_cal/head.jpg) are the standard tables at quality 90.

The camera model functions are ported from r1mc2 (v2) r1mc2/headcam.py at git r21_try 2a4a165 (`_undistort`,
`unproject`, `project`; the same code as the calibration tool's); changes: typed, a dataclass for the model.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
REPO = Path(__file__).resolve().parents[2]
# Session 11 (the user's decision, 12 Oct ~03:05): the SIMULATED head camera sits where the robot's own calibration
# file says - the 4 Oct capture's mount re-fitted in v3's base frame (session 10, r1sim.sysid.camera_mount; the file
# the robot installs as /etc/r1mc3/head_camera.json from HW2). The sim model's frame is the true body frame (its IMU
# carries the mounting offset as geometry, S4), which is v3's base convention, so the v3-frame mount is the sim's
# physical camera and r1d in the sim can be given the robot's file unchanged. The lens model is identical in both files.
CALIBRATION_FILE = REPO / "models" / "head_camera" / "calibration_s3_cal_20261004_v3frame.json"
# The 4 Oct file as calib_head.py wrote it: its mount absorbed v2's IMU-as-pelvis attitude (~1.1 deg), so it is right
# ONLY together with a v2-logged T_base_site (A5's "logged" camera, A6's real-frame poses) - never on the sim's head.
CALIBRATION_FILE_V2_FRAME = REPO / "models" / "head_camera" / "calibration_s3_cal_20261004.json"
HEAD_CAMERA_SITE = "head_camera"
VIGNETTING_FILE = REPO / "models" / "head_camera" / "vignetting.yaml"
JPEG_QUALITY = 90  # the real head frames' quantisation tables = the standard tables at quality 90
SUPERSAMPLE = 2.0  # render focal length / the output's central focal length
ZNEAR_M = 0.02  # the head camera's near plane: the robot's own body is ~9 cm from it (MuJoCo scales znear by the
# model's extent; without clip control - macOS OpenGL - depth precision ~ z^2 / znear, so no smaller than needed)

# OpenCV camera frame (x right, y down, z forward) -> MuJoCo camera frame (x right, y up, looks along -z)
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class CameraModel:
    """A calibrated camera (v2 headcam.py / calib_head.py model dict) + the image size."""

    lens: str  # "pinhole" | "equidistant"
    f: float
    cx: float
    cy: float
    k1: float = 0.0
    depth: str = "z"  # what a depth value measures: "z" (along the axis) or "range" (along the ray)
    width: int = 544
    height: int = 448

    def as_dict(self) -> dict[str, Any]:
        return {"lens": self.lens, "f": self.f, "cx": self.cx, "cy": self.cy, "k1": self.k1, "depth": self.depth}


@dataclass(frozen=True)
class HeadCalibration:
    model: CameraModel
    T_site_cam: FloatArray  # 4 x 4, camera (OpenCV frame) in the head_camera site's frame
    source: str
    note: str


def load_calibration(path: str | Path = CALIBRATION_FILE) -> HeadCalibration:
    d = json.loads(Path(path).read_text())
    m = d["model"]
    model = CameraModel(
        lens=str(m.get("lens", "pinhole")),
        f=float(m["f"]),
        cx=float(m["cx"]),
        cy=float(m["cy"]),
        k1=float(m.get("k1") or 0.0),
        depth=str(m.get("depth", "z")),
        width=int(d.get("width", 544)),
        height=int(d.get("height", 448)),
    )
    return HeadCalibration(model, np.asarray(d["T_site_cam"], dtype=np.float64), str(path), str(d.get("note", "")))


# ---------------------------------------------------------------- camera model (ported from v2 headcam.py)
def _undistort(model: CameraModel, xd: FloatArray, yd: FloatArray) -> tuple[FloatArray, FloatArray]:
    if not model.k1:
        return xd, yd
    x, y = xd, yd
    for _ in range(20):
        s = 1.0 + model.k1 * (x * x + y * y)
        x, y = xd / s, yd / s
    return x, y


def normalised_rays(model: CameraModel, px: npt.ArrayLike, py: npt.ArrayLike) -> FloatArray:
    """Pixels -> camera-frame directions (N x 3) with z = 1 (pinhole; the equidistant model's rays scaled to z = 1,
    valid in front of the camera)."""
    px_, py_ = np.atleast_1d(np.asarray(px, dtype=np.float64)), np.atleast_1d(np.asarray(py, dtype=np.float64))
    x, y = (px_ - model.cx) / model.f, (py_ - model.cy) / model.f
    if model.lens == "equidistant":
        th = np.hypot(x, y)
        s = np.where(th > 1e-9, np.sin(th) / np.maximum(th, 1e-9), 1.0)
        ray = np.stack([x * s, y * s, np.cos(th)], 1)
        return np.asarray(ray / ray[:, 2:3])
    x, y = _undistort(model, x, y)
    return np.stack([x, y, np.ones_like(x)], 1)


def unproject(model: CameraModel, px: npt.ArrayLike, py: npt.ArrayLike, d: npt.ArrayLike) -> FloatArray:
    """Pixel(s) + depth value(s) -> camera-frame point(s) (N x 3), v2 headcam.unproject."""
    ray = normalised_rays(model, px, py)
    if model.depth == "range":
        ray = ray / np.linalg.norm(ray, axis=1, keepdims=True)
    return np.asarray(ray * np.atleast_1d(np.asarray(d, dtype=np.float64))[:, None])


def project(model: CameraModel, P: npt.ArrayLike) -> FloatArray:
    """Camera-frame point(s) (N x 3, OpenCV frame) -> pixels (N x 2); NaN behind the camera. v2 headcam.project."""
    P_ = np.atleast_2d(np.asarray(P, dtype=np.float64))
    out = np.full((len(P_), 2), np.nan)
    if model.lens == "equidistant":
        ok = np.linalg.norm(P_, axis=1) > 1e-9
        r = np.hypot(P_[:, 0], P_[:, 1])
        th = np.arctan2(r, P_[:, 2])
        s = np.where(r > 1e-12, th / np.maximum(r, 1e-12), 1.0 / np.maximum(P_[:, 2], 1e-12))
        out[ok] = np.c_[P_[ok, 0] * s[ok], P_[ok, 1] * s[ok]] * model.f + [model.cx, model.cy]
        return out
    ok = P_[:, 2] > 1e-6
    xy = P_[ok, :2] / P_[ok, 2:3]
    if model.k1:
        xy = xy * (1.0 + model.k1 * np.sum(xy * xy, axis=1, keepdims=True))
    out[ok] = xy * model.f + [model.cx, model.cy]
    return out


# ---------------------------------------------------------------- the render frame and the remap
@dataclass(frozen=True)
class RenderFrame:
    """The centred pinhole MuJoCo renders: size, focal length (px), vertical field of view."""

    width: int
    height: int
    f: float

    @property
    def fovy_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.height / 2.0 / self.f))

    def pixel(self, xy: FloatArray) -> FloatArray:
        """Normalised (x/z, y/z) in the OpenCV frame -> render pixel coordinates (pixel centre at the integer index;
        MuJoCo's image rows are top first, so y down matches)."""
        return np.asarray(xy * self.f + [self.width / 2.0 - 0.5, self.height / 2.0 - 0.5])


def render_frame(model: CameraModel, supersample: float = SUPERSAMPLE, margin_px: int = 4) -> RenderFrame:
    """The smallest centred pinhole (at `supersample` x the output's central focal length) that covers every output
    pixel's ray, plus a margin for the interpolation."""
    # the outer edges of the border pixels (pixel centres at integers, so the image spans -0.5 .. size - 0.5)
    u = np.linspace(-0.5, model.width - 0.5, 4 * model.width)
    v = np.linspace(-0.5, model.height - 0.5, 4 * model.height)
    border_u = np.concatenate([u, u, np.full_like(v, -0.5), np.full_like(v, model.width - 0.5)])
    border_v = np.concatenate([np.full_like(u, -0.5), np.full_like(u, model.height - 0.5), v, v])
    r = normalised_rays(model, border_u, border_v)
    fr = supersample * model.f
    w = 2 * math.ceil(fr * float(np.abs(r[:, 0]).max()) + margin_px)
    h = 2 * math.ceil(fr * float(np.abs(r[:, 1]).max()) + margin_px)
    return RenderFrame(w, h, fr)


def remap_maps(model: CameraModel, frame: RenderFrame) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """For every output pixel (u, v): where its ray lands in the render frame (cv2.remap maps, float32)."""
    uu, vv = np.meshgrid(np.arange(model.width, dtype=np.float64), np.arange(model.height, dtype=np.float64))
    rays = normalised_rays(model, uu.ravel(), vv.ravel())
    p = frame.pixel(rays[:, :2])
    return (
        p[:, 0].reshape(model.height, model.width).astype(np.float32),
        p[:, 1].reshape(model.height, model.width).astype(np.float32),
    )


def remap(
    image: npt.NDArray[Any],
    maps: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
    *,
    prefilter_sigma: float = 0.0,
    nearest: bool = False,
) -> npt.NDArray[Any]:
    """Resample a render-frame image (H x W [x C]) into the output frame. prefilter_sigma: a Gaussian on the render
    first (anti-aliasing for the supersampled colour image); nearest: for depth (no mixing across edges)."""
    import cv2  # validation dependency (dev group); never imported on the robot

    img = image
    if prefilter_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), prefilter_sigma)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return np.asarray(cv2.remap(img, maps[0], maps[1], interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0))


def jpeg(rgb: npt.NDArray[np.uint8], quality: int = JPEG_QUALITY) -> bytes:
    """The frame as the robot sends it (baseline JPEG, standard tables at `quality`)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------- the camera in the MuJoCo model
def cam_quat_from_cv(R_parent_cv: FloatArray) -> FloatArray:
    """MuJoCo camera quaternion (w, x, y, z) in its parent frame from the rotation of an OpenCV camera frame."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R_parent_cv @ CV_TO_MJ).reshape(-1))
    return q


def add_head_camera(spec: Any, cal: HeadCalibration, frame: RenderFrame, name: str = "head_cam") -> str:
    """A MuJoCo camera on the body that carries the head_camera site, at site @ T_site_cam, with the render frame's
    field of view (call before compile; the model's offscreen buffer is enlarged to the render size)."""
    site = next((s for s in spec.sites if s.name == HEAD_CAMERA_SITE), None)
    if site is None:
        raise KeyError(f"no site {HEAD_CAMERA_SITE!r} in the spec")
    R_bs = np.zeros(9)
    mujoco.mju_quat2Mat(R_bs, np.asarray(site.quat, dtype=np.float64))
    T_bs = np.eye(4)
    T_bs[:3, :3], T_bs[:3, 3] = R_bs.reshape(3, 3), np.asarray(site.pos, dtype=np.float64)
    T_bc = T_bs @ cal.T_site_cam
    body = site.parent
    body.add_camera(name=name, pos=list(T_bc[:3, 3]), quat=list(cam_quat_from_cv(T_bc[:3, :3])), fovy=frame.fovy_deg)
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), frame.width)
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), frame.height)
    return name


def add_camera_at(spec: Any, T_world_cv: FloatArray, frame: RenderFrame, name: str) -> str:
    """A fixed world camera at an OpenCV-frame pose (e.g. a logged T_base_cam) with the render frame's field of view."""
    T = np.asarray(T_world_cv, dtype=np.float64)
    spec.worldbody.add_camera(
        name=name, pos=list(T[:3, 3]), quat=list(cam_quat_from_cv(T[:3, :3])), fovy=frame.fovy_deg
    )
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), frame.width)
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), frame.height)
    return name


def T_world_cam(m: Any, d: Any, name: str = "head_cam") -> FloatArray:
    """The camera's pose in the world as an OpenCV-frame 4 x 4 (from MuJoCo's camera frame)."""
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
    T = np.eye(4)
    T[:3, :3] = d.cam_xmat[i].reshape(3, 3) @ CV_TO_MJ
    T[:3, 3] = d.cam_xpos[i]
    return T


def vignetting_gain(model: CameraModel, path: str | Path = VIGNETTING_FILE) -> FloatArray:
    """The head camera's radial darkening per output pixel (H x W, 1 at the principal point): the real frames darken
    toward the image edge, the pinhole render + remap does not (models/head_camera/vignetting.yaml: measured on the
    paired A13 frames, session 4; an estimate until HW1's flat field)."""
    import yaml

    v = yaml.safe_load(Path(path).read_text())
    yy, xx = np.mgrid[0 : model.height, 0 : model.width].astype(np.float64)
    x2 = ((xx - model.cx) ** 2 + (yy - model.cy) ** 2) / float(v["r_max_px"]) ** 2
    g = 1.0 + float(v["a2"]) * x2 + float(v["a4"]) * x2**2 + float(v["a6"]) * x2**3
    return np.asarray(np.clip(g, 0.0, 1.0), dtype=np.float64)


class HeadCamera:
    """Renders the head camera's 544 x 448 frame (RGB + depth along the camera axis, metres) from (m, d) of a model
    built with `add_head_camera` (r1sim.world.build_world(head_camera=True)).

    backend "classic" (default): MuJoCo 3.5.0's OpenGL renderer in this process. "filament" (DESIGN D13): the wide
    pinhole colour frame comes from the Filament render worker (`worker` = an r1sim.render_client.RenderWorker of the
    SAME world, e.g. RenderWorker(world)); the remap, the JPEG and the ground-truth depth are unchanged (depth is
    geometry: it stays on the classic renderer, created on first use)."""

    def __init__(
        self,
        m: Any,
        cal: HeadCalibration | None = None,
        *,
        name: str = "head_cam",
        supersample: float = SUPERSAMPLE,
        backend: str = "classic",
        worker: Any = None,
        vignetting: bool = True,
    ) -> None:
        if backend not in ("classic", "filament"):
            raise ValueError(f"backend must be 'classic' or 'filament', not {backend!r}")
        if backend == "filament" and worker is None:
            raise ValueError("backend='filament' needs worker=r1sim.render_client.RenderWorker(<the same world>)")
        self.cal = cal or load_calibration()
        self.model = self.cal.model
        self.frame = render_frame(self.model, supersample)
        self.name = name
        self.maps = remap_maps(self.model, self.frame)
        # the real lens's darkening toward the edge (session 4, A13); vignetting=False for pure-geometry checks
        self.gain: FloatArray | None = vignetting_gain(self.model) if vignetting else None
        self.backend = backend
        self._worker = worker
        self._m = m
        m.vis.map.znear = min(float(m.vis.map.znear), ZNEAR_M / float(m.stat.extent))
        self._r: Any = mujoco.Renderer(m, self.frame.height, self.frame.width) if backend == "classic" else None

    def _classic(self) -> Any:
        if self._r is None:
            self._r = mujoco.Renderer(self._m, self.frame.height, self.frame.width)
        return self._r

    def close(self) -> None:
        if self._r is not None:
            self._r.close()
            self._r = None

    def __enter__(self) -> HeadCamera:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def render_wide(self, d: Any, *, depth: bool = False) -> npt.NDArray[Any]:
        if self.backend == "filament" and not depth:
            return np.asarray(
                self._worker.render(d.qpos, self.name, self.frame.width, self.frame.height, znear=ZNEAR_M),
                dtype=np.uint8,
            )
        r = self._classic()
        r.update_scene(d, camera=self.name)
        if depth:
            r.enable_depth_rendering()
            out = np.asarray(r.render(), dtype=np.float64).copy()
            r.disable_depth_rendering()
            return out
        return np.asarray(r.render(), dtype=np.uint8).copy()

    def rgb(self, d: Any) -> npt.NDArray[np.uint8]:
        """The 544 x 448 colour frame (anti-aliased: Gaussian of 0.5 x supersample render px before the remap), darkened
        toward the edge like the real lens (vignetting_gain) unless the camera was built with vignetting=False."""
        wide = self.render_wide(d)
        out = remap(wide, self.maps, prefilter_sigma=0.5 * SUPERSAMPLE)
        if self.gain is not None:
            out = np.clip(np.asarray(out, dtype=np.float64) * self.gain[..., None] + 0.5, 0.0, 255.0)
        return np.asarray(out, dtype=np.uint8)

    def depth_z(self, d: Any) -> FloatArray:
        """Ground-truth depth along the camera axis (m) per output pixel (nearest render sample: no mixing across
        edges); 0 where nothing was hit within the far plane. Always the classic renderer (geometry, not looks)."""
        wide = self.render_wide(d, depth=True).astype(np.float32)
        return np.asarray(remap(wide, self.maps, nearest=True), dtype=np.float64)


# ---------------------------------------------------------------- depth-only additions (session 5, A6)
# What the stereo depth model (r1sim.depth) needs to know of a scene: its PLAIN featureless surfaces (a depth step
# between two of them has nothing in the image to anchor it and spreads like a table's far edge against a wall:
# r1sim.depth.apply's `plain`), its SHINY ones (glare: a hook, off until a recording shows it) and the materials of its
# CLEAR PLASTIC parts (the stereo sees largely through them, r1sim.depth.stereo_input). Scene knowledge: the generic
# scenes' names are here; a scene from another package registers its own when it is imported
# (`register_depth_surfaces`), so this module never names it (session R).
_DEPTH_SURFACES: dict[str, set[str]] = {"plain": {"floor", "table_top"}, "glare": set(), "clear_materials": set()}


def register_depth_surfaces(
    *, plain: Sequence[str] = (), glare: Sequence[str] = (), clear_materials: Sequence[str] = ()
) -> None:
    """Add a scene's geom names (plain / glare surfaces) and material names (clear plastic) for the depth model."""
    _DEPTH_SURFACES["plain"].update(plain)
    _DEPTH_SURFACES["glare"].update(glare)
    _DEPTH_SURFACES["clear_materials"].update(clear_materials)


def depth_surfaces(kind: str) -> tuple[str, ...]:
    """The registered names of one kind ("plain", "glare", "clear_materials"), sorted."""
    return tuple(sorted(_DEPTH_SURFACES[kind]))


def clear_plastic_geoms(m: Any) -> list[int]:
    """Geom ids of the clear-plastic parts (the registered clear materials): the stereo sees largely through them
    (r1sim.depth.stereo_input)."""
    mats = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, n) for n in _DEPTH_SURFACES["clear_materials"]}
    mats.discard(-1)
    return [g for g in range(m.ngeom) if int(m.geom_matid[g]) in mats]


def depth_without(cam: HeadCamera, d: Any, geoms: list[int]) -> FloatArray:
    """Ground-truth depth (as HeadCamera.depth_z) with the given geoms left out of the render (their alpha set to 0
    for this one frame - MuJoCo skips invisible geoms - and restored): what lies BEHIND clear plastic."""
    m = cam._m
    if not geoms:
        return cam.depth_z(d)
    old = np.array(m.geom_rgba[geoms], copy=True)
    try:
        m.geom_rgba[geoms, 3] = 0.0
        return cam.depth_z(d)
    finally:
        m.geom_rgba[geoms] = old


def segmentation(cam: HeadCamera, d: Any) -> npt.NDArray[np.int32]:
    """The geom id seen at each output pixel (-1 = nothing), through the same lens chain as depth_z (classic renderer,
    nearest remap)."""
    r = cam._classic()
    r.update_scene(d, camera=cam.name)
    r.enable_segmentation_rendering()
    try:
        seg = np.asarray(r.render(), dtype=np.int32).copy()
    finally:
        r.disable_segmentation_rendering()
    gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1).astype(np.float32)
    return np.asarray(remap(gid, cam.maps, nearest=True), dtype=np.float64).round().astype(np.int32)


def plain_surface_mask(cam: HeadCamera, d: Any, names: Sequence[str] | None = None) -> npt.NDArray[np.bool_]:
    """True where the head camera sees one of the plain surfaces (`names`, default: the registered plain surfaces)."""
    m = cam._m
    names = depth_surfaces("plain") if names is None else names
    ids = [i for i in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names) if i >= 0]
    return np.asarray(np.isin(segmentation(cam, d), ids))
