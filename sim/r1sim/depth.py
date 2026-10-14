"""The head camera's depth model (DESIGN §12.5; row A6): ground-truth depth from the render
(r1sim.lens.HeadCamera.depth_z, metres along the camera axis) -> what the R1's native stereo depth delivers: a
544 x 448 uint16 image in mm, 0 = no depth (the format of the robot's depth_mm.pgm).

Session 4 (A6) fitted it to the recorded native depth (our own captures; the fit is an internal validation): six table
captures of 4-6 Oct against the known table plane, the far-edge step of 3 Oct, all 16 recorded frames for holes;
session 5 refitted it (schema 3) after finding why the error was smallest at the image centre and why the depth is
sharp at an object's outline but not at the table's far edge. Effects, in order, all from models/depth.yaml:
0. clear plastic (a clear plastic object) is partly seen through: `clear_stereo_opacity` of the plastic, the rest what
   lies behind it (`z_behind`, r1sim.lens.depth_without); a glare hook for shiny surfaces (off: no data);
1. the stereo's low-pass, by surface: a step between two PLAIN surfaces (table, walls, board, floor: `plain`,
   r1sim.lens.plain_surface_mask) spreads by `blur_sigma_px` (the far edge against the wall: 22-23 px, 10-90 %); an
   object's outline (the boxes and toys on the table, the robot) by `blur_object_sigma_px` (~6 px, measured on
   the recorded frame with objects). Without `plain`, every pixel gets `blur_sigma_px` (session 4);
2. optional edge fattening (`edge_fatten_px`, 0: the low-pass covers it);
3. the bias field of a disparity error that varies across the image: z^2 (k0 + k_lr lr + k_ud ud + k_r2 rho^2), lr -1
   (left) .. +1 (right), ud -1 (top) .. +1 (bottom), rho = the distance from the image centre in half-widths - the
   left reads nearer, the right farther (v2: "the left half of the table 3-6 cm high, the right up to 6 cm low") and
   the image's edges nearer than its centre (session 5: the same sign in all six frames); per seed each k is drawn
   around its mean with the spread seen across the recorded frames;
4. a spatially correlated error (two Gaussian-filtered noise fields, fine + coarse, and a small white part) with the
   amplitude sigma_a + sigma_b z^2, fitted on the residual after the four-term field;
5. holes (`hole_fraction`, measured 0: the native stream is dense - v2's "valid fraction 0.95" is the robot's own
   body nearer than 10 cm), optional holes at depth steps (`edge_hole_px`, 0);
6. mm quantisation.
Not modelled (no recorded data): glare on shiny surfaces (a glossy board, a metal sheet) - the native stream fills it
with SOME depth rather than leaving holes; the hook (`glare`, `glare_fill_fraction` 0) waits for recorded depth frames
of such surfaces; the temporal behaviour (one frame per capture was recorded). `DepthConfig.off()` = ground truth, only
quantised to mm (0 where nothing was hit); `DepthConfig.s4()` = session 4's fit and `DepthConfig.s3_placeholder()` =
session 3's first-cut values (for comparisons).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

DEPTH_YAML = Path(__file__).resolve().parents[2] / "models" / "depth.yaml"
FloatArray = npt.NDArray[np.float64]
STEP_M = 0.03  # a depth step this large counts as an edge


@dataclass(frozen=True)
class DepthConfig:
    quant_mm: float = 1.0
    blur_sigma_px: float = 8.97
    edge_fatten_px: int = 0
    field_k0_per_m: float = 0.0635
    field_k_lr_per_m: float = 0.4099
    field_k_ud_per_m: float = 0.048
    field_k0_spread_per_m: float = 0.0431
    field_k_lr_spread_per_m: float = 0.0359
    field_k_ud_spread_per_m: float = 0.1278
    sigma_a_m: float = 0.0045
    sigma_b_per_m: float = 0.01343
    corr_fine_px: float = 4.75
    corr_coarse_px: float = 15.0
    corr_fine_weight: float = 0.3592
    corr_coarse_weight: float = 0.5705
    hole_fraction: float = 0.0
    edge_hole_px: int = 0
    # session 5 (A6): the disparity field gains a radial term (the image's edges read nearer: z^2 k_r2 rho^2, rho = the
    # distance from the image centre in half-widths - the same sign and size in all six recorded frames); an OBJECT's
    # outline is low-passed by `blur_object_sigma_px` (fitted on the
    # opaque objects' edges in the recorded frames) while `blur_sigma_px` stays the spread of a step between two plain
    # surfaces (the table's far edge against the wall); clear plastic is partly seen through (`clear_stereo_opacity`:
    # 1 = an opaque surface, 0 = the stereo sees what is behind it)
    field_k_r2_per_m: float = -0.5639
    field_k_r2_spread_per_m: float = 0.1103
    blur_object_sigma_px: float = 2.34
    clear_stereo_opacity: float = 0.35
    # a HOOK, not a model (no recorded depth shows a shiny surface): the fraction of a glare region's pixels whose
    # depth the stream fills from the surroundings (as it fills the unmatched dark floor); 0 = off until recorded
    # depth frames of shiny surfaces
    glare_fill_fraction: float = 0.0
    # session 3's first cut (kept for comparisons: DepthConfig.s3_placeholder()): a fixed left-right bias at the
    # image edges and a smooth random field instead of the disparity field, white noise
    bias_lr_m: float = 0.0
    bias_smooth_m: float = 0.0

    @classmethod
    def from_yaml(cls, path: str | Path = DEPTH_YAML) -> DepthConfig:
        d = yaml.safe_load(Path(path).read_text())
        base = cls()
        kw: dict[str, Any] = {}
        for name, default in asdict(base).items():
            v = d[name]["value"] if name in d else default
            kw[name] = int(v) if isinstance(default, int) and not isinstance(default, bool) else float(v)
        return cls(**kw)

    @classmethod
    def off(cls) -> DepthConfig:
        return cls(
            blur_sigma_px=0.0,
            edge_fatten_px=0,
            field_k0_per_m=0.0,
            field_k_lr_per_m=0.0,
            field_k_ud_per_m=0.0,
            field_k0_spread_per_m=0.0,
            field_k_lr_spread_per_m=0.0,
            field_k_ud_spread_per_m=0.0,
            sigma_a_m=0.0,
            sigma_b_per_m=0.0,
            hole_fraction=0.0,
            edge_hole_px=0,
            field_k_r2_per_m=0.0,
            field_k_r2_spread_per_m=0.0,
            blur_object_sigma_px=0.0,
            clear_stereo_opacity=1.0,
            glare_fill_fraction=0.0,
        )

    @classmethod
    def s4(cls) -> DepthConfig:
        """Session 4's fit (models/depth.yaml schema 2): one Gaussian low-pass everywhere (8.97 px), a position-free
        amplitude (13.99 mm), clear plastic opaque - kept for the before / after comparison of the A6 row."""
        return cls(
            quant_mm=1.0,
            blur_sigma_px=8.97,
            edge_fatten_px=0,
            field_k0_per_m=-0.0338,
            field_k_lr_per_m=0.4404,
            field_k_ud_per_m=-0.2102,
            field_k0_spread_per_m=0.034,
            field_k_lr_spread_per_m=0.0383,
            field_k_ud_spread_per_m=0.1503,
            sigma_a_m=0.01399,
            sigma_b_per_m=0.0,
            corr_fine_px=5.75,
            corr_coarse_px=26.0,
            corr_fine_weight=0.2622,
            corr_coarse_weight=0.6969,
            hole_fraction=0.0,
            edge_hole_px=0,
            field_k_r2_per_m=0.0,
            field_k_r2_spread_per_m=0.0,
            blur_object_sigma_px=8.97,
            clear_stereo_opacity=1.0,
            glare_fill_fraction=0.0,
        )

    @classmethod
    def s3_placeholder(cls) -> DepthConfig:
        """Session 3's first cut (models/depth.yaml schema 1, every value an estimate): white noise 2 mm + 0.010 z^2,
        +-4.5 cm left-right, a 1 cm smooth field, 4.9 % holes (the misread valid fraction), 2 px edge holes and
        fattening, no low-pass."""
        return cls(
            blur_sigma_px=0.0,
            edge_fatten_px=2,
            field_k0_per_m=0.0,
            field_k_lr_per_m=0.0,
            field_k_ud_per_m=0.0,
            field_k0_spread_per_m=0.0,
            field_k_lr_spread_per_m=0.0,
            field_k_ud_spread_per_m=0.0,
            sigma_a_m=0.002,
            sigma_b_per_m=0.010,
            corr_fine_weight=0.0,
            corr_coarse_weight=0.0,
            hole_fraction=0.049,
            edge_hole_px=2,
            bias_lr_m=0.045,
            bias_smooth_m=0.01,
            field_k_r2_per_m=0.0,
            field_k_r2_spread_per_m=0.0,
            blur_object_sigma_px=0.0,
            clear_stereo_opacity=1.0,
            glare_fill_fraction=0.0,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def edges(z: FloatArray) -> npt.NDArray[np.bool_]:
    """Pixels next to a depth step > STEP_M (4-connected), including steps to 'no hit' (0)."""
    e = np.zeros(z.shape, dtype=bool)
    dx = np.abs(np.diff(z, axis=1)) > STEP_M
    dy = np.abs(np.diff(z, axis=0)) > STEP_M
    e[:, 1:] |= dx
    e[:, :-1] |= dx
    e[1:, :] |= dy
    e[:-1, :] |= dy
    return e


def _dilate(mask: npt.NDArray[np.bool_], r: int) -> npt.NDArray[np.bool_]:
    out = mask.copy()
    for axis in (0, 1):
        src = out.copy()
        for k in range(1, r + 1):
            out |= np.roll(src, k, axis=axis) | np.roll(src, -k, axis=axis)
    return out


def _min_filter(z: FloatArray, r: int) -> FloatArray:
    """Grey erosion: the minimum over a (2r+1)^2 window (separable; no scipy)."""
    out = z.copy()
    for axis in (0, 1):
        src = out.copy()
        for k in range(1, r + 1):
            out = np.minimum(out, np.minimum(np.roll(src, k, axis=axis), np.roll(src, -k, axis=axis)))
    return out


def _smooth_field(shape: tuple[int, int], rng: np.random.Generator, scale_px: float) -> FloatArray:
    """A smooth random field with unit std (a few random low-frequency cosines; deterministic per rng)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    f = np.zeros(shape)
    for _ in range(6):
        kx, ky = rng.normal(0.0, 1.0 / scale_px, 2)
        f += np.cos(2 * np.pi * (kx * xx + ky * yy) + rng.uniform(0, 2 * np.pi))
    return np.asarray(f / max(float(f.std()), 1e-9))


def _gauss(img: FloatArray, sigma_px: float) -> FloatArray:
    import cv2  # validation dependency (dev group); the simulator only

    return np.asarray(cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), sigma_px, borderType=cv2.BORDER_REFLECT))


def _unit_field(shape: tuple[int, int], rng: np.random.Generator, sigma_px: float) -> FloatArray:
    """Gaussian-filtered white noise with unit std (autocorrelation exp(-k^2 / (4 sigma_px^2)))."""
    f = rng.standard_normal(shape)
    if sigma_px > 0:
        f = _gauss(f, sigma_px)
    return np.asarray(f / max(float(f.std()), 1e-12))


def radial_rho2(h: int, w: int) -> FloatArray:
    """rho^2 per pixel: the squared distance from the image centre in units of the half-width (0 at the centre, 1 at
    the left / right edge's middle, ~1.68 at the corners of the 544 x 448 frame)."""
    half = (w - 1) / 2.0
    x = (np.arange(w, dtype=np.float64)[None, :] - (w - 1) / 2.0) / half
    y = (np.arange(h, dtype=np.float64)[:, None] - (h - 1) / 2.0) / half
    return np.asarray(x * x + y * y)


def stereo_input(z_true: npt.ArrayLike, z_behind: npt.ArrayLike | None, opacity: float) -> FloatArray:
    """What the stereo sees before its low-pass where clear plastic is in front: `z_behind` = the ground truth with
    the clear-plastic geoms removed (r1sim.lens.depth_without); a pixel whose first hit is clear plastic reads
    opacity x the plastic + (1 - opacity) x what is behind it (the matcher locks on the plastic's own rims and
    reflections a fraction of the time, otherwise on the texture seen through it)."""
    z = np.asarray(z_true, dtype=np.float64)
    if z_behind is None or opacity >= 1.0:
        return z
    zb = np.asarray(z_behind, dtype=np.float64)
    clear = (z > 0) & (zb > z + 1e-4)
    return np.asarray(np.where(clear, opacity * z + (1.0 - opacity) * zb, z))


def _norm_gauss(z: FloatArray, mask: npt.NDArray[np.bool_], sigma_px: float) -> FloatArray:
    """A normalised Gaussian of z over the pixels in `mask` (a pixel's value is the weighted mean of its masked
    neighbours only)."""
    num = _gauss(np.where(mask, z, 0.0), sigma_px)
    den = _gauss(mask.astype(np.float64), sigma_px)
    return np.asarray(num / np.maximum(den, 1e-9))


def apply(
    z_true: npt.ArrayLike,
    cfg: DepthConfig | None = None,
    *,
    seed: int = 0,
    plain: npt.ArrayLike | None = None,
    z_behind: npt.ArrayLike | None = None,
    glare: npt.ArrayLike | None = None,
) -> npt.NDArray[np.uint16]:
    """Ground-truth depth (m, 0 = no hit) -> the R1-like depth image (uint16 mm, 0 = no depth). Deterministic per
    seed (one numpy Generator, fixed draw order: the field's coefficients, the fine, coarse and white noise, the
    session-3 smooth field and noise, the holes).

    `plain` = the pixels that show a plain, featureless surface (table top, walls, board, floor: r1sim.lens.
    plain_surface_mask): they are low-passed by `blur_sigma_px` among themselves (a step between two plain surfaces
    spreads like the table's far edge against the wall); every other hit pixel (an object, the robot) by
    `blur_object_sigma_px` over all hit pixels (an object's outline stays sharp, as the objects' and the robot's in
    the recorded frames). Without `plain` every pixel is a plain surface (session 4's single Gaussian).
    `z_behind` = the ground truth without the clear-plastic geoms (`lens.depth_without`), mixed in by
    `clear_stereo_opacity`. `glare` = pixels of shiny surfaces (lens.depth_surfaces("glare")): a hook, off while
    `glare_fill_fraction` is 0 (no recorded depth of a shiny surface yet)."""
    c = cfg or DepthConfig.from_yaml()
    z = stereo_input(z_true, z_behind, c.clear_stereo_opacity)
    hit = z > 0
    if glare is not None and c.glare_fill_fraction > 0:  # the stream fills what it cannot match from around it
        gl = np.asarray(glare, dtype=bool) & hit
        fill = _norm_gauss(z, hit & ~gl, max(c.blur_sigma_px, 1.0))
        z = np.where(gl, (1.0 - c.glare_fill_fraction) * z + c.glare_fill_fraction * fill, z)
    h, w = z.shape
    rng = np.random.default_rng(seed)
    zz = np.where(hit, z, 0.0)
    pl = hit if plain is None else (np.asarray(plain, dtype=bool) & hit)
    obj = hit & ~pl
    if c.blur_sigma_px > 0 or (obj.any() and c.blur_object_sigma_px > 0):
        out = zz.copy()
        if c.blur_sigma_px > 0 and pl.any():  # the stereo's aggregation between plain surfaces
            out = np.where(pl, _norm_gauss(zz, pl, c.blur_sigma_px), out)
        if c.blur_object_sigma_px > 0 and obj.any():  # an object's outline: the narrow low-pass over every hit pixel
            out = np.where(obj, _norm_gauss(zz, hit, c.blur_object_sigma_px), out)
        zz = np.where(hit, out, 0.0)
    if c.edge_fatten_px > 0:  # near a step the nearer surface wins over a band of edge_fatten_px
        inf = np.where(hit, zz, np.inf)
        near = _dilate(edges(np.where(hit, zz, 0.0)), c.edge_fatten_px)
        inf = np.where(near, _min_filter(inf, c.edge_fatten_px), inf)
        hit = np.isfinite(inf)
        zz = np.where(hit, inf, 0.0)
    lr = np.broadcast_to((np.arange(w, dtype=np.float64)[None, :] - (w - 1) / 2.0) / ((w - 1) / 2.0), (h, w))
    ud = np.broadcast_to((np.arange(h, dtype=np.float64)[:, None] - (h - 1) / 2.0) / ((h - 1) / 2.0), (h, w))
    k = rng.standard_normal(3) * (c.field_k0_spread_per_m, c.field_k_lr_spread_per_m, c.field_k_ud_spread_per_m)
    k += (c.field_k0_per_m, c.field_k_lr_per_m, c.field_k_ud_per_m)
    bias = zz * zz * (k[0] + k[1] * lr + k[2] * ud)
    if c.field_k_r2_per_m or c.field_k_r2_spread_per_m:  # session 5: the radial term (drawn after the others, so
        # a model without it keeps session 4's random streams)
        k_r2 = c.field_k_r2_per_m + c.field_k_r2_spread_per_m * float(rng.standard_normal())
        bias = bias + zz * zz * k_r2 * radial_rho2(h, w)
    wf, wc = c.corr_fine_weight, c.corr_coarse_weight
    ww = max(0.0, 1.0 - wf - wc)
    unit = np.zeros((h, w))
    for weight, sigma in ((wf, c.corr_fine_px), (wc, c.corr_coarse_px), (ww, 0.0)):
        if weight > 0:
            unit += np.sqrt(weight) * _unit_field((h, w), rng, sigma)
    noise = unit * (c.sigma_a_m + c.sigma_b_per_m * zz * zz)
    if c.bias_lr_m or c.bias_smooth_m:  # session 3's first cut
        bias = bias + c.bias_lr_m * lr + c.bias_smooth_m * _smooth_field((h, w), rng, scale_px=w / 4.0)
    out = np.where(hit, zz + bias + noise, 0.0)
    holes = ~hit
    if c.edge_hole_px > 0:
        holes |= _dilate(edges(zz), c.edge_hole_px - 1)
    need = c.hole_fraction - float(holes.mean())
    if need > 0:  # random speckle holes up to the given fraction
        holes |= rng.random((h, w)) < need / max(1e-9, 1.0 - float(holes.mean()))
    mm = np.round(out * 1000.0 / c.quant_mm) * c.quant_mm
    mm = np.where(holes | (mm <= 0), 0.0, np.clip(mm, 0.0, 65535.0))
    return np.asarray(mm, dtype=np.uint16)


def pgm_bytes(depth_mm: npt.NDArray[np.uint16]) -> bytes:
    """The depth image as the robot's depth_mm.pgm: P5, 16 bit big-endian, maxval 65535."""
    h, w = depth_mm.shape
    return f"P5\n{w} {h}\n65535\n".encode() + np.asarray(depth_mm, dtype=">u2").tobytes()


def read_pgm(data: bytes) -> npt.NDArray[np.uint16]:
    parts = data.split(b"\n", 3)
    if parts[0] != b"P5":
        raise ValueError("not a P5 PGM")
    w, h = (int(v) for v in parts[1].split())
    if int(parts[2]) != 65535:
        raise ValueError("expected a 16-bit PGM")
    return np.frombuffer(parts[3][: w * h * 2], dtype=">u2").reshape(h, w).astype(np.uint16)
