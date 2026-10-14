"""The placement correction for a hand-placed camera (DESIGN §8.2; the user: "the top phone might not be positioned
perfectly and there needs to be a correction algorithm").

The phone's lens is calibrated once (an `OpenCVLens`: K + distortion, e.g. from the printed ChArUco board); every still
then gets its OWN camera pose from the table tags it sees (a full 6-degree-of-freedom fit of the camera to the tags'
known positions on the table plane: `register.locate_entity` with the table as the entity and the camera as the
reference), so tilt, rotation, height and drift between trials are corrected automatically. The result maps any pixel
to the table plane - or to a plane `z` above it (the top face of an object: no parallax error) - and reports its own
error in mm: the fit residual on the plane and a leave-one-tag-out check (each tag predicted by a fit without it - the
error a point elsewhere on the table can expect). It refuses, with the reason, when too few tags are visible, the view
is too oblique, or the error is above the tolerance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .lenses import Lens
from .register import View, locate_entity
from .tags import TagObs, TagPlacement

FloatArray = npt.NDArray[np.float64]
MIN_TAGS = 3  # 3 spread tags: a 6-DoF fit with redundancy, and a leave-one-out check that still has 2 left
MAX_TILT_DEG = 45.0  # the optical axis vs straight down: beyond this the far table is foreshortened > 1.4x
MAX_FIT_MM = 2.0  # on-plane RMS of the tag corners: room under a typical 5 mm object-placement rule


@dataclass(frozen=True)
class PhoneCorrection:
    ok: bool
    reason: str
    T_table_cam: FloatArray | None = None
    lens: Lens | None = None
    used: tuple[int, ...] = ()
    fit_rms_mm: float = math.nan  # the tag corners' rays on the table plane vs their known positions
    fit_max_mm: float = math.nan
    loo_max_mm: float = math.nan  # leave-one-tag-out: the worst tag centre predicted without that tag
    tilt_deg: float = math.nan  # the optical axis vs straight down
    height_m: float = math.nan  # the camera above the table plane
    mm_per_px: float = math.nan  # at the table point under the image centre
    details: dict[str, Any] = field(default_factory=dict)

    def to_plane(self, px: npt.ArrayLike, z: float = 0.0) -> FloatArray:
        """Pixels (N x 2) -> table-frame points (N x 3) on the plane at height z (m) above the table."""
        assert self.T_table_cam is not None and self.lens is not None
        uv = self.lens.normalise(np.atleast_2d(np.asarray(px, dtype=np.float64)))
        d = (self.T_table_cam[:3, :3] @ np.c_[uv, np.ones(len(uv))].T).T
        o = self.T_table_cam[:3, 3]
        t = (z - o[2]) / d[:, 2]
        return np.asarray(o + t[:, None] * d)

    def to_image(self, P_table: npt.ArrayLike) -> FloatArray:
        assert self.T_table_cam is not None and self.lens is not None
        P = np.atleast_2d(np.asarray(P_table, dtype=np.float64))
        T = np.linalg.inv(self.T_table_cam)
        return self.lens.project((T[:3, :3] @ P.T).T + T[:3, 3])

    def as_dict(self) -> dict[str, Any]:
        r: dict[str, Any] = {
            k: (None if isinstance(v, float) and math.isnan(v) else round(v, 3) if isinstance(v, float) else v)
            for k, v in (
                ("ok", self.ok),
                ("reason", self.reason),
                ("fit_rms_mm", self.fit_rms_mm),
                ("fit_max_mm", self.fit_max_mm),
                ("loo_max_mm", self.loo_max_mm),
                ("tilt_deg", self.tilt_deg),
                ("height_m", self.height_m),
                ("mm_per_px", self.mm_per_px),
            )
        }
        r["used"] = list(self.used)
        if self.T_table_cam is not None:
            r["T_table_cam"] = np.round(self.T_table_cam, 6).tolist()
        return r

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> PhoneCorrection:
        """The inverse of `as_dict` WITHOUT the lens (r1d's `measure.phone` reply on the client side: the checks read
        ok / reason / the errors; pixels are mapped by r1d, which has the lens)."""

        def f(k: str) -> float:
            v = d.get(k)
            return math.nan if v is None else float(v)

        T = d.get("T_table_cam")
        return cls(
            bool(d["ok"]),
            str(d.get("reason", "")),
            None if T is None else np.asarray(T, dtype=np.float64),
            None,
            tuple(int(i) for i in d.get("used", ())),
            f("fit_rms_mm"),
            f("fit_max_mm"),
            f("loo_max_mm"),
            f("tilt_deg"),
            f("height_m"),
            f("mm_per_px"),
        )


def _plane_errors(
    T_table_cam: FloatArray, lens: Lens, tags: Mapping[int, TagObs], pl: Mapping[int, TagPlacement]
) -> FloatArray:
    errs = []
    for tid, t in tags.items():
        if tid not in pl:
            continue
        uv = lens.normalise(t.corners_px)
        d = (T_table_cam[:3, :3] @ np.c_[uv, np.ones(4)].T).T
        o = T_table_cam[:3, 3]
        P = pl[tid].corners_entity()
        tt = (P[:, 2] - o[2]) / d[:, 2]
        hit = o + tt[:, None] * d
        errs += list(1000 * np.linalg.norm(hit[:, :2] - P[:, :2], axis=1))
    return np.asarray(errs)


def correct_phone(
    tags: Mapping[int, TagObs],
    lens: Lens,
    table_tags: Sequence[TagPlacement],
    *,
    min_tags: int = MIN_TAGS,
    max_tilt_deg: float = MAX_TILT_DEG,
    max_fit_mm: float = MAX_FIT_MM,
) -> PhoneCorrection:
    """The phone's pose over the table from the table tags in one still (`table_tags`: their placements in the table
    frame, the table top = z 0). The pose of the TABLE in the camera frame is fitted (6 DoF), then inverted."""
    pl = {p.tag_id: p for p in table_tags}
    seen = {k: v for k, v in tags.items() if k in pl}
    if len(seen) < min_tags:
        missing = sorted(set(pl) - set(seen))
        return PhoneCorrection(
            False,
            f"only {len(seen)} table tags in the phone still (need {min_tags}); not seen: "
            + (", ".join(str(i) for i in missing) or "-")
            + " - re-aim the phone so more tags are in view",
            used=tuple(sorted(seen)),
        )
    reg = locate_entity(
        "table",
        list(pl.values()),
        [View(seen, lens, np.eye(4), "phone")],
        level=False,
        min_tags=min_tags,
        max_rms_px=1e9,
    )
    assert reg.T_ref_entity is not None
    T_table_cam = np.asarray(np.linalg.inv(reg.T_ref_entity), dtype=np.float64)
    err = _plane_errors(T_table_cam, lens, seen, pl)
    # leave one tag out: fit without it, predict its centre on the plane
    loo: list[float] = []
    if len(seen) >= 3:
        for k in seen:
            rest = {i: t for i, t in seen.items() if i != k}
            if len(rest) < 2:
                continue
            r2 = locate_entity(
                "table",
                list(pl.values()),
                [View(rest, lens, np.eye(4))],
                level=False,
                min_tags=2,
                max_rms_px=1e9,
                prior=reg.T_ref_entity,
            )
            assert r2.T_ref_entity is not None
            T_loo = np.asarray(np.linalg.inv(r2.T_ref_entity), dtype=np.float64)
            e = _plane_errors(T_loo, lens, {k: seen[k]}, pl)
            loo.append(float(np.mean(e)))
    axis = T_table_cam[:3, 2]  # the optical axis in the table frame
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-axis[2])))))
    h = float(T_table_cam[2, 3])
    centre = np.array([[(lens.width - 1) / 2.0, (lens.height - 1) / 2.0]])
    pc = PhoneCorrection(True, "ok", T_table_cam, lens)
    p0 = pc.to_plane(centre)[0]
    p1 = pc.to_plane(centre + np.array([[1.0, 0.0]]))[0]
    mmpp = 1000 * float(np.linalg.norm(p1 - p0))
    fit_rms = float(np.sqrt(np.mean(err**2)))
    fit_max = float(np.max(err))
    loo_max = float(max(loo)) if loo else math.nan
    reasons = []
    if tilt > max_tilt_deg:
        reasons.append(f"the phone looks {tilt:.0f} deg off straight down (max {max_tilt_deg:.0f}): too oblique")
    if fit_rms > max_fit_mm:
        reasons.append(f"its fit error is {fit_rms:.1f} mm on the table (max {max_fit_mm:.1f})")
    if not math.isnan(loo_max) and loo_max > 2 * max_fit_mm:
        reasons.append(
            f"a tag left out is predicted {loo_max:.1f} mm off (max {2 * max_fit_mm:.1f}): move a tag / re-aim"
        )
    ok = not reasons
    return PhoneCorrection(
        ok,
        "ok" if ok else "; ".join(reasons),
        T_table_cam,
        lens,
        tuple(sorted(seen)),
        fit_rms,
        fit_max,
        loo_max,
        tilt,
        h,
        mmpp,
        {"rms_px": reg.rms_px, "loo_mm": loo},
    )
