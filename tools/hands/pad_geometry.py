"""Where the real Dex3-1 finger pads are, measured on Unitree's link meshes (session 4; r1sim.dex3 "real" pad layout).

The hand photos (sim_capture/20261010_hand_photos IMG_8775 / IMG_8779, 10 Oct) show a raised rectangular block on
the palm side of each PROXIMAL phalanx (the 2x2 pads) and a smooth bullet-shaped DISTAL phalanx with no raised pad (the
3x1 tip arrays sit under its shell). This tool measures both on the STLs (link frame, metres; the fingers run along
+x with the grasp side +g y, g = +1 right / -1 left; the thumb links run along g y with the grasp side +x):

* the proximal block: the mesh vertices standing more than BLOCK_MIN_M out on the grasp side -> its extent along the
  phalanx, across it and its face (the outermost grasp-side point);
* the distal shell: the grasp-side surface on the phalanx's centre line at the S1 tip pad's centre (along 0.034 m)
  by a ray cast (MeshRay, tools/hands/compare_collision.py) and the profile along the phalanx.

Usage: uv run python tools/hands/pad_geometry.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_collision import MeshRay, stl_mesh  # noqa: E402
from r1sim import dex3  # noqa: E402

FloatArray = npt.NDArray[np.float64]
BLOCK_MIN_M = 0.0135  # the block's face is at ~14.6 mm; the rest of the phalanx's grasp side stays below ~12.6 mm
TIP_PAD_ALONG_M = 0.034  # the S1 tip pad centre along the distal phalanx (dex3.hand_xml)
PROFILE_ALONG_M = (0.016, 0.020, 0.024, 0.028, 0.032, 0.034, 0.036, 0.038, 0.040, 0.042, 0.044, 0.046, 0.048, 0.050)


def _frame(side: str, link: str) -> tuple[int, int, int, float]:
    """(along axis, grasp axis, width axis, sign of the along axis) for a finger / thumb link."""
    g = -1.0 if side == "left" else 1.0
    if link.startswith("thumb"):
        return 1, 0, 2, g
    return 0, 1, 2, 1.0


def proximal_block(side: str, link: str = "index_0") -> dict[str, float]:
    """The raised pad block of a proximal phalanx (index_0 / middle_0 / thumb_1): along lo / hi, the width's
    lo / hi, its half width and the face (m from the joint axis, the grasp side)."""
    v, _ = stl_mesh(dex3.mesh_file(side, link))
    al, gr, wd, sa = _frame(side, link)
    gs = -1.0 if (side == "left" and not link.startswith("thumb")) else 1.0  # the grasp side's sign
    out = gs * v[:, gr]
    blk = v[out > BLOCK_MIN_M]
    along = sa * blk[:, al]
    return {
        "along_lo": float(along.min()),
        "along_hi": float(along.max()),
        "width_lo": float(blk[:, wd].min()),
        "width_hi": float(blk[:, wd].max()),
        "half_width": float((blk[:, wd].max() - blk[:, wd].min()) / 2),
        "face": float(out.max()),
    }


def distal_shell(side: str, link: str = "index_1") -> dict[str, Any]:
    """The distal phalanx's grasp-side shell on its centre line: the face at the S1 tip pad's centre and the profile
    along the phalanx (m from the joint axis; None where the ray misses)."""
    al, gr, _, sa = _frame(side, link)
    gs = -1.0 if (side == "left" and not link.startswith("thumb")) else 1.0
    ray = MeshRay(dex3.mesh_file(side, link))
    n = np.zeros(3)
    n[gr] = gs

    def face(a: float) -> float | None:
        p = np.zeros(3)
        p[al] = sa * a
        return ray.surface_along(p, n)

    return {
        "face_at_tip_pad_centre": face(TIP_PAD_ALONG_M),
        "profile": {f"{a:.3f}": face(a) for a in PROFILE_ALONG_M},
        "reach": float(np.max(sa * stl_mesh(dex3.mesh_file(side, link))[0][:, al])),
    }


def measure() -> dict[str, Any]:
    """Both hands, every pad-carrying phalanx (Unitree ships identical index / middle meshes per side)."""
    out: dict[str, Any] = {}
    for side in dex3.SIDES:
        out[side] = {
            "index_0": proximal_block(side, "index_0"),
            "thumb_1": proximal_block(side, "thumb_1"),
            "index_1": distal_shell(side, "index_1"),
            "thumb_2": distal_shell(side, "thumb_2"),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    r = measure()
    for side, links in r.items():
        for link, v in links.items():
            if "face" in v:
                print(
                    f"{side} {link} block: along {1000 * v['along_lo']:.1f}..{1000 * v['along_hi']:.1f} mm, width "
                    f"{1000 * v['width_lo']:.1f}..{1000 * v['width_hi']:.1f} mm, face {1000 * v['face']:.2f} mm"
                )
            else:
                prof = ", ".join(
                    f"{1000 * float(k):.0f}: {'-' if x is None else f'{1000 * x:.2f}'}" for k, x in v["profile"].items()
                )
                print(f"{side} {link} shell face at the tip pad centre {1000 * v['face_at_tip_pad_centre']:.2f} mm;")
                print(f"   profile (along mm: face mm) {prof}; reach {1000 * v['reach']:.1f} mm")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
