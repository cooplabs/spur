"""Phone photos -> the stills r1d's `measure.phone` / `measure.phone_calib` read (session 13; HW2 day 2's helper,
kept with that day's results).

    python tools/phone/prep_phone.py OUTDIR PHOTO... [--lens phone.json | --long-side 2016] [--quality 95]

What it does, and why (HW2 day 2's prep, results/hw2_20261013/STEPS.md "Before step 1"):
- **the lens model's size**: r1d refuses a still whose size differs from its intrinsics' width x height
  (measure_ops._phone) - a full-size iPhone photo (4032 x 3024) against the 2016 x 1512 lens model would be refused;
  `--lens FILE` takes the size from the lens model (phone.json's width / height), else the long side `--long-side`;
- **the SENSOR's own pixel layout**: iOS writes the same downward view with EXIF orientation 1, 3 or 6 depending on
  how the phone was held (day 2's photos from one phone: 1, 3 and 6), and OpenCV's decoder APPLIES that tag
  (cv2.imdecode rotates by it) - a still tagged 3 decodes upside down, its lens centre mirrored against the
  calibration. The stored pixels are kept as they are and the tag set to 1, so the calibration photos and every
  still agree however the phone was held;
- **one lens**: each photo's lens (EXIF LensModel + 35 mm focal length) is printed; a mix (0.5x and 1x) is refused -
  the lens model fits ONE lens (day 2: only the 0.5x saw every tag from the arm);
- HEIC (the iPhone's default) is converted with macOS `sips` first (this is a Mac-side tool: Pillow has no HEIC
  reader); JPEG / PNG are read directly.
r1d's request line now takes a still up to its schema's 40 MB (server.MAX_LINE), so the quality stays high (q95:
~1.5 MB at 2016 x 1512); day 2's 700 KB cap (the old 1 MiB line) is gone.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ORIENTATION = 0x0112  # EXIF tags
EXIF_IFD = 0x8769
LENS_MODEL = 0xA434
FOCAL_35MM = 0xA405
HEIC = (".heic", ".heif")


@dataclass(frozen=True)
class Prepared:
    name: str
    size: tuple[int, int]
    bytes: int
    orientation_was: int
    lens: str


def lens_size(path: Path) -> tuple[int, int]:
    d = json.loads(path.read_text())
    return int(d["width"]), int(d["height"])


def target_size(src: tuple[int, int], *, lens: tuple[int, int] | None, long_side: int) -> tuple[int, int]:
    """The output size: the lens model's (refused when the aspect or orientation differs), else the long side."""
    w, h = src
    if lens is not None:
        lw, lh = lens
        if (w >= h) != (lw >= lh):
            raise ValueError(f"the photo's stored pixels are {w} x {h}, the lens model {lw} x {lh}: rotated by 90 deg")
        if abs(w / h - lw / lh) > 0.01:
            raise ValueError(f"the photo's aspect {w / h:.3f} differs from the lens model's {lw / lh:.3f}")
        return lw, lh
    s = long_side / max(w, h)
    return round(w * s), round(h * s)


def prepare(im: Image.Image, *, lens: tuple[int, int] | None, long_side: int, quality: int) -> tuple[bytes, int, str]:
    """One photo (as stored: never transposed by its EXIF orientation) -> (JPEG bytes in the sensor frame at the target
    size with orientation 1, the orientation it had, its lens words)."""
    ex = im.getexif()
    orient = int(ex.get(ORIENTATION, 1))
    ifd = ex.get_ifd(EXIF_IFD)
    lens_words = f"{ifd.get(LENS_MODEL, '?')} (35 mm eq {ifd.get(FOCAL_35MM, '?')})"
    size = target_size(im.size, lens=lens, long_side=long_side)
    out = im.convert("RGB")
    if out.size != size:
        out = out.resize(size, Image.Resampling.LANCZOS)
    ex[ORIENTATION] = 1  # the stored (sensor) pixels as they are
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=quality, exif=ex.tobytes())
    return buf.getvalue(), orient, lens_words


def _open(src: Path, td: Path) -> Image.Image:
    if src.suffix.lower() in HEIC:
        if shutil.which("sips") is None:
            raise RuntimeError(f"{src.name}: HEIC needs macOS sips (export the photo as JPEG instead)")
        tmp = td / (src.stem + ".jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "100", str(src), "--out", str(tmp)],
                       check=True, capture_output=True)  # fmt: skip
        src = tmp
    im = Image.open(src)
    im.load()
    return im


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prep_phone.py", description=__doc__.split("\n")[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("photos", type=Path, nargs="+")
    ap.add_argument("--lens", type=Path, default=None, help="the lens model (phone.json): its width x height")
    ap.add_argument("--long-side", type=int, default=2016, help="without --lens: the long side in pixels")
    ap.add_argument("--quality", type=int, default=95)
    a = ap.parse_args(argv)
    lens = lens_size(a.lens) if a.lens else None
    a.outdir.mkdir(parents=True, exist_ok=True)
    done: list[Prepared] = []
    with tempfile.TemporaryDirectory() as td:
        for src in a.photos:
            data, orient, lens_words = prepare(_open(src, Path(td)), lens=lens, long_side=a.long_side,
                                               quality=a.quality)  # fmt: skip
            dst = a.outdir / (src.stem + ".jpg")
            dst.write_bytes(data)
            with Image.open(io.BytesIO(data)) as chk:
                p = Prepared(src.name, chk.size, len(data), orient, lens_words)
            done.append(p)
            print(f"{p.name}: {p.size[0]} x {p.size[1]}, {p.bytes // 1000} KB, iOS orientation was {orient} -> 1, "
                  f"{lens_words}")  # fmt: skip
    lenses = sorted({p.lens for p in done})
    print("LENSES:", "; ".join(lenses))
    if len(lenses) > 1:
        print("MIXED LENSES: the lens model fits one lens - take every photo with the same lens", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
