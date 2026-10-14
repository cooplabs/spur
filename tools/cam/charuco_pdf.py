"""The ChArUco board as a print-ready VECTOR PDF and a print-ready PNG (session 9; docs/hardware/CHARUCO.md).

    uv run python tools/cam/charuco_pdf.py [--board poster_6x4_90] [--png] [--dpi 300] [--out DIR]

Every black area is an exact rectangle (`r1sim.charuco_render.board_rects`, read from OpenCV's own board drawing), so
the PDF is exact at any printer resolution (no raster scaling). The page = the paper (landscape), the grid + its label
block centred on it (>= 30 mm of white above the grid), below the grid the label (board id, layout, dictionary,
"print at 100 %") and a 100 mm check bar with 10 mm ticks. Hand-written PDF 1.4 (one page, the built-in Helvetica): no
new dependency. `--png` (the user's print service takes only JPG / PNG, 11 Oct ~23:20): the same page rasterised at
`--dpi` (300: 9000 x 6000 px for 20 x 30 in) with the dpi in the file, every edge rounded to the nearest pixel
(<= 0.5 px = 0.04 mm at 300 dpi); the text in OpenCV's Hershey font; written by Pillow (a dev dependency: a tool).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))

from r1sim import charuco_render as cr  # noqa: E402

PT_PER_MM = 72.0 / 25.4
TOP_MM = 30.0  # white above the grid


BLOCK_BELOW_MM = 40.0  # the check bar + the label lines under the grid


def layout(b: cr.BoardSpec) -> tuple[float, float]:
    """The grid's top-left corner on the page (mm from the page's top-left): centred left-right; the grid + the label
    block centred top-bottom, >= TOP_MM of white above the grid."""
    return (b.paper.w_mm - b.width_mm) / 2.0, max(TOP_MM, (b.paper.h_mm - b.height_mm - BLOCK_BELOW_MM) / 2.0)


def default_name(b: cr.BoardSpec, ext: str) -> str:
    """e.g. r1mc3_charuco_poster_6x4_90_20x30in.png (the paper in the name: the size to order)."""
    return f"r1mc3_charuco_{b.name}_{round(b.paper.h_mm / 25.4)}x{round(b.paper.w_mm / 25.4)}in.{ext}"


def page_rects_mm(b: cr.BoardSpec) -> list[tuple[float, float, float, float]]:
    """Every black rectangle on the page (x, y from the top, w, h; mm): the board, the 100 mm bar, its 11 ticks."""
    gx, gy = layout(b)
    out = [(gx + r.x, gy + r.y, r.w, r.h) for r in cr.board_rects(b)]
    bx, by = gx, gy + b.height_mm + 14.0
    out.append((bx, by, 100.0, 1.5))
    for k in range(11):
        tall = 5.0 if k in (0, 5, 10) else 3.0
        out.append((bx + 10.0 * k - 0.15, by - tall, 0.3, tall))
    return out


def raster(b: cr.BoardSpec, dpi: float = 300.0) -> Any:
    """The page as a uint8 image (white 255 / black 0) at `dpi`: the PDF's rectangles, each edge rounded to the nearest
    pixel (abutting rectangles share their rounded edge: no hairlines), the label in OpenCV's Hershey font."""
    import cv2
    import numpy as np

    ppm = dpi / 25.4
    img = np.full((round(b.paper.h_mm * ppm), round(b.paper.w_mm * ppm)), 255, dtype=np.uint8)
    for x, y, w, h in page_rects_mm(b):
        img[round(y * ppm) : round((y + h) * ppm), round(x * ppm) : round((x + w) * ppm)] = 0
    gx, gy = layout(b)
    by = gy + b.height_mm + 14.0
    scale = 2.6 * ppm / 22.0  # Hershey simplex: ~22 px cap height at scale 1 -> ~2.6 mm letters
    th = max(1, round(0.3 * ppm))

    def put(x_mm: float, y_mm: float, s: str) -> None:
        cv2.putText(img, s, (round(x_mm * ppm), round(y_mm * ppm)), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, th, cv2.LINE_AA)

    put(gx + 104.0, by + 1.5, "100 mm")
    for i, line in enumerate(label_lines(b)):
        put(gx, by + 10.0 + 5.0 * i, line)
    return img


def write_png(b: cr.BoardSpec, path: Path, dpi: float = 300.0) -> Path:
    from PIL import Image

    Image.fromarray(raster(b, dpi)).save(path, dpi=(dpi, dpi), optimize=True)
    return path


def label_lines(b: cr.BoardSpec) -> list[str]:
    spans = (b.cols - 1) // 2 * 2 + 1  # an odd number of squares from a black square's left edge to a black right edge
    return [
        f"r1mc3 ChArUco board {b.name}: {b.cols} x {b.rows} squares of {b.square_mm:.1f} mm, ArUco markers "
        f"{b.marker_mm:.1f} mm, OpenCV {b.dict_name} (ids 0-{b.cols * b.rows // 2 - 1}), top-left square black",
        f"PRINT AT 100 % / ACTUAL SIZE on {b.paper.label}, matte or satin (not glossy / metallic), black on white - "
        f"never 'fit to page' or crop. Glue flat on a rigid board, no lamination.",
        f"CHECK with a ruler: one square = {b.square_mm:.1f} mm; along the top row, from the left edge of the first "
        f"black square to the right edge of black square {spans // 2 + 1} = {spans * b.square_mm:.1f} mm (+-0.5); "
        f"the bar below = 100.0 mm. Write the measured values on the back.",
    ]


def pdf_bytes(b: cr.BoardSpec) -> bytes:
    W, H = b.paper.w_mm * PT_PER_MM, b.paper.h_mm * PT_PER_MM
    gx, gy = layout(b)

    def rect(x_mm: float, y_mm: float, w_mm: float, h_mm: float) -> str:
        # page mm (y down from the top) -> PDF points (y up from the bottom)
        return (
            f"{x_mm * PT_PER_MM:.4f} {H - (y_mm + h_mm) * PT_PER_MM:.4f} "
            f"{w_mm * PT_PER_MM:.4f} {h_mm * PT_PER_MM:.4f} re"
        )

    # ONE path, ONE fill: abutting rectangles filled separately show anti-aliasing hairlines at their shared edges;
    # the board + the 100 mm check bar (a 1.5 mm bar + ticks every 10 mm: 3 mm, 5 mm at 0 / 50 / 100) = page_rects_mm
    ops = ["0 g"]
    ops += [rect(*r) for r in page_rects_mm(b)]
    bx, by = gx, gy + b.height_mm + 14.0
    ops.append("f")

    def text(x_mm: float, y_mm: float, size: float, s: str) -> str:
        s = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return f"BT /F1 {size:g} Tf {x_mm * PT_PER_MM:.3f} {H - y_mm * PT_PER_MM:.3f} Td ({s}) Tj ET"

    ops.append(text(bx + 104.0, by + 1.5, 10, "100 mm"))
    for i, line in enumerate(label_lines(b)):
        ops.append(text(gx, by + 9.0 + 5.0 * i, 9, line))
    content = "\n".join(ops).encode("ascii")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W:.4f} {H:.4f}] /Contents 4 0 R "
        f"/Resources << /Font << /F1 5 0 R >> >> >>".encode(),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Title (r1mc3 ChArUco {b.name}) /Producer (r1mc3 tools/cam/charuco_pdf.py) >>".encode(),
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R /Info 6 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def parse_rects_mm(pdf: bytes) -> tuple[tuple[float, float], list[tuple[float, float, float, float]]]:
    """(page size mm, every filled rectangle as (x, y-from-top, w, h) mm) - for the tests."""
    _head, _, rest = pdf.partition(b"/MediaBox [0 0 ")
    w_pt, h_pt = (float(v) for v in rest.split(b"]")[0].split())
    body = pdf.split(b"stream\n", 1)[1].split(b"\nendstream", 1)[0].decode()
    rects = []
    for line in body.splitlines():
        if line.endswith(" re"):
            x, y, w, h = (float(v) / PT_PER_MM for v in line.split()[:4])
            rects.append((x, h_pt / PT_PER_MM - y - h, w, h))
    return (w_pt / PT_PER_MM, h_pt / PT_PER_MM), rects


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", default=cr.CHOSEN.name)
    ap.add_argument("--out", default="", help="the folder (default docs/hardware/charuco)")
    ap.add_argument("--png", action="store_true", help="also the PNG (the print services that take only images)")
    ap.add_argument("--dpi", type=float, default=300.0)
    a = ap.parse_args(argv)
    b = next(x for x in cr.CANDIDATES if x.name == a.board)
    d = Path(a.out or REPO / "docs" / "hardware" / "charuco")
    d.mkdir(parents=True, exist_ok=True)
    out = d / default_name(b, "pdf")
    out.write_bytes(pdf_bytes(b))
    print(f"{out} ({out.stat().st_size} bytes): {b.cols} x {b.rows} x {b.square_mm} mm on {b.paper.label}")
    if a.png:
        png = write_png(b, d / default_name(b, "png"), a.dpi)
        print(f"{png} ({png.stat().st_size} bytes): {a.dpi:g} dpi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
