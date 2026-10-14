"""The ChArUco board the user prints (session 9; docs/hardware/CHARUCO.md): the geometry IS OpenCV's board, and the
committed PDF is exactly the generator's output at the stated size."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from r1sim import charuco_render as cr

REPO = Path(__file__).resolve().parents[2]
CHARUCO_DIR = REPO / "docs" / "hardware" / "charuco"


def _pdf_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("charuco_pdf", REPO / "tools" / "cam" / "charuco_pdf.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_rectangles_are_opencvs_board_every_corner_at_its_id() -> None:
    for b in (cr.CHOSEN, cr.DRAFT):
        img = cr.rasterize(b, cr.board_rects(b), px_per_mm=4.0, margin_mm=20.0)
        ids, px = cr.detect(img, b)
        assert sorted(ids.tolist()) == list(range(b.n_corners))
        truth = (cr.corners_mm(b) + 20.0) * 4.0
        assert float(np.abs(px - truth[ids]).max()) < 0.6  # px at 4 px / mm = 0.15 mm


def test_the_committed_pdf_is_the_generators_output_at_100_percent() -> None:
    tool = _pdf_tool()
    b = cr.CHOSEN
    data = tool.pdf_bytes(b)
    pdf = CHARUCO_DIR / tool.default_name(b, "pdf")
    assert pdf.name == "r1mc3_charuco_poster_6x4_90_20x30in.pdf"  # the user's print size (11 Oct: 20 x 30 in only)
    assert pdf.read_bytes() == data  # regenerate with: uv run python tools/cam/charuco_pdf.py --png
    (w, h), rects = tool.parse_rects_mm(data)
    assert abs(w - b.paper.w_mm) < 0.01 and abs(h - b.paper.h_mm) < 0.01
    gx, gy = tool.layout(b)
    board = [(r.x + gx, r.y + gy, r.w, r.h) for r in cr.board_rects(b)]
    assert len(rects) == len(board) + 1 + 11  # the board + the 100 mm bar + its 11 ticks
    assert np.allclose(np.array(rects[: len(board)]), np.array(board), atol=1e-3)
    bar = rects[len(board)]
    assert abs(bar[2] - 100.0) < 1e-3  # the check bar is 100.0 mm long
    mx, _my = b.margins_mm()
    assert mx >= 15.0 and gy >= 15.0 and h - gy - b.height_mm >= 15.0  # white around the grid
    assert b"PRINT AT 100 %" in data and f"{b.square_mm:.1f} mm".encode() in data


def test_the_committed_png_is_the_page_at_300_dpi_and_its_squares_measure_90_mm() -> None:
    """The user's print service takes only JPG / PNG at 20 x 30 in (11 Oct ~23:20): the PNG = the generator's raster
    of the same page (9000 x 6000 px, 300 dpi in the file); OpenCV finds every corner at its id and the squares are
    90.0 mm at that dpi - full size and after a 0.4x reduction (what a print service's preview / resampler sees)."""
    import cv2
    from PIL import Image

    tool = _pdf_tool()
    b = cr.CHOSEN
    png = CHARUCO_DIR / tool.default_name(b, "png")
    im = Image.open(png)
    assert im.size == (9000, 6000) and im.mode == "L"
    assert all(abs(float(v) - 300.0) < 0.01 for v in im.info["dpi"])
    a = np.asarray(im, dtype=np.uint8)
    assert np.array_equal(a, tool.raster(b, 300.0))  # regenerate with: uv run python tools/cam/charuco_pdf.py --png
    ppm = 300.0 / 25.4
    for f in (1.0, 0.4):
        g = a if f == 1.0 else np.asarray(cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), np.uint8)
        ids, px = cr.detect(g, b)
        assert sorted(ids.tolist()) == list(range(b.n_corners))
        c = dict(zip(ids.tolist(), px / f, strict=True))
        n = b.cols - 1
        dx = [np.linalg.norm(c[i + 1] - c[i]) / ppm for i in c if (i + 1) % n != 0]
        dy = [np.linalg.norm(c[i + n] - c[i]) / ppm for i in c if i + n in c]
        assert max(abs(d - b.square_mm) for d in dx + dy) < (0.01 if f == 1.0 else 0.05)
    ys, xs = np.where(a < 128)  # >= 50 mm of white all round: a print service's trim cannot reach the board
    assert min(xs.min(), ys.min(), a.shape[1] - xs.max(), a.shape[0] - ys.max()) / ppm > 50.0
