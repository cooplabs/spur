"""tools/phone/prep_phone.py (session 13; HW2 day 2's helper into the repo): a phone photo -> a still r1d reads in the
SENSOR's pixel layout at the lens model's size. The trap it closes: OpenCV's decoder (r1mc3.measure.capture.
decode_gray) APPLIES the EXIF orientation - an iPhone photo tagged 6 (held portrait) or 3 (upside down) decodes rotated
against the lens calibration (the centre mirrored), and a full-size photo is refused against a half-size lens model."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image
from r1mc3.measure.capture import decode_gray

REPO = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prep_phone", REPO / "tools" / "phone" / "prep_phone.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prep_phone"] = mod
    spec.loader.exec_module(mod)
    return mod


def _photo(w: int, h: int, orientation: int, lens: str = "iPhone back triple camera 2.22mm f/2.2") -> bytes:
    """A 'phone photo' whose stored pixels are landscape w x h: dark left half, a bright top-left marker - so a decoder
    that rotates it is caught - tagged with an EXIF orientation and a lens."""
    a = np.zeros((h, w, 3), np.uint8)
    a[:, w // 2 :] = 200
    a[: h // 8, : w // 8] = 255
    im = Image.fromarray(a)
    ex = im.getexif()
    ex[0x0112] = orientation
    ifd = ex.get_ifd(0x8769)
    ifd[0xA434] = lens
    ifd[0xA405] = 14
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95, exif=ex.tobytes())
    return buf.getvalue()


@pytest.mark.parametrize("orientation", [1, 3, 6])
def test_the_still_decodes_in_the_sensor_frame_at_the_lens_size(orientation: int) -> None:
    tool = _tool()
    raw = _photo(1008, 756, orientation)
    before = decode_gray(raw)
    if orientation == 6:  # what r1d saw without the helper: rotated to portrait (refused: not the lens model's size)
        assert before.shape == (1008, 756)
    if orientation == 3:  # ... or upside down at the right size (accepted, the lens centre mirrored)
        assert before.shape == (756, 1008) and before[-5, -5] > 240 and before[5, 5] < 240
    data, was, lens = tool.prepare(Image.open(io.BytesIO(raw)), lens=(504, 378), long_side=2016, quality=95)
    g = decode_gray(data)
    assert was == orientation and "2.22mm" in lens
    assert g.shape == (378, 504)  # the lens model's height x width
    assert g[3, 3] > 240 and g[3, -3] > 150 and g[-3, 3] < 60  # the marker top-left, bright right half: as stored


def test_a_rotated_or_other_aspect_photo_is_refused_against_the_lens() -> None:
    tool = _tool()
    with pytest.raises(ValueError, match="rotated by 90 deg"):
        tool.target_size((756, 1008), lens=(2016, 1512), long_side=2016)
    with pytest.raises(ValueError, match="aspect"):
        tool.target_size((1920, 1080), lens=(2016, 1512), long_side=2016)
    assert tool.target_size((4032, 3024), lens=None, long_side=2016) == (2016, 1512)


def test_the_cli_writes_the_stills_and_refuses_mixed_lenses(tmp_path: Path) -> None:
    tool = _tool()
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(_photo(1008, 756, 6))
    b.write_bytes(_photo(1008, 756, 1, lens="iPhone back triple camera 6.93mm f/1.48"))
    lens = tmp_path / "phone.json"
    lens.write_text(json.dumps({"width": 504, "height": 378, "K": [[400, 0, 252], [0, 400, 189], [0, 0, 1]]}))
    out = tmp_path / "out"
    assert tool.main([str(out), str(a), "--lens", str(lens)]) == 0
    assert decode_gray((out / "a.jpg").read_bytes()).shape == (378, 504)
    assert tool.main([str(out), str(a), str(b), "--lens", str(lens)]) == 2  # 0.5x + 1x: one lens model cannot fit
