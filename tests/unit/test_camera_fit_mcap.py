"""The HW2 camera fits read r1cam's RECORDINGS (session 11: `camera_fit vignetting | depth | latency`, the runbook's
C2 / C4 / C5 commands): synthetic frames from a known model are written by r1cam's own recorder (r1mc3.cam.recorder,
the robot's MCAP format) - and the head yaw by r1log's format - and each subcommand must give the model back. Base
(1e55bdd): camera_fit has only `lens` / `synthetic` (no MCAP reader for these: AttributeError)."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.cam.frames import Frame
from r1mc3.cam.recorder import CamRecorder
from r1mc3.cam.source import SourceBase
from r1sim import lens
from r1sim.sysid import camera_fit as CF
from r1sim.sysid import records as R

cv2 = pytest.importorskip("cv2")
W, H = 544, 448


class _Src(SourceBase):
    name = "synthetic"


def _jpeg(gray: np.ndarray) -> bytes:
    from PIL import Image

    b = io.BytesIO()
    Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).save(b, format="JPEG", quality=90)
    return b.getvalue()


def _record(tmp: Path, kind: str, frames: list[Frame], params: dict[str, Any], streams: tuple[str, ...]) -> Path:
    src = _Src(streams=streams)
    rec = CamRecorder(src, tmp, kind=kind, label=str(params.get("label", "")), params=params, streams=streams)
    rec.start()
    for f in frames:
        src.publish(f)
    rec.stop()
    return rec.mcap_path


def test_vignetting_from_a_flat_field_recording(tmp_path: Path) -> None:
    M = lens.load_calibration().model
    true = (-0.05, 0.01, -0.40)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    x = np.hypot(xx - M.cx, yy - M.cy) / 375.23  # the model file's r_max
    g = 1 + true[0] * x**2 + true[1] * x**4 + true[2] * x**6
    rng = np.random.default_rng(0)
    frames = [Frame("left", _jpeg(level * g + rng.normal(0, 1.5, g.shape)), 1_000_000_000 + k * 133_000_000, k + 1)
              for k, level in enumerate([120.0, 120.0, 180.0, 180.0, 230.0, 230.0])]  # fmt: skip
    p = _record(tmp_path, "flat_field", frames, {}, ("left",))
    res = CF.vignetting_from_mcap([p])
    xs = np.linspace(0.0, 1.0, 51)
    fit = 1 + res["a2"] * xs**2 + res["a4"] * xs**4 + res["a6"] * xs**6
    ref = 1 + true[0] * xs**2 + true[1] * xs**4 + true[2] * xs**6
    assert res["frames"] == 6 and float(np.max(np.abs(fit - ref))) < 0.02, res
    assert "model_now" in res and res["gain_max_diff_vs_model"] > 0.0


def test_depth_from_depth_target_recordings(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    paths = []
    for z, sig in ((0.45, 0.004), (0.60, 0.007)):
        frames = []
        for k in range(8):
            d = z + 0.0002 * (np.mgrid[0:H, 0:W][1] - 272) + rng.normal(0, sig, (H, W))
            mm = np.round(d * 1000.0)
            mm[rng.random(mm.shape) < 0.1] = 0  # holes
            frames.append(Frame("depth", mm.astype("<u2").tobytes(), 2_000_000_000 + k * 100_000_000, k + 1))
        paths.append(_record(tmp_path, "depth_targets", frames, {"label": f"board_{z}", "distance_m": z},
                             ("depth",)))  # fmt: skip
    res = CF.depth_from_mcap(paths)
    rows = res["recordings"]
    assert [r["distance_m"] for r in rows] == [0.45, 0.60] and all(r["usable"] == 8 for r in rows)
    assert abs(rows[0]["sigma_z_m"] - 0.004) < 0.0007 and abs(rows[1]["sigma_z_m"] - 0.007) < 0.0007
    assert all(abs(r["median_err_m"]) < 0.002 and 0.85 < r["valid_fraction"] < 0.95 for r in rows)
    assert set(res["bands"]) == {"0.45", "0.6"}
    # a recording without a tape distance says so
    p0 = _record(tmp_path, "depth_targets", frames[:2], {"label": "x"}, ("depth",))
    assert "error" in CF.depth_from_mcap([p0])["recordings"][0]
    assert CF.depth_from_mcap([p0], [0.6])["recordings"][0]["usable"] == 2


def test_latency_from_a_head_yaw_step_recording_and_r1log(tmp_path: Path) -> None:
    from r1sim import charuco_render as cr

    b = cr.CHOSEN
    board = cr.opencv_board(b)
    img = np.asarray(board.generateImage((300, 200), marginSize=10), dtype=np.float64)
    t0_ns = 5_000_000_000
    t_j = np.arange(0.0, 4.0, 0.002)
    yaw = np.where(t_j > 1.5, 0.15, 0.0) + np.where(t_j > 2.8, -0.1, 0.0) + 0.03 * np.sin(2 * math.pi * 0.9 * t_j)
    true_lag = 0.085
    frames = []
    for k, t in enumerate(np.arange(0.3, 4.0, 1 / 7.5)):
        shift = 120.0 - 400.0 * float(np.interp(t - true_lag, t_j, yaw))  # the board moves left as the head turns
        canvas = np.full((H, W), 235.0)
        M = np.array([[1.0, 0.0, shift], [0.0, 1.0, 120.0]], dtype=np.float32)
        canvas = cv2.warpAffine(img, M, (W, H), dst=canvas, borderMode=cv2.BORDER_TRANSPARENT)
        frames.append(Frame("left", _jpeg(canvas), t0_ns + round(t * 1e9), k + 1))
    cam = _record(tmp_path, "depth_targets", frames, {"label": "latency_board", "distance_m": 0.6}, ("left",))
    fields = {f: np.zeros((t_j.size, 26)) for f in R.BODY_FIELDS}
    from r1sim import joints as J

    fields["q"][:, J.IDX["head_yaw"]] = yaw
    log = tmp_path / "r1log.mcap"
    R.write_body_mcap(log, R.body_from_arrays(t_j, fields), t0_ns=t0_ns)
    res = CF.latency_from_mcap(cam, [log], b)
    assert "error" not in res, res
    assert abs(res["latency_ms"] - 85.0) <= 8.0 and res["frames"] > 20 and res["head_yaw_span_rad"] > 0.1, res
    # the clocks must agree: a 2 s offset not given -> the scan cannot find it (a wrong latency or no overlap)
    off = CF.latency_from_mcap(cam, [log], b, offset_ns=-int(2e9))
    assert "error" in off or abs(off["latency_ms"] - 85.0) > 8.0


def test_the_cli_words(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    frames = [Frame("depth", np.full((H, W), 450, "<u2").tobytes(), 1_000_000_000 + k, k + 1) for k in range(3)]
    p = _record(tmp_path, "depth_targets", frames, {"label": "flat", "distance_m": 0.45}, ("depth",))
    assert CF.main(["depth", "--mcap", str(p)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["recordings"][0]["median_err_m"] == 0.0
