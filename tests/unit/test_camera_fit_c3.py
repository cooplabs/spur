"""The lens fit on HW2 day 1's REAL ChArUco recording (C3, 12 Oct 2026; the detections in tests/assets/cam_hw2/).

Day 1's fit reported f 150.05 px with the principal point at the exact image centre (271.9, 223.9) and RMS 1.14 px over
1935 views - read as "Unitree's stereo service sends rectified images; the installed lens (f 126.94) is 18 % off".
Those numbers are the fit's STARTING guess (K0: f 150, the image centre): ONE view of 6 corners lying along a line
(minor-axis std 4.2 px) made OpenCV's calibrateCamera stop at its start. Without it the same recording converges to
f ~125 px, RMS ~0.34 px from any start - within ~1.6 % / ~1.3 px of the installed lens (the 4 Oct capture used the same
stream: "left udp 5002"). Every test here fails on the base 1cf3afd (noted per test)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1sim.sysid import camera_fit as cf

pytest.importorskip("cv2")
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "cam_hw2"
DETS = ASSETS / "c3_charuco_detections.npz"
RUN = ASSETS / "c3_charuco_run.json"


def _board() -> Any:
    from r1sim import charuco_render as cr

    return next(b for b in cr.CANDIDATES if b.name == "poster_6x4_90")


def _views() -> list[tuple[np.ndarray, np.ndarray]]:
    from r1sim import charuco_render as cr

    b = _board()
    C = np.c_[cr.corners_mm(b) * cr.MM, np.zeros(b.n_corners)]
    return [(C[i].astype(np.float64), p) for _t, i, p in cf.load_detections(DETS)]


def test_the_whole_c3_recording_converges_once_the_degenerate_view_is_dropped() -> None:
    """Base: f 150.05, cx 271.9, cy 223.9, RMS 1.14 (the start, not a fit) and b5_pass False."""
    v = _views()
    assert len(v) == 1935
    res = cf.fit_lens(v, kinds=("pinhole_k1",), loo=False)
    m = res["models"]["pinhole_k1"]
    assert m["converged"] and res["views_dropped"] >= 1 and res["b5_pass"]
    assert 124.0 < m["f"] < 126.5 and 257.0 < m["cx"] < 259.0 and 204.0 < m["cy"] < 206.5, m
    assert m["rms_px"] < 0.4 and abs(m["dist"][0]) < 0.02


def test_the_degenerate_view_is_the_one_that_stopped_the_fit() -> None:
    v = _views()
    bad = [k for k, (_o, p) in enumerate(v) if not cf.view_usable(p) and len(p) >= cf.VIEW_MIN_CORNERS]
    sv = np.linalg.svd(v[859][1] - v[859][1].mean(axis=0), compute_uv=False)
    assert not cf.view_usable(v[859][1]) and sv[1] / np.sqrt(len(v[859][1])) < cf.VIEW_MIN_SPREAD_PX
    assert len(bad) < 20  # only a handful of the usable-size views lie along a line
    # the old path (every view, no filter) on the even frames + 859 is stuck at its start; the same views WITH the
    # filter (session 12's path: 859 and the other line-like views dropped) converge. (The integrator, from the Linux
    # check: unfiltered, the even frames WITHOUT 859 converge on macOS's OpenCV but stop at f 137.8 on Linux's - other
    # line-like views among them trap that build - so "859 alone" is platform-dependent; the filter is not.)
    even = [v[k] for k in range(0, len(v), 2)]
    rms_ok, K_ok, *_ = cf._calibrate([x for x in even if cf.view_usable(x[1])], "pinhole_k1")
    rms_bad, K_bad, *_ = cf._calibrate([*even, v[859]], "pinhole_k1")
    # stuck AT its start (Linux's OpenCV moves 0.21 px off it, macOS's < 0.1; the converged f is 25 px away)
    assert abs(K_bad[0, 0] - cf.K0_F) < 1.0 and abs(K_ok[0, 0] - 125.0) < 1.5 and rms_ok < 0.4 < rms_bad


def test_the_hold_still_frames_from_the_plans_timeline() -> None:
    """Base: no hold-still selection (`hold_windows` / `still_views` did not exist)."""
    t0, wins = cf.hold_windows("unused.mcap", RUN)
    assert len(wins) == 81 and wins[0] == (2.0, 4.0)
    dets = cf.load_detections(DETS)
    views, sel = cf.still_views(dets, t0, wins, _board())
    assert sel["holds"] == 81 and sel["holds_used"] >= 30 and sel["views"] >= 60
    m = cf.fit_lens(views, kinds=("pinhole_k1",), loo=False)["models"]["pinhole_k1"]
    assert m["converged"] and 124.0 < m["f"] < 126.5 and m["rms_px"] < 0.5  # B5 on the hold-still frames: met
