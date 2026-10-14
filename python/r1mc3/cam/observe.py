"""Observations: a head frame from r1cam + the camera's pose in `base` at capture, kept under an obs_id (DESIGN §8.1;
v2 headcam.py's contract: a pixel always refers to the frame the client actually saw).

* The calibration (`HeadCalibration`): calib_head.py's JSON (`T_site_cam` 4x4 = the camera's optical frame in the
  head_camera site's frame, `model` = the lens dict, + provenance fields). On the robot `/etc/r1mc3/head_camera.json`
  (rtconfig writes it like r1rt.cfg); `R1MC3_HEAD_CALIB` overrides. Without one, pointing refuses (v2's rule: "the head
  camera is not calibrated"); observe still answers (the image, `calibration.geometry = "not calibrated"`).
* The pose at capture: the head_camera site from the measured joints + the IMU's roll / pitch (the IMU mount rotation
  undone - `r1mc3.world.Geometry.set_stance`, the planner's own geometry, so a pointed point and a planned hand use the
  same base frame), at the frame's receive time minus the camera latency (`latency_s`, unknown until HW2: 0), taken
  from a short history of measured states r1d samples each tick (the nearest sample at or before that time; the
  state at the request when the history does not reach back - v2 used the state at capture).
* `ObservationStore`: the last `keep` observations (v2 KEEP_FRAMES 8); an unknown / expired id is refused.
"""

from __future__ import annotations

import base64
import collections
import hashlib
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .pointing import check_model

FloatArray = npt.NDArray[np.float64]

ROBOT_CALIB_FILE = Path("/etc/r1mc3/head_camera.json")
KEEP_OBSERVATIONS = 8  # v2 headcam.py KEEP_FRAMES
HISTORY_S = 2.0  # how far back the measured-state history reaches
HEAD_SITE = "head_camera"


@dataclass(frozen=True)
class HeadCalibration:
    """The head camera's lens model + mount (calib_head.py's format) and where it came from."""

    model: dict[str, Any]
    T_site_cam: FloatArray
    source: str  # the file, or "sim" / "given"
    geometry: str = "calibrated"  # "calibrated" | "sim" (the simulator's own camera model)
    sha256: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """For status / observe replies. `sha256` = r1d's fingerprint of the parsed {T_site_cam, model} (canonical
        JSON: the same for any file with the same numbers); `file_sha256` = the file's own bytes (session 11: what
        `sha256sum /etc/r1mc3/head_camera.json` prints - the runbook's install check); `source` = where r1d read it
        (the file's own "source" field, its provenance, is kept as `provenance`)."""
        return {**self.meta, "geometry": self.geometry, "source": self.source, "sha256": self.sha256}


def calibration_from_dict(
    d: Mapping[str, Any], *, source: str = "given", geometry: str = "calibrated"
) -> HeadCalibration:
    T = np.asarray(d["T_site_cam"], dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-3) or abs(float(np.linalg.det(R)) - 1.0) > 1e-3:
        raise ValueError("T_site_cam is not a rigid transform")
    model = dict(d["model"])
    why = check_model(model)
    if why:
        raise ValueError(why)
    model.setdefault("lens", "pinhole")
    model.setdefault("depth", "z")
    model["k1"] = float(model.get("k1") or 0.0)
    raw = json.dumps({"T_site_cam": T.tolist(), "model": model}, sort_keys=True).encode()
    meta = {
        ("provenance" if k == "source" else k): v
        for k, v in d.items()
        if k not in ("T_site_cam", "model", "right") and isinstance(v, str | int | float)
    }
    return HeadCalibration(model, T, source, geometry, hashlib.sha256(raw).hexdigest(), meta)


def load_calibration(path: str | Path | None = None) -> HeadCalibration | None:
    """The calibration file (`path`, else $R1MC3_HEAD_CALIB, else /etc/r1mc3/head_camera.json); None when absent.
    A file that exists but does not parse raises ValueError (an operator must see it, never a silent default)."""
    p = Path(path) if path is not None else Path(os.environ.get("R1MC3_HEAD_CALIB") or ROBOT_CALIB_FILE)
    if not p.exists():
        return None
    try:
        raw = p.read_bytes()
        cal = calibration_from_dict(json.loads(raw), source=str(p))
        cal.meta["file_sha256"] = hashlib.sha256(raw).hexdigest()  # session 11: = `sha256sum <the file>`
        return cal
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"head camera calibration {p}: unreadable ({type(e).__name__}: {e})") from e


def calibration_mismatch(cal: HeadCalibration | None, *, stream_id: str | None, width: int, height: int) -> str | None:
    """Why this calibration must NOT be used for this frame (None = it may). Session 12 (HW2 day 1: a lens fit was
    read as "the stream changed under the calibration"): a calibration file may name the image size (`width` /
    `height`) and the stream it was made on (`stream`, r1cam's stream id, e.g. "stereo_patch_pc1/left" - the robot's
    left image through Unitree's stereo service); a frame of another size or from another stream is refused for
    pointing / measurement. A file that names no stream is not checked for it (the 4 Oct file: its capture was that
    same stream, `depth_source` "left udp 5002")."""
    if cal is None:
        return None
    w, h = cal.meta.get("width"), cal.meta.get("height")
    if w is not None and h is not None and (int(w), int(h)) != (int(width), int(height)):
        return f"the head-camera calibration is for {int(w)} x {int(h)} images, this frame is {width} x {height}"
    want = cal.meta.get("stream")
    if want and stream_id and str(want) != str(stream_id):
        return f"the head-camera calibration is for the stream {want!r}, this frame came from {stream_id!r}"
    return None


@dataclass
class Observation:
    obs_id: str
    t_capture_ns: int
    t_request_ns: int
    width: int
    height: int
    jpeg: bytes
    depth_m: FloatArray | None  # H x W metres (NaN = none), None when not asked for
    T_base_site: FloatArray
    calibration: HeadCalibration | None
    right_jpeg: bytes | None = None
    pose_source: str = "state at the request"
    frame: dict[str, Any] = field(default_factory=dict)  # r1cam's frame metadata (seq, receive time, counters)
    calibration_mismatch: str | None = None  # session 12: why the calibration does not fit this frame (refused)

    @property
    def T_base_cam(self) -> FloatArray | None:
        if self.calibration is None:
            return None
        return np.asarray(self.T_base_site @ self.calibration.T_site_cam, dtype=np.float64)

    def depth_valid_fraction(self) -> float | None:
        if self.depth_m is None:
            return None
        d = self.depth_m
        return round(float(np.mean(np.isfinite(d) & (d > 0.10) & (d < 3.0))), 3)


class ObservationStore:
    def __init__(self, keep: int = KEEP_OBSERVATIONS) -> None:
        self.keep = int(keep)
        self._obs: collections.OrderedDict[str, Observation] = collections.OrderedDict()
        self._n = 0
        self._lock = threading.Lock()

    def new_id(self) -> str:
        with self._lock:
            self._n += 1
            return f"h{self._n}"

    def put(self, o: Observation) -> None:
        with self._lock:
            self._obs[o.obs_id] = o
            while len(self._obs) > self.keep:
                self._obs.popitem(last=False)

    def get(self, obs_id: str) -> Observation | None:
        with self._lock:
            return self._obs.get(str(obs_id))

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._obs)


class StateHistory:
    """(t_ns, q26, imu_quat) samples of the measured state, the last HISTORY_S seconds (r1d's tick feeds it)."""

    def __init__(self, span_s: float = HISTORY_S) -> None:
        self.span_ns = int(span_s * 1e9)
        self._s: collections.deque[tuple[int, FloatArray, FloatArray | None]] = collections.deque()
        self._lock = threading.Lock()

    def add(self, t_ns: int, q26: npt.ArrayLike, imu_quat: npt.ArrayLike | None) -> None:
        q = np.asarray(q26, dtype=np.float64).copy()
        iq = None if imu_quat is None else np.asarray(imu_quat, dtype=np.float64).copy()
        with self._lock:
            if self._s and t_ns <= self._s[-1][0]:
                return
            self._s.append((int(t_ns), q, iq))
            while self._s and self._s[0][0] < t_ns - self.span_ns:
                self._s.popleft()

    def at(self, t_ns: int) -> tuple[int, FloatArray, FloatArray | None] | None:
        """The newest sample at or before t_ns (None when the history does not reach back that far)."""
        with self._lock:
            best = None
            for s in self._s:
                if s[0] <= t_ns:
                    best = s
                else:
                    break
            return best


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def depth_from_r1cam(entry: Mapping[str, Any]) -> FloatArray:
    """r1cam's depth entry (base64 raw uint16 LE millimetres, 0 / 65535 = none) -> H x W metres with NaN for none."""
    raw = base64.b64decode(str(entry["b64"]))
    h, w = int(entry["height"]), int(entry["width"])
    mm = np.frombuffer(raw, dtype="<u2").reshape(h, w).astype(np.float64)
    return np.where((mm > 0) & (mm < 65535), mm / 1000.0, np.nan)


def depth_pgm(depth_m: FloatArray) -> bytes:
    """A depth frame as a 16-bit PGM in millimetres (0 = none) - v2 HeadCam.depth_image."""
    mm = np.clip(np.nan_to_num(depth_m, nan=0.0) * 1000.0, 0, 65535).astype(">u2")
    h, w = mm.shape
    return f"P5\n{w} {h}\n65535\n".encode() + mm.tobytes()
