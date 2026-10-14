"""Frames for measurement: r1cam's head image with the camera's pose AT CAPTURE, a still from a phone, and the request
specs (tags / lenses) as the measurement types (DESIGN §8.2; r1d's `measure.setup` / `measure.phone`, session 10).

The head camera's pose at capture comes from the joints read just BEFORE r1cam's frame was taken and just AFTER it
arrived (both on CLOCK_MONOTONIC, the clock r1cam stamps its frames with on the backpack): the frame is accepted only
when it was taken after the first read (its receive time minus the camera's latency) and the head moved less than a
limit between the two reads - so the pose (the forward kinematics of the two reads' mean, `headpose.head_T_base_cam`)
is the pose of the picture to within the stillness limit. A measurement wants a still head anyway (the pre-flight
holds the head at the task pose; the balance policy's sway moves the whole body by < 1 mm here). r1cam's
`camera.observe` (pointing, the cam agent) stamps the pose the same way; a moving-head interpolation is not needed here.

Generic (D32): no task, no layout - the client declares its entities and their tags.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .lenses import OpenCVLens
from .tags import TagPlacement, flat_tag

FloatArray = npt.NDArray[np.float64]
# the head joints in the 26-joint SDK order (r1sim.joints: ... head_pitch, head_yaw are the last two)
HEAD_JOINTS = (24, 25)
CAMERA_LATENCY_NS = 50_000_000  # exposure -> r1cam's receive time: NOT measured yet (R1CAM_PLAN.md §3: HW2's latency
# step); v2's pipeline (rtpjitterbuffer latency=100 + the H.264 decode) suggests ~50-150 ms - a frame counts only when
# its receive time minus this lies after the first joint read; a larger true latency is still caught by the
# stillness rule (the head must not move between the reads around the frame)


@dataclass(frozen=True)
class HeadFrame:
    gray: npt.NDArray[np.uint8]
    T_base_cam: FloatArray
    t_rx_ns: int
    seq: int
    head_motion_rad: float  # the head joints' largest change between the joint reads around the frame
    q: FloatArray  # the 26 joints the pose was computed from (the two reads' mean)

    def summary(self) -> dict[str, Any]:
        return {"t_rx_ns": self.t_rx_ns, "seq": self.seq, "head_motion_rad": round(self.head_motion_rad, 5)}


class CaptureError(Exception):
    """A frame could not be taken: `code` is the r1d code (TIMEOUT / UNSUPPORTED / NOT_STILL), `details` its data."""

    def __init__(self, code: str, reason: str, **details: Any) -> None:
        super().__init__(reason)
        self.code = code
        self.details = {"reason": reason, **details}


def decode_gray(data: bytes) -> npt.NDArray[np.uint8]:
    """A JPEG / PNG -> its grey image (OpenCV's decoder: what the tag detector reads)."""
    import cv2

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("the image does not decode (JPEG / PNG expected)")
    return np.asarray(img, dtype=np.uint8)


Stance = tuple[FloatArray, FloatArray | None]  # (the 26 measured joints, the IMU's quaternion w x y z or None)


def head_frames(
    observe: Callable[[dict[str, Any]], dict[str, Any]],
    joints: Callable[[], Stance | None],
    pose_of: Callable[[FloatArray, FloatArray | None], FloatArray],
    n: int,
    *,
    wait_s: float = 5.0,
    max_motion_rad: float = 0.003,
    latency_ns: int = CAMERA_LATENCY_NS,
    clock: Callable[[], int] = time.monotonic_ns,
) -> list[HeadFrame]:
    """`n` NEW head frames (each taken after the joint read before it) with the camera pose from the joints.

    observe(req) = r1cam's `observe` reply (`{ok, code, data: {left: {b64, t_rx_ns, seq}}, details}`); joints() = the
    26 measured joints + the IMU's quaternion now (None = no state); pose_of(q26, imu) = the camera in base (OpenCV
    frame). Raises CaptureError."""
    out: list[HeadFrame] = []
    last_seq = 0
    for _ in range(n):
        deadline = clock() + int(wait_s * 1e9)
        s0 = joints()
        if s0 is None:
            raise CaptureError("FAULT", "no joint state from the core: the camera pose at capture is unknown")
        q0 = s0[0]
        w0 = clock()
        while True:
            try:
                rep = observe({"op": "observe", "max_age_s": 1.0, "wait_s": min(wait_s, 3.0)})
            except OSError as e:
                raise CaptureError("UNSUPPORTED", f"r1cam does not answer ({e})") from e
            if not rep.get("ok"):
                raise CaptureError(
                    str(rep.get("code", "FAULT")), "r1cam gave no head frame", r1cam=dict(rep.get("details", {}))
                )
            f = dict(dict(rep.get("data", {})).get("left") or {})
            seq, t_rx = int(f.get("seq", 0)), int(f.get("t_rx_ns", 0))
            if seq > last_seq and t_rx - latency_ns >= w0:
                break
            if clock() > deadline:
                raise CaptureError("TIMEOUT", f"no new head frame within {wait_s:g} s", last_seq=seq)
            time.sleep(0.02)
        s1 = joints()
        if s1 is None:
            raise CaptureError("FAULT", "no joint state from the core after the frame")
        q1 = s1[0]
        motion = float(np.max(np.abs(q1[list(HEAD_JOINTS)] - q0[list(HEAD_JOINTS)])))
        if motion > max_motion_rad:
            raise CaptureError(
                "NOT_STILL",
                f"the head moved {motion:.4f} rad while the frame was taken (max {max_motion_rad:g}): hold the head "
                "still and measure again",
                head_motion_rad=motion,
            )
        q = 0.5 * (q0 + q1)
        imu = _mean_quat(s0[1], s1[1])
        gray = decode_gray(base64.b64decode(str(f["b64"])))
        out.append(HeadFrame(gray, np.asarray(pose_of(q, imu), dtype=np.float64), t_rx, seq, motion, q))
        last_seq = seq
    return out


def _mean_quat(a: FloatArray | None, b: FloatArray | None) -> FloatArray | None:
    """The two reads' IMU quaternions averaged (they are ~a frame apart: the normalised sum is the mean)."""
    if a is None or b is None:
        return a if b is None else b
    qa, qb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if float(qa @ qb) < 0.0:
        qb = -qb
    m = qa + qb
    return np.asarray(m / np.linalg.norm(m), dtype=np.float64)


def tag_placement(spec: Mapping[str, Any]) -> TagPlacement:
    """A request's tag (`{id, size_m, T_entity_tag}` or `{id, size_m, xyz, up_yaw_deg}`) -> a TagPlacement."""
    if spec.get("T_entity_tag") is not None:
        T = np.asarray(spec["T_entity_tag"], dtype=np.float64)
        if T.shape != (4, 4) or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError(f"tag {spec.get('id')}: T_entity_tag is not a 4 x 4 pose")
        R = T[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0.0:
            raise ValueError(f"tag {spec.get('id')}: T_entity_tag's rotation is not a rotation")
        return TagPlacement(int(spec["id"]), float(spec["size_m"]), T)
    x, y, z = (float(v) for v in spec["xyz"])
    return flat_tag(int(spec["id"]), float(spec["size_m"]), x, y, z, float(spec.get("up_yaw_deg", 0.0)))


def tag_spec(t: TagPlacement) -> dict[str, Any]:
    """The inverse of `tag_placement` (a client builds its request from its plan)."""
    return {"id": int(t.tag_id), "size_m": float(t.size_m), "T_entity_tag": np.round(t.T_entity_tag, 9).tolist()}


def placements(entity: Mapping[str, Any]) -> list[TagPlacement]:
    ids = [int(t["id"]) for t in entity["tags"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"entity {entity.get('id')}: a tag id appears twice")
    return [tag_placement(t) for t in entity["tags"]]


def unique_ids(entities: Sequence[Mapping[str, Any]]) -> list[int]:
    """Every tag id of the request, each owned by ONE entity (a tag cannot belong to two entities)."""
    seen: dict[int, str] = {}
    for e in entities:
        for t in e["tags"]:
            tid = int(t["id"])
            if tid in seen and seen[tid] != e["id"]:
                raise ValueError(f"tag {tid} is declared on both {seen[tid]} and {e['id']}")
            seen[tid] = str(e["id"])
    return sorted(seen)


def opencv_lens(spec: Mapping[str, Any]) -> OpenCVLens:
    """A request's still-camera model (`{K | f_px, dist, width, height}`) -> an OpenCVLens."""
    w, h = int(spec["width"]), int(spec["height"])
    dist = np.asarray(spec.get("dist") or [0.0] * 5, dtype=np.float64)
    if spec.get("K") is not None:
        return OpenCVLens(np.asarray(spec["K"], dtype=np.float64), dist, w, h)
    return OpenCVLens.pinhole(float(spec["f_px"]), w, h, dist)
