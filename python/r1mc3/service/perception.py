"""r1d's camera ops (session 10, DESIGN §8.1; R1CAM_PLAN.md): `camera.observe` / `camera.point` /
`camera.object_extent` / `camera.project` / `camera.depth_image` - v2 headcam.py's observe -> obs_id -> point
contract, through r1cam (the frames) and r1d (the pose at capture, the world model's declared surfaces).

* observe: r1cam's newest head frame (+ depth, + the right image on request) -> an obs_id kept with the head
  camera's pose in `base` at capture (cam/observe.py: the measured joints + IMU at the frame's capture time) and the
  calibration in force. The reply carries what a client needs to point by itself (`T_base_cam`, `camera_model`).
* point: the pixel's calibrated line of sight x the declared surface it meets, depth only for heights
  (cam/pointing.py, v2's rules); refuses without a calibration (v2's rule) or for an unknown / expired obs_id.
* object_extent: what stands at the pixel, from the depth (v2's grasp sizing); the robot's own hands are dropped.
* project / depth_image: base points -> pixels of that frame; the frame's depth as a 16-bit PGM (mm).

No r1cam -> UNSUPPORTED (a dead camera fails perception with a code, nothing else - DESIGN §2.1). Generic (D32): the
surfaces are whatever the client declared (`world.declare`); nothing here knows a task.
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from r1mc3.cam import pointing as PT
from r1mc3.cam.observe import (
    HEAD_SITE,
    HeadCalibration,
    Observation,
    ObservationStore,
    StateHistory,
    b64,
    calibration_mismatch,
    depth_from_r1cam,
    depth_pgm,
    load_calibration,
)

from .codes import Code
from .goals import POINT_GRASP_CENTRE, hand_points, port_state
from .schemas import Result, fail, ok

if TYPE_CHECKING:
    from .service import Service

HAND_EXCLUDE_R_M = 0.06  # v2 HeadCam.object_extent: a 6 cm ball 3 cm above each grasp centre = the hand, not an object
HAND_EXCLUDE_UP_M = 0.03
OBSERVE_TIMEOUT_S = 6.0


_CAL_STATUS: dict[str, Any] = {}


def _cal_brief(c: HeadCalibration) -> dict[str, Any]:
    """The calibration in r1d's status: r1d's fingerprint (first 12 of the canonical {T_site_cam, model} sha256) and
    the file's own sha256 (session 11: compare with `sha256sum /etc/r1mc3/head_camera.json`)."""
    return {
        "geometry": c.geometry,
        "source": c.source,
        "sha256": c.sha256[:12],
        "file_sha256": c.meta.get("file_sha256"),
        "stream": c.meta.get("stream", "not named (not checked)"),  # session 12: the stream the file was made on
    }


def calibration_status() -> dict[str, Any]:
    """r1d's status before the first camera word (session 11; the perception is built lazily): the head-camera file
    r1d will load ($R1MC3_HEAD_CALIB, else /etc/r1mc3/head_camera.json) described like a live perception's - so the
    runbook's install check reads `r1 status` at once. Re-read only when the file's path / mtime change."""
    import os

    from r1mc3.cam.observe import ROBOT_CALIB_FILE

    path = os.environ.get("R1MC3_HEAD_CALIB") or str(ROBOT_CALIB_FILE)
    try:
        key = f"{path}:{os.stat(path).st_mtime_ns}"
    except OSError:
        key = f"{path}:absent"
    if _CAL_STATUS.get("key") != key:
        try:
            cal = load_calibration(path)
            desc: dict[str, Any] = (
                {"geometry": "not calibrated", "error": f"no file {path}"} if cal is None else _cal_brief(cal)
            )
        except ValueError as e:
            desc = {"geometry": "not calibrated", "error": str(e)}
        _CAL_STATUS.clear()
        _CAL_STATUS.update(key=key, status={"calibration": desc, "observations": [], "note": "no camera word yet"})
    out: dict[str, Any] = dict(_CAL_STATUS["status"])
    return out


class Perception:
    """One per r1d (`perception_of(svc)`); thread-safe enough for r1d's request lock (r1d calls it under its lock)."""

    def __init__(
        self, svc: Service, calibration: HeadCalibration | None = None, *, latency_s: float | None = None
    ) -> None:
        self.svc = svc
        self.store = ObservationStore()
        self.history = StateHistory()
        self.calibration_error: str | None = None
        if calibration is None:
            try:
                calibration = load_calibration()
            except ValueError as e:
                self.calibration_error = str(e)
        self.calibration = calibration
        # the camera's exposure -> receive latency: given, else the calibration file's "latency_s" (session 11: HW2's
        # latency step C5 writes the measured value there), else 0 (unknown - the frame's receive time is used)
        if latency_s is None:
            latency_s = float(calibration.meta.get("latency_s", 0.0)) if calibration is not None else 0.0
        self.latency_s = float(latency_s)
        self._geo: Any = None
        self._site = -1
        self._geo_lock = threading.Lock()
        self._pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="r1d-camera")

    # ------------------------------------------------------------------ state
    def set_calibration(self, cal: HeadCalibration | None) -> None:
        self.calibration, self.calibration_error = cal, None

    def tick(self, now_ns: int) -> None:
        """Sample the measured state for the pose-at-capture history (r1d's tick)."""
        st = port_state(self.svc.port)
        if st is not None and "q" in st:
            self.history.add(int(now_ns), st["q"], st.get("imu_quat"))

    def status(self) -> dict[str, Any]:
        c = self.calibration
        return {
            "calibration": None if c is None else _cal_brief(c),
            "calibration_error": self.calibration_error,
            "observations": self.store.ids(),
            "latency_s": self.latency_s,
        }

    def T_base_site(self, q26: Any, imu_quat: Any) -> np.ndarray:
        """The head_camera site in `base` (the planner's geometry: the IMU mount undone, base = the soles' frame)."""
        import mujoco

        if self._geo is None:
            from r1mc3.world import Geometry

            self._geo = Geometry(None)
            self._site = int(mujoco.mj_name2id(self._geo.m, mujoco.mjtObj.mjOBJ_SITE, HEAD_SITE))
            if self._site < 0:
                raise RuntimeError(f"the robot model has no {HEAD_SITE!r} site")
        geo = self._geo
        geo.set_stance(np.asarray(q26, dtype=np.float64), None if imu_quat is None else np.asarray(imu_quat, float))
        T = np.eye(4)
        T[:3, :3] = np.asarray(geo.d.site_xmat[self._site], dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.asarray(geo.d.site_xpos[self._site], dtype=np.float64)
        return T

    def _supports(self) -> list[PT.Support]:
        w = self.svc.world
        if w is None:
            return []
        return PT.supports_from_entities(w.snapshot().entities)

    def _client(self) -> Any:
        if self.svc.camera is not None:
            return self.svc.camera
        from r1mc3.record.runner import CameraClient

        return CameraClient()

    # ------------------------------------------------------------------ ops
    def _work(self, fn: Any, t: Any, *args: Any) -> None:
        """Run a slow op off r1d's request lock (r1d's tick - the goals' keepalives, the deadman - never waits for the
        camera: r1cam may take seconds for a cold stream) and finish its ticket."""

        def run() -> None:
            try:
                res = fn(*args)
            except Exception as e:  # a bug here must still answer the client
                res = fail(args[0].op, args[0].id, Code.FAULT, reason=f"{type(e).__name__}: {e}")
            t.finish(res)

        self._pool.submit(run)

    def observe_begin(self, r: Any, t: Any) -> Result | None:
        now = int(self.svc.port.now_ns())
        st = port_state(self.svc.port)
        if st is None or "q" not in st:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="no measured state from the core (the pose at capture)")
        self.history.add(now, st["q"], st.get("imu_quat"))
        self._work(self.observe, t, r, st, now, time.monotonic_ns())
        return None

    def observe(self, r: Any, st: dict[str, Any], now_req: int, wall_req: int) -> Result:
        """(the camera worker) r1cam's frame + the pose at its capture -> an observation."""
        req = {"op": "observe", "depth": bool(r.depth), "right": bool(r.right), "max_age_s": float(r.max_age_s)}
        t_wall0 = wall_req
        try:
            out = self._client().request(req)
        except OSError as e:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason=f"r1cam does not answer ({e})")
        if not out.get("ok"):
            try:
                code = Code(out.get("code", "FAULT"))
            except ValueError:
                code = Code.FAULT
            return fail(r.op, r.id, code, **dict(out.get("details", {})), camera="r1cam")
        data = dict(out.get("data", {}))
        left = data.get("left")
        if not left:
            return fail(r.op, r.id, Code.FAULT, reason="r1cam sent no head image")
        # the capture time on r1d's clock, anchored at the request: r1d's time then + (the frame's receive stamp - the
        # request's, both CLOCK_MONOTONIC on this machine) - the camera's latency (exposure -> receive; HW2 measures
        # it). On the robot r1d's clock IS CLOCK_MONOTONIC (= the receive stamp); in a lockstep simulation the request
        # anchor keeps a frame rendered after the request from reaching back past the request's own state sample (the
        # episode runner saw "1481.9 ms before the capture": the reply-time anchor minus a wall-clock age reached back
        # to the previous observe's sample when the simulation ran slower than the wall clock)
        t_rx = int(left.get("t_rx_ns", t_wall0))
        age_ns = max(0, time.monotonic_ns() - t_rx)
        t_cap = now_req + (t_rx - wall_req) - int(self.latency_s * 1e9)
        s = self.history.at(t_cap)
        if s is not None:
            q, iq, pose_src = s[1], s[2], f"measured state {round((t_cap - s[0]) / 1e6, 1)} ms before the capture"
        else:
            q, iq, pose_src = np.asarray(st["q"], float), st.get("imu_quat"), "state at the request (no older sample)"
        with self._geo_lock:
            T_bs = self.T_base_site(q, iq)
        depth = depth_from_r1cam(data["depth"]) if data.get("depth") else None
        import base64

        o = Observation(
            obs_id=self.store.new_id(),
            t_capture_ns=t_cap,
            t_request_ns=now_req,
            width=int(left.get("width", 544)),
            height=int(left.get("height", 448)),
            jpeg=base64.b64decode(str(left["b64"])),
            depth_m=depth,
            T_base_site=T_bs,
            calibration=self.calibration,
            right_jpeg=base64.b64decode(str(data["right"]["b64"])) if data.get("right") else None,
            pose_source=pose_src,
            frame={k: left.get(k) for k in ("seq", "t_rx_ns", "src_counter", "src_stamp", "stream_id")},
            calibration_mismatch=calibration_mismatch(
                self.calibration, stream_id=left.get("stream_id"), width=int(left.get("width", 544)),
                height=int(left.get("height", 448)),
            ),
        )  # fmt: skip
        self.store.put(o)
        cal = self.calibration
        T_bc = o.T_base_cam
        reply: dict[str, Any] = {
            "obs_id": o.obs_id,
            "t_capture_ns": o.t_capture_ns,
            "frame_age_ms": round(age_ns / 1e6, 1),
            "pose_source": pose_src,
            "jpeg_b64": b64(o.jpeg),
            "width": o.width,
            "height": o.height,
            "T_base_site": np.round(T_bs, 6).tolist(),
            "T_base_cam": None if T_bc is None else np.round(T_bc, 6).tolist(),
            "camera_model": None if cal is None else dict(cal.model),
            "calibration": (
                {"geometry": "not calibrated", **({"error": self.calibration_error} if self.calibration_error else {})}
                if cal is None
                else {**cal.describe(), **({"mismatch": o.calibration_mismatch} if o.calibration_mismatch else {})}
            ),
            "depth_valid_fraction": o.depth_valid_fraction(),
            "depth_source": None if depth is None else f"r1cam {data['depth'].get('source', 'depth')}",
            "stereo": o.right_jpeg is not None,
        }
        if o.right_jpeg is not None:
            reply["right_jpeg_b64"] = b64(o.right_jpeg)
        if not r.jpeg:
            reply.pop("jpeg_b64")
            reply.pop("right_jpeg_b64", None)
            reply["jpeg_bytes"] = len(o.jpeg)
        return ok(r.op, r.id, **reply)

    def _frame_or_fail(self, r: Any, *, need_cal: bool = True, need_depth: bool = False) -> Observation | Result:
        o = self.store.get(r.obs_id)
        if o is None:
            return fail(r.op, r.id, Code.UNKNOWN_FRAME, obs_id=r.obs_id, kept=self.store.ids(),
                        reason=f"no head frame {r.obs_id!r} (only the last {self.store.keep} are kept)")  # fmt: skip
        if need_cal and o.calibration is None:
            return fail(r.op, r.id, Code.NOT_CALIBRATED, reason="the head camera is not calibrated "
                        "(/etc/r1mc3/head_camera.json): pointing is refused")  # fmt: skip
        if need_cal and o.calibration_mismatch:
            return fail(r.op, r.id, Code.NOT_CALIBRATED, reason=f"{o.calibration_mismatch}: pointing is refused")
        if need_depth and o.depth_m is None:
            return fail(r.op, r.id, Code.NO_DEPTH, reason="that frame was observed without depth (observe depth=true)")
        return o

    def point(self, r: Any) -> Result:
        o = self._frame_or_fail(r)
        if isinstance(o, Result):
            return o
        assert o.calibration is not None and o.T_base_cam is not None
        model = {**o.calibration.model, "width": o.width, "height": o.height}
        res = PT.point(o.depth_m, model, o.T_base_cam, float(r.px), float(r.py), self._supports(),
                       patch=int(r.patch), use_depth=bool(r.use_depth))  # fmt: skip
        if not res.pop("ok"):
            hint = res.pop("code_hint", "OFF_SURFACE")
            code = {"INVALID_ARGS": Code.INVALID_ARGS, "UNKNOWN_FRAME": Code.OFF_SURFACE,
                    "NO_DEPTH": Code.NO_DEPTH}.get(hint, Code.OFF_SURFACE)  # fmt: skip
            return fail(r.op, r.id, code, **res)
        res["obs_id"] = o.obs_id
        res["geometry"] = f"{o.calibration.geometry}; ray + declared surface" if res.get("surface_id") else (
            f"{o.calibration.geometry}; depth")  # fmt: skip
        return ok(r.op, r.id, **res)

    def extent_begin(self, r: Any, t: Any) -> Result | None:
        o = self._frame_or_fail(r, need_depth=True)
        if isinstance(o, Result):
            return o
        exclude: list[tuple[float, float, float, float]] = []
        if r.exclude_hands:
            pts = hand_points(self.svc.port, self.svc.hands)
            if pts is not None:
                for side in range(pts.shape[0]):
                    c = pts[side, POINT_GRASP_CENTRE]
                    exclude.append((float(c[0]), float(c[1]), float(c[2]) + HAND_EXCLUDE_UP_M, HAND_EXCLUDE_R_M))
        self._work(self.object_extent, t, r, o, self._supports(), exclude)
        return None

    def object_extent(
        self, r: Any, o: Observation, sups: list[PT.Support], exclude: list[tuple[float, float, float, float]]
    ) -> Result:
        """(the camera worker) v2's extent from the frame's depth (~0.1 s of numpy: off r1d's lock)."""
        assert o.calibration is not None and o.T_base_cam is not None and o.depth_m is not None

        def surface_z(X: Any, Y: Any, grow: float) -> Any:
            return PT.top_at(sups, X, Y, grow)

        res = PT.object_extent(o.depth_m, o.calibration.model, o.T_base_cam, float(r.px), float(r.py),
                               surface_z if sups else None, exclude)  # fmt: skip
        if not res.pop("ok"):
            return fail(r.op, r.id, Code.OFF_SURFACE if res.get("flat_or_empty") else Code.NO_DEPTH, **res)
        if sups:
            c = res["centre_base"]
            hit = [s.id for s in sups if bool(s.contains(c[0], c[1], PT.EXT_EDGE_M))]
            res["surface_id"] = max(hit, key=lambda i: next(s.z for s in sups if s.id == i)) if hit else None
        res["obs_id"] = o.obs_id
        res["excluded_hands"] = len(exclude)
        return ok(r.op, r.id, **res)

    def project(self, r: Any) -> Result:
        o = self._frame_or_fail(r)
        if isinstance(o, Result):
            return o
        assert o.calibration is not None and o.T_base_cam is not None
        px = PT.project_points(o.calibration.model, o.T_base_cam, r.points, o.width, o.height)
        return ok(r.op, r.id, obs_id=o.obs_id, pixels=px)

    def depth_image(self, r: Any) -> Result:
        o = self._frame_or_fail(r, need_cal=False, need_depth=True)
        if isinstance(o, Result):
            return o
        assert o.depth_m is not None
        return ok(r.op, r.id, obs_id=o.obs_id, pgm_b64=b64(depth_pgm(o.depth_m)), width=o.width, height=o.height)

    def handle(self, r: Any, t: Any) -> Result | None:
        """r1d's dispatch (under its lock): observe / object_extent finish their ticket from the camera worker (None
        here = pending); point / project / depth_image answer at once (pure geometry on a kept frame)."""
        if r.op == "camera.observe":
            return self.observe_begin(r, t)
        if r.op == "camera.object_extent":
            return self.extent_begin(r, t)
        fn = {"camera.point": self.point, "camera.project": self.project, "camera.depth_image": self.depth_image}[r.op]
        return fn(r)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def perception_of(svc: Service) -> Perception:
    """r1d's Perception (made on first use; a test may set `svc._perception` first)."""
    p = getattr(svc, "_perception", None)
    if p is None:
        p = Perception(svc)
        svc._perception = p  # type: ignore[attr-defined]
    assert isinstance(p, Perception)
    return p
