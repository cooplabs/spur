"""r1d's measurement words (session 10; DESIGN §8.2): `measure.setup` (the head camera) and `measure.phone` (a still
from a hand-placed camera). GENERIC (D32): the client declares its entities and their tags; r1d detects, fits and
returns registrations (pose, fit RMS, 1-sigma, the tags used / missing, a refusal reason per entity); the tolerances,
the checks and the words that say what to move are the client's (a task's own pre-flight check).

Neither word moves anything. Both run OFF r1d's request lock in a worker thread (a few head frames at 7.5 / s + the
detection + the fits take ~0.5-2 s: holding the lock that long would stall r1d's 100 Hz tick - the goals' keepalives
and the hands): the request returns pending and the worker finishes its ticket. One measurement at a time
(RESOURCE_BUSY otherwise). OpenCV (the `perception` extra) is imported lazily: without it both words answer UNSUPPORTED.

`measure.setup`: `frames` NEW head frames from r1cam, each accepted only when taken after a joints + IMU read and with
the head still until the next read (`r1mc3.measure.capture.head_frames`); the camera's pose per frame = the head_camera
site in the PLANNER's base frame from those reads (`headpose.head_T_base_cam_stance` = Geometry.set_stance, the
convention camera.observe uses) @ the calibration's mount; every entity = one fit over all its tags in all frames
(`register.locate_entity`). The calibration is r1d's ONE head-camera calibration - the one pointing uses
(`perception_of(svc).calibration`; none -> NOT_CALIBRATED, v2's rule), its sha256 in the reply.
"""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .codes import Code
from .schemas import MeasurePhoneReq, MeasureSetupReq, Result, fail, ok

if TYPE_CHECKING:
    from .service import Service, Ticket


# Session 12 (S11 stand rehearsal finding: `r1 measure setup` gave no answer within the CLI's 20 s): a head
# measurement may take up to `frames` x `wait_s` waiting for new head frames (each frame: observe calls of <= 3 s until
# a NEW frame taken after the joint read arrives) + the detection and the fits - under load in the simulator ~6 s a
# frame, so 3 frames + the fits passed the CLI's 20 s and the operator read "r1d not reachable" while r1d was still
# measuring. Now the measurement has ONE budget (`setup_budget_s`): every r1cam call is bounded by what is left of it
# (the socket included: r1cam's client waited 15 s a call), it answers in words when the budget is spent, and the
# CLI waits for the same budget + a margin.
MEASURE_FIT_MARGIN_S = 10.0  # the detection + the fits after the frames (Orin: a few s; macOS under load: more)


def setup_budget_s(frames: int, wait_s: float) -> float:
    """The longest a `measure.setup` with these arguments may take before r1d answers (s)."""
    return float(frames) * float(wait_s) + MEASURE_FIT_MARGIN_S


def _cv2_missing() -> str | None:
    try:
        import cv2

        cv2.aruco.DetectorParameters()  # the contrib aruco module (opencv-python-headless >= 4.7 has it)
    except (ImportError, AttributeError) as e:
        return f"OpenCV with aruco is not installed ({type(e).__name__}): install r1mc3's `perception` extra"
    return None


class MeasureOps:
    def __init__(self, svc: Service) -> None:
        self.svc = svc
        self._busy = threading.Lock()
        self.last: dict[str, Any] | None = None  # the last measurement's summary (status)
        self.observe: Callable[[dict[str, Any]], dict[str, Any]] | None = None  # tests: r1cam's handle in-process
        self._deadline: float | None = None  # monotonic s: the running setup's frames must arrive by then (session 12)
        self._budget_s = 0.0

    # ------------------------------------------------------------------ dispatch (under r1d's request lock)
    def dispatch(self, r: MeasureSetupReq | MeasurePhoneReq, t: Ticket) -> Result | None:
        why = _cv2_missing()
        if why is not None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason=why)
        try:  # the specs are checked here, so a bad request fails at once (nothing starts)
            from r1mc3.measure import capture

            ents = [e.model_dump() for e in r.entities]
            capture.unique_ids(ents)
            for e in ents:
                capture.placements(e)
            if isinstance(r, MeasurePhoneReq):
                for tg in r.table_tags:
                    capture.tag_placement(tg.model_dump())
                capture.opencv_lens(r.intrinsics.model_dump())
                image = base64.b64decode(r.image_b64, validate=True)
        except (ValueError, KeyError) as e:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason=str(e))
        if not self._busy.acquire(blocking=False):
            return fail(r.op, r.id, Code.RESOURCE_BUSY, reason="a measurement is already running")
        if isinstance(r, MeasureSetupReq):
            job: Callable[[], Result] = lambda: self._setup(r)  # noqa: E731
        else:
            job = lambda: self._phone(r, image)  # noqa: E731
        threading.Thread(target=self._run, args=(r, t, job), daemon=True, name="r1d-measure").start()
        return None

    def _run(self, r: Any, t: Ticket, job: Callable[[], Result]) -> None:
        t0 = time.monotonic()
        try:
            res = job()
        except Exception as e:  # never leave a client waiting: a failure is a FAULT result with the reason
            res = fail(r.op, r.id, Code.FAULT, reason=f"{type(e).__name__}: {e}")
        finally:
            self._busy.release()
        res = res.model_copy(update={"data": {**res.data, "duration_s": round(time.monotonic() - t0, 3)}})
        self.last = {"op": r.op, "ok": res.ok, "code": str(res.code), "t_unix": round(time.time(), 3)}
        t.finish(res)

    # ------------------------------------------------------------------ the head
    def camera(self) -> tuple[Any, Any, dict[str, Any]]:
        """(lens, T_site_cam, info) from r1d's ONE head-camera calibration - the one pointing uses
        (`perception_of(svc).calibration`: $R1MC3_HEAD_CALIB, else /etc/r1mc3/head_camera.json; the simulator's
        camera in a simulation). None -> CaptureError NOT_CALIBRATED (v2's rule for pointing, the same here)."""
        from r1mc3.measure.capture import CaptureError
        from r1mc3.measure.lenses import ModelLens

        from .perception import perception_of

        p = perception_of(self.svc)
        cal = p.calibration
        if cal is None:
            raise CaptureError(
                "NOT_CALIBRATED",
                p.calibration_error
                or "no head-camera calibration in r1d (/etc/r1mc3/head_camera.json): measuring needs one",
            )
        return ModelLens(cal.model), np.asarray(cal.T_site_cam, dtype=np.float64), cal.describe()

    def _joints(self) -> Any:
        from .goals import port_state

        with self.svc._lock:  # a consistent read next to r1d's tick
            st = port_state(self.svc.port)
        if st is None or st.get("q") is None:
            return None
        q = np.asarray(st["q"], dtype=np.float64)
        if q.shape != (26,):
            return None
        imu = st.get("imu_quat")
        return q, None if imu is None else np.asarray(imu, dtype=np.float64)

    def _observe(self, req: dict[str, Any]) -> dict[str, Any]:
        """One r1cam `observe`, bounded by what is left of the measurement's budget (session 12): its own wait and,
        for r1cam over TCP, the socket's timeout (r1cam's client default waited 15 s a call); the budget spent ->
        CaptureError TIMEOUT in words (r1cam not streaming / too slow), never a silent wait past the CLI's patience."""
        from r1mc3.measure.capture import CaptureError

        left = self._deadline - time.monotonic() if self._deadline is not None else None
        if left is not None and left <= 0.0:
            raise CaptureError(
                "TIMEOUT",
                f"no usable head frame from r1cam within the measurement's {self._budget_s:g} s: is r1cam running and "
                "streaming? (`r1 camera status`; on the robot `sudo systemctl start r1cam`)",
            )
        if left is not None and "wait_s" in req:
            req = {**req, "wait_s": max(0.05, min(float(req["wait_s"]), left))}
        if self.observe is not None:
            return self.observe(req)
        cam = self.svc.camera
        if cam is None:
            from r1mc3.record.runner import CameraClient

            cam = CameraClient()
        port = getattr(cam, "port", None)
        if left is not None and isinstance(port, int):  # r1cam over TCP (CameraClient): bound the socket too
            from r1mc3.cam.service import request

            return request(port, req, timeout_s=max(0.5, left + 1.0))
        out: dict[str, Any] = cam.request(req)
        return out

    def _setup(self, r: MeasureSetupReq) -> Result:
        from r1mc3.measure import View, detect_tags, locate_entity
        from r1mc3.measure.capture import CaptureError, head_frames, placements, unique_ids
        from r1mc3.measure.headpose import head_T_base_cam_stance

        ents = [e.model_dump() for e in r.entities]
        ids = unique_ids(ents)
        self._budget_s = setup_budget_s(r.frames, r.wait_s)
        self._deadline = time.monotonic() + self._budget_s - MEASURE_FIT_MARGIN_S  # the frames' share of it
        try:
            lens, T_site_cam, cal = self.camera()
            frames = head_frames(
                self._observe,
                self._joints,
                lambda q, imu: head_T_base_cam_stance(q, imu, T_site_cam=T_site_cam),
                r.frames,
                wait_s=r.wait_s,
                max_motion_rad=r.max_head_motion_rad,
            )
        except CaptureError as e:
            try:
                code = Code(e.code)
            except ValueError:
                code = Code.FAULT
            return fail(r.op, r.id, code, **e.details)
        finally:
            self._deadline = None
        views = [View(detect_tags(f.gray, ids=ids), lens, f.T_base_cam, f"head{k}") for k, f in enumerate(frames)]
        regs = {e["id"]: locate_entity(e["id"], placements(e), views, level=bool(e["level"])) for e in ents}
        seen = sorted({i for v in views for i in v.tags})
        return ok(
            r.op,
            r.id,
            frame="base",
            entities={k: v.as_dict() for k, v in regs.items()},
            tags_seen=seen,
            frames=[f.summary() for f in frames],
            T_base_cam=[np.round(f.T_base_cam, 6).tolist() for f in frames],
            camera={
                **cal,
                "pose_from": "the joints + the IMU (the planner's base frame: Geometry.set_stance) @ the mount",
            },
            mode=self.svc.mode_name(),
        )

    # ------------------------------------------------------------------ a phone still
    def _phone(self, r: MeasurePhoneReq, image: bytes) -> Result:
        from r1mc3.measure import View, detect_tags, locate_entity
        from r1mc3.measure.capture import decode_gray, opencv_lens, placements, tag_placement, unique_ids
        from r1mc3.measure.phone import correct_phone

        lens = opencv_lens(r.intrinsics.model_dump())
        gray = decode_gray(image)
        if gray.shape != (lens.height, lens.width):
            return fail(r.op, r.id, Code.INVALID_ARGS,
                        reason=f"the image is {gray.shape[1]} x {gray.shape[0]}, the intrinsics {lens.width} x "
                        f"{lens.height}")  # fmt: skip
        table = [tag_placement(tg.model_dump()) for tg in r.table_tags]
        ents = [e.model_dump() for e in r.entities]
        ids = sorted(set(unique_ids(ents)) | {t.tag_id for t in table})
        tags = detect_tags(gray, ids=ids)
        pc = correct_phone(tags, lens, table, max_fit_mm=r.max_fit_mm)
        out: dict[str, Any] = {"frame": "table", "phone": pc.as_dict(), "tags_seen": sorted(tags)}
        if pc.ok and pc.T_table_cam is not None:
            view = View(tags, lens, pc.T_table_cam, "phone")
            out["entities"] = {
                e["id"]: locate_entity(e["id"], placements(e), [view], level=bool(e["level"])).as_dict() for e in ents
            }
            out["points"] = {
                p.name: [round(float(v), 5) for v in pc.to_plane(np.array([p.px]), z=p.z_m)[0]] for p in r.points
            }
        else:
            out["entities"] = {e["id"]: {"entity": e["id"], "ok": False, "reason": f"no phone pose: {pc.reason}"}
                               for e in ents}  # fmt: skip
            out["points"] = {}
        return ok(r.op, r.id, **out)

    def status(self) -> dict[str, Any]:
        return {"busy": self._busy.locked(), "last": self.last}


def measure_ops(svc: Service) -> MeasureOps:
    """The service's MeasureOps (made on first use: r1d without the measurement words pays nothing)."""
    m = getattr(svc, "_measure_ops", None)
    if m is None:
        m = MeasureOps(svc)
        svc._measure_ops = m  # type: ignore[attr-defined]
    return m
