"""r1d's measurement words (session 10; DESIGN §8.2): `measure.setup` (head frames from r1cam, the camera pose from the
joints + IMU around each frame) and `measure.phone` (a still from a hand-placed camera), GENERIC (the client declares
its entities + tags). Synthetic frames: tags warped into the image through an exact pinhole homography (no simulator),
so the numbers isolate r1d's path: frame acceptance, the stillness rule, the camera pose, the fit, the reply.
All of these answered UNKNOWN_COMMAND on the base (a041731: no measure.* op)."""

from __future__ import annotations

import base64
import json
import math
import threading
import time
from typing import Any

import cv2
import numpy as np
import pytest
from r1mc3.cli.main import build_request
from r1mc3.measure import OpenCVLens, Registration, TagPlacement, flat_tag
from r1mc3.measure.capture import tag_spec
from r1mc3.measure.headpose import head_T_base_cam_stance
from r1mc3.measure.phone import PhoneCorrection
from r1mc3.measure.register import rodrigues
from r1mc3.service import measure_ops as MO
from r1mc3.service.schemas import REQUEST, MeasurePhoneReq
from r1mc3.service.service import Service
from r1sim import joints as J
from r1sim import lens as lens_mod

from tests.unit.test_service_server import FakePort

HEAD_Q = (0.607, 0.0)  # the task view's head pitch / yaw
HEAD_LENS = OpenCVLens.pinhole(300.0, 544, 448)  # an exact pinhole stand-in for the head camera (homography-exact)
T_SITE_CAM = lens_mod.load_calibration().T_site_cam  # the real mount
# a generic slab in front of the robot (base frame): its top 0.75 m up, 0.45 m ahead; 4 flat tags on it
SLAB_T = np.eye(4)
SLAB_T[:3, :3] = rodrigues([0.0, 0.0, math.radians(3.0)])
SLAB_T[:3, 3] = [0.45, 0.02, 0.75]
SLAB_TAGS = [flat_tag(1, 0.08, -0.08, 0.12), flat_tag(2, 0.08, -0.08, -0.12), flat_tag(3, 0.10, 0.12, 0.15),
             flat_tag(4, 0.10, 0.12, -0.15)]  # fmt: skip


def _q() -> np.ndarray:
    q = J.POSE_STAND.copy()
    q[24], q[25] = HEAD_Q
    return q


def _render(lens: OpenCVLens, T_ref_cam: np.ndarray, tags: list[tuple[np.ndarray, TagPlacement]]) -> np.ndarray:
    """A grey image of flat tags (each: T_ref_entity, placement) seen by a pinhole camera at T_ref_cam."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = np.full((lens.height, lens.width), 150, np.uint8)
    Tc = np.linalg.inv(T_ref_cam)
    src = np.array([[99.5, 99.5], [899.5, 99.5], [899.5, 899.5], [99.5, 899.5]], np.float32)
    for T_e, t in tags:
        P = (Tc @ T_e @ np.c_[t.corners_entity(), np.ones(4)].T).T[:, :3]
        px = lens.project(P).astype(np.float32)
        sheet = np.full((1000, 1000), 255, np.uint8)
        sheet[100:900, 100:900] = cv2.aruco.generateImageMarker(d, t.tag_id, 800)
        H = cv2.getPerspectiveTransform(src, px)
        size = (lens.width, lens.height)
        warped = cv2.warpPerspective(sheet, H, size, flags=cv2.INTER_AREA, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(sheet, 255), H, size, flags=cv2.INTER_NEAREST, borderValue=0)
        img[mask > 0] = warped[mask > 0]
    return img


def _jpeg(gray: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return bytes(buf)


class StatePort(FakePort):
    """FakePort + the core's measured state (q, imu_quat) - the head may drift per read (`head_step` rad)."""

    def __init__(self, head_step: float = 0.0) -> None:
        super().__init__(mode=3)
        self.q = _q()
        self.head_step = head_step
        self.reads = 0

    def state(self) -> dict[str, Any]:
        self.reads += 1
        q = self.q.copy()
        q[24] += self.head_step * self.reads
        return {"q": q.tolist(), "imu_quat": [1.0, 0.0, 0.0, 0.0]}


class FakeCam:
    """r1cam's observe: a new frame per call (seq + 1, received now), or a stale one (received 1 s ago)."""

    def __init__(self, jpeg: bytes, *, stale: bool = False, delay_s: float = 0.0) -> None:
        self.jpeg = jpeg
        self.seq = 0
        self.stale = stale
        self.delay_s = delay_s

    def __call__(self, req: dict[str, Any]) -> dict[str, Any]:
        time.sleep(self.delay_s)
        if not self.stale:
            self.seq += 1
        t = time.monotonic_ns() - (1_000_000_000 if self.stale else 0)
        left = {"b64": base64.b64encode(self.jpeg).decode(), "t_rx_ns": t, "seq": max(self.seq, 1)}
        return {"ok": True, "code": "OK", "data": {"left": left}, "details": {}}


def _svc(port: StatePort, cam: FakeCam | None, monkeypatch: pytest.MonkeyPatch) -> Service:
    svc = Service(port)

    def camera(self: MO.MeasureOps) -> tuple[Any, Any, dict[str, Any]]:
        return HEAD_LENS, T_SITE_CAM, {"geometry": "test", "source": "pinhole stand-in", "sha256": "-"}

    monkeypatch.setattr(MO.MeasureOps, "camera", camera)
    ops = MO.measure_ops(svc)
    ops.observe = cam
    return svc


def _wait(svc: Service, req: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    t = svc.submit(req)
    assert t.done.wait(timeout_s), "no result"
    assert t.result is not None
    return t.result.model_dump(mode="json")


def _setup_req(frames: int = 2) -> dict[str, Any]:
    return {"op": "measure.setup", "frames": frames,
            "entities": [{"id": "slab", "tags": [tag_spec(t) for t in SLAB_TAGS]}]}  # fmt: skip


def _head_jpeg() -> bytes:
    T = head_T_base_cam_stance(_q(), [1.0, 0.0, 0.0, 0.0], T_site_cam=T_SITE_CAM)
    return _jpeg(_render(HEAD_LENS, T, [(SLAB_T, t) for t in SLAB_TAGS]))


def test_measure_setup_recovers_a_declared_entity_from_head_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _svc(StatePort(), FakeCam(_head_jpeg()), monkeypatch)
    out = _wait(svc, _setup_req(frames=3))
    assert out["ok"] and not out["moved"], out
    d = out["data"]
    assert d["frame"] == "base" and len(d["frames"]) == 3 and d["tags_seen"] == [1, 2, 3, 4]
    r = Registration.from_dict(d["entities"]["slab"])
    assert r.ok and r.used == (1, 2, 3, 4) and r.n_views == 3
    assert r.T_ref_entity is not None
    err_mm = 1000 * float(np.linalg.norm(r.xyz - SLAB_T[:3, 3]))
    assert err_mm < 1.0, err_mm  # the whole r1d path on exact frames: sub-millimetre (JPEG q95, the fit)
    assert abs(r.yaw_deg - 3.0) < 0.1
    assert d["camera"]["source"] == "pinhole stand-in" and "IMU" in d["camera"]["pose_from"]
    assert svc.status()["measure"]["last"]["ok"] is True  # status: the last measurement


def test_a_moving_head_is_refused_and_nothing_is_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _svc(StatePort(head_step=0.004), FakeCam(_head_jpeg()), monkeypatch)
    out = _wait(svc, _setup_req())
    assert not out["ok"] and out["code"] == "NOT_STILL" and out["details"]["head_motion_rad"] > 0.003


def test_no_new_frame_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _svc(StatePort(), FakeCam(_head_jpeg(), stale=True), monkeypatch)
    out = _wait(svc, {**_setup_req(), "wait_s": 0.5})
    assert not out["ok"] and out["code"] == "TIMEOUT"


def test_without_r1cam_the_word_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    from r1mc3.record.runner import CameraClient

    svc = _svc(StatePort(), None, monkeypatch)
    svc.camera = CameraClient(port=1)  # nothing listens on port 1
    out = _wait(svc, _setup_req())
    assert not out["ok"] and out["code"] == "UNSUPPORTED" and "r1cam" in out["details"]["reason"]


def test_without_a_calibration_the_word_refuses_like_pointing(monkeypatch: pytest.MonkeyPatch) -> None:
    """r1d's ONE head-camera calibration (the one pointing uses): none -> NOT_CALIBRATED (v2's rule)."""
    monkeypatch.setenv("R1MC3_HEAD_CALIB", "/nonexistent/head_camera.json")
    svc = Service(StatePort())
    MO.measure_ops(svc).observe = FakeCam(_head_jpeg())
    out = _wait(svc, _setup_req())
    assert not out["ok"] and out["code"] == "NOT_CALIBRATED", out


def test_one_camera_pose_and_one_lens_for_pointing_and_measurement() -> None:
    """measure's camera pose (headpose.head_T_base_cam_stance, its own geometry instance) = camera.observe's
    (Perception.T_base_site @ the mount), and ModelLens (the calibration dict through pointing's projection) = the
    simulator's HeadLens on the repository's calibration."""
    from r1mc3.cam.observe import calibration_from_dict
    from r1mc3.measure import HeadLens
    from r1mc3.measure.lenses import ModelLens
    from r1mc3.service.perception import Perception

    cal = lens_mod.load_calibration()
    q = _q()
    imu = np.array([0.9998, 0.012, -0.015, 0.003])
    imu /= np.linalg.norm(imu)
    svc = Service(StatePort())
    per = Perception(svc, calibration_from_dict({"T_site_cam": cal.T_site_cam.tolist(), "model": cal.model.as_dict()}))
    a = head_T_base_cam_stance(q, imu, T_site_cam=cal.T_site_cam)
    b = per.T_base_site(q, imu) @ cal.T_site_cam
    assert np.abs(a - b).max() < 1e-12
    ml, hl = ModelLens(cal.model.as_dict()), HeadLens(cal.model)
    px = np.array([[10.0, 12.0], [272.0, 224.0], [530.0, 440.0]])
    assert np.abs(ml.normalise(px) - hl.normalise(px)).max() < 1e-9
    P = np.array([[0.1, -0.05, 0.6], [-0.3, 0.2, 0.9]])
    assert np.abs(ml.project(P) - hl.project(P)).max() < 1e-9


def test_bad_specs_are_refused_before_anything_starts() -> None:
    svc = Service(StatePort())
    two = {"op": "measure.setup", "entities": [{"id": "a", "tags": [tag_spec(SLAB_TAGS[0])]},
                                               {"id": "b", "tags": [tag_spec(SLAB_TAGS[0])]}]}  # fmt: skip
    out = _wait(svc, two)
    assert out["code"] == "INVALID_ARGS" and "both a and b" in out["details"]["reason"]
    bent = tag_spec(SLAB_TAGS[0])
    bent["T_entity_tag"][0][0] = 2.0
    out = _wait(svc, {"op": "measure.setup", "entities": [{"id": "a", "tags": [bent]}]})
    assert out["code"] == "INVALID_ARGS" and "not a rotation" in out["details"]["reason"]
    both = {**tag_spec(SLAB_TAGS[0]), "xyz": [0.0, 0.0, 0.0]}
    out = _wait(svc, {"op": "measure.setup", "entities": [{"id": "a", "tags": [both]}]})
    assert out["code"] == "INVALID_ARGS"  # the schema: exactly one of T_entity_tag / xyz


def test_one_measurement_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _svc(StatePort(), FakeCam(_head_jpeg(), delay_s=0.3), monkeypatch)
    t1 = svc.submit(_setup_req(frames=2))
    second = svc.submit(_setup_req(frames=1)).result
    assert second is not None and second.code == "RESOURCE_BUSY"
    assert t1.done.wait(30.0) and t1.result is not None and t1.result.ok


# ---------------------------------------------------------------------------------------------- measure.phone
PHONE = OpenCVLens.pinhole(1400.0, 1600, 1200)
TABLE_TAGS = [flat_tag(20, 0.06, 0.08, 0.45), flat_tag(21, 0.06, 0.08, -0.45), flat_tag(22, 0.10, 0.55, 0.45),
              flat_tag(23, 0.10, 0.55, -0.45)]  # fmt: skip
BOX_T = np.eye(4)
BOX_T[:3, :3] = rodrigues([0.0, 0.0, math.radians(-4.0)])
BOX_T[:3, 3] = [0.30, -0.20, 0.012]  # a board 1.2 cm thick on the table, turned 4 deg
BOX_TAGS = [flat_tag(30, 0.05, -0.05, 0.08), flat_tag(31, 0.05, -0.05, -0.08), flat_tag(32, 0.05, 0.06, 0.0)]


def _phone_T(dx: float = 0.02, dy: float = -0.03, tilt: float = 7.0, roll: float = -3.0, yaw: float = 5.0) -> Any:
    down = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    R = rodrigues([0, 0, math.radians(yaw)]) @ rodrigues([math.radians(roll), math.radians(tilt), 0]) @ down
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([0.32 + dx, dy, 0.0]) - 0.95 * R[:, 2]
    return T


def test_measure_phone_corrects_the_phone_and_measures_on_the_table() -> None:
    T_cam = _phone_T()
    img = _render(PHONE, T_cam, [(np.eye(4), t) for t in TABLE_TAGS] + [(BOX_T, t) for t in BOX_TAGS])
    # a point on the box's top: where does the camera see it -> r1d maps it back at z = the top's height
    p_true = BOX_T @ np.array([0.0, 0.0, 0.0, 1.0])
    pc = np.linalg.inv(T_cam) @ p_true
    px = PHONE.project(pc[None, :3])[0]
    req = {
        "op": "measure.phone",
        "image_b64": base64.b64encode(_jpeg(img)).decode(),
        "intrinsics": {"f_px": PHONE.f_px, "width": PHONE.width, "height": PHONE.height},
        "table_tags": [tag_spec(t) for t in TABLE_TAGS],
        "entities": [{"id": "box", "tags": [tag_spec(t) for t in BOX_TAGS]}],
        "points": [{"name": "mark", "px": [float(px[0]), float(px[1])], "z_m": 0.012}],
    }
    svc = Service(StatePort())
    out = _wait(svc, req)
    assert out["ok"] and out["data"]["frame"] == "table", out
    phone = PhoneCorrection.from_dict(out["data"]["phone"])
    assert phone.ok and len(phone.used) >= 3 and phone.fit_rms_mm < 1.0  # (a far corner tag may be out of view)
    assert abs(phone.tilt_deg - math.hypot(7.0, 3.0)) < 1.0
    box = Registration.from_dict(out["data"]["entities"]["box"])
    assert box.ok and 1000 * float(np.linalg.norm(box.xyz[:2] - BOX_T[:2, 3])) < 1.0
    assert abs(box.yaw_deg + 4.0) < 0.2
    mark = np.asarray(out["data"]["points"]["mark"])
    assert 1000 * float(np.linalg.norm(mark[:2] - p_true[:2])) < 1.0


def test_measure_phone_refuses_a_size_mismatch_and_reports_too_few_tags() -> None:
    img = _render(PHONE, _phone_T(), [(np.eye(4), t) for t in TABLE_TAGS[:2]])
    base = {
        "op": "measure.phone",
        "image_b64": base64.b64encode(_jpeg(img)).decode(),
        "table_tags": [tag_spec(t) for t in TABLE_TAGS],
        "entities": [{"id": "box", "tags": [tag_spec(BOX_TAGS[0])]}],
    }
    svc = Service(StatePort())
    out = _wait(svc, {**base, "intrinsics": {"f_px": 1400.0, "width": 800, "height": 600}})
    assert out["code"] == "INVALID_ARGS" and "intrinsics" in out["details"]["reason"]
    out = _wait(svc, {**base, "intrinsics": {"f_px": 1400.0, "width": 1600, "height": 1200}})
    assert out["ok"] and not out["data"]["phone"]["ok"]  # a measurement report, not an error: the phone refused
    assert "only 2 table tags" in out["data"]["phone"]["reason"]
    assert out["data"]["entities"]["box"]["ok"] is False


def test_measure_phone_takes_the_clients_fit_limit_default_the_generic_one() -> None:
    """HW2 day 2: the phone's own check limit is a request field (a phone ~0.47 m above its table asks 3 mm):
    the default = r1mc3.measure.phone.MAX_FIT_MM (2 mm), bounded 0 < x <= 5 mm. Fails on d550f99 (no such field)."""
    from pydantic import ValidationError
    from r1mc3.measure.phone import MAX_FIT_MM

    base = {
        "op": "measure.phone",
        "image_b64": base64.b64encode(b"x" * 32).decode(),
        "intrinsics": {"f_px": 1400.0, "width": 1600, "height": 1200},
        "table_tags": [tag_spec(t) for t in TABLE_TAGS],
    }

    def parsed(r: dict[str, Any]) -> MeasurePhoneReq:
        m = REQUEST.validate_python(r)
        assert isinstance(m, MeasurePhoneReq)
        return m

    assert parsed(base).max_fit_mm == MAX_FIT_MM == 2.0
    assert parsed({**base, "max_fit_mm": 3.0}).max_fit_mm == 3.0
    for bad in (0.0, -1.0, 5.5):
        with pytest.raises(ValidationError):
            REQUEST.validate_python({**base, "max_fit_mm": bad})


def test_measure_phone_hands_the_limit_to_the_phone_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    """r1d's measure.phone runs correct_phone with the request's max_fit_mm (absent = 2 mm). Fails on d550f99."""
    import r1mc3.measure.phone as phone_mod

    seen: list[float] = []
    real = phone_mod.correct_phone

    def spy(*a: Any, **k: Any) -> PhoneCorrection:
        seen.append(float(k.get("max_fit_mm", phone_mod.MAX_FIT_MM)))
        return real(*a, **k)

    monkeypatch.setattr(phone_mod, "correct_phone", spy)
    img = _render(PHONE, _phone_T(), [(np.eye(4), t) for t in TABLE_TAGS])
    req = {
        "op": "measure.phone",
        "image_b64": base64.b64encode(_jpeg(img)).decode(),
        "intrinsics": {"f_px": PHONE.f_px, "width": PHONE.width, "height": PHONE.height},
        "table_tags": [tag_spec(t) for t in TABLE_TAGS],
    }
    svc = Service(StatePort())
    assert _wait(svc, req)["ok"] and _wait(svc, {**req, "max_fit_mm": 3.0})["ok"]
    assert seen == [2.0, 3.0]


# ---------------------------------------------------------------------------------------------- words / types
def test_the_registrations_survive_the_json_round_trip() -> None:
    r = Registration("e", True, "ok", SLAB_T.copy(), (1, 2), (3,), 2, 0.25, {"x_mm": 0.1}, {1: 0.5})
    back = Registration.from_dict(json.loads(json.dumps(r.as_dict())))
    assert back.ok and back.used == (1, 2) and back.missing == (3,) and back.tag_err_mm == {1: 0.5}
    assert back.T_ref_entity is not None and np.allclose(back.T_ref_entity, SLAB_T, atol=1e-6)


def test_the_r1_measure_words_build_requests_r1d_accepts(tmp_path: Any) -> None:
    plan = {"entities": _setup_req()["entities"], "table_tags": [tag_spec(t) for t in TABLE_TAGS]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    req, _ = build_request(["measure", "setup", str(tmp_path / "plan.json"), "--frames", "4"])
    assert REQUEST.validate_python(req).op == "measure.setup" and req["frames"] == 4
    (tmp_path / "still.jpg").write_bytes(_jpeg(np.full((12, 16), 200, np.uint8)))
    (tmp_path / "k.json").write_text(json.dumps({"f_px": 20.0, "width": 16, "height": 12}))
    req, _ = build_request(["measure", "phone", str(tmp_path / "plan.json"), "--image", str(tmp_path / "still.jpg"),
                            "--intrinsics", str(tmp_path / "k.json")])  # fmt: skip
    assert REQUEST.validate_python(req).op == "measure.phone"
    assert len(req["table_tags"]) == 4


def test_the_worker_does_not_hold_r1ds_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """While a measurement waits for frames, r1d's tick (which takes the request lock) keeps running."""
    svc = _svc(StatePort(), FakeCam(_head_jpeg(), delay_s=0.2), monkeypatch)
    t = svc.submit(_setup_req(frames=2))
    ticks = 0
    t_end = time.monotonic() + 0.3
    while time.monotonic() < t_end:
        got = svc._lock.acquire(timeout=0.05)
        assert got, "r1d's lock is held by the measurement"
        svc._lock.release()
        ticks += 1
    assert ticks > 3 and t.done.wait(30.0)
    assert threading.active_count() >= 1
