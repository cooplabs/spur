"""Pointing's pure parts (r1mc3.cam.pointing / observe, session 10): the camera model = v2's / the simulator's, the
declared supports from world entities, v2's point rules on synthetic depth (a flat table, a raised box, glare, off the
outline, nothing declared), v2's object extent, the observation store / state history / calibration loading, and the
lockstep frame source's owner-thread rendering. None of this existed on the base (a041731: no r1mc3.cam.pointing)."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.cam import pointing as PT
from r1mc3.cam.observe import (
    ObservationStore,
    StateHistory,
    calibration_from_dict,
    depth_from_r1cam,
    load_calibration,
)
from r1mc3.cam.sim import OnDemandSource, OwnerThreadCalls
from r1mc3.world import Container, Pose, Surface

MODEL = {
    "lens": "pinhole",
    "f": 126.94,
    "cx": 259.0,
    "cy": 203.9,
    "k1": 0.0249,
    "depth": "z",
    "width": 544,
    "height": 448,
}  # fmt: skip  (models/head_camera's s3_cal values)
TABLE_Z = 0.75


def _camera(pitch_deg: float = 35.0, pos: tuple[float, float, float] = (0.06, 0.0, 1.10)) -> np.ndarray:
    """A head camera looking forward and down (OpenCV frame: x right, y down, z forward) in base."""
    p = math.radians(pitch_deg)
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, pos
    return T


def _render_depth(T: np.ndarray, boxes: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    """Depth (z along the axis, m) of the table plane + axis-aligned boxes (x0, x1, y0, y1, top_z): the first hit."""
    v, u = np.mgrid[0:448, 0:544].astype(float)
    rays_c = PT.unproject(MODEL, u.ravel(), v.ravel(), np.ones(u.size))  # z = 1
    rays_b = rays_c @ T[:3, :3].T
    o = T[:3, 3]
    best = np.full(u.size, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (TABLE_Z - o[2]) / rays_b[:, 2]
        best = np.where((t > 0) & np.isfinite(t), t, best)
        for x0, x1, y0, y1, top in boxes:
            tt = (top - o[2]) / rays_b[:, 2]
            p = o + tt[:, None] * rays_b
            hit = (tt > 0) & (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
            best = np.where(hit & (tt < best), tt, best)
            for xs in (x0, x1):  # the near / far faces (enough for a box seen from above-front)
                tt = (xs - o[0]) / rays_b[:, 0]
                p = o + tt[:, None] * rays_b
                hit = (tt > 0) & (p[:, 1] >= y0) & (p[:, 1] <= y1) & (p[:, 2] >= TABLE_Z) & (p[:, 2] <= top)
                best = np.where(hit & (tt < best), tt, best)
    best[~np.isfinite(best)] = np.nan
    return best.reshape(448, 544)  # z = 1 rays: t is the depth along the axis


def _supports(*extra: object) -> list[PT.Support]:
    table = Surface("table", Pose((0.43, 0.0, TABLE_Z)), size=(0.7, 1.2, 0.02))
    return PT.supports_from_entities([table, *extra])


def test_the_camera_model_is_v2s_and_the_simulators() -> None:
    from r1sim import lens

    cm = lens.CameraModel("pinhole", 126.94, 259.0, 203.9, k1=0.0249)
    px, py = np.array([5.0, 270.0, 530.0]), np.array([10.0, 200.0, 440.0])
    for d in (0.4, 1.2):
        a = PT.unproject(MODEL, px, py, np.full(3, d))
        b = lens.unproject(cm, px, py, np.full(3, d))
        assert np.allclose(a, b, atol=1e-12)
        assert np.allclose(PT.project(MODEL, a), np.c_[px, py], atol=1e-6)
    eq = {**MODEL, "lens": "equidistant", "k1": 0.0}
    qx, qy = np.array([150.0, 270.0, 380.0]), np.array([120.0, 200.0, 300.0])  # (within 90 deg of the axis)
    P = PT.unproject(eq, qx, qy, np.ones(3))
    assert np.allclose(PT.project(eq, P), np.c_[qx, qy], atol=1e-6)
    assert PT.check_model({"f": 0, "cx": 1, "cy": 1}) and PT.check_model({"lens": "fisheye", "f": 1, "cx": 1, "cy": 1})
    assert PT.check_model(MODEL) is None


def test_supports_from_entities_surfaces_and_container_floors() -> None:
    board = Surface("board", Pose.xyz_yaw(0.26, 0.0, TABLE_Z + 0.012, math.radians(30)), size=(0.356, 0.356, 0.012))
    bin = Container("bin", Pose((0.40, -0.37, TABLE_Z)), size_outer=(0.35, 0.26, 0.045), wall_m=0.004, floor_m=0.002)
    s = _supports(board, bin)
    by = {x.id: x for x in s}
    assert by["bin"].kind == "container" and by["bin"].z == pytest.approx(TABLE_Z + 0.002)
    assert by["bin"].rim_z == pytest.approx(TABLE_Z + 0.045) and by["bin"].half == pytest.approx((0.171, 0.126))
    # the board's yawed outline: a corner of its axis-aligned box lies outside it
    assert bool(by["board"].contains(0.26 + 0.17, 0.0)) and not bool(by["board"].contains(0.26 + 0.17, 0.17))
    z = PT.top_at(s, np.array([0.26, 0.40, 0.70, 2.0]), np.array([0.0, -0.37, 0.3, 0.0]))
    assert z[:3] == pytest.approx([TABLE_Z + 0.012, TABLE_Z + 0.002, TABLE_Z]) and np.isnan(z[3])
    # the line of sight meets the highest face whose outline holds the hit
    hit = PT.ray_hit(s, np.array([0.06, 0.0, 1.1]), np.array([0.2, 0.0, -0.35]) / np.hypot(0.2, 0.35))
    assert hit is not None and hit[0].id == "board"


def test_point_on_a_flat_table_is_the_ray_and_the_plane() -> None:
    T = _camera()
    depth = _render_depth(T, [])
    for px, py in ((272.0, 300.0), (100.0, 250.0), (450.0, 300.0)):
        o, ray = PT.ray_base(MODEL, T, px, py)
        truth = o + (TABLE_Z - o[2]) / ray[2] * ray
        r = PT.point(depth, MODEL, T, px, py, _supports())
        assert r["ok"] and r["surface_id"] == "table" and r["via"] == "plane", r
        assert np.allclose(r["point_base"], truth, atol=1e-4) and r["height_above_surface_m"] == 0.0


def test_point_on_a_raised_box_takes_its_height_from_depth() -> None:
    T = _camera()
    box = (0.30, 0.38, -0.04, 0.04, TABLE_Z + 0.10)  # a 10 cm box on the table
    depth = _render_depth(T, [box])
    top = np.array([0.34, 0.0, TABLE_Z + 0.10])
    pc = np.linalg.inv(T) @ np.r_[top, 1.0]
    px, py = PT.project(MODEL, pc[None, :3])[0]
    r = PT.point(depth, MODEL, T, float(px), float(py), _supports())
    assert r["ok"] and r["via"] == "plane+depth" and "relative" in r["height_rule"], r
    # v2's halfway rule: (absolute + relative) / 2 with a clear table around = the box's height (here exact)
    assert r["height_above_surface_m"] == pytest.approx(0.10, abs=0.004)
    assert np.allclose(r["point_base"], top, atol=0.006)
    # a 3 cm thing is below v2's 4 cm "raised" line: on the surface (the point where the ray meets the table)
    low = (0.30, 0.38, -0.04, 0.04, TABLE_Z + 0.03)
    r2 = PT.point(_render_depth(T, [low]), MODEL, T, float(px), float(py) + 2.0, _supports())
    assert r2["ok"] and r2["height_above_surface_m"] == 0.0


def test_glare_inside_the_outline_is_the_surface_and_off_the_outline_is_refused() -> None:
    T = _camera()
    depth = _render_depth(T, [])
    v, u = np.mgrid[0:448, 0:544]
    glare = depth.copy()
    spot = np.hypot(u - 272, v - 300) < 10  # a small glare spot: the ring around the click reads the table
    glare[spot] = glare[spot] + 0.15  # the depth reads 15 cm too far (below the table) at the click
    r = PT.point(glare, MODEL, T, 272.0, 300.0, _supports())
    assert r["ok"] and r["height_from"] == PT.BELOW_INSIDE_LABEL, r
    # the table's outline ends at x 0.78: a pixel whose ray meets the plane beyond it is refused
    far = PT.point(depth, MODEL, T, 272.0, 60.0, _supports())
    assert far["ok"] is False and far["code_hint"] in ("UNREACHABLE",), far
    # nothing declared: the depth alone (v2's fallback)
    alone = PT.point(depth, MODEL, T, 272.0, 300.0, [])
    assert alone["ok"] and alone["via"] == "depth" and alone["point_base"][2] == pytest.approx(TABLE_Z, abs=1e-3)
    assert PT.point(depth, MODEL, T, 600.0, 10.0, _supports())["code_hint"] == "INVALID_ARGS"


def test_object_extent_measures_a_box_footprint() -> None:
    T = _camera()
    box = (0.30, 0.378, -0.0335, 0.0335, TABLE_Z + 0.112)  # 7.8 x 6.7 x 11.2 cm
    depth = _render_depth(T, [box])
    top = np.array([0.339, 0.0, TABLE_Z + 0.112])
    pc = np.linalg.inv(T) @ np.r_[top, 1.0]
    px, py = PT.project(MODEL, pc[None, :3])[0]
    s = _supports()
    e = PT.object_extent(depth, MODEL, T, float(px), float(py), lambda X, Y, g: PT.top_at(s, X, Y, g))
    assert e["ok"] and e["shape"] in ("box", "tall"), e
    assert e["length_m"] == pytest.approx(0.078, abs=0.012) and e["width_m"] == pytest.approx(0.067, abs=0.012)
    assert e["max_height_m"] == pytest.approx(0.112, abs=0.006)
    assert np.allclose(e["centre_base"], [0.339, 0.0], atol=0.01)
    flat = PT.object_extent(_render_depth(T, []), MODEL, T, float(px), float(py), lambda X, Y, g: PT.top_at(s, X, Y, g))
    assert flat["ok"] is False and flat["flat_or_empty"] is True


def test_object_height_rules_are_v2s() -> None:
    assert PT.object_height(0.10, 0.0) == (True, 0.10, "relative to the surface around it")
    assert PT.object_height(0.02, 0.0)[0] is False
    assert PT.object_height(0.09, 0.07) == (True, 0.09, "absolute (a large object)")
    assert PT.object_height(0.02, None) == (True, 0.02, "absolute (no surface around to compare with)")


def test_observation_store_history_and_calibration(tmp_path: Path) -> None:
    st = ObservationStore(keep=2)
    ids = [st.new_id() for _ in range(3)]
    assert ids == ["h1", "h2", "h3"]
    h = StateHistory(span_s=1.0)
    for k in range(5):
        h.add(k * 100_000_000, np.full(26, k), None)
    s = h.at(250_000_000)
    assert s is not None and s[0] == 200_000_000 and h.at(-1) is None
    good = {"T_site_cam": np.eye(4).tolist(), "model": {"f": 127.0, "cx": 259.0, "cy": 204.0}, "note": "x"}
    c = calibration_from_dict(good)
    assert c.model["lens"] == "pinhole" and c.model["k1"] == 0.0 and len(c.sha256) == 64 and c.meta["note"] == "x"
    bad = dict(good, T_site_cam=(2 * np.eye(4)).tolist())
    with pytest.raises(ValueError, match="rigid"):
        calibration_from_dict(bad)
    assert load_calibration(tmp_path / "none.json") is None
    (tmp_path / "broken.json").write_text("{")
    with pytest.raises(ValueError, match="unreadable"):
        load_calibration(tmp_path / "broken.json")
    (tmp_path / "ok.json").write_text(json.dumps(good))
    assert load_calibration(tmp_path / "ok.json") is not None
    import base64

    mm = np.array([[0, 750], [65535, 1200]], dtype="<u2")
    d = depth_from_r1cam({"b64": base64.b64encode(mm.tobytes()).decode(), "width": 2, "height": 2})
    assert np.isnan(d[0, 0]) and np.isnan(d[1, 0]) and d[0, 1] == pytest.approx(0.75)


def test_on_demand_frames_render_in_the_owner_thread() -> None:
    owner = OwnerThreadCalls(timeout_s=5.0)
    seen: list[int] = []

    def render(stream: str) -> object:
        seen.append(threading.get_ident())
        return b"\xff\xd8jpeg" if stream == "left" else np.zeros((448, 544), dtype="<u2")

    src = OnDemandSource(render, streams=("left", "depth"), run_in_owner=owner.call)
    src.demand(["left"])  # from the owner: at once
    assert src.slots["left"].peek() is not None and seen == [threading.get_ident()]
    done = threading.Event()

    def ask() -> None:
        src.demand(["depth"])
        done.set()

    threading.Thread(target=ask, daemon=True).start()
    time.sleep(0.1)
    assert not done.is_set() and src.slots["depth"].peek() is None  # waits for the owner's service()
    assert owner.service() == 1 and done.wait(2.0)
    assert seen[-1] == threading.get_ident() and src.slots["depth"].peek() is not None


def test_the_pose_at_capture_is_never_older_than_the_request() -> None:
    """The episode runner's finding (session 10): with the capture time = r1d's time at the REPLY minus the frame's
    wall-clock age, a simulation running slower than the wall clock reached back past the request to the previous
    observe's state ('measured state 1481.9 ms before the capture'). Anchored at the request it cannot."""
    import base64

    from r1mc3.cam.observe import calibration_from_dict
    from r1mc3.service.perception import Perception
    from r1mc3.service.schemas import CameraObserveReq
    from r1sim import joints as J

    class Port:
        t = 1_000_000_000

        def now_ns(self) -> int:
            return self.t

        def state(self) -> dict[str, object]:
            return {"q": J.POSE_STAND.copy(), "imu_quat": [1.0, 0.0, 0.0, 0.0]}

    class Cam:
        def __init__(self, port: Port) -> None:
            self.port = port

        def request(self, req: dict[str, object]) -> dict[str, object]:
            t_rx = time.monotonic_ns()  # the frame arrives at once ...
            time.sleep(0.1)  # ... the reply 100 ms (wall) later, while r1d's (simulated) clock moves 2 ms
            self.port.t += 2_000_000
            jpg = base64.b64encode(b"\xff\xd8x").decode()
            return {"ok": True, "data": {"left": {"b64": jpg, "t_rx_ns": t_rx, "width": 544, "height": 448}}}

    class Svc:
        def __init__(self) -> None:
            self.port = Port()
            self.camera = Cam(self.port)
            self.world = None
            self.hands = None

    class Ticket:
        result: object = None

        def finish(self, r: object) -> None:
            self.result = r

    svc = Svc()
    cal = calibration_from_dict({"T_site_cam": np.eye(4).tolist(), "model": {"f": 127.0, "cx": 259.0, "cy": 204.0}})
    per = Perception(svc, cal)  # type: ignore[arg-type]
    req = CameraObserveReq(op="camera.observe", depth=False, max_age_s=1.0)
    outs = []
    for t_req in (1_000_000_000, 2_500_000_000):  # two observes 1.5 s apart
        svc.port.t = t_req
        tk = Ticket()
        assert per.observe_begin(req, tk) is None
        per._pool.shutdown(wait=True)
        per._pool = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)
        outs.append(tk.result)
    assert all(getattr(o, "ok", False) for o in outs), outs
    second = outs[1].data  # type: ignore[attr-defined]
    assert second["t_capture_ns"] >= 2_500_000_000 - 1_000_000, second["t_capture_ns"]
    assert "1500" not in second["pose_source"] and "before the capture" in second["pose_source"]


def test_the_camera_latency_comes_from_the_calibration_file(tmp_path: Path) -> None:
    """Session 11: HW2's latency step (C5, `camera_fit latency`) writes the measured exposure -> receive latency into
    /etc/r1mc3/head_camera.json as "latency_s"; r1d's pose at capture subtracts it. No key = 0 (the receive time). The
    file's own sha256 is reported next to r1d's fingerprint (base: no latency from the file, no file_sha256)."""
    import hashlib

    from r1mc3.service.perception import Perception

    d = {"T_site_cam": np.eye(4).tolist(), "model": {"lens": "pinhole", "f": 127.0, "cx": 272.0, "cy": 224.0}}
    f0 = tmp_path / "plain.json"
    f0.write_text(json.dumps(d))
    f1 = tmp_path / "with_latency.json"
    f1.write_text(json.dumps({**d, "latency_s": 0.083, "source": "HW2 S12"}))
    c0, c1 = load_calibration(f0), load_calibration(f1)
    assert c0 is not None and c1 is not None
    assert c1.meta["file_sha256"] == hashlib.sha256(f1.read_bytes()).hexdigest() and c0.sha256 == c1.sha256
    desc = c1.describe()
    assert desc["source"] == str(f1) and desc["provenance"] == "HW2 S12"  # the file's field no longer hides the path
    svc: Any = object()
    assert Perception(svc, c0).latency_s == 0.0
    assert Perception(svc, c1).latency_s == 0.083
    assert Perception(svc, c1, latency_s=0.05).latency_s == 0.05  # an explicit value wins
