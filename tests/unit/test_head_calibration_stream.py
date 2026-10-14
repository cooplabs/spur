"""Session 12: a head-camera calibration names the stream it was made on, r1cam names the stream of every frame, and
r1d refuses pointing / measurement on a frame the calibration does not fit (another size, another stream) - HW2 day 1
read a fit artefact as "the stream changed under the calibration"; now a real change would be caught in words. A file
that names no stream (the installed 4 Oct file) is not checked for it. Fails on the base 1cf3afd (no
`calibration_mismatch`, no `stream_id`)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from r1mc3.cam.fake import FakeRobot
from r1mc3.cam.observe import Observation, ObservationStore, calibration_mismatch, load_calibration
from r1mc3.cam.robot import RobotSource
from r1mc3.cam.service import CamServer, CamService, request
from r1mc3.service.perception import Perception
from r1mc3.service.schemas import Result

HC = Path(__file__).resolve().parents[2] / "models" / "head_camera"
INSTALLED = HC / "calibration_s3_cal_20261004_v3frame.json"  # = the robot's /etc/r1mc3/head_camera.json (sha f18fe402)
CANDIDATE = HC / "calibration_c3_20261012_candidate_v3frame.json"


def test_a_frame_of_another_size_or_stream_does_not_fit_the_calibration() -> None:
    cand = load_calibration(CANDIDATE)
    assert cand is not None and cand.meta["stream"] == "stereo_patch_pc1/left"
    assert calibration_mismatch(cand, stream_id="stereo_patch_pc1/left", width=544, height=448) is None
    why = calibration_mismatch(cand, stream_id="sim/left", width=544, height=448)
    assert why is not None and "'stereo_patch_pc1/left'" in why and "'sim/left'" in why
    why = calibration_mismatch(cand, stream_id="stereo_patch_pc1/left", width=640, height=480)
    assert why is not None and "544 x 448" in why
    old = load_calibration(INSTALLED)
    assert old is not None and "stream" not in old.meta  # the installed file names no stream: not checked for it
    assert calibration_mismatch(old, stream_id="anything/left", width=544, height=448) is None


def test_r1cam_names_the_stream_of_every_frame(tmp_path: Path) -> None:
    fake = FakeRobot(depth_hz=10.0, image_hz=15.0, image_bytes=8_000).start()
    src = RobotSource(depth_addr=fake.depth_addr, image_command=fake.image_command, idle_s=1.0)
    srv = CamServer(CamService(src, tmp_path), port=0).start()
    try:
        ob = request(srv.port, {"op": "observe", "depth": True})
        assert ob["ok"], ob
        assert ob["data"]["left"]["stream_id"] == "stereo_patch_pc1/left"
        assert ob["data"]["depth"]["stream_id"] == "stereo_patch_pc1/depth"
    finally:
        srv.close()
        fake.close()


def test_r1d_refuses_pointing_on_a_frame_the_calibration_does_not_fit() -> None:
    cand = load_calibration(CANDIDATE)
    assert cand is not None
    per = object.__new__(Perception)  # only the frame lookup is exercised (no service, no camera)
    per.store = ObservationStore()
    why = calibration_mismatch(cand, stream_id="sim/left", width=544, height=448)
    o = Observation(per.store.new_id(), 0, 0, 544, 448, b"", None, np.eye(4), cand, calibration_mismatch=why)
    per.store.put(o)
    out = per._frame_or_fail(SimpleNamespace(op="camera.point", id=7, obs_id=o.obs_id))
    assert isinstance(out, Result) and out.code == "NOT_CALIBRATED" and "sim/left" in out.details["reason"]
    good = Observation(per.store.new_id(), 0, 0, 544, 448, b"", None, np.eye(4), cand)
    per.store.put(good)
    assert per._frame_or_fail(SimpleNamespace(op="camera.point", id=8, obs_id=good.obs_id)) is good


def test_the_candidate_is_c3s_hold_still_lens_and_not_the_installed_file() -> None:
    d = json.loads(CANDIDATE.read_text())
    assert d["model"]["f"] == 125.063 and d["model"]["cx"] == 257.698 and "NOT installed" in d["status"]
    assert "INTERIM" in d["mount_source"]
    assert CANDIDATE.read_bytes() != INSTALLED.read_bytes()
