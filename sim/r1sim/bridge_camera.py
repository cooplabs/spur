"""The head camera on the DDS-bridge process stack (session 11): the HW2 rehearsals run `r1 observe / point / extent /
measure setup` and a client's pre-flight against r1d as a PROCESS, the way the robot does.

On the robot, r1cam is its own service on 127.0.0.1:8781 and r1d asks it for frames (`R1CAM_PORT`); r1d reads the
calibration from /etc/r1mc3/head_camera.json (`R1MC3_HEAD_CALIB`). Here `BridgeCamera(bridge, records)` gives the
same two things on the DDS-bridge simulator:

* an r1cam server (`r1mc3.cam.service.CamServer`, the real one) on an ephemeral port whose source renders the head
  camera of the BRIDGE's simulated robot at the moment a frame is asked for (`OnDemandSource`, r1sim.headcam_feed's
  lens chain: render -> remap -> vignetting -> JPEG q90; depth as uint16 mm);
* the calibration file r1d is pointed at = the file the simulated camera renders through (default: lens.
  CALIBRATION_FILE = the v3-frame mount, i.e. the very file the robot installs as /etc/r1mc3/head_camera.json - the
  rehearsal exercises the install item, not a sim stand-in).

Why a render THREAD of its own (not the bridge's physics thread, not r1cam's request threads): an OpenGL renderer
belongs to the thread that made it (macOS: another thread hangs - session 10), and the bridge's physics runs in real
time (a 20-40 ms render inside its loop would starve rt/lowstate). The render thread owns the renderer and a PRIVATE
MjData: each frame copies the physics' last published qpos (the snapshot `Simulation.read_state()` is built from,
under the simulation's lock: microseconds), runs the kinematics on the copy and renders it - the physics never waits.
The frame's receive stamp is the snapshot's CLOCK_MONOTONIC time (r1d dates the pose at capture from it), not the end
of the render.

What it does NOT model: the robot's camera latency (exposure -> receive, ~50 ms guessed until HW2's latency step: the
`latency_s` option delays delivery like r1cam's SimSource), the right stereo image (the connector does not ask for it),
the stereo / video_hub service switch (`r1 camera services` answers that it is not available in the sim). The renders
share this process's GIL with the bridge's publisher thread: `stats()` reports the render times so a rehearsal can
show the camera's cost next to the bridge's lowstate gaps (the robot's own number = HW2's camera impact test, C1).

    from r1sim.stack import ProcStack
    with ProcStack(domain=100, workdir=tmp, bridge={"scene": "table"}, camera=True) as st:
        st.cli("observe")            # r1d -> this camera -> an obs_id + T_base_cam
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import json
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from r1mc3.cam.frames import Frame, monotonic_ns
from r1mc3.cam.service import CamServer, CamService
from r1mc3.cam.sim import SimSource

from r1sim import lens, world

CAMERA_NAME = "head_cam"
RIGHT_CAMERA_NAME = "head_cam_right"
# The right stereo image (S11 / S12 record it): a camera BASELINE_M along the left camera's +x (OpenCV: right) with the
# left's lens model - nominal: the session-10 stereo fit's synthetic check used 60 mm; the real baseline, the right
# lens and the left-right rotation are what HW2's S12 stereo fit (camera_fit) measures. v2's connector never asks.
BASELINE_M = 0.06
RENDER_TIMEOUT_S = 30.0


def right_calibration(cal: lens.HeadCalibration, baseline_m: float = BASELINE_M) -> lens.HeadCalibration:
    """The right camera = the left's mount moved baseline_m along the left camera's x axis, the same lens model."""
    T_lr = np.eye(4)
    T_lr[0, 3] = float(baseline_m)
    T = np.asarray(cal.T_site_cam @ T_lr, dtype=np.float64)
    return dataclasses.replace(cal, T_site_cam=T, note=f"the simulated right camera: the left's mount + {baseline_m} m")


def camera_world(
    *,
    scene: Any = "floor",
    stance: str = "centred",
    stand: Any = None,
    extra: Callable[[mujoco.MjSpec, Any], None] | None = None,
    right: bool = True,
) -> world.World:
    """The world DdsBridge would build for (scene, stance, stand) - the same robot, the same scene (r1sim.scenes'
    values, as DdsBridge's `scene`), the stand for "hanging" (HW1_STAND by default, as the bridge) - WITH the
    calibrated head camera (lens.CALIBRATION_FILE) and (right=True) the right stereo camera, plus an optional
    `extra(spec, info)` (e.g. a calibration board or tags). A camera adds no physics: the robot simulates exactly as
    without it."""
    from r1sim.dds_bridge import HW1_STAND, stand_extra

    extras: list[Callable[[mujoco.MjSpec, Any], None]] = []
    if stance == "hanging":
        extras.append(stand_extra(stand or HW1_STAND))
    if extra is not None:
        extras.append(extra)
    if right:
        cal_r = right_calibration(lens.load_calibration())

        def add_right(spec: mujoco.MjSpec, _info: Any) -> None:
            lens.add_head_camera(spec, cal_r, lens.render_frame(cal_r.model), name=RIGHT_CAMERA_NAME)

        extras.append(add_right)

    def ex(spec: mujoco.MjSpec, info: Any) -> None:
        for e in extras:
            e(spec, info)

    return world.build_world("real", scene=scene, extra=ex if extras else None, head_camera=True)


class _World:
    """The two attributes HeadFeed reads of a World, for a model loaded from the bridge's compiled model."""

    def __init__(self, m: mujoco.MjModel) -> None:
        self.model = m


class _SnapshotSource(SimSource):
    """r1cam's sim source (each stream rendered at the robot's rate WHILE WANTED - an observe's demand or a recording's
    hold, as on the robot: a recording gets frames without anyone observing) whose frames carry the physics snapshot's
    time (+ the modelled latency) as their receive stamp (module docstring)."""

    name = "sim-bridge"

    def __init__(
        self, render: Callable[[str], Any], stamp: Callable[[str], int], streams: tuple[str, ...], latency_s: float
    ) -> None:
        super().__init__(render, streams=streams, latency_s=latency_s)
        self._stamp = stamp
        self._lat_ns = round(float(latency_s) * 1e9)
        self.render_ms: collections.deque[float] = collections.deque(maxlen=500)

    def publish(self, f: Frame) -> None:
        self.render_ms.append(1000.0 * float(self.render_s.get(f.stream, 0.0)))
        super().publish(dataclasses.replace(f, t_rx_ns=self._stamp(f.stream) + self._lat_ns))

    def stats(self) -> dict[str, Any]:
        out = super().stats()
        if self.render_ms:
            r = np.asarray(self.render_ms)
            out["render_ms"] = {"p50": round(float(np.median(r)), 1), "max": round(float(r.max()), 1), "n": len(r)}
        return out


# ---------------------------------------------------------------------------------------- the camera process (child)
_HDR = struct.Struct("<qdII")  # the snapshot's stamp (r1cam's clock, ns), its sim time (s), len(qpos), nmocap
# then qpos (f8) and the mocap bodies' pos (3 x f8) + quat (4 x f8): a moved mocap body (the stand, a board) is not in
# qpos


class _Renderer:
    """In the camera process: ONE thread owns the OpenGL renderers (macOS: a renderer belongs to its thread) and a
    private MjData; every frame asks the bridge for the physics' last qpos (`snapshot(stream)` -> stamp, qpos), runs
    the kinematics on its copy and renders."""

    def __init__(
        self, m: mujoco.MjModel, cal: lens.HeadCalibration, snapshot: Callable[[str], tuple[int, Any, Any]], *,
        depth: str, vignetting: bool, seed: int,
    ) -> None:  # fmt: skip
        self.m, self.cal, self.snapshot = m, cal, snapshot
        self.depth, self.vignetting, self.seed = depth, vignetting, seed
        self.has_right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, RIGHT_CAMERA_NAME) >= 0
        self.stamps: dict[str, int] = {}
        self._calls: queue.Queue[tuple[str, threading.Event, list[Any]]] = queue.Queue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="r1cam-sim-render", daemon=True)
        self._thread.start()
        if not self._ready.wait(RENDER_TIMEOUT_S) or self._error is not None:
            raise RuntimeError(f"the camera's render thread did not start: {self._error!r}")

    def _loop(self) -> None:
        from r1sim.headcam_feed import HeadFeed

        try:
            d = mujoco.MjData(self.m)
            feed = HeadFeed(_World(self.m), lambda: d, depth=self.depth, seed=self.seed, vignetting=self.vignetting,
                            calibration=self.cal)  # fmt: skip
            right = (
                lens.HeadCamera(self.m, right_calibration(self.cal), name=RIGHT_CAMERA_NAME, vignetting=self.vignetting)
                if self.has_right
                else None
            )
        except BaseException as e:  # reported by __init__
            self._error = e
            self._ready.set()
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                try:
                    stream, done, box = self._calls.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    self.stamps[stream], qpos, mocap = self.snapshot(stream)
                    d.qpos[:] = qpos
                    if mocap.size:
                        d.mocap_pos[:] = mocap[:, :3]
                        d.mocap_quat[:] = mocap[:, 3:]
                    mujoco.mj_kinematics(self.m, d)
                    mujoco.mj_comPos(self.m, d)
                    mujoco.mj_camlight(self.m, d)
                    box.append(
                        lens.jpeg(right.rgb(d)) if stream == "right" and right is not None else feed.render(stream)
                    )
                except BaseException as e:  # the requester sees it (r1cam reports the stream's error)
                    box.append(e)
                done.set()
        finally:
            feed.close()
            if right is not None:
                right.close()

    def render(self, stream: str) -> Any:
        """SimSource's render (any thread): queued to the render thread."""
        done = threading.Event()
        box: list[Any] = []
        self._calls.put((stream, done, box))
        if not done.wait(RENDER_TIMEOUT_S):
            raise TimeoutError(f"the camera did not render {stream!r} within {RENDER_TIMEOUT_S} s")
        if isinstance(box[0], BaseException):
            raise box[0]
        return box[0]

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)


class _SnapshotClient:
    """The camera process's side of the snapshot link (one persistent TCP connection to the bridge's process)."""

    def __init__(self, port: int) -> None:
        self.sock = socket.create_connection(("127.0.0.1", int(port)), timeout=10.0)
        self.f = self.sock.makefile("rb")
        self._lock = threading.Lock()

    def __call__(self, stream: str) -> tuple[int, Any, Any]:
        with self._lock:
            self.sock.sendall(stream.encode() + b"\n")
            stamp, _t, n, nm = _HDR.unpack(self.f.read(_HDR.size))
            q = np.frombuffer(self.f.read(8 * n), dtype="<f8")
            mocap = np.frombuffer(self.f.read(8 * 7 * nm), dtype="<f8").reshape(nm, 7)
        return int(stamp), q, mocap


def serve(argv: Sequence[str] | None = None) -> int:
    """The camera process: `python -m r1sim.bridge_camera serve --model M.mjb --calib C.json --snapshot-port P
    --records DIR [--port 0] [--depth truth|r1] [--latency-s 0] [--no-vignetting] [--seed 0]` -> prints {"ready": true,
    "port": N} and serves r1cam's protocol until its stdin closes (the bridge's process went away) or SIGTERM."""
    ap = argparse.ArgumentParser(prog="python -m r1sim.bridge_camera serve")
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--snapshot-port", type=int, required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--depth", default="truth", choices=("truth", "r1"))
    ap.add_argument("--latency-s", type=float, default=0.0)
    ap.add_argument("--no-vignetting", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    m = mujoco.MjModel.from_binary_path(a.model)
    cal = lens.load_calibration(a.calib)
    snap = _SnapshotClient(a.snapshot_port)
    r = _Renderer(m, cal, snap, depth=a.depth, vignetting=not a.no_vignetting, seed=a.seed)
    streams = ("left", "right", "depth") if r.has_right else ("left", "depth")
    src = _SnapshotSource(r.render, r.stamps.__getitem__, streams, a.latency_s)
    server = CamServer(CamService(src, Path(a.records)), port=a.port).start()
    print(json.dumps({"ready": True, "port": server.port, "streams": list(streams), "pid": os.getpid()}), flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def watch_stdin() -> None:  # the bridge's process holds our stdin: EOF = it went away
        sys.stdin.read()
        stop.set()

    threading.Thread(target=watch_stdin, daemon=True).start()
    stop.wait()
    server.close()
    r.close()
    return 0


# ------------------------------------------------------------------------------------------ the bridge's side (parent)
class BridgeCamera:
    """The bridge robot's head camera behind a real r1cam server in its OWN process (module docstring): the bridge's
    process only answers the camera process's qpos requests (microseconds each). Close it (or use `with`)."""

    def __init__(
        self,
        bridge: Any,
        records_dir: str | Path,
        *,
        port: int = 0,
        depth: str = "truth",
        calibration: lens.HeadCalibration | None = None,
        latency_s: float = 0.0,
        vignetting: bool = True,
        seed: int = 0,
        workdir: str | Path | None = None,
    ) -> None:
        m = bridge.sim.m
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME) < 0:
            raise ValueError("the bridge's world has no head camera: build it with r1sim.bridge_camera.camera_world()")
        self.bridge = bridge
        self.cal = calibration or lens.load_calibration()
        self._m = m
        self._d = mujoco.MjData(m)
        self.records_dir = Path(records_dir)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.workdir = Path(workdir) if workdir is not None else self.records_dir.parent
        self.qpos_at_render: dict[str, np.ndarray] = {}
        self.stamp_at_render: dict[str, int] = {}
        self._history: collections.deque[tuple[int, str, np.ndarray]] = collections.deque(maxlen=600)
        self.frames = 0
        self._lock = threading.Lock()
        # the snapshot link: one connection from the camera process
        self._lsock = socket.create_server(("127.0.0.1", 0))
        self._snap_port = int(self._lsock.getsockname()[1])
        self._stop = threading.Event()
        self._serve_t = threading.Thread(target=self._serve_snapshots, name="bridge-camera-snap", daemon=True)
        self._serve_t.start()
        model_path = self.workdir / "bridge_camera_model.mjb"
        mujoco.mj_saveModel(m, str(model_path), None)
        cmd = [sys.executable, "-m", "r1sim.bridge_camera", "serve", "--model", str(model_path),
               "--calib", str(self.calibration_file(self.workdir)), "--snapshot-port", str(self._snap_port),
               "--records", str(self.records_dir), "--port", str(port), "--depth", depth,
               "--latency-s", str(latency_s), "--seed", str(seed)]  # fmt: skip
        if not vignetting:
            cmd.append("--no-vignetting")
        self._log = (self.workdir / "r1cam_sim.stderr").open("ab")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log)
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        try:
            ready = json.loads(line)
        except json.JSONDecodeError:
            self.close()
            raise RuntimeError(f"the camera process did not start (see {self.workdir / 'r1cam_sim.stderr'})") from None
        self.port = int(ready["port"])
        self.streams = tuple(ready["streams"])

    # ------------------------------------------------------------------ the snapshot link
    def _serve_snapshots(self) -> None:
        self._lsock.settimeout(0.5)
        conn: socket.socket | None = None
        while not self._stop.is_set() and conn is None:
            try:
                conn, _ = self._lsock.accept()
            except TimeoutError:
                continue
            except OSError:
                return
        if conn is None:
            return
        f = conn.makefile("rb")
        sim = self.bridge.sim
        try:
            while not self._stop.is_set():
                line = f.readline()
                if not line:
                    return
                stream = line.decode().strip()
                stamp = monotonic_ns()  # r1cam's clock (frames.monotonic_ns: what r1d anchors a frame's age on)
                with sim._lock:  # the snapshot read_state() is built from (engine.Simulation._publish)
                    snap = sim._snap
                t, qpos = float(snap[0]), np.ascontiguousarray(snap[1], dtype="<f8")
                with self._lock:
                    self.qpos_at_render[stream] = qpos.copy()  # the truth of the frame (tests, rehearsals)
                    self.stamp_at_render[stream] = stamp
                    self._history.append((stamp, stream, self.qpos_at_render[stream]))
                    self.frames += 1
                mocap = np.ascontiguousarray(np.c_[sim.d.mocap_pos, sim.d.mocap_quat], dtype="<f8")
                conn.sendall(_HDR.pack(stamp, t, qpos.size, mocap.shape[0]) + qpos.tobytes() + mocap.tobytes())
        except OSError:
            return
        finally:
            conn.close()

    # ------------------------------------------------------------------ truth / status
    def snapshot(self) -> float:
        """The physics' last published qpos into this side's private MjData (kinematics) -> its sim time."""
        sim = self.bridge.sim
        with sim._lock:
            snap = sim._snap
        self._d.qpos[:] = snap[1]
        mujoco.mj_kinematics(self._m, self._d)
        mujoco.mj_camlight(self._m, self._d)
        return float(snap[0])

    def data_at_render(self, stream: str = "left") -> Any:
        """An MjData with the kinematics of the last `stream` frame's physics snapshot (the frame's truth)."""
        d = mujoco.MjData(self._m)
        with self._lock:
            d.qpos[:] = self.qpos_at_render[stream]
        mujoco.mj_kinematics(self._m, d)
        mujoco.mj_camlight(self._m, d)
        return d

    def data_at_capture(self, t_capture_ns: int, stream: str = "left") -> Any:
        """The truth of the frame r1d reported (`camera.observe`'s t_capture_ns, r1d's clock = CLOCK_MONOTONIC on the
        process stack, latency subtracted): the kinematics at the `stream` snapshot nearest to it on r1cam's clock."""
        offset = time.clock_gettime_ns(time.CLOCK_MONOTONIC) - monotonic_ns()  # 0 on Linux; macOS: two clocks
        want = int(t_capture_ns) - offset
        with self._lock:
            cands = [(abs(st - want), q) for st, s_, q in self._history if s_ == stream]
        if not cands:
            raise KeyError(f"no {stream} frame rendered yet")
        d = mujoco.MjData(self._m)
        d.qpos[:] = min(cands, key=lambda c: c[0])[1]
        mujoco.mj_kinematics(self._m, d)
        mujoco.mj_camlight(self._m, d)
        return d

    def T_world_cam(self) -> np.ndarray:
        """The simulated camera's pose NOW (world, OpenCV frame)."""
        self.snapshot()
        return np.asarray(lens.T_world_cam(self._m, self._d, CAMERA_NAME))

    def call(self, fn: Callable[[], Any]) -> Any:
        """fn() in the caller's thread (kept for the session-11 flag's users; the renderers are in the camera
        process now, this side has no OpenGL)."""
        return fn()

    def calibration_dict(self) -> dict[str, Any]:
        return {
            "T_site_cam": np.asarray(self.cal.T_site_cam).tolist(),
            "model": self.cal.model.as_dict(),
            "width": self.cal.model.width,
            "height": self.cal.model.height,
            "note": f"the simulated head camera ({self.cal.source})",
        }

    def calibration_file(self, workdir: str | Path) -> Path:
        """The file r1d should read (`R1MC3_HEAD_CALIB`): the calibration's own file when the camera renders through
        it unchanged (the robot's file), else a copy written to `workdir/head_camera.json`."""
        src = Path(self.cal.source)
        if src.exists():
            try:
                own = lens.load_calibration(src)
                if own.model == self.cal.model and np.allclose(own.T_site_cam, self.cal.T_site_cam, atol=1e-12):
                    return src
            except (OSError, KeyError, ValueError):
                pass
        out = Path(workdir) / "head_camera.json"
        out.write_text(json.dumps(self.calibration_dict(), indent=1))
        return out

    def stats(self) -> dict[str, Any]:
        from r1mc3.cam.service import request

        out: dict[str, Any] = {"frames": self.frames, "port": self.port, "calibration": str(self.cal.source),
                               "streams": list(self.streams)}  # fmt: skip
        try:
            st = request(self.port, {"op": "status"}, timeout_s=5.0)["data"]
            if "render_ms" in st:
                out["render_ms"] = st["render_ms"]
        except (OSError, KeyError, ValueError):
            out["status_error"] = "r1cam (sim) did not answer"
        return out

    def close(self) -> None:
        self._stop.set()
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        with contextlib.suppress(OSError):
            self._lsock.close()
        if getattr(self, "_log", None) is not None:
            self._log.close()

    def __enter__(self) -> BridgeCamera:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def attach_env(cam: BridgeCamera, env: dict[str, str], workdir: str | Path) -> Mapping[str, str]:
    """The two environment variables an r1d process needs to use this camera (the robot's names)."""
    env["R1CAM_PORT"] = str(cam.port)
    env["R1MC3_HEAD_CALIB"] = str(cam.calibration_file(workdir))
    return env


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        raise SystemExit(serve(sys.argv[2:]))
    raise SystemExit("usage: python -m r1sim.bridge_camera serve ... (started by BridgeCamera)")
