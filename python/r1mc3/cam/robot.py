"""RobotSource: the R1 head stereo module on the backpack (ported from v2 r1mc2/native_cam.py, which ran 26 Sep - 9 Oct;
DESIGN §10.3 port map). No wrist cameras (the user, 11 Oct: Q1).

* Depth: TCP 192.168.123.161:5000; each frame = a 36-byte header `<IQQIIII` (magic 0x59313620, counter, timestamp,
  544, 448, magic, payload bytes) + 544 x 448 uint16 LE millimetres (0 / 65535 = none), ~10 FPS. The stream repeats
  some frames (same counter + timestamp): skipped. **~39 Mbit/s while connected** (544 x 448 x 2 B x 10/s) on the
  backpack's 100 Mbit/s link to the robot - so it connects only on demand.
* Left / right images: RTP H.264 on UDP 5002 / 5003, ~7.5 FPS, decoded by a gst-launch-1.0 subprocess into JPEGs on
  its stdout (v2's proven chain: software avdec_h264 - decodebin picks the Jetson NVMM decoder, whose output
  videoconvert cannot take; probe 3 Oct). Only with Unitree's `stereo_patch_pc1` ON and `video_hub` OFF (services.py).
* Full head video per trial (the user, 11 Oct: Q3): the incoming H.264 is ALREADY encoded - `record_pipeline` remuxes
  it into Matroska with no decode and no encode (no CPU encoder, no hardware encoder needed), next to the JPEG branch
  through a tee (one UDP socket per port).

Not on the Mac (no GStreamer): the image path is tested with a stand-in command that writes JPEGs to stdout.
"""

from __future__ import annotations

import os
import select
import shlex
import signal
import socket
import struct
import subprocess
import threading
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .frames import HEIGHT, WIDTH, Frame, monotonic_ns
from .source import IDLE_S, SourceBase
from .streams import DEPTH_RCVBUF_BYTES

DEPTH_ADDR = ("192.168.123.161", 5000)
HEADER = struct.Struct("<IQQIIII")
MAGIC = 0x59313620
DEPTH_BYTES = WIDTH * HEIGHT * 2
LEFT_PORT, RIGHT_PORT = 5002, 5003
STREAM_SERVICE = "stereo_patch_pc1"  # Unitree's service that sends these streams (the stream ids: "<service>/<stream>")
_RTP_CAPS = 'caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96"'
DECODE_TAIL = "avdec_h264 ! videoconvert ! jpegenc quality=95 ! fdsink fd=1 sync=false"


class DepthHeaderError(ValueError):
    pass


def pack_depth(counter: int, stamp: int, raw: npt.NDArray[np.uint16]) -> bytes:
    """One depth frame as the robot sends it (the fake robot of the tests; the golden layout of v2's probe captures)."""
    a = np.asarray(raw, dtype="<u2")
    if a.shape != (HEIGHT, WIDTH):
        raise ValueError(f"depth must be {HEIGHT} x {WIDTH}")
    return HEADER.pack(MAGIC, counter, stamp, WIDTH, HEIGHT, MAGIC, DEPTH_BYTES) + a.tobytes()


def parse_depth_header(b: bytes) -> tuple[int, int]:
    """(counter, timestamp) of a 36-byte header; DepthHeaderError if it is not the 544 x 448 uint16 stream."""
    m1, counter, ts, w, h, m2, n = HEADER.unpack(b)
    if (m1, m2, w, h, n) != (MAGIC, MAGIC, WIDTH, HEIGHT, DEPTH_BYTES):
        raise DepthHeaderError(f"unexpected depth header {(hex(m1), w, h, hex(m2), n)}")
    return int(counter), int(ts)


def depth_m(raw: bytes) -> npt.NDArray[np.float32]:
    """Raw uint16 mm -> metres (0 where the camera has none: 0 or 65535)."""
    a = np.frombuffer(raw, dtype="<u2").reshape(HEIGHT, WIDTH)
    out = a.astype(np.float32) / 1000.0
    out[(a == 0) | (a == 65535)] = 0.0
    return out


def decode_pipeline(port: int) -> str:
    """v2's LEFT_PIPELINE (native_cam.py) at JPEG quality 95 (R1CAM_PLAN.md §2: corner detection on the JPEG)."""
    return (
        f"udpsrc port={port} reuse=false {_RTP_CAPS} ! rtpjitterbuffer latency=100 drop-on-latency=true "
        f"! rtph264depay ! {DECODE_TAIL}"
    )


def record_pipeline(port: int, mkv_path: str) -> str:
    """The decode pipeline + a tee branch that writes the incoming H.264 into Matroska WITHOUT re-encoding (the full
    head video, Q3). The recording branch's queue is leaky: a slow disk drops video, it never stalls the decode."""
    return (
        f"udpsrc port={port} reuse=false {_RTP_CAPS} ! rtpjitterbuffer latency=100 drop-on-latency=true "
        f"! rtph264depay ! h264parse config-interval=-1 ! tee name=t "
        f"t. ! queue leaky=downstream max-size-buffers=30 ! {DECODE_TAIL} "
        f"t. ! queue leaky=downstream max-size-time=2000000000 ! matroskamux "
        f"! filesink location={shlex.quote(mkv_path)}"
    )


class DepthReader(threading.Thread):
    """The depth TCP stream, connected only while `source.wanted("depth")` (no socket otherwise). Session 12: the
    socket's receive buffer is set BEFORE connect (`rcvbuf`, streams.DEPTH_RCVBUF_BYTES; 0 = the OS default) - it is
    the TCP window the robot may fill, i.e. the most depth data that can queue ahead of the robot's state packets on
    the shared link (streams.py: HW2 C1)."""

    def __init__(
        self,
        source: SourceBase,
        addr: tuple[str, int] = DEPTH_ADDR,
        *,
        timeout_s: float = 3.0,
        rcvbuf: int = DEPTH_RCVBUF_BYTES,
    ) -> None:
        super().__init__(daemon=True, name="r1cam-depth")
        self.source, self.addr, self.timeout_s = source, addr, float(timeout_s)
        self.rcvbuf = int(rcvbuf)
        self.rcvbuf_effective: int | None = None  # what the OS reports (Linux: 2x the request, socket(7))
        self.connected = False
        self.repeats = 0
        self.connects = 0

    def _connect(self) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if self.rcvbuf > 0:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
            self.rcvbuf_effective = int(s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
            s.settimeout(self.timeout_s)
            s.connect(self.addr)
        except BaseException:
            s.close()
            raise
        self.connects += 1
        return s

    @staticmethod
    def _exact(s: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("depth stream closed")
            buf += chunk
        return bytes(buf)

    def run(self) -> None:
        src = self.source
        while not src.closed.is_set():
            if not src.wanted("depth"):
                time.sleep(0.05)
                continue
            try:
                with self._connect() as s:
                    self.connected, src.errors["depth"] = True, None
                    last: tuple[int, int] | None = None
                    while src.wanted("depth") and not src.closed.is_set():
                        counter, ts = parse_depth_header(self._exact(s, HEADER.size))
                        raw = self._exact(s, DEPTH_BYTES)
                        t = monotonic_ns()
                        if (counter, ts) == last:
                            self.repeats += 1
                            continue
                        last = (counter, ts)
                        src.publish(Frame("depth", raw, t, src.next_seq("depth"), src_counter=counter, src_stamp=ts))
            except Exception as ex:  # any failure: report, retry while wanted
                src.errors["depth"] = f"{type(ex).__name__}: {ex}"
                if src.wanted("depth"):
                    time.sleep(0.5)
            finally:
                self.connected = False


def split_jpegs(buf: bytes) -> tuple[list[bytes], bytes]:
    """Complete JPEGs (FFD8 ... FFD9; jpegenc writes no thumbnails) and the remainder."""
    out: list[bytes] = []
    while True:
        a = buf.find(b"\xff\xd8")
        b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
        if a < 0 or b < 0:
            return out, buf[a:] if a > 0 else buf
        out.append(buf[a : b + 2])
        buf = buf[b + 2 :]


class JpegStream(threading.Thread):
    """A subprocess writing JPEGs to stdout (gst-launch-1.0 on the robot), run only while the stream is wanted;
    killed after. `command` returns the argv for the current state (it changes when a recording starts / stops)."""

    def __init__(self, source: SourceBase, stream: str, command: Any) -> None:
        super().__init__(daemon=True, name=f"r1cam-{stream}")
        self.source, self.stream, self.command = source, stream, command
        self.proc: subprocess.Popen[bytes] | None = None
        self.restarts = 0
        self._argv: list[str] | None = None

    def run(self) -> None:
        src = self.source
        while not src.closed.is_set():
            if not src.wanted(self.stream):
                time.sleep(0.05)
                continue
            argv = list(self.command())
            try:
                p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            except FileNotFoundError as ex:
                src.errors[self.stream] = f"{ex}"
                time.sleep(1.0)
                continue
            self.proc, self._argv = p, argv
            buf, ended = b"", False
            try:
                assert p.stdout is not None
                fd = p.stdout.fileno()
                while src.wanted(self.stream) and not src.closed.is_set() and list(self.command()) == argv:
                    r, _, _ = select.select([fd], [], [], 0.1)
                    if not r:
                        if p.poll() is not None:
                            ended = True
                            break
                        continue
                    chunk = os.read(fd, 1 << 16)
                    if not chunk:
                        ended = True
                        break
                    jpegs, buf = split_jpegs(buf + chunk)
                    for j in jpegs:
                        src.publish(Frame(self.stream, j, monotonic_ns(), src.next_seq(self.stream)))
                    if len(buf) > 8 << 20:
                        buf = b""
            finally:
                if any("matroskamux" in a for a in argv):  # -e: SIGINT -> EOS -> the Matroska file is closed properly
                    p.send_signal(signal.SIGINT)
                    try:
                        p.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
                p.kill()
                try:
                    p.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
                if ended:
                    err = (p.stderr.read() or b"").decode("utf-8", "replace")[-300:] if p.stderr else ""
                    src.errors[self.stream] = f"pipeline ended: {err.strip() or 'no output'}"
                self.proc = None
                self.restarts += 1
            if ended and src.wanted(self.stream):
                time.sleep(1.0)


class RobotSource(SourceBase):
    """The head stereo module: depth (TCP) + left / right (GStreamer). `image_command(stream, record_path)` may be
    replaced (tests: a stand-in that writes JPEGs); `record_video(stream, path)` switches that stream's pipeline to the
    tee'd one that also writes the H.264 (None = back to decode only)."""

    name = "robot"

    def __init__(
        self,
        *,
        depth_addr: tuple[str, int] = DEPTH_ADDR,
        idle_s: float = IDLE_S,
        image_command: Any = None,
        streams: Sequence[str] = ("left", "right", "depth"),
        depth_rcvbuf: int = DEPTH_RCVBUF_BYTES,
    ) -> None:
        super().__init__(idle_s=idle_s, streams=streams)
        self._video: dict[str, str | None] = {"left": None, "right": None}
        self.image_command = image_command or self.gst_command
        self.threads: list[threading.Thread] = []
        if "depth" in self.streams:
            self.threads.append(DepthReader(self, depth_addr, rcvbuf=depth_rcvbuf))
        for s in ("left", "right"):
            if s in self.streams:
                self.threads.append(JpegStream(self, s, lambda s=s: self.image_command(s, self._video[s])))
        for t in self.threads:
            t.start()

    @staticmethod
    def gst_command(stream: str, record_path: str | None) -> list[str]:
        port = LEFT_PORT if stream == "left" else RIGHT_PORT
        pipe = decode_pipeline(port) if record_path is None else record_pipeline(port, record_path)
        return ["gst-launch-1.0", "-q", "-e", *shlex.split(pipe)]

    def record_video(self, stream: str, path: str | None) -> None:
        self._video[stream] = path

    def stream_id(self, stream: str) -> str:
        """The robot's frames come from Unitree's stereo service only (stereo_patch_pc1: left / right udp 5002 / 5003,
        depth tcp 5000; video_hub's path is not read by r1cam) - the id a calibration file names (session 12)."""
        return f"{STREAM_SERVICE}/{stream}"

    def stats(self) -> dict[str, Any]:
        out = super().stats()
        for t in self.threads:
            if isinstance(t, DepthReader):
                out["depth_reader"] = {"connects": t.connects, "repeats_skipped": t.repeats, "rcvbuf": t.rcvbuf,
                                       "rcvbuf_effective": t.rcvbuf_effective}  # fmt: skip
        return out
