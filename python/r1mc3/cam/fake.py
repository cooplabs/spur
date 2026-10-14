"""A stand-in for the R1's head stereo module, for tests and the bench (no robot, no GStreamer on the Mac).

* depth: a TCP server on 127.0.0.1 sending the robot's frame format (robot.pack_depth) at `depth_hz` (default 10:
  the real ~39 Mbit/s), a smooth synthetic depth (a table-like slope 0.6-1.2 m with a few holes), every
  `repeat_every`-th frame sent twice (the real stream repeats some frames);
* images: `image_command(stream, record_path)` = a Python child writing JPEG-SHAPED blobs (SOI + filler without any
  0xFF byte + EOI, `image_bytes` long: the real ~60 KB) to stdout at `image_hz` - exercises the JPEG splitter and the
  bandwidth, not a decoder.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from typing import Any

import numpy as np

from .frames import HEIGHT, WIDTH
from .robot import pack_depth

CHILD = (
    "import sys,time\n"
    "hz=float(sys.argv[1]); n=int(sys.argv[2]); tag=sys.argv[3].encode(); burn=float(sys.argv[4])/1e3\n"
    "fill=(bytes(range(0,255))*(n//255+1))[:max(0,n-4-len(tag))]\n"
    "blob=b'\\xff\\xd8'+tag+fill+b'\\xff\\xd9'\n"
    "out=sys.stdout.buffer; t=time.monotonic()\n"
    "while True:\n"
    "    b0=time.perf_counter()\n"
    "    while time.perf_counter()-b0<burn: pass\n"
    "    out.write(blob); out.flush(); t+=1.0/hz\n"
    "    time.sleep(max(0.0,t-time.monotonic()))\n"
)


def image_command(stream: str, hz: float = 7.5, nbytes: int = 60_000, burn_ms: float = 0.0) -> list[str]:
    """The stand-in image child: JPEG-shaped blobs at `hz`, burning `burn_ms` of CPU per frame (the decode)."""
    return [sys.executable, "-I", "-c", CHILD, str(hz), str(nbytes), stream, str(burn_ms)]


def synthetic_depth(counter: int) -> np.ndarray:
    yy, _xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    d = 1200.0 - 600.0 * yy / (HEIGHT - 1) + 5.0 * np.sin(counter / 10.0)
    out: np.ndarray = d.astype(np.uint16)
    out[::37, ::41] = 0  # holes
    return out


class FakeRobot:
    def __init__(
        self, *, depth_hz: float = 10.0, image_hz: float = 7.5, image_bytes: int = 60_000, repeat_every: int = 7
    ) -> None:
        self.depth_hz, self.image_hz, self.image_bytes = float(depth_hz), float(image_hz), int(image_bytes)
        self.repeat_every = int(repeat_every)
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(2)
        self.depth_addr: tuple[str, int] = ("127.0.0.1", int(self._srv.getsockname()[1]))
        self._stop = threading.Event()
        self.connections = 0
        self.frames_sent = 0
        self.bytes_sent = 0
        self._t = threading.Thread(target=self._serve, daemon=True, name="fake-depth")

    def start(self) -> FakeRobot:
        self._t.start()
        return self

    def _serve(self) -> None:
        self._srv.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn: socket.socket) -> None:
        counter = 0
        t = time.monotonic()
        with conn:
            while not self._stop.is_set():
                counter += 1
                frame = pack_depth(counter, time.time_ns() // 1000, synthetic_depth(counter))
                try:
                    for _ in range(2 if self.repeat_every and counter % self.repeat_every == 0 else 1):
                        conn.sendall(frame)
                        self.frames_sent += 1
                        self.bytes_sent += len(frame)
                except OSError:
                    return
                t += 1.0 / self.depth_hz
                time.sleep(max(0.0, t - time.monotonic()))

    def image_command(self, stream: str, record_path: Any = None) -> list[str]:
        return image_command(stream, self.image_hz, self.image_bytes)

    def close(self) -> None:
        self._stop.set()
        self._srv.close()
