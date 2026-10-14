"""`python -m r1mc3.cam [--source robot|fake|bench] [--port 8781] [--records DIR]` - the r1cam service (r1cam.service).

robot: the head stereo module (depth TCP + GStreamer); fake: a local stand-in in this process (a depth server on
127.0.0.1 + images from a Python child) for a quick bench; bench: depth from `--depth host:port` (a stand-in robot in
ANOTHER process, as the robot is) + images from the stand-in child, which can burn `--burn-ms` of CPU per frame to
stand for the robot's H.264 decode + JPEG encode (tools/cam/impact.py). The sim source is driven from a simulator
process (r1mc3.cam.sim.SimSource with a render callback), not from here.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from pathlib import Path

from .robot import DEPTH_ADDR, RobotSource
from .service import DEFAULT_PORT, CamServer, CamService
from .services import CameraServices, dds_rpc
from .streams import DEPTH_KEEP_S, DEPTH_RCVBUF_BYTES


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="r1cam", description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=("robot", "fake", "bench"), default="robot")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--records", default=os.environ.get("R1MC3_RECORD_DIR", "/var/lib/r1mc3/records"))
    ap.add_argument("--depth", default=f"{DEPTH_ADDR[0]}:{DEPTH_ADDR[1]}", help="host:port of the depth stream")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain of the robot's services (robot source)")
    ap.add_argument("--image-hz", type=float, default=7.5, help="bench: the stand-in image rate per stream")
    ap.add_argument("--image-bytes", type=int, default=60_000, help="bench: the stand-in JPEG size")
    ap.add_argument("--burn-ms", type=float, default=0.0, help="bench: CPU burnt per image (the decode's stand-in)")
    # session 12 (HW2 C1, streams.py): the depth's receive window and its keep after an observe - for the day-2 C1
    # comparison without a code change (`sudo systemctl edit r1cam`: Environment=R1CAM_DEPTH_RCVBUF=0 = the OS default)
    ap.add_argument("--depth-rcvbuf", type=int, default=int(os.environ.get("R1CAM_DEPTH_RCVBUF", DEPTH_RCVBUF_BYTES)),
                    help="the depth socket's receive buffer, bytes (0: the OS default)")  # fmt: skip
    ap.add_argument("--depth-keep-s", type=float, default=float(os.environ.get("R1CAM_DEPTH_KEEP_S", DEPTH_KEEP_S)),
                    help="the depth stream's keep after an observe's frame, s")  # fmt: skip
    a = ap.parse_args(argv)
    services: CameraServices | None = None
    if a.source == "robot":
        host, port = a.depth.rsplit(":", 1)
        src = RobotSource(depth_addr=(host, int(port)), depth_rcvbuf=a.depth_rcvbuf)
        services = CameraServices(lambda: dds_rpc(a.domain))  # made on the first services request only
    elif a.source == "bench":
        from .fake import image_command

        host, port = a.depth.rsplit(":", 1)
        src = RobotSource(
            depth_addr=(host, int(port)),
            image_command=lambda s, _p: image_command(s, a.image_hz, a.image_bytes, a.burn_ms),
            depth_rcvbuf=a.depth_rcvbuf,
        )
    else:
        from .fake import FakeRobot

        fake = FakeRobot().start()
        src = RobotSource(depth_addr=fake.depth_addr, image_command=fake.image_command)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    svc = CamService(src, Path(a.records), services=services, depth_keep_s=a.depth_keep_s)
    with CamServer(svc, port=a.port) as srv:
        print(json.dumps({"ready": True, "process": "r1cam", "source": a.source, "port": srv.port,
                          "records": a.records, "pid": os.getpid(), "depth_rcvbuf": a.depth_rcvbuf,
                          "depth_keep_s": a.depth_keep_s}), flush=True)  # fmt: skip
        stop.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
