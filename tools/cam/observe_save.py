"""Save a head frame through r1d (camera.observe) for the operator's pointing check (HW2 row C2): writes the JPEG and
prints the obs_id + the camera pose, so `r1 point <obs_id> <px> <py>` can be given pixels read off the saved picture.
Moves nothing.

    bash scripts/run.sh python tools/cam/observe_save.py --out /var/lib/r1mc3/records/c2_head.jpg [--port 8780]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from r1mc3.service.client import Client, R1dError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="where to write the head JPEG")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--no-depth", action="store_true")
    a = ap.parse_args(argv)
    try:
        with Client(a.host, a.port, timeout_s=30.0) as c:
            out = c.call("camera.observe", depth=not a.no_depth, requester="operator")
    except R1dError as e:
        print(json.dumps({"ok": False, "reason": str(e)}))
        return 2
    if not out.get("ok"):
        print(json.dumps(out))
        return 1
    d = out["data"]
    Path(a.out).write_bytes(base64.b64decode(d.pop("jpeg_b64")))
    d.pop("right_jpeg_b64", None)
    keys = ("obs_id", "calibration", "T_base_cam", "frame_age_ms", "pose_source")
    print(json.dumps({"ok": True, "jpeg": a.out, **{k: d[k] for k in keys}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
