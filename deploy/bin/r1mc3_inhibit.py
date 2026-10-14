#!/usr/bin/env python3
"""Hold a systemd-inhibit block lock (shutdown, sleep, idle) while r1rt / r1guard own the motors (DESIGN.md §2.1).

Every second it asks deploy/bin/r1mc3_motorcheck.py's decision (read-only; the same rules as the stop / deploy
refusal). While the answer is "refused" (or cannot tell) it keeps one `systemd-inhibit --mode=block ... sleep infinity`
child alive, so `systemctl reboot` / `poweroff` fail with the lock's reason instead of stopping r1rt under a robot that
depends on it; when the motors are free it ends the child. It publishes nothing and never touches the robot. The
hardware power switch and the E-stop are unaffected (they are the operator's).
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r1mc3_motorcheck as mc


def owned(shm_name: str, root: Path) -> bool:
    try:
        st = mc.read_fake(root, shm_name) if str(root) != "/" else mc.read_segment(shm_name)
    except Exception:
        return True
    return mc.decide(st)[0] != mc.EXIT_OK


def inhibit_cmd(reason: str) -> list[str]:
    return ["systemd-inhibit", "--what=shutdown:sleep:idle", "--who=r1mc3", f"--why={reason}", "--mode=block",
            "sleep", "infinity"]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shm-name", default="/r1mc3.rt")
    ap.add_argument("--root", default="/")
    ap.add_argument("--period-s", type=float, default=1.0)
    ap.add_argument("--cycles", type=int, default=0, help="stop after N checks (tests); 0 = forever")
    ap.add_argument("--print-only", action="store_true", help="print lock / unlock instead of running systemd-inhibit")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    child: subprocess.Popen[bytes] | None = None
    held = False
    running = True

    def _stop(_sig: int, _frm: FrameType | None) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    n = 0
    while running and (a.cycles == 0 or n < a.cycles):
        n += 1
        want = owned(a.shm_name, root)
        if want and not held:
            if a.print_only:
                print("lock", flush=True)
            else:
                child = subprocess.Popen(inhibit_cmd("r1rt / r1guard own the robot's motors: end the session first"))
            held = True
        elif not want and held:
            if a.print_only:
                print("unlock", flush=True)
            elif child is not None:
                child.terminate()
                child.wait(timeout=5)
                child = None
            held = False
        if child is not None and child.poll() is not None:  # the lock holder died: take it again next cycle
            child, held = None, False
        time.sleep(a.period_s)
    if child is not None:
        child.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
