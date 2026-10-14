"""`python -m r1mc3.service` - r1d, the minimal service (session 5).

    python -m r1mc3.service --shm-name /r1mc3.rt --domain 0 [--port 8780] [--no-hands] [--no-release] [--wait-s 10]
                            [--tick-hz 100] [--profile config/robot_profile.r1_dex3.yaml]

Attaches to r1rt's segment (waiting up to --wait-s for r1rt to create it), starts the hands' DDS loop (unless
--no-hands) and the motion-switcher client (unless --no-release: then takeover / on_stand answer UNSUPPORTED), serves
localhost:PORT, and prints one JSON line when ready: {"ready": true, "pid": ..., "port": ..., "shm": ..., "domain": N,
"missing": [...]} (`missing` = binding functions this build lacks; the ops that need them answer UNSUPPORTED).
SIGINT / SIGTERM: stops serving; r1d never publishes anything and never damps on the way out (its streams simply
stop: the core's TTL / deadman bring the arms to HOLD).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from .port import ShmPort
from .server import DEFAULT_PORT, R1dServer
from .service import Service

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = REPO / "config" / "robot_profile.r1_dex3.yaml"


def profile_limits(path: Path) -> dict[str, tuple[float, float]] | None:
    """Joint name -> (q_min, q_max) from the robot profile (None when the profile is absent)."""
    if not path.exists():
        return None
    from r1mc3.config import load_profile

    p = load_profile(path)
    return {str(k): (float(v[0]), float(v[1])) for k, v in p.limits.position.value.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m r1mc3.service", description=__doc__.split("\n")[0])
    ap.add_argument("--shm-name", default="/r1mc3.rt")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--wait-s", type=float, default=10.0)
    ap.add_argument("--tick-hz", type=float, default=100.0)
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--no-release", action="store_true")
    ap.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    ap.add_argument("--rt-config", type=Path, default=None, help="r1rt's config file (default: $R1RT_CONFIG)")
    ap.add_argument("--no-planner", action="store_true", help="no planner worker (planned arm goals: UNSUPPORTED)")
    ap.add_argument("--no-battery", action="store_true", help="no battery reader (status without `battery`)")
    a = ap.parse_args(argv)

    port = ShmPort.attach(a.shm_name, wait_s=a.wait_s)
    releaser: Any = None
    if not a.no_release:
        try:
            from .release import MotionSwitcherReleaser

            releaser = MotionSwitcherReleaser(a.domain)
        except ImportError as e:
            print(f"r1d: no motion-switcher client ({e}); takeover / on_stand answer UNSUPPORTED", file=sys.stderr)
    hands: Any = None
    if not a.no_hands:
        from r1mc3.hands.dex3 import Dex3Hands

        hands = Dex3Hands(a.domain).start()
    from r1mc3.hw1.ops import Hw1Ops
    from r1mc3.record.runner import Recorder

    contact: Any = None
    if hands is not None and hasattr(port.view, "write_hands_contact"):
        from r1mc3.hands.contact import ContactPublisher

        contact = ContactPublisher(hands, port.view.write_hands_contact)  # r1hands: the single writer of the region

    planner: Any = None
    world: Any = None
    if not a.no_planner:  # session 9: the planner in its own worker process (a slow plan never blocks r1d's tick)
        from r1mc3.plan import PlannerWorker
        from r1mc3.world import WorldModel

        planner, world = PlannerWorker(), WorldModel()
    svc = Service(
        port,
        planner=planner,
        world=world,
        releaser=releaser,
        hands=hands,
        limits=profile_limits(a.profile),
        tick_hz=a.tick_hz,
        recorder_factory=Recorder,
        hw1_factory=lambda s: Hw1Ops(s, domain=a.domain),  # the stand tests (r1d's own rt/lowstate recorder), shutdown
        contact=contact,
        rt_config=a.rt_config,
    )
    # session 13 (ops): the R1's battery in r1d's status - READ-ONLY (subscribes rt/lf/bmsstate; never required: without
    # the dds extra / a network the status says why and r1d runs on)
    battery: Any = None
    if not a.no_battery:
        from r1mc3.battery import BatteryReader

        battery = BatteryReader(a.domain).start()
        svc.battery = battery  # type: ignore[attr-defined]
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    with R1dServer(svc, port=a.port) as srv:
        print(
            json.dumps(
                {
                    "ready": True,
                    "pid": os.getpid(),
                    "port": srv.port,
                    "shm": a.shm_name,
                    "domain": a.domain,
                    "missing": port.missing(),
                }
            ),
            flush=True,
        )
        stop.wait()
    if hands is not None:
        hands.close()
    if battery is not None:
        battery.close()
    if planner is not None:
        planner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
