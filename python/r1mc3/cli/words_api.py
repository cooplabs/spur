"""The `r1` words for the api agent's session-10 ops (docs/operator/R1_CLI.md "session 10 (api)"): `move-path`,
`guarded-move`, `slide`, `pick`, `place`, `par`, `lease`, `twist`. Wired into r1mc3.cli.main through its api marker
blocks (`add_parsers` / `request`); pure (no r1d needed), tested in tests/unit/test_cli_s10_api.py.

Goal words wait for the goal's end unless `--no-wait` (as session 9's words). Every word here takes `--lease ID`
(the id `r1 lease acquire` printed) for a resource another client's lease holds.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

GOAL_WAIT_S = 60.0  # = cli.main.GOAL_WAIT_S (the goal's own timeout; the CLI adds a margin)
ARMS = ("left", "right")
GOAL_WORDS = ("move-path", "guarded-move", "slide", "pick", "place", "par")
WORDS = (*GOAL_WORDS, "lease", "twist")


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.replace(" ", "").split(",") if x]


def _words(s: str) -> list[str]:
    return [x for x in s.replace(" ", "").split(",") if x]


def add_parsers(sub: Any, goal_parser: Callable[[str], argparse.ArgumentParser]) -> None:
    p = goal_parser("move-path")
    p.add_argument("arm", choices=ARMS)
    p.add_argument("waypoints", type=json.loads, help="[[x, y, z], ...] (m, base frame)")
    p.add_argument("--linear", action="store_true", help="each leg a straight line")
    p.add_argument("--speed", type=float, default=0.4)
    p.add_argument("--blend", type=float, default=0.0, help="m: pass the waypoints within this, no stop (<= 0.05)")
    p = goal_parser("guarded-move")
    p.add_argument("arm", choices=ARMS)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--delta", type=_floats)
    g.add_argument("--position", type=_floats)
    g.add_argument("--stop-z", type=float)
    p.add_argument("--until", type=_words, default=["contact"], help="contact,blocked (comma list; '' = none)")
    p.add_argument("--touch", type=_words, default=[], help="declared entity ids the hand may touch")
    p.add_argument("--speed", type=float, default=0.2)
    p = goal_parser("slide")
    p.add_argument("arm", choices=ARMS)
    p.add_argument("delta", type=_floats)
    p.add_argument("--support", default=None)
    p.add_argument("--speed", type=float, default=0.2)
    p = goal_parser("pick")
    p.add_argument("arm", choices=ARMS)
    p.add_argument("position", type=_floats, help="the object's centre x,y,z (m, base frame)")
    p.add_argument("--height", type=float, default=0.05, help="the object's height (m)")
    p.add_argument("--grasp-z", type=float, default=None, help="the grasp centre's height (default: the centre)")
    p.add_argument("--grip", choices=("wrap", "pinch", "tip"), default="wrap")
    p.add_argument("--class", dest="object_class", choices=("rigid", "thin_flexible", "soft"), default="rigid")
    p.add_argument("--width", type=float, default=None)
    p.add_argument("--hover", type=float, default=0.03)
    p.add_argument("--lift", type=float, default=0.05)
    p.add_argument("--object-id", default=None)
    p = goal_parser("place")
    p.add_argument("arm", choices=ARMS)
    p.add_argument("position", type=_floats, help="where the grasp centre goes x,y,z (m, base frame)")
    p.add_argument("--lift", type=float, default=0.08)
    p.add_argument("--retreat", type=float, default=0.05)
    p = goal_parser("par")
    p.add_argument("steps", type=json.loads, help="[<goal request>, ...] on disjoint resources")
    p = sub.add_parser("lease")
    p.add_argument("action", choices=("acquire", "renew", "release", "list"))
    p.add_argument("arg", nargs="?", default=None, help="acquire: resources (comma list); renew / release: lease id")
    p.add_argument("--client", default="cli")
    p.add_argument("--ttl", type=float, default=30.0)
    p = sub.add_parser("twist")
    p.add_argument("arm", choices=ARMS)
    p.add_argument("v", type=_floats, help="vx,vy,vz (m/s, base frame; |v| <= 0.10)")
    p.add_argument("--ttl", type=float, default=0.3)
    for w in WORDS:
        sub.choices[w].add_argument("--lease", default=None, help="the lease id (a resource under a lease)")


def request(w: str, a: argparse.Namespace, ap: argparse.ArgumentParser) -> dict[str, Any] | None:
    """One of this module's words -> its request fields (None for any other word)."""
    if w not in WORDS:
        return None
    r: dict[str, Any]
    if w == "move-path":
        r = {"op": "arm.move_path", "arm": a.arm, "waypoints": a.waypoints, "linear": a.linear, "speed": a.speed}
        if a.blend:
            r["blend_m"] = a.blend
    elif w == "guarded-move":
        r = {"op": "arm.guarded_move", "arm": a.arm, "until": a.until, "touch": a.touch, "speed": a.speed}
        if a.delta is not None:
            r["delta"] = a.delta
        elif a.position is not None:
            r["position"] = a.position
        else:
            r["stop_z_m"] = a.stop_z
    elif w == "slide":
        r = {"op": "arm.slide", "arm": a.arm, "delta": a.delta, "speed": a.speed}
        if a.support is not None:
            r["support"] = a.support
    elif w == "pick":
        r = {"op": "skill.pick", "arm": a.arm, "position": a.position, "height_m": a.height, "grip": a.grip,
             "object_class": a.object_class, "hover_m": a.hover, "lift_m": a.lift}  # fmt: skip
        for k, v in (("grasp_z_m", a.grasp_z), ("width_m", a.width), ("object_id", a.object_id)):
            if v is not None:
                r[k] = v
    elif w == "place":
        r = {"op": "skill.place", "arm": a.arm, "position": a.position, "lift_m": a.lift, "retreat_m": a.retreat}
    elif w == "par":
        r = {"op": "par.run", "steps": a.steps}
    elif w == "lease":
        r = {"op": f"lease.{a.action}"}
        if a.action == "acquire":
            if a.arg is None:
                ap.error("r1 lease acquire <resource,resource,...>")
            r.update(resources=_words(a.arg), client=a.client, ttl_s=a.ttl)
        elif a.action in ("renew", "release"):
            if a.arg is None:
                ap.error(f"r1 lease {a.action} <lease id>")
            r["lease_id"] = a.arg
            if a.action == "renew":
                r["ttl_s"] = a.ttl
    else:
        r = {"op": "stream.arm_twist", "arm": a.arm, "v": a.v, "ttl": a.ttl}
    if w in GOAL_WORDS:
        r["wait"] = not a.no_wait
        r["timeout_s"] = GOAL_WAIT_S
    if a.lease is not None:
        r["lease"] = a.lease
    return r
