"""`r1` - words -> r1d requests (DESIGN.md §5.3; docs/operator/R1_CLI.md).

    r1 status                                   nothing moves
    r1 takeover | r1 on-stand                   Unitree's controller lets go; ours blends in (see R1_CLI.md)
    r1 stop | halt | resume | freeze            stop: arms HOLD, keep balancing; freeze: STAND_FREEZE
    r1 damp                                     LIMP - the operator's word only (hold the robot first)
    r1 joint GROUP Q1,Q2,... [--ttl 0.2] [--kp ...] [--kd ...]       one L0 setpoint (STREAM, TTL)
    r1 hold GROUP [Q1,Q2,...]                   HOLD where it is (or at Q)
    r1 jog GROUP DQ1,DQ2,... [--ttl 0.3]        L1 joint rates for TTL seconds
    r1 head PITCH,YAW [--ttl 0.3]               the head to a setpoint
    r1 hand SIDE Q1..Q7 [--ttl 1] | r1 hand-state [SIDE]
    r1 record list | r1 record plan KIND [k=v ...] | r1 record run KIND [k=v ...] --yes
         [--joint J] [--side S] [--label L] [--dry-run]           (k=v and the flags both set parameters)
    r1 reclaim                                  after the guard-takeover test: r1rt takes the motors back (operator)
    r1 board-hold --pause-ms 20|100|300|400|500 [--yes]   the board-hold test: LowCmd paused, the guard OFF,
                                                ON_STAND (dry run without --yes)
    r1 guard-takeover [--stop-s 3] [--yes]      the guard-takeover test: r1rt stopped in ON_STAND, the guard ARMED
                                                holds; then r1 reclaim
    r1 shutdown [--label L]                     session end (MONITOR / DAMP only): r1log finishes, logs saved + verified
                                                into a NEW folder <sessions>/<L>_<UTC> (L default "session")
    r1 stand-freeze [--yes]                     S10: ON_STAND, legs + waist ramp to the stand pose (dry run: checks)
    r1 stand-hold                               back from that freeze into the ON_STAND hold (then r1 resume)
  session 9 - L2 / L3 goals (each waits for its end; --no-wait returns at acceptance with the goal id):
    r1 move-joint ARM Q1,..,Q5 [--speed 0.5]    a planned joint move of one arm (world-checked with a planner)
    r1 reach ARM X,Y,Z [--speed] [--prefer front|inward|outward|across]      the hand (grasp centre) to a point
    r1 hover ARM X,Y,Z [--height 0.10] [--approach front|inward|around|low] [--target ID] [--approach-tol D]
                                                lift + transit + descend to above a point
    r1 lower ARM [--until contact|surface|depth] [--depth 0.10] [--stop-z Z]  a guarded descent
    r1 carry ARM X,Y,Z [--lift 0.10]            rise + travel + lower
    r1 plan '<goal json>'                       a dry run of any arm goal (nothing moves)
    r1 arm-stop [ARM]                           stop an arm's goal / stream: HOLD where it is
    r1 grasp SIDE [--grip wrap|pinch|tip] [--class rigid|thin_flexible|soft] [--width W] [--squeeze S]
    r1 release SIDE | r1 preshape SIDE SHAPE | r1 tare SIDE
    r1 look-at X,Y,Z                            the head camera at a point (base frame)
    r1 lean DEG                                 the policy's lean (its trained range; 0 = upright)
    r1 skill ready|table_ready [--arms both|left|right] [--speed 0.4]
    r1 seq '<steps json>'                       requests one after another (the first failure ends it)
    r1 goal status|wait|cancel ID | r1 goals    follow a goal
    r1 world list | r1 world declare '<entity json>' | r1 world clear [ID]

Output: the result as one JSON object (--pretty for indented); exit code 0 if ok, 1 if refused, 2 if r1d is not
reachable. Every request from the CLI carries requester "operator" (a human typed it): damp is accepted only so.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from typing import Any

from r1mc3.service.client import Client, R1dError
from r1mc3.service.server import DEFAULT_PORT

GROUPS = ("arm_left", "arm_right", "head", "leg_left", "leg_right", "waist")
# s the CLI waits (r1d's own wait + margin)
LONG_WORDS = {"board_hold.pause": 40.0, "guard_takeover.stall": 40.0, "shutdown": 70.0}
GOAL_WAIT_S = 60.0  # a goal word waits this long for the goal's end (the goal's own timeout_s; the CLI adds a margin)
ARMS = ("left", "right")


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.replace(" ", "").split(",") if x]


def _kv(items: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"r1: expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


_NEG_LIST = re.compile(r"^-(\d|\.\d)[\d.eE+-]*,")


def _protect_negative_lists(argv: Sequence[str]) -> list[str]:
    """argparse reads "-0.25,0.01,..." as an unknown option (only a single negative number is recognised): a leading
    space makes it a value again (`_floats` drops spaces). `r1 joint leg_left -0.25,0.01,...` then works as typed."""
    return [" " + a if _NEG_LIST.match(a) else a for a in argv]


def build_request(argv: Sequence[str]) -> tuple[dict[str, Any], argparse.Namespace]:
    """The words -> one request dict (op + fields) and the global options. Pure: tested without r1d."""
    argv = _protect_negative_lists(argv)
    ap = argparse.ArgumentParser(prog="r1", description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--timeout", type=float, default=20.0)
    sub = ap.add_subparsers(dest="word", required=True)
    sub.add_parser("status")
    for w in ("takeover", "on-stand"):
        p = sub.add_parser(w)
        p.add_argument("--release-timeout", type=float, default=4.0)
    for w in ("stop", "halt", "resume", "freeze", "damp", "reclaim", "stand-hold"):
        sub.add_parser(w)
    p = sub.add_parser("shutdown")
    p.add_argument("--label", default=None, help="the session folder's name prefix (default: session)")
    p = sub.add_parser("stand-freeze")
    p.add_argument("--yes", action="store_true", help="really freeze (without --yes: the ramp time + the far check)")
    p = sub.add_parser("board-hold")
    p.add_argument("--pause-ms", type=int, required=True, choices=(20, 100, 300, 400, 500))  # HW1 #34
    p.add_argument("--yes", action="store_true", help="really pause r1rt (without --yes: the plan + checks only)")
    p = sub.add_parser("guard-takeover")
    p.add_argument("--stop-s", type=float, default=3.0)
    p.add_argument("--yes", action="store_true", help="really stop r1rt (without --yes: the plan + checks only)")
    p = sub.add_parser("joint")
    p.add_argument("group", choices=GROUPS)
    p.add_argument("q", type=_floats)
    p.add_argument("--ttl", type=float, default=0.2)
    for k in ("dq", "kp", "kd", "tau-ff"):
        p.add_argument(f"--{k}", type=_floats, default=None)
    p = sub.add_parser("hold")
    p.add_argument("group", choices=GROUPS)
    p.add_argument("q", type=_floats, nargs="?", default=None)
    p = sub.add_parser("jog")
    p.add_argument("group", choices=GROUPS)
    p.add_argument("dq", type=_floats)
    p.add_argument("--ttl", type=float, default=0.3)
    p = sub.add_parser("head")
    p.add_argument("q", type=_floats)
    p.add_argument("--ttl", type=float, default=0.3)
    p = sub.add_parser("hand")
    p.add_argument("side", choices=("left", "right"))
    p.add_argument("q", type=_floats)
    p.add_argument("--ttl", type=float, default=1.0)
    for k in ("dq", "kp", "kd", "tau"):
        p.add_argument(f"--{k}", type=_floats, default=None)
    p = sub.add_parser("hand-state")
    p.add_argument("side", choices=("left", "right"), nargs="?", default=None)
    p = sub.add_parser("record")
    p.add_argument("action", choices=("list", "plan", "run"))
    p.add_argument("kind", nargs="?", default=None)
    p.add_argument("params", nargs="*", default=[])
    p.add_argument("--yes", action="store_true", help="really run it (record run without --yes is a dry run)")
    p.add_argument("--dry-run", action="store_true", help="only plan + check (the default without --yes)")
    for k in ("joint", "side", "label"):
        p.add_argument(f"--{k}", default=None)

    # session 9: L2 / L3
    def goal_parser(name: str) -> argparse.ArgumentParser:
        gp = sub.add_parser(name)
        gp.add_argument("--no-wait", action="store_true", help="return at acceptance (the goal id) instead of its end")
        return gp

    p = goal_parser("move-joint")
    p.add_argument("arm", choices=ARMS)
    p.add_argument("q", type=_floats)
    p.add_argument("--speed", type=float, default=0.5)
    for w in ("reach", "hover", "carry"):
        p = goal_parser(w)
        p.add_argument("arm", choices=ARMS)
        p.add_argument("position", type=_floats)
        p.add_argument("--speed", type=float, default=None)
        p.add_argument("--frame", default="base")
        if w == "reach":
            p.add_argument("--prefer", choices=("front", "inward", "outward", "across"), default=None)
        if w == "hover":
            p.add_argument("--height", type=float, default=0.10)
            p.add_argument("--prefer", choices=("front", "inward", "outward", "across"), default=None)
        if w == "carry":
            p.add_argument("--lift", type=float, default=0.10)
    p = goal_parser("lower")
    p.add_argument("arm", choices=ARMS)
    p.add_argument("--until", choices=("contact", "surface", "depth"), default="contact")
    p.add_argument("--depth", type=float, default=0.10)
    p.add_argument("--stop-z", type=float, default=None)
    p = sub.add_parser("plan")
    p.add_argument("goal", type=json.loads)
    p = sub.add_parser("arm-stop")
    p.add_argument("arm", choices=ARMS, nargs="?", default=None)
    p = goal_parser("grasp")
    p.add_argument("side", choices=ARMS)
    p.add_argument("--grip", choices=("wrap", "pinch", "tip"), default="wrap")
    p.add_argument("--class", dest="object_class", choices=("rigid", "thin_flexible", "soft"), default="rigid")
    p.add_argument("--width", type=float, default=None)
    p.add_argument("--squeeze", type=float, default=None)
    p = goal_parser("release")
    p.add_argument("side", choices=ARMS)
    p = goal_parser("preshape")
    p.add_argument("side", choices=ARMS)
    p.add_argument("shape")
    p = sub.add_parser("tare")
    p.add_argument("side", choices=ARMS)
    p = goal_parser("look-at")
    p.add_argument("position", type=_floats)
    p = sub.add_parser("lean")
    p.add_argument("deg", type=float)
    p = goal_parser("skill")
    p.add_argument("name", choices=("ready", "table_ready"))
    p.add_argument("--arms", choices=("both", "left", "right"), default="both")
    p.add_argument("--speed", type=float, default=0.4)
    p = goal_parser("seq")
    p.add_argument("steps", type=json.loads)
    p = sub.add_parser("goal")
    p.add_argument("action", choices=("status", "wait", "cancel"))
    p.add_argument("goal_id", type=int)
    sub.add_parser("goals")
    p = sub.add_parser("camera")  # session 9: r1cam (R1CAM_PLAN.md); `services` moves nothing, operator + --yes only
    p.add_argument("what", choices=("status", "services"))
    p.add_argument("--stereo", action="store_true", help="services: Unitree's stereo service ON, video_hub OFF")
    p.add_argument("--video-hub", action="store_true", help="services: back to video_hub")
    p.add_argument("--yes", action="store_true", help="services: really switch (without it r1cam refuses)")
    # session 10: each agent adds its words' parsers between its markers
    # >>> s10:cam
    p = sub.add_parser("observe", help="camera.observe: a head frame -> obs_id + the camera pose (no image printed)")
    p.add_argument("--no-depth", action="store_true")
    p = sub.add_parser("point", help="camera.point: a pixel of frame obs_id -> a base point on the declared surfaces")
    p.add_argument("obs_id")
    p.add_argument("px", type=float)
    p.add_argument("py", type=float)
    p = sub.add_parser("extent", help="camera.object_extent: what stands at the pixel (from the depth)")
    p.add_argument("obs_id")
    p.add_argument("px", type=float)
    p.add_argument("py", type=float)
    # <<< s10:cam
    # >>> s10:measure
    p = sub.add_parser("measure")  # session 10: tags -> entity poses (moves nothing); the plan = a JSON file
    p.add_argument("what", choices=("setup", "phone"))
    p.add_argument(
        "plan", help="JSON: {entities: [{id, tags: [{id, size_m, T_entity_tag | xyz}]}], table_tags?, points?}"
    )
    p.add_argument("--frames", type=int, default=3, help="setup: new head frames (each with the head still)")
    p.add_argument("--image", help="phone: the still (JPEG / PNG)")
    p.add_argument("--intrinsics", help="phone: JSON {K | f_px, dist, width, height}")
    # <<< s10:measure
    # >>> s10:grasp
    p = goal_parser("calibrate-pinch")  # session 10: hand.calibrate {side, kind: air_pinch}
    p.add_argument("side", choices=ARMS)
    # <<< s10:grasp
    # >>> s10:planner
    # <<< s10:planner
    # >>> s10:stale
    # <<< s10:stale
    # >>> s10:api
    from .words_api import add_parsers as _api_parsers  # move-path, guarded-move, slide, pick, place, par, ...

    _api_parsers(sub, goal_parser)
    # <<< s10:api
    # >>> s11:core
    sub.add_parser(  # session 11
        "balance", help="from a freeze of a BALANCING robot back into BALANCE (operator; then r1 resume)"
    )
    # <<< s11:core
    # >>> s11:grip
    # <<< s11:grip
    # >>> s11:cam
    # <<< s11:cam
    # >>> s11:api
    # <<< s11:api
    # >>> s11:episode
    # <<< s11:episode
    # >>> s12:approach
    hp = sub.choices.get("hover")  # session 12 (HW2 day 1): the requested approach on `r1 hover`
    if hp is not None:
        hp.add_argument(
            "--approach",
            choices=("front", "inward", "around", "low"),
            default=None,
            help="front / inward: the fingers' heading family; around: a TALL object from the side, the palm turned to "
            "it (needs --target or --grasp-z); low: a FLAT object, the grip's point down at it",
        )
        hp.add_argument(
            "--approach-tol",
            type=float,
            default=None,
            help="refuse when the style is more than this many degrees off (default: the nearest, reported)",
        )
        hp.add_argument("--grasp-z", type=float, default=None, help="where the grip's point will close (base z, m)")
        hp.add_argument("--target", default=None, help="the declared object this hover is for (target_id)")
        hp.add_argument("--grip", choices=("wrap", "pinch", "tip"), default=None)
        hp.add_argument("--grasp-yaw", type=float, default=None, help="the closing line seen from above (deg, base)")
    lp = sub.choices.get("lower")
    if lp is not None:  # the descent after such a hover may come within the hand's distance of the same object
        lp.add_argument("--target", default=None, help="the declared object this descent is for (target_id)")
    # <<< s12:approach
    # >>> s12:grip
    # <<< s12:grip
    # >>> s12:cam
    # <<< s12:cam
    # >>> s12:fixes
    # <<< s12:fixes
    # >>> s13:tools
    # <<< s13:tools
    # >>> s13:click
    # <<< s13:click
    # >>> s13:setup
    # <<< s13:setup
    # >>> s13:pose
    # <<< s13:pose
    # >>> s13:grip
    # <<< s13:grip
    # >>> s13:core
    # <<< s13:core
    # >>> s13:ops
    # <<< s13:ops
    p = sub.add_parser("world")
    p.add_argument("action", choices=("list", "declare", "clear"))
    p.add_argument("arg", nargs="?", default=None)
    a, extra = ap.parse_known_args(list(argv))
    if extra:  # k=v parameters after a flag (`r1 record run statics --label x duration_s=30`): argparse leaves them
        if a.word != "record" or not all("=" in x and not x.startswith("-") for x in extra):
            ap.error(f"unrecognized arguments: {' '.join(extra)}")
        a.params = list(a.params) + extra

    req: dict[str, Any] = {"requester": "operator"}
    w = a.word
    if w in ("status", "stop", "halt", "resume", "freeze", "damp", "reclaim", "shutdown"):
        req["op"] = w
        if w == "shutdown" and a.label:
            req["label"] = a.label
    elif w == "stand-freeze":
        req.update(op="stand_freeze", dry_run=not a.yes)
    elif w == "stand-hold":
        req["op"] = "stand_hold"
    elif w == "board-hold":
        req.update(op="board_hold.pause", pause_ms=a.pause_ms, dry_run=not a.yes)
    elif w == "guard-takeover":
        req.update(op="guard_takeover.stall", stop_s=a.stop_s, dry_run=not a.yes)
    elif w in ("takeover", "on-stand"):
        req.update(op=w.replace("-", "_"), release_timeout_s=a.release_timeout)
    elif w == "joint":
        req.update(op="joint.command", group=a.group, q=a.q, ttl=a.ttl)
        for k in ("dq", "kp", "kd", "tau_ff"):
            v = getattr(a, k)
            if v is not None:
                req[k] = v
    elif w == "hold":
        req.update(op="joint.hold", group=a.group)
        if a.q is not None:
            req["q"] = a.q
    elif w == "jog":
        req.update(op="stream.joint_jog", group=a.group, dq=a.dq, ttl=a.ttl)
    elif w == "head":
        req.update(op="stream.head", q=a.q, ttl=a.ttl)
    elif w == "hand":
        req.update(op="hand.command", side=a.side, q=a.q, ttl=a.ttl)
        for k in ("dq", "kp", "kd", "tau"):
            v = getattr(a, k)
            if v is not None:
                req[k] = v
    elif w == "hand-state":
        req["op"] = "hand.state"
        if a.side is not None:
            req["side"] = a.side
    elif w == "record":
        req["op"] = f"record.{a.action}"
        if a.kind is not None:
            req["kind"] = a.kind
        req["params"] = _kv(a.params)
        for k in ("joint", "side", "label"):
            v = getattr(a, k)
            if v is not None:
                req["params"][k] = v
        req["dry_run"] = not (a.action == "run" and a.yes and not a.dry_run)
    elif w == "camera":
        if a.what == "status":
            req["op"] = "camera.status"
        else:
            act = "stereo" if a.stereo else "video_hub" if a.video_hub else "status"
            req.update(op="camera.services", action=act, confirm=bool(a.yes))
    elif w in GOAL_WORDS:
        req.update(_goal_request(w, a))
        req["wait"] = not a.no_wait
        req["timeout_s"] = GOAL_WAIT_S
    elif w == "plan":
        req.update(op="arm.plan", goal=a.goal)
    elif w == "arm-stop":
        req["op"] = "arm.stop"
        if a.arm is not None:
            req["arm"] = a.arm
    elif w == "tare":
        req.update(op="hand.tare", side=a.side)
    elif w == "lean":
        req.update(op="body.pose", lean_deg=a.deg)
    elif w == "goal":
        req.update(op=f"goal.{a.action}", goal_id=a.goal_id)
        if a.action == "wait":
            req["timeout_s"] = GOAL_WAIT_S
    elif w == "goals":
        req["op"] = "goal.list"
    elif w == "world":
        req["op"] = f"world.{a.action}"
        if a.action == "declare":
            if a.arg is None:
                ap.error("r1 world declare '<entity json>'")
            req["entity"] = json.loads(a.arg)
        elif a.action == "clear" and a.arg is not None:
            req["entity_id"] = a.arg
    # session 10: each agent builds its words' requests between its markers (`elif w == ...` inside its own `if`)
    # >>> s10:cam
    if w == "observe":
        req.update(op="camera.observe", depth=not a.no_depth, jpeg=False)
    elif w in ("point", "extent"):
        req.update(op="camera.point" if w == "point" else "camera.object_extent", obs_id=a.obs_id, px=a.px, py=a.py)
    # <<< s10:cam
    # >>> s10:measure
    if w == "measure":
        import base64
        from pathlib import Path

        plan = json.loads(Path(a.plan).read_text())
        req["op"] = f"measure.{a.what}"
        req["entities"] = plan.get("entities", [])
        if a.what == "setup":
            req["frames"] = a.frames
        else:
            if not a.image or not a.intrinsics:
                ap.error("r1 measure phone <plan.json> --image still.jpg --intrinsics k.json")
            req["image_b64"] = base64.b64encode(Path(a.image).read_bytes()).decode("ascii")
            req["intrinsics"] = json.loads(Path(a.intrinsics).read_text())
            req["table_tags"] = plan.get("table_tags", [])
            req["points"] = plan.get("points", [])
    # <<< s10:measure
    # >>> s10:grasp
    if w == "calibrate-pinch":  # session 10: the hand's air-pinch reference (the hand empty, in free air)
        req.update(op="hand.calibrate", side=a.side, kind="air_pinch", wait=not a.no_wait, timeout_s=GOAL_WAIT_S)
    # <<< s10:grasp
    # >>> s10:planner
    # <<< s10:planner
    # >>> s10:stale
    # <<< s10:stale
    # >>> s10:api
    from .words_api import request as _api_request

    api_req = _api_request(w, a, ap)
    if api_req is not None:
        req.update(api_req)
    # <<< s10:api
    # >>> s11:core
    if w == "balance":
        req["op"] = "balance"
    # <<< s11:core
    # >>> s11:grip
    # <<< s11:grip
    # >>> s11:cam
    # <<< s11:cam
    # >>> s11:api
    # <<< s11:api
    # >>> s11:episode
    # <<< s11:episode
    # >>> s12:approach
    if w == "hover":
        for k_arg, k_req in (
            ("approach", "approach_style"),
            ("approach_tol", "approach_tol_deg"),
            ("grasp_z", "grasp_z_m"),
            ("target", "target_id"),
            ("grip", "grip"),
            ("grasp_yaw", "grasp_yaw_deg"),
        ):
            v = getattr(a, k_arg, None)
            if v is not None:
                req[k_req] = v
    if w == "lower" and getattr(a, "target", None) is not None:
        req["target_id"] = a.target
    # <<< s12:approach
    # >>> s12:grip
    # <<< s12:grip
    # >>> s12:cam
    # <<< s12:cam
    # >>> s12:fixes
    if req.get("op") == "measure.setup":  # session 12: wait as long as r1d's own budget for it (+ a margin) - at
        from r1mc3.service.measure_ops import setup_budget_s  # 20 s the CLI said "unreachable" while r1d measured
        from r1mc3.service.schemas import MeasureSetupReq

        wait_s = float(req.get("wait_s", MeasureSetupReq.model_fields["wait_s"].default))
        a.timeout = max(a.timeout, setup_budget_s(int(req.get("frames", 3)), wait_s) + 10.0)
    # <<< s12:fixes
    # >>> s13:tools
    # <<< s13:tools
    # >>> s13:click
    # <<< s13:click
    # >>> s13:setup
    # <<< s13:setup
    # >>> s13:pose
    # <<< s13:pose
    # >>> s13:grip
    # <<< s13:grip
    # >>> s13:core
    # <<< s13:core
    # >>> s13:ops
    # <<< s13:ops
    return req, a


GOAL_WORDS = (
    "move-joint",
    "reach",
    "hover",
    "carry",
    "lower",
    "grasp",
    "release",
    "preshape",
    "look-at",
    "skill",
    "seq",
)


def _goal_request(w: str, a: argparse.Namespace) -> dict[str, Any]:
    """A goal word -> its L2 / L3 request (DESIGN §5.1)."""
    if w == "move-joint":
        return {"op": "arm.move_joint", "arm": a.arm, "q": a.q, "speed": a.speed}
    if w in ("reach", "hover", "carry"):
        r: dict[str, Any] = {"op": f"arm.{w}", "arm": a.arm, "position": a.position, "frame": a.frame}
        if a.speed is not None:
            r["speed"] = a.speed
        if w in ("reach", "hover") and a.prefer is not None:
            r["prefer"] = a.prefer
        if w == "hover":
            r["height_m"] = a.height
        if w == "carry":
            r["lift_m"] = a.lift
        return r
    if w == "lower":
        r = {"op": "arm.lower", "arm": a.arm, "until": a.until, "max_depth_m": a.depth}
        if a.stop_z is not None:
            r["stop_z_m"] = a.stop_z
        return r
    if w == "grasp":
        r = {"op": "hand.grasp", "side": a.side, "grip": a.grip, "object_class": a.object_class}
        if a.width is not None:
            r["width_m"] = a.width
        if a.squeeze is not None:
            r["squeeze"] = a.squeeze
        return r
    if w == "release":
        return {"op": "hand.release", "side": a.side}
    if w == "preshape":
        return {"op": "hand.preshape", "side": a.side, "shape": a.shape}
    if w == "look-at":
        return {"op": "head.look_at", "position": a.position}
    if w == "skill":
        return {"op": f"skill.{a.name}", "arms": a.arms, "speed": a.speed}
    return {"op": "seq.run", "steps": a.steps}


def main(argv: Sequence[str] | None = None) -> int:
    req, a = build_request(sys.argv[1:] if argv is None else argv)
    op = req.pop("op")
    timeout = max(a.timeout, LONG_WORDS.get(op, 0.0))
    if req.get("wait") or op == "goal.wait":
        timeout = max(timeout, GOAL_WAIT_S + 5.0)
    try:
        with Client(a.host, a.port, timeout_s=timeout) as c:
            out = c.call(op, **req)
    except R1dError as e:
        print(json.dumps({"ok": False, "code": "UNREACHABLE", "op": op, "details": {"reason": str(e)}}))
        return 2
    print(json.dumps(out, indent=2 if a.pretty else None))
    return 0 if out.get("ok") else 1
