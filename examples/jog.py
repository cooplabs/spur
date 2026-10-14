"""Jog the robot by hand through r1d's streaming API, in the simulator: a hand in Cartesian space, an elbow joint, the
head, the fingers.

    bash scripts/run.sh python examples/jog.py             # the keyboard (in a terminal), the real processes
    bash scripts/run.sh python examples/jog.py --lockstep  # the same in one process (no DDS), paced to the clock
    bash scripts/run.sh python examples/jog.py --script    # a fixed sequence instead of the keyboard: prints what
                                                           # moved, exit 0 when every step moved what it should

The streaming ops are "dead man's" commands: each request carries a time to live (`ttl`, at most 0.3-0.5 s) and r1d
integrates it at its own tick until it runs out - a client that stops sending (a crash, a lost connection, a released
key) stops the motion within the ttl. So a teleop loop RE-SENDS the command while the key is held:

    stream.arm_twist {arm, v: [vx, vy, vz] m/s (|v| <= 0.10), ttl}  the hand's grasp point at a velocity (base frame)
    stream.joint_jog {group, dq: [...] rad/s, ttl}                   joint velocities for one group (arm_left, head ...)
    stream.head      {rate: [dpitch, dyaw] rad/s, ttl}               the head (pitch + = down, yaw + = left)
    hand.preshape    {side, shape: "open" | "fist" | ...}            the fingers to a named pose (a goal: waits)

Keys (the robot's base frame: x forward, y left, z up):
    1 / 2      the LEFT / RIGHT arm
    w / s      the hand forward / back        a / d   the hand left / right        r / f   the hand up / down
    e / q      the elbow more flexed / straighter (the elbow's SMALLER value = more flexed)
    i / k      the head down / up             j / l   the head left / right
    c / o      the hand closed (a fist) / open
    space      stop the arm (it holds where it is)        x   quit

Arm twists are refused while r1d's world model holds declared entities (no collision look-ahead for streams yet):
this example runs on the bare floor and declares nothing. The robot balances on its own policy throughout.
"""

from __future__ import annotations

import argparse
import contextlib
import select
import sys
import termios
import time
import tty
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from r1mc3.client import R1, Side

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

SPEED_M_S = 0.05  # the hand's jog speed
JOINT_RAD_S = 0.4  # the elbow's and the head's jog rate
TTL_S = 0.3  # each streamed command's time to live
RESEND_S = 0.1  # the teleop loop re-sends a held command this often (3 per ttl)
ELBOW = 3  # the elbow's index in an arm group's 5 joints (shoulder pitch, roll, yaw, elbow, wrist roll)

# key -> (what, argument): "twist" a unit direction, "elbow" a sign, "head" (dpitch, dyaw) signs
KEYS: dict[str, tuple[str, Any]] = {
    "w": ("twist", (1.0, 0.0, 0.0)), "s": ("twist", (-1.0, 0.0, 0.0)),
    "a": ("twist", (0.0, 1.0, 0.0)), "d": ("twist", (0.0, -1.0, 0.0)),
    "r": ("twist", (0.0, 0.0, 1.0)), "f": ("twist", (0.0, 0.0, -1.0)),
    "e": ("elbow", -1.0), "q": ("elbow", 1.0),
    "i": ("head", (1.0, 0.0)), "k": ("head", (-1.0, 0.0)), "j": ("head", (0.0, 1.0)), "l": ("head", (0.0, -1.0)),
}  # fmt: skip


def send(r: R1, arm: Side, what: str, arg: Any) -> Any:
    """One streamed command (it lasts TTL_S unless re-sent)."""
    if what == "twist":
        return r.twist(arm, [SPEED_M_S * float(v) for v in arg], ttl=TTL_S)
    if what == "elbow":
        dq = [0.0] * 5
        dq[ELBOW] = JOINT_RAD_S * float(arg)
        return r.jog(f"arm_{arm}", dq, ttl=TTL_S)
    if what == "head":
        return r.call("stream.head", rate=[JOINT_RAD_S * float(v) for v in arg], ttl=TTL_S)
    raise ValueError(what)


def hold_for(sim: C.Sim, arm: Side, what: str, arg: Any, seconds: float) -> list[Any]:
    """Re-send one command every RESEND_S for `seconds` (what a held key does); the replies."""
    out = []
    n = max(1, round(seconds / RESEND_S))
    for _ in range(n):
        out.append(send(sim.r1, arm, what, arg))
        sim.wait(RESEND_S)
    sim.wait(TTL_S + 0.2)  # the last command runs out, the joint settles
    return out


def hand_point(r: R1, arm: Side) -> np.ndarray:
    return np.asarray(((r.status().data.get("arms") or {}).get(arm) or {}).get("hand_point"), dtype=np.float64)


# ------------------------------------------------------------------------------------------------- the fixed sequence
def script(sim: C.Sim, say: Callable[..., None]) -> bool:
    """A fixed sequence (no keyboard): each step held for its time, what moved measured, checked against a minimum."""
    r = sim.r1
    ok = True

    def check(name: str, moved: float, need: float, unit: str) -> None:
        nonlocal ok
        good = moved >= need
        ok = ok and good
        say(f"{name:34s} moved {moved:6.3f} {unit} (at least {need}: {'ok' if good else 'NOT ENOUGH'})")

    for arm in ("left", "right"):
        side: Side = "left" if arm == "left" else "right"
        p0 = hand_point(r, side)
        hold_for(sim, side, "twist", (0.0, 0.0, 1.0), 1.5)
        p1 = hand_point(r, side)
        check(f"the {arm} hand up 1.5 s", float(p1[2] - p0[2]), 0.04, "m")
        hold_for(sim, side, "twist", (1.0, 0.0, 0.0), 1.5)
        p2 = hand_point(r, side)
        check(f"the {arm} hand forward 1.5 s", float(p2[0] - p1[0]), 0.04, "m")
        hold_for(sim, side, "twist", (-1.0, 0.0, -1.0), 1.5)
        p3 = hand_point(r, side)
        check(f"the {arm} hand back + down 1.5 s", float(np.linalg.norm(p3 - p2)), 0.06, "m")
    e0 = sim.joint_q("right_elbow")
    hold_for(sim, "right", "elbow", -1.0, 1.0)
    e1 = sim.joint_q("right_elbow")
    check("the right elbow more flexed 1 s", e0 - e1, 0.2, "rad")
    hold_for(sim, "right", "elbow", 1.0, 1.0)
    check("the right elbow back 1 s", sim.joint_q("right_elbow") - e1, 0.2, "rad")
    h0 = sim.joint_q("head_pitch")
    hold_for(sim, "left", "head", (1.0, 0.0), 1.0)
    h1 = sim.joint_q("head_pitch")
    check("the head down 1 s", h1 - h0, 0.2, "rad")
    y0 = sim.joint_q("head_yaw")
    hold_for(sim, "left", "head", (0.0, 1.0), 1.0)
    check("the head left 1 s", sim.joint_q("head_yaw") - y0, 0.2, "rad")
    hold_for(sim, "left", "head", (-1.0, -1.0), 1.0)
    rep = r.preshape("left", "fist")
    say(f"{'the left hand closed (a fist)':34s} {rep.code}")
    ok = ok and rep.ok
    rep = r.preshape("left", "open")
    say(f"{'the left hand open':34s} {rep.code}")
    ok = ok and rep.ok
    say(f"body tilt {sim.tilt_deg():.1f} deg (the policy balanced throughout)")
    return ok


# ------------------------------------------------------------------------------------------------- the keyboard
@contextlib.contextmanager
def raw_keys() -> Iterator[Callable[[float], str | None]]:
    """The terminal in cbreak mode (a key at a time, no echo); yields `key(timeout_s) -> the key or None`."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def key(timeout_s: float) -> str | None:
        rd, _, _ = select.select([sys.stdin], [], [], timeout_s)
        return sys.stdin.read(1) if rd else None

    try:
        yield key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def keyboard(sim: C.Sim, say: Callable[..., None]) -> bool:
    """The teleop loop: a held key (the terminal's auto-repeat) keeps its command alive; nothing held = nothing sent,
    and the motion stops within the ttl."""
    r = sim.r1
    arm: Side = "left"
    if __doc__:
        print("Keys" + __doc__.split("Keys", 1)[1].split("Arm twists")[0].rstrip(), flush=True)
    held: tuple[str, Any] | None = None
    last_key = 0.0
    with raw_keys() as key:
        while True:
            t = time.monotonic()
            k = key(RESEND_S)
            if k is not None:
                last_key = t
                if k == "x":
                    return True
                if k in "12":
                    arm = "left" if k == "1" else "right"
                    say(f"the {arm} arm")
                elif k == " ":
                    say("stop:", r.arm_stop(arm).code)
                    held = None
                elif k in "co":
                    say("hand:", r.preshape(arm, "fist" if k == "c" else "open").code)
                elif k in KEYS:
                    held = KEYS[k]
            if held is not None and time.monotonic() - last_key < 0.6:  # auto-repeat's first gap is ~0.5 s
                rep = send(r, arm, *held)
                if not rep.ok:
                    say(f"{held[0]}: {rep.code} {rep.details.get('reason', '')}")
                    held = None
            else:
                held = None
            if sim.mode == "lockstep":  # simulated time paced to the clock
                sim.wait(max(0.0, RESEND_S - (time.monotonic() - t)))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--lockstep", action="store_true", help="one process, simulated time (no DDS needed)")
    ap.add_argument("--script", action="store_true", help="a fixed sequence instead of the keyboard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--domain", type=int, default=0, help="the DDS domain id of the process stack (loopback only)")
    a = ap.parse_args(argv)
    if not a.script and not sys.stdin.isatty():
        ap.error("the keyboard needs a terminal: run it in one, or use --script")
    say = C.say(time.monotonic())
    say("starting the simulator and the controller" + (" (lockstep)" if a.lockstep else " (real processes)"))
    sim_cm = C.lockstep(seed=a.seed, table=False) if a.lockstep else C.processes(domain=a.domain, table=False)
    with sim_cm as sim:
        say(f"balancing ({sim.mode}); body tilt {sim.tilt_deg():.1f} deg")
        ok = script(sim, say) if a.script else keyboard(sim, say)
    say("DONE" if ok else "FAILED: a step did not move what it should")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
