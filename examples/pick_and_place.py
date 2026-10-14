"""Pick a box up from a table and set it down 12 cm to the side - through r1d's API, in the simulator.

    bash scripts/run.sh python examples/pick_and_place.py             # the real processes (r1rt, r1guard, r1d)
    bash scripts/run.sh python examples/pick_and_place.py --lockstep  # the same requests in one process (no DDS)

What it does, one API request at a time (every reply is printed in plain words):

  1. tell r1d's world model what is there: the table (a surface) and the box (an object to grasp);
  2. open the left hand into a pinch;
  3. `arm.hover` over the box - the planner picks a hand pose where the pinch closes at the box's middle height and a
     collision-free way there (around the table's edge, the hands start below its top);
  4. `arm.lower` down the planner's own line to that pose;
  5. `hand.grasp` - the grip state machine closes the fingers and answers HOLDING / NOT_HOLDING from the fingers' and
     the pads' evidence;
  6. the box leaves the world model (it is in the hand now) and `arm.carry` takes it up, 12 cm to the right and down
     until it is just above the table;
  7. `hand.release`, then the hand moves up and away;
  8. the simulator's truth says where the box ended up.

Exit code 0 when the box rests within 4 cm of where it was asked to go; 1 otherwise (each step's reply says why).
The robot balances on its own policy all the way through (the C++ core); nothing here touches the legs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from r1mc3.client import Side

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

PLACE_SHIFT_M = (0.0, -0.12)  # where the box goes: 12 cm to the robot's right of where it starts
# how close counts as "where it was asked to go": in the hand the box arrives within ~5 mm of the target, then the
# fingers opening nudge it - in the simulator it rests 2.0-2.7 cm off (lockstep, seeds 0-4) and 1.3-2.8 cm off (the
# real processes, 5 runs)
PLACE_TOL_M = 0.04
GRIP: Literal["pinch"] = "pinch"  # a 5 cm box: the thumb against the index and middle fingertips
# the line the fingers close along, seen from above (deg, base frame): across two opposite faces of the box - left to
# itself the planner may pick the diagonal, and a pinch across a 5 cm box's corners must open ~7 cm (wider than it can)
CLOSE_YAW_DEG = 0.0
# the carry aims the box's bottom AT the table's height: the planner keeps the hand itself clear of the table, so the
# carry ends with the box ~1 cm above it (sim) and the box drops that far when the fingers open
DROP_CLEAR_M = 0.0


def _words(rep: Any) -> str:
    """One reply in a line: the code, the goal's end state, what stopped it, the hand's error (mm)."""
    d = rep.data
    out = [rep.code]
    if d.get("state"):
        out.append(str(d["state"]))
    sb = d.get("stopped_by")
    if isinstance(sb, dict) and sb.get("code"):
        out.append(f"stopped by {sb['code']}")
    if d.get("error_mm") is not None:
        out.append(f"hand within {float(d['error_mm']):.1f} mm of the plan")
    v = d.get("verdict")
    if isinstance(v, dict) and v.get("state"):
        out.append(f"grip {v['state']}")
    if not rep.ok:
        why = rep.details.get("reason") or rep.details.get("errors") or rep.details or d.get("reason")
        out.append(str(why)[:200])
    return ", ".join(out)


def pick_and_place(sim: C.Sim, say: Callable[..., None], *, arm: Side = "left", obj: str = "box") -> dict[str, Any]:
    """The recipe (module docstring) on `sim`; returns {ok, failed_at, start, target, end, error_m, steps}."""
    r = sim.r1
    start = sim.body_pos(obj)
    spec = C.table_object(obj)
    hh = spec.half_height
    width = 2.0 * spec.size[0]
    target = np.array([start[0] + PLACE_SHIFT_M[0], start[1] + PLACE_SHIFT_M[1], C.TABLE_TOP_Z + hh])
    out: dict[str, Any] = {"ok": False, "failed_at": None, "start": start.round(4).tolist(),
                           "target": target.round(4).tolist(), "steps": {}}  # fmt: skip

    def step(name: str, rep: Any) -> Any:
        out["steps"][name] = {"ok": rep.ok, "code": rep.code}
        say(f"{name:9s}", _words(rep))
        return rep

    say(f"the {obj} is at x {start[0]:.3f} y {start[1]:+.3f} z {start[2]:.3f} m (the robot's base frame)")
    step("declare", r.world_declare(C.table_entity()))
    step("declare", r.world_declare(C.object_entity(obj, start)))
    step("preshape", r.preshape(arm, GRIP))
    hover = {"arm": arm, "position": [float(start[0]), float(start[1]), C.TABLE_TOP_Z], "height_m": 2.0 * hh,
             "grip": GRIP, "target_id": obj, "grasp_yaw_deg": CLOSE_YAW_DEG, "wait": True,
             "timeout_s": 60.0}  # fmt: skip
    rep = step("hover", r.call("arm.hover", **hover))
    sb = rep.data.get("stopped_by") or {}
    if not rep.ok and isinstance(sb, dict) and sb.get("code") in ("GUARD_LAG", "GUARD_CONTACT"):
        # the hand met something on its way (the arm's tracking guard stopped it): the same hover from where it
        # stopped, 1 cm higher
        rep = step("hover", r.call("arm.hover", **{**hover, "height_m": 2.0 * hh + 0.01}))
    if not rep.ok:
        out["failed_at"] = "hover"
        return out
    orient = (rep.data.get("plan_details") or {}).get("orientation") or {}
    descent = orient.get("descent_m")
    if descent is None:  # no grasp-height pose came back: lower until the fingers touch something, at most 4 cm
        lower = {"until": "contact", "max_depth_m": 0.04}
    else:
        lower = {"until": "depth", "max_depth_m": round(float(descent), 4), "target_id": obj}
    rep = step("lower", r.call("arm.lower", arm=arm, wait=True, timeout_s=30.0, **lower))
    sb = rep.data.get("stopped_by") or {}
    if not rep.ok and not (isinstance(sb, dict) and sb.get("code") in ("GUARD_LAG", "GUARD_CONTACT")):
        out["failed_at"] = "lower"  # (a guard stop = the fingers rest on the box: grasp from there)
        return out
    rep = step("grasp", r.grasp(arm, grip=GRIP, object_class="rigid", width_m=width))
    if not rep.ok:
        out["failed_at"] = "grasp"
        return out
    gc = rep.data.get("grasp_centre_m")
    if gc is None:
        gc = ((r.status().data.get("arms") or {}).get(arm) or {}).get("hand_point")
    r.world_clear(obj)  # in the hand now: the planner keeps the arm clear of the world, not of what it holds
    # the carry's end: the grasp point as high above the target's floor as it was above the table, + DROP_CLEAR_M
    z_end = C.TABLE_TOP_Z + DROP_CLEAR_M + (float(gc[2]) - C.TABLE_TOP_Z) if gc is not None else target[2] + 0.04
    rep = step("carry", r.carry(arm, [float(target[0]), float(target[1]), z_end], lift_m=0.08))
    d = rep.data
    sb = d.get("stopped_by") or {}
    legs, done = list(d.get("legs") or []), int(d.get("legs_done") or 0)
    # the box meeting the table in the carry's LAST leg (its way down) stops it on the tracking guard: it is placed
    guard = isinstance(sb, dict) and sb.get("code") in ("GUARD_LAG", "GUARD_CONTACT")
    placed = guard and bool(legs) and done >= len(legs) - 1
    if not rep.ok and not placed:
        out["failed_at"] = "carry"
        return out
    p = sim.body_pos(obj)
    held = p[2] - (C.TABLE_TOP_Z + hh)
    grip = ((r.status().data.get("grips") or {}).get(arm) or {}).get("state")
    say(
        f"the {obj} is in the hand at x {p[0]:.3f} y {p[1]:+.3f}, {100 * held:.1f} cm above the table (the grip says "
        f"{grip})"
    )
    out["held_above_table_m"], out["grip_before_release"] = round(float(held), 4), grip
    step("release", r.release(arm))
    step("retreat", r.move_relative(arm, [0.0, 0.0, 0.06]))
    sim.wait(1.0)  # the box settles
    end = sim.body_pos(obj)
    err = float(np.hypot(*(end[:2] - target[:2])))
    on_table = abs(float(end[2]) - (C.TABLE_TOP_Z + hh)) < 0.01
    out.update(end=end.round(4).tolist(), error_m=round(err, 4), on_table=on_table, ok=err <= PLACE_TOL_M and on_table)
    rest = "resting on the table" if on_table else "NOT on the table"
    say(
        f"the {obj} ended at x {end[0]:.3f} y {end[1]:+.3f} z {end[2]:.3f} m: {100 * err:.1f} cm from the target, "
        f"{rest}; body tilt {sim.tilt_deg():.1f} deg"
    )
    if not out["ok"]:
        out["failed_at"] = "placement"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--lockstep", action="store_true", help="one process, simulated time (no DDS needed)")
    ap.add_argument("--seed", type=int, default=0, help="the simulator's sensor-noise seed")
    ap.add_argument("--domain", type=int, default=0, help="the DDS domain id of the process stack (loopback only)")
    ap.add_argument("--json", type=Path, default=None, help="write the result (every step's code) to this file")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    say = C.say(t0)
    say("starting the simulator and the controller" + (" (lockstep)" if a.lockstep else " (real processes)"))
    with C.lockstep(seed=a.seed) if a.lockstep else C.processes(domain=a.domain, seed=a.seed) as sim:
        say(f"balancing ({sim.mode}); body tilt {sim.tilt_deg():.1f} deg")
        res = pick_and_place(sim, say)
    res["mode"], res["wall_s"] = sim.mode, round(time.monotonic() - t0, 1)
    if a.json is not None:
        a.json.write_text(json.dumps(res, indent=1))
    say("DONE: the box is where it was asked to go" if res["ok"] else f"FAILED at {res['failed_at']}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
