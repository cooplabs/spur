"""A lockstep r1d rig with the PHYSICAL Dex3 hands (session 10): the simulator (real profile + a scene with a table) +
the REAL v3 core (r1rt_py through r1sim.v3_core) + r1d (`r1mc3.service.Service`, ticked at 100 Hz in simulated time) +
the planner (inline: a pure function of the world snapshot) + both Dex3-1 hands simulated in MuJoCo and offered to r1d
through r1hands' interface (`SimHandsPort`: `command` / `state` in `r1mc3.hands.dex3.Dex3Hands`' format - raw tau_est,
absolute pressure taxels from the sensor model - at the hands' 100 Hz, exactly as r1d drives the real hands).

Session 9's lockstep rigs drove r1d's hands with a geometric fake (tests/unit/grip_fake.py) and the A8 rigs drove the
simulated hand with a replayed arm; this rig closes the loop: r1d's `arm.hover` / `arm.lower` / `hand.grasp` /
`arm.carry` / `hand.release` act on simulated objects through the same code paths as on the robot (the planner's aim,
the executor, the grip verdict, the lift check from the arm's measured motion).

Single-threaded: `call(req)` submits a request to r1d between simulation steps and steps the simulator until the
request's ticket finishes (r1d's tick runs inside the simulator's controllers, as in tests/integration's Rig).
Task-level code (a client's episode runner, a grasp validation grid) uses it; nothing here names a task object (the
scene's objects come from the scene's own info).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from . import dex3, world
from . import joints as J
from .engine import HandCmd, LowCmd, LowState, Simulation
from .sensors import SensorConfig, SensorModel

FloatArray = npt.NDArray[np.float64]

HAND_HZ = 100.0  # r1hands' state / command rate (Dex3Hands; dex3_server's 100 Hz)
SIDES = ("left", "right")


class SimHandsPort:
    """r1hands' interface (r1d's `HandsLike`: `command` / `state`) over the simulated Dex3 hands, both sides.

    `state(side)` gives Dex3Hands' keys from the last state the hands' controller saw: q, dq, tau_est (the sensor
    model's units: "raw" = 1e-6 N m, as the robot's dex3_server), temperature (7 x 2), pressure (9 x 12 raw pad units),
    the command on the wire, valid. `command(...)` takes Dex3Hands' arguments; a command lasts `ttl_s`, then the hand
    holds where it is (Dex3Hands' rule). The commands reach the simulator as hand-only writes (`write_cmd(body=False)`):
    the body's board and the hands' boards are separate input paths on the robot (engine.write_cmd)."""

    def __init__(self, sim: Simulation, *, tau_units: str = "raw") -> None:
        from r1mc3.hands import dex3 as hd

        self._hd = hd
        self.sim = sim
        self.tau_units = tau_units
        self._st: LowState | None = None
        self._cmd: dict[str, HandCmd] = {s: self._fixed(dex3.preset("open", s)) for s in SIDES}
        self._until: dict[str, float] = {s: float("inf") for s in SIDES}
        self._dirty = True
        self.commands = 0
        self.ticks = 0
        self.on_tick: Callable[[int], None] | None = None

    def _fixed(self, q: FloatArray) -> HandCmd:
        h = HandCmd()
        h.q[:] = q
        h.kp[:] = self._hd.KP_DEFAULT
        h.kd[:] = self._hd.KD_DEFAULT
        return h

    def __call__(self, st: LowState) -> LowCmd | None:
        """The hands' controller (registered at HAND_HZ): the fresh state in, the TTL rule, a hand-only write."""
        self._st = st
        self.ticks += 1
        t = float(st.t)
        for s in SIDES:
            if t > self._until[s]:  # TTL over: hold where it is
                self._cmd[s] = self._fixed(np.asarray(st.hand_q[s], dtype=np.float64).copy())
                self._until[s] = float("inf")
                self._dirty = True
        if self.on_tick is not None:
            self.on_tick(round(t * 1e9))
        if self._dirty:
            self.sim.write_cmd(LowCmd(hands={s: c.copy() for s, c in self._cmd.items()}), body=False)
            self._dirty = False
        return None

    def command(
        self,
        side: str,
        q: Any,
        *,
        dq: Any = None,
        kp: Any = None,
        kd: Any = None,
        tau: Any = None,
        ttl_s: float = 1.0,
    ) -> dict[str, Any]:
        hd = self._hd
        if side not in SIDES:
            raise ValueError(f"side: one of {SIDES}")
        if not 0.0 < ttl_s <= hd.TTL_MAX_S:
            raise ValueError(f"ttl_s: (0, {hd.TTL_MAX_S}]")
        qa = np.asarray(q, dtype=np.float64)
        qc = hd.clamp_q(qa, side)
        kpa = hd.KP_DEFAULT if kp is None else np.asarray(kp, dtype=np.float64)
        kda = hd.KD_DEFAULT if kd is None else np.asarray(kd, dtype=np.float64)
        if np.any(kpa < 0) or np.any(kpa > hd.KP_MAX) or np.any(kda < 0) or np.any(kda > hd.KD_MAX):
            raise ValueError(f"gains: kp in [0, {hd.KP_MAX}], kd in [0, {hd.KD_MAX}]")
        h = HandCmd()
        h.q[:] = qc
        h.dq[:] = np.zeros(hd.N) if dq is None else np.asarray(dq, dtype=np.float64)
        h.kp[:] = kpa
        h.kd[:] = kda
        h.tau_ff[:] = np.clip(np.zeros(hd.N) if tau is None else np.asarray(tau, dtype=np.float64), -hd.TAU_MAX,
                              hd.TAU_MAX)  # fmt: skip
        self._cmd[side] = h
        self._until[side] = float(self.sim.time) + float(ttl_s)
        self._dirty = True
        self.commands += 1
        # applied at once (the next physics step), not at the hands' next tick: r1d's tick and the hands' are
        # different processes on the robot, the command goes out on DDS when it is made
        self.sim.write_cmd(LowCmd(hands={side: h.copy()}), body=False)
        self._dirty = False
        return {"ok": True, "side": side, "q": qc.tolist(), "clamped_q": bool(np.any(qc != qa)), "ttl_s": ttl_s}

    def state(self, side: str) -> dict[str, Any]:
        st = self._st if self._st is not None else self.sim.read_state()
        temp = np.asarray(st.hand_temp.get(side, np.full(7, 30.0)), dtype=np.float64)
        pres = st.hand_pressure.get(side)
        c = self._cmd[side]
        return {
            "q": np.asarray(st.hand_q[side], dtype=np.float64).copy(),
            "dq": np.asarray(st.hand_dq[side], dtype=np.float64).copy(),
            "tau_est": np.asarray(st.hand_tau[side], dtype=np.float64).copy(),
            "temperature": np.stack([temp, temp], axis=1),
            "motorstate": [0] * 7,
            "pressure": np.zeros((9, 12)) if pres is None else np.asarray(pres, dtype=np.float64).copy(),
            "error": [0, 0],
            "power_v": 24.0,
            "q_cmd": c.q.copy(),
            "kp": c.kp.copy(),
            "kd": c.kd.copy(),
            "tau_ff": c.tau_ff.copy(),
            "valid": True,
            "invalid_reason": None,
            "age_ms": 0.0,
            "t": float(st.t),
            "tau_units": self.tau_units,
        }


def arms_pose(name: str) -> FloatArray:
    """POSE_STAND with both arms at r1mc3.skills' named posture (e.g. "table_ready": the hands above a table)."""
    from r1mc3.skills.postures import posture

    q = J.POSE_STAND.copy()
    q[14:19] = posture(name, "left")
    q[19:24] = posture(name, "right")
    return q


class ServiceRig:
    """The simulator + the v3 core + r1d (+ the planner, the physical hands) in lockstep (module docstring).

    `arms`: the arms' start posture (a r1mc3.skills posture name, or None = the policy's stand pose). `sensors`: the
    sensor model (needed for the hands' raw tau / pad taxels; True = the measured noise config of models/sensors.yaml).
    """

    def __init__(
        self,
        w: world.World,
        *,
        planner: bool = True,
        hands: bool = True,
        arms: str | None = "table_ready",
        seed: int = 0,
        sensors: bool = True,
        trace_upper: bool = False,
        contact: bool = True,
        contact_threshold: float | None = None,
    ) -> None:
        from r1mc3.service.port import LockstepPort
        from r1mc3.service.service import Service

        from .v3_core import V3CoreController

        self.w = w
        cfg = SensorConfig.from_yaml().replace(hand_tau_units="raw") if sensors else SensorConfig.off()
        self.sim = Simulation(w.model, w.index, seed=seed, sensors=SensorModel(cfg, seed=seed))
        q0 = J.POSE_STAND.copy() if arms is None else arms_pose(arms)
        qpos = w.initial_qpos(q0)
        for s in SIDES:
            qpos[w.index.hand_qadr[s]] = dex3.preset("open", s)
        self.sim.reset(qpos)
        self.t_ns = 0
        # the core does not command the hands (r1hands' business): the controller's own open-hand hold is off when the
        # rig's hands are r1d's (it would re-send "open" every writer tick over r1d's grasp)
        self.ctl = V3CoreController(w.model, trace_upper=trace_upper, hands=() if hands else dex3.SIDES)
        self.port = LockstepPort(
            self.ctl.core, self.ctl.mod, lambda: self.t_ns, targets_fn=lambda: self.ctl.last_q.copy()
        )
        # the hands' port always exists (tests read rig.hands.state); `hands=False` keeps it out of r1d and the sim loop
        self.hands = SimHandsPort(self.sim)
        # the pads' contact bits into the core's segment (r1hands' ContactPublisher, the single writer of the region on
        # the robot): the trajectory guards' GUARD_CONTACT and r1d's `arm.lower until contact` work in lockstep too
        self.contact: Any = None
        if hands and contact:
            from r1mc3.hands.contact import ContactPublisher
            from r1mc3.hands.grip import PAD_MIN_KPA

            seg = self.ctl.mod.SegmentView.of_core(self.ctl.core)
            # the threshold: r1hands' default (CONTACT_THRESHOLD: 40 raw since session 12, was 5 - the user's decision
            # on HW2's real pads) is below the pads' noise in the sensor model (absolute Pa, sigma 100 Pa: every pad
            # would read "contact" at rest) - the rig uses the grip's own pad rule (PAD_MIN_KPA above the tare, v2's
            # 1.5 kPa floor) unless told otherwise
            thr = PAD_MIN_KPA * 1000.0 if contact_threshold is None else float(contact_threshold)
            self.contact = ContactPublisher(self.hands, seg.write_hands_contact, threshold=thr)
        self.svc = Service(self.port, hands=self.hands if hands else None, contact=self.contact)
        if planner:
            import r1mc3.plan as plan_mod

            self.svc.planner = plan_mod
        self.script: list[tuple[float, Callable[[], object]]] = []
        self.sim.add_controller(self._r1d, 0.01, "r1d")
        self.sim.add_controller(self.ctl, self.ctl.period_s, "v3")
        if hands:
            self.sim.add_controller(self.hands, 1.0 / HAND_HZ, "hands")

    def _r1d(self, st: LowState) -> LowCmd | None:
        self.t_ns = round(float(st.t) * 1e9)
        while self.script and self.script[0][0] <= st.t + 1e-9:
            self.script.pop(0)[1]()
        self.svc.tick(self.t_ns)
        return None

    @property
    def time(self) -> float:
        return float(self.sim.time)

    def at(self, t: float, fn: Callable[[], object]) -> None:
        self.script.append((t, fn))
        self.script.sort(key=lambda e: e[0])

    def run(self, seconds: float) -> None:
        self.sim.run_for(seconds)

    def call(self, req: Mapping[str, Any], max_sim_s: float = 30.0, step_s: float = 0.02) -> dict[str, Any]:
        """Submit `req` to r1d now (goals with wait: the final result) and step the simulator until it answers; the
        reply as r1d's server would send it (Result.model_dump). TIMEOUT-shaped dict after max_sim_s of sim time."""
        t = self.svc.submit(dict(req))
        t0 = self.time
        while not t.done.is_set() and self.time - t0 < max_sim_s:
            self.sim.run_for(step_s)
        if t.result is None:
            return {"ok": False, "code": "TIMEOUT", "op": str(req.get("op")), "moved": False, "data": {},
                    "details": {"waited_sim_s": round(self.time - t0, 2)}}  # fmt: skip
        out: dict[str, Any] = t.result.model_dump(mode="json")
        return out


# ---------------------------------------------------------------------------------------------- a pick and place
PINCH_MAX_WIDTH_M = (
    0.055  # v2's grip choice (its mock connector: pinch <= 5.5 cm, wrap above; the connector's close_hand)
)
HOVER_CLEAR_M = 0.0  # the hover: the open hand's lowest fingertip at the object's top (the 15-run grid: 4 cm above it
#                      left the descent too short at the far slots; at the top the planner's own descent leg brings it)
HOVER_RETRY_CLEAR_M = 0.01  # session 11: a hover stopped by a guard (the transit met the object: a cylinder placed 5 mm
#                      off its spot - inside D35's tolerance - GUARD_LAG in the first leg) is planned once more this
#                      much higher (a 1 cm hover everywhere cost the right hand its far slot: the deeper descent from
#                      there was refused - a small round object at (20, +10) cm)
TIP_FLOOR_M = 0.004  # the descent never brings the hand's lowest collision point closer than this to the support
GRASP_FRAC = 0.5  # the grasp centre's target height: this fraction of the object's height above its support
DROP_CLEAR_M = 0.02  # the held object's bottom this far above the container's rim before the release (no floor given)
PLACE_CLEAR_M = 0.015  # ... or this far above the container's FLOOR (lowered between its walls: a short drop)


def grip_for(width_m: float) -> str:
    return "wrap" if width_m > PINCH_MAX_WIDTH_M else "pinch"


def deepest_lower(rig: ServiceRig, arm: str, depth_m: float, min_m: float = 0.005, iters: int = 7) -> float:
    """The deepest `arm.lower until depth` (<= depth_m) the planner accepts from where the arm is (dry runs: nothing
    moves), by bisection - a client that wants the hand as low as the world model allows asks the planner instead of
    guessing the hand's collision shape. 0.0 if not even min_m is possible."""

    def ok(d: float) -> bool:
        r = rig.call({"op": "arm.plan", "goal": {"op": "arm.lower", "arm": arm, "until": "depth", "max_depth_m": d}})
        return bool(r.get("ok"))

    if ok(depth_m):
        return depth_m
    lo, hi = 0.0, depth_m
    if not ok(min_m):
        return 0.0
    lo = min_m
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def pick_and_place(
    rig: ServiceRig,
    arm: str,
    obj_xy: tuple[float, float],
    support_z: float,
    height_m: float,
    width_m: float,
    place_xy: tuple[float, float],
    place_rim_z: float,
    *,
    grip: str | None = None,
    place_floor_z: float | None = None,
    lower_until: str = "depth",
    preshape: str | None = None,
    grasp_yaw_deg: float | None = None,
    hover_clear_m: float | None = None,
    aim_grip: bool | None = None,
    squeeze: float | None = None,
    log: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """One pick and place through r1d's public ops only (what a client - a task's connector through an adapter, the
    episode runner, an SDK user - can do): preshape -> arm.hover over the object (the planner's aim for the grip) ->
    arm.lower to the grasp height (the deepest descent the planner accepts down to GRASP_FRAC x height) ->
    hand.grasp (the verdict) -> arm.carry over the place -> hand.release. Returns each step's reply + a summary."""
    g = grip or grip_for(width_m)
    steps: dict[str, Any] = {}

    def step(name: str, req: dict[str, Any], max_s: float = 40.0) -> dict[str, Any]:
        r = rig.call({**req, "wait": True} if req["op"] not in ("arm.plan",) else req, max_sim_s=max_s)
        steps[name] = r
        if log is not None:
            log(name, r)
        return r

    out: dict[str, Any] = {"grip": g, "steps": steps, "ok": False, "failed_at": None}
    if not _approach_and_grasp(rig, arm, obj_xy, support_z, height_m, width_m, g, step, out, lower_until=lower_until,
                               preshape=preshape, grasp_yaw_deg=grasp_yaw_deg, hover_clear_m=hover_clear_m,
                               aim_grip=aim_grip, squeeze=squeeze):  # fmt: skip
        return out
    r = steps["grasp"]
    gc_now = r["data"].get("grasp_centre_m")
    pts = None
    if gc_now is None:
        from r1mc3.service.goals import hand_points

        pts = hand_points(rig.port, rig.hands)
    z_gc = float(gc_now[2]) if gc_now is not None else float(pts[1 if arm == "right" else 0, 4][2])  # type: ignore[index]
    # the object's bottom PLACE_CLEAR_M above the place's floor (inside a container: the hand may be between its walls,
    # the planner keeps it off them); never below the rim when no floor is given
    floor = place_rim_z if place_floor_z is None else place_floor_z
    target = [place_xy[0], place_xy[1], floor + (PLACE_CLEAR_M if place_floor_z is not None else DROP_CLEAR_M) +
              (z_gc - support_z)]  # fmt: skip
    r = step("carry", {"op": "arm.carry", "arm": arm, "position": target, "lift_m": 0.08}, 60.0)
    out["verdict_after_carry"] = (rig.svc.status().get("grips") or {}).get(arm, {}).get("state")
    d = r.get("data") or {}
    sb = d.get("stopped_by") or {}
    out["carry_stop"] = sb.get("code") if isinstance(sb, dict) else sb
    legs, done = list(d.get("legs") or []), int(d.get("legs_done") or 0)
    # the held object meeting the floor in the carry's LAST leg (its lower) stops it on the lag guard: placed (v2's
    # "placed" = held + touched) -> release there
    placed = out["carry_stop"] in ("GUARD_LAG", "GUARD_CONTACT") and legs and done == len(legs) - 1
    if not r["ok"] and not placed:
        out["failed_at"] = "carry"
        return out
    r = step("release", {"op": "hand.release", "side": arm}, 10.0)
    out["ok"] = bool(r["ok"])
    if not r["ok"]:
        out["failed_at"] = "release"
    return out


def _approach_and_grasp(
    rig: ServiceRig,
    arm: str,
    obj_xy: tuple[float, float],
    support_z: float,
    height_m: float,
    width_m: float,
    g: str,
    step: Callable[..., dict[str, Any]],
    out: dict[str, Any],
    *,
    lower_until: str = "depth",
    preshape: str | None = None,
    grasp_yaw_deg: float | None = None,
    hover_clear_m: float | None = None,
    aim_grip: bool | None = None,
    squeeze: float | None = None,
) -> bool:
    """preshape -> arm.hover -> arm.lower (the deepest the planner accepts) -> hand.grasp; False (out["failed_at"]
    set) when a step failed. The first half of every recipe here (pick_and_place, pick_and_lift)."""
    step("preshape", {"op": "hand.preshape", "side": arm, "shape": preshape or g}, 5.0)
    hv: dict[str, Any] = {
        "op": "arm.hover",
        "arm": arm,
        "position": [obj_xy[0], obj_xy[1], support_z],
        "height_m": height_m + (HOVER_CLEAR_M if hover_clear_m is None else hover_clear_m),
    }
    # the grip goes with the hover as the compat adapter sends it: the planner's reference point = where that grip
    # closes (HG-2) for pinch / tip; a wrap without the object's width aims at the default point (planner 8074a95: the
    # fist's fully closed point, 3.6 cm back along the fingers, closed every wrap too deep - grasp_TO_PLANNER_4.md)
    if aim_grip is None or aim_grip:
        hv["grip"] = g
    if grasp_yaw_deg is not None:  # the axis the fingers close along, seen from above (a box: across its narrow side)
        hv["grasp_yaw_deg"] = float((grasp_yaw_deg + 90.0) % 180.0 - 90.0)
    r = step("hover", hv)
    sb = (r.get("data") or {}).get("stopped_by") or {}
    if not r["ok"] and isinstance(sb, dict) and sb.get("code") in ("GUARD_LAG", "GUARD_CONTACT"):
        # the hand met something on the way (the object itself is not in the world model): the same hover from where
        # the arm stopped, HOVER_RETRY_CLEAR_M higher (v2's operator did the same: up a little, again)
        out["hover_retry"] = True
        r = step("hover", {**hv, "height_m": hv["height_m"] + HOVER_RETRY_CLEAR_M})
    if not r["ok"]:
        out["failed_at"] = "hover"
        return False
    pred = r["data"]["predicted"]
    gc = r["data"].get("grasp_centre_m") or pred["grasp_centre_m"]
    ref_z = float(gc[2])
    if g != "wrap" and pred.get("point_m") is not None:
        # session 11: a pinch closes at ITS point (between the thumb and index tips), not at the grasp centre (0.5 thumb
        # + 0.25 (index + middle)): the pinch's curled middle finger lifts the centre 2-3 cm above the tips - the
        # thin block's descent stopped with the tips AT its 3.1 cm top (they pressed its top edges: NOT_HOLDING, gap 20
        # of 24 mm). The planner's predicted pinch point, moved by r1d's measured hover error (the centre's).
        ref_z = float(pred["point_m"][2]) + (float(gc[2]) - float(pred["grasp_centre_m"][2]))
    out["closing_ref"] = "pinch_point" if ref_z != float(gc[2]) else "grasp_centre"
    want = ref_z - (support_z + GRASP_FRAC * height_m)
    depth = deepest_lower(rig, arm, max(0.005, want)) if want > 0.005 else 0.0
    out["lower_wanted_m"], out["lower_m"] = round(want, 4), round(depth, 4)
    if depth > 0.0:  # down to the grasp height (until "contact": or the first pad contact - `lower_until`)
        until = lower_until if (lower_until == "depth" or rig.contact is not None) else "depth"
        r = step("lower", {"op": "arm.lower", "arm": arm, "until": until, "max_depth_m": round(depth, 4)}, 30.0)
        sb = (r.get("data") or {}).get("stopped_by") or {}
        out["lower_stop"] = sb.get("code") if isinstance(sb, dict) else sb
        # the object itself is not in the world model: a descent the hand's fingers / palm meet it on is stopped by the
        # executor's lag guard (GUARD_LAG) - v2's "touched": the hand rests on / against the object, grasp from there
        if not r["ok"] and out["lower_stop"] not in ("GUARD_LAG", "GUARD_CONTACT"):
            out["failed_at"] = "lower"
            return False
    gq: dict[str, Any] = {"op": "hand.grasp", "side": arm, "grip": g, "object_class": "rigid", "width_m": width_m}
    if squeeze is not None:
        gq["squeeze"] = float(squeeze)
    r = step("grasp", gq, 15.0)
    v = (r.get("data") or {}).get("verdict") or {}
    out["verdict_at_grasp"] = v.get("state")
    if not r["ok"]:
        out["failed_at"] = "grasp"
        return False
    return True


LIFT_M = 0.08  # pick_and_lift: the scripted lift (the carry's 8 cm)
HOLD_S = 2.0  # ... held this long before the verdict is read (HW2 B4: a hold / slip / miss per grasp)


def pick_and_lift(
    rig: ServiceRig,
    arm: str,
    obj_xy: tuple[float, float],
    support_z: float,
    height_m: float,
    width_m: float,
    *,
    grip: str | None = None,
    preshape: str | None = None,
    grasp_yaw_deg: float | None = None,
    hover_clear_m: float | None = None,
    squeeze: float | None = None,
    lift_m: float = LIFT_M,
    hold_s: float = HOLD_S,
    log: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """A scripted grasp WITHOUT a place (session 11; the shape of HW2's 10 scripted grasps per object, B4, and the grid
    for the hand that cannot reach the container): preshape -> hover -> lower -> grasp -> arm.move_relative up `lift_m`
    -> hold `hold_s` -> the verdict (r1d's grip status) -> down again (1 cm short of where it was) -> hand.release ->
    up 6 cm. Public r1d ops only."""
    g = grip or grip_for(width_m)
    steps: dict[str, Any] = {}

    def step(name: str, req: dict[str, Any], max_s: float = 40.0) -> dict[str, Any]:
        r = rig.call({**req, "wait": True} if req["op"] not in ("arm.plan",) else req, max_sim_s=max_s)
        steps[name] = r
        if log is not None:
            log(name, r)
        return r

    out: dict[str, Any] = {"grip": g, "steps": steps, "ok": False, "failed_at": None}
    if not _approach_and_grasp(rig, arm, obj_xy, support_z, height_m, width_m, g, step, out, preshape=preshape,
                               grasp_yaw_deg=grasp_yaw_deg, hover_clear_m=hover_clear_m, squeeze=squeeze):  # fmt: skip
        return out
    r = step("lift", {"op": "arm.move_relative", "arm": arm, "delta": [0.0, 0.0, lift_m]}, 20.0)
    if not r["ok"]:
        out["failed_at"] = "lift"
        return out
    rig.run(hold_s)
    out["verdict_after_hold"] = (rig.svc.status().get("grips") or {}).get(arm, {}).get("state")
    if log is not None:  # the end of the hold (an observer reads the object's height here, before the set-down)
        log("hold", {"ok": True, "data": {"verdict": out["verdict_after_hold"]}})
    step("down", {"op": "arm.move_relative", "arm": arm, "delta": [0.0, 0.0, -(lift_m - 0.01)]}, 20.0)
    r = step("release", {"op": "hand.release", "side": arm}, 10.0)
    step("retreat", {"op": "arm.move_relative", "arm": arm, "delta": [0.0, 0.0, 0.06]}, 20.0)
    out["ok"] = bool(r["ok"])
    if not r["ok"]:
        out["failed_at"] = "release"
    return out
