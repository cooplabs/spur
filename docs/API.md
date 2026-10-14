# The command API (`r1d`)

`r1d` is the service every client talks to: the `r1` command line ([`OPERATOR.md`](OPERATOR.md)), the Python SDK
(`r1mc3.client`), your application. The design behind it: [`DESIGN.md`](DESIGN.md) §6.

## Transport

Newline-delimited JSON over TCP on the backpack's `127.0.0.1:8780` (from a laptop: through an SSH tunnel, e.g.
`ssh -L 8780:127.0.0.1:8780 <user>@<backpack>`). One request object per line -> one result object per line, in order.
Start it: `python -m r1mc3.service --shm-name NAME --domain N [--port 8780]` (on the robot the systemd unit does it).

- **Requests**: `{"op": "...", "id": <your id>, ...fields}`. The models are `r1mc3.service.schemas`
  (`request_schema()` = the JSON Schema of every op). Every request may carry `requester` (`"operator"` for a person at
  the command line) and `lease` (below).
- **Results**: `{"ok", "code", "op", "id", "moved", "data", "details"}`. `moved` says whether anything was commanded
  to move - a refusal is always `moved: false`. `code` is one of `r1mc3.service.codes.Code`: `OK`, `INVALID_ARGS`,
  `UNKNOWN_COMMAND`, `WRONG_MODE`, `NOT_OWNER`, `RESOURCE_BUSY`, `LIMIT`, `CONTROLLER_REJECTED`, `FAULT`, `TIMEOUT`,
  `HAND_FAULT`, `NOT_PERMITTED`, `UNSUPPORTED`, `RELEASE_FAILED`, `UNKNOWN_FRAME`, `STALE_FRAME`, `UNKNOWN_GOAL`,
  `UNREACHABLE`, `COLLISION`, `GUARD_STOP`, `SKILL_FAILED`, `STEP_FAILED`, `NOT_HOLDING`, `CANCELED`, `PREEMPTED`,
  `STOPPED`, `NOT_CALIBRATED`, `NO_DEPTH`, `OFF_SURFACE`, `CALIBRATION_FAILED`, ... - with the reason in `details`.
- **Frame**: `base` - on the floor between the feet, x forward, y LEFT, z up (metres, radians unless a field says
  `_deg`).
- **Groups**: `arm_left`, `arm_right`, `head` (in BALANCE or ON_STAND); `leg_left`, `leg_right`, `waist` (ON_STAND
  only); the hands `hand_left`, `hand_right`.

## System

| op | fields | what |
|---|---|---|
| `status` | - | the core's mode, faults, heartbeat age, rate, real-time grant; the guard (armed / active / absent); the token owner; the hands' state age; goals, grips, the body (tilt, lean), the arms' hand points; `state_age`, `stop` (latched?), `camera`, `battery`, `hands_heat`; `data.summary` says it in words |
| `stop` / `halt` | requester | always accepted, from any client: `stop` = every group HOLDs, the base command goes to zero, balance continues, latched until `resume`; `halt` = the trajectory clock halts (resumable). Every goal ends `aborted` with `OPERATOR_STOP` |
| `resume` | - | clears a stop (nothing moves by itself) |
| `damp` | requester = `"operator"` | **limp**: kp 0 + damping on every joint. Refused for any other requester. Reaches the motors through the guard too |
| `takeover` | release_timeout_s (4), reply_timeout_s (1) | the core's gates -> Unitree's controller releases (mode-switch RPC) -> MIRROR -> BLEND -> BALANCE. Refused under the stand-session lock and without a live guard |
| `on_stand` | as takeover | the same release, into a stiff hold of the measured pose (the robot hanging on its stand) |
| `freeze` | - | STAND_FREEZE: the legs ramp (0.4 s) to the stand pose at stand gains and hold |
| `balance` | requester = `"operator"` | from a STAND_FREEZE that came from BALANCE back into BALANCE (after the core's checks); the freeze's stop stays posted until `resume` |
| `stand_freeze` / `stand_hold` | - | on the stand: ramp the legs + waist to the stand pose / blend back into the ON_STAND hold |
| `reclaim` | requester = `"operator"` | after a guard takeover: `r1rt` takes the motors back (a blend from exactly what the guard sent); the guard re-arms |
| `shutdown` | label? | only in MONITOR or DAMP: the logs (r1log, recordings, r1d's request journal) saved into a new folder with checksums |

The two stand tests (the robot hanging on its stand; the operator's words `r1 board-hold` and `r1 guard-takeover`)
are described in [`OPERATOR.md`](OPERATOR.md).

## Goals

A goal op answers at once with `data.goal_id` and `data.state = "accepted"` (a refusal: `ok` false, `moved` false,
`data.state = "rejected"`); with `"wait": true` the reply is the goal's final result instead. Every goal has
`timeout_s` (default 30) and `preempt` (default false). States: `accepted -> active -> succeeded | aborted | canceled
| preempted | timed_out`. A goal holds its resources until it ends; a second goal on a held resource is
`RESOURCE_BUSY` (with `held_by`) unless it asks to `preempt` (the old goal brakes to HOLD and ends `preempted`).

| op | fields | what |
|---|---|---|
| `goal.status` / `goal.wait` / `goal.cancel` / `goal.list` | goal_id (+ timeout_s) | the goal table; cancel = canceled + HOLD |

**Results of an arm goal**, measured from the core's state: `moved`, `hand_start`, `hand_now`, `target`, `achieved`
(error <= 20 mm), `error_mm`, `fingertips_m` {thumb_tip, middle_tip, index_tip}, `grasp_centre_m`, `stopped_by`
{code, detail}, `duration_s`, `planned_on`, `plan_id`, `target_q`, `achieved_q`, the legs, the predicted points,
`clearance_m`, `planning_s`. `stopped_by.code` names what ended it: an executor guard (`GUARD_LAG`, `GUARD_TORQUE`,
`GUARD_CONTACT`, `DEADMAN`, `STARVED`, `RESUME_AFTER_SILENCE`), `OPERATOR_STOP`, `CANCELED` / `PREEMPTED` /
`TIMED_OUT`, `REFUSED_BY_CORE` (+ the reason), or `GUARD_TAKEOVER` (the guard holds the motors).

## L0 / L1: joints and streams

| op | fields | what |
|---|---|---|
| `joint.command` | group, q, dq?, kp?, kd?, tau_ff?, ttl (<= 0.3 s) | one setpoint; after the ttl the group decelerates to HOLD. Clamped to the profile's limits; at most 2 rad/s |
| `joint.hold` | group, q? | HOLD at q (or where it is) |
| `stream.joint_jog` | group, dq, ttl (<= 0.5 s) | joint rates, integrated at r1d's 100 Hz tick |
| `stream.head` | q (pitch, yaw) or a rate, ttl | the head (pitch + = nod down, yaw + = look left) |
| `stream.arm_twist` | arm, v [vx, vy, vz] (<= 0.10 m/s), ttl (<= 0.5 s) | resolved-rate Cartesian velocity (damped least squares on the arm model); **refused while the world has declared entities** (no collision look-ahead yet) |
| `hand.command` / `hand.state` | side, q (7), dq?, kp?, kd?, tau?, ttl (<= 5 s) | one Dex3-1 hand's 7 motors (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1); the state: joints, temperatures, the 9 x 12 pad readings, the state's age and validity |

Joint directions (what a positive value does) were verified on our robot:
[`hardware/JOINT_DIRECTIONS.md`](hardware/JOINT_DIRECTIONS.md). Never guess a direction from a name.

## L2: arm, hand and head goals

Every arm goal is planned against the world model as a whole before anything moves (`COLLISION` / `UNREACHABLE`
refusals move nothing).

| op | main fields | what |
|---|---|---|
| `arm.move_joint` | arm, q (5), speed, duration_s? | to these 5 joints (shoulder pitch, roll, yaw, elbow, wrist roll) |
| `arm.reach` | arm, position, approach?, prefer? | the hand's grasp centre to the point |
| `arm.move_linear` / `arm.move_relative` | arm, position / delta | a straight line of the hand point |
| `arm.move_path` | arm, waypoints (1-16), linear?, blend_m? (<= 0.05) | the whole path checked first; with `blend_m` the corners are cut and the hand does not stop at interior waypoints |
| `arm.hover` | arm, position, height_m, grip?, prefer?, grasp_yaw_deg?, approach?, approach_style? (`front` / `inward` / `around` / `low`), approach_tol_deg?, target_id?, grasp_z_m? | lift + transit + descend to `height_m` above the point, at a hand orientation ranked by the request: `grasp_yaw_deg` = the line the fingers close along (seen from above), `around` = the palm turned to a tall object, `low` = the fingertips kept over a flat object's bottom; with `target_id` (a declared object) the pose is chosen where the grip closes, the open hand clear of the object. `approach_tol_deg` refuses (nothing moves) when the style cannot be met within it, saying why |
| `arm.lower` | arm, until (`contact` / `surface` / `depth`), max_depth_m, stop_z_m?, target_id? | a guarded descent; after a styled hover, `until: depth` retraces the hover's own line to the grasp pose |
| `arm.guarded_move` | arm, one of position / delta / stop_z_m, until (["contact"], "blocked"), touch? | a straight line that SUCCEEDS when a named guard ends it |
| `arm.slide` | arm, delta (x, y), support?, max_gap_m | a straight line along the declared support under the lowest fingertip |
| `arm.carry` | arm, position, lift_m | rise + travel + lower, planned end to end |
| `arm.plan` | goal (any arm goal request) | the dry run: legs, durations, predicted points, clearance; nothing moves |
| `arm.stop` | arm? | the arm's goal canceled: HOLD |
| `hand.grasp` | side, grip (`wrap` / `pinch` / `tip`), object_class (`rigid` / `thin_flexible` / `soft`), width_m?, squeeze? | the grip state machine: preshape -> close -> per-finger contact stop -> squeeze -> verdict. `succeeded` with HOLDING / TENTATIVE (the grip keeps squeezing until `hand.release`); `aborted` with `NOT_HOLDING` and the reason. The reply carries the evidence and `hand_heat` |
| `hand.release` / `hand.preshape` | side (+ shape) | the hand to the pose |
| `hand.tare` | side | the pads re-tared over 0.5 s (the hand open, touching nothing) |
| `hand.calibrate` | side, kind (`air_pinch`) | a tip pinch on nothing (the hand empty, in free air): the reference a `thin_flexible` grasp needs |
| `head.look_at` | position | the head looks at the point (clipped to its limits) |
| `body.pose` | lean_deg (0-5) | the balance policy's forward lean (as trained), renewed by r1d (TTL 0.5 s: r1d silent -> upright); BALANCE only; refused `STOPPED` while a stop is latched |

## L3: skills and composites

| op | fields | what |
|---|---|---|
| `skill.ready` / `skill.table_ready` | arms (both / left / right), speed | named postures, one arm at a time; `table_ready` also from the hands under a declared table (out and around, fingers closed) |
| `skill.pick` | arm, position (the object's centre), height_m, grasp_z_m?, grip, object_class, width_m?, squeeze?, hover_m, lift_m, object_id? | hover -> straight down -> `hand.grasp` (the verdict) -> lift; any failed step = `SKILL_FAILED` |
| `skill.place` | arm, position, lift_m, retreat_m | carry -> release -> retreat |
| `seq.run` | steps (<= 64), stop_on_failure | the steps in order; the first failure -> `STEP_FAILED` + its index |
| `par.run` | steps (1-8 goals on different resources) | together; the first failure cancels the others |

## The world model

| op | fields | what |
|---|---|---|
| `world.declare` | entity {kind: `surface` / `container` / `obstacle` / `object`, id, pose {pos, yaw? or quat?}, size / size_outer, wall_m?, floor_m?, shape?} | adds or replaces the entity; the planner and the grip verdict read it |
| `world.list` / `world.clear` | entity_id? | |

A `surface`'s pose is the centre of its TOP face. A `container` has walls and a floor (a hand may be inside above the
floor). An `object` (box / cylinder / sphere) may be named as a goal's `target_id` / `object_id`: the hand may touch it.

## Leases

`lease.acquire {resources, client, ttl_s (<= 300)}` / `lease.renew` / `lease.release` / `lease.list`. Off until a
client takes a lease: while one is live, a request that moves its resources must carry `"lease": "<id>"`, else
`NOT_OWNER` (with `held_by`). `stop` / `halt` / `damp` / `arm.stop` / `status` / read-only ops / `arm.plan` are never
gated.

## Camera and measurement

| op | fields | what |
|---|---|---|
| `camera.observe` | depth (true), right (false), max_age_s, jpeg | the newest head frame -> `obs_id`, capture time, the camera pose at capture (base frame), the calibration in use, the image |
| `camera.point` | obs_id, px, py, patch, use_depth | the pixel -> a base point on the declared surfaces (the ray through the calibrated lens; depth adds the height); says which surface it hit; `NO_DEPTH` in words when there is none |
| `camera.object_extent` | obs_id, px, py | what stands at the pixel: centre, length / width / height, yaw, shape, confidence |
| `camera.project` / `camera.depth_image` | obs_id, points / - | base points -> pixels; the depth image |
| `measure.setup` | entities [{id, tags [{id, size_m, T_entity_tag}]}], frames | the head camera measures AprilTag-marked entities -> their poses in the base frame, or the reason it refused |
| `measure.phone` | image_b64, intrinsics, table_tags, entities, points | a phone still: its pose from tags on the table plane, the entities on it, points on the plane |

## Recordings (identification on the stand)

`record.list` / `record.plan {kind, params}` / `record.run {kind, params, dry_run (true)}`: steps, slow ramps, chirps,
static poses, IMU statics, pad presses, finger holds - each planned and checked first (limits, speed, what moves); a
manifest per run. See [`OPERATOR.md`](OPERATOR.md) and [`IDENTIFICATION.md`](IDENTIFICATION.md).

## The Python SDK

```python
from r1mc3.client import R1

with R1() as r:                                        # 127.0.0.1:8780
    r.world_declare({"kind": "surface", "id": "table", "pose": {"pos": [0.55, 0, 0.78]}, "size": [0.6, 1.2, 0.04]})
    res = r.reach("left", [0.35, 0.15, 0.85])          # waits for the goal's end (wait=False: the goal id at once)
    if not res.ok:
        print(res.code, res.details)
    r.grasp("left", grip="pinch", object_class="rigid", width_m=0.03).raise_for_error()
```

One typed method per op; every method returns a `Reply` (`ok`, `code`, `op`, `id`, `moved`, `data`, `details`);
`raise_for_error()` raises `R1Refused`. There is no `damp` method: the one limp state is the operator's word.
