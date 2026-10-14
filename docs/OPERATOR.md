# The operator's command line: `r1`

`r1` is a client of `r1d` ([`API.md`](API.md)): on the backpack `127.0.0.1:8780`, from a laptop through an SSH tunnel.
Run it as `r1 <words>` once installed (or `python -m r1mc3.cli <words>`). Every word sends ONE request and prints ONE
JSON result: `ok`, `code`, `moved` (whether anything was commanded to move - a refusal is always `false`), `data`,
`details` (`--pretty` indents it). Exit code 0 = ok, 1 = refused, 2 = `r1d` not reachable. Requests from the command
line carry the requester `operator` (a person typed them): some words are accepted only from the operator.

**Before any word that moves something: a person holds the robot (or it hangs on its stand), and knows how to stop
it.** Every motion ends by itself when its time-to-live runs out (the core decelerates the joint to HOLD within
0.15 s), and `r1 stop` stops all of them at once. Nothing in `r1d` ever damps; `r1 damp` is the one limp word.
Read [`SAFETY.md`](SAFETY.md) first.

## Modes and stops

| Word | What moves | Notes |
|---|---|---|
| `r1 status` | nothing | the core's mode, faults, heartbeat age, real-time grant; the guard (ARMED / OBSERVE-ONLY / ACTIVE / ABSENT); who owns the motors; the hands; the battery; `data.summary` says it in words |
| `r1 takeover` | the legs: Unitree's controller releases, ours mirrors its last command and blends (1 s) into the balance policy | the robot standing in Unitree's own balance mode, a person holding it. Refused under the stand-session lock, without a live guard, or when a gate fails (nothing is sent then) |
| `r1 on-stand` | the whole body (robot on its STAND, feet off the floor): the same release + mirror, then a blend into a stiff hold of the measured pose | then `joint` works for the legs and waist too (one group at a time) |
| `r1 stop` | everything stops where it is: arms / head HOLD, the base command zero, balance continues | any client; latched until `r1 resume`; commands issued before it are refused afterwards |
| `r1 halt` | the trajectory clock halts, motion decelerates to rest | `r1 resume` |
| `r1 resume` | clears a stop (nothing moves by itself) | |
| `r1 freeze` | the legs ramp (0.4 s) to the stand pose at stand gains and hold | nothing leaves it automatically |
| `r1 balance` | from a freeze of a BALANCING robot: the policy takes the legs again; the arms keep holding | operator only; then `r1 resume` |
| `r1 damp` | **LIMP**: kp 0 + damping on every joint | operator only; hold the robot first |
| `r1 reclaim` | nothing should move: `r1rt` takes the motors back from `r1guard` (a 1 s blend from exactly what the guard sent) | after a guard takeover only; refused with a plain reason otherwise |
| `r1 stand-freeze [--yes]` / `r1 stand-hold` | on the stand: the legs + waist ramp to the stand pose (0.4 s) and hold / blend back into the ON_STAND hold | without `--yes`: the plan and its checks only |
| `r1 shutdown [--label L]` | nothing | only in MONITOR or DAMP: the session's logs into a new folder with checksums |

**The session end, always:** hold the robot -> `r1 damp` -> support it standing -> `r1 shutdown` -> power off.

## The stand tests (the robot hanging on its stand)

| Word | What happens | Notes |
|---|---|---|
| `r1 board-hold --pause-ms 20\|100\|300\|400\|500 [--yes]` | **nothing is commanded for N ms**: `r1rt` is stopped (SIGSTOP), the motor board holds or lets go - the test measures which | needs the guard in observe-only mode for the test (otherwise it takes over at ~10 ms): disabling a safety function, so only with the robot on its stand and the guard re-armed and checked straight after. On our robot the board held ~0.4 s, then damped: longer pauses are refused |
| `r1 guard-takeover [--stop-s 3] [--yes]` | nothing should move: `r1rt` is stopped S s (0.5-8) in ON_STAND, `r1guard` must take over and hold the same pose; after SIGCONT `r1rt` stays silent | ON_STAND with the guard ARMED. If the guard has not taken the motors within 0.2 s, `r1d` continues `r1rt` at once. Then `r1 reclaim` |

Without `--yes` both print the plan and their checks only.

## Joints, hands, recordings

| Word | What moves | How it stops |
|---|---|---|
| `r1 joint GROUP Q1,Q2,... [--ttl 0.2] [--dq ..] [--kp ..] [--kd ..] [--tau-ff ..]` | one group (`arm_left` / `arm_right` / `head`; `leg_left` / `leg_right` / `waist` only in ON_STAND) to the setpoint, at most 2 rad/s | after `--ttl` (<= 0.3 s) the group decelerates to HOLD; outside the profile's limits: refused |
| `r1 hold GROUP [Q...]` | HOLD at Q (or where it is) | |
| `r1 jog GROUP DQ1,DQ2,... [--ttl 0.3]` | the group at these joint rates for `--ttl` s (repeat to keep going) | by itself; clamped at the limits |
| `r1 head PITCH,YAW [--ttl 0.3]` | the head | as `joint` |
| `r1 hand SIDE Q1..Q7 [--ttl 1] [--kp ..] [--kd ..] [--tau ..]` | one hand's 7 motors (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1) | after `--ttl` the hand holds where it is; if `r1d` dies the hand goes passive ~1 s later (a held object drops) |
| `r1 hand-state [SIDE]` | nothing | joints, temperatures, the 9 x 12 pad readings, the state's age; `invalid_reason` `stale` / `frozen` / `no state` |
| `r1 record list` / `r1 record plan KIND k=v ...` / `r1 record run KIND k=v ... --yes` | what the plan says (ON_STAND for body recordings) | without `--yes`: a dry run; `r1 stop` aborts (the joint decelerates to HOLD) |

Numbers in a word are comma lists without spaces (`r1 joint leg_left -0.25,0.01,0,0.5,-0.2,0`).

Recording kinds (the robot on its stand; small defaults, `r1 record plan` shows them):

| Kind | Parameters (defaults) | Moves |
|---|---|---|
| `step` | joint, amplitude_rad 0.05, hold_s 2, repeats 2 | one joint: base, +A, base, -A, base, ... |
| `ramp` | joint, amplitude_rad 0.2, rate_rad_s 0.02, hold_s 1 | one joint: a slow triangle out and back (friction) |
| `chirp` | joint, amplitude_rad 0.03, f0_hz 0.5, f1_hz 5, duration_s 20 | one joint: a sine sweep with smooth fades |
| `static_pose` | group, q, move_s 3, hold_s 10, payload_kg 0 | one group to q, hold |
| `imu_static` | duration_s 30 | nothing |
| `pad_press` | side, pad 0, weights_g [20, 50, 100, 200], press_s 5 | nothing (the hand holds open; you press) |
| `finger_hold` | side, finger, torque_nm 0.9, duration_s 60 | one finger closes with a constant torque |

Every run writes a manifest (the plan, the status, start / end in the logger's clock and in UTC, the log files of the
window).

## Joint directions - what a POSITIVE value does

Verified on our robot, joint by joint, against the model ([`hardware/JOINT_DIRECTIONS.md`](hardware/JOINT_DIRECTIONS.md)
has the full table and how to re-check). **Never guess a direction from a joint's name.**

| Joint | Positive = | Joint | Positive = |
|---|---|---|---|
| hip pitch, shoulder pitch | the limb swings BACKWARD | knee | BENDS more |
| hip roll, shoulder roll | toward the robot's LEFT, both sides (left limb out, right limb in) | ankle pitch | toes DOWN |
| hip yaw | toes toward the robot's LEFT (left foot out, right foot in) | ankle roll | the foot's LEFT side up |
| waist roll (torso held on the stand) | the legs swing to the robot's RIGHT | waist yaw (on the stand) | the pelvis turns RIGHT |
| shoulder yaw | the hand turns toward the robot's LEFT | wrist roll | the front of the hand turns toward the robot's RIGHT |
| **elbow** | **a LARGER value = STRAIGHTER; a SMALLER value = MORE FLEXED** - 0 is not straight (about 90 deg flexed); 1.40 is nearly hanging | head pitch / yaw | nod DOWN / turn LEFT |

Re-check after any firmware, model or joint-table change: `python -m r1mc3.hw1.dircheck` (prints the expected motion
per joint; `--plan` prints the step words and sends nothing).

## Goals: arms, hands, head, body

Each goal word waits for the goal's end and prints its result (`--no-wait`: the goal id at once; then `r1 goal wait
ID` / `r1 goal cancel ID` / `r1 goals`). `r1 stop` ends every goal (the arms HOLD). Positions are metres in the base
frame (between the feet, x forward, y LEFT, z up). Look at `r1 plan` first: expected motions are the planner's.

| Word | What moves |
|---|---|
| `r1 move-joint left 0.22,0.19,0.04,0.76,0 [--speed 0.5]` | the left arm to these 5 joints (shoulder pitch, roll, yaw, elbow, wrist roll) |
| `r1 reach left 0.35,0.15,0.85 [--prefer front]` | the hand's grasp centre to the point (planned against the world) |
| `r1 hover left 0.35,0.15,0.76 --height 0.05 [--grip wrap] [--grasp-yaw 90] [--approach front\|inward\|around\|low] [--target ID \| --grasp-z Z] [--approach-tol 10]` | up, across, down to 5 cm above the point, at the requested hand orientation; the reply's `plan_details.orientation` says what was delivered |
| `r1 lower left --until contact` / `--until depth --depth 0.03 [--target ID]` / `--stop-z 0.78` | a guarded descent |
| `r1 carry left 0.40,0.10,0.80 --lift 0.08` | up, across, down |
| `r1 move-path left '[[x,y,z], [x,y,z]]' [--linear] [--blend 0.02]` | the whole path planned first; with `--blend` the hand passes interior waypoints without stopping |
| `r1 guarded-move left --delta dx,dy,dz \| --position x,y,z \| --stop-z z [--until contact,blocked] [--touch id,...]` | a straight line that succeeds on a named guard |
| `r1 slide left dx,dy,0 [--support id]` | along the declared support under the hand |
| `r1 pick left x,y,z --height 0.05 [--grasp-z z] [--grip wrap] [--class rigid] [--width w] [--object-id id]` | hover -> down -> grasp -> lift |
| `r1 place left x,y,z [--lift 0.08] [--retreat 0.05]` | carry -> release -> retreat |
| `r1 par '[<goal>, <goal>]'` / `r1 seq '[<request>, ...]'` | goals together on different limbs / requests in order |
| `r1 plan '{"op": "arm.reach", "arm": "left", "position": [0.35, 0.15, 0.85]}'` | nothing (the dry run) |
| `r1 arm-stop [left]` | the arm HOLDs where it is |
| `r1 grasp right --grip pinch --class rigid --width 0.03` / `r1 release right` / `r1 preshape left pinch` / `r1 tare left` | the fingers (tare: nothing moves) |
| `r1 calibrate-pinch left` | the hand EMPTY in free air: a tip pinch on nothing, then open (needed once per hand before a `thin_flexible` grasp) |
| `r1 look-at 1.0,0.2,0.8` | the head (pitch + = nod down, yaw + = look left) |
| `r1 lean 3` / `r1 lean 0` | the body leans forward 3 deg (the policy's own lean, 0-5 deg) / upright. After any stop it answers `STOPPED` until `r1 resume` |
| `r1 skill table_ready [--arms left]` / `r1 skill ready` | the arms to a named posture, one arm at a time |
| `r1 twist left vx,vy,vz [--ttl 0.3]` | Cartesian velocity (refused while a world is declared) |
| `r1 lease acquire arm_left,hand_left [--client me] [--ttl 30]` / `renew ID` / `release ID` / `list` | leases |

## The world, the camera, measurement (nothing moves)

| Word | What |
|---|---|
| `r1 world declare '{"kind": "surface", "id": "table", "pose": {"pos": [0.55, 0, 0.78]}, "size": [0.6, 1.2, 0.04]}'` / `r1 world list` / `r1 world clear [ID]` | what the planner avoids (and what a goal may touch with `--target`) |
| `r1 observe [--no-depth]` | a head frame: its `obs_id`, the camera pose at capture, the calibration |
| `r1 point OBS_ID PX PY` | the pixel -> a base point on the declared surfaces |
| `r1 extent OBS_ID PX PY` | what stands at the pixel (size / shape from the depth) |
| `r1 measure setup plan.json [--frames 3]` | the head measures the plan's tag-marked entities (hold the head still first) |
| `r1 measure phone plan.json --image still.jpg --intrinsics phone.json` | a phone still: its pose from the table's tags, the entities on the table |

`plan.json` = `{entities: [{id, tags: [{id, size_m, T_entity_tag}]}], table_tags: [...], points: [...]}`.

## What `r1 status` says when something is or was wrong

| Line | Means | What to do |
|---|---|---|
| `r1rt: STATE_OLD ...` | the robot state the core balances on is older than its limit right now (the network, DDS, the receive thread) | watch it; with the state-age rung off (the default) the policy keeps balancing - stay ready to hold the robot |
| `r1rt: N STATE_OLD episode(s) since its start ...` | how often it happened | note it in the session log |
| the guard ACTIVE, `r1rt` SILENT | the guard holds the motors | hold the robot; find why the core stopped (the flight recorder, the guard's forensics line); `r1 reclaim` when it is safe |
| `stop.latched` | a stop is posted (any client's, or the core's own) | `r1 resume` when you want motion again |
| battery words (`~N min to 30 %`) | the battery's drain | plan the swap; nothing follows by itself |
