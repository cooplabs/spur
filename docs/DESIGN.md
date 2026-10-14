# Spur (r1mc3) design

How the controller and the simulator are built, and why. r1mc3 controls a Unitree R1 humanoid (26 body joints: legs,
waist, arms, head) with two Unitree Dex3-1 hands (7 motors each), from the R1's backpack computer (a Jetson Orin NX).
This document describes release 0.1.0. Where something is designed but not built or not proven on a robot, it says so.

Contents: 1 goals and the lesson behind them - 2 processes - 3 the real-time core - 4 the guard - 5 safety - 6 the
command API - 7 hands - 8 world model and planner - 9 perception and measurement - 10 telemetry - 11 deployment -
12 the simulator - 13 validation - 14 decisions - 15 risks and what is not solid.

## 1. Goals and the lesson behind them

1. **A general-purpose controller for the R1 + Dex3-1**: anyone with the robot can install it, run it in simulation and
   use it for their own tasks through one command API. Task knowledge (tables, objects, containers) enters only
   through that API.
2. **The robot must never be dropped by the software.** The previous controller ran the balance policy, the logging
   and a network health read on ONE thread. During a Wi-Fi reconnect, a network-driver read blocked that thread for
   about 4 seconds; the motor board kept the last command, the policy stopped, and the robot fell
   ([`case_studies/`](case_studies/)). Nothing that can block may share the loop that balances the robot, and a stall
   of that loop must be caught by something independent.
3. **Never limp.** No automatic path may ever damp the robot. Field practice is often "damp on any fault and accept
   the fall"; r1mc3 does not do that. Damping is the operator's action only.
4. **A simulator first, identified on the real robot**, running the same controller code, so that every fault path
   is tested before it is met on hardware.

Requirements that follow (each is answered below):

| Requirement | Answered in |
|---|---|
| Nothing that can block shares the loop that balances the robot | §2, §3.1 |
| A stall or crash of anything except the balance core leaves the robot balancing; a stall of the core is caught by an independent process; never limp | §2, §4, §5.3 |
| Stops never depend on the thread they stop | §3.3, §5 |
| Full-rate, timestamped telemetry, off the control path | §10 |
| One world model and one geometry implementation for every planner and check | §8 |
| The whole multi-step goal is planned before anything moves; a refusal moves nothing | §8.2 |
| "Holding" decided from independent evidence, never force alone | §7.2 |
| Typed error codes and result schemas | §6 |
| One command surface with leases | §6 |
| Safety thresholds as bounded, versioned configuration | §3.6 |
| Simulator physics independent of the controller, deterministic, with the robot's interfaces | §12 |

## 2. Processes

```
 a laptop (operator)                                     backpack: Jetson Orin NX (8 cores)
 ┌──────────────────────────┐   ssh tunnel   ┌───────────────────────────────────────────────────────────────┐
 │ `r1` CLI · Python SDK     │◄──────────────►│ r1d  (Python, CPU 0-4)  JSON over TCP :8780                    │
 │ your application          │      :8780     │   goals · leases · world model · planner workers (processes)   │
 │ simulator + viewer        │                │   skills · perception · measurement                            │
 └──────────────────────────┘                │        │ shared memory: command rings, stop channel, mailbox   │
                                              │        ▼                                                       │
                                              │ r1rt (C++17, SCHED_FIFO, isolated CPU) ── rt/lowcmd 500 Hz ────┼──► R1 body
                                              │   writer 500 Hz: state -> checks -> policy action / executor   │◄── rt/lowstate ~1 kHz
                                              │   -> limits -> LowCmd + CRC -> write                           │
                                              │   policy thread 50 Hz (fixed-size MLP)                         │
                                              │   heartbeat + mirror + ownership token ◄──┐                    │
                                              │ r1guard (C++, its own CPU) hot standby: own state reader,      │
                                              │   warm shadow policy; takes the token on a stale heartbeat     │
                                              │ r1hands (Python 100 Hz) ── rt/dex3/*/cmd ─► dex3_server ─USB─► hands
                                              │ r1cam (Python) head camera: images + depth on demand           │
                                              │ r1log (Python, nice) shared-memory rings -> MCAP, flight recorder│
                                              └───────────────────────────────────────────────────────────────┘
```

Principles:

1. **Only `r1rt` (or, after a takeover, `r1guard`) writes `rt/lowcmd`.** One writer, enforced by an ownership token
   (a compare-and-swap word) in shared memory; each writer checks it immediately before every write.
2. **Every component is a library with a `step()` first and a process second.** The process wrapper only schedules
   it. The simulator therefore runs the same code in lockstep (deterministic, faster than real time) or as the real
   processes over DDS (timing-faithful).
3. **Shared memory, not sockets, between our own processes**, lock-free (seqlocks, single-producer single-consumer
   rings). The real-time core never waits on anyone.
4. **Python never sits in the balance path.** A Python stall can only stop arm motion (smoothly) or delay a reply.
5. **Truthful results**: every goal reports what was measured, from the core's state.

| Process | Language | CPU / priority | Rate | Does | If it stalls or dies |
|---|---|---|---|---|---|
| `r1rt` | C++17 | writer FIFO 90 alone on CPU 7; policy thread FIFO 80 and DDS receive FIFO 85 on CPU 6 with the motor NIC's interrupt; memory locked | writer 500 Hz, policy 50 Hz | motor link, state estimate, safety checks, policy, arm / head executor, limits, telemetry records | `r1guard` takes over in about 10 ms (§4) |
| `r1guard` | C++17 | FIFO 88 on CPU 5 | 1 kHz cycle; writes at the core's rate when active | hot standby with its own state reader and a warm shadow policy; forensics | `r1rt` keeps running; a missing guard is reported and blocks a takeover |
| `r1d` | Python 3.11 | normal, CPUs 0-4 | executor tick 100 Hz | API, goals, leases, world model, planning in worker processes, skills, perception, measurement | arm motions decelerate to a hold when their keepalives stop; balance unaffected |
| `r1hands` | Python | normal | 100 Hz | the Dex3-1 hands through `dex3_server`, the pads, the grip state machine | the hands' own timeout makes them passive about 1 s later (a held object drops); the body is unaffected |
| `r1cam` | Python | normal | per frame | the head camera streams, capture timestamps | perception requests fail with a code |
| `r1log` | Python | nice 10 | drains rings | MCAP logs, flight recorder | rings overflow with a drop counter; nothing blocks |
| `dex3_server` | Unitree binary | normal | - | USB to DDS for the hands (not redistributable) | supervised |

All run as systemd services on the backpack. `r1rt` logs the scheduling it was granted and says so in its ready line
when real-time scheduling was refused - never silently.

**Stopping a service never damps.** `r1 shutdown` is accepted only when the core is in MONITOR (publishing nothing) or
DAMP. While we own the motors, systemd refuses a manual stop / restart of `r1rt` and `r1guard`, a reboot is blocked by
an inhibitor lock, and `r1rt` ignores SIGTERM in a publishing mode. If `r1rt` is killed anyway, `r1guard` takes over.
Nothing in any stop path publishes damp.

## 3. The real-time core `r1rt`

### 3.1 The writer loop (500 Hz; every iteration timed)

1. Wait for the next 2 ms deadline (absolute, CLOCK_MONOTONIC). Record the wake-up lateness.
2. Read the latest robot state from the DDS receive thread (a seqlock copy; never blocks). Check its age.
3. Read inputs, all non-blocking: the operator stop channel, the mode-request mailbox, the upper-body command ring,
   the base command (with a time-to-live), the hands' contact bits, the latest policy action with its timestamp.
4. Estimate: base orientation, angular velocity, projected gravity, tilt.
5. **Safety checks** (§5.2).
6. Legs + waist: the latest valid policy action (held between policy steps, as trained). If the policy output is older
   than two policy periods (40 ms), run the policy inline on this thread (rung 1 of the ladder, §5.3).
7. Arms + head: the upper-body executor (§3.3), with gravity feed-forward from a built-in arm model (§3.4).
8. Hard limits (position, velocity per tick, torque, gain bounds) from the versioned profile; every clamp is counted.
9. Fill LowCmd (the R1's slots, Unitree's CRC32), re-check the ownership token, write.
10. One telemetry record into the ring (§10); the heartbeat and the mirror for the guard.

Nothing else happens on this thread: no files, no logging calls, no network reads, no allocation, no Python.

After any stop of the writer longer than 100 ms (a deliberate SIGSTOP in a test, a whole-machine stall) the writer
does not resume from its old targets: it waits for live state and blends from the MEASURED pose, so a robot that
sagged during the stop is not snapped back.

### 3.2 The policy thread (50 Hz)

The balance policy is a small MLP (74 observations -> 512 -> 256 -> 128 -> 14 actions, ELU) trained with
[mjlab](https://github.com/mujocolab/mjlab) on Unitree's R1 model ([`POLICY.md`](POLICY.md)). The core runs it as
fixed-size Eigen code with weights converted from the ONNX file at install time; no allocation. A golden test holds it
to the Python reference implementation (max difference 2.8e-14 on 10,000 observations; target 1e-5). The policy's
inputs include a small forward-lean command (0-5 deg, as trained) that the API exposes as `body.pose`.

### 3.3 The upper-body executor (arms and head; the hands are separate)

Per group (`arm_left`, `arm_right`, `head`) exactly one source at a time:

| Source | What | Deadman |
|---|---|---|
| HOLD | hold the target with the hold gains + gravity feed-forward | none needed |
| TRAJECTORY | a quintic spline through knots from `r1d`, sampled at 500 Hz; may carry guards | `r1d` renews a keepalive; on expiry the group decelerates to a stop (0.15 s) and HOLDs |
| STREAM | per-joint setpoints with a time-to-live (<= 0.3 s for the arms) | on expiry: decelerate and HOLD |
| FREEDRIVE | kp 0, small kd, gravity feed-forward (hand-guiding) | operator only, timed |

**Guards evaluated every tick on a trajectory**: joint lag along the commanded motion (the arm is blocked), a torque
rise (on joints with a calibrated torque estimate), a pattern of hand pad contacts, and a tilt stop (the body tilting
4 deg beyond where the motion began). On a guard: decelerate within a few milliseconds, HOLD, report which guard and
the measured pose. Every stop uses one smooth deceleration profile per joint (continuous velocity and acceleration).
Speed control (halt / resume) acts on the trajectory clock inside the core, so it works even if `r1d` is slow.

### 3.4 Built-in models

The arm gravity model (two 5-joint chains + hand + backpack mass) and the hands' fingertip kinematics are small
generated C++ routines from the same MJCF the simulator uses, tested against MuJoCo on random poses.

### 3.5 Modes

| Mode | Legs + waist | Arms + head | Entered by |
|---|---|---|---|
| MONITOR | not published (Unitree's controller in charge) | not published | start |
| TAKEOVER (MIRROR -> BLEND) | mirror Unitree's last command, then blend into ours (1.0 s) | same | the operator's `takeover`, after gates (below) |
| BALANCE | the policy (base command with a time-to-live, zero by default) | executor | the end of the takeover |
| GUARDED | the guard owns the motors (§4) | HOLD | a guard takeover |
| STAND_FREEZE | ramp (0.4 s) to a double-support stand pose at stand gains, then hold | HOLD | no valid policy (rung 3), the operator's `freeze`, the optional state-age rung |
| ON_STAND | a stiff hold of the measured pose (per-joint commands allowed, legs included) | executor | the operator's `on-stand`, robot hanging on its stand: the same release + mirror / blend as a takeover, into a hold instead of the policy |
| DAMP | kp 0, damping | kp 0, damping | **the operator only** |

**The takeover gates** (from the previous controller, proven on hardware): tilt <= 12 deg, fresh state, motor
temperatures below 60 C, joint speeds small, Unitree's own command fresh and stiff, a live guard, real-time scheduling
granted; then Unitree's controller is asked to release (within 4 s) and ours mirrors and blends. **Failure branches
never damp**: a gate failing before the release stays in MONITOR (nothing of ours is published); a release that does
not happen leaves Unitree in charge with an alarm; a fault during MIRROR keeps mirroring and continues into BLEND; a
fault during BLEND finishes into BALANCE (the policy is the best balancer available). There is no hand-back to
Unitree's controller.

A **stand-session lock** (`r1rt --stand-session`) refuses every entry into BALANCE for the process lifetime: the first
sessions on a robot hanging on its stand run with it, so a mistyped word cannot start the balance policy on a
hanging robot.

`r1 balance` (operator only) returns from a STAND_FREEZE that came from BALANCE back into BALANCE, after the core's
checks (fresh state among them).

### 3.6 Configuration

One versioned robot profile (`config/robot_profile.r1_dex3.yaml`, validated and frozen at start): the joint map from
Unitree's SDK, limits (position, velocity, torque), gains with bounds, rates, guard defaults, stop times, the safety
thresholds. Every value carries provenance (source, date, status). At run time only fields marked `tunable` change,
within their bounds; safety fields need a restart and a version bump.

## 4. The guard `r1guard`

The guard is a separate process on its own CPU with its own DDS state reader. It shares one memory segment with the
core and touches only: the core's heartbeat (read), the mirror of the core's last command, policy state and targets
(read), the ownership token (compare-and-swap), and its own heartbeat (write).

- **Warm standby.** Every policy period it restores its own copy of the policy from the mirror and computes a shadow
  action on its own state reading, so a takeover continues from where the core was (in lockstep the shadow equals the
  core's action bit for bit).
- **Takeover.** When the core's heartbeat is older than 5 writer periods (10 ms at 500 Hz), the token is the core's
  and the core is publishing, the guard swaps the token and publishes from the same cycle.
- **What it does depends on the mode it took over from**: from BALANCE it runs its policy with a zero base command and
  holds the arms and head (with the core's gravity feed-forward); from a takeover blend or a freeze it continues the
  core's ramp from where it was; from ON_STAND it holds the core's last command (holding, not balancing); from DAMP it
  keeps damping (the operator's state); from MONITOR it does nothing. It never damps on its own.
- **Its own failures**: its own policy output invalid -> STAND_FREEZE; its own state stale -> the last command again.
- **Forensics**: at the takeover it records the core's last heartbeat, the token words, the tilt and, on Linux, the
  stalled process's threads from `/proc` (wchan, syscall, stack) - after the first write, never on the cycle path.
- **Giving the motors back** is the operator's decision (`r1 reclaim`): the core blends from exactly what the guard
  sent; the guard re-arms itself.
- **The operator's damp** reaches the motors through the guard too.

## 5. Safety

### 5.1 Hazards and the design answer

| Hazard | Example | Answer |
|---|---|---|
| The balance thread blocks | a network-driver read during a Wi-Fi drop | nothing blocking in `r1rt`; the guard; network health read only outside the core, and never through the blocking driver call |
| The balance process crashes or hangs | a bug, a page-fault storm | the guard takes the token in about 10 ms and balances in place |
| A Python service stalls or crashes | an exception, a slow planner | keepalive / time-to-live expiry -> the arms decelerate to HOLD; balance unaffected |
| The hands' logic stalls | an exception in the grip code | the hands' own timeout makes them passive; the body unaffected |
| The policy output is invalid | NaN, overrun | inline policy, then STAND_FREEZE |
| The robot state stops arriving | a cable fault | the last balanced command is kept (nothing better is possible without state); the board holds the last command about 0.4 s, then damps by itself - which is why the link is hardened (§11) |
| An arm or hand hits something | a table edge, the other arm | planner checks against one world model, executor guards (lag, contact), the tilt stop |
| Contact forces tip the robot | pressing on a table | force-limited skills; the arm tilt stop |
| Over-temperature | a hand motor at a hard squeeze (on our robot ~6 C/s at 0.9 N m) | the squeeze halves from 60 C; the grasp ends (relaxed, still powered) at 72 C; warnings in words from 65 C |
| Two writers on `rt/lowcmd` | a takeover race | the ownership token; MONITOR publishes nothing; no hand-back |
| A wrong configuration | an unbounded tuning value | bounded tunables; safety fields frozen at start |

### 5.2 Checks in the core (every tick)

| Check | Threshold | Action |
|---|---|---|
| Ownership token | taken by the guard | do not write |
| Operator stop channel | stop / halt / damp with sequence numbers | stop: all groups HOLD, base command zero, keep balancing (latched until `resume`); damp: DAMP (operator only) |
| The remote's L2+B | decoded from the wireless remote | DAMP (the operator's choice) |
| State age | late after 10 ms; resend the last balanced command after 100 ms; a fault after 1 s - never limp | counted, reported |
| Tilt | arm tilt stop 4 deg over the start of an arm motion; fall detection 30 deg | arm stop; at 30 deg the policy keeps running (it may catch it); no automatic damp |
| Joint limits, velocity, torque | profile values | clamp + count; repeated clamping on an arm -> HOLD that arm |
| Motor error word | any bit | event + alarm; an arm / head motor -> HOLD that limb; a leg / waist motor -> the policy keeps balancing and the operator decides |
| Policy age / validity | > 40 ms: inline; invalid: STAND_FREEZE | |
| Loop timing | wake-up lateness and iteration time histograms; a gap > 2 periods is an event | logged, never acted on blindly |

An optional **state-age rung** (off by default) freezes the robot into its stand pose when the state it balances on is
older than a limit for a sustained time; with it off, the age is measured and reported (`r1 status`).

### 5.3 The never-limp fallback ladder

| Rung | Trigger | Who acts | What the legs do | Time to act |
|---|---|---|---|---|
| 0 | normal | `r1rt` | the policy | - |
| 1 | the policy thread late (> 40 ms) | `r1rt`'s writer | the policy run inline, bumpless | the next tick |
| 2 | `r1rt`'s heartbeat older than 5 writer periods | `r1guard` | depends on the mode it took over from (§4); forensics | about 10 ms (measured on our robot: 9.8-10.7 ms) |
| 3 | no valid policy anywhere | `r1rt` or `r1guard` | STAND_FREEZE: ramp to the stand pose, hold | 0.4 s ramp |
| 4 | state lost | either | the last balanced command held | immediate |
| - | **never** | - | automatic damping or zero torque | - |
| operator | the person at the robot | `r1 damp`, the remote's L2+B, the hardware E-stop | DAMP / motor power off | - |

A whole-machine stall (both processes stopped) is not covered by software. It is why the network and kernel set-up
matters (§11), and why the board's own behaviour without commands was measured (it holds about 0.4 s).

**A fall already in progress** (tilt beyond what the policy can catch): the policy keeps running and nothing damps. A
protective fall posture is future work.

### 5.4 Start-up and session end

- Start: the services start in MONITOR (publishing nothing). Self-checks: real-time grants, DDS discovery (state rate,
  the board's mode), a live guard, the hands, the camera calibration, the configuration version.
- Session end, always the same: a person holds the robot -> `r1 damp` -> support it standing -> `r1 shutdown` (which
  saves the session's logs with checksums) -> power off.

### 5.5 Verification

Every rung and check has a simulator test with fault injection (§12.5), and the paths that involve the motor board
were tested on a robot hanging on its stand (the board-hold test, the guard-takeover test) before any balancing. The
stall that made the previous controller fall is a permanent regression test: in the simulator, the previous
architecture falls; r1mc3 with its real processes does not ([`SAFETY.md`](SAFETY.md)).

## 6. The command API

One API (`r1d`, JSON over TCP on the backpack's localhost, port 8780), one lease model, typed schemas (pydantic, JSON
Schema), a goal lifecycle (accepted -> active -> succeeded | aborted | canceled | preempted | timed_out, or rejected),
resource groups (`arm_left`, `arm_right`, `head`, `hand_left`, `hand_right`, the legs and waist in ON_STAND), SI
units, the base frame (on the floor between the feet, x forward, y left, z up). Errors are codes (`OK`,
`INVALID_ARGS`, `WRONG_MODE`, `NOT_OWNER`, `RESOURCE_BUSY`, `UNREACHABLE`, `COLLISION`, `LIMIT`, `GUARD_STOP`,
`NOT_HOLDING`, `TIMEOUT`, ...) with structured details, never prose to be matched.

| Layer | Commands |
|---|---|
| L0 joint | `joint.command` (q, dq, kp, kd, tau_ff, time-to-live), `joint.hold`, `hand.command` |
| L1 stream | `stream.joint_jog`, `stream.head`, `stream.arm_twist` (resolved-rate Cartesian velocity) |
| L2 goal | `arm.move_joint`, `arm.reach`, `arm.move_linear`, `arm.move_relative`, `arm.move_path` (optionally blended), `arm.hover`, `arm.lower` (guarded descent), `arm.guarded_move`, `arm.slide`, `arm.carry`, `arm.plan` (dry run), `arm.stop`, `hand.grasp`, `hand.release`, `hand.preshape`, `hand.tare`, `hand.calibrate`, `head.look_at`, `body.pose` (the policy's lean) |
| L3 skill | `skill.ready`, `skill.table_ready` (both arms to a ready posture, also from under a declared table), `skill.pick`, `skill.place`, `seq.run`, `par.run` |
| System | `status`, `stop` / `halt` / `resume` (always accepted), `damp` (operator only), `takeover`, `on_stand`, `freeze`, `balance`, `reclaim`, `shutdown`, `lease.*`, `world.*`, `camera.*`, `measure.*`, `record.*`, `goal.*` |

Every motion result carries, from the core's measured state: `moved`, `hand_start`, `hand_now`, `target`, `achieved`,
`error_mm`, the fingertip positions, the grasp centre, `stopped_by` (a code + the guard's detail), the duration and
the plan id. A refusal says whether anything moved - by construction nothing does.

Clients: the Python SDK `r1mc3.client` (typed, one method per command; there is no `damp` method on purpose), the
operator's `r1` command line, and `r1sim` in tests. Reference: [`API.md`](API.md), [`OPERATOR.md`](OPERATOR.md).

## 7. Hands (`r1hands`)

### 7.1 Structure

- Talks to Unitree's `dex3_server` over DDS (`rt/dex3/<side>/cmd|state`, 100 Hz, with the hands' own timeout bit
  set), checks the state's age and detects a frozen state stream.
- Per-finger commands with torque caps; heat: the squeeze halves from 60 C, the grasp ends (relaxed, still powered)
  at 72 C, and a grip at or above 65 C is reported "warm" in words.
- The pressure pads (9 pads x 12 taxels per hand): a tare per hand shape, a slow creep follower while the hand is open
  and empty (the pads drift by tens of counts per hour), spike rejection. On our robot the pads feel only firm contact
  on the fingertip's face - light touches give no signal, so the verdict cannot rest on the pads alone.
- Publishes hand state and a compact contact bitmask into shared memory every tick, for `r1d` and the executor's
  contact guard.

### 7.2 Grasp and the "holding" verdict

Grasp: preshape -> close -> per-finger stop on contact evidence (pad rise, stall, torque) -> squeeze (per grip) ->
verdict -> lift check. Grips: `wrap`, `pinch`, `tip`; object classes: `rigid`, `thin_flexible` (cloth-like, judged by
tip-to-tip pressure, closure and the lift after an air-pinch calibration of the hand), `soft`.

The verdict is a state machine with evidence, never force alone:

| Evidence | Holding needs |
|---|---|
| Opposition | contact on the thumb side AND the finger side |
| Finger gap vs the object | the gap consistent with the declared width (for a pinch: the aperture at least max(6 mm, half the width - 13 mm) and the tips not met) |
| Contact height | the contacts above the support surface (contacts AT the surface with no object evidence = pressing the table) |
| Lift check | after a short lift the contact persists and the fingers do not run on |

States: `OPEN -> CLOSING -> CONTACT -> TENTATIVE -> HOLDING | NOT_HOLDING (reason)`. **Contact decides**: once holding,
fingers closing further while the contact persists is not a slip (only closing on nothing is), and a NOT_HOLDING
verdict never relaxes the fingers by itself - the hand keeps its grip until it is told to open.

## 8. World model and planner

### 8.1 One world model

- The robot: the same MJCF as the simulator, posed from the live joint state (the IMU's own mounting rotation taken
  out, so the base frame is where the feet really are).
- Declared entities: `surface` (a table top), `container` (walls with thickness and a floor: a hand may be inside
  above the floor), `obstacle`, `object` (box / cylinder / sphere), each with a pose, a size and a source.
- **One geometry engine** for every query: MuJoCo collision and distance on one model.

### 8.2 The planner (worker processes; pure functions of a world snapshot)

1. Target: IK for the 5-DoF arm with the approach direction as a secondary task, and a hand-orientation search
   (elbow swivel x wrist roll) ranked by the requested approach style (`front`, `inward`, `around` - the palm turned
   to a tall object -, `low` - the fingertips kept just over a flat object's bottom) and a level-hand preference.
2. Path: a straight Cartesian line if valid, else RRT-Connect in joint space (seeded, reproducible) with shortcuts;
   every candidate checked densely against the robot itself, the other arm, the world and the balance envelope.
3. Timing: knots for the executor's quintic spline within the joint limits.
4. **Whole-goal validation before anything moves**: multi-leg goals (hover = lift + transit + descend; carry = rise +
   travel + lower) are planned end to end first.
5. Result: a plan id, the trajectory, the predicted contacts and landing, clearances, and why if refused.

The grasp aim is where the chosen grip's fingers actually close (from the hand model and the object's width), not the
open hand's midpoint. A held object is planned as part of the arm. Measured on a laptop: a hover plan takes about
0.3 s of CPU; on the Orin it is about 3-4x slower.

## 9. Perception and measurement

- **The head camera** (`r1cam`): the left image (544 x 448 on our robot), depth on demand (streaming depth all the time
  shares the motor link's USB bus: on our robot it cost state messages until depth was limited to the moment of an
  observation), the right image for calibration recordings. Capture timestamps; the robot's pose at capture.
- **Pointing**: a pixel's ray through the calibrated lens meets the declared surfaces; depth adds heights. A click on
  a thin, clear or small object can read the surface behind it: `camera.point` reports which surface it hit, and a
  refusal says there is no usable depth.
- **Measuring a set-up** (`measure.setup`, `measure.phone`): AprilTags (36h11) on the client's entities, seen by the
  head camera or by a phone still (the phone's pose from tags on the table plane, its lens calibrated once with a
  ChArUco board), give the entities' poses in the robot's base frame with a fit error; out-of-tolerance checks are the
  client's.

## 10. Telemetry and logging

| Stream | Content | Rate |
|---|---|---|
| Core records | send / receive times, board tick, wake-up lateness, q, dq, tau_est, temperatures, the command (q, dq, kp, kd, tau_ff), IMU, mode, faults, token owner | 500 Hz |
| Policy | observation and action | 50 Hz |
| Guard | heartbeat age, faults, modes, the command when active | 1 kHz |
| Hands | q, dq, tau, raw pads, contact bits, grip state, temperatures | 100 Hz |
| Events | goals, plans, refusals, faults, stops, takeovers | event |

Shared-memory rings -> `r1log` (low priority) -> MCAP files (zstd). The rings drop with a counter; the core never
waits. A flight recorder dumps the last seconds of everything on a fault, a takeover, a large tilt or the operator's
word. `r1 shutdown` saves the session's logs into a new folder with checksums.

## 11. Deployment on the backpack

- **Releases side by side**: a release tarball (with a manifest of every file's checksum) installs under
  `/opt/r1mc3/releases/<version>-<commit>` and builds there (its own Python venv by uv, the C++ core and guard);
  activation switches a symlink; rollback switches back. Both refuse while our processes own the motors.
- **Real-time set-up, staged** (each step measured and reversible): systemd units with FIFO priorities, memory
  locking and CPU affinity; a runtime CPU partition (system services, Python and other interrupts on CPUs 0-4, the
  guard on 5, the motor NIC's interrupt + the policy and DDS threads on 6, the writer alone on 7); the kernel's
  real-time budget opened (`kernel.sched_rt_runtime_us = -1`: container runtimes give every service a cgroup with a
  real-time budget of 0, so a FIFO request fails without it; our threads are pinned to CPUs 5-7, so a runaway cannot
  freeze CPUs 0-4); quiet serial consoles (`kernel.printk`), because a kernel notice printed to a 115200-baud console
  stalled a real-time thread for 24.5 ms on our robot. No bootloader change and no PREEMPT_RT kernel were needed:
  `cyclictest` under load p99.9 23 us, max 155 us after the set-up.
- **DDS**: CycloneDDS bound to the motor NIC; the real-time processes read through a 256 kB receive buffer, which
  bounds how much stale backlog can sit in front of a fresh sample after a stop.
- **The network**: the motor NIC never carries the default route; nothing reads the NIC's link speed through the
  driver while the controller runs (that read blocked for seconds on a Wi-Fi reconnect).

The step-by-step path: [`hardware/BRING_UP.md`](hardware/BRING_UP.md).

## 12. The simulator (`r1sim`)

A MuJoCo model of the R1 + two Dex3-1 hands + the backpack, physics independent of any controller, deterministic in
lockstep, real-time capable on a laptop, with the robot's interfaces. In brief (details: [`SIMULATOR.md`](SIMULATOR.md)):

- **Two profiles**: *training-equivalent* (bit-identical to the policy's mjlab training model) and
  *real-equivalent* (articulated head, the IMU at its real place, Dex3-1 hands with 33 pad cells each, the backpack
  and rear masses, a motor-board model, delays, sensor models). Every parameter has a source and a status in
  `models/provenance.yaml`; seeded variants draw parameters within their ranges.
- **The motor board without commands**: holds the last command 0.385 s, then damps - fitted on the robot.
- **Interfaces**: lockstep (the C++ core and guard through Python bindings, `r1d`'s executor, the hands, stepped in a
  fixed order on a simulated clock) and a **DDS bridge** (the simulator publishes `rt/lowstate` and the hands' state,
  serves Unitree's mode-switch RPC, subscribes `rt/lowcmd` with CRC checks: the real binaries run unchanged).
- **Scenes**: an empty floor (default), a plain table with simple objects, the robot on its stand; any MJCF; task
  scenes plug in from outside the package.
- **Fault injection** (§12.5): blocked threads and processes (SIGSTOP, sleeps at code points), DDS gaps, state delays,
  CRC errors, a second writer, a policy NaN, a frozen hand stream, motor error words, pushes - each scheduled, so a
  failure is reproducible.
- **The viewer**: [`sim/VIEWER.md`](sim/VIEWER.md).

## 13. Validation by identification

The simulator is trusted for a purpose only when the matching checks pass, each against named data:

- **Training equivalence**: the training-equivalent profile reproduces mjlab rollouts.
- **Identification on the robot** (hanging on its stand, feet off the floor; [`IDENTIFICATION.md`](IDENTIFICATION.md)):
  per-joint steps and chirps, slow torque ramps for friction, static poses with payloads, IMU statics, latency steps,
  the board's behaviour without commands, the head camera's lens (ChArUco) and depth; the fits with their residuals in
  `models/provenance.yaml`.
- **The policy on the identified model**: 400 / 400 runs passed across seeded variants (standing, arms forward, a
  4 deg lean with the arms out, pushes) - no retraining was needed.
- **The safety paths**: the stall that felled the previous controller, injected into the real processes on the
  simulated robot: upright in 23 / 23 runs, guard takeover 10.2-12.2 ms, one writer, never limp.
- **On hardware**: the read-only checks, the stand session, the board-hold and guard-takeover tests, the joint
  direction check, then balancing ([`hardware/BRING_UP.md`](hardware/BRING_UP.md)).

## 14. Decisions

The decisions that shape the code, with the reason (a short form of our internal decision record; the numbers "D<n>"
in code comments refer to it).

| # | Decision | Why |
|---|---|---|
| D1 | A C++ real-time core + an independent C++ guard + Python processes for everything else | a single control thread that also did I/O blocked and the robot fell |
| D2 | Our own RL policy on `rt/lowcmd` owns balance (Unitree's `arm_sdk` overlay only as an optional, unbuilt mode) | the field's norm for research control; the overlay is unproven on the R1 |
| D3 | The never-limp ladder: inline policy -> guard -> stand-and-freeze; no automatic damping ever; a falling robot keeps its policy | the operator's rule; damping a standing robot drops it |
| D4 | MuJoCo on the CPU as the simulator's truth (the policy's training engine); rendering in a separate worker | same engine as training; a laptop can run it |
| D5 | Gates: the simulator trusted on existing data; the core and guard proven on real processes against the simulator; the stand session before any balancing | each risky step validated before it is met on hardware |
| D7 | 500 Hz writer (200 Hz fallback), policy 50 Hz as trained, hands 100 Hz | Unitree's examples write at 500 Hz; the state arrives at ~1 kHz |
| D8 | Real-time set-up by runtime CPU partitioning first; no bootloader change unless measured necessary | measured: not necessary |
| D11 | Apache-2.0; CI entirely in simulation; no secrets or machine paths in the repository | compatible with MuJoCo and Unitree's SDK |
| D13 | Photo-like rendering in a render-only worker on a newer MuJoCo; the physics stays on the training version | Filament rendering needs MuJoCo >= 3.14; the policy was trained on 3.5.0 |
| D15 | STAND_FREEZE keeps a 0.4 s ramp | a faster ramp means a ~40 N m torque step |
| D16 | Every stall / fall safety test runs on the least stable plausible model (no leg stiction) | the nominal model's stiction hides marginal cases |
| D22 | ON_STAND for a hanging robot: the takeover's gates minus the policy, a 1 s blend into a stiff hold of the measured pose | the first hardware sessions need the motors without balancing |
| D23 | The stand-session lock refuses BALANCE for the process lifetime | a balance policy on a hanging robot is a hazard |
| D25 | Guard threshold 10 ms, inline-policy rung 40 ms | the board holds ~0.4 s; the sweeps showed margin |
| D30 | The kernel's real-time budget opened by a sysctl file | container runtimes give services a zero real-time budget |
| D31 | Quiet serial consoles (`kernel.printk = 1 4 1 7`) | a console print stalled a real-time thread 24.5 ms |
| D33 | A 256 kB DDS receive buffer for the real-time processes | bounds the stale backlog after a stop |
| D34 | The state-age rung off by default; the age always measured and reported | it would freeze the robot on short losses the core rides out |
| D36 | The freeze pose accounts for where the arms are (an ankle shift from the arms' centre of mass) | a freeze with the arms forward would otherwise tip |
| D37 / D40 | Contact decides: closing further on a persisting contact is not a slip; a NOT_HOLDING verdict never relaxes the fingers | real carries were judged not holding while held |
| D45 | The camera's depth streams only for an observation | full depth cost state messages on a shared USB bus |
| D46 | The pad tare follows the pads' slow creep while the hand is open and empty | with a fixed tare every pad reads "touching" within about an hour |
| D48 | The planner prefers a level hand (the closing line within 12 deg of level) | rolled hands hit the support with one finger first |
| D50 | The public repository is an export of the development repository (an allowlist + a word check + its own CI), not a fork | the development history and the site stay private; fixes re-export as 0.1.x |

## 15. Risks and what is not solid

| Risk | Mitigation |
|---|---|
| The board's behaviour without commands can change with firmware | the board-hold test on the stand after any firmware update |
| DDS wire compatibility of our own CycloneDDS types with the robot | the read-only checks on day one (passed on our robot: 0 bad CRCs) |
| MuJoCo contact for small-object grasps | identification per material where data allows; grasps are checked on the stand before use |
| The policy may not stand on another robot's identified model | the robustness batch on your identified model before balancing ([`IDENTIFICATION.md`](IDENTIFICATION.md)) |
| Kernel jitter on the Orin | the staged real-time set-up, measured with `cyclictest` |
| A whole-machine stall defeats both processes | network and USB hardening; the board's ~0.4 s hold; the flight recorder |
| `dex3_server` is closed source | supervised; its state age checked |
| Python stalls cause arm stops | planning in worker processes; keepalive margins (availability, not safety) |

Not solid (said plainly): every number is from one robot; the planner's speed on the Orin is at its 1 s target, not
safely under it; grasps from a camera click are not reliable on thin, clear or small objects; most L2 / L3 goals are
simulator-proven only; a protective fall posture, Unitree's `arm_sdk` overlay and a hand-back of the motors to
Unitree's controller are not built.
