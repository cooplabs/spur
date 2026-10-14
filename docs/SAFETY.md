# Safety

r1mc3 commands a humanoid of about 30 kg. This page is the safety model in plain words: what the software does by
itself when something goes wrong, what it never does, and what the operator must do. The design behind it:
[`DESIGN.md`](DESIGN.md) §4-§5.

## The one rule: never limp

"Limp" = damping: the joints' stiffness goes to zero and a standing robot collapses. **No automatic path in r1mc3 ever
damps the robot** - not a fault, not a stall, not a lost link, not a service stop, not a fall in progress. Damping
happens only when the operator asks for it:

- `r1 damp` (the command line; the SDK has no damp method on purpose),
- the wireless remote's L2+B,
- the hardware E-stop / motor power.

The reason: a standing robot that is damped falls. A robot whose software is stuck but whose motors hold the last
stiff command, or whose standby balances it in place, does not.

## What happens by itself (the fallback ladder)

| What goes wrong | What r1mc3 does | Time |
|---|---|---|
| The policy thread is late (> 40 ms) | the core runs the policy inline on its writer thread | next tick (2 ms) |
| The core process stalls, hangs or is killed | `r1guard`, already running with a warm copy of the policy, takes the motors and balances in place; the arms hold | ~10 ms (measured on our robot: 9.8-10.7 ms) |
| No valid policy output anywhere (NaN, out of range) | STAND_FREEZE: the legs ramp to a stand pose at stiff gains and hold | 0.4 s ramp |
| The robot state stops arriving | the last balanced command is held | immediate |
| A Python service (`r1d`, the planner, the hands) stalls or crashes | arm motions decelerate to a hold when their keepalives stop; balance is not affected | 0.15 s stop |
| An arm is blocked, touches something, or the body tilts during an arm motion | the executor's guards stop that arm smoothly and hold | a few ms |
| A hand motor runs hot | the squeeze halves from 60 C; the grasp ends (relaxed, still powered) at 72 C | - |
| Someone stops a service, reboots or deploys while we own the motors | refused (systemd refuses the stop, an inhibitor blocks the reboot, activation refuses) | - |

Nothing leaves STAND_FREEZE or a guard takeover by itself: the operator decides (`r1 balance` back into balancing
after a freeze from balancing; `r1 reclaim` to give the motors back to the core after a guard takeover).

**What is not covered**: a stall of the whole computer (both the core and the guard stop). Then the motor board holds
the last command for about 0.4 s (measured on our robot) and then releases into damping by itself. This is why the
backpack's network and kernel are set up as described in [`hardware/BRING_UP.md`](hardware/BRING_UP.md), and why the
logs keep the evidence (the flight recorder, the guard's forensics).

**A fall already in progress** (tilted beyond what the policy can catch): the policy keeps running, nothing damps. A
protective fall posture is not built.

## What the operator must do

1. **Before any word that moves something**: a person holds the robot, or it hangs on its stand with the feet off the
   floor, and knows how to stop it (`r1 stop`, the remote, the E-stop).
2. **One step at a time.** Every hardware step says what moves and how to stop it. New modes are used on a robot only
   after a simulator test and a stand test.
3. **First sessions on the stand, with the stand-session lock** (`r1rt --stand-session`): balancing is refused for the
   process lifetime, so the balance policy cannot start on a hanging robot.
4. **The session end is always the same**: hold the robot -> `r1 damp` -> support it standing -> `r1 shutdown` (saves
   the logs) -> power off.
5. **Safety-loosening changes** (a threshold, a limit, a disabled check, a longer test) need an explicit decision by the
   person responsible for the robot, and are recorded.
6. **Re-check after any firmware or model change**: the read-only checks, the joint directions
   ([`hardware/JOINT_DIRECTIONS.md`](hardware/JOINT_DIRECTIONS.md)), and the board-hold test on the stand.

## Stops

| Word | Effect |
|---|---|
| `r1 stop` | everything stops where it is: arms / head HOLD, the base command goes to zero, balance continues. Always accepted, from any client. Latched until `r1 resume` |
| `r1 halt` | the trajectory clock halts; motion decelerates to rest (resumable) |
| `r1 freeze` | the legs ramp to the stand pose at stand gains and hold (STAND_FREEZE) |
| `r1 damp` | **limp** - the operator's word only; hold the robot first |

Commands issued before a stop are refused by the core afterwards (a stop epoch), so a late command from a slow client
cannot restart a motion.

## How it was verified

| Check | Where | Result |
|---|---|---|
| Every rung of the ladder and every check | simulator fault injection (lockstep and the real processes over DDS) | tests in `tests/` |
| The stall that made the previous controller fall (a 4.0 s stall of the balance thread) | the real `r1rt` + `r1guard` against the simulated robot | upright 23 / 23 runs, guard takeover 10.2-12.2 ms, one writer, never limp (the previous architecture fell in 5 / 10 seeds) |
| What the motor board does without commands (the board-hold test) | our robot on its stand, the guard switched off for the test only | holds ~0.4 s, then damps |
| The guard takeover (the guard-takeover test: the core stopped with SIGSTOP) | our robot on its stand | 10.7 ms (twice), 9.8 / 10.0 ms another day; held stiff; motors back only on `r1 reclaim` |
| The operator's freeze while balancing, and back | our robot, standing, a person holding it | tilt 3.0-3.4 deg in the freeze, balancing again at once |

## Testing safely

- Reproduce problems in the simulator first (`sim/r1sim`, the fault injector), never first on a robot.
- On a robot: the operator procedures above, a person holding the robot or the robot on its stand, the E-stop in
  reach.
- If you find a path that damps the robot without an operator's command, or that lets two processes write to the
  motors, that is a safety defect: report it privately ([`SECURITY.md`](../SECURITY.md)).
