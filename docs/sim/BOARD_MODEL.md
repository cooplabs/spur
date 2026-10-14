# The motor board without commands: what the real board did, and the simulator's model

When the controller stops sending `rt/lowcmd`, what does Unitree's motor board do? Hold the last command? Let go? For
how long? It decides how long any software stall can be before the robot is in danger, and so how fast the guard must
take over. Code: `sim/r1sim/b6_board.py` (`python -m r1sim.b6_board [--out f.json]`, ~10 s: the simulated test), the
model in `models/driver.yaml` (read by `DriverConfig.real()`), the engine's timeout path; data: an excerpt of the real
test in `tests/assets/hw1/` (with a summary and a README saying how it was cut).

## What our robot's board did (the board-hold test, on the stand)

r1rt paused (SIGSTOP) in ON_STAND with the robot hanging on its stand, the guard switched to observe-only, r1d's state
recorder running ([`../hardware/BRING_UP.md`](../hardware/BRING_UP.md) "The board-hold test"). The timing comes from
the board's own 1 kHz tick in every state sample.

| Pause | What happened |
|---|---|
| 20 ms | nothing: max joint motion 2.7e-5 rad - the board HOLDS the last command |
| 100 ms | nothing: 3.2e-5 rad |
| 500 ms | **at ONE tick, ~384 ms after the last command, every loaded joint let go at once** (19 joints in the same tick): the left hip pitch's torque fell -3.55 -> -2.54 -> -1.92 N m in 2 ticks, its speed 0 -> 0.11 rad/s; then the loaded joints sank at a nearly CONSTANT speed with `tau_est = -kd x dq` - a **damped** release, not a free fall - until r1rt resumed |

**The hold from the last command: 383-387 ms.** One release event, on one robot and one firmware. A second session
(20 / 100 / 300 / 400 / 500 ms) agreed: 300 ms 0.00004 rad, 400 ms 0.0023 rad, 500 ms 0.028 rad.

Per joint over 395-495 ms:

| Joint | hold torque (N m) | sink (rad/s) | motion at 500 ms (rad) | -tau_est / dq |
|---|---|---|---|---|
| L / R hip pitch | -3.53 / -2.70 | 0.117 / 0.133 | 0.0135 / 0.0142 | 16.5 / 13.2 |
| L / R knee | 1.05 / 0.66 | -0.035 / -0.051 | 0.0031 / 0.0050 | 10.0 / 13.8 |
| waist roll | -2.36 | 0.020 (stops after ~40 ms) | 0.0028 | 16.5 |
| L / R shoulder pitch | -1.48 / -1.86 | 0.053 / 0.061 | 0.0052 / 0.0058 | 11.5 / 10.6 |
| L / R shoulder roll | 1.51 / -1.49 | -0.058 / 0.058 | 0.0064 / 0.0057 | 13.4 / 13.1 |
| L elbow | -0.43 | 0.043 | 0.0054 | (3.5: small, noisy) |

The ankles were released in the same tick but did not move (their friction, 2-3 N m, exceeds their gravity load).

## The model (`models/driver.yaml`, status measured)

| Parameter | Value | Range | Why |
|---|---|---|---|
| `cmd_timeout_s` | **0.385** | 0.375-0.400 | the release 383-387 ms after the last command |
| `on_cmd_timeout` | **damp** | - | a constant sink speed with tau = -kd dq: kp 0, kd > 0 (a free fall would reach ~0.03 rad by 500 ms; a hold would not move) |
| `damp_kd` | **12.5** N m s/rad, every body joint | 10-17 | the measured ratio, median 13.2 (10.0-16.5 over 9 joints); no group separates |
| `cmd_delay_s` / `state_delay_s` | 0.0 / 0.0005 | 0-0.002 / 0-0.001 | no latency beyond the joints' own dynamics; the state age at r1rt ([`REPLAY.md`](REPLAY.md)) |

The board's timeout counts BODY commands only: a hand command never refreshes it. The simulated stand's pole position
is fitted so the model hangs at our robot's measured pitch (that halves the hips' gravity load vs a default stand).

## The simulator reproduces the test

The hanging robot on the fitted stand, the measured pose, ON_STAND's gains, the identified leg friction, lockstep, the
command stopping 0-2 ms after the last 500 Hz tick:

| Number | Real | Simulated | Target |
|---|---|---|---|
| motion during a 20 / 100 ms pause | 2.7e-5 / 3.2e-5 rad | 0 / 0 | <= 1e-4 |
| release after the stop | 383.8 ms | 384.0 ms | +-10 ms |
| L / R hip pitch sink | 0.117 / 0.133 rad/s | 0.118 / 0.122 | +-30 % |
| L / R hip pitch motion at 500 ms | 0.0135 / 0.0142 rad | 0.0127 / 0.0130 | +-30 % |
| knees / shoulders / elbow sink | 0.035-0.061 rad/s | ~half as fast | (reported) |

The hips (the largest gravity load, the cleanest measurement) meet the target; the smaller joints sink about half as
fast in the simulator - their friction at low speed is the open item.

## What it means for the controller

- **The guard's deadline is far below the board's hold.** The guard takes over ~10 ms after r1rt's heartbeat stops
  (measured on our robot: 9.8-10.7 ms); the board would hold the robot for ~385 ms anyway. Between 10 ms and 385 ms the
  robot does not notice a core stall.
- **A resume after a long silence must not snap back.** After the release the joints have sagged; r1rt's resume blends
  from the measured pose instead of jumping back to its old target at full stiffness (this is why the board-hold test
  stops at 500 ms: past the release the legs sag for the rest of the pause).
- **A whole-machine stall longer than ~0.4 s is dangerous whatever the software does:** both real-time processes stalled
  together (r1rt and r1guard) leave the robot to the board. In the simulator, with this board and the least stable
  plausible robot, balancing: 0.5 s 1.2-2.6 deg, 1 s 9-19 deg without a fall, 2 s fell 2 / 2.

## Not solid

- One release event per session, one robot, one firmware; the hold's spread across power-ups and firmware versions is
  unknown.
- The damping was seen for ~116 ms only: whether the board stays damping, or later goes to zero torque, is not known.
- kd is one number for every joint (the data do not separate groups).
- What a real board does before its first command after power-up is not known (the simulator damps 0.385 s after it
  starts if nothing commands it).
