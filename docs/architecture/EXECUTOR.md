# The motion executor (`r1rt::UpperBodyExecutor`)

Inside the core's writer loop ([`../DESIGN.md`](../DESIGN.md) §3.3). Code: `rt/include/r1rt/executor.hpp`,
`rt/src/executor.cpp`; tests `rt/tests/test_executor.cpp`. One instance drives the arms and the head (joints 14-25 in
Unitree's SDK order); on the stand (ON_STAND) a second instance drives the legs and the waist (joints 0-13).

## Sources (one per group at a time)

| Source | Behaviour | Ends |
|---|---|---|
| HOLD | holds its target; a HOLD record moves the target at bounded velocity and deceleration (no overshoot); hold gains + gravity feed-forward | never |
| TRAJECTORY | quintic Hermite through the knots (q, dq, ddq match at every knot), sampled at the writer's tick; consecutive records extend it | the end; the deadman (no keepalive renewal); a guard; the tilt stop; an operator stop |
| STREAM | the target follows the setpoint, each tick's motion capped; held, not extrapolated, between records | the time-to-live (<= 300 ms after issue); the tilt stop; an operator stop |
| FREEDRIVE | operator only, timed (<= 120 s), kp 0, small kd, gravity feed-forward | the time; stale feed-forward; an operator stop |

## One stop profile

Every stop (deadman, time-to-live, guard, operator stop, starved) uses one profile per joint: v(t) = v0 (1 - 3 s^2 +
2 s^3), s = t / T, from the target's own velocity - continuous velocity and acceleration at both ends. T = 0.15 s
(0.02 s for guards), longer if the deceleration limit needs it; a joint that would pass its limit stops sooner, and
the position clamp is the last word. Then HOLD at the reached pose with the hold gains (ramped over 0.1 s).

## Guards (per trajectory) and the tilt stop

| Guard | Rule |
|---|---|
| lag | per joint, along its commanded motion only: the tracking error beyond friction / kp + a velocity term + a margin, after 0.2 s, lasting 0.06 s (the arm is blocked) |
| torque | the torque estimate's rise above its mean over the first 0.15 s, lasting 0.04 s - only on joints with a calibrated torque estimate (none of the R1's body joints yet, so it is refused on the robot profile) |
| contact | the chosen hand pads rose after the path began (any or all), lasting 0.03 s |
| tilt | (always, arms) the body tilt rises 4 deg over its value when the motion began |

A guard ends the motion with the guard stop and reports which guard and the measured pose (`stopped_by.code`:
`GUARD_LAG`, `GUARD_TORQUE`, `GUARD_CONTACT`; `DEADMAN`, `STARVED`, `RESUME_AFTER_SILENCE`).

## Limits and refusals

On every output: q within [q_min, q_max] (the model's limits minus 0.05 rad), the target's speed <= dq_max (2 rad/s)
- after a loop gap at most two periods' worth, no jump -, gains within bounds and kp > 0 outside FREEDRIVE. Clamps are
counted. Records are refused (with the reason) for: a bad group / source / joint count, a non-finite value, gains out
of bounds, a stop epoch older than the last operator stop (a late command after a stop cannot restart a motion), a
STREAM time-to-live <= 0 or > 300 ms or already expired, bad knots, a segment that does not continue the active
trajectory, more than 512 knots, a start more than 0.05 rad from the target, a torque guard on an uncalibrated joint.

## Real time

`init()` allocates the knot buffers; `submit`, `keepalive`, `stop`, `step` and the rest allocate nothing (tested with a
counting operator new), throw nothing and run in constant time. Deterministic: a pure function of its calls (two
random sessions with the same seed are bit-identical).

## In the core

Every tick in a mode that publishes the upper body (TAKEOVER blend, BALANCE, STAND_FREEZE, ON_STAND): the stop epoch
from the stop channel, up to 16 records from `r1d`'s ring, the keepalive stamps, the last good hand contact bits, the
arm gravity feed-forward, `step`, then the telemetry fields (guard event, clamps). The core's own stops (an operator
stop, entering STAND_FREEZE) go through the stop channel, so every client's later commands carry the right epoch.
Every tick the core publishes the executor's per-group status, from which `r1d` reports how a goal ended.
