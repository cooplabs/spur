# `r1guard` - the real-time core's hot standby

Rung 2 of the never-limp ladder ([`../DESIGN.md`](../DESIGN.md) §4, §5.3). Code: `rt/include/r1rt/guard.hpp`,
`rt/src/guard.cpp`, `rt/src/guard_forensics.cpp`; tests `rt/tests/test_guard.cpp` (C++), `tests/integration/
test_v3_guard*.py` (lockstep, the simulated robot), `tests/integration/test_rt_processes.py` (the real processes);
the Python binding `r1rt_py.Guard`; the lockstep driver `r1sim.v3_guard`.

## What it touches

`r1rt::Guard` is a library with a `cycle()`; the process `r1guard` runs it at 1 kHz on its own CPU with its own DDS
state reader. In the shared segment it touches only:

| Region | Guard | Why |
|---|---|---|
| `rt_heartbeat` | reads every cycle | the core's liveness: stale after 5 writer periods (10 ms at 500 Hz) |
| `mirror` | reads every cycle on standby | the core's last command, raw action, policy phase, lean, targets, gains, arm feed-forward, the ramp in progress: the warm shadow and the takeover's starting point |
| `token` | compare-and-swap on a stale heartbeat; checked before every write | one writer on `rt/lowcmd` |
| `guard_heartbeat` | writes every cycle | the core and `r1d` see that a standby exists and whether its shadow is healthy (the core refuses a takeover without it) |
| `guard_telemetry` | one record per cycle | `r1log` writes it to `/r1guard/telemetry` |

## The cycle (1 kHz)

1. Its own robot state (late after 10 ms, stale 100 ms, lost 1 s - the core's thresholds) and the tilt.
2. The core's heartbeat; on standby also the mirror.
3. **Standby, the warm shadow**: at the policy rate it restores its own policy runner from the mirror and runs one
   step on its own state with a zero base command. A non-finite shadow sets a fault on its heartbeat (nothing else
   changes). In lockstep the shadow equals the core's action bit for bit.
4. **The takeover**: the heartbeat older than 5 writer periods AND the token the core's AND the core in a publishing
   mode -> compare-and-swap; on success it publishes from the same cycle.
5. Active: the command by the mode it took over from (below), the core's hard limits, the token check, write.
6. Its heartbeat, status and telemetry record.

## What it does, by the mode it took over from (it never damps on its own)

| The core was in | The guard publishes |
|---|---|
| MONITOR | nothing (the token was never the core's) |
| TAKEOVER (mirror / blend) | the core's blend continued from where it was (its gains and timing from the mirror), into its own policy |
| BALANCE (incl. the inline rung) | its policy with a zero base command, legs + waist; arms + head HOLD at the core's last targets and gains, with the core's gravity feed-forward |
| STAND_FREEZE | the core's ramp to the stand pose continued from where it was, then hold |
| ON_STAND (or a blend into it) | the core's last command held / the core's gain ramp to the hold continued - holding, never a policy |
| DAMP | damp (the operator's state) |

Its own failures: its own policy output invalid -> STAND_FREEZE from what it was sending; its own state stale -> the
last command again; the token taken away -> it stops writing.

**After a whole-machine stop** (the core silent much longer than 10 ms means the guard was stopped too): it waits for a
live state sample - draining the stop's stale DDS backlog, at most 400 ms - and blends from the measured pose.

## The operator's words while the guard holds the motors

- **Damp** reaches the motors through the guard: an operator's damp request on the stop channel -> the guard damps
  (and stays damp); a non-operator's request is counted and ignored.
- **Reclaim** (`r1 reclaim`): when the core runs again and the operator asks, the core takes the token back and blends
  from exactly what the guard sent; the guard re-arms itself (a second takeover works). There is no automatic hand-back.

## Forensics

At the takeover (plain copies inside the cycle): the core's last heartbeat (sequence, time, mode, faults, board tick,
rate, real-time grant), the mirror's time and policy validity, the token words before / after, the core's pid, the
guard's state age and tilt, the shadow's validity and deviation. Then, after the first write and off the cycle thread,
`/proc/<pid>/task/*/{comm, wchan, syscall, stack}` of the stalled process (Linux; `stack` needs CAP_SYS_ADMIN). One
JSON `takeover` line per takeover.

## Numbers

| Case | Result |
|---|---|
| Standby 2 s next to a live core (lockstep) | publishes nothing; the robot's motion bit-identical to the core alone; the shadow equal to the core's actions |
| A 4.0 s stall of the core while balancing (lockstep) | takeover 11.0 ms after the core's last heartbeat (threshold 10 ms + one 1 kHz cycle); max tilt 0.37 deg; legs always stiff; the core silent afterwards (`NOT_OWNER`) |
| The same with the real processes over DDS on the simulated robot | upright in 23 / 23 runs; takeover 10.2-12.2 ms |
| A NaN on the guard's own network output, then a core stall | takeover in 11 ms -> its first action invalid -> STAND_FREEZE, legs stiff, never damp |
| Our robot on its stand (the guard-takeover test) | 10.7 ms (twice); 9.8 / 10.0 ms another day, the joints moving 0.00005 rad through the takeover (the first day, before the guard kept the arms' gravity feed-forward, the arms moved 0.004-0.020 rad) |
| The guard's cycle (Debug + UBSan, the network at -O2) | 19.7 us mean while active; 0 allocations per cycle incl. the takeover |

## Not solid

- Unitree's `arm_sdk` overlay mode has no guard behaviour (the mode is not built).
- `last_requester` on the stop channel is best effort (the last writer wins).
