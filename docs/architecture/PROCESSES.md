# `r1rt` and `r1guard` as processes

The libraries (`r1rt::Core` behind the C step ABI, `r1rt::Guard`) do the work; the processes only schedule them, feed
them DDS and report. Code: `rt/apps/{r1rt_main,r1guard_main}.cpp`, `rt/src/{dds_io,rt_thread,proc_config}.cpp`,
`rt/proc.cmake` (built with `-DR1RT_WITH_CYCLONEDDS=ON`: `scripts/ci.sh dds` -> `<build>/rt_dds/apps/{r1rt,r1guard}`),
`python/r1mc3/rtconfig.py`.

## Command lines

```
python -m r1mc3.rtconfig --out r1rt.cfg            # the config file (one source of numbers with the simulator)
r1rt    --config r1rt.cfg --shm-name /r1mc3.rt --domain N [--rate-hz 500] [--policy-weights PATH]
        [--stand-session] [--fault-writer-stall AT_S:DUR_S] [--fault-policy-nan AT_S:DUR_S] [--check-config]
r1guard --config r1rt.cfg --shm-name /r1mc3.rt --domain N [--observe-only] [--attach-timeout-s 10]
```

The DDS network comes from CycloneDDS's own `CYCLONEDDS_URI` (on the robot: the motor NIC; in the simulator and tests:
the loopback interface).

## Threads

| Process | Thread | Rate | Scheduling asked for | Does |
|---|---|---|---|---|
| r1rt | writer | 500 Hz, absolute deadlines | SCHED_FIFO 90, CPU 7 | the latest state (seqlock) -> `r1rt_core_step` (mailbox, checks, policy action or inline rung, executor, limits, heartbeat, mirror, telemetry) -> the token re-checked -> `rt/lowcmd` |
| r1rt | policy | 50 Hz | SCHED_FIFO 80, CPU 6 | `r1rt_core_policy_step` (a race-free hand-back to the writer) |
| r1rt | DDS receive | event | SCHED_FIFO 85, CPU 6 | `rt/lowstate` -> CRC -> joints -> seqlock; Unitree's own `rt/lowcmd` (foreign writers only) for the takeover's mirror |
| r1rt | main | 50 Hz poll | normal | signals, the ready / status / event lines |
| r1guard | cycle | 1 kHz | SCHED_FIFO 88, CPU 5 | `Guard::cycle` on its own state; while active writes `rt/lowcmd` at the core's rate, the token checked before each write |
| r1guard | DDS receive | event | SCHED_FIFO 85, CPU 5 | its OWN `rt/lowstate` reader |
| r1guard | main | 200 Hz poll | normal | the `/proc` forensics after the takeover's first write (file I/O off the cycle thread), signals, lines |

Memory is locked at start (`mlockall`). `rt_granted` (in the ready line and every heartbeat) = SCHED_FIFO on the writer
AND the memory locked; the grants come from the systemd unit's limits. Without them every request fails with its errno
in the ready line, and a takeover is refused on robot profiles. On macOS (the simulator only) the periodic threads use
the Mach time-constraint policy and `mach_wait_until`; that is not a real-time guarantee.

## The DDS layer

- `rt/lowstate`: best effort, keep-last 1; Unitree's CRC checked (a bad CRC is passed on flagged, counted and not
  used); CLOCK_MONOTONIC at receipt.
- `rt/lowcmd` out: the R1's slots, CRC, written with a DDS `user_data` tag `r1mc3:r1rt` / `r1mc3:r1guard`, so Unitree's
  own command can be told apart from ours (a sample without the tag is the built-in controller's).
- The real-time processes read through a 256 kB receive buffer (`deploy/config/cyclonedds_rt.xml` layered after the
  shared file): it bounds the stale backlog in front of a fresh sample after a stop.

## Output lines (stdout, one JSON object per line)

| Line | When |
|---|---|
| `{"ready": true, "process": "r1rt", ...}` / `"r1guard"` | once: pid, rt_granted, rates, domain, segment, scheduling per thread (+ errno), policy loaded, stand_session |
| `{"status": "r1rt", ...}` / `"r1guard"` | 1 Hz: mode, ticks, publishes, skipped deadlines, faults, token owner, state rate / bad CRC, foreign writers, lateness and iteration histograms, stops and resume blends, state age |
| `{"event": "takeover", "process": "r1guard", ...}` | after the takeover's first write: the silence, the mode taken over from, the tokens, the tilt, the shadow's validity, the stalled threads (Linux) |
| `{"event": "signal_refused" \| "exit" \| "rearmed" \| "operator_damp" \| ...}` | as they happen |

## Signals, stops, never limp

SIGTERM / SIGINT: `r1rt` exits only in MONITOR or DAMP; in a publishing mode the signal is refused and logged and the
process keeps publishing - the operator stops through `r1 damp` first. `r1guard` exits on a signal only on standby (or
in the operator's damp). Nothing in any exit path publishes damp. SIGKILL cannot be caught: if `r1rt` is killed,
`r1guard` takes over; if `r1guard` is killed, `r1rt` keeps running and reports the guard absent (and refuses a new
takeover). After any stop of the writer longer than 100 ms (SIGSTOP, a whole-machine stall) both processes wait for
live state and blend from the measured pose instead of resuming old targets.

`r1mc3-svc start` starts `r1guard` only after `r1rt`'s ready line and checks both ready lines.

## Fault hooks (simulation and tests)

`--fault-writer-stall AT_S:DUR_S` (a real sleep at the top of the writer loop), `--fault-policy-nan AT_S:DUR_S` (the
policy's output NaN -> rung 3), SIGSTOP / SIGCONT from outside (`r1sim.proc_faults`).

## Measured

| Case | Result |
|---|---|
| The core stopped with SIGSTOP for 4.0 s, fake robot over DDS loopback, 10 runs, a loaded laptop without real-time scheduling | command gap seen by the robot min 10.2, median 10.6, p90 12.5, max 22.6 ms (the max with a full test suite running alongside); `r1rt` silent after SIGCONT; the legs never limp |
| `kill -9` of `r1rt` | gap 10.6-15.7 ms |
| Our robot on its stand (real-time scheduling, the guard-takeover test) | 10.7 ms (twice), 9.8 / 10.0 ms another day |
| Our robot, the writer's own lateness after the CPU set-up | 5 min idle: 99.99 % < 50 us; loaded: 33 ticks 50-100 us, none above; 0 skipped deadlines |
