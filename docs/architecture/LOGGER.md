# `r1log` - the logger

`python -m r1mc3.log` (`python/r1mc3/log/`): a low-priority Python process (nice 10) that drains the core's and the
guard's rings into MCAP files, samples the hand-contact and heartbeat seqlocks, records events, and keeps a flight
recorder. It only pops single-consumer rings and reads seqlocks: the real-time core never waits on it. A logger that
falls behind loses records, counted.

## Reading the segment

Through `r1rt_py.SegmentView`, the same C++ ring and seqlock code the core uses (a plain memory view would have no
memory barriers: on the Orin's weakly ordered cores a reader could see a published head before the slot's bytes). At
start the logger compares every record's size and every field's offset with the C++ build and refuses to run on any
difference.

## Files and topics

`<out>/r1log_<UTC start>_<NNN>.mcap`, zstd chunks; a new file past a size or a duration. Every file starts with
metadata (source, host, layout / ABI versions, the wall clock next to the monotonic clock) and ends with a summary
(records, drops and sequence gaps per stream).

| Topic | Content | Rate |
|---|---|---|
| `/r1rt/telemetry` | every writer tick: times, board tick, lateness, iteration time, faults, clamps, mode, token owner, guard event, q / dq / tau_est / the command, IMU, temperatures, motor state | 500 Hz |
| `/r1rt/policy` | every policy step: observation, action, inline flag | 50 Hz |
| `/r1guard/telemetry` | every guard cycle: heartbeat age, faults, modes, published command, shadow, tilt | 1 kHz |
| `/r1hands/contact`, `/r1guard/heartbeat` | sampled when written | <= drain rate |
| `/r1log/event` | mode and owner changes, fault bits rising / clearing, stop requests, dump requests (JSON) | on change |
| `/r1log/stats` | records, drops, gaps, files, flight dumps (JSON) | 1 Hz |

The struct topics carry the record's raw C bytes; their schema (encoding `r1rt-struct`) is a JSON description of
every field, so an old log stays readable after the layout changes (`r1mc3.log.read`). Times are the records' own
monotonic times.

## Drops

The rings never block the producer: a full ring refuses the record and counts it. The logger reports both the ring's
drop counter and the holes it sees in the writer's tick numbers. The telemetry ring holds about 8 s at 500 Hz; the
default drain period (20 ms) leaves a large margin.

## The flight recorder

The last 30 s of every stream in memory. A trigger writes `flight_<UTC>_<NNN>_<reason>.mcap` with 30 s before and
5 s after; triggers arriving meanwhile join the same dump; a category that fired is held off for 10 s.

| Trigger (rising edge) | Reason name |
|---|---|
| a fault bit appears (state stale / lost, CRC, tilt stop, fall tilt, motor error, policy late / invalid, loop gap, not owner) | `fault:<bits>` |
| the mode enters GUARDED / STAND_FREEZE / DAMP | `mode:<MODE>` |
| the token goes to the guard | `takeover` |
| the absolute IMU tilt passes 10 deg (re-armed below 5) | `tilt` |
| an operator stop / halt / damp | `operator:<level>` |
| a request (SIGUSR1 to the logger) | `request:<reason>` |

```bash
python -m r1mc3.log run --out DIR [--segment /r1mc3.rt] [--flight-s 30] [--post-s 5] [--tilt-deg 10]
python -m r1mc3.log info FILE.mcap ...     # per topic: records, time span, sequence gaps; the metadata
kill -USR1 <pid>                           # a flight-recorder dump now
```

## The session save

`r1 shutdown` (accepted only in MONITOR or DAMP) lets the logger finish its final drain, then puts the session's files
(the logs, the recordings, `r1d`'s request journal) into a new folder with checksums.
