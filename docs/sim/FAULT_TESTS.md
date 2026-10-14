# The fault tests with real processes

Before our robot's first stand session we ran every fault that involves `r1rt`, `r1guard` or ON_STAND against the
REAL processes: one fresh process stack per row (the DDS-bridge simulator + r1rt + r1guard + r1d, the `r1` CLI per word
as the operator types it), ONE fault injected at a known wall-clock time, measured from both sides, the expected
behaviour from the design next to the result. This page is the method and our results.

**The full fault table is not in this release** (our runner builds its rows on our own task scenes). What you can run
yourself with the public tools:

```bash
bash scripts/run.sh python examples/robustness_batch.py --faults none cmd_gap state_delay policy_nan   # pushes x faults, real C++ core
bash scripts/run.sh python -m r1sim.viewer.batch --seeds 0-19 --keys 'S@4.0' --fail-on-fall           # seeded runs, a 0.5 s core stall at 4 s
```

and any single row by hand with the process stack and the bridge's scheduled faults
([`DDS_BRIDGE.md`](DDS_BRIDGE.md): `inject("lowstate_gap", duration_s=1.0)`, `second_writer`, `lowcmd_crc`, ...; a
SIGSTOP of `r1rt` for the guard). The examples: [`../../examples/README.md`](../../examples/README.md); the viewer's
batch mode: [`VIEWER.md`](VIEWER.md).

## How a row is measured

| Side | What |
|---|---|
| the robot (ground truth) | every applied `rt/lowcmd` sample and its writer (`r1rt` / `r1guard` by their DDS user data, the emulated built-in, anything else), the longest gap between two applied commands of ours, the smallest effective leg kp (**never limp** = > 0), dropped bad-CRC samples; the pelvis tilt at 200 Hz (**fell** = > 30 deg) |
| the core | its mode, fault bits, the token owner; the guard's mode and faults - sampled at 200 Hz from r1rt's shared memory (read-only) |
| the operator | r1d's typed results to the words typed |

Configurations: **balance** = standing in BALANCE after a takeover from the emulated built-in (the stall rows use the
least stable plausible robot: no leg stiction), 2 s before the fault; **on_stand** = hanging on the modelled stand in
ON_STAND; **monitor** = nothing taken over. Verdicts: **PASS** = as designed; **FAIL** = the design's behaviour broken
(or a physical limit - said so); **GAP** = a behaviour the design asks for is not built; **PENDING** = an injection
this build lacks. **A row whose robot had already fallen before the fault, or whose simulator fell behind real time,
is INVALID** - the table refuses to judge the controller on a starved simulator.

## The results (a laptop under load; one run per row unless counted)

| Fault | Config | Result | Verdict |
|---|---|---|---|
| state gap 0.1 / 1 s | balance | LATE / STALE / LOST flagged, r1rt keeps writing (board gap <= 4.5 ms), legs stiff, tilt <= 3.3 deg, back to normal | PASS |
| state gap 1.5 / 2 / 3 s (3 seeds each) | balance | fell 2 / 3, 2 / 3, 3 / 3 - after 100 ms of stale state r1rt resends the last policy command, which does not keep a balancing robot up for seconds | FAIL: a design limit. The opt-in "hold the stand" rung turns these into a freeze ([`STATE_AGE.md`](STATE_AGE.md)) |
| state gap 1 s | on_stand | held stiff, back to ON_STAND | PASS |
| state drops 50 % (3 s) / delay 5, 10 ms | balance | LATE only; tilt <= 1.3 deg | PASS |
| state delay 20 ms (3 s) | balance | no fall, but a sway of 12-18 deg | FAIL: a 20 ms state latency is a broken link (ours measured p99 1 ms) |
| state with a bad CRC 30 % | balance | counted, not used, tilt 0.5 deg | PASS |
| a foreign writer with a wrong CRC | balance | every sample dropped by the board | PASS |
| commands lost 50 % / corrupted 30 % before the board | balance | board gaps <= 30 ms, tilt 0.5 deg | PASS |
| the link down both ways 0.5 s | balance | STALE flagged, tilt 1.7 deg, recovered | PASS |
| a second valid writer (a stiff stand, 1 s) | balance | an alarm, r1rt kept writing, tilt 1.25 deg | PASS |
| a foreign DAMP writer (0.5 s) | balance | an alarm; the board applied the foreign damp interleaved with r1rt's commands: a sway to 5.8 deg, no fall | PASS (prevention is the ownership token for our own processes; a foreign program on the robot is the operator's to stop) |
| the policy's action NaN | balance | STAND_FREEZE 12 ms after the NaN, the legs stiff, stays frozen until the operator | PASS |
| r1d killed mid arm jog / mid knee jog | balance / on_stand | the joint stopped (the stream's time-to-live), balance / the hold kept | PASS |
| the hand service dead / a hand's state frozen | balance | reported (a frozen state = invalid after 0.1 s), body unaffected | PASS |
| the workstation link drops mid-jog | balance | the jog ended at its time-to-live, r1d up | PASS |
| a motor error word on a knee / a shoulder | balance | an alarm; an arm group with an error HOLDs stiff and refuses records until `r1 resume` | PASS |
| over-temperature (a leg joint to 75 C) | balance | a warning (>= 60 C) in `r1 status` | PASS |
| a takeover with a joint at 70 C | monitor | refused (HOT), nothing published | PASS |
| **r1rt paused 4.018 s** (the length of the stall that felled the previous controller) | balance | **the guard in 10.7 ms**, one writer, tilt 0.47 deg | PASS |
| r1rt paused 0.5 s x 5 seeds + 4.018 s x 3 seeds | balance | 8 / 8 upright, max tilt 0.49-0.55 deg, board gap 9.8-11.5 ms | PASS |
| r1rt killed (`kill -9`) | balance / on_stand | the guard took over (10.4 ms), one writer, never limp | PASS |
| r1rt paused 3 s | on_stand | the guard holds the core's last command (legs kp 200), gap 10.2 ms | PASS |
| r1guard killed | balance | GUARD_ABSENT after 52 ms (an alarm), r1rt kept publishing | PASS |
| SIGTERM to r1rt in BALANCE | balance | refused, still publishing | PASS |
| the operator's damp while the guard holds the token | on_stand / balance | a non-operator damp is ignored; the operator's `r1 damp` reaches the motors through the guard (kp 0) | PASS |
| `r1 reclaim` after a takeover from BALANCE | balance | refused: the guard keeps the token and balances (there is no hand-back from a balancing guard) | PASS |

## Whole-controller stalls and a starved simulator

A stall of BOTH real-time processes (r1rt and r1guard stopped together - what a whole-machine stall would do, which
software on the same machine cannot cover) leaves the robot with the board alone. With the identified board (holds
0.385 s, then damps) and the least stable plausible robot: 100-250 ms ridden through (< 0.6 deg), 500 ms 1.2-2.6 deg,
1 s 9-19 deg without a fall (4 / 4), **2 s fell 2 / 2** ([`BOARD_MODEL.md`](BOARD_MODEL.md)). This is why the backpack's
real-time set-up and the "never block" rules matter: the software can survive the core stalling, not the machine.

Under heavy load on a laptop, every fall during normal balance came with the SIMULATOR starved, not the controller:
the Python bridge's state stream dropped to 62-440 Hz with 100-356 ms gaps while r1rt kept its 2.0 ms ticks. A real
board streams at ~1 kHz in hardware, so these falls are a simulator artefact - but they showed a real property: between
the core's 10 ms "late" count and its 100 ms "stale" resend, the policy keeps stepping on old state (the reason for the
state-age limit, [`STATE_AGE.md`](STATE_AGE.md)). The guard's 10 ms threshold also gives a takeover from a core stop of
~9 ms (the heartbeat is up to one writer period old when a stop starts): safe (stiff, balancing), but one-way while
balancing.

## Not solid (said plainly)

1. A laptop without real-time scheduling: every gap / takeover time is a loaded laptop's, not the robot's (our robot:
   the guard in 9.8-10.7 ms).
2. One run per row by default: an outcome near its edge changes from run to run (we counted the edge rows over 3 seeds).
3. The bridge's board model is identified from one release event on one robot.
