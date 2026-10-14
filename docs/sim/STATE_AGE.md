# The state-age limit and the "hold the stand" rung

A balance policy stepping on old robot state is the quiet failure mode: the core keeps running, the commands keep
flowing, and the policy reacts to where the robot WAS. The robustness batch shows the margin: a constant extra 10 ms of
latency passes, 15 ms degrades, 20 ms falls ([`ROBUSTNESS_BATCH.md`](ROBUSTNESS_BATCH.md)). Our robot's state age at r1rt
was p50 0.49 / p99 1.02 / max 2.97 ms - but the question is what happens when the link misbehaves.

## What is built

| Piece | What it does |
|---|---|
| **the content age** (`r1rt::ContentAgeClock`) | how old the newest sample's CONTENT is, from the board's own 1 kHz tick inside every state sample - not from when it was received. The receive age cannot see a LATE receive path (samples queued after a receive-thread stall, a backlog being drained), which hands the core old poses with fresh receive times |
| **the alarm (always on)** | the age judged = max(receive age, content age); above `state_age_limit` -> the fault bit `STATE_OLD` every tick, and an EPISODE counted when it lasts `state_age_sustain` - in every mode, whatever the rung. `r1 status` shows the content age, its max, the old ticks and the episodes |
| **the rung** (`stale_rung`) | **0 = off** (the default): the policy runs on the old state until it is 100 ms stale, then the core resends the last balanced command; **1 = hold the stand**: a balancing robot whose state stayed old for the sustain time enters STAND_FREEZE (the 0.4 s ramp to a stand pose at the stand gains; it needs no state), back to balance only by the operator; **2** = hold the stand, then return to the policy by itself once the state is live again (not recommended: below) |
| **the guard** | r1guard applies the same rule to its OWN state while it balances after a takeover |
| **config** | `state_age_limit_ns` (10 ms), `state_age_sustain_ns` (the robot profile: 100 ms), `stale_return_live_ns` (0.5 s), `board_tick_hz` (1000), `stale_rung` (0) - every value in the robot profile with its source |
| **the sweep** | our sweep runner (lockstep, the C++ core on the identified variants, a late sample arriving with a fresh receive stamp and its old board tick, as on a robot) is **not in this release** (its cases are built on our own task scenes). To try a late state yourself: `python examples/robustness_batch.py --faults state_delay` (the state 40 ms late while a push lands) or the viewer's `D` key / `python -m r1sim.viewer.batch --keys 'D@4.0'` ([`VIEWER.md`](VIEWER.md)); the rung is the core's config (`stale_rung`, below) |

## The sweep (the identified model; 3 runs per cell; falls / runs)

| Event | the rung off | hold the stand at 10 ms for 100 ms (the recommendation) |
|---|---|---|
| state frozen 20-100 ms | no fall, <= 1 deg | no trigger (= the rung off) |
| state frozen 200 ms - 2 s | no fall; 2 s quiet: 1.3-3.4 deg; with a push: up to 12.8 deg, onto whatever is in front | freezes; stands at 3.6-8 deg (the freeze's own stand pose) |
| state 5-10 ms late (3 s) | no fall | no trigger |
| state 15-30 ms late (3 s) | no fall, <= 1 deg (also with a push) | freezes (stands at 3.7-6.9 deg) |
| state 40 / 50 ms late + a 5 N s push | **falls 4 / 6, 6 / 6** | **0 falls**, 4.2-5.8 deg |

Over 126 runs per response: the rung off 6 falls; hold the stand at 10 ms / 100 ms: 0 falls (also 0 at 10 ms / 60 ms
and 20 ms); at a 40 ms limit it misses the 40 ms case (3 falls). **The automatic return (rung 2) is unsafe** on a long
steady delay: after 2 s of a stable offset the content clock re-bases (a steady backlog reads as a new time base), the
age reads fresh, rung 2 returns to the policy while the state is still 15-50 ms late, and the policy staggers (8-18 deg,
steps of 5-74 cm, 1 / 3 falls at 50 ms + a push). The operator's return after a stale freeze needs the same care: "the
age reads fresh again" is not proof that the receive path is healthy.

## Our decision, and yours

On our robot the rung stays **off** until the alarm has logged the content age through real sessions with no false
episodes (the first balancing day: state age ~1 ms, 0 episodes); then `stale_rung 1` at 10 ms / 100 ms. It turns
every fall of the sweep into a stand, but it also freezes the robot on 200 ms - 2 s losses and steady 15-30 ms delays
that today's core rides out at ~1 deg: a freeze is safe but ends the task. Arm it with your own robot's numbers:
watch `r1 status`'s content age and `STATE_OLD` episodes over your sessions first.

## Not solid

- One sweep on one identified model; the floor contact is not identified.
- The content clock's re-base after 2 s of a stable offset hides a steady backlog (by design for clock drift).
- The simulated bridge itself can deliver old state under load (a laptop without real-time scheduling); every sweep is
  lockstep for that reason.
