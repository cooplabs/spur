# Joint directions: what a POSITIVE value does on the real R1

**The rule: a direction is never guessed.** Before an operator types a moving step, the expected motion is said in
words from this table (or computed from the model) - never from memory. Every word below comes from two independent
checks on our R1 (October 2026):

1. **A live check by eye** on the stand: each of the 26 joints stepped +0.10 rad (ankles / wrists / head +0.15, head
   pitch re-run at +0.20) and back, a person watching, compared with what r1mc3's real-profile model predicts for that
   joint at that pose: **26 / 26 match**.
2. **A gravity-sign check** over 179 quasi-static poses: the measured `tau_est` against the model's gravity torque at
   the measured joint angles - **no joint mirrored** (zero opposite-sign windows over 8,874 quasi-static windows).

Where a line is the model's prediction for a situation the check did not watch, it says "model only".

**Re-check after ANY Unitree firmware update, model change (`sim/r1sim/model.py`, the URDF / MJCF), joint-table change
(`sim/r1sim/joints.py`, the SDK order / slots) or on a new robot** - with the tool below. (During our first stand
session our own spoken descriptions had the ELBOW backwards; the robot always moved as commanded - the check is what
caught the words, not the robot.)

## The rules an operator needs

| Joint | A POSITIVE step (on the stand, the torso held) | Notes |
|---|---|---|
| hip pitch, shoulder pitch | the leg / arm swings **BACKWARD** | both sides |
| hip roll, shoulder roll | the leg / arm swings toward the **robot's LEFT** - for BOTH sides (left limb: outward; right limb: inward) | |
| hip yaw | the toes turn toward the robot's LEFT (left foot: toes OUT; right foot: toes IN) | |
| knee | the knee **BENDS** more | no hard stop inside r1mc3's limits toward straight - the knee goes ~6 deg past 0 (to -0.113) |
| ankle pitch | the toes go **DOWN** | both sides |
| ankle roll | the foot's LEFT side goes up (left foot: OUTER edge up; right foot: INNER edge up) | |
| waist roll | the legs swing to the robot's **RIGHT** | model only, the pelvis held (standing): the torso leans toward the robot's RIGHT |
| waist yaw | the pelvis turns to the robot's **RIGHT** (the toes turn right) | model only, the pelvis held (standing): the torso turns toward the robot's LEFT |
| shoulder yaw | the hand turns toward the robot's LEFT (both sides) | |
| **elbow** | **a LARGER value = STRAIGHTER; a SMALLER value = MORE FLEXED (forearm up)** | the elbow's 0 is NOT straight; rough visual estimates on our robot: 0.3 ~ forearm 90 deg out, 0.82 (Unitree's stand-lock preset) ~ 60 deg, 1.40 ~ 25-30 deg flexed (nearly hanging); straight somewhere above ~1.6-1.9 (limit 2.185). These angles are visual estimates, not measurements |
| wrist roll | the front of the hand turns toward the robot's RIGHT (left hand: inward; right hand: outward) | |
| head pitch | the head nods **DOWN** | |
| head yaw | the head turns **LEFT** | |

Body `tau_est` (`rt/lowstate`, N m): **+ = a torque in the +q direction** (gravity check: left hip pitch measured -2.94
vs the model's -2.94 N m). The hands' `tau_est` is raw (1 raw = 1e-6 N m).

## The 26 joints

"Watched point" = what the watcher judged by; "Model" = r1mc3's prediction for +0.10 rad at the check pose (torso
frame: x forward, y the robot's LEFT, z up; `tests/assets/hw1/dir_expect.json`); "Seen" = the watcher's answer;
"Gravity" = the sign check (confirmed = sign agreement over the poses where |model| exceeds the friction band;
correlation = a positive slope where too few poses; not testable = friction >> gravity on the stand).

| Joint | Watched point | Model (+0.10 rad) | Seen | Gravity (poses, agreement, slope) | Verdict |
|---|---|---|---|---|---|
| L hip pitch | sole | back 5.6 cm | the leg swings BACKWARD | confirmed (179, 100 %, 0.90) | confirmed x2 |
| L hip roll | sole | to the robot's LEFT 5.5 cm | OUTWARD | correlation (2, -, 1.11) | confirmed x2 |
| L hip yaw | toe | turns: front toward the robot's LEFT | toes OUT | not testable | visual |
| L knee | sole | back 3.5 cm | bends more | confirmed (7, 100 %, 1.15) | confirmed x2 |
| L ankle pitch | toe | pitches: front DOWN | toes DOWN (re-watched: the first answer "toes inward" was corrected) | not testable | visual |
| L ankle roll | outer edge | rolls: left side UP | OUTER edge up | not testable | visual |
| R hip pitch | sole | back 5.6 cm | BACKWARD | confirmed (179, 100 %, 1.02) | confirmed x2 |
| R hip roll | sole | to the robot's LEFT 5.5 cm | INWARD | correlation (2, -, 0.86) | confirmed x2 |
| R hip yaw | toe | turns: front toward the robot's LEFT | toes INWARD | not testable | visual |
| R knee | sole | back 3.5 cm | bends more | correlation (2, -, 1.29) | confirmed x2 |
| R ankle pitch | toe | pitches: front DOWN | toes DOWN | not testable | visual |
| R ankle roll | outer edge | rolls: left side UP (the outer edge goes down) | INNER edge up | not testable | visual |
| waist roll | left sole | to the robot's RIGHT 7.3 cm | the legs swing to the RIGHT | confirmed (4, 100 %, 1.66) | confirmed x2 |
| waist yaw | left toe | turns: front toward the robot's RIGHT | the pelvis rotates RIGHT | not testable | visual |
| L shoulder pitch | hand | back 4.0 cm | BACKWARD | confirmed (174, 100 %, 1.12) | confirmed x2 |
| L shoulder roll | hand | to the robot's LEFT 3.7 cm | OUTWARD | confirmed (166, 100 %, 0.99) | confirmed x2 |
| L shoulder yaw | front of the hand | turns: front toward the robot's LEFT | the hand toward the LEFT | correlation (2, -, 0.89) | confirmed x2 |
| L elbow | hand | back 1.9 cm (from 1.40: toward hanging) | STRAIGHTENS a little | confirmed (139, 100 %, 1.16) | confirmed x2 |
| L wrist roll | front of the hand | turns: front toward the robot's RIGHT | the front rotates INWARD | not testable | visual |
| R shoulder pitch | hand | back 4.0 cm | BACKWARD | confirmed (175, 100 %, 1.30) | confirmed x2 |
| R shoulder roll | hand | to the robot's LEFT 3.7 cm | INWARD | confirmed (176, 100 %, narrow range) | confirmed x2 |
| R shoulder yaw | front of the hand | turns: front toward the robot's LEFT | turns inward, to the LEFT | not testable | visual |
| R elbow | hand | back 1.9 cm | STRAIGHTENS | confirmed (141, 100 %, 0.88) | confirmed x2 |
| R wrist roll | front of the hand | turns: front toward the robot's RIGHT | turns OUTWARD, to the RIGHT | not testable | visual |
| head pitch | face | pitches: front DOWN | nods DOWN (re-run at +0.20: the first answer "up" was the return) | confirmed (10, 100 %, 0.83) | confirmed x2 |
| head yaw | face | turns: front toward the robot's LEFT | turns LEFT | not testable | visual |

15 / 26 joints are confirmed twice (by eye and by gravity); the other 11 carry no testable gravity load on the stand
(yaws, ankles, wrists) and rest on the live check.

## How to re-check (the tool and the step)

```bash
bash scripts/run.sh python -m r1mc3.hw1.dircheck                     # the table at the check pose; exit 1 if any joint differs
bash scripts/run.sh python -m r1mc3.hw1.dircheck --pose stand_lock    # at Unitree's stand-lock preset (STAND_LOCK.md)
bash scripts/run.sh python -m r1mc3.hw1.dircheck --plan --json out/dircheck.json   # + the step words (printed only)
```

`--plan` prints, for each joint, the expected motion and the two `r1 record run static_pose` words we used: out by the
joint's step in 1.5 s, hold 2 s, back in 1.5 s with `hold_s=1`. **What moves:** one joint, 0.10-0.20 rad, the robot
hanging on its stand in ON_STAND. **How to stop:** `r1 stop` (the group HOLDs where it is). One joint at a time: the
operator types the out-word, the watcher says what moved, compare with the printed expectation, then the back-word.
Any joint that does not match: stop, damp at the session end, nothing else moves until the cause is known (a mirrored
joint would push every controller the wrong way).

The unit test `tests/unit/test_hw1_dircheck.py` pins all of it: the model still predicts the 26 observed answers (at
the check pose and at the stand-lock preset), and a reversed joint is caught. A failure after a model / joint-table
change = re-check on the robot before anything moves.
