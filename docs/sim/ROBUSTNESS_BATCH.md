# The robustness batch: does the balance policy stand on YOUR identified robot?

Before the first balancing on the floor, the question is not "does the policy stand in the simulator" but "does it
stand on the robot model fitted to MY robot, across the uncertainty that remains". We answered it with seeded variants
around the identified values: 100 robots x 4 cases, every run in lockstep with the real C++ core. This page is the
method and our result.

**Our 100-seed batch runner is not in this release** (its cases are built on our own task scenes). Two public tools run
the same kind of check on the generic scenes, with the real C++ core balancing robot variants drawn within the
identified ranges:

```bash
bash scripts/run.sh python examples/robustness_batch.py                     # push sizes x faults -> falls / runs, the tilt, the feet
bash scripts/run.sh python examples/robustness_batch.py --seeds 100 --impulses 4 8 12 --faults none cmd_gap state_delay --workers 4
bash scripts/run.sh python -m r1sim.viewer.batch --seeds nominal,0-99 --duration 20 --pushes 3 --push-n 40,90 --out batch.json
```

([`../../examples/README.md`](../../examples/README.md), [`VIEWER.md`](VIEWER.md).) Every run is lockstep and deterministic
for its seed: a failing seed reproduces exactly and opens in the viewer. Build your own cases (your working pose, your
lean, your table) the same way: the pass rule and the sweep below are the parts worth copying.

## Profiles, cases, the pass rule

| Profile | Robot | Driver | Sensors |
|---|---|---|---|
| **identified** (the verdict) | `variants.sample_params("real", seed)`: masses, CoM, the IMU mounting, the hand mount, foot / floor friction, the arms ... incl. the legs' / waist's friction within the identified ranges (+-30 %, widened to the other side's value) | the board law, a command delay U(0, 2) ms, a state delay U(0, 1) ms, the identified board timeout (0.385 s, then damp kd 12.5) | the sensor model of the seed |
| old_real (reference) | the same seed's variant with the pre-identification friction (legs 1-5, ankles 0.8-3 N m) | board, no delay, no timeout | yes |
| train (reference) | the training's own randomisation | mjlab's `<position>` actuators, 5 ms step | no |

| Case | Duration | What |
|---|---|---|
| stance | 20 s | the policy's default arms, a table in front of the robot |
| arms forward | 15 s | both arms at a forward working pose over a table |
| lean and reach | 15 s | the policy's lean command 4 deg + both arms forward to a far reach (fingertips 0.36 m ahead of the feet, 0.79 m up), no table to lean on |
| pushes | 15 s | stance + 4 torso pushes of 5 N s over 0.2 s (~0.16 m/s for the ~32 kg robot): forward, back, left, right |

Pass per run: never fallen (pelvis tilt <= 30 deg, pelvis z >= 0.6 m); the steady cases: the pelvis tilt from t = 1 s
< 5 deg above the case's own commanded lean and the feet's slip < 20 mm; pushes: the tilt 2 s after the last push
< 5 deg (a recovery step is allowed). **A case passes with >= 99 / 100 runs; the batch passes when every identified
case passes.**

## Our result: 400 / 400 - no retraining needed

The identified profile, 100 seeds x 4 cases (the seeds spanned the ankle pitch friction 1.72-2.95 N m and a command
delay 0.1-1.9 ms): **400 / 400 pass, 0 falls** - on the first identified model and again after the arms' friction /
armature / damping and the legs' armature were identified (the same verdict each time).

| Case | Passed | Pelvis tilt from 1 s p50 / p99 / max (deg) | Feet slip max (mm) |
|---|---|---|---|
| stance | 100 / 100 | 0.75 / 0.86 / 0.87 | 5.6 |
| arms forward | 100 / 100 | 0.75 / 0.87 / 0.88 | 5.1 |
| lean and reach | 100 / 100 | 3.35 / 3.44 / 3.46 (the lean itself) | 7.1 |
| pushes | 100 / 100 | 2.34 / 2.82 / 2.93; 2 s after the last push <= 1.0 | 76 (a recovery step on some seeds) |

The references on the same seeds (old_real, train) also passed every run: the identified friction does not move the
policy's stance. On the real robot the first balancing matched: tilt 0.2-0.5 deg standing, 3.0-3.4 deg in an operator
freeze (the simulator said 2.4).

## How far from the limit: one parameter at a time

| Swept (the rest identified) | Range | Result |
|---|---|---|
| ankle pitch friction | 0.8 ... 6.0 N m | **all pass**: the standing (loaded) friction can be 2.4 x the hanging value before anything changes |
| hips / knees / waist friction | x 0.25, 0.5, 2 | all pass |
| the legs' armature | 0.008 ... 0.05 | all pass |
| **command delay (pure latency beyond the joints' dynamics)** | 0, 7, 10, 15, 20, 30 ms | **the limit**: 0-10 ms all pass; 15 ms passes by the rule but the pushes are recovered by walking 0.7-1.1 m; 20 ms: the arms-forward case fell 2 / 3, the lean case went over its limit 1 / 3, the pushes passed 0 / 3; 30 ms falls in every case |

**The margin is in the latency, not the friction.** Our robot added no measurable command latency beyond the joints'
own dynamics (0-1 ms) and its state age was p99 ~1 ms, far inside the ~10 ms margin. Anything that adds a constant
delay on a real robot - a slower DDS path, a policy thread scheduled late every step - eats this margin.

## When a retrain WOULD be needed

| Trigger (from your robot) | What changes in the training ([`../POLICY.md`](../POLICY.md)) |
|---|---|
| an end-to-end latency >= ~10 ms beyond the joints' dynamics, or sway / walking off after small pushes | randomise an action / observation delay of 0-20 ms (the training had none) |
| a standing ankle friction far above 6 N m, or the robot rocking on its ankles | the joint-friction randomisation per group (ankles 1-6 N m; the training drew 0.4-2.4) |
| the identified armature shown to matter on the floor | the armature randomisation down to ~0.004 |
| the lean with the arms forward unstable on the real robot | weight the arm term's envelope toward the forward reach with the lean |

## Not solid

- The friction was identified HANGING; the sweep shows the policy indifferent to the ankle friction up to 6 N m, but the
  standing friction was not measured.
- The delay margin is a simulator number on a model whose joint dynamics replay the stand recordings; the floor contact
  is not identified.
- The pushes are small (5 N s); the training pushed to 0.5 m/s. Larger pushes were not part of the batch.
- The arms are fixed poses: arm motion while leaned is not simulated here.
