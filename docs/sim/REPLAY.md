# Replaying the stand recordings: joint dynamics, latency, the IMU mounting

The stand recordings ([`../IDENTIFICATION.md`](../IDENTIFICATION.md)) are replayed in the simulator command for
command, so the simulated joints can be compared with the real ones sample for sample. Code: `sim/r1sim/tier_b/`
(`data.py` the record windows, `replay.py` joint dynamics + latency, `h1_imu.py` the IMU mounting); an excerpt of a
real knee step for the tests is in `tests/assets/hw1/`.

```bash
bash scripts/run.sh python -m r1sim.tier_b.replay --session <a session folder from r1 shutdown> --out out/replay [--config cfg.json] [--quick]
bash scripts/run.sh python -m r1sim.tier_b.h1_imu --session <the same folder> [--sim-config cfg.json]
```

The dataset of our robot's stand recordings ([`../DATASET.md`](../DATASET.md)) has the session layout these tools read.

## How the replay works

The real robot hung on its stand while r1rt commanded every joint at 500 Hz. The replay builds the same hanging robot
(the bridge's stand, the real profile, lockstep, sensors off), poses it at the MEASURED joint angles of the window's
first sample, settles it under the command r1rt was sending then, then feeds the simulated board the LOGGED command of
every tick (q_cmd, dq_cmd, kp, kd, tau_ff for all 26 joints) and samples the simulator at the same ticks. Per window,
for the moving joint: the normalised RMS (RMS(sim - real) / RMS(real - mean)), the delay to 10 % of each step, the gain /
lag of q against the command at the chirp's frequencies (a lock-in), the RMS of the torque against the real `tau_est`.
Targets: nRMS <= 10 %, delay within +-1 ms, phase within +-5 deg at 2 / 5 Hz, torque within +-15 %.

## Latency: the command path adds (almost) nothing

The real joints reach 10 % of a 0.05 rad step 16-44 ms after the command, depending on the joint (head yaw 16-18, waist
18-22, knee 18-24, hip 20-26, ankle 22-30, elbow 28-32, shoulder pitch 36-44 ms). **Replaying the same commands, the
simulated joints' own dynamics give the same delays with NO added latency**: on the joints the stand does not disturb,
the simulator's median delay minus the real one is +0.4 ms with 0 ms added (+1.5 ms with 1 ms). The spread by joint
is each joint's kp / inertia / friction, not a communication delay. So `models/driver.yaml`: command delay 0.0
(range 0-2 ms), state delay 0.0005 s (the measured state age p50 0.49 ms). A test pins it: 6 ms more puts the knee
outside the target.

## Joint dynamics: what the steps and chirps said about the model

| Parameter | Before (community-reported values) | Identified on our R1 | Evidence |
|---|---|---|---|
| armature, hips / knees / waist | 0.05 | **0.008** (0.005-0.011) | knee steps: the torque swing 1.9-2.0 x the real at 0.05, 1.03-1.10 x at 0.008 |
| armature, arms / head | 0.01 | **<= 0.0025** | shoulder pitch steps |
| friction, shoulder pitch / roll | 2.5 N m | **<= 0.05-0.1** | the real 0.05 rad steps overshoot 8-24 % at kp 60 - impossible with 2.5 N m |
| friction, shoulder yaw / elbow / wrist / head | 0.2 | **~0.03** (head pitch ~0.1) | the hysteresis of the step plateaus |
| damping, elbows / head pitch | 0.05 | **~1 N m s/rad** | the elbow lags 13-19 deg at 1 Hz (the model 4 deg): viscous, not Coulomb |
| friction, legs / waist | 2.5 (ankles 1.5) | hips 0.35-0.72, knees 0.43-0.63, ankle pitch 2.1-2.5, waist 0.41-0.52 | the friction ramps ([`../IDENTIFICATION.md`](../IDENTIFICATION.md)) |

**The stand rocks.** At every hip step our robot rocked on its stand at ~2-3 Hz; a rigid stand in the simulator could
not reproduce it, so the replay gives the pole a compliant slide (fitted: 35,000 N/m, 150 N s/m - fitted on the same
recordings it then improves; the stand is not the robot, but say it).

| Window (examples) | nRMS | delay real / sim (ms) | torque sim / real |
|---|---|---|---|
| left / right knee step | 2.1 % / 2.1 % | 20.1 / 20.2, 19.5 / 19.7 | 1.06 / 1.10 |
| left / right hip pitch step | 7.4 % / 7.1 % | 22.5 / 27.0, 21.7 / 26.5 | 1.00 / 1.06 |
| waist yaw step / chirp | 2.4 % / 8.4 % | 19.2 / 19.5 | 1.18 / 1.21 |
| knee chirps (phase at 2 / 5 Hz within 4 / 2 deg) | 10.3-10.6 % | - | 1.10-1.22 |
| ankle pitch steps | 40-46 % | - | 0.3-0.4 |
| elbow / shoulder pitch steps | 9-18 % | within 1-4 ms | 0.57-1.06 |

**Overall the joint-dynamics target is met on the knee steps and partly on the hips and the waist yaw, not on the ankle
pitch, the arms or the waist roll.** The ankle is a linkage on the R1 (moving the ankle pitch moved the roll), which a
serial joint with dry friction does not carry; the arms want low friction on steps and viscous damping on chirps (a
structural difference: the motor-side encoder with a compliant gear is a candidate); the waist roll needs the stand's
roll compliance (not modelled). The identified values make almost every row better than the model as built (shoulder
pitch 64 -> 17 % nRMS and a 104 ms delay error -> 2 ms; the knee chirps 18 -> 10.5 %), and they are in the `real`
profile. For balancing, what mattered was the legs' friction and armature: the robustness batch passes with them
([`ROBUSTNESS_BATCH.md`](ROBUSTNESS_BATCH.md)).

## The IMU-to-pelvis rotation from the waist axes (`h1_imu`)

The IMU sits on the pelvis; the robot hangs by the torso, so moving the waist joints rotates the pelvis about the waist
ROLL and YAW axes - two non-parallel axes the gyro sees in its own frame. The fit: gyro = C_roll qd_roll + C_yaw qd_yaw +
bias, normalised, rotated (Kabsch) onto the model's axes; standard errors by a block bootstrap. The same estimator on
the simulator's replay of the same commands (whose offset is known) measures the method's own bias. This needs no level
and no reference face - only slow waist sweeps on the stand (+-0.2 rad roll, +-0.3 rad yaw).

| | roll (deg) | pitch (deg) | yaw (deg) |
|---|---|---|---|
| first stand session (step / chirp / ramp data) | 0.42 +- 0.08 | 1.56 +- 0.48 | 0.12 +- 0.35 |
| the method's bias on the simulator | +0.15 | +0.55 | +0.02 |
| second stand session (dedicated waist sweeps) | **0.56 +- 0.07** | **0.82 +- 0.18** | **-0.39 +- 0.24** |
| the model's offset | 0.50 | 0.96 | 0 |

Within the method's accuracy the model's offset is confirmed; the yaw component was unobservable before.

## Not solid

- The stand's compliance is fitted on the same recordings; its roll compliance is not modelled.
- The armature / damping values come from a coarse grid per joint family, one parameter at a time.
- The ankles' and the arms' structure is not captured by the model.
- Sensors were off in the replay (the real traces carry their noise; nRMS includes it - small on the steps).
