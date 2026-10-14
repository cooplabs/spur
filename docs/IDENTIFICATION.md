# Identifying YOUR robot: the stand protocols, the fits, and our R1's numbers

The simulator ships with our R1's identified numbers. Yours will differ in the details - joint friction varies from
robot to robot, a board's firmware decides what happens when commands stop, a mount sits a little differently. This page
is how we measured ours on the stand, which tool fits what, and what we found, so you can do the same and replace our
numbers with yours before trusting a simulated margin.

**Where it fits in the bring-up:** after the read-only checks, with the robot hanging on its stand in ON_STAND
([`hardware/BRING_UP.md`](hardware/BRING_UP.md) section 7). Every recording moves ONE joint (or nothing) and is small by
default. Our stand recordings are published as a dataset ([`DATASET.md`](DATASET.md)): the tools below read it as they
read your own sessions.

## How a recording works

Every recording is a plan first, then a run, through r1d (the operator's `r1` CLI):

```bash
r1 record plan step joint=left_knee amplitude_rad=0.05 hold_s=2 repeats=2      # the setpoints, the limits, what moves
r1 record run  step joint=left_knee amplitude_rad=0.05 hold_s=2 repeats=2 --yes
```

The plan refuses BEFORE anything moves if it would leave a joint's limits (with a 0.05 rad margin), exceed the
executor's speed cap (2 rad/s), or move more than one joint where one is the rule; on a refusal it prints the range
against the limits. The run streams the setpoints with a short time-to-live: `r1 stop` (or a dead client) ends in a
HOLD, stiff. While it runs, r1log records the core's telemetry every 2 ms tick (per joint `q`, `dq`, `tau_est`, the
command `q_cmd` / `dq_cmd` / `kp` / `kd` / `tau_ff`) and the hands at 100 Hz into MCAP files; the run writes a manifest
(the plan, the times, its r1log files). `r1 shutdown` saves all of it into the session folder with checksums.

| Kind | What moves | What it is for |
|---|---|---|
| `step` | one joint: hold, +A, hold, base, hold, -A, ... | joint dynamics (the step response), command-to-motion delay |
| `chirp` | one joint: a sine sweep f0 -> f1 with 0.5 s fades | joint dynamics: gain / phase lag at 1-5 Hz |
| `ramp` | one joint under a SOFT servo: +A, -A, 0 at two speeds | joint friction (stiction) and, with a payload, static torques |
| `static_pose` | one group to a pose, hold (with a declared payload) | static torques against the model's gravity |
| `imu_static` | nothing | IMU statics, the DDS latency, a quiet baseline |
| `pad_press` | nothing (a person presses known weights; the hand holds open) | the pads' response |
| `finger_hold` | one finger closes with a constant torque (kp 0) | the finger motors' heating |
| `flat_field` / `charuco` / `depth_targets` | nothing (the head holds a pose) | the camera ([`hardware/CAMERA_CALIBRATION.md`](hardware/CAMERA_CALIBRATION.md)) |

## 1. Joint friction - slow soft-servo ramps (legs and waist)

**Why a ramp of the TARGET at low kp, not a ramp of torque:** a soft position servo turns a slow target ramp into a
torque ramp (the joint sticks while kp x the lag grows through the friction band, then follows). It is bounded - the
target never leaves hold +- A, the torque never exceeds kp x the lag - and never limp (kp > 0). A pure torque ramp on a
stuck joint becomes a run-away the moment it breaks free.

| Joint (left, then right; the waist last) | kp_test (N m/rad) | A (rad) | Speeds (rad/s) | Duration |
|---|---|---|---|---|
| hip pitch / roll / yaw | 40 | 0.25 / 0.15 / 0.20 | 0.1, 0.3 | 16-21 s |
| knee | 40 | 0.25 | 0.1, 0.3 | 21 s |
| ankle pitch / roll | 25 | 0.20 / 0.12 | 0.1, 0.3 | 14-18 s |
| waist roll | 150 (the torso is an inverted pendulum) | 0.10 | 0.1, 0.3 | 13 s |
| waist yaw (last: it turns the pelvis and legs under the hanging torso) | 40 | 0.20 | 0.1, 0.3 | 18 s |

`r1 record plan ramp joint=<j>` shows the defaults; `r1 record run ramp joint=<j> --yes` runs it. From the hanging pose a
knee hangs nearly straight: if the plan refuses (`LIMIT`), pre-pose the knee to ~0.40 rad first, or use a smaller
amplitude (>= 0.10 rad). All 14 joints: ~4.5 min of ramps.

**The fit** (`r1sim.sysid.stiction`): the torque from the logged PD law (or `tau_est`), the velocity over +-50 ms; the
steady window of each segment starts at the BREAKAWAY + 0.25 s; per speed the friction h = (tau_up - tau_down) / 2 from
tau = P4(q) + h sign(v) (gravity as a quartic in q, so a steep gravity term does not leak into the friction); the damping
b only when h RISES with speed (otherwise the prior). Ramps recorded through r1d have a ~100 Hz staircase setpoint, so
fit them from their manifest:

```bash
bash scripts/run.sh python -m r1sim.sysid stiction --log <r1log file> --provenance          # a models/provenance.yaml snippet
python -c "from r1sim.sysid import stiction as s; print(s.fit_joint(s.load_ramp('<manifest.json>'), 'left_knee'))"
```

**Our R1** (N m; Fs = breakaway, h at 0.1 / 0.3 rad/s, Fc = the value used):

| Joint | Fs | h(0.1) | h(0.3) | Fc | reliable |
|---|---|---|---|---|---|
| L / R hip pitch | 1.08 / 0.66 | 0.619 / 0.560 | 0.362 / 0.209 | 0.614 / 0.555 | yes |
| L / R hip roll | 1.03 / 0.79 | 0.395 / 0.358 | 0.109 / 0.106 | 0.391 / 0.354 | yes |
| L / R hip yaw | 0.73 / 0.78 | 0.725 / 0.494 | 0.597 / 0.409 | 0.720 / 0.489 | yes |
| L / R knee | 0.95 / 0.70 | 0.636 / 0.430 | 0.564 / 0.330 | 0.631 / 0.425 | yes |
| L / R ankle pitch | 2.22 / 1.88 | - / 2.49 | 2.531 / 2.089 | 2.516 / 2.074 | yes |
| L / R ankle roll | 1.95 / 1.29 | 1.26 / 1.19 | 1.46 / 1.36 | 1.445 / 1.181 | **no** |
| waist roll / yaw | - / 0.81 | 0.520 / 0.415 | - / 0.364 | 0.516 / 0.410 | yes |

The friction FALLS with speed on every joint with two valid speeds (Stribeck): damping is not identifiable from this
protocol (the prior 0.05 is kept). Compared with the community-reported values the model used before (2.5 N m on hips /
knees / waist, 1.5 on the ankles): the hips / knees / waist are 4-6 x lower, the ankle pitches 1.4-1.7 x higher - and
3-4 x the balance policy's TRAINING value. The robustness batch then showed the policy indifferent to it
([`sim/ROBUSTNESS_BATCH.md`](sim/ROBUSTNESS_BATCH.md)). Measured hanging: the friction under the standing load is
not known.

**A synthetic check comes with the fit** (`python -m r1sim.sysid demo stiction`): the protocol run on the simulated
hanging robot with known friction recovers it to +-0.02 N m on most joints (+-0.1 hip roll, +-0.15 waist roll).

## 2. Joint dynamics - steps and chirps

`step` (+-0.05 rad, hold 2 s, 2 repeats) and `chirp` (0.03 rad, 0.5 -> 5 Hz over 20 s) per joint, one at a time; an
extra fast step (+-0.02 rad) on an arm joint for the command-to-motion delay. Then replay them in the simulator command
for command ([`sim/REPLAY.md`](sim/REPLAY.md)).

**Our R1:** the delay to 10 % of a 0.05 rad step was 16-44 ms by joint (head yaw 16-18, waist 18-22, knee 18-24, hip
20-26, ankle 22-30, elbow 28-32, shoulder pitch 36-44), left = right in every pair - and the replay showed it is all
joint dynamics: **no added command latency** (0-1 ms). Chirp gain ~0.9-1.1 up to 2 Hz, lag 4-18 deg at 1-2 Hz, a mild
resonance near 5 Hz on the hanging limbs. The replay identified the legs' armature (0.008), the arms' friction (<= 0.1
N m on the shoulders, ~0.03 elsewhere) and the elbows' damping (~1 N m s/rad).

## 3. Static torques - the arm masses (slow sweeps with and without a payload)

A joint at rest under a position servo sits anywhere inside its friction band, and on the shoulders that band (~1 N m) is
as large as a 0.5 kg payload's effect - so static holds cannot show +-10 %. A slow up-and-down sweep under a soft servo
passes every angle once moving up and once moving down: the friction has opposite signs on the two passes, so their
mean is the holding torque g(q). The same sweep with a known payload gives the torque scale independently of the
model's masses.

```bash
r1 record run static_pose group=arm_left 'q=[<a base pose where the joints carry gravity>]' move_s=3 hold_s=1 --yes
r1 record run ramp joint=left_elbow amplitude_rad=0.3 'speeds=[0.1,0.2]' payload_kg=0 --yes      # then 0.5 kg at the wrist
bash scripts/run.sh python -m r1sim.sysid.b3 --sweep <manifest>@0 --sweep <manifest>@0.5 --json out/arm_masses.json
```

Per joint: the **mass scale** s (measured g vs the model's at 0 kg, with a constant offset) and the **payload scale** k
(the measured 0.5 kg increment vs the model's); the check passes when >= 20 points are within +-10 % AND both scales are
within 10 % of 1. **Our R1** (0.501 kg ~7 cm from the palm): left elbow s 1.011 / k 1.059, left shoulder pitch 1.046 /
1.030, right shoulder pitch 1.007 / 0.979 - shown; left shoulder roll 1.107 / 0.998, right elbow 1.087 / 1.111, right
shoulder roll 1.035 / 1.270 - marginal (median errors 1.1-3.7 %). The legs from the friction ramps: the model's leg
gravity torques are ~5-15 % LOW (hip pitch slope 1.06-1.07, knee 1.12-1.16).

## 4. The IMU's mounting rotation (waist sweeps, no level needed)

Slow waist sweeps on the stand rotate the pelvis (and its IMU) about two known axes; the gyro sees them in its own frame:

```bash
r1 record run ramp joint=waist_roll amplitude_rad=0.15 'speeds=[0.1,0.1,0.1]' --yes
r1 record run ramp joint=waist_yaw  amplitude_rad=0.3  'speeds=[0.1,0.1,0.1]' --yes
bash scripts/run.sh python -m r1sim.tier_b.h1_imu --session <the session folder>
```

**Our R1:** roll / pitch / yaw **0.56 / 0.82 / -0.39 deg** (standard errors 0.07 / 0.18 / 0.24) - the model's 0.50 /
0.96 / 0 confirmed within the method's accuracy ([`sim/REPLAY.md`](sim/REPLAY.md)). With a level reading 0.1 deg and a
reference face on the pelvis, `r1 record run imu_static duration_s=30 --yes` while reading the level twice (turned 180
deg between) is the classic alternative.

## 5. The board without commands (the board-hold test)

The one test that needs a safety function switched off (the guard to observe-only), on the stand only, with the
escalation 20 / 100 / 300 / 400 / 500 ms and a stop rule ([`hardware/BRING_UP.md`](hardware/BRING_UP.md) 7.1). Then
`python -m r1sim.b6_board` simulates the same test with your model. **Our R1's board holds the last command 383-387 ms,
then releases every loaded joint in one tick into damping kd ~12.5** ([`sim/BOARD_MODEL.md`](sim/BOARD_MODEL.md)).

## 6. The hands: pads and finger heat

**Pads** (`pad_press`): known weights (each weighed with its carrier) pressed at a pad's centre and along the finger,
the hand palm-up and open; the fit `python -m r1sim.sysid pads --hands <hands.mcap> --presses <presses.yaml> --side left`
gives each pad's gain, footprint and detection floor. **Our R1's pads respond only to FIRM fingertip contact:** nothing
from 24-200 g hung or ~400-640 g pressed, +100..330 raw at ~900 g, ~+1,200 at ~1,630 g; under a force gauge the tips
responded only above ~10 N and by WHERE the tip landed - a contact switch, not a gauge. The inner pads read negative when
a finger bends. The grip verdict therefore uses the pads as "touching / not" (a threshold of 40 raw over a streaming
tare, at rest the noise <= 16).

**Finger heat** (`finger_hold`): a finger closing on a rigid block at 0.45 and 0.9 N m, then cooling 10 min each; the
fit `python -m r1sim.sysid heat --hands <hands.mcap> --side left --motor 3 --windows 0:180,190:850` compares thermal
structures (one / two poles, commanded vs measured torque, the exponent) by BIC. **Our R1:** at 0.9 N m 41 -> 71 C in
~5 s (~6 C/s); at 0.45 N m a plateau ~60 C; cooling to 43 C in ~40 s. Abort holds above ~70 C.

## 7. The camera

The lens (ChArUco), vignetting, depth and latency: [`hardware/CAMERA_CALIBRATION.md`](hardware/CAMERA_CALIBRATION.md).
Our head camera: f 125.1 px, RMS 0.39 px (within 1.6 % of the calibration file in `models/head_camera/`).

## Where the results go

Each fit prints its result with a provenance snippet (value, source, date, status `measured`, a range from the fit's
standard errors). Putting it into the model is a deliberate change:

| Fit | Model change | Then re-run |
|---|---|---|
| friction | the real profile's per-joint friction (`r1sim.model`) + `models/provenance.yaml` | the robustness batch, the stall tests |
| replay (armature, arm friction, damping, delays) | the real profile, `models/driver.yaml` | the replay itself, the batch |
| the board | `models/driver.yaml` (`cmd_timeout_s`, `on_cmd_timeout`, `damp_kd`) | `python -m r1sim.b6_board`, the stall tests |
| the IMU mounting | `model.IMU_MOUNT_OFFSET_DEG` + provenance | the batch |
| pads / heat | `models/sensors.yaml` | the grasp tests |
| the camera | the calibration file (`models/head_camera/`; the robot's `/etc/r1mc3/head_camera.json`) | the camera tests |

`tests/unit/test_provenance_model.py` keeps `models/provenance.yaml` equal to the code; the robustness batch is the
go / no-go for balancing on the changed model.

## Not solid (said plainly)

- Everything was measured on the robot HANGING: the friction under load and the floor contact are not identified.
- One robot, one firmware, one or two sessions per protocol; the board's hold was seen releasing twice.
- The ankle is a linkage the serial model does not carry; the arms' structure (the motor-side encoder, a compliant
  gear) is approximated.
- The pads' model in the simulator is kinder than the real pads.
