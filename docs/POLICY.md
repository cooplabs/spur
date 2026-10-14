# The balance policy (R1MC-Manip2-Lean)

r1mc3 balances the R1 with a small neural network trained in simulation: the **balance policy**. It runs at 50 Hz
inside the real-time core (`r1rt`), commands the 12 leg joints and the 2 waist joints, and leaves the arms to the
rest of the controller (the arms are inputs to it, not outputs). This page says what the network is, how to check that
you have the exact file we ran, where it came from, and how it was trained - enough to retrain it or to train your
own.

The weights are ours and are released under this repository's licence (Apache-2.0). The R1 model it was trained on
is Unitree's (Apache-2.0, `models/r1/`, see `NOTICE`); the training framework is mjlab (Apache-2.0).

## The files and their checksums

`policies/r1mc2_manip2_lean_20261006T0421Z/` (the folder name keeps the training run's id: the previous version of our
controller, "r1mc2", trained it; 6 Oct 2026 04:21 UTC):

| File | sha256 | What |
|---|---|---|
| `policy.onnx` | `09bf2fd8b8d11d6219ca8a4323f1b4e7ef238bfe88d2bf330579c7e1a833b9be` | the observation normaliser + the MLP, ONNX opset 18, with the metadata the controller reads (below) |
| `r1mc_policy.json` | `70fa37d0d70a67474980af8ef3c62074adda41cbd949506de1576bf83ccabdba` | the run's summary: gait period, command limits, arms external, iterations, the unitree_rl_mjlab commit, the lean range, the stance, the checkpoint it was fine-tuned from |
| `params/env.yaml` | `72121ecdce7ba209cbf202f41b6b774b6c3483fb89684d60f70dd1bcd802d86c` | the exact training environment as mjlab dumped it: the simulation options, the actuators, the observations and their noise, the randomisation, the rewards, the commands, the terminations |
| `params/agent.yaml` | `25f536da1dedf60c7e529d192812a0f0037637927287e8bcc5433cbcd57a47c6` | the exact PPO settings (rsl_rl's on-policy runner) |

Check them with `shasum -a 256 policies/r1mc2_manip2_lean_20261006T0421Z/*.* policies/r1mc2_manip2_lean_20261006T0421Z/params/*`
(Linux: `sha256sum`). The C++ build converts `policy.onnx` into the core's fixed-size Eigen network at build time
(`rt/policy.cmake`, `tools/policy_convert.py`) and embeds the ONNX file's sha256; a golden test checks the C++ network
against the Python reference runner (`r1mc3.policy`) on recorded inputs (agreement ~1e-14).

## What the network is

| | |
|---|---|
| Architecture | input normaliser (fixed mean / std from training) -> MLP 74 -> 512 -> 256 -> 128 -> 14, ELU activations |
| Rate | 50 Hz (training: physics 200 Hz = 5 ms steps, decimation 4); the core runs it on its policy thread and its 500 Hz writer sends the latest targets |
| Output | 14 joint position offsets: left leg (hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll), right leg (same order), waist roll, waist yaw. Target = default pose + action x scale; scale 0.178 rad for hips / knees / waist, 0.398 rad for the ankles |
| Motor gains (ONNX metadata, used by the core) | hips / knees / waist kp 84.30, kd 5.37; ankles kp 31.39, kd 2.00; shoulder pitch / roll kp 60, kd 2; shoulder yaw / elbow kp 45, kd 2; wrist roll kp 30, kd 1.5 |
| Default pose (rad) | hip pitch -0.30, hip roll +-0.025, hip yaw -+0.13, knee 0.455, ankle pitch -0.14, ankle roll -+0.05 (left / right), waist 0, shoulder pitch 0.22, shoulder roll 0.193 / -0.194, shoulder yaw 0.039 / -0.033, elbow 0.76, wrist roll 0 - Unitree's stand-lock stance (feet ~0.22 m apart) |

The 74 inputs, in this order (ONNX metadata `observation_names`):

| Input | Size | Notes (training noise, uniform) |
|---|---|---|
| base angular velocity (IMU gyro, rad/s) | 3 | +-0.2 |
| projected gravity (the gravity direction in the pelvis frame) | 3 | +-0.05 |
| velocity command (forward m/s, sideways m/s, yaw rad/s) | 3 | none; the controller sends zeros when standing |
| lean command (rad, body pitch forward) | 1 | none; 0 - 5 deg trained |
| gait phase (sin, cos of a 0.86 s period) | 2 | none |
| joint positions relative to the default pose, all 24 body joints (legs, waist, both arms) | 24 | +-0.01 rad |
| joint velocities, all 24 body joints | 24 | +-1.5 rad/s |
| the previous action | 14 | none |

The arms appear only as inputs: whatever the arm controller does, the policy sees it and balances against it. The
hands (Dex3-1, ~0.7 kg each) are part of the trained robot's mass.

## Where it came from (provenance)

| Step | Run | Start | PPO iterations x parallel robots | Final numbers (training log) |
|---|---|---|---|---|
| 1. R1MC-Manip | 5 Oct 2026 01:51 UTC | from scratch | 3,605 x 8,192 | mean reward 43.7, falls 4-8 % of 20 s episodes |
| 2. R1MC-Manip2 | 6 Oct 2026 03:34 UTC | warm start from step 1's last checkpoint (the new lean input zero-weighted, the stance change folded into the normaliser and the output bias, so iteration 0 acts exactly like step 1) | 2,393 x 8,192 | mean reward 45.2, falls 4 %, lean error 3.6 deg (the lean was not learned) |
| 3. **R1MC-Manip2-Lean** (this policy) | 6 Oct 2026 04:21 UTC | fine-tune of step 2's checkpoint 2392 (the lean input re-normalised, a lean-tracking reward added) | 500 x 8,192 | mean reward 51.1, standing lean error 0.65 deg, falls 8 % (pushes and random arm swings included) |

Software: mjlab 1.2.0 (https://github.com/mujocolab/mjlab, Apache-2.0) inside Unitree's unitree_rl_mjlab at commit
`1425b15f73bd4095f0df53709d7c389c3eb9e790` (the R1 model in `models/r1/` is that commit's, unchanged), MuJoCo and
MuJoCo Warp 3.5.0 (the simulator in this repository pins the same MuJoCo version on purpose), PyTorch 2.x; trained
on one rented H100 GPU, about 1-2 hours per step. The ONNX file was exported by mjlab's exporter with our metadata.

On hardware: this policy balanced our R1 (a 32 kg R1 with two Dex3-1 hands and a backpack computer) under the previous
controller for four days of manipulation work, and under r1mc3 since its first balancing session (taking over from
Unitree's own controller, standing, operator freezes and resumes). It was **not** retrained for r1mc3: before r1mc3's
first balancing session the policy was checked on the robot model identified from stand recordings - 100 randomised
seeds x 4 cases (standing, arms forward at a table, a 4 deg lean with both arms reaching, torso pushes), every case
>= 99 / 100 (see `docs/IDENTIFICATION.md`; the example `examples/` robustness batch runs the same kind of check).

## The training recipe

Everything below is in `params/env.yaml` / `params/agent.yaml` exactly; this is the readable version.

**Task.** Stand (and walk when commanded) on legs + waist while the arms are moved by someone else. A velocity-tracking
task (mjlab's velocity task) with the arms taken out of the action space: a zero-size action term moves the arms
every physics step toward random targets at random speeds, and the policy only sees them.

**The arms during training.** Targets: 45 % from a library of arm poses over a tabletop workspace in front of the
robot (+-0.08 rad noise), the rest uniform within these ranges (left arm; the right arm mirrors roll, yaw and the
wrist):

| Joint | Range (rad) |
|---|---|
| shoulder pitch | -2.4 .. 0.75 |
| shoulder roll | -0.1 .. 1.85 |
| shoulder yaw | -1.87 .. 0.9 |
| elbow | -0.93 .. 2.05 |
| wrist roll | -1.87 .. 1.87 |

20 % of the time the arms hold still, 10 % they return to the default pose; a new target every 0.6-2.5 s at 0.3-1.5
rad/s, and 10 % of the moves are sudden swings at 1.5-3.0 rad/s (a planned fast arm swing once tipped the robot 20
degrees, so the policy has seen them). The controller's arm planner keeps the arms inside these ranges (they are in
the ONNX metadata as `arm_ranges_left`; `r1mc3.plan.ik.policy_envelope`).

**Forces at the hands.** Sustained forces on both hands for 0.5-4 s with 1-5 s pauses: 40 % an isotropic push up to
+-15 N, 60 % a "table reaction" - up to 20 N upward and 15 N backward plus up to 6 N sideways in the robot's heading
frame (what a table pushes back with while a hand presses or slides on it).

**Commands.** Velocity: forward -0.5 .. 0.8 m/s, sideways -0.4 .. 0.4 m/s, yaw -1.0 .. 1.0 rad/s, resampled every 3-8
s, half of the robots standing (zero command). Lean: 0 .. 5 deg forward body pitch, 25 % of the time zero, changing at
up to 4 deg/s, resampled every 4-10 s, zero while walking.

**Randomisation** (per robot at start-up unless said):

| What | Range |
|---|---|
| foot friction | 0.3 .. 1.6 (absolute, shared by all foot contacts of a robot) |
| encoder bias | +-0.015 rad per joint |
| torso centre of mass | +-5 cm per axis |
| torso mass | -0.5 .. +3.0 kg |
| payload per hand | -0.08 .. +0.5 kg |
| PD gains | kp x 0.85 .. 1.15, kd x 0.8 .. 1.2 |
| joint friction (frictionloss) | x 0.5 .. 3.0 |
| joint armature | x 0.7 .. 2.5 |
| stance width at reset | -3 .. +6 cm per side (as hip roll) |
| pushes (interval 5-6 s) | a velocity kick up to 0.5 m/s horizontal, 0.4 m/s vertical, 0.52 rad/s roll / pitch, 0.78 rad/s yaw |
| reset pose | +-0.5 m, any heading |

**The actuator model.** MuJoCo position actuators with the gains above (hips / knees / waist armature 0.0214,
frictionloss 1.0, effort limit 60 N m; ankles armature 0.00795, frictionloss 0.8, 50 N m; shoulders 0.01 / 0.2, 60 N m
for pitch / roll and 33 N m for yaw / elbow; wrist 33 N m), soft joint limits at 90 % of the range. Physics: 5 ms
steps, implicit-fast integrator, Newton solver (10 iterations), elliptic friction cones with impratio 10 (pyramidal
soft contacts let a pushed foot creep in simulation where a real sole does not).

**Rewards** (weight; mjlab scales them by the step time):

| Term | Weight | What |
|---|---|---|
| track linear velocity | 1.0 | exp kernel, std 0.5 m/s |
| track angular velocity | 1.0 | exp kernel, std 0.71 rad/s |
| track lean | 1.5 | exp kernel on the pelvis pitch vs the lean command, sigma 1.5 deg (5 deg wide term) |
| posture | 1.0 | the legs / waist near the default pose (standing std 0.05-0.10 rad, wider while walking / running), following the lean |
| torso orientation | -1.0 | squared tilt away from the commanded lean |
| foot gait | 0.5 | alternating contacts at the 0.86 s period when walking |
| foot clearance | -1.0 | swing foot 6 cm |
| foot slip | -0.25 | |
| stand still | -1.0 | joint motion when the command is zero |
| standing hip effort | -0.0002 | hip-roll torque while standing (pushing the feet apart cost a hip motor its protection once) |
| body angular velocity | -0.05 | |
| angular momentum | -0.025 | |
| joint acceleration | -2.5e-7 | legs + waist |
| action rate | -0.05 | |
| joint position limits | -10 | legs + waist |
| self collisions | -1.0 | leg contacts above 10 N |
| soft landing | -0.001 | |
| termination | -200 | falling (body tilt beyond 70 deg) |

Episodes of 20 s. **PPO** (rsl_rl on-policy runner): 24 steps per robot per iteration, 5 epochs, 4 mini-batches,
learning rate 1e-3 with the adaptive (KL 0.01) schedule, gamma 0.99, lambda 0.95, clip 0.2, entropy 0.01, value loss 1.0
(clipped), gradient norm 1.0, actor and critic 512-256-128 ELU with observation normalisation, initial action std 1.0;
the critic also sees the base linear velocity (+-0.5 m/s noise), foot heights, air times, contacts and contact forces.

**To retrain:** the training kit that produced this policy is a release asset,
`r1mc3-policy-training-0.1.0.tar.gz` (9.3 MB, sha256
`22a227d8f185dc46acb21414b129f56fd76367e4ce3d3cbced6a8cfb02f4122c`): the mjlab task definitions (R1MC-Manip,
R1MC-Manip2, R1MC-Manip2-Lean - the five custom terms above are in them), an install script for unitree_rl_mjlab at the
commit above, a one-command training script for a Linux machine with an NVIDIA GPU (pinned versions, a speed test,
training, the ONNX export with the controller's metadata) and two warm-start checkpoints. Its README gives the three
commands that reproduce the chain above. RL training is not deterministic across GPUs and versions: a rerun gives a
policy like this one, not the same weights - check any new policy in the simulator first (the robustness batch) and on
the stand before it balances the robot.

## Known limits

- **No sensing or actuation delay was modelled in training.** The real robot's measured delays are small (the
  command-to-motion delay identified at 0-1 ms beyond the joint dynamics, the state ~0.5 ms old), and the robustness
  check covers 0-2 ms of command delay and 0-1 ms of state delay; a robot or network with tens of milliseconds of delay is outside what the policy has seen.
- The ankle armature reported for the R1 by others (unitree_rl_mjlab issue #51) lies outside the trained
  randomisation; the identified model's check passed anyway.
- The policy commands a narrow stand-lock stance; it can walk in simulation, but r1mc3 uses it for standing and
  manipulation only - walking on hardware with it is untested.
- The arms must stay inside the trained ranges above (the planner enforces it) and should move as trained: the planner
  plans at <= 1 rad/s and the core's executor caps every arm joint at 2 rad/s; a client that streams its own arm
  targets faster than the trained swings (3 rad/s) is outside what the policy has seen.
