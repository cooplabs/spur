# The stand dataset (hardware recordings of one R1)

The numbers that make r1mc3's simulator behave like a real R1 - joint friction, armature and damping, what the motor
board does when commands stop, how fast the hot standby takes over - were fitted to recordings of our robot hanging on
its stand. We publish those recordings so you can check the fits, redo them for your own model, or compare your robot
with ours.

| | |
|---|---|
| File | `r1mc3-stand-dataset-0.1.0.tar.gz` (87.8 MB; 131 recordings) |
| Where | the assets of the repository's `v0.1.0` release |
| sha256 | `64d95d32febd920abe579422301f81f5ca7ad1b6f28bdffb0670fc15850d654b` |
| Licence | CC BY 4.0 (the data; the code in this repository is Apache-2.0) |
| Robot | one Unitree R1 (26 body joints) with two Dex3-1 hands and a backpack computer, ~32 kg, recorded 11 Oct 2026 |

```bash
shasum -a 256 r1mc3-stand-dataset-0.1.0.tar.gz     # Linux: sha256sum
tar -xzf r1mc3-stand-dataset-0.1.0.tar.gz
cd r1mc3-stand-dataset-0.1.0 && shasum -a 256 -c SHA256SUMS
```

## What is in it

| Folder | Recordings | What |
|---|---|---|
| `body/steps/` | 15 | one joint at a time: small position steps and back, the robot hanging, every other joint held stiff |
| `body/chirps/` | 5 | one joint at a time: a frequency sweep |
| `body/friction_ramps/` | 14 | each leg / waist joint: a slow target ramp at low stiffness, 0.1 and 0.3 rad/s, both directions |
| `body/static_poses/` | 26 | arm / leg / head poses held, some arm poses with a payload in the hand |
| `body/direction_check/` | 58 | every joint moved a little in its positive direction and back: the sign of each joint on the robot |
| `body/imu_statics/` | 2 | 120 s still: IMU noise and bias |
| `hands/` | 6 | the left Dex3-1: known weights on one pad; a finger holding a constant torque for 60 s (heating) |
| `board_hold/` | 3 | the real-time core paused 20 / 100 / 500 ms, the hot standby not allowed to take over |
| `guard_takeover/` | 2 | the real-time core paused 3 s with the hot standby armed |

The archive's `README.md` documents every array (units, joint order, rates); `index.csv` lists the files;
`identified.json` holds the fitted numbers.

## Loading it

Every recording is a NumPy `.npz` with a JSON `meta` string:

```python
import json
import numpy as np

z = np.load("body/steps/005_step_waist_yaw.npz")
meta = json.loads(str(z["meta"]))          # the plan: kind, joint, amplitude, duration, start pose
t, q, q_cmd = z["t"], z["q"], z["q_cmd"]   # t in s from the step's start; q / q_cmd (n, 26) in rad
tau = z["kp"] * (q_cmd - q) + z["kd"] * (z["dq_cmd"] - z["dq"]) + z["tau_ff"]   # the commanded joint torque (N m)
```

The body files use the keys of the simulator's hardware windows: `r1sim.tier_b.data.Window.load(path)` reads one as it
is (the replay of a step / chirp in the simulator with the logged commands starts from such a window).

## What it shows

| Quantity | Value | Recordings |
|---|---|---|
| leg / waist friction (hanging) | hips 0.37-0.72 N m, knees 0.43-0.63, ankle pitch 2.07-2.52, ankle roll ~1.2-1.4 (unreliable), waist 0.41-0.52 | `friction_ramps/` |
| joint armature | hips / knees / waist ~0.008 kg m^2; arms / head <= 0.0025 | `steps/`, `chirps/` |
| arm / head friction | shoulder pitch / roll ~0.1 N m, the rest ~0.03 | `steps/`, `static_poses/` |
| the motor board without commands | keeps the last targets for ~0.385 s, then damps (kp 0, kd ~12.5): the arms start to sink | `board_hold/` |
| the hot standby's takeover | it starts writing ~10.5 ms after the core stopped (both runs) | `guard_takeover/` |
| command delay beyond the joint dynamics | 0-1 ms | `steps/`, `chirps/` |

Caveats: one robot, one day. The friction was measured with the legs hanging, the bearings unloaded - a standing
robot's friction can only be higher. The ankle-roll ramps never reached a steady slide at the test stiffness.
