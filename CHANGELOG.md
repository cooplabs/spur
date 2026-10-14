# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-14

The first public release: a controller and simulator for the Unitree R1 with two Dex3-1 hands, running on one real
robot (balancing on the floor, both hands, the Jetson Orin backpack) and in simulation.

### Real-time core and guard (`rt/`)
- `r1rt`: the C++17 real-time core - the 500 Hz writer (absolute deadlines, the only writer of `rt/lowcmd`), the
  50 Hz policy thread (a fixed-size Eigen MLP, golden-tested against the reference implementation), the per-tick
  safety checks, the hard limits, the upper-body executor (HOLD / TRAJECTORY / STREAM / FREEDRIVE, deadmen, the lag /
  torque / contact / tilt guards, one smooth stop profile), the modes MONITOR, TAKEOVER (mirror -> blend), BALANCE,
  STAND_FREEZE, ON_STAND, DAMP (operator only), the stand-session lock, the resume blend from the measured pose after
  any stop, the arm gravity model and fingertip kinematics, the optional state-age rung (off by default).
- `r1guard`: the hot standby - its own state reader, a warm shadow policy, the takeover by an ownership-token swap
  (~10 ms), the behaviour by the mode it took over from, the operator's damp and reclaim, forensics from `/proc`.
- Shared memory: seqlocks, SPSC rings, the ownership token, the stop channel, the mode mailbox; Unitree hg message
  types + CRC32 (golden-tested); an opt-in CycloneDDS 0.10.2 build of the processes; Python bindings (`r1rt_py`).

### Service, hands, camera, logs (`python/r1mc3/`)
- `r1d`: the command API (JSON over TCP), goals with a lifecycle, leases, typed results and error codes; L0 joints,
  L1 streams (jog, head, Cartesian twist), L2 arm / hand / head goals (reach, linear, relative, path with blending,
  hover with approach styles, guarded lower, guarded move, slide, carry, dry-run plan), L3 skills (ready postures,
  pick, place, sequences, parallel goals); the operator words (takeover, on-stand, stop / halt / resume, freeze,
  balance, damp, reclaim, shutdown with the log save, the board-hold and guard-takeover stand tests); stand recordings
  for identification.
- The world model (surfaces, containers, obstacles, objects) and one MuJoCo geometry engine; the planner (5-DoF IK
  with an orientation search, straight lines or RRT-Connect, whole-goal validation before motion) in worker
  processes.
- `r1hands`: the Dex3-1 hands through `dex3_server`, state-age and frozen-stream checks, the pads (tare, creep
  follower), heat handling, the grip state machine with an evidence-based holding verdict ("contact decides").
- `r1cam`: the head camera (images, depth on demand), pointing through the calibrated lens, object extents;
  measurement of AprilTag-marked entities by the head camera or a phone still.
- `r1log`: shared-memory rings -> MCAP, a flight recorder, the session save; the battery in `r1 status`.
- The `r1` command line and the Python SDK `r1mc3.client`.

### Simulator (`sim/r1sim/`)
- A MuJoCo model of the R1 + two Dex3-1 hands + the backpack with every parameter's provenance; a
  training-equivalent profile (bit-identical to the policy's training model) and a real-equivalent profile identified
  on a robot (friction, latency, the motor board's behaviour without commands, the IMU's mounting); seeded variants.
- Lockstep and real-time engines; the C++ core and guard in lockstep; the DDS bridge (the real binaries against the
  simulated robot, Unitree's mode-switch RPC emulated); fault injection; generic scenes (an empty floor, a table with
  simple objects, the robot on its stand); task scenes plug in from outside the package.
- The viewer (`scripts/demo.sh`, `scripts/viewer.sh`): MuJoCo's window with the controller's own C++ core and
  `r1guard` in the loop (or, with `--stack`, the real processes over DDS); mouse zoom / pan / orbit, camera presets,
  overlays (mode, tilt, torques, contacts, pads, faults, the motors' owner), fault and push keys (stall the core and
  watch the guard take over), joint / hand jog, object drop-in, a stand scene, recording, headless frames and seeded
  batch runs (`python -m r1sim.viewer.batch`).
- Identification tools (steps, chirps, friction ramps, statics, the board model, the head camera's lens and depth).

### Examples (`examples/`)
- `pick_and_place.py`: a box picked from a table and placed 12 cm away through the goal API (the real processes on
  the simulated robot, or `--lockstep` in one process); `jog.py`: teleop through the streaming API (hand twist, joint
  jog, head, fingers; keyboard or a scripted run); `robustness_batch.py`: simulated robots drawn from the identified
  parameter ranges, balanced by the C++ core, under pushes and faults - a table of falls and tilts.

### Deployment (`deploy/`)
- Releases installed side by side on the backpack, activate / rollback (refused while we own the motors), systemd
  units with real-time priorities and CPU affinity, the runtime CPU partition, the kernel's real-time budget and quiet
  consoles, a stop that never damps (refused manual stops, an inhibitor for reboots).

### Documentation
- The design, the safety model, the API and operator references, the hardware bring-up guide (the joint directions
  verified on a robot), the simulator and identification guides, the balance policy, the stand dataset, and a case
  study of the fall of our previous controller.

### Release assets (attached to the GitHub release, not in the repository)
- `r1mc3-stand-dataset-0.1.0.tar.gz`: the stand recordings used to identify the simulator (steps, chirps, friction
  ramps, static poses, the direction check, IMU statics, hands, the board-hold and guard-takeover tests; CC BY 4.0) -
  `docs/DATASET.md`.
- `r1mc3-policy-training-0.1.0.tar.gz`: the training kit of the balance policy (the environment configuration, the
  PPO settings, two warm-start checkpoints) - `docs/POLICY.md`.

### Not solid in 0.1.0
- Every hardware number comes from one robot. Most L2 / L3 goals and the skills are simulator-proven only. Grasping
  from a camera click is not reliable on thin, clear or small objects. A protective fall posture, Unitree's `arm_sdk`
  overlay, walking commands and a hand-back of the motors to Unitree's controller are not built.
