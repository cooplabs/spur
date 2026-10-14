# Spur

An open-source controller and simulator for the **Unitree R1 humanoid with two Unitree Dex3-1 hands**, built around
one rule: **nothing automatic may ever make the robot go limp.**

> **Status: 0.1.0, an early release.** It runs on one real R1 (balancing on the floor, both hands, the backpack
> computer) and in its simulator. Many parts are proven only in simulation so far; the section
> [Limits](#limits-what-is-not-solid-yet) says which. Read [`docs/SAFETY.md`](docs/SAFETY.md) before you run anything
> on a robot.

## What it is

Spur separates what keeps the robot standing from everything else. (Its code is the Python package `r1mc3` -
the R1 motor controller, version 3 - with the simulator `r1sim`, the C++ core `r1rt` and the `r1` command line.)

The pieces:

- **`r1rt`, a small C++17 real-time core**, is the only process that talks to the motors (`rt/lowcmd` at 500 Hz). It
  runs the balance policy (a fixed-size MLP at 50 Hz), the safety checks, the joint limits and the arm / head motion
  executor. Nothing in it blocks: no files, no network reads, no memory allocation in the loop, no Python.
- **`r1guard`, an independent C++ hot standby**, keeps a warm copy of the balance policy and watches the core's
  heartbeat. If the core stalls it takes the motors over (in about 10 ms) and keeps the robot balancing in place.
  Exactly one process writes to the motors at any time, enforced by an ownership token in shared memory.
- **Python processes for everything else**: `r1d` (the command API with goals, leases, a world model, a motion
  planner, skills), `r1hands` (the Dex3-1 hands, their pressure pads and a grip verdict), `r1cam` (the head camera),
  `r1log` (full-rate logs and a flight recorder) and the `r1` command line. Any of them can stall or crash without
  the robot falling: arm motions slow to a hold, balance is unaffected.
- **`r1sim`, a MuJoCo simulator of the R1 + two Dex3-1 hands**, with every model parameter traced to a source and the
  key ones identified on a real robot (joint friction, latency, what the motor board does when commands stop). The
  same controller code runs in it in lockstep (deterministic, faster than real time) or as the real binaries talking
  DDS to a simulated robot. It comes with a viewer (zoom / pan / orbit, camera presets, overlays, fault keys,
  recording).

## The safety model in plain words

- **One writer.** Only `r1rt` - or, after a takeover, `r1guard` - commands the motors.
- **A standby that is already running.** `r1guard` computes the same balance action as the core every cycle, so if
  the core goes silent it can take over without a jump.
- **Never limp.** No automatic path ever damps the robot (switches its joints to zero stiffness). When something goes
  wrong the fallback ladder is: run the policy inline in the core -> the guard takes over and balances in place ->
  freeze into a stiff standing pose. Damping happens only when an operator asks for it (`r1 damp`, the remote's
  L2+B, the hardware E-stop).
- **Stops never depend on what they stop.** Arm motions carry deadlines; if the planner or the API goes quiet, the
  core brings the arms to a hold by itself. `r1 stop` is always accepted.
- **Refuse before moving.** Multi-step arm goals are planned and checked end to end against one world model before
  anything moves; a refusal says why and that nothing moved.
- **Holding is judged on evidence.** The grip verdict needs opposing contacts, a finger gap that matches the object
  and a lift check - never force alone.

The full design is in [`docs/DESIGN.md`](docs/DESIGN.md); the case study of the fall that shaped it is in
[`docs/case_studies/`](docs/case_studies/).

## Quick start in the simulator

Requirements: macOS (Apple silicon) or Linux (x86_64 or arm64), [uv](https://docs.astral.sh/uv/), a C++17 compiler,
git. uv installs Python 3.11, CMake and Ninja into the project's own environment; nothing is installed system-wide.

```bash
git clone https://github.com/cooplabs/spur spur && cd spur
bash scripts/demo.sh                # the real controller balancing the simulated R1, in a window
bash scripts/demo.sh --headless     # the same without a window (frames + numbers; what CI runs)
```

The first run builds the C++ core's Python binding (a couple of minutes; no DDS needed). What you see: the R1 on the
floor balanced by the controller's C++ real-time core with `r1guard` standing by - the same code as the `r1rt` /
`r1guard` binaries, stepped in lockstep with the simulator. The viewer's keys inject faults: `S` stalls the core and
`r1guard` takes the motors, `P` pushes the robot, `D` delays the robot's state; `J` and the arrow keys jog an arm, the
head or a hand - [`docs/sim/VIEWER.md`](docs/sim/VIEWER.md). `bash scripts/demo.sh --stack` runs the REAL processes
instead (`r1rt` + `r1guard` + `r1d` talking DDS to the simulated robot; needs CycloneDDS - see the viewer doc).

More:

```bash
bash scripts/viewer.sh --scene table         # the viewer on its own: a table with simple objects
bash scripts/ci.sh                           # lint, types, C++ build + tests, Python tests (what CI runs)
ls examples/                                 # pick-and-place through the API, jogging the arms, a robustness batch
```

Using the API from Python (against the simulator or a robot):

```python
from r1mc3.client import R1

with R1() as r:  # r1d on 127.0.0.1:8780
    print(r.status().data["summary"])
    # the simulator's table scene (r1sim.scenes.table_entities() gives these entities)
    r.world_declare({"kind": "surface", "id": "table", "pose": {"pos": [0.55, 0.0, 0.78]}, "size": [0.6, 1.2, 0.04]})
    res = r.reach("left", [0.35, 0.15, 0.85])  # planned against the world, waits for the end
    print(res.ok, res.code, res.data.get("error_mm"))
```

The API reference: [`docs/API.md`](docs/API.md); the operator's command line: [`docs/OPERATOR.md`](docs/OPERATOR.md).

## On a real robot

**Hardware it was built and tested on:** one Unitree R1 with two Dex3-1 hands and the backpack computer (Jetson Orin
NX, L4T R35.3.1 / Ubuntu 20.04; Python 3.11 installed by uv), the motor link on a USB Ethernet adapter, the head's
stereo camera. Unitree's `dex3_server` (the hands' USB bridge) is closed source and not included: install it from
Unitree.

**The path:** [`docs/hardware/BRING_UP.md`](docs/hardware/BRING_UP.md) - install a release side by side with
anything already on the backpack, the real-time set-up, read-only checks first (nothing moves), the hands, the robot
on its stand (a stiff hold of the measured pose, joint-by-joint direction checks), the guard and board tests, and only
then balancing on the floor. One step at a time, each saying what moves and how to stop it. The joint directions
verified on our robot: [`docs/hardware/JOINT_DIRECTIONS.md`](docs/hardware/JOINT_DIRECTIONS.md).

The operator rules the design is built around:

- Before any word that moves something, a person holds the robot (or it hangs on its stand) and knows how to stop it.
- Nothing automatic makes the robot limp; damping is the operator's action.
- A session always ends the same way: hold the robot -> `r1 damp` -> support it standing -> `r1 shutdown` (saves the
  logs) -> power off.
- Safety-loosening changes need an explicit decision by the person responsible for the robot.

### What has been shown on our robot

Measured on one R1 (October 2026); your robot, firmware and network may differ.

| What | Result |
|---|---|
| The core on the stock kernel, after the CPU set-up | real-time scheduling granted; `cyclictest` under load p99.9 23 us, max 155 us; the 500 Hz writer skipped 0 deadlines outside deliberate stops |
| What Unitree's motor board does when commands stop | it holds the last command for about 0.4 s (fitted 0.385 s), then releases into damping - the guard's ~10 ms is well inside that |
| The guard taking over from a stopped core (robot on its stand) | 10.7 ms (twice), and 9.8 / 10.0 ms on a second day; it held the pose stiff and gave the motors back only on the operator's word |
| Joint directions | 26 / 26 joints moved the way the model says (checked by eye against the model, plus a gravity-sign check) |
| Taking the motors from Unitree's controller (standing, from its walk mode) | 2 / 2, release in about 0.2 s, tilt 0.2-0.5 deg after the switch |
| The operator's freeze while balancing, then back to balancing | tilt 3.0-3.4 deg in the freeze (simulator 2.4), then balancing again at once |
| A network stall that froze the previous controller (a 3.8 s blocking read on a Wi-Fi reconnect) | reproduced on the backpack; the robot state reaching `r1rt` was untouched (about 1050 messages/s every second) |
| The head camera streaming next to the motor link | about 1052 state messages/s, state age p99 1.0 ms while streaming |
| Scripted grasps standing at a table, the approach style chosen per object | 20 / 20 held with the best style per object; 26 / 32 over all styles tried |

### What has been shown in the simulator

| What | Result |
|---|---|
| A 4.0 s stall of the balance process (the failure that made the previous controller fall) | the previous architecture fell in 5 / 10 seeds; r1mc3 (the real processes on the simulated robot) stayed upright 23 / 23, guard takeover 10.2-12.2 ms, never limp |
| The balance policy on the model identified from the robot | 400 / 400 runs passed (standing, arms forward, a 4 deg lean, pushes), 0 falls |
| The C++ policy against the reference implementation | max difference 2.8e-14 on 10,000 observations (target 1e-5) |

## Limits: what is not solid yet

Said plainly:

- **One robot.** Every hardware number above comes from one R1 in one room. Firmware updates can change the board's
  behaviour, the joint directions or the message layouts: re-run the read-only checks and the direction check after
  any change.
- **Many parts are simulator-proven only**: most L2 / L3 goals, the skills (`skill.pick`, `skill.place`), the
  planner's approach styles, the measurement words and the grip verdict's rules for new object classes. The simulator
  is kinder than the robot in places (grasps of small, soft objects hold more often in simulation than on the robot).
- **Grasping from a camera click is not reliable yet**: on our robot the depth at a clicked pixel was off by 2-6 cm on
  thin, clear or small objects; the controller holds well when it is told the object's pose and size.
- **Not built:** a protective fall posture (a robot already falling keeps balancing, it is not damped), Unitree's
  `arm_sdk` overlay mode, walking commands beyond standing in place, a collision look-ahead for Cartesian streaming
  (`stream.arm_twist` is refused while a world is declared), a hand-back of the motors to Unitree's controller.
- **The real-time numbers are for the stock L4T kernel** with runtime CPU partitioning; no PREEMPT_RT kernel was needed
  or tried.
- **Code comments cite our internal development notes** (session numbers, hardware sessions "HW1" / "HW2", decision
  numbers "D<n>"); those notes are not published. The public design doc keeps the decisions that matter.

## Repository layout

| Path | Contents |
|---|---|
| `rt/` | C++17 (CMake): the real-time core `r1rt`, the guard `r1guard`, the shared-memory channels (seqlock, SPSC ring, ownership token), the policy runner (Eigen), Unitree hg message types + CRC, the arm model; GoogleTest tests; Python bindings (`r1rt_py`) |
| `python/r1mc3/` | `r1d` (service, goals, leases), the world model and planner, skills, `r1hands`, `r1cam`, measurement, `r1log`, the `r1` CLI, the Python SDK (`r1mc3.client`) |
| `sim/r1sim/` | the simulator: model assembly with provenance, two profiles (training-equivalent, real-equivalent), sensors, the motor-board model, fault injection, the DDS bridge, the viewer, system identification tools |
| `render/` | an optional render worker (MuJoCo Filament) for photo-like head-camera frames |
| `models/` | the R1 MJCF and meshes, the Dex3-1 hands, the head-camera calibration, `provenance.yaml` (every simulator parameter with its source and status) |
| `policies/` | the balance policy (ONNX + metadata, checksums, provenance) |
| `config/` | the robot profile (limits, gains, rates; every value with provenance) |
| `deploy/` | install / activate / rollback on the backpack, the real-time set-up, systemd units |
| `tools/` | calibration, identification and conversion tools |
| `examples/` | API examples in the simulator |
| `tests/` | unit, integration (simulator, real processes over DDS) and regression tests |
| `docs/` | design, safety, API, operator guide, hardware bring-up, simulator, identification, case studies |

## Documentation

| Read | For |
|---|---|
| [`docs/SAFETY.md`](docs/SAFETY.md) | the safety model, the fallback ladder, the operator rules |
| [`docs/DESIGN.md`](docs/DESIGN.md) | the architecture and why it is built that way |
| [`docs/API.md`](docs/API.md), [`docs/OPERATOR.md`](docs/OPERATOR.md) | the command API and the `r1` command line |
| [`docs/hardware/BRING_UP.md`](docs/hardware/BRING_UP.md) | the first bring-up on your R1 |
| [`docs/SIMULATOR.md`](docs/SIMULATOR.md), [`docs/sim/VIEWER.md`](docs/sim/VIEWER.md) | the simulator and its viewer |
| [`docs/IDENTIFICATION.md`](docs/IDENTIFICATION.md) | how the model was identified on the robot, with our numbers |
| [`docs/POLICY.md`](docs/POLICY.md), [`docs/DATASET.md`](docs/DATASET.md) | the balance policy and the stand recordings (release assets) |
| [`docs/architecture/`](docs/architecture/) | the processes, the guard, the executor, the shared-memory interfaces, the logger |

## Contributing, security, licence

Contributions are welcome: [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
Safety defects and vulnerabilities are reported privately: [`SECURITY.md`](SECURITY.md). Changes:
[`CHANGELOG.md`](CHANGELOG.md).

Copyright 2026 Coop. Licensed under the Apache License, Version 2.0 ([`LICENSE`](LICENSE)). Third-party material
and its licences are listed in [`NOTICE`](NOTICE) (Unitree's R1 and Dex3-1 model files among them).
