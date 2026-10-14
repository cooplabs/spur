# r1sim - the simulator

`r1sim` is a MuJoCo model of the Unitree R1 with two Dex3-1 hands, **built to be trusted before the robot is touched**:
every model parameter is traced to a source, the uncertain ones carry ranges, and the parts that matter for safety
(friction, latency, what the motor board does when commands stop) are **identified from the real robot**. It runs the
real controller - the same C++ core in-process, or the real binaries over DDS - so a test in the simulator exercises
the code that runs on the robot.

| Piece | What | Where |
|---|---|---|
| model assembly | the R1 (Unitree's MJCF) + two articulated Dex3-1 hands with pads + the backpack mass, in two profiles | `sim/r1sim/model.py`, `models/` |
| provenance | every parameter: value, source, status (`official` / `measured` / `derived` / `reported` / `estimate` / `modelling_choice`), and for estimates what replaces them; ranges for the variants | `models/provenance.yaml`, `models/driver.yaml`, `models/sensors.yaml` |
| engine | physics independent of any controller: lockstep (deterministic) or real time (its own thread, paced by the wall clock) | `sim/r1sim/engine.py` |
| driver | the motor board's law on the LowCmd fields, delays, and what the board does without commands | `sim/r1sim/driver.py` |
| sensors | encoders, IMU, `tau_est`, the hands' pads, the fingers' temperature - what the robot REPORTS, not what the physics knows | `sim/r1sim/sensors.py` |
| faults | a reproducible schedule of faults (stalls, command gaps, delays, motor errors, sensor spikes, NaN actions) | `sim/r1sim/faults.py` |
| the core in lockstep | r1rt's C++ core (and r1guard) through their Python binding, called tick by tick with simulated time | `sim/r1sim/v3_core.py`, `sim/r1sim/v3_guard.py` |
| the DDS bridge | the simulated robot behind the robot's own DDS interface: the real binaries run against it unchanged | `sim/r1sim/dds_bridge.py`, [`sim/DDS_BRIDGE.md`](sim/DDS_BRIDGE.md) |
| the process stack | the bridge + r1rt + r1guard + r1d + r1log + the `r1` CLI as one object, for tests and rehearsals | `sim/r1sim/stack.py` |
| scenes | generic scenes ("floor", "table") and a hook for your own | `sim/r1sim/scenes.py` |
| variants | seeded robot variants within every parameter's range | `sim/r1sim/variants.py` |
| rendering | MuJoCo's renderer + a calibrated head-camera lens chain + a fitted depth model; an optional Filament worker | `sim/r1sim/render.py`, `lens.py`, `depth.py`, `render/` |
| identification | the fits from stand recordings (friction, pads, finger heat, arm masses, camera) | `sim/r1sim/sysid/`, [`IDENTIFICATION.md`](IDENTIFICATION.md) |
| viewer | an interactive window and headless frames | [`sim/VIEWER.md`](sim/VIEWER.md) |

## Profiles

| Profile | What it is | Use it for |
|---|---|---|
| `train` | the robot **exactly as the balance policy was trained** (mjlab 1.2.0 on MuJoCo 3.5.0): Unitree's `r1.xml` as is (head welded), a rigid 0.697 kg lump per hand, the training's collision and joint settings, MuJoCo `<position>` actuators, 5 ms. Bit-identical to mjlab's own model (a sim-to-sim test pins it) | checking that the policy runs as trained; comparing with training |
| `real` | **our robot as best known**: the head articulated, the articulated Dex3-1 hands with pads, the IMU at its URDF position and its measured mounting rotation, the rear computing module (0.454 kg, weighed), joint friction / armature / damping identified on the stand, the board's identified behaviour, 1 ms | everything else |

World frame (the controller's base frame): on the floor under the middle of the feet, x forward, y left, z up.

MuJoCo is pinned at **3.5.0** - the version the policy was trained with. Changing it needs the sim-to-sim regression
(the `train` profile against mjlab's reference rollouts).

## Lockstep and real time

- **Lockstep** (`Simulation.step(n)`): a simulated clock; controllers registered with `add_controller` are called at
  their rates, in registration order, at step boundaries. **Fully deterministic** (bit-identical reruns) - the mode
  of most tests, the robustness batch and the identification fits.
- **Real time** (`Simulation.start_realtime()`): physics runs in its OWN thread, paced by the wall clock, and **never
  waits for a controller**. A controller that stalls stops nothing but itself: the board keeps doing what a board
  does (hold the last command, then - with the identified board - release into damping after ~0.385 s). This is what
  makes a controller stall reproducible in simulation; a simulator that steps physics inside the control loop freezes
  the world with the controller and cannot show the fall.

**The driver** applies, every physics step and per joint, the board's law
`tau = clip(kp (q_des - q) + kd (dq_des - dq) + tau_ff, -tau_max, tau_max)`, evaluated inside MuJoCo as an affine
actuator (so physics advances many steps per call without Python, and the kd term is integrated implicitly). On top:
a command delay, a state delay, and the board's behaviour when commands stop (`cmd_timeout_s` + `hold` / `damp` /
`zero_torque`). Identified on our robot ([`sim/BOARD_MODEL.md`](sim/BOARD_MODEL.md), [`sim/REPLAY.md`](sim/REPLAY.md)):
no added command latency beyond the joints' own dynamics (0-1 ms), a state delay of 0.5 ms, a release 0.385 s after the
last command, then damping kd 12.5.

## Sensors

What the robot reports, with its imperfections (every parameter in `models/sensors.yaml`, with sources and ranges);
the exact values stay available as `LowState.truth` for tests, never for controllers:

- **encoders** (26 body joints, 7 per hand): a per-seed zero offset, noise on q and dq, quantisation;
- **IMU**: gyro / accelerometer noise and per-seed biases, the mounting rotation as geometry, a per-seed variation;
- **`tau_est`**: a scale of the applied torque + noise (hands: raw units optional, junk readings at a rate);
- **pads**: the Dex3's 9 pads x 12 cells, 33 active cells placed on the hand model's pad boxes; force split between
  cells, baseline, gain, floor, saturation, drift, noise, glitches. **Note:** our real pads respond only to FIRM
  fingertip contact (a switch more than a gauge) - the model's light-touch response is kinder than the robot;
- **finger temperature**: a two-pole thermal model on the applied torque.

## Fault injection

`FaultSchedule([Fault(kind, t_start, duration, params)])` - every fault has a scheduled start and a duration, so a
failure reproduces bit for bit in lockstep:

| Kind | Applied by | What |
|---|---|---|
| `controller_stall` | engine | a controller (or all) is not called in the window |
| `cmd_gap` / `cmd_drop` | engine | the board sees no commands / loses each with probability p |
| `state_delay` | engine | the controller reads states `delay_s` old |
| `hand_state_freeze` | engine | a hand's state stops updating |
| `motor_error` | engine | a motor error word on chosen joints |
| `sensor_spike` | sensors | a reading jumps by a magnitude |
| `policy_nan` / `policy_out_of_range` | the policy runner | the action is NaN / a fixed value |

With real processes, the bridge adds link faults (`lowstate_gap`, drops, corrupted CRCs, delays, a foreign writer, a
push) and the processes can be paused (SIGSTOP) - [`sim/FAULT_TESTS.md`](sim/FAULT_TESTS.md).

## Running the real controller against it

**In-process (lockstep):** `r1sim.v3_core.V3CoreController` runs r1rt's C++ core (the same library, through the
`r1rt_py` binding) tick by tick: the writer at 500 Hz, the policy thread's iteration every 10th tick, a takeover from an
emulated Unitree controller, the upper-body executor's trajectories / streams / holds. `r1sim.v3_guard` adds r1guard
on the same shared segment (1 kHz cycles; a stall of the whole core; the takeover by the ownership token).

**Real processes (real time):** `python -m r1sim.dds_bridge` serves the simulated robot on DDS exactly as the robot's
board does (`rt/lowstate`, `rt/lowcmd` with Unitree's CRC, the Dex3 hands, the motion-switcher RPC, an emulated
built-in controller), and the real `r1rt` / `r1guard` binaries, `r1d`, `r1log` and the `r1` CLI run against it
unchanged. `r1sim.stack.ProcStack` starts and stops the whole stack for a test:

```python
from r1sim.stack import ProcStack
with ProcStack(domain=31, workdir=tmp, bridge={"stance": "centred"}) as st:
    st.cli("status")                    # r1d's typed result (dict), as the operator's CLI prints it
    st.cli("takeover"); st.wait_mode("balance", 10)
    st.bridge.tilt_deg()
```

Every DDS test uses its own domain id on the loopback interface only. Teardown never damps anything.

## Scenes

| Scene | What |
|---|---|
| `"floor"` (the default) | the robot on an empty floor |
| `"table"` | a plain table (0.6 x 1.2 m, top 0.78 m, its near edge 0.25 m in front of the feet) with a box, a can and a ball loose on it; `r1sim.scenes.table_entities()` gives the matching r1d `world.declare` entities, so a client can declare to the controller what the simulator holds |
| `None` / `"none"` | the robot only |
| your own | a builder function, or a `"package.module:attr"` string naming one in an installed package |

```python
from r1sim import world
w = world.build_world("real", scene="table")                     # a generic table scene
w = world.build_world("real", scene="mypkg.scenes:kitchen")      # your own scene, by name
```

A scene builder is `builder(spec, *, seed=None, visual=True, look="photo") -> info`: it receives the `mujoco.MjSpec`
(the robot already in it), adds bodies / geoms / cameras before the compile, and returns a scene-info object (its
objects and surfaces) that the world keeps as `World.scene`. The built-ins return `r1sim.scenes.SceneInfo` (name,
objects with shape / size / pose / mass, the table's top height and edge). The same string works for the
DDS bridge (`--scene mypkg.scenes:kitchen`) and the process stack (`bridge={"scene": ...}`), so a project keeps its task
scenes in its own package and the simulator never needs to know them.

The bridge's stances: `centred` (standing, the emulated built-in a stiff stand), `hanging` (the robot hanging on a
modelled stand: a bar in a cradle, a pole as a one-sided stop, feet ~8 cm up; the emulated built-in in zero torque or
Unitree's stand-lock preset) and `9oct` (the stance and arm pose of the previous controller's fall, for the stall tests:
[`case_studies/V2_FALL.md`](case_studies/V2_FALL.md)).

## Variants (ranges, not single numbers)

Every estimate in `models/provenance.yaml` carries a range with its source. `build_world(profile, seed=None)` = the
nominal model (bit-identical, the deterministic tests); `seed=N` = one variant drawn uniformly within the ranges,
reproducible (each field from its own generator seeded with (seed, the field's name), so adding a field never changes
the others' draws). For `real`: the rear module's mass and placement, the fists, the hand mount, the camera site, the
joints' friction / armature / damping, the hands' joints and contact friction, the sole and floor friction, the IMU's
mounting rotation. For `train`: the training's own randomisation. The robustness batch draws 100 of them
([`sim/ROBUSTNESS_BATCH.md`](sim/ROBUSTNESS_BATCH.md)).

## Rendering

- **MuJoCo's own renderer** (in the physics process; the default and the only source of depth).
- **The head camera's lens chain** (`r1sim.lens`): a wide pinhole render remapped through the robot's calibrated lens
  (pinhole + k1) to the stream's 544 x 448, vignetting, JPEG - checked within 0.25 px of the camera model on a dot
  grid. **The depth model** (`r1sim.depth`): the depth stream's noise and holes, fitted to recorded depth frames.
- **Filament** in a render-only worker (`render/`: its own venv on MuJoCo 3.15 / Python 3.12, because the physics stays
  pinned to 3.5.0): it compiles the same MJCF and gets the poses per frame; a test proves both compiled models give
  identical body poses. Optional; used for better-looking frames.

## Identified from the real robot

| Quantity | Identified value (our R1) | How | Doc |
|---|---|---|---|
| leg / waist joint friction | hips 0.35-0.72, knees 0.43-0.63, waist 0.41-0.52, **ankle pitch 2.1-2.5**, ankle roll 1.2-1.4 N m | slow soft-servo ramps on the stand | [`IDENTIFICATION.md`](IDENTIFICATION.md) |
| arm / head friction, armature, damping | shoulder pitch / roll 0.1, the other arm joints ~0.03, head pitch 0.1 N m; armature legs 0.008, arms / head 0.0025; damping elbows / head pitch 1.0 N m s/rad | replaying the recorded steps and chirps | [`sim/REPLAY.md`](sim/REPLAY.md) |
| command / state latency | 0-1 ms beyond the joints' dynamics / 0.5 ms | the same replay; the state age at r1rt | [`sim/REPLAY.md`](sim/REPLAY.md) |
| the board without commands | holds 0.385 s, then damps (kd 12.5) | the board-hold test | [`sim/BOARD_MODEL.md`](sim/BOARD_MODEL.md) |
| the IMU's mounting rotation | measured roll / pitch / yaw 0.56 / 0.82 / -0.39 deg (the model carries roll / pitch 0.50 / 0.96 deg: the standing IMU tilt beyond what the leg encoders imply, the median of six sessions) | waist sweeps on the stand | [`IDENTIFICATION.md`](IDENTIFICATION.md) |
| the stand | the pole's position fitted to the measured hang | the board-hold test's pose | [`sim/BOARD_MODEL.md`](sim/BOARD_MODEL.md) |

## What is not solid (said plainly)

- The floor contact (feet, slip, the standing friction) is not identified: the friction was measured hanging.
- The pads' model is kinder than the real pads; the fingers' heat model is fitted to short holds.
- Unitree's own balance controller is not modelled (the emulated built-in is a stiff stand fixture).
- The simulated real-time stack runs without real-time scheduling on a laptop: under load the physics can fall behind
  real time (every result reports its real-time factor).
- The `real` profile is OUR robot: masses, the hand mount, friction and the board's timing may differ on yours. Fit
  your own ([`IDENTIFICATION.md`](IDENTIFICATION.md)) before trusting a margin.
