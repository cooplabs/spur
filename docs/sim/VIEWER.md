# The simulator's viewer

A window on the simulated R1 (with its two Dex3-1 hands) that you can watch, push, jog and break - driven by the
controller's own code. It runs on Linux and macOS; it also runs without a display (frames, a JSON log and an MP4).

## Quick start

```bash
scripts/demo.sh                 # the one-command demo: a window, the R1 standing on the floor, balanced by the
                                # controller's real-time core and its hot standby, r1guard
scripts/demo.sh --headless      # no window: a scripted run -> PNG frames, viewer.json and an MP4 in out/demo
scripts/viewer.sh --scene table # the viewer with a plain table and a few loose objects
scripts/viewer.sh --scene stand # the robot hanging on a stand, feet off the floor (the stand mode, ON_STAND)
```

The first `scripts/demo.sh` builds the C++ core's Python binding (`r1rt_py`) if it is not built yet (about two
minutes; it needs a C++17 compiler - CMake and Ninja come with the project's Python environment). On macOS the window
needs `mjpython` (it ships with the MuJoCo wheel); `scripts/viewer.sh` sets it up. Do not start it through `nice` or
another wrapper that drops `DYLD_*` variables - macOS strips them and `mjpython` then cannot find Python's library.

## What drives the robot

| `--controller` | What runs |
|---|---|
| `guarded` (default) | the real-time core AND r1guard - the same C++ code as the `r1rt` and `r1guard` programs that run on the robot, bound to Python and stepped in lockstep with the physics. Deterministic; several times faster than real time on a laptop |
| `core` | the core alone (no hot standby: a stall is left to the motor board) |
| `policy` | the balance policy through the Python reference runner (when the C++ binding is not built) |
| `hold` | a stiff joint hold at the standing pose (balances nothing - a fixture for comparisons) |
| `none` | passive physics |

`--stack` runs the real programs instead: `r1rt`, `r1guard` and the API service `r1d` as separate processes, talking
to the simulator over the robot's own DDS topics in real time (the simulator's DDS bridge). It needs CycloneDDS's
Python binding (`uv sync --extra dds`; on macOS build CycloneDDS first with `rt/idl/build_cyclonedds.sh`) and the DDS
build of the C++ tree (`cmake -S rt -B build/rt_dds -DR1RT_WITH_CYCLONEDDS=ON`); the viewer says which one is missing.
In this mode the viewer asks `r1d` for the takeover (on the stand: for ON_STAND) when it starts, and the keys go
through the real interfaces (below). A desktop does not run these processes with real-time scheduling, so a loaded
machine slows the simulator a little; the overlay shows its speed.

## Scenes

| `--scene` | What |
|---|---|
| `floor` (default) | the robot on an empty floor |
| `table` | a plain table (0.6 x 1.2 m, top 0.78 m above the floor, its near edge 0.25 m in front of the feet) with a box, a can and a ball on it |
| `stand` | the robot hanging on a stand, feet 10 cm off the floor; the core enters its stand mode (ON_STAND) the way the API service asks for it on a real robot. In-process the stand is a rigid hook at the pelvis; with `--stack` it is the simulator's model of a real stand (a padded cradle and a pole) |
| `package.module:attr` | your own scene: a function `builder(spec, *, seed, visual, look)` that adds bodies to the MuJoCo spec (see `r1sim.scenes`) |
| `--model FILE` | any MJCF file (passive physics, no robot controller) |

`--world FILE` draws declared world entities (the API's `world.declare` format: surfaces, containers, objects,
obstacles) as translucent boxes, so you can check what a client told the planner against what is really there.

## Keys

| Key | What it does |
|---|---|
| `1`-`6` | cameras: free orbit, follow the pelvis, front, side, overhead, the robot's head camera (mouse: orbit / pan / zoom as usual in MuJoCo's window) |
| `P` | push the pelvis (60 N for 0.15 s; forward, back, left, right in turn) |
| `S` | stall: with `guarded`, the core (its writer and its policy thread) stops for 2 s - r1guard takes the motors within about 10 ms and keeps the robot balancing; watch "motors owned by". With `core`, the whole controller stops for 0.5 s. With `--stack`, the `r1rt` process is stopped for 2 s (SIGSTOP) |
| `U` | the operator's hand-back from r1guard to the core: refused while balancing (the controller does not offer a balancing hand-back), accepted on the stand |
| `N` `G` `D` `M` | faults: the policy outputs NaN (the core freezes into its stand pose - it never goes limp); the motor board gets no commands for 0.3 s; the robot's state arrives 40 ms late for 2 s; a motor reports an error word |
| `J` | the jog target: left arm, right arm, head, left hand, right hand |
| `[` `]` (left / right arrows) | the joint within the target |
| `-` `=` (down / up arrows) | move one step: arms 0.1 rad, head 0.05 rad, a hand's closure a quarter of the way to a fist. In-process this is a 0.4 s minimum-jerk trajectory through the core's upper-body executor - the same records the API service sends; with `--stack` it is the API's joint jog (`stream.joint_jog`) and hand pre-shape (`hand.preshape`). The core accepts arm and head motions while balancing or on the stand, and not while r1guard holds the motors |
| `O` | drop an object in front of the robot (onto the table in the table scene): a box, a can, a ball, a block, a cup, a puck; after six the oldest comes back |
| `C` `F` `W` `H` | contact points and forces, body frames, the declared-world boxes, the text overlay |
| `R` `.` `K` | record (PNG frames + an MP4 through ffmpeg), a screenshot, pause |

The overlay shows the time, the scene, the controller and the core's mode, who owns the motors (the core or r1guard),
the base tilt and pelvis height, the largest joint torque, the hands' pad readings, the jog target, the contacts, the
active faults and the last event.

## Without a display

```bash
scripts/run.sh python -m r1sim.viewer --headless --scene table --frames 60 --every 0.1 \
    --keys 'P@1.0,J@2.0,=@2.1,S@4.0' --mp4 --out out/viewer
```

`--keys 'K@t,...'` presses the window's keys at simulated times. The output folder gets `frame_*.png`, `viewer.json`
(every frame's time, overlay and image statistics, the events) and `viewer.mp4` when ffmpeg is installed.

## Batch runs with seeds

```bash
scripts/run.sh python -m r1sim.viewer.batch --seeds nominal,0-9 --duration 10 --pushes 3 --push-n 40,90 --out batch.json
scripts/run.sh python -m r1sim.viewer.batch --seeds 0-19 --keys 'S@4.0' --fail-on-fall
```

Each seed is one robot variant inside the model's parameter ranges (masses, centres of mass, friction, delays - every
range with its source in `models/provenance.yaml`), with the key schedule and random pushes drawn from the same seed
(any direction, a force in the given range). Per seed the summary says whether the robot fell (the pelvis below
0.45 m or the base tilted beyond 30 degrees) and when, the largest tilt, the core's modes in order, r1guard's
takeovers and every event; `--mp4-dir` adds a video per seed. Runs are in lockstep, so the same seeds give the same
numbers on any machine. From Python: `from r1sim.viewer.batch import BatchConfig, run_batch`.

A pass here is the simulator's verdict on a model identified from one robot, not a guarantee for yours: see what is
and is not modelled in `models/provenance.yaml`.
