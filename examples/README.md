# examples/

Three programs that drive the controller through its API, in the simulator. Each one is a single command, prints
what it does in plain words and exits 0 when it did what it says.

| Example | What it shows | One command | What you should see | Time (macOS, a busy machine) |
|---|---|---|---|---|
| [`pick_and_place.py`](pick_and_place.py) | a pick-and-place through r1d's goal API: the world model, `arm.hover` / `arm.lower` / `hand.grasp` / `arm.carry` / `hand.release`, the grip verdict | `bash scripts/run.sh python examples/pick_and_place.py` | the robot takes over from the simulated built-in controller, the left hand picks the 5 cm box off the table and sets it down 12 cm to the right; the box ends within 4 cm of the target (in the hand it arrives within ~5 mm; opening the fingers nudges it) | ~25 s (real processes), 5-20 s (`--lockstep`) |
| [`jog.py`](jog.py) | teleoperation through the streaming API: a hand in Cartesian space (`stream.arm_twist`), an elbow joint (`stream.joint_jog`), the head (`stream.head`), the fingers (`hand.preshape`) - each a dead man's command with a time to live | `bash scripts/run.sh python examples/jog.py` (keys) or `... jog.py --script` | the keys move the hand (w s a d r f), the elbow (e q), the head (i k j l), the fingers (c o); `--script` runs a fixed sequence and prints how far each hand / joint moved | ~30 s (`--script`, real processes), ~8 s (`--script --lockstep`) |
| [`robustness_batch.py`](robustness_batch.py) | a robustness batch for the balance controller: many simulated robots (model parameters drawn within their identified ranges), pushed, with faults, the real C++ core balancing each | `bash scripts/run.sh python examples/robustness_batch.py` | a table: push size x fault -> falls / runs, the largest tilt, how far the feet moved, the core's mode at the end | ~4 min (72 runs, 2 workers), ~10 s (`--quick`) |

`common.py` is what they share: the simulator's built-in "table" scene (`r1sim.scenes`: a 0.6 x 1.2 m table, top
0.78 m, its near edge 0.25 m in front of the feet, a box, a can and a ball on it) and the two ways to run the
controller against it.

## Two ways to run: the real processes, or lockstep

* **The real processes (the default)** - what runs on the robot. The DDS-bridge simulator stands in for the robot
  (`rt/lowstate`, `rt/lowcmd`, the Dex3-1 hands, Unitree's motion switcher, its built-in standing controller), and the
  real `r1rt` (the 500 Hz real-time core), `r1guard` (the hot standby) and `r1d` (the API service) run as separate
  processes against it, talking DDS on the loopback interface. The example starts them, does the operator's takeover
  (`takeover`: the simulated built-in controller lets go and r1rt balances), and stops them all at the end (the
  simulator simply stops - nothing is damped). Real time: the timings wobble on a busy machine.
* **`--lockstep`** - one process, one simulated clock: the simulator, the SAME C++ core (`r1rt_py`, the code `r1rt`
  runs), r1d's service and the simulated hands, stepped together (`r1sim.hands_rig.ServiceRig`). Reproducible for a
  seed, no DDS needed.

Both hand the example an `r1mc3.client.R1` (the typed SDK: one method per API op), so the example code cannot tell
them apart. The robustness batch always runs in lockstep (it simulates many minutes of robot time).

## What you need

1. The Python environment: `uv sync` (Python 3.11, MuJoCo 3.5.0 and the rest, pinned by `uv.lock`).
2. The C++ core and its Python binding `r1rt_py` - every example needs it: `bash scripts/ci.sh cpp` (builds `rt/`
   into `build/rt` and runs its tests; a C++17 compiler; CMake and Ninja come with the venv).
3. For the real-processes mode only (`pick_and_place.py` and `jog.py` without `--lockstep`):
   * CycloneDDS 0.10.2 (the version Unitree's message types were generated with): `bash rt/idl/build_cyclonedds.sh`
     (about a minute; installs into `~/.cache/r1mc3/cyclonedds-0.10.2`, nothing system-wide);
   * the DDS build of `r1rt` / `r1guard`: `bash scripts/ci.sh dds` (into `build/rt_dds`);
   * CycloneDDS's Python binding: `scripts/run.sh` adds the `dds` extra by itself once that build exists (it sets
     `CYCLONEDDS_HOME`) - which is why the commands above go through `bash scripts/run.sh`.

Without step 3 the real-processes mode says so and stops; `--lockstep` works.

## Options

* `--lockstep` (pick_and_place, jog), `--seed N` (the simulator's sensor-noise seed), `--domain N` (the DDS domain
  of the process stack: give each concurrent run its own), `--json FILE` (pick_and_place: every step's reply code and
  the box's start / target / end).
* `robustness_batch.py --seeds N --impulses 4 8 12 --faults none cmd_gap state_delay policy_nan --workers N
  --json FILE`.

## What the numbers say (and do not say)

The pick-and-place in the simulator, 14 Oct 2026: 5 / 5 runs on the real processes (the box 1.3-2.8 cm from the
target) and 5 / 5 seeds in lockstep (2.0-2.7 cm). The default robustness batch (8 robots): no fall from a push alone
or after a 0.3 s command gap, up to 12 N s; with the robot's state arriving 40 ms late while the first push lands, 3 / 8
fell at 8 N s and 7 / 8 at 12 N s - a known limit of the balance policy with a late state (the core's optional
state-age rung, off by default, holds the stand instead of balancing on an old state). These are simulator numbers for
one identified R1: a push in the simulator is not a push on your robot, and a grasp that works on a simulated box is a
starting point, not a promise for your objects.
