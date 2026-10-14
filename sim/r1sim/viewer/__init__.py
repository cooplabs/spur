"""r1sim's viewer: a window on the simulated R1 that people can enjoy and debug with (DESIGN D32).

    scripts/demo.sh                                               # the one-command demo (a window)
    scripts/viewer.sh --scene table                               # the viewer (macOS: mjpython handled)
    scripts/run.sh python -m r1sim.viewer --headless --frames 30 --mp4    # frames + viewer.json, no display
    scripts/run.sh python -m r1sim.viewer.batch --seeds 0-9 --pushes 3    # robustness runs with seeds

What it is: MuJoCo's own window (`mujoco.viewer.launch_passive`: mouse zoom / pan / orbit, its panels) on a GENERIC
scene - `floor` (the robot on an empty floor, the default), `table` (r1sim.scenes' plain table with a box, a can and a
ball), `stand` (the robot hanging on a stand, feet off the floor) - or any MJCF file (`--model`), or another package's
scene builder by its "package.module:attr" string; with the R1 + both Dex3-1 hands driven by THE CONTROLLER'S OWN CODE:
`--controller guarded` (the default) = the C++ real-time core AND r1guard (`r1rt_py`: the code of the r1rt / r1guard
binaries) on one shared segment, in lockstep with the physics; `core` = the core alone; `policy` = the Python reference
policy; `hold` = a stiff stand fixture; `none`. On the stand the core enters ON_STAND through its mode mailbox, as r1d
does on the robot. `--stack` runs the REAL processes instead (r1rt, r1guard, r1d against the DDS-bridge simulator in
real time: `r1sim.viewer.stack`; needs CycloneDDS). Why MuJoCo's passive viewer: the window, the mouse camera and the
panels come with the pinned physics wheel (no second process, no second venv); the photo look of the head camera is
the render worker's job.

The keys (the window; headless: `--keys 'P@1.0,C@0'`): `1-6` camera presets (free, follow the pelvis, front, side,
overhead, the robot's head camera); `P` push the pelvis (60 N for 0.15 s, forward / back / left / right in turn);
faults through the simulator's injector (r1sim.faults): `N` the policy's action NaN (-> the core's STAND_FREEZE), `G` a
command gap (the board model holds ~0.4 s, then damps), `D` the state 40 ms late for 2 s, `M` a motor error word on
the left knee, `S` a stall (guarded: the core - writer and policy - for 2 s, r1guard takes the motors and keeps them;
otherwise the whole controller for 0.5 s); `U` the operator's hand-back from r1guard to the core (refused while
balancing - not offered; allowed on the stand); jog: `J` the target (left arm, right arm, head, left hand, right hand),
`[` `]` (the left / right arrows) the joint, `-` `=` (the down / up arrows) one step - a 0.4 s minimum-jerk TRAJECTORY
through the core's upper-body executor (what r1d sends), a hand's closure toward the fist; `O` drop an object (a pool of
generic shapes, onto the table in the table scene); `C` contact points + forces, `F` body frames, `W` the declared world
boxes (`--world`: r1d's world.declare entities, translucent), `H` the text overlay (time, the controller and the core's
mode, who owns the motors, the base tilt, the pelvis height, the largest joint torque, the hands' pads, the jog,
contacts, active faults, the last event), `R` record (PNG frames + an MP4 through ffmpeg), `.` a screenshot, `K` pause.
"""

from .app import CONTROLLERS, FAULT_KEYS, HELP, PRESETS, SCENES, App, ViewerConfig, load_entities

__all__ = ["CONTROLLERS", "FAULT_KEYS", "HELP", "PRESETS", "SCENES", "App", "ViewerConfig", "load_entities"]
