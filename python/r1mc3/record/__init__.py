"""The HW1 recording tools (session 5, DESIGN.md §13.2 Tier B, §14.1 step 5): scripted, dry-runnable plans executed
through r1d with the robot on its stand (ON_STAND), while r1log records the 500 Hz telemetry.

* `plans`  - what each recording does, as a pure function of its parameters and the start pose: per-joint steps and
             chirps, static poses (+ payload), IMU statics, pad presses with known weights, finger holds at 0.9 /
             0.45 N·m. `plan(kind, params, start_q)` -> a Plan (setpoints over time + checks + a summary).
* `runner` - `Recorder`: r1d's `record.list / record.plan / record.run` ops; runs one plan at a time in r1d's tick
             (STREAM setpoints with a short TTL - if r1d stops, the arm / leg stops), writes a JSON manifest (what
             moved, when, in both clocks, the r1log files of the window).
"""
