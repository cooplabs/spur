# Architecture notes

Short references for the pieces of [`../DESIGN.md`](../DESIGN.md). The code is the source of truth; each note names
the files.

| Note | What |
|---|---|
| [`PROCESSES.md`](PROCESSES.md) | `r1rt` and `r1guard` as processes: command lines, threads, priorities, the DDS layer, output lines, signals |
| [`GUARD.md`](GUARD.md) | the hot standby: the warm shadow, the takeover, what it does by mode, forensics, reclaim |
| [`EXECUTOR.md`](EXECUTOR.md) | the arm / head (and on the stand, leg) motion executor: sources, deadmen, guards, the stop profile, limits |
| [`SHARED_MEMORY.md`](SHARED_MEMORY.md) | the shared-memory segment: regions, who writes what, the ownership token, the stop channel, the mode mailbox, the step API |
| [`LOGGER.md`](LOGGER.md) | `r1log`: MCAP topics, drops, the flight recorder, the session save |
| [`PLANNER.md`](PLANNER.md) | the world model, the one geometry engine, the planner pipeline, the orientation search |
| [`GRIP.md`](GRIP.md) | the grasp sequence and the evidence-based holding verdict |
| [`PERCEPTION.md`](PERCEPTION.md) | pointing from head-image pixels, object extents, measuring a set-up with tags |
| [`ARM_MODEL.md`](ARM_MODEL.md) | the arm gravity model and fingertip kinematics inside the core |
