# docs/sim/ - the simulator in depth

The overview is [`../SIMULATOR.md`](../SIMULATOR.md).

| File | What |
|---|---|
| [`VIEWER.md`](VIEWER.md) | the interactive viewer and headless frames |
| [`DDS_BRIDGE.md`](DDS_BRIDGE.md) | the simulated robot behind the robot's own DDS interface: the real binaries against it; the stand; scheduled faults; the process stack |
| [`FAULT_TESTS.md`](FAULT_TESTS.md) | every fault involving the real-time core, the guard or ON_STAND, against the real processes - and the results |
| [`BOARD_MODEL.md`](BOARD_MODEL.md) | what the real motor board did when commands stopped, and the simulator's identified model of it |
| [`ROBUSTNESS_BATCH.md`](ROBUSTNESS_BATCH.md) | the go / no-go for balancing: the policy on 100 identified variants x 4 cases; where its margin is |
| [`REPLAY.md`](REPLAY.md) | the stand recordings replayed command for command: joint dynamics, latency, the IMU's mounting rotation |
| [`STATE_AGE.md`](STATE_AGE.md) | the content age of the robot state, the STATE_OLD alarm and the optional "hold the stand" rung |
