# docs/hardware/ - running r1mc3 on a real R1

| File | What |
|---|---|
| [`BRING_UP.md`](BRING_UP.md) | **start here**: the first bring-up on your R1 - what you need, the install, the read-only checks, the hands, the robot on its stand (ON_STAND, the joint-direction check, the board-hold and guard-takeover tests), the first balancing, what bit us |
| [`JOINT_DIRECTIONS.md`](JOINT_DIRECTIONS.md) | what a POSITIVE value does on the real robot, for all 26 joints (verified by eye and by gravity signs) + the re-check tool |
| [`STAND_LOCK.md`](STAND_LOCK.md) | Unitree's stand lock (a fixed preset pose) vs r1mc3's stand-session lock; lifting the lock before the floor |
| [`DEPLOY.md`](DEPLOY.md) | install / update / rollback on the backpack; the core's config; the site settings; "stopping never damps" |
| [`RT_SETUP.md`](RT_SETUP.md) | the real-time set-up (CPU partition, RT budget, clocks) and the RT-grant / RT-latency / policy-timing checks |
| [`NETWORK.md`](NETWORK.md) | the robot's network, the two CycloneDDS files, and what not to do on the backpack |
| [`CAMERA_CALIBRATION.md`](CAMERA_CALIBRATION.md) | the head camera: the ChArUco board ([`charuco/`](charuco/)), the lens / vignetting / depth / latency recordings and fits |
| [`E2FSCK.md`](E2FSCK.md) | checking the backpack's root file system (and why the easy way does not work on L4T R35.3.1) |

The methods and our robot's numbers: [`../IDENTIFICATION.md`](../IDENTIFICATION.md); the simulator that rehearses all of
it: [`../SIMULATOR.md`](../SIMULATOR.md).
