"""r1cam: the head camera process (DESIGN §8.1, §2.1; docs/plans/R1CAM_PLAN.md).

The R1's head stereo module: the left / right images (RTP H.264, 544 x 448, ~7.5 FPS) and the native depth (TCP,
uint16 mm, ~10 FPS). r1cam serves them to r1d (127.0.0.1:8781, newline JSON), records them (MCAP + a manifest per
recording), and stays OUT of the control's way (the user's warning, 11 Oct: the camera has slowed other things
before - v2 5 Oct 2026: the always-on depth stream (~39 Mbit/s) on the backpack's 100 Mbit/s link to the robot made
the control loop overrun): normal priority, nice, CPUs 0-4 only (deploy/units/r1cam.service), every stream ON DEMAND
(stopped `idle_s` after the last request), newest-frame slots and bounded queues that DROP frames instead of blocking.
No wrist cameras (the user, 11 Oct: Q1). The Unitree camera-services switch is an explicit operator request only.

Modules: `frames` (Frame, the dropping slots / queues), `robot` (the depth reader, the GStreamer streams, the pipelines,
RobotSource), `sim` (SimSource: frames from a render callback), `recorder` (MCAP), `services` (the camera-services
switch over Unitree's robot_state RPC), `service` (the requests + the TCP server; `python -m r1mc3.cam`).
"""
