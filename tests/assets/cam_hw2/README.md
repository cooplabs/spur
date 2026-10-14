# HW2 day 1 camera data (12 Oct 2026), small excerpts for tests (tests never read the session data in place)
| file | source | sha256 | how cut |
|---|---|---|---|
| cam_impact_20261013.json | our robot's HW2 day-1 records: cam/cam_impact.json (C1, the robot, 60 s windows) | 7080cd456eba06ab91511dd6fd5a1f19688af71335170a9cd124ea78d17556af | verbatim copy (6.3 KB) |
| c3_charuco_detections.npz | the ChArUco corners (ids + pixels + receive times) detected in every left frame of the HW2 day-1 recording cam/20261013T032404Z_cam_charuco.r1cam.mcap (C3, 325 s, 402 MB, sha256 59c650bf410e33fc...) with >= 6 corners: 1935 frames | 9e2dcaef28c2d73424eb267dc4d5a37ad3a45efb714c3435efee384cb9952d2b | `r1sim.sysid.camera_fit.save_detections(detections_timed(mcap, poster_6x4_90))` (180 KB) |
| c3_charuco_run.json | the HW2 day-1 record run's manifest stand_c/records/20261013T032404Z_charuco_all.json (its: its start + the plan's prompted holds) | a39a0f29b67cc0ebb3f7ef4cc5351db1c7bbbb2d529921a0beb433442f9b81d9 | verbatim copy (17 KB) |
