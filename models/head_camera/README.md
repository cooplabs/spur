# The R1 head camera's calibration (our robot)

`calibration_s3_cal_20261004.json` = our 4 Oct capture's `runs_calib/s3_cal/calibration.json` (sha256
128e96aa4a22ad2cba4da14ed1943ac7c1debb8b0a96280e933ac87698fbbc4d), copied unchanged on 10 Oct 2026 (v3 session 3).
Fitted by v2's head-calibration tool (`calib_head.py solve`) on 4 Oct 2026 from 16 taped marks on a table (2 raised), robot in STAND
at the table, head pitched to the task pose: pinhole + one radial term k1, f 126.94 px, centre (259.0, 203.9), k1 0.0249,
544 x 448 (the left stereo image as the robot sends it); `T_site_cam` = the camera (OpenCV frame: x right, y down, z
forward) in the `head_camera` site's frame (v2 model.py: on head_yaw_link at (0.08, 0, 0)). "16 marks, left-out median
3.3 mm, worst 8.3 mm". This is what v2 and its task client used for pointing. The lens model (pinhole + k1 vs
equidistant) is to be settled by the HW1 full-field ChArUco capture (DESIGN §12.5, B5).
Used by `r1sim.lens` (the simulated head camera: wide pinhole render -> remap -> 544 x 448 -> JPEG q90).

`calibration_s3_cal_20261004_v3frame.json` (session 10, the simulator's camera-mount fit): the SAME
capture's mount re-fitted in v3's base convention (`r1mc3.world.Geometry.set_stance`: the IMU's mounting rotation
undone - v2 took the IMU as the pelvis, and the 4 Oct `T_site_cam` absorbed that ~1.1 deg). The lens model is unchanged.
On the 16 marks: v3's pose with the OLD mount 4.43 px RMS / pointing 12.2 mm median; with the re-fitted mount 0.65 px /
2.3 mm, left-out median 3.0 mm (v2's fit: 0.69 px, 3.3 mm). **This is the file for the robot's
`/etc/r1mc3/head_camera.json` (r1d's pointing, measurement)** until HW2's S12 / mount fit replaces it.

**Session 11 (the user's decision, 12 Oct ~03:05): the simulator renders through the v3-frame file too**
(`r1sim.lens.CALIBRATION_FILE`); the 4 Oct file stays as `CALIBRATION_FILE_V2_FRAME`, used ONLY with a v2-logged
`T_base_site` (A5's "logged" camera and its left-out beliefs, A6's capture rebuild). On the process stack r1d is given
this very file (`ProcStack(camera=True)`: R1MC3_HEAD_CALIB). sha256 of the file: `f18fe402227a76671e8594d442007995dac294c7be3c9293dd82f82a32ef0c80`;
r1d's fingerprint of its numbers (`r1 status` data.camera.calibration.sha256): `03c007b72056...`. A `"latency_s"` key
(HW2's latency step) is read by r1d as the camera's exposure -> receive latency.

**Session 12 (after HW2 day 1's C3):** `calibration_c3_20261012_candidate_v3frame.json` = the C3 ChArUco lens (the plan's
hold-still frames: f 125.063, cx 257.698, cy 205.257, k1 -0.0007, RMS 0.389 px; B5 met) + an INTERIM mount (the 4 Oct
16 marks re-fitted under that lens: 1.16 px, left-out 4.1 mm), `"stream": "stereo_patch_pc1/left"`; sha256
`943cf9fcb2fe81386bee26db690643334661672133eca06cd6a2c870ba213dd0` (session R reworded its "source" text; before:
`02c50f8570cc24f94aa1117bdd5215151d24cdf36d137da7d112401b77a412d4`, the numbers unchanged). **NOT installed** - the installed file stays for HW2
day 2: day 1's "f 150.05 = rectified stream, 18 % off" was the lens fit stuck at its starting guess (one degenerate
view); the converged C3 lens is within 1.6 % / ~1.5 px of the installed one, and on the table at the working view the two
files agree within 3.5 mm median (11 mm max). The proper new file: this lens + a mount fitted on new data with
`python -m r1mc3.measure.head_mount` (a still board, a grid of head poses), then the user's OK.
