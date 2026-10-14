# s3_cal (4 Oct 2026): the head camera calibration capture, for the lens tests

From our 4 Oct 2026 calibration capture `runs_calib/s3_cal/` (read only): `calibration.json` (sha256 128e96aa...bbc4d, unchanged), `marks_used.txt`
(sha256 7c0e5f36...ff05, unchanged: "x_cm,y_cm[,z_cm]:pixel_x,pixel_y" in table cm - x from the table's near edge, y to
the robot's left, z above the table top - and the clicked pixel), `head_pose.json` = the keys obs_id, width, height,
T_base_cam, T_base_site, joints, status of `head.json` (sha256 20ad85ef...bb35; the depth / image payload left out).
Extraction: `cp calibration.json marks_used.txt` + the key subset of head.json (tests/unit/test_lens.py docstring).
