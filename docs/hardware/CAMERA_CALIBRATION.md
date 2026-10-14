# Calibrating the head camera

r1mc3 uses the R1's head camera (Unitree's stereo service: a left image, a right image and a depth stream) through
`r1cam`, a separate service that r1d asks for observations (`r1 observe`, `r1 point`, `r1 extent`) and recordings. The
camera model: a pinhole + one radial term (k1) for the left image (544 x 448), our lens f ~127 px (~130 deg across: a
wide lens), the mount on the head from the robot's own joint angles at the capture time, a vignetting model and a
fitted depth-noise model. The files: `models/head_camera/` (the calibration example of our robot and `README.md`); on
the robot `/etc/r1mc3/head_camera.json` (`r1 status` shows its sha256 and whether it is calibrated).

## The board

Print [`charuco/r1mc3_charuco_poster_6x4_90_20x30in.png`](charuco/r1mc3_charuco_poster_6x4_90_20x30in.png) (or the
vector [`.pdf`](charuco/r1mc3_charuco_poster_6x4_90_20x30in.pdf)): a ChArUco board of **6 x 4 squares of 90.0 mm with
68.0 mm ArUco markers (OpenCV DICT_4X4_50, ids 0-11)**, grid 540 x 360 mm, centred on a 20 x 30 in poster (9000 x
6000 px at 300 dpi), with >= 50 mm of white all round.

1. Print at **actual size** (no scaling, no "fit", no crop), black on white, matte or satin (glossy paper glares).
2. **Check the scale** with a ruler: one square = 90.0 mm; along the top row from the left edge of the first black
   square to the right edge of the third black square = **450.0 mm (+-0.5)**; the bar under the board = 100.0 mm.
   Reprint if off by more than 0.5 mm.
3. Glue it flat on a rigid board (foam board, spray adhesive, smoothed from the middle out, weighed flat for a few
   hours); a ruler on edge across it must show < ~1 mm of gap. No lamination. Write the measured 450 mm on the back.

**Why this size:** the head lens is so wide that at 0.6 m a 40 mm square spans ~8.5 px and its marker ~1 px per cell,
below what ArUco detection needs. We rendered candidate boards through the simulated head-camera chain (wide render ->
lens remap -> vignetting -> JPEG -> decode) at 0.3 / 0.45 / 0.6 m x 9 places x 3 tilts and re-fitted the lens from the
detections: `python -m r1sim.charuco_render` reproduces the comparison. The 90 mm board was the smallest that still
covers the image at 0.6 m.

## The procedure (the robot on its stand, ON_STAND; only the head moves)

First make sure the camera does not slow the motor link on your robot ([`NETWORK.md`](NETWORK.md): `tools/cam/impact.py`).

| ID | Where | Words | What moves |
|---|---|---|---|
| C0 | backpack, workstation | `sudo systemctl start r1cam`; `r1 camera services` (read-only: which of Unitree's camera services runs); if needed `r1 camera services --stereo --yes` (Unitree's stereo service on, video_hub off; on our robot this reverts at every boot) | nothing |
| C1 | workstation | `r1 observe`; `r1 --pretty status` (`data.camera`) | nothing |
| C2 | workstation + a person | the head to a still pose (`r1 head 0.57,0 --ttl 0.3`, repeated - a streamed word: it holds where it is when the time-to-live ends); `r1 record plan charuco` (325 s: 0.30 / 0.45 / 0.60 m x 9 places x 3 tilts, 2 s still + 2 s move each), then `r1 record run charuco --yes`; the person presents the board at each prompt; `r1 camera status` shows the live count ("hold still (0.45 m, centre, flat): 2/8" / "next: ..."); a spoken prompter for a second terminal: `python -m r1mc3.cam.prompter` | the head (to the pose; then nothing) |
| C3 | workstation | the fit on the hold-still frames: `bash scripts/run.sh python -m r1sim.sysid.camera_fit lens --mcap <stamp>_cam_charuco.r1cam.mcap --still --plan <stamp>_charuco_.json` -> per lens model the RMS (target <= 1 px), leave-one-out, BIC, f / cx / cy / k1, and **`converged`** | nothing |
| C4 (optional) | workstation + a person | vignetting: a plain matte white panel filling the view, `r1 record plan flat_field` / `run flat_field --yes` (3 light levels x straight / turned), then `camera_fit vignetting --mcap <file>` | nothing |
| C5 (optional) | workstation + a person | depth: a flat target at tape-measured distances, `r1 record run depth_targets label=board_0p45 distance_m=0.45 --yes` (and 0.30, 0.60, other surfaces), then `camera_fit depth --mcap <files>` -> the valid fraction, the median error and the noise per distance band | nothing |
| C6 (optional) | workstation + a person | the latency: the board still at ~0.6 m, `r1 record run depth_targets label=latency_board distance_m=0.6 duration_s=20 --yes` while the head steps +-0.08 rad every ~1.5 s, then `camera_fit latency --mcap <file> --r1log <the r1log files of that minute>` -> `latency_ms`; add `"latency_s"` to the robot's calibration file (r1d's pose at capture subtracts it) | the head yaw, small steps |
| C7 (optional) | workstation + a person | the mount on the real camera: the board propped still ~0.5 m in front of the head, `python -m r1mc3.measure.head_mount capture --out <dir> --yes` (12 head poses in a serpentine, ~30 s), then `python -m r1mc3.measure.head_mount fit <dir> --calib <a calibration file>` -> `file_as_is_rms_px` (the file exactly as it is) and `rms_px` (its lens, the mount re-fitted) | the head only |

Install a new calibration file on the robot only deliberately (it changes where the robot thinks things are): keep the
old file, write the new one to `/etc/r1mc3/head_camera.json`, restart r1d, check `r1 status` shows the new sha256.

## What our robot showed (and a trap)

- **A fit stuck at its starting guess looks like a real result.** Our first fit of a 325 s recording read f 150.05 px
  with the principal point at the image's exact centre and an RMS of 1.14 px - it looked as if the stream were
  rectified and ~18 % off our installed file. It was OpenCV's `calibrateCamera` stopped at its initial values by one
  degenerate view (6 corners along a line). The converged fit of the same recording: **f 125.06 px, RMS 0.39 px** -
  within 1.6 % of the installed lens, and within 3.5 mm median (11 mm max) at a table in front of the robot. Read
  `converged` and the principal point before believing a fit.
- On the robot the live coverage counter first read 0 frames: its JPEG decoder was a development-only dependency,
  absent from the release; it decodes with OpenCV now. If the count stays at 0, the plan's own timeline (2 s still, 2 s
  move) still works with the spoken prompter.

## External cameras

`python -m r1mc3.measure.phone_calib <photos> --out camera.json --report report.json` fits a lens model for any still
camera (10-20 photos of the board at full resolution, the board in different places including the image corners,
tilted up to ~40 deg; refused with fewer than 6 usable photos, mixed sizes or an RMS > 1.5 px). r1d's `measure` words
can then use such a camera's stills for measuring a setup with fiducial tags (the API docs).
