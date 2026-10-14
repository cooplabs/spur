# Pointing and measurement

[`../DESIGN.md`](../DESIGN.md) §9. Code: `python/r1mc3/cam/` (`r1cam`, pointing, extents), `python/r1mc3/measure/`
(tags, lenses, registration, the phone correction, pose checks), `python/r1mc3/service/perception.py` and
`measure_ops.py` (the `camera.*` / `measure.*` ops).

## Pointing: head-image pixels -> points in `base`

| Step | What |
|---|---|
| observe | `r1cam`'s newest head frame (+ depth on demand) -> an `obs_id` kept with the camera's pose at capture (the robot's joints and IMU at that moment, through the planner's geometry) |
| point | the pixel's calibrated line of sight meets the declared face it looks at (every surface top, every container floor); depth says how high the clicked point is above it; the reply names the surface it hit; no usable depth -> `NO_DEPTH` in words |
| object_extent | from the depth: the connected region above the local face, cut at 3-D jumps; footprint box + hull, heights, a shape class (round / box / tall / flat), a confidence; the robot's own hands excluded |
| project / depth_image | base points -> pixels of a kept frame; the frame's depth |

The calibration file (`/etc/r1mc3/head_camera.json` on the robot; an example in `models/head_camera/`) holds the lens
model and the camera's mount on the head; pointing is refused without it (`NOT_CALIBRATED`).

**What we saw on the robot**: the clicked pixel's depth can read the surface BEHIND a thin, clear or small object, and
extents from depth come out larger than the object (mixed pixels at the edges). The ops report what they used (which
surface, the depth source); a client that knows an object's size should pass it to the goals rather than trust the
extent.

## Measuring a set-up with tags

| Module | What |
|---|---|
| `r1mc3.measure.tags` | AprilTag 36h11 detection with edge refinement (sub-pixel corners) |
| `r1mc3.measure.lenses` | the head camera's lens model; OpenCV's model for a phone |
| `r1mc3.measure.register` | `locate_entity`: every corner of every tag of an entity, seen in one or more views (each with the camera's pose at capture), in one least-squares fit -> the entity's pose in the base frame, its fit error, and a refusal when the fit is not precise enough |
| `r1mc3.measure.phone` | a phone still's own pose from tags on the table plane (the phone does not need to sit perfectly), pixels -> the table plane or a plane above it; its error in mm |
| `r1mc3.measure.preflight` | pose checks against nominal poses with tolerances (refusing "not precise enough" when the fit's 1-sigma exceeds half the tolerance), the report in words |

`r1d`: `measure.setup` (the head measures a plan's entities; hold the head still), `measure.phone` (a still and the
phone's calibrated lens: `python -m r1mc3.measure.phone_calib <photos> --out phone.json` with a printed ChArUco
board). The plan - which entities, which tags where, the tolerances - is the client's.

In the simulator (ideal renders): a table's pose from its 4 tags within 0.1 mm / 0.01 deg; a phone placed 2-3 cm off
and tilted 6 deg still maps table points within ~1.2 mm. Measure the set-up before placing objects: they hide tags.
