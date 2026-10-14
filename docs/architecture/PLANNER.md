# The world model and the arm planner

[`../DESIGN.md`](../DESIGN.md) §8. Code: `python/r1mc3/world/` (entities, the world model, the one geometry engine),
`python/r1mc3/plan/` (IK, the orientation search, paths, timing, the pipeline, the worker, a speed check). Tests:
`tests/unit/test_world_*.py`, `tests/unit/test_plan_*.py`. Generic by construction: clients declare their tables,
containers and objects through `world.declare`.

## The world model

| Piece | What |
|---|---|
| entities | `Surface` (a box whose top is the surface: pose = the top face's centre), `Container` (walls + a floor; a hand may be INSIDE above the floor), `Obstacle`, `Object` (box / cylinder / sphere) |
| frames | `base` only: the feet's midpoint on the lower sole, x along the feet's mean yaw, y left, z up |
| `WorldModel` | `declare` (replaces the same id) / `remove` / `clear` / `snapshot()` -> an immutable snapshot; thread-safe |
| `Geometry` | ONE MuJoCo model: the simulator's robot MJCF (collision geoms) + the snapshot's entities as static geoms. `set_stance(q26, imu_quat, hands)` poses the measured robot so its base frame IS the model's world; the IMU's own mounting rotation in the pelvis is taken out (reading the IMU as the pelvis puts the hands 1-2.5 cm off) |

## The pipeline

`plan(snapshot, PlanGoal, q26, imu_quat=, hands=, ...) -> PlanResult` is a pure function: nothing moves until the
WHOLE goal is planned. `PlannerWorker` runs it in a separate process (a dead worker is replaced at the next request).

| Step | How |
|---|---|
| target | 5-DoF IK (damped least squares): the hand point first, the approach direction in the position task's null space; an orientation search over elbow swivel x wrist roll for hovers |
| the orientation search | candidate hand poses clear of the world, ranked by the request: a finger-heading family (`front` / `inward` / `outward` / `across`), the closing line (`grasp_yaw_deg`), an approach direction, an approach style (`around`: the palm turned to a tall object; `low`: the fingertips over a flat object's bottom), a level closing line (within 12 deg of level costs nothing) |
| path | a straight Cartesian line of the hand point in 1 cm IK steps, else RRT-Connect in joint space (seeded: reproducible) + random shortcuts; every candidate checked densely against the robot itself, the other arm, the world and the balance envelope |
| timing | knots for the executor's quintic spline (cruise times per segment, bounded knot velocities - no overshoot past a waypoint) |
| whole goals | `hover` = lift + transit (clearing what is under the path by 6 cm) + descend to the chosen pose; `carry` = rise + travel + lower; `lower` = a straight descent; `move_path` = every leg planned from the previous leg's planned end |
| result | `PlanResult(ok, code, plan_id, segments, predicted, clearance_m, reason, details, planning_s)`: one segment per leg = one executor trajectory |

**The grasp aim** is where the chosen grip's fingers actually close (the hand model swept from open to the grip's
closed preset at the object's width), not the open hand's midpoint. **A held object** is planned as a sphere at the
grasp centre, so a carry keeps it clear of the world like the hand.

**A start inside the clearance** (an arm already close to a declared entity) may leave it, as long as no checked pose
comes closer. `skill.table_ready` plans the way out from under a declared table (fingers closed, out and around).

## Speed

`python -m r1mc3.plan.bench --n 20` (a generic cluttered scene): on a laptop (Apple M4, under load) a hover plan
takes ~0.26 s of CPU (p50), a reach the same. The Orin's single-thread Python is roughly 3-4x slower: a hover there is
at the 1 s target, not safely under it.
