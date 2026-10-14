# models/

The robot and scene descriptions the simulator (`sim/r1sim`) assembles, and where every number comes from.

| Path | What | Owner of the numbers |
|---|---|---|
| `r1/` | Unitree's R1 MJCF (`r1_mjlab.xml`, unitree_rl_mjlab@1425b15) + meshes, Apache-2.0 - unmodified on disk | Unitree (`SOURCE.md`) |
| `dex3_1/` | Unitree's Dex3-1 URDFs (unitree_ros@5994d4f), BSD-3 - used as the reference the primitive hand model is tested against (no meshes) | Unitree (`SOURCE.md`) |
| `provenance.yaml` | every model parameter: value, source, status (official / measured / derived / reported / estimate / modelling_choice) and, for estimates, what replaces them | us; `tests/unit/test_provenance_model.py` keeps it equal to the code |

Profiles (`r1sim.model.PROFILES`): `train` = the robot exactly as the balance policy was trained (head welded,
rigid 0.697 kg hand lumps, mjlab collision and joint settings, 5 ms); `real` = our robot as best known (articulated
head and Dex3-1 hands, the IMU at its URDF position, the rear computing module (0.454 kg, weighed) inside the rear handle frame, 1 ms). Estimates are replaced by
measurements from the stand-only hardware session (DESIGN.md §12.7, §13.2).
