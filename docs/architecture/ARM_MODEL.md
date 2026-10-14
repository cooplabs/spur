# The built-in arm model

The real-time core needs two small models of the upper body that match the simulator exactly
([`../DESIGN.md`](../DESIGN.md) §3.4):

| Model | Used by | Why |
|---|---|---|
| the gravity torque of each arm (5 joints, with its hand and a declared payload) and of the head | the executor: added to every arm / head command's feed-forward; FREEDRIVE needs it | without it a held arm sags by its tracking error |
| fingertip forward kinematics | the executor's surface guard; the core's reports | the lowest fingertip vs a declared surface |

## How it is built

`tools/gen_arm_model.py` compiles the simulator's real-equivalent model and exports the kinematic tree the core needs
into `rt/src/arm_model_gen.cpp` (data only, stamped with a hash of the exported numbers): the bodies from the pelvis
through the waist, the torso, both arms and hands, the head and both legs; per body its parent, pose in the parent,
hinge axis / anchor, mass and centre of mass; the feet, palms, fingertips and IMU sites.

`rt/src/arm_model.cpp` walks the table exactly as MuJoCo's kinematics does. The holding torque of a hinge is the
subtree's first moment crossed with gravity, projected on the axis (+ the payload as a point mass at the grasp centre)
- MuJoCo's bias force at zero velocity.

```bash
uv run python tools/gen_arm_model.py          # regenerate after ANY change to the robot's kinematics / inertials
uv run python tools/gen_arm_model.py --check  # exit 1 if the committed table is stale
```

Tests fail when the committed table is stale or the built module is older than the table, so the MJCF and the core
cannot drift apart silently; the C++ against MuJoCo on random poses agrees to numerical precision.

## API (`rt/include/r1rt/arm_model.hpp`, namespace `r1rt::arm`)

No allocation, no exceptions, constant time; bad input returns `false` and writes zeros, never NaN.

| Function | Output |
|---|---|
| `arm_gravity(side, g_torso, q_arm5, q_hand7 or nullptr, payload_kg, tau_out5)` | holding torque of the 5 arm joints |
| `head_gravity(g_torso, q_head2, tau_out2)` | holding torque of head pitch / yaw |
| `gravity_in_torso(quat_imu_wxyz, waist_roll, waist_yaw, g_torso_out)` | gravity in the torso frame |
| `gravity_ff26(q26, quat_imu_wxyz, q_hand_left7, q_hand_right7, payload_kg2, tau26_out)` | the executor's input: arms + head, legs zero |
