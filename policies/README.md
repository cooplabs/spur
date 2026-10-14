# policies/

The balance policies r1mc3 can run: the exported network (ONNX with the controller's metadata) + the training run's
summary and its exact training configuration. The C++ build converts the ONNX weights into the real-time core's
fixed-size Eigen network (`rt/policy.cmake`, `tools/policy_convert.py`), and a golden test checks it against the Python
reference runner (`r1mc3.policy`).

| Folder | Policy | Status |
|---|---|---|
| `r1mc2_manip2_lean_20261006T0421Z/` | R1MC-Manip2-Lean: 74 observations -> 14 actions (legs + waist), 50 Hz, the arms external | the policy r1mc3 runs on the robot |

What it is, the sha256 of every file, where it came from and the full training recipe: `docs/POLICY.md`.
