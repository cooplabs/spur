# R1MC-Manip2-Lean (the live v2 balance policy, 6 Oct 2026 04:21 UTC)

The balance policy the r1mc2 (v2) controller ran on our R1 from 6 Oct 2026 until the 9 Oct fall, published here as
ours (DESIGN.md §10.1 `policies/`). Only the three files the controller and the simulator need are copied; the PyTorch
checkpoints (`model_*.pt`), tensorboard events and the training diff stay in the v2 package.

| File | sha256 | What |
|---|---|---|
| `policy.onnx` | `09bf2fd8b8d11d6219ca8a4323f1b4e7ef238bfe88d2bf330579c7e1a833b9be` | observation normaliser + MLP 74-512-256-128-14 (ELU), opset 18; metadata: joint names, kp/kd (`joint_stiffness` / `joint_damping`), `default_joint_pos`, `action_scale`, observation / command names, lean range |
| `r1mc_policy.json` | `70fa37d0d70a67474980af8ef3c62074adda41cbd949506de1576bf83ccabdba` | task R1MC-Manip2-Lean, gait period 0.86 s, command limits, arms external, 500 iterations x 8192 envs, unitree_rl_mjlab 1425b15, lean 0-5 deg, stance stand_lock, fine-tuned from Manip2 model_2392 |
| `params/env.yaml` | `72121ecdce7ba209cbf202f41b6b774b6c3483fb89684d60f70dd1bcd802d86c` | the exact training environment (mjlab 1.2.0 / MuJoCo + MuJoCo Warp 3.5.0): sim options, actuators, observations, noise, domain randomisation, rewards |

Source: the previous controller's (v2) `policies/r1mc2_manip2_lean_policy_20261006T0421Z/` (byte-identical to git
`r21_try` 2a4a165 `policies/...`), copied 9 Oct 2026 for v3 session 2. Trained on RunPod (H100) by us with the v2
training kit (`v2/training/`, V2_INVENTORY App. E §3); ONNX producer pytorch 2.14.1.

Live on hardware: v2's sessions 12-17 and its table-task runs (6-9 Oct 2026). Known limits: no latency was modelled in training
(App. E §3.5); the ankle armature reported for the R1 (unitree_rl_mjlab issue #51) lies outside its randomisation
(App. E §4.2 item 6).
