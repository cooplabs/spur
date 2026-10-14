# HW1 excerpts for tests (session 8; the source data stay in the live folder's results/hw1_20261010, never read by tests)

| File | Source (results/hw1_20261010/...) | How it was cut | Used by |
|---|---|---|---|
| s4_ramp_left_knee.npz | end/session_part3/records/20261011T195110Z_ramp_left_knee.json + end/session_part3/r1log/r1log_20261011T192459Z_002.mcap | `stiction.load_ramp(manifest)` (the plan's reference as q_cmd_ref), cut like `fit_joints` (the joint's moving window +- 50 samples), t from 0; only the left knee's columns (q, dq, tau_est, q_cmd, dq_cmd, kp, kd, tau_ff, q_cmd_ref) as float32, t float64; 9268 samples | tests/validation/test_stiction_hw1.py (fit agent) |
