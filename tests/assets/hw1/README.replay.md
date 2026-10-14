# HW1 excerpts for the B1 / B2 replay tests (replay agent, session 8)

| File | Source | How it was cut |
|---|---|---|
| `b1_step_left_knee.npz` | `results/hw1_20261010/end/session_part3/records/20261011T193715Z_step_left_knee.json` + its r1log `r1log_20261011T192459Z_001.mcap` (topic `/r1rt/telemetry`, 500 Hz) | `r1sim.tier_b.data.load_session` window, t = 1.5 .. 5.9 s of the plan (the +0.05 step at 2.0 s and the return at 4.0 s); every command field (q_cmd, dq_cmd, kp, kd, tau_ff) of all 26 joints as logged (float32); q / dq / tau_est as logged for the LEFT KNEE only - the other joints' sensor columns frozen at their first sample (the replay reads them only there: the start pose) |
| `b1_step_left_hip_pitch.npz` | `.../session_part3/records/20261011T193912Z_step_left_hip_pitch.json` + `r1log_20261011T192459Z_001.mcap` | the same cut: t = 1.5 .. 5.9 s (the +0.05 step at 2.0 s, the return at 4.0 s); commands of all 26 joints, the LEFT HIP PITCH's q / dq / tau_est, the other sensor columns frozen at the first sample |
