# Board-hold test excerpts (board agent, session 8; B6)

| File | Source (our robot's HW1 records, 11 Oct 2026, the stand session's part 4) | How it was cut |
|---|---|---|
| `board_hold_500ms_excerpt.npz` | the 500 ms pause's `.lowstate.npz` (+ its `.json` for `pause.t_stop_ns`, `actual_ms`) | unique board ticks (1 kHz) from -200 to +650 ms after the SIGSTOP; time `rel_ms` = (tick x 1 ms + min(t_rx - tick)) - t_stop (the tick clock aligned to the receive clock by the fastest delivery); q / dq / tau_est mapped slot -> body joint by `r1sim.joints.SLOT` (26, SDK order), float32; rpy of the IMU |
| `board_hold_summary.json` | the 20 / 100 / 500 ms records | the largest joint motion during each pause vs the mean of the last 100 ticks before it |
