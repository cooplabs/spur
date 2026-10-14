# HW2 day 1's hand data (12 Oct 2026) - test excerpts (session 12, grip)

Source: the live folder's `results/hw2_20261012/` (HW2_PART1.md H2 / H3 / H4 / G). Tests read only these copies.

| File | Source (cut) | sha256 |
|---|---|---|
| `h2_pads_rest.json` | `h2_pads_rest.json` verbatim (H2: the pads at rest, r1mc3.record.pads_rest, 10 s per hand, r1d's tare taken over zeros -> 511 / 511) | 072d82ee33fd2b382bcf592ef94b5910eedd5de768eab7eca4fe8ad85fae175a |
| `h2_pads_rest_after_tare.json` | `h2_pads_rest_after_tare.json` verbatim (H2 after `r1 tare`: noise sd 5-8, max +16 raw) | 37edc5dc339a83fdb7ba517af9c1fe6a5a53211ec62c37383deaf3e63c1781cd |
| `h3_pinch_right_fix.json`, `h3_pinch_left_fix.json` | verbatim (H3 on 5e0130b: `r1 calibrate-pinch` - the tips met (thumb + index "stall"), tip pads 0.00-0.02 kPa -> CALIBRATION_FAILED; layer -6.42 / -9.30 mm, aperture 1.55 / 5.48 mm) | 2bde2d17a437a2412b3c0d9ee5adf023a7328904f03fd698d27fbf2e691149b3 / 3a7de56ad91975344c2d458b020b46318e1a15e67390e3e51aca3940b6b52621 |
| `b4_grasps_hw2.json` | built from `stand_b/r1d_journal.jsonl` (the 40 hand.grasp requests: time, code; `grasp_reply_s` = the next request's receipt - this one's), `b4/b4_*_right*.jsonl` (the user's hold / slip / miss), `stand_b/r1log/r1log_20261013T012915Z_00*.mcap` `/r1rt/telemetry` (the measured 26 joints + IMU, median of 0.15-0.20 s after the grasp request; the robot's monotonic clock = UTC - 01:31:48.503 + 289.612 s, anchored on `on_stand` and `damp`, 3 ms apart) | dc6799091d53db89b0228a4665274cb1e995cf8427f4e10ced22e322cbda1777 |

**What HW2 did NOT record** (session 12's finding): the hands' raw state during the B4 grasps - r1log keeps only the
pads' contact bits (`/r1hands/contact`, at the then 5-raw threshold: saturated at 511 by the pads' drift ~30 min after
the tare) and its grip fields stayed 0 (r1d never registered its grasps with the contact publisher); r1d's journal keeps
the request + code, the B4 log the state words. So the verdict's terms for the tape pinch cannot be replayed from HW2.
From session 12 r1d writes a trace per grasp (`r1mc3.hands.trace`, `grasp_<UTC>_<side>.npz` in its records dir).
