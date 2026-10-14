# tests/assets/eth10 - v2's eth10 guard, copied byte for byte (read only; v2 is never edited)

| File | Source (r1-motor-controller-v2, git r21_try c3702ed) | sha256 |
|---|---|---|
| v2_eth10_guard.sh | robot_side/eth10_guard.sh | e9d42d6b45d17245d2cfa5def8635c1520bd1168220c83f6c5515a23a8a2989d |
| v2_r1-eth10-guard.service | robot_side/r1-eth10-guard.service | 1733c0d23d04051da0c455ff06d4b39db1f23db6c7e032703c0360ab2e3b61f5 |

Extraction: `cp $R1_V2_DIR/robot_side/eth10_guard.sh tests/assets/eth10/v2_eth10_guard.sh` (and the unit likewise).
Used by tests/unit/test_eth10_guard.py: v2's script runs in the same fake environment as the v3-aware guard (its
`/sys/class/net` path rewritten to the fake root in a temporary copy) so the two can be compared action for action
when no controller runs (DESIGN D18: "unchanged when no controller runs"); and by test_eth10_install.py as the v2
files that the install must never change.
