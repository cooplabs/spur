# Golden CRCs: hg LowCmd_ / LowState_

`crc_golden.txt` is written by `uv run python tools/crc_golden.py` (a re-implementation of unitree_sdk2_python's hg
CRC and pack formats, cross-checked against CRC-32/MPEG-2; sources in `rt/idl/SOURCE.md`) and read by the C++ test
`rt/tests/test_crc.cpp`, which must reproduce every CRC, the message bytes and a field-by-field repack.
One message per line: `<name> <lowcmd|lowstate> <crc, 8 hex digits> <message bytes as hex, crc field included>`.
Messages: zeros; the R1 stand command (26 joints in their hg slots, mode 1, `r1sim.joints.POSE_STAND`, stiff
gains); random fields from seeds 1-3 (LowCmd_) and 1-2 (LowState_); LowState_ zeros. `SHA256SUMS` pins the file;
`tools/crc_golden.py --check` regenerates and compares.
