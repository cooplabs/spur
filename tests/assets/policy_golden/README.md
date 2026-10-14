# Golden set for the C++ policy runner (DESIGN.md §3.2)

`golden_1000.bin` ("R1G1", format in our generator (internal: it needs the previous controller)): 1,000 observation vectors (74 values each, the
ONNX `observation_names` order) of the live policy R1MC-Manip2-Lean
(`policies/r1mc2_manip2_lean_20261006T0421Z/policy.onnx`, sha256 09bf2fd8...) and the actions v2's NumPy executor
(r1mc2 (v2) `r1mc2/policy.py` NumpyOnnx at git r21_try 2a4a165, float64) computes for them. Checksum in `SHA256SUMS`.

| Rows | Source |
|---|---|
| 0-249 | **v2's full stack through the backend adapter**: the simulator's v2 backend adapter (real-equivalent robot + a table scene, lockstep; internal) with v2's Runtime + Controller + WalkPolicy unchanged; v2 takes over (mirror -> blend -> WALK) and gets a command script through its own `Controller.submit` (stand; walk vx 0.3 for 2 s; vy 0.15; wz 0.4; a 3 deg lean; lean 0); every vector v2's controller fed its network recorded (901 in one 17 s episode, no fall, no fault), 250 taken evenly |
| 250-499 | built by **v2's own observation code** (`WalkPolicy.observe`) called directly: v2's WalkPolicy balanced the train-equivalent robot in r1sim (lockstep, robot only, 5 ms physics, v2's `step()` at 200 Hz, the network every 20 ms) through a 12 s command script (stand; vx 0.4; vy 0.2; wz 0.6; vx -0.3; a 4 deg lean; stand); one nominal episode and one perturbed (joint offsets +-0.05 rad, base tilt 0.5-2 deg, a 0.1 s sideways push of up to 60 N); 1,200 vectors recorded, 250 taken evenly; no fall |
| 500-999 | in-range random vectors per term (gyro +-3 rad/s, gravity within 40 deg of down, command within the policy's limits, lean 0..0.087 rad, phase or zero, joint_pos in the URDF ranges minus the default pose, joint_vel +-10, actions +-3) |

The header's `n_sim` (500) counts the adapter + sim rows.

Regenerate (bit-identical on this machine and MuJoCo 3.5.0 build; seed 1):

```bash
R1_V2_DIR=/path/to/r1-motor-controller-v2 uv run python <the generator>   # internal: it needs v2
```

Checked by `rt/tests/test_policy_golden.cpp` (ctest; tolerance 1e-5 absolute, prints the max error) and
the internal golden test (sha256; per-source max errors; with R1_V2_DIR: the expected actions are v2's, the
adapter rows regenerate bit-identically, and 10,000 fresh vectors - 900 adapter + 4,100 sim + 5,000 random, seed 7 -
go through the C++ runner). Measured 10 Oct 2026: max |C++ - v2| on this set 1.95e-14 (adapter 1.3e-15, sim 1.6e-15,
random 1.95e-14); on the 10,000 2.84e-14 (adapter 2.0e-15, sim 3.1e-15, random 2.84e-14).

**Regenerated 10 Oct 2026 (session 3)** after the scene's objects got their weighed masses: the adapter rows (0-249) run v2 in that scene, and MuJoCo's constraint solve over
the whole scene changes in the last bits when any body's mass changes, so the committed rows no longer regenerated bit
for bit. Same tool, same seed; the adapter rows moved by at most 4.6e-12 (observations), the sim and random rows are unchanged. The C++
check was re-run on the new set.

**Regenerated 11 Oct 2026 (session 8, refit2)** after the real profile took HW1's identified joint dynamics (the legs' /
waist's friction from the S4 ramps, the arms' / head's armature, friction and damping from the B1 replay - models/
provenance.yaml leg_joint_stiction_hw1, real_joint_*_hw1): v2's adapter episode now runs a different trajectory, so
rows 0-249 changed (max |obs| change 2.31, |action| 1.92; the episode still 1 / 0 falls / 0 faults, 901 recorded); the
sim rows (the train-equivalent robot) and the random rows are bit-identical. Same tool, same seed, sha256 92910cb3...;
the C++ check re-run on the new set.

**Regenerated 11 Oct 2026 (session 9, armature)** after the real profile took the legs' / waist's identified armature
0.008 (models/provenance.yaml real_joint_armature_hw1; issue #51's 0.05 before): v2's adapter episode runs a different
trajectory, so rows 0-249 changed (max |obs| change 2.12, |action| 1.33; the episode 1 / 0 falls / 0 faults, 901
recorded); the sim and random rows are bit-identical. Same tool, same seed, sha256 8c1775d3...; the C++ check
(PolicyGolden) and the internal golden test re-run on the new set (pass).
