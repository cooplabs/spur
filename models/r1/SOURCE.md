# R1 robot description (Unitree)

- Repository: https://github.com/unitreerobotics/unitree_rl_mjlab
- Commit: 1425b15f73bd4095f0df53709d7c389c3eb9e790 (also in SOURCE_COMMIT)
- Path: src/assets/robots/unitree_r1/xmls/r1.xml (here r1_mjlab.xml) and its meshes (meshes/)
- Copied from r1mc2 (v2) assets/r1/ (the policy's training asset; masses = unitree_ros R1.urdf).
- r1_mjlab.xml sha256 19c2291ce675217cc193092e59726bbd8d21c53bd5b2599912b60fe538f2de5f (checked by
  tests/unit/test_model.py).
- Licence: Apache-2.0 (LICENCE in this folder = the repository root LICENCE at that commit, byte for byte; checked
  14 Oct 2026 for the public release, with r1.xml and all 43 meshes byte-identical to that commit).
- Not modified on disk: every change is made at load time by r1sim/model.py and listed in models/provenance.yaml.
