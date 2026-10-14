# Dex3-1 hand description (Unitree)

- Repository: https://github.com/unitreerobotics/unitree_ros
- Commit: 5994d4faef0a9cadd3287f8de0199a67eeb2a259 (HEAD on 30 Sep 2026)
- Path: robots/dexterous_hand_description/dex3_1/{dex3_1_l.urdf, dex3_1_r.urdf}; LICENSE from the repository root
- Fetched 9 Oct 2026 (22:56 PDT) from raw.githubusercontent.com; read as data only.
- Licence: BSD 3-Clause (LICENSE in this folder = the repository root LICENSE at that commit, byte for byte - the only
  licence file in unitree_ros; checked 14 Oct 2026 for the public release: redistribution allowed with the notice).
- Meshes: the 16 link meshes `meshes/*.STL` (binary STL, metres, link frames) were fetched in session 3
  (10 Oct 2026 01:15 PDT) from the same commit, path robots/dexterous_hand_description/dex3_1/meshes/, raw files only
  (no clone), read as data (no scripts run). They are the hands' VISUAL geometry (session 3, the user's note "the
  hands are not super accurate"); collision stays primitives (r1sim/dex3.py) - any collision change needs a test and
  numbers and is recorded in models/provenance.yaml. tests/unit/test_dex3_urdf.py checks the masses, inertias, joint
  origins, axes and limits of the primitive hand against the URDFs; tests check the meshes' sha256 below.
  Unitree ships index_0 == middle_0 and index_1 == middle_1 (identical files per side).

| File | sha256 |
|---|---|
| dex3_1_l.urdf | c55f54d6d27b16a62b78a3dbebe85a85a6a4a4b8851dceb5e42339605f9aff30 |
| dex3_1_r.urdf | a388e69058ec6b84306ff05a7b2b613548c833c3f8455127ab3488324553da57 |
| LICENSE | 84aac59fd3246e3ddc49d1387644e8fcf43b0def4f0b9fe687f372e90446df2d |
| meshes/left_hand_index_0_link.STL | 6b35f2f77211d5a366f0b9a4e47c4ee35e536731266f1a34e9efa12db579b892 |
| meshes/left_hand_index_1_link.STL | 9e315ebc8a7a0cb98e033985b586b20d81cf8aa761181ae61ce56fcb14077a06 |
| meshes/left_hand_middle_0_link.STL | 6b35f2f77211d5a366f0b9a4e47c4ee35e536731266f1a34e9efa12db579b892 |
| meshes/left_hand_middle_1_link.STL | 9e315ebc8a7a0cb98e033985b586b20d81cf8aa761181ae61ce56fcb14077a06 |
| meshes/left_hand_palm_link.STL | 23a486b75bd78a9bf03cec25d84d87f97f3dae038cf21a743b6d469b337e4004 |
| meshes/left_hand_thumb_0_link.STL | a90c721661c0622685488a3c74d1e122c7da89242d3a1daef75edb83422d05e0 |
| meshes/left_hand_thumb_1_link.STL | 445c54a45bc13ce36001556f66bc0f49c83cb40321205ae4d676bb2874325684 |
| meshes/left_hand_thumb_2_link.STL | 3d8dbe5085acfc213d21aa8b0782e89cd79084e9678f3a85fc7b04a86b029db5 |
| meshes/right_hand_index_0_link.STL | 12ab4300c95e437e834f9aef772b3b431c671bf34338930b52ad11aef73bbd0d |
| meshes/right_hand_index_1_link.STL | c9c34efce4563cacdcfd29fc838a982976d40f5442c71219811dcbbf3923a33d |
| meshes/right_hand_middle_0_link.STL | 12ab4300c95e437e834f9aef772b3b431c671bf34338930b52ad11aef73bbd0d |
| meshes/right_hand_middle_1_link.STL | c9c34efce4563cacdcfd29fc838a982976d40f5442c71219811dcbbf3923a33d |
| meshes/right_hand_palm_link.STL | 86c0b231cc44477d64a6493e5a427ba16617a00738112dd187c652675b086fb9 |
| meshes/right_hand_thumb_0_link.STL | 544298d0ea1088f5b276a10cc6a6a9e533efdd91594955fdc956c46211d07f83 |
| meshes/right_hand_thumb_1_link.STL | 0a9a820da8dd10f298778b714f1364216e8a5976f4fd3a05689ea26327d44bf6 |
| meshes/right_hand_thumb_2_link.STL | 3f1bfb37668e8f61801c8d25f171fa1949e08666be86c67acad7e0079937cc45 |
