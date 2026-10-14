# config/

Robot profiles for the controller: joint map, rates, limits, gain sets, safety thresholds, executor settings. One
versioned YAML file per robot configuration, loaded by `r1mc3.config.load_profile` (YAML -> validated -> frozen at
start; DESIGN.md §3.6).

- Every value carries provenance: `status` (`official`, `measured`, `derived`, `reported`, `estimate`,
  `modelling_choice`), `source`, `date`, and `until` (what replaces it) for estimates and reports.
- Only values marked `tunable: true` (which must have `bounds`) may change at runtime, within their bounds. Safety
  values are never tunable; changing anything else needs a new file with a higher `version` (`check_upgrade`).
- Site settings (addresses, interface names) belong in a local, untracked file, never here.

| File | Robot |
|---|---|
| `robot_profile.r1_dex3.yaml` | Unitree R1 (26 body joints, mode_machine 1) + two Dex3-1 hands - first profile, session 1 |
