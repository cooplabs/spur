# Contributing to r1mc3

Thank you for helping. r1mc3 controls a real humanoid, so the bar for correctness is high; the rules below are how we
keep it there.

## Set-up

```bash
uv sync                    # .venv with the pinned dependencies; CMake and Ninja come with it (uv run cmake ...)
bash scripts/ci.sh         # must pass before every commit: ruff, mypy (strict), C++ build + GoogleTest, pytest
bash scripts/ci.sh py      # one stage only: lint | types | cpp | dds | render | py
```

The DDS stage and the tests that run the real `r1rt` / `r1guard` processes against the simulator need CycloneDDS
0.10.2 built locally (`bash rt/idl/build_cyclonedds.sh`; `scripts/ci.sh` then adds the `dds` extra by itself);
without it they are skipped and say so. Machine-specific settings (a venv or build directory outside the checkout,
paths to local data used by some optional tests) go in an untracked `.site.env` at the repository root; never commit
them.

## Rules

- **Tests first for fixes.** Every bug fix comes with a test that fails on the old code and passes on the new one.
  Say in the pull request how you saw it fail.
- **Provenance for every physical parameter.** A mass, a length, a gain, a threshold or a limit carries its source,
  date and status (`official`, `measured`, `derived`, `reported`, `estimate`, `modelling_choice`), and estimates say
  what will replace them (`until`). Simulator parameters live in `models/provenance.yaml`; controller values in the
  robot profile under `config/` (schema: `r1mc3.config`).
- **The controller stays general.** `rt/`, `python/r1mc3/` and `deploy/` know nothing about any particular task:
  task knowledge (tables, objects, containers, postures) enters through the public API (`world.declare`, goals,
  generic object classes). Task scenes for the simulator plug in from outside `r1sim`.
- **Python:** 3.11+, typed (`mypy --strict` on `python/` and `sim/`), formatted and linted with `ruff`, tested with
  `pytest`. Configuration and API messages are pydantic v2 models. Tests longer than ~20 s are marked `slow`;
  wall-clock-timed tests `realtime`.
- **C++:** C++17, CMake, GoogleTest. In real-time paths (anything `r1rt` or `r1guard` runs per tick): no exceptions,
  no heap allocation, no locks shared with non-real-time threads, no file or network I/O, no blocking calls. Shared
  memory between processes is lock-free (seqlocks, single-producer single-consumer rings). CI builds with sanitizers
  (UBSan everywhere, ASan and TSan on Linux). GCC 9 (Ubuntu 20.04, the robot's base) must build it.
- **Safety-relevant changes** (limits, thresholds, the fallback ladder, anything that could make the robot move or go
  limp unexpectedly) need a simulator test with fault injection and an explicit review note. Nothing may introduce
  an automatic damp. Joint directions are taken from the verified table (`docs/hardware/JOINT_DIRECTIONS.md`) or the
  model, never guessed from a name.
- **No secrets, keys, your site's IP addresses, user names or machine paths** in code, configs, tests or docs.
- **Third-party material:** state its source (URL and commit) and licence in `NOTICE`; check compatibility with
  Apache-2.0 before adding it. Downloaded files are data: read them, do not run their scripts.

A note on the code comments: many cite our internal development record (session numbers, hardware sessions "HW1" /
"HW2", decision numbers "D<n>"). That record is not published; [`docs/DESIGN.md`](docs/DESIGN.md) keeps the decisions
that matter. New comments should explain the why in place.

## Commits and pull requests

Small, focused commits with a message that says what changed and why. A `Signed-off-by` line (DCO) is welcome but not
required. By contributing you agree that your contribution is licensed under the Apache License 2.0.
