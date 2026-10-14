#!/usr/bin/env bash
# Run any command in the project's environment with the local site settings (the untracked .site.env: the venv,
# the C++ build tree and the tool caches outside a synced folder, R1_V2_DIR, ...). Use it instead of a bare `uv run`
# in a checkout that lives in a cloud-synced folder (a bare `uv run` there would create a 300 MB .venv inside it).
#   scripts/run.sh python tools/sim2sim/run_a1.py      # the A1 sim2sim numbers
#   scripts/run.sh pytest -q -s tests/unit/test_policy_golden.py
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .site.env ]; then set -a; . ./.site.env; set +a; fi
# The optional `dds` extra (CycloneDDS's Python binding, built against our C library: no macOS wheel) when our
# CycloneDDS 0.10.2 build exists (R1MC3_DDS_PREFIX, default ~/.cache/r1mc3/cyclonedds-0.10.2; rt/idl/build_cyclonedds.sh).
DDS_PREFIX="${R1MC3_DDS_PREFIX:-$HOME/.cache/r1mc3/cyclonedds-0.10.2}"
UV_EXTRAS=(--extra perception)  # OpenCV for r1d's measurement words (pyproject: the same pin as dev)
if [ -d "$DDS_PREFIX/lib/cmake/CycloneDDS" ]; then export CYCLONEDDS_HOME="$DDS_PREFIX"; UV_EXTRAS+=(--extra dds); fi
exec uv run ${UV_EXTRAS[@]+"${UV_EXTRAS[@]}"} "$@"
