#!/usr/bin/env bash
# Local CI: the same checks as .github/workflows/ci.yml, in one command.
#   scripts/ci.sh            -> everything: lint, types, C++ build + tests, Python tests (the Python golden
#                               test drives the fresh C++ build, so C++ runs first)
#   scripts/ci.sh lint|types|cpp|dds|render|py   -> one stage (dds: the hg DDS types, only when CycloneDDS is built;
#                               render: syncs the render worker's own venv, DESIGN D13, so the Filament tests run)
# Machine-specific settings go in a local, untracked .site.env (never committed), e.g.
#   (a project's own checks may add optional settings here; they are skipped when unset)
#   UV_PROJECT_ENVIRONMENT=...      the venv outside the checkout (e.g. when the checkout is in a synced folder)
#   R1MC3_BUILD_DIR=...             the C++ build tree outside the checkout (default: build/)
#   R1MC3_DDS_PREFIX=...            our CycloneDDS 0.10.2 install (default ~/.cache/r1mc3/cyclonedds-0.10.2)
#   R1_RENDER_VENV=...              the render worker's venv (default ~/.venvs/r1mc3-render; outside any synced folder)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .site.env ]; then set -a; . ./.site.env; set +a; fi
BUILD_DIR="${R1MC3_BUILD_DIR:-build}"
# The optional `dds` extra (CycloneDDS's Python binding, built against our C library: no macOS wheel) when our
# CycloneDDS 0.10.2 build exists (R1MC3_DDS_PREFIX, default ~/.cache/r1mc3/cyclonedds-0.10.2; rt/idl/build_cyclonedds.sh).
DDS_PREFIX="${R1MC3_DDS_PREFIX:-$HOME/.cache/r1mc3/cyclonedds-0.10.2}"
UV_EXTRAS=(--extra perception)  # OpenCV for r1d's measurement words (pyproject: the same pin as dev)
if [ -d "$DDS_PREFIX/lib/cmake/CycloneDDS" ]; then export CYCLONEDDS_HOME="$DDS_PREFIX"; UV_EXTRAS+=(--extra dds); fi

stage="${1:-all}"
run() { echo; echo "=== $* ==="; "$@"; }

uv sync --quiet ${UV_EXTRAS[@]+"${UV_EXTRAS[@]}"}

if [ "$stage" = all ] || [ "$stage" = lint ]; then
  run uv run ruff check .
  run uv run ruff format --check .
fi
if [ "$stage" = all ] || [ "$stage" = types ]; then
  run uv run mypy
fi
if [ "$stage" = all ] || [ "$stage" = cpp ]; then
  if [ -f rt/CMakeLists.txt ]; then
    run uv run cmake -S rt -B "$BUILD_DIR/rt" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DR1RT_SANITIZE=ON
    run uv run cmake --build "$BUILD_DIR/rt"
    run uv run ctest --test-dir "$BUILD_DIR/rt" --output-on-failure
  else
    echo "rt/CMakeLists.txt not present: C++ stage skipped"
  fi
fi
if [ "$stage" = all ] || [ "$stage" = dds ]; then
  # Optional: the hg DDS types against OUR CycloneDDS 0.10.2 build (rt/idl/build_cyclonedds.sh); runs when the prefix
  # exists (R1MC3_DDS_PREFIX, default ~/.cache/r1mc3/cyclonedds-0.10.2), otherwise it says so and skips.
  if [ -d "$DDS_PREFIX/lib/cmake/CycloneDDS-CXX" ]; then
    run uv run cmake -S rt -B "$BUILD_DIR/rt_dds" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DR1RT_SANITIZE=ON \
      -DR1RT_WITH_CYCLONEDDS=ON -DCMAKE_PREFIX_PATH="$DDS_PREFIX"
    run uv run cmake --build "$BUILD_DIR/rt_dds"
    run uv run ctest --test-dir "$BUILD_DIR/rt_dds" --output-on-failure
  else
    echo; echo "=== dds: skipped (no CycloneDDS-CXX 0.10.2 at $DDS_PREFIX; build it with rt/idl/build_cyclonedds.sh) ==="
  fi
fi
if [ "$stage" = all ] || [ "$stage" = render ]; then
  # The render-only worker's venv (DESIGN D13: MuJoCo 3.15.0 / Python 3.12, outside the checkout and any synced folder).
  RV="${R1_RENDER_VENV:-$HOME/.venvs/r1mc3-render}"
  run env UV_PROJECT_ENVIRONMENT="$RV" uv sync --project render --frozen --python 3.12 --quiet
  export R1_RENDER_VENV="$RV"
fi
if [ "$stage" = all ] || [ "$stage" = py ]; then
  run uv run pytest -q
fi
echo; echo "ci.sh: $stage OK"
