#!/usr/bin/env bash
# Build CycloneDDS + cyclonedds-cxx 0.10.2 (the version Unitree generated the hg types with) into a local prefix
# OUTSIDE the repository, then print how to build the hg DDS types and their tests against it.
#   bash rt/idl/build_cyclonedds.sh                 # prefix: $R1MC3_DDS_PREFIX or ~/.cache/r1mc3/cyclonedds-0.10.2
# Shallow clones of the two release tags, checked against pinned commits; no SSL, security, shared memory (iceoryx),
# examples or tests. ~10 MB installed, under a minute on an M4. Licence: EPL-2.0 OR EDL-1.0 (BSD-3) (DESIGN §10.2).
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f .site.env ]; then set -a; . ./.site.env; set +a; fi
PREFIX="${R1MC3_DDS_PREFIX:-$HOME/.cache/r1mc3/cyclonedds-0.10.2}"
SRC="${PREFIX}.src"
CDDS_SHA=9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd   # eclipse-cyclonedds/cyclonedds tag 0.10.2
CXX_SHA=2a372d2c4597faea54543b925755fa2d7cdd4232    # eclipse-cyclonedds/cyclonedds-cxx tag 0.10.2
if command -v uv >/dev/null 2>&1; then CMAKE=(uv run cmake); else CMAKE=(cmake); fi  # cmake + ninja: the dev venv

fetch() {  # repo dir sha
  if [ ! -d "$SRC/$2" ]; then
    git clone -q --depth 1 --branch 0.10.2 "https://github.com/eclipse-cyclonedds/$1.git" "$SRC/$2"
  fi
  local got
  got="$(git -C "$SRC/$2" rev-parse HEAD)"
  if [ "$got" != "$3" ]; then echo "$1: HEAD $got != pinned $3" >&2; exit 1; fi
}
mkdir -p "$SRC"
fetch cyclonedds cyclonedds "$CDDS_SHA"
fetch cyclonedds-cxx cyclonedds-cxx "$CXX_SHA"

"${CMAKE[@]}" -S "$SRC/cyclonedds" -B "$SRC/build_cdds" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DBUILD_IDLC=ON -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DDSPERF=OFF \
  -DENABLE_SSL=OFF -DENABLE_SECURITY=OFF -DENABLE_SHM=OFF -DENABLE_LTO=OFF
"${CMAKE[@]}" --build "$SRC/build_cdds" -j 6
"${CMAKE[@]}" --install "$SRC/build_cdds"
# Note: cyclonedds-cxx 0.10.2 does not compile with -DENABLE_TYPE_DISCOVERY=OFF (TTopicImpl.hpp needs getTypeInfo).
"${CMAKE[@]}" -S "$SRC/cyclonedds-cxx" -B "$SRC/build_cxx" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOCS=OFF -DENABLE_SHM=OFF
"${CMAKE[@]}" --build "$SRC/build_cxx" -j 6
"${CMAKE[@]}" --install "$SRC/build_cxx"

BUILD_DIR="${R1MC3_BUILD_DIR:-build}"
echo
echo "CycloneDDS + cyclonedds-cxx 0.10.2 installed in $PREFIX (sources in $SRC; delete that folder when done)."
echo "Build and test the hg DDS types:"
echo "  uv run cmake -S rt -B $BUILD_DIR/rt_dds -G Ninja -DR1RT_WITH_CYCLONEDDS=ON -DCMAKE_PREFIX_PATH=$PREFIX"
echo "  uv run cmake --build $BUILD_DIR/rt_dds && uv run ctest --test-dir $BUILD_DIR/rt_dds --output-on-failure"
