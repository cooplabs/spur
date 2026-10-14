#!/usr/bin/env bash
# The Linux check on Ubuntu 20.04 - the base of the Jetson Orin's L4T 35.3.1 that the R1's backpack runs (GCC 9; the
# script also runs on current Ubuntu releases): Python 3.11 from uv, CycloneDDS 0.10.2 built from source, both C++
# trees with the sanitizers (ASan + UBSan) and their tests, then the fast pytest set with headless OpenGL (OSMesa).
# Run it in a fresh container with the repository mounted at /work/src (read-only is fine; it builds in a copy):
#   docker run --rm -v "$PWD:/work/src:ro" ubuntu:20.04 bash /work/src/.github/ci/focal.sh
#   (arm64 = the robot's architecture; on an x86_64 machine add --platform linux/arm64, or test x86_64 natively)
# Optional: STAGES="cpp py" (default "cpp dds py"), JOBS=4, PYTEST_MARK="not realtime and not slow".
set -euo pipefail
STAGES="${STAGES:-cpp dds py}"
JOBS="${JOBS:-4}"
PYTEST_MARK="${PYTEST_MARK:-not realtime and not slow}"
has() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
ctest_run() {  # serial, as the timing tests (the guard's takeover, the RT loops) are written for; failures in full
  local log="/work/ctest_$(basename "$1").log"
  if uv run ctest --test-dir "$1" --output-on-failure > "$log" 2>&1; then tail -3 "$log"; else
    grep -B2 -A60 -E "\*\*\*Failed|FAILED|Failure" "$log" | head -150; tail -8 "$log"; return 1; fi
}
step() { echo; echo "##### $(date +%T) $*"; }
export DEBIAN_FRONTEND=noninteractive
# LeakSanitizer's slow unwinder: CycloneDDS's libddsc has no frame pointers, so the fast one cuts its known start-up
# leak's stack (rt/src/dds_io.cpp's suppression) to malloc <- ddsrt_malloc and the suppression cannot match
# (seen on Ubuntu 24.04 x86_64, GCC 13; 20.04 arm64 unwinds it either way)
export LSAN_OPTIONS="${LSAN_OPTIONS:-fast_unwind_on_malloc=0}"
step "apt (build tools, git, OSMesa for headless rendering)"
apt-get update -qq >/dev/null
GLIB=libglib2.0-0; apt-cache show libglib2.0-0t64 >/dev/null 2>&1 && GLIB=libglib2.0-0t64  # 24.04 renamed it
apt-get install -y -qq --no-install-recommends build-essential git curl ca-certificates python3-pip libosmesa6 libgl1 \
  "$GLIB" >/dev/null
. /etc/os-release && echo "$PRETTY_NAME, $(uname -m)" && gcc --version | head -1
step "uv"
pip3 install -q "uv>=0.5" 2>/dev/null || pip3 install -q --break-system-packages "uv>=0.5"  # PEP 668 on 23.04+
uv --version
rm -rf /work/r1 && cp -a /work/src /work/r1 && cd /work/r1
export UV_PROJECT_ENVIRONMENT=/work/venv PYTEST_ADDOPTS="-p no:cacheprovider" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
step "uv sync (Python 3.11, the locked dependencies)"
uv sync --locked --quiet --extra perception
uv run python --version
if has dds; then
  step "CycloneDDS 0.10.2 (rt/idl/build_cyclonedds.sh)"
  R1MC3_DDS_PREFIX=/work/cyclonedds bash rt/idl/build_cyclonedds.sh > /work/cyclonedds.log 2>&1 || { tail -30 /work/cyclonedds.log; exit 1; }
  export CYCLONEDDS_HOME=/work/cyclonedds
  uv sync --locked --quiet --extra perception --extra dds
fi
if has cpp; then
  step "C++: rt (Debug, sanitizers)"
  uv run cmake -S rt -B build/rt -G Ninja -DCMAKE_BUILD_TYPE=Debug -DR1RT_SANITIZE=ON > /work/cmake_rt.log
  uv run cmake --build build/rt -j "$JOBS" > /work/build_rt.log 2>&1 || { tail -40 /work/build_rt.log; exit 1; }
  ctest_run build/rt
fi
if has dds; then
  step "C++: rt with CycloneDDS (the real processes r1rt / r1guard)"
  uv run cmake -S rt -B build/rt_dds -G Ninja -DCMAKE_BUILD_TYPE=Debug -DR1RT_SANITIZE=ON -DR1RT_WITH_CYCLONEDDS=ON \
    -DCMAKE_PREFIX_PATH="$CYCLONEDDS_HOME" > /work/cmake_rt_dds.log
  uv run cmake --build build/rt_dds -j "$JOBS" > /work/build_rt_dds.log 2>&1 || { tail -40 /work/build_rt_dds.log; exit 1; }
  ctest_run build/rt_dds
fi
if has py; then
  step "pytest -m \"$PYTEST_MARK\""
  uv run pytest -q -m "$PYTEST_MARK" | tail -15
fi
step "focal.sh: $STAGES OK"
