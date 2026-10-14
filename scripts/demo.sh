#!/usr/bin/env bash
# The one-command demo: the real controller balancing the simulated R1.
#   scripts/demo.sh              # a window: the R1 with two Dex3-1 hands on the floor, balanced by the controller's
#                                #   C++ real-time core and r1guard - the code of the r1rt / r1guard binaries, stepped
#                                #   in lockstep with the simulator. Keys: S stalls the core (watch r1guard take the
#                                #   motors), P pushes, J / arrows jog an arm, the head or a hand, O drops an object,
#                                #   1-6 cameras, H the text (the full list is printed when the window opens)
#   scripts/demo.sh --headless   # no window: a scripted run (a push, an arm jog, a core stall, a push the guard
#                                #   rides out) -> PNG frames, viewer.json and an MP4 (with ffmpeg) in out/demo
#   scripts/demo.sh --stack      # the REAL processes: r1rt + r1guard + r1d against the DDS-bridge simulator in real
#                                #   time (needs CycloneDDS's Python binding and the DDS build of rt/: docs/sim/VIEWER.md)
# Any other option goes to the viewer (e.g. --scene table, --scene stand, --camera follow).
# The first run builds the C++ core's Python binding (r1rt_py, about two minutes) if it is not built yet.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .site.env ]; then set -a; . ./.site.env; set +a; fi
headless=0
for a in "$@"; do [ "$a" = "--headless" ] && headless=1; done
if ! bash scripts/run.sh python -c 'import sys; from r1sim import cpp_policy; sys.exit(cpp_policy.find_module() is None)'; then
  build="${R1MC3_BUILD_DIR:-build}/rt"
  echo "demo: building the C++ core's Python binding into $build (once) ..."
  bash scripts/run.sh cmake -S rt -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null
  bash scripts/run.sh cmake --build "$build" --target r1rt_py
fi
if [ "$headless" = 1 ]; then
  exec bash scripts/viewer.sh --camera follow --frames 90 --every 0.1 --width 640 --height 480 --out out/demo --mp4 \
    --keys 'P@1.5,J@2.5,=@2.6,=@3.2,=@3.8,S@5.0,P@7.0' "$@"
fi
exec bash scripts/viewer.sh "$@"
