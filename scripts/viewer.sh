#!/usr/bin/env bash
# The simulator's viewer (r1sim.viewer): MuJoCo's window on the R1 with two Dex3-1 hands, balanced by the controller's
# C++ real-time core and r1guard (or the real processes with --stack). Every r1sim.viewer option passes through:
#   scripts/viewer.sh                       # the floor; the core + r1guard in lockstep with the simulator
#   scripts/viewer.sh --scene table         # a plain table with loose objects; --scene stand = hanging on a stand
#   scripts/viewer.sh --stack               # the real r1rt / r1guard / r1d processes (needs CycloneDDS; see --help)
#   scripts/viewer.sh --headless --frames 30 --mp4    # no window: PNG frames + viewer.json (+ an MP4) in out/viewer
# The keys are printed when the window opens (docs/sim/VIEWER.md). Honours the local, untracked .site.env (scripts/run.sh).
# macOS: the window needs mjpython (it ships with the mujoco wheel), and mjpython needs uv's libpython on the loader
# path - set here. Never run this through `nice` or another wrapper that drops DYLD_* variables (macOS strips them).
set -euo pipefail
cd "$(dirname "$0")/.."
window=1
for a in "$@"; do
  case "$a" in --headless|-h|--help) window=0 ;; esac
done
if [ "$(uname)" = Darwin ] && [ "$window" = 1 ]; then
  libdir="$(bash scripts/run.sh python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
  export DYLD_FALLBACK_LIBRARY_PATH="${libdir}${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
  exec bash scripts/run.sh mjpython -m r1sim.viewer "$@"
fi
exec bash scripts/run.sh python -m r1sim.viewer "$@"
