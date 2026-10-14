#!/usr/bin/env bash
# Make an installed release the current one (DESIGN.md §11.2). REFUSES (exit 3) while r1rt / r1guard own the motors
# (deploy/bin/r1mc3_motorcheck.py; the same rule as a service stop). Installs that release's systemd units, switches
# the `current` symlink atomically, reloads systemd. Starts / stops / restarts NOTHING: the switched release runs from
# the next start (deploy/bin/r1mc3-svc start), which happens only after the session ended the standard way.
#   sudo bash deploy/activate.sh --id <id> [--root DIR] [--dry-run]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/lib.sh"
r1_parse_common "$@"
set -- ${R1_ARGS[@]+"${R1_ARGS[@]}"}
id=""
while [ $# -gt 0 ]; do
  case "$1" in
    --id) id="$2"; shift 2 ;;
    *) r1_die "unknown option $1" ;;
  esac
done
[ -n "$id" ] || r1_die "--id is required (deploy/status.sh lists the installed releases)"
prefix="$(r1_path "$R1_PREFIX")"
rel="$prefix/releases/$id"
[ -d "$rel" ] || r1_die "release $id is not installed ($rel)"
r1_require_motors_free "switching the release"

units="$(r1_path "$R1_UNIT_DIR")"
r1_run mkdir -p "$units"
for u in "$rel"/deploy/units/*.service; do
  [ -f "$u" ] || continue
  r1_run cp "$u" "$units/$(basename "$u")"
done
r1_switch_current "$id"
r1_sys systemctl daemon-reload
r1_log "active: $id (nothing restarted; it runs from the next start)"
