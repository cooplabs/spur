#!/usr/bin/env bash
# Switch back to the previous release (DESIGN.md §11.2: rollback = switch back). The previous one is read from
# <root>/opt/r1mc3/history.log (the release that was current before the present one), or given with --to.
# Goes through activate.sh, so it REFUSES (exit 3) while r1rt / r1guard own the motors and restarts nothing.
#   sudo bash deploy/rollback.sh [--to <id>] [--root DIR] [--dry-run]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/lib.sh"
r1_parse_common "$@"
set -- ${R1_ARGS[@]+"${R1_ARGS[@]}"}
to=""
while [ $# -gt 0 ]; do
  case "$1" in
    --to) to="$2"; shift 2 ;;
    *) r1_die "unknown option $1" ;;
  esac
done
[ -n "$to" ] || to="$(r1_previous_id)"
[ -n "$to" ] || r1_die "no previous release recorded in history.log (give --to <id>)"
args=(--root "$R1_ROOT" --id "$to"); [ "$R1_DRY_RUN" = 1 ] && args+=(--dry-run)
r1_log "rollback: $(r1_current_id) -> $to"
exec bash "$here/activate.sh" "${args[@]}"
