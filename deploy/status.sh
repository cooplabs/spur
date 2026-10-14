#!/usr/bin/env bash
# Read-only: the installed releases, the current and previous one, the units in place vs the current release's, the
# site config, the motor check (may a stop / switch happen now?), v2's folder present.
#   bash deploy/status.sh [--root DIR]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/lib.sh"
r1_parse_common "$@"
prefix="$(r1_path "$R1_PREFIX")"
cur="$(r1_current_id)"
echo "root:      $R1_ROOT"
echo "current:   ${cur:-<none>}"
echo "previous:  $(r1_previous_id)"
echo "releases:"
if [ -d "$prefix/releases" ]; then
  for d in "$prefix"/releases/*/; do
    [ -d "$d" ] || continue
    id="$(basename "$d")"; git="$(sed -n 's/^git=//p' "$d/MANIFEST" 2>/dev/null | cut -c1-12)"
    mark=" "; [ "$id" = "$cur" ] && mark="*"
    echo "  $mark $id  git $git"
  done
fi
echo "units:"
units="$(r1_path "$R1_UNIT_DIR")"
for u in r1rt r1guard r1d r1log r1mc3-inhibit; do
  f="$units/$u.service"
  if [ ! -f "$f" ]; then echo "  $u: not installed"
  elif [ -n "$cur" ] && cmp -s "$f" "$prefix/current/deploy/units/$u.service"; then echo "  $u: installed (= current release)"
  else echo "  $u: installed (DIFFERS from the current release's)"; fi
done
etc="$(r1_path "$R1_ETC")"
for f in r1rt.cfg cyclonedds.xml; do [ -f "$etc/$f" ] && echo "config:    $etc/$f" || echo "config:    $etc/$f MISSING"; done
echo -n "motors:    "; r1_motor_check || true
[ -d "$(r1_path "$R1_V2_DIR")" ] && echo "v2:        $(r1_path "$R1_V2_DIR") present (never touched by these scripts)" || true
