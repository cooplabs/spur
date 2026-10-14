#!/usr/bin/env bash
# Undo deploy/eth10/install.sh: uninstall.sh [--root DIR] [--dry-run] [--restore-v2]
# Stops + disables r1mc3-eth10-guard.service, removes its two files and the install record. v2's unit is enabled and
# started again when the record says it was enabled before the install, or with --restore-v2. v2's files are never
# written (checked).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../lib.sh"
r1_parse_common "$@"
RESTORE=0
for a in ${R1_ARGS[@]+"${R1_ARGS[@]}"}; do
  case "$a" in
    --restore-v2) RESTORE=1 ;;
    *) r1_die "unknown option $a (uninstall.sh [--root DIR] [--dry-run] [--restore-v2])" ;;
  esac
done
BIN="$(r1_path /usr/local/bin)"; UNITS="$(r1_path "$R1_UNIT_DIR")"; ETC="$(r1_path "$R1_ETC")"
REC="$ETC/eth10-guard.install"
if [ -f "$REC" ] && grep -q '^v2_unit_enabled_before=enabled$' "$REC"; then RESTORE=1; fi
sha() { if [ ! -f "$1" ]; then echo absent; elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
v2_before="$(sha "$BIN/r1-eth10-guard.sh") $(sha "$UNITS/r1-eth10-guard.service")"
r1_sys systemctl disable --now r1mc3-eth10-guard.service
for f in "$BIN/r1mc3-eth10-guard.sh" "$UNITS/r1mc3-eth10-guard.service" "$REC"; do
  if [ -e "$f" ]; then r1_run rm -f "$f"; [ "$R1_DRY_RUN" = 1 ] || r1_log "removed $f"; fi
done
r1_sys systemctl daemon-reload
if [ "$RESTORE" = 1 ]; then
  r1_sys systemctl enable --now r1-eth10-guard.service
else
  r1_log "v2's r1-eth10-guard.service left as it is (it was not recorded as enabled; --restore-v2 to enable it)"
fi
v2_after="$(sha "$BIN/r1-eth10-guard.sh") $(sha "$UNITS/r1-eth10-guard.service")"
[ "$v2_before" = "$v2_after" ] || r1_die "v2's eth10 guard files changed during the uninstall (must never happen)"
