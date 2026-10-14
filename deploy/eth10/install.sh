#!/usr/bin/env bash
# Install the r1mc3 (v3) eth10 guard ALONGSIDE v2's (DESIGN.md D18, U8; HW1 step B4.1, only with the user's OK):
#   install.sh [--root DIR] [--dry-run] [--replace-v2]
# Copies r1mc3-eth10-guard.sh -> /usr/local/bin/ and r1mc3-eth10-guard.service -> /etc/systemd/system/, writes
# /etc/r1mc3/eth10-guard.install (what was installed + v2's unit state before, for uninstall.sh), daemon-reload.
# v2's files (/usr/local/bin/r1-eth10-guard.sh, /etc/systemd/system/r1-eth10-guard.service) are NEVER written; their
# sha256 are recorded before and checked after.
# Without --replace-v2 nothing is started or stopped: it prints the two commands that switch over. With --replace-v2:
# `systemctl disable --now r1-eth10-guard.service` then `systemctl enable --now r1mc3-eth10-guard.service` (the new
# unit also Conflicts= v2's, so systemd never runs both). Undo: uninstall.sh (re-enables v2's unit if it was enabled).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/../lib.sh"
r1_parse_common "$@"
REPLACE=0
for a in ${R1_ARGS[@]+"${R1_ARGS[@]}"}; do
  case "$a" in
    --replace-v2) REPLACE=1 ;;
    *) r1_die "unknown option $a (install.sh [--root DIR] [--dry-run] [--replace-v2])" ;;
  esac
done
BIN="$(r1_path /usr/local/bin)"; UNITS="$(r1_path "$R1_UNIT_DIR")"; ETC="$(r1_path "$R1_ETC")"
V2_FILES="$BIN/r1-eth10-guard.sh $UNITS/r1-eth10-guard.service"
sha() { if [ ! -f "$1" ]; then echo absent; elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
v2_before=""; for f in $V2_FILES; do v2_before="$v2_before $(sha "$f")"; done
if r1_is_real_root; then
  v2_state="$(systemctl is-enabled r1-eth10-guard.service 2>/dev/null || true)"; [ -n "$v2_state" ] || v2_state=unknown
else
  v2_state="unknown (fake root)"
fi
put() {  # src dst mode: write a temporary file next to dst, then rename (a reader never sees half a file)
  local src="$1" dst="$2" mode="$3"
  if [ "$R1_DRY_RUN" = 1 ]; then r1_log "dry-run: install $src -> $dst ($mode)"; return 0; fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst.tmp.$$"; chmod "$mode" "$dst.tmp.$$"; mv -f "$dst.tmp.$$" "$dst"
  r1_log "installed $dst"
}
put "$HERE/r1mc3-eth10-guard.sh" "$BIN/r1mc3-eth10-guard.sh" 755
put "$HERE/r1mc3-eth10-guard.service" "$UNITS/r1mc3-eth10-guard.service" 644
if [ "$R1_DRY_RUN" != 1 ]; then
  mkdir -p "$ETC"
  {
    echo "installed=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "script_sha256=$(sha "$BIN/r1mc3-eth10-guard.sh")"
    echo "unit_sha256=$(sha "$UNITS/r1mc3-eth10-guard.service")"
    echo "v2_unit_enabled_before=$v2_state"
    echo "v2_files_sha256=${v2_before# }"
    echo "replace_v2=$REPLACE"
  } > "$ETC/eth10-guard.install"
  r1_log "recorded $ETC/eth10-guard.install (v2's unit was: $v2_state)"
fi
r1_sys systemctl daemon-reload
if [ "$REPLACE" = 1 ]; then
  r1_sys systemctl disable --now r1-eth10-guard.service
  r1_sys systemctl enable --now r1mc3-eth10-guard.service
else
  r1_log "not started (no --replace-v2). To switch over (the user's OK, HW1 B4.1):"
  r1_log "  sudo systemctl disable --now r1-eth10-guard.service && sudo systemctl enable --now r1mc3-eth10-guard.service"
fi
v2_after=""; for f in $V2_FILES; do v2_after="$v2_after $(sha "$f")"; done
[ "$v2_before" = "$v2_after" ] || r1_die "v2's eth10 guard files changed during the install (must never happen)"
r1_log "v2's files unchanged (${v2_after# })"
