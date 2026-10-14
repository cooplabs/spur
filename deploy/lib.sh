# Shared helpers for the deploy scripts (sourced, not run). DESIGN.md §11.2: a versioned release installed side by
# side with the previous one, a `current` symlink switched atomically, rollback = switch back; v2 stays installed and
# untouched. Every script takes:
#   --root DIR   prefix every system path with DIR (tests and rehearsals use a fake root under /tmp; "/" = the real
#                backpack). With a root other than "/", nothing outside DIR is touched and no systemctl / nvpmodel /
#                jetson_clocks command is executed (they are printed as "would run").
#   --dry-run    print what would change, change nothing.
# Written for bash 3.2 (the Mac) and bash 5 (the backpack's Ubuntu 20.04): no associative arrays, no mapfile.
# Nothing here ever publishes to the robot, and nothing in any stop / switch path damps it (DESIGN §2.1).

R1_ROOT="/"
R1_DRY_RUN=0
R1_PREFIX="/opt/r1mc3"            # releases/, current -> releases/<id>, history.log
R1_ETC="/etc/r1mc3"               # site config (r1rt.cfg, cyclonedds.xml): never overwritten by an install
R1_UNIT_DIR="/etc/systemd/system"
R1_SHM_NAME="${R1_SHM_NAME:-/r1mc3.rt}"
R1_V2_DIR="/home/unitree/r1-mc2"   # v2's own folder on the backpack: never written by these scripts
R1_STATE="/var/lib/r1mc3"          # the services' data (session 6): logs/ (r1log's MCAP = r1d's R1LOG_DIR), records/
R1_STATE_SUBDIRS="logs records sessions"   # (R1_RECORD_DIR), sessions/ (R1_SESSION_DIR: `r1 shutdown`'s log save)
R1_USER="unitree"                  # the units' User= (r1rt, r1guard, r1d, r1log)
R1_EXIT_REFUSED=3                 # a refusal because we own the motors (the motor check said so)

r1_log() { printf '%s\n' "$*" >&2; }
r1_die() { r1_log "ERROR: $*"; exit 2; }

# Parse the common options; the caller passes its own unknown options back through R1_ARGS (an array).
r1_parse_common() {
  R1_ARGS=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --root) [ $# -ge 2 ] || r1_die "--root needs a directory"; R1_ROOT="$2"; shift 2 ;;
      --root=*) R1_ROOT="${1#--root=}"; shift ;;
      --dry-run) R1_DRY_RUN=1; shift ;;
      *) R1_ARGS+=("$1"); shift ;;
    esac
  done
  [ -d "$R1_ROOT" ] || r1_die "root $R1_ROOT does not exist"
  R1_ROOT="$(cd "$R1_ROOT" && pwd)"
}

# A system path under the root ("/opt/r1mc3" -> "$R1_ROOT/opt/r1mc3"; the root "/" maps to itself).
r1_path() {
  if [ "$R1_ROOT" = "/" ]; then printf '%s' "$1"; else printf '%s%s' "$R1_ROOT" "$1"; fi
}

r1_is_real_root() { [ "$R1_ROOT" = "/" ]; }

# Run a command that changes something (honours --dry-run).
r1_run() {
  if [ "$R1_DRY_RUN" = 1 ]; then r1_log "dry-run: $*"; return 0; fi
  "$@"
}

# Run a SYSTEM command (systemctl, nvpmodel, ...): only on the real root and not in a dry run.
r1_sys() {
  if [ "$R1_DRY_RUN" = 1 ] || ! r1_is_real_root; then r1_log "would run: $*"; return 0; fi
  "$@"
}

# The release id of the current symlink ("" when none).
r1_current_id() {
  local cur
  cur="$(r1_path "$R1_PREFIX")/current"
  if [ -L "$cur" ]; then basename "$(readlink "$cur")"; fi
}

# Switch the current symlink atomically to releases/<id> (a temporary link renamed over the old one: a reader sees
# either the old or the new target, never a missing link). Appends one line to history.log.
r1_switch_current() {
  local id="$1" prefix old tmp
  prefix="$(r1_path "$R1_PREFIX")"
  [ -d "$prefix/releases/$id" ] || r1_die "release $id is not installed under $prefix/releases"
  old="$(r1_current_id)"
  if [ "$old" = "$id" ]; then r1_log "current is already $id"; return 0; fi
  if [ "$R1_DRY_RUN" = 1 ]; then r1_log "dry-run: switch current ${old:-<none>} -> $id"; return 0; fi
  tmp="$prefix/.current.tmp.$$"
  ln -s "releases/$id" "$tmp"
  # mv -T (GNU) renames over the old link atomically; BSD mv (macOS tests) has -h for the same purpose.
  if mv -T "$tmp" "$prefix/current" 2>/dev/null; then :; else mv -fh "$tmp" "$prefix/current"; fi
  printf '%s %s -> %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${old:-<none>}" "$id" >> "$prefix/history.log"
  r1_log "current: ${old:-<none>} -> $id"
}

# The previous release id: the "from" of the last switch whose "to" is the current id (history.log).
r1_previous_id() {
  local prefix cur
  prefix="$(r1_path "$R1_PREFIX")"
  cur="$(r1_current_id)"
  [ -f "$prefix/history.log" ] || return 0
  awk -v cur="$cur" '$4 == cur && $2 != "<none>" { prev = $2 } END { if (prev != "") print prev }' "$prefix/history.log"
}

# The motor check (deploy/bin/r1mc3_motorcheck.py): exit 0 = nothing of ours owns the motors (or MONITOR / DAMP),
# 3 = refused (we own them), 2 = cannot tell (treated as refused by the callers). Read-only.
r1_motor_check() {
  local here py
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  py="$(r1_path "$R1_PREFIX")/current/venv/bin/python"
  [ -x "$py" ] || py="$(command -v python3)"
  "$py" -I "$here/bin/r1mc3_motorcheck.py" --root "$R1_ROOT" --shm-name "$R1_SHM_NAME" "$@"
}

# Refuse (exit 3) unless the motor check says it is safe to stop / switch.
r1_require_motors_free() {
  local what="$1" rc=0
  r1_motor_check || rc=$?
  if [ "$rc" != 0 ]; then
    r1_log "REFUSED: $what while r1rt / r1guard own the motors (or their state cannot be read)."
    r1_log "  The operator first ends the session the standard way: hold the robot -> r1 damp -> stand -> r1 shutdown."
    exit "$R1_EXIT_REFUSED"
  fi
}
