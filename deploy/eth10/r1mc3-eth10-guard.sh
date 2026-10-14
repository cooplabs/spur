#!/usr/bin/env bash
# r1mc3 (v3) eth10 guard - v2's robot_side/eth10_guard.sh ("v4, 5 Oct"; r1-motor-controller-v2, git r21_try c3702ed)
# made v3-aware (DESIGN.md D18, the user's decision U8 of 10 Oct 2026; docs/hardware/USB_NETWORK.md "session 6").
# Backpack, root, systemd: r1mc3-eth10-guard.service (it CONFLICTS with v2's r1-eth10-guard.service: never both).
#
# WHEN NO CONTROLLER RUNS it does exactly what v2's guard does (the 5 Oct auto-recovery; the actions, their order and
# their timing are compared with v2's own script in tests/unit/test_eth10_guard.py):
#   1. on start and on every link-up: Energy Efficient Ethernet off on eth10 (ethtool), [100 Mbit/s only: off]; then
#      45 s grace;
#   2. every second a ping to the robot's main computer; link UP but no answer for 20 s in a row: eth10 off and on
#      (ip link), then 45 s grace. Link DOWN (robot off, on its stand): nothing.
#
# WHILE A CONTROLLER RUNS - v3's r1rt or r1guard, or v2's `r1mc2 serve` - it ONLY LOGS: no reset (the operator
# decides, v2's own rule), and NO `ethtool` call at all (no EEE change, no link-mode read, no `speed` read): ethtool's
# ioctls and the sysfs `speed` read take the kernel's rtnl lock and call into the USB NIC driver; on 9 Oct a `speed`
# read during a Wi-Fi drop blocked v2's control thread for ~4 s and the robot fell (HANDOFF 0.0o). An EEE-off owed by
# a link-up during a controller run is done once no controller runs any more (it is logged as deferred).
# What it still reads while a controller runs: /sys/class/net/eth10/carrier (no rtnl lock, DESIGN §2.1 r1mon's rule),
# ping (a socket bound to the device), pgrep (/proc), systemctl is-active (systemd over D-Bus).
#
# "A controller runs" (controller_running below) = ANY of:
#   * systemd: r1rt.service, r1rt-stand.service (the stand session's core) or r1guard.service active (deploy/units);
#   * a process named exactly r1rt or r1guard (pgrep -x: a manual start, a unit that is "deactivating" while r1rt
#     refuses SIGTERM and keeps publishing, r1guard holding the token after r1rt died);
#   * v2's `r1mc2 serve` (v2's own stand-down, kept);
#   * a check that FAILS (pgrep missing / erroring): counted as running - when in doubt it only logs.
# Journal: journalctl -t r1mc3-eth10-guard.   Read-only check of what it sees now: r1mc3-eth10-guard.sh --status
#
# Options: --root DIR (tests: /sys under DIR; commands come from PATH), --status (print and exit, no action).
IF=eth10; PC=192.168.123.161; MISS_MAX=20; GRACE=45
ADV100=0     # as v2 (5 Oct): eth10 is a USB adapter on a healthy path (gigabit OK); 1 = offer 100 Mbit/s only
TAG=r1mc3-eth10-guard
ROOT=/
STATUS_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --root=*) ROOT="${1#--root=}"; shift ;;
    --status) STATUS_ONLY=1; shift ;;
    *) echo "usage: $0 [--root DIR] [--status]" >&2; exit 2 ;;
  esac
done
if [ "$ROOT" = "/" ]; then SYS=/sys; else SYS="${ROOT%/}/sys"; fi

log() { logger -t "$TAG" "$*"; }

# Prints why a controller counts as running and returns 0; returns 1 (prints nothing) when none runs.
controller_running() {
  local u rc p
  for u in r1rt.service r1rt-stand.service r1guard.service; do
    systemctl is-active --quiet "$u" 2>/dev/null
    rc=$?
    if [ "$rc" = 0 ]; then echo "$u active"; return 0; fi
    # 3 = inactive / failed, 4 = no such unit, 127 = no systemctl (not on systemd): not running by systemd
  done
  for p in r1rt r1guard; do
    pgrep -x "$p" >/dev/null 2>&1
    rc=$?
    if [ "$rc" = 0 ]; then echo "process $p"; return 0; fi
    if [ "$rc" != 1 ]; then echo "pgrep failed ($rc): cannot tell, counted as running"; return 0; fi
  done
  pgrep -f "r1mc2 [s]erve" >/dev/null 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then echo "v2 r1mc2 serve"; return 0; fi
  if [ "$rc" != 1 ]; then echo "pgrep failed ($rc): cannot tell, counted as running"; return 0; fi
  return 1
}

eee_off() { ethtool --set-eee "$IF" eee off 2>/dev/null && log "EEE off on $IF" || log "EEE off on $IF: not supported / failed"; }
adv100() {   # only if gigabit is still offered: setting it restarts negotiation (one link down / up)
  [ "$ADV100" = "1" ] || return 0
  if ethtool "$IF" 2>/dev/null | sed -n '/Advertised link modes/,/Advertised pause/p' | grep -q "1000baseT"; then
    ethtool -s "$IF" advertise 0x008 && log "offering 100 Mbit/s full duplex only on $IF (renegotiating)"
  fi
}
# v2's link-up actions (EEE off, 100-only), unless a controller runs: then deferred (EEE_PENDING) and logged.
EEE_PENDING=0
link_actions() {
  local why
  if why=$(controller_running); then
    EEE_PENDING=1
    log "$1: $why - NOT calling ethtool (EEE off deferred until no controller runs)"
  else
    EEE_PENDING=0
    eee_off; adv100
  fi
}

if [ "$STATUS_ONLY" = 1 ]; then
  carrier=$(cat "$SYS/class/net/$IF/carrier" 2>/dev/null || echo "0 (no $IF / down)")
  if why=$(controller_running); then mode="LOG ONLY ($why): no reset, no ethtool"; else mode="v2 recovery (no controller runs)"; fi
  echo "r1mc3-eth10-guard: $IF carrier $carrier; mode: $mode; reset after ${MISS_MAX} s silence of $PC; grace ${GRACE} s"
  exit 0
fi

log "guard started (r1mc3 v3-aware, from v2 v4 5 Oct: reset after ${MISS_MAX} s silence; while r1rt / r1guard / r1mc2 serve run: log only, no ethtool; 100-only=${ADV100})"
link_actions "start"
miss=0; up_before=0
while true; do
  up=$(cat "$SYS/class/net/$IF/carrier" 2>/dev/null || echo 0)
  if [ "$up" = "1" ] && [ "$up_before" != "1" ]; then link_actions "link up"; miss=0; sleep "$GRACE"; up_before=1; continue; fi
  # the controller check only where it decides something (v2 checked only before a reset): a link change (logged
  # while a controller runs), a deferred EEE off, and right before a reset (below)
  if [ "$up" != "$up_before" ] && why=$(controller_running); then
    log "link $( [ "$up" = 1 ] && echo UP || echo DOWN ) while $why runs (only logged)"
  fi
  up_before=$up
  if [ "$up" = "1" ]; then
    if [ "$EEE_PENDING" = 1 ] && ! controller_running >/dev/null; then log "the deferred EEE off (no controller runs now)"; link_actions "deferred"; fi
    if ping -c1 -W1 -I "$IF" "$PC" >/dev/null 2>&1; then miss=0; else miss=$((miss + 1)); fi
    if [ "$miss" -ge "$MISS_MAX" ]; then
      if why=$(controller_running); then   # checked again right before the action
        log "link UP but $PC silent for ${MISS_MAX} s; $why running: NOT resetting (operator decides)"
        miss=0; sleep "$GRACE"; continue
      fi
      log "link UP but $PC silent for ${MISS_MAX} s: turning $IF off and on"
      ip link set "$IF" down; sleep 2; ip link set "$IF" up
      miss=0; up_before=0
      sleep "$GRACE"
      continue
    fi
  else
    miss=0
  fi
  sleep 1
done
