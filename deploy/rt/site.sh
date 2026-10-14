#!/usr/bin/env bash
# The backpack's kernel / systemd site settings v3 needs (HW1 fix list #3, #4, #24 - results/hw1_20261010/NOTES.md):
# at HW1 they were typed by hand after EVERY power-up (`sysctl kernel.sched_rt_runtime_us=-1 net.core.rmem_max=8388608`)
# and lost at every reboot. Here they become files (persist across reboots), applied now, saved first, undoable.
#   sudo bash deploy/rt/site.sh show
#   sudo bash deploy/rt/site.sh apply [--rt-budget sysctl|slice|none] [--console-printk "A B C D"|keep]
#        (defaults = the user's decisions of 11 Oct 2026: --rt-budget sysctl, --console-printk "1 4 1 7")
#   sudo bash deploy/rt/site.sh undo                  remove our files, restore the saved runtime values
#   (all: [--root DIR] [--dry-run]; install.sh calls `apply` with its --rt-budget / --console-printk)
#
# #4 (always): net.core.rmem_max = 8388608 - deploy/config/cyclonedds.xml asks SocketReceiveBufferSize min="4MB"; the
#    L4T default 425984 makes every DomainParticipant fail (r1rt exited at its first start on HW1).
# #3 the RT budget (THE USER'S DECISION, 11 Oct 2026: `sysctl`, the default - what HW1 ran on all day, both power-ups):
#    docker / containerd (Delegate=yes) make systemd realize the cpu controller for every service; on this kernel
#    (CONFIG_RT_GROUP_SCHED=y, cgroup v1) a service's own cpu cgroup gets cpu.rt_runtime_us = 0 and SCHED_FIFO fails
#    (EPERM) - the units' CPUSchedulingPolicy=fifo and r1rt's / r1guard's own threads alike.
#    - `sysctl`: kernel.sched_rt_runtime_us = -1 in /etc/sysctl.d - RT throttling OFF for the whole machine (the group
#      budgets are no longer checked). Simple, proven at HW1. Cost: a runaway SCHED_FIFO thread is never throttled -
#      ours are pinned (writer CPU 7, policy / DDS CPU 6, guard CPU 5), so CPUs 0-4 stay free for ssh / systemd.
#    - `slice`: the global throttle stays ON; our four RT units run in r1mc3.slice and an ExecStartPre=+ (root) gives
#      their cpu cgroups an explicit RT budget (deploy/rt/rtbudget.sh: the slice 700 ms / s, r1rt 400, r1guard 200 per
#      CPU) before the process starts. Because systemd would apply CPUSchedulingPolicy=fifo to that ExecStartPre too
#      (EPERM in a 0-budget cgroup), the drop-in sets the unit policy to `other`: r1rt / r1guard set SCHED_FIFO on
#      their own threads (they always have: rt_thread.cpp), which also ends the inheritance of FIFO 90 by CycloneDDS's
#      threads (#8 / #22). An OPT-IN alternative, not the user's choice: tested in a fake root only, NEVER run on the
#      robot - check `rt_granted` in the ready lines before relying on it.
#    - `none`: write nothing about the RT budget.
# #24 the console loglevel (THE USER'S DECISION, 11 Oct 2026: "1 4 1 7", the default; `keep` = no change): the backpack boots with
#    console=ttyTCU0,115200 console=ttyAMA0,115200 and kernel.printk 6 6 1 7, so every kernel line of level <= 5
#    (notice) is written to two 115 kbaud serial consoles synchronously; the ext4 periodic error report (3 lines)
#    stalled r1guard's cycle 24.5 ms at HW1. `--console-printk "4 4 1 7"` = only err / crit / alert / emerg reach the
#    consoles (an ext4 error is crit: still printed); `"1 4 1 7"` = only emergencies (a panic). dmesg / the journal keep
#    EVERY message either way - only the serial consoles go quiet. Nobody reads them on the robot.
# Nothing here moves a motor, publishes anything, restarts a service or touches v2.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
. "$here/lib.sh"
r1_parse_common "$@"
set -- ${R1_ARGS[@]+"${R1_ARGS[@]}"}
cmd="${1:-show}"; [ $# -gt 0 ] && shift
rt_budget="sysctl"; printk="1 4 1 7"   # the user's decisions, 11 Oct 2026 (#3, #24)
while [ $# -gt 0 ]; do
  case "$1" in
    --rt-budget) rt_budget="$2"; shift 2 ;;
    --console-printk) printk="$2"; shift 2 ;;
    *) r1_die "unknown option $1" ;;
  esac
done
case "$rt_budget" in sysctl|slice|none) ;; *) r1_die "--rt-budget must be sysctl, slice or none (got $rt_budget)" ;; esac
if [ "$printk" != keep ]; then
  printf '%s' "$printk" | grep -Eq '^[0-7] [0-7] [0-7] [0-7]$' || r1_die "--console-printk wants four levels 0-7, e.g. \"1 4 1 7\" (got $printk)"
fi

RMEM_MAX=8388608
SYSCTL_FILE="$(r1_path /etc/sysctl.d)/60-r1mc3-rt.conf"
UNITS_DIR="$(r1_path "$R1_UNIT_DIR")"
SLICE_FILE="$UNITS_DIR/r1mc3.slice"
RT_UNITS="r1rt r1rt-stand r1guard r1guard-observe"
backups="$(r1_path "$R1_STATE")/site_backup"
KEYS="net.core.rmem_max kernel.sched_rt_runtime_us kernel.printk"

proc_key() { printf '%s' "$(r1_path /proc/sys)/$(printf '%s' "$1" | tr . /)"; }
read_key() { local f; f="$(proc_key "$1")"; if [ -f "$f" ]; then tr -s ' \t' ' ' < "$f" | sed 's/^ //; s/ $//'; else echo '?'; fi; }
dropin() { printf '%s' "$UNITS_DIR/$1.service.d/60-r1mc3-rtbudget.conf"; }
leaf_us() { case "$1" in r1rt|r1rt-stand) echo 400000 ;; *) echo 200000 ;; esac; }

sysctl_content() {
  echo "# r1mc3 (v3) site settings - written by deploy/rt/site.sh apply (rt-budget $rt_budget, console-printk $printk)."
  echo "# Undo: sudo bash /opt/r1mc3/current/deploy/rt/site.sh undo. HW1 fix list #3 / #4 / #24 (NOTES.md)."
  echo "# #4: cyclonedds.xml asks a 4 MB receive buffer (SocketReceiveBufferSize min); the default 425984 fails DDS."
  echo "net.core.rmem_max = $RMEM_MAX"
  if [ "$rt_budget" = sysctl ]; then
    echo "# #3 (the user's decision, 11 Oct 2026): RT throttling off - our services' cpu cgroups get no RT budget otherwise."
    echo "kernel.sched_rt_runtime_us = -1"
  fi
  if [ "$printk" != keep ]; then
    echo "# #24 (the user's decision, 11 Oct 2026): only emergencies reach the 115 kbaud serial consoles (dmesg / journal: all)."
    echo "kernel.printk = $printk"
  fi
}

slice_content() {
  printf '%s\n' "# r1mc3 (v3): the slice of the RT units under --rt-budget slice (deploy/rt/site.sh); their cpu cgroups get an" \
    "# explicit RT budget from deploy/rt/rtbudget.sh (ExecStartPre=+). Undo: deploy/rt/site.sh undo." \
    "[Unit]" "Description=r1mc3 real-time units (r1rt, r1guard) with their own RT budget" "Before=slices.target"
}

dropin_content() {  # $1 = the unit name
  printf '%s\n' "# r1mc3 (v3) --rt-budget slice (deploy/rt/site.sh): the unit's cpu cgroup gets an RT budget before the start;" \
    "# the unit policy stays SCHED_OTHER (systemd would apply fifo to the ExecStartPre too: EPERM in a 0-budget cgroup);" \
    "# the process sets SCHED_FIFO on its own threads (rt_thread.cpp). Undo: deploy/rt/site.sh undo." \
    "[Service]" "Slice=r1mc3.slice" "CPUSchedulingPolicy=other" "CPUSchedulingPriority=0" \
    "ExecStartPre=+/bin/bash /opt/r1mc3/current/deploy/rt/rtbudget.sh --slice-us 700000 --leaf-us $(leaf_us "$1")"
}

ours_present() {
  [ -f "$SYSCTL_FILE" ] && return 0
  [ -f "$SLICE_FILE" ] && return 0
  for u in $RT_UNITS; do [ -f "$(dropin "$u")" ] && return 0; done
  return 1
}

case "$cmd" in
  show)
    for k in $KEYS; do echo "$k = $(read_key "$k")"; done
    echo "sysctl file: $([ -f "$SYSCTL_FILE" ] && echo "$SYSCTL_FILE" || echo none)"
    echo "rt-budget slice: $([ -f "$SLICE_FILE" ] && echo "$SLICE_FILE" || echo none)"
    if [ -f "$SYSCTL_FILE" ] && grep -q '^kernel.sched_rt_runtime_us' "$SYSCTL_FILE"; then echo "rt-budget method: sysctl"
    elif [ -f "$SLICE_FILE" ]; then echo "rt-budget method: slice"; else echo "rt-budget method: none of ours"; fi
    ;;
  apply)
    want="$(sysctl_content)"
    if ours_present; then
      same=1
      [ -f "$SYSCTL_FILE" ] && [ "$(cat "$SYSCTL_FILE")" = "$want" ] || same=0
      if [ "$rt_budget" = slice ]; then [ -f "$SLICE_FILE" ] || same=0; else [ ! -f "$SLICE_FILE" ] || same=0; fi
      if [ "$same" = 1 ]; then r1_log "site settings already applied (rt-budget $rt_budget, console-printk $printk): unchanged"; exit 0; fi
      r1_die "other r1mc3 site settings are in place (see: site.sh show): run 'site.sh undo' first, then apply"
    fi
    b="$backups/$(date -u +%Y%m%dT%H%M%SZ)"
    r1_run mkdir -p "$b"
    if [ "$R1_DRY_RUN" = 0 ]; then
      for k in $KEYS; do printf '%s=%s\n' "$k" "$(read_key "$k")"; done > "$b/sysctl_before.txt"
      printf '%s\n' "rt_budget=$rt_budget" "console_printk=$printk" > "$b/options.txt"
    fi
    r1_run mkdir -p "$(dirname "$SYSCTL_FILE")"
    if [ "$R1_DRY_RUN" = 1 ]; then r1_log "dry-run: write $SYSCTL_FILE:"; printf '%s\n' "$want" >&2
    else printf '%s\n' "$want" > "$SYSCTL_FILE"; fi
    if [ "$rt_budget" = slice ]; then
      if [ "$R1_DRY_RUN" = 1 ]; then r1_log "dry-run: write $SLICE_FILE and the drop-ins of: $RT_UNITS"
      else
        mkdir -p "$UNITS_DIR"; slice_content > "$SLICE_FILE"
        for u in $RT_UNITS; do mkdir -p "$(dirname "$(dropin "$u")")"; dropin_content "$u" > "$(dropin "$u")"; done
      fi
      r1_sys systemctl daemon-reload
    fi
    r1_sys sysctl -p "$SYSCTL_FILE"   # now, not only from the next boot
    if [ "$R1_DRY_RUN" = 1 ]; then r1_log "dry-run: site settings NOT applied - nothing was changed"
    else r1_log "site settings applied (rt-budget $rt_budget, console-printk $printk); backup: $b (undo: site.sh undo)"; fi
    ;;
  undo)
    b="$(ls -1d "$backups"/*/ 2>/dev/null | sort | tail -n 1 || true)"
    b="${b%/}"
    r1_run rm -f "$SYSCTL_FILE" "$SLICE_FILE"
    for u in $RT_UNITS; do r1_run rm -f "$(dropin "$u")"; done
    r1_sys systemctl daemon-reload
    if [ -n "$b" ] && [ -f "$b/sysctl_before.txt" ]; then
      while IFS='=' read -r k v; do
        [ -n "$k" ] && [ "$v" != '?' ] || continue
        r1_sys sysctl -w "$k=$v"
      done < "$b/sysctl_before.txt"
      r1_log "site settings undone; runtime values restored from $b"
    else
      r1_log "site settings removed; no backup with the runtime values (they reset at the next reboot)"
    fi
    ;;
  *) r1_die "unknown command $cmd (show | apply | undo)" ;;
esac
