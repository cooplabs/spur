#!/usr/bin/env bash
# ExecStartPre=+ (root) of the r1mc3 RT units under `site.sh apply --rt-budget slice` (HW1 fix #3, the slice method):
# give the unit's cpu cgroup - and its parent slice - an explicit RT budget BEFORE the process starts, so SCHED_FIFO
# is granted with the global RT throttle left on. On this kernel (CONFIG_RT_GROUP_SCHED=y, cgroup v1) every new cpu
# cgroup starts at cpu.rt_runtime_us = 0 (= no SCHED_FIFO at all); a parent must hold at least the sum of its
# children's budgets, so the slice is written first, then the unit's own group.
#   rtbudget.sh [--slice-us N] [--leaf-us N] [--pid PID] [--root DIR] [--dry-run]
# PID = whose cgroup (default: this script's own process = the unit's cgroup when systemd runs it as ExecStartPre).
# Nothing to do (exit 0) when the process sits in the root cpu cgroup (the controller not realized for the unit) or
# the kernel has no RT group scheduling (no cpu.rt_runtime_us files). Never moves a motor, never publishes anything.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
. "$here/lib.sh"
r1_parse_common "$@"
set -- ${R1_ARGS[@]+"${R1_ARGS[@]}"}
slice_us=700000; leaf_us=400000; pid="$$"
while [ $# -gt 0 ]; do
  case "$1" in
    --slice-us) slice_us="$2"; shift 2 ;;
    --leaf-us) leaf_us="$2"; shift 2 ;;
    --pid) pid="$2"; shift 2 ;;
    *) r1_die "unknown option $1" ;;
  esac
done
[ "$leaf_us" -le "$slice_us" ] || r1_die "the unit's budget ($leaf_us) exceeds the slice's ($slice_us)"
cgfile="$(r1_path /proc)/$pid/cgroup"
[ -f "$cgfile" ] || r1_die "no $cgfile"
# cgroup v1: "4:cpu,cpuacct:/r1mc3.slice/r1rt.service" (the cpu controller's line, alone or with cpuacct)
path="$(awk -F: '$2 ~ /(^|,)cpu(,|$)/ { print $3; exit }' "$cgfile")"
if [ -z "$path" ] || [ "$path" = "/" ]; then
  r1_log "rtbudget: the root cpu cgroup (or no cpu controller): nothing to do"; exit 0
fi
mnt=""
for m in /sys/fs/cgroup/cpu,cpuacct /sys/fs/cgroup/cpu; do
  if [ -d "$(r1_path "$m")" ]; then mnt="$(r1_path "$m")"; break; fi
done
[ -n "$mnt" ] || r1_die "no cgroup v1 cpu hierarchy under /sys/fs/cgroup"
if [ ! -f "$mnt/cpu.rt_runtime_us" ]; then
  r1_log "rtbudget: no RT group scheduling on this kernel (no cpu.rt_runtime_us): nothing to do"; exit 0
fi
# every ancestor below the root gets the slice's budget (if it has less), the unit's own group the leaf's
dir="$mnt"; IFS=/ read -r -a parts <<< "${path#/}"
n=${#parts[@]}; i=0
for c in "${parts[@]}"; do
  i=$((i + 1)); dir="$dir/$c"
  [ -f "$dir/cpu.rt_runtime_us" ] || r1_die "no $dir/cpu.rt_runtime_us"
  want="$slice_us"; [ "$i" = "$n" ] && want="$leaf_us"
  have="$(cat "$dir/cpu.rt_runtime_us")"
  if [ "$i" != "$n" ] && [ "$have" -ge "$want" ]; then continue; fi
  r1_run sh -c "echo $want > '$dir/cpu.rt_runtime_us'"
  r1_log "rtbudget: ${dir#"$mnt"} cpu.rt_runtime_us $have -> $want"
done
