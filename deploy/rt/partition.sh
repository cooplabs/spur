#!/usr/bin/env bash
# Real-time set-up STAGE 1, runtime part (DESIGN.md §11.1; RESEARCH.md §2.4-2.5, §7.1): no bootloader change, all
# reversible, every value saved first. The runbook is docs/hardware/RT_SETUP.md (one step at a time, measured with
# cyclictest before / after = bench test T4). The user's approval D8 (1) covers this stage; stage 2 (isolcpus on the
# kernel command line, a PREEMPT_RT kernel) is NOT done by this script and needs the user's OK after T4.
#   sudo bash deploy/rt/partition.sh show                     read-only
#   sudo bash deploy/rt/partition.sh apply [--nic-irq N] [--move-tasks]
#   sudo bash deploy/rt/partition.sh undo                     restore the newest backup
#   (all: [--root DIR] [--dry-run])
# CPU plan (DESIGN §2.1): system services, Python processes and every other IRQ on CPUs 0-4; r1guard on 5; the motor
# NIC's IRQ + r1rt's DDS receive and policy threads on 6; the r1rt writer alone on 7.
# apply: (1) backup -> /var/lib/r1mc3/partition_backup/<UTC>/; (2) every movable IRQ -> CPUs 0-4, the motor NIC's IRQ
# (default: the xHCI line in /proc/interrupts while eth10 is the USB adapter; --nic-irq for eth19 / r8169) -> CPU 6;
# (3) the unbound-workqueue cpumask -> CPUs 0-4; (4) a systemd drop-in CPUAffinity=0-4 for services started later
# (our units set their own); (5) with --move-tasks: running user-space tasks (not r1rt / r1guard) -> CPUs 0-4;
# (6) nvpmodel MAXN (mode 0) + jetson_clocks (clocks stored first).
# Session 8 (fix #12): deploy/units/r1mc3-partition.service runs `apply --move-tasks` once per boot, pulled in by
# r1rt.service / r1rt-stand.service (Wants= + After=): the partition is in place before the core starts, whether the
# stack starts at boot or by `r1mc3-svc start` - no hand step at power-up. `undo` stays a hand step.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
. "$here/lib.sh"
r1_parse_common "$@"
set -- ${R1_ARGS[@]+"${R1_ARGS[@]}"}
cmd="${1:-show}"; [ $# -gt 0 ] && shift
nic_irq=""; move_tasks=0
while [ $# -gt 0 ]; do
  case "$1" in
    --nic-irq) nic_irq="$2"; shift 2 ;;
    --move-tasks) move_tasks=1; shift ;;
    *) r1_die "unknown option $1" ;;
  esac
done
SYS_CPUS="0-4"; NIC_CPU="6"; WQ_MASK="1f"   # 0x1f = CPUs 0-4
proc_irq="$(r1_path /proc/irq)"
interrupts="$(r1_path /proc/interrupts)"
wq="$(r1_path /sys/devices/virtual/workqueue/cpumask)"
dropin="$(r1_path /etc/systemd/system.conf.d)/r1mc3-cpuaffinity.conf"
backups="$(r1_path /var/lib/r1mc3/partition_backup)"

detect_nic_irq() {
  # eth10 = the r8152 USB adapter: its interrupts arrive on the xHCI controller's line (tegra-xusb). HW1 (fix #12):
  # the backpack has THREE xhci lines - 228 and 230 "3610000.xhci" (0 interrupts) and 229 "xhci-hcd:usb1" (688258:
  # eth10, the Wi-Fi dongle and the hands) - and the first match (228) was the wrong one (the session used
  # --nic-irq 229). Now: the BUSIEST xhci line (the interrupt counts summed over the CPU columns); on a tie (e.g. all
  # 0 right after boot) the xhci-hcd:usb line, else the first.
  [ -f "$interrupts" ] || return 0
  awk 'NR == 1 { ncpu = NF; next }
       /xhci/ {
         irq = $1; sub(":", "", irq); n = 0
         for (i = 2; i <= ncpu + 1 && i <= NF; i++) if ($i ~ /^[0-9]+$/) n += $i
         usb = ($0 ~ /xhci-hcd/) ? 1 : 0
         if (best == "" || n > bestn || (n == bestn && usb && !bestusb)) { best = irq; bestn = n; bestusb = usb }
       }
       END { if (best != "") print best }' "$interrupts"
}

irq_dirs() { for d in "$proc_irq"/[0-9]*; do [ -f "$d/smp_affinity_list" ] && echo "$d"; done; }

case "$cmd" in
  show)
    echo "nic irq (auto): $(detect_nic_irq)"
    echo "workqueue cpumask: $(cat "$wq" 2>/dev/null || echo '?')"
    n=0; off=0
    for d in $(irq_dirs); do n=$((n + 1)); a="$(cat "$d/smp_affinity_list")"; [ "$a" = "$SYS_CPUS" ] || off=$((off + 1)); done
    echo "irqs: $n with an affinity file, $off not on $SYS_CPUS"
    [ -f "$dropin" ] && echo "systemd drop-in: $dropin" || echo "systemd drop-in: none"
    r1_sys nvpmodel -q
    ;;
  apply)
    [ -n "$nic_irq" ] || nic_irq="$(detect_nic_irq)"
    [ -n "$nic_irq" ] || r1_die "cannot find the motor NIC's IRQ (give --nic-irq N; see /proc/interrupts)"
    b="$backups/$(date -u +%Y%m%dT%H%M%SZ)"
    r1_run mkdir -p "$b"
    if [ "$R1_DRY_RUN" = 0 ]; then
      for d in $(irq_dirs); do printf '%s %s\n' "$(basename "$d")" "$(cat "$d/smp_affinity_list")"; done > "$b/irq_affinity.txt"
      cat "$wq" > "$b/workqueue_cpumask.txt" 2>/dev/null || true
      [ -f "$dropin" ] && cp "$dropin" "$b/dropin.conf" || true
      echo "$nic_irq" > "$b/nic_irq.txt"
    fi
    if r1_is_real_root && [ "$R1_DRY_RUN" = 0 ]; then
      nvpmodel -q > "$b/nvpmodel.txt" 2>&1 || true
      jetson_clocks --store "$b/jetson_clocks.conf" || true
    else
      r1_log "would run: nvpmodel -q > $b/nvpmodel.txt; jetson_clocks --store $b/jetson_clocks.conf"
    fi
    moved=0; fixed=0
    for d in $(irq_dirs); do
      irq="$(basename "$d")"; want="$SYS_CPUS"; [ "$irq" = "$nic_irq" ] && want="$NIC_CPU"
      if [ "$R1_DRY_RUN" = 1 ]; then r1_log "dry-run: irq $irq -> $want"; continue; fi
      if (echo "$want" > "$d/smp_affinity_list") 2>/dev/null; then moved=$((moved + 1)); else fixed=$((fixed + 1)); fi
    done
    r1_log "irqs moved: $moved (motor NIC irq $nic_irq -> CPU $NIC_CPU); not movable (per-CPU / EIO): $fixed"
    if [ -f "$wq" ]; then r1_run sh -c "echo $WQ_MASK > '$wq'"; fi
    r1_run mkdir -p "$(dirname "$dropin")"
    if [ "$R1_DRY_RUN" = 0 ]; then
      printf '%s\n' "# r1mc3 RT stage 1 (deploy/rt/partition.sh): services started from now on run on CPUs 0-4;" \
        "# r1rt / r1guard set their own CPUAffinity in their units. Undo: deploy/rt/partition.sh undo." \
        "[Manager]" "CPUAffinity=0 1 2 3 4" > "$dropin"
    fi
    if [ "$move_tasks" = 1 ]; then
      # HW1 (fix #21): `[ -s "$p/cmdline" ]` is false for EVERY process - procfs files report size 0 - so nothing was
      # ever moved (NetworkManager / sshd stayed on 0-7). The cmdline's CONTENT tells a kernel thread (empty) from a
      # user-space task.
      nt=0
      for p in $(r1_path /proc)/[0-9]*; do
        pid="$(basename "$p")"
        comm="$(cat "$p/comm" 2>/dev/null || true)"
        case "$comm" in r1rt|r1guard) continue ;; esac
        first=""; IFS= read -r -d '' first < "$p/cmdline" 2>/dev/null || true
        [ -n "$first" ] || continue   # a kernel thread (or a task that just exited)
        r1_sys taskset -a -p -c "$SYS_CPUS" "$pid" >/dev/null || true
        nt=$((nt + 1))
      done
      r1_log "user-space tasks moved to CPUs $SYS_CPUS: $nt (not r1rt / r1guard)"
    fi
    r1_sys nvpmodel -m 0
    r1_sys jetson_clocks
    if [ "$R1_DRY_RUN" = 1 ]; then   # fix #21: the dry run printed "stage 1 applied"
      r1_log "dry-run: stage 1 NOT applied - nothing was changed"
    else
      r1_log "stage 1 applied; backup: $b (undo: deploy/rt/partition.sh undo)"
    fi
    ;;
  undo)
    b="$(ls -1d "$backups"/*/ 2>/dev/null | sort | tail -n 1)"
    [ -n "$b" ] || r1_die "no backup under $backups"
    b="${b%/}"
    if [ -f "$b/irq_affinity.txt" ]; then
      while read -r irq aff; do
        f="$proc_irq/$irq/smp_affinity_list"; [ -f "$f" ] || continue
        if [ "$R1_DRY_RUN" = 1 ]; then r1_log "dry-run: irq $irq -> $aff"; else (echo "$aff" > "$f") 2>/dev/null || true; fi
      done < "$b/irq_affinity.txt"
    fi
    if [ -s "$b/workqueue_cpumask.txt" ] && [ -f "$wq" ]; then r1_run sh -c "cat '$b/workqueue_cpumask.txt' > '$wq'"; fi
    if [ -f "$b/dropin.conf" ]; then r1_run cp "$b/dropin.conf" "$dropin"; else r1_run rm -f "$dropin"; fi
    if [ -f "$b/jetson_clocks.conf" ]; then r1_sys jetson_clocks --restore "$b/jetson_clocks.conf"; fi
    mode=""   # `nvpmodel -q` prints "NV Power Mode: <name>" then the mode number on its own line
    if [ -f "$b/nvpmodel.txt" ]; then mode="$(grep -E '^[0-9]+$' "$b/nvpmodel.txt" | tail -n 1 || true)"; fi
    if [ -n "$mode" ]; then r1_sys nvpmodel -m "$mode"; else r1_log "nvpmodel: no saved mode in $b (set it by hand if it was changed)"; fi
    r1_log "stage 1 undone from $b (tasks moved with --move-tasks keep CPUs 0-4 until they restart: harmless)"
    ;;
  *) r1_die "unknown command $cmd (show | apply | undo)" ;;
esac
