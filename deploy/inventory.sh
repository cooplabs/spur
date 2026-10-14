#!/usr/bin/env bash
# The backpack inventory for HW1 (runbook step P1a): READ-ONLY - it installs, starts, stops and writes nothing (no sudo,
# no apt, no systemctl start/stop, never `ethtool` or a NIC `speed` read: U8 / DESIGN D18). It prints what v3's install
# needs and what is already there, with the fix for anything missing, so no robot time goes to discovering it.
#
#   from your computer (this checkout):  ssh unitree@<robot-ip> 'bash -s' < deploy/inventory.sh
#   on the backpack:                    bash inventory.sh
#
# Output: one line per check  "<check> | <found> | OK / ACTION / BLOCKER | <what to do>",  then a summary line.
# Exit 0 = ready (nothing to do), 3 = actions listed (none blocks the session), 4 = a BLOCKER (stop and decide).
# Known facts it confirms (RESEARCH.md §App R7 / V2_INVENTORY): Orin NX 8 cores, L4T R35.3.1 (Ubuntu 20.04, aarch64),
# kernel 5.10.104-tegra PREEMPT, system Python 3.8 (v3 uses uv's managed Python 3.11), v2's CycloneDDS C library.
# Written for bash 3.2 (the Mac, for tests) and bash 5 (the backpack): no associative arrays, no mapfile.
set -u
R1I_ROOT="${R1I_ROOT:-}"          # tests: a fake root prefixed to every file path
N_ACTION=0
N_BLOCKER=0

row() {  # check found status hint
  printf '%-28s | %-46s | %-7s | %s\n' "$1" "$2" "$3" "$4"
  case "$3" in ACTION) N_ACTION=$((N_ACTION + 1)) ;; BLOCKER) N_BLOCKER=$((N_BLOCKER + 1)) ;; esac
}
first_line() { head -n 1 2>/dev/null | cut -c1-46; }
have() { command -v "$1" >/dev/null 2>&1; }

printf '%-28s | %-46s | %-7s | %s\n' "check" "found" "status" "what to do"

# --- the platform (expected values: the 9 Oct journal + RESEARCH.md) -------------------------------------------------
rel="$(cat "$R1I_ROOT/etc/nv_tegra_release" 2>/dev/null | first_line)"
case "$rel" in
  *R35*REVISION:\ 3.1*) row "L4T release" "$rel" OK "R35.3.1 = JetPack 5.1.1 / Ubuntu 20.04 (the Linux build was checked on it)" ;;
  "") row "L4T release" "not found" ACTION "not a Jetson? (expected /etc/nv_tegra_release R35 rev 3.1) - tell the integrator" ;;
  *) row "L4T release" "$rel" ACTION "not R35.3.1: the session-6 Linux check used Ubuntu 20.04 - expect build surprises" ;;
esac
row "kernel" "$(uname -r 2>/dev/null) $(uname -v 2>/dev/null | grep -o 'PREEMPT[_A-Z]*' | head -n 1)" OK "PREEMPT (not RT) is expected; RT setup is block 4 (RT_SETUP.md)"
row "arch / cores" "$(uname -m 2>/dev/null) / $(getconf _NPROCESSORS_ONLN 2>/dev/null) CPUs" OK "aarch64 / 8 expected (the units pin CPUs 5-7)"
if have free; then
  row "memory" "$(free -m 2>/dev/null | awk '/^Mem:/ {print $2 " MB total, " $7 " MB available"}')" OK "-"
fi

# --- disk: the install + build need ~3 GB; the session's logs ~1 GB / h ----------------------------------------------
for d in / /opt /home /var; do
  avail_kb="$(df -Pk "$R1I_ROOT$d" 2>/dev/null | awk 'NR==2 {print $4}')"
  [ -n "$avail_kb" ] || continue
  gb=$((avail_kb / 1024 / 1024))
  if [ "$gb" -lt 5 ]; then
    row "disk free $d" "${gb} GB" BLOCKER "< 5 GB: free space first (logs back up: E2FSCK.md step 1) - the install needs ~3 GB"
  elif [ "$gb" -lt 10 ]; then
    row "disk free $d" "${gb} GB" ACTION "5-10 GB: enough for the install; the session's logs need ~1 GB / h - watch it"
  else
    row "disk free $d" "${gb} GB" OK "-"
  fi
done

# --- the tools the install uses (DEPLOY.md one-time prerequisites + "Linux / the Orin") -----------------------------
py="$(python3 --version 2>&1 | first_line)"
row "system python3" "${py:-missing}" OK "3.8 expected; v3's venv uses uv's managed Python 3.11 (downloaded by uv)"
if have uv; then
  row "uv" "$(uv --version 2>/dev/null | first_line)" OK "-"
  p311="$(uv python find 3.11 2>/dev/null | first_line)"
  if [ -n "$p311" ]; then row "uv Python 3.11" "$p311" OK "-"
  else row "uv Python 3.11" "not yet" OK "the install downloads it (~30 MB; internet below)"; fi
else
  row "uv" "MISSING" ACTION "DEPLOY.md P3: the uv static binary into /usr/local/bin (your OK; ~2 min)"
fi
for t in gcc g++ make; do
  if have "$t"; then row "$t" "$("$t" --version 2>/dev/null | first_line)" OK "-"
  else row "$t" "MISSING" ACTION "sudo apt-get install -y build-essential (your OK; ~5 min)"; fi
done
for t in git curl tar; do
  if have "$t"; then row "$t" "$(command -v "$t")" OK "-"
  else row "$t" "MISSING" ACTION "sudo apt-get install -y $t (your OK)"; fi
done
if have cmake; then
  row "system cmake" "$(cmake --version 2>/dev/null | first_line)" OK "not used: the release's build group brings cmake >= 3.24"
else
  row "system cmake" "none" OK "not needed: the release's build group brings cmake >= 3.24"
fi
for t in cyclictest stress-ng; do
  if have "$t"; then row "$t" "$(command -v "$t")" OK "-"
  else row "$t" "MISSING" ACTION "for T4 (block 4): sudo apt-get install -y rt-tests stress-ng (your OK; RT_SETUP.md)"; fi
done

# --- CycloneDDS: another controller's C library (optional, R1I_PREV_DDS = its prefix) and v3's C++ build ----------
PREV_DDS="${R1I_PREV_DDS:-}"  # e.g. a previous controller's own CycloneDDS on this backpack: reported, never modified
if [ -n "$PREV_DDS" ]; then
  if [ -d "$R1I_ROOT$PREV_DDS/lib" ]; then row "other CycloneDDS (C)" "$PREV_DDS" OK "not ours: never modified"
  else row "other CycloneDDS (C)" "not at $PREV_DDS" OK "v3 builds its own (next row); nothing else affected"; fi
fi
if [ -d "$R1I_ROOT/opt/r1mc3/cyclonedds-0.10.2/lib/cmake/CycloneDDS-CXX" ]; then
  row "v3 CycloneDDS-CXX 0.10.2" "/opt/r1mc3/cyclonedds-0.10.2" OK "-"
else
  row "v3 CycloneDDS-CXX 0.10.2" "not built" ACTION "DEPLOY.md P2: rt/idl/build_cyclonedds.sh on the backpack (your OK; 10-20 min, CPU only)"
fi

# --- v3 already on the backpack? (a re-run after a pre-stage, or leftovers) -----------------------------------------
if [ -L "$R1I_ROOT/opt/r1mc3/current" ]; then
  row "v3 current release" "$(readlink "$R1I_ROOT/opt/r1mc3/current" | first_line)" OK "check it is the session-6 release (status.sh)"
elif [ -d "$R1I_ROOT/opt/r1mc3/releases" ]; then
  row "v3 releases" "$(ls "$R1I_ROOT/opt/r1mc3/releases" 2>/dev/null | tr '\n' ' ' | cut -c1-46)" OK "installed, not activated: P2 continues from step 5"
else
  row "v3 release" "none" OK "P2 installs it (DEPLOY.md 'Install a release')"
fi
if [ -f "$R1I_ROOT/etc/r1mc3/r1rt.cfg" ]; then row "v3 robot config" "/etc/r1mc3/r1rt.cfg" OK "regenerate if the profile changed (P2 step 5)"
else row "v3 robot config" "none" OK "P2 step 5 writes it (rtconfig --profile robot)"; fi

# --- units and processes (read-only listings) -------------------------------------------------------------------------
if have systemctl && [ -z "$R1I_ROOT" ]; then
  units="$(systemctl list-unit-files 2>/dev/null | awk '{print $1}' | grep -E '^(r1|r1mc3)' | tr '\n' ' ' | cut -c1-46)"
  row "r1* systemd units" "${units:-none}" OK "v2's r1-eth10-guard.service expected; v3's appear after P2 / B4.1"
fi
if have pgrep; then
  v2="$(pgrep -fl 'r1mc2|mc\.sh serve' 2>/dev/null | head -n 1 | cut -c1-46)"
  if [ -n "$v2" ]; then row "v2 controller running" "$v2" BLOCKER "stop v2 the usual way first (v2 and v3 never run together)"
  else row "v2 controller running" "no" OK "-"; fi
  v3="$(pgrep -l -x 'r1rt|r1guard' 2>/dev/null | tr '\n' ' ' | cut -c1-46)"
  row "v3 processes running" "${v3:-no}" OK "none expected before B1.1"
fi

# --- the network: the motor NIC (name / state only) and the internet the install needs --------------------------------
if have ip; then
  e10="$(ip -br link show eth10 2>/dev/null | first_line)"
  if [ -n "$e10" ]; then row "eth10 (motor link)" "$e10" OK "name + state only (never ethtool / speed: U8)"
  else row "eth10 (motor link)" "not found" BLOCKER "the motor link is missing: check the USB adapter (USB_NETWORK.md)"; fi
fi
net_ok=0
if have curl; then
  for u in https://pypi.org https://github.com; do
    if curl -sI --max-time 6 "$u" >/dev/null 2>&1; then row "internet $u" "reachable" OK "-"; net_ok=$((net_ok + 1))
    else row "internet $u" "NOT reachable" BLOCKER "the install downloads Python 3.11 + build tools: fix Wi-Fi / DNS first"; fi
  done
fi
if have timedatectl && [ -z "$R1I_ROOT" ]; then
  row "clock synced (T8)" "$(timedatectl show -p NTPSynchronized --value 2>/dev/null | first_line)" OK "yes expected; T8 compares the clocks"
fi

printf 'SUMMARY | %d action(s), %d blocker(s) | %s\n' "$N_ACTION" "$N_BLOCKER" \
  "$( [ "$N_BLOCKER" -gt 0 ] && echo 'STOP: resolve the BLOCKER rows first' || { [ "$N_ACTION" -gt 0 ] && echo 'do the ACTION rows (each your OK), then P2' || echo 'ready for P2'; } )"
[ "$N_BLOCKER" -gt 0 ] && exit 4
[ "$N_ACTION" -gt 0 ] && exit 3
exit 0
