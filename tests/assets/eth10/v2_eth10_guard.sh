#!/usr/bin/env bash
# Backpack (root, systemd: r1-eth10-guard.service). 5 Oct 2026: the backpack <-> robot Ethernet (eth10) sometimes
# keeps its link UP while nothing gets through (the robot's main computer stops answering); turning the port off
# and on brought it back after ~20 s every time. This service:
#   1. turns Energy Efficient Ethernet off on eth10 (a known cause of link trouble with this Realtek chip on Jetsons)
#      and offers 100 Mbit/s full duplex only (5 Oct: with a broken pair the link kept trying gigabit, came up at 1000,
#      failed, renegotiated - 4 flaps in 2 min untouched, each one cutting the robot off; 100 Mbit uses only the 2
#      pairs that work and is plenty for the robot's data). Back to gigabit after the harness is fixed:
#      sudo ethtool -s eth10 advertise 0x03f   (and remove ADV100 below)
#   2. every second pings the robot's main computer; if the link is UP but there has been no answer for 20 s in a
#      row, it turns eth10 off and on, then waits 45 s before checking again. Link DOWN (robot off, on its stand):
#      it does nothing. Every action goes to the journal: journalctl -t r1-eth10-guard
#   Never while our controller (r1mc2 serve) runs (5 Oct: a reset 5 s into a silence fired while the session start
#   was switching the robot's camera services, just as the controller started, and its DDS never received anything;
#   a short stall the controller would have ridden out must not become a reset). It only logs then.
IF=eth10; PC=192.168.123.161; MISS_MAX=20; GRACE=45
log() { logger -t r1-eth10-guard "$*"; }
eee_off() { ethtool --set-eee "$IF" eee off 2>/dev/null && log "EEE off on $IF" || log "EEE off on $IF: not supported / failed"; }
ADV100=0     # 5 Oct: eth10 is now a USB adapter on a healthy path (gigabit OK); 1 = offer 100 Mbit/s only (old port)
adv100() {   # only if gigabit is still offered: setting it restarts negotiation (one link down / up)
  [ "$ADV100" = "1" ] || return 0
  if ethtool "$IF" 2>/dev/null | sed -n '/Advertised link modes/,/Advertised pause/p' | grep -q "1000baseT"; then
    ethtool -s "$IF" advertise 0x008 && log "offering 100 Mbit/s full duplex only on $IF (renegotiating)"
  fi
}
log "guard started (v4, 5 Oct: reset after ${MISS_MAX} s silence, never while our controller runs, 100-only=${ADV100})"
eee_off; adv100
miss=0; up_before=0
while true; do
  up=$(cat /sys/class/net/$IF/carrier 2>/dev/null || echo 0)
  if [ "$up" = "1" ] && [ "$up_before" != "1" ]; then eee_off; adv100; miss=0; sleep "$GRACE"; up_before=1; continue; fi
  up_before=$up
  if [ "$up" = "1" ]; then
    if ping -c1 -W1 -I "$IF" "$PC" >/dev/null 2>&1; then miss=0; else miss=$((miss + 1)); fi
    if [ "$miss" -ge "$MISS_MAX" ] && pgrep -f "r1mc2 [s]erve" >/dev/null; then
      log "link UP but $PC silent for ${MISS_MAX} s; our controller is running: NOT resetting (operator decides)"
      miss=0; sleep "$GRACE"; continue
    fi
    if [ "$miss" -ge "$MISS_MAX" ]; then
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
