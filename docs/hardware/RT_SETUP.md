# Real-time set-up of the backpack

What it is for: `r1rt`'s writer must wake every 2 ms (500 Hz; 200 Hz is the fallback rate) and never miss by much,
and the balance policy must get its 20 ms slots. The previous controller ran with 19-38 missed deadlines per second
and no real-time scheduling at all. r1mc3 runs on the **stock L4T kernel** (PREEMPT, not PREEMPT_RT) with a staged
set-up, each stage measured before and after and reversible:

- **Stage 1 (what we run):** systemd units with real-time limits (`deploy/units`), a runtime CPU partition
  (`deploy/rt/partition.sh`), fixed clocks (`nvpmodel` MAXN + `jetson_clocks`), the RT budget sysctl. No bootloader
  change.
- **Stage 2 (not needed on our robot):** `isolcpus` on the kernel command line, then a PREEMPT_RT kernel. Not scripted
  here; worth it only if stage 1's numbers are not good enough on your backpack.

**The CPU plan** (the Orin NX has 8 cores): system services, every Python process and every other interrupt on CPUs
0-4; `r1guard` on CPU 5; the motor network interface's interrupt + r1rt's DDS receive and policy threads on CPU 6;
r1rt's writer alone on CPU 7. The units: r1rt FIFO 90 (writer) / 85 (DDS) / 80 (policy), r1guard FIFO 88,
`LimitRTPRIO`, `LimitMEMLOCK=infinity` (the processes mlock their memory), no CPU accounting keys.

**Robot state for every step below:** on its stand, nothing of ours publishing (r1mc3 stopped or in MONITOR). Nothing in
this page moves the robot.

## The RT-grant check (read-only)

| # | Words | What to record |
|---|---|---|
| 1 | `uname -a; cat /proc/cmdline; zcat /proc/config.gz \| grep -E 'PREEMPT\|RT_GROUP_SCHED\|NO_HZ\|CGROUP'` | the preempt model; whether `CONFIG_RT_GROUP_SCHED=y` (then a service with its own cpu cgroup may get a real-time budget of 0) |
| 2 | `ulimit -r; ulimit -l; cat /proc/sys/kernel/sched_rt_runtime_us /proc/sys/kernel/sched_rt_period_us; stat -fc %T /sys/fs/cgroup` | the RT budget, cgroup v1 / v2 |
| 3 | `sudo nvpmodel -q; sudo jetson_clocks --show; cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` | the power mode and clocks (this is also the clock check) |
| 4 | `cat /proc/interrupts \| grep -E 'xhci\|eth'`; `sudo bash /opt/r1mc3/current/deploy/rt/partition.sh show` | which interrupt line carries the motor network interface |
| 5 | with r1mc3 in MONITOR: `journalctl -u r1rt-stand -u r1guard -b -o cat \| grep '"ready"'` (`rt_granted`, policy, priority, affinity per thread), `chrt -p $(pgrep -x r1rt)`, `taskset -cp $(pgrep -x r1rt)` | FIFO granted to r1rt / r1guard? On our backpack it was **refused** until the RT budget was set: docker / containerd (`Delegate=yes`) make systemd create a cpu cgroup for every service, each with `cpu.rt_runtime_us = 0`. The install's `/etc/sysctl.d/60-r1mc3-rt.conf` sets `kernel.sched_rt_runtime_us = -1` ([`DEPLOY.md`](DEPLOY.md)); our real-time threads are pinned to CPUs 5-7, so a runaway cannot freeze CPUs 0-4 |

## The RT-latency check (`cyclictest`, before and after the partition)

`cyclictest` (package `rt-tests`) and `stress-ng`. Two loads, 10 min each:

| # | Words | What moves |
|---|---|---|
| 1 | idle: `sudo cyclictest -m -p 90 -i 2000 -a 7 -t 1 -D 10m -h 3000 -q > /tmp/rtlat_idle.txt` | nothing |
| 2 | loaded: in a second shell `stress-ng --cpu 6 --io 2 --vm 1 --vm-bytes 256M -t 11m`, and the Wi-Fi busy (a large copy over it); then the same cyclictest -> `/tmp/rtlat_load.txt` | nothing (CPU load) |
| 3 | apply the partition: `sudo bash deploy/rt/partition.sh apply --dry-run`, then `apply --move-tasks` (every value saved under `/var/lib/r1mc3/partition_backup/<UTC>/` first): IRQs -> CPUs 0-4, the motor interface's line -> CPU 6, the unbound-workqueue mask -> 0-4, a systemd drop-in `CPUAffinity=0-4` for services started later, `nvpmodel -m 0` (MAXN), `jetson_clocks`; `--move-tasks` moves the running user-space tasks to 0-4 | nothing on the robot; the backpack's fan may speed up |
| 4 | restart r1mc3 in MONITOR, repeat the grant check's row 5, then 1-2 again on CPU 7 | nothing |
| 5 | with r1mc3 in MONITOR (its writer on CPU 7): r1rt's own wake-up lateness from its telemetry (r1log's `lateness` per tick) over 10 min - the number that matters | nothing |

The interrupt line: `partition.sh` picks the busiest line of the controller the motor interface hangs on (on our
backpack every USB device - the network adapter to the robot, the Wi-Fi dongle, the hands - shares one xHCI
controller and one interrupt; the first matching line had 0 interrupts). `--nic-irq N` names it explicitly. After a
reboot the partition unit (`r1mc3-partition.service`, pulled in by the core's units) re-applies it once before the core
starts. Check: `partition.sh show` -> the NIC line; `cat /proc/irq/<N>/smp_affinity_list` -> 6.

**Our numbers** (Orin NX, L4T R35.3.1, the stock kernel; `cyclictest` on CPU 7, 2 ms, FIFO 90; p99.9 / max):

| | Idle | Loaded |
|---|---|---|
| before the partition | 34 / 335 us | 25 / 51 us |
| after the partition | 12 / 42 us | 23 / 155 us |
| r1rt's writer (its own lateness), after | 99.99 % < 50 us, 2 ticks 250-500 us in 5 min | 33 ticks 50-100 us in 5 min, none above |

0 skipped deadlines all day outside deliberate stops; r1rt's lateness max 323-336 us. The deep idle state of the CPU
caused the idle worst case. Our pass rule was: under load, max <= 500 us and p99.9 <= 200 us (the 500 Hz budget leaves
>= 1.4 ms of slack). If your backpack misses it: drop the writer to 200 Hz first (a config value), then consider stage 2.

## The policy-timing check (`r1rt_policy_bench`; nothing moves)

The check times the policy **exactly as r1rt's policy thread calls it** (restore the runner state, the 74-value
observation, the 74-512-256-128-14 network as fixed-size Eigen, the 14 targets, the validity check), on a recorded
observation file, with NOTHING moving: no DDS, no shared memory, no motor command. It is built into every tree,
including the release's `build/rt` on the backpack. Run it with r1mc3 in MONITOR (nothing of ours publishes) and
**never at a priority >= the writer's (90) or on CPU 7 while r1rt runs**:

```bash
sudo /opt/r1mc3/current/build/rt/apps/r1rt_policy_bench \
  --weights /opt/r1mc3/current/build/rt/policy/policy_weights.r1w \
  --input /opt/r1mc3/current/tests/assets/t7/t7_policy_2000.r1pb \
  --mode periodic --rate-hz 50 --duration-s 120 --cpu 6 --fifo 80 --mlock \
  --out /var/lib/r1mc3/records/policy_timing_$(date -u +%Y%m%dT%H%M%SZ).json
```

`--mode periodic` = the thread's own pattern (one call per 20 ms wake: caches partly cold); `--mode burst --iters N`
= back-to-back calls (the inline fallback's pattern: when the thread's action is older than 40 ms the writer runs the
policy inside its 2 ms tick). The JSON says `pass_thread` (max compute <= 3 ms and lateness + compute <= the 20 ms
period), `pass_inline` (max compute <= 1.0 ms), `conditions_ok` (FIFO, the pin and mlock granted). Exit 1 = the C++
targets differ from the file's (the golden check), 2 = a usage / file error.

**Ours:** compute p50 / p99 / max **112 / 278 / 452 us** idle (total with the wake lateness max 508 us), **183 / 360 /
547 us** with `stress-ng` on CPUs 0-4 - both rules pass with a wide margin. (A laptop reference: ~57 us p50 periodic.)

## Undo everything

`sudo bash deploy/rt/partition.sh undo` (the newest backup); `sudo bash deploy/rt/site.sh undo` (the sysctl file); to
remove the units: stop r1mc3 (`r1mc3-svc stop`), remove `/etc/systemd/system/r1{rt,rt-stand,guard,guard-observe,d,log}
.service` + `r1mc3-inhibit.service` + `r1mc3-partition.service`, `systemctl daemon-reload`.

## Not solid

- One backpack, one kernel, one load pattern. Other JetPack versions are untested.
- The shared USB interrupt pins the Wi-Fi's and the hands' interrupts to CPU 6 together with the motor link's
  ([`NETWORK.md`](NETWORK.md)); a motor link on its own controller would be cleaner.
- The `slice` RT-budget method (throttling kept on) is built and tested in a fake root only.
