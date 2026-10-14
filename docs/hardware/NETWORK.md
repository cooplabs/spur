# The network: the motor link, DDS, and what not to do in a control loop

## The robot's network

The R1's computers talk over Unitree's internal Ethernet network with factory addresses that are the same on every
R1: **192.168.123.161** = the robot's own computer (the motor board's DDS endpoint, Unitree's controller, the motion
switcher, the camera services) and **192.168.123.164** = the backpack. r1mc3's real-time core talks to the motors over
this link with CycloneDDS: it reads `rt/lowstate` (on our robot 1052 samples/s) and writes `rt/lowcmd` (500 /s).
Nothing else matters as much: a late or lost state stream is a balancing robot without feedback.

Set the interface once in `/etc/r1mc3/cyclonedds.xml` (installed only if missing, never overwritten by an update):

```xml
<NetworkInterface name="<the backpack's interface on the robot's 192.168.123.x network>" priority="default" multicast="default"/>
```

Never the Wi-Fi. Between r1mc3's own processes nothing goes over DDS: they share memory (seqlocks and single-producer
rings on POSIX shared memory).

**On our backpack** the built-in Ethernet port had downshifted to 100 Mbit/s, so the robot link ran through a USB
network adapter, named by a udev rule on its MAC address so the name survives replugs and reboots. Two consequences we
lived with: the adapter shared a USB 2 bus (and one interrupt line) with the Wi-Fi dongle and the hands, and it needed
a replug after power-ups before its link came up. If you can, give the robot link its own controller.

Check at every power-up (read-only): `ip -br addr show <iface>` (192.168.123.164), `ping -c 20 -i 0.2 192.168.123.161`
(0 % loss), and r1rt's status line once it runs (`lowstate_rx` ~1000 /s, `crc_bad` 0).

## The two CycloneDDS files

| File | Used by | Receive buffer | Why |
|---|---|---|---|
| `/etc/r1mc3/cyclonedds.xml` | everything (r1d, r1hands, r1log, the tools) | 4 MB (`net.core.rmem_max` raised to 8 MB by the install) | headroom for the camera, the hands and bursts |
| `/etc/r1mc3/cyclonedds_rt.xml` (an overlay read AFTER the shared file; a later setting wins) | the real-time units only: r1rt, r1rt-stand, r1guard, r1guard-observe (`CYCLONEDDS_URI=file:///etc/r1mc3/cyclonedds.xml,file:///etc/r1mc3/cyclonedds_rt.xml`) | **256 kB** (Linux doubles it: 512 kB effective) | after a stop, r1rt / r1guard publish nothing until they hold a LIVE state, and they must first drain every sample queued during the stop. The 4 MB buffer holds ~1,700+ LowState samples (~1.6 s at 1052 /s) of stale state; 256 kB holds ~200 (~0.2 s). Normal operation needs a few: our state age stayed under 3 ms with the Wi-Fi busy |

`deploy/install.sh --rt-dds-rcvbuf bounded` (the default) installs the overlay; `shared` installs a no-op overlay
instead. After the board-hold / guard-takeover tests, `r1 status` shows the guard's `resume_wait_ms_last` /
`resume_wait_old_samples_last` - on our robot 0 waits and 0 stale samples.

QoS: r1mc3's writers RELIABLE + KEEP_LAST, readers BEST_EFFORT + KEEP_LAST. Unitree's CRC32 is checked both ways
(`crc_bad` counts the failures; a bad state is not used, a bad command is dropped by the board).

## What not to do on the backpack (lessons, each one learned the hard way)

| Lesson | What happened | What r1mc3 does |
|---|---|---|
| **Never read network or system state in the control loop** | the previous controller read the network adapter's link `speed` from sysfs in its control thread; during a Wi-Fi reconnect that read blocked ~4 s and the robot fell ([`../case_studies/V2_FALL.md`](../case_studies/V2_FALL.md)). On the stand we reproduced reads of 3.80 / 3.89 s | the real-time processes never touch sysfs, files, the network stack beyond their DDS sockets, or Python; logging is lock-free rings drained by another process |
| **A link watchdog must never reset the motor link while a controller runs** | a watchdog that turned the adapter off and on after 20 s of ping silence (and ran `ethtool` on every link-up) did not know about r1mc3's processes | if your backpack runs such a watchdog, make it log-only while `r1rt` / `r1guard` run (or remove it): a reset cuts the state and commands for many seconds |
| **A shared USB bus is a shared risk** | the camera, the Wi-Fi and the hands on the same USB 2 bus as the robot link; the Wi-Fi dongle dropped off USB twice in 20 min (its driver's power saving: off fixed it) | the state stream to r1rt did not change during the drops; still, keep heavy traffic off the motor link's bus |
| **The camera can starve the state stream** | streaming the head camera's left + right + depth through the same adapter (~74 Mbit/s) cut the state rate from 1052 to ~997 /s and raised the state age p99 from 1.0 to 2.9 ms | the depth stream is held only for an observation (+0.5 s), the right image only for calibration recordings, a 16 KB socket window: then 1052 /s and p99 1.0 ms with the camera working. Measure yours: `tools/cam/impact.py` (windows off / work / work_fast / calib / off_again) |
| **The robot network must never be a default route** | the robot-network interface carried a default route; while the Wi-Fi was in a "limited" state, DNS went out the robot link and failed | no gateway on the robot-network interface (netplan); the Wi-Fi profile with the lower route metric |
| **Serial consoles are slow** | a 3-line kernel report printed at 115200 baud stalled the guard's cycle 24.5 ms | `kernel.printk = 1 4 1 7` (installed by default) |

## Measuring the link (the DDS-latency check; read-only)

`r1 record run imu_static duration_s=120 --yes` in MONITOR records r1rt's telemetry every writer tick: `t_state_rx_ns`
(when the state used was received) against `t_wake_ns` (the tick) = the state's age at r1rt, plus the board's own 1 kHz
tick in every state (the content age: how old the newest sample's content is, which a late receive path cannot hide).
Do it once idle and once with the Wi-Fi busy (a large copy). Ours: state age p50 0.49 / p99 1.02 / max 2.97 ms, no gap
> 5 ms, both ways. For the margin this leaves the balance policy: [`../sim/ROBUSTNESS_BATCH.md`](../sim/ROBUSTNESS_BATCH.md)
(the policy degrades at ~15 ms of pure added delay) and [`../sim/STATE_AGE.md`](../sim/STATE_AGE.md).
