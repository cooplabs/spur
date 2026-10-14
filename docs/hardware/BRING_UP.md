# First bring-up on your R1

This is the order in which we brought r1mc3 up on our own Unitree R1 with two Dex3-1 hands, written as a guide for
yours: what you need, what to check before anything moves, how the robot is taken over on its stand, which safety tests
we ran there, what to record so the simulator can be fitted to your robot, and finally the first balancing on the
floor. Every number marked "our robot" was measured on one R1 (one backpack, one firmware) - expect yours to differ in
the details and to be similar in kind.

**Read first:** [`../SAFETY.md`](../SAFETY.md) (the safety model), [`JOINT_DIRECTIONS.md`](JOINT_DIRECTIONS.md) (what
a positive value does on the real robot), [`DEPLOY.md`](DEPLOY.md) (install / update / rollback).

**Rehearse in the simulator first.** Everything below except the physical steps can be typed against the simulated
robot: the DDS-bridge simulator runs the same binaries (`r1rt`, `r1guard`, `r1d`, `r1log`, the `r1` CLI) against a
simulated R1, standing or hanging on a modelled stand ([`../sim/DDS_BRIDGE.md`](../sim/DDS_BRIDGE.md),
[`../SIMULATOR.md`](../SIMULATOR.md)). Typing a whole session there once, with the same words, is the cheapest way to
find a wrong step.

---------------------------------------------------------------------------------------------------------------------

## 1. The rules for every step

These are not suggestions: the controller is built around them, and the steps below assume them.

- **Nothing automatic ever makes the robot limp.** Damping (kp 0, the joints go soft) is entered only by the operator:
  `r1 damp`, Unitree's remote (L2+B), or the hardware emergency stop. No r1mc3 process damps on a fault, a signal, a
  stop or a crash. The automatic fallbacks keep the robot stiff: the inline policy, then the guard taking over, then a
  freeze to a stand pose (see [`../SAFETY.md`](../SAFETY.md)).
- **One step at a time.** Each step is typed, its result read, and only then the next. Every moving step says, before
  it is typed, what will move and how to stop it.
- **Read-only first.** A new installation starts in MONITOR, where nothing of ours is published to the motors.
- **No unproven mode switch.** A mode is used on the robot only after it was rehearsed in the simulator.
- **The stop words.** `r1 stop` holds every arm / head group where it is, stiff (balance, or the stand hold, carries
  on). If that is not enough, the operator damps: `r1 damp`, the remote's L2+B, the emergency stop. Arm goals also end
  on `r1 arm-stop`. Every streamed command (`r1 jog`, `r1 head`, `r1 hand`, the recordings) carries a short time-to-live:
  a dead client ends in a hold, never in a limp.
- **Joint directions are never guessed.** Say what a step will do from the verified table
  ([`JOINT_DIRECTIONS.md`](JOINT_DIRECTIONS.md)) before it is typed. (We once described the elbow backwards in a session;
  the robot moved as commanded - the words were wrong. Since then every direction comes from the table or the model.)
- **A person is with the robot.** On the stand, someone watches it; on the floor, someone holds the robot's back frame
  for the takeover, every freeze, every guard test and the end, and the robot never stands unattended.
- **The session always ends the same way:** hold the robot -> `r1 damp` -> onto the stand -> `r1 shutdown` (it saves
  and verifies the logs) -> power off.
- **Never run two controllers' takeovers at once.** r1mc3's ownership token covers its own processes only.

## 2. What you need

| Item | For | Notes |
|---|---|---|
| a stand that holds the robot hanging, feet clear of the floor | every stand step | ours: a pole with a padded cradle; the bar of the robot's back frame rests in the cradle, the lower back against the pole. The bar can lift out of the cradle if the feet take weight: keep the feet ~20 cm clear and move one joint at a time |
| Unitree's remote, the emergency stop, the charger | all | start with the battery >= 80 %; our robot used ~0.7 % / min hanging and ~1 % / min balancing (~115 W) |
| a workstation (Linux or macOS) with this repository | the `r1` CLI, copies of the logs, the fits | the CLI reaches `r1d` through an SSH tunnel |
| SSH access to the backpack | the install, the services, read-only checks | Unitree's backpack: Jetson Orin NX, L4T R35.3.1 (Ubuntu 20.04, aarch64) - what we validated on |
| Unitree's `dex3_server` on the backpack | the hands | Unitree's hand service (closed source, not part of this repository: install it from Unitree) |
| weights 20 / 50 / 100 / 200 / 500 g, a kitchen scale | pad presses, static poses with a payload | weigh each with whatever carries it |
| a ChArUco board (printed flat, measured), a plain matte white panel (foam board) | the head camera's lens, vignetting, depth | [`CAMERA_CALIBRATION.md`](CAMERA_CALIBRATION.md) |
| a level reading 0.1 deg (optional) | the IMU's mounting angle | without one, slow waist sweeps on the stand give the IMU-to-pelvis rotation ([`../IDENTIFICATION.md`](../IDENTIFICATION.md) section 4) |
| tape measure, painter's tape | the floor part | |

**Two terminals.** *Workstation* = the `r1` CLI (and the copies / fits). *Backpack shell* = an SSH shell on the
backpack (the install, `sudo` lines, read-only system checks). Open the tunnel to `r1d` once and leave it:

```bash
ssh -N -L 8780:127.0.0.1:8780 unitree@<backpack>      # r1d listens on the backpack's 127.0.0.1:8780 only
r1() { (cd /path/to/r1mc3 && bash scripts/run.sh python -m r1mc3.cli "$@") }   # zsh / bash: a function, not a variable
```

Every `r1` word prints ONE JSON result (`ok`, `code`, `moved`, `data`; `r1 --pretty <word>` indents it); exit 0 = ok,
1 = refused (with the reason in words), 2 = `r1d` not reachable (check the tunnel). On the backpack the same CLI is
`/opt/r1mc3/current/bin/python -m r1mc3.cli <words>` - the fallback if the tunnel is down. In zsh, quote a list
argument (`'weights_g=[20,50,100,200]'`).

## 3. Unitree's own modes (what the robot does before r1mc3 takes it)

| Unitree mode | What it does | Notes from our robot |
|---|---|---|
| **zero torque** (the power-on mode) | every motor limp | the built-in controller still PUBLISHES `rt/lowcmd` (~650 /s, kp 0 / kd 0) |
| **stand lock** (remote / app) | every motor stiff at a **fixed preset pose** - not "where it is" | **selecting it moves the robot** into the preset (on the stand: the legs and arms swing to it). Details: [`STAND_LOCK.md`](STAND_LOCK.md) |
| a balancing / walking mode (remote / app) | Unitree's own balance | the floor takeover starts from it (section 9) |

r1mc3 never asks Unitree's controller to enter a mode. It takes the motors over with Unitree's motion-switcher
`ReleaseMode` call, only after its own entry checks, and mirrors Unitree's last command before blending into its own.
**A takeover needs a FRESH command from Unitree's controller to mirror** (that is how it starts without a jump), so the
built-in must be publishing when you ask. After a release, our robot's remote sometimes refused to re-enter the stand
lock in development mode until a power cycle - plan the day so that the stand part ends with `r1 damp` and the floor
part starts from a fresh power-up, or from MONITOR.

---------------------------------------------------------------------------------------------------------------------

## 4. Install (nothing moves)

[`DEPLOY.md`](DEPLOY.md) has the full procedure. In short:

| ID | Where | Words | What moves | Pass |
|---|---|---|---|---|
| I1 | workstation | `ssh unitree@<backpack> 'bash -s' < deploy/inventory.sh` (read-only: L4T, kernel, CPUs, disk, Python / uv, build tools, CycloneDDS, existing releases / units, the network interfaces, the clock) | nothing | every row OK, or an ACTION with its fix |
| I2 | backpack | the ACTION rows, each on its own: `uv` (a static binary), `build-essential`, CycloneDDS-CXX 0.10.2 built on the backpack with `rt/idl/build_cyclonedds.sh` (51 s on our Orin), optionally `rt-tests` + `stress-ng` for the latency check | nothing (CPU load) | installed |
| I3 | workstation | `bash deploy/release.sh --out out/releases` -> `r1mc3-<version>-<git7>.tar.gz` + `.sha256` | nothing | the tarball |
| I4 | backpack | copy it, then `sudo bash r1mc3-<id>/deploy/install.sh --tarball /tmp/r1mc3-<id>.tar.gz --dry-run`, then without `--dry-run` | nothing (a build: ~1 min on ours) | every file checked against the release's sha256 list; the venv, `r1rt`, `r1guard` built under `/opt/r1mc3/releases/<id>` |
| I5 | backpack | the robot's config from the robot profile: `/opt/r1mc3/releases/<id>/bin/python -m r1mc3.rtconfig --profile robot --out /etc/r1mc3/r1rt.cfg`, then `/opt/r1mc3/releases/<id>/bin/r1rt --config /etc/r1mc3/r1rt.cfg --check-config` | nothing | `{"config_ok": true, ...}`; every key with its source in `/etc/r1mc3/r1rt.cfg.provenance.json` |
| I6 | backpack | `sudo bash /opt/r1mc3/releases/<id>/deploy/activate.sh --id <id>`, then `bash /opt/r1mc3/current/deploy/status.sh` | nothing (refused while r1mc3 owns the motors) | `current` -> the new release; the units installed |
| I7 | backpack | edit `/etc/r1mc3/cyclonedds.xml`: the `NetworkInterface` = the backpack's interface on the robot's network (written once, never overwritten by an update) | nothing | [`NETWORK.md`](NETWORK.md) |

What bit us here: the units' real-time scheduling was refused until the RT budget was set (docker / containerd on the
backpack gave every service a CPU cgroup with a real-time budget of 0 - the install now writes
`/etc/sysctl.d/60-r1mc3-rt.conf`, [`RT_SETUP.md`](RT_SETUP.md)); CycloneDDS's 4 MB receive buffer failed against the
kernel's default `rmem_max` (the same file raises it to 8 MB); uv put its managed Python under `/root` when run with
`sudo`, unreadable by the services (the install now sets `UV_PYTHON_INSTALL_DIR`).

## 5. Read-only checks (MONITOR: nothing of ours is published)

Start the services with the **stand-session lock** for everything on the stand: the core then refuses every entry into
BALANCE for its whole lifetime ([`STAND_LOCK.md`](STAND_LOCK.md)). The robot hangs on its stand, Unitree in zero
torque.

| ID | Where | Words | What moves | Pass / what our robot showed |
|---|---|---|---|---|
| R1 | backpack | `sudo /opt/r1mc3/current/deploy/bin/r1mc3-svc start --stand`; `systemctl is-active r1rt-stand r1rt r1guard r1d r1log`; `journalctl -u r1rt-stand -u r1guard -b --no-pager -o cat \| grep '"ready"'` | nothing | both ready lines with `rt_granted: true`. Ours: writer FIFO 90 on CPU 7, policy 80 / DDS 85 on CPU 6, guard 88 on CPU 5 |
| R2 | backpack | DDS discovery: `CYCLONEDDS_URI=file:///etc/r1mc3/cyclonedds.xml /opt/r1mc3/current/venv/bin/cyclonedds ls -i 0 -r 10s` (who writes `rt/lowcmd`, who reads `rt/arm_sdk`) + r1rt's 1 Hz status line | nothing | ours: `rt/lowstate` 1052 /s, 0 bad CRC; only Unitree's controller writes `rt/lowcmd` |
| R3 | workstation | `r1 --pretty status` | nothing | MONITOR, token owner none, "r1guard: alive, ARMED", the stand lock shown, no alarms |
| R4 | workstation | `r1 takeover` (must be REFUSED under the stand lock) | nothing | refused WRONG_MODE, `moved: false`, Unitree still publishing |
| R5 | workstation | **the DDS-latency check**: `r1 record run imu_static duration_s=120 --yes` (read-only: r1log writes r1rt's telemetry every tick), then the state age from r1log | nothing | ours: state age p50 0.49 / p99 1.02 / max 2.97 ms, no gap > 5 ms, the same with the Wi-Fi busy |
| R6 | backpack | **the clock check**: `sudo nvpmodel -q; sudo jetson_clocks --show; timedatectl` | nothing | recorded (ours: MAXN, 1.98 GHz, chrony) |
| R7 | backpack | **the policy-timing check**: `r1rt_policy_bench` ([`RT_SETUP.md`](RT_SETUP.md)) idle, then with `stress-ng` on CPUs 0-4 | nothing (CPU) | ours: one policy step p50 / p99 / max 112 / 278 / 452 us idle, 183 / 360 / 547 us loaded - far inside the 20 ms period |
| R8 | backpack | **the RT-grant and RT-latency checks**: [`RT_SETUP.md`](RT_SETUP.md) (`cyclictest` before / after the CPU partition, then r1rt's own wake-up lateness from r1log) | nothing (CPU load) | ours after the partition: loaded p99.9 23 / max 155 us; r1rt 0 skipped deadlines, lateness max ~330 us |

**A Wi-Fi lesson worth checking on your backpack.** The previous controller fell over once because its control thread
read the network adapter's link speed from sysfs and that read blocked for ~4 s while the Wi-Fi reconnected
([`../case_studies/V2_FALL.md`](../case_studies/V2_FALL.md)). We reproduced it on the stand: with the robot damped, a
script that reads the adapter's `speed` file every 100 ms while the Wi-Fi is toggled saw reads of 3.80 / 3.89 s right
after the Wi-Fi came back. r1mc3's real-time processes never touch sysfs, the network or files, and during the same
Wi-Fi drops our robot's state stream to r1rt did not change (1052 /s every second). Run the same check if you add
anything to the backpack.

## 6. The hands (the body under Unitree; only the fingers move)

Start Unitree's `dex3_server`. **Its first start after each power-up moves the fingers** (a short calibration wiggle
of both hands): hands clear of people and objects. On our robot it ignored hand commands for ~20 s after starting, and
the fingertip pads streamed only after the first hand command.

| ID | Where | Words | What moves | Pass / what our robot showed |
|---|---|---|---|---|
| H1 | workstation | `r1 hand-state left`, `r1 hand-state right` | nothing | valid, state age < 50 ms, 9 pads x 12 cells, temperatures |
| H2 | workstation | open: `r1 hand left 0,0,0.05,-0.05,-0.05,-0.05,-0.05 --ttl 1` (right: `0,0,-0.05,0.05,0.05,0.05,0.05`); half close: `r1 hand left 0,0.25,0.35,-0.35,-0.35,-0.35,-0.35 --ttl 1`, then open | the fingers, slowly | within 0.05 rad - except thumb_1, which stops ~0.06 rad short of 0 on both of our hands every time (not a fault) |
| H3 | workstation | `r1 --pretty status` -> `hands_contact` (the pads' tare is taken from streaming pads of an OPEN hand) | nothing | valid, tared, no pad in contact at rest |
| H4 | workstation | the pad presses and finger holds of [`../IDENTIFICATION.md`](../IDENTIFICATION.md) (optional) | the fingers (holds) | - |

What our hands taught us: the pads are a **firm-contact switch, not a force gauge** (nothing below ~10 N on a tip,
then thousands of raw units, depending on where the tip lands) - the grip verdict uses them only as "touching / not";
a finger holding 0.9 N m heats ~6 C/s (41 -> 71 C in ~5 s), so long squeezes are limited by temperature; closing in free
air takes 0.34-0.41 s.

---------------------------------------------------------------------------------------------------------------------

## 7. The body on the stand (ON_STAND; feet off the floor)

**Everything in this section moves the robot's body.** The robot hangs on its stand, nobody under it, the stand-session
lock on. ON_STAND is r1mc3's hold for a hanging robot: it releases Unitree's controller, mirrors its last command and
blends in 1 s into a stiff hold of the MEASURED pose (the legs at the stand gains, the arms softer).

| ID | Where | Words | What moves | How to stop | Pass / what our robot showed |
|---|---|---|---|---|---|
| S0 | - | select Unitree's **stand lock** (or stay in zero torque; both publish a command to mirror) | **the stand lock moves the robot into its preset pose** - say so first | the remote | the robot at the preset ([`STAND_LOCK.md`](STAND_LOCK.md)) |
| S1 | workstation | `r1 status`, then `r1 on-stand` | the joints stiffen where they hang (no motion intended) | `r1 stop`; the operator: `r1 damp` / L2+B / emergency stop | ours, from the stand lock: release 0.16-0.22 s, settle 0.014-0.016 rad, one writer |
| S2 | workstation | **the joint-direction check** - one joint at a time, +0.10 rad and back, the watcher saying what moved vs the printed expectation: `bash scripts/run.sh python -m r1mc3.hw1.dircheck --plan --pose stand_lock` prints both words per joint | one joint, 0.10-0.20 rad | `r1 stop` | 26 / 26 on ours ([`JOINT_DIRECTIONS.md`](JOINT_DIRECTIONS.md)). **Any mismatch: stop, damp at the end, nothing else moves until the cause is known** |
| S3 | workstation | the identification recordings (steps, chirps, friction ramps, static poses, IMU statics): [`../IDENTIFICATION.md`](../IDENTIFICATION.md). Every recording is planned first (`r1 record plan ...`: the limits, the speed, what moves - refused before anything moves if it leaves a limit), then run with `--yes` | one joint at a time, small | `r1 stop` (the recording also ends in a hold when its stream's time-to-live runs out) | the manifest `done` |
| S4 | workstation | gentle joint-limit checks: `r1 jog <group> <rates> --ttl 0.3` repeated toward a limit in small steps, reading `r1 hold <group>` | one joint, <= 0.2 rad/s | stop typing (the time-to-live ends it) | the core refuses beyond the profile's limit (`LIMIT`, `moved: false`); ours had no hard stop inside the limits on the joints checked |
| S5 | workstation | **the freeze from the stand**: both legs to ~0.10 rad off the stand pose (`r1 record run static_pose group=leg_left 'q=[...]' move_s=3 hold_s=1 --yes`, then `leg_right`), `r1 stand-freeze` (a dry run: the ramp time and the check that every target is within 0.15 rad), `r1 stand-freeze --yes`, `r1 stand-hold`, `r1 resume` | both legs, 0.1 rad in 0.4 s | `r1 stop` | ours: the 0.4 s ramp measured 0.400 s, torque step hip 3.0 / knee 2.0 N m |

### 7.1 The board-hold test - what Unitree's motor board does when our commands stop (explicit OK; on the stand only)

The guard takes over ~10 ms after the core stops, so the board's own behaviour can only be seen with the guard
switched to OBSERVE-ONLY for the test. That disables a safety function: do it only on the stand, arms down, and switch
the armed guard back straight after.

| ID | Where | Words | What moves | Pass / stop if |
|---|---|---|---|---|
| BH1 | backpack | `r1 status` (ON_STAND, the guard on standby); `sudo systemctl start r1guard-observe` (the units conflict: systemd stops the armed guard); `systemctl is-active r1guard r1guard-observe` | nothing (the guard can no longer take over) | `inactive` / `active` |
| BH2 | workstation | `r1 board-hold --pause-ms 20` (a dry run: the plan + its checks), then `r1 board-hold --pause-ms 20 --yes` - r1d records the state from 1 s before, pauses r1rt (SIGSTOP) for the time, resumes it (SIGCONT), records 2 s after. Then 100, 300, 400, 500 ms | what the board does without commands | **stop at the first pause that moves a joint > 0.03 rad** (`data.lowstate.during.max_joint_motion_rad`) or if you see the legs drop. Longer pauses are refused by the word |
| BH3 | backpack | `sudo systemctl start r1guard`; `systemctl is-active r1guard r1guard-observe`; `r1 status` | nothing | `active` / `inactive`; "r1guard: alive, ARMED". **No further moving step until the guard is armed** |

**What our board did:** it HOLDS the last command for ~385 ms (20 / 100 / 300 ms: < 0.0001 rad of motion), then every
loaded joint lets go in the same 1 ms tick into a **damped** release (kp 0, kd ~12.5: the hips sank ~0.12 rad/s - not
a free fall). At 400 ms ours moved 0.002 rad, at 500 ms 0.014-0.028 rad. r1rt's resume blends back from the measured
pose (it does not snap back to its old target). This is why the guard's deadline (10 ms) is set far below the board's
hold, and why a whole-machine stall longer than ~0.4 s is dangerous whatever the software does
([`../sim/BOARD_MODEL.md`](../sim/BOARD_MODEL.md)).

### 7.2 The guard-takeover test - the guard takes the motors; only the operator gives them back

| ID | Where | Words | What moves | Pass |
|---|---|---|---|---|
| GT1 | workstation | `r1 status` (ON_STAND, token r1rt, the guard armed); `r1 guard-takeover --stop-s 3` (the dry run), then `r1 guard-takeover --stop-s 3 --yes` (r1d pauses r1rt for 3 s; if the guard has not taken the token within 0.2 s it resumes r1rt at once and reports FAULT) | nothing should: the guard holds the core's last command at the stand gains | the guard ACTIVE; r1rt silent after it resumes |
| GT2 | workstation | `r1 status` | nothing | "r1rt: ON_STAND, running but SILENT - the guard holds the motors"; token r1guard |
| GT3 | workstation | `r1 reclaim` (operator only; ON_STAND only, r1rt alive with fresh state) | nothing should: r1rt blends from exactly what the guard was sending | ON_STAND, token r1rt; the guard re-arms by itself |

**What our robot did:** the guard took over **9.8-10.7 ms** after r1rt stopped (four runs on two days), the legs moved
< 0.003 rad, one writer at a time, `r1 reclaim` gave the motors back with a command step of exactly 0, and the guard
re-armed by itself.

### 7.3 The end of a stand session

| ID | Where | Words | What moves |
|---|---|---|---|
| E1 | - | the operator holds the robot (on the stand: hands on it) | nothing |
| E2 | workstation | `r1 damp` (only on the operator's word) | **LIMP**: kp 0 + damping on every joint |
| E3 | workstation | `r1 shutdown --label <name>` (accepted only in DAMP / MONITOR): r1d stops r1log, copies the session's logs, recordings and its request journal into a new session folder, sha256 on both sides, prints "safe to power off now"; then copy the folder to the workstation (`rsync -a unitree@<backpack>:<data.saved.folder>/ <dest>/`) and check the sha256 again | nothing |
| E4 | - | power off | already limp |

---------------------------------------------------------------------------------------------------------------------

## 8. Before the first balancing

Do not take the robot to the floor until:

1. **The stand part is clean:** ON_STAND, the board-hold and guard-takeover tests, the freeze from the stand, the
   joint-direction check 26 / 26.
2. **The policy stands on YOUR identified model.** Fit the simulator to your robot's stand recordings
   ([`../IDENTIFICATION.md`](../IDENTIFICATION.md)), then run a robustness batch on it - the method and our result:
   [`../sim/ROBUSTNESS_BATCH.md`](../sim/ROBUSTNESS_BATCH.md); the public tools: `python examples/robustness_batch.py`
   and `python -m r1sim.viewer.batch`. Ours (100 seeds x 4 cases) passed 400 / 400 with no retraining; its margin was
   in latency (it degrades at 15 ms of pure added delay, falls at 20 ms), not in friction.
3. **The latency on the backpack is far inside that margin** (ours: state age p99 ~1 ms, r1rt's lateness < 0.5 ms).
4. **The camera does not starve the motor link** if your camera shares a bus with it: on our robot the head camera
   and the robot link shared one USB 2 adapter, and streaming three camera streams cut the state rate from 1052 to
   ~997 /s until the depth was held only for an observation ([`NETWORK.md`](NETWORK.md)).

## 9. The first balancing (the floor)

Lift the stand-session lock: the whole stack restarts WITHOUT `--stand` (never a lone `systemctl start r1rt`: the
other processes attach to the core's shared memory once, at their start) - [`STAND_LOCK.md`](STAND_LOCK.md) "Lifting
it". The robot stands under Unitree's own balancing mode, a person holding its back frame.

| ID | Where | Words | What moves | How to stop | Pass / what our robot showed |
|---|---|---|---|---|---|
| F1 | workstation | `r1 --pretty status` | nothing | - | MONITOR, the guard ARMED, Unitree's command fresh AND stiff (the takeover's gate: legs kp >= 10, the command <= 100 ms old), no stand lock, tilt small |
| F2 | workstation (**a person holding**) | `r1 takeover` (the core's gates -> `ReleaseMode` -> mirror Unitree's last command -> a 1 s blend into the balance policy) | the legs: Unitree lets go, ours blends in; the arms / head hold | `r1 stop` (balance continues); the operator: `r1 damp` / L2+B / emergency stop | ours, twice: the release in one call, 0.19-0.21 s; BALANCE, tilt 0.2-0.5 deg; Unitree silent afterwards |
| F3 | workstation, backpack | **the health watch**, 2-5 min standing still: `r1 --pretty status` every ~30 s + r1rt's status line (`journalctl -u r1rt -o cat -n 3 --no-pager \| grep '"status":"r1rt"'`) | nothing intended | `r1 stop`; the person holds | ours: 0 skipped deadlines, 0 overruns, 0 faults, state age ~1 ms, the guard ARMED throughout |
| F4 | workstation (**a person holding**) | **the operator freeze and back**: `r1 freeze` (the legs + waist ramp to a stand pose in 0.4 s at the stand gains, the ankles shifted for where the arms are; the arms hold; never limp) -> `r1 balance` (the policy takes the legs again from the stand pose) -> `r1 resume` (clears the freeze's stop) | the legs | the person keeps holding; `r1 damp` only on the operator's word | ours: tilt 0.4 -> 3.0-3.4 deg during the freeze (the simulator said 2.4), `r1 balance` accepted at once, tilt back to ~1 deg |
| F5 | workstation | arm / head goals, slowly at first (`r1 hover ...`, `r1 move-joint ...`, `r1 skill ...`: the API docs) | one arm at a time | `r1 stop`, `r1 arm-stop` | planned before anything moves; refusals in words |
| F6 (optional, **the last step**) | backpack (**explicit OK; a person holding**) | the guard while balancing: `sudo kill -STOP $(pidof r1rt); sleep 3; sudo kill -CONT $(pidof r1rt)`; `r1 --pretty status` | nothing should: the guard balances in place with its own (warm) policy copy | `r1 damp` (it reaches the motors through the guard) | there is no hand-back from a balancing guard: end the session after it |

The end on the floor: the person holds the robot -> `r1 damp` (**limp**: the person takes the weight) -> onto the stand
-> `r1 shutdown` + the copy -> power off.

---------------------------------------------------------------------------------------------------------------------

## 10. What bit us (so it does not bite you)

| What happened | What to do |
|---|---|
| A different USB network adapter had been plugged in since the last session: the interface name changed and nothing found the robot | check the motor-link interface at every power-up (`ip -br addr`, `ping -c 20 192.168.123.161`, Unitree's robot computer); name the adapter by a udev rule on its MAC |
| Our USB network adapter needed a replug after every power-up before the link came up | the same check; a scripted replug of the USB device is possible (unbind / bind), but test it on your hardware first |
| The Wi-Fi dongle dropped off USB twice in 20 minutes | turning off the driver's power saving fixed it on ours (`/etc/modprobe.d/`); the real-time processes never noticed the drops |
| The motor link's interface carried a default route; during a weak Wi-Fi moment DNS went out the wrong way | the robot-network interface should never be a default route (netplan: no gateway on it) |
| The real-time units were refused FIFO scheduling | the RT budget in `/etc/sysctl.d/60-r1mc3-rt.conf` (installed by default; [`RT_SETUP.md`](RT_SETUP.md)) |
| A kernel message printed to the two 115200-baud serial consoles stalled the guard's cycle 24.5 ms | `kernel.printk = 1 4 1 7` in the same file (only emergencies reach the serial consoles; installed by default) |
| The CPU partition picked the wrong interrupt line for the USB controller (the first match had 0 interrupts) | it now picks the busiest line of the controller; check `partition.sh show` |
| The filesystem of the backpack's NVMe carried ext4 errors from months before, never checked | back the logs up and check it ([`E2FSCK.md`](E2FSCK.md)); until then copy and verify every session's logs |
| The stand lock is a fixed preset, not "stiff where it is" | say so before selecting it: the robot moves into it |
| The fingers' first ~20 s after the hand service starts | wait before the first hand command |
| A hand stopped a grasp at its 72 C limit after ~30 back-to-back grasps | let the fingers cool; `r1 hand-state` shows the temperatures |
| Planning on the Orin is ~5-6 x slower than on a laptop (a default hover p50 0.44 s) | fine for single goals; measure yours with `python -m r1mc3.plan.bench` |
| The camera and the robot link on one USB 2 adapter: the camera cost ~5 % of the state packets | stream only what you use; hold the depth stream only for an observation; measure with `tools/cam/impact.py` |

## 11. What is not solid (said plainly)

- Everything here was validated on **one** R1 with **one** backpack and firmware, in October 2026. Unitree's firmware
  updates can change the board's behaviour without commands, the stand-lock preset, the motion switcher's behaviour or
  a joint's sign: re-run the board-hold test, the joint-direction check and the read-only checks after any update.
- The board-hold test stopped at 500 ms on purpose: what the board does after its damped release is seen for ~100 ms
  only.
- The friction was identified on the hanging robot (the bearings unloaded); the standing friction is not measured. The
  robustness batch shows the policy indifferent to the ankle friction up to 6 N m.
- The floor part was validated standing in place (balance, freezes, arm motions, small pushes in simulation). Walking is
  not part of this controller.
- The guard's takeover while BALANCING was rehearsed in the simulator; we did not run step F6 on our robot.
