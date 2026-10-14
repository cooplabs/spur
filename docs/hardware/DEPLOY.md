# Deploying r1mc3 to the backpack: install, update, rollback

The scripts are in `deploy/` ([`../../deploy/README.md`](../../deploy/README.md) lists them). Target: Unitree's
backpack computer (Jetson Orin NX, L4T R35.3.1 = JetPack 5.1.1 = Ubuntu 20.04, aarch64) - what we validated on. Every
script takes `--dry-run` (prints what it would change) and `--root DIR` (a fake root: nothing outside it is touched;
how the test suite runs them on any machine).

Rules that hold for every step:
- Nothing in an install, switch, rollback or service stop publishes anything to the robot or damps it.
- A switch / rollback / service stop is **refused (exit 3) while r1rt or r1guard own the motors** (any publishing mode:
  a takeover, BALANCE, ON_STAND, STAND_FREEZE, GUARDED). End the session first the standard way: hold the robot ->
  `r1 damp` -> stand -> `r1 shutdown`.
- Releases install side by side under `/opt/r1mc3/releases/<version>-<git7>`; `current` is a symlink switched
  atomically; the previous release stays installed for a rollback.
- Site configuration under `/etc/r1mc3/` is written only where missing - never overwritten by an update.

## Prerequisites (first install only)

| # | Where | What | What moves |
|---|---|---|---|
| P1 | workstation | the read-only inventory: `ssh unitree@<backpack> 'bash -s' < deploy/inventory.sh` - L4T, kernel, CPUs, memory, disk, Python, uv, build tools, `cyclictest` / `stress-ng`, CycloneDDS, any r1mc3 release / config / units / processes, the network interfaces, internet, the clock - each row OK / ACTION / BLOCKER with the fix | nothing |
| P2 | backpack | CycloneDDS-CXX 0.10.2 for the C++ build, built ON the backpack into `/opt/r1mc3/cyclonedds-0.10.2` with `rt/idl/build_cyclonedds.sh` (51 s on our Orin) | nothing (CPU load) |
| P3 | backpack | `uv` (a single static binary in `/usr/local/bin`), `build-essential` | nothing |

The install needs network access for the release's locked `build` dependency group (cmake, ninja, nanobind, onnx
wheels) and uv's managed Python 3.11, unless they are cached. The system cmake / Python do not matter.

## Install a release (an update = the same with a newer release)

| # | Where | What | What moves |
|---|---|---|---|
| 1 | workstation | `bash deploy/release.sh --out out/releases` (from a git checkout: `git archive` of HEAD or `--ref`, a MANIFEST with the sha256 of every file, the tarball's own `.sha256`; no extended attributes / platform metadata, owner root) -> `out/releases/r1mc3-<id>.tar.gz` | nothing |
| 2 | workstation | `scp out/releases/r1mc3-<id>.tar.gz* unitree@<backpack>:/tmp/` | nothing |
| 3 | backpack | `cd /tmp && tar -xzf r1mc3-<id>.tar.gz r1mc3-<id>/deploy && sudo bash r1mc3-<id>/deploy/install.sh --tarball /tmp/r1mc3-<id>.tar.gz --dry-run` | nothing |
| 4 | backpack | the same without `--dry-run`: checks the sha256 and every file, installs under `/opt/r1mc3/releases/<id>` (renamed into place only when complete), builds there (the venv + `r1rt` + `r1guard` + the Python binding of the core, Release), links `bin/`, puts uv's Python in `/opt/r1mc3/python` (readable by the services), writes the site settings (below). Does NOT switch | CPU load only |
| 5 | backpack | the robot's core config from the robot profile: `/opt/r1mc3/releases/<id>/bin/python -m r1mc3.rtconfig --profile robot --out /etc/r1mc3/r1rt.cfg`, then `/opt/r1mc3/releases/<id>/bin/r1rt --config /etc/r1mc3/r1rt.cfg --check-config` -> `{"config_ok": true, ...}` (keep the old file as `r1rt.cfg.<date>` first) | nothing |
| 6 | backpack | `sudo bash /opt/r1mc3/releases/<id>/deploy/activate.sh --id <id>` - refused while r1mc3 owns the motors; installs the units, switches `current`, `systemctl daemon-reload`; restarts nothing | nothing |
| 7 | backpack | `bash /opt/r1mc3/current/deploy/status.sh` (releases, current / previous, the units vs the current release, the config, the motor check); the first time also `systemd-analyze verify` on the units | nothing |
| 8 | backpack | start in MONITOR: `sudo /opt/r1mc3/current/deploy/bin/r1mc3-svc start` (or `start --stand` for a session on the stand: [`STAND_LOCK.md`](STAND_LOCK.md)) - r1rt + r1guard + r1log + r1d + the inhibitor; **publishes nothing** until a takeover / on-stand request | nothing |

`r1mc3-svc enable` (start at boot) is your decision; the default is not enabled.

### What the core's config holds

`python -m r1mc3.rtconfig --profile robot` writes the rates, the joint limits (position / effort x 0.6 / velocity),
the gain bounds, the stand / hold / arm gains, every safety threshold (the state ages, the tilt stops, the policy age,
the takeover gates, the guard's periods), the executor's stop time, the freeze ramp, the takeover / ON_STAND blend, the
damping kd, the policy interface and the converted weights - from the robot profile `config/robot_profile.r1_dex3.yaml`
(every value with its source, status and date). Every key gets a comment line in the file and an entry in
`/etc/r1mc3/r1rt.cfg.provenance.json`. The robot's file differs from the simulator's (the tests') only in the velocity
limits (the URDF's), the gain bounds and the requirement that the real-time grants are present for a takeover. Where
the profile is LOOSER than the simulator's tested value, the tested value is kept and listed under
`kept_simulator_value_where_looser`: loosening a limit is a deliberate, reviewed change. Run step 5 again whenever the
profile changes, from the release that will be active.

### The site settings the install writes (`deploy/rt/site.sh`; `site.sh show` prints them)

| Setting | Why | Option |
|---|---|---|
| `/etc/sysctl.d/60-r1mc3-rt.conf`: `kernel.sched_rt_runtime_us = -1` | without it the units' SCHED_FIFO was refused on our backpack (docker / containerd make systemd give every service a CPU cgroup with a real-time budget of 0). Our real-time threads are pinned to CPUs 5-7, so a runaway cannot freeze CPUs 0-4 | `--rt-budget sysctl` (default); `slice` = keep the throttle on and give the RT units their own budgets (built, tested in a fake root only); `none` |
| the same file: `net.core.rmem_max = 8388608` | CycloneDDS's 4 MB receive buffer (the shared config) fails against the kernel's default | always |
| the same file: `kernel.printk = 1 4 1 7` | only emergency messages reach the serial consoles (a 3-line kernel report at 115200 baud stalled the guard's cycle 24.5 ms on ours); dmesg / the journal keep everything | `--console-printk keep` = no change |
| `/etc/r1mc3/cyclonedds.xml` (only if missing) | CycloneDDS bound to the backpack's interface on the robot's network ([`NETWORK.md`](NETWORK.md)) - **edit the interface name for your backpack** | - |
| `/etc/r1mc3/cyclonedds_rt.xml` (every install) | the real-time units' bounded 256 kB receive buffer ([`NETWORK.md`](NETWORK.md)) | `--rt-dds-rcvbuf bounded` (default) / `shared` |
| the CPU partition unit `r1mc3-partition.service` | runs `partition.sh apply --move-tasks` once per boot before the core ([`RT_SETUP.md`](RT_SETUP.md)) | `sudo bash deploy/rt/partition.sh undo` |

## Rollback

| # | Where | What | What moves |
|---|---|---|---|
| R1 | - | the session's standard end first if r1mc3 owns the motors (hold -> `r1 damp` -> stand -> `r1 shutdown`) | the robot is damped by the operator |
| R2 | backpack | `sudo /opt/r1mc3/current/deploy/bin/r1mc3-svc stop` (refused while r1mc3 owns the motors) | nothing |
| R3 | backpack | `sudo bash /opt/r1mc3/current/deploy/rollback.sh` (= the release that was current before; `--to <id>` for another) | nothing |
| R4 | backpack | `sudo /opt/r1mc3/current/deploy/bin/r1mc3-svc start` (MONITOR) | nothing |

## Stopping never damps

systemd cannot refuse a stop from `ExecStop`, so the rule "stopping the software never makes the robot limp" is built
from three parts: `RefuseManualStop=yes` (systemd refuses `systemctl stop / restart` of r1rt and r1guard),
`SendSIGKILL=no` + `TimeoutStopSec=infinity` (a stop that does happen - e.g. a forced shutdown - never escalates to
SIGKILL), and r1rt itself ignoring SIGTERM in a publishing mode. The sanctioned path is `r1 shutdown` (accepted only in
DAMP / MONITOR) or `r1mc3-svc stop` after the motor check (`deploy/bin/r1mc3_motorcheck.py`: exit 0 = nothing of ours
owns the motors, or r1rt is in MONITOR / DAMP; 3 = refused; 2 = cannot tell, refused). The inhibitor
(`r1mc3_inhibit.py`) holds a systemd-inhibit lock while r1mc3 owns the motors, so `systemctl reboot` / `poweroff` are
refused then.

If a test pause ever outlives r1d (the board-hold / guard-takeover tests pause r1rt from a child of r1d with a resume
backstop; `r1 status` unreachable, or r1rt's heartbeat old while no pause runs): `sudo kill -CONT $(pidof r1rt)` - r1rt
resumes (silent if the guard holds the token; `r1 reclaim` then). `r1d.service` has `KillMode=process` so that an r1d
crash stops only r1d's main process and the backstop child still sends its resume.

## The observe-only guard (for the board-hold test only)

`r1guard.service` refuses a manual stop, so "the guard off" for the board-hold test is a swap to an OBSERVE-ONLY guard
(heartbeat + shadow policy, never takes the token: `r1guard --observe-only`, `r1guard-observe.service`, installed with
the other units, never enabled at boot). The two units `Conflicts=` each other: starting one stops the other.

| # | Where | What |
|---|---|---|
| G1 | backpack (explicit OK, robot on the stand) | `r1 status` -> the guard on standby (not holding the motors); `sudo systemctl start r1guard-observe`; `systemctl is-active r1guard r1guard-observe` = `inactive` / `active`; `r1 status` -> the guard observe-only (NOT armed) |
| G2 | backpack, right after the test | `sudo systemctl start r1guard` -> the other way round; `r1 status` -> the guard armed. **No moving step while it is not armed** |

## Where a session's files live on the backpack

| Folder | Written by |
|---|---|
| `/var/lib/r1mc3/logs` | r1log (MCAP: the core's telemetry every tick, the hands, the guard, the flight recorder) |
| `/var/lib/r1mc3/records` | r1d (`r1 record ...` manifests and data, r1d's request journal) |
| `/var/lib/r1mc3/sessions` | `r1 shutdown [--label L]`: a NEW folder `<L>_<UTC>/` each time with every file of the last 24 h not already saved, sha256 on both sides, `manifest.json` |
| `/etc/r1mc3/r1rt.cfg` | the config step above |

All owned by the units' user; `install.sh` creates them and the units' `StateDirectory=` re-creates them at start.

## Not proven / limits

- The units' settings are checked by tests for their keys, not by systemd itself (`systemd-analyze verify` on the
  backpack the first time).
- The `slice` RT-budget method and the e2fsck procedures are tested in a fake root only.
- Other JetPack versions and other Linux images are untested; the C++ builds with GCC 9 (Ubuntu 20.04) and current
  compilers, and the Linux build is checked in an Ubuntu 20.04 arm64 container by our CI.
