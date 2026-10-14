# deploy/

Installation on the robot's backpack computer (Jetson Orin NX, L4T R35.3.1 / Ubuntu 20.04), DESIGN.md §11 and §2.1.
**Written and tested in session 5 against a fake root on the Mac; nothing has run on the backpack yet** - the first
real install is a step of the stand session (HW1), one step at a time: `docs/hardware/DEPLOY.md`.

| Path | What |
|---|---|
| `release.sh` | Mac: a release tarball from a git ref (`git archive`, a MANIFEST with the sha256 of every file, the tarball's own `.sha256`) |
| `install.sh` | backpack: install a release side by side under `/opt/r1mc3/releases/<version>-<git7>` (checked against its MANIFEST, renamed into place only when complete), build it there (venv + the C++ core and guard), `bin/` links, the site config `/etc/r1mc3` only where missing; `--activate` to switch |
| `activate.sh` | switch `current` to an installed release (atomic symlink rename, `history.log`), install its systemd units; **refused while r1rt / r1guard own the motors**; restarts nothing |
| `rollback.sh` | switch back to the previous release (from `history.log`, or `--to`); the same refusal |
| `status.sh` | read-only: releases, current / previous, units vs the current release, config, the motor check |
| `units/` | `r1rt`, `r1guard` (FIFO 90 / 88, CPUs 6-7 / 5, `LimitRTPRIO`, `LimitMEMLOCK=infinity`, `RefuseManualStop=yes`, `SendSIGKILL=no`, `TimeoutStopSec=infinity`, `Restart=no`, no CPU accounting), `r1d` (CPUs 0-4), `r1log` (nice 10), `r1mc3-inhibit` |
| `bin/r1mc3_motorcheck.py` | the ONE rule (read-only): exit 0 = nothing of ours owns the motors, or r1rt is in MONITOR / DAMP; 3 = refused; 2 = cannot tell (refused) |
| `bin/r1mc3-svc` | `start` (MONITOR: publishes nothing) / `stop` (refused while we own the motors; never damps) / `status` / `enable` / `disable` |
| `bin/r1mc3_inhibit.py` | holds a systemd-inhibit block lock while we own the motors: `systemctl reboot` / `poweroff` are refused then |
| `rt/partition.sh` | RT stage 1, runtime part (`show` / `apply` / `undo`; every value backed up first): IRQs to CPUs 0-4, the motor NIC's IRQ to 6, the workqueue mask, a systemd CPUAffinity drop-in, nvpmodel MAXN + jetson_clocks |
| `config/cyclonedds.xml` | CycloneDDS bound to eth10 (the motor NIC), installed to `/etc/r1mc3/` only if missing |

Every script takes `--root DIR` (a fake root: nothing outside it is touched and no system command runs - they are
printed as "would run") and `--dry-run`. Tests: `tests/unit/test_deploy_*.py`.

**Stopping never damps** (DESIGN §2.1). systemd cannot refuse a stop from `ExecStop` (it runs, then kills anyway), so
the refusal is built from three parts: `RefuseManualStop=yes` (systemd refuses `systemctl stop / restart` of r1rt and
r1guard), `SendSIGKILL=no` + `TimeoutStopSec=infinity` (a stop that does happen - e.g. a forced shutdown - never
escalates to SIGKILL), and r1rt itself ignoring SIGTERM in a publishing mode (the process wrapper's rule). The
sanctioned path is `r1 shutdown` (accepted only in DAMP / MONITOR) or `r1mc3-svc stop` after the motor check. A reboot
request is blocked by the inhibitor while we own the motors. v2 (`/home/unitree/r1-mc2`) is never written.
