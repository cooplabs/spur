# Two "locks": Unitree's stand lock and r1mc3's stand-session lock

Two different things are called a "lock" - do not mix them up:

| Name | What it is | Who owns it |
|---|---|---|
| **Unitree's stand lock** | a mode of Unitree's own controller, selected with the remote / app: every motor enabled, stiff, held at a **fixed preset pose**; not balancing | Unitree (before r1mc3 takes the motors) |
| **r1mc3's stand-session lock** | `r1rt --stand-session` (unit `r1rt-stand.service`, started by `r1mc3-svc start --stand`): the core refuses every entry into BALANCE for its whole lifetime | r1mc3 |

## Unitree's stand lock is a FIXED preset pose (measured on our R1)

It is easy to assume the stand lock holds the joints "stiff where they are". **It does not: it commands a fixed
preset pose**, the same one every time, and **the robot MOVES INTO that pose when the stand lock is selected** - on a
stand the legs and arms swing to it from wherever they hung. Say so before anyone selects it.

| Group | Preset (rad; SDK order) | kp / kd |
|---|---|---|
| left leg | hip pitch -0.305, hip roll +0.04, hip yaw -0.14, knee 0.41, ankle pitch -0.137, ankle roll -0.06 | 200 / 3 |
| right leg | hip pitch -0.305, hip roll -0.04, hip yaw +0.14, knee 0.41, ankle pitch -0.137, ankle roll +0.06 | 200 / 3 |
| waist (slots 12, 13) | roll 0, yaw 0 | 300 / 5 |
| left arm | shoulder pitch 0.2, shoulder roll +0.2, shoulder yaw 0, elbow 0.8, wrist roll 0 | 100 / 2 (wrist 50 / 2) |
| right arm | shoulder pitch 0.2, shoulder roll -0.2, shoulder yaw 0, elbow 0.8, wrist roll 0 | 100 / 2 (wrist 50 / 2) |
| head (slots 29, 30) | pitch 0, yaw 0 | pitch 50 / 2, yaw 10 / 0.1 |
| unused slots 14, 20, 21, 27, 28, 31-34 | - | 0 / 0 |

What the preset means on the robot (by the verified directions, [`JOINT_DIRECTIONS.md`](JOINT_DIRECTIONS.md)): the
hips slightly FORWARD (pitch -0.305), the knees bent 0.41, the toes slightly UP (ankle pitch -0.137); the arms slightly
BACK (shoulder pitch 0.2) and out from the body (shoulder roll toward each side), the elbows at 0.8 (roughly 60 deg
flexed - a SMALLER elbow value is MORE flexed).

Other facts of the mode on our robot: Unitree **publishes `rt/lowcmd` at ~643 /s** in it, `mode` 1 (enabled) on all
35 slots. Hanging, every joint settled within 0.017 rad of the preset (hip pitch -0.290 vs -0.305: gravity at kp 200;
elbows 0.817 vs 0.8); the pelvis tilt 5.4 deg. In ZERO TORQUE (the power-on mode) Unitree's controller ALSO publishes
`rt/lowcmd` (~648 /s, kp 0 / kd 0 on every slot).

The gains of the stand lock are the stand gains r1mc3's robot profile uses for its own stand hold, so ON_STAND's entry
from the stand lock has no gain step: `r1 on-stand` released Unitree's controller in 0.19-0.22 s on our robot and then
held the MEASURED pose (not the preset), the arms at r1mc3's softer arm gains. Settle: 0.014-0.016 rad on the hips
(the new target is the measured pose, so gravity at kp 200 moves the leg another ~0.015 rad), arms <= 0.007 rad.

The joint directions at this pose: `python -m r1mc3.hw1.dircheck --pose stand_lock` (the same 26 answers).

The simulator's hanging stance commands this preset when its emulated built-in is in `stand_lock`
([`../sim/DDS_BRIDGE.md`](../sim/DDS_BRIDGE.md)).

## r1mc3's stand-session lock

For every session with the robot on its stand, run the core as **`r1rt-stand.service` = `r1rt ... --stand-session`**:
the core refuses EVERY entry into BALANCE (takeover, enter-balance, ...) for the process's whole lifetime; nothing
clears it at run time; `r1 status` shows it ("stand session: balance REFUSED"). A hanging robot must never run the
balance policy: the policy would push against the stand.

`r1rt-stand.service` and `r1rt.service` `Conflicts=` each other (systemd never runs both), share the limits, the
refused manual stop, the never-kill stop settings and the motor-ownership check, and no other unit `Wants=` a core
(starting one would otherwise pull in the UNLOCKED core). `r1mc3-svc start` refuses while either core runs.

| # | Where | Words | What moves | How to undo |
|---|---|---|---|---|
| SL1 | backpack | `sudo /opt/r1mc3/current/deploy/bin/r1mc3-svc start --stand` -> r1rt-stand, r1guard, r1log, r1d, the inhibitor; `systemctl is-active r1rt-stand r1rt` -> `active` / `inactive` | nothing (MONITOR) | `sudo .../r1mc3-svc stop` (allowed in MONITOR / DAMP) |
| SL2 | workstation | `r1 status` -> the stand lock shown; MONITOR; the guard armed | nothing | - |

## Lifting it (before the floor)

Only with the robot on its stand or held. **Never `sudo systemctl start r1rt` alone:** r1rt creates its shared-memory
segment and unlinks it when it exits, while r1guard / r1d / r1log attach once at their start - after a lone core swap
the old r1guard keeps running on the dead segment and the new core has no guard heartbeat. So the WHOLE stack
restarts:

| # | Where | Words | What moves |
|---|---|---|---|
| L1 | workstation | `r1 status`: MONITOR -> L3; r1mc3 owns the motors (ON_STAND / GUARDED) -> L2 | nothing |
| L2 | - + workstation | the standard end: hold the robot -> `r1 damp` -> on the stand -> `r1 shutdown` (the log save) | LIMP at `r1 damp` |
| L3 | backpack | `sudo /opt/r1mc3/current/deploy/bin/r1mc3-svc stop` (refused unless MONITOR / DAMP), `r1mc3-svc status` -> all inactive | nothing |
| L4 | - | after L2 only: Unitree's controller back (remote / app; if it does not come back after the release: power off / on) - the takeover needs its fresh command | the robot under Unitree |
| L5 | backpack | `sudo /opt/r1mc3/current/deploy/bin/r1mc3-svc start` (no `--stand`) -> `systemctl is-active r1rt r1rt-stand` = `active` / `inactive` | nothing (MONITOR) |
| L6 | workstation | `r1 status`: MONITOR, the guard armed, no stand lock | nothing |

Tested: `tests/unit/test_deploy_units.py`, `tests/unit/test_deploy_scripts.py` (the units' keys, the conflicts both
ways, `r1mc3-svc start --stand` starts r1rt-stand and never r1rt). Not tested by the suite: systemd itself - run
`systemd-analyze verify` on the backpack once.
