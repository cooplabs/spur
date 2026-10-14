# Security and safety policy

r1mc3 commands a humanoid robot. We treat two kinds of problems as security issues:

1. **Classic vulnerabilities**: anything that lets an unauthorised party send commands, read data, or crash a process
   (the API, the shared-memory channels, the command line, the deployment scripts).
2. **Safety defects**: anything that could make a robot move unexpectedly, ignore a stop, go limp without an
   operator's command, let two processes write to the motors, or defeat the real-time core's or the guard's checks.

## Reporting

Please report privately through GitHub's **private vulnerability reporting** on this repository (the **Security** tab
-> **Report a vulnerability**). Do not open a public issue for an unfixed problem. Include what you saw, how to
reproduce it (in the simulator where possible), the version or commit, and - for a safety defect - what the robot did
or would do. We aim to acknowledge reports within a week and will credit reporters who want it.

## Testing safely

- Reproduce problems in the simulator (`sim/r1sim`, its fault injector), never first on a robot.
- Never test on a robot without the operator procedures ([`docs/SAFETY.md`](docs/SAFETY.md),
  [`docs/OPERATOR.md`](docs/OPERATOR.md)): one step at a time, each saying what moves and how to stop it, with a
  person holding the robot or the robot on its stand, and the hardware E-stop in reach.
- No automatic damping exists by design; if you find a path that damps the robot without an operator's command, that
  is a safety defect - report it.

## Exposure

`r1d` listens on the backpack's localhost only (port 8780) and has no authentication of its own: reach it through an
SSH tunnel, never by exposing the port on a network. The real-time processes take no network input other than the
robot's DDS topics on the motor link.

## Supported versions

Only the latest release and the main branch are supported.
