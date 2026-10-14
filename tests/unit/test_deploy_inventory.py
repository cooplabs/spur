"""`deploy/inventory.sh` (HW1 runbook step P1a): the backpack inventory is READ-ONLY; it reports what v3's install
needs.

Run in a fake root (R1I_ROOT) with stub tools on PATH, so the backpack's state is set per test: a ready backpack
(exit 0), missing tools (ACTION rows, exit 3), a running v2 controller / no internet / no eth10 (BLOCKER rows, exit 4).
Every mutating command (sudo, apt-get, ethtool, systemctl, rm, ...) is a stub that leaves a marker if it is ever
called: none may be (the script prints the fix; the operator runs it with an OK). Written after the user's request
(10 Oct 2026, after session 6): "build this into the runbook ... with these checks"."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy" / "inventory.sh"
FORBIDDEN = ("sudo", "apt-get", "apt", "ethtool", "systemctl", "rm", "reboot", "nvpmodel", "jetson_clocks", "kill")


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _backpack(tmp: Path, *, uv: bool = True, rt_tests: bool = True, v2_running: bool = False, online: bool = True,
              eth10: bool = True, cxx_built: bool = True) -> tuple[Path, dict[str, str]]:  # fmt: skip
    root = tmp / "root"
    for d in ("etc", "opt", "var"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "etc/nv_tegra_release").write_text(
        "# R35 (release), REVISION: 3.1, GCID: 32827747, BOARD: t186ref, EABI: aarch64, "
        "DATE: Sun Mar 19 15:19:21 UTC 2023\n"
    )
    if cxx_built:
        (root / "opt/r1mc3/cyclonedds-0.10.2/lib/cmake/CycloneDDS-CXX").mkdir(parents=True)
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    marker = tmp / "MUTATED"
    for name in FORBIDDEN:
        _stub(bin_dir, name, f'echo "{name} $*" >> "{marker}"; exit 0')
    _stub(bin_dir, "python3", 'echo "Python 3.8.10"')
    for t in ("gcc", "g++", "make", "git", "tar"):
        _stub(bin_dir, t, f'echo "{t} (stub) 9.4.0"')
    _stub(bin_dir, "curl", "exit 0" if online else "exit 6")
    _stub(bin_dir, "ip", 'echo "eth10            UP             00:00:5e:00:53:01"' if eth10 else "exit 1")
    _stub(bin_dir, "pgrep", 'echo "4242 python3 -m r1mc2 serve"; exit 0' if v2_running else "exit 1")
    if uv:
        _stub(bin_dir, "uv", 'case "$1" in --version) echo "uv 0.4.30";; python) exit 1;; esac')
    if rt_tests:
        _stub(bin_dir, "cyclictest", "exit 0")
        _stub(bin_dir, "stress-ng", "exit 0")
    # the system utilities the script reads with (head, cut, awk, df, grep, uname, getconf, ...) stay on PATH
    env = dict(os.environ, R1I_ROOT=str(root), PATH=f"{bin_dir}:/usr/bin:/bin")
    return marker, env


def _run(env: dict[str, str]) -> tuple[int, str]:
    out = subprocess.run(["/bin/bash", str(SCRIPT)], capture_output=True, text=True, env=env, timeout=60)
    return out.returncode, out.stdout


def _row(stdout: str, check: str) -> list[str]:
    for line in stdout.splitlines():
        cols = [c.strip() for c in line.split("|")]
        if cols and cols[0] == check:
            return cols
    raise AssertionError(f"no row {check!r} in:\n{stdout}")


def test_a_ready_backpack_reports_ready_and_changes_nothing(tmp_path: Path) -> None:
    marker, env = _backpack(tmp_path)
    code, out = _run(env)
    assert code == 0, out
    assert _row(out, "L4T release")[2] == "OK"
    assert _row(out, "uv")[2] == "OK" and _row(out, "v3 CycloneDDS-CXX 0.10.2")[2] == "OK"
    assert "ready for P2" in out
    assert not marker.exists(), marker.read_text()


def test_missing_tools_are_actions_with_their_fix(tmp_path: Path) -> None:
    marker, env = _backpack(tmp_path, uv=False, rt_tests=False, cxx_built=False)
    code, out = _run(env)
    assert code == 3, out
    uv = _row(out, "uv")
    assert uv[1] == "MISSING" and uv[2] == "ACTION" and "P3" in uv[3]
    assert _row(out, "cyclictest")[2] == "ACTION" and "rt-tests" in _row(out, "cyclictest")[3]
    assert _row(out, "v3 CycloneDDS-CXX 0.10.2")[2] == "ACTION"
    assert "do the ACTION rows" in out
    assert not marker.exists(), marker.read_text()


def test_blockers_stop_the_session(tmp_path: Path) -> None:
    marker, env = _backpack(tmp_path, v2_running=True, online=False, eth10=False)
    code, out = _run(env)
    assert code == 4, out
    assert _row(out, "v2 controller running")[2] == "BLOCKER"
    assert _row(out, "internet https://pypi.org")[2] == "BLOCKER"
    assert _row(out, "eth10 (motor link)")[2] == "BLOCKER"
    assert "STOP" in out
    assert not marker.exists(), marker.read_text()


def test_the_script_runs_from_stdin_as_over_ssh(tmp_path: Path) -> None:
    """The runbook runs it as `ssh unitree@<robot-ip> 'bash -s' < deploy/inventory.sh` (no file on the robot)."""
    marker, env = _backpack(tmp_path)
    out = subprocess.run(["/bin/bash", "-s"], stdin=SCRIPT.open(), capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ready for P2" in out.stdout
    assert not marker.exists()
