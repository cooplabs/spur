"""deploy/eth10/{install,uninstall}.sh in a FAKE root (DESIGN.md D18, U8): the v3-aware guard installed ALONGSIDE v2's,
v2's files never written (byte for byte), nothing started without --replace-v2, the switch-over and its undo printed
as the systemctl commands they would run (a fake root runs no system command), --dry-run changes nothing."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1].parent
ETH10 = REPO / "deploy" / "eth10"
ASSETS = REPO / "tests" / "assets" / "eth10"


def _run(script: str, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "LC_ALL": "C"}
    return subprocess.run(
        ["bash", str(ETH10 / script), "--root", str(root), *args], capture_output=True, text=True, env=env, check=False
    )


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    r = tmp_path / "root"
    (r / "usr" / "local" / "bin").mkdir(parents=True)
    (r / "etc" / "systemd" / "system").mkdir(parents=True)
    shutil.copy(ASSETS / "v2_eth10_guard.sh", r / "usr" / "local" / "bin" / "r1-eth10-guard.sh")
    shutil.copy(ASSETS / "v2_r1-eth10-guard.service", r / "etc" / "systemd" / "system" / "r1-eth10-guard.service")
    return r


def _v2_bytes(root: Path) -> tuple[bytes, bytes]:
    return (
        (root / "usr/local/bin/r1-eth10-guard.sh").read_bytes(),
        (root / "etc/systemd/system/r1-eth10-guard.service").read_bytes(),
    )


def test_install_alongside_v2_without_starting_anything(root: Path) -> None:
    before = _v2_bytes(root)
    out = _run("install.sh", root)
    assert out.returncode == 0, out.stderr
    s = root / "usr/local/bin/r1mc3-eth10-guard.sh"
    u = root / "etc/systemd/system/r1mc3-eth10-guard.service"
    assert s.read_bytes() == (ETH10 / "r1mc3-eth10-guard.sh").read_bytes() and os.access(s, os.X_OK)
    assert u.read_bytes() == (ETH10 / "r1mc3-eth10-guard.service").read_bytes()
    assert _v2_bytes(root) == before  # v2's files untouched
    rec = (root / "etc/r1mc3/eth10-guard.install").read_text()
    assert "v2_unit_enabled_before=unknown (fake root)" in rec and "replace_v2=0" in rec
    assert "would run: systemctl daemon-reload" in out.stderr
    assert "disable" not in out.stderr.replace("sudo systemctl disable --now r1-eth10-guard.service", "")
    assert "would run: systemctl enable" not in out.stderr  # nothing started without --replace-v2
    assert "v2's files unchanged" in out.stderr


def test_replace_v2_switches_over_and_uninstall_restores(root: Path) -> None:
    before = _v2_bytes(root)
    out = _run("install.sh", root, "--replace-v2")
    assert out.returncode == 0, out.stderr
    assert "would run: systemctl disable --now r1-eth10-guard.service" in out.stderr
    assert "would run: systemctl enable --now r1mc3-eth10-guard.service" in out.stderr
    assert out.stderr.index("disable --now r1-eth10-guard") < out.stderr.index("enable --now r1mc3-eth10-guard")
    un = _run("uninstall.sh", root, "--restore-v2")
    assert un.returncode == 0, un.stderr
    assert "would run: systemctl disable --now r1mc3-eth10-guard.service" in un.stderr
    assert "would run: systemctl enable --now r1-eth10-guard.service" in un.stderr
    assert not (root / "usr/local/bin/r1mc3-eth10-guard.sh").exists()
    assert not (root / "etc/systemd/system/r1mc3-eth10-guard.service").exists()
    assert not (root / "etc/r1mc3/eth10-guard.install").exists()
    assert _v2_bytes(root) == before


def test_uninstall_restores_v2_when_it_was_recorded_enabled(root: Path) -> None:
    assert _run("install.sh", root).returncode == 0
    rec = root / "etc/r1mc3/eth10-guard.install"
    rec.write_text(
        rec.read_text().replace("v2_unit_enabled_before=unknown (fake root)", "v2_unit_enabled_before=enabled")
    )
    un = _run("uninstall.sh", root)
    assert "would run: systemctl enable --now r1-eth10-guard.service" in un.stderr
    root2 = root.parent / "root2"
    shutil.copytree(root, root2)
    (root2 / "etc/r1mc3").mkdir(parents=True, exist_ok=True)
    assert _run("install.sh", root2).returncode == 0
    un2 = _run("uninstall.sh", root2)  # recorded unknown, no flag: v2 left as it is
    assert "systemctl enable --now r1-eth10-guard.service" not in un2.stderr


def test_dry_run_changes_nothing(root: Path) -> None:
    out = _run("install.sh", root, "--dry-run", "--replace-v2")
    assert out.returncode == 0, out.stderr
    assert not (root / "usr/local/bin/r1mc3-eth10-guard.sh").exists()
    assert not (root / "etc/r1mc3").exists()
    assert "dry-run: install" in out.stderr


def test_unknown_option_refused(root: Path) -> None:
    out = _run("install.sh", root, "--force")
    assert out.returncode == 2 and "unknown option" in out.stderr
