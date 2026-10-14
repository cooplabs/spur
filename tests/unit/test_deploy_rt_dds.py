"""Session 9 (#29's open race; the user's decision 11 Oct 2026 "(e) smaller buffer"): the real-time units' CycloneDDS
overlay - r1rt / r1guard read the shared /etc/r1mc3/cyclonedds.xml AND /etc/r1mc3/cyclonedds_rt.xml (a 256 kB socket
receive buffer: after a stop they must drain the lowstate backlog of the stop before a live sample, and the shared 4 MB
holds ~1.6 s+ of it). install.sh writes the overlay from `--rt-dds-rcvbuf bounded` (default) | `shared` (no override).
Fake root only (tests/unit/test_deploy_scripts.py's fixtures). Fails on 6449aee: no overlay, no option, the units read
the shared file alone. The overlay's effect on the drain is measured with processes in
tests/integration/test_v3_resume_race.py (`race_run(uri=...)`)."""

from __future__ import annotations

import re
from pathlib import Path

from tests.unit.test_deploy_scripts import DEPLOY, _release, _run, root, source_repo  # noqa: F401  (fixtures)

RT_UNITS = ("r1rt.service", "r1rt-stand.service", "r1guard.service", "r1guard-observe.service")
OTHER_UNITS = ("r1d.service", "r1log.service")
BOTH = "file:///etc/r1mc3/cyclonedds.xml,file:///etc/r1mc3/cyclonedds_rt.xml"


def _uri(unit: str) -> str | None:
    m = re.search(r"^Environment=CYCLONEDDS_URI=(\S+)$", (DEPLOY / "units" / unit).read_text(), re.M)
    return m.group(1) if m else None


def test_the_rt_units_read_the_shared_file_then_the_rt_overlay_and_nothing_else_changes() -> None:
    for u in RT_UNITS:
        assert _uri(u) == BOTH, u
    for u in OTHER_UNITS:
        assert _uri(u) in (None, "file:///etc/r1mc3/cyclonedds.xml"), u  # r1d / r1log keep the shared 4 MB


def test_the_bounded_overlay_asks_for_256_kb_and_the_shared_file_keeps_4_mb() -> None:
    rt = (DEPLOY / "config" / "cyclonedds_rt.xml").read_text()
    assert re.search(r'<SocketReceiveBufferSize min="256kB" max="256kB"/>', rt)
    assert "NetworkInterface" not in rt  # the interface stays the site file's (eth10)
    assert '<SocketReceiveBufferSize min="4MB"/>' in (DEPLOY / "config" / "cyclonedds.xml").read_text()
    assert "SocketReceiveBufferSize" not in (DEPLOY / "config" / "cyclonedds_rt_shared.xml").read_text()
    # Linux doubles a SO_RCVBUF request: 512 kB effective, far below the site's rmem_max 8 MB (deploy/rt/site.sh)
    assert "net.core.rmem_max = 8388608" in (DEPLOY / "rt" / "site.sh").read_text()


def test_install_writes_the_overlay_from_the_option(source_repo: Path, root: Path, tmp_path: Path) -> None:  # noqa: F811
    tb = _release(source_repo, "HEAD", tmp_path / "out")
    etc = root / "etc" / "r1mc3"
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb, "--no-build", "--no-site"])
    assert r.returncode == 0, r.stderr
    ov = etc / "cyclonedds_rt.xml"
    assert ov.read_text() == (DEPLOY / "config" / "cyclonedds_rt.xml").read_text()  # the default: bounded
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb, "--no-build", "--no-site",
              "--rt-dds-rcvbuf", "shared"])  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert ov.read_text() == (DEPLOY / "config" / "cyclonedds_rt_shared.xml").read_text()
    assert len(list(etc.glob("cyclonedds_rt.xml.bak-*"))) == 1  # the replaced overlay kept
    assert (etc / "cyclonedds.xml").read_text() == (DEPLOY / "config" / "cyclonedds.xml").read_text()  # untouched
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb, "--no-build", "--no-site",
              "--rt-dds-rcvbuf", "4MB"])  # fmt: skip
    assert r.returncode != 0 and "--rt-dds-rcvbuf wants bounded or shared" in r.stderr
