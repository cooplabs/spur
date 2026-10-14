"""deploy/ (DESIGN.md §11.2, §2.1): release -> install -> activate -> update -> rollback in a FAKE root, the refusal
while r1rt / r1guard own the motors, the site config never overwritten, v2's folder untouched, the RT stage-1
partition script (apply / undo) - all against a fake root under the test's tmp dir; nothing touches this machine's
system or any robot (the scripts run no system command when --root is not "/")."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy"
SCRIPTS = [*sorted(DEPLOY.glob("*.sh")), DEPLOY / "bin" / "r1mc3-svc", DEPLOY / "rt" / "partition.sh"]
REFUSED = 3


def _run(args: list[str | Path], **kw: object) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "LC_ALL": "C"}
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, env=env, check=False, **kw)  # type: ignore[call-overload,no-any-return]


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return r.stdout.strip()


@pytest.fixture()
def source_repo(tmp_path: Path) -> Path:
    """A tiny repository with two commits (two releases), each with this worktree's deploy/ (units, config)."""
    repo = tmp_path / "src"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "test")
    shutil.copytree(DEPLOY, repo / "deploy")
    (repo / "pyproject.toml").write_text('[project]\nname = "r1mc3"\nversion = "0.1.0.dev0"\n')
    (repo / "VERSION_NOTE").write_text("one\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "one")
    (repo / "VERSION_NOTE").write_text("two\n")
    _git(repo, "commit", "-q", "-am", "two")
    return repo


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    r = tmp_path / "root"
    v2 = r / "home" / "unitree" / "r1-mc2"
    v2.mkdir(parents=True)
    (v2 / "server.py").write_text("# v2 - must never change\n")
    return r


def _release(repo: Path, ref: str, out: Path) -> Path:
    r = _run(["bash", DEPLOY / "release.sh", "--repo", repo, "--ref", ref, "--out", out])
    assert r.returncode == 0, r.stderr
    return Path(r.stdout.strip().splitlines()[-1])


def _tree_hash(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(d)).encode() + f.read_bytes())
    return h.hexdigest()


def _set_motors(root: Path, state: dict[str, object] | None) -> None:
    """The fake segment + state the motor check reads under a fake root (deploy/bin/r1mc3_motorcheck.py)."""
    shm = root / "dev" / "shm" / "r1mc3.rt"
    st = root / "run" / "r1mc3" / "motorcheck_state.json"
    if state is None:
        shm.unlink(missing_ok=True)
        st.unlink(missing_ok=True)
        return
    shm.parent.mkdir(parents=True, exist_ok=True)
    shm.write_bytes(b"\0")
    st.parent.mkdir(parents=True, exist_ok=True)
    st.write_text(json.dumps(state))


def test_every_script_parses() -> None:
    for s in SCRIPTS:
        r = _run(["bash", "-n", s])
        assert r.returncode == 0, f"{s}: {r.stderr}"
    if shutil.which("shellcheck"):
        r = _run(["shellcheck", "-S", "error", *SCRIPTS])
        assert r.returncode == 0, r.stdout


def test_release_tarball_has_a_manifest_and_a_checksum(source_repo: Path, tmp_path: Path) -> None:
    tb = _release(source_repo, "HEAD", tmp_path / "out")
    sha = (tb.parent / (tb.name + ".sha256")).read_text().split()[0]
    assert hashlib.sha256(tb.read_bytes()).hexdigest() == sha
    with tarfile.open(tb) as t:
        names = t.getnames()
        top = names[0].split("/")[0]
        man = t.extractfile(f"{top}/MANIFEST")
        assert man is not None
        text = man.read().decode()
    git7 = _git(source_repo, "rev-parse", "--short=7", "HEAD")
    assert top == f"r1mc3-0.1.0.dev0-{git7}"
    assert f"git={_git(source_repo, 'rev-parse', 'HEAD')}" in text
    assert "./deploy/units/r1rt.service" in text and "./VERSION_NOTE" in text


def test_install_activate_update_rollback_cycle(source_repo: Path, root: Path, tmp_path: Path) -> None:
    v2_before = _tree_hash(root / "home")
    tb1 = _release(source_repo, "HEAD~1", tmp_path / "out")
    tb2 = _release(source_repo, "HEAD", tmp_path / "out")
    id1, id2 = (t.name.removeprefix("r1mc3-").removesuffix(".tar.gz") for t in (tb1, tb2))
    prefix = root / "opt" / "r1mc3"

    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb1, "--no-build", "--activate"])
    assert r.returncode == 0, r.stderr
    assert (prefix / "current").is_symlink() and os.readlink(prefix / "current") == f"releases/{id1}"
    assert (prefix / "releases" / id1 / "VERSION_NOTE").read_text() == "one\n"
    unit = root / "etc" / "systemd" / "system" / "r1rt.service"
    assert unit.read_text() == (DEPLOY / "units" / "r1rt.service").read_text()
    dds = root / "etc" / "r1mc3" / "cyclonedds.xml"
    assert dds.exists()
    dds.write_text("<!-- the operator's own edit -->\n")
    assert (prefix / "releases" / id1 / "bin" / "r1rt").is_symlink()

    # update: a second release installed side by side, then activated; the operator's config is kept
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb2, "--no-build"])
    assert r.returncode == 0, r.stderr
    assert os.readlink(prefix / "current") == f"releases/{id1}"  # installing does not switch
    r = _run(["bash", DEPLOY / "activate.sh", "--root", root, "--id", id2])
    assert r.returncode == 0, r.stderr
    assert os.readlink(prefix / "current") == f"releases/{id2}"
    assert dds.read_text() == "<!-- the operator's own edit -->\n"
    assert (prefix / "releases" / id1).is_dir()  # side by side

    # rollback = switch back to the previous release
    r = _run(["bash", DEPLOY / "rollback.sh", "--root", root])
    assert r.returncode == 0, r.stderr
    assert os.readlink(prefix / "current") == f"releases/{id1}"
    hist = (prefix / "history.log").read_text().splitlines()
    assert [ln.split()[1:] for ln in hist] == [["<none>", "->", id1], [id1, "->", id2], [id2, "->", id1]]

    # status is read-only and says what is current; v2's folder never changed
    r = _run(["bash", DEPLOY / "status.sh", "--root", root])
    assert r.returncode == 0, r.stderr
    assert f"current:   {id1}" in r.stdout and "motors:" in r.stdout
    assert _tree_hash(root / "home") == v2_before


def test_dry_run_changes_nothing(source_repo: Path, root: Path, tmp_path: Path) -> None:
    tb = _release(source_repo, "HEAD", tmp_path / "out")
    before = _tree_hash(root)
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb, "--no-build", "--activate", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert _tree_hash(root) == before


def test_a_tampered_release_is_refused(source_repo: Path, root: Path, tmp_path: Path) -> None:
    tb = _release(source_repo, "HEAD", tmp_path / "out")
    bad = tmp_path / "bad.tar.gz"
    shutil.copy(tb, bad)
    (tmp_path / "bad.tar.gz.sha256").write_text("0" * 64 + "  bad.tar.gz\n")
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", bad, "--no-build"])
    assert r.returncode != 0 and "sha256 mismatch" in r.stderr
    # a file changed inside a re-packed tarball with a matching outer checksum: the MANIFEST catches it
    work = tmp_path / "repack"
    with tarfile.open(tb) as t:
        t.extractall(work, filter="data")
    top = next(work.iterdir())
    (top / "VERSION_NOTE").write_text("tampered\n")
    re = tmp_path / "re.tar.gz"
    with tarfile.open(re, "w:gz") as t:
        t.add(top, arcname=top.name)
    (tmp_path / "re.tar.gz.sha256").write_text(hashlib.sha256(re.read_bytes()).hexdigest() + "  re.tar.gz\n")
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", re, "--no-build"])
    assert r.returncode != 0 and "MANIFEST check failed" in r.stderr
    assert not (root / "opt" / "r1mc3" / "releases" / top.name.removeprefix("r1mc3-")).exists()


@pytest.mark.parametrize(
    ("state", "refused"),
    [
        (None, False),  # no segment: nothing of ours runs
        ({"token_owner": "none"}, False),
        ({"token_owner": "r1rt", "rt": {"mode": "MONITOR", "age_ms": 1}}, False),
        ({"token_owner": "r1rt", "rt": {"mode": "DAMP", "age_ms": 1}}, False),
        ({"token_owner": "r1rt", "rt": {"mode": "BALANCE", "age_ms": 1}}, True),
        ({"token_owner": "r1rt", "rt": {"mode": "ON_STAND", "age_ms": 1}}, True),
        ({"token_owner": "r1rt", "rt": {"mode": "TAKEOVER_BLEND", "age_ms": 1}}, True),
        ({"token_owner": "r1rt", "rt": {"mode": "STAND_FREEZE", "age_ms": 1}}, True),
        ({"token_owner": "r1rt", "rt": {"mode": "MONITOR", "age_ms": 500}}, True),  # stale heartbeat: stalled?
        ({"token_owner": "r1guard", "guard": {"mode": "GUARDED", "age_ms": 1}}, True),
        ({"token_owner": "r1rt", "rt": {"mode": "BALANCE", "age_ms": 9000}, "processes_alive": False}, False),
        ({}, False),  # a segment whose token was never taken
    ],
)
def test_switch_and_stop_refused_while_we_own_the_motors(
    source_repo: Path, root: Path, tmp_path: Path, state: dict[str, object] | None, refused: bool
) -> None:
    tb1 = _release(source_repo, "HEAD~1", tmp_path / "out")
    tb2 = _release(source_repo, "HEAD", tmp_path / "out")
    assert (
        _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb1, "--no-build", "--activate"]).returncode
        == 0
    )
    assert _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb2, "--no-build"]).returncode == 0
    id2 = tb2.name.removeprefix("r1mc3-").removesuffix(".tar.gz")
    cur = root / "opt" / "r1mc3" / "current"
    before = os.readlink(cur)
    _set_motors(root, state)
    r = _run(["bash", DEPLOY / "activate.sh", "--root", root, "--id", id2])
    s = _run(["bash", DEPLOY / "bin" / "r1mc3-svc", "stop", "--root", root])
    if refused:
        assert r.returncode == REFUSED and "REFUSED" in r.stderr, r.stderr
        assert os.readlink(cur) == before  # nothing switched
        assert s.returncode == REFUSED, s.stderr
        assert "systemctl" not in s.stderr  # refused before any stop was even printed
    else:
        assert r.returncode == 0, r.stderr
        assert os.readlink(cur) == f"releases/{id2}"
        assert s.returncode == 0, s.stderr
        assert "would run: systemctl kill --signal=SIGTERM r1rt.service" in s.stderr
    # no command a stop / switch path runs (or would run) ever damps
    cmds = [ln for ln in (r.stderr + s.stderr).splitlines() if ln.startswith(("would run:", "dry-run:"))]
    assert not [c for c in cmds if "damp" in c.lower()], cmds


def test_a_segment_that_cannot_be_read_counts_as_refused(source_repo: Path, root: Path, tmp_path: Path) -> None:
    tb = _release(source_repo, "HEAD", tmp_path / "out")
    assert _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb, "--no-build"]).returncode == 0
    shm = root / "dev" / "shm" / "r1mc3.rt"
    shm.parent.mkdir(parents=True)
    shm.write_bytes(b"\0")  # a segment, but no readable state
    r = _run(["bash", DEPLOY / "bin" / "r1mc3-svc", "stop", "--root", root])
    assert r.returncode == REFUSED, r.stderr


def test_install_makes_the_services_data_folders_the_units_name(source_repo: Path, root: Path, tmp_path: Path) -> None:
    """Session 6: /var/lib/r1mc3/{logs,records,sessions} exist after an install (fake root) and are given to the units'
    user (printed: a fake root runs no chown); they are exactly r1d.service's R1LOG_DIR / R1_RECORD_DIR /
    R1_SESSION_DIR."""
    tb = _release(source_repo, "HEAD", tmp_path / "out")
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb, "--no-build"])
    assert r.returncode == 0, r.stderr
    env = {}
    for line in (DEPLOY / "units" / "r1d.service").read_text().splitlines():
        if line.startswith("Environment=") and "=" in line[len("Environment=") :]:
            k, v = line[len("Environment=") :].split("=", 1)
            env[k] = v
    for key in ("R1LOG_DIR", "R1_RECORD_DIR", "R1_SESSION_DIR"):
        d = root / env[key].lstrip("/")
        assert d.is_dir(), key
        assert f"would run: chown unitree:unitree {d}" in r.stderr, (key, r.stderr)
    assert f"would run: chown unitree:unitree {root / 'var/lib/r1mc3'}" in r.stderr


def test_svc_start_with_and_without_the_stand_lock(root: Path) -> None:
    """Session 6: `r1mc3-svc start --stand` starts r1rt-stand (r1rt --stand-session) and never r1rt.service; `start`
    the unlocked core; `stop` stops whichever core runs (both named in a fake root); bad words refused."""
    svc = DEPLOY / "bin" / "r1mc3-svc"
    r = _run(["bash", svc, "start", "--stand", "--root", root])
    assert r.returncode == 0, r.stderr
    assert "would run: systemctl start r1rt-stand.service" in r.stderr
    assert "would run: systemctl start r1rt.service" not in r.stderr
    assert r.stderr.index("start r1rt-stand.service") < r.stderr.index("start r1guard.service")
    assert "STAND LOCK" in r.stderr
    r = _run(["bash", svc, "start", "--root", root])
    assert "would run: systemctl start r1rt.service" in r.stderr and "r1rt-stand.service" not in r.stderr.replace(
        "would check: no core running (r1rt r1rt-stand)", ""
    )
    r = _run(["bash", svc, "stop", "--root", root])
    assert r.returncode == 0, r.stderr
    assert "would run: systemctl kill --signal=SIGTERM r1rt.service" in r.stderr
    assert "would run: systemctl kill --signal=SIGTERM r1rt-stand.service" in r.stderr
    assert _run(["bash", svc, "stop", "--stand", "--root", root]).returncode == 2


def test_partition_apply_and_undo_in_a_fake_root(root: Path) -> None:
    irqs = {"10": "0-7", "229": "0-7", "300": "0-7"}
    for irq, a0 in irqs.items():
        d = root / "proc" / "irq" / irq
        d.mkdir(parents=True)
        (d / "smp_affinity_list").write_text(a0 + "\n")
    (root / "proc" / "interrupts").write_text(
        "           CPU0       CPU1\n 10:  1 2  GICv3  27 Level  arch_timer\n"
        "229:  5 6  GICv3 195 Level  3610000.xhci\n300:  0 0  GICv3 300 Level  other\n"
    )
    wq = root / "sys" / "devices" / "virtual" / "workqueue" / "cpumask"
    wq.parent.mkdir(parents=True)
    wq.write_text("ff\n")
    script = DEPLOY / "rt" / "partition.sh"
    r = _run(["bash", script, "show", "--root", root])
    assert r.returncode == 0 and "nic irq (auto): 229" in r.stdout, r.stdout + r.stderr
    r = _run(["bash", script, "apply", "--root", root])
    assert r.returncode == 0, r.stderr
    aff = {i: (root / "proc" / "irq" / i / "smp_affinity_list").read_text().strip() for i in irqs}
    assert aff == {"10": "0-4", "229": "6", "300": "0-4"}  # the motor NIC's line on CPU 6, all else on 0-4
    assert wq.read_text().strip() == "1f"
    dropin = root / "etc" / "systemd" / "system.conf.d" / "r1mc3-cpuaffinity.conf"
    assert "CPUAffinity=0 1 2 3 4" in dropin.read_text()
    assert "would run: nvpmodel -m 0" in r.stderr and "would run: jetson_clocks" in r.stderr
    r = _run(["bash", script, "undo", "--root", root])
    assert r.returncode == 0, r.stderr
    aff = {i: (root / "proc" / "irq" / i / "smp_affinity_list").read_text().strip() for i in irqs}
    assert aff == irqs and wq.read_text().strip() == "ff" and not dropin.exists()


def test_partition_dry_run_changes_nothing(root: Path) -> None:
    d = root / "proc" / "irq" / "229"
    d.mkdir(parents=True)
    (d / "smp_affinity_list").write_text("0-7\n")
    before = _tree_hash(root)
    r = _run(["bash", DEPLOY / "rt" / "partition.sh", "apply", "--nic-irq", "229", "--root", root, "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert _tree_hash(root) == before


def test_the_backpack_build_uses_the_release_venvs_tools_and_builds_what_r1d_needs(
    source_repo: Path, root: Path, tmp_path: Path
) -> None:
    """Session 6, the Linux build check (Ubuntu 20.04 arm64 = the Orin's L4T 35.3.1): the backpack's system cmake 3.16
    refuses rt/CMakeLists (3.24 needed), its python3 is 3.8 (3.11 needed) and a --no-dev venv had no nanobind / onnx,
    so the release would have had no r1rt_py (r1d / r1log cannot attach) and no converted policy weights (rtconfig).
    The build now syncs the `build` group, runs the venv's cmake / ninja with the venv's Python 3.11 and builds into
    build/rt (where r1mc3.log._native and r1sim.cpp_policy look); bin/ points there. Fails on the session-6 base."""
    tb = _release(source_repo, "HEAD", tmp_path / "out")
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb])  # build ON (a fake root: printed)
    assert r.returncode == 0, r.stderr
    rel = next((root / "opt" / "r1mc3" / "releases").iterdir())
    runs = [ln for ln in r.stderr.splitlines() if ln.startswith("would run:")]
    sync = next(ln for ln in runs if " uv sync " in ln)
    assert "--group build" in sync and "--no-dev" in sync and "--extra dds" in sync
    assert "--extra perception" in sync  # session 10: OpenCV on the robot for measure.setup (no dev group there)
    cfg = next(ln for ln in runs if " -S " in ln and "/rt " in ln)
    assert f"{rel}/venv/bin/cmake -S {rel}/rt -B {rel}/build/rt " in cfg  # the venv's cmake, the build/rt tree
    assert f"-DPython_EXECUTABLE={rel}/venv/bin/python" in cfg and f"-DPython3_EXECUTABLE={rel}/venv/bin/python" in cfg
    assert f"-DCMAKE_MAKE_PROGRAM={rel}/venv/bin/ninja" in cfg and "-DR1RT_WITH_CYCLONEDDS=ON" in cfg
    assert any(ln.endswith(f"{rel}/venv/bin/cmake --build {rel}/build/rt") for ln in runs)
    assert not any(ln.startswith("would run: cmake ") for ln in runs)  # never the system cmake
    assert os.readlink(rel / "bin" / "r1rt") == "../build/rt/apps/r1rt"
    assert os.readlink(rel / "bin" / "r1guard") == "../build/rt/apps/r1guard"
    group = tomllib.loads((REPO / "pyproject.toml").read_text())["dependency-groups"]["build"]
    names = {g.split(">")[0].split("=")[0].split("<")[0] for g in group}
    assert {"cmake", "ninja", "nanobind", "onnx"} <= names


def test_install_puts_uvs_python_where_the_units_can_read_it(source_repo: Path, root: Path, tmp_path: Path) -> None:
    """HW2 day 1 (12 Oct 2026, the P2 install): install.sh ran `uv sync` under sudo without UV_PYTHON_INSTALL_DIR, so
    uv would put its managed Python 3.11 under /root (unreadable by the units' user: the venv's interpreter links into
    it) - HW1 and HW2 typed `sudo env UV_PYTHON_INSTALL_DIR=/opt/r1mc3/python ...` by hand. The build now sets it to
    <prefix>/python itself; a caller's own value still wins. Fails on 1cf3afd (no UV_PYTHON_INSTALL_DIR in the sync)."""
    tb = _release(source_repo, "HEAD", tmp_path / "out")
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb])  # build ON (a fake root: printed)
    assert r.returncode == 0, r.stderr
    sync = next(ln for ln in r.stderr.splitlines() if ln.startswith("would run:") and " uv sync " in ln)
    assert f"UV_PYTHON_INSTALL_DIR={root}/opt/r1mc3/python " in sync
    env = {**os.environ, "LC_ALL": "C", "UV_PYTHON_INSTALL_DIR": "/somewhere/else"}
    r2 = subprocess.run(
        ["bash", str(DEPLOY / "install.sh"), "--root", str(root), "--tarball", str(tb)],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    assert r2.returncode == 0, r2.stderr
    sync2 = next(ln for ln in r2.stderr.splitlines() if ln.startswith("would run:") and " uv sync " in ln)
    assert "UV_PYTHON_INSTALL_DIR=/somewhere/else " in sync2


def test_install_reads_a_full_size_release_without_sigpipe(source_repo: Path, root: Path, tmp_path: Path) -> None:
    """Session 6, the Linux build check: install.sh took the tarball's top directory from `tar -tzf | head -n 1` under
    `set -o pipefail` - with a full release (~600 files) head exits first, tar dies of SIGPIPE, the pipeline is 141 and
    `set -e` aborts the install (every time on Ubuntu 20.04 / GNU tar; the small test tarballs fitted in the pipe
    buffer). A release with 3000 files installs; fails on the session-6 base."""
    many = source_repo / "many"
    many.mkdir()
    for i in range(3000):
        (many / f"file_{i:04d}_with_a_long_enough_name_to_fill_the_listing.txt").write_text(f"{i}\n")
    _git(source_repo, "add", "-A")
    _git(source_repo, "commit", "-q", "-m", "a full-size release")
    tb = _release(source_repo, "HEAD", tmp_path / "out")
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb, "--no-build"])
    assert r.returncode == 0, (r.returncode, r.stderr[-600:])
    assert any((root / "opt" / "r1mc3" / "releases").iterdir())


def test_bin_python_runs_inside_the_release_venv(source_repo: Path, root: Path, tmp_path: Path) -> None:
    """Session 6, the Linux build check (install.sh run for real in an Ubuntu 20.04 container): bin/python was a
    symlink to ../venv/bin/python; Python finds its venv by the path it is started with, so the units' ExecStart
    (/opt/r1mc3/current/bin/python) ran WITHOUT the venv and r1d / r1log could not import r1mc3. bin/python is now a
    wrapper that starts venv/bin/python by its own path: sys.prefix is the venv. Fails on the session-6 base."""
    import sys

    tb = _release(source_repo, "HEAD", tmp_path / "out")
    r = _run(["bash", DEPLOY / "install.sh", "--root", root, "--tarball", tb, "--no-build"])
    assert r.returncode == 0, r.stderr
    rel = next((root / "opt" / "r1mc3" / "releases").iterdir())
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(rel / "venv")], check=True)
    out = _run([rel / "bin" / "python", "-c", "import sys; print(sys.prefix)"])
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()).resolve() == (rel / "venv").resolve(), out.stdout
