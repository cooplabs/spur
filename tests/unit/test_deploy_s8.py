"""Session 8: the HW1 deploy fixes (fix list items 1-38 in results/hw1_20261010/NOTES.md) - each test FAILS on bad5a5c
(the base) and passes now. All against fake roots / temp repositories; nothing touches this machine's system or any
robot (the scripts run no system command when --root is not "/")."""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from tests.unit.test_deploy_scripts import DEPLOY, _release, _run


@pytest.fixture()
def repo_with_xattrs(tmp_path: Path) -> Path:
    repo = tmp_path / "src"
    repo.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@example.invalid"], ["config", "user.name", "test"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    shutil.copytree(DEPLOY, repo / "deploy")
    (repo / "pyproject.toml").write_text('[project]\nname = "r1mc3"\nversion = "0.1.0.dev0"\n')
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "one"], check=True)
    return repo


# ---------------------------------------------------------------------------------------------- #6 release tarball
def test_the_release_tarball_carries_no_xattrs_and_root_ownership(repo_with_xattrs: Path, tmp_path: Path) -> None:
    """HW1 fix #6: the Mac's tarball stored com.apple.provenance xattrs (731 GNU tar warnings on the backpack) and the
    Mac user (uid 501) as every file's owner. On bad5a5c (macOS): 'LIBARCHIVE.xattr.com.apple.provenance' in the
    stream and uid 501; on Linux: the builder's uid."""
    tb = _release(repo_with_xattrs, "HEAD", tmp_path / "out")
    raw = gzip.decompress(tb.read_bytes())
    assert b"LIBARCHIVE.xattr" not in raw and b"SCHILY.xattr" not in raw, "extended attributes in the tarball"
    with tarfile.open(tb) as t:
        owners = {(m.uid, m.gid) for m in t.getmembers()}
    assert owners == {(0, 0)}, owners


def test_install_unpacks_without_the_tarballs_owner() -> None:
    text = (DEPLOY / "install.sh").read_text()
    assert 'tar --no-same-owner -xzf "$tarball"' in text


# ---------------------------------------------------------------------------------------------- #12 / #21 partition
# /proc/interrupts on the backpack at HW1 (the RT-grant check's read-out, row 4; the xhci lines + a neighbour)
HW1_INTERRUPTS = """\
           CPU0       CPU1       CPU2       CPU3       CPU4       CPU5       CPU6       CPU7
 10:     123456     100000      90000      80000      70000      60000      50000      40000     GICv3  27 Level     arch_timer
228:          0          0          0          0          0          0          0          0     GICv3 199 Level     3610000.xhci
229:     688258          0          0          0          0          0          0          0     GICv3 195 Level     xhci-hcd:usb1
230:          0          0          0          0          0          0          0          0     GICv3 196 Level     3610000.xhci
307:          0          0          0          0          0          0          0          0       MSI 1074266112 Edge      eth19
"""  # noqa: E501


def _fake_irqs(root: Path, text: str = HW1_INTERRUPTS) -> None:
    for irq in ("10", "228", "229", "230", "307"):
        d = root / "proc" / "irq" / irq
        d.mkdir(parents=True, exist_ok=True)
        (d / "smp_affinity_list").write_text("0-7\n")
    (root / "proc" / "interrupts").write_text(text)


def test_partition_picks_the_busiest_xhci_line(tmp_path: Path) -> None:
    """HW1 fix #12: the auto-detection took the FIRST xhci line (228, 0 interrupts); the live line is 229
    (xhci-hcd:usb1: eth10 + Wi-Fi + hands). On bad5a5c: 'nic irq (auto): 228'."""
    root = tmp_path / "root"
    _fake_irqs(root)
    r = _run(["bash", DEPLOY / "rt" / "partition.sh", "show", "--root", root])
    assert r.returncode == 0, r.stderr
    assert "nic irq (auto): 229" in r.stdout, r.stdout
    r = _run(["bash", DEPLOY / "rt" / "partition.sh", "apply", "--root", root])
    assert r.returncode == 0, r.stderr
    assert (root / "proc" / "irq" / "229" / "smp_affinity_list").read_text().strip() == "6"
    assert (root / "proc" / "irq" / "228" / "smp_affinity_list").read_text().strip() == "0-4"


def test_partition_tie_prefers_the_xhci_hcd_line(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_irqs(root, HW1_INTERRUPTS.replace("688258", "     0"))  # right after boot: every xhci line still 0
    r = _run(["bash", DEPLOY / "rt" / "partition.sh", "show", "--root", root])
    assert "nic irq (auto): 229" in r.stdout, r.stdout


def test_move_tasks_moves_user_space_tasks_whose_proc_files_report_size_zero(tmp_path: Path) -> None:
    """HW1 fix #21: `--move-tasks` moved nothing (NetworkManager / sshd stayed on CPUs 0-7): the script skipped every
    task whose /proc/<pid>/cmdline has size 0 - and procfs reports size 0 for ALL of them. Here a user-space task's
    cmdline is a FIFO (size 0, content on read: as procfs), a kernel thread's an empty file, r1rt's skipped by name.
    On bad5a5c: no `taskset ... 1234` at all."""
    root = tmp_path / "root"
    _fake_irqs(root)
    procs = {"1234": ("NetworkManager", "fifo"), "77": ("kworker/0:1", "empty"), "4321": ("r1rt", "fifo")}
    writers = []
    for pid, (comm, kind) in procs.items():
        d = root / "proc" / pid
        d.mkdir(parents=True)
        (d / "comm").write_text(comm + "\n")
        if kind == "empty":
            (d / "cmdline").write_bytes(b"")
        else:
            os.mkfifo(d / "cmdline")
            if comm != "r1rt":  # r1rt is skipped by name before its cmdline is read
                line = f"printf '/usr/sbin/{comm}\\0--no-daemon\\0' > '{d}/cmdline'"
                writers.append(subprocess.Popen(["sh", "-c", line]))
    try:
        r = _run(["bash", DEPLOY / "rt" / "partition.sh", "apply", "--move-tasks", "--root", root], timeout=20)
    finally:
        for w in writers:
            w.kill()
            w.wait()
    assert r.returncode == 0, r.stderr
    assert "would run: taskset -a -p -c 0-4 1234" in r.stderr, r.stderr
    assert "taskset -a -p -c 0-4 77" not in r.stderr and "taskset -a -p -c 0-4 4321" not in r.stderr


def test_partition_dry_run_does_not_say_applied(tmp_path: Path) -> None:
    """HW1 fix #21: the dry run printed "stage 1 applied". On bad5a5c it does."""
    root = tmp_path / "root"
    _fake_irqs(root)
    r = _run(["bash", DEPLOY / "rt" / "partition.sh", "apply", "--root", root, "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert "stage 1 applied" not in r.stderr and "NOT applied" in r.stderr


def test_the_partition_runs_once_per_boot_before_either_core() -> None:
    """HW1 fix #12: a oneshot unit, pulled in by both cores (Wants + After), never by anything else, never blocking."""
    from tests.unit.test_deploy_units import _one, _parse

    u = _parse("r1mc3-partition.service")
    assert _one(u, "Service", "Type") == "oneshot" and _one(u, "Service", "RemainAfterExit") == "yes"
    assert "partition.sh apply --move-tasks" in _one(u, "Service", "ExecStart")
    assert "Install" not in u  # not enabled on its own: only a v3 core pulls it in
    for core in ("r1rt.service", "r1rt-stand.service"):
        c = _parse(core)["Unit"]
        assert "r1mc3-partition.service" in " ".join(c["Wants"]) and "r1mc3-partition.service" in " ".join(c["After"])
        assert "r1mc3-partition.service" not in " ".join(c.get("Requires", []))


# ---------------------------------------------------------------------------------------------- #3 / #4 / #24 site
def _site(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["bash", DEPLOY / "rt" / "site.sh", *args, "--root", root])


def _fake_sysctls(root: Path) -> None:
    for key, val in (
        ("net/core/rmem_max", "425984"),
        ("kernel/sched_rt_runtime_us", "950000"),
        ("kernel/printk", "6\t6\t1\t7"),
    ):  # fmt: skip  (the backpack at HW1)
        f = root / "proc" / "sys" / key
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(val + "\n")


def test_site_apply_default_writes_the_hw1_runtime_fixes_as_a_file(tmp_path: Path) -> None:
    """HW1 fixes #3 / #4 / #24: the runtime sysctls typed by hand after every power-up become a sysctl.d file, applied
    now, the old values saved, undone. The defaults are the user's decisions of 11 Oct 2026: #3 sched_rt_runtime_us -1
    (the method HW1 ran on), #24 kernel.printk "1 4 1 7" (only emergencies to the serial consoles); #4 rmem_max 8 MB
    for cyclonedds.xml's 4 MB buffer. On bad5a5c there is no site.sh."""
    root = tmp_path / "root"
    _fake_sysctls(root)
    r = _site(root, "apply")
    assert r.returncode == 0, r.stderr
    conf = (root / "etc" / "sysctl.d" / "60-r1mc3-rt.conf").read_text()
    assert "net.core.rmem_max = 8388608" in conf.splitlines()
    assert "kernel.sched_rt_runtime_us = -1" in conf.splitlines()
    assert "kernel.printk = 1 4 1 7" in conf.splitlines()  # #24: the user's decision (11 Oct 2026)
    assert f"would run: sysctl -p {root}/etc/sysctl.d/60-r1mc3-rt.conf" in r.stderr
    assert not (root / "etc" / "systemd" / "system" / "r1mc3.slice").exists()
    backup = next((root / "var" / "lib" / "r1mc3" / "site_backup").iterdir())
    saved = (backup / "sysctl_before.txt").read_text().splitlines()
    assert "net.core.rmem_max=425984" in saved and "kernel.sched_rt_runtime_us=950000" in saved
    assert "kernel.printk=6 6 1 7" in saved
    assert _site(root, "apply").returncode == 0  # the same options again: unchanged
    assert _site(root, "apply", "--rt-budget", "slice").returncode == 2  # another method: undo first
    assert _site(root, "apply", "--console-printk", "keep").returncode == 2  # other options: undo first
    r = _site(root, "undo")
    assert r.returncode == 0, r.stderr
    assert not (root / "etc" / "sysctl.d" / "60-r1mc3-rt.conf").exists()
    assert "would run: sysctl -w kernel.sched_rt_runtime_us=950000" in r.stderr
    assert "would run: sysctl -w net.core.rmem_max=425984" in r.stderr


def test_site_apply_console_printk_and_none(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_sysctls(root)
    r = _site(root, "apply", "--rt-budget", "none", "--console-printk", "keep")
    assert r.returncode == 0, r.stderr
    lines = (root / "etc" / "sysctl.d" / "60-r1mc3-rt.conf").read_text().splitlines()
    assert not any(x.startswith(("kernel.printk", "kernel.sched_rt_runtime_us")) for x in lines)
    assert _site(root, "undo").returncode == 0
    r = _site(root, "apply", "--rt-budget", "none", "--console-printk", "1 4 1 7")
    assert r.returncode == 0, r.stderr
    lines = (root / "etc" / "sysctl.d" / "60-r1mc3-rt.conf").read_text().splitlines()
    assert "kernel.printk = 1 4 1 7" in lines and "net.core.rmem_max = 8388608" in lines
    assert not any(x.startswith("kernel.sched_rt_runtime_us") for x in lines)
    assert _site(root, "apply", "--console-printk", "9 9 9 9").returncode == 2
    assert _site(root, "apply", "--rt-budget", "cgroup").returncode == 2


def test_site_apply_slice_method(tmp_path: Path) -> None:
    """#3's other method: the throttle stays on, the four RT units go into r1mc3.slice with an ExecStartPre=+ budget and
    the unit policy `other` (r1rt / r1guard set FIFO on their own threads)."""
    root = tmp_path / "root"
    _fake_sysctls(root)
    r = _site(root, "apply", "--rt-budget", "slice")
    assert r.returncode == 0, r.stderr
    units = root / "etc" / "systemd" / "system"
    assert (units / "r1mc3.slice").exists()
    conf = (root / "etc" / "sysctl.d" / "60-r1mc3-rt.conf").read_text()
    assert "sched_rt_runtime_us" not in conf and "net.core.rmem_max = 8388608" in conf
    for u, leaf in (("r1rt", 400000), ("r1rt-stand", 400000), ("r1guard", 200000), ("r1guard-observe", 200000)):
        d = (units / f"{u}.service.d" / "60-r1mc3-rtbudget.conf").read_text().splitlines()
        assert "Slice=r1mc3.slice" in d and "CPUSchedulingPolicy=other" in d and "CPUSchedulingPriority=0" in d
        pre = f"ExecStartPre=+/bin/bash /opt/r1mc3/current/deploy/rt/rtbudget.sh --slice-us 700000 --leaf-us {leaf}"
        assert pre in d
    assert "would run: systemctl daemon-reload" in r.stderr
    assert _site(root, "undo").returncode == 0
    assert not (units / "r1mc3.slice").exists() and not (units / "r1rt.service.d" / "60-r1mc3-rtbudget.conf").exists()


def test_site_dry_run_changes_nothing(tmp_path: Path) -> None:
    from tests.unit.test_deploy_scripts import _tree_hash

    root = tmp_path / "root"
    _fake_sysctls(root)
    before = _tree_hash(root)
    r = _run(["bash", DEPLOY / "rt" / "site.sh", "apply", "--rt-budget", "slice", "--root", root, "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert _tree_hash(root) == before and "NOT applied" in r.stderr


def _fake_cgroup(root: Path, path: str, budgets: dict[str, int]) -> Path:
    mnt = root / "sys" / "fs" / "cgroup" / "cpu,cpuacct"
    for rel, v in budgets.items():
        d = mnt / rel.lstrip("/")
        d.mkdir(parents=True, exist_ok=True)
        (d / "cpu.rt_runtime_us").write_text(f"{v}\n")
    proc = root / "proc" / "4242"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text(f"12:memory:/system.slice\n4:cpu,cpuacct:{path}\n1:name=systemd:{path}\n")
    return mnt


def test_rtbudget_writes_the_slice_then_the_unit(tmp_path: Path) -> None:
    root = tmp_path / "root"
    mnt = _fake_cgroup(root, "/r1mc3.slice/r1rt.service",
                       {"/": 950000, "/r1mc3.slice": 0, "/r1mc3.slice/r1rt.service": 0})  # fmt: skip
    r = _run(["bash", DEPLOY / "rt" / "rtbudget.sh", "--pid", "4242", "--leaf-us", "400000", "--root", root])
    assert r.returncode == 0, r.stderr
    assert (mnt / "r1mc3.slice" / "cpu.rt_runtime_us").read_text().strip() == "700000"
    assert (mnt / "r1mc3.slice" / "r1rt.service" / "cpu.rt_runtime_us").read_text().strip() == "400000"
    assert (mnt / "cpu.rt_runtime_us").read_text().strip() == "950000"  # the root untouched


def test_rtbudget_nothing_to_do_in_the_root_cgroup(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_cgroup(root, "/", {"/": 950000})
    r = _run(["bash", DEPLOY / "rt" / "rtbudget.sh", "--pid", "4242", "--root", root])
    assert r.returncode == 0 and "nothing to do" in r.stderr, r.stderr


def test_new_scripts_parse() -> None:
    for s in (DEPLOY / "rt" / "site.sh", DEPLOY / "rt" / "rtbudget.sh"):
        r = _run(["bash", "-n", s])
        assert r.returncode == 0, f"{s}: {r.stderr}"
        if shutil.which("shellcheck"):
            r = _run(["shellcheck", "-S", "error", s])
            assert r.returncode == 0, r.stdout


def test_install_applies_the_site_settings_with_its_options(repo_with_xattrs: Path, tmp_path: Path) -> None:
    """install.sh (4c): the site settings by default (the user's decisions of 11 Oct 2026: rt-budget sysctl, printk
    "1 4 1 7"), the options passed through, --no-site skips them. On bad5a5c: no sysctl.d file after an install."""
    tb = _release(repo_with_xattrs, "HEAD", tmp_path / "out")
    root = tmp_path / "root"
    root.mkdir()
    _fake_sysctls(root)
    r = _run(["bash", DEPLOY / "install.sh", "--tarball", tb, "--no-build", "--root", root])
    assert r.returncode == 0, r.stderr
    conf = (root / "etc" / "sysctl.d" / "60-r1mc3-rt.conf").read_text().splitlines()
    assert "kernel.sched_rt_runtime_us = -1" in conf and "net.core.rmem_max = 8388608" in conf
    assert "kernel.printk = 1 4 1 7" in conf
    root2 = tmp_path / "root2"
    root2.mkdir()
    r = _run(["bash", DEPLOY / "install.sh", "--tarball", tb, "--no-build", "--root", root2, "--rt-budget", "slice",
              "--console-printk", "4 4 1 7"])  # fmt: skip
    assert r.returncode == 0, r.stderr
    assert (root2 / "etc" / "systemd" / "system" / "r1mc3.slice").exists()
    assert "kernel.printk = 4 4 1 7" in (root2 / "etc" / "sysctl.d" / "60-r1mc3-rt.conf").read_text()
    root3 = tmp_path / "root3"
    root3.mkdir()
    r = _run(["bash", DEPLOY / "install.sh", "--tarball", tb, "--no-build", "--root", root3, "--no-site"])
    assert r.returncode == 0 and not (root3 / "etc" / "sysctl.d").exists(), r.stderr


# ---------------------------------------------------------------------------------------------- #13 / #20 shutdown
def test_each_shutdown_saves_into_a_new_folder_with_every_unsaved_file_of_the_day(tmp_path: Path) -> None:
    """HW1 #13: `r1 shutdown` wrote FLAT into R1_SESSION_DIR (a 2nd save would overwrite manifest.json; the operator
    moved each save aside by hand). HW1 #20: it took only files since r1d's LAST start (after a restart the earlier
    r1log files of the day were in no folder). Now: a new `<label>_<UTC>` folder per shutdown, every file of the last
    24 h not already saved byte for byte. On bad5a5c: both saves land in the same folder (the first manifest
    overwritten) and the file from before r1d's start is missing."""
    import json
    import time

    from tests.unit.test_hw1_ops import FakePort, _op, _svc
    from tests.unit.test_hw1_ops import _run as run_req

    logs, recs, sessions = tmp_path / "logs", tmp_path / "records", tmp_path / "sessions"
    logs.mkdir()
    recs.mkdir()
    before_restart = logs / "r1log_morning_000.mcap"  # written by the r1log of an earlier r1d start, 2 h ago
    before_restart.write_bytes(os.urandom(2048))
    os.utime(before_restart, (time.time() - 7200, time.time() - 7200))
    (logs / "r1log_now_000.mcap").write_bytes(os.urandom(1024))
    svc = _svc(FakePort("damp"), tmp_path, session_dir=sessions)
    r1 = run_req(svc, _op("shutdown"))
    assert r1["ok"], r1
    f1 = Path(r1["data"]["saved"]["folder"])
    assert f1.parent == sessions and f1.name.startswith("session_"), f1  # bad5a5c: f1 == sessions (flat)
    m1 = json.loads((f1 / "manifest.json").read_text())
    files1 = {e["file"] for e in m1["files"]}
    assert {"r1log/r1log_morning_000.mcap", "r1log/r1log_now_000.mcap"} <= files1, files1
    time.sleep(1.1)  # a new UTC second: a different folder name even without the suffix rule
    (logs / "r1log_afternoon_000.mcap").write_bytes(os.urandom(512))
    r2 = run_req(svc, {**_op("shutdown"), "label": "part4"})
    assert r2["ok"], r2
    f2 = Path(r2["data"]["saved"]["folder"])
    assert f2 != f1 and f2.parent == sessions and f2.name.startswith("part4_")
    files2 = {e["file"] for e in json.loads((f2 / "manifest.json").read_text())["files"]}
    assert "r1log/r1log_afternoon_000.mcap" in files2
    assert "r1log/r1log_morning_000.mcap" not in files2 and "r1log/r1log_now_000.mcap" not in files2  # saved before
    assert json.loads((f1 / "manifest.json").read_text()) == m1  # the first save untouched


def test_hw1s_flat_save_counts_as_saved_and_folders_are_never_reused(tmp_path: Path) -> None:
    import json

    from r1mc3.hw1.session import new_session_folder, previous_saves

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    src = tmp_path / "logs" / "a.mcap"
    src.parent.mkdir()
    src.write_bytes(b"abc")
    (sessions / "manifest.json").write_text(json.dumps({"files": [{"src": str(src), "bytes": 3, "verified": True}]}))
    assert previous_saves(sessions) == {str(src.resolve()): 3}
    (sessions / "s_20261012T000000Z").mkdir()
    assert new_session_folder(sessions, "s", "20261012T000000Z").name == "s_20261012T000000Z_2"


def test_cli_shutdown_label() -> None:
    from r1mc3.cli.main import build_request

    assert build_request(["shutdown"])[0] == {"requester": "operator", "op": "shutdown"}
    assert build_request(["shutdown", "--label", "hw2_stand"])[0]["label"] == "hw2_stand"


# ---------------------------------------------------------------------------------------------- #31 r1 status
def test_status_shows_the_cores_stops_apart_from_overruns(tmp_path: Path) -> None:
    """HW1 fix #31: the deliberate SIGSTOP windows (the board-hold + guard-takeover tests) were inside
    skipped_deadlines. The core counts its long silences apart (stops, stop_ms_max in its heartbeat / rt_info); `r1
    status` shows them in numbers and words. On bad5a5c: no `stops` key, no words."""
    from tests.unit.test_hw1_ops import FakePort, _svc

    port = FakePort("on_stand")
    port.rt = {"stand_session": True, "stops": 4, "stop_ms_max": 3000.08}
    st = _svc(port, tmp_path).status()
    assert st["core"]["stops"] == 4 and st["core"]["stop_ms_max"] == 3000.08
    assert any("4 long silence(s)" in x and "3000 ms" in x for x in st["summary"]), st["summary"]
    port.rt = {"stand_session": True}  # a core / binding that does not report them
    st = _svc(port, tmp_path).status()
    assert st["core"]["stops"] is None and not any("long silence" in x for x in st["summary"])


# ---------------------------------------------------------------------------------------------- #34 board-hold
# durations
def test_board_hold_escalation_is_20_100_300_400_500_and_refuses_1000_2000(tmp_path: Path) -> None:
    """HW1 #34: the board holds the last LowCmd ~400 ms then releases (the board-hold test, the user stopped at 500 ms):
    `r1 board-hold` takes 20 / 100 / 300 / 400 / 500 and refuses 1000 / 2000 until #29 is proven on the stand (a
    tightening). On bad5a5c: 300 / 400 refused, 1000 / 2000 accepted (CLI and schema)."""
    from pydantic import TypeAdapter, ValidationError
    from r1mc3.cli.main import build_request
    from r1mc3.hw1.pause import BOARD_HOLD_PAUSES_MS
    from r1mc3.service.schemas import BoardHoldReq

    assert BOARD_HOLD_PAUSES_MS == (20, 100, 300, 400, 500)
    for ms in (300, 400):
        assert build_request(["board-hold", "--pause-ms", str(ms)])[0]["pause_ms"] == ms
        assert TypeAdapter(BoardHoldReq).validate_python({"op": "board_hold.pause", "pause_ms": ms}).pause_ms == ms
    for ms in (1000, 2000):
        with pytest.raises(SystemExit):
            build_request(["board-hold", "--pause-ms", str(ms)])
        with pytest.raises(ValidationError):
            TypeAdapter(BoardHoldReq).validate_python({"op": "board_hold.pause", "pause_ms": ms})
