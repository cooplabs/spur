"""Session 13 (HW2 day 2, 08:35 CST): `r1mc3-svc start` started r1guard 8 ms after r1rt; r1guard exited "attach
/r1mc3.rt: segment smaller than expected" (r1rt was still sizing its segment) and the start said "started" anyway - the
operator found it in `r1 status` (r1guard ABSENT) and fixed it by hand (`sudo systemctl start r1guard`). Now the start
waits for r1rt's READY line before it starts r1guard (the segment is made by then), waits for r1guard's, and says which
one is missing. In a fake root with a fake journal (R1_SVC_JOURNAL); nothing touches this machine's system. On the base
(dcc01a2) the start never looked at a ready line: the missing-guard case exited 0 ("started").
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SVC = REPO / "deploy" / "bin" / "r1mc3-svc"
RT_READY = '{"ready":true,"process":"r1rt","pid":101,"rt_granted":true,"rate_hz":500,"policy_hz":50}'
GUARD_READY = '{"ready":true,"process":"r1guard","pid":102,"rt_granted":true,"cycle_hz":1000}'
GUARD_FAIL = "r1guard: attach /r1mc3.rt: segment smaller than expected"


def _fake_journal(tmp: Path, journals: dict[str, str]) -> Path:
    """A `journal <unit> <t0>` command printing the given text per unit (empty for the others)."""
    for unit, text in journals.items():
        (tmp / f"{unit}.journal").write_text(text + "\n")
    script = tmp / "journal.sh"
    script.write_text(f'#!/usr/bin/env bash\ncat "{tmp}/$1.journal" 2>/dev/null || true\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _start(tmp: Path, journal: Path | None, *extra: str) -> subprocess.CompletedProcess[str]:
    root = tmp / "root"
    root.mkdir(exist_ok=True)
    env = {**os.environ, "LC_ALL": "C", "R1_SVC_READY_WAIT_S": "1"}
    if journal is not None:
        env["R1_SVC_JOURNAL"] = str(journal)
    return subprocess.run(["bash", str(SVC), "start", *extra, "--root", str(root)], capture_output=True, text=True,
                          env=env, check=False)  # fmt: skip


def test_start_waits_for_r1rt_before_the_guard_and_checks_both(tmp_path: Path) -> None:
    j = _fake_journal(tmp_path, {"r1rt": RT_READY, "r1guard": GUARD_READY})
    r = _start(tmp_path, j)
    assert r.returncode == 0, r.stderr
    e = r.stderr
    assert e.index("systemctl start r1rt.service") < e.index("ready: r1rt") < e.index("systemctl start r1guard.service")
    assert e.index("systemctl start r1guard.service") < e.index("ready: r1guard") < e.index("start r1d.service")
    assert e.count("systemctl start r1guard.service") == 1  # once, not again in the rest
    assert "ready: r1rt + r1guard" in e


def test_a_guard_without_its_ready_line_is_said_by_name(tmp_path: Path) -> None:
    j = _fake_journal(tmp_path, {"r1rt": RT_READY, "r1guard": GUARD_FAIL})
    r = _start(tmp_path, j)
    assert r.returncode == 2, r.stderr  # base: 0 ("started") with the guard absent
    assert "NO READY LINE from r1guard" in r.stderr and "segment smaller than expected" in r.stderr
    assert "systemctl start r1d.service" not in r.stderr  # nothing started after the missing guard


def test_the_stand_core_is_waited_for_too_and_a_fake_root_only_says_it_would(tmp_path: Path) -> None:
    j = _fake_journal(tmp_path, {"r1rt-stand": RT_READY, "r1guard": GUARD_READY})
    r = _start(tmp_path, j, "--stand")
    assert r.returncode == 0, r.stderr
    assert r.stderr.index("start r1rt-stand.service") < r.stderr.index("ready: r1rt")
    r = _start(tmp_path, None)
    assert r.returncode == 0, r.stderr
    assert "would wait <= 1 s for r1rt's ready line" in r.stderr and "would wait <= 1 s for r1guard's" in r.stderr
