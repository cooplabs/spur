"""The session-end log save built into `r1 shutdown` (DESIGN.md §4.4, §9).

v2 lost a session's log once because the save was a separate script that needed the right serve name (NI-3). Here the
save is part of the shutdown word itself and checks its own work:

1. r1log is asked to finish (SIGINT, its unit's KillSignal: a final drain, the open MCAP closed) when r1d can find it
   (`R1LOG_PID` / `R1LOG_PIDFILE`, else the process table: `python -m r1mc3.log run ... --segment <our segment>`), and
   r1d waits for it to exit (<= 20 s, the unit's TimeoutStopSec). Under systemd it does not restart (exit 0).
2. The session's files - r1log's (R1LOG_DIR), the recordings' manifests and data (R1_RECORD_DIR) and r1d's own request
   journal - go into a NEW session folder `<R1_SESSION_DIR>/<label>_<UTC>` (default parent `<records>/sessions`,
   label "session"; a folder that exists is never reused): hard links where the filesystem allows (no second copy of
   a GB-sized log), else copies. Session 8 (HW1 fixes #13 / #20): HW1's `r1 shutdown` wrote FLAT into R1_SESSION_DIR
   (a second shutdown would have overwritten manifest.json; the operator moved each save aside by hand) and took only
   the files modified since r1d's LAST start (after a v3 restart the earlier r1log files of the day were in no
   session folder: 43 files copied by hand). Now "the session's files" = every file of the last
   `R1_SESSION_LOOKBACK_H` hours (24) that no earlier session folder under the same parent already holds byte for
   byte (the earlier manifests' source path + size; a file that grew after an earlier snapshot is saved again).
3. Every file's size + sha256 is computed on the source and re-read from the session folder; `manifest.json` lists
   them. Only when every file verifies (and r1log has stopped or was not running) does the result say
   "safe to power off now".

Nothing here touches the motors: `r1 shutdown` is accepted only in MONITOR / DAMP (the service checks that first), and
stopping r1rt / r1guard is systemd's job (`r1mc3-svc stop`, which refuses while we own the motors) or the power-off.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

R1LOG_STOP_TIMEOUT_S = 20.0  # deploy/units/r1log.service TimeoutStopSec
MTIME_SLACK_S = 5.0
LOOKBACK_H = 24.0  # R1_SESSION_LOOKBACK_H: how far back `r1 shutdown` looks for files not saved yet
DEFAULT_LABEL = "session"


def lookback_s() -> float:
    try:
        return 3600.0 * float(os.environ.get("R1_SESSION_LOOKBACK_H", LOOKBACK_H))
    except ValueError:
        return 3600.0 * LOOKBACK_H


def new_session_folder(parent: Path, label: str | None, stamp: str) -> Path:
    """`<parent>/<label>_<stamp>`, never an existing one (a suffix _2, _3, ... if needed); not created here."""
    base = f"{label or DEFAULT_LABEL}_{stamp}"
    folder, n = parent / base, 1
    while folder.exists():
        n += 1
        folder = parent / f"{base}_{n}"
    return folder


def previous_saves(parent: Path) -> dict[str, int]:
    """Source path -> size of every file an earlier session folder under `parent` holds (their manifest.json; the
    HW1 flat layout `parent/manifest.json` too). Unreadable manifests are skipped (they then protect nothing)."""
    seen: dict[str, int] = {}
    if not parent.is_dir():
        return seen
    for man in [parent / "manifest.json", *sorted(parent.glob("*/manifest.json"))]:
        if not man.is_file():
            continue
        try:
            files = json.loads(man.read_text()).get("files", [])
        except (OSError, ValueError):
            continue
        for e in files:
            src, size = e.get("src"), e.get("bytes")
            if isinstance(src, str) and isinstance(size, int) and e.get("verified", True):
                seen[str(Path(src).resolve())] = size
    return seen


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pid_alive(pid: int) -> bool:
    """Running (a zombie - exited, not yet reaped by its parent - counts as exited: under systemd the parent reaps at
    once; in the simulator rehearsal r1log's parent is the harness)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(st.strip()) and not st.strip().startswith("Z")


def find_r1log(shm_name: str) -> list[int]:
    """r1log's pid(s) for OUR segment: R1LOG_PID / R1LOG_PIDFILE, else the process table (`ps`), matching
    `r1mc3.log run` with `--segment <shm_name>` (the unit passes --segment /r1mc3.rt explicitly)."""
    if os.environ.get("R1LOG_PID"):
        return [int(os.environ["R1LOG_PID"])]
    pf = os.environ.get("R1LOG_PIDFILE")
    if pf and Path(pf).exists():
        with contextlib.suppress(ValueError):
            return [int(Path(pf).read_text().strip())]
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, cmd = parts
        words = cmd.split()
        if "r1mc3.log" not in words or "run" not in words:
            continue
        seg = words[words.index("--segment") + 1] if "--segment" in words[:-1] else "/r1mc3.rt"
        if seg == shm_name and int(pid_s) != os.getpid():
            pids.append(int(pid_s))
    return pids


def stop_r1log(pids: Iterable[int], timeout_s: float = R1LOG_STOP_TIMEOUT_S) -> dict[str, Any]:
    """SIGINT each r1log (its final drain) and wait for it to exit; what happened per pid."""
    res: dict[str, Any] = {}
    pids = list(pids)
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGINT)
    end = time.monotonic() + timeout_s
    for pid in pids:
        while _pid_alive(pid) and time.monotonic() < end:
            with contextlib.suppress(ChildProcessError, OSError):
                os.waitpid(pid, os.WNOHANG)  # our own child in tests: reap it so it does not linger as a zombie
            time.sleep(0.05)
        res[str(pid)] = "exited" if not _pid_alive(pid) else "still running"
    return res


def _session_files(
    dirs: Iterable[tuple[str, Path]],
    since_wall: float,
    exclude: Iterable[Path],
    saved: dict[str, int] | None = None,
) -> list[tuple[str, Path]]:
    ex = [e.resolve() for e in exclude]
    saved = saved or {}
    out: list[tuple[str, Path]] = []
    for label, d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            rp = p.resolve()
            if not p.is_file() or any(e == rp or e in rp.parents for e in ex) or p.name.startswith("."):
                continue
            st = p.stat()
            if st.st_mtime < since_wall - MTIME_SLACK_S or saved.get(str(rp)) == st.st_size:
                continue  # older than the look-back, or already in an earlier session folder byte for byte
            out.append((label, p))
    return out


def save_session(
    *,
    session_dir: Path,
    log_dir: Path | None,
    record_dir: Path,
    since_wall: float,
    journal: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    live: Iterable[Path] = (),
    log_closed: bool = True,
    sessions_root: Path | None = None,
) -> dict[str, Any]:
    """Link / copy the session's files into `session_dir`, hash them, write manifest.json.

    A hard link shares the file with its writer, so only CLOSED files are linked (hashed on the source, re-read from
    the session folder). A file whose writer may still run - `live` (r1d's own journal, appended after every request
    until r1d stops) and r1log's files when r1log was not stopped (`log_closed=False`) - is COPIED and marked a
    `snapshot`: its hash is the copy's own, which nobody writes again (rehearsal R4: a linked journal changed after its
    hash was recorded). `sessions_root` (session 8): the parent of every session folder - its folders are excluded
    and the files their manifests already hold (same source, same size) are skipped. Returns {folder, files, bytes,
    verified, failed: [...]}."""
    saved = previous_saves(sessions_root) if sessions_root is not None else {}
    session_dir.mkdir(parents=True, exist_ok=True)
    jpath = session_dir / "r1d_journal.jsonl"
    jpath.write_text("".join(json.dumps(e) + "\n" for e in journal))
    sources: list[tuple[str, Path]] = []
    if log_dir is not None:
        sources.append(("r1log", log_dir))
    sources.append(("records", record_dir))
    entries: list[dict[str, Any]] = []
    failed: list[str] = []
    live_set = {Path(x).resolve() for x in live}
    excluded = [session_dir, record_dir / "sessions"]  # earlier sessions' folders are not this session's files
    if sessions_root is not None:
        excluded.append(sessions_root)
    for label, src in _session_files(sources, since_wall, exclude=excluded, saved=saved):
        root = log_dir if label == "r1log" and log_dir is not None else record_dir
        rel = Path(label) / src.relative_to(root)
        dst = session_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        growing = src.resolve() in live_set or (label == "r1log" and not log_closed)
        if growing:  # a snapshot: copied, hashed on the copy (its writer may append to the source afterwards)
            shutil.copy2(src, dst)
            how = "snapshot"
            h_src = sha256_of(dst)
            size = dst.stat().st_size
        else:
            h_src = sha256_of(src)
            size = src.stat().st_size
            how = "link"
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
                how = "copy"
        h_dst = sha256_of(dst)
        ok = h_dst == h_src and dst.stat().st_size == size
        if not ok:
            failed.append(str(rel))
        entries.append({"file": str(rel), "src": str(src), "bytes": size, "sha256": h_src, "how": how, "verified": ok})
    entries.append(
        {
            "file": jpath.name,
            "src": None,
            "bytes": jpath.stat().st_size,
            "sha256": sha256_of(jpath),
            "how": "written",
            "verified": True,
        }
    )
    manifest = {
        "schema": "r1mc3.session_manifest/1",
        "utc_saved": datetime.now(UTC).isoformat(),
        "since_utc": datetime.fromtimestamp(since_wall, UTC).isoformat(),
        "rule": "every file since since_utc not already held byte for byte by an earlier session folder",
        "earlier_saved_files": len(saved),
        "files": entries,
        "verified": not failed,
        **(extra or {}),
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return {
        "folder": str(session_dir),
        "files": len(entries),
        "bytes": int(sum(e["bytes"] for e in entries)),
        "verified": not failed,
        "failed": failed,
        "r1log_files": sum(1 for e in entries if e["file"].startswith("r1log/")),
        "record_files": sum(1 for e in entries if e["file"].startswith("records/")),
    }
