"""The systemd units in deploy/units (DESIGN.md §2.1, §11.1): the RT limits and CPU plan, a stop that can never damp
or escalate to SIGKILL for r1rt / r1guard, no unit dependency that would stop the guard with the core, and no CPU
accounting on the RT units (the stock L4T kernel's RT_GROUP_SCHED would deny SCHED_FIFO, RESEARCH.md §2.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

UNITS = Path(__file__).resolve().parents[2] / "deploy" / "units"


def _parse(name: str) -> dict[str, dict[str, list[str]]]:
    """A systemd unit as {section: {key: [values]}} (systemd allows repeated keys; configparser does not)."""
    out: dict[str, dict[str, list[str]]] = {}
    sec = ""
    for raw in (UNITS / name).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1]
            out.setdefault(sec, {})
            continue
        key, _, val = line.partition("=")
        assert sec, f"{name}: key outside a section: {line}"
        out[sec].setdefault(key.strip(), []).append(val.strip())
    return out


def _one(u: dict[str, dict[str, list[str]]], sec: str, key: str) -> str:
    vals = u.get(sec, {}).get(key)
    assert vals is not None and len(vals) == 1, f"[{sec}] {key}: {vals}"
    return vals[0]


def test_the_five_units_exist() -> None:
    # session 6: + r1guard-observe.service (the observe-only guard of the board-hold test), r1rt-stand.service (HW1's
    # stand lock)
    assert {p.name for p in UNITS.glob("*.service")} == {
        "r1rt.service",
        "r1guard.service",
        "r1d.service",
        "r1log.service",
        "r1mc3-inhibit.service",
        "r1guard-observe.service",
        "r1rt-stand.service",
        "r1mc3-partition.service",
        "r1cam.service",  # session 9 (cam): the camera service, off the RT CPUs, started explicitly (not by r1mc3-svc)
    }  # fmt: skip  (session 8, fix #12: the RT partition once per boot before the core)


def test_the_board_hold_observe_only_guard() -> None:
    """Session 6: r1guard-observe.service (the board-hold test only) runs the same guard with --observe-only, conflicts
    with the armed one (systemd never runs both; starting either stops the other), is never installed to start at boot,
    and keeps the guard's never-kill stop settings."""
    obs, armed = _parse("r1guard-observe.service"), _parse("r1guard.service")
    assert _one(obs, "Service", "ExecStart") == _one(armed, "Service", "ExecStart") + " --observe-only"
    assert _one(obs, "Unit", "Conflicts") == "r1guard.service"
    assert "Install" not in obs  # not enabled at boot: started by hand for the board-hold test, explicit OK
    for key in ("SendSIGKILL", "TimeoutStopSec", "Restart", "KillSignal", "CPUSchedulingPriority", "CPUAffinity"):
        assert _one(obs, "Service", key) == _one(armed, "Service", key), key
    for key in ("BindsTo", "PartOf", "Requires"):
        assert key not in obs["Unit"]


@pytest.mark.parametrize(("name", "prio", "cpus"), [("r1rt.service", "90", "6 7"), ("r1guard.service", "88", "5")])
def test_rt_units_limits_and_a_stop_that_never_kills(name: str, prio: str, cpus: str) -> None:
    u = _parse(name)
    s = "Service"
    assert _one(u, s, "CPUSchedulingPolicy") == "fifo"
    assert _one(u, s, "CPUSchedulingPriority") == prio
    assert _one(u, s, "CPUAffinity") == cpus
    assert int(_one(u, s, "LimitRTPRIO")) >= int(prio)
    assert _one(u, s, "LimitMEMLOCK") == "infinity"
    assert _one(u, "Unit", "RefuseManualStop") == "yes"  # systemctl stop / restart refused by systemd itself
    assert _one(u, s, "SendSIGKILL") == "no"
    assert _one(u, s, "TimeoutStopSec") == "infinity"
    assert _one(u, s, "Restart") == "no"
    for key in ("CPUAccounting", "CPUQuota", "CPUWeight", "CPUShares", "Slice"):
        assert key not in u[s], f"{name}: {key} would move it out of the root cpu cgroup (no SCHED_FIFO)"
    assert "--shm-name /r1mc3.rt" in _one(u, s, "ExecStart")


def test_r1d_stop_never_kills_its_pause_backstops() -> None:
    """Session 6: `r1 board-hold` / `r1 guard-takeover` SIGSTOP r1rt from a child of r1d whose SIGCONT backstop must
    survive an r1d crash / restart: KillMode=process (systemd's default control-group would kill the children with r1d's
    main process)."""
    assert _one(_parse("r1d.service"), "Service", "KillMode") == "process"


def _env(u: dict[str, dict[str, list[str]]]) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in u["Service"].get("Environment", []))


def _arg(cmd: str, flag: str) -> str:
    words = cmd.split()
    return words[words.index(flag) + 1]


STATE = "/var/lib/r1mc3"


def test_r1d_finds_r1logs_files_its_records_its_sessions_and_r1rts_config() -> None:
    """Session 6 (found by the runbook agent): r1d had no R1LOG_DIR / R1_RECORD_DIR / R1_SESSION_DIR / R1RT_CONFIG - on
    the robot `r1 shutdown`'s log save would not find r1log's files. r1d's R1LOG_DIR = r1log's --out, its R1RT_CONFIG
    = r1rt's --config, every folder under /var/lib/r1mc3 and in r1d's StateDirectory= (systemd creates + owns it)."""
    d, lg, rt = _parse("r1d.service"), _parse("r1log.service"), _parse("r1rt.service")
    env = _env(d)
    assert env["R1LOG_DIR"] == _arg(_one(lg, "Service", "ExecStart"), "--out") == _env(lg)["R1LOG_DIR"]
    assert env["R1RT_CONFIG"] == _arg(_one(rt, "Service", "ExecStart"), "--config")
    state_dirs = set(_one(d, "Service", "StateDirectory").split())
    for key in ("R1LOG_DIR", "R1_RECORD_DIR", "R1_SESSION_DIR"):
        assert env[key].startswith(STATE + "/"), (key, env[key])
        assert env[key].removeprefix("/var/lib/") in state_dirs, (key, state_dirs)
    assert _env(lg)["R1LOG_DIR"].removeprefix("/var/lib/") in _one(lg, "Service", "StateDirectory").split()
    assert len({env["R1LOG_DIR"], env["R1_RECORD_DIR"], env["R1_SESSION_DIR"]}) == 3
    assert _one(d, "Service", "User") == _one(lg, "Service", "User")  # one owner for the shared folders


def test_the_stand_session_core() -> None:
    """Session 6 (the user's approval): r1rt-stand.service = r1rt.service + `--stand-session` (BALANCE refused for the
    process's lifetime), the two cores conflict (systemd never runs both), the same limits / never-kill stop / motor
    check, not enabled at boot."""
    st, rt = _parse("r1rt-stand.service"), _parse("r1rt.service")
    assert _one(st, "Service", "ExecStart") == _one(rt, "Service", "ExecStart") + " --stand-session"
    assert _one(st, "Unit", "Conflicts") == "r1rt.service" and _one(rt, "Unit", "Conflicts") == "r1rt-stand.service"
    assert "Install" not in st
    assert _one(st, "Unit", "RefuseManualStop") == "yes"
    for key in (
        "ExecStop",
        "SendSIGKILL",
        "TimeoutStopSec",
        "Restart",
        "KillSignal",
        "User",
        "CPUSchedulingPolicy",
        "CPUSchedulingPriority",
        "CPUAffinity",
        "LimitRTPRIO",
        "LimitMEMLOCK",
        "OOMScoreAdjust",
        "Environment",
    ):
        assert st["Service"].get(key) == rt["Service"].get(key), key  # fmt: skip


def test_no_unit_pulls_in_a_core() -> None:
    """Session 6: starting r1guard / r1d / r1log must never start r1rt.service by dependency - it Conflicts= the stand
    session's core and would replace it (lifting the stand lock behind the operator's back). Ordering only."""
    for p in UNITS.glob("*.service"):
        u = _parse(p.name)["Unit"]
        for key in ("Wants", "Requires", "BindsTo", "Requisite", "Upholds"):
            for v in u.get(key, []):
                assert "r1rt" not in v, f"{p.name}: {key}={v}"
        if p.name not in ("r1rt.service", "r1rt-stand.service") and "r1rt.service" in " ".join(u.get("After", [])):
            assert "r1rt-stand.service" in " ".join(u["After"]), p.name


def test_the_guard_never_stops_with_the_core() -> None:
    u = _parse("r1guard.service")["Unit"]
    for key in ("BindsTo", "PartOf", "Requires", "Requisite", "StopWhenUnneeded", "PropagatesStopTo"):
        assert key not in u, f"r1guard.service: {key} could stop the guard together with r1rt"


def test_no_stop_path_damps_or_publishes() -> None:
    for p in UNITS.glob("*.service"):
        u = _parse(p.name)
        for key, vals in u.get("Service", {}).items():
            if key.startswith("Exec"):
                for v in vals:
                    assert "damp" not in v.lower(), f"{p.name} {key}={v}"
        stops = [v for k, vals in u.get("Service", {}).items() if k.startswith("ExecStop") for v in vals]
        for v in stops:  # the only stop hook is the read-only motor check (logged; its result is ignored: "-")
            assert v.startswith("-") and "r1mc3_motorcheck.py" in v, f"{p.name}: {v}"


def test_python_services_stay_off_the_rt_cpus() -> None:
    for name in ("r1d.service", "r1log.service", "r1mc3-inhibit.service"):
        cpus = {int(c) for c in _one(_parse(name), "Service", "CPUAffinity").split()}
        assert cpus <= {0, 1, 2, 3, 4}, name
    assert _one(_parse("r1log.service"), "Service", "Nice") == "10"


def test_r1cam_stays_off_the_rt_cpus_below_normal_priority_and_is_not_started_by_the_stack() -> None:
    """Session 9 (the user's warning: the camera has slowed other things before): CPUs 0-4 only, never RT, nice,
    idle-class disk I/O, memory-capped; r1mc3-svc does not start it (explicit, after the HW2 impact test)."""
    u = _parse("r1cam.service")
    assert _one(u, "Service", "CPUAffinity") == "0 1 2 3 4"
    assert _one(u, "Service", "CPUSchedulingPolicy") == "other"
    assert int(_one(u, "Service", "Nice")) >= 10
    assert _one(u, "Service", "IOSchedulingClass") == "idle"
    assert _one(u, "Service", "MemoryMax") == "1G"
    svc = (UNITS.parent / "bin" / "r1mc3-svc").read_text()
    assert "r1cam" not in svc
