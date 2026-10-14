"""The v3-aware eth10 guard (deploy/eth10/r1mc3-eth10-guard.sh; DESIGN.md D18, the user's decision U8) in a FAKE
environment with a VIRTUAL clock: a fake root's /sys/class/net/eth10/carrier, and fake `sleep` (advances the clock and
applies the scenario's events), `ping` (answers or times out after 1 virtual s), `ip`, `ethtool`, `logger`, `pgrep`,
`systemctl` that record every call with the virtual time. Each case is a timeline:

  * no controller: the guard does EXACTLY what v2's own script does (v2's file from tests/assets/eth10, run in the same
    fake world): the same ethtool / ip calls at the same virtual times - the 5 Oct auto-recovery kept;
  * r1rt running: only logs - no reset, no ethtool call at all (start, link-up, silence);
  * r1rt starting mid-outage: no reset while it runs, the recovery resumes after it stops;
  * a link flap while r1rt runs (systemd's unit only): no ethtool until it stops, then the deferred EEE off;
  * v2's `r1mc2 serve` (v2's own stand-down kept, now also without ethtool), r1guard alone (r1rt died), pgrep failing
    (counted as running);
  * `--status` (the runbook's read-only check).

Nothing here touches this machine's network; the guard only ever runs the fakes (PATH) against the fake root."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1].parent
GUARD = REPO / "deploy" / "eth10" / "r1mc3-eth10-guard.sh"
V2_GUARD = REPO / "tests" / "assets" / "eth10" / "v2_eth10_guard.sh"
BASH = shutil.which("bash") or "/bin/bash"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash scripts")

FAKELIB = r"""
import json, os, re, signal, sys
ROOT = os.environ["FAKE_ROOT"]
STATE = os.path.join(ROOT, "state.json")
ACTIONS = os.path.join(ROOT, "actions.jsonl")

def load():
    with open(STATE) as f:
        return json.load(f)

def save(s):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f)
    os.replace(tmp, STATE)

def act(s, cmd, args):
    with open(ACTIONS, "a") as f:
        f.write(json.dumps({"t": round(s["t"], 3), "cmd": cmd, "args": args}) + "\n")

def write_carrier(s):
    eff = 0 if s.get("admin_down") else int(s["carrier"])
    p = os.path.join(ROOT, "sys", "class", "net", "eth10")
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "carrier"), "w") as f:
        f.write(f"{eff}\n")

def advance(s, dt):
    s["t"] += dt
    for ev in sorted(s["events"], key=lambda e: e[0]):
        t, key, val = ev
        if t <= s["t"] and not ev in s["applied"]:
            s["applied"].append(ev)
            if key == "proc+":
                s["procs"].append(val)
            elif key == "proc-":
                s["procs"] = [p for p in s["procs"] if p != val]
            elif key == "unit":
                s["units"][val[0]] = val[1]
            else:
                s[key] = val
    write_carrier(s)
    save(s)
    if s["t"] >= s["end"]:
        # the scenario is over: end the guard (the fake's parent is the guard's bash)
        os.kill(os.getppid(), signal.SIGTERM)
        sys.exit(0)
"""

FAKES: dict[str, str] = {
    "sleep": """
s = load(); advance(s, float(sys.argv[1]))
""",
    "ping": """
s = load()
act(s, "ping", sys.argv[1:])
ok = (not s.get("admin_down")) and int(s["carrier"]) == 1 and s["pc"] in ("1", 1)
if ok:
    sys.exit(0)
advance(s, 1.0)   # -W1: no answer within 1 s
sys.exit(1)
""",
    "ip": """
s = load()
act(s, "ip", sys.argv[1:])
if sys.argv[1:4] == ["link", "set", "eth10"]:
    if sys.argv[4] == "down":
        s["admin_down"] = True
    else:
        s["admin_down"] = False
        if s["pc"] == "until_reset":
            s["pc"] = "1"   # 5 Oct: turning the port off and on brought the robot back
    s["resets"] = s.get("resets", 0) + (1 if sys.argv[4] == "up" else 0)
    write_carrier(s); save(s)
""",
    "ethtool": """
s = load(); act(s, "ethtool", sys.argv[1:])
""",
    "logger": """
s = load()
args = sys.argv[1:]
if args[:1] == ["-t"]:
    args = args[2:]
act(s, "log", [" ".join(args)])
""",
    "pgrep": """
s = load()
act(s, "pgrep", sys.argv[1:])
if s.get("pgrep_error"):
    sys.exit(2)
a = sys.argv[1:]
if a[0] == "-x":
    sys.exit(0 if a[1] in s["procs"] else 1)
if a[0] == "-f":
    pat = re.compile(a[1])
    sys.exit(0 if any(pat.search(p) for p in s["procs"]) else 1)
sys.exit(1)
""",
    "systemctl": """
s = load()
a = [x for x in sys.argv[1:] if x != "--quiet"]
if a[0] == "is-active":
    sys.exit(0 if s["units"].get(a[1]) == "active" else 3)
act(s, "systemctl", sys.argv[1:])
""",
}


def _world(tmp: Path, *, end: float, events: list[list[Any]], carrier: int = 1, pc: str = "1", **extra: Any) -> Path:
    root = tmp / "root"
    (root / "sys" / "class" / "net" / "eth10").mkdir(parents=True)
    bin_ = tmp / "bin"
    bin_.mkdir()
    (bin_ / "fakelib.py").write_text(FAKELIB)
    for name, body in FAKES.items():
        p = bin_ / name
        p.write_text(
            f"#!{sys.executable} -S\nimport sys\nsys.path.insert(0, {str(bin_)!r})\nfrom fakelib import *\n{body}"
        )
        p.chmod(0o755)
    state: dict[str, Any] = {"t": 0.0, "end": end, "events": events, "applied": [], "carrier": carrier, "pc": pc,
                             "procs": [], "units": {}, "admin_down": False}  # fmt: skip
    state.update(extra)
    # events at t <= 0 apply before the guard starts
    for ev in [e for e in events if e[0] <= 0]:
        state["applied"].append(ev)
        k, v = ev[1], ev[2]
        if k == "proc+":
            state["procs"].append(v)
        elif k == "unit":
            state["units"][v[0]] = v[1]
        else:
            state[k] = v
    (root / "state.json").write_text(json.dumps(state))
    (root / "sys" / "class" / "net" / "eth10" / "carrier").write_text(f"{carrier}\n")
    return root


def _run(script: Path, root: Path, *args: str) -> list[dict[str, Any]]:
    env = {
        "PATH": f"{root.parent / 'bin'}:/usr/bin:/bin",
        "FAKE_ROOT": str(root),
        "LC_ALL": "C",
    }
    subprocess.run([BASH, str(script), *args], env=env, capture_output=True, text=True, timeout=120, check=False)
    lines = (root / "actions.jsonl").read_text().splitlines() if (root / "actions.jsonl").exists() else []
    return [json.loads(x) for x in lines]


def _v2_copy(tmp: Path, root: Path) -> Path:
    """v2's script with its /sys path pointed at the fake root (a temporary copy; v2's file is never changed)."""
    src = V2_GUARD.read_text()
    assert "/sys/class/net/$IF/carrier" in src
    out = tmp / "v2_eth10_guard.sh"
    out.write_text(src.replace("/sys/class/net/$IF/carrier", f"{root}/sys/class/net/$IF/carrier"))
    return out


def _actions(rows: list[dict[str, Any]], *cmds: str) -> list[tuple[float, str, tuple[str, ...]]]:
    return [(r["t"], r["cmd"], tuple(r["args"])) for r in rows if r["cmd"] in cmds]


def _logs(rows: list[dict[str, Any]]) -> list[tuple[float, str]]:
    return [(r["t"], r["args"][0]) for r in rows if r["cmd"] == "log"]


# --------------------------------------------------------------------------------------------------------------------
# no controller: identical to v2

SCENARIOS_NO_CONTROLLER: dict[str, dict[str, Any]] = {
    "silence_then_reset": {"end": 200.0, "events": [[50.0, "pc", "until_reset"]]},
    "link_flap_then_silence": {
        "end": 250.0,
        "events": [[60.0, "carrier", 0], [70.0, "carrier", 1], [120.0, "pc", "until_reset"]],
    },
    "robot_off": {"end": 120.0, "events": [[30.0, "carrier", 0]]},
}


@pytest.mark.parametrize("name", sorted(SCENARIOS_NO_CONTROLLER))
def test_no_controller_does_exactly_what_v2_does(tmp_path: Path, name: str) -> None:
    sc = SCENARIOS_NO_CONTROLLER[name]
    r3 = _world(tmp_path / "v3", **sc)
    rows3 = _run(GUARD, r3, "--root", str(r3))
    r2 = _world(tmp_path / "v2", **sc)
    rows2 = _run(_v2_copy(tmp_path / "v2", r2), r2)
    a3 = _actions(rows3, "ethtool", "ip")
    a2 = _actions(rows2, "ethtool", "ip")
    assert a3 == a2, f"v3 {a3}\nv2 {a2}"
    assert _actions(rows3, "ping") == _actions(rows2, "ping")  # the same pings at the same times
    assert a3[0] == (0.0, "ethtool", ("--set-eee", "eth10", "eee", "off"))  # EEE off at start, as v2
    if name != "robot_off":
        assert ("ip", ("link", "set", "eth10", "down")) in [(c, a) for _, c, a in a3]  # the recovery ran
    # no controller ran: no log-only lines
    assert not [m for _, m in _logs(rows3) if "NOT" in m]


def test_reset_timing_matches_the_5_oct_rule(tmp_path: Path) -> None:
    """Silence from t=50: 20 unanswered pings of 1 s + 1 s sleep each -> the reset ~40 virtual s later; 45 s grace."""
    r = _world(tmp_path, end=150.0, events=[[50.0, "pc", "until_reset"]])
    rows = _run(GUARD, r, "--root", str(r))
    downs = [t for t, c, a in _actions(rows, "ip") if a[-1] == "down"]
    assert len(downs) == 1 and 85.0 <= downs[0] <= 95.0, downs


# --------------------------------------------------------------------------------------------------------------------
# a controller runs: log only, never ethtool


def test_r1rt_running_only_logs_and_never_calls_ethtool(tmp_path: Path) -> None:
    r = _world(tmp_path, end=250.0, events=[[0.0, "proc+", "r1rt"], [50.0, "pc", "until_reset"]])
    rows = _run(GUARD, r, "--root", str(r))
    assert _actions(rows, "ethtool") == []
    assert _actions(rows, "ip") == []
    logs = [m for _, m in _logs(rows)]
    assert any("start: process r1rt - NOT calling ethtool" in m for m in logs), logs
    not_reset = [m for m in logs if "NOT resetting" in m]
    assert len(not_reset) >= 2 and all("process r1rt running" in m for m in not_reset), logs


def test_r1rt_starting_mid_outage_blocks_the_reset_until_it_stops(tmp_path: Path) -> None:
    r = _world(
        tmp_path,
        end=300.0,
        events=[[50.0, "pc", "until_reset"], [55.0, "proc+", "r1rt"], [55.0, "unit", ["r1rt.service", "active"]],
                [200.0, "proc-", "r1rt"], [200.0, "unit", ["r1rt.service", "inactive"]]],
    )  # fmt: skip
    rows = _run(GUARD, r, "--root", str(r))
    eth = _actions(rows, "ethtool")
    assert eth and all(t < 55.0 or t >= 200.0 for t, _, _ in eth), eth
    ips = _actions(rows, "ip")
    assert ips and all(t >= 200.0 for t, _, _ in ips), ips  # the recovery resumes only after r1rt stopped
    logs = _logs(rows)
    assert any(55.0 <= t < 200.0 and "NOT resetting" in m for t, m in logs), logs


def test_link_flap_while_r1rt_runs_defers_eee_off_until_it_stops(tmp_path: Path) -> None:
    """Only systemd says r1rt runs (no process of that name: the unit check alone must hold the guard off)."""
    r = _world(
        tmp_path,
        end=180.0,
        events=[[0.0, "unit", ["r1rt.service", "active"]], [50.0, "carrier", 0], [60.0, "carrier", 1],
                [150.0, "unit", ["r1rt.service", "inactive"]]],
    )  # fmt: skip
    rows = _run(GUARD, r, "--root", str(r))
    eth = _actions(rows, "ethtool")
    assert eth, "the deferred EEE off never ran"
    assert all(t >= 150.0 for t, _, _ in eth), eth
    assert len(eth) == 1  # once: the deferred one
    logs = [m for _, m in _logs(rows)]
    assert any("link DOWN while r1rt.service active runs" in m for m in logs), logs
    assert any("link up: r1rt.service active - NOT calling ethtool" in m for m in logs), logs
    assert any("the deferred EEE off" in m for m in logs), logs


@pytest.mark.parametrize(
    ("procs", "extra", "why"),
    [
        (["r1mc2 serve --port 8771"], {}, "v2 r1mc2 serve"),
        (["r1guard"], {}, "process r1guard"),
        ([], {"pgrep_error": True}, "pgrep failed"),
    ],
)
def test_other_controllers_and_failed_checks_only_log(
    tmp_path: Path, procs: list[str], extra: dict[str, Any], why: str
) -> None:
    r = _world(tmp_path, end=120.0, events=[[0.0, "proc+", p] for p in procs] + [[30.0, "pc", "until_reset"]], **extra)
    rows = _run(GUARD, r, "--root", str(r))
    assert _actions(rows, "ethtool", "ip") == []
    assert any("NOT resetting" in m and why in m for _, m in _logs(rows)), _logs(rows)


def test_the_stand_sessions_core_unit_counts_as_running(tmp_path: Path) -> None:
    """Session 6: HW1 runs r1rt as r1rt-stand.service (`--stand-session`): systemd's state alone holds the guard off."""
    r = _world(
        tmp_path, end=120.0, events=[[0.0, "unit", ["r1rt-stand.service", "active"]], [30.0, "pc", "until_reset"]]
    )
    rows = _run(GUARD, r, "--root", str(r))
    assert _actions(rows, "ethtool", "ip") == []
    assert any("NOT resetting" in m and "r1rt-stand.service active" in m for _, m in _logs(rows)), _logs(rows)


def test_status_prints_the_mode_and_acts_on_nothing(tmp_path: Path) -> None:
    r = _world(tmp_path, end=10.0, events=[])
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "FAKE_ROOT": str(r), "LC_ALL": "C"}
    out = subprocess.run([BASH, str(GUARD), "--root", str(r), "--status"], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and "carrier 1" in out.stdout and "v2 recovery (no controller runs)" in out.stdout
    st = json.loads((r / "state.json").read_text())
    st["procs"] = ["r1rt"]
    (r / "state.json").write_text(json.dumps(st))
    out = subprocess.run([BASH, str(GUARD), "--root", str(r), "--status"], env=env, capture_output=True, text=True)
    assert "LOG ONLY (process r1rt): no reset, no ethtool" in out.stdout, out.stdout
    rows = (
        [json.loads(x) for x in (r / "actions.jsonl").read_text().splitlines()]
        if (r / "actions.jsonl").exists()
        else []
    )
    assert _actions(rows, "ethtool", "ip", "ping") == []


def test_the_script_is_valid_bash_and_points_at_the_installed_path() -> None:
    assert subprocess.run([BASH, "-n", str(GUARD)], capture_output=True).returncode == 0
    unit = (GUARD.parent / "r1mc3-eth10-guard.service").read_text()
    assert "ExecStart=/bin/bash /usr/local/bin/r1mc3-eth10-guard.sh" in unit
    assert "Conflicts=r1-eth10-guard.service" in unit
    assert os.access(GUARD, os.X_OK)


def test_v2s_guard_does_what_u8_forbids_while_r1rt_runs(tmp_path: Path) -> None:
    """The contrast (why U8 exists): v2's own script, with r1rt running, calls ethtool at start and resets eth10 after
    20 s of silence - it stands down only for `r1mc2 serve`. The v3-aware guard does neither (test above)."""
    r = _world(tmp_path, end=150.0, events=[[0.0, "proc+", "r1rt"], [50.0, "pc", "until_reset"]])
    rows = _run(_v2_copy(tmp_path, r), r)
    assert _actions(rows, "ethtool"), "v2 calls ethtool while r1rt runs"
    assert [a for _, _, a in _actions(rows, "ip")], "v2 resets eth10 while r1rt runs"
