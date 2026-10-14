"""deploy/bin/r1mc3_motorcheck.py: the one rule every stop / switch / reboot path uses (DESIGN.md §2.1: refused while
we own the motors; shutdown only from DAMP or MONITOR), its heartbeat decoding against r1log's layout, and the
shutdown inhibitor's lock / unlock decision."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

BIN = Path(__file__).resolve().parents[2] / "deploy" / "bin"


def _load(name: str) -> ModuleType:
    sys.path.insert(0, str(BIN))
    spec = importlib.util.spec_from_file_location(name, BIN / f"{name}.py")
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mc = _load("r1mc3_motorcheck")
NOW = 10_000_000_000


def _hb(mode: str, age_ms: float = 1.0) -> dict[str, object]:
    return {"mode": mode, "t_ns": NOW - int(age_ms * 1e6)}


@pytest.mark.parametrize(
    ("state", "status"),
    [
        ({"segment": False}, 0),
        ({"segment": True, "token_owner": "none"}, 0),
        ({"segment": True, "token_owner": "r1rt", "rt": _hb("MONITOR")}, 0),
        ({"segment": True, "token_owner": "r1rt", "rt": _hb("DAMP")}, 0),
        *[({"segment": True, "token_owner": "r1rt", "rt": _hb(m)}, 3)
          for m in ("TAKEOVER_MIRROR", "TAKEOVER_BLEND", "BALANCE", "STAND_FREEZE", "ON_STAND", "OVERLAY_ARM_SDK")],
        ({"segment": True, "token_owner": "r1rt", "rt": _hb("MONITOR", age_ms=150)}, 3),
        ({"segment": True, "token_owner": "r1rt", "rt": None}, 3),
        ({"segment": True, "token_owner": "r1guard", "guard": _hb("GUARDED")}, 3),
        ({"segment": True, "token_owner": "unknown(7)"}, 3),
        ({"segment": True, "token_owner": "r1guard", "processes_alive": False}, 0),
        ({"segment": True, "token_owner": "r1guard", "processes_alive": None}, 3),
    ],
)  # fmt: skip
def test_decide(state: dict[str, object], status: int) -> None:
    got, reason = mc.decide({"now_ns": NOW, **state})
    assert got == status, reason


def test_heartbeat_layout_matches_r1logs() -> None:
    from r1mc3.log import records

    dt = records.HEARTBEAT
    assert dt.itemsize == mc.HEARTBEAT_SIZE == struct.calcsize(mc.HEARTBEAT_FMT)
    rec = __import__("numpy").zeros(1, dtype=dt)
    rec["seq"], rec["t_ns"], rec["mode"], rec["faults"], rec["rt_granted"] = 7, 123456789, 3, 0x10, 1
    rec["blend_progress"] = 0.5
    hb = mc.decode_heartbeat(rec.tobytes())
    assert (hb["seq"], hb["t_ns"], hb["mode"], hb["faults"], hb["rt_granted"]) == (7, 123456789, "BALANCE", 0x10, 1)
    assert hb["blend_progress"] == 0.5


def test_modes_follow_types_h() -> None:
    types_h = (Path(__file__).resolve().parents[2] / "rt" / "include" / "r1rt" / "types.h").read_text()
    for code, name in mc.MODES.items():
        assert f"R1RT_MODE_{name} = {code}" in types_h.replace("  ", " ") or f"R1RT_MODE_{name} = {code}," in types_h


def test_cli_json_line_and_exit_status(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "dev" / "shm").mkdir(parents=True)
    (root / "dev" / "shm" / "r1mc3.rt").write_bytes(b"\0")
    (root / "run" / "r1mc3").mkdir(parents=True)
    (root / "run" / "r1mc3" / "motorcheck_state.json").write_text(
        json.dumps({"token_owner": "r1rt", "rt": {"mode": "BALANCE", "age_ms": 2}})
    )
    r = subprocess.run([sys.executable, "-I", str(BIN / "r1mc3_motorcheck.py"), "--root", str(root)],
                       capture_output=True, text=True, check=False)  # fmt: skip
    assert r.returncode == 3
    out = json.loads(r.stdout)
    assert out["ok"] is False and out["owner"] == "r1rt" and out["rt_mode"] == "BALANCE"


def test_the_inhibitor_locks_only_while_owned(tmp_path: Path) -> None:
    root = tmp_path / "root"
    st = root / "run" / "r1mc3" / "motorcheck_state.json"
    st.parent.mkdir(parents=True)
    (root / "dev" / "shm").mkdir(parents=True)
    (root / "dev" / "shm" / "r1mc3.rt").write_bytes(b"\0")
    cmd = [sys.executable, "-I", str(BIN / "r1mc3_inhibit.py"), "--root", str(root), "--print-only", "--cycles", "1",
           "--period-s", "0"]  # fmt: skip
    st.write_text(json.dumps({"token_owner": "r1rt", "rt": {"mode": "ON_STAND", "age_ms": 1}}))
    assert subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split() == ["lock"]
    st.write_text(json.dumps({"token_owner": "r1rt", "rt": {"mode": "MONITOR", "age_ms": 1}}))
    assert subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split() == []
    inh = _load("r1mc3_inhibit")
    assert inh.inhibit_cmd("x")[:5] == ["systemd-inhibit", "--what=shutdown:sleep:idle", "--who=r1mc3", "--why=x",
                                         "--mode=block"]  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Session 6 (found by the rehearsal with real processes): the tool did `int(view.token())`, but the binding returns
# (owner, generation) - on a LIVE segment it answered "cannot tell" (exit 2 = refused) in every mode, so every ExecStop
# / deploy / `r1 shutdown` check would have refused at the session end on the robot. These read REAL segments.


def _binding() -> ModuleType:
    from r1sim import cpp_policy

    if cpp_policy.find_module() is None:
        pytest.skip("r1rt_py not built (scripts/ci.sh cpp)")
    return cpp_policy.import_r1rt_py()


def test_owner_of_accepts_the_tuple_and_the_old_word() -> None:
    assert mc.owner_of((1, 7)) == 1 and mc.owner_of([2, 3]) == 2
    assert mc.owner_of((5 << 8) | 2) == 2 and mc.owner_of(0) == 0


def test_a_live_named_segment_in_monitor_is_not_owned(capsys: pytest.CaptureFixture[str]) -> None:
    """The real path: `main()` on the real root attaches by name (r1rt's created segment: the token free, as r1rt in
    MONITOR) -> exit 0. On the old code: exit 2 (int() of a tuple)."""
    mod = _binding()
    import os

    name = f"/r1mc3.ops.{os.getpid()}"
    seg = mod.SegmentView.create(name, True)
    try:
        status = mc.main(["--shm-name", name])
        out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    finally:
        seg.close()  # created here: the name is unlinked on close
    assert status == 0, out
    assert out["owner"] == "none" and out["ok"] is True


def test_a_live_core_that_owns_the_motors_is_refused_and_its_damp_is_allowed() -> None:
    """A lockstep core (the real C++ core and its segment, through the binding's view): in BALANCE it owns the token
    -> refused (3); after the operator's DAMP -> allowed (0, DESIGN §2.1: shutdown from DAMP / MONITOR)."""
    _binding()
    from r1sim import joints as J
    from r1sim import world
    from r1sim.engine import Simulation
    from r1sim.v3_core import REQ_DAMP, V3CoreController

    w = world.build_world("real", visual=False)
    sim = Simulation(w.model, w.index)
    sim.reset(w.initial_qpos(J.POSE_STAND.copy()))
    ctl = V3CoreController(w.model)
    sim.add_controller(ctl, ctl.period_s, "v3")
    sim.run_for(0.2)
    view = ctl.mod.SegmentView.of_core(ctl.core)
    t_ns = round(sim.time * 1e9)
    st = mc.state_from_view(view, now_ns=t_ns, alive=True)
    assert st["token_owner"] == "r1rt" and st["rt"]["mode"] == "BALANCE", st
    assert mc.decide(st)[0] == 3
    requester = ctl.mod.requesters()["operator"]
    seq = view.request_mode(REQ_DAMP, requester, t_ns)
    sim.run_for(0.05)
    assert view.mode_reply(seq)[0] == 0
    st = mc.state_from_view(view, now_ns=round(sim.time * 1e9), alive=True)
    assert st["rt"]["mode"] == "DAMP", st
    assert mc.decide(st)[0] == 0


# --------------------------------------------------------------------------------------------------------------------
# Session 8 (HW1 fix #1): in the REAL install the units run `python -I <release>/deploy/bin/r1mc3_motorcheck.py`; the
# binding sits in <release>/build/rt (install.sh builds it there), on no sys.path - the bare `import r1rt_py` failed,
# the check answered "cannot tell" (exit 2 = refused) and every stop was refused; the robot needed a .pth by hand.


def test_the_check_finds_the_releases_own_binding_in_an_installed_layout(tmp_path: Path) -> None:
    """A fake installed release (deploy/bin/<the check> + build/rt/<the binding>) and a LIVE segment in MONITOR (the
    token free): the check, run exactly as the unit runs it (a separate `python -I`, no PYTHONPATH, no
    R1MC3_BUILD_DIR) -> exit 0. On bad5a5c: exit 2, "cannot read the segment: No module named 'r1rt_py'"."""
    import os
    import shutil

    from r1sim import cpp_policy

    built = cpp_policy.find_module()
    if built is None:
        pytest.skip("r1rt_py not built (scripts/ci.sh cpp)")
    mod = _binding()
    rel = tmp_path / "opt" / "r1mc3" / "releases" / "0.1.0-test"
    (rel / "deploy" / "bin").mkdir(parents=True)
    (rel / "build" / "rt").mkdir(parents=True)
    shutil.copy2(BIN / "r1mc3_motorcheck.py", rel / "deploy" / "bin" / "r1mc3_motorcheck.py")
    shutil.copy2(built, rel / "build" / "rt" / Path(built).name)
    name = f"/r1mc3.dep.{os.getpid()}"
    seg = mod.SegmentView.create(name, True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "R1MC3_BUILD_DIR")}
    try:
        r = subprocess.run(
            [sys.executable, "-I", str(rel / "deploy" / "bin" / "r1mc3_motorcheck.py"), "--shm-name", name],
            capture_output=True, text=True, check=False, env=env, cwd=tmp_path,
        )  # fmt: skip
    finally:
        seg.close()
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert r.returncode == 0, out
    assert out["owner"] == "none" and out["ok"] is True


def test_binding_dirs_start_at_the_release_next_to_the_script(tmp_path: Path) -> None:
    script = tmp_path / "rel" / "deploy" / "bin" / "r1mc3_motorcheck.py"
    assert mc.binding_dirs(script)[0] == (tmp_path / "rel" / "build" / "rt").resolve()
