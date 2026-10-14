"""The HW1 recording tools through r1d against the REAL core in lockstep (python/r1mc3/record; real profile on the
floor).

The recordings run on ON_STAND on the robot. Until the core's ON_STAND is merged, the rehearsal runs the SAME runner on
an arm in BALANCE (the plan's required mode switched by the test only): the commanded target follows the step and the
chirp exactly, every tick's record is accepted, the run ends with a HOLD at the last setpoint, a stop aborts it (the
joint decelerates to HOLD), and the manifest says what moved, when and how it ended.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import r1mc3.record.runner as runner_mod
from r1mc3.record.plans import plan as real_plan
from r1mc3.record.runner import Recorder
from r1sim import cpp_policy, world
from r1sim.v3_core import GROUP_JOINTS, SOURCE_HOLD, SOURCE_STREAM

from tests.integration.test_service_lockstep import COL, Rig

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
ELBOW = 17  # left_elbow, SDK order
K = GROUP_JOINTS[0].index(ELBOW)


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


@pytest.fixture
def in_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    def plan_in_balance(*a: Any, **kw: Any) -> Any:
        p = real_plan(*a, **kw)
        p.needs_mode = "balance"
        return p

    monkeypatch.setattr(runner_mod, "plan", plan_in_balance)


def _rig(w: world.World, tmp_path: Path) -> Rig:
    rig = Rig(w)
    rig.svc.recorder = Recorder(rig.svc, out_dir=tmp_path)
    return rig


def test_step_recording_follows_and_writes_its_manifest(w: world.World, tmp_path: Path, in_balance: None) -> None:
    rig = _rig(w, tmp_path)
    res: dict[str, Any] = {}
    params = {"joint": "left_elbow", "amplitude_rad": 0.05, "hold_s": 0.5, "repeats": 1}

    def go() -> None:
        res["base"] = rig.ctl.last_q[ELBOW]
        res["dry"] = rig.svc.submit({"op": "record.plan", "kind": "step", "params": params}).result
        res["run"] = rig.svc.submit({"op": "record.run", "kind": "step", "params": params, "dry_run": False}).result
        res["busy"] = rig.svc.submit({"op": "record.run", "kind": "step", "params": params, "dry_run": False}).result

    rig.at(1.0, go)
    tr = rig.run(4.2)
    assert res["dry"].ok and res["dry"].data["dry_run"] and not res["dry"].moved
    assert res["run"].ok and res["run"].moved and res["busy"].code == "RESOURCE_BUSY"
    col = COL["q"] + ELBOW - 14
    base = res["base"]
    for t0, off in ((1.2, 0.0), (1.7, 0.05), (2.2, 0.0), (2.7, -0.05), (3.2, 0.0)):  # mid-hold windows
        rows = tr[(tr[:, 0] >= t0) & (tr[:, 0] < t0 + 0.1)]
        assert np.max(np.abs(rows[:, col] - (base + off))) < 1e-9, (t0, off)
    assert np.all(tr[tr[:, 0] > 3.55][:, COL["src"]] == SOURCE_HOLD)  # the run ended with a HOLD
    (m,) = list(tmp_path.glob("*_step_left_elbow.json"))
    doc = json.loads(m.read_text())
    assert doc["status"] == "done" and doc["plan"]["moves"] == ["left_elbow"] and doc["refused"] == 0
    assert doc["pushes"] >= 245 and doc["t_end_mono_ns"] - doc["t_start_mono_ns"] == pytest.approx(2.5e9, abs=2e7)
    assert int(rig.ctl.core.status()["upper_rejected"]) == 0


def test_chirp_recording_and_a_stop_aborts_it(w: world.World, tmp_path: Path, in_balance: None) -> None:
    rig = _rig(w, tmp_path)
    res: dict[str, Any] = {}
    params = {"joint": "left_elbow", "amplitude_rad": 0.03, "f0_hz": 0.5, "f1_hz": 3.0, "duration_s": 3.0}

    def go() -> None:
        res["base"] = rig.ctl.last_q[ELBOW]
        res["t0"] = rig.t_ns * 1e-9
        res["run"] = rig.svc.submit({"op": "record.run", "kind": "chirp", "params": params, "dry_run": False}).result

    rig.at(1.0, go)
    rig.at(2.5, lambda: res.__setitem__("stop", rig.svc.submit({"op": "stop"}).result))
    tr = rig.run(3.2)
    assert res["run"].ok and res["stop"].ok
    p = real_plan("chirp", params, np.zeros(26))
    col = COL["q"] + ELBOW - 14
    rows = tr[(tr[:, 0] >= 1.2) & (tr[:, 0] < 2.4)]
    # the core's target = the setpoint of r1d's latest tick (10 ms steps; the executor holds a STREAM setpoint)
    want = np.array([res["base"] + p.at(np.floor((t - res["t0"]) * 100 + 1e-6) / 100)[0][K] for t in rows[:, 0]])
    assert np.max(np.abs(rows[:, col] - want)) < 1e-6
    assert np.all(rows[:, COL["src"]] == SOURCE_STREAM)
    assert np.all(tr[tr[:, 0] > 2.75][:, COL["src"]] == SOURCE_HOLD)  # the stop: decelerated to HOLD
    (m,) = list(tmp_path.glob("*_chirp_left_elbow.json"))
    doc = json.loads(m.read_text())
    assert doc["status"] == "aborted" and doc["reason"] == "stop"
