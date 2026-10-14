"""The examples (examples/README.md) run as a user runs them and do what they say: the pick-and-place moves the box
where it was asked (lockstep, and the real processes on the DDS-bridge simulator), the jog's fixed sequence moves every
joint it names, the robustness batch's quick run balances; their world-model entities match the simulator's scene."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import mujoco
import numpy as np
import pytest
from r1sim import cpp_policy, scenes

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
needs_core = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")


def _load(name: str) -> ModuleType:
    """An example script as a module (examples/ is not a package: the scripts import `common` from beside them)."""
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(f"r1mc3_example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # (dataclasses look their module up there)
    spec.loader.exec_module(mod)
    return mod


def test_the_examples_declare_what_the_simulator_holds() -> None:
    """The entities the examples give r1d's world model are the "table" scene's own (r1sim.scenes): the table's top
    face at the table's height, the box where the scene puts it, with its MuJoCo half sizes."""
    C = _load("common")
    info = scenes.add_table(mujoco.MjSpec())
    t = C.table_entity()
    assert t["kind"] == "surface" and t["pose"]["pos"][2] == pytest.approx(info.table_top_z)
    box = info.object("box")
    e = C.object_entity("box", np.asarray(box.pos))
    assert e["kind"] == "object" and e["shape"] == "box" and e["size"] == pytest.approx(list(box.size))
    assert e["pose"]["pos"] == pytest.approx(list(box.pos), abs=1e-4)
    readme = (EXAMPLES / "README.md").read_text()
    for script in sorted(EXAMPLES.glob("*.py")):
        if script.name != "common.py":
            assert script.name in readme, f"examples/README.md does not mention {script.name}"


@needs_core
@pytest.mark.slow
def test_pick_and_place_lockstep(tmp_path: Path) -> None:
    pp = _load("pick_and_place")
    out = tmp_path / "pp.json"
    assert pp.main(["--lockstep", "--json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["ok"] and res["failed_at"] is None, res
    assert [k for k in res["steps"]] == ["declare", "preshape", "hover", "lower", "grasp", "carry", "release",
                                         "retreat"]  # fmt: skip
    assert res["grip_before_release"] == "HOLDING" and res["held_above_table_m"] > 0.005  # carried, not dragged
    assert res["error_m"] <= pp.PLACE_TOL_M and res["on_table"]


@needs_core
@pytest.mark.slow
def test_jog_script_lockstep() -> None:
    assert _load("jog").main(["--lockstep", "--script"]) == 0


@needs_core
@pytest.mark.slow
def test_robustness_batch_quick(tmp_path: Path) -> None:
    out = tmp_path / "rb.json"
    assert _load("robustness_batch").main(["--quick", "--workers", "1", "--json", str(out)]) == 0
    rows = json.loads(out.read_text())
    assert len(rows) == 2 and not any(r["fallen"] for r in rows), rows
    assert all(r["mode_end"] == "BALANCE" and r["tilt_max_deg"] < 5.0 for r in rows), rows


@pytest.mark.needs_dds
@pytest.mark.realtime
@pytest.mark.slow
def test_pick_and_place_on_the_real_processes(tmp_path: Path) -> None:
    from r1sim import stack

    if not stack.available():
        pytest.skip("the process stack needs CycloneDDS's Python binding and the DDS build of r1rt / r1guard")
    pp = _load("pick_and_place")
    out = tmp_path / "pp.json"
    rc = pp.main(["--domain", "85", "--json", str(out)])
    res = json.loads(out.read_text())
    assert rc == 0 and res["ok"], res
    assert res["mode"] == "processes" and res["grip_before_release"] == "HOLDING"
