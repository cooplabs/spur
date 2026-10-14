"""r1rt::PolicyRunner (C++, through r1rt_py) == r1mc3.policy.ReferencePolicy (= v2's WalkPolicy) on S2's scripted
1,600-tick sequence in float64: every 26-joint command, the observations' consequences (target, action, lean ramp,
phase clock) within 1e-12; and the committed golden file for the C++-only test regenerates bit for bit."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from r1mc3.policy import POLICY_DIR, ReferencePolicy
from r1sim import cpp_policy
from r1sim.v3_core import OBS_TERMS, pack_params, policy_params

from tests.unit.test_policy_reference import _cmd_script, _state

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "assets" / "policy_runner" / "sequence_1600.r1pr"
needs_cpp = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
TOL = 1e-12


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("policy_runner_golden", REPO / "tools" / "policy_runner_golden.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_params_are_the_metadata() -> None:
    pol = ReferencePolicy.load(POLICY_DIR)
    p = policy_params(pol)
    assert p["n_joints"] == 24 and p["n_act"] == 14 and p["has_lean"] == 1 and p["has_waist"] == 0
    names = ["base_ang_vel", "projected_gravity", "command", "lean", "phase", "joint_pos", "joint_vel", "actions"]
    assert p["obs_terms"] == [OBS_TERMS[n] for n in names]
    assert p["period_s"] == 0.86 and p["step_dt_s"] == 0.02 and (p["lean_lo"], p["lean_hi"]) == (0.0, 0.087)
    assert p["kp"][:6] == [84.302] * 4 + [31.385] * 2
    assert len(pack_params(p)) == 215  # r1sim.v3_core.PARAMS_LAYOUT


def test_golden_file_regenerates_bit_identically() -> None:
    data, info = _tool().build()
    assert info["n_policy"] == 160 and info["n_obs"] == 74
    want = (GOLDEN.parent / "SHA256SUMS").read_text().split()[0]
    assert hashlib.sha256(GOLDEN.read_bytes()).hexdigest() == want
    if hashlib.sha256(data).hexdigest() != want:
        # The committed file was generated with macOS numpy (Accelerate BLAS); another BLAS (Linux / OpenBLAS: session
        # 6, the Linux build check) sums the network's products in another order - the INPUTS (header, params, ticks)
        # must still be byte-identical, the float64 outputs equal within 1e-12 (the C++ runner's own golden test
        # tolerance; measured 2.4e-14 on Ubuntu 20.04 arm64)
        tool = _tool()
        k = tool.HEADER.size + 8 * info["n_params"] + tool.N_TICKS * (16 + 4 * 62)
        gold = GOLDEN.read_bytes()
        assert len(data) == len(gold) and data[:k] == gold[:k]
        out = np.frombuffer(data[k:], "<f8") - np.frombuffer(gold[k:], "<f8")
        assert float(np.max(np.abs(out))) <= 1e-12


@needs_cpp
def test_cpp_runner_matches_the_reference_on_the_s2_sequence() -> None:
    mod = cpp_policy.import_r1rt_py()
    ref = ReferencePolicy.load(POLICY_DIR)
    cpp = mod.PolicyRunner(str(cpp_policy.weights_for(POLICY_DIR / "policy.onnx")), policy_params(ref))
    rng = np.random.default_rng(7)
    t = 0.0
    worst = {"q26": 0.0, "kp": 0.0, "target": 0.0, "action": 0.0, "lean": 0.0, "phase": 0.0}
    for k in range(1600):
        if k == 100:
            assert ref.set_body_targets(lean_deg=3.0) is None and cpp.set_lean_target(np.radians(3.0))
        if k == 1300:
            assert ref.set_body_targets(lean_deg=7.0) is not None and not cpp.set_lean_target(np.radians(7.0))
        s, c = _state(k), _cmd_script(k)
        q_r, kp_r, kd_r = ref.step(s, c, t)
        _ran, q_c, kp_c, kd_c = cpp.step(s["q"], s["dq"], s["quat"], s["gyro"], c, t)
        worst["q26"] = max(worst["q26"], float(np.abs(q_r - q_c).max()))
        worst["kp"] = max(worst["kp"], float(np.abs(kp_r - kp_c).max() + np.abs(kd_r - kd_c).max()))
        worst["target"] = max(worst["target"], float(np.abs(ref.target - cpp.target).max()))
        worst["action"] = max(worst["action"], float(np.abs(ref.last_action - cpp.last_action).max()))
        worst["lean"] = max(worst["lean"], abs(ref.lean - cpp.lean))
        worst["phase"] = max(worst["phase"], abs(ref.phase_t - cpp.phase_t))
        t += 0.002 + float(rng.uniform(-2e-4, 2e-4))
    print("C++ PolicyRunner vs ReferencePolicy, 1,600 ticks:", {k: f"{v:.2e}" for k, v in worst.items()})
    assert ref.policy_steps == cpp.steps > 150
    assert worst["kp"] == 0.0
    assert all(v <= TOL for v in worst.values()), worst
