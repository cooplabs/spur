"""The C++ policy runner, through its Python binding, keeps the R1 standing in lockstep in both profiles (train with
mjlab's <position> actuators, real with the board block), and its trajectory stays with the pure-Python reference
runner's: the networks differ by <= ~5e-14 per call (tests/unit/test_cpp_policy_binding.py), and over a 10 s stand
that must not grow beyond 1e-9 rad in any joint. Skipped when r1rt_py is not built (scripts/ci.sh cpp builds it)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from r1mc3.policy import POLICY_DIR, ReferencePolicy
from r1sim import cpp_policy, world
from r1sim.control import PolicyController
from r1sim.driver import driver_for_profile
from r1sim.engine import Simulation

from tests.integration.test_engine import _report

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")

SECONDS = 10.0
SAMPLE_S = 0.1
CONTROL_DT = {"train": 0.005, "real": 0.002}


def _tilt_deg(quat: np.ndarray) -> float:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(quat, dtype=np.float64))
    return float(np.degrees(np.arccos(np.clip(r[8], -1.0, 1.0))))


def _rollout(w: world.World, policy: ReferencePolicy) -> tuple[np.ndarray, float, float]:
    """Joint positions (26, SDK order) every SAMPLE_S for SECONDS; the worst tilt; the final pelvis height."""
    sim = Simulation(w.model, w.index, driver=driver_for_profile(w.profile))
    sim.reset(w.initial_qpos())
    sim.add_controller(PolicyController(policy), CONTROL_DT[w.profile], "policy")
    qs, worst = [], 0.0
    for _ in range(round(SECONDS / SAMPLE_S)):
        sim.run_for(SAMPLE_S)
        s = sim.read_state()
        qs.append(s.q.copy())
        worst = max(worst, _tilt_deg(s.base_quat))
    return np.array(qs), worst, float(sim.read_state().base_pos[2])


@pytest.mark.parametrize("profile", ["train", "real"])
def test_cpp_policy_stands_and_tracks_the_python_runner(profile: str) -> None:
    w = world.build_world(profile, visual=False, with_scene=False)
    q_cc, tilt_cc, z_cc = _rollout(w, cpp_policy.load_cpp_policy(POLICY_DIR))
    q_py, tilt_py, _ = _rollout(w, ReferencePolicy.load(POLICY_DIR))
    diff = float(np.abs(q_cc - q_py).max())
    _report(
        f"cpp_policy_stand_{profile}",
        {"max_joint_diff_rad": diff, "tilt_max_cpp_deg": round(tilt_cc, 3), "tilt_max_py_deg": round(tilt_py, 3)},
    )
    assert tilt_cc < 3.0 and z_cc > 0.6  # stands
    assert diff <= 1e-9
