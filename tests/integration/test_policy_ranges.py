"""The balance policy holds the robot at the corners of the parameter ranges and over seeded variants (session 2,
items 1b + 5; the policy version of test_stand_ranges.py).

Unlike the S1 fixture, the policy starts from its own default pose (joints.POSE_STAND), the pose it was trained to
balance around. The training randomisation's heavy-front corner, which tips the balance-free fixture
(test_stand_ranges.FIXTURE_LIMIT), is included here and must stand: the policy was trained on it. Each profile gets
its own driver (train: mjlab's <position> actuators; real: the board block)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from r1mc3.policy import POLICY_DIR, ReferencePolicy
from r1sim import model as M
from r1sim import variants, world
from r1sim.control import PolicyController
from r1sim.driver import driver_for_profile
from r1sim.engine import Simulation

from tests.integration.test_engine import _report, _tilt_deg

SECONDS = 4.0
N_SEEDS = 10
CONTROL_DT = {"train": 0.005, "real": 0.002}  # the train profile's physics step / the v3 writer rate


def _stand(w: world.World, policy: ReferencePolicy) -> dict[str, float]:
    sim = Simulation(w.model, w.index, driver=driver_for_profile(w.profile))
    sim.reset(w.initial_qpos())
    policy.reset()
    sim.add_controller(PolicyController(policy), CONTROL_DT[w.profile], "policy")
    feet0 = np.array([sim.d.site_xpos[f][:2] for f in w.index.foot_sites])
    sim.run_for(SECONDS)
    s = sim.read_state()
    feet = np.array([sim.d.site_xpos[f][:2] for f in w.index.foot_sites])
    return {
        "mass_kg": round(w.index.total_mass, 3),
        "tilt_end_deg": round(_tilt_deg(s.base_quat), 3),
        "pelvis_z_m": round(float(s.base_pos[2]), 4),
        "feet_slip_mm": round(1000.0 * float(np.linalg.norm(feet - feet0, axis=1).max()), 2),
    }


def _world(profile: str, *, seed: int | None = None, corner: str | None = None) -> world.World:
    if corner is None:
        return world.build_world(profile, visual=False, seed=seed)
    m = M.build_model(profile, visual=False, params=variants.corner_params(profile, corner))
    return world.World(profile=profile, model=m, index=M.RobotIndex(m), scene=None)


def _assert_stands(r: dict[str, float]) -> None:
    assert r["tilt_end_deg"] < 5.0, r
    assert r["pelvis_z_m"] > 0.6, r


@pytest.fixture(scope="module")
def policy() -> ReferencePolicy:
    return ReferencePolicy.load(POLICY_DIR)


@pytest.mark.parametrize("profile", M.PROFILES)
def test_policy_stands_at_the_range_corners(profile: str, policy: ReferencePolicy) -> None:
    t0 = time.perf_counter()
    rows = {c: _stand(_world(profile, corner=c), policy) for c in variants.CORNERS}
    _report(f"policy_ranges_corners_{profile}", {**rows, "wall_s": round(time.perf_counter() - t0, 1)})
    for r in rows.values():
        _assert_stands(r)


@pytest.mark.slow
@pytest.mark.parametrize("profile", M.PROFILES)
def test_policy_stands_over_seeded_variants(profile: str, policy: ReferencePolicy) -> None:
    """N seeds: real = robot variants within the estimate ranges (the empty floor); train = the training
    randomisation."""
    t0 = time.perf_counter()
    rows = {f"seed_{s}": _stand(_world(profile, seed=s), policy) for s in range(N_SEEDS)}
    tilts = [r["tilt_end_deg"] for r in rows.values()]
    _report(
        f"policy_ranges_seeds_{profile}",
        {**rows, "tilt_min_max_deg": (min(tilts), max(tilts)), "wall_s": round(time.perf_counter() - t0, 1)},
    )
    for r in rows.values():
        _assert_stands(r)
