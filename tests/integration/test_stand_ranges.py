"""The S1 stand fixture holds the robot at the corners of the parameter ranges and over seeded variants (session 2,
item 1b). The fixture balances nothing: each variant gets its own CoM-centred stand pose. The policy version of this
test comes with the policy runner (integration)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from r1sim import joints as J
from r1sim import model as M
from r1sim import variants, world
from r1sim.control import StandHold
from r1sim.engine import Simulation

from tests.integration.test_engine import _report, _tilt_deg

SECONDS = 3.0
N_SEEDS = 10


def _stand(w: world.World) -> dict[str, float]:
    ctrl = StandHold.for_model(w.model)
    sim = Simulation(w.model, w.index)
    sim.reset(w.initial_qpos(ctrl.pose))
    sim.add_controller(ctrl, 0.002, "stand")
    s0 = sim.read_state()
    feet0 = np.array([sim.d.site_xpos[f][:2] for f in w.index.foot_sites])
    sim.run_for(SECONDS)
    s = sim.read_state()
    feet = np.array([sim.d.site_xpos[f][:2] for f in w.index.foot_sites])
    return {
        "mass_kg": round(w.index.total_mass, 3),
        "ankle_pitch": round(float(ctrl.pose[J.IDX["left_ankle_pitch"]]), 4),
        "tilt_change_deg": round(_tilt_deg(s.base_quat) - _tilt_deg(s0.base_quat), 3),
        "pelvis_dz_m": round(float(s.base_pos[2] - s0.base_pos[2]), 4),
        "feet_slip_mm": round(1000.0 * float(np.linalg.norm(feet - feet0, axis=1).max()), 2),
    }


def _world(profile: str, *, seed: int | None = None, corner: str | None = None) -> world.World:
    if corner is None:
        return world.build_world(profile, visual=False, seed=seed)
    m = M.build_model(profile, visual=False, params=variants.corner_params(profile, corner))
    return world.World(profile=profile, model=m, index=M.RobotIndex(m), scene=None)


def _assert_stands(r: dict[str, float]) -> None:
    assert abs(r["tilt_change_deg"]) < 5.0
    assert abs(r["pelvis_dz_m"]) < 0.03


# The training randomisation's heavy-front corner (torso +3 kg with its COM 5 cm forward / left / up, +0.5 kg per hand)
# creeps forward and tips under the balance-free fixture (s2, 10 Oct: tilt 1.4 -> 25 deg in 3 s; with foot friction
# 1.6 still 16 deg). Every other corner and seed stands. Holding that corner is the policy's job (it was trained on
# it): the policy version of this test must include it. Reported here, not asserted.
FIXTURE_LIMIT = {("train", "heavy_front")}


@pytest.mark.parametrize("profile", M.PROFILES)
def test_fixture_stands_at_the_range_corners(profile: str) -> None:
    t0 = time.perf_counter()
    rows = {c: _stand(_world(profile, corner=c)) for c in variants.CORNERS}
    _report(f"stand_ranges_corners_{profile}", {**rows, "wall_s": round(time.perf_counter() - t0, 1)})
    for c, r in rows.items():
        if (profile, c) not in FIXTURE_LIMIT:
            _assert_stands(r)


@pytest.mark.parametrize("profile", M.PROFILES)
def test_fixture_stands_over_seeded_variants(profile: str) -> None:
    """N seeds, 3 s each. real: robot variants within the estimate ranges (the empty floor); train: the training's
    randomisation of the model, plus the same seed's scene variant (scene draws do not depend on the profile)."""
    t0 = time.perf_counter()
    rows = {f"seed_{s}": _stand(_world(profile, seed=s)) for s in range(N_SEEDS)}
    tilts = [r["tilt_change_deg"] for r in rows.values()]
    _report(
        f"stand_ranges_seeds_{profile}",
        {**rows, "tilt_min_max_deg": (min(tilts), max(tilts)), "wall_s": round(time.perf_counter() - t0, 1)},
    )
    for r in rows.values():
        _assert_stands(r)
