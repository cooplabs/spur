"""B3's static-torque fit (r1sim.sysid.statics, session 8): the friction band's sign from the approach direction."""

from __future__ import annotations

import numpy as np
import pytest
from r1sim.sysid import statics


def _poses(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-6.0, 6.0, n), rng.choice([-1, 1], n)


def test_recovers_the_mass_scale_and_the_friction_band() -> None:
    g, d = _poses(40, 1)
    tau = 1.05 * g + 0.8 * d + 0.2 + np.random.default_rng(2).normal(0.0, 0.05, g.size)
    f = statics.fit_static(g, tau, d)
    assert f.slope == pytest.approx(1.05, abs=0.02)
    assert f.friction == pytest.approx(0.8, abs=0.05) and f.offset == pytest.approx(0.2, abs=0.05)
    # the model's masses (s = 1) are within 10 % once the band is out (a 5 % mass error)
    assert f.within_tol > 0.9 and f.shown


def test_without_the_direction_the_band_hides_the_masses() -> None:
    """The wrap-up's situation: a ~1 N m band, no sign - the plain fit cannot show +-10 %."""
    g, d = _poses(40, 3)
    g = g / 3.0  # shoulder-sized torques (|g| ~ 0-2 N m)
    tau = g + 0.9 * d
    f0 = statics.fit_static(g, tau, np.zeros_like(d), min_g=0.5)
    assert not f0.shown and np.isnan(f0.friction)
    f1 = statics.fit_static(g, tau, d, min_g=0.5)
    assert f1.friction == pytest.approx(0.9, abs=1e-6) and f1.shown


def test_too_few_poses_is_never_shown() -> None:
    g, d = _poses(10, 4)
    f = statics.fit_static(g, g + 0.5 * d, d)
    assert not f.shown and any("< 20" in n for n in f.notes)


def test_one_direction_only_cannot_split_friction_and_offset() -> None:
    g, _ = _poses(30, 5)
    d = np.ones_like(g)
    f = statics.fit_static(g, g + 0.7 + 0.3 * d, d)
    assert f.offset == 0.0 and f.friction == pytest.approx(1.0, abs=1e-6)
    assert any("not separable" in n for n in f.notes)
