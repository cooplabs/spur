"""B6 (DESIGN §13.2): the simulator's board model reproduces what the real board did without LowCmd (HW1's board-hold
test).
 Real (tests/assets/hw1/board_hold_500ms_excerpt.npz, the board's 1 kHz tick): 20 / 100 ms pauses move nothing; in the
500 ms pause the board releases at one tick 383.8 ms after the stop and the loaded joints sink at a nearly constant
speed (the hips 0.117 / 0.133 rad/s) = damping. Simulated: the hanging robot on the stand at HW1's attitude
(dds_bridge.HW1_STAND), the board-hold pose and gains, lockstep, the real profile's board (models/driver.yaml). The
targets: the hold (<= 1e-4 rad at 20 / 100 ms), the release within +-10 ms, the hips' sink speed and their motion at 500
ms within +-30 %. (r1sim.b6_board; docs/sim/B6_BOARD.md has the whole table, the knees / shoulders included.)"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from r1sim import b6_board as B
from r1sim.driver import DriverConfig

from tests.integration.test_engine import _report

HIPS = ("left_hip_pitch", "right_hip_pitch")
SUMMARY = Path(__file__).resolve().parents[1] / "assets" / "hw1" / "board_hold_summary.json"


@pytest.fixture(scope="module")
def real() -> B.BoardHoldNumbers:
    return B.board_hold_reference()


@pytest.fixture(scope="module")
def sim500() -> B.BoardHoldNumbers:
    run = B.simulate_board_hold(500.0)
    assert int(run["timeouts"][0]) == 1
    return B.sim_numbers(run, 500.0)


def test_the_real_board_hold_recording(real: B.BoardHoldNumbers) -> None:
    """What the excerpt says (the numbers B6 is held to)."""
    assert abs(real.release_ms - 383.8) < 1.0
    assert real.motion_100ms_rad < 1e-4
    for n in HIPS:
        assert 0.11 <= real.sink_rad_s[n] <= 0.14
    s = json.loads(SUMMARY.read_text())
    assert (
        s["board_hold_20ms"]["max_motion_during_rad"] < 1e-4 and s["board_hold_100ms"]["max_motion_during_rad"] < 1e-4
    )


@pytest.mark.parametrize("pause_ms", [20.0, 100.0])
def test_the_simulated_board_holds_a_short_pause(pause_ms: float) -> None:
    assert B.hold_motion(pause_ms) <= 1e-4


def test_the_simulated_board_releases_and_damps_like_the_real_one(
    real: B.BoardHoldNumbers, sim500: B.BoardHoldNumbers
) -> None:
    rows = {n: dict(real_sink=real.sink_rad_s[n], sim_sink=sim500.sink_rad_s[n], real_end=real.motion_end_rad[n],
                    sim_end=sim500.motion_end_rad[n]) for n in B.COMPARED}  # fmt: skip
    _report("b6_board_hold", dict(real_release_ms=real.release_ms, sim_release_ms=sim500.release_ms, rows=rows))
    assert abs(sim500.release_ms - real.release_ms) <= 10.0
    for n in HIPS:
        assert sim500.sink_rad_s[n] == pytest.approx(real.sink_rad_s[n], rel=0.30), n
        assert sim500.motion_end_rad[n] == pytest.approx(real.motion_end_rad[n], rel=0.30), n


def test_the_board_before_session_8_does_not_reproduce_the_real_one(real: B.BoardHoldNumbers) -> None:
    """The base's board (no timeout: holds the last command forever) - what every bridge / lockstep run assumed before
    HW1: no release, the hips stay put (< 10 % of the real sag)."""
    old = B.sim_numbers(B.simulate_board_hold(500.0, driver=DriverConfig()), 500.0)
    assert old.release_ms != old.release_ms  # NaN: never released
    for n in HIPS:
        assert old.motion_end_rad[n] < 0.1 * real.motion_end_rad[n]
