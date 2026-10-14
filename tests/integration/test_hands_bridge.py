"""r1hands minimal (python/r1mc3/hands/dex3.py) against the DDS-bridge simulator (r1sim.dds_bridge, real time, domain
63): the hand state arrives at ~100 Hz with the 9 x 12 taxels, a commanded finger closes in the simulated hand, and
when the command's TTL expires the hand holds where it is (the bridge receives the hold, not the old target)."""

from __future__ import annotations

import time

import numpy as np
import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1mc3.hands import dex3  # noqa: E402

DOMAIN = 63


def test_hands_over_the_bridge() -> None:
    from r1sim.dds_bridge import DdsBridge

    with DdsBridge(domain=DOMAIN, scene="none", builtin="stand") as br:
        hands = dex3.Dex3Hands(DOMAIN, sides=("left",)).start()
        try:
            t_end = time.monotonic() + 10.0
            while time.monotonic() < t_end and not hands.state("left")["valid"]:
                time.sleep(0.05)
            st0 = hands.state("left")
            assert st0["valid"], st0
            assert st0["pressure"].shape == (9, 12)
            n0 = st0["states"]
            time.sleep(1.0)
            rate = hands.state("left")["states"] - n0
            assert rate > 50  # ~100 Hz (under load: a margin)
            q_goal = st0["q"].copy()
            q_goal[5:7] = [-0.6, -0.6]  # the left index closes (negative on the left hand)
            hands.command("left", q_goal, kp=[1.5] * 7, ttl_s=1.5)
            time.sleep(1.2)
            q_mid = hands.state("left")["q"]
            assert np.all(np.abs(q_mid[5:7] - (-0.6)) < 0.15), q_mid  # the simulated finger followed
            assert br.stats["hand_cmd_rx"] > 50
            time.sleep(0.8)  # the TTL expired at 1.5 s: holding where it is
            st = hands.state("left")
            assert st["held"] and st["valid"]
            assert np.all(np.abs(st["q"][5:7] - q_mid[5:7]) < 0.1)  # no snap back to the open hand
        finally:
            hands.close()
