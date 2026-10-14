"""The state-age rung with REAL processes (session 9, U19 / D25; docs/sim/STATE_AGE.md): the DDS-bridge sim (the S1
stand fixture as Unitree's built-in) + r1rt (stale_rung 1 in its config) + r1guard + r1d + the `r1` CLI, the takeover
into BALANCE, then the bridge's robot-side faults on rt/lowstate:

  * `lowstate_delay` 40 ms (every sample published 40 ms late: fresh receive stamps at r1rt, the board tick behind) ->
    r1rt reads the board-tick content age, raises R1RT_FAULT_STATE_OLD and freezes into the stand (STAND_FREEZE) -
    the robot-like path of the lockstep test; never limp (every applied command keeps the legs stiff), one writer;
  * `lowstate_gap` 0.5 s (nothing published) -> the same freeze from the receive age.

Fails on 6449aee (no content age, no rung: r1rt keeps BALANCE). Wall-clock on a Mac without real-time scheduling: the
bridge's physics lags under load (RTF 0.4-0.6 with 11 agents) - so it runs here with `board_tick="wall"` (the real
board's tick is a real-time clock; the default simulated-time tick makes a slow bridge read as OLD state, an artefact);
the test reports the bridge's lag next to any early freeze.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]

from r1sim import stack  # noqa: E402

if not stack.available():
    pytest.skip("needs the dds extra and the DDS build of r1rt / r1guard (scripts/ci.sh dds)", allow_module_level=True)

DOMAIN = 52
FAULT_STATE_OLD = 1 << 19
# the board's tick in real time, as on the robot (the bridge's default ticks simulated time: a bridge slower than real
# time would read as old state - docs/sim/STATE_AGE.md)
BRIDGE: dict[str, Any] = {"scene": "none", "builtin": "stand", "board_tick": "wall"}


def _cfg() -> dict[str, Any]:
    from r1mc3 import rtconfig

    cfg = rtconfig.core_dict()
    cfg["stale_rung"] = 1  # armed here (the robot profile ships 0 until the user's OK)
    return cfg


def _r1rt_faults_seen(st: stack.ProcStack, after: float) -> int:
    assert st.rt is not None
    word = 0
    for d in st.rt.json_lines():
        if d.get("status") == "r1rt" and d["_t"] > after:
            word |= int(d["faults"])
    return word


def _health(st: stack.ProcStack) -> str:
    lg = st.bridge.log()
    gaps = f"{lg['lowstate_gap_p99_ms']} / {lg['lowstate_gap_max_ms']} ms"
    return f"bridge physics {lg['physics']}, lowstate gap p99 / max {gaps}"


@pytest.mark.parametrize(
    ("kind", "params", "domain"),
    [("lowstate_delay", {"delay_ms": 40}, DOMAIN), ("lowstate_gap", {}, DOMAIN + 1)],
)
def test_old_state_freezes_a_balancing_robot_into_the_stand_with_real_processes(
    tmp_path: Path, kind: str, params: dict[str, Any], domain: int
) -> None:
    with stack.ProcStack(domain=domain, workdir=tmp_path, bridge=BRIDGE, cfg=_cfg()) as st:
        st.wait_mode("monitor", 10)
        tk = st.cli("takeover")
        assert tk["ok"], tk
        st.wait_mode("balance", 10)
        time.sleep(1.0)
        assert st.mode() == "balance", f"froze before the fault: {_health(st)}"
        br = st.bridge
        t0 = time.monotonic()
        n_before = len(br.applied)
        br.inject(kind, duration_s=1.5 if kind == "lowstate_delay" else 0.5, **params)
        st.wait_mode("stand_freeze", 4)
        time.sleep(2.0)
        assert st.mode() == "stand_freeze"  # rung 1: the operator brings BALANCE back
        if kind == "lowstate_delay":  # r1rt's 1 Hz status line samples the fault word: a 1.5 s window is always seen
            assert _r1rt_faults_seen(st, t0) & FAULT_STATE_OLD
        after = list(br.applied)[n_before:]
        writers = {a.writer for a in after}
        assert len(writers) == 1, f"writers after the fault: {writers}"  # r1rt alone (the guard never took over)
        assert min(a.min_leg_kp for a in after) > 10.0  # never limp: the freeze holds the legs at stand gains
        assert br.tilt_deg() < 5.0, _health(st)
        # the operator's way back (the same ENTER_BALANCE as after any STAND_FREEZE): r1d's word for it is session 10's
