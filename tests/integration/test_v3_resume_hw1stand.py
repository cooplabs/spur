"""Fix #29 follow-up (session 8): the resume keeps the pre-stop HOLDING TORQUE - on HW1's stand, identified board.

The integrator's defaults re-run of the HW1 rehearsal (the DDS bridge on the identified board model + HW1_STAND, where
the hanging hips carry ~3 N m of gravity at kp 200) found that after a board-hold pause of 100 / 300 ms - the board
HELD, the legs moved <= 0.001 rad - the resume blend of 0bfe1ca gave a 3.05 / 3.08 N m hip torque step and a 0.013 rad
dip: it started at the bare measured pose and dropped the pre-stop holding offset (the base had no step there). Now the
blend starts at the measured pose + (the last command - the pose measured before the stop): no sag -> the start is the
old target (nothing moves); a sag -> the holding torque is continuous and the extra kp x sag builds over the blend.

Lockstep on the bridge's own robot (`DdsBridge(stance="hanging")` without DDS: HW1_STAND, DriverConfig.real() = the
identified board, 0.385 s hold then damp kd 12.5, the refit friction), ON_STAND through the mailbox from the stand-lock
built-in, the core stalled ("v3"), the largest 1 ms change of the applied torque on legs / waist / arms after the
resume and the largest dip of a joint below its pre-stop pose. Fails on 0bfe1ca (0.2 / 0.3 s: ~3 N m, ~0.013 rad).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1sim import cpp_policy  # noqa: E402
from r1sim.dds_bridge import DdsBridge  # noqa: E402
from r1sim.faults import Fault, FaultSchedule  # noqa: E402
from r1sim.v3_core import MODE_ON_STAND, EmulatedBuiltin, TakeoverPlan, V3CoreController  # noqa: E402

from tests.integration.test_v3_core_onstand import RELEASE_S, T_ON_STAND, T_REQUEST  # noqa: E402

if cpp_policy.find_module() is None:
    pytest.skip("r1rt_py is not built (scripts/ci.sh cpp)", allow_module_level=True)

BODY = slice(0, 24)
T_SILENCE = T_ON_STAND + 1.0


def _run(silence_s: float) -> dict[str, Any]:
    br = DdsBridge(domain=99, scene="none", stance="hanging")  # never started: its settled hanging sim, in lockstep
    sim = br.sim
    sim.faults = FaultSchedule([Fault("controller_stall", T_SILENCE + sim.time, silence_s, {"name": "v3"})], seed=1)
    t0 = sim.time
    builtin = EmulatedBuiltin(br.world.model)  # stiff at the stand pose = Unitree's stand lock
    plan = TakeoverPlan(builtin, t_request_s=t0 + T_REQUEST, release_s=RELEASE_S, target="on_stand", via_mailbox=True)
    ctl = V3CoreController(br.world.model, faults=sim.faults, takeover=plan)
    sim.add_controller(builtin, ctl.period_s, "builtin")
    sim.add_controller(ctl, ctl.period_s, "v3")
    t_sil = t0 + T_SILENCE
    rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    while sim.time < t_sil + silence_s + 1.5:
        sim.run_for(0.001)
        if sim.time >= t_sil - 0.05:
            st = sim.read_state()
            rows.append(
                (
                    sim.time,
                    np.asarray(st.truth.get("q", st.q)).copy(),
                    np.asarray(st.truth.get("tau_est", st.tau_est)).copy(),
                )
            )
    t = np.array([r[0] for r in rows])
    q = np.array([r[1] for r in rows])
    tau = np.array([r[2] for r in rows])
    t_res = t_sil + silence_s
    win = (t >= t_res - 0.002) & (t <= t_res + 0.3)
    dtau = np.abs(np.diff(tau[:, BODY], axis=0))[win[1:]]
    pre = q[(t >= t_sil - 0.05) & (t < t_sil)][-1, BODY]
    k_res = int(np.argmin(np.abs(t - t_res)))
    sag = np.abs(q[k_res, BODY] - pre)
    after = q[(t >= t_res) & (t <= t_res + 1.0)][:, BODY]
    dip = float(np.abs(after - q[k_res, BODY]).max())  # the motion caused by the resume itself (the blend's dip / kick)
    return {
        "ctl": ctl,
        "step_nm": float(dtau.max()),
        "sag_max": float(sag.max()),
        "dip": dip,
        "end_err": float(np.abs(q[-1, BODY] - pre).max()),
    }


def _public(r: dict[str, Any]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}


@pytest.mark.parametrize("silence_s", [0.2, 0.3])
def test_a_pause_inside_the_boards_hold_resumes_without_a_step_or_a_dip(silence_s: float) -> None:
    r = _run(silence_s)
    print(f"\n[ info ] HW1 stand, identified board, {silence_s} s stop: {_public(r)}")
    st = r["ctl"].status()
    assert st["mode"] == MODE_ON_STAND and st["stops"] == 1 and st["resume_blends"] == 1, st
    assert r["sag_max"] < 0.002, r  # the board held (0.385 s)
    assert r["step_nm"] < 0.5, r  # 0bfe1ca: ~3 N m on the hips (the holding offset dropped)
    assert r["dip"] < 0.002, r  # 0bfe1ca: ~0.013 rad


@pytest.mark.parametrize("silence_s", [1.0, 2.0])
def test_a_long_stop_on_the_hw1_stand_still_resumes_without_a_snap(silence_s: float) -> None:
    r = _run(silence_s)
    print(f"\n[ info ] HW1 stand, identified board, {silence_s} s stop: {_public(r)}")
    assert r["sag_max"] > 0.02, r  # the board released after 0.385 s: the hips sagged
    # session 9 (the legs' identified armature 0.008, armature agent): the 2 s stop sags 0.136 rad and the first
    # step after the resume reads 2.055 N m (session 8 on 0.05: 1.96; the base before #29: 11.6) -> < 2.5
    assert r["step_nm"] < 2.5, r
    assert r["end_err"] < 0.03, r
