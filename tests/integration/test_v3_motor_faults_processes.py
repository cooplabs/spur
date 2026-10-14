"""DESIGN §4.2 motor error words and the over-temperature warning with REAL processes (session 6): the DDS-bridge sim
(the S1 stand fixture as Unitree's built-in, centred stance) + r1rt + r1guard + r1d + the `r1` CLI (r1sim.stack), the
takeover into BALANCE, then the bridge's injections (hang's `motor_error` / `over_temp`, as the board would report them
in rt/lowstate):

  * a LEFT-ARM motor's error word -> the core HOLDs the left arm (its commanded targets stop changing) and refuses its
    records (`r1 joint arm_left ...` moves nothing), while `r1 joint arm_right ...` still moves the right arm and the
    legs keep balancing; after the word clears and the operator's `r1 resume`, the left arm follows records again;
  * a hip-roll temperature ramp to 80 C -> r1rt's faults carry R1RT_FAULT_OVER_TEMP (a warning) and nothing in the
    command changes stiffness (the legs' gains as before).

Session 5 raised MOTOR_ERROR only (the left arm kept following) and had no temperature warning: fails on 26e8bce.
Wall-clock on a Mac without real-time scheduling: the asserted facts are the states, with generous waits.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]

from r1sim import stack  # noqa: E402

if not stack.available():
    pytest.skip("needs the dds extra and the DDS build of r1rt / r1guard (scripts/ci.sh dds)", allow_module_level=True)

DOMAIN = 72
FAULT_MOTOR_ERROR = 1 << 9
FAULT_OVER_TEMP = 1 << 18
L_SHOULDER_ROLL = 15  # SDK index (arm_left = 14-18)
R_SHOULDER_ROLL = 20  # arm_right = 19-23
L_HIP_ROLL = 1


def _mirror(st: stack.ProcStack) -> dict[str, Any]:
    from r1sim.cpp_policy import import_r1rt_py

    v = import_r1rt_py().SegmentView.attach(st.shm)
    try:
        m = v.mirror()
        assert m is not None
        return dict(m)
    finally:
        v.close()


def _joint(st: stack.ProcStack, group: str, q: list[float]) -> dict[str, Any]:
    return st.cli("joint", group, ",".join(f"{v:.4f}" for v in q), "--ttl", "0.3")


def _r1rt_faults(st: stack.ProcStack, after: float) -> int:
    assert st.rt is not None
    lines = [d for d in st.rt.json_lines() if d.get("status") == "r1rt" and d["_t"] > after]
    return int(lines[-1]["faults"]) if lines else 0


def test_an_arm_motor_error_holds_that_arm_and_over_temperature_warns(tmp_path: Path) -> None:
    with stack.ProcStack(domain=DOMAIN, workdir=tmp_path, bridge={"scene": "none", "builtin": "stand"}) as st:
        st.wait_mode("monitor", 10)
        tk = st.cli("takeover")
        assert tk["ok"], tk
        st.wait_mode("balance", 10)
        time.sleep(0.5)
        ql = st.cli("hold", "arm_left")["data"]["q"]
        qr = st.cli("hold", "arm_right")["data"]["q"]
        # the error word on the left shoulder roll (5 s window)
        st.bridge.inject("motor_error", joint=L_SHOULDER_ROLL, word=1, duration_s=5.0)
        t0 = time.monotonic()
        time.sleep(1.2)  # > the stop time; r1rt's 1 Hz status line
        assert _r1rt_faults(st, t0) & FAULT_MOTOR_ERROR
        held = float(_mirror(st)["q_target"][L_SHOULDER_ROLL])
        ql2 = list(ql)
        ql2[1] += 0.10
        qr2 = list(qr)
        qr2[1] -= 0.10
        for _ in range(4):  # 1.2 s of setpoints (TTL 0.3 s each)
            assert _joint(st, "arm_left", ql2)["ok"]
            assert _joint(st, "arm_right", qr2)["ok"]
            time.sleep(0.3)
        m = _mirror(st)
        assert abs(float(m["q_target"][L_SHOULDER_ROLL]) - held) < 1e-6  # the left arm held: its records refused
        assert float(m["q_target"][R_SHOULDER_ROLL]) < qr[1] - 0.05  # the right arm followed
        assert min(float(k) for k in np.asarray(m["kp"])[14:19]) > 1.0  # held stiff, never limp
        assert st.mode() == "balance" and st.bridge.tilt_deg() < 5.0
        # the word clears (end of the window), then the operator's resume: the left arm follows again
        time.sleep(max(0.0, 5.3 - (time.monotonic() - t0)))
        assert st.cli("resume")["ok"]
        for _ in range(4):
            assert _joint(st, "arm_left", ql2)["ok"]
            time.sleep(0.3)
        assert float(_mirror(st)["q_target"][L_SHOULDER_ROLL]) > held + 0.05
        # over-temperature: the left hip roll ramps to 80 C -> a warning, the legs' gains unchanged
        kp_before = np.asarray(_mirror(st)["kp"], dtype=np.float64)[:12].copy()
        st.bridge.inject("over_temp", joint=L_HIP_ROLL, to_c=80, rate_c_s=40, duration_s=4.0)
        t1 = time.monotonic()
        time.sleep(2.5)
        assert _r1rt_faults(st, t1) & FAULT_OVER_TEMP
        kp_now = np.asarray(_mirror(st)["kp"], dtype=np.float64)[:12]
        assert np.array_equal(kp_now, kp_before)  # a warning only: no action on stiffness
        assert st.mode() == "balance" and st.bridge.tilt_deg() < 5.0


def test_on_stand_from_zero_torque_blends_the_gains_up_with_real_processes(tmp_path: Path) -> None:
    """Rehearse R7 (session 6): ON_STAND from Unitree's ZERO TORQUE (the hanging bridge's default built-in: kp 0 / kd 0,
    its targets at the standing pose, far from the hanging legs). Until 6c59cd6 r1rt's first command held the measured
    pose at the ON_STAND gains: the legs went kp 0 -> 200 in one packet. Now the snapshot is the measured pose at the
    built-in's gains (kp 0) and the blend ramps the legs' kp up over the 1 s blend (U13 / D22)."""
    with stack.ProcStack(domain=DOMAIN + 1, workdir=tmp_path, bridge={"scene": "none", "stance": "hanging"}) as st:
        st.wait_mode("monitor", 10)
        res = st.cli("on-stand")
        assert res["ok"], res
        st.wait_mode("on_stand", 10)
        time.sleep(0.3)
        br = st.bridge
        rt_h = [h for h, w in br.writers().items() if w["label"] == "r1rt"]
        assert len(rt_h) == 1
        cmds = [a for a in list(br.applied) if a.writer == rt_h[0]]
        kps = np.array([a.min_leg_kp for a in cmds])
        walls = np.array([a.wall for a in cmds])
        assert kps[0] < 1.0, kps[:5]  # r1rt's first packet: the built-in's kp 0 (6c59cd6: 200)
        assert float(np.max(np.diff(kps))) < 20.0  # no one-packet stiffening (6c59cd6: one jump of ~200)
        t_stiff = float(walls[np.argmax(kps >= 0.95 * kps.max())] - walls[0])
        assert t_stiff > 0.6  # stiff only at the end of the ~1 s blend (+ MIRROR while ReleaseMode runs)
        assert kps.max() > 30.0  # the stiff ON_STAND hold at the end
        assert min(t.foot_z_min for t in list(br.truth)[-200:]) > 0.02  # still hanging
