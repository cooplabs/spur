"""The leg-stiction ramp (HW1 S4) recorded THROUGH r1d fits as recorded (rehearsal R6), in lockstep: the robot hanging
on its stand (the welded fixture of test_record_on_stand), `on_stand`, then `record.run ramp joint=left_knee` (the
soft-servo protocol of r1sim.sysid.stiction), r1log's view = the core's /r1rt/telemetry ring drained every 10 ms.

r1d streams the setpoint at its 100 Hz tick, so the 500 Hz q_cmd in the telemetry is a staircase: on 7d9c62f
`fit_joint` found "no ramp segments (q_cmd constant)". `stiction.ramp_recording` adds the PLAN's reference (from the
manifest) for the segments - the torque stays the logged command's - and the fit recovers the model's knee
frictionloss (the nominal real profile: 2.5 N·m, issue #51's value) within 5 %."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.log.records import TELEMETRY
from r1sim import cpp_policy
from r1sim import joints as J
from r1sim.engine import LowState
from r1sim.sysid import stiction
from r1sim.sysid.records import BODY_FIELDS, body_from_arrays

from tests.integration.test_record_on_stand import HangRig
from tests.integration.test_v3_core_onstand import _hanging_world

pytestmark = [
    pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp"),
    pytest.mark.slow,
]


def test_knee_ramp_recorded_through_r1d_fits(tmp_path: Path) -> None:
    rig = HangRig(tmp_path)
    raw: list[bytes] = []

    def r1log(_st: LowState) -> None:  # r1log's drain of the core's telemetry ring (the full 500 Hz record)
        b = rig.port.view.pop_telemetry()
        if b:
            raw.append(bytes(b))

    rig.sim.add_controller(r1log, 0.01, "r1log")
    res: dict[str, Any] = {}
    rig.at(1.5, lambda: res.__setitem__("on", rig.svc.submit({"op": "on_stand"})))
    ramp = {"joint": "left_knee"}
    rig.at(3.0, lambda: res.__setitem__("ramp", rig.svc.submit(
        {"op": "record.run", "kind": "ramp", "params": ramp, "dry_run": False}).result))  # fmt: skip
    dur = stiction.ramp_plan("left_knee").duration_s
    rig.sim.run_for(3.0 + dur + 0.6)
    assert res["on"].result is not None and res["on"].result.ok
    assert res["ramp"] is not None and res["ramp"].ok, res["ramp"]
    man_path = next(p for p in tmp_path.glob("*_ramp_left_knee.json"))
    man = json.loads(man_path.read_text())
    assert man["status"] == "done" and man["refused"] == 0, man["status"]

    tel = np.frombuffer(b"".join(raw), dtype=TELEMETRY)
    t_abs = tel["t_wake_ns"].astype(np.int64)
    fields = {k: np.asarray(tel[k], dtype=np.float64) for k in BODY_FIELDS}
    knee = J.IDX["left_knee"]
    w, _ = _hanging_world()  # the rig's model (the same build): its knee dof's frictionloss
    true_fc = float(rig.sim.m.dof_frictionloss[int(w.index.vadr[knee])])

    # as recorded, without the plan: the staircase has no constant-gradient runs (what 7d9c62f's path did)
    sel = (t_abs >= man["t_start_mono_ns"]) & (t_abs <= man["t_end_mono_ns"])
    plain = body_from_arrays(t_abs[sel] * 1e-9, {k: v[sel] for k, v in fields.items()})
    with pytest.raises(ValueError, match="no ramp segments"):
        stiction.fit_joint(plain, "left_knee")

    rec = stiction.ramp_recording(man, t_abs, fields)
    fit = stiction.fit_joint(rec, "left_knee")
    print(f"knee frictionloss {fit.frictionloss:.3f} +- {fit.frictionloss_se:.3f} (model {true_fc}), "
          f"static {fit.static:.3f}, damping {fit.damping:.3f}, speeds {fit.speeds_measured}")  # fmt: skip
    assert abs(fit.frictionloss - true_fc) / true_fc < 0.05, (fit.frictionloss, true_fc, fit.notes)
    assert len(fit.speeds_measured) == 2  # both protocol speeds (0.1 / 0.3 rad/s commanded) found as segments

    # the same from the FILES (the manifest + an r1log MCAP of the telemetry): stiction.load_ramp, as HW1 runs it
    from r1sim.sysid.records import write_body_mcap

    whole = body_from_arrays((t_abs - t_abs[0]) * 1e-9, fields)
    mcap = tmp_path / "r1log_telemetry.mcap"
    write_body_mcap(mcap, whole, t0_ns=int(t_abs[0]))
    fit2 = stiction.fit_joint(stiction.load_ramp(man_path, [mcap]), "left_knee")
    assert abs(fit2.frictionloss - fit.frictionloss) < 1e-6
