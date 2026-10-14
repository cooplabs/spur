"""B3 from HW2's arm sweeps (session 11, core; r1sim.sysid.b3): the analysis HW1_FITS.md "Session 8" item 3 asked for.

Synthetic checks on the simulated HANGING robot (the sweep = `r1 record run ramp joint=left_elbow ... payload_kg=KG`'s
protocol, a soft servo kp 20; the payload a point at the palm site):

* the model's payload share is the point mass's r x f about the joint axis (the J^T m g it uses);
* a robot that IS the model: mass scale s ~ 1, payload scale k ~ 1, every point within 10 % -> B3 shown - also
  through the files (an r1log MCAP with the IMU quaternion + a `record run ramp` manifest -> `load_sweep`);
* the same sweeps on an arm whose links (the hand's included) are 20 % heavier: s ~ 1.19 and NOT shown (with the
  free offset alone 90 % of its points would pass - the scales decide too);
* only an empty sweep: no torque scale -> NOT shown (B3 needs 0 / 0.5 kg).

Fails on the base 1e55bdd (no r1sim.sysid.b3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from r1sim import joints as J
from r1sim.sysid import b3

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def gm() -> b3.GravityModel:
    return b3.GravityModel()


@pytest.fixture(scope="module")
def sweeps(gm: b3.GravityModel) -> dict[str, list[b3.Sweep]]:
    out: dict[str, list[b3.Sweep]] = {}
    for scale in (1.0, 1.2):
        row = []
        for kg in (0.0, 0.5):
            sw = b3.synthetic_sweep("left_elbow", payload_kg=kg, arm_mass_scale=scale)
            sw.imu_quat = gm.q_site_imu.copy()  # the welded pelvis is upright: the IMU reads its own mount
            row.append(sw)
        out[f"x{scale}"] = row
    return out


def test_the_payload_share_is_the_point_mass_moment_about_the_axis(gm: b3.GravityModel) -> None:
    q = J.POSE_STAND.copy()
    q[J.IDX["left_elbow"]] = 0.6
    tau, share = gm.holding(q, gm.q_site_imu, "left_elbow", payload_kg=0.5)
    empty, zero = gm.holding(q, gm.q_site_imu, "left_elbow")
    assert zero == 0.0 and tau == pytest.approx(empty + share)
    m, d = gm.m, gm.d
    v = int(gm.ix.vadr[J.IDX["left_elbow"]])
    jid = int(m.dof_jntid[v])
    sid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_hand_palm"))
    r = d.site_xpos[sid] - d.xanchor[jid]
    assert share == pytest.approx(-float(d.xaxis[jid] @ np.cross(r, [0.0, 0.0, -0.5 * 9.81])), rel=1e-9)
    assert abs(share) > 0.3  # a 0.5 kg payload at the palm matters on the elbow (~0.75 N m here)


def test_a_robot_that_is_the_model_shows_b3(sweeps: dict[str, list[b3.Sweep]], gm: b3.GravityModel) -> None:
    res = b3.analyse(sweeps["x1.0"], gm=gm)
    r = res["joints"]["left_elbow"]
    print(b3.table(res))
    assert r["n_points"] >= b3.MIN_POSES
    assert abs(r["mass_scale"] - 1.0) < 0.03  # 0.992
    assert abs(r["payload_scale"] - 1.0) < 0.08  # 0.961 (the loaded sweep's friction / dynamics)
    assert r["within_tol"] == 1.0 and r["shown"] and res["shown_all"]


def test_a_20_percent_heavier_arm_is_not_shown(sweeps: dict[str, list[b3.Sweep]], gm: b3.GravityModel) -> None:
    res = b3.analyse(sweeps["x1.2"], gm=gm)
    r = res["joints"]["left_elbow"]
    print(b3.table(res))
    assert 1.15 < r["mass_scale"] < 1.25  # 1.186
    assert not r["shown"] and any(n.startswith("mass scale") for n in r["notes"])


def test_an_empty_sweep_alone_gives_no_torque_scale(sweeps: dict[str, list[b3.Sweep]], gm: b3.GravityModel) -> None:
    res = b3.analyse(sweeps["x1.0"][:1], gm=gm)
    r = res["joints"]["left_elbow"]
    assert np.isnan(r["payload_scale"]) and not r["shown"]
    assert any("no payload sweep" in n for n in r["notes"])


def _write(tmp: Path, sw: b3.Sweep, quat: np.ndarray, tag: str) -> Path:
    """The sweep as r1d + r1log leave it on the robot: the r1log MCAP (/r1rt/telemetry with the IMU quaternion) and a
    `record run ramp` manifest (plan kind / params, the group's base, its start time, its r1log files)."""
    from r1mc3.log import mcapfile
    from r1mc3.log import records as R
    from r1mc3.service.codes import GROUP_JOINTS

    t0 = 5_000_000_000
    rec = sw.rec
    recs = np.zeros(rec.t.size, dtype=R.TELEMETRY)
    recs["seq"] = np.arange(rec.t.size)
    recs["t_wake_ns"] = t0 + np.round(rec.t * 1e9).astype(np.int64)
    for k in rec.fields:
        recs[k] = rec.fields[k]
    recs["imu_quat"] = quat
    mc = tmp / f"r1log_{tag}.mcap"
    f = mcapfile.McapFile(mc, metadata={"source": "test_sysid_b3"})
    for r in recs:
        f.write("telemetry", int(r["t_wake_ns"]), r.tobytes(), int(r["seq"]))
    f.close()
    cols = list(GROUP_JOINTS["arm_left"])
    params: dict[str, Any] = {"joint": sw.joint, "amplitude_rad": 0.3, "speeds": [0.1, 0.2], "kp": 20.0, "kd": 1.0}
    if sw.payload_kg:
        params["payload_kg"] = sw.payload_kg
    man = {
        "plan": {"kind": "ramp", "params": params},
        "group_joints": cols,
        "base": [float(J.POSE_STAND[c]) for c in cols],
        "t_start_mono_ns": t0,
        "t_end_mono_ns": int(recs["t_wake_ns"][-1]),
        "r1log_files": [str(mc)],
    }
    mp = tmp / f"manifest_{tag}.json"
    mp.write_text(json.dumps(man))
    return mp


def test_the_files_path_gives_the_same_verdict(
    tmp_path: Path, sweeps: dict[str, list[b3.Sweep]], gm: b3.GravityModel
) -> None:
    quat = gm.q_site_imu.copy()
    m0 = _write(tmp_path, sweeps["x1.0"][0], quat, "0kg")
    m5 = _write(tmp_path, sweeps["x1.0"][1], quat, "05kg")
    loaded = [b3.load_sweep(str(m0)), b3.load_sweep(str(m5))]
    assert [s.payload_kg for s in loaded] == [0.0, 0.5] and loaded[0].joint == "left_elbow"
    assert np.allclose(loaded[0].imu_quat, quat / np.linalg.norm(quat), atol=1e-6)
    res = b3.analyse(loaded, gm=gm)
    r = res["joints"]["left_elbow"]
    print(b3.table(res))
    assert r["shown"] and abs(r["mass_scale"] - 1.0) < 0.03
    # the same file with the payload given on the command line instead (MANIFEST@KG)
    assert b3.load_sweep(f"{m0}@0.5").payload_kg == 0.5
