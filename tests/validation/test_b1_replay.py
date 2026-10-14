"""B1 / B2 (DESIGN §13.2): the replay tool (r1sim.tier_b) - its metrics on synthetic signals with known answers, its
record loader on an r1log-format file, and the replay of a REAL HW1 step (the left knee, S2, 11 Oct 2026; an excerpt
in tests/assets/hw1/b1_step_left_knee.npz, README.replay.md) against the B1 / B2 targets.

The identified values (docs/sim/TIER_B.md): no added command delay (B2: the sim's own joint dynamics give the real
20 ms "delay to 10 %"; the wrap-up's "+5-8 ms" came from comparing with a sim that was not replaying the same
command), the lowstate age 0.5 ms (T5), knee armature 0.008 (the model's 0.05, issue #51, gives 1.5-2x the real torque
swing), knee friction 0.55 N m (the HW1 S4 re-fit)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from r1sim.tier_b import replay as R
from r1sim.tier_b.data import Window, load_session

ASSET = Path(__file__).resolve().parents[1] / "assets" / "hw1" / "b1_step_left_knee.npz"
HIP_ASSET = ASSET.with_name("b1_step_left_hip_pitch.npz")
IDENTIFIED_KNEE = R.ReplayConfig(
    cmd_delay_s=0.0, state_delay_s=0.0005, friction=R.PROVISIONAL_FRICTION, armature={"knee": 0.008}
)


# ------------------------------------------------------------------------------------------------ metrics
def _ramp_step(t: np.ndarray, t0: float, a: float, rate: float = 2.0) -> np.ndarray:
    return np.asarray(np.clip((t - t0) * rate, 0.0, abs(a)) * np.sign(a), dtype=np.float64)


def _first_order(t: np.ndarray, u: np.ndarray, tau: float) -> np.ndarray:
    y = np.zeros_like(u)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        y[i] = y[i - 1] + dt / tau * (u[i - 1] - y[i - 1])
    return y


def test_step_timing_measures_a_known_added_delay() -> None:
    t = np.arange(0.0, 4.0, 0.002)
    cmd = _ramp_step(t, 1.0, 0.05) - _ramp_step(t, 3.0, 0.05)
    q = _first_order(t, cmd, 0.02)
    q_late = np.interp(t - 0.003, t, q)  # the same response 3 ms later
    ev = R.step_events(t, cmd)
    assert [round(e[0], 3) for e in ev] == [1.0, 3.0]
    d0 = R.step_timing(t, q, ev)
    d3 = R.step_timing(t, q_late, ev)
    assert all(abs((b - a) - 0.003) < 2e-4 for a, b in zip(d0, d3, strict=True)), (d0, d3)


def test_chirp_response_recovers_gain_and_lag() -> None:
    t = np.arange(0.0, 20.0, 0.002)
    params = {"f0_hz": 0.5, "f1_hz": 5.0, "duration_s": 20.0}
    ph = R.chirp_phase(t, 0.5, 5.0, 20.0)
    cmd = 0.03 * np.sin(ph)
    delay = 0.010  # a pure 10 ms delay and a gain of 0.8: lag = 360 f delay
    q = 0.8 * 0.03 * np.sin(R.chirp_phase(t - delay, 0.5, 5.0, 20.0))
    out = R.chirp_response(t, cmd, q, params, (2.0, 4.0))
    for f in (2.0, 4.0):
        assert out[f]["gain"] == pytest.approx(0.8, abs=0.02)
        assert out[f]["lag_deg"] == pytest.approx(360.0 * out[f]["f_centre"] * delay, abs=1.0)


def test_nrms_ignores_offsets_and_scales_with_the_error() -> None:
    real = np.sin(np.linspace(0, 10, 500))
    assert R.nrms(real + 0.3, real) == pytest.approx(0.0, abs=1e-12)
    assert R.nrms(1.1 * real, real) == pytest.approx(0.1, rel=0.02)


# ------------------------------------------------------------------------------------------------ the loader
def test_load_session_cuts_a_manifest_window_from_an_r1log_file(tmp_path: Path) -> None:
    from r1sim.sysid.records import body_from_arrays, write_body_mcap

    n = 2000
    t = np.arange(n) * 0.002
    f = {k: np.zeros((n, 26)) for k in ("q", "dq", "tau_est", "q_cmd", "dq_cmd", "kp", "kd", "tau_ff")}
    f["q_cmd"][:, 3] = np.where(t > 1.0, 0.45, 0.40)
    t0_ns = 5_000_000_000
    (tmp_path / "r1log").mkdir()
    (tmp_path / "records").mkdir()
    write_body_mcap(tmp_path / "r1log" / "r1log_x_000.mcap", body_from_arrays(t, f), t0_ns=t0_ns)
    man = {
        "schema": "r1mc3.record_manifest/1",
        "status": "done",
        "plan": {"kind": "step", "params": {"joint": "left_knee", "amplitude_rad": 0.05}, "duration_s": 2.0},
        "t_start_mono_ns": t0_ns + 1_000_000_000,
        "t_end_mono_ns": t0_ns + 3_000_000_000,
        "r1log_files": ["/var/lib/r1mc3/logs/r1log_x_000.mcap"],
    }
    (tmp_path / "records" / "20261011T000000Z_step_left_knee.json").write_text(json.dumps(man))
    (w,) = load_session(tmp_path, pad_s=0.5)
    assert w.joint == "left_knee" and w.kind == "step"
    assert w.t[0] == pytest.approx(-0.5, abs=0.003) and w.t[-1] == pytest.approx(2.5, abs=0.003)
    assert np.all(w.j("q_cmd", 3)[w.t > 0.01] == pytest.approx(0.45))


# ------------------------------------------------------------------------------------------------ the real step
@pytest.fixture(scope="module")
def knee() -> Window:
    return Window.load(ASSET)


def test_the_identified_model_meets_b1_and_b2_on_the_real_knee_step(knee: Window) -> None:
    """B1: nRMS <= 10 %, torque swing within +-15 %; B2: the delay-to-10 % median within +-1 ms."""
    sc = R.score(knee, R.replay(knee, IDENTIFIED_KNEE))
    assert sc.nrms <= R.B1_NRMS_MAX, sc.nrms
    assert abs(sc.delay_median_diff_ms) <= R.B2_DELAY_MS, (sc.delay10_real_ms, sc.delay10_sim_ms)
    assert abs(sc.torque["rms_ratio"] - 1.0) <= R.B1_TORQUE_TOL, sc.torque
    assert sc.hang["foot_z_min_m"] > 0.05  # the feet hang off the floor, as on the stand


def test_an_added_command_delay_breaks_b2_on_the_knee(knee: Window) -> None:
    """The wrap-up's '+5-8 ms' sim delay: the replay says no - 6 ms more puts the knee's delay outside +-1 ms."""
    late = R.ReplayConfig(cmd_delay_s=0.006, state_delay_s=0.0005, friction=R.PROVISIONAL_FRICTION,
                          armature={"knee": 0.008})  # fmt: skip
    sc = R.score(knee, R.replay(knee, late))
    assert sc.delay_median_diff_ms > 4.0, sc.delay_median_diff_ms


def test_the_real_profile_as_built_meets_b1_on_the_knee_step(knee: Window) -> None:
    """Session 9 (armature): the legs' identified 0.008 is in the model (session 8: a strict xfail - issue #51's 0.05
    gave the replayed knee step a torque swing 1.63x the real one). As built now: nRMS 3.0 %, torque swing 1.055x,
    delay to 10 % -0.4 ms (macOS)."""
    sc = R.score(knee, R.replay(knee, R.ReplayConfig(state_delay_s=0.0005)))
    assert sc.nrms <= R.B1_NRMS_MAX, sc.nrms
    assert abs(sc.torque["rms_ratio"] - 1.0) <= R.B1_TORQUE_TOL and abs(sc.delay_median_diff_ms) <= R.B2_DELAY_MS


def test_verdicts_and_table(knee: Window) -> None:
    sc = R.score(knee, R.replay(knee, IDENTIFIED_KNEE))
    v = R.verdicts(sc)
    assert v["nrms"] is True and v["phase"] is None and v["delay"] is True
    md = R.table([sc])
    assert "left_knee" in md and md.count("\n") == 2
    assert math.isfinite(sc.offset_rad)


def test_the_hip_step_needs_the_compliant_stand() -> None:
    """The stand, not the hip: with the bridge's RIGID pole the replayed hip needs ~1.3x the real torque and misses
    nRMS; with the HW1-fitted compliant pole (the real robot rocks on its stand) both are inside B1's targets. (The
    hip's delay to 10 % stays ~5 ms late in both: open, TIER_B.md.)"""
    hip = Window.load(HIP_ASSET)
    cfg = dict(state_delay_s=0.0005, friction=R.PROVISIONAL_FRICTION, armature={"hip_pitch": 0.008})
    rigid = R.score(hip, R.replay(hip, R.ReplayConfig(**cfg, pole_k=None)))  # type: ignore[arg-type]
    soft = R.score(hip, R.replay(hip, R.ReplayConfig(**cfg)))  # type: ignore[arg-type]
    assert rigid.torque["rms_ratio"] > 1.0 + R.B1_TORQUE_TOL or rigid.nrms > R.B1_NRMS_MAX, rigid
    assert soft.nrms <= R.B1_NRMS_MAX, soft.nrms
    assert abs(soft.torque["rms_ratio"] - 1.0) <= R.B1_TORQUE_TOL, soft.torque
