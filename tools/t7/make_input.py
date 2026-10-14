"""T7's input file: the policy observations the Orin's timing check replays (docs/hardware/T7_BENCH.md).

    uv run python tools/t7/make_input.py            # -> tests/assets/t7/t7_policy_2000.r1pb (+ SHA256SUMS)

**Where the inputs come from:** the live policy (R1MC-Manip2-Lean, `r1mc3.policy.ReferencePolicy`) balancing the
real-equivalent robot in r1sim (lockstep, the empty floor, 1 kHz physics, the controller at the v3 writer's 500 Hz, the
network every 20 ms) with the session-2 sensor model (models/sensors.yaml: encoder / IMU noise, biases, quantisation),
through a 41 s command script: stand 8 s; a 3 deg lean request (this policy's range is 0..5 deg);
stand leaned until 20 s; walk vx 0.2 m/s for 8 s (the policy ramps the lean to 0 while walking); stand again (the lean
ramps back). Every argument the policy step was called with is recorded - policy-order q / dq (24), the IMU
quaternion and gyro, the raw command, the step's time (the phase clock) and the lean target - for the first 2,000
policy steps (40 s; the robot stands throughout: tilt checked < 5 deg). v2's 8 Hz logs carry no dq / IMU rates and
HW1 never ran the policy, so a simulated stance is the realistic source; the network's cost does not depend on the
values anyway (a dense MLP, constant time), only the correctness check does.

**Expected outputs:** the inputs are rounded to float32 (the file stores them in 4 bytes), then a FRESH
ReferencePolicy replays them exactly as r1rt's policy thread drives its runner (per step: the lean target when it
changed, `advance_clock(t)`, `act(q, dq, quat, gyro, cmd)`); its 14 joint targets per step are stored as float32.
The timing program (rt/apps/r1rt_policy_bench.cpp) replays the same sequence through the C++ PolicyRunner and reports
the largest |C++ - stored| (float32 storage: <= ~2e-7 rad expected; the check allows 2e-6).

File "R1PB" v1 (little-endian):
    header    magic b"R1PB", uint32 version 1, n_steps, n_params, n_joints, n_act
    params    float64[n_params]: r1sim.v3_core.PARAMS_LAYOUT (the r1rt_policy_params fields, in order)
    steps     n_steps x (float64 t, float32 lean_request_rad (NaN = none), float32 q[n_joints], dq[n_joints],
              quat[4], gyro[3], cmd[3])
    expected  n_steps x float32 target[n_act]
Read by rt/apps/r1rt_policy_bench.cpp and tests/unit/test_t7_bench.py (`read_r1pb`).
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO / "python"))

from r1mc3.policy import POLICY_DIR, ReferencePolicy  # noqa: E402
from r1sim.v3_core import pack_params, policy_params  # noqa: E402

OUT = REPO / "tests" / "assets" / "t7" / "t7_policy_2000.r1pb"
MAGIC = b"R1PB"
VERSION = 1
N_STEPS = 2000
HEADER = struct.Struct("<4s5I")
LEAN_DEG = 3.0
SCRIPT = (  # (until t s, (vx, vy, wz), lean request deg or None)
    (8.0, (0.0, 0.0, 0.0), None),
    (20.0, (0.0, 0.0, 0.0), LEAN_DEG),
    (28.0, (0.2, 0.0, 0.0), None),
    (41.0, (0.0, 0.0, 0.0), None),
)

FloatArray = npt.NDArray[np.float64]


@dataclass
class R1pb:
    """One R1PB file in memory (float32 fields widened to float64, exactly as the C++ reader does)."""

    params: FloatArray
    n_joints: int
    n_act: int
    t: FloatArray  # (n,)
    lean_req: FloatArray  # (n,), NaN = none
    q: FloatArray  # (n, n_joints)
    dq: FloatArray
    quat: FloatArray  # (n, 4)
    gyro: FloatArray  # (n, 3)
    cmd: FloatArray  # (n, 3)
    expected: FloatArray  # (n, n_act)

    @property
    def n(self) -> int:
        return int(self.t.size)


def _f32(a: npt.ArrayLike) -> FloatArray:
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def record(seconds: float = SCRIPT[-1][0]) -> dict[str, Any]:
    """Run the policy on the real-equivalent robot through SCRIPT; every policy step's arguments (float64)."""
    from r1sim import world
    from r1sim.control import PolicyController
    from r1sim.driver import driver_for_profile
    from r1sim.engine import Simulation
    from r1sim.sensors import SensorConfig, SensorModel

    w = world.build_world("real", visual=False)
    sim = Simulation(w.model, w.index, driver=driver_for_profile("real"), sensors=SensorModel(SensorConfig.from_yaml()))
    sim.reset(w.initial_qpos())
    pol = ReferencePolicy.load(POLICY_DIR)
    ctl = PolicyController(pol)
    sim.add_controller(ctl, 0.002, "policy")
    rows: list[tuple[Any, ...]] = []
    act = pol.act

    def rec_act(q: Any, dq: Any, quat: Any, gyro: Any, cmd: Any) -> Any:
        rows.append((float(pol.t_last or 0.0), float(pol.lean_target), np.array(q), np.array(dq), np.array(quat),
                     np.array(gyro), np.array(cmd, dtype=np.float64)))  # fmt: skip
        return act(q, dq, quat, gyro, cmd)

    pol.act = rec_act  # type: ignore[method-assign]
    tilt_max = 0.0
    for until, cmd, lean in SCRIPT:
        ctl.cmd[:] = cmd
        if lean is not None:
            why = pol.set_body_targets(lean_deg=lean)
            if why is not None:
                raise RuntimeError(why)
        while sim.time < min(until, seconds) - 1e-9:
            sim.run_for(0.05)
            qw = sim.d.xquat[w.index.pelvis_body]
            tilt_max = max(tilt_max, float(np.degrees(2.0 * np.arccos(min(1.0, np.hypot(qw[0], qw[3]))))))
    return {"rows": rows, "tilt_max_deg": tilt_max, "pelvis_z_end_m": float(sim.d.xpos[w.index.pelvis_body][2])}


def replay(pol: ReferencePolicy, f: R1pb) -> FloatArray:
    """The reference on the stored (float32-rounded) inputs, driven as r1rt drives its runner; (n, n_act) targets."""
    pol.reset()
    pol.set_body_targets(lean_deg=0.0)
    out = np.zeros((f.n, f.n_act))
    for k in range(f.n):
        if not np.isnan(f.lean_req[k]):
            lo, hi = pol.lean_range or (0.0, 0.0)
            if lo - 1e-6 <= f.lean_req[k] <= hi + 1e-6:  # = PolicyRunner::set_lean_target (refused: kept)
                pol.lean_target = float(f.lean_req[k])
        if pol.t_last is not None:  # = PolicyRunner::advance_clock
            pol.phase_t += float(f.t[k]) - pol.t_last
        pol.t_last = float(f.t[k])
        out[k] = pol.act(f.q[k], f.dq[k], f.quat[k], f.gyro[k], f.cmd[k])
    return out


def write_r1pb(path: Path, f: R1pb) -> None:
    with path.open("wb") as fh:
        fh.write(HEADER.pack(MAGIC, VERSION, f.n, f.params.size, f.n_joints, f.n_act))
        fh.write(f.params.astype("<f8").tobytes())
        for k in range(f.n):
            fh.write(struct.pack("<d", float(f.t[k])))
            fields = np.concatenate([[f.lean_req[k]], f.q[k], f.dq[k], f.quat[k], f.gyro[k], f.cmd[k]])
            fh.write(fields.astype("<f4").tobytes())
        fh.write(f.expected.astype("<f4").tobytes())


def read_r1pb(path: Path) -> R1pb:
    b = path.read_bytes()
    magic, version, n, n_params, nj, na = HEADER.unpack_from(b, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"{path}: not an R1PB v{VERSION} file")
    off = HEADER.size
    params = np.frombuffer(b, "<f8", n_params, off).astype(np.float64)
    off += 8 * n_params
    per = 1 + 2 * nj + 4 + 3 + 3
    rec = np.dtype([("t", "<f8"), ("f", "<f4", (per,))])
    steps = np.frombuffer(b, rec, n, off)
    off += rec.itemsize * n
    expected = np.frombuffer(b, "<f4", n * na, off).astype(np.float64).reshape(n, na)
    off += 4 * n * na
    if off != len(b):
        raise ValueError(f"{path}: {len(b) - off} trailing bytes")
    fl = steps["f"].astype(np.float64)
    c = np.cumsum([0, 1, nj, nj, 4, 3, 3])
    return R1pb(params=params, n_joints=nj, n_act=na, t=steps["t"].astype(np.float64), lean_req=fl[:, 0],
                q=fl[:, c[1]:c[2]], dq=fl[:, c[2]:c[3]], quat=fl[:, c[3]:c[4]], gyro=fl[:, c[4]:c[5]],
                cmd=fl[:, c[5]:c[6]], expected=expected)  # fmt: skip


def build(n_steps: int = N_STEPS) -> tuple[R1pb, dict[str, Any]]:
    run = record()
    rows = run["rows"][:n_steps]
    if len(rows) < n_steps:
        raise RuntimeError(f"only {len(rows)} policy steps recorded")
    pol = ReferencePolicy.load(POLICY_DIR)
    lean_prev: float | None = None
    lean_req = np.full(n_steps, np.nan)
    for k, r in enumerate(rows):
        if lean_prev is None or r[1] != lean_prev:
            lean_req[k] = r[1]
            lean_prev = r[1]
    f = R1pb(
        params=pack_params(policy_params(pol)),
        n_joints=int(pol.n),
        n_act=int(pol.n_act),
        t=np.array([r[0] for r in rows]),
        lean_req=_f32(lean_req),
        q=_f32([r[2] for r in rows]),
        dq=_f32([r[3] for r in rows]),
        quat=_f32([r[4] for r in rows]),
        gyro=_f32([r[5] for r in rows]),
        cmd=_f32([r[6] for r in rows]),
        expected=np.zeros((n_steps, int(pol.n_act))),
    )
    f.expected = _f32(replay(pol, f))
    info = {k: v for k, v in run.items() if k != "rows"}
    info["policy_steps_recorded"] = len(run["rows"])
    return f, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--steps", type=int, default=N_STEPS)
    a = ap.parse_args(argv)
    f, info = build(a.steps)
    if info["tilt_max_deg"] >= 5.0:
        raise RuntimeError(f"the recorded stance tilted {info['tilt_max_deg']:.2f} deg (>= 5): not a stance")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_r1pb(a.out, f)
    back = read_r1pb(a.out)
    assert np.array_equal(back.expected, f.expected) and np.array_equal(back.q, f.q)
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    (a.out.parent / "SHA256SUMS").write_text(f"{digest}  {a.out.name}\n")
    print(f"{a.out}: {f.n} steps, {a.out.stat().st_size} bytes, sha256 {digest[:16]}..., {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
