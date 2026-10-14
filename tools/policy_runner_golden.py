"""The golden sequence for the C++ policy runner (r1rt::PolicyRunner, DESIGN.md §3.2).

    uv run python tools/policy_runner_golden.py            # -> tests/assets/policy_runner/sequence_1600.r1pr

The sequence is S2's scripted reference test (tests/unit/test_policy_reference.py::test_bit_identical_to_v2_walkpolicy):
1,600 ticks at ~500 Hz with jitter (seed 7), states from `_state(k)` (POSE_STAND + noise, a wobbling quaternion), the
command script (0 -> below the 0.1 phase threshold -> walking beyond the limits (clipped) -> turning -> 0), a 3 deg
lean request at tick 100 and a refused 7 deg request at tick 1300 - run through r1mc3.policy.ReferencePolicy (= v2's
WalkPolicy, bit-identical). The inputs are rounded to float32 first (and fed to the reference as those float64
values), so the file stores them in 4 bytes; the outputs at every policy step are float64.

File "R1PR" v1 (little-endian):
    header   magic b"R1PR", uint32 version 1, n_ticks, n_policy, n_obs, n_params
    params   float64[n_params]: r1sim.v3_core.PARAMS_LAYOUT (the r1rt_policy_params fields, in order)
    ticks    n_ticks x (float64 t, float64 lean_request_rad (NaN = none), float32 q26, dq26, quat4, gyro3, cmd3)
    policy   n_policy x float64 (tick, obs[n_obs], target[n_act], last_action[n_act], lean, phase_t)
Read by rt/tests/test_policy_runner.cpp (ctest, <= 1e-12) and tests/unit/test_policy_runner.py (regenerates it bit
for bit and checks the C++ runner through the binding on the same sequence in float64).
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO / "python"))

from r1mc3.policy import POLICY_DIR, ReferencePolicy  # noqa: E402
from r1sim import joints as J  # noqa: E402
from r1sim.v3_core import pack_params, policy_params  # noqa: E402

OUT = REPO / "tests" / "assets" / "policy_runner" / "sequence_1600.r1pr"
MAGIC = b"R1PR"
VERSION = 1
N_TICKS = 1600
HEADER = struct.Struct("<4s5I")


def _state(k: int) -> dict[str, Any]:
    """= tests/unit/test_policy_reference.py::_state (S2), rounded to float32."""
    rng = np.random.default_rng(k)
    quat = np.array([1.0, 0.02 * np.sin(k / 30), 0.03 * np.cos(k / 40), 0.01])
    s = {
        "q": J.POSE_STAND + 0.05 * rng.normal(size=26),
        "dq": 0.5 * rng.normal(size=26),
        "quat": quat / np.linalg.norm(quat),
        "gyro": 0.2 * rng.normal(size=3),
    }
    return {k2: v.astype(np.float32).astype(np.float64) for k2, v in s.items()}


def _cmd_script(k: int) -> np.ndarray:
    """= tests/unit/test_policy_reference.py::_cmd_script (S2)."""
    t = k * 0.002
    if t < 0.6:
        return np.zeros(3)
    if t < 1.0:
        return np.array([0.05, 0.03, 0.0])
    if t < 1.8:
        return np.array([0.9, -0.5, 0.0])
    if t < 2.4:
        return np.array([0.0, 0.0, 0.6])
    return np.zeros(3)


class _Recorder:
    """Wraps the policy network: records every observation it is called with."""

    def __init__(self, net: Any) -> None:
        self.net = net
        self.meta = net.meta
        self.obs: list[np.ndarray] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.obs.append(np.array(x, dtype=np.float64))
        out: np.ndarray = self.net(x)
        return out


def build() -> tuple[bytes, dict[str, Any]]:
    pol = ReferencePolicy.load(POLICY_DIR)
    rec = _Recorder(pol.net)
    pol.net = rec  # type: ignore[assignment]
    params = pack_params(policy_params(pol))
    rng = np.random.default_rng(7)
    t = 0.0
    ticks = bytearray()
    policy = bytearray()
    lean_req = {100: np.radians(3.0), 1300: np.radians(7.0)}  # 7 deg: refused (outside 0..5 deg)
    for k in range(N_TICKS):
        req = float(lean_req.get(k, np.nan))
        if k in lean_req:
            pol.set_body_targets(lean_deg=float(np.degrees(req)))
        s, c = _state(k), _cmd_script(k).astype(np.float32).astype(np.float64)  # stored as float32 too
        n0 = pol.policy_steps
        pol.step(s, c, t)
        ticks += struct.pack("<dd", t, req)
        ticks += np.concatenate([s["q"], s["dq"], s["quat"], s["gyro"], c]).astype("<f4").tobytes()
        if pol.policy_steps != n0:
            row = np.concatenate([[float(k)], rec.obs[-1], pol.target, pol.last_action, [pol.lean, pol.phase_t]])
            policy += row.astype("<f8").tobytes()
        t += 0.002 + float(rng.uniform(-2e-4, 2e-4))
    n_policy = pol.policy_steps
    head = HEADER.pack(MAGIC, VERSION, N_TICKS, n_policy, len(rec.obs[0]), len(params))
    data = head + params.astype("<f8").tobytes() + bytes(ticks) + bytes(policy)
    return data, {"n_policy": n_policy, "n_obs": len(rec.obs[0]), "n_params": len(params)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    data, info = build()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if out.resolve() == OUT.resolve():
        (OUT.parent / "SHA256SUMS").write_text(f"{digest}  {OUT.name}\n")
    print(
        f"policy_runner_golden: {out}: {N_TICKS} ticks, {info['n_policy']} policy steps, {len(data)} bytes, "
        f"sha256 {digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
