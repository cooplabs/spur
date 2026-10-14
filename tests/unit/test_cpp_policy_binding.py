"""The Python binding of the C++ policy runner (module r1rt_py, rt/bindings; r1sim.cpp_policy) gives the C++ runner's
numbers: the golden set (v2's NumPy executor) within 1e-12, the C++ CLI on the same file, our Python executor on
random vectors, inside ReferencePolicy; it refuses bad input with clear errors and is safe to call from threads.
Skipped (with the reason) when the module is not built - scripts/ci.sh builds it in its C++ stage, before pytest."""

from __future__ import annotations

import struct
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3.policy import POLICY_DIR, OnnxModel, ReferencePolicy
from r1sim import cpp_policy

GOLDEN = cpp_policy.REPO / "tests" / "assets" / "policy_golden" / "golden_1000.bin"
TOL = 1e-12

pytestmark = pytest.mark.skipif(
    cpp_policy.find_module() is None,
    reason=f"r1rt_py not built (looked in {[str(d) for d in cpp_policy.build_dirs()]}): run scripts/ci.sh cpp",
)


def _golden() -> tuple[np.ndarray, np.ndarray, int]:
    raw = GOLDEN.read_bytes()
    magic, ver, n, n_obs, n_act, n_sim = struct.unpack_from("<4s5I", raw, 0)
    assert (magic, ver, n_obs, n_act) == (b"R1G1", 1, 74, 14)
    obs = np.frombuffer(raw, "<f8", n * n_obs, 32).reshape(n, n_obs)
    act = np.frombuffer(raw, "<f8", n * n_act, 32 + 8 * n * n_obs).reshape(n, n_act)
    return obs, act, int(n_sim)


@pytest.fixture(scope="module")
def net() -> cpp_policy.CppPolicyNet:
    return cpp_policy.CppPolicyNet(POLICY_DIR / "policy.onnx")


def test_binding_matches_the_golden_set(net: cpp_policy.CppPolicyNet) -> None:
    obs, expected, n_sim = _golden()
    got = np.array([net(o) for o in obs])
    err = np.abs(got - expected)
    print(f"\nr1rt_py vs v2's NumPy executor, {len(obs)} golden vectors ({n_sim} from the sim): max {err.max():.3e}")
    assert err.max() <= TOL


def test_binding_equals_the_cpp_cli(net: cpp_policy.CppPolicyNet) -> None:
    """Same source (policy_mlp.cpp), same flags (UBSan, -O2) as the CLI's r1policy: the same bits are expected; the
    assertion allows 1e-12 in case a compiler ever differs between the two targets."""
    cli = next((d / "r1rt_policy_eval" for d in cpp_policy.build_dirs() if (d / "r1rt_policy_eval").exists()), None)
    if cli is None:
        pytest.skip("r1rt_policy_eval not built")
    obs, _, _ = _golden()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "act.f64"
        subprocess.run([str(cli), str(net.weights_path), str(GOLDEN), str(out)], check=True, capture_output=True)
        cli_act = np.fromfile(out, "<f8").reshape(len(obs), 14)
    got = np.array([net(o) for o in obs])
    diff = float(np.abs(got - cli_act).max())
    print(f"\nr1rt_py vs r1rt_policy_eval: max {diff:.3e} (bit-identical: {diff == 0.0})")
    assert diff <= TOL


def test_binding_matches_the_python_executor_on_random_vectors(net: cpp_policy.CppPolicyNet) -> None:
    rng = np.random.default_rng(20261010)
    xs = rng.uniform(-3.0, 3.0, size=(1000, 74))
    err = max(float(np.abs(net(x) - net.reference(x)).max()) for x in xs)
    ref = OnnxModel.load(POLICY_DIR / "policy.onnx")
    assert np.array_equal(net.reference(xs[0]), ref(xs[0]))  # reference() is our executor, unchanged
    print(f"\nr1rt_py vs OnnxModel, 1,000 random vectors: max {err:.3e}")
    assert err <= TOL


def test_cpp_net_inside_reference_policy() -> None:
    py = ReferencePolicy.load(POLICY_DIR)
    cc = cpp_policy.load_cpp_policy(POLICY_DIR)
    assert isinstance(cc.net, cpp_policy.CppPolicyNet)
    assert cc.sha256 == py.sha256 == cc.net.onnx_sha256
    rng = np.random.default_rng(3)
    q = py.default + rng.normal(0.0, 0.02, 24)
    dq = rng.normal(0.0, 0.3, 24)
    quat, gyro = np.array([0.999, 0.01, -0.02, 0.0]) / np.linalg.norm([0.999, 0.01, -0.02, 0.0]), np.zeros(3)
    for cmd in ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0)):
        assert np.abs(cc.act(q, dq, quat, gyro, cmd) - py.act(q, dq, quat, gyro, cmd)).max() <= TOL


def test_errors_are_clear(net: cpp_policy.CppPolicyNet) -> None:
    mod: Any = cpp_policy.import_r1rt_py()
    p = mod.PolicyMlp()
    assert not p.loaded
    with pytest.raises(RuntimeError, match="no weights loaded"):
        p.run(np.zeros(74))
    with pytest.raises(RuntimeError, match="cannot open"):
        p.load("/nonexistent/policy_weights.r1w")
    with pytest.raises(TypeError):
        net.mlp.run(np.zeros(73))  # wrong shape: nanobind refuses it
    out = np.zeros(13)
    with pytest.raises(TypeError):
        net.mlp.run_into(np.zeros(74), out)


def test_weights_from_another_onnx_are_refused(tmp_path: Path, net: cpp_policy.CppPolicyNet) -> None:
    raw = bytearray(net.weights_path.read_bytes())
    raw[40] ^= 0xFF  # the ONNX sha256 recorded by the converter (R1W1 header offset 40)
    bad = tmp_path / "other.r1w"
    bad.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="was converted from ONNX"):
        cpp_policy.CppPolicyNet(POLICY_DIR / "policy.onnx", weights=bad)


def test_concurrent_calls_give_the_serial_results(net: cpp_policy.CppPolicyNet) -> None:
    """run() releases the GIL and PolicyMlp keeps no mutable state: four threads on one instance give the same bits
    as one thread."""
    xs = np.random.default_rng(5).uniform(-2.0, 2.0, size=(200, 74))
    serial = np.array([net(x) for x in xs])
    results: dict[int, np.ndarray] = {}

    def work(k: int) -> None:
        results[k] = np.array([net.mlp.run(x) for x in xs])

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for k in range(4):
        assert np.array_equal(results[k], serial)
