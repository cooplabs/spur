"""The C++ policy runner (r1rt::PolicyMlp, DESIGN.md §3.2) in the simulator, through its Python binding `r1rt_py`
(rt/bindings, nanobind; built by the C++ stage of scripts/ci.sh).

    from r1sim.cpp_policy import load_cpp_policy
    policy = load_cpp_policy()                 # a ReferencePolicy whose network is the C++ MLP
    ctrl = PolicyController(policy)            # v2's observation / schedule logic around the RT core's network

`CppPolicyNet` is an `OnnxModel` (same metadata, same call signature: (74,) float64 -> (14,) float64) whose
`__call__` runs the C++ network, so it drops into `ReferencePolicy(net=...)` unchanged; the Python executor stays
available as `CppPolicyNet.reference(x)` for comparisons. The golden test bounds the difference (max |C++ - v2|
2.8e-14 on 10,000 vectors, tests/unit/test_policy_golden.py); trajectories stay within that over a 10 s stand
(tests/integration/test_cpp_policy_stand.py).

Where things come from:
* the module: the first `r1rt_py*.so` in $R1MC3_BUILD_DIR/rt, then <repo>/build/rt (the trees scripts/ci.sh builds);
* the weights: the build's converted file (<build>/rt/policy/policy_weights.r1w) when the sha256 of the ONNX recorded
  in it matches the requested policy.onnx, else a fresh conversion with tools/policy_convert.py (needs the `onnx`
  package, a dev dependency) into a cache outside the repository ($R1MC3_CACHE_DIR, default ~/.cache/r1mc3/policy).
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt
from r1mc3.policy import POLICY_DIR, ReferencePolicy
from r1mc3.policy.onnx_np import OnnxModel, parse_model

FloatArray = npt.NDArray[np.float64]

REPO = Path(__file__).resolve().parents[2]
MODULE = "r1rt_py"
_WEIGHTS_SHA_OFFSET = 40  # R1W1 header: magic 4, version 4, dims 20, activation 4, elu_alpha 8 -> sha256[32]


def build_dirs() -> list[Path]:
    """Where the C++ build trees are, in search order."""
    out: list[Path] = []
    env = os.environ.get("R1MC3_BUILD_DIR")
    if env:
        out.append(Path(env) / "rt")
    out.append(REPO / "build" / "rt")
    return out


def find_module() -> Path | None:
    for d in build_dirs():
        hits = sorted(d.glob(f"{MODULE}*.so")) + sorted(d.glob(f"{MODULE}*.pyd"))
        if hits:
            return hits[0]
    return None


def import_r1rt_py() -> ModuleType:
    """Import the binding from the build tree (ImportError with what to do when it is not built)."""
    if MODULE in sys.modules:
        return sys.modules[MODULE]
    path = find_module()
    if path is None:
        where = ", ".join(str(d) for d in build_dirs())
        raise ImportError(
            f"{MODULE} is not built (looked in {where}): run `bash scripts/ci.sh cpp` (needs nanobind from the dev "
            "dependencies; CMake must run under `uv run`)"
        )
    loader = importlib.machinery.ExtensionFileLoader(MODULE, str(path))
    spec = importlib.util.spec_from_file_location(MODULE, path, loader=loader)
    if spec is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[MODULE] = mod
    return mod


def _sha256(path: Path) -> bytes:
    return hashlib.sha256(path.read_bytes()).digest()


def _weights_sha(path: Path) -> bytes:
    with open(path, "rb") as f:
        head = f.read(_WEIGHTS_SHA_OFFSET + 32)
    return head[_WEIGHTS_SHA_OFFSET:] if len(head) == _WEIGHTS_SHA_OFFSET + 32 and head[:4] == b"R1W1" else b""


def cache_dir() -> Path:
    env = os.environ.get("R1MC3_CACHE_DIR")
    return Path(env) / "policy" if env else Path.home() / ".cache" / "r1mc3" / "policy"


def weights_for(onnx_path: str | Path) -> Path:
    """An R1W1 weights file converted from exactly this policy.onnx (see the module docstring)."""
    onnx_path = Path(onnx_path)
    want = _sha256(onnx_path)
    for d in build_dirs():
        cand = d / "policy" / "policy_weights.r1w"
        if cand.exists() and _weights_sha(cand) == want:
            return cand
    cached = cache_dir() / f"{want.hex()}.r1w"
    if cached.exists() and _weights_sha(cached) == want:
        return cached
    spec = importlib.util.spec_from_file_location("r1mc3_policy_convert", REPO / "tools" / "policy_convert.py")
    if spec is None or spec.loader is None:
        raise ImportError("tools/policy_convert.py not found")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)  # needs the `onnx` package (dev dependency)
    cached.parent.mkdir(parents=True, exist_ok=True)
    conv.convert(onnx_path, cached)
    return cached


class CppPolicyNet(OnnxModel):
    """The policy network evaluated by the C++ runner; metadata and the Python executor from the same ONNX file."""

    def __init__(self, onnx_path: str | Path, weights: str | Path | None = None) -> None:
        onnx_path = Path(onnx_path)
        super().__init__(parse_model(onnx_path.read_bytes()))
        mod = import_r1rt_py()
        self.weights_path = Path(weights) if weights is not None else weights_for(onnx_path)
        self.mlp: Any = mod.PolicyMlp()
        self.mlp.load(str(self.weights_path))
        sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
        if self.mlp.onnx_sha256 != sha:
            raise ValueError(
                f"{self.weights_path} was converted from ONNX {self.mlp.onnx_sha256[:12]}..., not {onnx_path} "
                f"({sha[:12]}...)"
            )
        self.onnx_sha256: str = sha
        self._out = np.zeros(int(mod.K_ACT), dtype=np.float64)

    def __call__(self, x: npt.ArrayLike) -> FloatArray:
        obs = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        out = np.empty_like(self._out)
        self.mlp.run_into(obs, out)
        return out

    def reference(self, x: npt.ArrayLike) -> FloatArray:
        """The pure-Python executor (OnnxModel) on the same input, for comparisons."""
        return super().__call__(x)


def load_cpp_policy(policy_dir: str | Path = POLICY_DIR, **kw: Any) -> ReferencePolicy:
    """`ReferencePolicy.load` with the C++ network: policy.onnx (+ r1mc_policy.json) -> a runner whose net is the
    RT core's MLP (`policy.net` is the CppPolicyNet)."""
    p = Path(policy_dir)
    onnx_path = p / "policy.onnx" if p.is_dir() else p
    pol = ReferencePolicy.load(onnx_path, **kw)
    net = CppPolicyNet(onnx_path)
    pol.net = net
    return pol
