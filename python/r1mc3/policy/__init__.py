"""The balance policy: a dependency-free ONNX executor and the Python reference runner (v2's WalkPolicy).

The C++ runner of the RT core (rt/, DESIGN.md §3.2) must match the reference executor within 1e-5 on the golden set;
the reference itself is bit-identical to v2's NumPy executor (tests/unit/test_policy_*.py)."""

from __future__ import annotations

from pathlib import Path

from .onnx_np import OnnxModel
from .reference import ReferencePolicy

#: The live policy (R1MC-Manip2-Lean, 74 observations -> 14 actions), published in policies/ with its provenance.
POLICY_DIR: Path = Path(__file__).resolve().parents[3] / "policies" / "r1mc2_manip2_lean_20261006T0421Z"

__all__ = ["POLICY_DIR", "OnnxModel", "ReferencePolicy"]
