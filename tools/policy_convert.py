"""Convert an exported balance policy (policy.onnx) into the weights file of the C++ policy runner (DESIGN.md §3.2).

    uv run python tools/policy_convert.py policies/<name>/policy.onnx out/policy_weights.r1w

The ONNX file is read with the `onnx` package and every initializer cast to float64, exactly as v2's NumPy executor
reads it (r1mc2 (v2) r1mc2/policy.py:53-66 NumpyOnnx.__init__ at git r21_try 2a4a165). The graph must be exactly the
chain mjlab / rsl_rl exports for an observation-normalised 3-hidden-layer ELU MLP:

    obs -> Sub(mean) -> Div(std) -> Gemm -> Elu -> Gemm -> Elu -> Gemm -> Elu -> Gemm -> actions

with Gemm alpha = beta = 1, transA = 0 (transB 0 or 1) and one Elu alpha; anything else is refused, because the C++
runner (rt/src/policy_mlp.cpp) implements only that chain. The values are stored unchanged (no folding of the
normaliser into the first layer: the runner computes (obs - sub) / div exactly as the graph does).

Weights file "R1W1" (little-endian; read by r1rt::PolicyMlp::load):

    offset  size  field
    0       4     magic "R1W1"
    4       4     uint32 version = 1
    8       20    uint32 dims[5] = n_obs, n_h0, n_h1, n_h2, n_act   (74, 512, 256, 128, 14 for R1MC-Manip2-Lean)
    28      4     uint32 activation = 1 (ELU)
    32      8     float64 elu_alpha
    40      32    uint8 sha256 of the source ONNX file
    72      8     reserved (zero)
    80      ...   float64 arrays in this order: sub[n_obs], div[n_obs], W0[n_h0][n_obs], b0[n_h0],
                  W1[n_h1][n_h0], b1[n_h1], W2[n_h2][n_h1], b2[n_h2], W3[n_act][n_h2], b3[n_act]
                  (W row-major as [out][in]: y = W x + b)

The file is generated at build / install time from the published ONNX (never committed).
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"R1W1"
VERSION = 1
ACT_ELU = 1
HEADER = struct.Struct("<4sI5IId32s8s")  # 80 bytes
EXPECTED_OPS = ("Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm")


class ConvertError(ValueError):
    """The ONNX graph is not the chain the C++ runner implements."""


def _attrs(node: Any) -> dict[str, Any]:
    import onnx

    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def read_onnx(path: str | Path) -> dict[str, Any]:
    """The policy's parameters from the ONNX graph, as float64 (v2 NumpyOnnx semantics), after checking the graph.

    Returns dict(sub, div, layers=[(W [out, in], b [out]), ...], elu_alpha, sha256, meta)."""
    import onnx
    from onnx import numpy_helper

    raw = Path(path).read_bytes()
    m = onnx.load_from_string(raw)
    g = m.graph
    consts: dict[str, np.ndarray] = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t).astype(np.float64)
    nodes = [n for n in g.node if n.op_type != "Constant"]
    ops = tuple(n.op_type for n in nodes)
    if ops != EXPECTED_OPS:
        raise ConvertError(f"unexpected op chain {ops}; the runner implements {EXPECTED_OPS}")
    if len(g.input) != 1 or len(g.output) != 1:
        raise ConvertError("the graph must have exactly one input and one output")
    # The chain must be connected: each node consumes the previous node's output (the first one the graph input).
    prev = g.input[0].name
    for n in nodes:
        if n.input[0] != prev:
            raise ConvertError(f"{n.op_type} {n.name!r} does not consume the previous output {prev!r}")
        for extra in n.input[1:]:
            if extra and extra not in consts:
                raise ConvertError(f"{n.op_type} input {extra!r} is not a constant")
        prev = n.output[0]
    if prev != g.output[0].name:
        raise ConvertError("the last node does not produce the graph output")

    sub = consts[nodes[0].input[1]].reshape(-1)
    div = consts[nodes[1].input[1]].reshape(-1)
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    alphas: list[float] = []
    for n in nodes[2:]:
        at = _attrs(n)
        if n.op_type == "Elu":
            alphas.append(float(at.get("alpha", 1.0)))
            continue
        if float(at.get("alpha", 1.0)) != 1.0 or float(at.get("beta", 1.0)) != 1.0 or int(at.get("transA", 0)):
            raise ConvertError(f"Gemm {n.name!r}: only alpha = beta = 1 and transA = 0 are supported, got {at}")
        if len(n.input) < 3 or not n.input[2]:
            raise ConvertError(f"Gemm {n.name!r} has no bias")
        w = consts[n.input[1]]
        w = w if int(at.get("transB", 0)) else w.T  # store as [out, in]
        b = consts[n.input[2]].reshape(-1)
        if w.ndim != 2 or b.shape != (w.shape[0],):
            raise ConvertError(f"Gemm {n.name!r}: weight {w.shape} and bias {b.shape} do not fit")
        layers.append((np.ascontiguousarray(w, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64)))
    if len(set(alphas)) != 1:
        raise ConvertError(f"the Elu nodes use different alphas {alphas}")
    dims = [layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers]
    if sub.shape != (dims[0],) or div.shape != (dims[0],):
        raise ConvertError(f"normaliser shapes {sub.shape} / {div.shape} do not match {dims[0]} observations")
    for (w, _), n_in in zip(layers, dims[:-1], strict=True):
        if w.shape[1] != n_in:
            raise ConvertError(f"layer input {w.shape[1]} != previous output {n_in}")
    return dict(
        sub=sub,
        div=div,
        layers=layers,
        elu_alpha=alphas[0],
        dims=dims,
        sha256=hashlib.sha256(raw).digest(),
        meta={p.key: p.value for p in m.metadata_props},
    )


def to_bytes(p: dict[str, Any]) -> bytes:
    """The R1W1 file content for read_onnx's result."""
    dims = p["dims"]
    if len(dims) != 5:
        raise ConvertError(f"the runner needs 3 hidden layers, got dims {dims}")
    head = HEADER.pack(MAGIC, VERSION, *dims, ACT_ELU, float(p["elu_alpha"]), p["sha256"], b"\0" * 8)
    parts = [head, p["sub"].astype("<f8").tobytes(), p["div"].astype("<f8").tobytes()]
    for w, b in p["layers"]:
        parts += [w.astype("<f8").tobytes(order="C"), b.astype("<f8").tobytes()]
    return b"".join(parts)


def mlp_forward(p: dict[str, Any], obs: np.ndarray) -> np.ndarray:
    """Reference forward pass on read_onnx's result (float64), for self-checks."""
    x = (np.asarray(obs, dtype=np.float64) - p["sub"]) / p["div"]
    a = float(p["elu_alpha"])
    n = len(p["layers"])
    for k, (w, b) in enumerate(p["layers"]):
        x = w @ x + b
        if k < n - 1:
            x = np.where(x > 0, x, a * (np.exp(np.minimum(x, 0)) - 1))
    return np.asarray(x)


def convert(onnx_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    p = read_onnx(onnx_path)
    data = to_bytes(p)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("onnx", help="policy.onnx")
    ap.add_argument("out", help="output weights file (.r1w)")
    a = ap.parse_args(argv)
    try:
        p = convert(a.onnx, a.out)
    except ConvertError as e:
        print(f"policy_convert: REFUSED: {e}", file=sys.stderr)
        return 2
    print(
        f"policy_convert: {a.onnx} -> {a.out}: dims {p['dims']}, elu alpha {p['elu_alpha']}, "
        f"onnx sha256 {p['sha256'].hex()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
