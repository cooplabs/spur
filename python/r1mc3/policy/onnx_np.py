"""A tiny ONNX reader + NumPy executor for the feed-forward graphs mjlab exports (obs normaliser + MLP).

Ported from r1mc2 (v2) r1mc2/policy.py NumpyOnnx (lines 53-150) at git r21_try 2a4a165; changes: the model file is
read by our own minimal protobuf wire-format reader instead of the `onnx` package (so the simulator and the demo need
neither onnx nor onnxruntime; `scripts/demo.sh` runs without the dev dependencies), typed, no npz export. The
evaluation is v2's, operation for operation, in float64 - the arithmetic is kept identical so the outputs are
bit-identical to v2's executor (tests/unit/test_policy_onnx.py), which is the golden reference of DESIGN.md §3.2.

Only what these graphs need is decoded: ModelProto.metadata_props / graph; GraphProto nodes, initializers, input and
output names and dims; NodeProto inputs, outputs, op_type, attributes (float, int, string, tensor, floats, ints);
TensorProto dims, data_type, raw_data / float_data / double_data / int32_data / int64_data. Field numbers from
onnx/onnx.proto (ONNX IR version 8, opset 18 - what this policy was exported with).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# TensorProto.DataType values we decode (onnx.proto)
_DTYPES: dict[int, Any] = {1: np.float32, 6: np.int32, 7: np.int64, 11: np.float64}


# ---------------------------------------------------------------- protobuf wire format
def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7
        if shift > 63:
            raise ValueError("protobuf: varint too long")


def _fields(buf: bytes) -> list[tuple[int, int, Any]]:
    """(field number, wire type, value) for every field of one message; value = int (varint / fixed) or bytes."""
    out: list[tuple[int, int, Any]] = []
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fno, wt = key >> 3, key & 7
        v: int | bytes
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v = buf[i : i + 8]
            i += 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v = buf[i : i + ln]
            i += ln
        elif wt == 5:
            v = buf[i : i + 4]
            i += 4
        else:
            raise ValueError(f"protobuf: unsupported wire type {wt} (field {fno})")
        if i > n:
            raise ValueError("protobuf: truncated message")
        out.append((fno, wt, v))
    return out


def _signed64(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


def _ints(wt: int, v: Any) -> list[int]:
    """A repeated int64 field: one varint, or a packed run of varints."""
    if wt == 0:
        return [_signed64(int(v))]
    vals, i, b = [], 0, bytes(v)
    while i < len(b):
        x, i = _varint(b, i)
        vals.append(_signed64(x))
    return vals


def _floats(wt: int, v: Any) -> list[float]:
    """A repeated float field: one fixed32, or a packed run."""
    b = bytes(v)
    return list(struct.unpack(f"<{len(b) // 4}f", b))


def _doubles(wt: int, v: Any) -> list[float]:
    b = bytes(v)
    return list(struct.unpack(f"<{len(b) // 8}d", b))


def _str(v: Any) -> str:
    return bytes(v).decode("utf-8")


# ---------------------------------------------------------------- ONNX messages
def parse_tensor(buf: bytes) -> tuple[str, npt.NDArray[Any]]:
    """TensorProto -> (name, array in its stored dtype and shape)."""
    dims: list[int] = []
    dtype = 0
    name = ""
    raw: bytes | None = None
    f32: list[float] = []
    f64: list[float] = []
    i32: list[int] = []
    i64: list[int] = []
    for fno, wt, v in _fields(buf):
        if fno == 1:
            dims += _ints(wt, v)
        elif fno == 2:
            dtype = int(v)
        elif fno == 4:
            f32 += _floats(wt, v)
        elif fno == 5:
            i32 += _ints(wt, v)
        elif fno == 7:
            i64 += _ints(wt, v)
        elif fno == 8:
            name = _str(v)
        elif fno == 9:
            raw = bytes(v)
        elif fno == 10:
            f64 += _doubles(wt, v)
        elif fno in (13, 14):
            raise ValueError(f"tensor {name!r}: external data is not supported")
    if dtype not in _DTYPES:
        raise ValueError(f"tensor {name!r}: unsupported data type {dtype}")
    np_t = np.dtype(_DTYPES[dtype]).newbyteorder("<")
    if raw is not None:
        arr = np.frombuffer(raw, dtype=np_t).astype(_DTYPES[dtype])
    elif dtype == 1:
        arr = np.array(f32, dtype=np.float32)
    elif dtype == 11:
        arr = np.array(f64, dtype=np.float64)
    elif dtype == 6:
        arr = np.array(i32, dtype=np.int32)
    else:
        arr = np.array(i64, dtype=np.int64)
    return name, arr.reshape(dims) if dims else arr.reshape(())


def _parse_attribute(buf: bytes) -> tuple[str, Any]:
    name, kind = "", 0
    f: float | None = None
    i: int | None = None
    s: bytes | None = None
    t: npt.NDArray[Any] | None = None
    floats: list[float] = []
    ints: list[int] = []
    for fno, wt, v in _fields(buf):
        if fno == 1:
            name = _str(v)
        elif fno == 2:
            f = struct.unpack("<f", bytes(v))[0]
        elif fno == 3:
            i = _signed64(int(v))
        elif fno == 4:
            s = bytes(v)
        elif fno == 5:
            t = parse_tensor(bytes(v))[1]
        elif fno == 7:
            floats += _floats(wt, v)
        elif fno == 8:
            ints += _ints(wt, v)
        elif fno == 20:
            kind = int(v)
    # AttributeProto.AttributeType: FLOAT 1, INT 2, STRING 3, TENSOR 4, FLOATS 6, INTS 7
    value: Any = {1: f, 2: i, 3: s, 4: t, 6: floats, 7: ints}.get(kind)
    if value is None:
        raise ValueError(f"attribute {name!r}: unsupported type {kind}")
    return name, value


@dataclass
class OnnxNode:
    op_type: str
    input: list[str]
    output: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


def _parse_node(buf: bytes) -> OnnxNode:
    n = OnnxNode("", [], [])
    for fno, _wt, v in _fields(buf):
        if fno == 1:
            n.input.append(_str(v))
        elif fno == 2:
            n.output.append(_str(v))
        elif fno == 4:
            n.op_type = _str(v)
        elif fno == 5:
            k, a = _parse_attribute(bytes(v))
            n.attrs[k] = a
    return n


def _parse_value_info(buf: bytes) -> tuple[str, list[int]]:
    """ValueInfoProto -> (name, dims) (a symbolic dim reads 0)."""
    name, dims = "", []
    for fno, _wt, v in _fields(buf):
        if fno == 1:
            name = _str(v)
        elif fno == 2:  # TypeProto
            for f2, _w2, v2 in _fields(bytes(v)):
                if f2 != 1:  # tensor_type
                    continue
                for f3, _w3, v3 in _fields(bytes(v2)):
                    if f3 != 2:  # shape
                        continue
                    for f4, _w4, v4 in _fields(bytes(v3)):
                        if f4 != 1:  # dim
                            continue
                        dv = 0
                        for f5, _w5, v5 in _fields(bytes(v4)):
                            if f5 == 1:
                                dv = int(v5)
                        dims.append(dv)
    return name, dims


@dataclass
class OnnxGraph:
    meta: dict[str, str]
    initializers: dict[str, npt.NDArray[Any]]
    nodes: list[OnnxNode]  # every node, Constant nodes included
    inputs: list[tuple[str, list[int]]]
    outputs: list[tuple[str, list[int]]]


def parse_model(data: bytes) -> OnnxGraph:
    """ModelProto bytes -> the parts the executor needs."""
    meta: dict[str, str] = {}
    graph: bytes | None = None
    for fno, _wt, v in _fields(data):
        if fno == 7:
            graph = bytes(v)
        elif fno == 14:
            k = val = ""
            for f2, _w2, v2 in _fields(bytes(v)):
                if f2 == 1:
                    k = _str(v2)
                elif f2 == 2:
                    val = _str(v2)
            meta[k] = val
    if graph is None:
        raise ValueError("ONNX model without a graph")
    g = OnnxGraph(meta, {}, [], [], [])
    for fno, _wt, v in _fields(graph):
        if fno == 1:
            g.nodes.append(_parse_node(bytes(v)))
        elif fno == 5:
            name, arr = parse_tensor(bytes(v))
            g.initializers[name] = arr
        elif fno == 11:
            g.inputs.append(_parse_value_info(bytes(v)))
        elif fno == 12:
            g.outputs.append(_parse_value_info(bytes(v)))
    return g


# ---------------------------------------------------------------- the executor (v2 NumpyOnnx.__call__)
class OnnxModel:
    """Evaluates the small feed-forward graphs mjlab exports (Sub, Div, Gemm, Elu, ...) in float64."""

    def __init__(self, graph: OnnxGraph) -> None:
        self.meta: dict[str, str] = dict(graph.meta)
        self.consts: dict[str, FloatArray] = {k: v.astype(np.float64) for k, v in graph.initializers.items()}
        for n in graph.nodes:
            if n.op_type == "Constant":
                self.consts[n.output[0]] = np.asarray(next(iter(n.attrs.values()))).astype(np.float64)
        self.nodes: list[OnnxNode] = [n for n in graph.nodes if n.op_type != "Constant"]
        self.inp: str = graph.inputs[0][0]
        self.out: str = graph.outputs[0][0]
        self.n_in: int = graph.inputs[0][1][-1]
        self.n_out: int = graph.outputs[0][1][-1]

    @classmethod
    def load(cls, path: str | Path) -> OnnxModel:
        return cls(parse_model(Path(path).read_bytes()))

    def __call__(self, x: npt.ArrayLike) -> FloatArray:
        v: dict[str, Any] = dict(self.consts)
        v[self.inp] = np.asarray(x, np.float64).reshape(1, -1)
        for n in self.nodes:
            at = n.attrs
            a = [v[i] for i in n.input if i]
            op = n.op_type
            r: Any
            if op == "Sub":
                r = a[0] - a[1]
            elif op == "Add":
                r = a[0] + a[1]
            elif op == "Mul":
                r = a[0] * a[1]
            elif op == "Div":
                r = a[0] / a[1]
            elif op == "Gemm":
                A = a[0].T if at.get("transA", 0) else a[0]
                B = a[1].T if at.get("transB", 0) else a[1]
                r = at.get("alpha", 1.0) * A @ B
                if len(a) > 2:
                    r = r + at.get("beta", 1.0) * a[2]
            elif op == "MatMul":
                r = a[0] @ a[1]
            elif op == "Elu":
                al = at.get("alpha", 1.0)
                r = np.where(a[0] > 0, a[0], al * (np.exp(np.minimum(a[0], 0)) - 1))
            elif op == "Relu":
                r = np.maximum(a[0], 0)
            elif op == "Tanh":
                r = np.tanh(a[0])
            elif op == "Clip":
                lo = a[1] if len(a) > 1 else -np.inf
                hi = a[2] if len(a) > 2 else np.inf
                r = np.clip(a[0], lo, hi)
            elif op in ("Identity", "Cast"):
                r = a[0]
            elif op == "Reshape":
                r = a[0].reshape(a[1].astype(int))
            else:
                raise NotImplementedError(f"ONNX op {op}")
            v[n.output[0]] = r
        return np.asarray(v[self.out], dtype=np.float64).reshape(-1)
