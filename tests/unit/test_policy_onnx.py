"""r1mc3.policy.onnx_np: our own ONNX reader and NumPy executor.

The reader is checked against the `onnx` package (a dev dependency, never needed at run time). (The executor's
bit-for-bit equality with the previous controller's executor is an internal validation.)"""

from __future__ import annotations

import numpy as np
import pytest
from r1mc3.policy import POLICY_DIR, OnnxModel
from r1mc3.policy.onnx_np import parse_model

ONNX = POLICY_DIR / "policy.onnx"


def test_policy_file_is_the_published_one() -> None:
    import hashlib

    assert hashlib.sha256(ONNX.read_bytes()).hexdigest() == (
        "09bf2fd8b8d11d6219ca8a4323f1b4e7ef238bfe88d2bf330579c7e1a833b9be"
    )


def test_reader_equals_the_onnx_package() -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper

    ref = onnx.load(str(ONNX))
    g = parse_model(ONNX.read_bytes())
    assert g.meta == {p.key: p.value for p in ref.metadata_props}
    ref_init = {t.name: numpy_helper.to_array(t) for t in ref.graph.initializer}
    assert set(g.initializers) == set(ref_init)
    for k, v in ref_init.items():
        assert g.initializers[k].dtype == v.dtype and g.initializers[k].shape == v.shape, k
        assert np.array_equal(g.initializers[k], v), k  # bit-equal
    assert [n.op_type for n in g.nodes] == [n.op_type for n in ref.graph.node]
    for mine, theirs in zip(g.nodes, ref.graph.node, strict=True):
        assert mine.input == list(theirs.input) and mine.output == list(theirs.output)
        assert mine.attrs == {a.name: helper.get_attribute_value(a) for a in theirs.attribute}
    assert g.inputs == [("obs", [1, 74])] and g.outputs == [("actions", [1, 14])]


def test_executor_shapes_and_graph() -> None:
    m = OnnxModel.load(ONNX)
    assert (m.n_in, m.n_out) == (74, 14)
    assert [n.op_type for n in m.nodes] == ["Sub", "Div"] + ["Gemm", "Elu"] * 3 + ["Gemm"]
    a = m(np.zeros(74))
    assert a.shape == (14,) and a.dtype == np.float64 and np.all(np.isfinite(a))
    x = np.random.default_rng(3).normal(size=74)
    assert np.array_equal(m(x), m(x))  # deterministic


def test_reader_rejects_garbage() -> None:
    with pytest.raises((ValueError, IndexError)):
        parse_model(b"\x0f\xff\xff")
    with pytest.raises(ValueError):
        parse_model(b"")  # no graph
