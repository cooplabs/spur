"""r1mc3.client (session 9): every SDK method sends a request r1d's schemas accept, and Reply / raise_for_error."""

from __future__ import annotations

from typing import Any

import pytest
from r1mc3.client import R1, R1Refused
from r1mc3.service.schemas import REQUEST


class _Rec:
    """A transport that records the request and answers OK (or a refusal for op 'arm.reach')."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.timeout_s = 20.0

    def call(self, op: str, **kw: Any) -> dict[str, Any]:
        req = {"op": op, **kw}
        self.sent.append(req)
        if op == "arm.reach":
            return {"ok": False, "code": "UNREACHABLE", "op": op, "moved": False, "data": {}, "details": {"why": "x"}}
        return {"ok": True, "code": "OK", "op": op, "moved": False, "data": {"goal_id": 1}, "details": {}}

    def close(self) -> None:
        pass


CALLS = [
    ("status", (), {}),
    ("stop", (), {}),
    ("move_joint", ("left", [0.2, 0.3, 0, 0.5, 0]), {}),
    ("reach", ("left", [0.3, 0.2, 0.8]), {"prefer": "front"}),
    ("move_linear", ("right",), {"delta": [0, 0, -0.02]}),
    ("move_relative", ("right", [0.02, 0, 0]), {}),
    ("hover", ("left", [0.3, 0.2, 0.76], 0.05), {}),
    ("lower", ("left",), {"until": "depth", "max_depth_m": 0.03}),
    ("carry", ("left", [0.35, 0.1, 0.8]), {}),
    ("plan", ({"op": "arm.reach", "arm": "left", "position": [0.3, 0.2, 0.8]},), {}),
    ("arm_stop", (), {}),
    ("grasp", ("right",), {"grip": "tip", "object_class": "thin_flexible"}),
    ("release", ("right",), {}),
    ("preshape", ("left", "pinch"), {}),
    ("tare", ("left",), {}),
    ("look_at", ([1.0, 0.2, 0.8],), {}),
    ("lean", (3.0,), {}),
    ("skill", ("table_ready",), {"arms": "left"}),
    ("seq", ([{"op": "status"}],), {}),
    ("goal_status", (1,), {}),
    ("goal_wait", (1,), {}),
    ("goal_cancel", (1,), {}),
    ("goals", (), {}),
    ("world_declare", ({"kind": "surface", "id": "t", "pose": {"pos": [0.5, 0, 0.7]}, "size": [0.4, 1, 0.04]},), {}),
    ("world_list", (), {}),
    ("world_clear", ("t",), {}),
    ("joint_hold", ("arm_left",), {}),
    ("jog", ("head", [0.1, 0.0]), {}),
]


@pytest.mark.parametrize(("name", "args", "kw"), CALLS, ids=[c[0] for c in CALLS])
def test_every_method_sends_a_valid_request(name: str, args: tuple[Any, ...], kw: dict[str, Any]) -> None:
    t = _Rec()
    getattr(R1(transport=t), name)(*args, **kw)
    REQUEST.validate_python(t.sent[-1])


def test_reply_and_raise_for_error() -> None:
    r = R1(transport=_Rec())
    assert r.status().ok
    bad = r.reach("left", [2.0, 0, 0])
    assert not bad.ok and bad.code == "UNREACHABLE" and bad.details == {"why": "x"}
    with pytest.raises(R1Refused) as e:
        bad.raise_for_error()
    assert e.value.reply.code == "UNREACHABLE"
    assert not hasattr(r, "damp")  # the one limp state stays the operator's word (r1 damp)
