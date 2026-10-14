"""r1mc3.client (session 10): the api agent's new SDK methods send requests r1d's schemas accept; a lease taken
through the SDK rides on every later request until released (base a041731: the methods did not exist)."""

from __future__ import annotations

from typing import Any

import pytest
from r1mc3.client import R1
from r1mc3.service.schemas import OPS, REQUEST

from tests.unit.test_client_sdk import _Rec

CALLS = [
    ("move_path", ("left", [[0.3, 0.2, 0.8], [0.32, 0.2, 0.8]]), {"linear": True}),
    ("guarded_move", ("left",), {"delta": [0.0, 0.0, -0.02], "until": ["blocked"]}),
    ("guarded_move", ("right",), {"stop_z_m": 0.75, "until": [], "touch": ["box"]}),
    ("slide", ("left", [0.03, 0.0, 0.0]), {"support": "table"}),
    ("pick", ("left", [0.35, 0.2, 0.78]), {"height_m": 0.06, "width_m": 0.04, "object_id": "box"}),
    ("place", ("left", [0.3, -0.1, 0.8]), {}),
    ("par", ([{"op": "arm.move_joint", "arm": "left", "q": [0, 0, 0, 0.5, 0]}],), {}),
    ("lease_acquire", (["arm_left", "hand_left"],), {"client": "me"}),
    ("leases", (), {}),
    ("twist", ("left", [0.02, 0.0, 0.0]), {}),
    ("calibrate_pinch", ("right",), {}),  # grasp's hand.calibrate (merged s10_grasp 057190a)
]


@pytest.mark.parametrize(("name", "args", "kw"), CALLS, ids=[f"{c[0]}{i}" for i, c in enumerate(CALLS)])
def test_every_new_method_sends_a_valid_request(name: str, args: tuple[Any, ...], kw: dict[str, Any]) -> None:
    t = _Rec()
    getattr(R1(transport=t), name)(*args, **kw)
    REQUEST.validate_python(t.sent[-1])


class _LeaseRec(_Rec):
    def call(self, op: str, **kw: Any) -> dict[str, Any]:
        out = super().call(op, **kw)
        if op == "lease.acquire":
            out["data"] = {"lease_id": "L7"}
        return out


def test_the_lease_rides_on_every_request_until_released() -> None:
    t = _LeaseRec()
    r = R1(transport=t)
    r.lease_acquire(["arm_left"])
    r.move_joint("left", [0.2, 0.3, 0, 0.5, 0])
    assert t.sent[-1]["lease"] == "L7"
    REQUEST.validate_python(t.sent[-1])
    r.lease_renew(10.0)
    assert t.sent[-1]["lease_id"] == "L7"
    r.lease_release()
    assert t.sent[-1]["op"] == "lease.release" and t.sent[-1]["lease_id"] == "L7"
    r.status()
    assert "lease" not in t.sent[-1]


TAG = {"id": 3, "size_m": 0.06, "xyz": [0.1, 0.1, 0.0]}
OTHERS = [  # the other agents' ops (cam, measure): validated once their request models are in this tree
    ("observe", (), {}),
    ("point", ("o1", 270.0, 300.0), {}),
    ("object_extent", ("o1", 270.0, 300.0), {}),
    ("project", ("o1", [[0.4, 0.0, 0.75]]), {}),
    ("measure_setup", ([{"id": "table", "tags": [TAG]}],), {"frames": 2}),
    ("measure_phone", ("A" * 32, {"f_px": 3000.0, "width": 4032, "height": 3024}, [TAG, TAG, TAG]), {}),
]


@pytest.mark.parametrize(("name", "args", "kw"), OTHERS, ids=[c[0] for c in OTHERS])
def test_the_other_agents_ops_send_valid_requests(name: str, args: tuple[Any, ...], kw: dict[str, Any]) -> None:
    t = _Rec()
    getattr(R1(transport=t), name)(*args, **kw)
    if t.sent[-1]["op"] not in OPS:
        pytest.skip(f"{t.sent[-1]['op']} is not in this tree yet (cam / measure merged later)")
    REQUEST.validate_python(t.sent[-1])
