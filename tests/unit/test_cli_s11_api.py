"""Session 11 (api): `r1 move-path ... --blend M` and the SDK's move_path(blend_m=...) send arm.move_path's blend_m
(base 1e55bdd: argparse refused --blend; the SDK had no blend_m; r1d's schema refused the field)."""

from __future__ import annotations

from typing import Any

import pytest
from r1mc3.cli.main import build_request
from r1mc3.client import R1
from r1mc3.service.schemas import REQUEST

from tests.unit.test_client_sdk import _Rec


def test_move_path_blend_word() -> None:
    req, _ = build_request(["move-path", "left", "[[0.3, 0.2, 0.8], [0.32, 0.2, 0.8]]", "--blend", "0.02"])
    assert req["op"] == "arm.move_path" and req["blend_m"] == 0.02
    assert REQUEST.validate_python(req).blend_m == 0.02  # type: ignore[union-attr]
    req, _ = build_request(["move-path", "left", "[[0.3, 0.2, 0.8]]"])
    assert "blend_m" not in req  # unblended: the request an older r1d accepts


@pytest.mark.parametrize("blend", [0.0, 0.03])
def test_sdk_move_path_blend(blend: float) -> None:
    t = _Rec()
    R1(transport=t).move_path("left", [[0.3, 0.2, 0.8], [0.32, 0.2, 0.8]], blend_m=blend)
    sent: dict[str, Any] = t.sent[-1]
    assert ("blend_m" in sent) is (blend > 0)
    REQUEST.validate_python(sent)
