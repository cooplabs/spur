"""The `r1` words for the L2 / L3 ops (session 9): every word builds a request r1d's schemas accept (no r1d needed)."""

from __future__ import annotations

import pytest
from r1mc3.cli.main import build_request
from r1mc3.service.schemas import REQUEST

WORDS = [
    (["move-joint", "left", "0.2,0.3,0,0.5,0", "--speed", "0.3"], "arm.move_joint"),
    (["reach", "right", "0.3,-0.2,0.8", "--prefer", "front"], "arm.reach"),
    (["hover", "left", "0.3,0.2,0.76", "--height", "0.05"], "arm.hover"),
    (["lower", "left", "--until", "depth", "--depth", "0.03"], "arm.lower"),
    (["lower", "right", "--stop-z", "0.78"], "arm.lower"),
    (["carry", "left", "0.35,0.1,0.8", "--lift", "0.08", "--no-wait"], "arm.carry"),
    (["plan", '{"op": "arm.reach", "arm": "left", "position": [0.3, 0.2, 0.8]}'], "arm.plan"),
    (["arm-stop", "left"], "arm.stop"),
    (["arm-stop"], "arm.stop"),
    (["grasp", "right", "--grip", "pinch", "--class", "thin_flexible", "--width", "0.003"], "hand.grasp"),
    (["release", "left"], "hand.release"),
    (["preshape", "left", "pinch"], "hand.preshape"),
    (["tare", "right"], "hand.tare"),
    (["look-at", "1.0,0.2,0.8"], "head.look_at"),
    (["lean", "3"], "body.pose"),
    (["skill", "table_ready", "--arms", "left"], "skill.table_ready"),
    (["seq", '[{"op": "status"}, {"op": "arm.stop"}]'], "seq.run"),
    (["goal", "wait", "4"], "goal.wait"),
    (["goal", "cancel", "4"], "goal.cancel"),
    (["goals"], "goal.list"),
    (["world", "list"], "world.list"),
    (
        ["world", "declare", '{"kind": "surface", "id": "t", "pose": {"pos": [0.5, 0, 0.7]}, "size": [0.4, 1, 0.04]}'],
        "world.declare",
    ),
    (["world", "clear", "t"], "world.clear"),
]


@pytest.mark.parametrize(("argv", "op"), WORDS, ids=[" ".join(w[:2]) for w, _ in WORDS])
def test_each_word_builds_a_valid_request(argv: list[str], op: str) -> None:
    req, _ = build_request(argv)
    assert req["op"] == op and req["requester"] == "operator"
    REQUEST.validate_python(req)  # raises if r1d would refuse its shape


def test_goal_words_wait_unless_told_not_to() -> None:
    assert build_request(["move-joint", "left", "0,0,0,0.5,0"])[0]["wait"] is True
    assert build_request(["reach", "left", "0.3,0.2,0.8", "--no-wait"])[0]["wait"] is False
    assert build_request(["lower", "left", "--stop-z", "0.78"])[0]["stop_z_m"] == 0.78
