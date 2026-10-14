"""The `r1` words for the api agent's session-10 ops (cli/words_api.py): every word builds a request r1d's schemas
accept (no r1d needed). Base a041731: the words did not exist (argparse: invalid choice)."""

from __future__ import annotations

import pytest
from r1mc3.cli.main import build_request
from r1mc3.service.schemas import REQUEST

WORDS = [
    (["move-path", "left", "[[0.3, 0.2, 0.8], [0.32, 0.2, 0.8]]", "--linear"], "arm.move_path"),
    (["guarded-move", "left", "--delta", "-0.02,0,0", "--until", "blocked"], "arm.guarded_move"),
    (["guarded-move", "right", "--stop-z", "0.75", "--until", "", "--touch", "box,table"], "arm.guarded_move"),
    (["slide", "left", "0.03,0,0", "--support", "table"], "arm.slide"),
    (["pick", "left", "0.35,0.2,0.78", "--height", "0.06", "--width", "0.04", "--object-id", "box"], "skill.pick"),
    (["place", "left", "0.3,-0.1,0.8", "--no-wait"], "skill.place"),
    (["par", '[{"op": "arm.move_joint", "arm": "left", "q": [0, 0, 0, 0.5, 0]}]'], "par.run"),
    (["lease", "acquire", "arm_left,hand_left", "--client", "me", "--ttl", "60"], "lease.acquire"),
    (["lease", "renew", "L3"], "lease.renew"),
    (["lease", "release", "L3"], "lease.release"),
    (["lease", "list"], "lease.list"),
    (["twist", "left", "0.02,0,-0.01", "--lease", "L3"], "stream.arm_twist"),
]


@pytest.mark.parametrize(("argv", "op"), WORDS, ids=[f"{w[0][0]}{i}" for i, w in enumerate(WORDS)])
def test_each_word_builds_a_valid_request(argv: list[str], op: str) -> None:
    req, _ = build_request(argv)
    assert req["op"] == op
    REQUEST.validate_python(req)


def test_goal_words_wait_and_carry_the_lease() -> None:
    req, _ = build_request(["slide", "left", "0.03,0,0", "--lease", "L9"])
    assert req["wait"] is True and req["lease"] == "L9"
    req, _ = build_request(["pick", "left", "0.35,0.2,0.78", "--no-wait"])
    assert req["wait"] is False and "lease" not in req
    with pytest.raises(SystemExit):
        build_request(["lease", "acquire"])
