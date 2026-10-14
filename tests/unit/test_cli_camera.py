"""The `r1` words for r1d's camera ops (session 10): `r1 observe` (obs_id + the camera pose; no image printed),
`r1 point <obs_id> <px> <py>`, `r1 extent <obs_id> <px> <py>`. On the base (a041731) argparse refused the words."""

from __future__ import annotations

import pytest
from r1mc3.cli.main import build_request
from r1mc3.service.schemas import REQUEST


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        (["observe"], {"op": "camera.observe", "depth": True, "jpeg": False}),
        (["observe", "--no-depth"], {"op": "camera.observe", "depth": False, "jpeg": False}),
        (["point", "h3", "270", "250.5"], {"op": "camera.point", "obs_id": "h3", "px": 270.0, "py": 250.5}),
        (["extent", "h7", "10", "20"], {"op": "camera.object_extent", "obs_id": "h7", "px": 10.0, "py": 20.0}),
    ],
)
def test_camera_words_build_requests_r1d_accepts(argv: list[str], want: dict[str, object]) -> None:
    req, _ = build_request(argv)
    assert {k: req[k] for k in want} == want
    REQUEST.validate_python(req)  # the request models take it as is
