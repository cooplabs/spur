"""The demo command (`python -m r1sim.demo`) runs end to end: the robot stands in the default scene (the table) under
the balance policy (default) and under the session-1 fixture (`--fixture`), the PNG is written and the numbers
printed."""

from __future__ import annotations

from pathlib import Path

import pytest
from r1sim import demo


@pytest.mark.realtime
def test_demo_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "demo.png"
    rc = demo.main(["--seconds", "1.0", "--realtime-seconds", "0.5", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "RESULT: the R1 stands in the scene under the balance policy" in text
    assert "balance policy r1mc2_manip2_lean_20261006T0421Z/policy.onnx" in text
    assert "real-time factor, lockstep (real)" in text and "real-time mode, achieved" in text
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 50_000


@pytest.mark.needs_gl
def test_demo_policy_and_fixture_lockstep(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """No real-time part (not timing-dependent): the policy demo and the --fixture comparison both stand."""
    for extra, what in (([], "the balance policy"), (["--fixture"], "the session-1 stand-hold fixture")):
        rc = demo.main(["--seconds", "1.0", "--realtime-seconds", "0", "--out", str(tmp_path / "d.png"), *extra])
        text = capsys.readouterr().out
        assert rc == 0, text
        assert f"RESULT: the R1 stands in the scene under {what}" in text
        assert ("FIXTURE" in text) == bool(extra)
