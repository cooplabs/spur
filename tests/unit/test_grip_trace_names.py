"""Session 12 (the integrator, from the Linux CI): every grasp keeps its own trace file. The name carried whole seconds
only (`grasp_<UTC>_<side>.npz`), so two grasps of one side within a wall second wrote the SAME file and the second
replaced the first (the lockstep `test_service_grasp_trace` on Linux: the first file was the second grasp's). Now the
UTC stamp has milliseconds and a name already held gets `-1`, `-2` ... before the side."""

from __future__ import annotations

from pathlib import Path

from r1mc3.hands import trace as T

from tests.unit.grip_fake import FakeHand


def test_two_traces_of_one_side_in_the_same_instant_are_both_kept(tmp_path: Path) -> None:
    h = FakeHand()
    a = T.TracingHands(h, h.side, {"grip": "wrap", "width_m": 0.04})
    b = T.TracingHands(h, h.side, {"grip": "wrap", "width_m": None})
    b.utc = a.utc  # the worst case: the same stamp
    pa = a.save(tmp_path, background=False)
    pb = b.save(tmp_path, background=False)
    assert pa != pb
    files = sorted(tmp_path.glob(f"grasp_*_{h.side}.npz"))
    assert len(files) == 2, files
    assert {T.load(f)["meta"]["params"]["width_m"] for f in files} == {0.04, None}


def test_the_stamp_has_milliseconds() -> None:
    h = FakeHand()
    utc = T.TracingHands(h, h.side, {}).utc
    assert len(utc) == len("20261013T031124.123Z") and utc[15] == "." and utc.endswith("Z"), utc
