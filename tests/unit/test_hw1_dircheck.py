"""The joint-direction check (HW1 fix list #38, python/r1mc3/hw1/dircheck.py): v3's model still predicts, for every
joint, the motion the user saw on the robot at HW1 (11 Oct 2026: 26 / 26), and the check catches a mirrored joint.

Data: tests/assets/hw1/dir_expect.json = the session's own prediction (the numbers the live answers were compared
with; README_dircheck.md). A failure here after a model / joint-table change means a direction changed: re-check on the
robot before anything moves (the user's rule, docs/hardware/JOINT_DIRECTIONS.md)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from r1mc3.hw1 import dircheck as D
from r1mc3.service.codes import GROUP_JOINTS
from r1sim import joints as J

ASSET = Path(__file__).resolve().parents[1] / "assets" / "hw1" / "dir_expect.json"


@pytest.fixture(scope="module")
def at_hw1_pose() -> dict[str, D.Expectation]:
    return D.expectations(D.HW1_POSE, delta=0.10)


def test_tables_match_the_simulator_and_r1d() -> None:
    assert D.JOINTS == J.JOINTS
    assert {g: tuple(v) for g, v in D.GROUPS.items()} == GROUP_JOINTS
    assert set(D.HW1_VERIFIED) == set(J.JOINTS)  # every one of the 26 joints was checked at HW1


def test_the_model_predicts_what_the_user_saw_at_hw1(at_hw1_pose: dict[str, D.Expectation]) -> None:
    ok = D.check_against_hw1(at_hw1_pose)
    assert all(ok.values()), {j: (at_hw1_pose[j].key, D.HW1_VERIFIED[j]) for j, v in ok.items() if not v}


def test_the_repo_tool_reproduces_the_session_prediction(at_hw1_pose: dict[str, D.Expectation]) -> None:
    """The numbers the 26 live answers were compared with (dir_expect.py at HW1), to 0.02 cm / 0.002 rad."""
    session = json.loads(ASSET.read_text())
    assert set(session) == set(at_hw1_pose)
    for jn, s in session.items():
        e = at_hw1_pose[jn]
        assert e.point == s["point"], jn
        assert e.d_cm == pytest.approx(s["d_cm"], abs=0.02), jn
        assert e.rot == pytest.approx(s["rot"], abs=0.002), jn
        assert e.words == s["words"] and e.rot_words == s["rot_words"], jn


def test_the_same_directions_at_unitrees_stand_lock_preset() -> None:
    """The robot enters ON_STAND from the stand lock's preset pose (docs/hardware/STAND_LOCK.md): no direction flips
    there either, at the step sizes HW1 used."""
    exp = D.expectations(D.STAND_LOCK_PRESET, delta=D.STEP_RAD)
    ok = D.check_against_hw1(exp)
    assert all(ok.values()), [j for j, v in ok.items() if not v]


def test_a_mirrored_joint_is_caught(at_hw1_pose: dict[str, D.Expectation]) -> None:
    """A joint whose sign is reversed (modelled here as a NEGATIVE step) reads the opposite motion: every joint fails
    the check - so the check can tell a mirrored joint from a correct one."""
    mirrored = D.expectations(D.HW1_POSE, delta=-0.10)
    ok = D.check_against_hw1(mirrored)
    assert not any(ok.values()), [j for j, v in ok.items() if v]
    one = {**at_hw1_pose, "left_knee": mirrored["left_knee"]}
    assert [j for j, v in D.check_against_hw1(one).items() if not v] == ["left_knee"]


def test_the_operator_words_say_what_the_user_said(at_hw1_pose: dict[str, D.Expectation]) -> None:
    """Every capitalised word the tool prints (BACKWARD, OUTWARD, INWARD, DOWN, LEFT, STRAIGHTENS, ...) is in the
    user's own answer at HW1 for that joint."""
    for jn, e in at_hw1_pose.items():
        caps = set(re.findall(r"\b[A-Z]{2,}\b", e.operator))
        assert caps, (jn, e.operator)
        assert caps <= set(re.findall(r"\b[A-Z]{2,}\b", D.HW1_VERIFIED[jn].seen)), (jn, e.operator)


def test_elbow_sign_a_larger_value_is_straighter(at_hw1_pose: dict[str, D.Expectation]) -> None:
    """The HW1 correction (NOTES.md 14:5x): a SMALLER elbow value = MORE flexed. +0.10 from 1.40 moves the hand back
    toward hanging (straighter) and the words say STRAIGHTENS - never 'flexes'."""
    for side in ("left", "right"):
        e = at_hw1_pose[f"{side}_elbow"]
        assert e.key == "back" and e.d_cm[0] < -1.0 and "STRAIGHTENS" in e.operator


def test_the_step_plan_moves_one_joint_by_the_hw1_step() -> None:
    plan = D.step_plan(D.HW1_POSE)
    assert [s["joint"] for s in plan] == [J.JOINTS[j] for idx in D.GROUPS.values() for j in idx]
    for s in plan:
        q_out = [float(x) for x in re.search(r"q=\[([^\]]*)\]", s["out"]).group(1).split(",")]  # type: ignore[union-attr]
        q_back = [float(x) for x in re.search(r"q=\[([^\]]*)\]", s["back"]).group(1).split(",")]  # type: ignore[union-attr]
        base = [D.HW1_POSE[j] for j in D.GROUPS[s["group"]]]
        assert q_back == pytest.approx(base)
        diff = [i for i, (a, b) in enumerate(zip(q_out, base, strict=True)) if abs(a - b) > 1e-9]
        assert len(diff) == 1 and q_out[diff[0]] - base[diff[0]] == pytest.approx(D.STEP_RAD[s["joint"]])
        assert "hold_s=1" in s["back"] and "--yes" in s["out"]  # HW1: a return with hold_s 0.5 was refused
    assert (D.STEP_RAD["head_pitch"], D.STEP_RAD["left_ankle_pitch"], D.STEP_RAD["left_knee"]) == (0.20, 0.15, 0.10)
