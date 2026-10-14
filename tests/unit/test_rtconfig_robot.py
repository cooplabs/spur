"""The robot's processes config from the robot profile (session 6; `python -m r1mc3.rtconfig --profile robot`):
every written key has its provenance (the profile entry's source + date, or why it stays the repository's / the
model's), the values are the profile's, the file differs from the simulator's only where the profile says so (pinned:
a profile change shows up here), the guard gate and `takeover_require_rt` are on, a profile whose policy gains differ
from the policy's metadata is refused, and the C++ reader (`r1rt --check-config`) and the core accept the file."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest
import yaml
from r1mc3 import rtconfig

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "config" / "robot_profile.r1_dex3.yaml"


@pytest.fixture(scope="module")
def robot(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str], dict[str, dict[str, object]]]:
    d = tmp_path_factory.mktemp("robotcfg")
    p = rtconfig.write_robot(d / "r1rt.cfg", PROFILE)
    keys = dict(x.split(" = ", 1) for x in p.read_text().splitlines() if x and not x.startswith("#"))
    prov = json.loads(Path(f"{p}.provenance.json").read_text())
    return p, keys, prov


def _floats(v: str) -> list[float]:
    return [float(x) for x in v.split()]


def test_every_key_has_its_provenance(robot: tuple[Path, dict[str, str], dict[str, dict[str, object]]]) -> None:
    p, keys, prov = robot
    assert keys.pop("format") == rtconfig.FORMAT
    assert set(keys) == set(prov) - {"_profile"}
    for k, pv in prov.items():
        if k == "_profile":
            continue
        assert pv["from"] in ("profile", "repository", "fixed"), (k, pv)
        if pv["from"] == "profile":
            assert pv["status"] and pv["source"] and pv["date"], (k, pv)
        else:
            assert pv["why"], (k, pv)
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):  # a comment line names each key's source right above it
        if line and not line.startswith("#") and not line.startswith("format"):
            assert lines[i - 1].startswith(f"# {line.split(' = ')[0]}: "), (lines[i - 1], line)
    assert prov["_profile"]["version"] == 2 and len(str(prov["_profile"]["sha256"])) == 64


def test_the_values_are_the_profiles(robot: tuple[Path, dict[str, str], dict[str, dict[str, object]]]) -> None:
    _, keys, _ = robot
    prof = yaml.safe_load(PROFILE.read_text())
    names = [j["name"] for j in prof["joint_map"]["value"]]
    eff = prof["limits"]["effort"]["value"]
    frac = prof["limits"]["torque_clip_fraction"]["value"]
    assert _floats(keys["tau_max"]) == pytest.approx([frac * eff[n] for n in names])
    # the URDF's velocity, except where it is looser than the simulator's tested 30 rad/s dq clamp (kept, listed)
    vel = prof["limits"]["velocity"]["value"]
    assert _floats(keys["dq_max"]) == pytest.approx([min(vel[n], 30.0) for n in names])
    assert _floats(keys["q_min"]) == pytest.approx([prof["limits"]["position"]["value"][n][0] for n in names])
    stiff = prof["gains"]["stiff"]["value"]
    assert _floats(keys["stand_kp"]) == pytest.approx([stiff[n][0] for n in names])
    arms = prof["gains"]["arms_manip"]["value"]
    assert _floats(keys["upper_kp"]) == pytest.approx([(arms.get(n) or stiff[n])[0] for n in names])
    sf, pr = prof["safety"], prof["processes"]
    assert int(keys["lowstate_stale_ns"]) == sf["lowstate_age_resend_ms"]["value"] * 1_000_000
    assert float(keys["fall_tilt_rad"]) == pytest.approx(math.radians(sf["fall_detect_deg"]["value"]))
    assert float(keys["stand_ramp_s"]) == pr["stand_freeze_ramp_s"]["value"]
    assert float(keys["takeover_blend_s"]) == pr["takeover_blend_stand_s"]["value"]
    assert int(keys["guard.stale_periods"]) == sf["guard_heartbeat_writer_periods"]["value"]
    assert _floats(keys["damp_kd"]) == pytest.approx([pr["damp_kd"]["value"][n] for n in names])
    # the robot's arrangement: no takeover without real-time grants, none without a live guard; the guard armed
    assert keys["takeover_require_rt"] == "1" and keys["takeover_require_guard"] == "1"
    assert keys["guard.takeover_enabled"] == "1"


def test_never_looser_than_the_simulators_tested_limits(
    robot: tuple[Path, dict[str, str], dict[str, dict[str, object]]], tmp_path: Path
) -> None:
    """A profile limit beyond the simulator's tested value keeps the simulator's, per joint, and the provenance lists
    the joints (loosening a safety limit needs the user's OK). Today: the URDF's 33.4 rad/s joints vs the 30 rad/s
    clamp; a made-up 100 N·m knee effort keeps the knee's torque clip at the simulator's 36 N·m."""
    _, _, prov = robot
    assert prov["dq_max"]["kept_simulator_value_where_looser"] == [
        "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
        "head_pitch", "head_yaw",
    ]  # fmt: skip

    def strong_knee(d: dict) -> None:  # type: ignore[type-arg]
        d["limits"]["effort"]["value"]["left_knee"] = 100

    cfg, _, pv = rtconfig.robot_config(_edited_profile(tmp_path, strong_knee))
    assert cfg["tau_max"][3] == pytest.approx(36.0)  # 0.6 x 60, not 0.6 x 100
    assert pv["tau_max"]["kept_simulator_value_where_looser"] == ["left_knee"]


def test_differs_from_the_simulators_file_only_where_the_profile_says() -> None:
    """Pinned (session 6): the profile carries the simulator's numbers except the velocity / gain bounds (tighter, from
    the URDF and the gain sets) and the real-time requirement. A profile change (e.g. HW1's stand pose or gains) must
    show up here - update the pin deliberately, with the reason."""
    cfg, guard, _ = rtconfig.robot_config(PROFILE)
    sim = rtconfig.core_dict()
    diff = rtconfig.differences(cfg, sim)
    assert set(diff) == {"dq_max", "kd_max", "kp_max", "takeover_require_rt"}, diff
    assert guard == rtconfig.GUARD_DEFAULTS  # the same guard parameters (1 kHz, 5 periods, 0.3 s, armed)
    for k in ("dq_max", "kp_max", "kd_max"):  # only ever tighter
        r, s = diff[k][0], diff[k][1]
        assert r is not None and s is not None
        assert all(float(a) <= float(b) for a, b in zip(r.split(), s.split(), strict=True)), k


def _edited_profile(tmp: Path, edit: object) -> Path:
    d = yaml.safe_load(PROFILE.read_text())
    edit(d)  # type: ignore[operator]
    p = tmp / "profile.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


def test_policy_gains_must_match_the_policy(tmp_path: Path) -> None:
    def bump(d: dict) -> None:  # type: ignore[type-arg]
        d["gains"]["policy_legs_waist"]["value"]["left_knee"] = [90.0, 5.367]

    with pytest.raises(ValueError, match="trained"):
        rtconfig.robot_config(_edited_profile(tmp_path, bump))


def test_a_version_1_profile_is_refused(tmp_path: Path) -> None:
    def v1(d: dict) -> None:  # type: ignore[type-arg]
        d.pop("processes")
        d["version"] = 1

    with pytest.raises(ValueError, match="processes"):
        rtconfig.robot_config(_edited_profile(tmp_path, v1))


def test_observe_only_and_cli(tmp_path: Path) -> None:
    out = tmp_path / "obs.cfg"
    assert rtconfig.main(["--out", str(out), "--profile", "robot", "--observe-only"]) == 0
    assert "guard.takeover_enabled = 0" in out.read_text()
    with pytest.raises(SystemExit):  # the simulator-only options are refused for the robot
        rtconfig.main(["--out", str(out), "--profile", "robot", "--no-guard-gate"])
    sim = tmp_path / "sim.cfg"
    assert rtconfig.main(["--out", str(sim)]) == 0  # the default stays the simulator's file
    assert sim.read_text() == rtconfig.write(tmp_path / "sim2.cfg", rtconfig.core_dict()).read_text()
    assert "takeover_require_rt = 0" in sim.read_text()


def test_r1rt_check_config_accepts_the_robot_file(
    robot: tuple[Path, dict[str, str], dict[str, dict[str, object]]],
) -> None:
    from r1sim import stack

    apps = stack.apps_dir()
    if apps is None:
        pytest.skip("r1rt not built (scripts/ci.sh dds)")
    out = subprocess.run(
        [str(apps / "r1rt"), "--config", str(robot[0]), "--check-config"], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr[-500:]
    d = json.loads(out.stdout.strip().splitlines()[-1])
    assert d["config_ok"] is True and d["takeover_require_guard"] == 1 and d["writer_hz"] == 500


def test_the_core_accepts_the_robot_config() -> None:
    try:
        from r1sim.cpp_policy import import_r1rt_py

        mod = import_r1rt_py()
    except ImportError:
        pytest.skip("the r1rt_py binding is not built (scripts/ci.sh cpp)")
    cfg, _, _ = rtconfig.robot_config(PROFILE)
    core = mod.Core(cfg)
    st = core.status()
    assert st["mode"] == 0  # MONITOR: publishes nothing
