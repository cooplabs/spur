"""r1sim.viewer, session R: what the first cut left out - the guard in the window (the C++ core + r1guard in lockstep,
`--controller guarded`, the default), jog through the core's upper-body executor, the hands' closure, object drop-in,
the stand scene (ON_STAND through the mode mailbox), another package's scene by its "module:attr" string, the arrows in
the window, and the headless batch runs with seeds (r1sim.viewer.batch).

Base 3f87b85: `ViewerConfig(controller="guarded")`, the keys J / [ ] / - = / O / U, `scene="stand"`, a "module:attr"
scene and `r1sim.viewer.batch` did not exist (ValueError / ImportError there).
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from r1sim import cpp_policy
from r1sim.viewer import App, ViewerConfig
from r1sim.viewer.app import JOG_STEP_RAD, POOL
from r1sim.viewer.batch import BatchConfig, parse_seeds, run_batch

needs_core = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")


def _foot_floor_contacts(app: App) -> int:
    n = 0
    for i in range(app.d.ncon):
        c = app.d.contact[i]
        names = {mujoco.mj_id2name(app.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in (c.geom1, c.geom2)}
        n += int("floor" in names)
    return n


@needs_core
def test_the_guard_takes_the_motors_when_the_core_stalls_and_keeps_balancing() -> None:
    app = App(ViewerConfig(scene="floor"))  # the default controller: the core + r1guard
    assert app.controller == "guarded"
    app.advance(1.0, max_steps=10**7)
    ov = dict(app.overlay())
    assert ov["controller"] == "guarded  BALANCE" and ov["motors owned by"].startswith("r1rt")
    assert app.key("S").startswith("fault controller_stall of the core for 2.0 s")
    app.advance(1.2, max_steps=10**7)
    assert app.token_owner() == 2 and app.guard_takeovers() == 1
    assert dict(app.overlay())["motors owned by"].startswith("r1guard")
    app.advance(4.0, max_steps=10**7)  # the core is back from its stall but silent: the guard keeps the motors
    assert app.token_owner() == 2 and app.tilt_deg() < 5.0 and app.pelvis_z() > 0.65
    assert app.jog_step(1.0).startswith("jog: r1guard holds the motors")
    app.key("U")  # the operator's hand-back: refused while balancing (not offered: the guard keeps the robot)
    app.advance(4.1, max_steps=10**7)
    assert app.events[-1].what.startswith("reclaim refused (mode)") and app.token_owner() == 2


@needs_core
def test_jog_moves_the_selected_joint_through_the_cores_executor() -> None:
    app = App(ViewerConfig(scene="floor"))
    app.advance(1.0, max_steps=10**7)
    assert app.key("J").startswith("jog target arm_right: right_shoulder_pitch")
    assert app.key("]").startswith("jog target arm_right: right_shoulder_roll")
    j = 20  # right_shoulder_roll (SDK order, r1sim.joints)
    q0 = float(app.ctl.last_q[j])
    assert app.key("-").startswith("jog right_shoulder_roll ->")
    app.advance(1.6, max_steps=10**7)  # the 0.4 s trajectory has ended
    assert float(app.ctl.last_q[j]) == pytest.approx(q0 - JOG_STEP_RAD["arm_right"], abs=1e-6)
    st = app.sim.read_state()
    assert abs(float(st.q[j]) - (q0 - JOG_STEP_RAD["arm_right"])) < 0.03  # the arm followed its target
    assert app.tilt_deg() < 3.0


def test_a_hands_closure_goes_to_its_command() -> None:
    app = App(ViewerConfig(scene="floor", controller="hold"))
    app.advance(0.3)
    for _ in range(3):
        app.key("J")  # left arm -> right arm -> head -> the left hand
    assert app.jog.name == "hand_left"
    q_open = np.asarray(app.sim.read_state().hand_q["left"]).copy()
    for _ in range(4):
        app.key("=")
    assert app.jog.closure["left"] == 1.0 and dict(app.overlay())["jog"] == "hand_left: closure 1.00"
    app.advance(1.3)
    q = np.asarray(app.sim.read_state().hand_q["left"])
    assert np.abs(q - q_open).max() > 0.8  # the fingers closed (toward the fist preset)


def test_object_drop_in_lands_and_the_pool_waits_unseen() -> None:
    app = App(ViewerConfig(scene="table", controller="hold"))
    k = 0
    b = mujoco.mj_name2id(app.m, mujoco.mjtObj.mjOBJ_BODY, f"viewer_drop_{k}_{POOL[k][0]}")
    g = mujoco.mj_name2id(app.m, mujoco.mjtObj.mjOBJ_GEOM, f"viewer_drop_{k}")
    assert app.m.geom_rgba[g, 3] == 0.0 and app.m.body_contype[b] == 0  # parked: unseen, no contacts
    app.advance(0.2)
    z_parked = float(app.d.xpos[b][2])
    assert app.key("O").startswith("dropped a box at")
    app.advance(1.5)
    assert app.m.geom_rgba[g, 3] == 1.0
    assert float(app.d.xpos[b][2]) == pytest.approx(0.78 + 0.025, abs=0.003)  # resting on the table top
    assert z_parked == pytest.approx(0.5, abs=1e-3)  # it floated where it waited (gravity compensated)
    app2 = App(ViewerConfig(scene="floor", controller="hold"))
    b2 = mujoco.mj_name2id(app2.m, mujoco.mjtObj.mjOBJ_BODY, f"viewer_drop_{k}_{POOL[k][0]}")
    app2.key("O")
    app2.advance(1.0)
    assert float(app2.d.xpos[b2][2]) == pytest.approx(0.025, abs=0.003)  # on the floor in front of the robot


@needs_core
def test_the_stand_scene_enters_on_stand_and_the_guard_hands_back_there() -> None:
    app = App(ViewerConfig(scene="stand"))
    app.advance(3.5, max_steps=10**7)  # the legs settle, ON_STAND is asked at 1.5 s, the blend ends at ~2.8 s
    assert dict(app.overlay())["controller"] == "guarded  ON_STAND"
    assert _foot_floor_contacts(app) == 0 and app.pelvis_z() > 0.8  # hanging, feet off the floor
    assert app.key("=").startswith("jog left_shoulder_pitch ->")
    app.key("S")
    app.advance(4.0, max_steps=10**7)
    assert app.token_owner() == 2
    app.advance(6.0, max_steps=10**7)
    app.key("U")  # on the stand the operator's hand-back is allowed
    app.advance(6.1, max_steps=10**7)
    assert app.events[-1].what.startswith("reclaim accepted")
    app.advance(7.5, max_steps=10**7)
    assert app.token_owner() == 1 and app.mode_name() == "ON_STAND"


def test_another_packages_scene_by_its_module_string() -> None:
    app = App(ViewerConfig(scene="r1sim.scenes:add_table", controller="none"))
    assert app.w is not None and app.w.scene is not None and app.w.scene.name == "table"
    with pytest.raises(ValueError, match="scene"):
        App(ViewerConfig(scene="no_such_scene", controller="none"))


def test_the_arrows_in_the_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mujoco.viewer
    from r1sim.viewer.__main__ import _parser, interactive

    from tests.unit.test_viewer import _FakeWindow

    class Arrows(_FakeWindow):
        def is_running(self) -> bool:
            self.n += 1
            if self.n in self.keys:
                self.cb(int(self.keys[self.n]))  # type: ignore[operator]
            return self.n <= self.loops

    a = _parser().parse_args(["--controller", "hold", "--out", str(tmp_path / "o")])
    app = App(ViewerConfig(scene="floor", controller="hold"))
    monkeypatch.setattr(mujoco.viewer, "launch_passive",
                        lambda m, d, key_callback=None: Arrows(m, key_callback, 6, {2: "262", 4: "262"}))  # fmt: skip
    assert interactive(a, app) == 0
    assert app.jog.joint["arm_left"] == 2  # right arrow twice = the next joint twice


def test_batch_runs_with_seeds_are_reproducible(tmp_path: Path) -> None:
    assert parse_seeds("nominal,0-2,5") == [None, 0, 1, 2, 5]
    cfg = BatchConfig(seeds=[0, 1], duration_s=1.5, controller="hold", pushes=1, push_n=(30.0, 40.0))
    a, b = run_batch(cfg), run_batch(cfg)
    assert a["n"] == 2 and a["falls"] == 0
    rows = [(r["seed"], r["max_tilt_deg"], [e["what"] for e in r["events"]]) for r in a["runs"]]
    assert rows == [(r["seed"], r["max_tilt_deg"], [e["what"] for e in r["events"]]) for r in b["runs"]]
    assert all(len(r["events"]) == 1 and r["events"][0]["what"].startswith("push") for r in a["runs"])
    assert a["runs"][0]["events"] != a["runs"][1]["events"]  # each seed draws its own push


@needs_core
def test_batch_cli_core_alone_vs_with_the_guard(tmp_path: Path) -> None:
    from r1sim.viewer.batch import main

    rows = {}
    for ctl in ("core", "guarded"):
        out = tmp_path / f"{ctl}.json"
        rc = main(["--seeds", "0", "--duration", "3", "--controller", ctl, "--keys", "S@1.0", "--out", str(out),
                   "--fail-on-fall"])  # fmt: skip
        res = json.loads(out.read_text())
        assert rc == 0 and res["n"] == 1 and res["falls"] == 0
        rows[ctl] = res["runs"][0]
    # the core alone: the 0.5 s stall is the board's to ride out (its ~0.4 s hold); no standby took anything over
    assert rows["core"]["guard_takeovers"] == 0 and [m for _, m in rows["core"]["modes"]] == ["BALANCE"]
    # with r1guard: the stall (2 s, the core only) hands the motors to the guard, which keeps the robot up
    assert rows["guarded"]["guard_takeovers"] == 1 and rows["guarded"]["modes"][-1][1] == "r1guard"
