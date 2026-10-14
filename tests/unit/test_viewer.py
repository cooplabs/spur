"""r1sim.viewer (session 11, D32's viewer - first cut): the headless path end to end, the keys (faults through the
simulator's injector, a push, camera presets, toggles), the declared world boxes, any MJCF file.

The window itself (MuJoCo's passive viewer) needs a display; everything it shows comes from `App`, tested here.
Base 1e55bdd: the package did not exist (S10: "the simulator viewer for session R - slipped").
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from r1sim import cpp_policy
from r1sim.viewer import FAULT_KEYS, App, ViewerConfig
from r1sim.viewer.__main__ import main

ENTITIES = {"entities": [
    {"kind": "surface", "id": "table", "pose": {"pos": [0.55, 0.0, 0.76]}, "size": [0.6, 1.2, 0.04]},
    {"kind": "object", "id": "box", "pose": {"pos": [0.42, 0.2, 0.805]}, "size": [0.025, 0.025, 0.025]},
]}  # fmt: skip


def _render(app: App, w: int = 160, h: int = 120) -> np.ndarray:
    r = mujoco.Renderer(app.m, height=h, width=w)
    try:
        r.update_scene(app.d, camera=app.camera(), scene_option=app.scene_option())
        app.decorate(r.scene)
        return np.asarray(r.render(), dtype=np.float64)
    finally:
        r.close()


def test_headless_frames_with_the_overlay(tmp_path: Path) -> None:
    out = tmp_path / "v"
    rc = main(["--headless", "--scene", "floor", "--controller", "hold", "--frames", "3", "--every", "0.2",
               "--keys", "C@0,P@0.1", "--out", str(out), "--width", "160", "--height", "120"])  # fmt: skip
    assert rc == 0
    res = json.loads((out / "viewer.json").read_text())
    assert [f["file"] for f in res["frames"]] == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    assert all((out / f["file"]).stat().st_size > 1000 and f["std"] > 5.0 for f in res["frames"])
    assert [round(f["t"], 3) for f in res["frames"]] == [0.0, 0.2, 0.4]
    ov = res["frames"][-1]["overlay"]
    assert ov["controller"].startswith("hold") and "tilt / pelvis z" in ov and "contacts" in ov
    assert [e["what"].split(" ")[0] for e in res["events"]] == ["contacts", "push"]


def test_the_push_key_pushes_the_pelvis() -> None:
    def pelvis_vx(push: bool) -> float:
        app = App(ViewerConfig(scene="floor", controller="hold"))
        app.advance(0.3)
        if push:
            assert app.key("p").startswith("push 60 N")
        app.advance(0.4)
        assert app.w is not None
        b = app.w.index.pelvis_body
        return float(app.d.cvel[b][3])  # the pelvis' linear velocity along x (world)

    assert pelvis_vx(True) - pelvis_vx(False) > 0.02  # m/s: the first push is forward


def test_fault_keys_go_to_the_simulators_injector() -> None:
    app = App(ViewerConfig(scene="floor", controller="hold"))
    app.advance(0.1)
    for k, (kind, dur, _) in FAULT_KEYS.items():
        what = app.key(k)
        if kind.startswith("policy_"):
            assert "only with the C++ core" in what  # the hold fixture has no policy
        else:
            assert what.startswith(f"fault {kind} for {dur} s"), what
    kinds = {f.kind for f in app.faults.faults}
    assert kinds == {v[0] for v in FAULT_KEYS.values() if not v[0].startswith("policy_")}
    assert any(r[0] == "faults" and "cmd_gap" in r[1] for r in app.overlay())


@pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py not built: run scripts/ci.sh cpp")
def test_the_policy_nan_key_freezes_the_real_core() -> None:
    app = App(ViewerConfig(scene="floor", controller="core"))
    app.advance(0.5)
    assert dict(app.overlay())["controller"] == "core  BALANCE"
    app.key("N")
    app.advance(1.0)
    assert dict(app.overlay())["controller"] == "core  STAND_FREEZE"  # rung 3: never limp, the operator resumes


def test_camera_presets_and_the_world_boxes(tmp_path: Path) -> None:
    wj = tmp_path / "w.json"
    wj.write_text(json.dumps(ENTITIES))
    app = App(ViewerConfig(scene="floor", controller="none", world_json=str(wj)))
    imgs = {}
    for k, name in zip("1356", ("free", "front", "overhead", "head"), strict=True):
        assert app.key(k) == f"camera: {name}"
        imgs[name] = _render(app)
    assert app.camera("head").type == mujoco.mjtCamera.mjCAMERA_FIXED
    names = list(imgs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert np.abs(imgs[a] - imgs[b]).mean() > 3.0, (a, b)
    app.key("1")
    with_boxes = _render(app)
    assert app.key("W") == "world boxes off"
    without = _render(app)
    assert np.abs(with_boxes - without).mean() > 1.0
    r = mujoco.Renderer(app.m, height=60, width=80)
    try:
        r.update_scene(app.d, camera=app.camera())
        app.key("W")
        n0 = r.scene.ngeom
        assert app.decorate(r.scene) == 2 and r.scene.ngeom == n0 + 2
    finally:
        r.close()


def test_any_mjcf_file(tmp_path: Path) -> None:
    xml = tmp_path / "drop.xml"
    xml.write_text("""<mujoco><worldbody><geom type="plane" size="1 1 0.1"/>
      <body pos="0 0 0.3"><freejoint/><geom type="box" size="0.05 0.05 0.05"/></body></worldbody></mujoco>""")
    app = App(ViewerConfig(model_path=str(xml)))
    app.advance(1.0)
    ov = dict(app.overlay())
    assert app.controller == "none" and ov["contacts"].split(" ")[0] != "0"  # the box landed on the plane
    assert app.key("P") == "push: no robot in this model"
    assert _render(app).std() > 5.0


class _FakeWindow:
    """MuJoCo's passive-viewer handle as `interactive` uses it (no display needed); presses keys at given loops."""

    def __init__(self, m: mujoco.MjModel, key_callback: object, loops: int, keys: dict[int, str]) -> None:
        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        self.user_scn = mujoco.MjvScene(m, maxgeom=64)
        self.cb, self.loops, self.keys, self.n = key_callback, loops, keys, 0
        self.texts: list[object] = []
        self.syncs = 0

    def __enter__(self) -> _FakeWindow:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def is_running(self) -> bool:
        self.n += 1
        if self.n in self.keys:
            self.cb(ord(self.keys[self.n]))  # type: ignore[operator]
        return self.n <= self.loops

    def lock(self) -> object:
        import contextlib

        return contextlib.nullcontext()

    def set_texts(self, t: object) -> None:
        self.texts.append(t)

    def sync(self) -> None:
        self.syncs += 1


def test_the_window_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mujoco.viewer
    from r1sim.viewer.__main__ import _parser, interactive

    wj = tmp_path / "w.json"
    wj.write_text(json.dumps(ENTITIES))
    a = _parser().parse_args(["--controller", "hold", "--world", str(wj), "--out", str(tmp_path / "o"),
                              "--width", "160", "--height", "120"])  # fmt: skip
    app = App(ViewerConfig(scene="floor", controller="hold", world_json=str(wj)))
    win: list[_FakeWindow] = []

    def launch(m: mujoco.MjModel, d: mujoco.MjData, key_callback: object = None) -> _FakeWindow:
        win.append(_FakeWindow(m, key_callback, 24, {3: "2", 5: "c", 7: "r", 12: "r", 15: ".", 17: "k"}))
        return win[0]

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    assert interactive(a, app) == 0
    w = win[0]
    assert w.syncs == 24 and w.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING  # '2' = follow the pelvis
    assert w.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] == 1  # 'c'
    assert w.user_scn.ngeom == 2 and w.texts and isinstance(w.texts[-1], tuple)  # the world boxes, the overlay
    out = tmp_path / "o"
    assert len(list(out.glob("frame_*.png"))) == 5 and len(list(out.glob("screenshot_*.png"))) == 1  # 'r' ... 'r', '.'
    assert app.toggles.paused and app.t > 0.0  # 'k' paused it after it ran
