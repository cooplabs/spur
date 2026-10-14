"""`python -m r1sim.viewer` - the simulator's viewer (see the package docstring). Interactive (MuJoCo's window; on
macOS under `mjpython`: `scripts/viewer.sh`) or headless (`--headless --frames N --out DIR`); `--stack` = the real
processes (r1rt, r1guard, r1d) against the DDS-bridge simulator (`r1sim.viewer.stack`)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..render import write_png
from .app import CONTROLLERS, HELP, PRESETS, SCENES, SPECIAL_KEYS, App, ViewerConfig


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m r1sim.viewer", description=__doc__.splitlines()[0])
    ap.add_argument(
        "--scene",
        default="floor",
        help=f"{' | '.join(SCENES)}: the floor (default), a plain table with loose objects, the robot hanging on a "
        "stand; or 'package.module:attr' = another package's scene builder (r1sim.scenes.resolve)",
    )
    ap.add_argument("--model", default=None, help="any MJCF file instead of a scene (no controller: passive physics)")
    ap.add_argument("--controller", choices=CONTROLLERS, default="guarded",
                    help="guarded = the C++ core + r1guard in lockstep (default; the policy if r1rt_py is not built) | "
                    "core = the core alone | policy | hold (a stiff stand fixture) | none")  # fmt: skip
    ap.add_argument("--stack", action="store_true",
                    help="the real r1rt / r1guard / r1d processes against the DDS-bridge simulator (needs CycloneDDS: "
                    "the dds extra + the DDS build of rt/); --controller is ignored")  # fmt: skip
    ap.add_argument("--domain", type=int, default=None, help="--stack: the DDS domain id (default: from the pid)")
    ap.add_argument("--profile", choices=("real", "train"), default="real")
    ap.add_argument("--world", default=None, help="a JSON of declared entities (r1d world.declare) drawn as boxes")
    ap.add_argument("--camera", choices=PRESETS, default="free")
    ap.add_argument("--seed", type=int, default=None, help="a robot / scene variant (r1sim.variants)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--headless", action="store_true", help="render frames offscreen instead of opening a window")
    ap.add_argument("--frames", type=int, default=1, help="headless: the number of frames")
    ap.add_argument("--every", type=float, default=0.1, help="headless: simulated seconds between frames")
    ap.add_argument("--start", type=float, default=0.0, help="headless: simulated seconds before the first frame")
    ap.add_argument("--keys", default="", help="headless: keys at times, e.g. 'P@1.0,N@2.0,C@0' (the window's keys)")
    ap.add_argument("--out", default="out/viewer", help="headless frames / recordings / screenshots")
    ap.add_argument("--mp4", action="store_true", help="headless: also an MP4 of the frames (needs ffmpeg)")
    return ap


def _keys(spec: str) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        k, _, t = item.partition("@")
        if len(k) != 1 or not t:
            raise SystemExit(f"--keys: 'K@t' items (one character, a time), not {item!r}")
        out.append((float(t), k))
    return sorted(out)


def _text_on(img: np.ndarray, left: str, right: str) -> np.ndarray:
    """The overlay text drawn on a headless frame (Pillow, a dev / validation dependency; without it: no text)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - the release venv without the dev group
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    lines = list(zip(left.split("\n"), right.split("\n"), strict=False))
    h = 14 * len(lines) + 8
    dr.rectangle((4, 4, 420, 4 + h), fill=(0, 0, 0))
    for i, (a, b) in enumerate(lines):
        dr.text((10, 8 + 14 * i), f"{a:<16} {b}", fill=(255, 255, 255))
    out: np.ndarray = np.asarray(im)
    return out


class Recorder:
    """Frames of the current camera to PNG files (+ an MP4 through ffmpeg when asked and present)."""

    def __init__(self, app: Any, out: Path, width: int, height: int) -> None:
        self.app, self.out = app, out
        self.r = mujoco.Renderer(app.m, height=height, width=width)
        self.frames: list[Path] = []

    def frame(self, path: Path, *, text: bool = True) -> np.ndarray:
        a = self.app
        self.r.update_scene(a.d, camera=a.camera(), scene_option=a.scene_option())
        a.decorate(self.r.scene)
        img: np.ndarray = np.asarray(self.r.render())
        if text and a.toggles.text:
            img = _text_on(img, *a.overlay_text())
        write_png(path, img)
        self.frames.append(path)
        return img

    def mp4(self, path: Path, fps: float) -> Path | None:
        ff = shutil.which("ffmpeg")
        if ff is None or not self.frames:
            return None
        pattern = str(self.frames[0].parent / "frame_%05d.png")
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", pattern, "-pix_fmt", "yuv420p",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(path)], check=True)  # fmt: skip
        return path

    def close(self) -> None:
        self.r.close()


def headless(a: argparse.Namespace, app: Any) -> dict[str, Any]:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(app, out, a.width, a.height)
    keys = _keys(a.keys)
    frames: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    try:
        for k in range(a.frames):
            t_frame = a.start + k * a.every
            while keys and keys[0][0] <= t_frame + 1e-9:  # each key at its time (the sim advanced to it first)
                tk, ch = keys.pop(0)
                app.advance(tk, max_steps=10**7)
                app.key(ch)
            app.advance(t_frame, max_steps=10**7)
            img = rec.frame(out / f"frame_{k:05d}.png")
            frames.append({"file": f"frame_{k:05d}.png", "t": round(app.t, 4), "mean": round(float(img.mean()), 2),
                           "std": round(float(img.std()), 2), "overlay": dict(app.overlay())})  # fmt: skip
        movie = rec.mp4(out / "viewer.mp4", 1.0 / max(a.every, 1e-3)) if a.mp4 else None
    finally:
        rec.close()
    res = {**app.summary(), "frames": frames, "camera": app.preset, "width": a.width, "height": a.height,
           "wall_s": round(time.perf_counter() - t0, 3), "mp4": None if movie is None else str(movie)}  # fmt: skip
    (out / "viewer.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def interactive(a: argparse.Namespace, app: Any) -> int:  # pragma: no cover - needs a display (and mjpython on macOS)
    import mujoco.viewer

    out = Path(a.out)
    pending: list[str] = []

    def on_key(code: int) -> None:
        if 32 <= code < 127:
            pending.append(chr(code))
        elif code in SPECIAL_KEYS:  # the arrows: the jog's joint (left / right) and step (down / up)
            pending.append(SPECIAL_KEYS[code])

    rec: Recorder | None = None
    n_rec = 0
    with mujoco.viewer.launch_passive(app.m, app.d, key_callback=on_key) as v:
        cam = app.camera()
        v.cam.type, v.cam.trackbodyid, v.cam.fixedcamid = cam.type, cam.trackbodyid, cam.fixedcamid
        v.cam.distance, v.cam.azimuth, v.cam.elevation = cam.distance, cam.azimuth, cam.elevation
        v.cam.lookat[:] = cam.lookat
        wall0, sim0 = time.perf_counter(), app.t
        last_preset = app.preset
        print(HELP)
        while v.is_running():
            while pending:
                ch = pending.pop(0).upper()
                if ch == "R":
                    if rec is None:
                        rec, n_rec = Recorder(app, out, a.width, a.height), 0
                        print(f"recording to {out}/frame_*.png")
                    else:
                        movie = rec.mp4(out / "viewer.mp4", 30.0)
                        rec.close()
                        rec = None
                        print(f"recording stopped ({n_rec} frames){'; ' + str(movie) if movie else ''}")
                elif ch == ".":
                    shot = Recorder(app, out, a.width, a.height)
                    path = out / f"screenshot_{time.strftime('%Y%m%dT%H%M%S')}.png"
                    shot.frame(path)
                    shot.close()
                    print(f"screenshot {path}")
                else:
                    what = app.key(ch)
                    if what:
                        print(f"[{app.t:7.2f} s] {what}")
                if app.toggles.paused:
                    wall0, sim0 = time.perf_counter(), app.t
            if app.preset != last_preset:
                cam = app.camera()
                v.cam.type, v.cam.trackbodyid, v.cam.fixedcamid = cam.type, cam.trackbodyid, cam.fixedcamid
                v.cam.distance, v.cam.azimuth, v.cam.elevation = cam.distance, cam.azimuth, cam.elevation
                v.cam.lookat[:] = cam.lookat
                last_preset = app.preset
            if getattr(app, "realtime", False):  # --stack: the bridge's physics runs on its own; draw its latest state
                app.refresh()
            else:
                app.advance(sim0 + (time.perf_counter() - wall0), max_steps=200)
            with v.lock():
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = app.toggles.contacts
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = app.toggles.contacts
                v.opt.frame = mujoco.mjtFrame.mjFRAME_BODY if app.toggles.frames else mujoco.mjtFrame.mjFRAME_NONE
                v.user_scn.ngeom = 0
                app.decorate(v.user_scn)
                if app.toggles.text:
                    left, right = app.overlay_text()
                    v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
                else:
                    v.set_texts([])
            v.sync()
            if rec is not None:
                rec.frame(out / f"frame_{n_rec:05d}.png")
                n_rec += 1
            time.sleep(1.0 / 60.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    if not a.headless and sys.platform == "darwin":
        import mujoco.viewer as mv

        if getattr(mv, "_MJPYTHON", None) is None:  # how MuJoCo itself tells (launch_passive refuses otherwise)
            print("macOS: the window needs mjpython - scripts/viewer.sh (or --headless); docs/sim/VIEWER.md")
            return 2
    app: Any
    if a.stack:
        from .stack import StackApp, StackConfig, why_not

        reason = why_not()
        if reason:
            print(f"--stack cannot run here: {reason}")
            return 2
        app = StackApp(StackConfig(scene=a.scene, seed=a.seed or 0, domain=a.domain, world_json=a.world,
                                   camera=a.camera, width=a.width, height=a.height))  # fmt: skip
    else:
        app = App(ViewerConfig(scene=a.scene, controller=a.controller, profile=a.profile, model_path=a.model,
                               world_json=a.world, seed=a.seed, camera=a.camera, width=a.width,
                               height=a.height))  # fmt: skip
    try:
        if a.headless:
            res = headless(a, app)
            print(json.dumps({k: res[k] for k in ("t", "scene", "controller", "camera", "wall_s", "mp4")}))
            print(json.dumps(res["overlay"]))
            return 0
        return interactive(a, app)
    finally:
        close = getattr(app, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    raise SystemExit(main())
