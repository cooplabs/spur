"""r1render worker: draws r1sim worlds with MuJoCo's Filament renderer in a process of its own (DESIGN D13).

Why a separate process: the physics, the controllers and every test run on MuJoCo 3.5.0 / Python 3.11 (the version
the balance policy was trained with; the A1 bit-match depends on it). MuJoCo's Filament renderer reached the Python
wheels only in 3.14 (`mujoco._render_filament`, Python >= 3.12). So this worker - and nothing else - runs on the
pinned newer MuJoCo (render/pyproject.toml): it compiles the SAME world (the MJCF r1sim exports with
`spec.to_xml()`, textures and meshes by absolute path), takes qpos from the physics process, runs forward kinematics
and renders. It never steps physics. The client is `r1sim.render_client.RenderWorker` (main project).

Protocol (stdin / stdout, binary, one request -> one reply): each message is a little-endian uint32 header length, a
UTF-8 JSON header, then `header["nbytes"]` bytes of payload (0 if absent).
  {"op": "init"}                                    -> {"version", "nq", "nbody", "ncam", "cams", "bodies"}
  {"op": "sync", "fields": [[name, shape], ...]} + float64 arrays back to back
                                                    -> {"n": fields set}: the physics model's own values of the
                                                       kinematic / camera / light fields (see below)
  {"op": "poses", "n": k} + k*nq float64 (qpos)     -> k*nbody*7 float64 (xpos 3 + xquat 4 per body, after
                                                       mj_kinematics)
  {"op": "render", "width", "height", "camera": name | null, "pos", "forward", "up", "fovy_deg", "znear", "zfar",
   "shadows": bool, "headlight": scale} + nq float64 -> height*width*3 uint8 RGB (rows top first)
  {"op": "quit"}                                    -> {} and exit
Errors come back as {"error": "..."} (no payload). stdout carries the protocol only: diagnostics go to stderr.

Why "sync": `MjSpec.to_xml()` prints 6 significant digits, so any value r1sim computes (a phone's fitted pose, a
placed object) comes back from the XML up to ~5e-7 off (measured: 4e-7 m on a body position, session 4). The client
therefore sends the compiled physics model's body / joint / geom / site / camera / light placement arrays once, and
the worker copies them over its own compile: same sizes are checked, the values are then the physics model's to the
last bit, and the pose identity holds exactly (meshes come from the same files; inline mesh vertices keep 6 digits).

No depth: in 3.15.0 Filament's DRAW_MODE_DEPTH gives a normalised visualisation (values on a 1/255 grid, not metres)
and reading a DEPTH32F target aborts the process ("Unsupported pixel format", probe 10 Oct). Ground-truth depth is
geometry, not looks: r1sim keeps rendering it with MuJoCo 3.5.0's classic renderer (r1sim.lens.HeadCamera.depth_z).

This file is self-contained (stdlib + numpy + mujoco only) and is started as a script by the client, so the render
venv needs no install of this project.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from typing import Any, BinaryIO

import mujoco
import numpy as np

HEADLESS_BACKEND = "opengl_headless"  # Filament over OpenGL (Metal on macOS); Vulkan headless had driver issues (#2979)
# Filament has no headlight; r1sim's photo look relies on MuJoCo's (textures.HEADLIGHT: ambient 0.40 + diffuse 0.12,
# standing in for the light the white walls and ceiling bounce). The worker emulates its DIFFUSE term with a
# directional light along each view's camera axis (intensity = this factor x mean(diffuse)) and its AMBIENT term with a
# uniform image-based light (below) - look parameters, not measurements.
HEADLIGHT_INTENSITY_PER_UNIT = 30000.0
# MuJoCo's headlight AMBIENT term lights every surface the same whatever its normal. Filament's stand-in is an
# image-based light; mjrf's DEFAULT one is a warm, directional environment: it lit the robot's torso (seen from the
# head camera, at the bottom of every head frame) dark brown (mean grey 59, RGB 71 / 56 / 50 at the A13 pose) where
# the classic renderer draws it grey (127) and the real frames show a light-grey body (117) - session 4, found by the
# A13 click test (a model clicked ~26 px low on sim frames only). The worker replaces it with a UNIFORM white IBL of
# intensity this factor x mean(headlight ambient) (ambient 0.40 -> 6000), and keeps the directional headlight for the
# diffuse term at HEADLIGHT_INTENSITY_PER_UNIT x mean(diffuse) (0.12 -> 3600). Matched to the classic frame's torso
# and table at the A13 pose (a look parameter, like the classic headlight itself).
AMBIENT_IBL_INTENSITY_PER_UNIT = 15000.0


def _read_exact(f: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("client closed the pipe")
        buf += chunk
    return bytes(buf)


def read_msg(f: BinaryIO) -> tuple[dict[str, Any], bytes]:
    (hlen,) = struct.unpack("<I", _read_exact(f, 4))
    header = json.loads(_read_exact(f, hlen).decode())
    n = int(header.get("nbytes", 0))
    return header, (_read_exact(f, n) if n else b"")


def write_msg(f: BinaryIO, header: dict[str, Any], payload: bytes = b"") -> None:
    header = dict(header, nbytes=len(payload))
    h = json.dumps(header).encode()
    f.write(struct.pack("<I", len(h)) + h)
    if payload:
        f.write(payload)
    f.flush()


class FilamentWorld:
    """One compiled world + a headless Filament context; targets are created per image size on first use."""

    def __init__(self, xml_path: str) -> None:
        import mujoco._render_filament as mjrf
        from mujoco.experimental.studio import window

        self.mjrf = mjrf
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        # the window only provides the GL context (its own size is irrelevant: we read targets, not the window)
        self._win = window.Window("r1render", 64, 64, HEADLESS_BACKEND)
        self.ctx = mjrf.Context(mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL))
        self.ctx.set_clear_color([0.0, 0.0, 0.0])
        self.objs = mjrf.ModelObjects(self.ctx, self.m)
        self.scene = self.ctx.create_scene(mjrf.SceneParams())
        self.scene.configure_from_model(self.m)
        self.lights = mjrf.ModelLights(self.scene, self.objs)
        self.rend = mjrf.ModelRenderables(self.scene, self.objs)
        self._targets: dict[tuple[int, int], tuple[Any, Any, np.ndarray]] = {}
        hl = self.m.vis.headlight
        on = bool(hl.active)
        self.headlight_intensity = HEADLIGHT_INTENSITY_PER_UNIT * float(np.mean(hl.diffuse)) if on else 0.0
        self.ambient_intensity = AMBIENT_IBL_INTENSITY_PER_UNIT * float(np.mean(hl.ambient)) if on else 0.0
        self.ambient = self._uniform_ibl(self.ambient_intensity)
        self.headlight = self.ctx.create_light(
            mjrf.LightParams(
                type=mjrf.LightType.LIGHT_DIRECTIONAL,
                color=np.ones(3, np.float32),
                intensity=self.headlight_intensity,
                cast_shadows=False,
            )
        )
        self.scene.add_light(self.headlight)

    def _uniform_ibl(self, intensity: float) -> Any:
        """A uniform white image-based light (a 16 x 16 cube of 1.0): direction-free ambient, replacing mjrf's default
        environment light (which tints and darkens surfaces facing away from its bright side)."""
        mjrf, n = self.mjrf, 16
        tex = self.ctx.create_texture(
            mjrf.TextureConfig(
                width=n,
                height=n,
                format=mjrf.PixelFormat.PIXEL_FORMAT_RGB8,
                color_space=mjrf.ColorSpace.COLORSPACE_LINEAR,
                sampler_type=mjrf.SamplerType.TEXTURE_CUBE,
            )
        )
        data = mjrf.TextureData()
        data.set_image(np.full((6 * n, n, 3), 255, np.uint8).tobytes())
        tex.set_data(data)
        light = self.ctx.create_light(
            mjrf.LightParams(type=mjrf.LightType.LIGHT_IMAGE, texture=tex, intensity=intensity)
        )
        self.scene.add_light(light)
        self._ambient_tex = (tex, data)  # keep the texture alive with the light
        return light

    # ------------------------------------------------------------------ kinematics
    def set_qpos(self, qpos: np.ndarray) -> None:
        if qpos.shape != (self.m.nq,):
            raise ValueError(f"qpos has {qpos.shape}, the model nq = {self.m.nq}")
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)

    def sync(self, fields: list[tuple[str, list[int]]], payload: bytes) -> int:
        off = 0
        for name, shape in fields:
            dst = getattr(self.m, name)
            n = int(np.prod(shape)) * 8
            src = np.frombuffer(payload[off : off + n], dtype=np.float64).reshape(shape)
            if tuple(dst.shape) != tuple(src.shape):
                raise ValueError(f"{name}: the worker's model has {dst.shape}, the client sent {src.shape}")
            dst[...] = src
            off += n
        if off != len(payload):
            raise ValueError(f"sync payload has {len(payload)} bytes, the fields need {off}")
        mujoco.mj_forward(self.m, self.d)
        return len(fields)

    def poses(self, qpos: np.ndarray) -> np.ndarray:
        self.set_qpos(qpos)
        return np.concatenate([self.d.xpos, self.d.xquat], axis=1).astype(np.float64)

    # ------------------------------------------------------------------ rendering
    def _target(self, w: int, h: int) -> tuple[Any, Any, np.ndarray]:
        key = (w, h)
        if key not in self._targets:
            mjrf = self.mjrf
            fmt, buf = mjrf.PixelFormat.PIXEL_FORMAT_RGB8, np.zeros((h, w, 3), np.uint8)
            tgt = self.ctx.create_render_target(mjrf.RenderTargetConfig(color_format=fmt))
            tgt.resize(w, h)
            rd = mjrf.ReadPixelsRequest()
            rd.target = tgt
            rd.set_buffer(buf)
            self._targets[key] = (tgt, rd, buf)
        return self._targets[key]

    def camera(self, req: dict[str, Any], width: int, height: int) -> Any:
        """A centred pinhole: the named model camera (pose from the kinematics, the model's fovy) or an explicit pose
        (pos, forward, up in the world) with fovy_deg. Square pixels: the horizontal extent follows the aspect."""
        m, d = self.m, self.d
        name = req.get("camera")
        if name is not None:
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise KeyError(f"no camera named {name!r}")
            R = d.cam_xmat[cid].reshape(3, 3)
            pos, forward, up = d.cam_xpos[cid].copy(), -R[:, 2], R[:, 1]
            fovy = float(req.get("fovy_deg") or m.cam_fovy[cid])
        else:
            pos, forward, up = (np.asarray(req[k], dtype=np.float64) for k in ("pos", "forward", "up"))
            fovy = float(req["fovy_deg"])
        extent = float(m.stat.extent)
        near = float(req.get("znear") or m.vis.map.znear * extent)
        far = float(req.get("zfar") or m.vis.map.zfar * extent)
        top = near * np.tan(np.radians(fovy) / 2.0)
        return self.mjrf.Camera(
            pos=pos,
            forward=forward,
            up=up,
            frustum_center=0.0,
            frustum_width=top * width / height,
            frustum_bottom=-top,
            frustum_top=top,
            frustum_near=near,
            frustum_far=far,
        )

    def render(self, req: dict[str, Any], qpos: np.ndarray) -> bytes:
        mjrf = self.mjrf
        w, h = int(req["width"]), int(req["height"])
        self.set_qpos(qpos)
        self.lights.update(self.d)
        self.rend.update(self.d)
        cam = self.camera(req, w, h)
        self.headlight.set_transform(np.asarray(cam.pos, np.float32), np.asarray(cam.forward, np.float32))
        self.headlight.set_intensity(self.headlight_intensity * float(req.get("headlight", 1.0)))
        self.ambient.set_intensity(self.ambient_intensity * float(req.get("headlight", 1.0)))
        tgt, rd, buf = self._target(w, h)
        rr = mjrf.RenderRequest(camera=cam, draw_mode=mjrf.DrawMode.DRAW_MODE_DEFAULT)
        rr.scene = self.scene
        rr.target = tgt
        rr.viewport.left, rr.viewport.bottom, rr.viewport.width, rr.viewport.height = 0, 0, w, h
        if "shadows" in req:
            rr.enable_shadows = bool(req["shadows"])
        self._win.NewFrame()
        frame = self.ctx.render([rr], [rd])
        self.ctx.wait_for_frame(frame)
        return buf.tobytes()


def serve(xml_path: str, fin: BinaryIO, fout: BinaryIO) -> int:
    t0 = time.perf_counter()
    world = FilamentWorld(xml_path)
    print(f"r1render: mujoco {mujoco.__version__}, world loaded in {time.perf_counter() - t0:.2f} s", file=sys.stderr)
    m = world.m
    while True:
        try:
            header, payload = read_msg(fin)
        except EOFError:
            return 0
        op = header.get("op")
        try:
            if op == "init":
                write_msg(
                    fout,
                    {
                        "version": mujoco.__version__,
                        "nq": int(m.nq),
                        "nbody": int(m.nbody),
                        "ncam": int(m.ncam),
                        "cams": [m.cam(i).name for i in range(m.ncam)],
                        "bodies": [m.body(i).name for i in range(m.nbody)],
                    },
                )
            elif op == "sync":
                write_msg(fout, {"n": world.sync([(f[0], list(f[1])) for f in header["fields"]], payload)})
            elif op == "poses":
                q = np.frombuffer(payload, dtype=np.float64).reshape(int(header["n"]), m.nq)
                out = np.stack([world.poses(qi) for qi in q])
                write_msg(fout, {"n": len(q)}, out.tobytes())
            elif op == "render":
                q = np.frombuffer(payload, dtype=np.float64)
                write_msg(fout, {"width": header["width"], "height": header["height"]}, world.render(header, q))
            elif op == "quit":
                write_msg(fout, {})
                return 0
            else:
                write_msg(fout, {"error": f"unknown op {op!r}"})
        except Exception as e:  # report and keep serving: the client decides
            write_msg(fout, {"error": f"{type(e).__name__}: {e}"})


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: worker.py WORLD.xml  (requests on stdin, replies on stdout)", file=sys.stderr)
        return 2
    return serve(args[0], sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    sys.exit(main())
