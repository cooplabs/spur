"""Client of the Filament render worker (DESIGN D13; the worker is render/r1render/worker.py).

The physics process (MuJoCo 3.5.0, Python 3.11) exports the world it simulates as MJCF (`spec.to_xml()`; textures and
meshes are referenced by absolute path) and starts the worker in the render venv (MuJoCo 3.15.0, Python 3.12, its own
pinned project `render/`). Per frame it sends qpos and receives the picture; the worker runs forward kinematics on its
own compile of the same MJCF. `body_poses` exists so a test can prove both compiles put every body in the same place
for the same qpos (the condition for the pictures to show what the physics did).

Where the render venv lives (never inside the repository or a synced folder): $R1_RENDER_VENV, default
~/.venvs/r1mc3-render. Create / update it with
    UV_PROJECT_ENVIRONMENT="$R1_RENDER_VENV" uv sync --project render --frozen
(`scripts/ci.sh render` does this). Without it `available()` is False and the Filament paths are skipped.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import time
from functools import cache
from pathlib import Path
from typing import IO, Any

import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parents[2]
RENDER_PROJECT = REPO / "render"
WORKER_SCRIPT = RENDER_PROJECT / "r1render" / "worker.py"
RENDER_MUJOCO_VERSION = "3.15.0"  # render/pyproject.toml pins it exactly
DEFAULT_VENV = Path.home() / ".venvs" / "r1mc3-render"


def venv_dir() -> Path:
    return Path(os.environ.get("R1_RENDER_VENV") or DEFAULT_VENV)


def venv_python() -> Path:
    return venv_dir() / "bin" / "python"


@cache
def available() -> bool:
    """True if the render venv exists and carries the pinned MuJoCo with the Filament module."""
    py = venv_python()
    if not py.exists():
        return False
    probe = "import mujoco, mujoco._render_filament; print(mujoco.__version__)"
    try:
        out = subprocess.run([str(py), "-I", "-c", probe], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and out.stdout.strip() == RENDER_MUJOCO_VERSION


# The model fields the worker takes from the physics model after compiling the exported MJCF (`MjSpec.to_xml()` prints
# 6 significant digits: computed placements come back up to ~5e-7 off; see the worker's docstring): everything that
# places a body, a geom, a site, a camera or a light. Sizes must match (the same MJCF); values are then bit-identical.
SYNC_FIELDS = (
    "body_pos", "body_quat", "body_ipos", "body_iquat", "jnt_pos", "jnt_axis", "geom_pos", "geom_quat", "geom_size",
    "site_pos", "site_quat", "site_size", "cam_pos", "cam_quat", "cam_fovy", "light_pos", "light_dir",
)  # fmt: skip


def sync_payload(m: Any) -> tuple[list[tuple[str, list[int]]], bytes]:
    """The physics model's placement arrays for the worker's "sync" (names + shapes, float64 bytes back to back)."""
    fields, chunks = [], []
    for name in SYNC_FIELDS:
        a = np.ascontiguousarray(np.asarray(getattr(m, name), dtype=np.float64))
        fields.append((name, list(a.shape)))
        chunks.append(a.tobytes())
    return fields, b"".join(chunks)


def apply_sync(m: Any, src: Any) -> None:
    """The same copy on a model in THIS process (tests: an exported MJCF recompiled here then equals the original)."""
    for name in SYNC_FIELDS:
        getattr(m, name)[...] = getattr(src, name)


def export_world(world: Any, out_dir: str | Path | None = None, name: str = "world.xml") -> Path:
    """Write the world's MJCF (from the MjSpec it was compiled from) for the worker. `world` is an r1sim.world.World
    (built by build_world, which keeps the spec) or an MjSpec. Returns the file's path."""
    spec = getattr(world, "spec", world)
    if spec is None or not hasattr(spec, "to_xml"):
        raise ValueError("export_world needs a World built by r1sim.world.build_world (with its spec) or an MjSpec")
    d = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="r1render_"))
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(spec.to_xml())
    return p


class WorkerError(RuntimeError):
    pass


class RenderWorker:
    """A running render worker for one exported world. Use as a context manager.

    `world` may be a World / MjSpec (exported to a temp dir) or the path of an exported MJCF. `model`: the compiled
    physics model whose placements the worker copies (default: world.model for a World; give it for a spec / path,
    or None to keep the XML's 6-digit values)."""

    def __init__(
        self, world: Any, *, model: Any = "auto", python: str | Path | None = None, timeout_s: float = 120.0
    ) -> None:
        self.xml = Path(world) if isinstance(world, (str, Path)) else export_world(world)
        if isinstance(model, str) and model == "auto":
            model = getattr(world, "model", None)
        py = Path(python) if python is not None else venv_python()
        if not py.exists():
            raise WorkerError(f"no render venv python at {py} (set R1_RENDER_VENV; scripts/ci.sh render creates it)")
        self.timeout_s = timeout_s
        t0 = time.perf_counter()
        # -I: isolated (no user site, no PYTHONPATH, no cwd on sys.path); the worker is a self-contained script
        self._p = subprocess.Popen(
            [str(py), "-I", str(WORKER_SCRIPT), str(self.xml)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.xml.parent),
        )
        info, _ = self._call({"op": "init"})
        self.startup_s = time.perf_counter() - t0
        self.version: str = info["version"]
        self.nq: int = int(info["nq"])
        self.nbody: int = int(info["nbody"])
        self.cameras: list[str] = list(info["cams"])
        self.bodies: list[str] = list(info["bodies"])
        if model is not None:
            fields, payload = sync_payload(model)
            self._call({"op": "sync", "fields": fields}, payload)

    # ------------------------------------------------------------------ plumbing
    def _call(self, header: dict[str, Any], payload: bytes = b"") -> tuple[dict[str, Any], bytes]:
        p = self._p
        assert p.stdin is not None and p.stdout is not None
        h = json.dumps(dict(header, nbytes=len(payload))).encode()
        try:
            p.stdin.write(struct.pack("<I", len(h)) + h + payload)
            p.stdin.flush()
            (hlen,) = struct.unpack("<I", _read_exact(p.stdout, 4))
            reply = json.loads(_read_exact(p.stdout, hlen).decode())
            n = int(reply.get("nbytes", 0))
            data = _read_exact(p.stdout, n) if n else b""
        except (BrokenPipeError, EOFError) as e:
            err = p.stderr.read().decode(errors="replace") if p.stderr is not None and p.poll() is not None else ""
            raise WorkerError(f"render worker died: {e}\n{err[-2000:]}") from e
        if "error" in reply:
            raise WorkerError(reply["error"])
        return reply, data

    def _qpos(self, qpos: npt.ArrayLike) -> bytes:
        q = np.ascontiguousarray(np.asarray(qpos, dtype=np.float64))
        if q.shape != (self.nq,):
            raise ValueError(f"qpos has shape {q.shape}, the world nq = {self.nq}")
        return q.tobytes()

    # ------------------------------------------------------------------ API
    def body_poses(self, qpos: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """(xpos (k, nbody, 3), xquat (k, nbody, 4)) from the worker's own forward kinematics for k qpos rows."""
        q = np.atleast_2d(np.asarray(qpos, dtype=np.float64))
        if q.shape[1] != self.nq:
            raise ValueError(f"qpos rows have {q.shape[1]} values, the world nq = {self.nq}")
        _, data = self._call({"op": "poses", "n": len(q)}, np.ascontiguousarray(q).tobytes())
        a = np.frombuffer(data, dtype=np.float64).reshape(len(q), self.nbody, 7)
        return a[:, :, :3].copy(), a[:, :, 3:].copy()

    def render(
        self,
        qpos: npt.ArrayLike,
        camera: str | None,
        width: int,
        height: int,
        *,
        pose: tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike] | None = None,
        fovy_deg: float | None = None,
        znear: float | None = None,
        zfar: float | None = None,
        shadows: bool | None = None,
        headlight: float | None = None,
    ) -> npt.NDArray[np.uint8]:
        """RGB (height, width, 3) uint8, rows top first. `camera`: a model camera by name (its pose from the worker's
        kinematics, its fovy unless fovy_deg is given), or None with pose = (pos, forward, up) in the world and
        fovy_deg - a centred pinhole with square pixels either way. No depth from the worker (see its docstring):
        ground-truth depth stays on the classic renderer (r1sim.lens.HeadCamera.depth_z)."""
        req: dict[str, Any] = {"op": "render", "width": int(width), "height": int(height), "camera": camera}
        if pose is not None:
            req["pos"], req["forward"], req["up"] = ([float(v) for v in np.asarray(x).ravel()] for x in pose)
        if fovy_deg is not None:
            req["fovy_deg"] = float(fovy_deg)
        if znear is not None:
            req["znear"] = float(znear)
        if zfar is not None:
            req["zfar"] = float(zfar)
        if shadows is not None:
            req["shadows"] = bool(shadows)
        if headlight is not None:  # scale of the worker's emulated headlight (1 = the model's headlight)
            req["headlight"] = float(headlight)
        _, data = self._call(req, self._qpos(qpos))
        return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()

    def close(self) -> None:
        p = getattr(self, "_p", None)
        if p is None or p.poll() is not None:
            return
        try:
            self._call({"op": "quit"})
        except WorkerError:
            pass
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        for f in (p.stdin, p.stdout, p.stderr):
            if f is not None:
                f.close()

    def __enter__(self) -> RenderWorker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _read_exact(f: IO[bytes], n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("render worker closed its output")
        buf += chunk
    return bytes(buf)
