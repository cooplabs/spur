# r1render - the render-only worker (DESIGN D13)

MuJoCo's Filament renderer reached the Python wheels in 3.14 (`mujoco._render_filament`, Python >= 3.12). The physics,
the controllers and every test stay on MuJoCo 3.5.0 / Python 3.11 (the version the balance policy was trained with);
this small project pins **mujoco==3.15.0 on Python 3.12** for drawing pictures only.

- `r1render/worker.py` - a self-contained script (stdlib + numpy + mujoco): compiles the MJCF the physics process
  exports (`r1sim.render_client.export_world`), takes the physics model's placement arrays once ("sync"), then per
  request: qpos -> forward kinematics -> a Filament frame (headless, OpenGL backend), or body poses (for the identity
  test). Protocol in its docstring. Never steps physics.
- The client: `r1sim.render_client.RenderWorker` (main project); the head chain: `r1sim.lens.HeadCamera(m,
  backend="filament", worker=...)`.

The venv lives outside the repository and any synced folder: `$R1_RENDER_VENV` (default `~/.venvs/r1mc3-render`).

```bash
UV_PROJECT_ENVIRONMENT="${R1_RENDER_VENV:-$HOME/.venvs/r1mc3-render}" uv sync --project render --frozen --python 3.12
```
(`scripts/ci.sh render` does this.) Without the venv, `r1sim.render_client.available()` is False and the Filament
tests skip. The project is not installed into its venv (`package = false`): the client starts the worker as a script
with `python -I`, so one venv serves every checkout.
