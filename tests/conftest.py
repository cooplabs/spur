"""Shared test hooks: the rendering tests (`needs_gl`) need an OpenGL context.

MuJoCo renders offscreen through CGL (macOS), EGL or OSMesa (Linux). A host that cannot make any context - GitHub's
macOS runners are VMs without a GPU ("CGLError: invalid pixel format") - skips these tests with the reason; the Linux
CI jobs render through OSMesa, so the rendering itself stays tested.
"""

from __future__ import annotations

import functools

import pytest


@functools.cache
def _gl_unavailable() -> str | None:
    """None when an 8 x 8 offscreen renderer can be made here, else why not."""
    try:
        import mujoco

        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>")
        with mujoco.Renderer(m, 8, 8):
            pass
    except Exception as e:  # any failure to make a context means: no rendering on this host
        return f"{type(e).__name__}: {e}"
    return None


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("needs_gl") is not None and (why := _gl_unavailable()) is not None:
        pytest.skip(f"no OpenGL context on this host ({why})")
