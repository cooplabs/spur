"""DDS on the Python side (session 5): Unitree's hg message types for CycloneDDS's Python binding + the hg CRC.

The real-time core and the guard speak DDS from C++ (`rt/`, our CycloneDDS-CXX 0.10.2 build); the simulator's DDS
bridge (DESIGN.md §12.4), r1d's hands / motion-switcher clients and the HW1 recording tools speak it from Python
with the same wire types. The optional dependency group `dds` (`cyclonedds==0.10.2`) provides the binding; on macOS
there is no wheel, so it builds against our C library: export `CYCLONEDDS_HOME` (= `R1MC3_DDS_PREFIX`, default
`~/.cache/r1mc3/cyclonedds-0.10.2`) before `uv sync --extra dds` (scripts/ci.sh and run.sh do it).

`available()` says whether the binding imports; tests that need it are marked `needs_dds`.
"""

from __future__ import annotations


def available() -> bool:
    """True when CycloneDDS's Python binding imports (the optional `dds` dependency group is installed)."""
    try:
        import cyclonedds.domain  # noqa: F401
    except ImportError:
        return False
    return True
