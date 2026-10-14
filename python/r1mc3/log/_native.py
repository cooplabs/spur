"""Load `r1rt_py`, the C++ binding that holds the segment reader (`SegmentView`, rt/bindings/r1rt_py_shm.cpp).

r1log needs the C++ ring / seqlock code (acquire / release ordering) - a Python mmap view has no barriers and could
read a published slot's old bytes on the Orin's weakly ordered cores. On the robot the module is installed next to
r1log; in a checkout it is found in the C++ build trees scripts/ci.sh builds ($R1MC3_BUILD_DIR/rt, then
<repo>/build/rt), as r1sim.cpp_policy does (r1mc3 must not import the simulator).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

MODULE = "r1rt_py"
REPO = Path(__file__).resolve().parents[3]


def build_dirs() -> list[Path]:
    """Where the C++ build trees are, in search order."""
    out: list[Path] = []
    env = os.environ.get("R1MC3_BUILD_DIR")
    if env:
        out.append(Path(env) / "rt")
    out.append(REPO / "build" / "rt")
    return out


def find_module() -> Path | None:
    for d in build_dirs():
        hits = sorted(d.glob(f"{MODULE}*.so")) + sorted(d.glob(f"{MODULE}*.pyd"))
        if hits:
            return hits[0]
    return None


def native() -> ModuleType:
    """The binding (already imported, installed, or from a build tree); ImportError says what to do."""
    if MODULE in sys.modules:
        return sys.modules[MODULE]
    path = find_module()
    if path is None:
        try:
            return importlib.import_module(MODULE)
        except ImportError:
            where = ", ".join(str(d) for d in build_dirs())
            raise ImportError(
                f"{MODULE} is not built (looked in {where}): run `bash scripts/ci.sh cpp`, or install the module"
            ) from None
    loader = importlib.machinery.ExtensionFileLoader(MODULE, str(path))
    spec = importlib.util.spec_from_file_location(MODULE, path, loader=loader)
    if spec is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = mod
    loader.exec_module(mod)
    return mod


def has_segment_reader() -> bool:
    """True when the binding is importable and carries SegmentView (tests skip otherwise)."""
    try:
        return hasattr(native(), "SegmentView")
    except ImportError:
        return False
