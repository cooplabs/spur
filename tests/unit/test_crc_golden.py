"""The committed hg CRC golden file is reproducible from the Python reference (tools/crc_golden.py: a
re-implementation of unitree_sdk2_python's CRC and pack formats, anchored on CRC-32/MPEG-2). The C++ side
(rt/tests/test_crc.cpp) must reproduce the same file; together they pin r1rt's CRC to Unitree's."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "tests" / "assets" / "crc_golden"


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("crc_golden", ROOT / "tools" / "crc_golden.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crc_golden"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_committed_golden_file_regenerates_identically() -> None:
    assert _tool().generate() == (ASSETS / "crc_golden.txt").read_text()


def test_golden_file_sha256_pinned() -> None:
    sums: dict[str, str] = {}
    for line in (ASSETS / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest_, name = line.split()
            sums[name] = digest_
    digest = hashlib.sha256((ASSETS / "crc_golden.txt").read_bytes()).hexdigest()
    assert sums.get("crc_golden.txt") == digest


def test_reference_crc_matches_the_catalogue_check_value() -> None:
    """CRC-32/MPEG-2 of "123456789" is 0x0376E6E7 (the CRC catalogue); Unitree's word CRC is that CRC over
    big-endian words, so the anchor proves the reference is the real algorithm, not a stand-in."""
    assert _tool().crc_mpeg2(b"123456789") == 0x0376E6E7
