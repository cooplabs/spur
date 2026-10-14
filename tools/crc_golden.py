"""Golden CRCs for the hg LowCmd_ / LowState_ messages (the C++ test rt/tests/test_crc.cpp must reproduce them).

    uv run python tools/crc_golden.py            # rewrites tests/assets/crc_golden/crc_golden.txt (+ SHA256SUMS)
    uv run python tools/crc_golden.py --check    # exit 1 if the committed file differs from a fresh generation

The reference is a re-implementation, from reading (never running) unitree_sdk2_python's
`unitree_sdk2py/utils/crc.py` at 814556d15970dd2ecf1c9984e845ca02ab07e206 (BSD-3-Clause, Copyright (c) Unitree
Robotics): the struct formats `__packFmtHGLowCmd` / `__packFmtHGLowState`, the word conversion `__Trans` (little-endian
words, all but the last) and the pure-Python CRC `_crc_py`. It is what r1mc2 (v2) ran on the robot:
`r1mc2/backends/robot.py:357-391` calls `unitree_sdk2py.utils.crc.CRC().Crc(msg)` before every write (v2's own
tests/fake_sdk/.../crc.py is a zlib stand-in, not the real algorithm). Its Linux path calls a bundled .so with the same
algorithm (unitree_sdk2 crc.h crc32_core).

Independent anchor: the word CRC equals CRC-32/MPEG-2 (catalogue check value "123456789" -> 0x0376E6E7) over the
big-endian bytes of each word; this script asserts both on every message.

Messages: all zeros; the R1 stand command (the 26 joints in their hg slots, mode 1, POSE_STAND, the stiff gains);
random fields from fixed seeds; LowState_ zeros and random. File format, one message per line:
    <name> <lowcmd|lowstate> <crc as 8 hex digits> <message bytes as hex, crc field included>
"""

from __future__ import annotations

import argparse
import hashlib
import random
import struct
import sys
from collections.abc import Sequence
from pathlib import Path

from r1sim import joints as J

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tests" / "assets" / "crc_golden"
OUT = OUT_DIR / "crc_golden.txt"

# unitree_sdk2py utils/crc.py: "#size 1004" / "#size 2092"
FMT_HG_LOWCMD = "<2B2x" + "B3x5fI" * 35 + "5I"
FMT_HG_LOWSTATE = "<2I2B2xI" + "13fh2x" + "B3x4f2hf7I" * 35 + "40B5I"
N_SLOTS = 35
POLY = 0x04C11DB7


def words_of(packed: bytes) -> list[int]:
    """unitree_sdk2py __Trans: little-endian 32-bit words, all but the last (the crc)."""
    n = (len(packed) >> 2) - 1
    return [int.from_bytes(packed[4 * i : 4 * i + 4], "little") for i in range(n)]


def crc_py(data: Sequence[int]) -> int:
    """unitree_sdk2py _crc_py (bit-wise, MSB first per word)."""
    crc = 0xFFFFFFFF
    for current in data:
        bit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ POLY
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if current & bit:
                crc ^= POLY
            bit >>= 1
    return crc


def _mpeg2_table() -> list[int]:
    t = []
    for i in range(256):
        c = i << 24
        for _ in range(8):
            c = ((c << 1) ^ POLY) & 0xFFFFFFFF if c & 0x80000000 else (c << 1) & 0xFFFFFFFF
        t.append(c)
    return t


_T = _mpeg2_table()


def crc_mpeg2(data: bytes) -> int:
    """CRC-32/MPEG-2 (catalogue: init 0xFFFFFFFF, poly 0x04C11DB7, no reflection, no xorout)."""
    crc = 0xFFFFFFFF
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _T[((crc >> 24) ^ b) & 0xFF]
    return crc


def f32(x: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", x))[0])


def pack_lowcmd(mode_pr: int, mode_machine: int, motors: Sequence[Sequence[float | int]], reserve: Sequence[int],
                crc: int) -> bytes:  # fmt: skip
    """motors: 35 x (mode, q, dq, tau, kp, kd, reserve) - the order of __PackHGLowCmd."""
    vals: list[float | int] = [mode_pr, mode_machine]
    for m in motors:
        vals.extend(m)
    vals.extend(reserve)
    vals.append(crc)
    return struct.pack(FMT_HG_LOWCMD, *vals)


def pack_lowstate(fields: list[float | int]) -> bytes:
    return struct.pack(FMT_HG_LOWSTATE, *fields)


def with_crc(fmt: str, packed: bytes) -> tuple[bytes, int]:
    """Compute the CRC, check it against the independent MPEG-2 form, and write it into the crc field."""
    words = words_of(packed)
    crc = crc_py(words)
    be = b"".join(w.to_bytes(4, "big") for w in words)
    if crc != crc_mpeg2(be):
        raise AssertionError("bit-wise word CRC != CRC-32/MPEG-2 over big-endian words")
    out = packed[:-4] + struct.pack("<I", crc)
    if words_of(out) != words:
        raise AssertionError("the crc field must not be covered")
    return out, crc


def lowcmd_zeros() -> bytes:
    return pack_lowcmd(0, 0, [(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)] * N_SLOTS, [0, 0, 0, 0], 0)


def lowcmd_stand() -> bytes:
    """The R1 stand command as v2's robot backend fills it (robot.py:376-391): slot = J.SLOT[i], mode 1."""
    motors: list[tuple[float | int, ...]] = [(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)] * N_SLOTS
    for i, s in enumerate(J.SLOT):
        motors[int(s)] = (1, f32(J.POSE_STAND[i]), 0.0, 0.0, f32(J.KP_STIFF[i]), f32(J.KD_STIFF[i]), 0)
    return pack_lowcmd(J.MODE_PR, J.MODE_MACHINE, motors, [0, 0, 0, 0], 0)


def lowcmd_random(seed: int) -> bytes:
    r = random.Random(seed)
    motors = [
        (r.randrange(256), f32(r.uniform(-3, 3)), f32(r.uniform(-20, 20)), f32(r.uniform(-60, 60)),
         f32(r.uniform(0, 300)), f32(r.uniform(0, 10)), r.getrandbits(32))
        for _ in range(N_SLOTS)
    ]  # fmt: skip
    return pack_lowcmd(r.randrange(256), r.randrange(256), motors, [r.getrandbits(32) for _ in range(4)],
                       r.getrandbits(32))  # fmt: skip


def lowstate_random(seed: int | None) -> bytes:
    """Field order of __PackHGLowState; seed None = all zeros."""
    r = random.Random(seed)
    z = seed is None

    def u32() -> int:
        return 0 if z else r.getrandbits(32)

    def u8() -> int:
        return 0 if z else r.randrange(256)

    def i16() -> int:
        return 0 if z else r.randrange(-32768, 32768)

    def fl(a: float) -> float:
        return 0.0 if z else f32(r.uniform(-a, a))

    vals: list[float | int] = [u32(), u32(), u8(), u8(), u32()]
    vals += [fl(1.0) for _ in range(4)] + [fl(5.0) for _ in range(3)] + [fl(20.0) for _ in range(3)]
    vals += [fl(3.2) for _ in range(3)] + [i16()]
    for _ in range(N_SLOTS):
        vals += [u8(), fl(3.0), fl(20.0), fl(100.0), fl(60.0), i16(), i16(), fl(60.0)]
        vals += [u32() for _ in range(7)]  # sensor[2], motorstate, reserve[4]
    vals += [u8() for _ in range(40)] + [u32() for _ in range(4)] + [u32()]
    return pack_lowstate(vals)


def generate() -> str:
    if crc_mpeg2(b"123456789") != 0x0376E6E7:
        raise AssertionError("CRC-32/MPEG-2 check value")
    msgs: list[tuple[str, str, bytes]] = [
        ("lowcmd_zeros", "lowcmd", lowcmd_zeros()),
        ("lowcmd_r1_stand", "lowcmd", lowcmd_stand()),
        *[(f"lowcmd_random_{s}", "lowcmd", lowcmd_random(s)) for s in (1, 2, 3)],
        ("lowstate_zeros", "lowstate", lowstate_random(None)),
        *[(f"lowstate_random_{s}", "lowstate", lowstate_random(s)) for s in (1, 2)],
    ]
    lines = []
    for name, kind, packed in msgs:
        fmt = FMT_HG_LOWCMD if kind == "lowcmd" else FMT_HG_LOWSTATE
        if len(packed) != struct.calcsize(fmt):
            raise AssertionError(f"{name}: size")
        msg, crc = with_crc(fmt, packed)
        lines.append(f"{name} {kind} {crc:08x} {msg.hex()}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the committed file instead of writing")
    a = ap.parse_args(argv)
    text = generate()
    if a.check:
        ok = OUT.exists() and OUT.read_text() == text
        print("crc_golden: committed file matches" if ok else "crc_golden: committed file DIFFERS")
        return 0 if ok else 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    (OUT_DIR / "SHA256SUMS").write_text(f"{digest}  crc_golden.txt\n")
    for line in text.splitlines():
        name, kind, crc, hexbytes = line.split()
        print(f"{name:20s} {kind:8s} {len(hexbytes) // 2:5d} B  crc {crc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
