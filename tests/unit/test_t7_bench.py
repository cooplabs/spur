"""The policy-timing check for the Orin (session 8; docs/hardware/T7_BENCH.md).

* the committed input (tests/assets/t7/t7_policy_2000.r1pb, tools/t7/make_input.py) is intact, belongs to the live
  policy (its params block = the policy's r1rt_policy_params) and its expected targets are what the Python reference
  computes from its inputs;
* the C++ timing program (rt/apps/r1rt_policy_bench.cpp from the C++ build: $R1MC3_BUILD_DIR/rt, else build/rt; skipped
  when absent) runs in both modes, writes a well-formed result, and its targets equal the golden policy's (the
  file's) - and a corrupted expected target IS caught (the check is real), a broken file is refused.
Timing numbers are not asserted here (macOS under load is not the Orin): the pass rule is evaluated by the tool.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from r1mc3.policy import POLICY_DIR, ReferencePolicy
from r1sim.v3_core import pack_params, policy_params

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "tests" / "assets" / "t7" / "t7_policy_2000.r1pb"
TOL = 2e-6  # float32 storage of the expected targets


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("r1mc3_tools_t7_make_input", ROOT / "tools" / "t7" / "make_input.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclass looks its module up there
    spec.loader.exec_module(mod)
    return mod


def _bench() -> tuple[Path, Path]:
    """(r1rt_policy_bench, converted weights) from the C++ build, or skip."""
    dirs = []
    if os.environ.get("R1MC3_BUILD_DIR"):
        dirs.append(Path(os.environ["R1MC3_BUILD_DIR"]) / "rt")
    dirs.append(ROOT / "build" / "rt")
    for d in dirs:
        exe, w = d / "apps" / "r1rt_policy_bench", d / "policy" / "policy_weights.r1w"
        if exe.exists() and w.exists():
            return exe, w
    pytest.skip("no C++ build with r1rt_policy_bench + converted weights (scripts/ci.sh cpp builds it)")


def _run(args: list[str]) -> tuple[int, dict[str, Any]]:
    p = subprocess.run(args, capture_output=True, text=True, timeout=120, check=False)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    assert lines, (p.returncode, p.stdout, p.stderr)
    return p.returncode, json.loads(lines[-1])


def test_the_input_file_is_intact_and_belongs_to_the_live_policy() -> None:
    want = (ASSET.parent / "SHA256SUMS").read_text().split()[0]
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == want
    f = _tool().read_r1pb(ASSET)
    pol = ReferencePolicy.load(POLICY_DIR)
    assert f.n >= 2000
    assert (f.n_joints, f.n_act) == (pol.n, pol.n_act) == (24, 14)
    # the params block = what r1rt would load for this policy (regenerate the file when the policy changes)
    np.testing.assert_array_equal(f.params, pack_params(policy_params(pol)))
    # the recorded stance: 50 Hz steps, the 3 deg lean request, a walking segment
    dt = np.diff(f.t)
    assert abs(float(np.median(dt)) - 0.02) < 1e-6 and float(dt.min()) > 0.0
    reqs = f.lean_req[~np.isnan(f.lean_req)]
    assert np.any(np.abs(reqs - np.radians(3.0)) < 1e-6)
    assert float(np.abs(f.cmd[:, 0]).max()) > 0.1
    assert np.allclose(np.linalg.norm(f.quat, axis=1), 1.0, atol=1e-6)


def test_the_expected_targets_are_the_python_references() -> None:
    tool = _tool()
    f = tool.read_r1pb(ASSET)
    got = tool.replay(ReferencePolicy.load(POLICY_DIR), f)
    err = float(np.abs(got - f.expected).max())
    assert err <= TOL, err
    assert float(np.abs(got).max()) > 0.1  # real targets, not zeros


def test_the_bench_runs_in_both_modes_and_matches_the_golden_policy(tmp_path: Path) -> None:
    exe, w = _bench()
    out = tmp_path / "burst.json"
    rc, r = _run([str(exe), "--weights", str(w), "--input", str(ASSET), "--mode", "burst", "--iters", "300",
                  "--warmup", "5", "--out", str(out)])  # fmt: skip
    assert rc == 0, r
    assert json.loads(out.read_text()) == r
    assert r["t7"] == 1 and r["mode"] == "burst" and r["calls"] == 300 and r["input_steps"] == 2000
    assert r["check"]["ok"] is True and r["check"]["invalid"] == 0 and r["check"]["invalid_timed"] == 0
    assert r["check"]["max_abs_err_rad"] <= TOL
    c = r["compute_us"]
    assert c["n"] == 300 and 0.0 < c["p50"] <= c["p99"] <= c["p999"] <= c["max"]
    assert r["lateness_us"] is None and r["total_us"] == c
    assert r["budget_us"] == {"thread": 3000, "inline": 1000, "period": 20000, "inline_rung_after_ms": 40}
    assert r["pass"] == (r["check"]["ok"] and r["pass_thread"] and r["pass_inline"])
    assert r["pass_inline"] == (c["max"] <= 1000.0)
    for k in ("os", "build", "sanitized", "grant", "mlock", "conditions_ok", "skipped_periods"):
        assert k in r, k
    if r["os"] != "linux":
        assert r["conditions_ok"] is False  # no SCHED_FIFO / affinity: a reference, never the T7 answer

    rc, p = _run([str(exe), "--weights", str(w), "--input", str(ASSET), "--mode", "periodic", "--rate-hz", "100",
                  "--duration-s", "0.3", "--warmup", "2"])  # fmt: skip
    assert rc == 0, p
    assert p["mode"] == "periodic" and p["calls"] == 30
    lat, tot = p["lateness_us"], p["total_us"]
    assert lat["n"] == 30 and tot["n"] == 30 and tot["max"] >= p["compute_us"]["max"]
    assert p["budget_us"]["period"] == 10000


def test_the_bench_catches_a_wrong_target_and_refuses_a_broken_file(tmp_path: Path) -> None:
    exe, w = _bench()
    b = bytearray(ASSET.read_bytes())
    tool = _tool()
    f = tool.read_r1pb(ASSET)
    # the last step's first expected target + 1e-3 rad: the check must see it
    off = len(b) - 4 * f.n_act
    v = np.frombuffer(bytes(b[off : off + 4]), "<f4")[0]
    b[off : off + 4] = np.array([v + 1e-3], "<f4").tobytes()
    bad = tmp_path / "wrong.r1pb"
    bad.write_bytes(bytes(b))
    rc, r = _run([str(exe), "--weights", str(w), "--input", str(bad), "--mode", "burst", "--iters", "10"])
    assert rc == 1 and r["check"]["ok"] is False and r["pass"] is False
    assert abs(r["check"]["max_abs_err_rad"] - 1e-3) < 1e-5

    short = tmp_path / "short.r1pb"
    short.write_bytes(ASSET.read_bytes()[:-8])
    rc, r = _run([str(exe), "--weights", str(w), "--input", str(short)])
    assert rc == 2 and "size" in r["error"]
