"""Unitree's RPC over DDS + the motion switcher (python/r1mc3/dds/rpc.py, api_types.py): our reconstructed unitree_api
IDL regenerates Unitree's C++ headers (incl. the XTypes type blobs DDS peers match); a client and a server agree in
one process and across two processes over the loopback; re-sends are answered once; unknown api ids are refused."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from r1mc3 import dds

pytestmark = pytest.mark.needs_dds
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1mc3.dds import rpc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IDL = ROOT / "rt" / "idl" / "src"
UNITREE = ROOT / "rt" / "idl" / "unitree_api"
TYPES = (
    "Request_",
    "RequestHeader_",
    "RequestIdentity_",
    "RequestLease_",
    "RequestPolicy_",
    "Response_",
    "ResponseHeader_",
    "ResponseStatus_",
)
SHA256 = {  # rt/idl/src/unitree/idl/api/SOURCE.md
    "Request_": "889137406890f4d91cc3061f84aa3e54fececfa52eda14c7f14262677e2b36db",
    "Response_": "f767bb234ce49ab9406abf23feca939737fb9128a67aa6e1ff19a6f0acef8fab",
    "motion_switcher_api": "50d1ddd901bb72ae354cce9cdaae5c1173c06d03d3c59c0baf0d7bd2a9ee3366",
}


def _body(text: str) -> list[str]:
    """The generated header from line 10 on, without the include guard and #include lines (they name files)."""
    return [ln for ln in text.splitlines()[9:] if "DDSCXX_" not in ln and not ln.startswith("#include")]


def test_the_copies_are_unitrees_files() -> None:
    for name, sha in SHA256.items():
        assert hashlib.sha256((UNITREE / f"{name}.hpp").read_bytes()).hexdigest() == sha


def test_our_idl_regenerates_unitrees_headers(tmp_path: Path) -> None:
    prefix = Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / ".cache/r1mc3/cyclonedds-0.10.2"))
    idlc, cxx = prefix / "bin" / "idlc", next(iter((prefix / "lib").glob("libcycloneddsidlcxx.*")), None)
    if not idlc.exists() or cxx is None or shutil.which(str(idlc)) is None:
        pytest.skip("no CycloneDDS 0.10.2 idlc (rt/idl/build_cyclonedds.sh)")
    for n in TYPES:
        subprocess.run(
            [str(idlc), "-x", "final", "-Wno-implicit-extensibility", "-l", str(cxx), "-I", str(IDL),
             str(IDL / "unitree" / "idl" / "api" / f"{n}.idl")],
            cwd=tmp_path, check=True, capture_output=True,
        )  # fmt: skip
        ours = _body((tmp_path / f"{n}.hpp").read_text())
        theirs = _body((UNITREE / f"{n}.hpp").read_text())
        assert ours == theirs, n
        assert any("type_map_blob" in ln for ln in ours), n  # the type identifiers DDS peers match are compared too


def test_motion_switcher_in_one_process() -> None:
    rpc.use_loopback()
    state = {"name": "ai", "released": 0, "selected": ""}

    def release() -> bool:
        state["released"] += 1  # type: ignore[operator]
        state["name"] = ""
        return True

    def select(name: str) -> bool:
        state["selected"] = name
        state["name"] = name
        return True

    with rpc.MotionSwitcherServer(mode=lambda: str(state["name"]), release=release, select=select, domain=41) as srv:
        cli = rpc.MotionSwitcherClient(41)
        assert cli.check_mode() == (0, {"form": "0", "name": "ai"})
        assert cli.release_mode() == 0
        assert cli.check_mode() == (0, {"form": "", "name": ""})
        assert cli.select_mode("ai") == 0 and state["selected"] == "ai"
        assert cli.rpc.call(4242, "")[0] == rpc.ERR_UNKNOWN_API
        assert [e["api_id"] for e in srv.log] == [1001, 1003, 1001, 1002, 4242]
    assert state["released"] == 1


def test_a_resent_request_is_answered_once() -> None:
    rpc.use_loopback()
    calls = []

    def release() -> bool:
        calls.append(time.monotonic())
        time.sleep(0.6)  # slower than the client's re-send interval
        return True

    with rpc.MotionSwitcherServer(mode=lambda: "", release=release, domain=42):
        cli = rpc.MotionSwitcherClient(42)
        assert cli.rpc.call(rpc.API_RELEASE_MODE, "", timeout_s=3.0, resend_s=0.1)[0] == 0
        assert cli.rpc.resends >= 2
    assert len(calls) == 1


def test_timeout_without_a_server() -> None:
    rpc.use_loopback()
    cli = rpc.MotionSwitcherClient(43)
    t0 = time.monotonic()
    assert cli.release_mode(timeout_s=0.3) == rpc.RPC_TIMEOUT
    assert 0.25 < time.monotonic() - t0 < 2.0


def test_across_two_processes_over_the_loopback() -> None:
    """The server in another process: discovery over lo0 (unicast) + the client's re-send until matched."""
    code = textwrap.dedent(
        """
        import sys, time
        from r1mc3.dds import rpc
        rpc.use_loopback()
        st = {"name": "ai"}
        def rel():
            st["name"] = ""
            return True
        srv = rpc.MotionSwitcherServer(mode=lambda: st["name"], release=rel, domain=44).start()
        print("up", flush=True)
        t_end = time.monotonic() + 20
        while time.monotonic() < t_end and len(srv.log) < 3:
            time.sleep(0.01)
        time.sleep(0.5)
        srv.stop()
        """
    )
    env = rpc.use_loopback(dict(os.environ))
    p = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout is not None and p.stdout.readline().strip() == "up"
        rpc.use_loopback()
        cli = rpc.MotionSwitcherClient(44)
        assert cli.check_mode(timeout_s=10.0) == (0, {"form": "0", "name": "ai"})
        assert cli.release_mode(timeout_s=10.0) == 0
        assert cli.check_mode(timeout_s=10.0)[1]["name"] == ""
    finally:
        p.wait(timeout=30)
