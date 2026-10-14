"""The takeover's ReleaseMode step (python/r1mc3/service/release.py) against the motion switcher's service side over
DDS (r1mc3.dds.rpc.MotionSwitcherServer, loopback, domain 62): a built-in that lets go -> released; one that does not
-> not released within the timeout (the takeover then aborts); a CheckMode that fails is never read as "released"."""

from __future__ import annotations

from typing import Any

import pytest
from r1mc3 import dds

pytestmark = pytest.mark.needs_dds
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1mc3.dds import rpc  # noqa: E402
from r1mc3.service.release import MotionSwitcherReleaser  # noqa: E402

DOMAIN = 62
rpc.use_loopback()


class Builtin:
    def __init__(self, lets_go: bool) -> None:
        self.name = "ai_sport"
        self.lets_go = lets_go
        self.releases = 0

    def mode(self) -> str:
        return self.name

    def release(self) -> bool:
        self.releases += 1
        if self.lets_go:
            self.name = ""
        return self.lets_go


@pytest.mark.parametrize("lets_go", [True, False], ids=["released", "refuses"])
def test_release_against_the_switcher(lets_go: bool) -> None:
    b = Builtin(lets_go)
    srv = rpc.MotionSwitcherServer(mode=b.mode, release=b.release, domain=DOMAIN).start()
    try:
        ok, info = MotionSwitcherReleaser(DOMAIN).release(timeout_s=1.0)
    finally:
        srv.stop()
    assert ok is lets_go, info
    assert info["before"] == "ai_sport"
    if lets_go:
        assert info["calls"] == 1 and b.releases == 1
    else:
        assert info["after"] == "ai_sport" and info["calls"] >= 2  # retried every 50 ms until the timeout


class _FailingCheck:
    def check_mode(self, **_: Any) -> tuple[int, dict[str, Any]]:
        return 3104, {}  # an RPC timeout: no answer, an empty dict

    def release_mode(self, **_: Any) -> int:
        return 0


def test_a_failed_check_is_not_released() -> None:
    ok, info = MotionSwitcherReleaser(0, client=_FailingCheck()).release(timeout_s=0.15)
    assert ok is False and "unknown" in str(info["after"])
