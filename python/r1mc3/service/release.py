"""The motion switcher's ReleaseMode for r1d's takeover (DESIGN.md §3.5: an RPC in r1d, never in an RT thread).

`MotionSwitcherReleaser(domain)` wraps the DDS client of `r1mc3.dds.rpc` (the bridge agent's, wire-compatible with
Unitree's `rt/api/motion_switcher/request|response`): `release(timeout_s) -> (released, info)`, v2's loop (the
official R1 example's logic, v2 r1mc2/backends/robot.py:469-485): ReleaseMode, then CheckMode; repeat every 50 ms
until CheckMode reports no active mode, at most `timeout_s` (4 s). A controller still active at the end is NOT
released: the takeover aborts (nothing of ours published) - v2 damped there, v3 does not (DESIGN §3.5).
"""

from __future__ import annotations

import time
from typing import Any


class MotionSwitcherReleaser:
    def __init__(self, domain: int, *, client: Any | None = None) -> None:
        if client is None:
            from r1mc3.dds.rpc import MotionSwitcherClient  # the bridge agent's module (session 5)

            client = MotionSwitcherClient(domain)
        self.client = client

    @staticmethod
    def _active(check: Any) -> str | None:
        """The active mode's name from a CheckMode answer (`MotionSwitcherClient.check_mode()` -> (code, {"form",
        "name"}); also a dict or a bare name); None = nothing active (released). A failed CheckMode (code != 0, e.g. a
        timeout, whose dict is empty) is NOT "released": it reads as an unknown active mode."""
        if isinstance(check, tuple) and len(check) == 2:
            code, check = check
            if int(code) != 0:
                return f"unknown (CheckMode code {int(code)})"
        if isinstance(check, dict):
            name = check.get("name", check.get("mode", ""))
            return str(name) if name else None
        if isinstance(check, str):
            return check or None
        return None

    def release(self, timeout_s: float) -> tuple[bool, dict[str, Any]]:
        t0 = time.monotonic()
        info: dict[str, Any] = {}
        calls = 0

        def left() -> float:  # each RPC waits at most what is left of the timeout (>= 0.2 s)
            return max(0.2, min(1.0, timeout_s - (time.monotonic() - t0)))

        try:
            info["before"] = self._active(self.client.check_mode(timeout_s=left()))
            while True:
                self.client.release_mode(timeout_s=left())
                calls += 1
                active = self._active(self.client.check_mode(timeout_s=left()))
                if active is None:
                    info.update(calls=calls, seconds=round(time.monotonic() - t0, 3))
                    return True, info
                if time.monotonic() - t0 > timeout_s:
                    info.update(after=active, calls=calls, seconds=round(time.monotonic() - t0, 3))
                    return False, info
                time.sleep(0.05)
        except Exception as e:  # the RPC failed: treat as not released (the takeover aborts)
            info.update(error=repr(e), calls=calls)
            return False, info
