"""The client side of r1d's JSON-lines protocol (the `r1` CLI and scripts use it; DESIGN.md §5.3)."""

from __future__ import annotations

import itertools
import json
import socket
from typing import Any

from .server import DEFAULT_PORT


class R1dError(RuntimeError):
    """r1d could not be reached or answered something that is not a result."""


class Client:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, *, timeout_s: float = 20.0) -> None:
        self.addr = (host, port)
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._rf: Any = None
        self._ids = itertools.count(1)

    def _connect(self) -> None:
        try:
            self._sock = socket.create_connection(self.addr, timeout=self.timeout_s)
        except OSError as e:
            raise R1dError(f"r1d not reachable at {self.addr[0]}:{self.addr[1]} ({e})") from None
        self._rf = self._sock.makefile("rb")

    def call(self, op: str, **kw: Any) -> dict[str, Any]:
        """One request -> its result dict (ok, code, op, id, moved, data, details)."""
        if self._sock is None:
            self._connect()
        assert self._sock is not None
        req = {"op": op, "id": next(self._ids), **kw}
        self._sock.sendall(json.dumps(req).encode() + b"\n")
        line = self._rf.readline()
        if not line:
            self.close()
            raise R1dError("r1d closed the connection")
        out = json.loads(line)
        if not isinstance(out, dict) or "code" not in out:
            raise R1dError(f"not a result: {out!r}")
        return out

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
        self._sock = None
        self._rf = None

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
