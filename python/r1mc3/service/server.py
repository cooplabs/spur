"""r1d's transport: newline-delimited JSON over TCP on localhost (DESIGN.md §5.3; the Mac reaches it through the SSH
tunnel), plus the tick thread that advances the service at its rate.

One request per line, one result per line, in order per connection. The server binds 127.0.0.1 only (never a
public interface: the robot's API is reached through the tunnel). A malformed line gets an INVALID_ARGS result; a
client that disconnects mid-request loses only its own answer. Nothing here can block the RT core (the segment is
the only path to it), and a stalled r1d only lets the TTLs / deadmen stop arm motion.
"""

from __future__ import annotations

import json
import socketserver
import threading
import time
from typing import Any

from .codes import Code
from .schemas import fail
from .service import Service

DEFAULT_PORT = 8780


def _largest_request_field() -> int:
    """The longest string any request may carry: measure.phone's base64 still (schemas.MeasurePhoneReq.image_b64)."""
    from annotated_types import MaxLen

    from .schemas import MeasurePhoneReq

    meta = MeasurePhoneReq.model_fields["image_b64"].metadata
    return max(int(m.max_length) for m in meta if isinstance(m, MaxLen))


# session 13 (HW2 day 2's prep): the line limit was 1 MiB while measure.phone's schema accepts a 40 MB base64 still - a
# phone still of ~1.5 MB (2016 x 1512, JPEG q95: ~2 MB base64) was refused, and the rest of the over-long line was read
# as further "requests" (one INVALID_ARGS per MiB: every later reply on that connection answered the wrong request).
# Now the limit follows the schema's largest field (+ 1 MiB for the rest of the request) and an over-long line is read
# to its end and answered ONCE. Memory: one such line per connection at a time (~41 MB); r1d binds 127.0.0.1 only.
MAX_LINE = _largest_request_field() + (1 << 20)
_DRAIN_CHUNK = 1 << 20


def read_request_line(rfile: Any, max_line: int) -> tuple[bytes, int]:
    """One request line -> (the line, 0) or, when it is longer than `max_line`, (b"", its length in bytes: the rest was
    read and dropped up to its newline, so the next line is the next request). (b"", 0) = the client closed."""
    line: bytes = rfile.readline(max_line + 1)
    if len(line) <= max_line or line.endswith(b"\n"):
        return line, 0
    n = len(line)
    while True:
        chunk: bytes = rfile.readline(_DRAIN_CHUNK)
        n += len(chunk)
        if not chunk or chunk.endswith(b"\n"):
            return b"", n


def too_long_words(n_bytes: int, max_line: int) -> str:
    mb, lim = n_bytes / 1e6, max_line / 1e6
    return (
        f"the request line is {mb:.1f} MB, longer than r1d's limit of {lim:.0f} MB (the largest request field the "
        "schema accepts + 1 MiB); a phone still: downscale it to its lens model's size first (python "
        "tools/phone/prep_phone.py OUTDIR PHOTO...)"
    )


class _Handler(socketserver.StreamRequestHandler):
    server: _Server

    def handle(self) -> None:
        while True:
            line, too_long = read_request_line(self.rfile, MAX_LINE)
            if too_long:
                out = fail("?", None, Code.INVALID_ARGS, reason=too_long_words(too_long, MAX_LINE)).model_dump(
                    mode="json"
                )
                try:
                    self.wfile.write(json.dumps(out).encode() + b"\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
                continue
            if not line:
                return
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("a request is a JSON object")
                out = self.server.service.handle(req, timeout_s=self.server.timeout_s)
            except (ValueError, json.JSONDecodeError) as e:
                out = fail("?", None, Code.INVALID_ARGS, reason=str(e)).model_dump(mode="json")
            try:
                self.wfile.write(json.dumps(out).encode() + b"\n")
            except (BrokenPipeError, ConnectionResetError):
                return


class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], service: Service, timeout_s: float) -> None:
        self.service = service
        self.timeout_s = timeout_s
        super().__init__(addr, _Handler)


class R1dServer:
    """The service + its TCP server + its tick thread (context manager)."""

    def __init__(self, service: Service, *, port: int = DEFAULT_PORT, timeout_s: float = 15.0) -> None:
        self.service = service
        self.tcp = _Server(("127.0.0.1", port), service, timeout_s)
        self.port = int(self.tcp.server_address[1])
        self._stop = threading.Event()
        self._tick = threading.Thread(target=self._tick_loop, name="r1d-tick", daemon=True)
        self._serve = threading.Thread(target=self.tcp.serve_forever, name="r1d-tcp", daemon=True)
        self.tick_overruns = 0

    def _tick_loop(self) -> None:
        period = 1.0 / self.service.tick_hz
        nxt = time.monotonic()
        while not self._stop.is_set():
            self.service.tick()
            nxt += period
            delay = nxt - time.monotonic()
            if delay < 0:
                self.tick_overruns += 1
                nxt = time.monotonic()
            else:
                self._stop.wait(delay)

    def start(self) -> R1dServer:
        self._tick.start()
        self._serve.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self.tcp.shutdown()
        self.tcp.server_close()
        self._tick.join(timeout=2.0)
        self.service.close()

    def __enter__(self) -> R1dServer:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()
