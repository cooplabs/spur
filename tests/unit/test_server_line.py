"""r1d's request line limit (session 13, HW2 day 2's prep): measure.phone's schema accepts a 40 MB base64 still but the
transport read at most 1 MiB per line - a 2016 x 1512 phone still at JPEG q95 (~2 MB base64) was refused and the rest of
the over-long line was answered as further requests (every later reply on that connection belonged to the wrong
request). Now the limit follows the schema's largest field, and an over-long line gets ONE answer."""

from __future__ import annotations

import io
import json
import socket
from typing import Any

import pytest
from r1mc3.service import server as srv_mod
from r1mc3.service.schemas import MeasurePhoneReq
from r1mc3.service.server import R1dServer, read_request_line
from r1mc3.service.service import Service

from tests.unit.test_service_server import FakePort, _NoRelease


@pytest.fixture
def server() -> Any:
    svc = Service(FakePort(), releaser=_NoRelease())
    with R1dServer(svc, port=0) as srv:
        yield srv


def _exchange(port: int, lines: list[bytes]) -> list[dict[str, Any]]:
    """Send each line, read one reply per line (in order), on ONE connection."""
    with socket.create_connection(("127.0.0.1", port), timeout=20) as s:
        f = s.makefile("rb")
        out = []
        for ln in lines:
            s.sendall(ln + b"\n")
            out.append(json.loads(f.readline()))
        return out


def _padded_status(n_bytes: int, rid: int) -> bytes:
    """A `status` request with an unknown padding key, about n_bytes long (the schema refuses the extra key: ONE
    INVALID_ARGS reply - the point is that the whole line is ONE request)."""
    head = json.dumps({"op": "status", "id": rid, "pad": ""})[:-2]
    return (head + "x" * max(0, n_bytes - len(head) - 2) + '"}').encode()


def test_the_limit_follows_measure_phones_largest_still() -> None:
    lim = next(m.max_length for m in MeasurePhoneReq.model_fields["image_b64"].metadata if hasattr(m, "max_length"))
    assert srv_mod.MAX_LINE == lim + (1 << 20)
    assert srv_mod.MAX_LINE > 40_000_000  # base 1 MiB: the 40 MB the schema accepts could never arrive


def test_a_2mb_line_is_one_request_and_the_connection_stays_in_step(server: Any) -> None:
    """Base (1 MiB): the 2 MB line came back as 2-3 INVALID_ARGS replies (the truncated JSON, then its remainder as
    'requests'), so the `status` after it read a stale INVALID_ARGS instead of its own answer."""
    big, nxt = _exchange(server.port, [_padded_status(2_000_000, 1), b'{"op": "status", "id": 2}'])
    assert big["code"] == "INVALID_ARGS"  # the unknown key (a whole request was parsed), not a broken line
    assert nxt["ok"] and nxt["id"] == 2


def test_an_over_long_line_gets_one_answer_in_words(server: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srv_mod, "MAX_LINE", 4096)
    big, nxt = _exchange(server.port, [_padded_status(50_000, 1), b'{"op": "status", "id": 2}'])
    assert big["code"] == "INVALID_ARGS"
    why = big["details"]["reason"]
    assert "0.1 MB, longer than r1d's limit" in why and "prep_phone" in why
    assert nxt["ok"] and nxt["id"] == 2


def test_read_request_line_drains_to_the_newline() -> None:
    f = io.BytesIO(b"a" * 10 + b"\n" + b"b" * 5000 + b"\n" + b"c\n")
    assert read_request_line(f, 64) == (b"a" * 10 + b"\n", 0)
    assert read_request_line(f, 64) == (b"", 5001)
    assert read_request_line(f, 64) == (b"c\n", 0)
    assert read_request_line(f, 64) == (b"", 0)  # closed
    # the limit counts the request, not its newline: 64 bytes + newline is one request, 65 is too long
    assert read_request_line(io.BytesIO(b"d" * 64 + b"\n"), 64) == (b"d" * 64 + b"\n", 0)
    assert read_request_line(io.BytesIO(b"d" * 65 + b"\n"), 64) == (b"", 66)
