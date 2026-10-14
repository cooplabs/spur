"""r1cam's requests and its TCP server (127.0.0.1:8781, newline JSON, r1d's own transport pattern; R1CAM_PLAN.md §2).
r1d is the only client (the `r1` CLI and the compat adapter go through r1d). A dead r1cam fails perception requests
with a code; nothing else depends on it (DESIGN §2.1).

| op | what |
|---|---|
| `status` | the source, each stream's state / frames / age / error, the recordings, the services' last answer |
| `observe` | the newest left JPEG (+ right, + depth on request), base64, with the receive times; starts idle streams |
| `record_start` | {kind, label, streams, depth_every, video, params} -> a recording id + its MCAP path |
| `record_stop` | {rec_id} -> the manifest (counts, drops, files with sha256) |
| `services` | {action: "status"} read-only; {action: "stereo" / "video_hub", requester: "operator", confirm: true} |

Results have r1d's shape: ok, code (r1mc3.service.codes.Code), op, id, data, details.
"""

from __future__ import annotations

import base64
import itertools
import json
import socketserver
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from r1mc3.service.codes import Code

from .frames import STREAMS
from .recorder import CamRecorder
from .services import CameraServices, ServiceRefused
from .source import SourceBase
from .streams import DEPTH_KEEP_S, RIGHT_REFUSED, record_refusal, right_allowed
from .streams import policy as stream_policy

DEFAULT_PORT = 8781
MAX_LINE = 1 << 20
OBSERVE_WAIT_S = 4.0  # a cold image stream needs a key frame (v2: STEREO_COLD_WAIT_S 8 s for the right one)


def _res(op: str, rid: Any, code: Code = Code.OK, data: dict[str, Any] | None = None, **details: Any) -> dict[str, Any]:
    return {"ok": code == Code.OK, "code": str(code), "op": op, "id": rid, "data": data or {}, "details": details}


class CamService:
    def __init__(
        self,
        source: SourceBase,
        records_dir: str | Path,
        *,
        services: CameraServices | None = None,
        max_recordings: int = 2,
        depth_keep_s: float = DEPTH_KEEP_S,
    ) -> None:
        self.source = source
        self.depth_keep_s = float(depth_keep_s)
        self.records_dir = Path(records_dir)
        self.services = services
        self.max_recordings = max_recordings
        self.recordings: dict[int, CamRecorder] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.last_services: dict[str, Any] | None = None

    def handle(self, req: Mapping[str, Any]) -> dict[str, Any]:
        op = str(req.get("op", ""))
        rid = req.get("id")
        try:
            if op == "status":
                return _res(op, rid, data=self.status())
            if op == "observe":
                return self._observe(req)
            if op == "record_start":
                return self._record_start(req)
            if op == "record_stop":
                return self._record_stop(req)
            if op == "services":
                return self._services(req)
        except (ValueError, TypeError, KeyError) as e:
            return _res(op, rid, Code.INVALID_ARGS, reason=str(e))
        return _res(op, rid, Code.UNKNOWN_COMMAND, reason=f"unknown op {op!r}")

    def status(self) -> dict[str, Any]:
        with self._lock:
            recs = {i: {"kind": r.kind, "label": r.label, "frames": dict(r.counts), "dropped": r.queue.dropped,
                        **({"coverage": r.coverage.status()} if r.coverage is not None else {})}
                    for i, r in self.recordings.items()}  # fmt: skip
        return {
            **self.source.stats(),
            "recordings": recs,
            "services": self.last_services,
            "policy": stream_policy(self.depth_keep_s),
        }

    def _calibrating(self) -> bool:
        """A calibration recording runs (it holds the right image: an observe may then return it too)."""
        with self._lock:
            return any(right_allowed(r.kind) and "right" in r.streams for r in self.recordings.values())

    def _observe(self, req: Mapping[str, Any]) -> dict[str, Any]:
        """The newest frames. Session 12 (streams.py, HW2 C1): the right image only while a calibration recording runs
        (else the reply says why it is missing: `right_refused`); the depth stream is HELD until this observe's frame
        arrived, then closes DEPTH_KEEP_S later (before: every stream kept IDLE_S = 3 s after each request)."""
        op, rid = "observe", req.get("id")
        notes: dict[str, Any] = {}
        right = bool(req.get("right"))
        if right and not self._calibrating():
            right, notes["right_refused"] = False, RIGHT_REFUSED
        want = ["left"] + (["right"] if right else []) + (["depth"] if req.get("depth") else [])
        want = [s for s in want if s in self.source.streams]
        depth = "depth" in want
        self.source.demand([s for s in want if s != "depth"])
        if depth:
            self.source.hold(["depth"], True)
            self.source.demand(["depth"], keep_s=0.0)  # an on-demand (lockstep) source renders it now
        max_age = float(req.get("max_age_s", 1.0))
        wait = float(req.get("wait_s", OBSERVE_WAIT_S))
        out: dict[str, Any] = {}
        try:
            for s in want:
                f = self.source.slots[s].get(max_age_s=max_age, timeout_s=wait)
                if f is None:
                    return _res(op, rid, Code.TIMEOUT, stream=s, error=self.source.errors.get(s),
                                reason=f"no {s} frame younger than {max_age} s within {wait} s")  # fmt: skip
                out[s] = {"seq": f.seq, "t_rx_ns": f.t_rx_ns, "bytes": len(f.data), "width": f.width,
                          "height": f.height, "src_counter": f.src_counter, "src_stamp": f.src_stamp,
                          "stream_id": self.source.stream_id(s),
                          "b64": base64.b64encode(f.data).decode("ascii")}  # fmt: skip
        finally:
            if depth:
                self.source.hold(["depth"], False)
                self.source.keep(["depth"], keep_s=self.depth_keep_s)
        return _res(op, rid, data={**out, **notes})

    def _record_start(self, req: Mapping[str, Any]) -> dict[str, Any]:
        op, rid = "record_start", req.get("id")
        streams = tuple(req.get("streams", ("left",)))  # session 12: the right image only for calibration kinds
        if any(s not in STREAMS for s in streams):
            raise ValueError(f"streams: from {STREAMS}")
        refused = record_refusal(str(req.get("kind", "manual")), streams)
        if refused is not None:
            return _res(op, rid, Code.NOT_PERMITTED, reason=refused)
        with self._lock:
            if len(self.recordings) >= self.max_recordings:
                return _res(op, rid, Code.RESOURCE_BUSY, recordings=list(self.recordings))
            rec = CamRecorder(
                self.source,
                self.records_dir,
                kind=str(req.get("kind", "manual")),
                label=str(req.get("label", "")),
                params=dict(req.get("params", {})),
                streams=streams,
                depth_every=int(req.get("depth_every", 1)),
                video=bool(req.get("video", False)),
            ).start()
            i = next(self._ids)
            self.recordings[i] = rec
        return _res(op, rid, data={"rec_id": i, "mcap": str(rec.mcap_path), "manifest": str(rec.manifest_path)})

    def _record_stop(self, req: Mapping[str, Any]) -> dict[str, Any]:
        op, rid = "record_stop", req.get("id")
        with self._lock:
            rec = self.recordings.pop(int(req["rec_id"]), None)
        if rec is None:
            return _res(op, rid, Code.INVALID_ARGS, reason="no such recording")
        return _res(op, rid, data=rec.stop(str(req.get("status", "done")), str(req.get("reason", ""))))

    def _services(self, req: Mapping[str, Any]) -> dict[str, Any]:
        op, rid = "services", req.get("id")
        if self.services is None:
            return _res(op, rid, Code.UNSUPPORTED, reason="no robot camera services here (sim / bench)")
        action = str(req.get("action", "status"))
        if action == "status":
            self.last_services = {"status": self.services.status(), "t": time.time()}
            return _res(op, rid, data=self.last_services)
        try:
            r = self.services.switch(action, requester=str(req.get("requester", "r1d")),
                                     confirm=req.get("confirm") is True)  # fmt: skip
        except ServiceRefused as e:
            return _res(op, rid, Code.NOT_PERMITTED, reason=str(e))
        self.last_services = {**r, "t": time.time()}
        return _res(op, rid, Code.OK if r["ok"] else Code.FAULT, data=r)

    def close(self) -> None:
        with self._lock:
            recs = list(self.recordings.values())
            self.recordings.clear()
        for r in recs:
            r.stop("aborted", "r1cam closed")
        self.source.close()


class _Handler(socketserver.StreamRequestHandler):
    server: _Server

    def handle(self) -> None:
        while True:
            line = self.rfile.readline(MAX_LINE)
            if not line:
                return
            if not line.strip():
                continue
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("a request is a JSON object")
                out = self.server.svc.handle(req)
            except (ValueError, json.JSONDecodeError) as e:
                out = _res("?", None, Code.INVALID_ARGS, reason=str(e))
            try:
                self.wfile.write(json.dumps(out).encode() + b"\n")
            except (BrokenPipeError, ConnectionResetError):
                return


class _Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], svc: CamService) -> None:
        self.svc = svc
        super().__init__(addr, _Handler)


class CamServer:
    """The service on 127.0.0.1 only (context manager)."""

    def __init__(self, svc: CamService, *, port: int = DEFAULT_PORT) -> None:
        self.svc = svc
        self.tcp = _Server(("127.0.0.1", port), svc)
        self.port = int(self.tcp.server_address[1])
        self._t = threading.Thread(target=self.tcp.serve_forever, name="r1cam-tcp", daemon=True)

    def start(self) -> CamServer:
        self._t.start()
        return self

    def close(self) -> None:
        self.tcp.shutdown()
        self.tcp.server_close()
        self.svc.close()

    def __enter__(self) -> CamServer:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()


def request(port: int, req: Mapping[str, Any], timeout_s: float = 15.0) -> dict[str, Any]:
    """One request to a running r1cam (r1d's client side; the CLI hook)."""
    import socket

    with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as s:
        s.sendall(json.dumps(dict(req)).encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
    out: dict[str, Any] = json.loads(buf)
    return out
