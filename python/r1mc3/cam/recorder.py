"""r1cam recordings: one MCAP file per recording (r1log's container, zstd chunks) + a manifest JSON beside it, in the
records folder like every `r1 record` (so `r1 shutdown`'s save carries both; HW1 fixes #13 / #20).

Topics: `/r1cam/left`, `/r1cam/right` (the JPEG as decoded on the robot, encoding "jpeg"), `/r1cam/depth` (the raw
544 x 448 uint16 LE mm frame, encoding "r1cam.depth_u16le_544x448"; sequence = the header's counter),
`/r1cam/depth_meta` (per depth frame: the header's counter + timestamp), `/r1cam/meta` (start, 1 Hz stats, end).
log_time = our CLOCK_MONOTONIC receive time (r1log's clock: frames line up with the head joints). `depth_every` keeps
every n-th depth frame (the ChArUco sweep at ~2 FPS: a raw depth frame is 487 KB, 4.9 MB/s at the full 10 FPS).

The recorder taps the source through a `DropQueue` (bounded): a slow disk drops the OLDEST frames (counted in the
manifest), it never blocks the camera threads, r1d or anything else. The full head video of a trial (Q3) is the
robot's H.264 remuxed by GStreamer (robot.record_pipeline: no decode, no encode) - `video=True` asks the source for it.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from mcap.writer import CompressionType, Writer

from .frames import STREAMS, DropQueue, Frame, monotonic_ns
from .source import SourceBase

ENCODING = {"left": "jpeg", "right": "jpeg", "depth": "r1cam.depth_u16le_544x448"}
SCHEMA = "r1mc3.r1cam_manifest/1"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CamRecorder:
    def __init__(
        self,
        source: SourceBase,
        out_dir: str | Path,
        *,
        kind: str,
        label: str = "",
        params: dict[str, Any] | None = None,
        streams: tuple[str, ...] = ("left", "right"),
        depth_every: int = 1,
        video: bool = False,
        queue_len: int = 256,
    ) -> None:
        bad = [s for s in streams if s not in STREAMS or s not in source.streams]
        if bad or not streams:
            raise ValueError(f"streams: from {source.streams}, got {streams}")
        self.source = source
        self.out_dir = Path(out_dir)
        self.kind, self.label = kind, label
        self.params = dict(params or {})
        self.streams = tuple(streams)
        self.depth_every = max(1, int(depth_every))
        self.video = bool(video)
        self.queue = DropQueue(queue_len)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.stem = f"{stamp}_cam_{kind}" + (f"_{label}" if label else "")
        self.mcap_path = self.out_dir / f"{self.stem}.r1cam.mcap"
        self.manifest_path = self.out_dir / f"{self.stem}.r1cam.json"
        self.video_paths: dict[str, Path] = {}
        self.counts = dict.fromkeys(self.streams, 0)
        self.skipped_depth = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="r1cam-rec")
        self._f: IO[bytes] | None = None
        self._w: Writer | None = None
        self._ch: dict[str, int] = {}
        self.t0_ns = 0
        self.t1_ns = 0
        self.wall0 = 0.0
        self.write_errors = 0
        self._depth_seen = 0
        # session 10: the live S12 count for a charuco recording (coverage.py; off without OpenCV, never blocking)
        self.coverage: Any = None
        if kind == "charuco":
            from .coverage import BOARDS, CharucoCoverage

            board = str(self.params.get("board", "poster_6x4_90"))
            dists = tuple(float(z) for z in self.params.get("distances_m", (0.3, 0.45, 0.6)))
            self.coverage = CharucoCoverage(board if board in BOARDS else "poster_6x4_90", distances_m=dists)

    # ---- the source side (camera threads): never blocks
    def _tap(self, f: Frame) -> None:
        if f.stream not in self.streams:
            return
        if f.stream == "depth":
            self._depth_seen += 1
            if (self._depth_seen - 1) % self.depth_every:
                self.skipped_depth += 1
                return
        self.queue.put(f)

    def start(self) -> CamRecorder:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._f = self.mcap_path.open("wb")
        w = Writer(self._f, compression=CompressionType.ZSTD)
        w.start(profile="", library="r1mc3.cam")
        for s in self.streams:
            self._ch[s] = w.register_channel(f"/r1cam/{s}", ENCODING[s], 0)
        self._ch["depth_meta"] = w.register_channel("/r1cam/depth_meta", "json", 0)
        self._ch["meta"] = w.register_channel("/r1cam/meta", "json", 0)
        self._w = w
        self.t0_ns, self.wall0 = monotonic_ns(), time.time()
        self._meta({"event": "start", "kind": self.kind, "label": self.label, "params": self.params,
                    "streams": list(self.streams), "depth_every": self.depth_every, "source": self.source.name,
                    "wall_utc": datetime.fromtimestamp(self.wall0, UTC).isoformat()})  # fmt: skip
        if self.video and hasattr(self.source, "record_video"):
            for s in self.streams:
                if s in ("left", "right"):
                    p = self.out_dir / f"{self.stem}.{s}.h264.mkv"
                    self.source.record_video(s, str(p))
                    self.video_paths[s] = p
        self.source.hold(self.streams, True)
        self.source.add_listener(self._tap)
        self._thread.start()
        return self

    def _meta(self, obj: dict[str, Any]) -> None:
        assert self._w is not None
        t = monotonic_ns()
        self._w.add_message(self._ch["meta"], log_time=t, data=json.dumps(obj).encode(), publish_time=t)

    def _run(self) -> None:
        last_stats = time.monotonic()
        while not (self._stop.is_set() and len(self.queue) == 0):
            f = self.queue.get(0.1)
            if f is not None:
                self._write(f)
            if time.monotonic() - last_stats >= 1.0:
                last_stats = time.monotonic()
                self._meta({"event": "stats", "counts": dict(self.counts), "dropped": self.queue.dropped})

    def _write(self, f: Frame) -> None:
        assert self._w is not None
        try:
            seq = f.src_counter if (f.stream == "depth" and f.src_counter is not None) else f.seq
            self._w.add_message(self._ch[f.stream], log_time=f.t_rx_ns, data=f.data, publish_time=f.t_rx_ns,
                                sequence=int(seq) & 0xFFFFFFFF)  # fmt: skip
            if f.stream == "depth":
                m = {"seq": f.seq, "counter": f.src_counter, "stamp": f.src_stamp}
                self._w.add_message(self._ch["depth_meta"], log_time=f.t_rx_ns, data=json.dumps(m).encode(),
                                    publish_time=f.t_rx_ns)  # fmt: skip
            self.counts[f.stream] += 1
        except Exception:  # a write failure is counted in the manifest, the camera goes on
            self.write_errors += 1
        # the live count only while the queue is short: the recording always wins over the count (a skipped frame is
        # still recorded; the count just sees fewer frames)
        if self.coverage is not None and f.stream == "left" and len(self.queue) <= 4:
            try:
                self.coverage.feed(f.data)
            except Exception:  # an undecodable frame is not counted; the recording goes on
                pass

    def stop(self, status: str = "done", reason: str = "") -> dict[str, Any]:
        self.source.remove_listener(self._tap)
        self.source.hold(self.streams, False)
        for s in self.video_paths:
            self.source.record_video(s, None)  # type: ignore[attr-defined]
        self._stop.set()
        self._thread.join(timeout=10.0)
        self.t1_ns = monotonic_ns()
        self._meta({"event": "end", "counts": dict(self.counts), "dropped": self.queue.dropped, "status": status})
        assert self._w is not None and self._f is not None
        self._w.finish()  # type: ignore[no-untyped-call]
        self._f.close()
        files = [self.mcap_path, *self.video_paths.values()]
        doc = {
            "schema": SCHEMA,
            "kind": self.kind,
            "label": self.label,
            "params": self.params,
            "status": status,
            "reason": reason,
            "source": self.source.name,
            "streams": list(self.streams),
            "depth_every": self.depth_every,
            "frames": dict(self.counts),
            "dropped_queue": self.queue.dropped,
            "skipped_depth": self.skipped_depth,
            "write_errors": self.write_errors,
            "t_start_mono_ns": self.t0_ns,
            "t_end_mono_ns": self.t1_ns,
            "duration_s": round((self.t1_ns - self.t0_ns) / 1e9, 3),
            "utc_start": datetime.fromtimestamp(self.wall0, UTC).isoformat(),
            "files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files if p.exists()],
            "source_stats": self.source.stats(),
        }
        if self.coverage is not None:
            doc["coverage"] = self.coverage.status()
        self.manifest_path.write_text(json.dumps(doc, indent=1))
        return doc
