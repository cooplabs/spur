"""`Recorder`: r1d's record.* ops - one HW1 recording at a time, executed in r1d's tick (r1mc3.record.plans).

    r1 record plan chirp joint=left_knee amplitude_rad=0.03 f1_hz=5      # dry run: the plan + its checks, nothing moves
    r1 record run chirp joint=left_knee --yes                           # runs it (ON_STAND), writes the manifest

While a body plan runs, every r1d tick sends the group's setpoint as a STREAM record with a short TTL (0.1 s), so a
stalled r1d stops the joint (the executor decelerates to HOLD); `r1 stop` (any stop level) or the core leaving
ON_STAND ends it the same way. At the end the group gets a HOLD at the plan's last setpoint. Hand plans renew the
hand command every tick with a 0.2 s TTL (Dex3Hands then holds where it is).

The manifest (JSON, `<out_dir>/<UTC start>_<kind>_<joint|side>.json`): the plan as described, the status (done /
aborted + why), the start / end in CLOCK_MONOTONIC ns (the clock of r1log's records) and UTC, the pushes and refusals,
and the r1log files whose time span overlaps the window (`log_dir`, r1log's --out). The fits (docs/hardware/
HW1_FITS.md) cut the 500 Hz telemetry by the monotonic window and read the commanded signal from it (q_cmd / tau_ff
as the core sent them), not from the plan.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from r1mc3.service.codes import GROUP_JOINTS, Code
from r1mc3.service.schemas import RecordReq, Result, fail, ok

from .plans import ALIASES, CAMERA_KINDS, FINGERS, KINDS, Plan, PlanError, plan

if TYPE_CHECKING:
    from r1mc3.service.service import Service, Ticket

HANDS_TOPIC = "/r1hands/state"  # r1sim.sysid.records.HANDS_TOPIC (r1hands' full-rate ring does not exist yet)
HAND_FIELDS = ("q", "dq", "tau_est", "temperature", "pressure", "q_cmd", "kp", "kd", "tau_ff")  # sysid.records
RECORD_TTL_S = 0.1
HAND_TTL_S = 0.2
HAND_INVALID_ABORT_NS = 250_000_000  # a hand recording aborts when its hand's state stays invalid this long
CLOSE_DIR_LEFT = np.array([0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])  # v2 r1mc2/dex3.py CLOSE_DIR
MIRROR = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
OPEN_LEFT = np.array([0.0, 0.0, 0.05, -0.05, -0.05, -0.05, -0.05])  # v2 dex3 PRESETS_LEFT["open"]


def close_dir(side: str) -> np.ndarray:
    return CLOSE_DIR_LEFT.copy() if side == "left" else MIRROR * CLOSE_DIR_LEFT


def open_preset(side: str) -> np.ndarray:
    return OPEN_LEFT.copy() if side == "left" else MIRROR * OPEN_LEFT


@dataclass
class _Run:
    plan: Plan
    cmd_id: int
    t0_ns: int
    wall0: float
    pushes: int = 0
    refused: int = 0
    status: str = "running"
    reason: str = ""
    t_end_ns: int = 0
    last_q: np.ndarray | None = None
    manifest: Path | None = None
    hand_cmds: int = 0
    hand_samples: int = 0
    hand_log: Any = None  # HandLog while a hand plan runs
    hand_invalid_since: int | None = None  # the hand's state went invalid (frozen / stale) at this time
    extra: dict[str, Any] = field(default_factory=dict)
    cam_rec_id: int | None = None  # session 9: the r1cam recording of a camera plan
    cam_manifest: dict[str, Any] | None = None


class CameraClient:
    """r1d's side of r1cam (127.0.0.1:R1CAM_PORT, default 8781): one request per call; raises OSError when r1cam does
    not answer (the camera plans then refuse; nothing else depends on it)."""

    def __init__(self, port: int | None = None) -> None:
        self.port = int(port if port is not None else os.environ.get("R1CAM_PORT", "8781"))

    def request(self, req: dict[str, Any]) -> dict[str, Any]:
        from r1mc3.cam.service import request

        return request(self.port, req)


def hand_schema() -> dict[str, Any]:
    """The /r1hands/state message (JSON): t_ns, side + the HAND_FIELDS arrays (= r1sim.sysid.records.hand_schema)."""
    arr = {"type": "array"}
    props: dict[str, Any] = {"t_ns": {"type": "integer"}, "side": {"enum": ["left", "right"]}}
    props.update({k: arr for k in HAND_FIELDS})
    return {"title": "r1hands_state", "type": "object", "properties": props, "required": ["t_ns", "side"]}


class HandLog:
    """The hand recordings' MCAP (topic /r1hands/state, JSON, one message per hand sample at r1d's tick): what the
    HW1 fits read (r1sim.sysid.records.load_hand) - the state next to the command that was on the wire."""

    def __init__(self, path: Path) -> None:
        from mcap.writer import Writer

        self.path = path
        self._f = open(path, "wb")  # closed in close()
        self._w = Writer(self._f)
        self._w.start(profile="", library="r1mc3.record")
        sid = self._w.register_schema(
            name="r1hands_state", encoding="jsonschema", data=json.dumps(hand_schema()).encode()
        )
        self._cid = self._w.register_channel(topic=HANDS_TOPIC, message_encoding="json", schema_id=sid)
        self.seq = 0
        self.t0_ns: int | None = None

    def write(self, t_ns: int, side: str, st: dict[str, Any]) -> bool:
        if not st.get("valid") or "q_cmd" not in st:
            return False
        msg: dict[str, Any] = {"t_ns": int(t_ns), "side": side}
        for k in HAND_FIELDS:
            v = st.get(k)
            if v is not None:
                msg[k] = np.asarray(v, dtype=np.float64).tolist()
        if self.t0_ns is None:
            self.t0_ns = int(t_ns)
        self._w.add_message(
            channel_id=self._cid,
            log_time=int(t_ns),
            publish_time=int(t_ns),
            data=json.dumps(msg).encode(),
            sequence=self.seq,
        )
        self.seq += 1
        return True

    def close(self) -> None:
        self._w.finish()  # type: ignore[no-untyped-call]
        self._f.close()


class Recorder:
    def __init__(
        self,
        svc: Service,
        *,
        out_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        camera: Any = None,
    ) -> None:
        self.svc = svc
        self.camera = camera if camera is not None else CameraClient()  # session 9: r1cam for the camera kinds
        self.out_dir = Path(out_dir or os.environ.get("R1_RECORD_DIR", "out/records"))
        self.log_dir = (
            Path(log_dir) if log_dir else (Path(os.environ["R1LOG_DIR"]) if "R1LOG_DIR" in os.environ else None)
        )
        self.run: _Run | None = None
        self.done: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- requests
    def _start_q(self) -> np.ndarray | None:
        for src in (self.svc.port.targets(), self.svc.port.measured()):
            if src is not None:
                return np.asarray(src, dtype=np.float64)
        return None

    def request(self, r: RecordReq, t: Ticket) -> Result:
        if r.op == "record.list":
            active = None if self.run is None else self._summary(self.run)
            kinds = list(KINDS + CAMERA_KINDS)
            return ok(r.op, r.id, kinds=kinds, aliases=ALIASES, active=active, done=self.done[-10:])
        if r.kind is None:
            return fail(r.op, r.id, Code.INVALID_ARGS, reason=f"kind: one of {', '.join(KINDS + tuple(ALIASES))}")
        try:
            p = plan(r.kind, r.params, self._start_q(), self.svc.limits)
        except PlanError as e:
            return fail(
                r.op, r.id, Code.LIMIT if "limit" in str(e) or "dq_max" in str(e) else Code.INVALID_ARGS, **e.details
            )
        if r.op == "record.plan" or r.dry_run:
            return ok(r.op, r.id, dry_run=True, plan=p.describe())
        return self._start(r, p)

    def _start(self, r: RecordReq, p: Plan) -> Result:
        if self.run is not None:
            return fail(r.op, r.id, Code.RESOURCE_BUSY, active=self._summary(self.run))
        mode = self.svc.mode_name()
        if p.needs_mode is not None and mode != p.needs_mode:
            return fail(r.op, r.id, Code.WRONG_MODE, mode=mode, needs=p.needs_mode)
        if p.group is not None and self.svc.port.tables.groups.get(p.group) is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason=f"the core has no '{p.group}' group")
        if p.hand is not None and self.svc.hands is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="no hands in this r1d")
        if p.group is not None and self._start_q() is None:
            return fail(r.op, r.id, Code.UNSUPPORTED, reason="the core's current targets are not readable")
        cam_id: int | None = None
        if p.camera is not None:  # session 9: r1cam records for the plan's duration (nothing moves)
            try:
                req = {"op": "record_start", "kind": p.kind, "label": str(p.params.get("label", "")),
                       "streams": p.camera["streams"], "depth_every": p.camera["depth_every"],
                       "video": p.camera["video"], "params": p.params}  # fmt: skip
                out = self.camera.request(req)
            except OSError as e:
                return fail(r.op, r.id, Code.UNSUPPORTED, reason=f"r1cam does not answer ({e})")
            if not out.get("ok"):
                return fail(r.op, r.id, Code.FAULT, reason="r1cam refused the recording", r1cam=out)
            cam_id = int(out["data"]["rec_id"])
        now = self.svc.port.now_ns()
        self.run = _Run(plan=p, cmd_id=self.svc.new_cmd_id(), t0_ns=now, wall0=time.time(), cam_rec_id=cam_id)
        if p.hand is not None:
            # session 11 (contact decides: a NOT_HOLDING grip keeps re-sending its grip until told to open): a hand
            # recording takes the hand like the operator's `r1 hand` does - the side's grip ends, so it never re-sends
            # its command over the recording's (grip's finding; before, the runbook had to say "r1 release first")
            l2 = getattr(self.svc, "l2", None)
            if l2 is not None:
                l2.end_grip(str(p.hand["side"]))
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.run.hand_log = HandLog(self.out_dir / (self._stem(self.run) + ".mcap"))
        self.tick(now)
        return ok(r.op, r.id, moved=p.group is not None or p.hand is not None, started=self._summary(self.run))

    def abort(self, reason: str) -> None:
        if self.run is not None and self.run.status == "running":
            self._finish("aborted", reason, self.svc.port.now_ns())

    # ---------------------------------------------------------------- tick
    def tick(self, now: int) -> None:
        run = self.run
        if run is None:
            return
        p = run.plan
        t = (now - run.t0_ns) * 1e-9
        if p.needs_mode is not None and self.svc.mode_name() != p.needs_mode:
            self._finish("aborted", f"the core left {p.needs_mode} ({self.svc.mode_name()})", now)
            return
        if t >= p.duration_s:
            self._finish("done", "", now)
            return
        if p.group is not None:
            q, dq = p.at(t)
            rc = self.svc.stream_setpoint(p.group, q, dq, RECORD_TTL_S, run.cmd_id, now, p.gains)
            run.pushes += 1
            run.refused += int(rc != 0)
            run.last_q = q
        if p.hand is not None and self.svc.hands is not None:
            self._hand_tick(run)

    def _hand_tick(self, run: _Run) -> None:
        assert self.svc.hands is not None and run.plan.hand is not None
        h = run.plan.hand
        side = h["side"]
        if "torque_nm" in h:  # finger_hold: the finger's motors torque-only (kp 0) in the closing direction
            st = self.svc.hands.state(side)
            q = np.asarray(st.get("q", open_preset(side)), dtype=np.float64)
            kp = np.full(7, 1.5)
            tau = np.zeros(7)
            for m in FINGERS[h["finger"]]:
                kp[m] = 0.0
                tau[m] = h["torque_nm"] * close_dir(side)[m]
            self.svc.hands.command(side, q, kp=kp, tau=tau, ttl_s=HAND_TTL_S)
        else:  # pad_press: the hand holds open
            self.svc.hands.command(side, open_preset(side), ttl_s=HAND_TTL_S)
        run.hand_cmds += 1
        now = self.svc.port.now_ns()
        st = self.svc.hands.state(side)
        if st.get("valid"):
            run.hand_invalid_since = None
        elif run.hand_invalid_since is None:
            run.hand_invalid_since = now
        elif now - run.hand_invalid_since > HAND_INVALID_ABORT_NS:  # a frozen / stale hand is not data
            self._finish("aborted", f"the {side} hand's state is not valid ({st.get('invalid_reason')})", now)
            return
        if run.hand_log is not None:
            run.hand_samples += int(run.hand_log.write(now, side, st))

    def _finish(self, status: str, reason: str, now: int) -> None:
        run = self.run
        assert run is not None
        run.status, run.reason, run.t_end_ns = status, reason, now
        p = run.plan
        if p.group is not None and status == "done" and run.last_q is not None:
            gid = self.svc.port.tables.groups[p.group]
            self.svc._push(p.group, gid, "hold", {"cmd_id": run.cmd_id, "t_issue_ns": now, "q": run.last_q.tolist()})
        if p.hand is not None and self.svc.hands is not None and "torque_nm" in p.hand:
            side = p.hand["side"]
            self.svc.hands.command(side, open_preset(side), ttl_s=1.0)  # the finger lets go
        if run.hand_log is not None:
            run.hand_log.close()
        if run.cam_rec_id is not None:
            try:
                out = self.camera.request({"op": "record_stop", "rec_id": run.cam_rec_id, "status": status,
                                           "reason": reason})  # fmt: skip
                run.cam_manifest = out.get("data") if out.get("ok") else {"error": out}
            except OSError as e:
                run.cam_manifest = {"error": f"r1cam does not answer ({e})"}
        run.manifest = self._write_manifest(run)
        self.done.append(self._summary(run))
        self.run = None

    # ---------------------------------------------------------------- manifest
    def _summary(self, run: _Run) -> dict[str, Any]:
        p = run.plan
        return {
            "kind": p.kind,
            "status": run.status,
            "reason": run.reason,
            "t_start_ns": run.t0_ns,
            "t_end_ns": run.t_end_ns or None,
            "duration_s": p.duration_s,
            "pushes": run.pushes,
            "refused": run.refused,
            "hand_cmds": run.hand_cmds,
            "manifest": None if run.manifest is None else str(run.manifest),
        }

    def _log_files(self, run: _Run) -> list[str]:
        if self.log_dir is None or not self.log_dir.is_dir():
            return []
        t0 = run.wall0 - 1.0
        return sorted(str(f) for f in self.log_dir.glob("r1log_*.mcap") if f.stat().st_mtime >= t0)

    def _stem(self, run: _Run) -> str:
        p = run.plan
        tag = p.params.get("joint") or p.params.get("side") or p.group or "all"
        label = f"_{p.params['label']}" if p.params.get("label") else ""
        return f"{datetime.fromtimestamp(run.wall0, UTC).strftime('%Y%m%dT%H%M%SZ')}_{p.kind}_{tag}{label}"

    def _press_notes(self, run: _Run) -> Path | None:
        """pad_press: the operator's notes in the fit's format (r1sim.sysid.records.load_presses): one entry per
        SCHEDULED press, times in the recording's clock (s from its first hand sample). Edit them if a press was late
        or a weight differed (the notes are the truth the fit uses)."""
        p = run.plan
        if p.kind != "pad_press" or p.hand is None or run.hand_log is None or run.hand_log.t0_ns is None:
            return None
        import yaml

        off = (run.hand_log.t0_ns - run.t0_ns) * 1e-9
        rows = [
            {
                "side": p.hand["side"],
                "pad": int(p.hand["pad"]),
                "x_mm": float(e["x_mm"]),
                "weight_g": float(e["weight_g"]),
                "t_start": round(e["t_start"] - off, 4),
                "t_end": round(e["t_end"] - off, 4),
            }
            for e in p.hand["schedule"]
        ]
        path = self.out_dir / (self._stem(run) + ".presses.yaml")
        path.write_text(yaml.safe_dump(rows, sort_keys=False))
        return path

    def _write_manifest(self, run: _Run) -> Path:
        p = run.plan
        wall1 = run.wall0 + (run.t_end_ns - run.t0_ns) * 1e-9
        name = self._stem(run) + ".json"
        doc = {
            "schema": "r1mc3.record_manifest/1",
            "plan": p.describe(),
            "base": None if p.base is None else p.base.tolist(),
            "group_joints": list(GROUP_JOINTS[p.group]) if p.group else [],
            "status": run.status,
            "reason": run.reason,
            "t_start_mono_ns": run.t0_ns,
            "t_end_mono_ns": run.t_end_ns,
            "utc_start": datetime.fromtimestamp(run.wall0, UTC).isoformat(),
            "utc_end": datetime.fromtimestamp(wall1, UTC).isoformat(),
            "pushes": run.pushes,
            "refused": run.refused,
            "hand_cmds": run.hand_cmds,
            "r1log_files": self._log_files(run),
            "hands_mcap": None if run.hand_log is None else str(run.hand_log.path),
            "hand_samples": run.hand_samples,
            "press_notes": None if (pn := self._press_notes(run)) is None else str(pn),
            "camera": None if run.cam_rec_id is None else {"plan": run.plan.camera, "r1cam": run.cam_manifest},
            "core_mode_at_end": self.svc.mode_name(),
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / name
        path.write_text(json.dumps(doc, indent=1))
        return path
