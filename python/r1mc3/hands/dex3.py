"""Dex3-1 low level over DDS (r1hands minimal; DESIGN.md §2.1, §6.1; R-l: stale / timeout paths that work).

    hands = Dex3Hands(domain=0).start()                 # DDS participant, readers, writers, the 100 Hz thread
    hands.command("left", q7, kp=..., kd=..., ttl_s=1.0)
    hands.state("left")                                 # q, dq, tau_est, temperature, pressure (9 x 12), age, valid
    hands.close()

Wire: unitree_hg HandCmd_ / HandState_ on rt/dex3/{left,right}/cmd|state (r1mc3.dds.hg; the topics dex3_server
serves), 7 motors in Dex3 DDS order (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1), the mode
byte RIS_Mode_t = id | status << 4 | timeout << 7 (status 1 = FOC; Dex3-1 User Manual; v2 r1mc2/dex3.py:93,
r1sim.dex3.mode_byte).

Timeout semantics (three layers, none of them limp by our choice):
1. The motors' own deadman: every command carries the timeout bit, so a hand goes passive ~1 s after commands stop
   (this process dying or stalling; the manual's 1 s timeout). That is the hardware's, as DESIGN §2.1 says (a held
   object drops). A motor that hit its timeout keeps the flag until a command WITHOUT the bit clears it, so the first
   CLEAR_TICKS commands after (re)starting a hand go without it (v2 r1mc2/hands.py:822-826).
2. A command's TTL (<= 5 s): when it expires the loop holds the measured position with the default gains (kp 1.5,
   kd 0.2: Unitree's own clients' values, v2 dex3.py) - the fingers stop where they are, still powered.
3. State staleness: a hand whose state is older than STATE_STALE_S is reported `valid: False`; a TTL-expired hand
   without fresh state keeps its last held command (never a jump to a stale reading).
4. A FROZEN state stream (session 6; DESIGN §12.6 "hand state freeze"): samples keep arriving but repeat one reading
   bit for bit. Real samples never repeat exactly (sensor noise on q / tau_est / the pads' cells), so each sample's
   serialized bytes are compared with the previous one's; no change for longer than FROZEN_S -> `valid: False`,
   `frozen: True`, `invalid_reason: "frozen"` (the recordings then store nothing and abort; the contact bits go
   invalid; a TTL hold does not take a frozen q).
A hand that was never commanded is left alone (nothing published for it), as v2 did.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

SIDES: tuple[str, str] = ("left", "right")
NAMES: tuple[str, ...] = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
N = 7
N_PADS = 9
# Limits [rad], Dex3 DDS order: Unitree's URDFs (unitree_ros dex3_1_{l,r}.urdf; v2 r1mc2/dex3.py; models/dex3_1/)
Q_MIN: dict[str, FloatArray] = {
    "left": np.array([-1.0472, -0.6109, 0.0, -1.5708, -1.7453, -1.5708, -1.7453]),
    "right": np.array([-1.0472, -1.0472, -1.7453, 0.0, 0.0, 0.0, 0.0]),
}
Q_MAX: dict[str, FloatArray] = {
    "left": np.array([1.0472, 1.0472, 1.7453, 0.0, 0.0, 0.0, 0.0]),
    "right": np.array([1.0472, 0.6109, 0.0, 1.5708, 1.7453, 1.5708, 1.7453]),
}
MARGIN = 0.03  # commands stay this far inside the limits (v2 dex3.MARGIN)
TAU_MAX = np.array([2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])  # URDF effort [N m]
KP_DEFAULT = np.full(N, 1.5)  # Unitree's clients (xr_teleoperate Dex3_1_Controller), v2 dex3.KP
KD_DEFAULT = np.full(N, 0.2)
KP_MAX = 5.0  # bound on a commanded per-finger kp (v2 never exceeded ~3; an ESTIMATE until HW1's finger steps)
KD_MAX = 1.0
RATE_HZ = 100.0
STATE_STALE_S = 0.1
FROZEN_S = (
    0.1  # unchanged this long (>= ~10 samples at the state rate) = frozen; the bridge's noise: see test_hands_frozen
)
CLEAR_TICKS = 3
TTL_MAX_S = 5.0
TOPIC_CMD = {"left": "rt/dex3/left/cmd", "right": "rt/dex3/right/cmd"}
TOPIC_STATE = {"left": "rt/dex3/left/state", "right": "rt/dex3/right/state"}


def mode_byte(motor_id: int, status: int = 0x01, timeout: int = 1) -> int:
    """Unitree RIS_Mode_t: id (4 bits) | status (3 bits) << 4 | timeout (1 bit) << 7 (Dex3-1 User Manual)."""
    return (motor_id & 0x0F) | ((status & 0x07) << 4) | ((timeout & 0x01) << 7)


def clamp_q(q: FloatArray, side: str) -> FloatArray:
    lo = Q_MIN[side] + MARGIN * (Q_MIN[side] < 0)
    hi = Q_MAX[side] - MARGIN * (Q_MAX[side] > 0)
    return np.asarray(np.clip(q, lo, hi), dtype=np.float64)


@dataclass
class _Cmd:
    q: FloatArray
    dq: FloatArray
    kp: FloatArray
    kd: FloatArray
    tau: FloatArray
    until: float  # monotonic s
    held: bool = False  # the TTL expired: holding the measured position


@dataclass
class _Hand:
    cmd: _Cmd | None = None
    ticks_since_start: int = 0
    writes: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    t_state: float | None = None
    states: int = 0
    fp: bytes | None = None  # the last sample's serialized bytes (a frozen stream repeats them exactly)
    t_change: float | None = None  # when a sample last differed from the one before it
    repeats: int = 0  # consecutive identical samples now
    repeats_max: int = 0  # the longest run seen (normal operation: the noise floor of the threshold)
    sent: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None  # last published (q, kp, kd, tau)


def _vec(v: Any, name: str, default: FloatArray) -> FloatArray:
    if v is None:
        return default.copy()
    a = np.asarray(v, dtype=np.float64)
    if a.shape != (N,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: {N} finite values expected (Dex3 DDS order {', '.join(NAMES)})")
    return a


class Dex3Hands:
    """The two hands over DDS (one participant; readers with listeners-free polling in the loop thread)."""

    def __init__(self, domain: int = 0, *, rate_hz: float = RATE_HZ, sides: tuple[str, ...] = SIDES) -> None:
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        from r1mc3.dds import hg

        self.hg = hg
        self.rate_hz = rate_hz
        self.sides = sides
        self.dp = DomainParticipant(domain)
        self._w: dict[str, Any] = {}
        self._r: dict[str, Any] = {}
        self._msg: dict[str, Any] = {}
        for s in sides:  # entities created here, never inside the loop (v2's rule)
            self._w[s] = DataWriter(self.dp, Topic(self.dp, TOPIC_CMD[s], hg.HandCmd_))
            self._r[s] = DataReader(self.dp, Topic(self.dp, TOPIC_STATE[s], hg.HandState_))
            self._msg[s] = hg.hand_cmd_zero(N)
        self._h = {s: _Hand() for s in sides}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.overruns = 0

    # ---------------------------------------------------------------- API
    def start(self) -> Dex3Hands:
        self._thread = threading.Thread(target=self._loop, name="r1hands-dex3", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def command(
        self,
        side: str,
        q: Any,
        *,
        dq: Any = None,
        kp: Any = None,
        kd: Any = None,
        tau: Any = None,
        ttl_s: float = 1.0,
    ) -> dict[str, Any]:
        """Per-motor setpoints for one hand, published every loop tick until `ttl_s` (then hold where it is)."""
        if side not in self._h:
            raise ValueError(f"side: one of {self.sides}")
        if not 0.0 < ttl_s <= TTL_MAX_S:
            raise ValueError(f"ttl_s: (0, {TTL_MAX_S}]")
        qa = _vec(q, "q", np.zeros(N))
        qc = clamp_q(qa, side)
        kpa = _vec(kp, "kp", KP_DEFAULT)
        kda = _vec(kd, "kd", KD_DEFAULT)
        if np.any(kpa < 0) or np.any(kpa > KP_MAX) or np.any(kda < 0) or np.any(kda > KD_MAX):
            raise ValueError(f"gains: kp in [0, {KP_MAX}], kd in [0, {KD_MAX}]")
        taua = _vec(tau, "tau", np.zeros(N))
        tau_c = np.clip(taua, -TAU_MAX, TAU_MAX)
        with self._lock:
            h = self._h[side]
            h.cmd = _Cmd(
                q=qc, dq=_vec(dq, "dq", np.zeros(N)), kp=kpa, kd=kda, tau=tau_c, until=time.monotonic() + ttl_s
            )
        return {
            "ok": True,
            "side": side,
            "q": qc.tolist(),
            "clamped_q": bool(np.any(qc != qa)),
            "clamped_tau": bool(np.any(tau_c != taua)),
            "ttl_s": ttl_s,
        }

    def _frozen(self, h: _Hand, now: float) -> bool:
        return h.t_change is not None and h.repeats > 0 and now - h.t_change > FROZEN_S

    def state(self, side: str) -> dict[str, Any]:
        with self._lock:
            h = self._h[side]
            now = time.monotonic()
            age = None if h.t_state is None else now - h.t_state
            frozen = self._frozen(h, now)
            stale = age is None or age >= STATE_STALE_S
            out = dict(h.state)
            if h.sent is not None:  # the command on the wire (the HW1 recordings log it next to the state)
                out.update(q_cmd=h.sent[0].copy(), kp=h.sent[1].copy(), kd=h.sent[2].copy(), tau_ff=h.sent[3].copy())
            out.update(
                valid=not stale and not frozen,
                invalid_reason=("no state" if age is None else "stale" if stale else "frozen" if frozen else None),
                frozen=frozen,
                unchanged_ms=None if h.t_change is None else (now - h.t_change) * 1e3,
                repeats=h.repeats,
                repeats_max=h.repeats_max,
                age_ms=None if age is None else age * 1e3,
                commanding=h.cmd is not None,
                held=bool(h.cmd is not None and h.cmd.held),
                writes=h.writes,
                states=h.states,
            )
            return out

    # ---------------------------------------------------------------- loop
    def _read(self, now: float) -> None:
        for s in self.sides:
            got = self._r[s].take(N=16)
            if not got:
                continue
            with self._lock:
                h = self._h[s]
                for g in got:  # every sample: a frozen stream repeats one reading bit for bit
                    fp = bytes(g.serialize())
                    if fp != h.fp:
                        h.fp, h.t_change, h.repeats = fp, now, 0
                    else:
                        h.repeats += 1
                        h.repeats_max = max(h.repeats_max, h.repeats)
            m = got[-1]
            ms = list(m.motor_state)[:N]
            st = {
                "q": np.array([x.q for x in ms]),
                "dq": np.array([x.dq for x in ms]),
                "tau_est": np.array([x.tau_est for x in ms]),
                "temperature": np.array([list(x.temperature) for x in ms], dtype=np.float64),
                "motorstate": [int(x.motorstate) for x in ms],
                "pressure": np.array([list(p.pressure) for p in list(m.press_sensor_state)[:N_PADS]]),
                "error": [int(e) for e in m.error],
                "power_v": float(m.power_v),
            }
            with self._lock:
                h = self._h[s]
                h.state = st
                h.t_state = now
                h.states += len(got)

    def _publish(self, now: float) -> None:
        for s in self.sides:
            with self._lock:
                h = self._h[s]
                c = h.cmd
                if c is None:
                    continue
                if now > c.until and not c.held:
                    fresh = h.t_state is not None and now - h.t_state < STATE_STALE_S and not self._frozen(h, now)
                    if fresh:  # hold where it is, still powered, default gains
                        c.q = clamp_q(np.asarray(h.state["q"], dtype=np.float64), s)
                    c.dq = np.zeros(N)
                    c.tau = np.zeros(N)
                    c.kp = KP_DEFAULT.copy()
                    c.kd = KD_DEFAULT.copy()
                    c.held = True
                tb = 0 if h.ticks_since_start < CLEAR_TICKS else 1
                h.ticks_since_start += 1
                q, dq, kp, kd, tau = c.q.copy(), c.dq.copy(), c.kp.copy(), c.kd.copy(), c.tau.copy()
            msg = self._msg[s]
            for i in range(N):
                mc = msg.motor_cmd[i]
                mc.mode = mode_byte(i, 0x01, tb)
                mc.q, mc.dq, mc.tau = float(q[i]), float(dq[i]), float(tau[i])
                mc.kp, mc.kd = float(kp[i]), float(kd[i])
            self._w[s].write(msg)
            with self._lock:
                h.writes += 1
                h.sent = (q, kp, kd, tau)

    def step(self, now: float | None = None) -> None:
        """One loop iteration (the thread calls it at rate_hz; tests may call it directly)."""
        t = time.monotonic() if now is None else now
        self._read(t)
        self._publish(t)

    def _loop(self) -> None:
        period = 1.0 / self.rate_hz
        nxt = time.monotonic()
        while not self._stop.is_set():
            self.step()
            nxt += period
            d = nxt - time.monotonic()
            if d < 0:
                self.overruns += 1
                nxt = time.monotonic()
            else:
                self._stop.wait(d)
