"""r1d's `hand.calibrate {side, kind: "air_pinch"}` (session 10; DESIGN §6.2 / D27): measure one hand's air-pinch
reference and keep it (r1mc3.hands.calibration), so that `hand.grasp {object_class: "thin_flexible"}` stops refusing.

The goal: the hand (empty, in free air - the operator's / client's job, as for `hand.tare`) closes a `tip` pinch on
nothing with the grip state machine (layer_zero_m 0: the layer term then reads the raw tip-shell clearance where the
tips stopped), the reference = that clearance (`air_pinch_layer`), checked: the tips met each other - both tip pads
pressed, or (session 12, HW2 H3: the real tip pads are a ~10 N switch) the thumb and the index both stalled with the
tips' points at their meeting distance (`calibration.accept_air_pinch`) - and the value inside AIR_PINCH_MIN_M ..
AIR_PINCH_MAX_M; then the hand opens. A refused calibration keeps the old
reference (code CALIBRATION_FAILED + why). Nothing but the hand moves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .codes import Code
from .schemas import Result, fail

if TYPE_CHECKING:
    from .l2 import L2
    from .service import Ticket

VERDICT_MAX_S = 8.0  # the close ends by its own TIMEOUT_S (6 s); this bounds the goal


def store(l2: L2) -> Any:
    """The r1d's HandCalibration (created on first use; R1_HAND_CAL_FILE loads / keeps it)."""
    hc = getattr(l2, "_hand_cal", None)
    if hc is None:
        from r1mc3.hands.calibration import HandCalibration

        hc = HandCalibration()
        l2._hand_cal = hc  # type: ignore[attr-defined]
    return hc


def hand_calibrate(l2: L2, r: Any, t: Ticket) -> Result | None:
    from .goals import HAND_RESOURCES

    bad = l2._hands_or_fail(r.op, r.id)
    if bad is not None:
        return bad
    if r.side in l2.grips:
        return fail(r.op, r.id, Code.RESOURCE_BUSY, reason="the hand holds a grip: hand.release first")
    return l2.accept(r, t, frozenset([HAND_RESOURCES[r.side]]), CalibrateRunner(l2, r))


class CalibrateRunner:
    """The air pinch (a GraspRunner of its own, ticked here - not one of L2's held grips), then the hand opens."""

    def __init__(self, l2: L2, r: Any) -> None:
        self.l2, self.r = l2, r
        self.gr: Any = None
        self.opener: Any = None
        self.t0 = 0
        self.end_state: str | None = None
        self.code = Code.OK
        self.stopped_by: dict[str, Any] | None = None
        self.value: float | None = None
        self.previous: float | None = None
        self.verdict: dict[str, Any] | None = None
        self.why: str | None = None
        self.basis = ""  # how the tips' meeting was shown: "tip pressure" | "stall" (session 12)

    def ready(self, now_ns: int) -> bool:
        return True

    def start(self, now_ns: int) -> Result | None:
        from r1mc3.hands.grip import GraspRunner

        self.previous = store(self.l2).layer_zero(self.r.side)
        self.gr = GraspRunner(self.l2.svc.hands, self.r.side, "tip", "thin_flexible", layer_zero_m=0.0)
        self.gr.start(now_ns)
        self.t0 = now_ns
        return None

    def _judge(self) -> None:
        from r1mc3.hands.calibration import accept_air_pinch

        v = self.gr.verdict()
        self.verdict = v.as_dict()
        # session 12: the tips' meeting shown by the pads OR by the stall of both tips (HW2 H3: the real tip pads read
        # 0.00-0.02 kPa while the tips touched - r1mc3.hands.calibration.accept_air_pinch)
        z, self.basis, why = accept_air_pinch(self.verdict)
        if z is None:
            self.why = why
            return
        try:
            store(self.l2).set_air_pinch(self.r.side, z)
        except ValueError as e:
            self.why = str(e)
            return
        self.value = z

    def tick(self, now_ns: int) -> str | None:
        if self.end_state is not None:
            return self.end_state
        if self.opener is None:
            st = self.gr.tick(now_ns)
            if st in ("CLOSING", "CONTACT") and (now_ns - self.t0) * 1e-9 < VERDICT_MAX_S:
                return None
            self._judge()
            from r1mc3.hands import grip as G

            from .l2 import HandMoveRunner

            self.gr.release(now_ns)
            self.opener = HandMoveRunner(self.l2, self.r.side, G.preset("open", self.r.side), release=True)
            self.opener.start(now_ns)
            return None
        st = self.opener.tick(now_ns)
        if st is None:
            return None
        if self.value is None:
            self.end_state, self.code = "aborted", Code.CALIBRATION_FAILED
            self.stopped_by = {"code": "CALIBRATION_FAILED", "detail": self.why}
        else:
            self.end_state = "succeeded"
        return self.end_state

    def stop(self, now_ns: int, why: str) -> None:
        if self.end_state is not None:
            return
        if self.gr is not None and self.opener is None:
            self.gr.release(now_ns)
        self.end_state = why
        self.code = {"preempted": Code.PREEMPTED, "timed_out": Code.TIMEOUT}.get(why, Code.CANCELED)
        self.stopped_by = {"code": why.upper(), "detail": "r1d stopped the calibration"}

    def end(self, state: str, code: Code, stopped_by: dict[str, Any] | None) -> None:
        if self.end_state is None:
            self.end_state, self.code, self.stopped_by = state, code, stopped_by

    def outcome(self) -> tuple[Code, dict[str, Any], dict[str, Any]]:
        data: dict[str, Any] = {
            "moved": self.gr is not None,
            "side": self.r.side,
            "kind": self.r.kind,
            "layer_zero_m": None if self.value is None else round(self.value, 5),
            "previous_layer_zero_m": None if self.previous is None else round(self.previous, 5),
            "kept": self.value is None,
            "met_by": self.basis or None,
            "reason": self.why,
            "verdict": self.verdict,
            "stopped_by": self.stopped_by,
        }
        return self.code, data, ({} if self.why is None else {"reason": self.why})
