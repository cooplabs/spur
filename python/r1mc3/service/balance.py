"""`balance` - the operator's way from a freeze back into BALANCE (session 11; `r1 balance`).

Why: a balancing robot can end up in STAND_FREEZE (legs ramped to the stand at stand gains, arms / head held - never
limp) three ways: the operator's `r1 freeze`, an invalid policy action (rung 3), or the state-age rung (D34's
`stale_rung 1`: the robot state stayed older than its limit). The core has always accepted ENTER_BALANCE from such a
freeze (core.cpp `R1RT_REQ_ENTER_BALANCE`: fresh state, a valid first action), but r1d had no word for it - so neither
HW2's operator freeze with the arms forward (the runbook's F5: the arm-aware freeze pose, the user holding the robot)
nor rung 1 (which needs the operator to bring BALANCE back) could be used (S10.md / STATE_AGE.md "Session 10").

The word is OPERATOR ONLY (a human typed it, holding the robot) and r1d refuses it before the core sees it when the
freeze is not one a balancing robot can leave: not in STAND_FREEZE, the stand's freeze (ON_STAND's `r1 stand-freeze`:
a hanging robot never balances - `r1 stand-hold` is that way back), a stand session (r1rt `--stand-session`). The core
then has the final say (session 11): policy loaded, the upper executor ready, a FRESH state - and never on a state older
than the state-age limit now (the board-tick content age does not re-base during a freeze any more: a state that stayed
late reads late) - and the policy's first action valid (else it freezes again). A refusal moves nothing: the robot stays
in its freeze. After BALANCE the freeze's STOP is still posted (arms HOLD, base command zero): `r1 resume` clears it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .codes import Code
from .schemas import BalanceReq, Result, fail

if TYPE_CHECKING:
    from .service import Service, Ticket

WAY_BACK = "`r1 balance` (the operator, holding the robot), then `r1 resume`"
NEXT_AFTER = "`r1 resume` clears the freeze's STOP (the arms hold and the base command is zero until then)"


def handles(r: Any) -> bool:
    return isinstance(r, BalanceReq)


def request(svc: Service, r: BalanceReq, t: Ticket) -> Result | None:
    """r1d's checks, then the core's ENTER_BALANCE through the mailbox (answered in tick(); words added on the way)."""
    if r.requester != "operator":
        return fail(
            r.op,
            r.id,
            Code.NOT_PERMITTED,
            reason="balance from a freeze is the operator's word (you hold the robot): `r1 balance`",
        )
    mode = svc.mode_name()
    if svc.stand_session() is True:
        return fail(
            r.op,
            r.id,
            Code.WRONG_MODE,
            mode=mode,
            reason="stand session: balance REFUSED (r1rt runs with --stand-session; lifting the stand lock is the "
            "runbook's L1-L6)",
        )
    if mode != "stand_freeze":
        why = "already balancing" if mode == "balance" else f"the core is in {mode.upper()}, not STAND_FREEZE"
        return fail(
            r.op,
            r.id,
            Code.WRONG_MODE,
            mode=mode,
            needs="stand_freeze from BALANCE (`r1 freeze`, an invalid policy action or the state-age rung)",
            reason=why,
        )
    if svc.freeze_from_stand:
        return fail(
            r.op,
            r.id,
            Code.WRONG_MODE,
            mode=mode,
            reason="this freeze came from the stand (ON_STAND): a hanging robot never balances - `r1 stand-hold`",
        )
    state_names = {"reclaim_state", "state"}

    def words(res: Result) -> Result:
        if res.ok:
            return res.model_copy(update={"data": {**res.data, "next": NEXT_AFTER}})
        det = dict(res.details)
        rname = det.get("refusal_name")
        if rname in state_names:
            sa = _state_age(svc)
            age = None if sa is None else sa.get("content_age_ms")
            if sa is not None and sa.get("old_now"):
                det["reason"] = (
                    "the robot state is older than the state-age limit"
                    + (f" (content age {float(age):.0f} ms)" if age is not None else "")
                    + ": the robot stays frozen - try again once `r1 status` shows no STATE_OLD"
                )
            else:
                det["reason"] = (
                    "no fresh robot state in r1rt (or its upper executor not ready): the robot stays frozen - "
                    "check `r1 status`, then try again"
                )
            det["state_age"] = sa
        elif rname == "stand_session":
            det["reason"] = "stand session: balance REFUSED (r1rt runs with --stand-session)"
        elif rname == "mode":
            det["reason"] = (
                "the core refused: no policy loaded, or the policy's first action was invalid (it froze again) - "
                "the robot stays in STAND_FREEZE; `r1 status`"
            )
        return res.model_copy(update={"details": det})

    from .service import Ticket as _Ticket

    class _WordsTicket(_Ticket):
        """The mailbox sequence's ticket: the core's reply gets the balance words, then the caller's ticket (the
        server waits on it) is finished."""

        def finish(self, res: Result) -> None:
            out = words(res)
            super().finish(out)
            t.finish(out)

    return svc._mode_request_now(r.op, r.id, "enter_balance", r.requester, _WordsTicket(request=t.request))


def status_hint(out: dict[str, Any], core_mode: str | None) -> None:
    """`r1 status` (session 11): a STAND_FREEZE that a balancing robot can leave carries its way back in
    `state_age.freeze_way_back` (the words line comes from state_age.state_age_words); the stand's freeze keeps its own
    line (`r1 stand-hold`)."""
    sa = out.get("state_age")
    if core_mode != "stand_freeze" or out.get("freeze_from_stand") or not isinstance(sa, dict):
        return
    sa["freeze_way_back"] = WAY_BACK


def _state_age(svc: Service) -> dict[str, Any] | None:
    from .service import _fault_names
    from .state_age import state_age_status

    hb = svc.port.heartbeat()
    faults = None if hb is None else _fault_names(hb.faults)
    return state_age_status(svc.port, faults)


__all__ = ["NEXT_AFTER", "WAY_BACK", "handles", "request", "status_hint"]
