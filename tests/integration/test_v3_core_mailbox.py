"""The mode-request mailbox (session 5, layout 2 / ABI 5) from Python: the codes, a request through
`r1rt_py.SegmentView` answered at the core's next writer tick, BUSY while one pends, DAMP only from the operator, and
the takeover of the simulated robot driven entirely through the mailbox (as r1d / the CLI do) - the same result as
the direct calls of session 4. C++ side (races, abandoned claims): rt/tests/test_core_mailbox.cpp."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from r1sim import cpp_policy, world
from r1sim.engine import Simulation
from r1sim.model import centred_stand_pose
from r1sim.v3_core import (
    MODE_BALANCE,
    MODE_DAMP,
    MODE_MONITOR,
    REQ_DAMP,
    REQ_RESUME,
    REQ_STOP,
    REQUESTER_OPERATOR,
    REQUESTER_R1D,
    REQUESTER_SIM,
    EmulatedBuiltin,
    TakeoverPlan,
    V3CoreController,
)

pytestmark = pytest.mark.skipif(cpp_policy.find_module() is None, reason="r1rt_py is not built (scripts/ci.sh cpp)")


@pytest.fixture(scope="module")
def w() -> world.World:
    return world.build_world("real", visual=False)


def _mod() -> Any:
    return cpp_policy.import_r1rt_py()


def test_the_codes_by_name() -> None:
    m = _mod()
    assert m.ABI_VERSION == 8 and m.LAYOUT_VERSION == 4  # session 10: ABI 8 (unchanged), layout 4 (the executor status)
    req = m.requests()
    assert req["reclaim"] == 10
    assert req["takeover"] == 6 and req["takeover_released"] == 7 and req["takeover_abort"] == 8
    assert req["on_stand"] == 9 and req["damp"] == REQ_DAMP and req["stop"] == REQ_STOP
    who = m.requesters()
    assert who["operator"] == REQUESTER_OPERATOR and who["r1d"] == REQUESTER_R1D and who["sim"] == REQUESTER_SIM
    assert m.modes()["on_stand"] == 6 and m.modes()["balance"] == MODE_BALANCE
    assert m.status_codes()["again"] == -8 and m.status_codes()["abandoned"] == -9
    assert m.refusals()["not_operator"] == 20
    assert m.refusals()["reclaim_token"] == 23 and m.refusals()["reclaim_state"] == 24


def test_a_request_is_answered_at_the_next_tick_and_damp_needs_the_operator(w: world.World) -> None:
    sim = Simulation(w.model, w.index)
    sim.reset(w.initial_qpos(centred_stand_pose(w.model)))
    ctl = V3CoreController(w.model)  # enters BALANCE at its first call (the single-core lockstep tests' way)
    sim.add_controller(ctl, ctl.period_s, "v3")
    sim.run_for(0.2)
    view = ctl.core.segment_view()  # = SegmentView.of_core(core)
    assert view.kind == "core"
    seq = view.request_mode(REQ_STOP, REQUESTER_R1D, 0)
    assert view.mode_reply(seq) is None  # pending until the writer tick
    with pytest.raises(BlockingIOError):
        view.request_mode(REQ_RESUME, REQUESTER_R1D, 0)  # one request at a time
    sim.run_for(0.002)
    assert view.mode_reply(seq) == (0, 0, MODE_BALANCE)
    assert ctl.status()["stop_active"] == 1
    # DAMP from the simulator's requester id: refused (not the operator), the robot stays balanced
    seq = view.request_mode(REQ_DAMP, REQUESTER_SIM, 0)
    sim.run_for(0.002)
    status, refusal, mode = view.mode_reply(seq)
    assert (status, refusal, mode) == (-6, 20, MODE_BALANCE)
    seq = view.request_mode(REQ_DAMP, REQUESTER_OPERATOR, 0)
    sim.run_for(0.002)
    assert view.mode_reply(seq) == (0, 0, MODE_DAMP)
    st = ctl.status()
    assert st["mode_requests"] == 3 and st["mode_last_seq"] == seq
    with pytest.raises(ValueError):
        view.mode_reply(seq + 5)  # never issued


def _run_takeover(w: world.World, *, via_mailbox: bool, seconds: float = 2.6) -> dict[str, Any]:
    sim = Simulation(w.model, w.index)
    sim.reset(w.initial_qpos(centred_stand_pose(w.model)))
    builtin = EmulatedBuiltin(w.model)
    plan = TakeoverPlan(builtin, t_request_s=1.0, release_s=0.3, via_mailbox=via_mailbox)
    ctl = V3CoreController(w.model, takeover=plan, trace_cmd=True)
    sim.add_controller(builtin, ctl.period_s, "builtin")
    sim.add_controller(ctl, ctl.period_s, "v3")
    sim.run_for(seconds)
    return {"plan": plan, "ctl": ctl, "status": ctl.status(), "trace": ctl.cmd_trace}


def test_the_takeover_through_the_mailbox_matches_the_direct_calls(w: world.World) -> None:
    """r1d's takeover (TAKEOVER, ReleaseMode, RELEASED) posted through the mailbox: the core runs the same sequence
    (MIRROR at 1.0 s, BLEND from 1.3 s, BALANCE after 1.0 s of blend) and publishes the same commands as with the
    direct requests up to a state one tick fresher: the mailbox is read inside the writer step after this tick's
    lowstate, a direct call before the step acts on the previous tick's (2 ms older) - the entry action differs by
    ~2e-5 rad."""
    a = _run_takeover(w, via_mailbox=True)
    b = _run_takeover(w, via_mailbox=False)
    pa, pb = a["plan"], b["plan"]
    assert pa.request_status == 0 and pa.refusal == "ok" and pa.decision_status == 0
    assert pa.seqs == [1, 2] and a["status"]["mode_requests"] == 2
    assert a["status"]["mode"] == MODE_BALANCE == b["status"]["mode"]
    assert pa.t_requested == pb.t_requested and pa.t_decided == pb.t_decided
    ta, tb = a["trace"], b["trace"]
    assert len(ta) == len(tb)
    first_pub = next(i for i, r in enumerate(ta) if r[1])
    assert first_pub == next(i for i, r in enumerate(tb) if r[1])
    assert abs(ta[first_pub][0] - 1.0) < 1e-9  # MIRROR publishes from the request's tick
    dq = max(float(np.max(np.abs(ra[4] - rb[4]))) for ra, rb in zip(ta, tb, strict=True))
    assert dq < 1e-3, dq  # the same commands, tick for tick (up to the fresher state of the mailbox's gates)
    assert [r[2] for r in ta] == [r[2] for r in tb]
    assert a["status"]["faults_seen"] == b["status"]["faults_seen"]
    assert MODE_MONITOR in {r[2] for r in ta}


def test_r1ds_client_side_on_a_segment_view(w: world.World) -> None:
    """SegmentView (what r1d attaches to r1rt's POSIX segment with) has the client side the lockstep Core has on
    itself: push_upper / upper_keepalive / stop_epoch, and the core's state / mirror as dicts."""
    from r1sim.v3_core import GROUP_ARM_LEFT, SOURCE_STREAM

    sim = Simulation(w.model, w.index)
    sim.reset(w.initial_qpos(centred_stand_pose(w.model)))
    ctl = V3CoreController(w.model)
    sim.add_controller(ctl, ctl.period_s, "v3")
    sim.run_for(0.3)
    view = _mod().SegmentView.of_core(ctl.core)
    st = view.state()
    mi = view.mirror()
    assert st is not None and mi is not None
    assert st["q"].shape == (26,) and mi["q_target"].shape == (26,) and mi["kp"].shape == (26,)
    assert st["mode"] == MODE_BALANCE == mi["mode"]
    np.testing.assert_allclose(mi["q_target"], ctl.last_q, atol=0.0)  # the last published command
    assert view.stop_epoch() == ctl.core.stop_epoch()
    q_arm = ctl.last_q[14:19].copy()
    q_arm[0] += 0.1
    t_ns = round(sim.time * 1e9)
    rec = {"cmd_id": 77, "group": GROUP_ARM_LEFT, "source": SOURCE_STREAM, "n_joints": 5, "t_issue_ns": t_ns,
           "ttl_ns": 300_000_000, "q": list(map(float, q_arm)), "dq": [0.0] * 5}  # fmt: skip
    assert view.push_upper(rec) == 0
    assert view.upper_keepalive(GROUP_ARM_LEFT, 77, t_ns) == 0
    assert view.upper_keepalive(6, 77, t_ns) != 0  # no group 6
    sim.run_for(0.2)
    s = ctl.status()
    assert s["upper_accepted"] == 1 and s["upper_cmd_id"][0] == 77
    assert abs(ctl.last_q[14] - q_arm[0]) < 1e-9  # the stream reached (0.1 rad at <= 2 rad/s)
    # a stop through the view's channel moves the epoch the next record must carry
    view.request_stop(0, REQUESTER_R1D, t_ns)
    assert view.stop_epoch() == ctl.core.stop_epoch() > 0


def test_a_fresh_posix_segment_has_no_state_yet() -> None:
    import os

    name = f"/r1mc3.core.{os.getpid()}"
    view = _mod().SegmentView.create(name, True)
    try:
        assert view.state() is None and view.mirror() is None
        assert view.stop_epoch() == 0
        rec = {"cmd_id": 1, "group": 0, "source": 0, "n_joints": 5, "q": [0.0] * 5}
        assert view.push_upper(rec) == 0
    finally:
        view.close()


def test_r1hands_writers_on_a_segment_view() -> None:
    """r1hands' side (single writer): the contact summary the trajectory guards read and the hands' pose + payloads
    the core's arm model uses (session 5) - written through a SegmentView, read back as the core reads them."""
    import os

    from r1mc3.log import records as R

    view = _mod().SegmentView.create(f"/r1mc3.coreh.{os.getpid()}", True)
    try:
        assert view.hands_contact() is None and view.hands_pose() is None
        assert view.write_hands_contact(0b101, 0b10, stalled_left=1, grip_right=3, t_ns=123) == 1
        assert view.write_hands_contact(0b111, 0, t_ns=456) == 2
        writes, raw = view.hands_contact()
        rec = np.frombuffer(raw, dtype=R.HANDS_CONTACT)[0]
        assert writes == 2 and int(rec["seq"]) == 2 and int(rec["t_ns"]) == 456
        assert list(rec["pad_contact"]) == [0b111, 0] and int(rec["valid"]) == 1
        q = [0.1 * k for k in range(7)]
        assert view.write_hands_pose(q, q[::-1], [0.5, 0.0], 789) == 1
        writes, raw = view.hands_pose()
        assert writes == 1 and len(raw) == 152
        arr = np.frombuffer(raw[16 : 16 + 16 * 8], dtype="<f8")  # q[2][7] then payload_kg[2]
        np.testing.assert_allclose(arr[:7], q)
        np.testing.assert_allclose(arr[7:14], q[::-1])
        np.testing.assert_allclose(arr[14:16], [0.5, 0.0])
        with pytest.raises(ValueError):
            view.write_hands_pose(q[:6], q, [0.0, 0.0], 0)
    finally:
        view.close()
