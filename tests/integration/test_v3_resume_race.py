"""#29's open race (session 9 FIRST 0), made DETERMINISTIC: a whole-machine stop that lands between r1rt's last checks
and its DDS write.

Session 8's live CI caught it once in ~9 loaded runs (S8.md "#29's open race"): after a 2 s stop of r1rt AND r1guard,
ONE r1rt packet computed BEFORE the stop reached the board (its pre-stop target at full stiffness: a 33.6 N m hip step
on the sagged hanging robot) and the board kept it until a resume blend replaced it - because during the resume wait
(the first live lowstate, at least R1RT_RESUME_SETTLE_NS = 10 ms after the stop) the core and the guard publish NOTHING.

Here r1rt's test hook (`R1RT_TEST_STALL_BEFORE_WRITE`, rt/apps/r1rt_main.cpp) stops r1rt itself exactly there (after its
5 ms stale check and the token check, before `bus.write`); the test stops r1guard within ~1 ms of it (a whole-machine
stop), waits 2 s (the board's 0.385 s hold ran out: the hanging limbs sag in its damping), and continues both (the
guard first, as test_v3_resume_processes). The stale packet goes out EVERY run. What the fix bounds is how long it acts:

* `stale_ms` = how long the stale packet stayed the board's command (the board's clock: the next applied packet);
* the PHYSICAL effect on the robot (the bridge's ground truth sampled every ~1 ms): the largest joint speed on the legs,
  waist and arms in the 0.1 s after the resume, and the largest motion back toward the old target in its first 50 ms;
* `step_nm` = the largest 1 ms change of the applied torque (the stale packet's step itself: it stays ~30 N m - no
  user-space check can stop a packet already inside the write; the fix makes it act for ~1-2 ms instead of >= 10).
"""

from __future__ import annotations

import os
import signal
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from r1mc3 import dds

pytestmark = [pytest.mark.needs_dds]
if not dds.available():
    pytest.skip("CycloneDDS's Python binding is not installed (uv sync --extra dds)", allow_module_level=True)

from r1sim import stack  # noqa: E402
from r1sim.dds_bridge import DdsBridge  # noqa: E402
from r1sim.driver import DriverConfig  # noqa: E402

from tests.integration.test_v3_resume_processes import BODY, _hw1_world, _Sampler  # noqa: E402

STOP_S = 2.0
# The backlog: after a stop the DDS receive path first drains the lowstate samples queued in the socket during it (each
# stamped fresh, its board tick old: not live). On the Mac's loopback default (~768 KB) that drain took 16-34 ms (the
# guard's resume_wait_ms_last, macOS under load) - longer than the old 10 ms floor, so the floor's removal shows only
# with a bounded backlog: RCVBUF pins r1rt / r1guard / r1d's socket receive buffer (the proposal for the robot, whose
# 4 MB buffer holds ~1-2 s of its 1052/s stream: core_RACE_NUMBERS.md).
RCVBUF = "64kB"
# What the numbers say (64 kB, macOS; core_RACE_NUMBERS.md): with ~3 agents running, base vs fix separated cleanly
# (stale 7 / 7 / 7 vs 3 / 3 / 4 ms on the board's clock; joint speed ~1.1 vs ~0.55 rad/s; kick ~0.017 vs ~0.010
# rad); with 11 agents the wall-clock window is the drain PLUS the guard's own scheduling delays (a normal-priority
# process on macOS: 4 samples seen over 4-16 ms) and the two overlap (fix 4 / 9 / 16 ms, base 10 / 17 / 37 ms). So the
# load-independent property is asserted - the resuming guard's wait ends at its FIRST live sample (base: it skips the
# live samples that arrive inside the 10 ms floor) - and the joint motion only as a sanity bound. The deterministic
# fail-before / pass-after timing is test_v3_resume_wait.py (lockstep: 10 -> 0 ms).
# The 50 ms expiry: the bridge's real-time sim can lag the wall clock by more than the ResumeClock's 20 ms slack over a
# 2 s stop (its board tick then never "catches up": ~1 run in 5 here) - a sim artefact (the robot's tick is a steady
# clock); such a run ends by the expiry (resume_wait_ms_last >= 50) and is allowed once in 3.
WAIT_EXPIRED_MS = 49.0
KICK_SANITY_MAX = 0.03  # rad, the median: the default buffer's 16-34 ms drain gave 0.02-0.06


def loopback_uri_rcvbuf(size: str) -> str:
    """The loopback configuration with the socket receive buffer pinned to `size` (e.g. "64kB"): after a stop the DDS
    receive path drains at most what that buffer held (the samples of the stop's start; newer ones were dropped)."""
    from r1mc3.dds.rpc import LOOPBACK_URI

    tag = f'<Internal><SocketReceiveBufferSize min="{size}" max="{size}"/></Internal>'
    return LOOPBACK_URI.replace("</General>", "</General>" + tag, 1)


def _stopped(pid: int, timeout_s: float) -> bool:
    """True once the child `pid` is stopped (r1rt's own raise(SIGSTOP)); os.waitpid(WUNTRACED) reports it at once."""
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        try:
            p, status = os.waitpid(pid, os.WUNTRACED | os.WNOHANG)
        except ChildProcessError:
            return False
        if p == pid and os.WIFSTOPPED(status):
            return True
        time.sleep(0.0002)
    return False


def race_run(
    tmp_path: Path,
    domain: int,
    stop_s: float = STOP_S,
    rcvbuf: str | None = None,
    *,
    whole: bool = True,
    uri: str | None = None,
) -> dict[str, Any]:
    """ON_STAND from the stand lock on the hanging robot (the identified board, HW1_STAND, the HW1 friction), then the
    hooked stop of `stop_s`: the whole machine (r1guard stopped too) or, `whole=False`, the board-hold test's case
    (r1guard observe-only, r1rt alone stopped: its own resume ends the stale packet). Returns the numbers above + the
    resume counters."""
    trigger = tmp_path / "stall_before_write"
    br = DdsBridge(domain=domain, scene="none", stance="hanging", builtin="stand_lock", w=_hw1_world())
    assert br.sim.driver == DriverConfig.real()  # the board: holds 0.385 s, then damps at kd 12.5 (B6)
    br.start()
    try:
        env = {"R1RT_TEST_STALL_BEFORE_WRITE": str(trigger)}
        st = stack.ProcStack(domain=domain, workdir=tmp_path, bridge=None, env=env)
        if rcvbuf is not None:  # r1rt / r1guard / r1d with a bounded DDS socket receive buffer (the queued backlog)
            st.env["CYCLONEDDS_URI"] = loopback_uri_rcvbuf(rcvbuf)
        if uri is not None:  # the processes' whole CycloneDDS configuration (e.g. the deploy's shared file + overlay)
            st.env["CYCLONEDDS_URI"] = uri
        with st:
            st.wait_mode("monitor", 15)
            time.sleep(0.5)
            r = st.cli("on-stand")
            assert r["ok"], r
            st.wait_mode("on_stand", 15)
            time.sleep(1.5)
            if not whole:  # the board-hold test: the guard observe-only (the user's sudo at HW1: r1guard-observe)
                assert st.stop_guard() == 0
                st.start_guard("--observe-only")
                time.sleep(0.5)
            assert st.rt is not None and st.guard is not None
            rt_pid, guard_pid = st.rt.p.pid, st.guard.p.pid
            with _Sampler(br) as smp:
                time.sleep(0.3)
                trigger.write_text("1")
                assert _stopped(rt_pid, 5.0), "r1rt did not stop itself at the hook"
                if whole:
                    os.kill(guard_pid, signal.SIGSTOP)  # the whole machine: the guard within ~1 ms of r1rt
                wall_stop = time.monotonic()
                t_sim_stop = float(br.sim.read_state().t)
                time.sleep(stop_s)
                if whole:
                    os.kill(guard_pid, signal.SIGCONT)
                os.kill(rt_pid, signal.SIGCONT)
                wall_cont = time.monotonic()
                # 2.5 s: past the drain limit (0.4 s) + the 1 s blend + one status line (1 Hz, late under load)
                time.sleep(2.5)
            rt_line = st.rt.last_status() or {}
            guard_line = st.guard.last_status() or {}
            hook = st.rt.events("test_stall_before_write")
            applied = [a for a in list(br.applied) if a.wall > wall_stop - 0.5]
            labels = {h: v["label"] for h, v in br.writers().items()}
    finally:
        br.stop()
    assert hook, "the hook never fired"
    t = np.array([x[0] for x in smp.rows])
    q = np.array([x[1] for x in smp.rows])
    tau = np.array([x[2] for x in smp.rows])
    dq = np.vstack([np.zeros((1, q.shape[1])), np.diff(q, axis=0) / np.maximum(np.diff(t), 1e-6)[:, None]])
    after = [a for a in applied if a.wall >= wall_cont - 0.001]
    assert len(after) >= 2, "nothing published after the stop"
    t_first = after[0].sim_t  # the stale packet (the hooked write right after SIGCONT)
    stale_ms = (after[1].wall - after[0].wall) * 1e3  # wall clock: the bridge's sim can lag it by a few ms under load
    stale_sim_ms = (after[1].sim_t - t_first) * 1e3
    pre = t < t_sim_stop
    q_target = q[pre][-1, BODY]  # ~ the pre-stop command (the hold settled within ~0.016 rad: HW1 S1)
    k0 = int(np.flatnonzero(t <= t_first)[-1])
    q0 = q[k0, BODY]
    w100 = (t >= t_first) & (t <= t_first + 0.1)
    w50 = (t >= t_first) & (t <= t_first + 0.05)
    toward = np.sign(q_target - q0)  # + = back toward the old target (the stale packet's pull)
    kick = float(np.max((q[w50][:, BODY] - q0) * toward)) if w50.any() else float("nan")
    win = (t >= t_first - 0.003) & (t <= t_first + 0.3)
    dtau = np.abs(np.diff(tau[:, BODY], axis=0))[win[1:]]
    return {
        "stale_ms": stale_ms,
        "stale_sim_ms": stale_sim_ms,
        "dq_peak": float(np.max(np.abs(dq[w100][:, BODY]))) if w100.any() else float("nan"),
        "kick_rad": kick,
        "step_nm": float(dtau.max()) if dtau.size else float("nan"),
        "sag_max": float(np.max(np.abs(q0 - q_target))),
        "first_writer": labels.get(after[0].writer, "?"),
        "next_writer": labels.get(after[1].writer, "?"),
        "writers_after": sorted({labels.get(a.writer, "?") for a in after}),
        "min_leg_kp": min(a.min_leg_kp for a in applied),
        "sample_dt_ms": float(np.median(np.diff(t[w100])) * 1e3) if w100.sum() > 2 else None,
        "rt": {k: rt_line.get(k) for k in ("stops", "resume_blends", "resume_waits_expired", "stale_skips")},
        "guard": {
            k: guard_line.get(k)
            for k in (
                "takeovers",
                "resume_takeovers",
                "resume_blends",
                "act_mode",
                "resume_wait_ms_last",
                "resume_wait_samples_last",
                "resume_wait_old_samples_last",
                "resume_drain_waits",
                "resume_uncaught",
            )
        },
    }


@pytest.mark.realtime
@pytest.mark.slow
def test_a_stale_pre_stop_packet_acts_only_until_the_first_live_sample(tmp_path: Path) -> None:
    """The deterministic race, 3 runs with a bounded backlog (RCVBUF): every run the stale packet reaches the board (the
    hook guarantees it; its ~30 N m step is unavoidable - it is already inside the write) and stays its command until
    the guard's first command after its resume wait; that wait must end at the guard's FIRST live sample (or, in one run
    at most, the 50 ms expiry of the sim's tick-lag artefact); never limp, one hand-over (r1rt's stale packet, then the
    guard). Base (the 10 ms floor): a run whose drain is shorter than 10 ms skips the live samples inside the floor."""
    if not stack.available():
        pytest.skip("needs the DDS build of r1rt / r1guard (scripts/ci.sh dds)")
    runs = [race_run(tmp_path / f"r{i}", 41 + i, rcvbuf=RCVBUF) for i in range(3)]
    for r in runs:
        print(f"\n[ info ] hooked whole-machine 2 s, {RCVBUF} (macOS, under load): {r}")
    expired = 0
    for r in runs:
        g = r["guard"]
        assert r["first_writer"] == "r1rt" and r["next_writer"] == "r1guard", r  # the race happened; the guard ended it
        assert r["sag_max"] > 0.01, r  # the board let go during the stop: there was something to kick back
        assert r["min_leg_kp"] > 0.0, r  # never limp
        assert g["resume_takeovers"] == 1 and g["resume_blends"] == 1, r
        if g["resume_wait_ms_last"] >= WAIT_EXPIRED_MS:
            expired += 1  # the sim's tick lag (above)
            continue
        live_seen = g["resume_wait_samples_last"] - g["resume_wait_old_samples_last"]
        assert live_seen <= 1, r  # the wait ended at the first live sample (base: more, inside its 10 ms floor)
    assert expired <= 1, runs
    med = statistics.median(r["kick_rad"] for r in runs)
    print(f"[ info ] stale_ms {[round(r['stale_ms'], 1) for r in runs]}, kick median {med:.4f} rad")
    assert med <= KICK_SANITY_MAX, runs


def robot_like_uri(overlay: str | None) -> str:
    """The deploy's pattern on the loopback: the shared file's 4 MB receive buffer, then (optionally) a real-time
    overlay from deploy/config/ (r1rt / r1guard units: CYCLONEDDS_URI = the shared file, the overlay)."""
    from r1mc3.dds.rpc import LOOPBACK_URI

    shared = LOOPBACK_URI.replace(
        "</General>", '</General><Internal><SocketReceiveBufferSize min="4MB"/></Internal>', 1
    )
    if overlay is None:
        return shared
    return shared + ",file://" + str(Path(__file__).resolve().parents[2] / "deploy" / "config" / overlay)


@pytest.mark.realtime
@pytest.mark.slow
def test_the_deploys_rt_overlay_shortens_the_backlog_the_guard_drains(tmp_path: Path) -> None:
    """(e), the user's decision 11 Oct 2026: deploy/config/cyclonedds_rt.xml (256 kB) read after a 4 MB shared file -
    CycloneDDS takes the overlay (the URI list: a later setting overrides) and the hooked race's backlog shrinks. Paired
    runs under the same load (macOS, 11 agents, 21:25): the 4 MB shared file alone 38 / 35 old samples drained by the
    guard over 40 / 37 ms; + the overlay 10 / 17 over 19 / 48 ms (the wall time carries the guard's own scheduling);
    the test, 21:30: 31.5 vs 14 (medians)."""
    if not stack.available():
        pytest.skip("needs the DDS build of r1rt / r1guard (scripts/ci.sh dds)")
    order = [None, "cyclonedds_rt.xml", None, "cyclonedds_rt.xml"]
    runs = [race_run(tmp_path / f"o{i}", 45 + i, uri=robot_like_uri(ov)) for i, ov in enumerate(order)]
    for ov, r in zip(order, runs, strict=True):
        print(f"\n[ info ] {ov or '4 MB shared only'}: {r}")
    drained = [(ov, r["guard"]["resume_wait_old_samples_last"]) for ov, r in zip(order, runs, strict=True)]
    # the smaller of each pair: the sim's tick-lag artefact (see WAIT_EXPIRED_MS) only ever INFLATES a run's count
    old = {ov: min(n for o, n in drained if o == ov) for ov in (None, "cyclonedds_rt.xml")}
    print(f"[ info ] old samples drained (the smaller of each pair): {old}")
    for r in runs:
        assert r["first_writer"] == "r1rt" and r["min_leg_kp"] > 0.0, r
        if r["guard"]["resume_wait_ms_last"] < 390.0:  # inside the drain limit: never a blend from a backlog pose
            assert r["guard"]["resume_uncaught"] == 0, r
    assert old["cyclonedds_rt.xml"] < 0.7 * old[None], old
