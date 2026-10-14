"""The session-6 words with REAL processes on a HANGING robot (DESIGN §14.1 items 5-8): the DDS-bridge simulator on its
stand (hang's `stance="hanging"`), r1rt + r1guard + r1d + r1log as processes, the `r1` CLI per word:

    r1 on-stand                           -> ON_STAND; r1 status says the guard is alive and ARMED
    r1 board-hold --pause-ms 100 --yes    -> refused (the guard is armed)
    r1 guard-takeover --stop-s 1.5 --yes  -> the guard takes the token; after SIGCONT r1rt is silent (NOT_OWNER)
    r1 status                             -> the guard ACTIVE, taken over from ON_STAND; r1rt SILENT
    r1 reclaim                            -> r1rt owns again; the guard re-arms by itself
    (the guard stopped) r1 board-hold --pause-ms 100 --yes -> r1rt paused 100 ms, r1d's own lowstate recording, the
                                     token stays r1rt; (then) the guard started again -> armed
    r1 shutdown                   -> refused in ON_STAND; r1 damp; r1 shutdown -> r1log finished, the logs saved +
                                     verified, "safe to power off now"

Wall-clock on a loaded Mac without real-time scheduling: the asserted facts are the sequence, the owners and the end
states; the bridge's applied-command log is the board's view (one writer at a time, never limp before the damp)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.needs_dds, pytest.mark.realtime, pytest.mark.slow]

from r1sim import stack  # noqa: E402

if not stack.available():
    pytest.skip("needs the dds extra and the DDS build of r1rt / r1guard (scripts/ci.sh dds)", allow_module_level=True)

DOMAIN = 61


def _wait(st: stack.ProcStack, pred: Callable[[dict[str, Any]], bool], timeout_s: float) -> dict[str, Any]:
    end = time.monotonic() + timeout_s
    s: dict[str, Any] = {}
    while time.monotonic() < end:
        s = st.status()
        if pred(s):
            return s
        time.sleep(0.2)
    raise AssertionError(f"condition not reached in {timeout_s} s: {s.get('data', {}).get('summary')}")


def test_takeover_reclaim_board_hold_shutdown_with_real_processes(tmp_path: Path) -> None:
    with stack.ProcStack(
        domain=DOMAIN, workdir=tmp_path, bridge={"scene": "none", "stance": "hanging"}, r1log=True
    ) as st:
        st.wait_mode("monitor", 15)
        on = st.cli("on-stand")
        assert on["ok"] and on["_exit"] == 0, on
        s = st.wait_mode("on_stand", 15)
        g = s["data"]["guard"]
        assert g["alive"] and g["armed"] is True and not g["active"], s["data"]["summary"]

        # r1hands' contact summary reaches the segment (r1d writes it at its tick; the bridge's hands publish state)
        import struct

        from r1mc3.log._native import native

        view = native().SegmentView.attach(st.shm)
        writes, raw = view.hands_contact()
        _seq, t_ns, _pad_l, _pad_r, *_rest, valid, _pad = struct.unpack_from("<Qq2I2B2B2BBB", bytes(raw))
        assert int(writes) > 10 and valid == 1 and t_ns > 0
        assert s["data"]["hands_contact"]["writes"] > 10

        # the board-hold test must refuse while the guard is armed
        r = st.cli("board-hold", "--pause-ms", "100", "--yes")
        assert r["code"] == "WRONG_MODE" and r["details"]["failed"] == ["guard OFF"] and r["_exit"] == 1

        # the guard-takeover test: SIGSTOP r1rt in ON_STAND, the guard holds; after SIGCONT r1rt silent
        t_mark = time.monotonic()
        r = st.cli("guard-takeover", "--stop-s", "1.5", "--yes")
        assert r["ok"], r
        d = r["data"]
        assert d["takeover"]["token_seen_r1guard_ms_after_stop"] is not None
        assert d["takeover"]["token_seen_r1guard_ms_after_stop"] < 100.0  # r1d's 1 ms poll; the guard ~10-12 ms
        assert d["after_sigcont"]["r1rt_silent"] and d["after_sigcont"]["token"] == "r1guard"
        assert 1490.0 <= d["pause"]["actual_ms"] < 1700.0
        assert d["lowstate"]["recorded"] and d["lowstate"]["during"]["samples"] > 100  # the board kept publishing
        assert Path(d["files"]["lowstate"]).exists() and Path(d["files"]["manifest"]).exists()
        s = st.status()["data"]
        assert s["guard"]["active"] and s["guard"]["taken_over_from"] == "on_stand", s["summary"]
        assert any("SILENT" in w for w in s["summary"])
        after = [a for a in list(st.bridge.applied) if a.wall >= t_mark and a.crc_ok]
        assert after and min(a.min_leg_kp for a in after) > 20.0  # never limp through the stall (the hold gains)

        # reclaim: r1rt owns again; the guard re-arms by itself
        rc = st.cli("reclaim")
        assert rc["ok"], rc
        s = _wait(st, lambda x: x["data"]["token"]["owner"] == "r1rt" and x["data"]["core"]["mode"] == "on_stand", 10)
        s = _wait(st, lambda x: x["data"]["guard"]["armed"] is True and not x["data"]["guard"]["active"], 10)
        again = st.cli("reclaim")
        assert again["code"] == "WRONG_MODE" and again["_exit"] == 1  # nothing to reclaim now

        # the board-hold test: the guard OFF (stopped), LowCmd paused 100 ms, back ON + armed
        assert st.stop_guard() == 0
        s = _wait(st, lambda x: x["data"]["guard"] is None or not x["data"]["guard"]["alive"], 5)
        r = st.cli("board-hold", "--pause-ms", "100", "--yes")
        assert r["ok"], r
        d = r["data"]
        assert 99.0 <= d["pause"]["actual_ms"] < 140.0
        assert d["after"]["owner"] == "r1rt" and "warning" not in d
        assert d["lowstate"]["during"]["samples"] >= 20  # ~500 Hz from the bridge while r1rt was stopped
        man = json.loads(Path(d["files"]["manifest"]).read_text())
        assert man["schema"] == "r1mc3.hw1_board_hold/1"
        st.start_guard()
        _wait(st, lambda x: x["data"]["guard"] is not None and x["data"]["guard"]["armed"] is True, 10)

        # the session end: shutdown refused while ON_STAND, then damp -> shutdown saves the logs
        sd = st.cli("shutdown")
        assert sd["code"] == "WRONG_MODE" and not sd["moved"]
        dp = st.cli("damp")
        assert dp["ok"], dp
        st.wait_mode("damp", 10)
        sd = st.cli("shutdown")
        assert sd["ok"], sd
        d = sd["data"]
        assert d["message"] == "safe to power off now" and d["saved"]["verified"]
        assert d["r1log"]["found"] and all(v == "exited" for v in d["r1log"]["stop"].values())
        assert d["saved"]["r1log_files"] >= 1 and d["saved"]["record_files"] >= 2  # both tests' files at least
        folder = Path(d["saved"]["folder"])
        man = json.loads((folder / "manifest.json").read_text())
        assert all(e["verified"] for e in man["files"])
        journal = [json.loads(x) for x in (folder / "r1d_journal.jsonl").read_text().splitlines()]
        assert [
            e["request"]["op"] for e in journal if e["request"]["op"] in ("guard_takeover.stall", "reclaim", "damp")
        ] == [
            "guard_takeover.stall",
            "reclaim",
            "reclaim",
            "damp",
        ]


def test_reclaim_refused_after_a_balancing_takeover(tmp_path: Path) -> None:
    """The guard took over from BALANCE (the 9 Oct stall's case): `r1 reclaim` is refused by the core (refusal "mode":
    a balancing hand-back is not built, DESIGN §4.4) and the guard keeps balancing; `r1 guard-takeover` refuses outside
    ON_STAND."""
    import os
    import signal

    with stack.ProcStack(domain=DOMAIN + 1, workdir=tmp_path, bridge={"scene": "none", "builtin": "stand"}) as st:
        st.wait_mode("monitor", 15)
        assert st.cli("takeover")["ok"]
        st.wait_mode("balance", 15)
        r = st.cli("guard-takeover", "--stop-s", "1", "--yes")
        assert r["code"] == "WRONG_MODE" and "ON_STAND" in r["details"]["failed"]
        assert st.rt is not None
        os.kill(st.rt.pid, signal.SIGSTOP)
        time.sleep(0.5)
        os.kill(st.rt.pid, signal.SIGCONT)
        s = _wait(st, lambda x: x["data"]["guard"]["active"] and x["data"]["core"]["alive"], 5)
        assert s["data"]["guard"]["taken_over_from"] == "balance", s["data"]["summary"]
        rc = st.cli("reclaim")
        assert rc["code"] == "WRONG_MODE" and rc["details"]["refusal_name"] == "mode", rc
        assert "balancing hand-back is not built" in rc["details"]["reason"]
        time.sleep(0.5)
        s = st.status()["data"]
        assert s["token"]["owner"] == "r1guard" and st.bridge.tilt_deg() < 5.0  # the guard still balances


def _static_pose(st: stack.ProcStack, group: str, q: list[float]) -> None:
    r = st.cli(
        "record", "run", "static_pose", f"group={group}", f"q={json.dumps(q)}", "move_s=1.5", "hold_s=1", "--yes"
    )
    assert r["ok"], r
    end = time.monotonic() + 15
    while time.monotonic() < end and st.cli("record", "list")["data"]["active"] is not None:
        time.sleep(0.2)


def test_stand_freeze_words_on_the_hanging_robot(tmp_path: Path) -> None:
    """HW1 S10 (U3) rehearsed: ON_STAND -> `r1 stand-freeze` (dry run: ramp 0.4 s from r1rt's config + the far check)
    -> a knee 0.25 rad off the stand pose: refused (LIMIT, far) -> the legs 0.10 rad off (ADDITIONS S5 step 1) ->
    `--yes` -> STAND_FREEZE (from the stand) -> `r1 stand-hold` -> ON_STAND -> `r1 resume`."""
    cfg_path = tmp_path / "r1rt.cfg"  # ProcStack writes r1rt's config there; r1d reads the same file
    with stack.ProcStack(
        domain=DOMAIN + 2,
        workdir=tmp_path,
        bridge={"scene": "none", "stance": "hanging"},
        env={"R1RT_CONFIG": str(cfg_path)},
    ) as st:
        st.wait_mode("monitor", 15)
        assert st.cli("on-stand")["ok"]
        st.wait_mode("on_stand", 15)
        dry = st.cli("stand-freeze")
        assert dry["ok"] and dry["data"]["dry_run"] and dry["data"]["plan"]["ramp_s"] == 0.4, dry
        line = next(ln for ln in cfg_path.read_text().splitlines() if ln.startswith("stand_q"))
        stand = [float(x) for x in line.split("=")[1].split()]
        # session 8: with HW1's measured leg friction (0.4-0.6 N m, was issue #51's 2.5) the legs hang nearly straight
        # in zero torque (knees ~0.0 vs 0.38 before), ~0.45 rad off the stand pose - the right leg to the stand pose
        # first, so the far check sees the one joint this step moves
        _static_pose(st, "leg_right", stand[6:12])
        q = stand[0:6]
        q[3] += 0.25  # the left knee 0.25 rad off the stand pose
        _static_pose(st, "leg_left", q)
        dry = st.cli("stand-freeze")
        assert dry["data"]["far_check"]["ok"] is False and dry["data"]["far_check"]["worst_joint"] == "left_knee"
        far = st.cli("stand-freeze", "--yes")
        assert far["code"] == "LIMIT" and not far["moved"] and "0.15 rad" in far["details"]["reason"], far
        assert st.mode() == "on_stand"
        for group, cols in (("leg_left", range(0, 6)), ("leg_right", range(6, 12))):
            q = [stand[j] for j in cols]
            q[0] += 0.10  # hip pitch and knee 0.10 rad off the stand pose (ADDITIONS S5 step 1)
            q[3] -= 0.10
            _static_pose(st, group, q)
        dry = st.cli("stand-freeze")
        assert dry["data"]["far_check"]["ok"] is True and dry["data"]["far_check"]["worst_dev_rad"] <= 0.12, dry
        fr = st.cli("stand-freeze", "--yes")
        assert fr["ok"] and fr["moved"] and fr["data"]["mode"] == "stand_freeze", fr
        s = st.status()["data"]
        assert s["freeze_from_stand"] and any("STAND_FREEZE (from the stand)" in w for w in s["summary"]), s["summary"]
        time.sleep(1.0)
        assert st.bridge.hang_state()["foot_z_min"] > 0.03  # still hanging, feet off the floor
        hd = st.cli("stand-hold")
        assert hd["ok"], hd
        st.wait_mode("on_stand", 5)
        assert st.cli("stand-hold")["code"] == "WRONG_MODE"  # only from the stand's freeze
        assert st.cli("resume")["ok"]


def test_stand_session_refuses_takeover_with_real_processes(tmp_path: Path) -> None:
    """r1rt --stand-session (HW1: balance refused for r1rt's lifetime) on the hanging robot with Unitree's STAND LOCK:
    `r1 takeover` is refused by r1d before anything is sent - the built-in is NOT released and keeps publishing, the
    core stays in MONITOR; `r1 status` says so; `r1 on-stand` still works."""
    with stack.ProcStack(
        domain=DOMAIN + 3,
        workdir=tmp_path,
        bridge={"scene": "none", "stance": "hanging", "builtin": "stand_lock"},
        rt_args=["--stand-session"],
    ) as st:
        st.wait_mode("monitor", 15)
        s = st.status()["data"]
        assert s["core"]["stand_session"] is True, s["core"]
        assert "stand session: balance REFUSED (r1rt --stand-session)" in s["summary"]
        n0 = st.bridge.writers()[st.bridge.builtin_handle]["n"]
        tk = st.cli("takeover")
        assert tk["code"] == "WRONG_MODE" and not tk["moved"] and tk["_exit"] == 1, tk
        assert "--stand-session" in tk["details"]["reason"]
        time.sleep(0.3)
        assert st.bridge.builtin_active  # no ReleaseMode reached the motion switcher
        assert st.bridge.writers()[st.bridge.builtin_handle]["n"] > n0 + 50  # Unitree's command still flowing
        assert st.mode() == "monitor" and st.status()["data"]["token"]["owner"] == "none"
        on = st.cli("on-stand")
        assert on["ok"], on
        st.wait_mode("on_stand", 15)
