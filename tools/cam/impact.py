"""The camera impact test (R1CAM_PLAN.md last section; the user's warning, 11 Oct: "the camera thing ... messed with
some of the other metrics and speed and processing"): the RT numbers with the camera OFF, then WORKING (left + depth as
r1d's camera.observe asks while working: one observe every 3 s = a trial's pace), WORKING FAST (an observe every 0.5 s:
the depth almost always connected = the worst working case), a CALIBRATION recording (left + right + depth, as the
ChArUco recording), then OFF again (drift), each for `--seconds`.

Session 12 (HW2 day 1's C1: the state rate 1052 -> 997 /s and the state age p99 1.0 -> 2.9 ms while all three streams
ran; the user's decision: fewer streams - left + depth while working, all three only for calibration recordings;
r1mc3.cam.streams has the traffic per stream: the depth = ~87 %): the windows follow that rule (day 1's `stream` /
`record` windows streamed all three continuously), and the lateness check reads each window's OWN ticks - day 1's
check read r1rt's cumulative `max_us` (`max_ms_so_far`: T6's deliberate 3 s pause, minutes earlier) and failed every
window, the OFF baseline included.

    # the sim / the Mac (the DDS-bridge stack: the hanging robot in ON_STAND, r1rt + r1guard + r1d as processes;
    # a stand-in robot sends the real depth stream (~39 Mbit/s) over the loopback; r1cam as its own process)
    uv run python tools/cam/impact.py sim --domain 100 --seconds 20 --out out/cam_impact_sim.json [--burn-ms 15]
    # session 11: --camera rendered = the SIMULATED head camera instead (r1sim.bridge_camera: r1cam's sim process
    # renders the bridge robot's left + right + depth at the robot's rates while wanted; no network stand-in) - the
    # numbers include the renderer's CPU on this Mac, i.e. the sim harness's cost, not the robot's camera
    uv run python tools/cam/impact.py sim --camera rendered --domain 100 --seconds 20

    # the robot (HW2 day 2's C1 re-run, BEFORE any trial; the stack already running in ON_STAND, r1cam NOT started;
    # the operator starts / stops r1cam when told - `sudo systemctl start r1cam` / `stop`)
    python tools/cam/impact.py robot --shm /r1mc3.rt --seconds 60 --out /var/lib/r1mc3/records/cam_impact.json

Per window: r1rt's ticks, skipped deadlines (minus deliberate stops), overruns, lateness (ticks late > 0.5 / 1 / 2 ms
from the cumulative histogram's difference; the window's max = the new max when the window set one, else bounded by
the highest histogram bucket the window's ticks reached), the lowstate rate at r1rt (lowstate_rx), the lowstate age at
r1rt's tick (SegmentView.state() sampled at ~100 Hz: p50 / p99 / max), r1guard's lateness and state age, the CPU (per
CPU from /proc/stat on Linux; per process elsewhere), the link traffic (eth10's counters on the robot; the stand-in's
depth bytes in the sim), r1cam's frames, drops and depth connects. The verdict (robot only; the proposed pass rule, the
user's OK): r1rt 0 skipped deadlines, no tick of the window later than 0.5 ms (the histogram's edge) and a new maximum
<= 0.4 ms (HW1's range), lowstate age p99 <= 2 ms and rate >= 1000/s - in the WORKING windows (`work`, `work_fast`;
the calibration window is reported, not judged: calibration recordings are not trials); on the Mac the camera
windows are compared with the OFF baseline (no RT scheduling: the numbers are relative, "macOS, under load").
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "python", REPO / "sim"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from r1mc3.cam.service import request  # noqa: E402

CONDITIONS = ("off", "work", "work_fast", "calib", "off_again")
JUDGED = ("work", "work_fast")  # the working windows (the user's rule: left + depth while working)
PASS = {"skipped": 0, "late_max_ms": 0.4, "late_bucket_ms": 0.5, "age_p99_ms": 2.0, "rate_min_hz": 1000.0}  # proposed
EDGES_US = ("<50", "<100", "<250", "<500", "<1000", "<2000", "<5000", "<10000", "<20000", ">=20000")
WORK_PERIOD_S = 3.0  # one observe every 3 s: a trial's pace (a model turn / a teleop click needs one frame)
FAST_PERIOD_S = 0.5  # back to back: the depth stream almost always connected (the worst working case)
WORK_OBSERVE = {"op": "observe", "depth": True, "wait_s": 4.0}  # left + depth = r1d's camera.observe while working
CALIB_RECORD = {"op": "record_start", "kind": "calibration", "label": "impact", "streams": ["left", "right", "depth"]}


def _late_counts(h0: dict[str, Any] | None, h1: dict[str, Any] | None) -> dict[str, Any]:
    """Ticks late beyond 0.5 / 1 / 2 ms in the window (the cumulative histograms' difference) + the max if it grew."""
    if not h0 or not h1:
        return {}
    a, b = h0.get("hist_us", {}), h1.get("hist_us", {})
    keys = list(b)
    d = {k: int(b.get(k, 0)) - int(a.get(k, 0)) for k in keys}

    def over(us: int) -> int:
        out = 0
        for k, v in d.items():
            lo = k.lstrip("<>=")
            if k.startswith(">=") or (lo.isdigit() and int(lo) > us):
                out += v
        return out

    grew = float(h1.get("max_us", 0.0)) > float(h0.get("max_us", 0.0))
    bound: float | None = None  # the upper edge of the highest bucket the window's ticks reached (ms; None: open)
    for k in EDGES_US:
        if d.get(k, 0) > 0:
            bound = float(k.lstrip("<")) / 1000.0 if k.startswith("<") else None
    return {
        "n": int(h1.get("n", 0)) - int(h0.get("n", 0)),
        "over_0p5ms": over(500),
        "over_1ms": over(1000),
        "over_2ms": over(2000),
        "max_ms": round(float(h1["max_us"]) / 1000.0, 3) if grew else None,
        "window_max_bound_ms": round(float(h1["max_us"]) / 1000.0, 3) if grew else bound,
        "max_ms_so_far": round(float(h1.get("max_us", 0.0)) / 1000.0, 3),  # the process's whole life: NOT judged
    }


def _pct(v: list[float], q: float) -> float | None:
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


class AgeSampler(threading.Thread):
    """r1rt's lowstate age at its own tick (the segment's state mirror), sampled at ~rate_hz."""

    def __init__(self, view: Any, rate_hz: float = 100.0) -> None:
        super().__init__(daemon=True, name="age-sampler")
        self.view, self.period = view, 1.0 / rate_hz
        self.samples: list[tuple[float, float]] = []
        # session 13: NOT `_stop` - that name is threading.Thread's own method (join() calls it: the Event in its place
        # made join() raise TypeError, so the sampler was never joined and its SegmentView outlived the interpreter:
        # nanobind's "leaked 1 instances ... r1rt_py.SegmentView" + 36 functions at every exit, HW2 day 2's C1)
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            st = self.view.state()
            if st is not None:
                self.samples.append((time.monotonic(), float(st["lowstate_age_ns"]) * 1e-6))
            time.sleep(self.period)

    def window(self, t0: float, t1: float) -> dict[str, Any]:
        v = [a for t, a in self.samples if t0 <= t <= t1]
        return {"n": len(v), "p50_ms": _pct(v, 0.5), "p99_ms": _pct(v, 0.99), "max_ms": max(v) if v else None}

    def stop(self) -> None:
        """Stop sampling, wait for the thread, let go of the view (the caller closes its port: `close_port`)."""
        self._halt.set()
        if self.is_alive():
            self.join(2.0)
        self.view = None


def close_port(port: Any) -> None:
    """Close the r1rt segment view (detaches; r1rt keeps running) - before the interpreter exits, so nanobind finds no
    live r1rt_py object at its teardown (session 13: the 'leaked' warnings)."""
    view = getattr(port, "view", None)
    if view is not None and hasattr(view, "close") and not getattr(view, "closed", False):
        view.close()


def _cpu_snapshot(pids: dict[str, list[int]]) -> dict[str, Any]:
    """/proc/stat per CPU (Linux) + each process group's CPU seconds (ps; children included by the caller)."""
    snap: dict[str, Any] = {"t": time.monotonic(), "proc": {}}
    stat = Path("/proc/stat")
    if stat.exists():
        cpus = {}
        for line in stat.read_text().splitlines():
            if line.startswith("cpu") and line[3:4].isdigit():
                f = line.split()
                vals = [int(x) for x in f[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                cpus[f[0]] = (sum(vals), idle)
        snap["cpus"] = cpus
    for name, ps in pids.items():
        tot = 0.0
        for pid in ps:
            out = subprocess.run(["ps", "-o", "time=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
            if out:
                parts = [float(x) for x in out.replace("-", ":").split(":")]
                sec = 0.0
                for x in parts:
                    sec = sec * 60 + x
                tot += sec
        snap["proc"][name] = tot
    return snap


def _cpu_window(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    dt = b["t"] - a["t"]
    out: dict[str, Any] = {"proc_pct": {k: round(100.0 * (b["proc"].get(k, 0.0) - v) / dt, 1)
                                        for k, v in a["proc"].items()}}  # fmt: skip
    if "cpus" in a and "cpus" in b:
        out["cpu_busy_pct"] = {
            c: round(100.0 * (1 - (b["cpus"][c][1] - a["cpus"][c][1]) / max(1, b["cpus"][c][0] - a["cpus"][c][0])), 1)
            for c in a["cpus"]
            if c in b["cpus"]
        }
    return out


def _children(pid: int) -> list[int]:
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def _net_bytes(iface: str) -> int | None:
    p = Path(f"/sys/class/net/{iface}/statistics")
    if not p.exists():
        return None
    return int((p / "rx_bytes").read_text()) + int((p / "tx_bytes").read_text())


class Measure:
    """The windowed numbers from the status-line sources + the age sampler + CPU + link."""

    def __init__(
        self,
        rt_status: Callable[[], dict[str, Any] | None],
        guard_status: Callable[[], dict[str, Any] | None],
        ages: AgeSampler,
        pids: Callable[[], dict[str, list[int]]],
        link_bytes: Callable[[], int | None],
    ) -> None:
        self.rt_status, self.guard_status, self.ages, self.pids, self.link_bytes = (
            rt_status, guard_status, ages, pids, link_bytes)  # fmt: skip

    def window(self, seconds: float, during: Callable[[], None] | None = None, period_s: float = 1.0) -> dict[str, Any]:
        """`during` runs every `period_s` (a working window: one observe per period)."""
        time.sleep(1.2)  # a fresh status line on both sides
        r0, g0, l0 = self.rt_status(), self.guard_status(), self.link_bytes()
        c0 = _cpu_snapshot(self.pids())
        t0 = time.monotonic()
        t_end = t0 + seconds
        t_next = t0
        while time.monotonic() < t_end:
            if during is not None:
                during()
            t_next += period_s
            time.sleep(min(period_s, max(0.0, min(t_end, t_next) - time.monotonic())))
        time.sleep(1.2)
        r1, g1, l1 = self.rt_status(), self.guard_status(), self.link_bytes()
        c1 = _cpu_snapshot(self.pids())
        t1 = time.monotonic()
        out: dict[str, Any] = {"seconds": round(t1 - t0, 2)}
        if r0 and r1:
            dt = (int(r1["t_ns"]) - int(r0["t_ns"])) * 1e-9
            out["r1rt"] = {
                "mode": r1.get("mode"),
                "ticks": int(r1["ticks"]) - int(r0["ticks"]),
                "skipped_deadlines": (int(r1["skipped_deadlines"]) - int(r0["skipped_deadlines"]))
                - (int(r1.get("stop_skipped", 0)) - int(r0.get("stop_skipped", 0))),
                "overruns": int(r1.get("overruns", 0)) - int(r0.get("overruns", 0)),
                "lateness": _late_counts(r0.get("lateness"), r1.get("lateness")),
                "lowstate_rate_hz": round((int(r1["lowstate_rx"]) - int(r0["lowstate_rx"])) / dt, 1),
            }
        if g0 and g1:
            out["r1guard"] = {
                "lateness": _late_counts(g0.get("lateness"), g1.get("lateness")),
                "lowstate_age_ms_at_end": round(float(g1["lowstate_age_ns"]) * 1e-6, 3),
            }
        out["lowstate_age_at_r1rt"] = self.ages.window(t0, t1)
        out["cpu"] = _cpu_window(c0, c1)
        if l0 is not None and l1 is not None:
            out["link_mbit_s"] = round((l1 - l0) * 8 / 1e6 / (t1 - t0), 2)
        return out


def window_lateness_ok(late: dict[str, Any] | None) -> bool:
    """The window's OWN ticks (session 12): none later than 0.5 ms (the histogram's edge) and, when the window set a
    new maximum, that maximum <= 0.4 ms. Never r1rt's cumulative max (`max_ms_so_far`: day 1's T6 pause of 3 s)."""
    if not late or "over_0p5ms" not in late:
        return False
    new_max = late.get("max_ms")
    return int(late["over_0p5ms"]) == 0 and (new_max is None or float(new_max) <= PASS["late_max_ms"])


def verdict(win: dict[str, Any]) -> dict[str, Any]:
    r = win.get("r1rt", {})
    age = win.get("lowstate_age_at_r1rt", {}).get("p99_ms")
    checks = {
        "no_skipped_deadlines": r.get("skipped_deadlines") == PASS["skipped"],
        "lateness_max_le_0p4ms": window_lateness_ok(r.get("lateness")),
        "age_p99_le_2ms": age is not None and age <= PASS["age_p99_ms"],
        "rate_ge_1000": (r.get("lowstate_rate_hz") or 0) >= PASS["rate_min_hz"],
    }
    return {"checks": checks, "pass": all(checks.values())}


def camera_windows(
    m: Measure,
    cport: int,
    seconds: float,
    out: dict[str, Any],
    *,
    after: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    """The three camera windows (session 12): `work` (an observe of left + depth every WORK_PERIOD_S), `work_fast`
    (every FAST_PERIOD_S) and `calib` (a calibration recording of all three streams + the fast observes); `after(name,
    window)` adds the mode's own numbers. Observe answers are counted per window (`observes_ok`)."""
    for name, period in (("work", WORK_PERIOD_S), ("work_fast", FAST_PERIOD_S), ("calib", FAST_PERIOD_S)):
        oks: list[bool] = []

        def obs(oks: list[bool] = oks) -> None:
            try:
                oks.append(bool(request(cport, WORK_OBSERVE).get("ok")))
            except OSError:
                oks.append(False)

        rec = request(cport, CALIB_RECORD) if name == "calib" else None
        out["windows"][name] = w = m.window(seconds, obs, period_s=period)
        w["observes_ok"] = f"{sum(oks)} / {len(oks)}"
        w["observe_period_s"] = period
        if rec is not None:
            if rec.get("ok"):
                w["r1cam_recording"] = request(cport, {"op": "record_stop", "rec_id": rec["data"]["rec_id"]})["data"]
            else:
                w["r1cam_recording"] = {"refused": rec}
        st = request(cport, {"op": "status"}).get("data", {})
        w["r1cam"] = {"streams": st.get("streams"), "depth_reader": st.get("depth_reader")}
        if after is not None:
            after(name, w)


def observes_ok(out: dict[str, Any]) -> str:
    return " | ".join(f"{k} {out['windows'][k]['observes_ok']}" for k in ("work", "work_fast", "calib"))


# ---------------------------------------------------------------------------------------------------------------- sim
@contextlib.contextmanager
def _sim_stack(domain: int, workdir: Path, camera: bool = False) -> Iterator[Any]:
    from r1sim.stack import ProcStack

    bridge = {"stance": "hanging", "scene": "none", "safety": False, "builtin": "stand_lock"}
    with ProcStack(domain=domain, workdir=workdir / "stack", bridge=bridge, shm_name=f"/r1mc3.cam.{os.getpid()}",
                   camera=camera) as st:  # fmt: skip
        res = st.cli("on-stand")
        if not res.get("ok"):
            raise RuntimeError(f"r1 on-stand refused: {res}")
        st.wait_mode("on_stand", 20)
        yield st


FAKE_ROBOT_CHILD = (
    "import json, time\n"
    "from r1mc3.cam.fake import FakeRobot\n"
    "f = FakeRobot(depth_hz=10.0, repeat_every=0).start()\n"
    "print(json.dumps({'port': f.depth_addr[1]}), flush=True)\n"
    "while True: time.sleep(1)\n"
)


def run_sim_rendered(a: argparse.Namespace) -> dict[str, Any]:
    """The four windows with the SIMULATED head camera (session 11): r1cam's sim process (r1sim.bridge_camera) idles
    in the off windows (nothing wanted, nothing rendered) and renders left + right + depth while observed / recorded."""
    from r1mc3.service.port import ShmPort

    workdir = Path(tempfile.mkdtemp(prefix="cam_impact_r_", dir=a.scratch))
    out: dict[str, Any] = {"mode": "sim-rendered", "seconds": a.seconds, "windows": {}}
    with _sim_stack(a.domain, workdir, camera=True) as st:
        port = ShmPort.attach(st.shm)
        ages = AgeSampler(port.view)
        ages.start()
        cport = int(st.camera.port)
        frames0 = {"n": 0}

        def link_bytes() -> int:  # the frames the camera rendered (no network in the sim): counted, not bytes
            return int(st.camera.frames) - frames0["n"]

        def pids() -> dict[str, list[int]]:
            return {"r1rt": [st.rt.pid], "r1guard": [st.guard.pid], "r1cam": [st.camera.proc.pid]}

        m = Measure(st.rt.last_status, st.guard.last_status, ages, pids, lambda: None)

        out["windows"]["off"] = m.window(a.seconds)
        f0 = {"n": link_bytes()}

        def rendered(name: str, w: dict[str, Any]) -> None:
            w["frames_rendered"] = link_bytes() - f0["n"]
            f0["n"] = link_bytes()

        camera_windows(m, cport, a.seconds, out, after=rendered)
        time.sleep(4.0)  # the streams go idle (3 s after the last request)
        out["windows"]["off_again"] = m.window(a.seconds)
        out["observes_ok"] = observes_ok(out)
        out["camera"] = st.camera.stats()
        ages.stop()
        close_port(port)
    return out


def run_sim(a: argparse.Namespace) -> dict[str, Any]:
    from r1mc3.service.port import ShmPort

    if getattr(a, "camera", "bench") == "rendered":
        return run_sim_rendered(a)
    workdir = Path(tempfile.mkdtemp(prefix="cam_impact_", dir=a.scratch))
    # the robot side: the real depth stream (~39 Mbit/s) over the loopback, from its OWN process (in this process it
    # would share the GIL with the bridge's physics and slow the simulated robot - a harness artefact, seen in the
    # first run: the bridge's state age max 34 -> 128 ms with the stand-in in-process)
    robot = subprocess.Popen(
        [sys.executable, "-c", FAKE_ROBOT_CHILD], stdout=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": f"{REPO / 'python'}"},
    )  # fmt: skip
    assert robot.stdout is not None
    fake_port = int(json.loads(robot.stdout.readline())["port"])
    sent = {"bytes": 0}
    cam: dict[str, Any] = {}

    def link_bytes() -> int:
        """r1cam's received depth bytes (its depth frame count x the frame size) - read now, not at the last observe."""
        if "port" in cam and cam["proc"].poll() is None:
            st_ = request(int(cam["port"]), {"op": "status"})["data"]["streams"]
            sent["bytes"] = int(st_["depth"]["frames"]) * (36 + 544 * 448 * 2)
        return int(sent["bytes"])

    out: dict[str, Any] = {"mode": "sim", "seconds": a.seconds, "burn_ms": a.burn_ms, "windows": {}}
    with _sim_stack(a.domain, workdir) as st:
        port = ShmPort.attach(st.shm)
        ages = AgeSampler(port.view)
        ages.start()

        def pids() -> dict[str, list[int]]:
            d = {"r1rt": [st.rt.pid], "r1guard": [st.guard.pid]}
            if "proc" in cam and cam["proc"].poll() is None:
                d["r1cam"] = [cam["proc"].pid, *_children(cam["proc"].pid)]
            return d

        m = Measure(st.rt.last_status, st.guard.last_status, ages, pids, link_bytes)

        def start_cam() -> int:
            p = subprocess.Popen(
                [sys.executable, "-m", "r1mc3.cam", "--source", "bench", "--port", "0", "--records",
                 str(workdir / "records"), "--depth", f"127.0.0.1:{fake_port}", "--burn-ms", str(a.burn_ms)],
                stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": f"{REPO / 'python'}"},
            )  # fmt: skip
            assert p.stdout is not None
            ready = json.loads(p.stdout.readline())
            cam["proc"], cam["port"] = p, int(ready["port"])
            return int(ready["port"])

        out["windows"]["off"] = m.window(a.seconds)
        cport = start_cam()
        camera_windows(m, cport, a.seconds, out)
        man = out["windows"]["calib"].get("r1cam_recording", {})
        if man.get("files"):
            man["mbytes"] = round(man["files"][0]["bytes"] / 1e6, 1)
        cam["proc"].terminate()
        cam["proc"].wait(10)
        out["windows"]["off_again"] = m.window(a.seconds)
        out["observes_ok"] = observes_ok(out)
        ages.stop()
        close_port(port)
    robot.kill()
    out["link_note"] = "sim: link_mbit_s = r1cam's received depth bytes from the stand-in robot (the robot: eth10)"
    return out


# -------------------------------------------------------------------------------------------------------------- robot
def _journal_status(unit_names: tuple[str, ...], key: str) -> Callable[[], dict[str, Any] | None]:
    """The newest {"status": key} line of a systemd unit's journal (r1rt / r1guard print one per second)."""

    def f() -> dict[str, Any] | None:
        args = ["journalctl", "-o", "cat", "-n", "50", "--no-pager"]
        for u in unit_names:
            args += ["-u", u]
        lines = subprocess.run(args, capture_output=True, text=True).stdout.splitlines()
        for line in reversed(lines):
            with contextlib.suppress(ValueError):
                d = json.loads(line)
                if d.get("status") == key:
                    return dict(d)
        return None

    return f


def run_robot(a: argparse.Namespace) -> dict[str, Any]:
    from r1mc3.service.port import ShmPort

    port = ShmPort.attach(a.shm)
    ages = AgeSampler(port.view)
    ages.start()

    def pids() -> dict[str, list[int]]:
        d: dict[str, list[int]] = {}
        for name in ("r1rt", "r1guard", "r1cam"):
            got = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True).stdout.split()
            if got:
                d[name] = [int(x) for x in got]
        return d

    m = Measure(_journal_status(("r1rt", "r1rt-stand"), "r1rt"), _journal_status(("r1guard",), "r1guard"), ages,
                pids, lambda: _net_bytes(a.iface))  # fmt: skip
    out: dict[str, Any] = {"mode": "robot", "seconds": a.seconds, "windows": {}}

    def ask(msg: str) -> None:
        input(f"\n>>> {msg} - press Enter when done ")

    out["windows"]["off"] = m.window(a.seconds)
    ask("start r1cam:  sudo systemctl start r1cam")

    camera_windows(m, a.cam_port, a.seconds, out)
    ask("stop r1cam:  sudo systemctl stop r1cam")
    out["windows"]["off_again"] = m.window(a.seconds)
    ages.stop()
    close_port(port)
    out["verdict"] = {k: verdict(w) for k, w in out["windows"].items()}
    out["judged"] = list(JUDGED)
    out["pass"] = all(out["verdict"][k]["pass"] for k in JUDGED)
    return out


def table(res: dict[str, Any]) -> list[str]:
    """One compact row per window (the report's table)."""
    rows = ["| window | r1rt skipped / overruns | r1rt late > 0.5 / 1 / 2 ms (window max <= ms) | lowstate rate /s | "
            "age at r1rt p50 / p99 / max ms | guard late > 0.5 ms | CPU % r1rt / r1guard / r1cam | link Mbit/s |",
            "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for k, w in res["windows"].items():
        r, g = w.get("r1rt", {}), w.get("r1guard", {})
        lt, gl = r.get("lateness", {}), g.get("lateness", {})
        ag, cpu = w.get("lowstate_age_at_r1rt", {}), w.get("cpu", {}).get("proc_pct", {})

        def f(x: Any) -> str:
            return "-" if x is None else f"{x:.2f}" if isinstance(x, float) else str(x)

        rows.append(
            f"| {k} | {r.get('skipped_deadlines')} / {r.get('overruns')} | {lt.get('over_0p5ms')} / "
            f"{lt.get('over_1ms')} / {lt.get('over_2ms')} ({f(lt.get('window_max_bound_ms'))}) | "
            f"{f(r.get('lowstate_rate_hz'))} | "
            f"{f(ag.get('p50_ms'))} / {f(ag.get('p99_ms'))} / {f(ag.get('max_ms'))} | {gl.get('over_0p5ms')} | "
            f"{f(cpu.get('r1rt'))} / {f(cpu.get('r1guard'))} / {f(cpu.get('r1cam'))} | {f(w.get('link_mbit_s'))} |"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="where", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--domain", type=int, default=100)
    s.add_argument("--burn-ms", type=float, default=0.0, help="CPU per image frame in r1cam's decoder stand-in")
    s.add_argument("--scratch", default=None)
    s.add_argument("--camera", choices=("bench", "rendered"), default="bench",
                   help="bench: r1cam's bench source + a depth-stream stand-in (session 9); rendered: the simulated "
                   "head camera (session 11)")  # fmt: skip
    r = sub.add_parser("robot")
    r.add_argument("--shm", default="/r1mc3.rt")
    r.add_argument("--cam-port", type=int, default=8781)
    r.add_argument("--iface", default="eth10")
    for p in (s, r):
        p.add_argument("--seconds", type=float, default=20.0)
        p.add_argument("--out", default="")
    a = ap.parse_args(argv)
    res = run_sim(a) if a.where == "sim" else run_robot(a)
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)
    print("\n".join(table(res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
