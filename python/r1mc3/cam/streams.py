"""Which head-camera streams r1cam runs, and when (session 12: the user's decision after HW2 day 1's C1, 12 Oct 2026
~21:00 PDT: "fewer streams - the left image + depth while working, all three only for calibration recordings").

What C1 measured on the robot (`results/hw2_20261012/cam/cam_impact.json`; the camera and the robot's state share the
backpack's USB-2 eth10 adapter, a 100 Mbit/s link): with r1cam streaming left + right + depth the lowstate rate fell
1052 -> 997 /s (~5 % of the state packets lost) and r1rt's state age p99 rose 1.0 -> 2.9 ms (max 5.1); eth10 carried
29.5 Mbit/s with the camera off and 74.1 Mbit/s with it on (+44.6). Where the +44.6 Mbit/s comes from (the recording of
the same window, `20261013T031124Z_cam_impact`):
  - depth: one 544 x 448 x 2 B = 487 KB frame per send at ~10 /s (471 unique frames in 62.8 s = 7.5 /s; the stream
    repeats ~1 in 4 frames, robot.py) = **~39 Mbit/s = ~87 %** of the camera's traffic;
  - left / right images: H.264 RTP, 8.68 MB per stream in 62.8 s = **~1.1 Mbit/s each** (the Matroska files).
So "fewer streams" (no right image while working) saves ~1.1 Mbit/s (~2.5 % of the camera's load) - on its own it
cannot move C1's numbers. The work path already asked for left + depth only (r1d's camera.observe: right=False by
default; v2's task client asks for the right image only with R1_HEAD_RIGHT=1); C1's tool streamed all three. The levers
that do reduce what C1 measured, both on r1cam's side of the link and generic (no robot-side change):
  1. **depth only while it is needed**: an observe holds the depth stream until ITS fresh frame arrived and then lets
     it close after DEPTH_KEEP_S (before: kept IDLE_S = 3 s after every request -> ~30 frames = ~14.6 MB per observe
     at the robot's rate; now ~1-5 frames). The image streams keep IDLE_S (an H.264 stream needs a key frame to
     restart; 1.1 Mbit/s each);
  2. **a bounded TCP receive window on the depth socket** (DEPTH_RCVBUF_BYTES, set before connect): the robot's
     sender can then have at most that many depth bytes in flight, so at most that much queues IN FRONT of a lowstate
     packet on the shared path (a whole 487 KB frame = ~39 ms of line time at 100 Mbit/s; 16 KB = ~1.3 ms). The rate
     it still allows: 16 KB per round trip = 16 MB/s at a 1 ms RTT, ~3x the 4.9 MB/s the depth needs (break-even
     RTT ~3.3 ms; the robot's link answers in well under 1 ms). Linux doubles the requested value internally
     (socket(7)); macOS keeps it.
Which of these brings C1 back inside the proposed rule (rate >= 1000 /s, age p99 <= 2 ms) is day 2's C1 re-run (the
impact tool's `work` / `work_fast` / `calib` windows) - not provable on the Mac.
"""

from __future__ import annotations

from collections.abc import Iterable

WORK_STREAMS = ("left", "depth")  # what an observe may stream while working (the user, 12 Oct)
CALIBRATION_KINDS = ("charuco", "flat_field", "calibration")  # recordings that may stream the right image too
DEPTH_KEEP_S = 0.5  # the depth stream's idle time after an observe got its frame (before: IDLE_S 3 s)
DEPTH_RCVBUF_BYTES = 16384  # the depth socket's receive buffer = the TCP window the robot may fill (0: the OS default)
RIGHT_REFUSED = (
    "the right image streams only during calibration recordings (kind charuco / flat_field / calibration; the user's "
    "decision 12 Oct after HW2's C1: fewer streams - the left image + depth while working)"
)


def right_allowed(kind: str) -> bool:
    """A recording of this kind may stream the right image."""
    return str(kind) in CALIBRATION_KINDS


def record_refusal(kind: str, streams: Iterable[str]) -> str | None:
    """Why a recording of `kind` with `streams` is refused (None = allowed)."""
    if "right" in tuple(streams) and not right_allowed(kind):
        return f"recording kind {kind!r} asked for the right image: {RIGHT_REFUSED}"
    return None


def policy(depth_keep_s: float = DEPTH_KEEP_S) -> dict[str, object]:
    """What `status` shows (`r1 camera status`; the depth reader's own numbers are in `depth_reader`)."""
    return {
        "work_streams": list(WORK_STREAMS),
        "right": "calibration recordings only (" + " / ".join(CALIBRATION_KINDS) + ")",
        "depth": f"on demand: held until an observe's frame, then closed after {depth_keep_s:g} s",
    }
