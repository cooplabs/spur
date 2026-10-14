"""HW1 (the stand-only session, DESIGN.md §14.1) operator tools that r1d runs on the backpack, next to r1rt:

* `pause`: the board-hold test (LowCmd paused for 20 / 100 / 300 / 400 / 500 ms with the guard OFF: how long the board
holds a command)
  and T6 (r1rt stopped in ON_STAND with the guard ARMED: the guard holds, ownership returns only by `r1 reclaim`) -
  SIGSTOP / SIGCONT of r1rt with the actual times, a detached SIGCONT backstop, and r1d's own rt/lowstate recorder.
* `session`: the session-end log save built into `r1 shutdown` (DESIGN §4.4 / §9: v2 once lost a session's log to a
  wrong serve name).
"""
