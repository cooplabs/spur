# The shared-memory segment and the step API

How our processes talk to the real-time core without ever making it wait ([`../DESIGN.md`](../DESIGN.md) §2). Code:
`rt/include/r1rt/{types.h, seqlock.hpp, spsc_ring.hpp, shm_layout.h, shm.hpp, step_api.h}`; the Python view
`r1rt_py.SegmentView` (`rt/bindings/`). The layout's source of truth is `shm_layout.h` (`R1RT_LAYOUT_VERSION`); every
reader checks the header (magic, layout version, ABI version, size, joint count) and refuses a mismatch.

## One segment

One POSIX shared-memory object (`/r1mc3.rt` on the robot), created by `r1rt` at start - in MONITOR, before anything is
published. In the simulator's lockstep mode the same struct lives in ordinary memory, so the same code is tested.

| Region | Type | Writer | Readers | Rate |
|---|---|---|---|---|
| `header` | fixed header | `r1rt`, once | all | once |
| `token` | one 64-bit word | compare-and-swap: `r1rt` (takeover), `r1guard` (stale heartbeat), the operator via `r1d` (reclaim) | `r1rt` / `r1guard` before every write | event |
| `rt_heartbeat` | seqlock | `r1rt` | `r1guard` (1 kHz), `r1d`, tools | every writer tick |
| `guard_heartbeat` | seqlock | `r1guard` | `r1rt` (no takeover without a live guard), `r1d` | every guard cycle |
| `state` | seqlock | `r1rt` | `r1d`, `r1guard`, `r1hands` | every writer tick |
| `mirror` | seqlock | `r1rt` | `r1guard` (the warm standby's input) | every writer tick |
| `base_cmd` | seqlock | `r1d` | `r1rt` (time-to-live <= 0.5 s; expiry = a zero command) | on change |
| `hands_contact`, `hands_pose` | seqlocks | `r1hands` | `r1rt` (contact guards, the arm model) | 100 Hz |
| `stop` | the stop channel | **any client** (atomic add) | `r1rt` every tick (acknowledges), `r1guard` | event |
| `keepalive` | per-group stamps | `r1d` | `r1rt` (the trajectory deadman) | 20 Hz+ |
| `upper_cmd` | SPSC ring | `r1d` | `r1rt` | per goal / stream message |
| `mode_req` | the mode mailbox | clients (compare-and-swap) | `r1rt` every tick; each client its reply | event |
| `exec_status` | seqlock | `r1rt` | `r1d` (how goals ended) | every writer tick |
| `telemetry`, `policy`, `guard_telemetry` | SPSC rings | `r1rt` / `r1guard` | `r1log` | every tick / 50 Hz / 1 kHz |

Rules: fixed size, no pointers, every region on its own 64-byte line, only lock-free (address-free) atomics; a
seqlock reader copies and retries, it never blocks the writer; a full ring refuses the new record and counts it -
the producer never waits.

## The ownership token

One word = (owner, generation). `r1rt` takes it at a takeover (from "nobody"), `r1guard` swaps it from `r1rt` when the
heartbeat is stale, the operator's reclaim gives it back to `r1rt`. Each writer checks it immediately before every
write, so a race is bounded to one packet.

## The stop channel

Any client posts a request (level stop / halt / damp, who, when) by an atomic add on a counter; `r1rt` reads it every
tick and acknowledges the count it acted on. Damp is honoured only from the operator. A stop raises the stop epoch:
commands stamped with an older epoch are refused afterwards. A stop or halt stays latched until `resume` (the
heartbeat says so).

## The mode mailbox

Several clients (`r1d`, the CLI, tests) send mode requests (takeover, on-stand, freeze, balance, reclaim, ...) to
the core through one mailbox: a client claims it by compare-and-swap, writes the request, and reads the core's reply
(status, refusal code, the mode after) from a reply slot. The core acts on it inside its tick with that tick's state.

## The step API (C ABI, `step_api.h`)

`r1rt_core_create`, `r1rt_core_step`, `r1rt_core_policy_step`, `r1rt_core_destroy` (+ status and setter functions) and the segment helpers: the process wrapper and the
Python binding (`r1rt_py.Core`) call the same functions, so the simulator's lockstep and the robot run the same core
code. Every function returns a code (`R1RT_OK`, `R1RT_E_REFUSED`, `R1RT_E_INVALID_ARG`, `R1RT_E_NOT_OWNER`, ...);
nothing throws, nothing allocates after `init`.
