"""r1d - the minimal service (session 5, DESIGN.md §2.1, §5; M5a).

r1d is the one process clients talk to (the `r1` CLI, the Python SDK, task clients). It never sits in the
balance path: everything it asks of the real-time core goes through the core's shared segment (DESIGN §2 principle
3) - the mode-request mailbox (takeover, ON_STAND, freeze, ...), the stop channel (stop / halt / damp, any client),
and the upper_cmd ring (L0 / L1 records, stamped with the stop epoch) with the TRAJECTORY keepalives (the executor's
deadman, DESIGN §3.3). A stalled r1d therefore stops arm motion (TTL / deadman -> HOLD), never balance.

Layout (a library with a `tick()` first and a process second, DESIGN §2 principle 2):

* `codes`   - error codes (DESIGN §5), the core's modes / requests / groups / requesters (from the C++ binding);
* `schemas` - the typed requests (pydantic, one model per op) and the one result shape;
* `port`    - how r1d reaches the core: `ShmPort` (r1rt's POSIX segment through `r1rt_py.SegmentView`) or
              `LockstepPort` (an `r1rt_py.Core` driven by the simulator; tests and rehearsals);
* `service` - `Service`: request dispatch + `tick(now_ns)` (streams, jogs, keepalives) + the takeover / ON_STAND
              sequences (the motion-switcher release is r1d's job, never an RT thread's);
* `server`  - localhost JSON lines over TCP (port 8780) + the 100 Hz tick thread; `client` - the matching client;
* `python -m r1mc3.service --shm-name NAME --domain N [--port 8780]` - the process.
"""
