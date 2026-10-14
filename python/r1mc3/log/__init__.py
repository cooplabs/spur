"""r1log (DESIGN.md §9, §2.1): drains r1rt's telemetry and policy rings into MCAP files (zstd chunks), samples the
hands-contact and guard-heartbeat seqlocks, records mode / token / fault / stop events, and keeps a 30 s flight
recorder that dumps its own MCAP file on a fault, a takeover, a large tilt or an operator request.

Low priority by design: it only pops single-consumer rings and reads seqlocks, so the RT core never waits on it; a
logger that falls behind loses records with a counter (docs/architecture/LOGGER.md).
"""

from .flight import FlightRecorder, Trigger
from .logger import DEFAULT_TRIGGER_FAULTS, Logger, LoggerConfig, run_forever
from .mcapfile import STREAMS, McapFile, RotatingLog
from .reader import LogData, info, read

__all__ = [
    "DEFAULT_TRIGGER_FAULTS",
    "STREAMS",
    "FlightRecorder",
    "LogData",
    "Logger",
    "LoggerConfig",
    "McapFile",
    "RotatingLog",
    "Trigger",
    "info",
    "read",
    "run_forever",
]
