"""The HW1 fits from the command line (a thin wrapper around `python -m r1sim.sysid`; docs/hardware/HW1_FITS.md).

uv run python tools/hw1fit/hw1fit.py stiction --log r1log_XXX.mcap --provenance
uv run python tools/hw1fit/hw1fit.py pads --hands hands.mcap --presses presses.yaml --side left
uv run python tools/hw1fit/hw1fit.py heat --hands hands.mcap --side left --motor 3 --windows 10:190,200:380
uv run python tools/hw1fit/hw1fit.py demo heat
"""

from __future__ import annotations

import sys

from r1sim.sysid.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
