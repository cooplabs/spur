"""System identification for the HW1 recordings (session 5): the fits that need HW1 data, PREPARED so the data drops in.

**Nothing here changes the simulator's model.** Each module is a protocol + a loader-independent fit + a synthetic
recording from the simulator with KNOWN parameters (the tests recover them); the fitted values go into the model as a
reviewed change with provenance once the real recordings exist (docs/hardware/HW1_FITS.md):

* `stiction` - the legs' dry friction and damping from a slow ramp per joint on the stand (A7 nominal, U4);
* `pads` - each Dex3 pad's gain and footprint from presses with known weights + a pinch check on a thin block (A10 rows
2-3);
* `heat` - the finger motors' thermal model STRUCTURE and parameters from full-rate holds (A10's fast pole);
* `records` - the one in-memory form and the loaders (r1log MCAP, the hands' state topic, .npz, the press notes).

CLI: `python -m r1sim.sysid {stiction,pads,heat} ...` (= tools/hw1fit/hw1fit.py); `... demo` runs on synthetic data.
"""
