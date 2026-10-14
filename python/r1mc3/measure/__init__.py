"""Measuring the setup with fiducial tags (DESIGN §8.2; the user's decision: tags measured before each trial, covered
during it).

Generic pieces (no task knowledge, D32): `tags` (tag36h11 detection, tag placements on an entity), `lenses` (the head
camera's calibrated model, a phone's OpenCV model), `register` (an entity's pose from its tags seen in one or more
views), `phone` (a hand-placed camera's pose from table tags: the placement correction), `preflight` (checks with
tolerances -> pass / refuse with the words that say what to move). The client supplies the entities, their tag plans,
the nominal layout and the tolerances through this API.

OpenCV (cv2.aruco, solvePnP, undistortPoints) is imported inside the functions that need it: it is a dev dependency
today; the runtime extra is proposed in docs/architecture/MEASUREMENT.md.
"""

from .lenses import HeadLens, Lens, OpenCVLens
from .preflight import Check, CheckResult, PoseCheck, PreflightReport, ValueCheck, run_checks
from .register import Registration, View, locate_entity
from .tags import TagObs, TagPlacement, detect_tags, flat_tag

__all__ = [
    "Check",
    "CheckResult",
    "HeadLens",
    "Lens",
    "OpenCVLens",
    "PoseCheck",
    "PreflightReport",
    "Registration",
    "TagObs",
    "TagPlacement",
    "ValueCheck",
    "View",
    "detect_tags",
    "flat_tag",
    "locate_entity",
    "run_checks",
]
