"""r1mc3.plan: the arm planner (DESIGN.md §7.2) - a pure function of a world snapshot, the measured joints and a goal.

    res = plan(world.snapshot(), PlanGoal("reach", "left", position=(0.35, 0.2, 0.85)), q_now26)
    res.ok, res.code, res.segments   # Segment = one executor TRAJECTORY command (t, q, dq, ddq, guards)

`PlannerWorker` runs `plan` in a worker process (a slow plan never blocks r1d's tick).
"""

from .core import KINDS, PlanGoal, PlanResult, Segment, geometry_for, plan
from .ik import IKResult, arm_limits, policy_envelope, solve
from .orient import FAMILIES, OrientResult, best_orientation
from .timing import Timed, time_path
from .worker import PlannerWorker

__all__ = [
    "FAMILIES",
    "KINDS",
    "IKResult",
    "OrientResult",
    "PlanGoal",
    "PlanResult",
    "PlannerWorker",
    "Segment",
    "Timed",
    "arm_limits",
    "best_orientation",
    "geometry_for",
    "plan",
    "policy_envelope",
    "solve",
    "time_path",
]
