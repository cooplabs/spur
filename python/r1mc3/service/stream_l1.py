"""L1 `stream.arm_twist` (DESIGN.md §5.1; session 10): the hand's grasp centre at a Cartesian velocity.

Resolved rate: at every request r1d computes the arm's position Jacobian at the arm's CURRENT target (the jog's own
setpoint while a stream runs) from the core's built-in arm model (numerical, 5 forward kinematics; the model r1d
measures every result with), takes damped least squares joint rates for the requested velocity, caps them, and hands
them to the joint jog (`stream.joint_jog`'s integrator at r1d's tick, its TTL: silence decelerates the arm to HOLD).
A client streams twists (e.g. 20 Hz, TTL 0.3 s); each request re-linearises.

Not built (said plainly): the collision look-ahead (DESIGN §5.1) - so a twist is REFUSED while the world model has
declared entities (the same rule as an unchecked arm.move_joint); orientation rates (5-DoF arm: position only);
`stream.arm_impedance`, `stream.base_velocity` (walking is not in scope).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .codes import GROUP_JOINTS, Code
from .goals import ARM_GROUPS, POINT_GRASP_CENTRE, SIDE_INDEX, hand_points, port_state
from .schemas import ArmTwistReq, Result, fail

if TYPE_CHECKING:
    from .service import Service

FloatArray = npt.NDArray[np.float64]
V_MAX_M_S = 0.10  # the request's |v| cap (schemas.ArmTwistReq)
DQ_CAP_RAD_S = 1.0  # the joint rates' cap (v2 planned at <= 1 rad/s peak; the executor's dq_max is 2)
DAMPING = 0.02  # the least-squares damping (m): rates stay bounded near a singular pose
EPS_RAD = 1e-4  # the finite-difference step


def handles(r: Any) -> bool:
    return isinstance(r, ArmTwistReq)


def _centre(svc: Service, st: dict[str, Any], side: str, group: str, q_arm: FloatArray) -> FloatArray | None:
    q = np.asarray(st["q"], dtype=np.float64).copy()
    q[list(GROUP_JOINTS[group])] = q_arm
    pts = hand_points(svc.port, svc.hands, {**st, "q": q})
    return None if pts is None else np.asarray(pts[SIDE_INDEX[side], POINT_GRASP_CENTRE], dtype=np.float64)


def jacobian(svc: Service, side: str, q_arm: FloatArray) -> FloatArray | None:
    """The grasp centre's position Jacobian (3 x 5, base frame) at the arm joints q_arm (the other joints measured)."""
    st = port_state(svc.port)
    if st is None or "q" not in st:
        return None
    group = ARM_GROUPS[side]
    p0 = _centre(svc, st, side, group, q_arm)
    if p0 is None:
        return None
    jac = np.zeros((3, 5))
    for k in range(5):
        dq = q_arm.copy()
        dq[k] += EPS_RAD
        pk = _centre(svc, st, side, group, dq)
        if pk is None:
            return None
        jac[:, k] = (pk - p0) / EPS_RAD
    return jac


def arm_twist(svc: Service, r: ArmTwistReq) -> Result:
    group = ARM_GROUPS[r.arm]
    if svc.l2.world_has_entities():
        return fail(r.op, r.id, Code.UNSUPPORTED, reason="no collision look-ahead for twists yet: refused while the "
                    "world has declared entities (use the planned arm goals)")  # fmt: skip
    jog = svc._jogs.get(group)
    q_arm = jog.q_set.copy() if jog is not None else svc._current(group)
    if q_arm is None:
        return fail(r.op, r.id, Code.UNSUPPORTED, reason="the core's current targets are not readable")
    jac = jacobian(svc, r.arm, np.asarray(q_arm, dtype=np.float64))
    if jac is None:
        return fail(r.op, r.id, Code.UNSUPPORTED, reason="no arm model / measured state for the Jacobian")
    v = np.asarray(r.v, dtype=np.float64)
    dq = jac.T @ np.linalg.solve(jac @ jac.T + DAMPING**2 * np.eye(3), v)
    peak = float(np.max(np.abs(dq)))
    scale = min(1.0, DQ_CAP_RAD_S / peak) if peak > 0 else 1.0
    out = svc._jog(r.op, r.id, group, dq * scale, r.ttl)
    if out.ok:
        v_pred = jac @ (dq * scale)
        out = out.model_copy(update={"data": {**out.data, "dq": [round(float(x), 5) for x in dq * scale],
                                              "v_predicted": [round(float(x), 5) for x in v_pred],
                                              "scaled": scale < 1.0}})  # fmt: skip
    return out
