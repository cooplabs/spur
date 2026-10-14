"""A small floating 'robot' with motors, standing in for the R1 in engine tests that must not depend on the model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt

TOY_XML = """
<mujoco model="toy">
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" friction="0.8 0.02 0.01"/>
    <body name="pelvis" pos="0 0 0.86">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.15 0.05" mass="4"/>
      <site name="imu"/>
      <body name="leg" pos="0 0 -0.05">
        <joint name="j0_joint" axis="0 1 0" range="-1 1" armature="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03" mass="1"/>
        <body pos="0 0 -0.25">
          <joint name="j1_joint" axis="0 1 0" range="-1 1" armature="0.01"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03" mass="1"/>
          <geom type="box" pos="0.03 0 -0.27" size="0.12 0.1 0.02" mass="0.5"/>
        </body>
      </body>
      <body name="arm" pos="0 0.2 0">
        <joint name="j2_joint" axis="0 1 0" range="-3 3" armature="0.005"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="j0" joint="j0_joint" ctrlrange="-60 60" ctrllimited="true"/>
    <motor name="j1" joint="j1_joint" ctrlrange="-60 60" ctrllimited="true"/>
    <motor name="j2" joint="j2_joint" ctrlrange="-5 5" ctrllimited="true"/>
  </actuator>
  <sensor>
    <gyro name="imu_ang_vel" site="imu"/>
    <accelerometer name="imu_lin_acc" site="imu"/>
  </sensor>
</mujoco>
"""

# The three toy joints sit in SDK slots 3 (j0), 9 (j1) and 14 (j2); every other slot is absent.
TOY_SLOTS = {3: "j0", 9: "j1", 14: "j2"}


@dataclass
class ToyIndex:
    present: npt.NDArray[np.bool_]
    qadr: npt.NDArray[np.int64]
    vadr: npt.NDArray[np.int64]
    act: npt.NDArray[np.int64]
    tau_max: npt.NDArray[np.float64]
    imu_site: int
    hand_qadr: Mapping[str, npt.NDArray[Any]] = field(default_factory=dict)
    hand_vadr: Mapping[str, npt.NDArray[Any]] = field(default_factory=dict)
    hand_act: Mapping[str, npt.NDArray[Any]] = field(default_factory=dict)
    hand_tau_max: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(7))


def toy_model() -> Any:
    return mujoco.MjModel.from_xml_string(TOY_XML)


def toy_index(m: Any) -> ToyIndex:
    present = np.zeros(26, dtype=bool)
    qadr = np.full(26, -1, dtype=np.int64)
    vadr = np.full(26, -1, dtype=np.int64)
    act = np.full(26, -1, dtype=np.int64)
    tau = np.zeros(26)
    for slot, name in TOY_SLOTS.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        present[slot] = True
        qadr[slot], vadr[slot], act[slot] = m.jnt_qposadr[j], m.jnt_dofadr[j], a
        tau[slot] = m.actuator_ctrlrange[a, 1]
    imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    return ToyIndex(present, qadr, vadr, act, tau, int(imu))
