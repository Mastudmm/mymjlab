"""Static check of the Go1 depth-camera mount (head site).

Prints the camera site's world position and forward direction (-Z axis of the
site frame, matching RayCastSensor's pinhole convention) for the robot in its
default standing pose. Run from anywhere:

    uv run python src/mjlab/scripts/check_camera.py
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.asset_zoo.robots import get_go1_robot_cfg

EXPECTED_FORWARD = np.array([1.0, 0.0, 0.0])  # +x, horizontal (no fixed pitch)
EXPECTED_PITCH_DEG = 0.0  # arcsin(forward_z); training randomizes -25°..-15°


def main() -> None:
  robot = get_go1_robot_cfg().build()
  model = robot.spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head")
  pos = data.site_xpos[site_id]
  xmat = data.site_xmat[site_id].reshape(3, 3)
  forward = -xmat[:, 2]  # site -Z is the camera forward direction

  pitch_deg = np.degrees(np.arcsin(forward[2]))
  yaw_deg = np.degrees(np.arctan2(forward[1], forward[0]))

  print(f"head site 世界位置 : ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
  print(f"相机 forward 向量  : ({forward[0]:.4f}, {forward[1]:.4f}, {forward[2]:.4f})")
  print(f"俯仰角(下俯为负)  : {pitch_deg:+.2f}° (期望 {EXPECTED_PITCH_DEG:+.2f}°)")
  print(f"偏航角             : {yaw_deg:+.2f}° (期望 0.00°)")

  ok_pos = abs(pos[2] - 0.595) < 1e-3  # trunk base 0.445 + site z 0.15
  ok_forward = np.allclose(forward, EXPECTED_FORWARD, atol=1e-3)
  print("位置检查:", "PASS" if ok_pos else "FAIL")
  print("朝向检查:", "PASS" if ok_forward else "FAIL")


if __name__ == "__main__":
  main()
