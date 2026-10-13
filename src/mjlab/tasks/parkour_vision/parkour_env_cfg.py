"""Parkour vision env cfg factory (robot-agnostic overrides).

Reuses :func:`mjlab.tasks.velocity_vision.velocity_env_cfg.make_velocity_env_cfg`
with the parkour terrain set, adds the ``reward_point_bonus`` milestone reward,
and switches the velocity command to pure forward (x-only).
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.parkour_vision.mdp import reward_point_bonus
from mjlab.tasks.velocity_vision.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import PARKOUR_TERRAINS_CFG


def apply_parkour_overrides(
  cfg: ManagerBasedRlEnvCfg,
  terrain_cfg=PARKOUR_TERRAINS_CFG,
) -> None:
  """在 velocity_vision env cfg 上叠加 parkour 任务专属配置（机器人无关）。

  - 地形替换为 ``PARKOUR_TERRAINS_CFG``（梅花桩 / 高平台 / 沟壑，各占一列）。
  - 加 ``reward_point_bonus`` 奖励（基座穿越奖励点 + 速度门控）。
  - command 切纯 x 前进（``rel_pure_x_envs=1.0``，lin_vel_y=ang_vel_z=0，关闭
    heading/standing），固定初始 yaw 朝世界 x。
  - command curriculum 改为 parkour 速度阶段（仅 x）。
  """
  # 地形：替换为 parkour 集合（replace 浅拷贝 cfg 本身，sub_terrains 共享）。
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(terrain_cfg)

  # 奖励点 milestone bonus（robot parkour 风格）。
  cfg.rewards["reward_point_bonus"] = RewardTermCfg(
    func=reward_point_bonus,
    weight=1.0,
    params={
      "command_name": "twist",
      "hit_radius": 0.4,
      "speed_tol": 0.5,
      "command_threshold": 0.1,
      "bonus_value": 1.0,
    },
  )

  # 纯 x 前进 command。
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_pure_x_envs = 1.0
  twist.rel_pure_y_envs = 0.0
  twist.heading_command = False
  twist.ranges.heading = None
  twist.ranges.lin_vel_x = (0.3, 1.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.terrain_command_ranges = {}

  # 固定初始 yaw 朝世界 x，收紧出生随机范围（障碍链附近需留出自由半径）。
  cfg.events["reset_base"].params["pose_range"]["x"] = (-0.2, 0.2)
  cfg.events["reset_base"].params["pose_range"]["y"] = (-0.3, 0.3)
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)

  # Command curriculum：仅 x 方向速度阶段。
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (0.3, 0.8),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (0.0, 0.0),
    },
    {
      "step": 2000 * 24,
      "lin_vel_x": (0.4, 1.0),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (0.0, 0.0),
    },
  ]
  cfg.curriculum["command_vel"].params["terrain_velocity_stages"] = None
