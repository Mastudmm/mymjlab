"""Unitree Go1 parkour vision env cfg (Go1 specialization on the parkour task)."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.parkour_vision.parkour_env_cfg import apply_parkour_overrides
from mjlab.tasks.velocity_vision.config.go1.env_cfgs import (
  unitree_go1_rough_env_cfg,
)


def unitree_go1_parkour_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """构建 AME parkour Go1 环境配置（Go1 特化 + parkour 任务配置）。

  复用 ``unitree_go1_rough_env_cfg`` 的全部 Go1 特化（robot、depth_camera、
  terrain_scan、contact sensors、foot friction、camera DR、reward std、
  terminations 等），仅叠加 parkour 任务专属覆盖（见
  :func:`apply_parkour_overrides`）：翻墙地形（单地形）、奖励点 bonus、
  航点命令（WaypointVelocityCommand）。
  """
  cfg = unitree_go1_rough_env_cfg(play=play)
  apply_parkour_overrides(cfg)

  # Play / eval：关掉深度整帧随机失效（训练期才需要），否则部署演示时相机会
  # 随机"瞎掉"。
  if play:
    cfg.observations["actor"].terms["depth"].params["dropout_prob"] = 0.0

  # Play 覆盖：保持 curriculum=True（课程模式下生成器是确定性的——每种地形
  # 独占一列、行=难度递增；若置 False 会切成随机模式，patch 随机抽地形+难度，
  # 布局就乱了）。env 的初始难度行由 max_init_terrain_level 控制。
  if (
    play
    and cfg.scene.terrain is not None
    and cfg.scene.terrain.terrain_generator is not None
  ):
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.terrain_generator.num_rows = 10
    cfg.scene.terrain.terrain_generator.num_cols = 3
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
