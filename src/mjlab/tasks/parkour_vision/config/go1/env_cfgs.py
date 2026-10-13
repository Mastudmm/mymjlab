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
  :func:`apply_parkour_overrides`）：parkour 地形、奖励点 bonus、纯 x command。
  """
  cfg = unitree_go1_rough_env_cfg(play=play)
  apply_parkour_overrides(cfg)

  # Play 覆盖：关闭课程，行数减半便于可视化；三种地形各保留一列。
  if (
    play
    and cfg.scene.terrain is not None
    and cfg.scene.terrain.terrain_generator is not None
  ):
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_rows = 6
    cfg.scene.terrain.terrain_generator.num_cols = 3
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
