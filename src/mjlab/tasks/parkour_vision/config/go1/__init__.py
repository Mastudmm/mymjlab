"""Task registration for the Go1 parkour vision task."""

from mjlab.tasks.parkour_vision.rl import ParkourAmpOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity_vision.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_go1_parkour_env_cfg
from .rl_cfg import (
  unitree_go1_parkour_amp_runner_cfg,
  unitree_go1_parkour_runner_cfg,
)

# Parkour-terrain variant: straight-line obstacle courses (step stones /
# platforms / gaps) with milestone reward points, depth camera, and a pure
# forward (x-only) command.
register_mjlab_task(
  task_id="Mjlab-ParkourVision-Unitree-Go1",
  env_cfg=unitree_go1_parkour_env_cfg(),
  play_env_cfg=unitree_go1_parkour_env_cfg(play=True),
  rl_cfg=unitree_go1_parkour_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# AMP 变体（用户决定 2026-10-30）：env cfg 与原任务完全共享（AMP 不需要
# 任何环境侧观测/奖励项，判别器奖励在 runner 层混合），仅换 AMP runner
# 与 AMP 训练配置。原任务保留作 A/B 对照。
# 注意：AMP 变体从零训练——旧纯 PPO checkpoint 缺判别器状态且 optimizer
# 参数组不同，直接 resume 会 KeyError。
register_mjlab_task(
  task_id="Mjlab-ParkourVisionAMP-Unitree-Go1",
  env_cfg=unitree_go1_parkour_env_cfg(),
  play_env_cfg=unitree_go1_parkour_env_cfg(play=True),
  rl_cfg=unitree_go1_parkour_amp_runner_cfg(),
  runner_cls=ParkourAmpOnPolicyRunner,
)
