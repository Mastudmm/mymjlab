from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity_amp.rl import VelocityAmpOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import unitree_g1_amp_runner_cfg

register_mjlab_task(
  task_id="Mjlab-VelocityAmp-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_amp_runner_cfg(),
  runner_cls=VelocityAmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-VelocityAmp-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_amp_runner_cfg(),
  runner_cls=VelocityAmpOnPolicyRunner,
)
