"""Task registration for the Go1 parkour vision task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity_vision.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_go1_parkour_env_cfg
from .rl_cfg import unitree_go1_parkour_runner_cfg

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
