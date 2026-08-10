"""Register AME Unitree Go1 velocity tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity_ame.config.go1.env_cfgs import unitree_go1_ame_env_cfg
from mjlab.tasks.velocity_ame.config.go1.rl_cfg import unitree_go1_ame_runner_cfg
from mjlab.tasks.velocity_ame.config.variants import iter_task_specs
from mjlab.tasks.velocity_ame.rl import AmeOnPolicyRunner

for _task_spec in iter_task_specs():
  register_mjlab_task(
    task_id=_task_spec.task_id,
    env_cfg=unitree_go1_ame_env_cfg(phase=_task_spec.phase, play=False),
    play_env_cfg=unitree_go1_ame_env_cfg(phase=_task_spec.phase, play=True),
    rl_cfg=unitree_go1_ame_runner_cfg(phase=_task_spec.phase),
    runner_cls=AmeOnPolicyRunner,
  )

del _task_spec
