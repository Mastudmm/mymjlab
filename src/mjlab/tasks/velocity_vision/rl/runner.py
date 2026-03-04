import os

import wandb
import rsl_rl.runners.on_policy_runner as rsl_rl_runner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity_vision.rl.modules import DepthActorCritic
rsl_rl_runner.DepthActorCritic = DepthActorCritic  # Inject to eval scope

from mjlab.tasks.velocity_vision.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env: RslRlVecEnvWrapper, train_cfg: dict, log_dir=None, device="cpu"):
      # 动态获取环境中的 history_length，传给 policy
      if "policy" in train_cfg and "policy" in env.unwrapped.cfg.observations:
          # Try to get the group history length; if it's None (because we set per-term length), read it from a concrete term like joint_pos
          history_length = env.unwrapped.cfg.observations["policy"].history_length
          if history_length is None:
              history_length = env.unwrapped.cfg.observations["policy"].terms["joint_pos"].history_length
          train_cfg["policy"]["obs_history_num"] = history_length
      super().__init__(env, train_cfg, log_dir, device)

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    if self.alg.policy.actor_obs_normalization:
      normalizer = self.alg.policy.actor_obs_normalizer
    else:
      normalizer = None
    export_velocity_policy_as_onnx(
      self.alg.policy,
      normalizer=normalizer,
      path=policy_path,
      filename=filename,
    )
    # Attach metadata (use "local" for run_path if not using wandb)
    run_name = wandb.run.name if self.logger_type == "wandb" and wandb.run else "local"
    attach_onnx_metadata(
      self.env.unwrapped,
      run_name,  # type: ignore
      path=policy_path,
      filename=filename,
    )
    if self.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
