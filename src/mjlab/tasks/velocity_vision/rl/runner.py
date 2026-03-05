import os

import wandb
import rsl_rl.runners.on_policy_runner as rsl_rl_runner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.tasks.velocity_vision.rl import modules
rsl_rl_runner.modules = modules  # Inject the entire module to try and trick resolution if it looks up by module
# But rsl_rl uses `resolve_callable`.
# We need to make sure "DepthActorCritic" is importable from where rsl_rl tries to find it,
# OR we update the config to use the full path "mjlab.tasks.velocity_vision.rl.modules:DepthActorCritic"

class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env: RslRlVecEnvWrapper, train_cfg: dict, log_dir=None, device="cpu"):
      # 动态获取环境中的 history_length，传给 policy
      if "actor" in train_cfg and "actor" in env.unwrapped.cfg.observations:
          # Try to get the group history length; if it's None (because we set per-term length), read it from a concrete term like joint_pos
          history_length = env.unwrapped.cfg.observations["actor"].history_length
          if history_length is None:
              history_length = env.unwrapped.cfg.observations["actor"].terms["joint_pos"].history_length
          train_cfg["actor"]["obs_history_num"] = history_length
      super().__init__(env, train_cfg, log_dir, device)

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    try:
      self.export_policy_to_onnx(policy_path, filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      onnx_path = os.path.join(policy_path, filename)
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(onnx_path, metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
