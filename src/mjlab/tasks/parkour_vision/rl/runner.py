"""AMP-enabled runner for the parkour vision task.

AMP（Adversarial Motion Priors）奖励在本仓库中的真实形态（重要）：
- 它**不是**环境奖励项（不要在 rewards.py / parkour_env_cfg.py 里加
  "amp_reward" 之类的 RewardTermCfg，velocity_amp 的 env cfg 注释明确禁止）；
- RslRlVecEnvWrapper 每步额外产出 43 维 AMP 观测（joint_pos_rel / joint_vel /
  足端位置 / 机体线速度 / 角速度 / 射线离地高度，布局与 dataset/*.json 一致）；
- rsl_rl 的 AmpOnPolicyRunner 在训练循环里用判别器对 (s_t, s_{t+1}) 转换
  打风格分，再与任务奖励线性混合后交给 PPO：
      r = lerp * r_task + (1 - lerp) * r_style
  其中 r_style = amp_reward_coef * clamp(1 - (D-1)^2 / 4, min=0)。
- 注意量纲：任务奖励经 scale_rewards_by_dt=True 缩放（rate * dt ≈ 0.02 倍），
  r_style 不乘 dt。因此 parkour 的 lerp 取 0.98（名义 2%），实际风格占比
  约 20% 量级；若照搬 velocity_amp 的 0.8，判别器会主导训练。

本模块为跑酷接入末帧采集和归一化统计持久化。
保存 checkpoint 时同步导出 policy ONNX（判别器只用于训练）。
"""

import warnings
from pathlib import Path

import torch
import wandb
from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class ParkourAmpOnPolicyRunner(AmpOnPolicyRunner):
  """Capture AMP terminal transitions and export the Parkour policy."""

  env: RslRlVecEnvWrapper

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    self.env.set_terminal_amp_capture(True)
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      self.env.set_terminal_amp_capture(False)

  def save(self, path: str, infos=None):
    """Save checkpoint (含判别器状态，AMPPPO.save 已加入) + policy ONNX 导出."""
    super().save(path, infos)
    _export_parkour_policy(self, path)


class ParkourAMPPPO(AMPPPO):
  """Keep the discriminator's input statistics with Parkour AMP checkpoints."""

  def save(self) -> dict:
    saved = super().save()
    normalizer = self.amp_normalizer
    if normalizer is not None:
      saved["amp_normalizer_state"] = {
        "mean": torch.from_numpy(normalizer.mean.copy()),
        "var": torch.from_numpy(normalizer.var.copy()),
        "count": float(normalizer.count),
      }
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    normalizer = self.amp_normalizer
    if normalizer is not None and (load_cfg is None or load_cfg.get("discriminator")):
      state = loaded_dict.get("amp_normalizer_state")
      if state is None:
        warnings.warn(
          "This Parkour AMP checkpoint has no input normalization statistics. "
          "The discriminator will use the current normalizer statistics.",
          RuntimeWarning,
          stacklevel=2,
        )
      else:
        # The running normalizer uses NumPy even if torch.load mapped to CUDA.
        normalizer.mean = state["mean"].cpu().numpy().copy()
        normalizer.var = state["var"].cpu().numpy().copy()
        normalizer.count = float(state["count"])
    return load_iteration


class ParkourOnPolicyRunner(MjlabOnPolicyRunner):
  """Save and export the policy for the pure PPO Parkour task."""

  def save(self, path: str, infos=None):
    super().save(path, infos)
    _export_parkour_policy(self, path)


def _export_parkour_policy(runner: OnPolicyRunner, checkpoint_path: str) -> None:
  run_dir = Path(checkpoint_path).parent
  filename = f"{run_dir.name}.onnx"
  try:
    runner.export_policy_to_onnx(str(run_dir), filename)
    wandb_active = runner.logger.logger_type == "wandb" and wandb.run is not None
    run_name = wandb.run.name if wandb_active else "local"
    onnx_path = str(run_dir / filename)
    metadata = get_base_metadata(runner.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if wandb_active and runner.cfg["upload_model"]:
      wandb.save(onnx_path, base_path=str(run_dir))
  except Exception as e:
    print(f"[WARN] ONNX export failed (training continues): {e}")
