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

本类只做两件 mjlab 特有的事（AMP 全部逻辑都在 rsl_rl 子模块里）：
1. 构造前把 env actor 观测组的 history_length 注入 train_cfg["actor"]
   ["obs_history_num"]：DepthActorCritic 依赖它切分 [proprio | depth]，
   缺失/错位会**静默**损坏两条分支（与 velocity_vision
   VelocityOnPolicyRunner 的行为保持一致）；
2. 保存 checkpoint 时同步导出 policy ONNX（判别器只用于训练，不导出）。
"""

import os

import wandb
from rsl_rl.runners import AmpOnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)


class ParkourAmpOnPolicyRunner(AmpOnPolicyRunner):
  """AmpOnPolicyRunner + parkour 的观测历史注入与 ONNX 导出。"""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ):
    # 动态获取环境中 actor 观测组的 history_length，传给 policy。
    # 与 VelocityOnPolicyRunner 逐行一致：组级 history_length 优先；
    # 为 None（历史设在具体项上）时回退读 joint_pos 项的值。
    if "actor" in train_cfg and "actor" in env.unwrapped.cfg.observations:
      history_length = env.unwrapped.cfg.observations["actor"].history_length
      if history_length is None:
        history_length = (
          env.unwrapped.cfg.observations["actor"].terms["joint_pos"].history_length
        )
      train_cfg["actor"]["obs_history_num"] = history_length
    super().__init__(env, train_cfg, log_dir, device)

  def save(self, path: str, infos=None):
    """Save checkpoint (含判别器状态，AMPPPO.save 已加入) + policy ONNX 导出."""
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
