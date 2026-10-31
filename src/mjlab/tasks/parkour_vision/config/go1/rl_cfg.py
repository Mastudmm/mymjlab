"""RL configuration for Unitree Go1 parkour vision task.

Mirrors the velocity_vision Go1 config; ``depth_shape`` is fixed to match the
80x80 depth camera pattern (the vision task's (1, 50, 80) is a known mismatch).

``depth_history_num`` MUST equal the number of depth frames the environment
stacks in the actor observation group: ``DepthActorCritic`` splits the actor
observation into ``[proprio | depth]`` using ``depth_vol * depth_history_num``,
so a mismatch silently shifts the split point and corrupts both branches. The
parkour env cfg sets ``history_length=3`` on the depth term -> keep this at 3.
"""

from dataclasses import dataclass

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_AMP_JOINT_POS_OFFSET
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

# AMP 数据类目前只存在于 velocity_amp 的 go1 rl_cfg（虽挂在 go1 目录下，
# 本身是机器人无关的通用配置）。跑酷直接复用，避免复制两份漂移；
# 若未来第二个机器人接 AMP，应把这两个类提升到 mjlab/rl/config.py。
from mjlab.tasks.velocity_amp.config.go1.rl_cfg import (
  RslRlAmpAlgorithmCfg,
  RslRlAmpOnPolicyRunnerCfg,
)


@dataclass
class DepthActorCriticCfg(RslRlModelCfg):
  depth_shape: tuple = (1, 80, 80)
  # Number of stacked depth frames consumed by the shared CNN; must match the
  # depth observation term's history_length (3 in the parkour env cfg).
  depth_history_num: int = 3
  obs_history_num: int = 1
  # width of the critic's height_scan branch. MUST equal the terrain_scan ray
  # count: size (0.8, 0.8) / resolution 0.05 -> 17 x 17 = 289.
  # 2026-10-26: this used to be omitted, so DepthActorCritic fell back to its
  # stale default of 256 and the critic's [proprio | scan] split was off by 33
  # dimensions — silently (no error, no crash), which polluted the value input.
  scan_dim: int = 289


def unitree_go1_parkour_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the Go1 parkour vision task."""
  actor_cfg = DepthActorCriticCfg(
    class_name="mjlab.tasks.velocity_vision.rl.modules:DepthActorCritic",
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    # Actor normalization stays OFF: DepthActorCritic applies the normalizer to
    # the WHOLE actor observation before splitting off the depth frames, so
    # enabling it would also per-pixel standardize the depth image (already
    # bounded in [-0.5, 0.5]); pixels that are constant most of the time get a
    # near-zero std and their noise gets amplified. One-line A/B if wanted.
    obs_normalization=False,
    hidden_dims=(512, 256, 128),
    activation="elu",
    depth_shape=(1, 80, 80),
    depth_history_num=3,
  )

  # Critic normalization ON (2026-10-25): the critic mixes physically unrelated
  # units — base_lin_vel (m/s), foot_contact_forces (N, up to hundreds),
  # height_scan (m, scaled 1/3), joint velocities — so the value MLP sees an
  # input whose scale spans orders of magnitude. Empirical normalization is the
  # cheap fix and matters more now that the milestone bonus raised the return
  # scale to ~0-70. The critic has no depth frames, so this is safe.
  critic_cfg = RslRlModelCfg(
    obs_normalization=True,
    hidden_dims=(512, 256, 128),
    activation="elu",
  )

  return RslRlOnPolicyRunnerCfg(
    actor=actor_cfg,
    critic=critic_cfg,
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # v8（2026-10-27）：0.01 -> 0.02。v7 的 Policy/mean_std 在 0.17~1.17
      # 间反复摆动（利用/探索失衡），提高熵奖励稳定探索。
      entropy_coef=0.02,
      num_learning_epochs=5,
      num_mini_batches=4,
      # v8：1e-3 -> 3e-4 作为续训起点（自适应 schedule 仍按 desired_kl
      # 自动调节，这里只影响首批更新，避免从 4400 续训时的冲击）。
      learning_rate=3.0e-4,
      schedule="adaptive",
      # γ=0.996（用户 2026-10-30）：视野 1/(1-γ) = 250 步 = 5s（v11 的 0.99
      # 只有 2s）。H2 里程碑在 450 步外的现值从 1.1 分升到 100×0.996^450 ≈
      # 16.5 分，翻完墙"继续走"的净现值由 ≈0 转正——削弱"领完就自杀"的
      # 动机。代价：GAE 方差变大、critic 回归更难，前中期 value loss 走高、
      # std 有膨胀风险，总轮数需求约 +10~30%（5h/5000 预算内可控）。
      gamma=0.996,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go1_parkour_vision",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=2200,
  )


def unitree_go1_parkour_amp_runner_cfg() -> RslRlAmpOnPolicyRunnerCfg:
  """Create AMP runner configuration for the Go1 parkour vision task.

  与 unitree_go1_parkour_runner_cfg 的差异只在两点：
  1. algorithm 换成 RslRlAmpAlgorithmCfg（class_name=AMPPPO + AMP 回放缓冲），
     PPO 超参逐项沿用 v12 纯 PPO 配置（entropy 0.02 / lr 3e-4 / gamma 0.996），
     保证除 AMP 外训练动力学一致，A/B 才有意义；
  2. 叠加 AMP 字段（数据集 / 观测布局 / 判别器 / 混合系数）。

  关键量纲说明（scale_rewards_by_dt=True）：
  - 任务奖励按 dt 缩放：行走期速率 ~1.3/s -> 每步 ~0.026；
  - 判别器风格奖励不乘 dt：amp_reward_coef=0.3 时满值 0.3/步；
  - 混合 r = lerp*r_task + (1-lerp)*r_style，故 lerp=0.98 名义 2%，
    实际风格占比 ~20% 量级。velocity_amp 的 0.8 在跑酷上等于让
    判别器主导训练（专家数据没有攀爬姿态），严禁照搬。
  """
  # Actor/critic 网络与纯 PPO 版逐字段一致（DepthActorCritic 需要的
  # obs_history_num 由 ParkourAmpOnPolicyRunner 构造时从 env cfg 注入）。
  actor_cfg = DepthActorCriticCfg(
    class_name="mjlab.tasks.velocity_vision.rl.modules:DepthActorCritic",
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    obs_normalization=False,
    hidden_dims=(512, 256, 128),
    activation="elu",
    depth_shape=(1, 80, 80),
    depth_history_num=3,
  )
  critic_cfg = RslRlModelCfg(
    obs_normalization=True,
    hidden_dims=(512, 256, 128),
    activation="elu",
  )

  return RslRlAmpOnPolicyRunnerCfg(
    actor=actor_cfg,
    critic=critic_cfg,
    algorithm=RslRlAmpAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.02,  # v12 同款
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,  # v12 同款（自适应 schedule 起点）
      schedule="adaptive",
      gamma=0.996,  # v12 同款
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    # ---- AMP 专属字段 ----
    # 专家数据：仓库根 dataset/*.json（CWD 必须在仓库根启动训练）。
    # 4 段平地 trot（前进/后退/左转/右转），FrameDim=43 与 env 侧
    # _build_amp_obs_43 布局逐维对应，FrameDuration=0.02 与
    # step_dt(0.005*4) 一致。
    amp_motion_files=["dataset/*.json"],
    amp_num_preload_transitions=1_000_000,
    # 风格奖励幅度系数：只缩放 r_style 幅度，不影响判别器训练强度。
    amp_reward_coef=0.1,
    # 混合系数：0.98 = 名义 98% 任务 + 2% 风格；实际占比见 docstring。
    amp_task_reward_lerp=0.92,
    # 判别器结构沿用 velocity_amp。
    amp_discr_hidden_dims=(512, 256),
    # 启动即核对 env AMP 观测维度/顺序 vs 数据集，错位立即报错。
    amp_preflight_check=True,
    amp_preflight_strict=True,
    amp_preflight_max_files=8,
    # 43D 布局（36D 基础 + 机体线/角速度 + 射线离地高度）。
    amp_expected_obs_dim=43,
    # 数据集存绝对关节角；用默认姿态偏移转成相对值，与 env 侧
    # joint_pos_rel（减 default_joint_pos）语义对齐。GO1_AMP_JOINT_POS_OFFSET
    # 与 Go1 INIT_STATE 默认关节角逐项一致（已实证）。
    amp_joint_pos_mode="absolute",
    amp_joint_pos_offset=list(GO1_AMP_JOINT_POS_OFFSET),
    # ---- runner 常规字段 ----
    experiment_name="go1_parkour_vision_amp",
    save_interval=200,
    num_steps_per_env=24,
    # 训练预算（7h 规则，用户 2026-10-30 决定）：与实际训练习惯对齐取
    # 5000（cfg 里的 2200 只是 v0.1 出厂默认，实际推送一直 CLI 覆盖 5000）。
    # 折算：PPO v12 5000 轮 ≈ 5h（集群实测基准）；AMP 每轮额外做判别器
    # 前向/反向 + AMP 回放缓冲采样，估 +15~30% -> 5000 轮 ≈ 5.7~6.5h，
    # 7h 墙钟内。save_interval=200 保证被砍时留有 checkpoint；
    # 若集群首轮实测 iter 时间 > ~4.8s（7h/5250），下调到 4500。
    max_iterations=5000,
  )
