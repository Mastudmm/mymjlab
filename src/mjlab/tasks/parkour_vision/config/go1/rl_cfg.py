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

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


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
