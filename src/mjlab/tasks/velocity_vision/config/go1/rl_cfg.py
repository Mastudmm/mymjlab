"""RL configuration for Unitree Go1 velocity task."""

from dataclasses import dataclass
from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlModelCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class DepthActorCriticCfg(RslRlModelCfg):
    depth_shape: tuple = (1, 50, 80)
    obs_history_num: int = 1
    # scan_shape is automatically derived


def unitree_go1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go1 velocity task."""
  actor_cfg = DepthActorCriticCfg(
      class_name="mjlab.tasks.velocity_vision.rl.modules:DepthActorCritic",
      init_noise_std=1.0,
      obs_normalization=False,
      hidden_dims=(512, 256, 128),
      activation="elu",
      depth_shape=(1, 50, 80),
  )

  critic_cfg = RslRlModelCfg(
      obs_normalization=False,
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
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go1_velocity_vision",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=2200,
  )
