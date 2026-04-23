"""RL configuration for Unitree Go1 velocity_amp task."""

from dataclasses import dataclass, field

from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_AMP_JOINT_POS_OFFSET
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class RslRlAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Config for AMPPPO algorithm on top of PPO settings."""

  class_name: str = "rsl_rl.algorithms.amp_ppo.AMPPPO"
  amp_replay_buffer_size: int = 100_000


@dataclass
class RslRlAmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config extended with AMP data and discriminator parameters."""

  class_name: str = "AmpOnPolicyRunner"

  amp_num_preload_transitions: int = 1_000_000
  amp_motion_files: list[str] = field(default_factory=lambda: ["dataset/*.json"])
  amp_reward_coef: float = 0.3
  amp_discr_hidden_dims: tuple[int, ...] = (1024, 512)
  amp_task_reward_lerp: float = 0.6
  amp_preflight_check: bool = True
  amp_preflight_strict: bool = True
  amp_preflight_max_files: int = 8
  # AMP observation layout selector.
  #
  # 36D (legacy, default for old datasets/checkpoints):
  #   [0:12]   joint_pos_rel      (FR,FL,RR,RL each [hip,thigh,calf])
  #   [12:24]  joint_vel_rel      (same joint order as above)
  #   [24:36]  foot_pos_b_xyz     (FR,FL,RR,RL each [x,y,z], in body/base frame)
  #
  # 43D (extended):
  #   [0:36]   same as legacy 36D
  #   [36:39]  root_lin_vel_b     ([vx, vy, vz] in body/base frame)
  #   [39:42]  root_ang_vel_b     ([wx, wy, wz] in body/base frame)
  #   [42:43]  root_z             (root height term, see vecenv_wrapper implementation)
  amp_expected_obs_dim: int = 43
  amp_joint_pos_mode: str = "absolute"
  amp_joint_pos_offset: list[float] = field(default_factory=lambda: list(GO1_AMP_JOINT_POS_OFFSET))


def unitree_go1_amp_runner_cfg() -> RslRlAmpOnPolicyRunnerCfg:
  """Create AMP runner configuration for Unitree Go1 velocity_amp task."""
  return RslRlAmpOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlAmpAlgorithmCfg(
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
    experiment_name="go1_velocity_amp",
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=2000,
  )
