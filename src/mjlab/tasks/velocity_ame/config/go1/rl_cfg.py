"""RSL-RL runner config for AME on Unitree Go1.

Extends mjlab's base RSL-RL configs with the AME attention-encoder model
fields and a ``symmetry_cfg`` slot on the algorithm (kept ``None`` for the
first port; symmetry augmentation is a follow-up once a Go1 mirror map is
written).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.velocity_ame.config.variants import AmePhase, resolve_task_spec

AME_ACTOR_CLASS = "mjlab.tasks.velocity_ame.rl.modules:AmeActorModel"
AME_CRITIC_CLASS = "mjlab.tasks.velocity_ame.rl.modules:AmeCriticModel"


@dataclass
class AmeModelCfg(RslRlModelCfg):
  """Actor/critic config for the AME attention-based map encoder."""

  map_latent_dim: int = 64
  """Dimension of the attention query/value embeddings."""
  num_attention_heads: int = 16
  """Number of multi-head attention heads."""
  proprio_key: str = "proprio"
  """Proprioceptive key used in nested-obs mode (unused for Go1 multi-group)."""
  terrain_key: str = "terrain_points"
  """Terrain point-cloud key used in nested-obs mode."""
  proprio_obs_groups: tuple[str, ...] = ()
  """Observation groups concatenated into the proprioceptive input."""
  terrain_obs_group: str | None = None
  """Observation group holding the [B, H, W, 3] terrain point cloud."""
  attention_dump_key: str = "attention_weights"
  """Key under which attention weights are dumped for visualization."""
  terrain_input_mode: str = "z"
  """CNN input channels: ``"z"`` (height only) or ``"xyz"`` (full point)."""
  concat_coords_post_cnn: bool = True
  """Concatenate token coordinates to CNN features before attention."""
  cnn_downsample: bool = False
  """Stride the first CNN layer to shorten the attention token sequence."""
  attach_global: bool = False
  """Add a global-context branch that pools tokens into the query."""
  encoder_variant: str = "paper"
  """Encoder variant label recorded in attention metadata."""


@dataclass
class AmePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO config with an optional symmetry slot for AME."""

  symmetry_cfg: dict[str, Any] | None = None
  """Symmetry data-augmentation config. ``None`` disables symmetry (default)."""


@dataclass
class AmeRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config specialized for AME."""

  onnx_export_policy: Literal["never", "final_only", "every_save"] = "final_only"


def unitree_go1_ame_runner_cfg(phase: AmePhase) -> AmeRunnerCfg:
  """Create the RL runner config for an AME Go1 training phase."""

  task_spec = resolve_task_spec(phase)
  entropy_coef = 0.005 if phase == "base" else 0.002
  distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
  }
  return AmeRunnerCfg(
    actor=AmeModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      proprio_obs_groups=("actor_proprio",),
      terrain_obs_group="actor_terrain",
      distribution_cfg=distribution_cfg,
      class_name=AME_ACTOR_CLASS,
      terrain_input_mode="xyz",
      concat_coords_post_cnn=False,
      cnn_downsample=True,
      attach_global=False,
      encoder_variant="g1",
    ),
    critic=AmeModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      proprio_obs_groups=("critic_proprio",),
      terrain_obs_group="critic_terrain",
      distribution_cfg=None,
      class_name=AME_CRITIC_CLASS,
      terrain_input_mode="xyz",
      concat_coords_post_cnn=False,
      cnn_downsample=True,
      attach_global=False,
      encoder_variant="g1",
    ),
    algorithm=AmePpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=3,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=entropy_coef,
      desired_kl=0.01,
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=False,
      optimizer="adam",
      share_cnn_encoders=True,
      symmetry_cfg=None,
    ),
    experiment_name=task_spec.experiment_name,
    logger="tensorboard",
    num_steps_per_env=24,
    max_iterations=15000 if phase == "base" else 3200,
    obs_groups={
      "actor": ("actor_proprio", "actor_terrain"),
      "critic": ("critic_proprio", "critic_terrain"),
    },
    save_interval=100,
    wandb_project="ame_go1",
  )
