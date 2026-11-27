"""Parkour actor using RSL-RL's active model interface."""

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import CNN, MLP, HiddenState
from tensordict import TensorDict
from torch import nn

from mjlab.managers.observation_manager import ObservationManager

VISUAL_FEATURES = 32
HISTORY_FEATURES = 32
LEGACY_PARKOUR_TASK = "Mjlab-ParkourVision-LegacyMLP-Unitree-Go1"


class ParkourVisionActor(MLPModel):
  """Encode stacked depth and proprioception before the Gaussian action head."""

  _needs_obs_manager = True

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    *,
    observation_manager: ObservationManager,
    depth_shape: tuple[int, int],
    **kwargs,
  ) -> None:
    if tuple(obs_groups[obs_set]) != ("actor",):
      raise ValueError("ParkourVisionActor requires the flat 'actor' group.")
    terms = observation_manager.active_terms["actor"]
    if not terms or terms[-1] != "depth":
      raise ValueError("ParkourVisionActor requires 'depth' as the last actor term.")

    current_indices: list[int] = []
    proprio_dim = 0
    term_shapes = observation_manager.group_obs_term_dim["actor"]
    for term, shape in zip(terms[:-1], term_shapes[:-1], strict=True):
      history = max(1, observation_manager.get_term_cfg("actor", term).history_length)
      if len(shape) != 1 or shape[0] < history or shape[0] % history:
        raise ValueError(f"Invalid Parkour history layout for '{term}': {shape}.")
      width = shape[0] // history
      proprio_dim += shape[0]
      # Each term stores its own oldest-to-newest history before concatenation.
      current_indices.extend(range(proprio_dim - width, proprio_dim))

    depth_frames = max(
      1, observation_manager.get_term_cfg("actor", "depth").history_length
    )
    height, width = depth_shape
    depth_dim = depth_frames * height * width
    if (
      not current_indices
      or height <= 0
      or width <= 0
      or term_shapes[-1] != (depth_dim,)
      or obs["actor"].shape[-1] != proprio_dim + depth_dim
    ):
      raise ValueError(
        f"Invalid Parkour actor layout: proprio={proprio_dim}, "
        f"depth shape={depth_frames}x{height}x{width}, "
        f"depth term={term_shapes[-1]}, actor width={obs['actor'].shape[-1]}."
      )

    # MLPModel constructs its policy head using _get_latent_dim().
    self.latent_dim = len(current_indices) + HISTORY_FEATURES + VISUAL_FEATURES
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    self.encoder = _ParkourObservationEncoder(
      proprio_dim, current_indices, depth_frames, depth_shape
    )

  def _get_latent_dim(self) -> int:
    return self.latent_dim

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    # MLPModel.forward calls this hook for both rollout sampling and PPO updates.
    return self.encoder(super().get_latent(obs, masks, hidden_state))

  def as_jit(self) -> nn.Module:
    return _ParkourActorExport(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _ParkourActorExport(self)

  def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
    if "shared_cnn.0.weight" in state_dict:
      raise RuntimeError(
        "This checkpoint uses the old flat Parkour MLP. "
        f"Replay it with {LEGACY_PARKOUR_TASK}. "
        "Start a new run to train the CNN actor."
      )
    return super().load_state_dict(state_dict, strict=strict, assign=assign)


class _ParkourObservationEncoder(nn.Module):
  """Share the same observation split and feature transforms with exports."""

  def __init__(
    self,
    proprio_dim: int,
    current_indices: list[int],
    depth_frames: int,
    depth_shape: tuple[int, int],
  ) -> None:
    super().__init__()
    self.proprio_dim = proprio_dim
    self.depth_frames = depth_frames
    self.depth_height, self.depth_width = depth_shape
    self.register_buffer(
      "current_indices",
      torch.tensor(current_indices, dtype=torch.long),
      persistent=False,
    )
    # Stack time as channels and downsample early to fit large PPO minibatches.
    self.cnn = CNN(
      input_dim=depth_shape,
      input_channels=depth_frames,
      output_channels=(8, 16, 16),
      kernel_size=(5, 3, 3),
      stride=(4, 2, 2),
      padding="zeros",
    )
    cnn_dim = self.cnn.output_dim
    assert isinstance(cnn_dim, int)
    self.visual_projection = nn.Linear(cnn_dim, VISUAL_FEATURES)
    self.history_encoder = MLP(proprio_dim, HISTORY_FEATURES, (128,))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    proprio = obs[:, : self.proprio_dim]
    current = proprio.index_select(1, self.current_indices)
    depth = obs[:, self.proprio_dim :].reshape(
      -1, self.depth_frames, self.depth_height, self.depth_width
    )
    vision = self.visual_projection(self.cnn(depth))
    history = self.history_encoder(proprio)
    return torch.cat((current, history, vision), dim=-1)


class _ParkourActorExport(nn.Module):
  """Deterministic policy on the same flat observation vector as training."""

  is_recurrent: bool = False
  input_names = ["obs"]
  output_names = ["actions"]

  def __init__(self, actor: ParkourVisionActor) -> None:
    super().__init__()
    self.input_size = actor.obs_dim
    self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
    self.encoder = copy.deepcopy(actor.encoder)
    self.mlp = copy.deepcopy(actor.mlp)
    self.deterministic_output = (
      actor.distribution.as_deterministic_output_module()
      if actor.distribution is not None
      else nn.Identity()
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    latent = self.encoder(self.obs_normalizer(obs))
    return self.deterministic_output(self.mlp(latent))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @torch.jit.export
  def reset(self) -> None:
    pass
