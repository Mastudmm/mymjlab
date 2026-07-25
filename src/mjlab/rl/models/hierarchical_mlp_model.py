"""Hierarchical MLP model with optional proprio history / terrain encoders.

Extends ``MLPModel`` to split the flattened observation into:
- proprio current frame (direct)
- proprio history -> history_encoder (e.g. ->64)
- height_scan (with its own history) -> terrain_encoder (e.g. ->64, critic only)
- foot current frame (direct, critic only)
and concatenates them as the main MLP input.

Term layout (offset/frame_dim/history per term) is auto-extracted from the
observation_manager, so changing history length or adding/removing terms needs
no manual dimension edits.

A subclass declares it needs the manager via class attr ``_needs_obs_manager =
True``; ``construct_algorithm`` then injects ``observation_manager=`` kwarg.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict


@dataclass
class _TermLayout:
  """Per-term slice metadata within the flattened observation."""

  name: str
  offset: int  # start index in flat obs
  block_dim: int  # hist * frame_dim
  frame_dim: int  # single-frame dim
  hist: int  # >= 1
  kind: str  # "proprio" | "terrain" | "foot"


class HierarchicalMLPModel(MLPModel):
  """MLPModel + optional proprio history encoder / terrain encoder.

  Layout is auto-extracted from observation_manager. Actor (no height_scan)
  builds no terrain_encoder; critic does. Inheritable for future vision encoders.
  """

  _needs_obs_manager = True

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    *,
    observation_manager: Any | None = None,
    history_encoder_dims: tuple[int, ...] = (128, 64),
    terrain_encoder_dims: tuple[int, ...] = (256, 128, 64),
    terrain_term_names: tuple[str, ...] = ("height_scan",),
    foot_term_names: tuple[str, ...] = (
      "foot_height",
      "foot_air_time",
      "foot_contact",
      "foot_contact_forces",
    ),
    **kwargs,
  ) -> None:
    # Fallback: no manager -> degrade to plain MLPModel.
    if observation_manager is None:
      super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
      self._layout_terms: list[_TermLayout] = []
      self.history_encoder: nn.Module | None = None
      self.terrain_encoder: nn.Module | None = None
      return

    # Phase 1: pure-data attrs (safe before nn.Module.__init__).
    self._layout_terms = self._extract_layout(
      observation_manager, obs_set, terrain_term_names, foot_term_names
    )
    self._history_encoder_dims = tuple(history_encoder_dims)
    self._terrain_encoder_dims = tuple(terrain_encoder_dims)
    self._has_history_enc = any(
      t.kind == "proprio" and t.hist > 1 for t in self._layout_terms
    )
    self._has_terrain_enc = any(t.kind == "terrain" for t in self._layout_terms)
    self._proprio_history_input_dim = sum(
      t.block_dim - t.frame_dim
      for t in self._layout_terms
      if t.kind == "proprio" and t.hist > 1
    )
    self._terrain_input_dim = sum(
      t.block_dim for t in self._layout_terms if t.kind == "terrain"
    )
    self._current_dim = sum(
      t.frame_dim for t in self._layout_terms if t.kind in ("proprio", "foot")
    )

    # Precompute flat column indices for vectorized slicing (no python loop
    # in forward). Each list is a plain python list[int] used as x[:, idx].
    self._proprio_hist_idx: list[int] = []
    self._terrain_idx: list[int] = []
    self._proprio_cur_idx: list[int] = []
    self._foot_cur_idx: list[int] = []
    for t in self._layout_terms:
      if t.kind == "terrain":
        self._terrain_idx.extend(range(t.offset, t.offset + t.block_dim))
      elif t.kind == "foot":
        self._foot_cur_idx.extend(
          range(t.offset + t.block_dim - t.frame_dim, t.offset + t.block_dim)
        )
      else:  # proprio
        if t.hist > 1:
          self._proprio_hist_idx.extend(range(t.offset, t.offset + t.block_dim - t.frame_dim))
        self._proprio_cur_idx.extend(
          range(t.offset + t.block_dim - t.frame_dim, t.offset + t.block_dim)
        )

    # Phase 2: super().__init__ triggers _get_obs_dim + _get_latent_dim.
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

    # Phase 3: encoders (after super so submodules register correctly).
    activation = kwargs.get("activation", "elu")
    self.history_encoder = (
      MLP(
        self._proprio_history_input_dim,
        self._history_encoder_dims[-1],
        self._history_encoder_dims[:-1],
        activation,
      )
      if self._has_history_enc and self._proprio_history_input_dim > 0
      else None
    )
    self.terrain_encoder = (
      MLP(
        self._terrain_input_dim,
        self._terrain_encoder_dims[-1],
        self._terrain_encoder_dims[:-1],
        activation,
      )
      if self._has_terrain_enc and self._terrain_input_dim > 0
      else None
    )

  def _extract_layout(
    self,
    obs_manager,
    group_name: str,
    terrain_term_names: tuple[str, ...],
    foot_term_names: tuple[str, ...],
  ) -> list[_TermLayout]:
    """Auto-build per-term slice metadata from the observation_manager."""
    names = obs_manager.active_terms[group_name]
    dims = obs_manager.group_obs_term_dim[group_name]
    cfgs = obs_manager._group_obs_term_cfgs[group_name]

    layout: list[_TermLayout] = []
    offset = 0
    for name, dim_tuple, cfg in zip(names, dims, cfgs, strict=False):
      block_dim = int(np.prod(dim_tuple))
      hist = max(1, getattr(cfg, "history_length", 0) or 0)
      if block_dim % hist != 0:
        raise ValueError(
          f"Term '{name}' block_dim={block_dim} not divisible by hist={hist}; "
          f"check flatten_history_dim / shape {dim_tuple}."
        )
      frame_dim = block_dim // hist
      if name in terrain_term_names:
        kind = "terrain"
      elif name in foot_term_names:
        kind = "foot"
      else:
        kind = "proprio"
      layout.append(_TermLayout(name, offset, block_dim, frame_dim, hist, kind))
      offset += block_dim
    return layout

  def _get_latent_dim(self) -> int:
    """Main MLP input dim = current frames + encoder output dims."""
    if not getattr(self, "_layout_terms", None):
      return super()._get_latent_dim()
    dim = self._current_dim
    if self._has_history_enc:
      dim += self._history_encoder_dims[-1]
    if self._has_terrain_enc:
      dim += self._terrain_encoder_dims[-1]
    return dim

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    """Slice obs per term, encode history/terrain, concat with current frames."""
    if not self._layout_terms:
      return super().get_latent(obs, masks, hidden_state)
    obs = (
      unpad_trajectories(obs, masks)
      if masks is not None and not self.is_recurrent
      else obs
    )
    x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
    x = self.obs_normalizer(x)

    # Vectorized slicing: no python loop, minimal cat.
    parts: list[torch.Tensor] = []
    if self.history_encoder is not None and self._proprio_hist_idx:
      parts.append(self.history_encoder(x[:, self._proprio_hist_idx]))
    if self.terrain_encoder is not None and self._terrain_idx:
      parts.append(self.terrain_encoder(x[:, self._terrain_idx]))
    parts.append(x[:, self._proprio_cur_idx])
    if self._foot_cur_idx:
      parts.append(x[:, self._foot_cur_idx])
    return torch.cat(parts, dim=-1)

  def as_jit(self) -> nn.Module:
    if not self._layout_terms:
      return super().as_jit()
    return _TorchHierarchicalMLPModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    if not self._layout_terms:
      return super().as_onnx(verbose)
    return _OnnxHierarchicalMLPModel(self, verbose)


class _TorchHierarchicalMLPModel(nn.Module):
  """JIT/ONNX export wrapper: encoders + slicing + mlp self-contained."""

  def __init__(self, model: HierarchicalMLPModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.history_encoder = (
      copy.deepcopy(model.history_encoder)
      if model.history_encoder is not None
      else None
    )
    self.terrain_encoder = (
      copy.deepcopy(model.terrain_encoder)
      if model.terrain_encoder is not None
      else None
    )
    self.mlp = copy.deepcopy(model.mlp)
    self.deterministic_output = (
      model.distribution.as_deterministic_output_module()
      if model.distribution is not None
      else nn.Identity()
    )

    # Reuse the precomputed flat column indices from the parent model
    # (vectorized slicing, no python loop in forward).
    self._proprio_hist_idx = model._proprio_hist_idx
    self._terrain_idx = model._terrain_idx
    self._proprio_cur_idx = model._proprio_cur_idx
    self._foot_cur_idx = model._foot_cur_idx
    self._has_hist = model.history_encoder is not None
    self._has_terr = model.terrain_encoder is not None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    parts: list[torch.Tensor] = []
    if self._has_hist and self._proprio_hist_idx:
      parts.append(self.history_encoder(x[:, self._proprio_hist_idx]))
    if self._has_terr and self._terrain_idx:
      parts.append(self.terrain_encoder(x[:, self._terrain_idx]))
    parts.append(x[:, self._proprio_cur_idx])
    if self._foot_cur_idx:
      parts.append(x[:, self._foot_cur_idx])
    latent = torch.cat(parts, dim=-1)
    return self.deterministic_output(self.mlp(latent))

  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxHierarchicalMLPModel(_TorchHierarchicalMLPModel):
  """ONNX export wrapper sharing forward with the JIT version."""

  is_recurrent: bool = False

  def __init__(self, model: HierarchicalMLPModel, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose
    self.input_size = model.obs_dim

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
