"""Per-terrain velocity command for AME.

Extends :class:`UniformVelocityCommand` to override the command sampling
ranges per sub-terrain (e.g. constrain lateral velocity on narrow beams),
while keeping heading random for generalization.

The override is applied at the end of ``_resample_command``, after the base
class has sampled global ranges and applied forward/standing/world logic, so
it takes precedence over those per-env specializations. Standing envs are
still zeroed every step by the base ``_update_command`` and are unaffected.

Terrain lookup relies on ``terrain.terrain_types`` (the sub-terrain column
index per env). In curriculum mode ``num_cols == len(sub_terrains)``, so the
column index maps directly to ``list(terrain_generator.sub_terrains.keys())``.
Command resample runs after reset events (incl. ``randomize_terrain``), so
``terrain_types`` already reflects the post-reset terrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class TerrainCommandOverride:
  """Per-sub-terrain command range overrides.

  Each field replaces the global range when sampling the command for envs
  currently on that sub-terrain. ``None`` keeps the global range.
  """

  lin_vel_x: tuple[float, float] | None = None
  lin_vel_y: tuple[float, float] | None = None
  ang_vel_z: tuple[float, float] | None = None
  heading: tuple[float, float] | None = None
  """Heading target range (world yaw). Only used when ``heading_command=True``;
  overrides ``ranges.heading`` for envs on this sub-terrain."""


@dataclass(kw_only=True)
class AmeVelocityCommandCfg(UniformVelocityCommandCfg):
  """:class:`UniformVelocityCommandCfg` with per-sub-terrain range overrides."""

  per_terrain_overrides: dict[str, TerrainCommandOverride] = field(default_factory=dict)
  """Map sub-terrain name -> range override. Unknown names are warned and
  ignored at runtime."""

  def build(self, env: ManagerBasedRlEnv) -> AmeVelocityCommand:
    return AmeVelocityCommand(self, env)


class AmeVelocityCommand(UniformVelocityCommand):
  """:class:`UniformVelocityCommand` with per-sub-terrain range overrides."""

  cfg: AmeVelocityCommandCfg

  def __init__(self, cfg: AmeVelocityCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    # Lazy name -> column index map, built on first resample.
    self._terrain_col_idx: dict[str, int] | None = None

  def _ensure_terrain_index(self) -> None:
    """Build the sub-terrain name -> column index map once."""
    if self._terrain_col_idx is not None:
      return
    self._terrain_col_idx = {}
    terrain = self._env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
      return
    names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    for name in self.cfg.per_terrain_overrides:
      if name in names:
        self._terrain_col_idx[name] = names.index(name)
      else:
        print(f"[WARN] per_terrain_overrides 地形 '{name}' 不在 sub_terrains 中,忽略")

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    if not self.cfg.per_terrain_overrides:
      return
    self._ensure_terrain_index()
    if not self._terrain_col_idx:
      return
    terrain = self._env.scene.terrain
    if terrain is None:
      return
    types = terrain.terrain_types
    for name, col_idx in self._terrain_col_idx.items():
      grp = env_ids[types[env_ids] == col_idx]
      n = len(grp)
      if n == 0:
        continue
      ov = self.cfg.per_terrain_overrides[name]
      if ov.lin_vel_x is not None:
        self.vel_command_b[grp, 0] = torch.empty(n, device=self.device).uniform_(
          *ov.lin_vel_x
        )
      if ov.lin_vel_y is not None:
        self.vel_command_b[grp, 1] = torch.empty(n, device=self.device).uniform_(
          *ov.lin_vel_y
        )
      if ov.ang_vel_z is not None:
        self.vel_command_b[grp, 2] = torch.empty(n, device=self.device).uniform_(
          *ov.ang_vel_z
        )
      if ov.heading is not None and self.cfg.heading_command:
        self.heading_target[grp] = torch.empty(n, device=self.device).uniform_(
          *ov.heading
        )
