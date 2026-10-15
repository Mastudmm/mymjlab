"""Parkour-specific termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def parkour_completed(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  margin: float = 0.5,
  max_y_offset: float = 0.6,
) -> torch.Tensor:
  """Success termination: the robot has passed the whole obstacle chain.

  Fires when the base x position moves beyond the patch's last valid reward
  point plus ``margin`` AND the base stays within ``max_y_offset`` of that
  point's y (prevents brushing success by skirting the chain sideways). Since
  reward points sit on/at each obstacle, this means every obstacle in the
  chain was crossed, so the episode ends immediately (non-timeout) instead of
  letting the robot idle until the time limit. Terrain curriculum then
  decides the next patch (passing this distance threshold also satisfies the
  move-up condition).
  """
  asset = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None
  points_t = terrain.reward_points
  counts_t = terrain.reward_counts
  assert points_t is not None and counts_t is not None, (
    "parkour_completed requires terrain with reward points."
  )

  rows = terrain.terrain_levels
  cols = terrain.terrain_types
  points = points_t[rows, cols]  # [B, max, 3]
  counts = counts_t[rows, cols]  # [B]

  # Position of the last valid reward point for each env (world frame).
  idx = (counts - 1).clamp(min=0)
  batch_idx = torch.arange(env.num_envs, device=env.device)
  last_point = points[batch_idx, idx]  # [B, 3]

  base_pos = asset.data.root_link_pos_w
  passed_x = base_pos[:, 0] > last_point[:, 0] + margin
  near_y = (base_pos[:, 1] - last_point[:, 1]).abs() < max_y_offset
  return passed_x & near_y


def vel_violate(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  threshold: float = 1.5,
  min_level: int = 3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate on excessive velocity violation vs the command (wmp_go1 style).

  Catches a robot charging BACKWARDS (or standing while commanded forward) by
  more than ``threshold`` m/s on body-frame x. Only active on terrain levels
  above ``min_level``: on the easy rows the robot is still learning to walk,
  and disqualifying it there would just reset it constantly. Complements
  :func:`out_of_bounds`, which only fires on spatial escape.
  """
  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  vel_error = asset.data.root_link_lin_vel_b[:, 0] - command[:, 0]
  violated = ((vel_error > threshold) & (command[:, 0] < 0.0)) | (
    (vel_error < -threshold) & (command[:, 0] > 0.0)
  )
  terrain = env.scene.terrain
  assert terrain is not None
  return violated & (terrain.terrain_levels > min_level)


def base_height_below(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_height: float = -0.3,
) -> torch.Tensor:
  """Failure termination: the base fell into a pit / trench.

  Robots that drop into the stone-pit (2 m) or trench (1.5 m) may land on
  their feet at the pit floor without triggering the non-foot contact or
  orientation terminations; catch them by base height.
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] < min_height


def _exempt_mask(env: ManagerBasedRlEnv, exempt_terrain_types: tuple[int, ...]):
  """Boolean mask [B] that is True on envs whose terrain column is exempt."""
  terrain = env.scene.terrain
  assert terrain is not None
  if not exempt_terrain_types:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return torch.isin(
    terrain.terrain_types,
    torch.tensor(list(exempt_terrain_types), device=env.device),
  )


def illegal_contact_excluding(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  exempt_terrain_types: tuple[int, ...] = (),
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Terminate on non-foot body contact with the terrain, except on exempt
  terrain columns.

  On the platform ("low wall") column the trunk inevitably touches the wall
  while climbing, so contact must not end the episode there; falls are still
  caught by the (relaxed) orientation limit. On the stones/gaps columns
  trunk contact remains a terminating failure.
  """
  from mjlab.tasks.velocity_vision.mdp.terminations import illegal_contact

  collided = illegal_contact(env, sensor_name, force_threshold)
  return collided & ~_exempt_mask(env, exempt_terrain_types)


def bad_orientation_excluding(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  relaxed_angle: float,
  exempt_terrain_types: tuple[int, ...] = (),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Orientation-limit termination with a relaxed limit on exempt columns.

  Climbing the 0.2-0.5 m wall pitches the trunk far beyond the flat-ground
  75-degree limit, so the wall column uses ``relaxed_angle`` (still catches
  actual falls); all other columns keep ``limit_angle``.
  """
  from mjlab.envs.mdp.terminations import bad_orientation

  exceeded = bad_orientation(env, limit_angle, asset_cfg)
  relaxed = bad_orientation(env, relaxed_angle, asset_cfg)
  exempt = _exempt_mask(env, exempt_terrain_types)
  return torch.where(exempt, relaxed, exceeded)


def out_of_bounds(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  max_y_offset: float = 3.5,
  max_back_x: float = 3.5,
) -> torch.Tensor:
  """Terminate when the robot leaves its training patch area.

  The obstacle chains run along world +x from the spawn origin; wandering
  sideways (into neighbouring terrain columns) or backwards escapes the
  remaining reward points entirely. Terminating on escape makes detours a
  hard loss instead of a slow drift. Forward x is not limited here —
  ``parkour_completed`` already ends the episode at the end of the chain.
  """
  asset = env.scene[asset_cfg.name]
  pos = asset.data.root_link_pos_w
  origin = env.scene.env_origins
  y_out = (pos[:, 1] - origin[:, 1]).abs() > max_y_offset
  x_back = pos[:, 0] < origin[:, 0] - max_back_x
  return y_out | x_back
