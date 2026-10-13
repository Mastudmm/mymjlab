"""Parkour-specific reward terms for the depth-camera parkour task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity_vision.mdp.rewards import (
  flat_orientation,
  self_collision_cost,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


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


def contact_penalty_excluding(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
  exempt_terrain_types: tuple[int, ...] = (),
) -> torch.Tensor:
  """Body-contact penalty with per-column exemption.

  On the platform ("low wall") column the head/thigh/trunk inevitably brush
  the wall while climbing, so those contacts must not be penalized there;
  the stones/gaps columns keep the original penalty.
  """
  value = self_collision_cost(env, sensor_name, force_threshold)
  exempt = _exempt_mask(env, exempt_terrain_types)
  return value * (~exempt).float()


def flat_orientation_excluding(
  env: ManagerBasedRlEnv,
  std: float,
  exempt_terrain_types: tuple[int, ...] = (),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Upright-orientation reward with per-column exemption.

  Climbing the wall pitches the trunk far beyond flat, so the upright reward
  would suppress climbing; exempt columns receive the neutral 1.0 (no
  orientation signal at all).
  """
  value = flat_orientation(env, std, asset_cfg)
  exempt = _exempt_mask(env, exempt_terrain_types)
  return value * (~exempt).float() + exempt.float()


class reward_point_bonus:
  """One-time milestone bonus for traversing terrain reward points.

  The terrain generator records reward points per (row, col) patch (exposed as
  ``TerrainEntity.reward_points`` / ``reward_counts``). A point is "hit" when

  - the robot base xy lies within ``hit_radius`` of the point,
  - the base z is within ``z_tol`` of the point z (prevents crediting a robot
    that is still on the ground in front of a wall, or standing in a pit
    below the stones), and
  - the robot makes forward progress: body-x speed is positive (climbing is
    slow — the strict speed-tracking gate would deny the wall-top bonus) or
    close to the commanded speed.

  Points only pay while the commanded speed exceeds ``command_threshold`` (no
  bonus while standing). Each point pays at most once per episode (per-env
  ``claimed`` buffer cleared on reset), naturally guiding the robot along the
  obstacle chain. On terrains without reward points the reward is zero.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    terrain = env.scene.terrain
    points = None if terrain is None else terrain.reward_points
    if points is not None:
      self._max_points = int(points.shape[2])
      self._has_points = True
    else:
      self._max_points = 0
      self._has_points = False
    self.claimed = torch.zeros(
      (env.num_envs, self._max_points), dtype=torch.bool, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    hit_radius: float = 0.4,
    z_tol: float = 0.1,
    min_forward_speed: float = 0.05,
    speed_tol: float = 0.5,
    command_threshold: float = 0.1,
    bonus_value: float = 1.0,
  ) -> torch.Tensor:
    if not self._has_points:
      return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    assert terrain is not None

    # Per-env reward points for the current (row, col) sub-terrain. Cache to
    # locals so the None-check narrows (property access is not narrowed by ty).
    rows = terrain.terrain_levels
    cols = terrain.terrain_types
    points_t = terrain.reward_points
    counts_t = terrain.reward_counts
    assert points_t is not None and counts_t is not None, (
      "reward_point_bonus requires terrain with reward points."
    )
    points = points_t[rows, cols]  # [B, max, 3]
    counts = counts_t[rows, cols]  # [B]

    base_pos_w = asset.data.root_link_pos_w  # [B, 3]

    # Hit test (xy): base within hit_radius of the point.
    dist = torch.norm(points[..., :2] - base_pos_w[:, None, :2], dim=-1)
    near = dist < hit_radius  # [B, max]

    # Hit test (z): base high enough to actually be on/at the obstacle top.
    high_enough = base_pos_w[:, 2:3] > points[..., 2] - z_tol  # [B, max]

    # Forward-progress gate: climbing is slow, so accept any positive
    # forward speed OR a speed close to the command.
    command = env.command_manager.get_command(command_name)
    assert command is not None
    vel_b = asset.data.root_link_lin_vel_b  # [B, 3]
    forward = vel_b[:, 0] > min_forward_speed
    tracking = (vel_b[:, 0] - command[:, 0]).abs() < speed_tol
    speed_ok = forward | tracking
    moving = command[:, 0] > command_threshold

    idx = torch.arange(self._max_points, device=env.device)
    valid = idx[None, :] < counts[:, None]  # [B, max]
    hit = valid & near & high_enough & speed_ok[:, None] & moving[:, None]

    newly = hit & (~self.claimed)
    reward = newly.sum(dim=1).float() * bonus_value  # [B]

    self.claimed |= hit
    env.extras["log"]["Metrics/reward_point_count_mean"] = (
      newly.sum(dim=1).float().mean()
    )
    return reward

  def reset(self, env_ids: torch.Tensor) -> None:
    self.claimed[env_ids] = False
