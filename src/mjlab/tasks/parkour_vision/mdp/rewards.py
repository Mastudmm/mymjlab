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
  - the base z is at least ``z_margin`` above the point z (prevents crediting
    a robot that is still on the ground in front of a wall, or standing in a
    pit below the stones), and
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
    z_margin: float = 0.05,
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

    # Hit test (z): base must be ON/above the obstacle top — a robot on the
    # ground in front of a wall or in a pit below the stones earns nothing.
    high_enough = base_pos_w[:, 2:3] > points[..., 2] + z_margin  # [B, max]

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


class heading_alignment:
  """Reward for aligning the base yaw with the world +x heading.

  The parkour chains run along world +x and the command is pure forward-x, so
  a robot that yaws away drifts off the obstacle line. This keeps the heading
  pinned to the chain direction.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    pass

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float = 0.4,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    # Project world +x into the body frame: the body-frame x component of a
    # world +x vector equals cos(yaw_error).
    world_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(
      env.num_envs, 3
    )
    from mjlab.utils.lab_api.math import quat_apply_inverse

    body_x_of_world_x = quat_apply_inverse(quat, world_x)
    cos_err = body_x_of_world_x[:, 0]  # 1.0 when yaw aligned, -1.0 when flipped
    return torch.clamp(cos_err, min=0.0)  # no reward for facing backwards


class lateral_track:
  """Penalize lateral deviation from the spawn centerline (world y).

  Keeps the robot on the straight obstacle line; escaping sideways earns no
  forward progress under :class:`forward_progress` and loses this reward.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    pass

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    origin_y = env.scene.env_origins[:, 1]
    y_err = asset.data.root_link_pos_w[:, 1] - origin_y
    return torch.exp(-(y_err**2) / std**2)


class forward_progress:
  """Reward the world-frame +x displacement per step.

  Directly pays for advancing along the obstacle-chain direction, so climbing
  a wall (slow but forward) still earns, while sideways detours earn nothing.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self.last_x = env.scene.env_origins[:, 0].clone()

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    x = asset.data.root_link_pos_w[:, 0]
    progress = x - self.last_x
    self.last_x = x.clone()
    return progress

  def reset(self, env_ids: torch.Tensor) -> None:
    # Anchor to the (possibly curriculum-updated) spawn origin so the first
    # step after reset measures real progress from the start line.
    self.last_x[env_ids] = self._env.scene.env_origins[env_ids, 0]


def track_linear_velocity_excluding_z(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  exempt_terrain_types: tuple[int, ...] = (),
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Velocity tracking without the vertical-speed penalty on exempt columns.

  The base ``track_linear_velocity`` penalizes body-frame vz (commanded vz is
  zero), which is correct for walking but makes jumping — required to climb
  the 0.2-0.5 m wall — extremely expensive. On the wall column the vz penalty
  is dropped; the xy tracking term stays everywhere.
  """
  from mjlab.tasks.velocity_vision.mdp.rewards import track_linear_velocity

  value = track_linear_velocity(env, std, command_name, asset_cfg)
  exempt = _exempt_mask(env, exempt_terrain_types)
  # On exempt columns drop the vz part only: recompute xy-only term and blend.
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  xy_only = torch.exp(-xy_error / std**2)
  return torch.where(exempt, xy_only, value)
