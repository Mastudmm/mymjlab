"""Parkour-specific reward terms for the depth-camera parkour task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity_vision.mdp.rewards import self_collision_cost
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_FRONT_FEET_CFG = SceneEntityCfg("robot", site_names=("FR", "FL"))


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


def wall_climb_shaping(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  asset_cfg: SceneEntityCfg = _FRONT_FEET_CFG,
  approach_weight: float = 0.3,
  pitch_weight: float = 0.3,
  paw_weight: float = 0.5,
  near_max: float = 1.5,
  max_pitch: float = 0.8,
  paw_min_height: float = 0.1,
) -> torch.Tensor:
  """Dense shaping gradient toward the wall-climb behavior (user request
  2026-10-20: approach + pitch + front paws on the wall).

  Three components, all gated by a NEAR-WALL ZONE around the FIRST wall's
  reward point (wall-top center, world frame from ``terrain.reward_points``)
  so nothing is payable away from the wall:

  - zone: horizontal distance to the point < ``near_max`` AND base below the
    point's top (not yet climbed) AND base x before the wall center — i.e.
    the robot is on the approach side of wall 1;
  - approach: linear ramp ``1 - d_xy / near_max`` — pays for closing in on
    the wall face (gradient exists from the spawn onward);
  - pitch: nose-up tilt ``clamp(body-x world z, 0, max_pitch) / max_pitch``
    — the climb-preparation posture (front of the trunk rising);
  - paws: fraction of FRONT feet (FR/FL, ``feet_ground_contact`` sensor — its
    secondary match covers the wall geoms) in contact with terrain while the
    foot site is above ``paw_min_height`` (elevated = on the wall face, not
    the ground).

  Magnitudes: full-value sum ~1.1 with term weight 1.0 — comparable to but
  below the tracking reward; the milestone bonus (2.0/point on top) remains
  the primary climb payment. Component means are logged to TensorBoard via
  ``Metrics/wall_shaping/*``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None
  points_t = terrain.reward_points
  counts_t = terrain.reward_counts
  assert points_t is not None and counts_t is not None, (
    "wall_climb_shaping requires terrain with reward points."
  )

  base_pos = asset.data.root_link_pos_w
  rows = terrain.terrain_levels
  cols = terrain.terrain_types

  # Wall-1 top center for each env; envs without points get no shaping.
  has_point = counts_t[rows, cols] > 0
  batch = torch.arange(env.num_envs, device=env.device)
  idx0 = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  wall_xy = points_t[rows, cols][batch, idx0, :2]  # [B, 2]
  wall_top_z = points_t[rows, cols][batch, idx0, 2]  # [B]

  d_xy = torch.norm(wall_xy - base_pos[:, :2], dim=-1)
  # "Not yet on top" gate: for TALL walls require the base to still be below
  # the top; for LOW walls (top < 0.15 m — the training-start rows where the
  # standing base is already above the top) the x-side gate alone delimits
  # the approach side, otherwise the zone would be empty there.
  low_wall = wall_top_z < 0.15
  zone = (
    (d_xy < near_max)
    & ((base_pos[:, 2] < wall_top_z + 0.1) | low_wall)
    & (base_pos[:, 0] < wall_xy[:, 0] + 0.3)
    & has_point
  ).float()

  # (a) approach ramp: 0 at near_max -> 1 at the wall.
  approach = torch.clamp(1.0 - d_xy / near_max, min=0.0)

  # (b) nose-up pitch: body +x axis world z component (0 = level, 1 = 90°).
  quat = asset.data.root_link_quat_w
  world_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
  pitch_sin = quat_apply(quat, world_x)[:, 2]
  pitch_n = torch.clamp(pitch_sin, min=0.0, max=max_pitch) / max_pitch

  # (c) front paws on the wall: contact while the foot site is elevated.
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  # Sensor geom order is (FR, FL, RR, RL) — front feet are columns [:, :2],
  # matching asset_cfg site order ("FR", "FL").
  in_contact = (contact_sensor.data.found[:, :2] > 0).float()  # [B, 2]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, 2]
  elevated = (foot_z > paw_min_height).float()
  paws_n = (in_contact * elevated).mean(dim=1)  # [B]

  combined = zone * (
    approach_weight * approach + pitch_weight * pitch_n + paw_weight * paws_n
  )

  env.extras["log"]["Metrics/wall_shaping/approach"] = approach.mean()
  env.extras["log"]["Metrics/wall_shaping/pitch"] = pitch_n.mean()
  env.extras["log"]["Metrics/wall_shaping/paws"] = paws_n.mean()
  env.extras["log"]["Metrics/wall_shaping/zone"] = zone.mean()
  return combined


def contact_penalty_excluding(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
  exempt_terrain_types: tuple[int, ...] = (),
) -> torch.Tensor:
  """Body-contact penalty with per-column exemption.

  On the platform ("low wall") column the head/thigh/trunk inevitably brush
  the wall while climbing, so those contacts must not be penalized there;
  the stones/gaps columns keep the original penalty. (This exemption — and
  the matching contact termination / orientation-limit relaxations — is the
  ONLY per-column exemption kept in the wmp_go1-aligned reward set: unlike
  wmp_go1's heightfield terrain, a true vertical box wall forces body contact
  during the climb.)
  """
  value = self_collision_cost(env, sensor_name, force_threshold)
  exempt = _exempt_mask(env, exempt_terrain_types)
  return value * (~exempt).float()


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
    self._cfg = cfg
    self._env = env
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

  def debug_vis(self, visualizer) -> None:
    """Draw the reward points of each visualized env in the 3D scene.

    Per point (for the env's CURRENT terrain row/col):

    - small solid sphere: green = unclaimed, gray = claimed this episode;
    - large transparent sphere of ``hit_radius``: the xy capture zone;
    - short stem of height ``z_margin``: how far ABOVE the point's z the
      base must climb before the hit test passes.

    Toggle with the viewer's debug-visualizer switch (``R`` in the native
    viewer, on by default)."""
    if not self._has_points:
      return
    terrain = self._env.scene.terrain
    assert terrain is not None
    points_t = terrain.reward_points
    counts_t = terrain.reward_counts
    assert points_t is not None and counts_t is not None
    params = self._cfg.params
    hit_radius = float(params["hit_radius"])
    z_margin = float(params["z_margin"])
    stem = np.array([0.0, 0.0, z_margin])

    for env_idx in visualizer.get_env_indices(self._env.num_envs):
      row = terrain.terrain_levels[env_idx]
      col = terrain.terrain_types[env_idx]
      points = points_t[row, col]
      count = int(counts_t[row, col])
      claimed = self.claimed[env_idx]
      for i in range(min(count, self._max_points)):
        p = points[i].detach().cpu().numpy()
        if claimed[i]:
          marker_color = (0.4, 0.4, 0.4, 0.35)
          zone_color = (0.4, 0.4, 0.4, 0.05)
        else:
          marker_color = (0.15, 0.9, 0.2, 0.9)
          zone_color = (0.15, 0.9, 0.2, 0.10)
        visualizer.add_sphere(p, 0.06, marker_color)
        visualizer.add_sphere(p, hit_radius, zone_color)
        visualizer.add_cylinder(p, p + stem, 0.012, (0.95, 0.7, 0.1, 0.85))

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
    world_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
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


def cheat_penalty(
  env: ManagerBasedRlEnv,
  max_yaw_dev: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize large world-frame yaw deviation from the chain direction (+x).

  wmp_go1 applies the same penalty to its forward-only terrain group: heading
  beyond +-``max_yaw_dev`` rad in the WORLD frame means the robot turned away
  from the obstacle line (the only way around a full-width chain), so the
  detour is penalized directly instead of relying on lost progress alone.
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.root_link_quat_w
  from mjlab.utils.lab_api.math import quat_apply

  world_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
  forward = quat_apply(quat, world_x)  # body +x direction in world frame
  heading = torch.atan2(forward[:, 1], forward[:, 0])
  return ((heading > max_yaw_dev) | (heading < -max_yaw_dev)).float()


def track_linear_velocity_clipped(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  lin_vel_clip: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Velocity tracking with speed overspeed clipping (wmp_go1 style).

  The actual body-frame xy velocity is clipped to the commanded speed plus
  ``lin_vel_clip`` before the exponential error is computed: moving FASTER
  than commanded (charging over a gap, leaping onto a wall) costs nothing,
  while slow/erratic motion is still punished. This is what lets wmp_go1
  train obstacle traversal with a single uniform tracking reward and no
  per-terrain exemptions — the obstacle columns reward speed instead of
  punishing it.

  Only the body-frame xy error enters the exponential — the z velocity is
  NOT penalized (deliberately: climbing needs vz > 0; the base term's
  ``z_error`` does not apply here).
  """

  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b[:, :2]
  upper = torch.where(
    command[:, :2] < 0,
    torch.full_like(command[:, :2], 1e5),
    command[:, :2] + lin_vel_clip,
  )
  lower = torch.where(
    command[:, :2] > 0,
    torch.full_like(command[:, :2], -1e5),
    command[:, :2] - lin_vel_clip,
  )
  clipped = torch.clip(actual, lower, upper)
  xy_error = torch.sum(torch.square(command[:, :2] - clipped), dim=1)
  return torch.exp(-xy_error / std**2)
