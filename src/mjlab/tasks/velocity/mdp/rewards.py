from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
import os
import hashlib

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions.

  Returns the number of self-collisions detected by the specified contact sensor.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  # Accept shapes: [B], [B,1], [B,N], [B,N,1]; always reduce to [B].
  if found.dim() == 3:  # [B, N, 1]
    found = found.squeeze(-1)  # -> [B, N]
  if found.dim() == 2:  # [B, N] or [B,1]
    if found.shape[1] == 1:
      return found.squeeze(1)
    return torch.sum(found, dim=1)
  # [B] already
  return found


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.75,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, N, 3]
  terrain_heights = _get_terrain_heights(env, foot_pos)  # [B, N]
  foot_height_rel = foot_pos[..., 2] - terrain_heights
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  # Changed to one-sided penalty: only penalize if foot is LOWER than target.
  # This allows the robot to lift feet higher (e.g. for stairs) without penalty from this term.
  delta = torch.clamp(target_height - foot_height_rel, min=0.0)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost




class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing.
  
  Calculates swing height relative to the foot's position at liftoff.
  This prevents 'crouch-walking' and adapts to uneven terrain.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    # Track max height relative to liftoff (starts at 0)
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    # Track the foot height at the moment of liftoff
    self.liftoff_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    
    # 1. Get Data
    foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
    foot_z = foot_pos[..., 2]
    
    # Handle contact sensor dimensions
    found = contact_sensor.data.found
    assert found is not None
    if found.dim() == 3:
      found = found.squeeze(-1)
    in_contact = found > 0.5
    in_air = ~in_contact

    # 2. Update Liftoff Reference
    # While on ground, the "liftoff height" tracks the current foot height.
    # When it leaves the ground, this value freezes.
    self.liftoff_heights = torch.where(
      in_contact,
      foot_z,
      self.liftoff_heights
    )

    # 3. Calculate Swing Height & Track Peak
    # Height relative to where we started this step
    swing_height = foot_z - self.liftoff_heights
    
    # Only update peak if we are in the air
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, swing_height),
      self.peak_heights
    )

    # 4. Evaluate at Landing (First Contact)
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    
    # Error calculation:
    # If peak < target (Undershoot): error < 0
    # If peak > target (Overshoot): error > 0
    error = self.peak_heights / target_height - 1.0
    
    # Asymmetric Penalty:
    # We penalize Undershoot (too low) heavily -> 1.0
    # We penalize Overshoot (too high) lightly -> 0.1 (allow stepping over things)
    penalty_weight = torch.where(error < 0, 1.0, 0.1)
    
    # Mask by command (only penalize when moving)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    is_moving = (linear_norm + angular_norm) > command_threshold
    
    cost = torch.sum(torch.square(error) * penalty_weight * first_contact.float(), dim=1)
    cost *= is_moving.float()

    # Logging
    num_landings = torch.sum(first_contact.float())
    if num_landings > 0:
      peak_heights_at_landing = self.peak_heights * first_contact.float()
      mean_peak_height = torch.sum(peak_heights_at_landing) / num_landings
      env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height

    # 5. Reset
    # Reset peak to 0 for the next step
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights
    )
    
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float() # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose, with tighter constraints when standing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = ( #std 越小（数值小）：曲线越尖锐。意味着约束非常严格。关节角度只要稍微偏离默认姿态一点点，奖励就会迅速下降归零。
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def track_base_height(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Reward for tracking the base height relative to the terrain using L2 penalty.
  
  Note:
    This returns a cost (positive value). The weight in the config should be negative.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_pos = asset.data.root_link_pos_w
  
  # Get terrain height
  # If we had a sensor, we would use it. Since we don't, we compute it.
  terrain_heights = _get_terrain_heights(env, root_pos.unsqueeze(1)).squeeze(1)
  
  # Adjusted target height (target + terrain)
  adjusted_target_height = target_height + terrain_heights
  
  # Compute L2 penalty
  error = torch.square(root_pos[:, 2] - adjusted_target_height)
  
  # Apply gravity scaling (only penalize when upright)
  # Penalize only if the robot is roughly upright (projected gravity z < 0)
  # clamp(-proj_grav_z, 0, 0.7) / 0.7 -> 1.0 when upright (-1), 0.0 when tilted > 45 deg
  proj_grav_z = asset.data.projected_gravity_b[:, 2]
  scale = torch.clamp(-proj_grav_z, min=0.0, max=0.7) / 0.7
  
  return error * scale


def stumble_penalty(
  env: ManagerBasedRlEnv,
  sensor_names: str | list[str],
) -> torch.Tensor:
  """Penalize stumbling (hitting obstacles horizontally) across multiple sensors."""
  if isinstance(sensor_names, str):
    sensor_names = [sensor_names]

  total_penalty = torch.zeros(env.num_envs, device=env.device)

  for name in sensor_names:
    contact_sensor: ContactSensor = env.scene[name]
    forces = contact_sensor.data.force  # [B, N, 3]
    
    if forces is None:
      continue

    # Horizontal force magnitude (xy plane)
    horizontal_forces = torch.norm(forces[:, :, :2], dim=-1)
    # Vertical force magnitude (z axis)
    vertical_forces = torch.abs(forces[:, :, 2])
    
    # Stumble condition:
    # 1. Geometric check: Horizontal force > Vertical force (implies hitting vertical surface).
    # 2. Magnitude check: Horizontal force must be > 10.0N to ignore light scuffing/drag.
    stumble = ((horizontal_forces > 4.0*vertical_forces) & (horizontal_forces > 5.0)).float()
    
    # Penalize by horizontal impact, but CLIP it to prevent reward explosion.
    penalty = torch.sum(stumble * torch.clamp(horizontal_forces, max=20.0), dim=1)
    total_penalty += penalty

  return total_penalty


def feet_clearance_body(
  env: ManagerBasedRlEnv,
  target_height: float,
  tanh_mult: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target foot height in body frame when moving.
  
  This is a computationally efficient alternative to feet_clearance that avoids raycasting.
  It encourages a 'high-stepping' gait relative to the robot's body.
  
  Args:
    target_height: Target height of the foot in the body frame (usually negative, e.g. -0.2).
    tanh_mult: Scaling factor for the velocity mask.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Get foot positions and velocities in world frame
  # Shape: [B, N, 3]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  foot_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
  
  # Get root position and orientation
  # Shape: [B, 3] and [B, 4]
  root_pos_w = asset.data.root_link_pos_w
  root_quat_w = asset.data.root_link_quat_w
  
  # Calculate relative position/velocity in world frame
  # Broadcast root: [B, 1, 3]
  rel_pos_w = foot_pos_w - root_pos_w.unsqueeze(1)
  rel_vel_w = foot_vel_w - asset.data.root_link_lin_vel_w.unsqueeze(1)
  
  # Transform to body frame
  # We need to apply inverse rotation to each foot.
  # quat_apply_inverse handles [B, 4] and [B, 3], so we flatten the feet dimension first.
  B, N, _ = foot_pos_w.shape
  
  # Flatten: [B*N, 3]
  rel_pos_w_flat = rel_pos_w.reshape(-1, 3)
  rel_vel_w_flat = rel_vel_w.reshape(-1, 3)
  
  # Repeat quat: [B*N, 4]
  root_quat_w_expanded = root_quat_w.repeat_interleave(N, dim=0)
  
  # Rotate
  foot_pos_b_flat = quat_apply_inverse(root_quat_w_expanded, rel_pos_w_flat)
  foot_vel_b_flat = quat_apply_inverse(root_quat_w_expanded, rel_vel_w_flat)
  
  # Reshape back: [B, N, 3]
  foot_pos_b = foot_pos_b_flat.reshape(B, N, 3)
  foot_vel_b = foot_vel_b_flat.reshape(B, N, 3)
  
  # Calculate cost
  # 1. Height error: deviation from target Z in body frame
  foot_z_error = torch.square(foot_pos_b[..., 2] - target_height)
  
  # 2. Velocity mask: only penalize when foot is moving horizontally (swing phase)
  foot_vel_xy_norm = torch.norm(foot_vel_b[..., :2], dim=-1)
  velocity_mask = torch.tanh(tanh_mult * foot_vel_xy_norm)
  
  cost = torch.sum(foot_z_error * velocity_mask, dim=1)
  
  return cost


def _get_terrain_heights(env: ManagerBasedRlEnv, positions: torch.Tensor) -> torch.Tensor:
  """Get terrain heights at specified (x, y) positions using a pre-computed height map.
  
  Args:
    env: The environment instance.
    positions: Tensor of shape (B, N, 3) or (B, N, 2) containing positions.
    
  Returns:
    Tensor of shape (B, N) containing terrain heights.
  """
  # Allow skipping via env attribute (useful for play/inference)
  if getattr(env, "skip_terrain_height_map", False):
    return torch.zeros(positions.shape[:2], device=env.device)

  # If terrain is a plane, height is 0.
  # Use getattr to safely access terrain in case it's not defined in the interface
  terrain = getattr(env.scene, "terrain", None)
  if terrain is None or terrain.cfg.terrain_type == "plane":
    return torch.zeros(positions.shape[:2], device=env.device)

  # Initialize map if needed
  if not hasattr(env, "_terrain_height_map"):
    _init_terrain_height_map(env)
    
  # Lookup
  return _lookup_terrain_height_map(env, positions)


def _init_terrain_height_map(env: ManagerBasedRlEnv, resolution: float = 0.02):
  """Pre-compute terrain height map for fast lookup with caching."""
  terrain = getattr(env.scene, "terrain", None)
  if terrain is None:
    raise ValueError("Terrain is None, cannot generate height map.")
    
  cfg = terrain.cfg
  gen_cfg = cfg.terrain_generator
  
  if gen_cfg is None:
    raise ValueError("Terrain generator config is None.")

  # Try to load from cache
  cache_path = None
  try:
    # Create a hash based on the string representation of the config and resolution
    config_str = str(gen_cfg) + f"_res_{resolution}"
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    cache_dir = os.path.join(os.getcwd(), "logs", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"terrain_map_{config_hash}.pt")
    
    if os.path.exists(cache_path):
      print(f"Loading terrain height map from cache: {cache_path}")
      map_data = torch.load(cache_path, map_location=env.device)
      setattr(env, "_terrain_height_map", map_data)
      return
  except Exception as e:
    print(f"Cache lookup failed: {e}. Proceeding to generate map.")

  print(f"Generating global terrain height map (res={resolution}m)...")
  
  # Calculate bounds
  # Assuming grid is centered at 0,0
  total_width = gen_cfg.num_rows * gen_cfg.size[0]
  total_height = gen_cfg.num_cols * gen_cfg.size[1]
  
  # Add border width
  border = gen_cfg.border_width
  
  min_x = -total_width / 2 - border
  max_x = total_width / 2 + border
  min_y = -total_height / 2 - border
  max_y = total_height / 2 + border
  
  x_points = int((max_x - min_x) / resolution)
  y_points = int((max_y - min_y) / resolution)
  
  # Create grid of points
  x = np.linspace(min_x, max_x, x_points)
  y = np.linspace(min_y, max_y, y_points)
  xx, yy = np.meshgrid(x, y)
  
  # Flatten for raycasting
  points = np.stack([xx.flatten(), yy.flatten()], axis=1)
  num_points = points.shape[0]
  
  # Raycast
  model = env.sim._mj_model
  data = env.sim._mj_data
  ray_pnt = np.zeros(3, dtype=np.float64)
  ray_vec = np.array([0, 0, -1], dtype=np.float64)
  geom_id_arr = np.zeros(1, dtype=np.int32)
  
  heights = np.zeros(num_points, dtype=np.float32)
  
  # Loop for raycasting
  for i in range(num_points):
    ray_pnt[0] = points[i, 0]
    ray_pnt[1] = points[i, 1]
    ray_pnt[2] = 100.0 # High enough
    
    dist = mujoco.mj_ray(model, data, ray_pnt, ray_vec, None, 1, -1, geom_id_arr)
    if dist > -1:
      heights[i] = 100.0 - dist
    else:
      heights[i] = -10.0 # Default to low if missed
      
  # Reshape to (H, W) -> (y, x)
  height_map = torch.from_numpy(heights.reshape(y_points, x_points)).float().to(env.device)
  
  map_data = {
    "map": height_map.unsqueeze(0).unsqueeze(0), # (1, 1, H, W)
    "min_x": min_x,
    "max_x": max_x,
    "min_y": min_y,
    "max_y": max_y
  }

  # Save to cache if path was determined
  if cache_path:
    try:
      print(f"Saving terrain height map to cache: {cache_path}")
      torch.save(map_data, cache_path)
    except Exception as e:
      print(f"Failed to save cache: {e}")

  # Use setattr to avoid linter errors about unknown attributes
  setattr(env, "_terrain_height_map", map_data)
  print("Terrain map generated.")


def _lookup_terrain_height_map(env: ManagerBasedRlEnv, positions: torch.Tensor) -> torch.Tensor:
  """Lookup heights using bilinear interpolation."""
  tm = getattr(env, "_terrain_height_map")
  
  x = positions[..., 0]
  y = positions[..., 1]
  
  # Normalize to [-1, 1]
  norm_x = 2 * (x - tm["min_x"]) / (tm["max_x"] - tm["min_x"]) - 1
  norm_y = 2 * (y - tm["min_y"]) / (tm["max_y"] - tm["min_y"]) - 1
  
  B, N = x.shape
  # Stack for grid_sample: (1, 1, B*N, 2)
  grid = torch.stack([norm_x, norm_y], dim=-1).reshape(1, 1, B * N, 2)
  
  # Sample
  sampled = torch.nn.functional.grid_sample(
    tm["map"], 
    grid, 
    mode='bilinear', 
    padding_mode='border', 
    align_corners=True
  )
  
  return sampled.reshape(B, N)
