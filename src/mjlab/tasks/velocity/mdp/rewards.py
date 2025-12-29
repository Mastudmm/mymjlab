from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

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
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
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

    std = (
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
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the base height."""
  asset: Entity = env.scene[asset_cfg.name]
  root_z = asset.data.root_link_pos_w[:, 2]
  # Penalize deviation from target height
  error = torch.square(root_z - target_height)
  return torch.exp(-error / std**2)


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
  """Get terrain heights at specified (x, y) positions using raycasting.
  
  Args:
    env: The environment instance.
    positions: Tensor of shape (B, N, 3) or (B, N, 2) containing positions.
    
  Returns:
    Tensor of shape (B, N) containing terrain heights.
  """
  # If terrain is a plane, height is 0.
  if env.scene.terrain is None or env.scene.terrain.cfg.terrain_type == "plane":
    return torch.zeros(positions.shape[:2], device=env.device)

  # For rough terrain, we use raycasting.
  # Note: This is a CPU-based implementation and might be slow for large batches.
  # We cast rays from (x, y, high_z) downwards.
  
  B, N = positions.shape[:2]
  positions_flat = positions.reshape(-1, positions.shape[-1]).cpu().numpy()
  heights = np.zeros(B * N, dtype=np.float32)
  
  # Ray parameters
  ray_pnt = np.zeros(3, dtype=np.float64)
  ray_vec = np.array([0, 0, -1], dtype=np.float64)
  
  # Access simulation data
  # We need the raw mujoco model/data for mj_ray.
  # env.sim is the Simulation instance.
  # It has _mj_model and _mj_data which are the raw MuJoCo objects.
  model = env.sim._mj_model
  data = env.sim._mj_data

  # Optimization: Cache terrain heights to reduce CPU overhead.
  # We update only every 5 steps. This assumes terrain doesn't change abruptly
  # under the feet within ~0.1s.
  update_period = 10
  cache_name = "_terrain_heights_cache"
  
  if (
    hasattr(env, cache_name) 
    and env.common_step_counter % update_period != 0
  ):
    cached = getattr(env, cache_name)
    if cached.shape == (B, N):
      return cached
  
  # Iterate and cast rays
  geom_id_arr = np.zeros(1, dtype=np.int32) # Output buffer for geomid
  
  for i in range(B * N):
    ray_pnt[0] = positions_flat[i, 0]
    ray_pnt[1] = positions_flat[i, 1]
    ray_pnt[2] = 10.0 # Start from high enough
    
    # geomgroup=None means all groups. 
    # flg_static=1 means only static geoms (terrain is usually static).
    # bodyexclude=-1 means no exclusion.
    dist = mujoco.mj_ray(model, data, ray_pnt, ray_vec, None, 1, -1, geom_id_arr)
    
    if dist > -1:
      heights[i] = 10.0 - dist
    else:
      heights[i] = 0.0

  result = torch.from_numpy(heights).to(device=env.device).reshape(B, N)
  setattr(env, cache_name, result)
  return result
