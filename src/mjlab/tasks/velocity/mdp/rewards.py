from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
import os
import hashlib

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
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

  # --- Original Code (Commented Out) ---
  # Penalizes both Roll (index 0) and Pitch (index 1) rates.
  # xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  # -------------------------------------

  # --- Modification for Stair Climbing ---
  # Only penalizes Roll rate (Index 0).
  # We IGNORE Pitch rate (Index 1) because the robot needs to pitch up/down significantly
  # to climb stairs. Penalizing it forces the robot to stay flat, causing failure on slopes/stairs.
  xy_error = 0.01 * torch.square(actual[:, 0])
  # -------------------------------------

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


def self_collision_cost(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1.0
) -> torch.Tensor:
  """Penalize self-collisions / undesired contacts exceeding a force threshold.

  Returns the number of contacts where the net force magnitude exceeds the threshold.
  Useful for allowing 'grazing' (light touches) without penalty.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  
  # Try to use force data if available for thresholding to allow grazing (light touch)
  if hasattr(sensor.data, "force") and sensor.data.force is not None:
    forces = sensor.data.force  # [B, N, 3]
    force_mag = torch.norm(forces[..., :3], dim=-1) # [B, N]
    violation = (force_mag > threshold).float()
  else:
    # Fallback to binary 'found' logic if no force data
    found = sensor.data.found
    assert found is not None
    if found.dim() == 3:  # [B, N, 1]
      found = found.squeeze(-1)
    violation = (found > 0.5).float()

  # Sum violations across sensor contact points (e.g. all calf segments)
  if violation.dim() > 1:
    collision_count = torch.sum(violation, dim=1)
  else:
    collision_count = violation # [B]

  # Apply upright mask to avoid 'punishing the dead' (robot already fell over)
  # Only punish collisions when the robot is trying to walk (upright)
  try:
    asset: Entity = env.scene["robot"]
    proj_grav_z = asset.data.projected_gravity_b[:, 2]
    # Scale: 1.0 when upright (-1), goes to 0 when tilted > ~45 deg
    upright_scale = torch.clamp(-proj_grav_z, min=0.0, max=0.7) / 0.7
  except KeyError:
    upright_scale = torch.ones_like(collision_count)

  return collision_count * upright_scale


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
  sensor_name: str | list[str] | None = None,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity.
  
  Uses body-frame kinematics to determine foot height, avoiding noisy RayCasts.
  Target height should be specified in the body frame (e.g. -0.2).
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # -- Kinematic Logic (Body Frame) --
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  root_pos_w = asset.data.root_link_pos_w
  root_quat_w = asset.data.root_link_quat_w

  # Calculate relative position (Foot - Root) in World Frame
  # [B, N, 3] - [B, 1, 3] broadcast
  rel_pos_w = foot_pos_w - root_pos_w.unsqueeze(1)

  # Transform to Body Frame
  B, N, _ = foot_pos_w.shape
  rel_pos_w_flat = rel_pos_w.reshape(-1, 3)
  root_quat_w_expanded = root_quat_w.repeat_interleave(N, dim=0)
  
  rel_pos_b_flat = quat_apply_inverse(root_quat_w_expanded, rel_pos_w_flat)
  foot_pos_b = rel_pos_b_flat.reshape(B, N, 3)
  
  # Use Z-coordinate in body frame as height
  foot_height_rel = foot_pos_b[..., 2]

  env.extras["log"]["Debug/Clearance_Body_Mean"] = torch.mean(foot_height_rel)

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
  
  Calculates swing height relative to the foot's position at liftoff in the body frame.
  This prevents 'crouch-walking' and adapts to uneven terrain without needing raycasts.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    # Track max height relative to liftoff (starts at 0)
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    # Track the foot height at the moment of liftoff (in body frame)
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
    height_sensor_name: str | list[str] | None = None, # Deprecated but kept for signature compatibility
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    
    # -- 1. Calculate Foot Position in Body Frame --
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :] # [B, N, 3]
    root_pos_w = asset.data.root_link_pos_w
    root_quat_w = asset.data.root_link_quat_w
    
    B, N, _ = foot_pos_w.shape
    
    # Broadcast root
    rel_pos_w = foot_pos_w - root_pos_w.unsqueeze(1) #root_pos_w.unsqueeze(1) 把 [B, 3] 变成了 [B, 1, 3]。
    
    # Flatten for quat operation
    rel_pos_w_flat = rel_pos_w.reshape(-1, 3)#：变成了 [B * N, 3]。 #reshape(-1, 3)把数据重新排列成只有 3 列的矩阵，行数自动计算
    root_quat_w_expanded = root_quat_w.repeat_interleave(N, dim=0)#把每一行复制 N 次。
    
    # Transform to body frame
    foot_pos_b_flat = quat_apply_inverse(root_quat_w_expanded, rel_pos_w_flat)
    foot_pos_b = foot_pos_b_flat.reshape(B, N, 3)
    
    # We only care about Z in body frame
    foot_z_body = foot_pos_b[..., 2]
    
    # Logging for debug
    env.extras["log"]["Debug/Swing_Body_Height_Min"] = torch.min(foot_z_body)
    env.extras["log"]["Debug/Swing_Body_Height_Max"] = torch.max(foot_z_body)

    # -- 2. Contact State --
    found = contact_sensor.data.found
    assert found is not None
    if found.dim() == 3:
      found = found.squeeze(-1)
    in_contact = found > 0.5
    in_air = ~in_contact

    # -- 3. Update Liftoff Reference --
    # While on ground, the "liftoff height" tracks current body-frame Z.
    # When it leaves the ground, this value freezes.
    self.liftoff_heights = torch.where(
      in_contact,
      foot_z_body,
      self.liftoff_heights
    )

    # -- 4. Calculate Swing Height & Track Peak --
    # Height gained since liftoff
    swing_height = foot_z_body - self.liftoff_heights
    
    # Only update peak if we are in the air
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, swing_height),
      self.peak_heights
    )

    # -- 5. Evaluate at Landing --
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    
    # Error: difference from target
    # If target is 0.1m, and we reached 0.08m, error < 0.
    error = self.peak_heights / target_height - 1.0
    
    # Asymmetric Penalty: 
    # Penalize Undershoot (too low) heavily -> 1.0
    # Penalize Overshoot (too high) lightly -> 0.025
    penalty_weight = torch.where(error < 0, 1.0, 0.35)
    
    # Command Mask: Only penalize when moving
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    is_moving = (linear_norm + angular_norm) > command_threshold
    
    cost = torch.sum(torch.square(error) * penalty_weight * first_contact.float(), dim=1)
    cost *= is_moving.float()

    # Upright Mask: Don't punish if fallen
    proj_grav_z = asset.data.projected_gravity_b[:, 2]
    upright_scale = torch.clamp(-proj_grav_z, min=0.0, max=0.7) / 0.7
    cost *= upright_scale

    # Logging
    num_landings = torch.sum(first_contact.float())
    if num_landings > 0:
      peak_heights_at_landing = self.peak_heights * first_contact.float()
      mean_peak_height = torch.sum(peak_heights_at_landing) / num_landings
      env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height

    # -- 6. Reset --
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
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

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
  sensor_name: str | None = None,
) -> torch.Tensor:
  """Reward for tracking the base height relative to the terrain.

  Modified to heavily penalize operating below the target height to prevent
  'crouching' behavior often induced by foot clearance rewards.
  
  Returns a cost (positive value). Weight in config should be negative.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_pos = asset.data.root_link_pos_w
  
  # Get terrain height
  if sensor_name is not None:
     terrain_heights = _get_heights_from_single_ray_sensors(env, sensor_name)
  else:
     terrain_heights = torch.zeros_like(root_pos[:, 2])

  # Logging for debug: 
  # Check Min to see if rays are missing terrain (hitting -10.0)
  env.extras["log"]["Debug/Terrain_Height_Min"] = torch.min(terrain_heights)
  env.extras["log"]["Debug/Terrain_Height_Mean"] = torch.mean(terrain_heights)
  
  # Check relative height (Robot Z - Terrain Z)
  # If this is consistently huge or negative, the rays are wrong.
  rel_h_world = root_pos[:, 2] - terrain_heights
  
  # --- Tilt Correction ---
  # On slopes/stairs, the vertical distance is not the clearance.
  # We project the vertical distance to the body's local Z axis.
  # tilt_cos = -projected_gravity_z (approx)
  proj_grav_z = asset.data.projected_gravity_b[:, 2]
  tilt_cos = torch.clamp(-proj_grav_z, min=0.0, max=1.0)
  
  # Effective clearance height
  rel_h_body = rel_h_world * tilt_cos
  
  # Log mean clearance to check overall behavior
  env.extras["log"]["Debug/Base_Height_Above_Terrain_Mean"] = torch.mean(rel_h_body)
  # Log MIN clearance to detect individual "crashes" or "crawling" (Critical Safety Check)
  # If this hits ~0.0, someone is scraping the floor.
  env.extras["log"]["Debug/Base_Height_Above_Terrain_Min"] = torch.min(rel_h_body)

  # Target is usually "desired clearance" (e.g. 0.3m)
  # Error = current_clearance - target
  deviation = rel_h_body - target_height
  
  # Reward calculation: (Positive reward)
  # Formula: exp(-error^2 / sigma)
  # We still want to penalize crouching (deviation < 0) more than standing tall.
  # So we use a smaller sigma (sharper drop) for negative deviation.
  
  # sigma for "too low": 0.05 (very strict)
  # sigma for "too high": 0.1 (more lenient)
  sigma = torch.where(deviation < 0.0, 0.05, 0.1)
  
  reward = torch.exp(-torch.square(deviation) / (2 * torch.square(sigma)))
  
  # Apply gravity scaling (only reward when upright)
  # Scale goes to 0 if robot falls over (grav_z > -0.1 approx)
  upright_scale = torch.clamp(tilt_cos, min=0.0, max=0.7) / 0.7
  
  return reward * upright_scale


def stumble_penalty(
  env: ManagerBasedRlEnv,
  sensor_names: str | list[str],
) -> torch.Tensor:
  """Penalize stumbling (hitting obstacles horizontally) across multiple sensors.
  
  Improved logic:
  1. Checks if horizontal force > vertical force (impact vs support).
  2. Masks penalty when robot is not upright to avoid 'punishing the dead'.
  """
  if isinstance(sensor_names, str):
    sensor_names = [sensor_names]

  total_penalty = torch.zeros(env.num_envs, device=env.device)
  
  # Get robot asset for gravity projection
  # We assume the first asset in the scene is the robot for simplicity 
  # or rely on a known name if passed, but here we scan env.scene
  # A safer way is to assume typical 'robot' name or pass asset_cfg
  # Fallback to 'robot'
  try:
      asset: Entity = env.scene["robot"]
      proj_grav_z = asset.data.projected_gravity_b[:, 2]
      # Scale: 1.0 when upright (-1), 0.0 when tilted > 45 deg (>-0.707)
      # -(-1) = 1. clamp(1, 0, 0.7)/0.7 = 1.
      # -(-0.5) = 0.5. clamp(0.5, 0, 0.7)/0.7 = 0.71
      upright_scale = torch.clamp(-proj_grav_z, min=0.0, max=0.7) / 0.7
  except KeyError:
      upright_scale = torch.ones(env.num_envs, device=env.device)

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
    # Relaxed condition: horizontal > vertical (angle > 45 deg)
    # Plus minimal force check (>1.0N) to ignore noise
    stumble = ((horizontal_forces > 3.0 * vertical_forces) & (horizontal_forces > 1.0)).float()
    
    # Penalize by horizontal impact magnitude
    # We sum over all contacts (e.g. all 4 legs if passed conceptually, or body parts)
    penalty = torch.sum(stumble * horizontal_forces, dim=1)
    total_penalty += penalty

  # Apply upright mask
  return total_penalty * upright_scale


def feet_clearance_body(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  tanh_mult: float = 2.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target foot height in body frame when moving.
  
  This is a computationally efficient alternative to feet_clearance that avoids raycasting.
  It encourages a 'high-stepping' gait relative to the robot's body.
  
  Args:
    target_height: Target height of the foot in the body frame (usually negative, e.g. -0.2).
    tanh_mult: Scaling factor for the velocity mask.
    command_name: Name of the command to check for movement (optional).
    command_threshold: Threshold for the command magnitude to activate the penalty.
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

  # 3. Command mask: only penalize when the robot is commanded to move
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      is_moving = (linear_norm + angular_norm) > command_threshold
      cost *= is_moving.float()

  # 4. Upright mask: do not penalize if the robot has fallen
  proj_grav_z = asset.data.projected_gravity_b[:, 2]
  upright_scale = torch.clamp(-proj_grav_z, min=0.0, max=0.7) / 0.7
  cost *= upright_scale
  
  return cost


def _get_heights_from_sensor(
  env: ManagerBasedRlEnv, sensor_name: str, positions: torch.Tensor
) -> torch.Tensor:
  """Get terrain heights at specified positions using a RayCastSensor with GridPattern."""
  try:
    sensor = env.scene[sensor_name]
  except KeyError:
    return torch.zeros(positions.shape[:2], device=env.device)
    
  # Basic check for sensor type and availability
  if not hasattr(sensor, "data") or not hasattr(sensor, "cfg") or not hasattr(sensor.cfg, "pattern"):
    return torch.zeros(positions.shape[:2], device=env.device)

  # Check if pattern has necessary attributes for grid
  pattern = sensor.cfg.pattern
  if not hasattr(pattern, "size") or not hasattr(pattern, "resolution"):
     return torch.zeros(positions.shape[:2], device=env.device)

  # Reconstruct grid dimensions
  res = pattern.resolution
  size_x, size_y = pattern.size
  
  # Calculate dimensions (assuming GridPattern generation logic)
  W = int(torch.arange(-size_x / 2, size_x / 2 + res * 0.5, res).shape[0])
  H = int(torch.arange(-size_y / 2, size_y / 2 + res * 0.5, res).shape[0])

  if H * W != sensor.num_rays:
    # Fallback if dimensions don't match (e.g. pattern changed)
    return torch.zeros(positions.shape[:2], device=env.device)

  # Get Z heights from sensor world hit positions
  # hit_pos_w is [B, N, 3]
  # We extract Z: [B, N]
  # Handle misses: if distance < 0, the ray missed (sky/hole).
  # RayCastSensor usually sets hit_pos to origin on miss, which is high (robot body).
  # We want misses to be treated as "far below" (deep pit) so clearance is valid (safe).
  hit_pos_z = sensor.data.hit_pos_w[..., 2].clone()
  distances = sensor.data.distances
  # Set misses to -10.0m (effectively infinite depth for locomotion)
  hit_pos_z[distances < 0] = -10.0

  height_map = hit_pos_z.reshape(env.num_envs, 1, H, W)

  # Transform query positions to sensor local frame
  sensor_pos = sensor.data.pos_w
  sensor_quat = sensor.data.quat_w
  
  B, N_query, _ = positions.shape
  rel_pos = positions - sensor_pos.unsqueeze(1) # [B, Nq, 3]
  
  # Apply inverse rotation
  q_expanded = sensor_quat.repeat_interleave(N_query, dim=0)
  p_flat = rel_pos.reshape(-1, 3)
  local_pos_flat = quat_apply_inverse(q_expanded, p_flat)
  local_pos = local_pos_flat.reshape(B, N_query, 3)
  
  local_x = local_pos[..., 0]
  local_y = local_pos[..., 1]

  # Normalize coordinates to [-1, 1] for grid_sample
  min_x = -size_x / 2
  max_x = min_x + (W - 1) * res
  min_y = -size_y / 2
  max_y = min_y + (H - 1) * res

  norm_x = 2 * (local_x - min_x) / (max_x - min_x) - 1
  norm_y = 2 * (local_y - min_y) / (max_y - min_y) - 1

  # Create sampling grid: [B, H_out, W_out, 2]
  grid = torch.stack([norm_x, norm_y], dim=-1).reshape(B, 1, N_query, 2)
  
  sampled = torch.nn.functional.grid_sample(
    height_map,
    grid,
    mode='bilinear',
    padding_mode='border',
    align_corners=True
  )
  # sampled is [B, 1, 1, Nq]
  return sampled.reshape(B, N_query)


def _get_heights_from_single_ray_sensors(
  env: ManagerBasedRlEnv, sensor_names: str | list[str]
) -> torch.Tensor:
  """Get terrain heights by reading Z-hit positions from one or more single-ray sensors.
  
  Args:
      env: Environment instance.
      sensor_names: Single sensor name or list of sensor names (e.g. one per foot).
      
  Returns:
      Tensor of shape [B, N] (if list) or [B] (if single string), where N is len(list).
  """
  if isinstance(sensor_names, str):
      sensor_names = [sensor_names]
      is_single = True
  else:
      is_single = False
      
  heights_list = []
  for name in sensor_names:
      try:
          sensor = env.scene[name]
      except KeyError:
          # Fallback: return 0.0 if sensor not found
          heights_list.append(torch.zeros(env.num_envs, device=env.device))
          continue
          
      # Read hit Z
      # sensor.data.hit_pos_w is [B, ray_num, 3].
      hit_pos_w = sensor.data.hit_pos_w[..., 0, :] # [B, 3]
      hit_z = hit_pos_w[..., 2]
      dist = sensor.data.distances[..., 0]
      
      # Handle misses: if dist < 0, it means we scanned "sky" or hole.
      safe_z = torch.where(dist < 0, torch.tensor(-10.0, device=env.device), hit_z)
      
      heights_list.append(safe_z)
      
  # Stack: [B, N]
  result = torch.stack(heights_list, dim=1)
  
  if is_single:
      return result.squeeze(1) # [B]
  return result


def energy_saving ( #只需要关节传感器
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG  #类型注解
) -> torch.Tensor:
  """let the joint do less effort"""

  asset: Entity = env.scene[asset_cfg.name]
  torques = asset.data.actuator_force[:,asset_cfg.joint_ids]
  joint_vel = asset.data.joint_vel[:,asset_cfg.joint_ids]
  return torch.sum(torch.abs(torques*joint_vel),dim=1)


def feet_air_time_variance_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data

  # Check if data is available
  if sensor_data.last_air_time is None or sensor_data.last_contact_time is None:
    return torch.zeros(env.num_envs, device=env.device)

  last_air_time = sensor_data.last_air_time
  last_contact_time = sensor_data.last_contact_time

  # Compute variance across feet (dim=1)计算方差
  # penalize high variance in air time and contact time between feet to encourage symmetry
  reward = torch.var(torch.clamp(last_air_time, max=0.6), dim=1) + \
       torch.var(torch.clamp(last_contact_time, max=0.6), dim=1)
  # Upright mask (using asset_cfg to find robot)
  try:
    asset: Entity = env.scene[asset_cfg.name]
    proj_grav_z = asset.data.projected_gravity_b[:, 2]
    upright_scale = torch.clamp(-proj_grav_z, min=0.0, max=0.7) / 0.7
  except KeyError:
    upright_scale = torch.ones_like(reward)


  return reward * upright_scale


class progress_reward:
  """Reward for making progress since the last command update.
  
  Encourages the robot to actually move away from the starting point of the current command,
  preventing it from getting stuck (e.g., at stairs).
  
  Logic:
  - When command changes (or env resets), define a "checkpoint" P0 at current position.
  - Calculate distance D = |P_current - P0|.
  - Calculate expected distance D_exp = |V_cmd| * time_elapsed.
  - Ratio R = D / D_exp.
  - If R < 0.2 (stuck) and time > 0.5s: PENALTY.
  - If R > 0.2 (moving): REWARD (scaled by R).
  """
  
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.command_name = cfg.params["command_name"]
    self.threshold = cfg.params.get("threshold", 0.1)
    
    # State: [B, 3] last command (vx, vy, wz)
    self.last_command = torch.zeros((env.num_envs, 3), device=env.device)
    # State: [B, 2] checkpoint position (x, y)
    self.checkpoint_pos_w = torch.zeros((env.num_envs, 2), device=env.device)
    # State: [B] time since last command change
    self.accumulated_time = torch.zeros(env.num_envs, device=env.device)
    
  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    
    # 1. Get current command
    current_command = env.command_manager.get_command(command_name) # [B, 3]
    if current_command is None:
        return torch.zeros(env.num_envs, device=env.device)
        
    # Command magnitude (linear only)
    line_vel_cmd = current_command[:, :2] 
    cmd_norm = torch.norm(line_vel_cmd, dim=1) # [B]
    
    # 2. Detect Command Change or Environment Reset
    # Change if command vector differs significantly
    cmd_changed = torch.norm(current_command - self.last_command, dim=1) > 1e-4
    # Also treat episode reset (step 0) as a change point
    # We can check env.episode_length_buf, but ideally the caller relies on standard reset mechanisms.
    # However, for stateful rewards, we must handle restarts manually if not using built-in reset hooks.
    # Luckily, when env resets, usually commands correspond to a new set, or we can just rely on position jump.
    # A robust check: if accumulated_time > episode_length * dt (impossible), or just rely on cmd change.
    
    # Let's trust command change detection. 
    # Important: On env.reset(), `last_command` in memory might be stale but robot pos is new. 
    # We should probably reset if `env.episode_length_buf == 0`.
    is_reset = (env.episode_length_buf == 0)
    should_reset = cmd_changed | is_reset

    # 3. Update Checkpoints
    current_pos_w_xy = asset.data.root_link_pos_w[:, :2] # [B, 2]
    
    reset_indices = torch.nonzero(should_reset).squeeze(-1)
    if len(reset_indices) > 0:
        self.checkpoint_pos_w[reset_indices] = current_pos_w_xy[reset_indices]
        self.last_command[reset_indices] = current_command[reset_indices]
        self.accumulated_time[reset_indices] = 0.0
        
    # 4. Update Time
    self.accumulated_time += env.step_dt
    
    # 5. Calculate Metrics
    # Displacement from checkpoint
    displacement = current_pos_w_xy - self.checkpoint_pos_w
    dist_travelled = torch.norm(displacement, dim=1)
    
    expected_dist = cmd_norm * self.accumulated_time
    
    # Avoid division by zero
    progress_ratio = dist_travelled / (expected_dist + 1e-5)
    
    # 6. Compute Reward
    # Only active if command is significant (e.g. > threshold)
    active_mask = (cmd_norm > threshold).float()
    
    # Logic Refined: Use Soft Constraints instead of Hard Penalties (Tanh)
    # We want the ratio to be closer to 1.0. 
    # Use tanh to bound the reward smoothly between 0 and 1.
    # tanh(1.0) ~= 0.76, tanh(0.25) ~= 0.24, tanh(inf) -> 1.0
    
    # Check for stuck condition
    # If stuck (ratio < 0.25 after 0.5s), we force reward to 0.0.
    # Otherwise, we give the tanh(ratio).
    is_stuck = (progress_ratio < 0.25) & (self.accumulated_time > 0.5)
    
    # We remove explicit negative penalties to avoid value function collapse.
    # 0.0 is "bad enough" compared to positive rewards elsewhere.
    reward_term = torch.tanh(progress_ratio)
    
    final_reward = torch.where(is_stuck, torch.zeros_like(reward_term), reward_term)
    
    return final_reward * active_mask

