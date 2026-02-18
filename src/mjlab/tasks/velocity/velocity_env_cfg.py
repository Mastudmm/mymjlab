"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Observations
  ##

  # Policy 不再包含 base_lin_vel，让其只由 critic 访问。
  policy_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.015, n_max=0.015),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }
  # 为 joint_pos 增加延迟
  policy_terms["joint_pos"].delay_min_lag = 1
  policy_terms["joint_pos"].delay_max_lag = 3
  policy_terms["joint_pos"].delay_per_env = True
  policy_terms["joint_pos"].delay_hold_prob = 0.7      # 70% 的概率保持当前延迟不变
  policy_terms["joint_pos"].delay_update_period = 50   # 每 50 步尝试更新一次延迟
  policy_terms["joint_pos"].delay_per_env_phase = True # 每个 env 的刷新相位不同

  # 为 joint_vel 增加延迟
  policy_terms["joint_vel"].delay_min_lag = 1
  policy_terms["joint_vel"].delay_max_lag = 3
  policy_terms["joint_vel"].delay_per_env = True
  policy_terms["joint_vel"].delay_hold_prob = 0.7
  policy_terms["joint_vel"].delay_update_period = 50
  policy_terms["joint_vel"].delay_per_env_phase = True
  '''
  如果是单独添加历史，就在这里处理：policy_terms["joint_pos"].history_length = 3
    policy_terms["joint_pos"].flatten_history_dim = True
    policy_terms["joint_vel"].history_length = 3
    policy_terms["joint_vel"].flatten_history_dim = True
    为什么不在上面直接加？因为 critic_terms = 复用了policy_terms,如果直接加critic也会有历史
  '''
  critic_terms = {
    **policy_terms,
    # 将 base_lin_vel 单独保留在 critic，用于价值估计。
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # Apply history to the entire policy observation group:
  # stack current + past 2 frames for all policy terms, flatten history dims for MLP inputs.
  observations["policy"].history_length = 10
  observations["policy"].flatten_history_dim = True
  observations["critic"].history_length = 10
  observations["critic"].flatten_history_dim = True

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(15.0, 18.0),
      rel_standing_envs=0.1,
      rel_heading_envs=1.0, #给一个期望的朝向角（heading），不要求前进/横移速度；通过朝向控制把机体转到目标朝向
      rel_pure_x_envs=0.8, # 10% 的概率只产生 X 方向速度 (vy=0)
      rel_pure_y_envs=0.1, # 10% 的概率只产生 Y 方向速度 (vx=0)
      heading_command=True,
      heading_control_stiffness=1.0,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 1.5),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.0),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=mdp.randomize_encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  ##
  # Rewards
  ##
 
  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=4.0,
      params={"command_name": "twist", "std": math.sqrt(0.09)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.2)},
    ),
    "progress": RewardTermCfg(
      func=mdp.progress_reward,
      weight=0.01,
      params={
        "command_name": "twist",
        "threshold": 0.1,
      },
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "pose": RewardTermCfg( #保持关节角度，threshold代表在行走或者是跑动的时候限制将被放宽
      func=mdp.variable_posture,
      weight=0.75,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.10,
        "running_threshold": 1.25,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,  # Override per-robot
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,  # Override per-robot
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "j_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.0013),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,  # Increased to encourage lifting legs
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.125,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.10,
      },
    ),
    "base_height": RewardTermCfg(
      func=mdp.track_base_height,
      weight=0.0, # Highly negative to punish crouching. Adjusted for asymmetric penalty function.
      params={
         "target_height": 0.28, # Nominal height for Go1
         "asset_cfg": SceneEntityCfg("robot"),
         "sensor_name": "ray_base", # Ensure sensor is used
       },
     ),
    "stumble": RewardTermCfg(
      func=mdp.stumble_penalty,
      weight=-0.075,
      params={
        "sensor_names": "calf_ground_contact",
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=0.0,
      params={
        "target_height": -0.2,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "target_height": 0.15,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "height_sensor_name": None, # Explicitly allow override
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.45,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-4,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.10,
      },
    ),
    "calf_collision": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=0.0,  # Override per-robot
      params={"sensor_name": "", "threshold": 1.0},  # Set per-robot (e.g. calf_ground_contact)
    ),
    "thigh_collision": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=0.0,  # Override per-robot
      params={"sensor_name": "", "threshold": 1.0},  # Set per-robot (e.g. thigh_ground_contact)
    ),
    "energy_save":RewardTermCfg(
      func=mdp.energy_saving,
      weight= -0.000001,
      params={"asset_cfg":SceneEntityCfg("robot")}
    ),
    "feet_air_time_variance": RewardTermCfg(
      func=mdp.feet_air_time_variance_penalty,
      weight=-0.4, # Default to 0.0, to be tuned per robot/task
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
  }


  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(75.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.5, 1.25), "ang_vel_z": (-1.25, 1.25)},
          {"step": 1800 * 24, "lin_vel_x": (-0.75, 1.5), "ang_vel_z": (-1.75, 1.75)},
          {"step": 4000 * 24, "lin_vel_x": (-1.0, 2.0)},  
        ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
