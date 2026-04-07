"""Unitree Go1 velocity environment configurations."""

from typing import Literal

from mjlab.asset_zoo.robots import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.envs.mdp import dr
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go1_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 50 # Disable or keep low for performance 防止穿模MuJoCo 的标准物理步是离散的。如果一个物体速度很快，在一个时间步内穿过了一堵薄墙，离散检测可能会漏掉碰撞（“穿模”）。
  cfg.sim.contact_sensor_maxmatch = 64 # Sufficient for most terrains

  cfg.scene.entities = {"robot": get_go1_robot_cfg()}

  # Set raycast sensor frame to Go1 trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "trunk"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  # Foot collision geoms have exact names (no numeric suffix).
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)
  # Calf/Thigh collision geoms are split into numbered segments in XML.
  # Consistency: use explicit enumerations for both matching and exclusion.
  calf_geom_names = tuple(
    f"{name}_calf_collision{i}" for name in foot_names for i in (1, 2)
  )
  thigh_geom_names = tuple(
    f"{name}_thigh_collision{i}" for name in foot_names for i in (1, 2, 3)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Calf contact: allowed (no termination, typically no penalty unless added later).
  calf_ground_cfg = ContactSensorCfg(
    name="calf_ground_contact",
    primary=ContactMatch(mode="geom", pattern=calf_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found","force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # Thigh contact: used for penalty (no termination).
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_contact",
    primary=ContactMatch(mode="geom", pattern=thigh_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  head_ground_cfg = ContactSensorCfg(
    name="head_ground_contact",
    primary=ContactMatch(mode="geom", pattern="head_collision", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      # Exclude feet + calves + thighs explicitly to leave body-only contacts.
      exclude=tuple(geom_names) + tuple(calf_geom_names) + tuple(thigh_geom_names)+ ("head_collision",) ,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
    calf_ground_cfg,
    thigh_ground_cfg,
    head_ground_cfg
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO1_ACTION_SCALE

  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

  # Added Domain Randomization: Mass, Damping, Friction Loss
  # Now physically accurate: pseudo_inertia smoothly scales both mass and moment of inertia
  cfg.events["body_inertia_mass"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
          # Use alpha_range to describe the symmetric scale limit. 
          # alpha param is the log-scale: [-0.2, 0.2] is roughly equivalent to scaling ~[0.8, 1.2]
          "alpha_range": (-0.2, 0.2), 
      },
  )
  cfg.events["dof_friction"] = EventTermCfg(
      func=dr.dof_frictionloss,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
          "operation": "add",
          "ranges": (0.0, 0.2), # Add random friction loss
      },
  )
  cfg.events["dof_damping"] = EventTermCfg(
      func=dr.dof_damping,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
          "operation": "scale",
          "ranges": (0.8, 1.2), # Damping +/- 20%
      },
  )

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.315,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.325,  
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.625,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.375,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.45,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.75,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
  
  # Configure rewards to use single-ray sensors
  # We pass a list of sensor names that match the order of the sites (FR, FL, RR, RL)
  foot_ray_names = [f"ray_{name}" for name in site_names]
  
  # Switch foot_clearance to use body-frame logic (no raycasts)
  # Target height in body frame (e.g. -0.2m means 20cm below body center)
  # Go1 body is ~0.28m high, so -0.18m implies ~10cm ground clearance.
  
  # Update base height tracking if it exists
  if "base_height" in cfg.rewards:
     cfg.rewards["base_height"].params["sensor_name"] = "ray_base"
     
  # Update foot swing height if it exists
  if "foot_swing_height" in cfg.rewards:
      cfg.rewards["foot_swing_height"].params["height_sensor_name"] = None

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip", "feet_swing_height_variance"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.475
  # Override base placeholder reward: bind sensor + weight.
  cfg.rewards["calf_collision"].params["sensor_name"] = calf_ground_cfg.name
  #cfg.rewards["calf_collision"].params["threshold"] = 2.0  # Allow grazing contacts < 15N
  cfg.rewards["calf_collision"].params["force_threshold"] = 2.0 
  cfg.rewards["calf_collision"].weight = -1.0  # tweak within [-1.0, -3.0]
  cfg.rewards["thigh_collision"].params["sensor_name"] = thigh_ground_cfg.name
  cfg.rewards["thigh_collision"].params["force_threshold"] = 2.0
  #cfg.rewards["thigh_collision"].params["threshold"] = 2.0
  cfg.rewards["thigh_collision"].weight = -0.75  # tweak within [-1.0, -3.0]
  cfg.rewards["stumble"].params["sensor_names"] = [
    calf_ground_cfg.name,   
    thigh_ground_cfg.name,
  ]
  cfg.rewards["stumble"].weight = -0.3 # Increased penalty to force leg lifting
  cfg.rewards["foot_clearance"].weight = -1.25
  cfg.rewards["foot_clearance"].params["target_height"]=-0.145
  cfg.rewards["foot_swing_height"].weight = -0.125 # Penalty for deviation from target height (MUST BE NEGATIVE)
  cfg.rewards["stumble"].params={
        "sensor_names": ["feet_ground_contact"],
      }
  cfg.rewards["foot_slip"].weight = -0.3

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": nonfoot_ground_cfg.name,
      "force_threshold": 10.0,  # Required by recent history-based contact API updates
    },
  )


  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 6
        cfg.scene.terrain.terrain_generator.num_rows = 6
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 flat terrain velocity configuration."""
  cfg = unitree_go1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  #del cfg.observations["actor"].terms["height_scan"]
  #del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.2, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
