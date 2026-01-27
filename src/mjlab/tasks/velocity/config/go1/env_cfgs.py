"""Unitree Go1 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg, ObjRef, GridPatternCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_go1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 50 # Disable or keep low for performance 防止穿模MuJoCo 的标准物理步是离散的。如果一个物体速度很快，在一个时间步内穿过了一堵薄墙，离散检测可能会漏掉碰撞（“穿模”）。
  cfg.sim.contact_sensor_maxmatch = 64 # Sufficient for most terrains

  cfg.scene.entities = {"robot": get_go1_robot_cfg()}

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
  )
  # Thigh contact: used for penalty (no termination).
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_contact",
    primary=ContactMatch(mode="geom", pattern=thigh_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  nonfootleg_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      # Exclude feet + calves + thighs explicitly to leave body-only contacts.
      exclude=tuple(geom_names) + tuple(calf_geom_names) + tuple(thigh_geom_names) ,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  
  # Minimalistic Per-Foot Ray Sensors (1 ray per foot)
  # This dramatically reduces VRAM usage and computation vs a full grid.
  foot_ray_sensors = []
  for site_name in site_names: # FR, FL, RR, RL
      sensor = RayCastSensorCfg(
          name=f"ray_{site_name}",
          frame=ObjRef(type="site", name=site_name, entity="robot"),
          # Single ray pointing down
          pattern=GridPatternCfg(size=(0.0, 0.0), resolution=1.0, direction=(0.0, 0.0, -1.0)),
          ray_alignment="world", # Always point down in world frame, ignore foot rotation
          max_distance=1.5,
          debug_vis=False, # Ensure debug visualization is off
          # OPTIMIZATION: Only hit terrain (usually group 0), ignore robot parts (usually 1, 2...)
          # If this causes rays to pass through floor, remove this line.
          include_geom_groups=(0,), 
          exclude_parent_body=True, 
      )
      foot_ray_sensors.append(sensor)
  
  # Base height sensor (1 ray at trunk center)
  base_ray_sensor = RayCastSensorCfg(
      name="ray_base",
      frame=ObjRef(type="body", name="trunk", entity="robot"),
      pattern=GridPatternCfg(size=(0.0, 0.0), resolution=1.0, direction=(0.0, 0.0, -1.0)),
      ray_alignment="world", 
      max_distance=1.5,
      debug_vis=False,
      include_geom_groups=(0,),
      exclude_parent_body=True,
  )

  # Register sensors
  cfg.scene.sensors = (
    feet_ground_cfg,
    calf_ground_cfg,
    thigh_ground_cfg,
    nonfootleg_ground_cfg,
    *foot_ray_sensors, 
    base_ray_sensor,
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
  cfg.events["body_mass"] = EventTermCfg(
      func=mdp.randomize_field,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
          "field": "body_mass",
          "operation": "scale",
          "ranges": (0.8, 1.2), # Mass +/- 20%
      },
  )
  cfg.events["dof_friction"] = EventTermCfg(
      func=mdp.randomize_field,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
          "field": "dof_frictionloss",
          "operation": "add",
          "ranges": (0.0, 0.2), # Add random friction loss
      },
  )
  cfg.events["dof_damping"] = EventTermCfg(
      func=mdp.randomize_field,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
          "field": "dof_damping",
          "operation": "scale",
          "ranges": (0.8, 1.2), # Damping +/- 20%
      },
  )

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.4,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.7,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.375,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.425,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.75,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
  
  # Configure rewards to use single-ray sensors
  # We pass a list of sensor names that match the order of the sites (FR, FL, RR, RL)
  foot_ray_names = [f"ray_{name}" for name in site_names]
  cfg.rewards["foot_clearance"].params["sensor_name"] = foot_ray_names
  
  # Update base height tracking if it exists
  if "base_height" in cfg.rewards:
     cfg.rewards["base_height"].params["sensor_name"] = "ray_base"
     
  # Update foot swing height if it exists
  if "foot_swing_height" in cfg.rewards:
      cfg.rewards["foot_swing_height"].params["height_sensor_name"] = foot_ray_names

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = 0.0
  cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"].weight = 0.15
  # Override base placeholder reward: bind sensor + weight.
  cfg.rewards["calf_collision"].params["sensor_name"] = calf_ground_cfg.name
  cfg.rewards["calf_collision"].weight = -0.25  # tweak within [-1.0, -3.0]
  cfg.rewards["thigh_collision"].params["sensor_name"] = thigh_ground_cfg.name
  cfg.rewards["thigh_collision"].weight = -0.25  # tweak within [-1.0, -3.0]
  cfg.rewards["stumble"].params["sensor_names"] = [
    calf_ground_cfg.name,   
    thigh_ground_cfg.name,
  ]
  cfg.rewards["stumble"].weight = -0.5 # Increased penalty to force leg lifting
  cfg.rewards["foot_clearance"].weight = -0.001
  cfg.rewards["foot_swing_height"].weight = -0.175 # Penalty for deviation from target height (MUST BE NEGATIVE)
  cfg.rewards["stumble"].params={
        "sensor_names": ["feet_ground_contact","calf_ground_contact"],
      }
  cfg.rewards["foot_slip"].weight = -0.875

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfootleg_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

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

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  del cfg.curriculum["terrain_levels"]

  return cfg
