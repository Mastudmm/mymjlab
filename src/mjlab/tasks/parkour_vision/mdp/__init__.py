"""Parkour-specific reward, termination, observation and curriculum terms."""

from mjlab.tasks.parkour_vision.mdp.curriculums import (
  commands_vel_score as commands_vel_score,
)
from mjlab.tasks.parkour_vision.mdp.curriculums import (
  terrain_levels_by_pass_rate as terrain_levels_by_pass_rate,
)
from mjlab.tasks.parkour_vision.mdp.curriculums import (
  waypoint_timeout_schedule as waypoint_timeout_schedule,
)
from mjlab.tasks.parkour_vision.mdp.observations import (
  depth_camera_normalized as depth_camera_normalized,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  cheat_penalty as cheat_penalty,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  contact_penalty_excluding as contact_penalty_excluding,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  contact_penalty_force_scaled as contact_penalty_force_scaled,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  forward_progress as forward_progress,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  heading_alignment as heading_alignment,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  lateral_track as lateral_track,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  reward_point_bonus as reward_point_bonus,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  track_goal_vel as track_goal_vel,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  track_linear_velocity_clipped as track_linear_velocity_clipped,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  wall_climb_shaping as wall_climb_shaping,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  bad_orientation_excluding as bad_orientation_excluding,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  base_height_below as base_height_below,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  illegal_contact_excluding as illegal_contact_excluding,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  out_of_bounds as out_of_bounds,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  parkour_completed as parkour_completed,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  vel_violate as vel_violate,
)
from mjlab.tasks.parkour_vision.mdp.waypoint_command import (
  WaypointVelocityCommand as WaypointVelocityCommand,
)
from mjlab.tasks.parkour_vision.mdp.waypoint_command import (
  WaypointVelocityCommandCfg as WaypointVelocityCommandCfg,
)

__all__ = [
  "reward_point_bonus",
  "wall_climb_shaping",
  "forward_progress",
  "heading_alignment",
  "lateral_track",
  "track_linear_velocity_clipped",
  "track_goal_vel",
  "cheat_penalty",
  "contact_penalty_excluding",
  "contact_penalty_force_scaled",
  "parkour_completed",
  "base_height_below",
  "out_of_bounds",
  "bad_orientation_excluding",
  "illegal_contact_excluding",
  "vel_violate",
  "depth_camera_normalized",
  "commands_vel_score",
  "terrain_levels_by_pass_rate",
  "waypoint_timeout_schedule",
  "WaypointVelocityCommand",
  "WaypointVelocityCommandCfg",
]
