"""Parkour-specific reward and termination terms."""

from mjlab.tasks.parkour_vision.mdp.rewards import (
  contact_penalty_excluding as contact_penalty_excluding,
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
  flat_orientation_excluding as flat_orientation_excluding,
)
from mjlab.tasks.parkour_vision.mdp.rewards import (
  reward_point_bonus as reward_point_bonus,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  bad_orientation_excluding as bad_orientation_excluding,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  base_height_below as base_height_below,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  out_of_bounds as out_of_bounds,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  illegal_contact_excluding as illegal_contact_excluding,
)
from mjlab.tasks.parkour_vision.mdp.terminations import (
  parkour_completed as parkour_completed,
)

__all__ = [
  "reward_point_bonus",
  "forward_progress",
  "heading_alignment",
  "lateral_track",
  "contact_penalty_excluding",
  "flat_orientation_excluding",
  "parkour_completed",
  "base_height_below",
  "out_of_bounds",
  "bad_orientation_excluding",
  "illegal_contact_excluding",
]
