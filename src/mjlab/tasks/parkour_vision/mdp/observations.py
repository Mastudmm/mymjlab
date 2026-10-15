"""Parkour-specific observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def depth_camera_normalized(
  env: ManagerBasedRlEnv,
  sensor_cfg: SceneEntityCfg,
  far_clip: float = 3.0,
) -> torch.Tensor:
  """Depth image normalized to wmp_go1's [-0.5, 0.5] range.

  wmp_go1 (sim2real validated) maps clipped depth [0, far] to [0.5, -0.5]:
  near surfaces read positive, far/missed rays read -0.5. Compared to the
  previous scale=1/4 on a 4 m range, this halves the per-meter gain decay and
  bounds the observation range regardless of the sensor's max distance.

  Missed rays (< 0) are replaced by ``far_clip`` (i.e. -0.5 after mapping).
  """
  sensor: RayCastSensor = env.scene[sensor_cfg.name]
  distances = sensor.data.distances
  distances = torch.where(distances < 0, far_clip, distances)
  distances = torch.clamp(distances, max=far_clip)
  return (0.5 - distances / far_clip).flatten(start_dim=1)
